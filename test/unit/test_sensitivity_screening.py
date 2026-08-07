"""Unit tests for the sensitivity-screening decision layer.

Two surfaces:

  1. `sensitivity.parse_verdict` — the pure answer parser, including
     the ambiguity rules (contradictory answers resolve SENSITIVE,
     unknown categories are dropped, garbage raises so the caller's
     fail-closed path takes over).
  2. `clip_archival.resolve_clip_for_upload` — the fail-closed decision
     matrix. The VLM call and the censor pipeline are monkeypatched at
     the module seams (`clip_archival.sensitivity` /
     `clip_archival.censor`); everything else is real, including the
     tempfile round-trip (so we also assert no raw temp file survives).

The matrix under test (docs/plans/2026-07-27-room-privacy-and-clip-
censoring.md):

  screen → not sensitive  → raw bytes pass through
  screen → sensitive      → censored bytes only (person-mask mode)
  screen error            → censored bytes only (FULL-FRAME mode)
  censor error            → DROP (upload_bytes=None), never raw
  kill-switch off         → raw passes through, screen never called
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import pytest

from src.services import clip_archival
from src.services.detection.censor import CensorError
from src.services.detection.sensitivity import (
    SENSITIVITY_CATEGORIES,
    ScreenResult,
    parse_verdict,
)
from src.settings import Settings

RAW_MP4 = b"\x00\x00\x00\x18ftypisom-raw-clip-bytes"
CENSORED_MP4 = b"\x00\x00\x00\x18ftypisom-censored-bytes"

EVENT_ID = uuid4()
CAMERA_ID = uuid4()


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "db_url": "postgresql://x",
        "supabase_url": "https://example.supabase.co",
        "supabase_publishable_key": "anon",
        "supabase_jwt_secret": "jwtsecret",
        "supabase_service_role_key": "service-role-key",
        "sensitivity_screening_enabled": True,
    }
    base.update(overrides)
    return Settings(**base)


# ── parse_verdict ─────────────────────────────────────────────────────


def test_parse_verdict_clean() -> None:
    res = parse_verdict('{"sensitive": false, "categories": [], "confidence": 0.9}')
    assert res == ScreenResult(sensitive=False, categories=[], confidence=0.9)


def test_parse_verdict_sensitive_with_categories() -> None:
    res = parse_verdict(
        '{"sensitive": true, "categories": ["toileting", "nudity_or_undress"],'
        ' "confidence": 0.8}'
    )
    assert res.sensitive is True
    assert res.categories == ["toileting", "nudity_or_undress"]


def test_parse_verdict_contradiction_resolves_sensitive() -> None:
    """sensitive=false but categories named → the model saw something
    worth naming; ambiguity resolves SENSITIVE."""
    res = parse_verdict(
        '{"sensitive": false, "categories": ["bathing_showering"], "confidence": 0.4}'
    )
    assert res.sensitive is True
    assert res.categories == ["bathing_showering"]


def test_parse_verdict_drops_unknown_categories() -> None:
    res = parse_verdict(
        '{"sensitive": true, "categories": ["toileting", "made_up_thing"],'
        ' "confidence": 1.0}'
    )
    assert res.categories == ["toileting"]
    assert all(c in SENSITIVITY_CATEGORIES for c in res.categories)


def test_parse_verdict_clamps_confidence() -> None:
    assert parse_verdict(
        '{"sensitive": true, "categories": [], "confidence": 7}'
    ).confidence == 1.0
    assert parse_verdict(
        '{"sensitive": true, "categories": [], "confidence": "bogus"}'
    ).confidence == 0.0


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "{}",                                # missing 'sensitive'
        '{"sensitive": true, "categories": "toileting"}',  # not a list
        "",
    ],
)
def test_parse_verdict_garbage_raises(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_verdict(raw)


# ── resolve_clip_for_upload fail-closed matrix ────────────────────────


class _FakeCensor:
    """censor_clip stand-in: records the mode + paths, writes canned
    bytes (or raises)."""

    def __init__(self, *, raises: bool = False, write: bytes = CENSORED_MP4):
        self.raises = raises
        self.write = write
        self.calls: list[dict[str, Any]] = []

    async def censor_clip(
        self, in_path: str, out_path: str, *, full_frame: bool = False,
        keep_audio: bool = True, mode: str = "pixelate",
    ) -> None:
        self.calls.append(
            {"in_path": in_path, "out_path": out_path,
             "full_frame": full_frame, "keep_audio": keep_audio, "mode": mode}
        )
        if self.raises:
            raise CensorError("boom")
        with open(out_path, "wb") as f:
            f.write(self.write)


async def _resolve(mp4: bytes, settings: Settings) -> clip_archival.ClipResolution:
    return await clip_archival.resolve_clip_for_upload(
        mp4, event_id=EVENT_ID, camera_id=CAMERA_ID, settings=settings,
    )


async def test_kill_switch_off_bypasses_screen(monkeypatch) -> None:
    """The kill-switch is honored in env=local/test only (here: the
    default env, local)."""
    async def _never_called(mp4: bytes, *, settings: Settings) -> ScreenResult:
        raise AssertionError("screen_clip must not run when the switch is off")

    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _never_called)
    res = await _resolve(RAW_MP4, _settings(sensitivity_screening_enabled=False))
    assert res.upload_bytes == RAW_MP4
    assert res.censored is False
    assert res.categories is None


@pytest.mark.parametrize("env", ["staging", "prod"])
async def test_kill_switch_ignored_outside_local_test(
    monkeypatch, caplog, env: str,
) -> None:
    """Outside local/test a false kill-switch is IGNORED (fail-closed by
    env, mirroring internal_service_key): screening still runs, and a
    loud warning surfaces the misconfiguration."""
    ran = {"n": 0}

    async def _clean(mp4: bytes, *, settings: Settings) -> ScreenResult:
        ran["n"] += 1
        return ScreenResult(sensitive=False, categories=[], confidence=0.95)

    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _clean)
    with caplog.at_level("WARNING", logger="src.services.clip_archival"):
        res = await _resolve(
            RAW_MP4, _settings(sensitivity_screening_enabled=False, env=env),
        )
    assert ran["n"] == 1  # the screen ran despite the switch
    assert res.upload_bytes == RAW_MP4  # clean verdict → raw still passes
    assert any(
        "sensitivity_screening_enabled=false ignored outside local/test"
        in r.message
        for r in caplog.records
    )


async def test_not_sensitive_passes_raw_through(monkeypatch) -> None:
    async def _clean(mp4: bytes, *, settings: Settings) -> ScreenResult:
        return ScreenResult(sensitive=False, categories=[], confidence=0.95)

    fake = _FakeCensor()
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _clean)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes == RAW_MP4
    assert res.censored is False
    assert fake.calls == []  # clean clip → censor never touched


async def test_sensitive_uploads_censored_bytes_only(monkeypatch) -> None:
    async def _sensitive(mp4: bytes, *, settings: Settings) -> ScreenResult:
        return ScreenResult(
            sensitive=True, categories=["toileting"], confidence=0.9,
        )

    fake = _FakeCensor()
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _sensitive)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes == CENSORED_MP4
    assert res.censored is True
    assert res.categories == ["toileting"]
    # Person-mask mode: a real verdict means the censor may target persons.
    assert [c["full_frame"] for c in fake.calls] == [False]
    # The raw + censored tempfiles are gone after the decision.
    for call in fake.calls:
        assert not os.path.exists(call["in_path"])
        assert not os.path.exists(call["out_path"])


async def test_low_confidence_not_sensitive_is_censored(monkeypatch) -> None:
    """Doubt → censor, in code: a well-formed sensitive=false verdict
    BELOW the confidence floor (default 0.6) must not clear the clip."""

    async def _hesitant_clear(mp4: bytes, *, settings: Settings) -> ScreenResult:
        return ScreenResult(sensitive=False, categories=[], confidence=0.05)

    fake = _FakeCensor()
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _hesitant_clear)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes == CENSORED_MP4
    assert res.censored is True
    # We DO have a verdict (just a doubted one) → person-mask mode.
    assert [c["full_frame"] for c in fake.calls] == [False]


async def test_confident_not_sensitive_passes_raw(monkeypatch) -> None:
    """The floor pair: sensitive=false at 0.9 (≥ 0.6 floor) is a real
    clearance — raw passes."""

    async def _clean(mp4: bytes, *, settings: Settings) -> ScreenResult:
        return ScreenResult(sensitive=False, categories=[], confidence=0.9)

    fake = _FakeCensor()
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _clean)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes == RAW_MP4
    assert res.censored is False
    assert fake.calls == []


async def test_screen_failure_censors_full_frame(monkeypatch) -> None:
    """VLM error/timeout → no trusted verdict → FULL-FRAME censor,
    categories unknown (None). Raw never passes."""

    async def _boom(mp4: bytes, *, settings: Settings) -> ScreenResult:
        raise RuntimeError("gemini 500")

    fake = _FakeCensor()
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _boom)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes == CENSORED_MP4
    assert res.censored is True
    assert res.categories is None
    assert [c["full_frame"] for c in fake.calls] == [True]


async def test_censor_failure_drops_clip(monkeypatch) -> None:
    """Sensitive + censor pipeline dies → DROP: upload_bytes is None
    (caller stamps clip_suppressed); the raw bytes are never returned."""

    async def _sensitive(mp4: bytes, *, settings: Settings) -> ScreenResult:
        return ScreenResult(
            sensitive=True, categories=["nudity_or_undress"], confidence=0.7,
        )

    fake = _FakeCensor(raises=True)
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _sensitive)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes is None
    assert res.censored is True
    assert res.categories == ["nudity_or_undress"]
    # Tempfiles cleaned even on the failure path.
    for call in fake.calls:
        assert not os.path.exists(call["in_path"])
        assert not os.path.exists(call["out_path"])


async def test_screen_and_censor_both_failing_drops_clip(monkeypatch) -> None:
    """The worst path: no verdict AND the full-frame fallback dies →
    still a drop, never a raw upload."""

    async def _boom(mp4: bytes, *, settings: Settings) -> ScreenResult:
        raise TimeoutError("screen timeout")

    fake = _FakeCensor(raises=True)
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _boom)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes is None
    assert res.censored is True
    assert res.categories is None
    assert [c["full_frame"] for c in fake.calls] == [True]


async def test_censor_writing_zero_bytes_drops_clip(monkeypatch) -> None:
    """A censor that 'succeeds' but produces an empty file is a
    failure — zero bytes must not be uploaded as a clip."""

    async def _sensitive(mp4: bytes, *, settings: Settings) -> ScreenResult:
        return ScreenResult(sensitive=True, categories=[], confidence=0.6)

    fake = _FakeCensor(write=b"")
    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _sensitive)
    monkeypatch.setattr(clip_archival.censor, "censor_clip", fake.censor_clip)
    res = await _resolve(RAW_MP4, _settings())
    assert res.upload_bytes is None
    assert res.censored is True


def test_parse_verdict_distress_only_downgrades_to_clean() -> None:
    """Record-only categories (emotional_distress) never censor by
    themselves (2026-07-28: a staged fall with a cry was blurred because
    the model HEARD the shout — blur is for bodily privacy)."""
    r = parse_verdict(
        '{"sensitive": true, "categories": ["emotional_distress"], '
        '"confidence": 0.9}'
    )
    assert r.sensitive is False
    assert r.categories == ["emotional_distress"]  # still reported


def test_parse_verdict_distress_plus_exposure_stays_sensitive() -> None:
    r = parse_verdict(
        '{"sensitive": true, "categories": ["emotional_distress", '
        '"nudity_or_undress"], "confidence": 0.9}'
    )
    assert r.sensitive is True


def test_parse_verdict_sensitive_with_no_categories_stays_sensitive() -> None:
    """Unknown exposure — fail closed."""
    r = parse_verdict('{"sensitive": true, "categories": [], "confidence": 0.4}')
    assert r.sensitive is True


def test_parse_verdict_distress_contradiction_does_not_promote() -> None:
    """sensitive=false + only record-only categories stays clean (the
    contradiction rule promotes on TRIGGER categories only)."""
    r = parse_verdict(
        '{"sensitive": false, "categories": ["emotional_distress"], '
        '"confidence": 0.8}'
    )
    assert r.sensitive is False
