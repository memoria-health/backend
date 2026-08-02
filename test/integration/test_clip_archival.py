"""Integration tests for `src.services.clip_archival.archive_event`.

Per `feedback_no_mocking_async_wiring`: the supporting objects are
all real — real Event/Patient/Caregiver rows, real FrameBuffer /
AudioBuffer instances pre-populated with synthetic frames. Only the
storage HTTP layer is swapped out (via the `fake_httpx` pattern from
test_storage.py) so we can assert the exact wire shape and avoid
touching real Supabase Storage.

What we assert:

  - The full pipeline lands `events.has_clip=true` and
    `events.clip_storage_key=events/{event_id}.mp4`.
  - The Supabase upload is a single POST to the right
    `incident-clips/events/{id}.mp4` URL with the combined MP4 body.
  - If the post-event frame snapshot is empty, the row stays as-is
    (the events row is the user-facing surface; the clip is optional).
  - the single-encode assembly produces a valid MP4 (uses
    ffmpeg; skipped if not on PATH).
  - The sensitivity screen (2026-07-27) runs on EVERY clip before
    upload — private cameras included (they get clips now; the old
    private-camera suppression is gone). A sensitive verdict uploads a
    CENSORED clip + stamps `clip_censored`/`sensitivity_categories`; a
    censor failure DROPS the clip + stamps `clip_suppressed`; a screen
    failure censors full-frame. The VLM + censor are monkeypatched at
    the `clip_archival.sensitivity` / `clip_archival.censor` seams so
    no real Gemini/YOLO runs in the suite (the decision matrix itself
    is unit-tested in test/unit/test_sensitivity_screening.py).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.camera import Camera
from src.db.caregiver import Caregiver
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services import clip_archival, storage as storage_module
from src.services.clip_archival import archive_event
from src.services.detection.audio_buffer import AudioBuffer
from src.services.detection.censor import CensorError
from src.services.detection.frame_buffer import FrameBuffer
from src.services.detection.mp4_remux import frames_to_mp4
from src.services.detection.sensitivity import ScreenResult
from src.settings import Settings, get_settings

_pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)

_CENSORED_BYTES = b"\x00\x00\x00\x18ftypisom-censored-stand-in"


def _patch_screen(monkeypatch, result: ScreenResult | Exception) -> None:
    """Swap the pinned-model VLM call at the clip_archival seam."""

    async def _screen(mp4: bytes, *, settings: Settings) -> ScreenResult:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(clip_archival.sensitivity, "screen_clip", _screen)


def _patch_censor(monkeypatch, *, raises: bool = False) -> list[dict[str, Any]]:
    """Swap the censor pipeline; returns the recorded calls."""
    calls: list[dict[str, Any]] = []

    async def _censor(
        in_path: str, out_path: str, *, full_frame: bool = False,
        keep_audio: bool = True,
    ) -> None:
        # keep_audio records the intimate-situation carve-out: censored
        # clips carry sound EXCEPT on a positive intimacy verdict, where
        # the audio is itself the exposure.
        calls.append({"full_frame": full_frame, "keep_audio": keep_audio})
        if raises:
            raise CensorError("censor stack down")
        with open(out_path, "wb") as f:
            f.write(_CENSORED_BYTES)

    monkeypatch.setattr(clip_archival.censor, "censor_clip", _censor)
    return calls


_CLEAN = ScreenResult(sensitive=False, categories=[], confidence=0.99)


# ── HTTP fake (mirrors test_storage.py's pattern) ─────────────────────


class _FakeAsyncClient:
    state: dict[str, Any] = {"responses": [], "captured": []}

    def __init__(self, *, timeout: float | None = None, **_kw: Any) -> None:
        self._timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, url: str, **kw: Any) -> httpx.Response:
        return self._record("POST", url, kw)

    async def get(self, url: str, **kw: Any) -> httpx.Response:
        return self._record("GET", url, kw)

    async def delete(self, url: str, **kw: Any) -> httpx.Response:
        return self._record("DELETE", url, kw)

    @classmethod
    def _record(cls, method: str, url: str, kw: dict[str, Any]) -> httpx.Response:
        cls.state["captured"].append({"method": method, "url": url, "kw": kw})
        if not cls.state["responses"]:
            raise RuntimeError(f"no canned response for {method} {url}")
        return cls.state["responses"].pop(0)


def _install_fake_httpx(monkeypatch) -> dict[str, Any]:
    """Install the fake AFTER any seeding step that hits real
    Supabase. The fake replaces httpx.AsyncClient globally, so any
    other code path that constructs one (e.g. conftest's
    `auth_user_factory` hitting `/auth/v1/admin/users`) gets the
    fake too — which would deadlock seeding."""
    state: dict[str, Any] = {"responses": [], "captured": []}
    _FakeAsyncClient.state = state
    monkeypatch.setattr(storage_module.httpx, "AsyncClient", _FakeAsyncClient)
    return state


# ── seeding ───────────────────────────────────────────────────────────


async def _seed_event(
    session: AsyncSession, auth_user_factory
) -> tuple[UUID, UUID, UUID]:
    uid, email = await auth_user_factory()
    now = datetime.now(timezone.utc)
    session.add(
        Caregiver(
            id=uid, display_name="Clip Tester", email=email,
            created_at=now, updated_at=now,
        )
    )
    patient_id = uuid4()
    session.add(
        Patient(
            id=patient_id, caregiver_id=uid,
            full_name="Margaret O'Brien", preferred_name="Maggie",
            relation="parent", state="calm", status_line="OK",
            cameras_paused=False, created_at=now, updated_at=now,
        )
    )
    await session.flush()
    camera_id = uuid4()
    session.add(
        Camera(
            id=camera_id, patient_id=patient_id, display_id=1,
            room="Bedroom", icon_name="tv", online=True, created_at=now,
        )
    )
    await session.flush()
    event_id = uuid4()
    session.add(
        Event(
            id=event_id, patient_id=patient_id,
            detector_id=None, camera_id=camera_id,
            category="safety", source="preset", icon_name="fall",
            title="Fall", location="Bedroom",
            occurred_at=now, has_clip=False,
            narrative="Maggie fell.", opened=False, reviewed=False,
            review_choice=None, review_notes=None, reviewed_at=None,
            clip_storage_key=None, created_at=now,
        )
    )
    await session.commit()
    return event_id, patient_id, camera_id


def _i420_blob(width: int, height: int) -> bytes:
    return b"\x80" * (width * height * 3 // 2)


def _build_pre_frames() -> list:
    """Pre-roll frame snapshots as the tick now hands them to
    archive_event (clip_fps refs; single-encode assembly). Past-dated
    captured_at so the boundary-overlap filter keeps the (future-dated)
    post fixture frames."""
    from src.services.detection.frame_buffer import FrameSnapshot
    return [
        FrameSnapshot(
            width=320, height=240,
            data=_i420_blob(320, 240),
            captured_at=100.0 + i / 5,
        )
        for i in range(15)  # 3s @ 5 fps
    ]


@pytest_asyncio.fixture
async def filled_buffers() -> AsyncIterator[tuple[FrameBuffer, AudioBuffer]]:
    """A pre-populated post-event frame + audio buffer.

    archive_event() sleeps for clip_post_event_wait_seconds and then
    calls snapshot_last_seconds(clip_post_event_wait_seconds) against
    time.monotonic(). In production the room-worker is continuously
    pushing fresh frames during that sleep, so the snapshot always
    finds recent ones. In the test we have no such background pusher,
    so we future-date the fixture's captured_at values to outlive any
    setup delay between fixture build and archive_event call (ffmpeg
    pre-mp4 build can take a few hundred ms).
    """
    import time
    base = time.monotonic() + 2.0
    frames = FrameBuffer(window_seconds=15.0, target_fps=5)
    audio = AudioBuffer(window_seconds=15.0)
    for i in range(15):
        frames.push(
            width=320, height=240, data=_i420_blob(320, 240),
            captured_at=base + i / 5,
        )
    yield frames, audio


# ── tests ─────────────────────────────────────────────────────────────


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_uploads_and_flips_has_clip(
    auth_user_factory, filled_buffers, monkeypatch,
):
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(monkeypatch, _CLEAN)
    frame_buffer, audio_buffer = filled_buffers
    pre_frames = _build_pre_frames()
    # Override the wait so the test doesn't pause for 15s.
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(httpx.Response(200, text="ok"))

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # One upload, to the final key.
    assert len(fake_httpx["captured"]) == 1
    call = fake_httpx["captured"][0]
    assert call["method"] == "POST"
    expected_key = f"events/{event_id}.mp4"
    assert call["url"].endswith(f"/incident-clips/{expected_key}")
    assert call["kw"]["headers"]["Content-Type"] == "video/mp4"

    # Body is a valid MP4 (starts with ftyp box past the size prefix).
    body = call["kw"]["content"]
    assert isinstance(body, bytes)
    assert body[4:8] == b"ftyp"

    # Row updated. Clean verdict → nothing censored/suppressed.
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is True
    assert row.clip_storage_key == expected_key
    assert row.clip_censored is False
    assert row.sensitivity_categories is None
    assert row.clip_suppressed is False


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_private_camera_screens_like_any_other(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """A PRIVATE camera (room-derived privacy) GETS clips since the
    censor pipeline (2026-07-27): privacy_enabled no longer suppresses
    the upload — the clip goes through the same screen as every other
    camera. Clean verdict → normal upload, nothing stamped."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)
        cam = await s.get(Camera, camera_id)
        assert cam is not None
        cam.privacy_enabled = True
        await s.commit()

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(monkeypatch, _CLEAN)
    frame_buffer, audio_buffer = filled_buffers
    pre_frames = _build_pre_frames()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(httpx.Response(200, text="ok"))

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # The clip uploaded exactly like a Full camera's.
    assert len(fake_httpx["captured"]) == 1
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is True
    assert row.clip_storage_key == f"events/{event_id}.mp4"
    assert row.clip_censored is False
    assert row.clip_suppressed is False


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_sensitive_clip_uploads_censored_and_stamps(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """Sensitive verdict → ONLY the censored rendition is uploaded
    (person-mask mode) and the row is stamped clip_censored +
    sensitivity_categories."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(
        monkeypatch,
        ScreenResult(
            sensitive=True,
            categories=["toileting", "nudity_or_undress"],
            confidence=0.9,
        ),
    )
    censor_calls = _patch_censor(monkeypatch)
    frame_buffer, audio_buffer = filled_buffers
    pre_frames = _build_pre_frames()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(httpx.Response(200, text="ok"))

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # The uploaded body is the CENSORED bytes, not the raw concat.
    assert len(fake_httpx["captured"]) == 1
    assert fake_httpx["captured"][0]["kw"]["content"] == _CENSORED_BYTES
    # A real verdict → person-mask mode (full-frame is for no-verdict).
    assert [c["full_frame"] for c in censor_calls] == [False]

    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is True
    assert row.clip_censored is True
    assert row.sensitivity_categories == ["toileting", "nudity_or_undress"]
    assert row.clip_suppressed is False


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_screen_failure_censors_full_frame(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """VLM error → fail closed: censor FULL-FRAME, upload only that,
    stamp clip_censored with no categories (no trusted verdict)."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(monkeypatch, RuntimeError("gemini 500"))
    censor_calls = _patch_censor(monkeypatch)
    frame_buffer, audio_buffer = filled_buffers
    pre_frames = _build_pre_frames()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(httpx.Response(200, text="ok"))

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    assert len(fake_httpx["captured"]) == 1
    assert fake_httpx["captured"][0]["kw"]["content"] == _CENSORED_BYTES
    assert [c["full_frame"] for c in censor_calls] == [True]

    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is True
    assert row.clip_censored is True
    assert row.sensitivity_categories is None
    assert row.clip_suppressed is False


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_censor_failure_drops_clip(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """Sensitive clip + censor pipeline down → the clip is DROPPED
    (never uploaded raw): no storage POST, has_clip stays False,
    clip_suppressed stamped so the apps show a 'no clip' placeholder."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(
        monkeypatch,
        ScreenResult(sensitive=True, categories=["bathing_showering"], confidence=0.8),
    )
    _patch_censor(monkeypatch, raises=True)
    frame_buffer, audio_buffer = filled_buffers
    pre_frames = _build_pre_frames()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # No upload was ever attempted — raw bytes never left the box.
    assert fake_httpx["captured"] == []

    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is False
    assert row.clip_storage_key is None
    assert row.clip_suppressed is True


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_pre_only_clip_when_post_frames_missing(
    auth_user_factory, monkeypatch,
):
    """Post-roll unavailable (empty buffer / stale window) but the
    pinned pre-roll exists → a pre-only clip is shipped (2026-07-28:
    a fall clip without aftermath beats no clip). Screened like any
    other clip."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(monkeypatch, _CLEAN)
    # Empty buffers (no frames pushed) — only the pinned pre-roll exists.
    frame_buffer = FrameBuffer(window_seconds=15.0, target_fps=5)
    audio_buffer = AudioBuffer(window_seconds=15.0)
    pre_frames = _build_pre_frames()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(httpx.Response(200, text="ok"))
    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # The pre-only clip uploads and the row gets its clip.
    assert len(fake_httpx["captured"]) == 1
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is True


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_noops_when_no_frames_at_all(
    auth_user_factory, monkeypatch,
):
    """No post-roll AND no pre-roll → nothing to ship; no upload, no
    crash, events row stays clipless (alert text still reaches the
    caregiver)."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    frame_buffer = FrameBuffer(window_seconds=15.0, target_fps=5)
    audio_buffer = AudioBuffer(window_seconds=15.0)
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=[],
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    assert fake_httpx["captured"] == []
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is False
    assert row.clip_storage_key is None
    # Terminal skip must stamp clip_suppressed so the apps show the
    # no-clip state instead of "Generating clip…" forever.
    assert row.clip_suppressed is True



@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_swallows_upload_failure(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """Upload errors must NOT crash the camera tick. The archiver is
    a fire-and-forget task; we log and move on."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id, camera_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    _patch_screen(monkeypatch, _CLEAN)
    frame_buffer, audio_buffer = filled_buffers
    pre_frames = _build_pre_frames()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(
        httpx.Response(503, text="boom", request=httpx.Request("POST", "https://x")),
    )

    # Must not raise.
    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        camera_id=camera_id,
        pre_frames=pre_frames,
        pre_pcm=b"",
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is False
