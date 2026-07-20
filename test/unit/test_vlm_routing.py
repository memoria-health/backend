"""run_vlm fallback-chain order for the gemini path:

    gemini(primary model) → gemini(fallback model) → dashscope → openrouter

with IMMEDIATE failover (no retry), and the flash tier reported as the existing
``"gemini"`` provider value (the analyses.provider CHECK constraint has no
separate value for it).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.services.detection.vlm_routing as vlm_routing


def _identity() -> SimpleNamespace:
    return SimpleNamespace(
        patient_in_clip=False,
        patient_bbox=None,
        extracted_frames=[],
        per_frame_bboxes=[],
        sample_fps=4,
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(gemini_model="flash-lite", gemini_fallback_model="flash")


def _patch_all(monkeypatch, calls, *, gemini_ok_on=None) -> None:
    """Record every provider attempt. gemini records the model it received; a
    provider raises unless it's the designated gemini success model."""
    async def fake_gemini(*, model=None, **_kw):
        calls.append(("gemini", model))
        if gemini_ok_on is not None and model == gemini_ok_on:
            return "GEMINI_RESULT"
        raise RuntimeError("gemini boom")

    def _make(name):
        async def fake(**_kw):
            calls.append((name, None))
            raise RuntimeError(f"{name} boom")
        return fake

    monkeypatch.setattr(vlm_routing, "gemini_provider", fake_gemini)
    monkeypatch.setattr(vlm_routing, "dashscope_provider", _make("dashscope"))
    monkeypatch.setattr(vlm_routing, "openrouter_provider", _make("openrouter"))
    monkeypatch.setattr(vlm_routing, "openai_provider", _make("openai"))


@pytest.mark.asyncio
async def test_gemini_chain_order_on_total_failure(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    _patch_all(monkeypatch, calls)  # every provider fails
    with pytest.raises(RuntimeError, match="all VLM providers failed"):
        await vlm_routing.run_vlm(
            choice="gemini", mp4=b"", detectors=[], identity=_identity(),
            log_tag="t", settings=_settings(),
        )
    assert calls == [
        ("gemini", "flash-lite"),   # primary model
        ("gemini", "flash"),        # fallback model — SAME provider, immediate
        ("dashscope", None),        # native-video Qwen
        ("openrouter", None),       # last-resort net
    ]


@pytest.mark.asyncio
async def test_flash_tier_success_reported_as_gemini(monkeypatch) -> None:
    calls: list[tuple[str, str | None]] = []
    _patch_all(monkeypatch, calls, gemini_ok_on="flash")  # flash-lite fails, flash ok
    used, result = await vlm_routing.run_vlm(
        choice="gemini", mp4=b"", detectors=[], identity=_identity(),
        log_tag="t", settings=_settings(),
    )
    assert used == "gemini"          # existing constraint value, not a new one
    assert result == "GEMINI_RESULT"
    assert calls == [("gemini", "flash-lite"), ("gemini", "flash")]  # stops at flash
