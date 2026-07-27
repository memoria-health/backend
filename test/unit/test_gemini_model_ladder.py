"""Gemini model ladder — primary 3.6-flash, fallback 3.1-flash-lite.

Aryaman 2026-07-27: a primary-model failure (transport, empty
response, schema garbage) must degrade QUALITY (cheaper model), never
AVAILABILITY (no verdict). One retry, then raise.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.services.detection.providers import gemini as gp


def _settings(**over):
    base = dict(
        gemini_api_key="k", gemini_model="gemini-3.6-flash",
        gemini_fallback_model="gemini-3.1-flash-lite",
        gemini_media_resolution="", gemini_video_fps="4",
        motion_overlay_enabled=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _Resp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None


def _client_with(models_called, behavior):
    class _Models:
        async def generate_content(self, *, model, contents, config):
            models_called.append(model)
            return behavior(model)

    return SimpleNamespace(aio=SimpleNamespace(models=_Models()))


@pytest.mark.asyncio
async def test_primary_failure_falls_back_once():
    called = []

    def behavior(model):
        if model == "gemini-3.6-flash":
            raise RuntimeError("primary down")
        return _Resp(
            '{"description": "ok", "people_present": false,'
            ' "people_count": 0, "speech_heard": false,'
            ' "speech_transcript": ""}'
        )

    with patch.object(gp.genai, "Client",
                      return_value=_client_with(called, behavior)):
        res = await gp.gemini_provider(
            mp4=b"x", settings=_settings(),
            detectors=[], log_tag="t",
        )
    assert called == ["gemini-3.6-flash", "gemini-3.1-flash-lite"]
    assert res is not None


@pytest.mark.asyncio
async def test_both_models_failing_raises():
    called = []

    def behavior(model):
        raise RuntimeError("all down")

    with patch.object(gp.genai, "Client",
                      return_value=_client_with(called, behavior)):
        with pytest.raises(RuntimeError):
            await gp.gemini_provider(
                mp4=b"x", settings=_settings(),
                detectors=[], log_tag="t",
            )
    assert called == ["gemini-3.6-flash", "gemini-3.1-flash-lite"]


@pytest.mark.asyncio
async def test_no_fallback_configured_single_attempt():
    called = []

    def behavior(model):
        raise RuntimeError("down")

    with patch.object(gp.genai, "Client",
                      return_value=_client_with(called, behavior)):
        with pytest.raises(RuntimeError):
            await gp.gemini_provider(
                mp4=b"x", settings=_settings(gemini_fallback_model=""),
                detectors=[], log_tag="t",
            )
    assert called == ["gemini-3.6-flash"]
