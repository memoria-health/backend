"""Unit tests for the RAPID seizure call path (detection v2, P0).

The module under test is deliberately provider-injected: these tests drive
it with a fake provider and assert the wiring — detector splitting, the
churn pre-gate short-circuit, settings override, and field filtering —
without any network or model.
"""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.prompt import DetectorSpec
from src.services.detection.rapid_call import (
    RAPID_SEIZURE_PROMPT,
    filter_rapid_fields,
    run_rapid_seizure,
    split_rapid,
)
from src.settings import get_settings

W, H = 160, 120


def _i420(y: np.ndarray) -> bytes:
    h, w = y.shape
    uv = np.full((h // 2) * (w // 2) * 2, 128, dtype=np.uint8)
    return y.tobytes() + uv.tobytes()


def _frames(n: int = 12, *, flicker: bool) -> list[FrameSnapshot]:
    out = []
    for i in range(n):
        y = np.full((H, W), 60, dtype=np.uint8)
        if flicker:
            y[40:70, 60:90] = 190 if i % 2 else 60
        out.append(FrameSnapshot(
            width=W, height=H, data=_i420(y), captured_at=100.0 + i / 8))
    return out


def _spec(ident: str) -> DetectorSpec:
    return DetectorSpec(
        identifier=ident, name=ident, prompt_fragment="x",
        schema_fields={f"{ident}_detected": {"type": "boolean"}},
    )


class _FakeProvider:
    def __init__(self, fields: dict, usage: dict | None = None):
        self.fields = fields
        self.usage = usage
        self.calls: list[dict] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(detector_fields=dict(self.fields), usage=self.usage)


def test_split_rapid_flag_off_is_identity() -> None:
    s = get_settings().model_copy(update={"rapid_seizure_enabled": False})
    dets = [_spec("fall"), _spec("seizure")]
    main, active = split_rapid(dets, s)
    assert main == dets
    assert active is False


def test_split_rapid_extracts_seizure_when_enabled() -> None:
    s = get_settings().model_copy(update={"rapid_seizure_enabled": True})
    main, active = split_rapid([_spec("fall"), _spec("seizure")], s)
    assert [d.identifier for d in main] == ["fall"]
    assert active is True
    # No seizure assigned → nothing to extract, path stays off.
    main2, active2 = split_rapid([_spec("fall")], s)
    assert [d.identifier for d in main2] == ["fall"]
    assert active2 is False


def test_filter_rapid_fields_keeps_only_namespaced_keys() -> None:
    fields = {
        "seizure_detected": True,
        "seizure_confidence": 0.9,
        "seizure_reasoning": "rhythmic",
        "rapid_motion_description": "whole-body shaking",
        "people_present": True,          # must NOT leak over the main call's
        "speech_transcript": "hi",       # must NOT leak
        "description": "scene",          # must NOT leak
    }
    kept = filter_rapid_fields(fields)
    assert set(kept) == {
        "seizure_detected", "seizure_confidence", "seizure_reasoning",
        "rapid_motion_description",
    }


def test_prompt_carries_the_load_bearing_constraints() -> None:
    p = RAPID_SEIZURE_PROMPT.lower()
    assert "never sufficient" in p       # red tint alone can't fire it
    assert "shivering" in p              # the mutual confounder is named
    assert "any position" in p           # not predicated on lying down


async def test_gate_closed_short_circuits_before_any_call() -> None:
    """A still window never reaches artifacts or the provider — this is the
    cost lever: most motion ticks exit here for free."""
    s = get_settings().model_copy(update={"rapid_seizure_enabled": True})
    fake = _FakeProvider({"seizure_detected": True})
    out = await run_rapid_seizure(
        frames=_frames(flicker=False), per_person_boxes=[[(40, 30, 100, 80)]],
        settings=s, provider=fake, log_tag="t",
    )
    assert out is None
    assert fake.calls == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
async def test_fires_end_to_end_with_flash_settings() -> None:
    s = get_settings().model_copy(update={"rapid_seizure_enabled": True})
    fake = _FakeProvider({
        "seizure_detected": True, "seizure_confidence": 0.88,
        "seizure_reasoning": "sustained rhythmic shaking",
        "rapid_motion_description": "arms oscillating",
        "people_present": True,
    }, usage={"input": 20900, "output": 310})
    out = await run_rapid_seizure(
        frames=_frames(flicker=True), per_person_boxes=[[(40, 20, 110, 100)]],
        settings=s, provider=fake, log_tag="t",
    )
    assert out is not None
    assert out["seizure_detected"] is True
    assert "people_present" not in out
    assert out["rapid_input_tokens"] == 20900   # ledger data rides along
    assert out["rapid_output_tokens"] == 310
    assert len(fake.calls) == 1
    call = fake.calls[0]
    # The call runs on the RAPID recipe: flash-tier model, native fps,
    # overlay note on, exactly [motion-description, seizure] specs.
    assert call["settings"].gemini_model == s.rapid_model
    assert call["settings"].gemini_video_fps == str(s.rapid_fps)
    assert call["settings"].motion_overlay_enabled is True
    assert [d.identifier for d in call["detectors"]] == ["rapid_motion", "seizure"]
    assert call["mp4"][4:8] == b"ftyp"
