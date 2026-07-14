"""Unit: call-plan executor merge semantics (detection v2 P3 T3).

Scripted per-call provider results pin: scene-anchored shared block,
best-answer-wins for multi-call detectors, per-call failure isolation
(scene fatal, crop survivable), audio degradation marking, artifact
degradation fallback, and the summed token ledger.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import src.services.detection.scheduling.executor as ex
from src.services.detection.frame_buffer import FrameSnapshot
from src.services.detection.providers.types import ProviderResult
from src.services.detection.prompt import DetectorSpec
from src.services.detection.scheduling.dims import EffectiveDims
from src.services.detection.scheduling.packer import pack_calls
from src.services.detection.scheduling.scheduler import DueDetector, DueSet
from src.services.reid.body_pipeline import TrackObservation
from src.services.reid.inference import IdentityResult


def _due(
    ident: str, *, bucket: str = "scene", severity: int = 50,
    targets: tuple[int, ...] | None = None, needs_audio: bool = False,
    confirm: tuple[int, int] = (0, 0),
) -> DueDetector:
    return DueDetector(
        spec=DetectorSpec(
            identifier=ident, name=ident, prompt_fragment="frag",
            schema_fields={f"{ident}_detected": {"type": "boolean"}},
        ),
        dims=EffectiveDims(
            evidence_branch="person_motion", subject_scope="anyone",
            bucket=bucket, duty_cycle="every_eligible_tick",
            needs_audio=needs_audio, severity=severity,
            shape="point_in_time", dims_source="row",
            confirm_k=confirm[0], confirm_n=confirm[1],
        ),
        target_track_ids=targets,
    )


def _plans(*due: DueDetector):
    return pack_calls(
        DueSet(due=tuple(due), crowd_fallback=False, skipped=()),
        settings=SimpleNamespace(),
    )


def _identity(track_ids=()) -> IdentityResult:
    return IdentityResult(
        reid_ran=True, patient_in_clip=False, confidence=0.0,
        best_frame_index=None, face_count=0,
        tracks=tuple(
            TrackObservation(
                track_id=t, label="UNKNOWN", confidence=0.0,
                bbox_series=((0, (10.0, 10.0, 50.0, 90.0)),),
                face_locked=False, pre_labeled=False,
            ) for t in track_ids
        ),
    )


class _NS(SimpleNamespace):
    """Settings stub that supports the executor's model_copy(update=…)
    (used by the flash/overlay paths)."""

    def model_copy(self, update=None):
        d = dict(self.__dict__)
        d.update(update or {})
        return _NS(**d)


def _settings(**over) -> SimpleNamespace:
    base = dict(
        reid_inference_sample_width=480,
        livekit_worker_target_fps=4,
        detection_v2_decision_enabled=False,
        rapid_model="flash-model",
    )
    base.update(over)
    return _NS(**base)


class _ScriptedVLM:
    """run_vlm stand-in: pops one scripted (provider, fields) per call."""

    def __init__(self, script) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    async def __call__(self, **kw):
        self.calls.append(kw)
        provider, fields, usage = self.script.pop(0)
        if fields is None:
            raise RuntimeError("scripted provider failure")
        return provider, ProviderResult(
            description=f"desc-{len(self.calls)}",
            detector_fields=dict(fields),
            people_present=bool(fields.get("people_present", True)),
            people_count=int(fields.get("people_count", 1)),
            usage=usage,
        )


async def _run(plans, vlm, *, identity=None, monkeypatch=None, crop=b"crop",
               settings=None, flash=None, window=None,
               native_window=None, native_fps=None):
    if monkeypatch is not None:
        async def fake_crop(*a, **k):
            return crop
        monkeypatch.setattr(ex, "build_person_crop_mp4", fake_crop)
    return await ex.run_scheduled_calls(
        plans, scene_mp4=b"scene", window=window or [object()], pcm=None,
        pcm_sample_rate=None, pcm_num_channels=None,
        native_window=native_window, native_fps=native_fps,
        identity=identity or _identity(), settings=settings or _settings(),
        vlm_choice="gemini", run_main_vlm=vlm,
        flash_provider=flash, log_tag="t",
    )


@pytest.mark.asyncio
async def test_scene_anchors_shared_block_and_ledger_sums(monkeypatch) -> None:
    plans = _plans(
        _due("fall", severity=95),
        _due("medication", bucket="person", targets=(1,)),
    )
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": False, "people_present": True},
         {"input": 1000, "output": 50}),
        ("gemini", {"medication_detected": True, "people_present": False},
         {"input": 300, "output": 20}),
    ])
    out = await _run(plans, vlm, identity=_identity((1,)), monkeypatch=monkeypatch)
    assert out.description == "desc-1"                # scene call's
    assert out.people_present is True                 # never the crop's
    assert out.detector_fields["medication_detected"] is True
    assert out.detector_fields["people_present"] is True
    assert out.sched_fields["vlm_input_tokens"] == 1300
    assert out.sched_fields["vlm_output_tokens"] == 70
    assert len(out.sched_fields["sched_calls"]) == 2
    # Crop call received the crop artifact, not the scene mp4.
    assert vlm.calls[1]["mp4"] == b"crop"


@pytest.mark.asyncio
async def test_multi_call_detector_keeps_best_answer(monkeypatch) -> None:
    plans = _plans(
        _due("fall"),
        _due("medication", bucket="person", targets=(1, 2)),
    )
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": False}, None),
        ("gemini", {"medication_detected": False, "medication_confidence": 0.2}, None),
        ("gemini", {"medication_detected": True, "medication_confidence": 0.8}, None),
    ])
    out = await _run(plans, vlm, identity=_identity((1, 2)), monkeypatch=monkeypatch)
    assert out.detector_fields["medication_detected"] is True
    assert out.detector_fields["medication_confidence"] == 0.8


@pytest.mark.asyncio
async def test_crop_call_failure_survives_scene_failure_fatal(monkeypatch) -> None:
    plans = _plans(_due("fall"), _due("medication", bucket="person", targets=(1,)))
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": True}, None),
        ("gemini", None, None),                       # crop call dies
    ])
    out = await _run(plans, vlm, identity=_identity((1,)), monkeypatch=monkeypatch)
    assert out.detector_fields["fall_detected"] is True
    assert out.sched_fields["sched_call_errors"]      # recorded, not fatal

    vlm2 = _ScriptedVLM([("gemini", None, None)])     # scene call dies
    with pytest.raises(RuntimeError):
        await _run(_plans(_due("fall")), vlm2, monkeypatch=monkeypatch)


@pytest.mark.asyncio
async def test_missing_crop_degrades_to_scene_artifact(monkeypatch) -> None:
    plans = _plans(_due("fall"), _due("medication", bucket="person", targets=(9,)))
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": False}, None),
        ("gemini", {"medication_detected": False}, None),
    ])
    # Track 9 doesn't exist in identity ⇒ builder can't run ⇒ the call
    # rides the scene mp4 and the tick is marked, detectors kept.
    out = await _run(plans, vlm, identity=_identity((1,)), monkeypatch=monkeypatch)
    assert vlm.calls[1]["mp4"] == b"scene"
    assert out.sched_fields["sched_degraded_artifacts"] == 1
    assert "medication_detected" in out.detector_fields


@pytest.mark.asyncio
async def test_needs_audio_on_frame_sampling_provider_marks_degraded(monkeypatch) -> None:
    plans = _plans(_due("calling_for_help", needs_audio=True))
    vlm = _ScriptedVLM([
        ("openrouter", {"calling_for_help_detected": False}, None),
    ])
    out = await _run(plans, vlm, monkeypatch=monkeypatch)
    assert out.sched_fields["sched_degraded_audio"] is True


# ─── k-of-N confirmation (P4) ────────────────────────────────────


@pytest.mark.asyncio
async def test_confirm_suppresses_on_failed_vote(monkeypatch) -> None:
    """fall fires, confirm k=2/n=2, votes yes/no ⇒ suppressed before
    promotion; votes + spend land in sched_confirm."""
    plans = _plans(_due("fall", confirm=(2, 2)))
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": True, "fall_confidence": 0.9},
         {"input": 100, "output": 10}),                       # main call
        ("gemini", {"fall_detected": True}, {"input": 50, "output": 5}),   # vote 1
        ("gemini", {"fall_detected": False}, {"input": 50, "output": 5}),  # vote 2
    ])
    out = await _run(
        plans, vlm, monkeypatch=monkeypatch,
        settings=_settings(detection_v2_decision_enabled=True),
    )
    assert out.detector_fields["fall_detected"] is False
    assert out.detector_fields["fall_confirm_suppressed"] is True
    c = out.sched_fields["sched_confirm"]["fall"]
    assert (c["votes"], c["k"], c["n"], c["passed"]) == (1, 2, 2, False)
    assert out.sched_fields["vlm_input_tokens"] == 200   # main + 2 votes
    assert len(vlm.calls) == 3
    # Re-checks are SOLO calls on the same artifact.
    assert [d.identifier for d in vlm.calls[1]["detectors"]] == ["fall"]
    assert vlm.calls[1]["mp4"] == b"scene"


@pytest.mark.asyncio
async def test_confirm_passes_and_never_runs_without_flag_or_fire(monkeypatch) -> None:
    plans = _plans(_due("fall", confirm=(2, 2)))
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": True}, None),
        ("gemini", {"fall_detected": True}, None),
        ("gemini", {"fall_detected": True}, None),
    ])
    out = await _run(
        plans, vlm, monkeypatch=monkeypatch,
        settings=_settings(detection_v2_decision_enabled=True),
    )
    assert out.detector_fields["fall_detected"] is True
    assert out.sched_fields["sched_confirm"]["fall"]["passed"] is True

    # Flag off ⇒ zero extra calls even with config present.
    vlm2 = _ScriptedVLM([("gemini", {"fall_detected": True}, None)])
    out2 = await _run(plans, vlm2, monkeypatch=monkeypatch)
    assert len(vlm2.calls) == 1
    assert "sched_confirm" not in out2.sched_fields

    # Flag on but the detector did NOT fire ⇒ no votes spent.
    vlm3 = _ScriptedVLM([("gemini", {"fall_detected": False}, None)])
    out3 = await _run(
        plans, vlm3, monkeypatch=monkeypatch,
        settings=_settings(detection_v2_decision_enabled=True),
    )
    assert len(vlm3.calls) == 1
    assert "sched_confirm" not in out3.sched_fields


# ─── overlay / RAPID as a regular bucket call (Ledger row 5) ──────


def _i420_frames(n: int = 12, *, flicker: bool, w: int = 160, h: int = 120):
    """Synthetic native-rate window: the flicker block sits INSIDE the
    _identity() track box so the region-scoped gate sees it."""
    out = []
    for i in range(n):
        y = np.full((h, w), 60, dtype=np.uint8)
        if flicker:
            y[40:70, 12:48] = 190 if i % 2 else 60
        uv = np.full((h // 2) * (w // 2) * 2, 128, dtype=np.uint8)
        out.append(FrameSnapshot(
            width=w, height=h, data=y.tobytes() + uv.tobytes(),
            captured_at=100.0 + i / 8,
        ))
    return out


class _FlashRecorder:
    def __init__(self, fields, usage=None):
        self.fields, self.usage, self.calls = dict(fields), usage, []

    async def __call__(self, **kw):
        self.calls.append(kw)
        return ProviderResult(
            description="rapid", detector_fields=dict(self.fields),
            usage=self.usage,
        )


async def _fake_tint(frames):
    return list(frames)


async def _fake_overlay_mp4(*a, **k):
    return b"ovl"


@pytest.mark.asyncio
async def test_overlay_gate_closed_never_spends() -> None:
    """Still native frames: the rapid-energy gate short-circuits before
    tint/artifact/provider — the P0 lane's cost lever, kept verbatim —
    and the gate readouts still land on the row for soak tuning."""
    plans = _plans(_due("seizure", bucket="rapid", targets=None, severity=96))
    vlm = _ScriptedVLM([("gemini", {"people_present": True}, None)])
    flash = _FlashRecorder({"seizure_detected": True})
    out = await _run(
        plans, vlm, flash=flash,
        native_window=_i420_frames(flicker=False), native_fps=8,
    )
    assert flash.calls == []
    assert "seizure_detected" not in out.detector_fields
    gate_entries = [c for c in out.sched_fields["sched_calls"] if c.get("gate")]
    assert gate_entries and gate_entries[0]["gate"] == "closed"
    assert "rapid_gate_active_fraction" in out.detector_fields


@pytest.mark.asyncio
async def test_overlay_call_runs_the_validated_recipe(monkeypatch) -> None:
    plans = _plans(
        _due("fall"),
        _due("seizure", bucket="rapid", targets=(1,), severity=96),
    )
    vlm = _ScriptedVLM([
        ("gemini", {"fall_detected": False}, {"input": 10, "output": 1}),
    ])
    monkeypatch.setattr(ex, "tint_window", _fake_tint)
    monkeypatch.setattr(ex, "build_overlay_crop_mp4", _fake_overlay_mp4)
    flash = _FlashRecorder({
        "seizure_detected": True, "seizure_confidence": 0.9,
        "rapid_motion_description": "high-amplitude oscillation",
        "people_present": True,   # not a member key — must never merge
    }, usage={"input": 200, "output": 20})
    out = await _run(
        plans, vlm, flash=flash, identity=_identity((1,)),
        native_window=_i420_frames(flicker=True), native_fps=8,
    )
    assert len(flash.calls) == 1
    call = flash.calls[0]
    assert call["mp4"] == b"ovl"
    assert call["settings"].gemini_model == "flash-model"
    assert call["settings"].gemini_video_fps == "8"
    assert call["settings"].motion_overlay_enabled is True
    idents = [d.identifier for d in call["detectors"]]
    assert idents == ["rapid_motion", "seizure"]
    # The seizure member runs its OVERLAY prompt variant, not the row's.
    assert "MOTION-OVERLAY" in call["detectors"][1].prompt_fragment
    assert out.detector_fields["seizure_detected"] is True
    assert (
        out.detector_fields["rapid_motion_description"]
        == "high-amplitude oscillation"
    )
    assert out.detector_fields["rapid_gate_active_fraction"] > 0
    assert out.sched_fields["vlm_input_tokens"] == 210   # scene + overlay
    assert out.people_present is True                    # scene's, not rapid's


@pytest.mark.asyncio
async def test_overlay_artifact_failure_recorded_never_clean_judged(
    monkeypatch,
) -> None:
    plans = _plans(_due("seizure", bucket="rapid", targets=None, severity=96))
    vlm = _ScriptedVLM([("gemini", {}, None)])

    async def _no_mp4(*a, **k):
        return None

    monkeypatch.setattr(ex, "tint_window", _fake_tint)
    monkeypatch.setattr(ex, "build_overlay_crop_mp4", _no_mp4)
    flash = _FlashRecorder({"seizure_detected": True})
    out = await _run(
        plans, vlm, flash=flash,
        native_window=_i420_frames(flicker=True), native_fps=8,
    )
    assert flash.calls == []
    assert out.sched_fields["sched_degraded_artifacts"] == 1
    assert any(
        c.get("gate") == "artifact_failed"
        for c in out.sched_fields["sched_calls"]
    )


@pytest.mark.asyncio
async def test_overlay_without_native_window_clamps_and_marks(
    monkeypatch,
) -> None:
    """Transition tick (buffer just retargeted / restart): no native
    window yet — the overlay plan runs on the decimated window at its
    real rate and the tick is marked degraded, never skipped."""
    plans = _plans(_due("seizure", bucket="rapid", targets=None, severity=96))
    vlm = _ScriptedVLM([("gemini", {}, None)])
    monkeypatch.setattr(ex, "tint_window", _fake_tint)
    monkeypatch.setattr(ex, "build_overlay_crop_mp4", _fake_overlay_mp4)
    flash = _FlashRecorder({"seizure_detected": False})
    out = await _run(
        plans, vlm, flash=flash, window=_i420_frames(flicker=True),
    )
    assert out.sched_fields["sched_degraded_fps"] is True
    assert flash.calls
    assert flash.calls[0]["settings"].gemini_video_fps == "4"


@pytest.mark.asyncio
async def test_overlay_confirm_votes_carry_recipe(monkeypatch) -> None:
    """k-of-N votes on an overlay fire re-check the SAME artifact at the
    SAME fps with the SAME prompt variant + tint note (P4 finding 3,
    extended to overlay)."""
    plans = _plans(
        _due("seizure", bucket="rapid", targets=None, severity=96,
             confirm=(2, 2)),
    )
    vlm = _ScriptedVLM([("gemini", {}, None)])
    monkeypatch.setattr(ex, "tint_window", _fake_tint)
    monkeypatch.setattr(ex, "build_overlay_crop_mp4", _fake_overlay_mp4)
    flash = _FlashRecorder({"seizure_detected": True, "seizure_confidence": 0.9})
    out = await _run(
        plans, vlm, flash=flash,
        native_window=_i420_frames(flicker=True), native_fps=8,
        settings=_settings(detection_v2_decision_enabled=True),
    )
    assert len(flash.calls) == 3          # fire + 2 votes
    for vote in flash.calls[1:]:
        assert vote["mp4"] == b"ovl"
        assert vote["settings"].gemini_video_fps == "8"
        assert vote["settings"].motion_overlay_enabled is True
        assert [d.identifier for d in vote["detectors"]] == ["seizure"]
        assert "MOTION-OVERLAY" in vote["detectors"][0].prompt_fragment
    assert out.detector_fields["seizure_detected"] is True
    assert out.sched_fields["sched_confirm"]["seizure"]["passed"] is True
