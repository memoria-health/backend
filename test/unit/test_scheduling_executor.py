"""Unit: call-plan executor merge semantics (detection v2 P3 T3).

Scripted per-call provider results pin: scene-anchored shared block,
best-answer-wins for multi-call detectors, per-call failure isolation
(scene fatal, crop survivable), audio degradation marking, artifact
degradation fallback, and the summed token ledger.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.services.detection.scheduling.executor as ex
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


def _settings(**over) -> SimpleNamespace:
    base = dict(
        reid_inference_sample_width=480,
        livekit_worker_target_fps=4,
        detection_v2_decision_enabled=False,
    )
    base.update(over)
    return SimpleNamespace(**base)


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
               settings=None):
    if monkeypatch is not None:
        async def fake_crop(*a, **k):
            return crop
        monkeypatch.setattr(ex, "build_person_crop_mp4", fake_crop)
    return await ex.run_scheduled_calls(
        plans, scene_mp4=b"scene", window=[object()], pcm=None,
        pcm_sample_rate=None, pcm_num_channels=None,
        identity=identity or _identity(), settings=settings or _settings(),
        vlm_choice="gemini", run_main_vlm=vlm,
        flash_provider=None, log_tag="t",
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
