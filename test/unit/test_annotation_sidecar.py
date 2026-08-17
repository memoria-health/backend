"""The sidecar's determination: when each detector started and stopped.

Not a dump of model output. Someone using this corpus should not have to
know our field-naming conventions or re-implement our firing rules to
answer "what happened in this clip".
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from src.services.annotations.sidecar import (
    ClipFacts,
    DetectorSpec,
    build_sidecar,
    sidecar_key,
)

_T0 = datetime(2026, 8, 5, 22, 0, 0, tzinfo=timezone.utc)
_NOW = datetime(2026, 8, 5, 23, 0, 0, tzinfo=timezone.utc)


def _clip(duration: float = 900.0, **kw) -> ClipFacts:
    base = dict(
        key="data/v1/cg/pt/2026-08-05/220000Z__bedroom.mp4",
        started_at=_T0, duration_seconds=duration,
        caregiver_id="cg", patient_id="pt", camera_id="cam", room="Bedroom",
    )
    base.update(kw)
    return ClipFacts(**base)


def _obs(offset: float, fields: dict, *, people=True, motion=True):
    return SimpleNamespace(
        captured_at=_T0 + timedelta(seconds=offset),
        people_present=people, motion_in_clip=motion,
        state_after="open", description="", detector_fields=fields,
    )


_FALL = DetectorSpec(
    identifier="fall", detected_field="fall_detected",
    confidence_field="fall_confidence", name="Fall", category="safety",
    alert_trigger="leading", is_sustained=False,
    threshold=0.70, reasoning_field="fall_reasoning",
)
_TV = DetectorSpec(
    identifier="tv", detected_field="tv_detected",
    confidence_field="tv_confidence", name="Watching TV", category="routine",
    alert_trigger="leading", is_sustained=True, exit_gap_seconds=300.0,
)
_STOVE = DetectorSpec(
    identifier="stove", detected_field="stove_detected",
    confidence_field="stove_confidence", name="Stove on", category="safety",
    alert_trigger="duration", is_sustained=True, exit_gap_seconds=120.0,
    duration_threshold_seconds=100.0,
)
_BED = DetectorSpec(
    identifier="outofbed", detected_field="outofbed_detected",
    confidence_field="outofbed_confidence", name="Out of bed",
    category="routine", alert_trigger="schedule",
)


def test_a_span_carries_its_start_and_stop() -> None:
    obs = [
        _obs(0, {}),
        _obs(10, {"fall_detected": True, "fall_confidence": 0.8}),
        _obs(20, {"fall_detected": True, "fall_confidence": 0.9}),
        _obs(30, {}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert len(out["detections"]) == 1
    d = out["detections"][0]
    assert (d["start_offset_seconds"], d["end_offset_seconds"]) == (10.0, 20.0)
    assert d["tick_count"] == 2
    assert d["max_confidence"] == 0.9
    assert d["open_start"] is False and d["open_end"] is False


def test_consecutive_ticks_are_one_occurrence() -> None:
    """A fall seen on three ticks is one fall, not three."""
    obs = [_obs(0, {})] + [
        _obs(t, {"fall_detected": True, "fall_confidence": 0.8})
        for t in (10, 20, 30)
    ] + [_obs(40, {})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert len(out["detections"]) == 1


def test_a_wide_gap_splits_into_two_occurrences() -> None:
    obs = [
        _obs(10, {"fall_detected": True, "fall_confidence": 0.8}),
        _obs(200, {"fall_detected": True, "fall_confidence": 0.8}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert len(out["detections"]) == 2


def test_a_state_detector_survives_a_lull() -> None:
    """A brief lull inside a Watching-TV session is not the end of it —
    the detector's own exit gap decides, not adjacency."""
    obs = [
        _obs(0, {"tv_detected": True, "tv_confidence": 0.7}),
        _obs(60, {}),          # lull, well under the 300s exit gap
        _obs(120, {"tv_detected": True, "tv_confidence": 0.7}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_TV])
    assert len(out["detections"]) == 1
    assert out["detections"][0]["observed_seconds"] == 120.0


def test_same_lull_splits_a_point_detector() -> None:
    """Identical ticks, different detector: the rules are per-detector."""
    obs = [
        _obs(0, {"fall_detected": True, "fall_confidence": 0.7}),
        _obs(120, {"fall_detected": True, "fall_confidence": 0.7}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert len(out["detections"]) == 2


def test_still_running_at_the_clip_end_is_marked_open() -> None:
    """The stitch signal: someone pacing when the clip rolls over."""
    obs = [
        _obs(0, {}),
        _obs(880, {"tv_detected": True, "tv_confidence": 0.7}),
        _obs(890, {"tv_detected": True, "tv_confidence": 0.7}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_TV])
    d = out["detections"][0]
    assert d["open_end"] is True
    assert d["open_start"] is False


def test_already_running_at_the_clip_start_is_marked_open() -> None:
    obs = [
        _obs(0, {"tv_detected": True, "tv_confidence": 0.7}),
        _obs(10, {"tv_detected": True, "tv_confidence": 0.7}),
        _obs(500, {}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_TV])
    d = out["detections"][0]
    assert d["open_start"] is True and d["open_end"] is False


def test_duration_detector_reports_crossing_its_bar() -> None:
    obs = [
        _obs(0, {}),
        *[_obs(t, {"stove_detected": True, "stove_confidence": 0.9})
          for t in range(10, 200, 10)],
        _obs(400, {}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_STOVE])
    d = out["detections"][0]
    assert d["exceeded_duration_threshold"] is True
    assert d["duration_threshold_seconds"] == 100.0


def test_absence_detector_gets_no_span() -> None:
    """There is no footage of a thing not happening."""
    obs = [_obs(10, {"outofbed_detected": True, "outofbed_confidence": 0.9})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_BED])
    assert out["detections"] == []


def test_absence_alert_says_it_has_no_footage() -> None:
    alert = SimpleNamespace(
        id=uuid4(), detector_id=uuid4(), occurred_at=_T0 + timedelta(seconds=5),
        title="Not out of bed", category="routine", narrative="",
        alert_trigger="schedule",
    )
    out = build_sidecar(_clip(), [], [alert], generated_at=_NOW)
    assert out["alerts"][0]["no_footage"] is True


def test_backward_pointing_alert_is_flagged() -> None:
    """A trailing/duration alert is about footage that may be in an
    earlier clip."""
    alert = SimpleNamespace(
        id=uuid4(), detector_id=uuid4(), occurred_at=_T0 + timedelta(seconds=5),
        title="Stove left on", category="safety", narrative="",
        alert_trigger="duration",
    )
    out = build_sidecar(_clip(), [], [alert], generated_at=_NOW)
    assert out["alerts"][0]["may_precede_clip"] is True
    assert out["alerts"][0]["no_footage"] is False


def test_a_miswired_truthy_value_is_not_a_firing() -> None:
    """Must match the alerting path exactly, or the sidecar contradicts
    the events beside it."""
    obs = [_obs(10, {"fall_detected": "true", "fall_confidence": 0.9}),
           _obs(20, {"fall_detected": 1, "fall_confidence": 0.9})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert out["detections"] == []


def test_edges_are_null_when_no_tick_is_near_the_boundary() -> None:
    """"No evidence" and "evidence of nobody" are different claims."""
    out = build_sidecar(_clip(), [_obs(400, {})], [], generated_at=_NOW)
    assert out["edges"]["people_present_at_start"] is None
    assert out["edges"]["people_present_at_end"] is None


def test_raw_ticks_are_kept_whole() -> None:
    obs = [_obs(10, {"fall_detected": True, "fall_reasoning": "on the floor"})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert out["observations"][0]["detectors"]["fall_reasoning"] == "on the floor"


def test_clip_reliability_travels() -> None:
    out = build_sidecar(_clip(timeline_exact=False, masked=True), [], [],
                        generated_at=_NOW)
    assert out["clip"]["timeline_exact"] is False
    assert out["clip"]["masked"] is True


def test_sidecar_key_sits_beside_the_clip() -> None:
    assert sidecar_key("data/v1/a/b/2026-08-05/220000Z__bedroom.mp4") == (
        "data/v1/a/b/2026-08-05/220000Z__bedroom.json"
    )


# ── evidence is not the same as an alert ────────────────────────────


def _alert(offset: float, identifier: str, **kw):
    base = dict(
        id=uuid4(), detector_id=uuid4(),
        occurred_at=_T0 + timedelta(seconds=offset),
        title="t", category="safety", narrative="n",
        identifier=identifier, alert_trigger="leading",
    )
    base.update(kw)
    return SimpleNamespace(**base)


def test_a_span_below_threshold_is_not_an_alert() -> None:
    """The model saw something we would never have told a caregiver
    about. Recording it is useful; reading it as an alert is wrong."""
    obs = [_obs(10, {"fall_detected": True, "fall_confidence": 0.3})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    d = out["detections"][0]
    assert d["crossed_threshold"] is False
    assert d["promoted"] is False
    assert d["alert_event_ids"] == []


def test_a_span_that_did_alert_links_to_it() -> None:
    obs = [_obs(10, {"fall_detected": True, "fall_confidence": 0.95})]
    alert = _alert(12, "fall")
    out = build_sidecar(_clip(), obs, [alert], generated_at=_NOW,
                        detector_specs=[_FALL])
    d = out["detections"][0]
    assert d["crossed_threshold"] is True
    assert d["promoted"] is True
    assert d["alert_event_ids"] == [str(alert.id)]


def test_above_threshold_but_not_promoted_is_visible() -> None:
    """Cleared the bar and still no alert -- k-of-N unfilled, patient
    unconfirmed, mid-episode, or it lost the one-per-clip pick. That gap
    is the most instructive thing in the file."""
    obs = [_obs(10, {"fall_detected": True, "fall_confidence": 0.95})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    d = out["detections"][0]
    assert d["crossed_threshold"] is True
    assert d["promoted"] is False


def test_justification_comes_from_the_most_confident_tick() -> None:
    obs = [
        _obs(10, {"fall_detected": True, "fall_confidence": 0.4,
                  "fall_reasoning": "maybe knelt down"}),
        _obs(20, {"fall_detected": True, "fall_confidence": 0.95,
                  "fall_reasoning": "lying flat on the floor"}),
    ]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert out["detections"][0]["justification"] == "lying flat on the floor"


def test_operator_rejection_is_carried_as_a_negative_label() -> None:
    """A human calling it a false positive with the footage in front of
    them is the best negative label this system produces."""
    alert = _alert(10, "fall", gate_state="rejected")
    out = build_sidecar(_clip(), [], [alert], generated_at=_NOW)
    assert out["alerts"][0]["gate_state"] == "rejected"
    assert out["alerts"][0]["operator_false_positive"] is True


def test_auto_release_is_not_a_judgement() -> None:
    """`auto` means the timer released it with nobody looking. Reading
    that as 'a human approved it' would poison the labels."""
    alert = _alert(10, "fall", gate_state="auto")
    out = build_sidecar(_clip(), [], [alert], generated_at=_NOW)
    assert out["alerts"][0]["operator_false_positive"] is False
    assert out["alerts"][0]["gate_state"] == "auto"


def test_duration_on_an_open_start_is_unknowable() -> None:
    """The activity may have been running long before the clip began, so
    the visible span is a floor. Saying False would be a claim we cannot
    support."""
    obs = [_obs(0, {"stove_detected": True, "stove_confidence": 0.9}),
           _obs(10, {"stove_detected": True, "stove_confidence": 0.9})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_STOVE])
    d = out["detections"][0]
    assert d["open_start"] is True
    assert d["exceeded_duration_threshold"] is None


def test_no_private_fields_leak_into_the_sidecar() -> None:
    obs = [_obs(10, {"fall_detected": True, "fall_confidence": 0.9})]
    out = build_sidecar(_clip(), obs, [], generated_at=_NOW,
                        detector_specs=[_FALL])
    assert not [k for k in out["detections"][0] if k.startswith("_")]
