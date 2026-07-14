"""Unit tests for the L2 packer (detection v2 P3 T1).

Pins D3/D4 exactly: uncapped SCENE, ≤3 crop buckets chunked
severity-first, one call per (bucket, track), degraded-person merges
into scene, rapid-members-without-the-P0-lane ride the wide call,
needs_audio propagates to the call.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.services.detection.prompt import DetectorSpec
from src.services.detection.scheduling.dims import EffectiveDims
from src.services.detection.scheduling.packer import pack_calls
from src.services.detection.scheduling.scheduler import DueDetector, DueSet


def _due(
    ident: str,
    *,
    bucket: str = "scene",
    severity: int = 50,
    targets: tuple[int, ...] | None = None,
    needs_audio: bool = False,
) -> DueDetector:
    return DueDetector(
        spec=DetectorSpec(
            identifier=ident, name=ident, prompt_fragment="frag",
            schema_fields={},
        ),
        dims=EffectiveDims(
            evidence_branch="person_motion", subject_scope="anyone",
            bucket=bucket, duty_cycle="every_eligible_tick",
            needs_audio=needs_audio, severity=severity,
            shape="point_in_time", dims_source="row",
        ),
        target_track_ids=targets,
    )


def _pack(*due: DueDetector):
    return pack_calls(
        DueSet(due=tuple(due), crowd_fallback=False, skipped=()),
        settings=SimpleNamespace(),
    )


def _idents(plan) -> list[str]:
    return [d.spec.identifier for d in plan.detectors]


def test_scene_call_is_uncapped_and_first() -> None:
    scene = [_due(f"s{i}", severity=i) for i in range(6)]
    person = _due("medication", bucket="person", targets=(1,), severity=20)
    plans = _pack(*scene, person)
    assert len(plans) == 2
    assert plans[0].crop == "full" and plans[0].fps == 4
    assert len(plans[0].detectors) == 6            # no cap on the wide call
    assert plans[1].crop == "person" and plans[1].target_track_id == 1


def test_crop_bucket_caps_at_three_severity_first() -> None:
    dets = [
        _due(f"p{i}", bucket="person", targets=(7,), severity=sev)
        for i, sev in enumerate([10, 90, 40, 70])
    ]
    plans = _pack(*dets)
    assert len(plans) == 3                           # guaranteed scene + 2 chunks
    assert _idents(plans[0]) == []                   # the always-emitted scene anchor
    assert _idents(plans[1]) == ["p1", "p3", "p2"]   # 90, 70, 40
    assert _idents(plans[2]) == ["p0"]               # 10 overflows
    assert all(p.target_track_id == 7 for p in plans[1:])


def test_one_call_per_person_never_a_union(monkeypatch=None) -> None:
    d = _due("medication", bucket="person", targets=(1, 2))
    plans = _pack(d)
    assert len(plans) == 3                            # scene anchor + 2 crop calls
    crops = [p for p in plans if p.crop == "person"]
    assert sorted(p.target_track_id for p in crops) == [1, 2]
    assert all(_idents(p) == ["medication"] for p in crops)


def test_degraded_person_merges_into_scene_call() -> None:
    plans = _pack(
        _due("fall", bucket="scene", severity=95),
        _due("medication", bucket="person", targets=None),  # no box this tick
    )
    (plan,) = plans
    assert plan.crop == "full" and plan.tier == "lite"
    assert _idents(plan) == ["fall", "medication"]


def test_degraded_fine_motion_keeps_its_own_flash_call() -> None:
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("fidget", bucket="fine_motion", targets=None),
    )
    assert len(plans) == 2
    fm = [p for p in plans if p.tier == "flash"]
    assert len(fm) == 1 and fm[0].fps == 8 and fm[0].crop == "full"


def test_rapid_member_rides_the_wide_call_when_p0_lane_off() -> None:
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("seizure", bucket="rapid", targets=(3,), severity=96),
    )
    (plan,) = plans
    assert plan.crop == "full" and plan.channel == "clean"
    assert set(_idents(plan)) == {"fall", "seizure"}


def test_snapshot_rides_the_wide_call_until_zones_exist() -> None:
    """Full-frame stills measurably FAIL object checks (spec §1.4);
    the zone-crop frame call arrives with per-camera zones in P6.
    Until then snapshot members pack into the wide video call."""
    dets = [_due(f"z{i}", bucket="snapshot", severity=i) for i in range(4)]
    plans = _pack(_due("fall", bucket="scene"), *dets)
    (plan,) = plans
    assert plan.frames == "video" and plan.crop == "full"
    assert len(plan.detectors) == 5


def test_needs_audio_propagates_to_the_call() -> None:
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("calling_for_help", bucket="scene", needs_audio=True),
        _due("medication", bucket="person", targets=(1,)),
    )
    scene = plans[0]
    assert scene.needs_audio is True                # any member ⇒ call
    person = [p for p in plans if p.crop == "person"][0]
    assert person.needs_audio is False


def test_face_bucket_recipe() -> None:
    plans = _pack(_due("grimace", bucket="face", targets=(2,)))
    (plan,) = [p for p in plans if p.crop == "face"]
    assert (plan.frames, plan.fps, plan.channel, plan.tier, plan.crop) == (
        "video", 8, "clean", "flash", "face",
    )
    assert plan.target_track_id == 2


def test_deterministic_ordering_scene_first_then_severity() -> None:
    plans = _pack(
        _due("low_face", bucket="face", targets=(1,), severity=10),
        _due("hi_person", bucket="person", targets=(1,), severity=90),
        _due("fall", bucket="scene", severity=95),
    )
    assert _idents(plans[0]) == ["fall"]
    assert _idents(plans[1]) == ["hi_person"]
    assert _idents(plans[2]) == ["low_face"]


# ─── F1/F4 review pins ───────────────────────────────────────────


def test_scene_call_is_always_emitted_even_with_nothing_due() -> None:
    """The presence machine needs the wide look every tick — an empty
    due set still packs one empty-detector scene call, exactly like
    flag-off with every detector toggled off (T3/T4 review F1)."""
    (plan,) = _pack()
    assert plan.crop == "full" and plan.fps == 4 and plan.tier == "lite"
    assert plan.detectors == ()
    assert plan.needs_audio is False


def test_crop_only_loadout_still_gets_the_scene_anchor() -> None:
    plans = _pack(_due("medication", bucket="person", targets=(1,)))
    assert len(plans) == 2
    assert plans[0].crop == "full" and plans[0].detectors == ()
    assert plans[1].crop == "person"


def test_multi_target_scene_collapse_lists_the_member_once() -> None:
    """A rapid/snapshot member fanning out per target must not appear
    twice in the wide call's ledger (T3/T4 review F4)."""
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("seizure", bucket="rapid", targets=(1, 2), severity=96),
    )
    (plan,) = plans
    assert _idents(plan).count("seizure") == 1
