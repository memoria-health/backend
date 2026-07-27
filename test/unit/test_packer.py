"""Unit tests for the L2 packer (detection v2 P3 T1).

Pins D3/D4 exactly: uncapped SCENE, ≤3 crop buckets chunked
severity-first, one call per (bucket, track), degraded-person merges
into scene, rapid packs as a first-class overlay/flash crop bucket
(Ledger row 5), needs_audio propagates to the call.
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
    windowed: bool = False,
) -> DueDetector:
    return DueDetector(
        spec=DetectorSpec(
            identifier=ident, name=ident, prompt_fragment="frag",
            schema_fields={},
        ),
        dims=EffectiveDims(
            requires_motion=True, subject_scope="anyone",
            bucket=bucket, duty_cycle="every_eligible_tick",
            needs_audio=needs_audio, severity=severity,
            alert_trigger="leading", dims_source="row",
        ),
        target_track_ids=targets,
        windowed=windowed,
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


def test_rapid_member_gets_its_own_overlay_flash_call() -> None:
    """Ledger row 5 (Aryaman 2026-07-14): the P0 standalone lane is
    retired — RAPID packs like any crop bucket, on its validated
    recipe, one call per person, never a clean-clip ride."""
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("seizure", bucket="rapid", targets=(3,), severity=96),
    )
    assert len(plans) == 2
    (rapid,) = [p for p in plans if p.channel == "overlay"]
    assert (rapid.frames, rapid.fps, rapid.tier, rapid.crop) == (
        "video", 8, "flash", "person",
    )
    assert rapid.target_track_id == 3
    assert _idents(rapid) == ["seizure"]
    assert "seizure" not in _idents(plans[0])
    # Two targets ⇒ two per-person overlay calls (D3: never a union).
    plans2 = _pack(_due("seizure", bucket="rapid", targets=(1, 2), severity=96))
    crops = [p for p in plans2 if p.channel == "overlay"]
    assert sorted(p.target_track_id for p in crops) == [1, 2]


def test_degraded_rapid_keeps_its_own_full_frame_overlay_call() -> None:
    """No box / crowd tick: RAPID falls back to the full tinted frame —
    the P0 lane's exact no-box behavior — never a clean-clip ride."""
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("seizure", bucket="rapid", targets=None, severity=96),
    )
    assert len(plans) == 2
    (rapid,) = [p for p in plans if p.channel == "overlay"]
    assert rapid.crop == "full" and rapid.fps == 8 and rapid.tier == "flash"
    assert rapid.target_track_id is None


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
    """A member fanning out per target whose targets all collapse into
    the SAME group must not appear twice in that call's ledger (T3/T4
    review F4). Rapid used to be this case; since Ledger row 5 it packs
    per person, so the pin rides the unknown-bucket → wide fallback."""
    plans = _pack(
        _due("fall", bucket="scene"),
        _due("mystery", bucket="not_a_bucket_yet", targets=(1, 2), severity=96),
    )
    (plan,) = plans
    assert _idents(plan).count("mystery") == 1


# ── when_scope windowed subgroup (recalibration 2026-07-14) ──────────


def test_windowed_detector_folds_into_base_call() -> None:
    # 2026-07-19 (Aryaman): 3 media classes, no extra call shapes.
    # A windowed detector rides the SAME call as always-on ones under
    # an identical recipe — when_scope eligibility is enforced
    # upstream, so reaching the packer means "run it this tick".
    plans = _pack(
        _due("fall", bucket="scene", severity=95),
        _due("medication", bucket="person", severity=20, windowed=True),
    )
    assert len(plans) == 1
    (scene,) = plans
    assert not scene.windowed
    assert _idents(scene) == ["fall", "medication"]
    assert not scene.plan_key.endswith(":win")


def test_windowed_and_always_on_share_one_crop_call() -> None:
    # The old :win split sent the identical person crop TWICE per tick
    # (~1/3 of occupied-hour cost). Folded 2026-07-19.
    plans = _pack(
        _due("medication", bucket="person", targets=(1,), windowed=True),
        _due("choking", bucket="person", targets=(1,)),
    )
    keys = [p.plan_key for p in plans if p.target_track_id == 1]
    assert keys == ["person:video4:clean:lite:track1"]
    (crop,) = [p for p in plans if p.target_track_id == 1]
    assert sorted(_idents(crop)) == ["choking", "medication"]
