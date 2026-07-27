"""Unit truth table for the L1 scheduler (detection v2 P3 T0).

`resolve_due` is the ONE generic function evaluated per detector row —
these tests pin the whole decision surface: branch gates, the fixed
subject rule (fail-open; suppression only on verified absence),
per-person crop targets (D3), crowd hardening, duty cycles, and the
parity contract (default dials ⇒ every non-scheduled detector due on
every tick).
"""

from __future__ import annotations

from types import SimpleNamespace

from src.services.detection.prompt import DetectorSpec
from src.services.detection.scheduling.dims import EffectiveDims
from src.services.detection.scheduling.scheduler import (
    ScheduledDetector,
    resolve_due,
)
from src.services.detection.signals.tick_signals import TickSignals, TrackMotion
from src.services.reid.observations import TrackObservation
from src.services.reid.result import IdentityResult


def _settings(**over) -> SimpleNamespace:
    base = dict(
        scheduler_person_motion_floor=0.0,
        scheduler_env_motion_floor=0.0,
        scheduler_person_requires_track=False,
        scheduler_sample_every_k=1,
        scheduler_crowd_fallback_tracks=4,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _entry(
    ident: str = "fall",
    *,
    branch: str = "person_motion",
    subject: str = "patient",
    bucket: str = "scene",
    duty: str = "every_eligible_tick",
    shape: str = "point_in_time",
    when_scope: str | None = None,
    category: str = "safety",
    severity: int = 50,
) -> ScheduledDetector:
    # `branch` kept as the fixture vocabulary (tests read naturally);
    # EffectiveDims carries its collapsed form: person_motion ⇒
    # requires_motion=True, environment/person_still ⇒ False.
    return ScheduledDetector(
        spec=DetectorSpec(
            identifier=ident, name=ident, prompt_fragment="frag",
            schema_fields={},
        ),
        dims=EffectiveDims(
            requires_motion=(branch == "person_motion"),
            subject_scope=subject, bucket=bucket,
            duty_cycle=duty, needs_audio=False, severity=severity, shape=shape,
            dims_source="row",
        ),
        when_scope=when_scope,
        category=category,
    )


def _track(tid: int, label: str = "UNKNOWN") -> TrackObservation:
    return TrackObservation(
        track_id=tid, label=label, confidence=0.0,
        bbox_series=((0, (10.0, 10.0, 50.0, 90.0)),),
        face_locked=False, pre_labeled=False,
    )


def _identity(
    tracks,
    *,
    attribution_ran: bool = False,
    patient_in_clip: bool = False,
    only_other: bool = False,
) -> IdentityResult:
    return IdentityResult(
        reid_ran=tracks is not None, patient_in_clip=patient_in_clip,
        confidence=0.0, best_frame_index=None, face_count=0,
        attribution_ran=attribution_ran, tracks=tracks,
        has_only_verified_other_tracks=only_other,
        faces_enrolled=attribution_ran,
    )


def _signals(
    *,
    env: float = 0.0,
    amounts: dict[int, float] | None = None,
    frames_used: int = 12,
) -> TickSignals:
    return TickSignals(
        person_present=None, people_count=None, patient_present=None,
        track_motion=tuple(
            TrackMotion(track_id=tid, label="UNKNOWN", amount=a, churn=0.0)
            for tid, a in (amounts or {}).items()
        ),
        env_motion=env, face_valid=None, frames_used=frames_used,
    )


def _due_idents(ds) -> list[str]:
    return [d.spec.identifier for d in ds.due]


def _skip_reasons(ds) -> dict[str, str]:
    return dict(ds.skipped)


# ─── Parity contract (D2) ────────────────────────────────────────


def test_parity_defaults_everything_due_even_with_reid_down() -> None:
    entries = [
        _entry("fall"),
        _entry("stove_left_on", branch="environment", subject="environment",
               bucket="snapshot", duty="thinned"),
        _entry("medication", bucket="person"),
    ]
    ds = resolve_due(
        entries, signals=_signals(), identity=_identity(None),
        settings=_settings(), tick_index=7,
    )
    assert _due_idents(ds) == ["fall", "stove_left_on", "medication"]
    assert ds.skipped == ()
    # No boxes anywhere ⇒ every artifact is scene-level.
    assert all(d.target_track_ids is None for d in ds.due)


def test_parity_empty_room_still_runs_person_detectors() -> None:
    ds = resolve_due(
        [_entry("fall")], signals=_signals(), identity=_identity(()),
        settings=_settings(), tick_index=0,
    )
    assert _due_idents(ds) == ["fall"]


# ─── Scheduled shape: evidence still collected per tick ──────────


def test_scheduled_shape_stays_due_only_scheduled_only_duty_skips() -> None:
    """The absence runner only QUERIES analyses for `<id>_detected`
    fields — the per-tick calls are its evidence stream. A scheduled-
    shape detector must keep riding calls or absence events fire
    falsely (T3/T4 review F3)."""
    entries = [
        _entry("meal_check", shape="scheduled"),
        _entry("night_check", duty="scheduled_only"),
    ]
    ds = resolve_due(
        entries, signals=_signals(), identity=_identity(None),
        settings=_settings(), tick_index=0,
    )
    assert _due_idents(ds) == ["meal_check"]
    assert _skip_reasons(ds) == {"night_check": "scheduled_only_duty"}


# ─── Duty cycle sampling ─────────────────────────────────────────


def test_sampled_duty_runs_every_kth_tick() -> None:
    e = [_entry("agitation", duty="thinned")]
    s = _settings(scheduler_sample_every_k=3)
    results = [
        bool(resolve_due(e, signals=_signals(), identity=_identity(None),
                         settings=s, tick_index=i).due)
        for i in range(6)
    ]
    assert results == [True, False, False, True, False, False]


# ─── Branch gates ────────────────────────────────────────────────


def test_env_floor_gates_environment_branch_only() -> None:
    entries = [
        _entry("stove_left_on", branch="environment", subject="environment",
               bucket="snapshot"),
        _entry("fall"),
    ]
    ds = resolve_due(
        entries, signals=_signals(env=0.01),
        identity=_identity(None),
        settings=_settings(scheduler_env_motion_floor=0.05), tick_index=0,
    )
    assert _due_idents(ds) == ["fall"]
    assert _skip_reasons(ds)["stove_left_on"] == "env_floor"


def test_requires_track_skips_person_branch_in_verified_empty_room() -> None:
    ds = resolve_due(
        [_entry("fall")], signals=_signals(), identity=_identity(()),
        settings=_settings(scheduler_person_requires_track=True), tick_index=0,
    )
    assert ds.due == ()
    assert _skip_reasons(ds)["fall"] == "no_person"


def test_motion_floor_selects_moving_tracks_person_still_ignores_it() -> None:
    tracks = (_track(1), _track(2))
    signals = _signals(amounts={1: 0.01, 2: 0.30})
    settings = _settings(
        scheduler_person_motion_floor=0.05,
        scheduler_person_requires_track=True,
    )
    moving = resolve_due(
        [_entry("medication", bucket="person")], signals=signals,
        identity=_identity(tracks), settings=settings, tick_index=0,
    )
    (d,) = moving.due
    assert d.target_track_ids == (2,)   # only the moving track qualifies
    still = resolve_due(
        [_entry("device", branch="person_still", bucket="person",
                subject="anyone")],
        signals=signals, identity=_identity(tracks), settings=settings,
        tick_index=0,
    )
    (d2,) = still.due
    assert d2.target_track_ids == (1, 2)  # presence-only branch


# ─── Subject rule (fixed; fail-open) ─────────────────────────────


def test_patient_scoped_suppressed_only_on_verified_absence() -> None:
    tracks = (_track(1, "OTHER"), _track(2, "OTHER"))
    ds = resolve_due(
        [_entry("fall"), _entry("possible_intruder", subject="anyone")],
        signals=_signals(),
        identity=_identity(tracks, attribution_ran=True, only_other=True),
        settings=_settings(), tick_index=0,
    )
    # Patient-scoped suppressed; anyone-scoped still runs (locked
    # decision: intruder/environment run for verified others).
    assert _due_idents(ds) == ["possible_intruder"]
    assert _skip_reasons(ds)["fall"] == "verified_not_patient"


def test_no_attribution_never_suppresses_patient_scope() -> None:
    tracks = (_track(1, "OTHER"),)   # poisoned label, attribution OFF
    ds = resolve_due(
        [_entry("fall")], signals=_signals(),
        identity=_identity(tracks, attribution_ran=False, only_other=True),
        settings=_settings(), tick_index=0,
    )
    assert _due_idents(ds) == ["fall"]


def test_patient_targets_prefer_patient_track_and_drop_verified_others() -> None:
    tracks = (_track(1, "PATIENT"), _track(2, "OTHER"), _track(3))
    ds = resolve_due(
        [_entry("medication", bucket="person")], signals=_signals(),
        identity=_identity(tracks, attribution_ran=True, patient_in_clip=True),
        settings=_settings(), tick_index=0,
    )
    (d,) = ds.due
    assert d.target_track_ids == (1,)   # locked patient track only

    no_lock = (_track(2, "OTHER"), _track(3))
    ds2 = resolve_due(
        [_entry("medication", bucket="person")], signals=_signals(),
        identity=_identity(no_lock, attribution_ran=True),
        settings=_settings(), tick_index=0,
    )
    (d2,) = ds2.due
    assert d2.target_track_ids == (3,)  # UNKNOWN kept, verified OTHER dropped


def test_anyone_scope_targets_every_track_per_person() -> None:
    tracks = (_track(1), _track(2), _track(3))
    ds = resolve_due(
        [_entry("possible_intruder", subject="anyone", bucket="person")],
        signals=_signals(), identity=_identity(tracks),
        settings=_settings(), tick_index=0,
    )
    (d,) = ds.due
    assert d.target_track_ids == (1, 2, 3)   # one crop call per person (D3)


# ─── Crowd hardening (D3) ────────────────────────────────────────


def test_crowd_fallback_drops_all_crops() -> None:
    tracks = tuple(_track(i) for i in range(1, 7))   # 6 > 4
    ds = resolve_due(
        [_entry("medication", bucket="person"), _entry("fall")],
        signals=_signals(), identity=_identity(tracks),
        settings=_settings(), tick_index=0,
    )
    assert ds.crowd_fallback is True
    assert all(d.target_track_ids is None for d in ds.due)
    assert len(ds.due) == 2   # everything still runs — full-scene


def test_scene_bucket_never_gets_track_targets() -> None:
    ds = resolve_due(
        [_entry("fall", bucket="scene")], signals=_signals(),
        identity=_identity((_track(1),)), settings=_settings(), tick_index=0,
    )
    (d,) = ds.due
    assert d.target_track_ids is None


# ─── Fail-open guarantees (T0 review F1/F3) ──────────────────────


def test_tracker_outage_fails_open_even_with_requires_track() -> None:
    """THE dial-path safety invariant: tracks=None means the tracker
    crashed/timed out — never a verified empty room. requires_track
    must not silence person detectors on an outage."""
    ds = resolve_due(
        [_entry("fall")], signals=_signals(), identity=_identity(None),
        settings=_settings(scheduler_person_requires_track=True), tick_index=0,
    )
    assert _due_idents(ds) == ["fall"]


def test_unmeasurable_window_fails_open_past_raised_floors() -> None:
    """frames_used < 4 ⇒ tick_signals returned zeros, not readings.
    Raised floors must treat that as 'unknowable', not 'still' —
    otherwise a truncated window gates seizure/fall off."""
    tracks = (_track(1),)
    settings = _settings(
        scheduler_person_motion_floor=0.10,
        scheduler_env_motion_floor=0.10,
        scheduler_person_requires_track=True,
    )
    ds = resolve_due(
        [
            _entry("fall", bucket="person"),
            _entry("fire_smoke", branch="environment", subject="environment"),
        ],
        signals=_signals(frames_used=0),   # zeros are absence-of-data
        identity=_identity(tracks), settings=settings, tick_index=0,
    )
    assert _due_idents(ds) == ["fall", "fire_smoke"]
    (fall, _) = ds.due
    assert fall.target_track_ids == (1,)   # every track fails open


def test_emptied_patient_targets_fall_open_to_scene() -> None:
    """All qualifying tracks verified-OTHER but the room is NOT
    verified patient-free (an UNKNOWN could be off-frame): detector
    stays due, degraded to the full scene."""
    tracks = (_track(2, "OTHER"),)
    ds = resolve_due(
        [_entry("medication", bucket="person")], signals=_signals(),
        identity=_identity(tracks, attribution_ran=True, only_other=False),
        settings=_settings(), tick_index=0,
    )
    (d,) = ds.due
    assert d.target_track_ids is None


# ── when_scope gating (recalibration 2026-07-14) ─────────────────────


def _resolve_at(entries, hh, mm=0):
    from datetime import datetime

    return resolve_due(
        entries, signals=_signals(), identity=_identity(None),
        settings=_settings(), tick_index=0,
        local_now=datetime(2026, 7, 14, hh, mm),
    )


def test_when_scope_outside_window_skips_with_reason() -> None:
    out = _resolve_at([_entry("medication", when_scope="8 AM, 8 PM ±1H")], 17)
    assert not out.due
    assert ("medication", "outside_when_scope") in out.skipped


def test_when_scope_inside_window_is_due_and_windowed() -> None:
    out = _resolve_at([_entry("medication", when_scope="8 AM, 8 PM ±1H")], 8, 30)
    assert [d.spec.identifier for d in out.due] == ["medication"]
    assert out.due[0].windowed is True


def test_always_scope_is_due_and_not_windowed() -> None:
    out = _resolve_at([_entry("fall", when_scope="ALWAYS")], 3)
    assert out.due and out.due[0].windowed is False


def test_scheduled_shape_exempt_from_window_gate() -> None:
    # Absence rows keep their all-day per-tick collection regardless of
    # any when_scope text — their schedule_config governs timing.
    out = _resolve_at(
        [_entry("stretching", shape="scheduled", when_scope="6 PM ±1H")], 3,
    )
    assert [d.spec.identifier for d in out.due] == ["stretching"]
    assert out.due[0].windowed is False


def test_no_local_now_fails_open() -> None:
    out = resolve_due(
        [_entry("medication", when_scope="8 AM, 8 PM ±1H")],
        signals=_signals(), identity=_identity(None),
        settings=_settings(), tick_index=0,
    )
    assert [d.spec.identifier for d in out.due] == ["medication"]
    assert out.due[0].windowed is False


# ─── Cross-camera patient exclusivity (2026-07-18) ───────────────


def test_patient_elsewhere_skips_patient_scoped_only() -> None:
    entries = [
        _entry("tv_time", category="routine"),
        _entry("stove_left_on", branch="environment", subject="environment"),
        _entry("intruder", subject="anyone"),
    ]
    ds = resolve_due(
        entries, signals=_signals(), identity=_identity(None),
        settings=_settings(), tick_index=0,
        patient_confirmed_elsewhere=True,
    )
    assert _due_idents(ds) == ["stove_left_on", "intruder"]
    assert _skip_reasons(ds) == {"tv_time": "patient_elsewhere"}


def test_safety_detectors_exempt_from_exclusivity_and_floor() -> None:
    """The enacted-choking miss (2026-07-18): a misidentified face on
    another camera must NEVER suppress safety detectors, and fidget-
    scale distress must never fall under the motion floor."""
    tracks = (_track(1),)
    ds = resolve_due(
        [_entry("choking", bucket="person", severity=100)],
        signals=_signals(amounts={1: 0.005}),   # fidget-scale motion
        identity=_identity(tracks, attribution_ran=False),
        settings=_settings(
            scheduler_person_motion_floor=0.03,
            scheduler_person_requires_track=True,
        ),
        tick_index=0,
        patient_confirmed_elsewhere=True,
    )
    (d,) = ds.due
    assert d.spec.identifier == "choking"
    assert d.target_track_ids == (1,)   # floor ignored for safety


def test_patient_seen_here_overrides_elsewhere() -> None:
    ds = resolve_due(
        [_entry("fall")], signals=_signals(),
        identity=_identity((), patient_in_clip=True),
        settings=_settings(), tick_index=0,
        patient_confirmed_elsewhere=True,
    )
    assert _due_idents(ds) == ["fall"]
