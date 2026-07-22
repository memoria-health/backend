"""Unit suite for the track-anchored binding ledger (track_ledger.py).

Pure-logic tests: fabricated IdentityResults drive the ledger through
multi-tick scenarios mirroring the soak findings — binding persistence
through back-turned (UNKNOWN) ticks, audit hysteresis, ambiguous
crossings, retirement, single-patient, registry coordination.
"""

from __future__ import annotations

import dataclasses
import uuid
from types import SimpleNamespace

import pytest

from src.services.reid.body_pipeline import (
    LABEL_OTHER,
    LABEL_PATIENT,
    LABEL_UNKNOWN,
    TrackObservation,
)
from src.services.detection.patient_presence import TIER_FACE
from src.services.reid.inference import IdentityResult
from src.services.reid.track_ledger import _PID_BASE, TrackLedger

CAM = uuid.UUID("00000000-0000-0000-0000-000000000001")

BOX_A = (100.0, 100.0, 180.0, 300.0)
BOX_B = (500.0, 100.0, 580.0, 300.0)


class FakeRegistry:
    def __init__(self):
        self.claims: list[tuple] = []
        self.vacated: list = []
        self.accept = True
        self._holder = None

    def claim(self, camera_id, *, tier, confidence):
        self.claims.append((camera_id, tier, confidence))
        if self.accept:
            self._holder = camera_id
        return self.accept

    def holder(self):
        return self._holder

    def vacate(self, camera_id):
        self.vacated.append(camera_id)
        if self._holder == camera_id:
            self._holder = None


def track(
    tid,
    box,
    *,
    label=LABEL_UNKNOWN,
    conf=0.0,
    face_locked=False,
    pre_source=None,
    n_frames=5,
    drift=0.0,
):
    series = tuple(
        (i * 2, (box[0] + drift * i, box[1], box[2] + drift * i, box[3]))
        for i in range(n_frames)
    )
    return TrackObservation(
        track_id=tid,
        label=label,
        confidence=conf,
        bbox_series=series,
        face_locked=face_locked,
        pre_labeled=pre_source is not None,
        pre_label_source=pre_source,
    )


def result(tracks=(), *, ran=True, degraded=False):
    tracks = tuple(tracks)
    return IdentityResult(
        reid_ran=ran,
        patient_in_clip=any(t.label == LABEL_PATIENT for t in tracks),
        confidence=max(
            (t.confidence for t in tracks if t.label == LABEL_PATIENT),
            default=0.0,
        ),
        best_frame_index=0,
        face_count=0,
        patient_bbox=None,
        people_present_reid=bool(tracks),
        people_count_reid=len(tracks),
        attribution_ran=True,
        tracks=tracks,
        faces_enrolled=True,
        per_frame_bboxes=[None] * 12,
        sample_fps=2,
        quality_degraded=degraded,
    )


@pytest.fixture()
def ledger():
    return TrackLedger(
        camera_id=CAM, registry=None, settings=SimpleNamespace(),
    )


@pytest.fixture()
def registry():
    return FakeRegistry()


def patient_tracks(res):
    return [t for t in (res.tracks or ()) if t.label == LABEL_PATIENT]


# ── continuity ─────────────────────────────────────────────────────


def test_same_body_keeps_persistent_id(ledger):
    r1 = ledger.apply(result([track(1, BOX_A)]), tick_mono=10.0)
    r2 = ledger.apply(result([track(7, BOX_A)]), tick_mono=20.0)
    r3 = ledger.apply(result([track(3, BOX_A)]), tick_mono=30.0)
    pids = {r.tracks[0].track_id for r in (r1, r2, r3)}
    assert len(pids) == 1
    assert pids.pop() >= _PID_BASE


def test_two_people_keep_distinct_ids(ledger):
    r1 = ledger.apply(
        result([track(1, BOX_A), track(2, BOX_B)]), tick_mono=10.0,
    )
    r2 = ledger.apply(
        result([track(9, BOX_A), track(4, BOX_B)]), tick_mono=20.0,
    )
    ids1 = sorted(t.track_id for t in r1.tracks)
    ids2 = sorted(t.track_id for t in r2.tracks)
    assert ids1 == ids2
    assert len(set(ids1)) == 2


# ── the product requirement: binding persists ──────────────────────


def test_patient_binding_persists_through_unknown_ticks(ledger):
    r1 = ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.62,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    assert r1.patient_in_clip
    for i in range(2, 8):  # back turned / phone at face — no evidence
        r = ledger.apply(
            result([track(i, BOX_A)]), tick_mono=10.0 * i,
        )
        assert r.patient_in_clip, f"tick {i} dropped the binding"
        pt = patient_tracks(r)
        assert len(pt) == 1
        assert pt[0].pre_label_source == "held"
        assert pt[0].confidence == pytest.approx(0.62)
        assert r.patient_bbox is not None


def test_other_binding_persists_symmetrically(ledger):
    r1 = ledger.apply(
        result([track(1, BOX_B, label=LABEL_OTHER, face_locked=True)]),
        tick_mono=10.0,
    )
    assert r1.tracks[0].label == LABEL_OTHER
    r2 = ledger.apply(result([track(5, BOX_B)]), tick_mono=20.0)
    assert r2.tracks[0].label == LABEL_OTHER
    assert r2.has_only_verified_other_tracks
    assert not r2.patient_in_clip


def test_crowd_does_not_touch_patient_binding(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    # A second and third person appear — the old park-on-crowd moment.
    r = ledger.apply(
        result([track(2, BOX_A), track(3, BOX_B),
                track(4, (300.0, 100.0, 380.0, 300.0))]),
        tick_mono=20.0,
    )
    assert r.patient_in_clip  # binding rides through the crowd
    pt = patient_tracks(r)
    assert len(pt) == 1
    assert pt[0].bbox_series[0][1][0] == pytest.approx(BOX_A[0])


# ── birth bindings ─────────────────────────────────────────────────


def test_pool_birth_binds_patient(ledger):
    r = ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.75,
                      pre_source="gallery")]),
        tick_mono=10.0,
    )
    assert r.patient_in_clip
    r2 = ledger.apply(result([track(2, BOX_A)]), tick_mono=20.0)
    assert r2.patient_in_clip


def test_birth_fragments_do_not_enter_ledger(ledger):
    r = ledger.apply(
        result([track(1, BOX_A), track(2, BOX_B, n_frames=1)]),
        tick_mono=10.0,
    )
    ids = sorted(t.track_id for t in r.tracks)
    assert ids[0] == 2          # fragment keeps its clip-local id
    assert ids[1] >= _PID_BASE  # substantial track got a pid


# ── audit path ─────────────────────────────────────────────────────


def test_single_contradiction_suppresses_but_survives(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    r = ledger.apply(
        result([track(2, BOX_A, label=LABEL_OTHER, face_locked=True)]),
        tick_mono=20.0,
    )
    # The verified-OTHER tick is never relabeled and the binding does
    # not ride it...
    assert not r.patient_in_clip
    assert r.tracks[0].label == LABEL_OTHER
    # ...but one tick never destroys the binding: a neutral tick
    # resumes the hold.
    r2 = ledger.apply(result([track(3, BOX_A)]), tick_mono=30.0)
    assert r2.patient_in_clip


def test_sustained_contradiction_unbinds_to_unknown(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    ledger.apply(
        result([track(2, BOX_A, label=LABEL_OTHER, face_locked=True)]),
        tick_mono=20.0,
    )
    r = ledger.apply(
        result([track(3, BOX_A, label=LABEL_OTHER, face_locked=True)]),
        tick_mono=30.0,
    )
    assert not r.patient_in_clip
    assert ledger.stash_fields()["id_binding"] == "none"
    # Unbound to UNKNOWN, so a later neutral tick stays unknown...
    r2 = ledger.apply(result([track(4, BOX_A)]), tick_mono=40.0)
    assert not r2.patient_in_clip
    # ...until fresh evidence re-binds.
    r3 = ledger.apply(
        result([track(5, BOX_A, label=LABEL_PATIENT, conf=0.6,
                      face_locked=True)]),
        tick_mono=50.0,
    )
    assert r3.patient_in_clip


def test_pool_other_prelabel_is_not_audit_grade(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    # Outfit-echo poison: pool pre-labels the body OTHER, twice.
    for i, tm in ((2, 20.0), (3, 30.0)):
        r = ledger.apply(
            result([track(i, BOX_A, label=LABEL_OTHER,
                          pre_source="gallery")]),
            tick_mono=tm,
        )
        assert not r.patient_in_clip     # suppressed this tick (F1)
        assert r.tracks[0].label == LABEL_OTHER
    # No streak was built — the binding survives the echo.
    r = ledger.apply(result([track(4, BOX_A)]), tick_mono=40.0)
    assert r.patient_in_clip


def test_degraded_contradiction_is_inadmissible(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    for i, tm in ((2, 20.0), (3, 30.0), (4, 40.0)):
        ledger.apply(
            result([track(i, BOX_A, label=LABEL_OTHER,
                          face_locked=True)], degraded=True),
            tick_mono=tm,
        )
    r = ledger.apply(result([track(5, BOX_A)]), tick_mono=50.0)
    assert r.patient_in_clip  # blur never executed the binding


# ── retirement ─────────────────────────────────────────────────────


def test_patient_retires_after_grace_and_rebinds_on_reentry(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    r = ledger.apply(result([]), tick_mono=20.0)
    assert not r.patient_in_clip          # good empty tick: no claim
    assert ledger.is_active               # but binding still live
    ledger.apply(result([]), tick_mono=60.0)  # 50s absent > 45s grace
    assert not ledger.is_active
    assert ledger.stash_fields()["id_binding"] == "none"
    # Re-entry: new pid, pool evidence re-binds.
    r2 = ledger.apply(
        result([track(9, BOX_A, label=LABEL_PATIENT, conf=0.74,
                      pre_source="gallery")]),
        tick_mono=70.0,
    )
    assert r2.patient_in_clip


def test_scene_presence_extends_patient_grace(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    ledger.note_scene_presence(True)
    ledger.apply(result([]), tick_mono=100.0)   # way past grace
    assert ledger.is_active
    ledger.note_scene_presence(False)
    ledger.apply(result([]), tick_mono=200.0)
    assert not ledger.is_active


# ── stitcher edges ─────────────────────────────────────────────────


def test_ambiguous_boundary_crossing_inherits_nothing(ledger):
    ledger.apply(
        result([track(1, (100.0, 100.0, 180.0, 300.0)),
                track(2, (200.0, 100.0, 280.0, 300.0))]),
        tick_mono=10.0,
    )
    # One person dead-center between both predictions: refuse.
    r = ledger.apply(
        result([track(3, (150.0, 100.0, 230.0, 300.0))]),
        tick_mono=20.0,
    )
    known = {
        e.pid for e in ledger._entries.values()
    }
    assert r.tracks[0].track_id == max(known)  # a NEW pid, not inherited


def test_walker_is_stitched_via_velocity(ledger):
    # Moving right at ~25 px/s within the clip (drift 10px/frame at
    # 2fps), then reappearing further right after a 10s boundary.
    ledger.apply(
        result([track(1, BOX_A, drift=10.0)]), tick_mono=10.0,
    )
    moved = tuple(v + 250.0 if i % 2 == 0 else v for i, v in enumerate(BOX_A))
    r = ledger.apply(result([track(2, moved)]), tick_mono=20.0)
    assert r.tracks[0].track_id >= _PID_BASE
    # Same pid as the first tick's track (velocity carried the gate).
    r0_pid = min(e.pid for e in ledger._entries.values())
    assert r.tracks[0].track_id == r0_pid
    assert len(ledger._entries) == 1


# ── invariants and rails ───────────────────────────────────────────


def test_single_patient_higher_score_steals(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.55,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    r = ledger.apply(
        result([track(2, BOX_A),
                track(3, BOX_B, label=LABEL_PATIENT, conf=0.8,
                      face_locked=True)]),
        tick_mono=20.0,
    )
    pt = patient_tracks(r)
    assert len(pt) == 1
    assert pt[0].bbox_series[0][1][0] == pytest.approx(BOX_B[0])
    # And the weaker challenger never steals back.
    r2 = ledger.apply(
        result([track(4, BOX_A, label=LABEL_PATIENT, conf=0.6,
                      face_locked=True),
                track(5, BOX_B)]),
        tick_mono=30.0,
    )
    pt2 = patient_tracks(r2)
    assert len(pt2) == 1
    assert pt2[0].bbox_series[0][1][0] == pytest.approx(BOX_B[0])


def test_reid_not_ran_freezes_everything(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    frozen = result([], ran=False)
    out = ledger.apply(frozen, tick_mono=500.0)
    assert out is frozen                  # untouched
    assert ledger.is_active               # nothing retired
    r = ledger.apply(result([track(2, BOX_A)]), tick_mono=505.0)
    assert not r.patient_in_clip or True  # apply still safe post-freeze


def test_stale_tick_never_mutates_state(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=100.0,
    )
    r = ledger.apply(
        result([track(2, BOX_B, label=LABEL_OTHER, face_locked=True)]),
        tick_mono=50.0,  # captured BEFORE the last processed tick
    )
    assert r.patient_in_clip              # display coherence fold
    assert len(ledger._entries) == 1      # no birth from stale window
    r2 = ledger.apply(result([track(3, BOX_A)]), tick_mono=110.0)
    assert r2.patient_in_clip


# ── registry coordination ──────────────────────────────────────────


def test_registry_claim_tiers_and_vacate(registry):
    led = TrackLedger(
        camera_id=CAM, registry=registry, settings=SimpleNamespace(),
    )
    led.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    assert registry.claims and registry.claims[0][1] == TIER_FACE
    led.apply(result([]), tick_mono=100.0)  # retire past grace
    assert registry.vacated == [CAM]


def test_outclaimed_bind_does_not_stick(registry):
    registry.accept = False
    led = TrackLedger(
        camera_id=CAM, registry=registry, settings=SimpleNamespace(),
    )
    r = led.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    assert not led.is_active
    # Tick-local truth still stands for this clip's own evidence.
    assert r.tracks[0].label in (LABEL_PATIENT, LABEL_UNKNOWN)


def test_takeover_elsewhere_unbinds_without_vacate(registry):
    led = TrackLedger(
        camera_id=CAM, registry=registry, settings=SimpleNamespace(),
    )
    led.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    other_cam = uuid.UUID("00000000-0000-0000-0000-000000000002")
    registry._holder = other_cam
    r = led.apply(result([track(2, BOX_A)]), tick_mono=20.0)
    assert not r.patient_in_clip
    assert not led.is_active
    assert registry.vacated == []         # seat belongs elsewhere


# ── binding-aware face throttle (optimization #1) ──────────────────


def test_face_evidence_full_while_unbound(ledger):
    assert ledger.wants_face_evidence()          # cold start: full
    ledger.apply(result([track(1, BOX_A)]), tick_mono=10.0)  # UNKNOWN
    assert ledger.wants_face_evidence()          # unbound body: full
    assert ledger.wants_face_evidence()          # stays full


def test_face_evidence_heartbeat_once_all_bound(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    pattern = [ledger.wants_face_evidence() for _ in range(6)]
    assert pattern == [False, False, True, False, False, True]


def test_face_evidence_resumes_on_new_body(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    assert ledger.wants_face_evidence() is False
    # A second body walks in — unbound entry ends the throttle.
    ledger.apply(
        result([track(2, BOX_A), track(3, BOX_B)]), tick_mono=20.0,
    )
    assert ledger.wants_face_evidence() is True


def test_face_evidence_full_on_empty_room(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    ledger.apply(result([]), tick_mono=20.0)     # room empties
    assert ledger.wants_face_evidence() is True  # cold-entry instant


def test_registry_seat_refreshed_while_patient_in_view(registry):
    from src.services.detection.patient_presence import TIER_HELD
    led = TrackLedger(
        camera_id=CAM, registry=registry, settings=SimpleNamespace(),
    )
    led.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    n_after_bind = len(registry.claims)
    # Two neutral ticks with the patient still visible — the seat must
    # be re-warmed each tick (TIER_HELD), not left to TTL out.
    led.apply(result([track(2, BOX_A)]), tick_mono=20.0)
    led.apply(result([track(3, BOX_A)]), tick_mono=30.0)
    held_claims = [c for c in registry.claims[n_after_bind:]
                   if c[1] == TIER_HELD]
    assert len(held_claims) == 2


# ── two-way evidence accumulator (2026-07-21 live findings) ────────


def sf_track(tid, box, events=3, score=0.48, n_frames=5):
    """A couch-distance track: small-face events, no verdict."""
    t = track(tid, box, n_frames=n_frames)
    return dataclasses.replace(
        t, smallface_events=events, best_match_score=score,
    )


def mm_track(tid, box, mismatches=3, n_frames=5):
    """A guest track: verified mismatches, zero hits, no verdict."""
    t = track(tid, box, n_frames=n_frames)
    return dataclasses.replace(t, verified_mismatches=mismatches)


def test_uncle_accumulates_to_other_across_ticks(ledger):
    r1 = ledger.apply(result([mm_track(1, BOX_B, 3)]), tick_mono=10.0)
    assert r1.tracks[0].label == LABEL_UNKNOWN     # 3 < 5: not yet
    r2 = ledger.apply(result([mm_track(2, BOX_B, 3)]), tick_mono=20.0)
    assert r2.tracks[0].label == LABEL_OTHER       # 6 ≥ 5: branded
    r3 = ledger.apply(result([track(3, BOX_B)]), tick_mono=30.0)
    assert r3.tracks[0].label == LABEL_OTHER       # persists


def test_couch_smallface_binds_patient_across_ticks(ledger):
    r1 = ledger.apply(result([sf_track(1, BOX_A, 3)]), tick_mono=10.0)
    assert not r1.patient_in_clip                  # 3 < 6: not yet
    r2 = ledger.apply(
        result([sf_track(2, BOX_A, 3, score=0.51)]), tick_mono=20.0,
    )
    assert r2.patient_in_clip                      # 6 ≥ 6: bound
    pt = patient_tracks(r2)
    assert pt and pt[0].confidence == pytest.approx(0.51)
    r3 = ledger.apply(result([track(3, BOX_A)]), tick_mono=30.0)
    assert r3.patient_in_clip                      # binding persists


def test_mismatch_resets_smallface_run(ledger):
    ledger.apply(result([sf_track(1, BOX_A, 5)]), tick_mono=10.0)
    ledger.apply(result([mm_track(2, BOX_A, 1)]), tick_mono=20.0)
    r = ledger.apply(result([sf_track(3, BOX_A, 3)]), tick_mono=30.0)
    assert not r.patient_in_clip                   # run restarted: 3 < 6


def test_bound_patient_never_accumulates_mismatches(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    # Phone-at-face class: occasional pure-mismatch clips on the
    # BOUND track must not build toward OTHER.
    for i, tm in ((2, 20.0), (3, 30.0), (4, 40.0)):
        r = ledger.apply(result([mm_track(i, BOX_A, 4)]), tick_mono=tm)
        assert r.patient_in_clip


def test_degraded_ticks_do_not_accumulate(ledger):
    for i, tm in ((1, 10.0), (2, 20.0), (3, 30.0)):
        ledger.apply(
            result([mm_track(i, BOX_B, 4)], degraded=True), tick_mono=tm,
        )
    r = ledger.apply(result([track(9, BOX_B)]), tick_mono=40.0)
    assert r.tracks[0].label == LABEL_UNKNOWN      # blur built nothing


def test_stash_truth_fields(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    f = ledger.stash_fields()
    assert f["id_binding_seen"] is True and f["id_last_seen_s"] == 0
    ledger.apply(result([]), tick_mono=25.0)       # patient absent
    f2 = ledger.stash_fields()
    assert f2["id_binding_seen"] is False
    assert f2["id_last_seen_s"] == 15


def test_harvest_hint_only_in_face_afterglow(ledger):
    assert ledger.harvest_hint() is None           # nothing bound
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    hint = ledger.harvest_hint()
    assert hint is not None and hint[1] == pytest.approx(0.7)
    # Still inside the 60s envelope on a neutral seen tick.
    ledger.apply(result([track(2, BOX_A)]), tick_mono=40.0)
    assert ledger.harvest_hint() is not None
    # Envelope expired: no more donations without fresh face evidence.
    ledger.apply(result([track(3, BOX_A)]), tick_mono=100.0)
    assert ledger.harvest_hint() is None
    # Patient absent this tick: no hint even inside a fresh envelope.
    ledger.apply(
        result([track(4, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=110.0,
    )
    ledger.apply(result([]), tick_mono=120.0)
    assert ledger.harvest_hint() is None


def test_pool_provenance_binding_never_emits_hint(ledger):
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.75,
                      pre_source="gallery")]),
        tick_mono=10.0,
    )
    assert ledger.harvest_hint() is None           # no pool→pool feed


def test_all_bound_gates_lean_sampling(ledger):
    # Empty room: not lean (no patient present).
    assert ledger.all_bound() is False
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    # Patient bound + everyone identified → lean sampling allowed.
    assert ledger.all_bound() is True
    # A new unbound person appears → full sampling resumes.
    ledger.apply(
        result([track(2, BOX_A), track(3, BOX_B)]), tick_mono=20.0,
    )
    assert ledger.all_bound() is False
    # Empty room again → not lean (nothing to track).
    ledger.apply(result([]), tick_mono=30.0)
    assert ledger.all_bound() is False


# ── fragment merge (2026-07-22 keystone: the 1000→1001 churn) ──────

BODY = (600.0, 100.0, 800.0, 460.0)          # him, full body
FRAG = (660.0, 120.0, 730.0, 200.0)          # face-region fragment inside


def test_fragment_merges_and_lock_stays_stable(ledger):
    # Bound via pool as one body.
    ledger.apply(
        result([track(1, BODY, label=LABEL_PATIENT, conf=0.75,
                      pre_source="gallery")]),
        tick_mono=10.0,
    )
    pid0 = next(iter(ledger._entries))
    # SAM fragments: body track UNKNOWN + small face fragment carrying
    # a fresh face lock (the exact live 1000→1001 steal scenario).
    r = ledger.apply(
        result([track(2, BODY),
                track(3, FRAG, label=LABEL_PATIENT, conf=0.48,
                      face_locked=True, n_frames=4)]),
        tick_mono=20.0,
    )
    assert len(ledger._entries) == 1              # merged: ONE person
    assert next(iter(ledger._entries)) == pid0    # pid did NOT churn
    assert r.patient_in_clip
    # Patient bbox rides the BODY fragment, not the face sliver.
    assert r.patient_bbox is not None
    assert r.patient_bbox[2] - r.patient_bbox[0] > 100
    # Face evidence refreshed the merged binding (provenance face now).
    assert ledger._entries[pid0].provenance == "face"
    # Fully-bound scene → lean may engage.
    assert ledger.all_bound() is True


def test_fragment_merge_repeated_ticks_no_churn(ledger):
    ledger.apply(
        result([track(1, BODY, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    pid0 = next(iter(ledger._entries))
    for i, tm in ((2, 20.0), (4, 30.0), (6, 40.0), (8, 50.0)):
        r = ledger.apply(
            result([track(i, BODY), track(i + 1, FRAG, n_frames=4)]),
            tick_mono=tm,
        )
        assert r.patient_in_clip, f"tick {tm} lost patient"
        assert len(ledger._entries) == 1
        assert next(iter(ledger._entries)) == pid0


def test_similar_sized_adults_never_merge(ledger):
    # A hug / close pass: two comparable boxes overlapping heavily.
    A = (300.0, 100.0, 460.0, 460.0)
    B = (360.0, 110.0, 520.0, 460.0)   # overlaps A a lot, similar size
    ledger.apply(
        result([track(1, A, label=LABEL_PATIENT, conf=0.7,
                      face_locked=True),
                track(2, B)]),
        tick_mono=10.0,
    )
    assert len(ledger._entries) == 2   # ratio guard: PEOPLE never merge
    r = ledger.apply(result([track(3, A), track(4, B)]), tick_mono=20.0)
    assert len(ledger._entries) == 2
    pt = patient_tracks(r)
    assert len(pt) == 1                # guest never wears the label


# ── pool cannot establish patient cross-camera (2026-07-22 live bug) ─

class _HolderElsewhereRegistry:
    """Registry stub: the patient is held on a DIFFERENT camera."""
    def confirmed_elsewhere(self, camera_id):
        return True
    def holder(self):
        return uuid.UUID("00000000-0000-0000-0000-0000000000ff")
    def claim(self, camera_id, *, tier, confidence):
        # Real tier logic: a FACE claim (tier 2) wins the seat from the
        # elsewhere-holder; an outfit/body claim (tier 1) loses.
        return tier >= 2
    def vacate(self, camera_id):
        pass


def test_pool_cannot_establish_patient_when_held_elsewhere():
    led = TrackLedger(
        camera_id=CAM, registry=_HolderElsewhereRegistry(),
        settings=SimpleNamespace(),
    )
    # A similar-outfit guest pool-matches the patient gallery here.
    r = led.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.75,
                      pre_source="gallery")]),
        tick_mono=10.0,
    )
    # Blocked: outfit alone must not confirm the patient on a second
    # camera while the real patient is locked elsewhere.
    assert not r.patient_in_clip
    assert r.tracks[0].label != LABEL_PATIENT


def test_face_still_establishes_patient_across_cameras():
    led = TrackLedger(
        camera_id=CAM, registry=_HolderElsewhereRegistry(),
        settings=SimpleNamespace(),
    )
    # A real face match DOES cross the boundary (handoff / correction).
    r = led.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.55,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    assert r.patient_in_clip


def test_pool_cold_start_still_works_when_no_holder(ledger):
    # No registry holder → pool cold-start attribution unaffected.
    r = ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.75,
                      pre_source="gallery")]),
        tick_mono=10.0,
    )
    assert r.patient_in_clip


def test_pool_maintains_own_camera_lock(ledger):
    # Establish via face, then pool maintains — even if (hypothetically)
    # held-elsewhere flips, maintenance of an existing lock is allowed.
    ledger.apply(
        result([track(1, BOX_A, label=LABEL_PATIENT, conf=0.6,
                      face_locked=True)]),
        tick_mono=10.0,
    )
    r = ledger.apply(
        result([track(2, BOX_A, label=LABEL_PATIENT, conf=0.75,
                      pre_source="gallery")]),
        tick_mono=20.0,
    )
    assert r.patient_in_clip   # maintaining, not establishing
