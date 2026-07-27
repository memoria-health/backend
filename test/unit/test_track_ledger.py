"""Unit suite for the bind-once-hold ledger (track_ledger.py).

The EdgeTAM-continuous pod hands the backend STABLE track ids across
ticks, so the ledger keys entries by the pod id and never stitches. Two
labels only: PATIENT or UNKNOWN. Tests drive fabricated IdentityResults
through multi-tick scenarios: stable-id hold, hold-through-distance (the
northstar), re-entry rebind, the single-patient FP guard, cross-camera
single-holder, retirement + bounded scene-extension.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from src.services.identity.observations import (
    LABEL_OTHER,
    LABEL_PATIENT,
    LABEL_UNKNOWN,
    TrackObservation,
)
from src.services.detection.patient_presence import TIER_FACE
from src.services.identity.result import IdentityResult
from src.services.identity.track_ledger import TrackLedger

CAM = uuid.UUID("00000000-0000-0000-0000-000000000001")
OTHER_CAM = uuid.UUID("00000000-0000-0000-0000-0000000000ff")

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

    def confirmed_elsewhere(self, camera_id):
        return self._holder is not None and self._holder != camera_id


def track(
    tid, box, *, label=LABEL_UNKNOWN, conf=0.0, face_locked=False,
    pre_source=None, n_frames=5,
):
    series = tuple((i * 2, box) for i in range(n_frames))
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


def face(tid, box, *, conf=0.55):
    return track(tid, box, label=LABEL_PATIENT, conf=conf, face_locked=True)


def pool(tid, box, *, conf=0.75):
    return track(tid, box, label=LABEL_PATIENT, conf=conf, pre_source="gallery")


@pytest.fixture()
def ledger():
    return TrackLedger(camera_id=CAM, registry=None, settings=SimpleNamespace())


@pytest.fixture()
def registry():
    return FakeRegistry()


def label_of(res, tid):
    return next(t.label for t in res.tracks if t.track_id == tid)


# ── stable-id continuity (pod does the work now) ────────────────────────

def test_same_pod_id_is_one_entry(ledger):
    for i, ts in enumerate((10.0, 20.0, 30.0)):
        ledger.apply(result([track(5, BOX_A)]), tick_mono=ts)
    assert len(ledger._entries) == 1
    assert 5 in ledger._entries          # keyed by the pod id, no 1000+ space


def test_two_pod_ids_two_entries(ledger):
    ledger.apply(result([track(1, BOX_A), track(2, BOX_B)]), tick_mono=10.0)
    assert set(ledger._entries) == {1, 2}


# ── bind + HOLD (the northstar) ─────────────────────────────────────────

def test_face_lock_binds_patient(ledger):
    r = ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    assert label_of(r, 1) == LABEL_PATIENT
    assert ledger.is_active


def test_pool_match_binds_patient(ledger):
    r = ledger.apply(result([pool(1, BOX_A)]), tick_mono=10.0)
    assert label_of(r, 1) == LABEL_PATIENT


def test_patient_held_through_unknown_ticks(ledger):
    # Lock at entry, then the SAME pod id keeps arriving UNKNOWN (face too
    # small at the couch). The binding must HOLD — the whole northstar.
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    for ts in (20.0, 30.0, 40.0, 50.0):
        r = ledger.apply(result([track(1, BOX_A)]), tick_mono=ts)
        assert label_of(r, 1) == LABEL_PATIENT, f"unbound at {ts}"
    assert ledger.is_active


def test_no_mid_track_audit_other_never_unbinds(ledger):
    # A verified-OTHER clip on the bound patient's own track (poison /
    # distance mis-read) must NOT unbind — there is no audit anymore.
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    r = ledger.apply(
        result([track(1, BOX_A, label=LABEL_OTHER)]), tick_mono=20.0,
    )
    assert label_of(r, 1) == LABEL_PATIENT


# ── no OTHER ────────────────────────────────────────────────────────────

def test_other_label_collapses_to_unknown(ledger):
    r = ledger.apply(result([track(9, BOX_A, label=LABEL_OTHER)]), tick_mono=10.0)
    assert label_of(r, 9) == LABEL_UNKNOWN
    assert ledger._entries[9].binding == LABEL_UNKNOWN


def test_guest_stays_unknown_never_other(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    r = ledger.apply(
        result([face(1, BOX_A), track(2, BOX_B, label=LABEL_OTHER)]),
        tick_mono=20.0,
    )
    assert label_of(r, 1) == LABEL_PATIENT
    assert label_of(r, 2) == LABEL_UNKNOWN
    assert r.has_only_verified_other_tracks is False


# ── single-patient FP guard ─────────────────────────────────────────────

def test_face_outranks_pool_for_patient_seat(ledger):
    # A strong pool (0.85) must NOT steal the seat from a weaker face
    # (0.50) — face categorically outranks pool.
    ledger.apply(result([face(1, BOX_A, conf=0.50)]), tick_mono=10.0)
    r = ledger.apply(
        result([face(1, BOX_A, conf=0.50), pool(2, BOX_B, conf=0.85)]),
        tick_mono=20.0,
    )
    assert label_of(r, 1) == LABEL_PATIENT
    assert label_of(r, 2) == LABEL_UNKNOWN


def test_higher_face_score_takes_seat_loser_unknown(ledger):
    ledger.apply(result([face(1, BOX_A, conf=0.50)]), tick_mono=10.0)
    r = ledger.apply(
        result([face(1, BOX_A, conf=0.50), face(2, BOX_B, conf=0.70)]),
        tick_mono=20.0,
    )
    assert label_of(r, 2) == LABEL_PATIENT
    assert label_of(r, 1) == LABEL_UNKNOWN   # loser -> UNKNOWN, never OTHER


def test_pool_cannot_establish_when_confirmed_elsewhere(registry):
    registry._holder = OTHER_CAM             # patient confirmed on another cam
    led = TrackLedger(camera_id=CAM, registry=registry, settings=SimpleNamespace())
    r = led.apply(result([pool(1, BOX_A)]), tick_mono=10.0)
    assert label_of(r, 1) == LABEL_UNKNOWN   # outfit alone can't cross cameras


def test_face_still_establishes_when_confirmed_elsewhere(registry):
    registry._holder = OTHER_CAM
    led = TrackLedger(camera_id=CAM, registry=registry, settings=SimpleNamespace())
    r = led.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    assert label_of(r, 1) == LABEL_PATIENT   # face crosses the boundary


def test_face_reentry_beats_stale_scene_extended_phantom(ledger):
    # Patient face-binds, leaves; a lingering scene keeps the OLD id
    # scene-extended (a phantom holding the seat at its peak 0.62). The
    # patient returns as a NEW pod id and face-locks LOWER (0.55). The
    # physically-present, face-verified return must win over the stale
    # phantom (review Finding 3).
    ledger.apply(result([face(1, BOX_A, conf=0.62)]), tick_mono=10.0)
    ledger.note_scene_presence(True)
    ledger.apply(result([]), tick_mono=20.0)          # id1 scene-extends
    assert ledger.is_active
    r = ledger.apply(result([face(2, BOX_A, conf=0.55)]), tick_mono=30.0)
    assert label_of(r, 2) == LABEL_PATIENT
    assert ledger._entries[1].binding == LABEL_UNKNOWN


def test_pool_challenger_cannot_steal_from_stale_phantom(ledger):
    # The Finding-3 fix must NOT open a pool-steal hole: a POOL challenger
    # (weaker outfit evidence) can't take a held seat just because the
    # holder wasn't seen this tick.
    ledger.apply(result([face(1, BOX_A, conf=0.62)]), tick_mono=10.0)
    ledger.note_scene_presence(True)
    ledger.apply(result([]), tick_mono=20.0)          # id1 scene-extends
    r = ledger.apply(result([pool(2, BOX_A, conf=0.90)]), tick_mono=30.0)
    assert label_of(r, 2) == LABEL_UNKNOWN            # face-held seat stands
    assert ledger._entries[1].binding == LABEL_PATIENT


def test_face_reconfirm_resets_pool_scale_confidence(ledger):
    # Pool bind (~0.75), then a face re-confirm at 0.55 must RESET the
    # stored confidence to the face-scale 0.55, not keep the pool 0.75 —
    # else a legit 0.60 face challenger would wrongly lose (Finding low-sev).
    ledger.apply(result([pool(1, BOX_A, conf=0.75)]), tick_mono=10.0)
    assert ledger._entries[1].confidence >= 0.72
    ledger.apply(result([face(1, BOX_A, conf=0.55)]), tick_mono=20.0)
    assert ledger._entries[1].provenance == "face"
    assert abs(ledger._entries[1].confidence - 0.55) < 1e-6


# ── retire + re-entry rebind ────────────────────────────────────────────

def test_patient_retires_after_grace(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    ledger.apply(result([]), tick_mono=20.0)          # gone, within grace
    assert ledger.is_active
    ledger.apply(result([]), tick_mono=100.0)         # > 45s grace
    assert not ledger.is_active
    assert 1 not in ledger._entries


def test_reentry_new_pod_id_rebinds_patient(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    ledger.apply(result([]), tick_mono=100.0)         # retired
    # returns as a fresh pod id — a new entry, re-bound by face.
    r = ledger.apply(result([face(2, BOX_A)]), tick_mono=110.0)
    assert label_of(r, 2) == LABEL_PATIENT
    assert 1 not in ledger._entries


def test_scene_presence_extends_within_bound(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    ledger.note_scene_presence(True)
    ledger.apply(result([]), tick_mono=100.0)         # 90s unseen, within 150s
    assert ledger.is_active                            # held within the bound


def test_scene_extension_bounded_phantom_ages_out(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    ledger.note_scene_presence(True)
    # Scene keeps saying "present" long past the 150s bound (drift phantom).
    ledger.apply(result([]), tick_mono=180.0)
    ledger.apply(result([]), tick_mono=250.0)          # > 150s since body seen
    ledger.apply(result([]), tick_mono=320.0)          # + grace
    assert not ledger.is_active                         # Fix B: it ages out


# ── registry / cross-camera single holder ───────────────────────────────

def test_bind_claims_face_tier_and_retire_vacates(registry):
    led = TrackLedger(camera_id=CAM, registry=registry, settings=SimpleNamespace())
    led.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    assert any(c[1] == TIER_FACE for c in registry.claims)
    led.apply(result([]), tick_mono=100.0)
    assert CAM in registry.vacated


def test_outclaimed_bind_does_not_stick(registry):
    registry.accept = False                            # a better claim rules
    led = TrackLedger(camera_id=CAM, registry=registry, settings=SimpleNamespace())
    r = led.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    assert label_of(r, 1) == LABEL_UNKNOWN
    assert not led.is_active


def test_takeover_elsewhere_unbinds_without_fresh_evidence(registry):
    led = TrackLedger(camera_id=CAM, registry=registry, settings=SimpleNamespace())
    led.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    registry._holder = OTHER_CAM                       # seat moved elsewhere
    r = led.apply(result([track(1, BOX_A)]), tick_mono=20.0)   # no fresh evidence
    assert label_of(r, 1) == LABEL_UNKNOWN


# ── face throttle ───────────────────────────────────────────────────────

def test_face_evidence_full_until_patient_locked(ledger):
    assert ledger.wants_face_evidence() is True        # nobody locked
    ledger.apply(result([track(1, BOX_A)]), tick_mono=10.0)
    assert ledger.wants_face_evidence() is True        # still unbound


def test_face_evidence_heartbeat_once_locked(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    ledger.apply(result([face(1, BOX_A)]), tick_mono=20.0)  # stable, no birth
    seq = [ledger.wants_face_evidence() for _ in range(4)]
    assert seq == [False, False, True, False]          # 1-in-3 heartbeat


def test_new_birth_resumes_full_face_cadence(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    ledger.apply(result([face(1, BOX_A), track(2, BOX_B)]), tick_mono=20.0)
    assert ledger.wants_face_evidence() is True        # a new body appeared


# ── harvest hint ────────────────────────────────────────────────────────

def test_harvest_hint_only_in_face_afterglow(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    assert ledger.harvest_hint() is not None
    # pool-provenance never emits a hint (pool-corruption guard)
    led2 = TrackLedger(camera_id=CAM, registry=None, settings=SimpleNamespace())
    led2.apply(result([pool(1, BOX_A)]), tick_mono=10.0)
    assert led2.harvest_hint() is None


# ── infra safety ────────────────────────────────────────────────────────

def test_reid_not_ran_freezes_everything(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=10.0)
    r = ledger.apply(result([track(2, BOX_B)], ran=False), tick_mono=20.0)
    assert r.tracks[0].track_id == 2                   # returned untouched
    assert 2 not in ledger._entries                    # no mutation


def test_stale_tick_never_mutates_state(ledger):
    ledger.apply(result([face(1, BOX_A)]), tick_mono=20.0)
    ledger.apply(result([track(2, BOX_B)]), tick_mono=10.0)   # stale window
    assert 2 not in ledger._entries
    assert set(ledger._entries) == {1}


def test_stash_truth_fields(ledger):
    ledger.apply(result([face(1, BOX_A, conf=0.6)]), tick_mono=10.0)
    s = ledger.stash_fields()
    assert s["id_binding"] == "patient"
    assert s["id_binding_source"] == "face"
    assert s["id_tracks"] == {"1": LABEL_PATIENT}
    assert s["id_binding_seen"] is True
