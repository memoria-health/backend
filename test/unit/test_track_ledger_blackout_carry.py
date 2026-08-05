"""Carrying the patient lock through a lights-off / IR blackout.

The scenario, from a real bedroom (Aryaman, 2026-08-05): someone is in
bed, far from the lens. The light goes off, the camera flips to IR, and
the entire frame changes at once. EdgeTAM loses the mask and re-grounds
the same body under a NEW track_id.

This ledger keys by pod track_id and never re-associates, so the bound
entry sits un-sighted until it retires while the live body next to it is
labelled UNKNOWN. It is the worst possible place to lose the lock: at
distance, in the dark, in bed, neither a face nor an outfit match will
re-establish the patient for the rest of the night — so every sleep,
waking and disturbance for that night is attributed to nobody.

The carry is a PHYSICAL claim, not a similarity one: inside a couple of
seconds a person cannot have left the room and a different person cannot
have taken their place in the same box. Hence the hard time bound, and
hence the refusal to act when two bodies could both be the heir.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.services.identity.track_ledger import (
    LABEL_PATIENT,
    LABEL_UNKNOWN,
    TrackLedger,
)
from test.unit.test_track_ledger import CAM, face, label_of, result, track

BED = (0.30, 0.40, 0.55, 0.75)          # where the sleeper is
BED_SHIFTED = (0.32, 0.42, 0.57, 0.77)  # same body, box moved a little on IR
DOORWAY = (0.80, 0.10, 0.95, 0.60)      # a different part of the room


def _ledger(**overrides):
    return TrackLedger(
        camera_id=CAM, registry=None, settings=SimpleNamespace(**overrides),
    )


def _bind_patient(led, tid=1, box=BED, *, t=100.0):
    """Establish the patient on `tid` via a face lock."""
    led.apply(result([face(tid, box)]), tick_mono=t)
    assert label_of(led.apply(result([face(tid, box)]), tick_mono=t + 1), tid) == (
        LABEL_PATIENT
    )


# ── the carry itself ────────────────────────────────────────────────


def test_lock_moves_to_the_new_track_id_after_a_blackout():
    led = _ledger()
    _bind_patient(led, 1)
    # Lights off: track 1 is gone, track 2 is the same body re-grounded.
    res = led.apply(result([track(2, BED_SHIFTED)]), tick_mono=102.0)
    assert label_of(res, 2) == LABEL_PATIENT, (
        "the re-acquired body should inherit the lock, not be born UNKNOWN"
    )


def test_the_dead_track_does_not_linger_holding_a_second_patient():
    """Two entries both claiming PATIENT would let the takeover check and the
    home registry disagree about which camera holds the seat."""
    led = _ledger()
    _bind_patient(led, 1)
    led.apply(result([track(2, BED_SHIFTED)]), tick_mono=102.0)
    patients = [e for e in led._entries.values() if e.binding == LABEL_PATIENT]
    assert len(patients) == 1
    assert patients[0].pid == 2


def test_the_heir_inherits_the_lock_it_did_not_earn():
    """Provenance and bound_at come from the holder: the heir has proven
    nothing itself, it is the same body under a new id. Resetting them would
    restart the hold and re-open mid-track face re-litigation."""
    led = _ledger()
    _bind_patient(led, 1)
    before = dict(
        provenance=led._entries[1].provenance,
        bound_at=led._entries[1].bound_at,
        confidence=led._entries[1].confidence,
    )
    led.apply(result([track(2, BED_SHIFTED)]), tick_mono=102.0)
    heir = led._entries[2]
    assert heir.provenance == before["provenance"]
    assert heir.bound_at == before["bound_at"]
    assert heir.confidence == before["confidence"]


# ── the guards ──────────────────────────────────────────────────────


def test_refuses_when_two_bodies_could_both_be_the_heir():
    """Mislabelling someone ELSE as the patient is the one failure this
    ledger has always refused to risk. Doing nothing degrades to the old
    behaviour; guessing does not."""
    led = _ledger()
    _bind_patient(led, 1)
    res = led.apply(
        result([track(2, BED_SHIFTED), track(3, (0.31, 0.41, 0.56, 0.76))]),
        tick_mono=102.0,
    )
    assert label_of(res, 2) == LABEL_UNKNOWN
    assert label_of(res, 3) == LABEL_UNKNOWN


def test_refuses_a_body_somewhere_else_in_the_room():
    """The claim is that the body did not MOVE. A track born across the room
    is a different person walking in, not a re-acquisition."""
    led = _ledger()
    _bind_patient(led, 1)
    res = led.apply(result([track(2, DOORWAY)]), tick_mono=102.0)
    assert label_of(res, 2) == LABEL_UNKNOWN


def test_refuses_once_the_physical_claim_expires():
    """After the window the argument evaporates: given long enough, the
    person really can have left and been replaced."""
    led = _ledger(reid_blackout_carry_max_gap_s=5.0)
    _bind_patient(led, 1)
    res = led.apply(result([track(2, BED_SHIFTED)]), tick_mono=140.0)
    assert label_of(res, 2) == LABEL_UNKNOWN


def test_does_not_fire_while_the_bound_track_is_still_visible():
    """A second body appearing next to a patient who is STILL tracked is just
    another person in the room."""
    led = _ledger()
    _bind_patient(led, 1)
    res = led.apply(
        result([face(1, BED), track(2, BED_SHIFTED)]), tick_mono=102.0,
    )
    assert label_of(res, 1) == LABEL_PATIENT
    assert label_of(res, 2) == LABEL_UNKNOWN


def test_disabled_by_zero_window():
    led = _ledger(reid_blackout_carry_max_gap_s=0.0)
    _bind_patient(led, 1)
    res = led.apply(result([track(2, BED_SHIFTED)]), tick_mono=102.0)
    assert label_of(res, 2) == LABEL_UNKNOWN


def test_does_not_fire_on_a_degraded_tick():
    """Degraded frames are the case where boxes are least trustworthy, which
    is exactly when a geometric claim should not be believed."""
    led = _ledger()
    _bind_patient(led, 1)
    res = led.apply(
        result([track(2, BED_SHIFTED)], degraded=True), tick_mono=102.0,
    )
    assert label_of(res, 2) == LABEL_UNKNOWN


def test_never_invents_a_patient_from_nothing():
    """No prior lock means nothing to carry — a body appearing in the dark is
    UNKNOWN until it earns the seat."""
    led = _ledger()
    res = led.apply(result([track(2, BED)]), tick_mono=102.0)
    assert label_of(res, 2) == LABEL_UNKNOWN


@pytest.mark.parametrize("bad", [None, "x", float("nan")])
def test_a_carry_bug_can_never_break_a_tick(bad):
    """apply() is wrapped precisely so identity logic degrades instead of
    taking the tick down. Feeding it nonsense must return, not raise."""
    led = _ledger()
    _bind_patient(led, 1)
    led._entries[1].last_bbox = bad  # corrupt the geometry the carry reads
    out = led.apply(result([track(2, BED_SHIFTED)]), tick_mono=102.0)
    assert out is not None
