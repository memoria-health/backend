"""The patient seat — Aryaman's rule, 2026-07-24.

Seat = highest combined score. A challenger takes it only by beating
the seated reference score by a margin δ. The reference lives and dies
with the track. Silence never costs the seat (the tracker carries it).
Two plausible candidates on an empty seat → lock neither.
"""

from __future__ import annotations

from identity.seat import Candidate, update_seat

FLOOR = 0.5
DELTA = 0.1


def _c(cam, tid, score, anchor="face"):
    return Candidate(camera_id=cam, track_id=tid, score=score, anchor=anchor)


def _live(*keys):
    return set(keys)


def test_empty_seat_locks_clear_winner():
    seat = update_seat(
        None, [_c("a", 7, 0.8), _c("a", 9, 0.3)],
        live=_live(("a", 7), ("a", 9)), floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat is not None
    assert (seat.camera_id, seat.track_id) == ("a", 7)
    assert seat.score == 0.8
    assert seat.anchor == "face"


def test_empty_seat_below_floor_stays_empty():
    seat = update_seat(
        None, [_c("a", 7, 0.4)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat is None


def test_two_plausible_lock_neither():
    # The wrong-lock guard: both clear the floor, within δ of each
    # other → refuse to choose. UNKNOWN is recoverable; a wrong patient
    # is not.
    seat = update_seat(
        None, [_c("a", 7, 0.72), _c("a", 9, 0.68)],
        live=_live(("a", 7), ("a", 9)), floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat is None


def test_silence_holds_seat():
    seat = update_seat(
        None, [_c("a", 7, 0.8)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    # Next batches: no scores at all (patient far away, no evidence).
    held = update_seat(
        seat, [], live=_live(("a", 7)), floor=FLOOR, delta=DELTA, now=200.0,
    )
    assert held is seat


def test_challenger_must_beat_by_delta():
    seat = update_seat(
        None, [_c("a", 7, 0.8)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    live = _live(("a", 7), ("b", 4))
    # 0.85 < 0.8 + δ → seat stays
    kept = update_seat(
        seat, [_c("b", 4, 0.85)], live=live,
        floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert (kept.camera_id, kept.track_id) == ("a", 7)
    # 0.95 ≥ 0.8 + δ → seat moves
    moved = update_seat(
        kept, [_c("b", 4, 0.95)], live=live,
        floor=FLOOR, delta=DELTA, now=102.0,
    )
    assert (moved.camera_id, moved.track_id) == ("b", 4)
    assert moved.score == 0.95


def test_same_track_refreshes_reference_upward():
    seat = update_seat(
        None, [_c("a", 7, 0.6, anchor="body")], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    seat = update_seat(
        seat, [_c("a", 7, 0.9, anchor="face")], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert seat.score == 0.9
    assert seat.anchor == "face"              # anchor upgrades with evidence
    assert seat.confirmed_at == 101.0


def test_same_track_lower_score_does_not_erode_reference():
    seat = update_seat(
        None, [_c("a", 7, 0.9)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    seat = update_seat(
        seat, [_c("a", 7, 0.55)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert seat.score == 0.9                  # weak later evidence ≠ demotion


def test_seat_dies_with_track():
    seat = update_seat(
        None, [_c("a", 7, 0.8)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    gone = update_seat(
        seat, [], live=_live(("a", 9)), floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert gone is None


def test_after_track_death_new_candidate_locks_fresh():
    seat = update_seat(
        None, [_c("a", 7, 0.9)], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    # Track 7 dies; challenger on cam b needs only the floor now (no
    # stale 0.9 reference to beat — it died with the track).
    seat = update_seat(
        seat, [_c("b", 4, 0.6)], live=_live(("b", 4)),
        floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert seat is not None
    assert (seat.camera_id, seat.track_id) == ("b", 4)


def test_dead_track_candidate_is_ignored():
    # A score for a track that died between scoring and folding must
    # not take the seat (review 2026-07-24: dead-track exclusion).
    seat = update_seat(
        None, [_c("a", 7, 0.9)], live=_live(("a", 9)),   # 7 is dead
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat is None


def test_sub_floor_runner_up_does_not_block_lock():
    # Ambiguity only applies between PLAUSIBLE candidates: a runner-up
    # below the floor is not a plausible patient and must not veto the
    # clear winner (review 2026-07-24: the highest-stakes branch).
    seat = update_seat(
        None, [_c("a", 7, 0.72), _c("a", 9, 0.45)],
        live=_live(("a", 7), ("a", 9)), floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat is not None
    assert (seat.camera_id, seat.track_id) == ("a", 7)


def test_same_track_body_never_ratchets_reference():
    # Once harvests fill the pool, the track matches its OWN crops at
    # ~1.0 — self-referential. That evidence must not raise the bar a
    # challenger must beat (live incident 2026-07-24: unassailable 1.0
    # seat) — and face_confirmed is untouched by it.
    seat = update_seat(
        None, [_c("a", 7, 0.56, anchor="face")], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    seat = update_seat(
        seat, [_c("a", 7, 1.0, anchor="body")], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert seat.score == 0.56 and seat.anchor == "face"
    assert seat.face_confirmed is True


def test_similar_clothes_separate_at_full_resolution():
    # The bug the cap caused: two body-only candidates whose raw
    # cosines genuinely separate must lock the stronger one, not tie.
    seat = update_seat(
        None,
        [_c("a", 7, 1.0, anchor="body"), _c("a", 9, 0.75, anchor="body")],
        live=_live(("a", 7), ("a", 9)), floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat is not None
    assert (seat.camera_id, seat.track_id) == ("a", 7)


def test_body_locked_seat_gains_confirmation_on_face_evidence():
    seat = update_seat(
        None, [_c("a", 7, 0.7, anchor="body")], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=100.0,
    )
    assert seat.face_confirmed is False
    # Weaker face evidence doesn't beat the reference but DOES confirm.
    seat = update_seat(
        seat, [_c("a", 7, 0.55, anchor="face")], live=_live(("a", 7)),
        floor=FLOOR, delta=DELTA, now=101.0,
    )
    assert seat.score == 0.7                  # reference unchanged
    assert seat.face_confirmed is True        # confirmation recorded
