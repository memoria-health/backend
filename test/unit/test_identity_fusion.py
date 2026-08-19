"""Track-level face evidence + both live-incident replays (2026-07-24).

Two failures in one evening shaped the statistic:
- The AUNT: one 0.46 outlier among twenty near-zeros seated an
  impostor (memoryless gate), then harvest poisoned the gallery.
- The T11 WALK-AROUND: mean-fusion diluted the real patient — half a
  walking track's sightings are back-of-head garbage, and ten genuine
  0.42-0.49 sightings fused to a stuck 0.23.

The replay tests below use the ACTUAL numbers from both incidents'
logs; the seat/harvest tests pin the gating defenses.
"""

from uuid import UUID

from identity.fusion import SightingHistory
from identity.harvest import harvest_eligible
from identity.seat import Candidate, Seat, update_seat

# T11's real sightings, 17:57-17:58 log (walk-around: genuine faces
# interleaved with back-of-head detections the mean-fusion choked on).
ARYAMAN_WALK = [
    (0.08, 34), (0.49, 38), (0.32, 45), (0.35, 99), (0.09, 44),
    (0.18, 36), (0.46, 35), (0.35, 36), (0.06, 45), (0.47, 34),
    (0.49, 36), (0.45, 34),
]


def test_patient_walk_around_clears_the_floor():
    h = SightingHistory()
    for cos, px in ARYAMAN_WALK:
        h.add(cos, px, now=1000.0)
    # Consistency waiver: many marginal-but-consistent sightings
    # substitute for one big face. Measured 0.488 on the real replay.
    assert h.evidence(now=1001.0) >= 0.40


def test_impostor_spike_stays_buried():
    h = SightingHistory()
    for _ in range(20):                       # the aunt's twenty zeros
        h.add(0.05, 36, now=1000.0)
    h.add(0.46, 53, now=1000.0)                           # her one flattering angle
    # Measured 0.147 on the real replay — the outlier needs K-1
    # friends it doesn't have, and one qualifier can't open the waiver.
    assert h.evidence(now=1001.0) < 0.40


def test_lone_outlier_pads_against_zeros_not_itself():
    h = SightingHistory()
    h.add(0.55, 50, now=1000.0)                           # single strong-ish frame
    assert h.evidence(now=1001.0) < 0.40                # 1-of-K, padded with 0s


def test_back_of_head_garbage_cannot_dilute():
    # A walking track alternates genuine faces with high-det-confidence
    # back-of-head detections scoring ~0 — top-K must read through them
    # (the exact mean-fusion failure).
    h = SightingHistory()
    for cos, px in [(0.03, 45), (0.50, 40)] * 6:
        h.add(cos, px, now=1000.0)
    assert h.evidence(now=1001.0) >= 0.40


# ── seat gating ──────────────────────────────────────────────────────

_P = UUID("11111111-1111-1111-1111-111111111111")


def _cand(tid=6, score=0.43, anchor="face", provisional=True):
    return Candidate(
        camera_id="a", track_id=tid, patient_id=_P, score=score, anchor=anchor,
        provisional=provisional,
    )


def test_provisional_cannot_take_empty_seat():
    seat = update_seat(
        None, [_cand()], live={("a", 6)}, floor=0.4, delta=0.1, now=0.0,
    )
    assert seat is None                       # the aunt's 0.46 spike


def test_non_provisional_takes_empty_seat():
    seat = update_seat(
        None, [_cand(provisional=False)],
        live={("a", 6)}, floor=0.4, delta=0.1, now=0.0,
    )
    assert seat is not None and seat.track_id == 6


def test_provisional_cannot_steal_occupied_seat():
    held = Seat(
        camera_id="a", track_id=1, score=0.5, anchor="face",
        confirmed_at=0.0, since=0.0, face_confirmed=True, face_confirms=2,
    )
    out = update_seat(
        held, [_cand(tid=6, score=0.95)],
        live={("a", 1), ("a", 6)}, floor=0.4, delta=0.1, now=1.0,
    )
    assert out.track_id == 1                  # holder survives the spike


def test_provisional_still_holds_and_deepens_own_seat():
    held = Seat(
        camera_id="a", track_id=6, score=0.43, anchor="face",
        confirmed_at=0.0, since=0.0, face_confirmed=True, face_confirms=1,
    )
    out = update_seat(
        held, [_cand(tid=6, score=0.41)],
        live={("a", 6)}, floor=0.4, delta=0.1, now=1.0,
    )
    assert out.track_id == 6
    assert out.face_confirms == 2             # confirmation deepens


# ── harvest depth ────────────────────────────────────────────────────

def _seat(confirms: int) -> Seat:
    return Seat(
        camera_id="a", track_id=6, score=0.43, anchor="face",
        confirmed_at=0.0, since=0.0, face_confirmed=True,
        face_confirms=confirms,
    )


def test_single_confirmation_never_harvests():
    # The aunt at 16:26:48: locked once, never re-confirmed. Under the
    # depth gate her clothes never enter the gallery.
    assert not harvest_eligible(
        _seat(1), camera_id="a", track_id=6, solo=True,
        track_age_s=10.0, min_age_s=3.0,
    )


def test_depth_two_harvests():
    assert harvest_eligible(
        _seat(2), camera_id="a", track_id=6, solo=True,
        track_age_s=10.0, min_age_s=3.0,
    )


def test_empty_fold_reaps_dead_track_seat():
    # reap_seat path: an empty candidate list still applies
    # seat-dies-with-track (the ghost-seat fix, 2026-07-24).
    held = Seat(
        camera_id="a", track_id=10, score=0.82, anchor="body",
        confirmed_at=0.0, since=0.0,
    )
    assert update_seat(
        held, [], live=set(), floor=0.4, delta=0.1, now=1.0,
    ) is None


def test_uncle_t13_replay_deep_history_beats_single_flash():
    # T13, 18:11-18:12 live: 25 near-zeros (including at 80-100px,
    # close-up!) then ONE 0.46@79px flash. The history's verdict
    # (0.147) must be the evidence — the flash locked live because
    # max(history, single) bypassed it. Sub-instant singles never
    # override the track's history.
    h = SightingHistory()
    zeros = [(0.05, 44), (0.0, 43), (0.07, 45), (0.06, 41), (0.09, 49),
             (0.13, 74), (0.13, 94), (0.11, 101), (0.15, 83), (0.14, 86),
             (0.12, 92), (0.06, 60)]
    for cos, px in zeros:
        h.add(cos, px, now=1000.0)
    h.add(0.46, 79, now=1000.0)
    assert h.evidence(now=1001.0) < 0.40


def test_sub_floor_face_round_is_not_a_confirmation():
    # 2026-07-24: a wrongly-seated track's 0.31 face round must not
    # count toward the harvest gate.
    held = Seat(
        camera_id="a", track_id=13, score=0.44, anchor="face",
        confirmed_at=0.0, since=0.0, face_confirmed=True, face_confirms=1,
    )
    out = update_seat(
        held, [_cand(tid=13, score=0.31, provisional=True)],
        live={("a", 13)}, floor=0.4, delta=0.1, now=1.0,
    )
    assert out.face_confirms == 1             # unchanged — no harvest


def test_contested_position_is_nobodys_evidence():
    # 2026-07-24, the 19:13 aunt lock: the patient stood in front of
    # the aunt, his face landed inside HER box's crop, and the hit was
    # credited to her track. A face position inside more than one
    # track's (padded) box must be discarded — whoever's crop found it.
    from identity.loop import _contested
    overlapped = [(1, 0, 0, 100, 200), (2, 30, 20, 130, 220)]
    assert _contested(60.0, 60.0, overlapped, own=2) is True
    assert _contested(60.0, 60.0, overlapped, own=1) is True
    separated = [(1, 0, 0, 100, 200), (2, 500, 0, 600, 200)]
    assert _contested(50.0, 100.0, separated, own=1) is False


def test_ghost_evidence_expires_by_time():
    # 2026-07-25: the patient left, his track welded onto a relative,
    # and his frozen 0.4+ sightings held the seat for 16 minutes —
    # faceless probes never displaced them from a count-based window.
    # Evidence is a claim about NOW: old sightings stop counting.
    h = SightingHistory()
    for cos, px in ARYAMAN_WALK:
        h.add(cos, px, now=1000.0)
    assert h.evidence(now=1010.0) >= 0.40     # fresh → holds
    assert h.evidence(now=1000.0 + 120.0) == 0.0   # ghost → silent
