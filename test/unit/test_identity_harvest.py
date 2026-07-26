"""Harvest rule — one condition (Aryaman, 2026-07-24).

Face-confirmed patient, alone in frame → harvest liberally. The face
confirmation lives on the TRACK (locked at the door, still valid on the
couch 3 hours later). Anything else — body-anchored seat, second person
in frame, newborn track — never harvests. This deletes the poisoning
loop instead of guarding it: a body-won seat can never self-feed, and
multi-person harvesting simply doesn't exist.
"""

from __future__ import annotations

from src.services.identity.harvest import harvest_eligible
from src.services.identity.seat import Seat

MIN_AGE = 3.0


def _seat(
    cam="a", tid=7, anchor="face", at=100.0, face_confirmed=None,
    face_confirms=None,
):
    if face_confirmed is None:
        face_confirmed = anchor == "face"
    if face_confirms is None:
        # A settled, repeatedly-confirmed patient seat (the depth gate
        # ≥2 is tested explicitly in test_identity_fusion.py).
        face_confirms = 2 if face_confirmed else 0
    return Seat(
        camera_id=cam, track_id=tid, score=0.8, anchor=anchor,
        confirmed_at=at, since=at, face_confirmed=face_confirmed,
        face_confirms=face_confirms,
    )


def test_face_anchored_solo_mature_track_harvests():
    assert harvest_eligible(
        _seat(), camera_id="a", track_id=7, solo=True,
        track_age_s=10.0, min_age_s=MIN_AGE,
    )


def test_body_anchored_seat_never_harvests():
    assert not harvest_eligible(
        _seat(anchor="body"), camera_id="a", track_id=7, solo=True,
        track_age_s=10.0, min_age_s=MIN_AGE,
    )


def test_multi_person_never_harvests():
    assert not harvest_eligible(
        _seat(), camera_id="a", track_id=7, solo=False,
        track_age_s=10.0, min_age_s=MIN_AGE,
    )


def test_other_track_never_harvests():
    assert not harvest_eligible(
        _seat(tid=7), camera_id="a", track_id=9, solo=True,
        track_age_s=10.0, min_age_s=MIN_AGE,
    )


def test_other_camera_never_harvests():
    assert not harvest_eligible(
        _seat(cam="a"), camera_id="b", track_id=7, solo=True,
        track_age_s=10.0, min_age_s=MIN_AGE,
    )


def test_newborn_track_waits():
    # Births are the messy moments (crossing/occlusion re-entry) —
    # don't learn from them.
    assert not harvest_eligible(
        _seat(), camera_id="a", track_id=7, solo=True,
        track_age_s=1.0, min_age_s=MIN_AGE,
    )


def test_no_seat_never_harvests():
    assert not harvest_eligible(
        None, camera_id="a", track_id=7, solo=True,
        track_age_s=10.0, min_age_s=MIN_AGE,
    )


def test_face_confirmed_survives_body_anchor():
    # The confirmation lives on the TRACK: a later body-score ratchet
    # flips the anchor but must NOT stop harvesting (live incident
    # 2026-07-24 — harvest silently froze when anchor went body).
    assert harvest_eligible(
        _seat(anchor="body", face_confirmed=True), camera_id="a",
        track_id=7, solo=True, track_age_s=10.0, min_age_s=MIN_AGE,
    )
