"""IdentityClient — implementation A adapter (2026-07-26).

The adapter's whole job: translate the identity service's /state into
the exact IdentityResult shape the TrackLedger can bind from, and
fail OPEN (reid_ran=False) on any doubt. These tests pin the mapping
and the doors: seat→PATIENT lanes (face vs pool), prefix camera
matching, and the fail-open paths (down, not-ready, unknown camera).
"""

import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest

from src.services.identity.identity_client import IdentityClient

CAM = UUID("9271358c-0000-4000-8000-000000000000")
CAM_KEY = "9271358c"
# The patient this tick is about, and a second resident the same camera
# covers — the shared-lounge case (2026-08-20).
PATIENT = UUID("9271358c-0000-4000-8000-000000000000")
OTHER_RESIDENT = UUID("44444444-4444-4444-4444-444444444444")


def _sidecar(state):
    sc = IdentityClient(settings=SimpleNamespace(
        identity_state_url="http://x:1", identity_provider="sidecar",
    ))

    async def _fake_state():
        return state

    sc._state = _fake_state
    return sc


def _label(sc):
    return asyncio.run(sc.label_clip(
        mp4_bytes=b"", patient_id=CAM, camera_id=CAM,
    ))


def _state(seat=None, tracks=None, seats=None):
    """state-v2: `seats` is keyed by patient uuid. `seat=` is the
    convenience shorthand for "this tick's patient is located"."""
    if seats is None:
        seats = {str(PATIENT): seat} if seat is not None else {}
    return {
        "ready": True,
        "seats": seats,
        "cameras": {CAM_KEY: {"tracks": tracks or {}}},
    }


def test_state_down_fails_open():
    sc = _sidecar(None)
    r = _label(sc)
    assert r.reid_ran is False and r.patient_in_clip is False


def test_unknown_camera_fails_open():
    sc = _sidecar({"ready": True, "seats": {}, "cameras": {"deadbeef": {}}})
    r = _label(sc)
    assert r.reid_ran is False


def test_empty_room_runs_with_no_people():
    sc = _sidecar(_state())
    r = _label(sc)
    assert r.reid_ran is True
    assert r.people_present_reid is False and r.tracks == ()
    assert r.patient_in_clip is False


def test_face_confirmed_seat_binds_on_face_lane():
    sc = _sidecar(_state(
        seat={"camera": CAM_KEY, "track": 3, "score": 0.72,
              "face_confirmed": True, "anchor": "face"},
        tracks={"3": {"bbox": [10, 10, 50, 100]}},
    ))
    r = _label(sc)
    assert r.patient_in_clip is True and r.confidence == 0.72
    (t,) = r.tracks
    assert t.label == "PATIENT" and t.face_locked is True
    assert t.bbox_series == ((0, (10.0, 10.0, 50.0, 100.0)),)


def test_body_anchored_seat_binds_on_pool_lane():
    sc = _sidecar(_state(
        seat={"camera": CAM_KEY, "track": 1, "score": 0.93,
              "face_confirmed": False, "anchor": "body"},
        tracks={"1": {"bbox": [1, 2, 3, 4]}},
    ))
    (t,) = _label(sc).tracks
    assert t.label == "PATIENT" and t.face_locked is False
    assert t.pre_labeled is True and t.pre_label_source == "gallery"


def test_seat_on_other_camera_leaves_tracks_unknown():
    sc = _sidecar(_state(
        seat={"camera": "3c6570af", "track": 9, "score": 0.9,
              "face_confirmed": True, "anchor": "face"},
        tracks={"5": {"bbox": [1, 2, 3, 4]}},
    ))
    r = _label(sc)
    assert r.patient_in_clip is False and r.confidence == 0.0
    (t,) = r.tracks
    assert t.label == "UNKNOWN" and t.face_locked is False


def test_suppression_can_never_fire_from_sidecar():
    sc = _sidecar(_state(
        seat={"camera": "3c6570af", "track": 9, "score": 0.9,
              "face_confirmed": True, "anchor": "face"},
        tracks={"5": {"bbox": [1, 2, 3, 4]}},
    ))
    assert _label(sc).has_only_verified_other_tracks is False


def test_label_clip_never_raises_on_garbage_state():
    sc = _sidecar({"ready": True,
                   "seats": {str(PATIENT): {"camera": CAM_KEY, "track": "x"}},
                   "cameras": {CAM_KEY: {"tracks": {
                       "a": {"bbox": [1]},
                       "2": {"bbox": [1, 2, 3, 4]},
                   }}}})
    try:
        r = _label(sc)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"label_clip raised: {e!r}")
    assert r.reid_ran in (True, False)


def test_outsized_payload_is_capped_not_allocated():
    from src.services.identity import identity_client as sc_mod

    tracks = {str(i): {"bbox": [1, 2, 3, 4]} for i in range(500)}
    sc = _sidecar(_state(tracks=tracks))
    r = _label(sc)
    assert r.reid_ran is True
    assert len(r.tracks) <= sc_mod._MAX_TRACKS


def test_home_wide_track_count_sums_all_cameras():
    """Crowd fallback is a GLOBAL rule: the sidecar reports live
    tracks across the WHOLE home so a packed living room disables
    crops on every camera, not just the crowded one."""
    state = {
        "ready": True,
        "seats": {},
        "cameras": {
            CAM_KEY: {"tracks": {"1": {"bbox": [1, 2, 3, 4]}}},
            "3c6570af": {"tracks": {
                str(i): {"bbox": [1, 2, 3, 4]} for i in range(5)
            }},
        },
    }
    r = _label(_sidecar(state))
    assert r.people_count_reid == 1          # this camera
    assert r.people_count_home == 6          # whole home


# ── Shared cameras: one camera, several residents (2026-08-20) ───────


def test_reads_only_this_patients_seat():
    """A shared lounge locates Alice on track 3 and Bob on track 8.
    A tick about Alice must bind track 3 and leave Bob's track
    UNKNOWN — Bob's presence is not Alice's evidence."""
    sc = _sidecar(_state(
        seats={
            str(PATIENT): {"camera": CAM_KEY, "track": 3, "score": 0.80,
                           "face_confirmed": True, "anchor": "face"},
            str(OTHER_RESIDENT): {"camera": CAM_KEY, "track": 8,
                                  "score": 0.91, "face_confirmed": True,
                                  "anchor": "face"},
        },
        tracks={"3": {"bbox": [1, 2, 3, 4]}, "8": {"bbox": [5, 6, 7, 8]}},
    ))
    r = _label(sc)
    assert r.patient_in_clip is True
    assert r.confidence == 0.80          # Alice's score, not Bob's 0.91
    bound = {t.track_id: t.label for t in r.tracks}
    assert bound == {3: "PATIENT", 8: "UNKNOWN"}


def test_another_residents_seat_is_never_borrowed():
    """THE SHADY-BUG GUARD. Bob is located on this camera, Alice is
    not. A tick about Alice must come back "not located" — reading the
    only seat present, or the first one, would file Bob's activity onto
    Alice's record. Unlike a missed observation that is not
    recoverable."""
    sc = _sidecar(_state(
        seats={
            str(OTHER_RESIDENT): {"camera": CAM_KEY, "track": 8,
                                  "score": 0.95, "face_confirmed": True,
                                  "anchor": "face"},
        },
        tracks={"8": {"bbox": [5, 6, 7, 8]}},
    ))
    r = _label(sc)
    assert r.reid_ran is True            # the service answered
    assert r.patient_in_clip is False    # ...and the answer was "not Alice"
    assert r.confidence == 0.0
    (t,) = r.tracks
    assert t.label == "UNKNOWN" and t.face_locked is False
    assert t.pre_labeled is False
    # Still never "verified other": an unrecognized body must not let
    # identity suppression skip the model on a shared camera.
    assert r.has_only_verified_other_tracks is False


def test_v1_seat_key_is_not_read_as_this_patients_answer():
    """A state-v1 node (single `seat`, no `seats`) reaching a v2 client
    must read as no verdict. The v1 shape carried whichever patient
    that process served, and the router now sends a tick here based on
    the CAMERA — so trusting `seat` would attribute one resident's lock
    to another. The contract check fails this first in production; this
    pins the parser too, since fail-open is only safe if it is total."""
    sc = _sidecar({
        "ready": True,
        "seat": {"camera": CAM_KEY, "track": 3, "score": 0.9,
                 "face_confirmed": True, "anchor": "face"},
        "cameras": {CAM_KEY: {"tracks": {"3": {"bbox": [1, 2, 3, 4]}}}},
    })
    r = _label(sc)
    assert r.patient_in_clip is False
    (t,) = r.tracks
    assert t.label == "UNKNOWN"
