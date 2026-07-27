"""SidecarReID — implementation A adapter (2026-07-26).

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

from src.services.reid.sidecar import SidecarReID

CAM = UUID("9271358c-0000-4000-8000-000000000000")
CAM_KEY = "9271358c"


def _sidecar(state):
    sc = SidecarReID(settings=SimpleNamespace(
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


def _state(seat=None, tracks=None):
    return {
        "ready": True,
        "seat": seat,
        "cameras": {CAM_KEY: {"tracks": tracks or {}}},
    }


def test_state_down_fails_open():
    sc = _sidecar(None)
    r = _label(sc)
    assert r.reid_ran is False and r.patient_in_clip is False


def test_unknown_camera_fails_open():
    sc = _sidecar({"ready": True, "seat": None, "cameras": {"deadbeef": {}}})
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
    sc = _sidecar({"ready": True, "seat": {"camera": CAM_KEY, "track": "x"},
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
    from src.services.reid import sidecar as sc_mod

    tracks = {str(i): {"bbox": [1, 2, 3, 4]} for i in range(500)}
    sc = _sidecar(_state(tracks=tracks))
    r = _label(sc)
    assert r.reid_ran is True
    assert len(r.tracks) <= sc_mod._MAX_TRACKS
