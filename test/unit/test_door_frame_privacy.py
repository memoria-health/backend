"""The still served to the exit-door picker for a PRIVATE camera.

Context (2026-08-04): the picker used to subscribe to the camera's LiveKit
track from the client. For a private camera that can never work — privacy
mode restricts subscription to `detector_worker_*` identities — so it waited
six seconds and reported "this sensor isn't streaming right now", which was
false. It was streaming; the caregiver may not watch it.

The fix serves one still from the server. That only stays safe if people are
masked, because a still the client can re-fetch is a slideshow of the room.

The rule this file pins is the one that differs from every other censor path:
NO PERSON PRESENT → RETURN THE FRAME UNCHANGED. `_censor_frame` pixelates
full-frame in that case, correctly, because a clip with a sensitive verdict
must not leak a missed body. Here the room IS the subject and full-frame
pixelation would defeat the feature. An empty bedroom is architecture.

WHAT MOVED (2026-08-06). The censor used to segment the still itself, so it
could see for itself that nobody was in shot. Its masks now come from the
perception pod, which records nothing at all for a tick with no people — so
"no masks" is ambiguous between an empty room and a blind sidecar, and only
one of those may be served. The endpoint (`api/cameras.door_frame`) settles
it against the identity service's /state first, and calls the censor only
when someone is actually tracked. Two files, therefore two halves of this
file: the censor's half (mask what is there, raise otherwise) and the
endpoint's half (who is allowed to skip masking at all).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.detection import censor, pod_masks

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")

CAMERA_ID = "cafe1234-0000-4000-8000-000000000000"
POD_URL = "http://pod.invalid"


def _jpeg(w: int = 320, h: int = 240, fill: int = 200) -> bytes:
    """A synthetic room: flat wall, a dark 'door', and a high-contrast
    checkerboard where the 'person' stands.

    The checkerboard matters. An earlier version of this fixture left the
    person region flat, so pixelating it produced identical pixels and the
    "was it censored?" assertion compared 0.0 to 0.0 and passed vacuously.
    Detail is what pixelation destroys, so the region has to have some.
    """
    frame = np.full((h, w, 3), fill, dtype=np.uint8)
    frame[40:200, 20:90] = 30  # the "door" — must survive untouched
    ys, xs = np.mgrid[100:160, 240:300]
    frame[100:160, 240:300] = np.where(
        ((ys // 4 + xs // 4) % 2)[..., None], 255, 0
    ).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    assert ok
    return bytes(buf)


def _decode(jpeg: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert arr is not None
    return arr


def _mask_window(mask: "np.ndarray | None", *, w: int = 320, h: int = 240):
    """A one-frame `MaskWindow` carrying `mask`, RLE-encoded exactly the
    way the pod encodes it — so this exercises the real decode path
    rather than a stand-in for it."""
    if mask is None:
        return pod_masks.MaskWindow([])
    from gpu.perception.continuous import rle_encode

    return pod_masks.MaskWindow([{
        "captured_at": 1_786_000_000.0,
        "width": w, "height": h,
        "masks": {"1": rle_encode(mask)},
    }])


def _serve(monkeypatch, window) -> bytes:
    monkeypatch.setattr(pod_masks, "fetch_window", lambda *a, **kw: window)
    return censor.mask_people_in_still_sync(
        _jpeg(), camera_id=CAMERA_ID, pod_url=POD_URL,
    )


# ── the censor's half ────────────────────────────────────────────────


def test_person_region_is_censored_but_the_room_survives(monkeypatch) -> None:
    """A person is pixelated; pixels well outside the dilated mask are left
    alone so the room stays usable for picking doors."""
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[100:160, 240:300] = 1        # person, far from the "door"

    src = _jpeg()
    before = _decode(src)
    after = _decode(_serve(monkeypatch, _mask_window(mask)))

    # The person region loses its detail. Pixelation flattens the
    # checkerboard, so variance drops sharply — that is what "censored"
    # means here, and it is measurable rather than merely "changed".
    person_before = float(before[100:160, 240:300].std())
    person_after = float(after[100:160, 240:300].std())
    assert person_before > 80, "fixture must have real detail to destroy"
    assert person_after < person_before / 2

    # The door survives. It sits far from the mask and its dilation, so a
    # caregiver can still see what they came here to tap.
    assert np.allclose(
        before[40:200, 20:90], after[40:200, 20:90], atol=6
    ), "the door must not be censored — the picker is unusable without it"


def test_track_resolution_masks_are_upscaled_to_the_still(monkeypatch) -> None:
    """Pod masks are computed at EdgeTAM's track resolution, not the
    still's. Decoding at the RECORDED dims and scaling up is the whole
    contract; decoding at the still's dims would reshape garbage."""
    small = np.zeros((60, 80), dtype=np.uint8)   # quarter scale
    small[25:40, 60:75] = 1                      # same region, scaled

    after = _decode(
        _serve(monkeypatch, _mask_window(small, w=80, h=60))
    )
    before = _decode(_jpeg())
    assert float(after[100:160, 240:300].std()) < float(
        before[100:160, 240:300].std()
    ) / 2


def test_no_mask_raises_so_the_caller_fails_closed(monkeypatch) -> None:
    """Reaching this function means the endpoint established someone IS
    tracked. Finding no mask then means the window is stale, not that
    the room is empty — serve nothing rather than an unmasked room."""
    with pytest.raises(censor.CensorError):
        _serve(monkeypatch, _mask_window(None))


def test_empty_mask_also_raises(monkeypatch) -> None:
    """An all-zero mask carries no person either. Same treatment."""
    with pytest.raises(censor.CensorError):
        _serve(monkeypatch, _mask_window(np.zeros((240, 320), np.uint8)))


def test_pod_unreachable_raises(monkeypatch) -> None:
    """Cannot-ask is NOT the same as nobody-there. An unreachable pod
    must raise so the endpoint serves 503, never an unmasked private
    room."""
    def boom(*a, **kw):
        raise pod_masks.PodMaskError("tunnel is dead")

    monkeypatch.setattr(pod_masks, "fetch_window", boom)
    with pytest.raises(pod_masks.PodMaskError):
        censor.mask_people_in_still_sync(
            _jpeg(), camera_id=CAMERA_ID, pod_url=POD_URL,
        )


def test_undecodable_input_raises() -> None:
    with pytest.raises(censor.CensorError):
        censor.mask_people_in_still_sync(
            b"this is not a jpeg", camera_id=CAMERA_ID, pod_url=POD_URL,
        )


# ── the endpoint's half: who may skip masking ────────────────────────


class _FakeReid:
    def __init__(self, answer):
        self.answer = answer

    async def tracked_people(self, camera_id):
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def _request(reid):
    from types import SimpleNamespace

    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                detector_worker=SimpleNamespace(_reid=reid),
            ),
        ),
    )


async def test_presence_probe_reports_the_track_count() -> None:
    from src.api.cameras import _tracked_people

    assert await _tracked_people(_request(_FakeReid(2)), CAMERA_ID) == 2
    assert await _tracked_people(_request(_FakeReid(0)), CAMERA_ID) == 0


def test_an_empty_room_is_answered_with_a_settled_frame() -> None:
    """A presence verdict describes what perception saw `lag_s` ago
    (1.2-2.6s measured live), so it cannot clear the NEWEST frame — a
    person who just walked in is in that picture and not yet in the
    verdict. The old segmenter read these very pixels and had no such
    gap. `min_age_seconds` asks for a frame the verdict actually covers,
    and answers None when the buffer holds none.
    """
    import time as _time
    from types import SimpleNamespace

    from src.api.cameras import (
        _PRESENCE_VERDICT_LAG_MARGIN_S, _latest_frame_jpeg,
    )
    from src.services.detection.frame_buffer import FrameSnapshot

    now = _time.monotonic()
    fresh = FrameSnapshot(width=4, height=4, data=_jpeg(), captured_at=now)
    settled = FrameSnapshot(
        width=4, height=4, data=_jpeg(fill=120),
        captured_at=now - _PRESENCE_VERDICT_LAG_MARGIN_S - 1.0,
    )

    class _Buf:
        def __init__(self, frames):
            self.frames = frames

        def snapshot_last_seconds(self, _s):
            return list(self.frames)

    def _req(frames):
        return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
            detector_worker=SimpleNamespace(_rooms={"r": SimpleNamespace(
                _ingests_by_identity={"cam_x": SimpleNamespace(
                    frame_buffer=_Buf(frames))})}))))

    # Newest wins when no age floor is asked for (the public path).
    assert _latest_frame_jpeg(_req([settled, fresh]), "cam_x") == fresh.data
    # With the floor, the too-fresh frame is refused in favour of one the
    # verdict covers.
    assert _latest_frame_jpeg(
        _req([settled, fresh]), "cam_x",
        min_age_seconds=_PRESENCE_VERDICT_LAG_MARGIN_S,
    ) == settled.data
    # And nothing old enough is a 503, not the fresh frame.
    assert _latest_frame_jpeg(
        _req([fresh]), "cam_x",
        min_age_seconds=_PRESENCE_VERDICT_LAG_MARGIN_S,
    ) is None


async def test_no_verdict_is_not_an_empty_room() -> None:
    """Every way of failing to get an answer must read as None, which
    the endpoint turns into 503. A sidecar that is down, a camera it has
    never heard of, a worker that isn't running, a probe that raises —
    none of them are evidence the bedroom is empty."""
    from src.api.cameras import _tracked_people

    assert await _tracked_people(_request(_FakeReid(None)), CAMERA_ID) is None
    assert await _tracked_people(
        _request(_FakeReid(RuntimeError("state unreachable"))), CAMERA_ID,
    ) is None
    assert await _tracked_people(_request(None), CAMERA_ID) is None
    assert await _tracked_people(_request(object()), CAMERA_ID) is None
