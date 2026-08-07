"""`censor.mask_people_in_still_sync` — the still served to the exit-door
picker for a PRIVATE camera.

Context (2026-08-04): the picker used to subscribe to the camera's LiveKit
track from the client. For a private camera that can never work — privacy
mode restricts subscription to `detector_worker_*` identities — so it waited
six seconds and reported "this sensor isn't streaming right now", which was
false. It was streaming; the caregiver may not watch it.

The fix serves one still from the server. That only stays safe if people are
masked, because a still the client can re-fetch is a slideshow of the room.

The rule this file pins is the one that differs from every other censor path:
NO PERSON FOUND → RETURN THE FRAME UNCHANGED. `_censor_frame` pixelates
full-frame in that case, correctly, because a clip with a sensitive verdict
must not leak a missed body. Here the room IS the subject and full-frame
pixelation would defeat the feature. An empty bedroom is architecture.

Segmentation being unable to RUN is different from finding nobody, and must
still raise so the endpoint fails closed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.services.detection import censor

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")


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


def test_no_person_returns_the_frame_untouched(monkeypatch) -> None:
    """The whole point: an empty room stays fully visible, or the caregiver
    cannot see the doors they came here to tap."""
    monkeypatch.setattr(censor, "_get_seg_model", lambda: object())
    monkeypatch.setattr(censor, "_person_mask", lambda model, frame: None)

    src = _jpeg()
    assert censor.mask_people_in_still_sync(src) == src


def test_empty_mask_also_returns_untouched(monkeypatch) -> None:
    """A mask array that is all zeros means the same thing as None — nobody
    in shot. It must not be treated as 'mask everything'."""
    monkeypatch.setattr(censor, "_get_seg_model", lambda: object())
    monkeypatch.setattr(
        censor, "_person_mask",
        lambda model, frame: np.zeros(frame.shape[:2], dtype=np.uint8),
    )

    src = _jpeg()
    assert censor.mask_people_in_still_sync(src) == src


def test_person_region_is_censored_but_the_room_survives(monkeypatch) -> None:
    """A person is pixelated; pixels well outside the dilated mask are left
    alone so the room stays usable for picking doors."""
    monkeypatch.setattr(censor, "_get_seg_model", lambda: object())

    def fake_mask(model, frame):
        m = np.zeros(frame.shape[:2], dtype=np.uint8)
        m[100:160, 240:300] = 1        # person, far from the "door"
        return m

    monkeypatch.setattr(censor, "_person_mask", fake_mask)

    src = _jpeg()
    before = _decode(src)
    after = _decode(censor.mask_people_in_still_sync(src))

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


def test_segmentation_failure_raises_so_the_caller_fails_closed(monkeypatch) -> None:
    """Cannot-run is NOT the same as nobody-there. A broken CV stack must
    raise, so the endpoint serves 503 rather than an unmasked private room."""
    def boom():
        raise RuntimeError("ultralytics missing")

    monkeypatch.setattr(censor, "_get_seg_model", boom)

    with pytest.raises(Exception):
        censor.mask_people_in_still_sync(_jpeg())


def test_undecodable_input_raises() -> None:
    with pytest.raises(censor.CensorError):
        censor.mask_people_in_still_sync(b"this is not a jpeg")
