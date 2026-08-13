"""Raw frame wire (perception/rawwire.py) — pack/parse round trip.

The wire replaced the JPEG hop between identity and perception
(2026-08-13, multi-camera CPU wall). One module owns both directions;
these tests are what keeps a format change from shipping half-made.
"""

import numpy as np
import pytest

from perception.rawwire import pack_frames, parse_frames


def _frame(w: int, h: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def test_round_trip_pixels_indices_and_header():
    frames = [(4, _frame(64, 36, 1)), (9, _frame(64, 36, 2))]
    body = pack_frames(
        {"v": 1, "camera_id": "cam_a", "person_thr": 0.4}, frames,
    )
    header, out = parse_frames(body)
    assert header["camera_id"] == "cam_a"
    assert header["person_thr"] == 0.4
    assert [idx for idx, _ in out] == [4, 9]
    for (_, sent), (_, got) in zip(frames, out):
        assert got.shape == sent.shape
        assert np.array_equal(got, sent)


def test_pack_overwrites_caller_frames_table():
    # The table is derived from the arrays — a caller-supplied lie
    # about dims must not survive packing.
    frames = [(0, _frame(32, 16, 3))]
    body = pack_frames({"frames": [[0, 999, 999]]}, frames)
    header, out = parse_frames(body)
    assert header["frames"] == [[0, 32, 16]]
    assert out[0][1].shape == (16, 32, 3)


def test_non_contiguous_input_survives():
    # A downscale/crop can hand pack a strided view; pixels must still
    # arrive intact.
    base = _frame(64, 32, 4)
    view = base[::2, ::2]
    body = pack_frames({}, [(0, view)])
    _, out = parse_frames(body)
    assert np.array_equal(out[0][1], view)


def test_mixed_dims_round_trip():
    frames = [(0, _frame(64, 36, 5)), (1, _frame(32, 18, 6))]
    _, out = parse_frames(pack_frames({}, frames))
    assert out[0][1].shape == (36, 64, 3)
    assert out[1][1].shape == (18, 32, 3)


@pytest.mark.parametrize("mutate", [
    lambda b: b[:4],                                   # truncated length
    lambda b: b[:len(b) - 10],                         # truncated pixels
    lambda b: (2 ** 30).to_bytes(8, "big") + b[8:],    # absurd header len
    lambda b: b[:8] + b"not json" + b[16:],            # broken header
])
def test_malformed_bodies_raise(mutate):
    body = pack_frames({}, [(0, _frame(16, 16, 7))])
    with pytest.raises(ValueError):
        parse_frames(mutate(body))


def test_caps_enforced():
    small = _frame(8, 8, 8)
    body = pack_frames({}, [(i, small) for i in range(3)])
    with pytest.raises(ValueError):
        parse_frames(body, max_frames=2)
    with pytest.raises(ValueError):
        parse_frames(pack_frames({}, [(0, _frame(64, 8, 9))]),
                     max_side_px=32)
