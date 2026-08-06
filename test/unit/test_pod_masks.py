"""`src.services.detection.pod_masks` — the censor's mask source.

The pod computes person masks, run-length encodes them, and serves them
from `GET /masks`; the censor decodes them and covers people with them.
Encoder and decoder live in two different deployment targets (the pod
ships `gpu/perception/` only), so the thing that actually protects this
is a round-trip against the REAL encoder — imported here from
`gpu.perception.continuous`, never reimplemented. If the two ever drift,
the censor covers whole frames green and every gate alert auto-releases
unjudged, which is exactly the failure this module was written to end.

Also pinned:

  - decode happens at the RECORDED dims and is upscaled afterwards.
    Masks arrive at EdgeTAM's track resolution; decoding at the clip's
    dims would reshape the run lengths into garbage.
  - window lookup picks the NEAREST retained frame, and refuses one
    that is too far away rather than covering the wrong place.
  - an unreachable pod raises `PodMaskError`; a reachable pod with
    nothing retained returns an EMPTY window. Callers must be able to
    tell those apart — the door-frame still serves a room for one and
    503s for the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu.perception.continuous import rle_encode
from src.services.detection import pod_masks

cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")


def _blob(h: int, w: int, box=(10, 20, 40, 60)) -> np.ndarray:
    y0, x0, y1, x1 = box
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


# ── RLE round-trip against the pod's real encoder ────────────────────


@pytest.mark.parametrize("mask", [
    np.zeros((8, 8), dtype=bool),                       # all background
    np.ones((8, 8), dtype=bool),                        # all foreground
    _blob(48, 64),                                      # a person-ish blob
    np.eye(16, dtype=bool),                             # every run length 1
])
def test_round_trip_is_lossless(mask) -> None:
    h, w = mask.shape
    out = pod_masks.rle_decode(rle_encode(mask), w, h)
    assert out.shape == mask.shape
    assert np.array_equal(out, mask)


def test_a_mask_starting_with_one_round_trips() -> None:
    """The encoder's convention is 'first run is ZEROs', so a mask whose
    very first pixel is set emits a leading zero-length run. Getting the
    parity wrong here inverts the entire mask — everything BUT the
    person gets covered — which still looks like 'a mask', so nothing
    downstream would catch it."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    encoded = rle_encode(mask)
    assert encoded[0] == 0, "leading zero-run is the parity marker"
    assert np.array_equal(pod_masks.rle_decode(encoded, 4, 4), mask)


def test_a_truncated_tail_decodes_as_background() -> None:
    """The encoder omits a trailing zero run. The decoder must read that
    as background, not as an error or as coverage."""
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, :2] = True
    assert np.array_equal(pod_masks.rle_decode(rle_encode(mask), 4, 4), mask)


@pytest.mark.parametrize("bad", [
    ([100], 4, 4),          # runs overflow the frame
    ([-1, 2], 4, 4),        # negative run
    ([1], 0, 4),            # zero dimension
])
def test_malformed_input_raises_rather_than_guessing(bad) -> None:
    rle, w, h = bad
    with pytest.raises(ValueError):
        pod_masks.rle_decode(rle, w, h)


# ── decode at recorded dims, then upscale ────────────────────────────


def _frame(ts: float, mask: np.ndarray, w: int, h: int) -> dict:
    return {
        "captured_at": ts, "width": w, "height": h,
        "masks": {"1": rle_encode(mask)},
    }


def test_recorded_dims_come_from_the_mask_not_the_frame() -> None:
    """The pod records masks at EdgeTAM's TRACK resolution while the
    frame is full-res, and `record_masks` used to take the dimensions as
    a parameter — which the caller filled in from the frame. Every mask
    was then labelled 1920x1080 while actually being 640x360, so
    decoding put the whole silhouette in a band across the top of the
    frame: person-sized coverage, nowhere near the person, and nothing
    downstream able to notice (a mask is just a mask). Found 2026-08-06
    by rendering one, not by reading numbers.

    Dimensions now come off the array, so the disagreement cannot be
    expressed. This drives the real `record_masks`.
    """
    from gpu.perception import continuous

    continuous.forget_masks("dimcheck")
    track_res = _blob(360, 640, box=(100, 200, 300, 400))
    continuous.record_masks("dimcheck", 5000.0, {7: track_res})
    frames = continuous.masks_between("dimcheck", 4999.0, 5001.0)
    continuous.forget_masks("dimcheck")

    assert len(frames) == 1
    assert (frames[0]["width"], frames[0]["height"]) == (640, 360)

    # And it round-trips through the consumer to the same silhouette.
    out = pod_masks.MaskWindow(frames).mask_at(5000.0, (360, 640))
    assert out is not None
    assert np.array_equal(out.astype(bool), track_res)


def test_a_mask_of_the_wrong_shape_is_dropped_not_mislabelled() -> None:
    """One odd mask must not be encoded against another's dimensions —
    that is the same class of failure, one track at a time."""
    from gpu.perception import continuous

    continuous.forget_masks("dimcheck2")
    continuous.record_masks("dimcheck2", 5000.0, {
        1: _blob(360, 640, box=(10, 10, 50, 50)),
        2: _blob(180, 320, box=(5, 5, 25, 25)),      # different shape
    })
    frames = continuous.masks_between("dimcheck2", 4999.0, 5001.0)
    continuous.forget_masks("dimcheck2")
    assert list(frames[0]["masks"]) == ["1"]


def test_mask_is_decoded_at_recorded_dims_and_upscaled() -> None:
    """Track resolution in, frame resolution out. Decoding at the output
    dims would reshape the runs into a diagonal smear, so this checks
    the region actually lands where it should."""
    small = _blob(60, 80, box=(15, 20, 30, 40))
    window = pod_masks.MaskWindow([_frame(1000.0, small, 80, 60)])

    out = window.mask_at(1000.0, (240, 320))
    assert out is not None
    assert out.shape == (240, 320)
    # The blob quadruples: rows 60-120, cols 80-160.
    assert out[80:110, 100:150].all()
    assert not out[0:50, 0:50].any()


def test_tracks_are_unioned() -> None:
    """Two people in the room means both get covered — the censor masks
    every person, not just the patient (dignity applies to a caregiver
    mid-bathing-assist too)."""
    a, b = _blob(48, 64, (2, 2, 10, 10)), _blob(48, 64, (30, 40, 44, 60))
    window = pod_masks.MaskWindow([{
        "captured_at": 1000.0, "width": 64, "height": 48,
        "masks": {"1": rle_encode(a), "2": rle_encode(b)},
    }])
    out = window.mask_at(1000.0, (48, 64))
    assert out is not None
    assert out[5, 5] and out[35, 50]


def test_an_all_background_frame_reads_as_no_mask() -> None:
    window = pod_masks.MaskWindow([
        _frame(1000.0, np.zeros((48, 64), dtype=bool), 64, 48),
    ])
    assert window.mask_at(1000.0, (48, 64)) is None


# ── window indexing ──────────────────────────────────────────────────


def _three_frames() -> pod_masks.MaskWindow:
    return pod_masks.MaskWindow([
        _frame(1002.0, _blob(48, 64, (30, 30, 40, 40)), 64, 48),
        _frame(1000.0, _blob(48, 64, (0, 0, 10, 10)), 64, 48),
        _frame(1001.0, _blob(48, 64, (15, 15, 25, 25)), 64, 48),
    ])


@pytest.mark.parametrize("t,probe", [
    (1000.0, (5, 5)),
    (1000.4, (5, 5)),      # rounds down to 1000
    (1001.0, (20, 20)),
    (1001.6, (35, 35)),    # rounds up to 1002
    (1002.0, (35, 35)),
])
def test_nearest_frame_wins_regardless_of_arrival_order(t, probe) -> None:
    """Frames arrive out of order (the pod's retention tolerates it), so
    the lookup sorts rather than trusting position."""
    out = _three_frames().mask_at(t, (48, 64))
    assert out is not None and out[probe]


def test_a_frame_beyond_tolerance_is_not_used() -> None:
    """Better to cover the whole frame than to cover where the person
    WAS. A stale mask leaves them uncovered where they now are, which is
    the one outcome worse than over-covering."""
    window = _three_frames()
    assert window.mask_at(1000.0 - 60.0, (48, 64)) is None
    assert window.mask_at(1002.0 + 60.0, (48, 64)) is None


def test_newest_ignores_the_clock() -> None:
    """The live recorder path has no frame time to match — the mask for
    the frame being encoded has not been computed yet. It takes whatever
    the pod has most recently managed."""
    out = _three_frames().newest((48, 64))
    assert out is not None and out[35, 35]


def test_an_empty_window_answers_nothing_rather_than_everything() -> None:
    empty = pod_masks.MaskWindow([])
    assert empty.empty
    assert empty.mask_at(1000.0, (48, 64)) is None
    assert empty.newest((48, 64)) is None


def test_malformed_frames_are_skipped_not_fatal() -> None:
    """A corrupt payload must not take the censor down — it degrades to
    'no mask here', which covers the frame."""
    good = _frame(1000.0, _blob(48, 64, (0, 0, 10, 10)), 64, 48)
    window = pod_masks.MaskWindow([
        {"captured_at": "not a number", "width": 64, "height": 48, "masks": {}},
        {"width": 64, "height": 48, "masks": {"1": [0, 4]}},   # no timestamp
        {"captured_at": 999.0, "width": 0, "height": 48, "masks": {"1": [1]}},
        good,
    ])
    assert len(window) == 1
    assert window.mask_at(1000.0, (48, 64)) is not None


# ── camera key + clock conversion ────────────────────────────────────


def test_pod_camera_key_matches_the_identity_service() -> None:
    """The pod keys sessions by `identity_<8 hex>` (identity/loop.py
    sends `identity_{camera_id}`, where the id is the LiveKit identity
    suffix = the UUID's first 8 hex chars). Getting this wrong returns
    an empty window forever, which looks exactly like 'nobody home'."""
    assert pod_masks.pod_camera_key(
        "cafe1234-1111-4000-8000-000000000000"
    ) == "identity_cafe1234"


def test_monotonic_to_epoch_preserves_intervals() -> None:
    """Frame buffers time frames on the monotonic clock, pod masks are
    stamped in epoch. Absolute values differ by an arbitrary offset; the
    thing that must survive is the spacing between frames."""
    import time

    now_mono = time.monotonic()
    a = pod_masks.monotonic_to_epoch(now_mono - 10.0)
    b = pod_masks.monotonic_to_epoch(now_mono)
    assert b - a == pytest.approx(10.0, abs=0.05)
    assert abs(b - time.time()) < 0.5


# ── fetch: unreachable vs. nothing retained ──────────────────────────


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status != 200:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


def test_fetch_parses_frames_and_targets_the_pod_key(monkeypatch) -> None:
    seen = {}

    def fake_get(url, **kw):
        seen["url"] = url
        seen["params"] = kw["params"]
        seen["headers"] = kw["headers"]
        return _Resp({"count": 1, "frames": [
            _frame(1000.0, _blob(48, 64), 64, 48),
        ]})

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)
    window = pod_masks.fetch_window(
        "cafe1234-1111-4000-8000-000000000000", 1000.0, 1010.0,
        base_url="http://pod.invalid/", token="hunter2",
    )
    assert len(window) == 1
    assert seen["url"] == "http://pod.invalid/masks/identity_cafe1234"
    assert seen["headers"]["Authorization"] == "Bearer hunter2"
    # Slack on both ends so the first and last clip frames still find a
    # neighbour despite the two services decimating differently.
    assert seen["params"]["t0"] < 1000.0
    assert seen["params"]["t1"] > 1010.0


def test_no_token_sends_no_auth_header(monkeypatch) -> None:
    """The pod runs open today (loopback, tunnel-only). An empty token
    must send NO header rather than an empty bearer, so both sides agree
    on what 'unset' means."""
    seen = {}

    def fake_get(url, **kw):
        seen["headers"] = kw["headers"]
        return _Resp({"frames": []})

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)
    pod_masks.fetch_window(
        "cafe1234", 1000.0, 1010.0, base_url="http://pod.invalid",
    )
    assert seen["headers"] == {}


def test_an_unreachable_pod_raises(monkeypatch) -> None:
    """Distinct from 'nothing retained'. The door-frame still serves an
    empty room for one of these and 503s for the other."""
    def boom(url, **kw):
        raise OSError("connection refused")

    import httpx
    monkeypatch.setattr(httpx, "get", boom)
    with pytest.raises(pod_masks.PodMaskError):
        pod_masks.fetch_window(
            "cafe1234", 1000.0, 1010.0, base_url="http://pod.invalid",
        )


def test_a_reachable_pod_with_nothing_retained_is_an_empty_window(
    monkeypatch,
) -> None:
    import httpx
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: _Resp({"count": 0, "frames": []}),
    )
    window = pod_masks.fetch_window(
        "cafe1234", 1000.0, 1010.0, base_url="http://pod.invalid",
    )
    assert window.empty


def test_a_malformed_payload_raises(monkeypatch) -> None:
    """A pod serving something other than the mask contract is a broken
    pod, not an empty room."""
    import httpx
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: _Resp({"oops": True}),
    )
    with pytest.raises(pod_masks.PodMaskError):
        pod_masks.fetch_window(
            "cafe1234", 1000.0, 1010.0, base_url="http://pod.invalid",
        )


def test_the_fetch_span_is_bounded(monkeypatch) -> None:
    """A bad clip_start must not turn into a request for a decade of
    masks. The pod only retains 45s anyway."""
    seen = {}

    def fake_get(url, **kw):
        seen["params"] = kw["params"]
        return _Resp({"frames": []})

    import httpx
    monkeypatch.setattr(httpx, "get", fake_get)
    pod_masks.fetch_window(
        "cafe1234", 1000.0, 1000.0 + 10_000_000,
        base_url="http://pod.invalid",
    )
    span = seen["params"]["t1"] - seen["params"]["t0"]
    assert span <= pod_masks.MAX_WINDOW_SPAN_S + 2 * pod_masks.WINDOW_SLACK_S


# ── the live feed ────────────────────────────────────────────────────


def test_feed_drops_a_stale_window_rather_than_carrying_it(
    monkeypatch,
) -> None:
    """A mask that has aged out covers where the person USED to be,
    which is worse than covering nothing — the caller's bounded bridge
    then covers the frame entirely."""
    windows = [
        pod_masks.MaskWindow([_frame(1000.0, _blob(48, 64), 64, 48)]),
        pod_masks.MaskWindow([]),
    ]
    monkeypatch.setattr(
        pod_masks, "fetch_window", lambda *a, **kw: windows.pop(0),
    )
    feed = pod_masks.MaskFeed("cafe1234", base_url="http://pod.invalid")
    feed._poll_once()
    assert feed.freshest((48, 64)) is not None
    feed._poll_once()
    assert feed.freshest((48, 64)) is None


def test_feed_survives_a_poll_failure_and_records_it(monkeypatch) -> None:
    """A dead pod must not kill a recording session; it must make every
    frame get covered, and say so once."""
    def boom(*a, **kw):
        raise pod_masks.PodMaskError("tunnel is dead")

    monkeypatch.setattr(pod_masks, "fetch_window", boom)
    feed = pod_masks.MaskFeed("cafe1234", base_url="http://pod.invalid")
    feed._poll_once()  # must not raise
    assert feed.last_error is not None
    assert feed.freshest((48, 64)) is None


def test_a_poll_failure_drops_the_carried_window(monkeypatch) -> None:
    """A pod outage must not freeze the last mask in place.

    Holding it would cover where the person WAS when the pod went away
    and leave them uncovered wherever they are now, for the whole length
    of the outage — unbounded, and exactly backwards. The failure path
    has to decay to 'no mask' like the empty-room path does.
    """
    state = {"fail": False}

    def flaky(*a, **kw):
        if state["fail"]:
            raise pod_masks.PodMaskError("tunnel is dead")
        return pod_masks.MaskWindow([
            _frame(1000.0, _blob(48, 64), 64, 48),
        ])

    monkeypatch.setattr(pod_masks, "fetch_window", flaky)
    feed = pod_masks.MaskFeed("cafe1234", base_url="http://pod.invalid")
    feed._poll_once()
    assert feed.freshest((48, 64)) is not None
    state["fail"] = True
    feed._poll_once()
    assert feed.freshest((48, 64)) is None


def test_start_does_not_block_on_the_first_poll(monkeypatch) -> None:
    """`start()` runs on the recorder's path to launching its encoder
    child. Waiting for HTTP there stalls the pacer feeding that child and
    it drops frames — so the first poll belongs on the thread."""
    import threading as _t
    import time as _t2

    released = _t.Event()

    def slow(*a, **kw):
        released.wait(5.0)
        return pod_masks.MaskWindow([])

    monkeypatch.setattr(pod_masks, "fetch_window", slow)
    feed = pod_masks.MaskFeed("cafe1234", base_url="http://pod.invalid")
    t0 = _t2.monotonic()
    feed.start()
    elapsed = _t2.monotonic() - t0
    released.set()
    feed.close()
    assert elapsed < 0.5, f"start() blocked for {elapsed:.2f}s"
