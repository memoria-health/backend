"""Person masks read from the perception pod (2026-08-06).

The censor's mask source. EdgeTAM on the pod already computes a full
person mask for every frame of every tick and tracks the person reliably
at seated distance; those masks used to be thrown away and the backend
re-segmented the same pixels with yolo11n-seg, which scored ~0.05 at that
distance. Both censor failure paths treat "no mask" as "cover
everything", so the operator's 60s gate page came back a solid green
rectangle for its whole duration — unjudgeable, so every alert
auto-released. This module is the consumer that closes that loop.

Three pieces:

  `fetch_window` — one HTTP read of GET /masks for a bounded time range,
  returned as a `MaskWindow` that answers "the mask at time T" by
  nearest-frame lookup. Used by the file-based censor, which knows its
  clip's start and length up front and must not fetch per frame.

  `MaskFeed` — a rolling window for the LIVE recorder path, where the
  session runs for up to 900s, the pod only retains 45s, and the mask for
  the frame being encoded right now does not exist yet (it arrives
  ~`lag_s` later). It re-reads a short trailing window on a background
  thread and hands out the freshest mask it has.

  `rle_decode` — the inverse of `continuous.rle_encode` on the pod. The
  two must never drift; a round-trip test pins them against the real
  encoder rather than a copy of it.

CLOCKS. Mask stamps are the capture times the identity service measured
on the backend's own host and sent through /observe, so they are directly
comparable to backend wall clock. Frame buffers time frames with
`time.monotonic()`, so callers convert once via `monotonic_to_epoch` and
everything downstream is epoch.

RESOLUTION. Masks are computed at EdgeTAM's track resolution, which is
not the clip's. Every payload carries the width/height it was recorded
at; decode there, then upscale — the same track-res→frame-res step
`sam_track.mask_crop` does on the pod.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class PodMaskError(RuntimeError):
    """The pod could not be asked for masks (down, unreachable, refused,
    unparseable). Distinct from "asked, and it has none": callers that
    fail closed need to tell an empty room from a blind sidecar."""


# How far from a clip frame a pod mask frame may sit and still be used.
# The identity service ships tracking batches at TARGET_FPS = 3.0, so
# retained masks are ~0.33s apart in the normal case; this is ~3 of those
# gaps, enough to ride out a skipped batch or a pod 429 without reaching
# for a mask old enough to be somewhere else in the room. Beyond it the
# censor's own gap bridge takes over, and beyond THAT the frame is
# covered entirely.
MATCH_TOLERANCE_S = 1.0

# Slack added to both ends of a clip's fetch window. Absorbs the
# decimation offset between the clip's frames and the identity service's
# (they are separate LiveKit subscriptions sampling the same camera), so
# the first and last clip frames still find a neighbour. Costs nothing:
# the pod retains 45s regardless and a whole tick of masks is a few
# hundred KB run-length encoded.
WINDOW_SLACK_S = 3.0

# Ceiling on a clip's fetch window. The pod retains 45s, so asking for
# more than that is free but pointless; this only exists so a caller with
# a bad clip length cannot ask for a decade of masks.
MAX_WINDOW_SPAN_S = 120.0

# Payload sanity caps, mirroring the discipline in identity_client and
# pod_client: a corrupt or hostile response must not allocate unbounded
# backend memory inside a censor child that is already the memory-tight
# part of the system.
_MAX_FRAMES = 2048
_MAX_TRACKS_PER_FRAME = 64
_MAX_MASK_PIXELS = 8192 * 8192


def pod_camera_key(camera_id: object) -> str:
    """The pod's key for a backend camera UUID.

    The identity service keys its cameras by the LiveKit identity suffix
    (`cam_c1d3aeaa` → `c1d3aeaa`, which is the UUID's first 8 hex chars —
    see `identity/contract.py`) and prefixes `identity_` when it calls
    the pod (`identity/loop.py`). Sessions, and therefore retained masks,
    live under that name.
    """
    return f"identity_{str(camera_id)[:8]}"


def monotonic_to_epoch(t_monotonic: float) -> float:
    """Convert a `time.monotonic()` stamp to wall-clock epoch.

    Frame buffers time frames on the monotonic clock (it cannot jump);
    pod masks are stamped in epoch. Both readings are taken here, back to
    back, so the conversion is only as wrong as the gap between them.
    """
    return time.time() - (time.monotonic() - t_monotonic)


def rle_decode(rle: list, width: int, height: int) -> "np.ndarray":
    """Inverse of `gpu/perception/continuous.rle_encode`.

    Row-major run lengths, first run is ZEROs (a mask that starts with a
    1 carries a leading 0-length run so the parity is unambiguous).
    Returns a bool array of shape (height, width).
    """
    import numpy as np

    if width <= 0 or height <= 0:
        raise ValueError(f"bad mask dims {width}x{height}")
    if width * height > _MAX_MASK_PIXELS:
        raise ValueError(f"mask too large: {width}x{height}")
    out = np.zeros(width * height, dtype=bool)
    pos = 0
    value = False
    for run in rle:
        run = int(run)
        if run < 0:
            raise ValueError("negative run length")
        end = pos + run
        if end > out.size:
            raise ValueError("run lengths overflow the mask dimensions")
        if value and run:
            out[pos:end] = True
        pos = end
        value = not value
    # A short encoding leaves the tail zero, which is what a trailing
    # ZERO run would have said anyway — the encoder omits it.
    return out.reshape(height, width)


class MaskWindow:
    """Pod masks for a bounded time range, answered by capture time.

    `frames` is the pod's payload: dicts of {captured_at, width, height,
    masks: {track_id: rle}}, in any order.
    """

    __slots__ = ("_frames",)

    def __init__(self, frames: list[dict]) -> None:
        parsed: list[tuple[float, int, int, list]] = []
        for f in frames[:_MAX_FRAMES]:
            try:
                ts = float(f["captured_at"])
                w, h = int(f["width"]), int(f["height"])
                rles = list((f.get("masks") or {}).values())
            except (KeyError, TypeError, ValueError):
                continue
            if not rles or w <= 0 or h <= 0:
                continue
            parsed.append((ts, w, h, rles[:_MAX_TRACKS_PER_FRAME]))
        parsed.sort(key=lambda e: e[0])
        self._frames = parsed

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def empty(self) -> bool:
        return not self._frames

    def newest(
        self, shape: tuple[int, int],
    ) -> "np.ndarray | None":
        """The most recently captured retained mask, scaled to `shape`.
        The live feed's lookup: it has no frame time to match against,
        only "whatever the pod has managed to compute by now"."""
        if not self._frames:
            return None
        return _union_at(self._frames[-1], shape)

    def mask_at(
        self,
        frame_time: float,
        shape: tuple[int, int],
        *,
        tolerance: float = MATCH_TOLERANCE_S,
    ) -> "np.ndarray | None":
        """Union of every track's mask on the retained frame nearest
        `frame_time`, as a uint8 {0,1} array at `shape` (height, width).
        None when nothing is retained within `tolerance`."""
        nearest = self._nearest(frame_time, tolerance)
        if nearest is None:
            return None
        return _union_at(nearest, shape)

    def match_gap(
        self,
        frame_time: float,
        *,
        tolerance: float = MATCH_TOLERANCE_S,
    ) -> float | None:
        """Seconds between `frame_time` and the mask actually used for
        it, or None when nothing matched.

        The caller needs this to know HOW STALE the mask it just got is.
        Masks arrive at the identity service's tracking cadence (~3fps)
        while clips are encoded at `clip_fps` (8), so most frames are
        masked with a silhouette from up to ~170ms away — enough for a
        walking person to leave it. The old local segmenter ran on the
        frame itself and never had this problem, which is why the
        margins downstream were tuned without it.
        """
        nearest = self._nearest(frame_time, tolerance)
        return None if nearest is None else abs(nearest[0] - frame_time)

    def _nearest(
        self, frame_time: float, tolerance: float,
    ) -> "tuple[float, int, int, list] | None":
        best = None
        best_gap = tolerance
        for entry in self._frames:
            gap = abs(entry[0] - frame_time)
            if gap <= best_gap:
                best, best_gap = entry, gap
            elif best is not None and entry[0] > frame_time:
                break  # sorted: gaps only grow from here
        return best


def _union_at(
    entry: "tuple[float, int, int, list]", shape: tuple[int, int],
) -> "np.ndarray | None":
    """Union one retained frame's tracks and scale it to `shape`.

    Union in FLOAT at the RECORDED dims, resize in float, threshold last.
    Thresholding first and resizing with INTER_NEAREST turns each source
    pixel into a hard rectangular block, which is where a staircase
    silhouette comes from; interpolating the soft mask and cutting at 0.5
    recovers a smooth boundary at no real cost. Masks arrive at EdgeTAM's
    track resolution, so this scaling step is the norm, not the
    exception.
    """
    import cv2
    import numpy as np

    _ts, w, h, rles = entry
    union_f = np.zeros((h, w), dtype=np.float32)
    for rle in rles:
        try:
            union_f = np.maximum(union_f, rle_decode(rle, w, h))
        except ValueError:
            log.warning("pod_masks.bad_rle dims=%dx%d — skipped", w, h)
            continue
    out_h, out_w = shape
    if (h, w) != (out_h, out_w):
        union_f = cv2.resize(
            union_f, (out_w, out_h), interpolation=cv2.INTER_LINEAR,
        )
    union = (union_f > 0.5).astype(np.uint8)
    return union if union.any() else None


def fetch_window(
    camera_id: object,
    t0: float,
    t1: float,
    *,
    base_url: str,
    token: str = "",
    timeout: float = 10.0,
) -> MaskWindow:
    """Read GET /masks once for [t0, t1] (epoch), with slack.

    Raises `PodMaskError` when the pod cannot be asked. An EMPTY window
    is a successful answer meaning "nothing retained", which callers read
    differently depending on what fail-closed means for them.
    """
    import httpx

    if t1 < t0:
        t0, t1 = t1, t0
    t1 = min(t1, t0 + MAX_WINDOW_SPAN_S)
    key = pod_camera_key(camera_id)
    url = f"{base_url.rstrip('/')}/masks/{key}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        resp = httpx.get(
            url,
            params={
                "t0": t0 - WINDOW_SLACK_S, "t1": t1 + WINDOW_SLACK_S,
            },
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — every failure reads the same
        raise PodMaskError(f"pod masks unavailable at {url}: {exc}") from exc
    frames = payload.get("frames") if isinstance(payload, dict) else None
    if not isinstance(frames, list):
        raise PodMaskError(f"pod masks malformed from {url}")
    window = MaskWindow(frames)
    # DEBUG, not info: `MaskFeed` calls this ~3x a second for the whole
    # life of a recording session, so an info line here is a few hundred
    # thousand lines a day per camera. The paths where an empty window
    # actually matters log it themselves, with the context to explain it.
    log.debug(
        "pod_masks.fetched cam=%s span=%.1fs frames=%d",
        key, t1 - t0, len(window),
    )
    return window


class MaskFeed:
    """A rolling `MaskWindow` for the live recorder path.

    The file censor knows its clip's bounds and reads once. A recorder
    session does not: it runs up to 900s, the pod retains 45s of masks,
    and the mask for the frame being encoded right now has not been
    computed yet — it lands ~`lag_s` (≈2.6s live) later. So this polls a
    short trailing window on a background thread and hands out the
    freshest mask available, which the caller dilates to cover the lag.

    A poll failure is not fatal and not silent: `last_error` stays set
    until a read succeeds, and `latest` keeps returning the previous
    window until it ages out. What ages out gets covered entirely, which
    is the same fail-closed rule the rest of the censor uses.
    """

    def __init__(
        self,
        camera_id: object,
        *,
        base_url: str,
        token: str = "",
        timeout: float = 10.0,
        trailing_seconds: float = 10.0,
        interval_seconds: float = 0.33,
    ) -> None:
        self._camera_id = camera_id
        self._base_url = base_url
        self._token = token
        self._timeout = timeout
        self._trailing = trailing_seconds
        self._interval = interval_seconds
        self._lock = threading.Lock()
        self._window = MaskWindow([])
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_error: str | None = "not started"

    def start(self) -> None:
        """Begin polling. Returns immediately — it does NOT wait for the
        first window.

        Blocking here would stall the recorder's encoder child for up to
        the HTTP timeout at the start of every masked session, and the
        pacer feeding it drops frames while it waits. The frames that
        arrive before the first poll lands simply have no mask, and no
        mask means the frame is covered — safe, and over in a fraction of
        a second."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="pod-mask-feed", daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=self._timeout + 1.0)

    def __enter__(self) -> "MaskFeed":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def freshest(self, shape: tuple[int, int]) -> "np.ndarray | None":
        """The newest retained mask, scaled to `shape`. None when the
        feed has nothing recent — the caller covers the frame."""
        with self._lock:
            window = self._window
        return window.newest(shape)

    def _run(self) -> None:
        # Poll FIRST, then wait — `start()` deliberately does not block,
        # so the sooner this lands the fewer frames get covered wholesale.
        self._poll_once()
        while not self._stop.wait(self._interval):
            self._poll_once()

    def _poll_once(self) -> None:
        now = time.time()
        try:
            window = fetch_window(
                self._camera_id, now - self._trailing, now,
                base_url=self._base_url, token=self._token,
                timeout=self._timeout,
            )
        except PodMaskError as exc:
            if self.last_error is None:
                log.warning(
                    "pod_masks.feed_poll_failed — %s; covering frames "
                    "until it recovers", exc,
                )
            self.last_error = str(exc)
            # DROP the carried window. Keeping it would mask where the
            # person was at the moment the pod went away and leave them
            # uncovered wherever they are now — for as long as the outage
            # lasts, which is unbounded. Same rule as the empty case
            # below; the caller's bounded gap bridge absorbs a blip and
            # covers the frame beyond that.
            with self._lock:
                self._window = MaskWindow([])
            return
        self.last_error = None
        with self._lock:
            if not window.empty:
                self._window = window
            else:
                # Nothing retained in the trailing window: the room may
                # genuinely be empty, or the sidecar may have stopped
                # observing. Either way the carried masks are now stale,
                # and a stale mask covers where the person USED to be —
                # worse than useless. Drop them and let the caller cover.
                self._window = MaskWindow([])
