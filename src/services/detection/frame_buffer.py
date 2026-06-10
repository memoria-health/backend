"""Per-camera ring buffer of recent video frames.

Owned by `CameraIngest`; the room-worker's `VideoStream` consumer
pushes each subscribed frame in, the tick reads
`snapshot_last_seconds(window)` to assemble the LLM input.

Design notes:

- **Pure data structure.** No LiveKit types in the signature — the
  room-worker is responsible for converting `rtc.VideoFrame` to the
  I420 byte copy this buffer stores. Lets the tests pass synthetic
  bytes without spinning up the SDK.
- **Time-based downsampling.** `target_fps` translates to a minimum
  gap between accepted frames. Frames that arrive before the gap
  elapses are dropped at push time — cheaper than buffering at the
  publish rate and decimating later.
- **Time-based trim.** Every push drops frames older than
  `window_seconds`, keeping the buffer at most
  ``window_seconds * target_fps + 1`` entries.
- **I420 storage cost.** A 720p I420 frame is ~1.4 MB; at the default
  4 fps × 11 s that's ~62 MB per camera in steady state. Acceptable for v0 (one
  or two cameras per patient on a backend with modest RAM); revisit if
  load tests show pressure (downsample to 480p on push, or store JPEGs).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSnapshot:
    """One frame captured off the LiveKit video track.

    `data` is a fresh `bytes` copy (memoryview lifetime in the SDK
    expires when the next frame arrives, so callers must copy before
    handing the frame to `FrameBuffer.push`).

    `captured_at` is monotonic seconds (`time.monotonic()`), not
    wall-clock. Used only for relative-window math inside the buffer;
    `analyses.captured_at` on the DB side is wall-clock and set
    separately by the tick.
    """

    width: int
    height: int
    data: bytes
    captured_at: float


class FrameBuffer:
    """Bounded ring of recent I420 frames.

    Two knobs:
      - `window_seconds`: how much history to keep (e.g. 11s).
      - `target_fps`: sampling rate after publish-side downsample.

    Capacity is ``window_seconds * target_fps + 1`` frames (the +1
    covers the slack between the last drop and the next push).
    """

    def __init__(self, *, window_seconds: float, target_fps: int) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        self._window_seconds = float(window_seconds)
        self._min_gap_seconds = 1.0 / target_fps
        self._capacity = int(window_seconds * target_fps) + 1
        # `maxlen` enforces the hard cap as a safety net; the
        # time-based `_trim_older_than` is the primary trim mechanism.
        self._frames: deque[FrameSnapshot] = deque(maxlen=self._capacity)
        self._last_accepted_at: float = float("-inf")

    def push(
        self,
        *,
        width: int,
        height: int,
        data: bytes,
        captured_at: float,
    ) -> bool:
        """Accept or drop one frame.

        Returns True if the frame was accepted, False if it was
        dropped to honor `target_fps`. Callers don't normally need
        the return value — it exists for the test that asserts the
        sampler actually drops frames between samples.
        """
        if captured_at - self._last_accepted_at < self._min_gap_seconds:
            return False
        self._frames.append(
            FrameSnapshot(
                width=width, height=height, data=data, captured_at=captured_at,
            )
        )
        self._last_accepted_at = captured_at
        self._trim_older_than(captured_at - self._window_seconds)
        return True

    def snapshot_last_seconds(
        self,
        seconds: float,
        *,
        now: float | None = None,
    ) -> list[FrameSnapshot]:
        """Return frames with `captured_at >= now - seconds`, oldest
        first. `now` defaults to `time.monotonic()`; tests can pin it.
        """
        if seconds <= 0:
            return []
        ref = time.monotonic() if now is None else now
        cutoff = ref - seconds
        return [f for f in self._frames if f.captured_at >= cutoff]

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def capacity(self) -> int:
        return self._capacity

    def _trim_older_than(self, cutoff: float) -> None:
        # Frames in the deque are time-ordered; pop from the left
        # until the head is within the window.
        while self._frames and self._frames[0].captured_at < cutoff:
            self._frames.popleft()
