"""Per-camera ring buffer of recent PCM audio.

Sibling to `FrameBuffer`. The room-worker pushes int16 PCM chunks
(16 kHz mono — `rtc.AudioStream` auto-resamples to whatever the
consumer asks for). The tick reads `snapshot_last_seconds(window)` to
assemble the audio track that goes into the MP4 handed to the LLM.

Sample rate / channel count are fixed at construction; the room-worker
asks LiveKit for one shape and pushes that shape only. Storing those
two as buffer-level fields keeps every chunk identically shaped so the
tick body can pass the concatenated bytes straight to ffmpeg.

Storage cost: 16 kHz x 2 bytes/sample x 1 channel = 32 kB/s. The
default 11s window is ~350 kB per camera — a rounding error vs the
I420 frame buffer's ~60 MB.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioSnapshot:
    """One chunk of int16 PCM captured off the LiveKit audio track.

    `data` is a fresh `bytes` copy (the SDK reuses its own buffer when
    the next chunk arrives, so the room-worker must copy before
    handing the chunk to `AudioBuffer.push`).

    `captured_at` is monotonic seconds (`time.monotonic()`), aligned
    with `FrameSnapshot.captured_at` for cross-modality slicing.
    Sample rate + channel count live on the parent buffer; they're
    invariant per buffer instance.
    """

    data: bytes
    captured_at: float


class AudioBuffer:
    """Bounded ring of recent PCM audio chunks.

    Knobs:
      - `window_seconds`: how much history to keep.
      - `sample_rate` / `num_channels`: fixed shape every chunk must
        match. Defaults are 16 kHz mono, the shape the LLM is happiest
        consuming and the cheapest to encode.

    Capacity is not bounded by chunk count (chunks vary in size — the
    SDK emits ~10-20 ms per event); the time-window trim is the only
    backstop. A pathological flood is bounded by the window: 32 kB/s
    times window_seconds.
    """

    def __init__(
        self,
        *,
        window_seconds: float,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be > 0, got {sample_rate}")
        if num_channels <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}")
        self._window_seconds = float(window_seconds)
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self._chunks: deque[AudioSnapshot] = deque()

    def push(self, *, data: bytes, captured_at: float) -> None:
        """Append a PCM chunk and trim out-of-window chunks."""
        if not data:
            return
        self._chunks.append(AudioSnapshot(data=data, captured_at=captured_at))
        self._trim_older_than(captured_at - self._window_seconds)

    def snapshot_last_seconds(
        self,
        seconds: float,
        *,
        now: float | None = None,
    ) -> bytes:
        """Return concatenated PCM bytes for chunks within the last
        `seconds`. Oldest first (same temporal direction as the audio
        plays back). `now` defaults to `time.monotonic()`; tests pin
        it.
        """
        return self.snapshot_window(seconds, now=now)[0]

    def snapshot_window(
        self,
        seconds: float,
        *,
        now: float | None = None,
    ) -> tuple[bytes, float | None]:
        """Same window as `snapshot_last_seconds`, but ALSO returns the
        capture time of the first chunk included (None when empty).

        Anything aligning this audio against video must use that
        timestamp rather than working the start time back from
        len(pcm) / bytes-per-second. That arithmetic assumes the buffer
        is gapless AND ends exactly at the caller's anchor; a muted
        track, an unpublished track, or an SDK stall breaks both, and
        the mis-inferred offset then displaces the ENTIRE track (16s,
        measured) while still producing a file that looks structurally
        normal — so nothing downstream can detect it.
        """
        if seconds <= 0:
            return b"", None
        ref = time.monotonic() if now is None else now
        cutoff = ref - seconds
        picked = [c for c in self._chunks if c.captured_at >= cutoff]
        if not picked:
            return b"", None
        return b"".join(c.data for c in picked), picked[0].captured_at

    def __len__(self) -> int:
        return len(self._chunks)

    def _trim_older_than(self, cutoff: float) -> None:
        while self._chunks and self._chunks[0].captured_at < cutoff:
            self._chunks.popleft()
