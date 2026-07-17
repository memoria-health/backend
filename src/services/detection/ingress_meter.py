"""Per-camera video ingress telemetry — what the LiveKit SDK actually
DELIVERS to the worker callback, before any decimation.

Soak #1 (2026-07-14, finding 3): the gateway published CFR 15 but the
backend's video callback saw only 4–6 fps, so every RAPID tick ran
`degraded_fps`. The cause was left unproven — SFU/temporal-layer drop
vs event-loop starvation. This meter settles it with numbers stamped
on every analyses row:

  ingress_recv_fps        frames/s the SDK handed our callback
                          (pre-decimation — the disputed boundary)
  ingress_kept_fps        frames/s FrameBuffer accepted after its
                          min-gap decimation (the detector's reality)
  ingress_max_gap_ms      largest inter-arrival gap in the window,
                          including the gap from the last frame to
                          "now" so a stalled stream reads as a
                          growing gap, not a frozen healthy number
  ingress_convert_ms_max  worst per-frame convert+copy cost in our
                          own callback — the event-loop tax we add
                          at native res
  ingress_res             delivered WxH — a drop here means the SFU
                          switched simulcast/adaptive layers
                          (publisher/SFU side), not our loop

How to read it during a soak: smooth-but-slow delivery (low recv_fps,
small max gap, full res) points at the SFU or publisher; normal
average with large gaps points at event-loop starvation on our side
(the A/B is reid on/off); shrunken res means adaptive stream demoted
the layer.

Pure Python, O(1) amortized per frame, no locks — single writer
within the asyncio loop, same contract as FrameBuffer.
"""

from __future__ import annotations

from collections import deque

# Rolling horizon. Covers a full ~10s tick interval with margin so
# every stamped row describes roughly "since the previous row".
WINDOW_SECONDS = 15.0


class IngressMeter:
    """Rolling window over frame arrivals at the SDK→worker boundary."""

    def __init__(self, *, window_seconds: float = WINDOW_SECONDS) -> None:
        self._window = window_seconds
        # (arrived_at, accepted, convert_ms) — ≤ window × delivered fps
        # entries (~450 at 30 fps), so snapshot-time scans stay trivial.
        self._samples: deque[tuple[float, bool, float]] = deque()
        self._res: tuple[int, int] | None = None

    def record(
        self,
        arrived_at: float,
        *,
        accepted: bool,
        convert_ms: float,
        width: int,
        height: int,
    ) -> None:
        """One SDK-delivered frame. `arrived_at` is time.monotonic()."""
        self._samples.append((arrived_at, accepted, convert_ms))
        self._res = (width, height)
        self._prune(arrived_at)

    def snapshot(self, now: float) -> dict[str, object] | None:
        """Row-ready fields, or None when the window holds fewer than
        two arrivals — paths that bypass the video consumer (tests
        pushing straight into FrameBuffer, audio-only cameras) must
        stamp nothing rather than a lying 0.0."""
        self._prune(now)
        n = len(self._samples)
        if n < 2:
            return None
        first_t = self._samples[0][0]
        span = max(now - first_t, 1e-3)
        kept = 0
        convert_max = 0.0
        max_gap = 0.0
        prev_t: float | None = None
        for t, accepted, convert_ms in self._samples:
            if accepted:
                kept += 1
            if convert_ms > convert_max:
                convert_max = convert_ms
            if prev_t is not None and t - prev_t > max_gap:
                max_gap = t - prev_t
            prev_t = t
        # Tail gap: silence since the newest frame counts too.
        if prev_t is not None and now - prev_t > max_gap:
            max_gap = now - prev_t
        w, h = self._res if self._res else (0, 0)
        return {
            "ingress_recv_fps": round(n / span, 2),
            "ingress_kept_fps": round(kept / span, 2),
            "ingress_max_gap_ms": int(max_gap * 1000),
            "ingress_convert_ms_max": round(convert_max, 1),
            "ingress_res": f"{w}x{h}",
        }

    def _prune(self, now: float) -> None:
        cutoff = now - self._window
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
