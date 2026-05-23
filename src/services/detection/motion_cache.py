"""Per-camera latest motion state cache.

The wall phone keeps motion detection client-side (vImage SAD on the
Y plane — see `nemara-camera-app/NemaraCam/Camera/MotionDetector.swift`)
and publishes its current state to the backend over LiveKit data
packets every ~5s on the `motion_state` topic.

The backend's `RoomWorker.on_data_packet` parses each packet and calls
`MotionCache.put(...)`. The per-camera tick reads `get(...)` and
treats anything missing or older than `ttl_seconds` as
`motion_in_clip=False` — that's the "phone hasn't checked in lately,
assume no motion" semantic that matches what a missed packet would
look like.

Pure in-process state, no I/O. Lives once per `DetectorWorker`
instance and is passed by reference to each `CameraIngest` so the
camera reads its own row without locking (single asyncio event loop ⇒
single-threaded reads + writes; no race window).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class _Entry:
    in_motion: bool
    received_at: float  # monotonic seconds


class MotionCache:
    """Latest-state cache keyed by camera_id.

    A "miss" (no entry) and a "stale hit" (entry older than the TTL)
    are equivalent from the consumer's perspective — both resolve to
    `in_motion=False`. The distinction only matters for diagnostics.
    """

    def __init__(self, *, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be > 0, got {ttl_seconds}"
            )
        self._ttl_seconds = float(ttl_seconds)
        self._entries: dict[UUID, _Entry] = {}

    def put(
        self,
        camera_id: UUID,
        *,
        in_motion: bool,
        received_at: float | None = None,
    ) -> None:
        """Record the camera's freshly-reported motion state.
        `received_at` defaults to `time.monotonic()`; tests pin it.
        """
        ts = time.monotonic() if received_at is None else received_at
        self._entries[camera_id] = _Entry(in_motion=in_motion, received_at=ts)

    def get(
        self,
        camera_id: UUID,
        *,
        now: float | None = None,
    ) -> bool:
        """Return the cached motion state, or False if missing/stale.

        The tick treats False as "no motion in clip" — drives the
        closed-state short-circuit decision. False on missing state
        means a camera that goes quiet is assumed idle, which is the
        safe (LLM-cost-saving) default.
        """
        ref = time.monotonic() if now is None else now
        entry = self._entries.get(camera_id)
        if entry is None:
            return False
        if ref - entry.received_at > self._ttl_seconds:
            return False
        return entry.in_motion

    def drop(self, camera_id: UUID) -> None:
        """Forget the camera's state — called when its `CameraIngest`
        is torn down (participant left).
        """
        self._entries.pop(camera_id, None)

    def __contains__(self, camera_id: UUID) -> bool:
        # Diagnostic convenience — does NOT honor TTL. Use `get` for
        # the consumer-facing "is this camera moving?" question.
        return camera_id in self._entries
