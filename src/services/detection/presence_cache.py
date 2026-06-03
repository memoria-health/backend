"""In-memory 'is the Memoria Sensor app alive?' heartbeat cache.

Every paired iPhone sensor polls `GET /v1/cameras/me/desired-state`
every 1s the whole time the Memoria Sensor app is running
(`DesiredStateMonitor` on the phone, NOT tied to Start/Stop). That poll
is our "app is open" signal. The poll handler calls `touch(camera_id)`
— pure in-memory, NO DB write — so the 1s hot path never touches the
connection pool (a per-poll write is what exhausted it before).

`PresenceSweeper` reads `alive_ids()` every few seconds and flips the
`cameras.app_present` column only on transitions. A camera unseen for
longer than `stale_seconds` is "app closed" (force-quit, phone off,
crashed) — distinct from "disconnected but app still running" (which
keeps polling and stays present).

Pi cameras never reach the desired-state endpoint (they auth with an
ingest token, not an api_key), so they never appear here — which is
correct: a Pi has no app to close.

Pure in-process state. A single asyncio event loop ⇒ single-threaded
reads + writes, no lock needed. Modeled on `motion_cache.py`.

SINGLE REPLICA ONLY: this lives in one process's memory. Prod runs a
single uvicorn process / single Railway replica, so the poll endpoint
and the sweep share the same cache. If the backend is ever scaled to
>1 replica, polls would round-robin across processes and each process's
sweep would see only its share → false "app closed" flapping. Move this
to Redis or a throttled DB `last_seen_at` before scaling horizontally.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True)
class _Entry:
    received_at: float  # monotonic seconds


class PresenceCache:
    """Last-heartbeat cache keyed by camera_id.

    A "miss" (never polled) and a "stale hit" (last poll older than
    `stale_seconds`) are equivalent — both mean "app not alive".
    """

    def __init__(self, *, stale_seconds: float) -> None:
        if stale_seconds <= 0:
            raise ValueError(
                f"stale_seconds must be > 0, got {stale_seconds}"
            )
        self._stale_seconds = float(stale_seconds)
        self._entries: dict[UUID, _Entry] = {}

    def touch(self, camera_id: UUID, *, received_at: float | None = None) -> None:
        """Record a heartbeat for this camera. `received_at` defaults to
        `time.monotonic()`; tests pin it.
        """
        ts = time.monotonic() if received_at is None else received_at
        self._entries[camera_id] = _Entry(received_at=ts)

    def is_alive(self, camera_id: UUID, *, now: float | None = None) -> bool:
        """True iff this camera polled within `stale_seconds`."""
        ref = time.monotonic() if now is None else now
        entry = self._entries.get(camera_id)
        if entry is None:
            return False
        return ref - entry.received_at <= self._stale_seconds

    def alive_ids(self, *, now: float | None = None) -> set[UUID]:
        """The set of cameras currently considered app-alive."""
        ref = time.monotonic() if now is None else now
        return {
            cid
            for cid, entry in self._entries.items()
            if ref - entry.received_at <= self._stale_seconds
        }

    def seed(
        self, camera_ids: Iterable[UUID], *, received_at: float | None = None
    ) -> None:
        """Startup grace: mark these cameras alive WITHOUT overwriting a
        fresher live entry (`setdefault`). Used so a backend restart
        doesn't flash every paired iPhone sensor to "app closed" before
        its next 1s poll lands — genuinely-closed ones go stale within
        `stale_seconds` and the next sweep corrects them.
        """
        ts = time.monotonic() if received_at is None else received_at
        for camera_id in camera_ids:
            self._entries.setdefault(camera_id, _Entry(received_at=ts))

    def drop(self, camera_id: UUID) -> None:
        """Forget a camera (e.g. on unpair/delete)."""
        self._entries.pop(camera_id, None)

    def __contains__(self, camera_id: UUID) -> bool:
        # Diagnostic convenience — does NOT honor staleness. Use
        # `is_alive` for the consumer-facing question.
        return camera_id in self._entries
