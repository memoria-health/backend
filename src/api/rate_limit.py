"""Reusable rate limiting for any endpoint.

Audit finding M1: the API had no limiter, no 429 path, nothing. Two
concrete consequences — pairing codes are six digits and therefore
exhaustible, and detector creation drives paid VLM calls with no
per-account cap.

Usage is one line on any route:

    from src.api.rate_limit import RateLimit, RateLimitPolicy

    PAIR_LIMIT = RateLimitPolicy(name="pair", limit=10, window_seconds=60)

    @router.post("/cameras/pair", dependencies=[Depends(RateLimit(PAIR_LIMIT))])
    async def pair_camera(...): ...

The pieces are deliberately separate so each can change alone:

  RateLimitPolicy   what the limit IS       (data, no behavior)
  RateLimitBackend  where counters LIVE     (protocol)
  InMemorySlidingWindow  the default store  (one implementation)
  RateLimit         how it is ENFORCED      (FastAPI dependency)
  client_key        who the caller IS       (pure function)

Swapping the store for Redis when the backend outgrows one replica
means writing a second RateLimitBackend and passing it in — no route,
policy, or test changes. That seam is the reason for the indirection;
today there is exactly one implementation.

SINGLE REPLICA ONLY, like PresenceCache. Counters live in this
process, so N replicas permit N x the limit. That is still a bound,
and prod runs one instance (no Redis in the Railway project), but it
is the reason the backend is an interface rather than a module-level
dict.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request, status

__all__ = [
    "InMemorySlidingWindow",
    "RateLimit",
    "RateLimitBackend",
    "RateLimitPolicy",
    "client_key",
]


@dataclass(frozen=True)
class RateLimitPolicy:
    """What a limit is, with no opinion on how it is stored or enforced.

    `name` namespaces the counters so two policies never share a bucket
    even when the same caller hits both.
    """

    name: str
    limit: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be >= 1")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")


class RateLimitBackend(Protocol):
    """Where counters live. One method, so a Redis implementation is
    small and a fake in tests is trivial.
    """

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Record a hit for `key`; return False if it exceeds the limit."""
        ...

    def retry_after(self, key: str, *, now: float | None = None) -> int:
        """Whole seconds until `key` is allowed again (>= 1)."""
        ...


class InMemorySlidingWindow:
    """Per-key sliding window over hit timestamps.

    A true sliding window rather than a fixed bucket: fixed buckets let
    a caller send `limit` requests at 0:59 and `limit` again at 1:01,
    which is 2x the intended rate through the boundary — precisely the
    burst a brute-forcer wants.

    Idle keys are swept on write. Without that, an attacker rotating
    source IPs turns the limiter itself into a memory-exhaustion
    vector: every new key would allocate a deque that nothing frees.
    """

    __slots__ = ("_hits", "_last_sweep", "_limit", "_lock", "_window")

    # Sweep at most this often; the sweep is O(keys) and callers hit
    # `allow` on every request, so doing it unconditionally would make
    # the hot path scale with the number of distinct clients.
    _SWEEP_INTERVAL_SECONDS = 60.0

    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = limit
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = {}
        # uvicorn runs sync endpoints in a threadpool, so this is
        # reachable from more than one thread even though the async
        # paths share an event loop.
        self._lock = threading.Lock()
        self._last_sweep = 0.0

    def allow(self, key: str, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        cutoff = now - self._window
        with self._lock:
            self._maybe_sweep(now)
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True

    def retry_after(self, key: str, *, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.get(key)
            if not hits:
                return 1
            # The oldest hit is the one whose expiry frees a slot.
            wait = (hits[0] + self._window) - now
        return max(1, int(wait) + 1)

    def tracked_keys(self) -> int:
        """Number of keys currently held. Exposed for tests and future
        diagnostics — not part of RateLimitBackend."""
        with self._lock:
            return len(self._hits)

    def _maybe_sweep(self, now: float) -> None:
        """Drop keys with no hits inside the window. Caller holds the lock."""
        if now - self._last_sweep < self._SWEEP_INTERVAL_SECONDS:
            return
        self._last_sweep = now
        cutoff = now - self._window
        dead = [k for k, h in self._hits.items() if not h or h[-1] <= cutoff]
        for k in dead:
            del self._hits[k]


def client_key(forwarded_for: str | None, peer_host: str | None) -> str:
    """Identify the caller for limiting purposes.

    Railway's edge APPENDS the real client address to X-Forwarded-For,
    so the RIGHT-most entry is the one a proxy vouched for. Reading the
    left-most (the usual "real client IP" idiom) would be a bypass: a
    caller can send any X-Forwarded-For they like, mint a fresh
    identity per request, and never hit the limit.

    Falls back to the socket peer, then to a single shared bucket —
    failing closed onto one bucket is better than failing open.
    """
    if forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return peer_host or "unknown"


class RateLimit:
    """FastAPI dependency enforcing one policy.

    Depends on the RateLimitBackend protocol, not on the in-memory
    store, so a Redis-backed limiter drops in without touching any
    endpoint. Each instance owns its own default store, which is what
    keeps two policies from sharing counters.
    """

    def __init__(
        self,
        policy: RateLimitPolicy,
        backend: RateLimitBackend | None = None,
    ) -> None:
        self._policy = policy
        self._backend: RateLimitBackend = backend or InMemorySlidingWindow(
            limit=policy.limit, window_seconds=policy.window_seconds,
        )

    @property
    def policy(self) -> RateLimitPolicy:
        return self._policy

    @property
    def backend(self) -> RateLimitBackend:
        return self._backend

    async def __call__(self, request: Request) -> None:
        key = f"{self._policy.name}:{self._identify(request)}"
        if self._backend.allow(key):
            return
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate_limited",
            headers={"Retry-After": str(self._backend.retry_after(key))},
        )

    @staticmethod
    def _identify(request: Request) -> str:
        return client_key(
            request.headers.get("x-forwarded-for"),
            request.client.host if request.client else None,
        )
