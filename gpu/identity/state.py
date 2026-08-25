"""Per-camera track state mirrored from the pod's persistent session.

The pod owns id continuity (EdgeTAM carry + SAM3 ground); this mirror
just records what is currently live so the identity layer can reason
about births, deaths, solo-ness, and track age.

A track absent from ONE batch is not dead. It used to be, and that
single line was the loudest source of instability in the system:
everything above it — the seat, the identity lock, every downstream
consumer — is chained to a track id, so one noisy batch from the
detector destroyed a face match that had cost 20-60 seconds to earn.
Measured 2026-08-25 in the bedroom, with the resident sitting still:
seat lifetimes of 81s, 6s and 97s, and re-acquisition taking 22, 57 and
443 seconds.

A track now has to be missing for `MISSES_BEFORE_DEATH` consecutive
batches before it dies. SAM3 is still the authority on existence; this
only declines to believe it on the strength of a single frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Bbox = tuple[float, float, float, float]

# Consecutive batches a track may be missing before it is declared dead.
#
# 2 rather than 1, and not more. Batches arrive a few hundred
# milliseconds apart, so this buys roughly half a second of tolerance —
# enough to ride out the single-batch dropouts that were killing seats,
# and far too short to keep a ghost of someone who has actually left.
# The cost of being wrong in this direction is one stale box for one
# extra batch; the cost of the old behaviour was a destroyed identity.
MISSES_BEFORE_DEATH = 2


@dataclass
class TrackView:
    track_id: int
    bbox: Bbox
    born_at: float
    last_seen: float
    batches_seen: int = 1
    # Consecutive batches this track has been absent from. Reset to 0
    # every time it is seen; at MISSES_BEFORE_DEATH the track dies.
    misses: int = 0
    # Recent (wall_ts, bbox) samples, newest last — the follow-crop
    # series (2026-08-18). Capped in apply(); ~15s of history.
    series: tuple[tuple[float, Bbox], ...] = ()

    def age(self, *, now: float) -> float:
        return now - self.born_at


@dataclass(frozen=True)
class TrackEvents:
    births: list[int]
    deaths: list[int]


@dataclass
class CameraState:
    camera_id: str
    tracks: dict[int, TrackView] = field(default_factory=dict)
    updated_at: float = 0.0
    # Atomic snapshot of live ids for cross-thread readers (the seat
    # fold runs in worker threads while apply() mutates `tracks` on the
    # event loop — reading a frozenset ref is GIL-atomic, iterating the
    # live dict is not).
    live_ids: frozenset[int] = field(default_factory=frozenset)

    @property
    def solo(self) -> bool:
        """Exactly one live person on this camera."""
        return len(self.live_ids) == 1

    def apply(self, batch_tracks: dict[int, Bbox], *, now: float,
              batch_series: dict[int, list[tuple[float, Bbox]]] | None = None,
              ) -> TrackEvents:
        """Fold one batch's pod output (tid -> newest bbox) into state.

        SWAP-NOT-MUTATE: a NEW dict is built and assigned in one
        GIL-atomic ref write. Cross-thread readers (seat fold in worker
        threads, the /state HTTP threadpool) grab the ref and iterate
        their own immutable snapshot — in-place mutation here would
        raise "dict changed size during iteration" under them.
        """
        births: list[int] = []
        deaths: list[int] = []
        new_tracks: dict[int, TrackView] = {}
        for tid, bbox in batch_tracks.items():
            t = self.tracks.get(tid)
            fresh = tuple((batch_series or {}).get(tid) or ())
            if t is None:
                t = TrackView(
                    track_id=tid, bbox=bbox, born_at=now, last_seen=now,
                    series=fresh[-120:],
                )
                births.append(tid)
            else:
                t.bbox = bbox
                t.last_seen = now
                t.batches_seen += 1
                # Seen again, so whatever gap it had is over. A track
                # that flickers every other batch must not accumulate
                # its way to a death.
                t.misses = 0
                t.series = tuple(
                    e for e in (*t.series, *fresh) if now - e[0] <= 15.0
                )[-120:]
            new_tracks[tid] = t
        for tid, t in self.tracks.items():
            if tid in batch_tracks:
                continue
            t.misses += 1
            if t.misses >= MISSES_BEFORE_DEATH:
                deaths.append(tid)
            else:
                # Still ours: carried forward with its last known box,
                # so the seat above it survives the gap and every
                # consumer keeps seeing the person who has not moved.
                new_tracks[tid] = t
        self.tracks = new_tracks
        self.live_ids = frozenset(new_tracks)
        self.updated_at = now
        return TrackEvents(births=sorted(births), deaths=sorted(deaths))
