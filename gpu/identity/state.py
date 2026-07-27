"""Per-camera track state mirrored from the pod's persistent session.

The pod owns id continuity (EdgeTAM carry + SAM3 ground); this mirror
just records what is currently live so the identity layer can reason
about births, deaths, solo-ness, and track age. A track absent from a
batch's output is dead — SAM3 is the authority on existence and the
session clears departed masks (continuous.py "empty-scene clear").
"""

from __future__ import annotations

from dataclasses import dataclass, field

Bbox = tuple[float, float, float, float]


@dataclass
class TrackView:
    track_id: int
    bbox: Bbox
    born_at: float
    last_seen: float
    batches_seen: int = 1

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

    def apply(self, batch_tracks: dict[int, Bbox], *, now: float) -> TrackEvents:
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
            if t is None:
                new_tracks[tid] = TrackView(
                    track_id=tid, bbox=bbox, born_at=now, last_seen=now,
                )
                births.append(tid)
            else:
                t.bbox = bbox
                t.last_seen = now
                t.batches_seen += 1
                new_tracks[tid] = t
        for tid in self.tracks:
            if tid not in batch_tracks:
                deaths.append(tid)
        self.tracks = new_tracks
        self.live_ids = frozenset(new_tracks)
        self.updated_at = now
        return TrackEvents(births=sorted(births), deaths=sorted(deaths))
