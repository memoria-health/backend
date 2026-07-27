"""Display-veto — rendered people are not people (2026-07-21).

The 8:44 screen-face incident: a face on Aryaman's laptop birthed
person tracks, parked his lock for 11 minutes and got branded OTHER.
Every brain layer behaved correctly; the defect was upstream — a
person rendered ON A DISPLAY must never become a person observation.
Homes have TVs: without this, a living-room TV mints phantom crowds,
OTHER brands, and pool pollution all day.

Measured on the incident frames (probe_continuity.py): SAM concept
"television, laptop, or phone screen" finds displays reliably (stable
region, temporal union covers per-frame misses) and separation is
binary — the phantom's containment in the screen region was 1.0 on
every frame, the real patient maxed at 0.13.

Mechanics: the backend asks the pod for a screen pass every couple of
minutes (screens are furniture); regions accumulate here in a
per-camera temporal-union map with a TTL. A track is vetoed only when
EVERY observed frame of it sits ≥ `containment` inside some live
region — a rendered person is pinned inside the glass on every frame,
while a real person crossing in front of a TV dips out within a clip.
Vetoed tracks are dropped before the brain: no ledger entry, no
binding, no pool harvest, no face checks downstream.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Veto only tracks with at least this many observed frames — a 1-frame
# flicker inside a region is not enough signal to suppress.
_MIN_FRAMES = 2


def containment(inner, outer) -> float:
    """area(inner ∩ outer) / area(inner)."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    a = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (iw * ih) / a


@dataclass
class ScreenMap:
    """Per-camera temporal union of detected display regions."""

    ttl_s: float = 900.0
    containment_floor: float = 0.8
    _regions: list[tuple[tuple[float, float, float, float], float]] = field(
        default_factory=list,
    )

    def add(self, regions, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for r in regions:
            box = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            # Refresh an existing region if this is essentially the
            # same display seen again; else append.
            for i, (old, _ts) in enumerate(self._regions):
                if containment(box, old) > 0.7 and containment(old, box) > 0.7:
                    self._regions[i] = (box, now)
                    break
            else:
                self._regions.append((box, now))
        # Bound the map — a pathological prompt can't grow it forever.
        if len(self._regions) > 16:
            self._regions = sorted(
                self._regions, key=lambda e: e[1], reverse=True,
            )[:16]

    def live(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        return [b for b, ts in self._regions if now - ts <= self.ttl_s]

    def veto_track_ids(
        self, rows_by_frame: dict[int, np.ndarray],
        now: float | None = None,
    ) -> set[int]:
        """Track ids whose EVERY observed bbox is contained in a live
        screen region."""
        live = self.live(now)
        if not live:
            return set()
        per_tid: dict[int, list[float]] = {}
        for rows in rows_by_frame.values():
            for row in rows:
                tid = int(row[4])
                box = (float(row[0]), float(row[1]),
                       float(row[2]), float(row[3]))
                best = max(containment(box, s) for s in live)
                per_tid.setdefault(tid, []).append(best)
        return {
            tid for tid, scores in per_tid.items()
            if len(scores) >= _MIN_FRAMES
            and min(scores) >= self.containment_floor
        }


def apply_veto(obs, screen_map: ScreenMap, *, now: float | None = None):
    """Drop screen-contained tracks from a RemoteObservations bundle
    (rows + face obs). Returns (obs, dropped_tids) — obs is mutated in
    place (it is per-tick and not shared)."""
    dropped = screen_map.veto_track_ids(obs.rows_by_frame, now)
    if not dropped:
        return obs, dropped
    for fi in list(obs.rows_by_frame):
        rows = obs.rows_by_frame[fi]
        keep = np.array(
            [int(r[4]) not in dropped for r in rows], dtype=bool,
        )
        if not keep.all():
            obs.rows_by_frame[fi] = rows[keep]
        if len(obs.rows_by_frame[fi]) == 0:
            del obs.rows_by_frame[fi]
    for key in list(obs.face_obs):
        if key[0] in dropped:
            del obs.face_obs[key]
    log.info(
        "reid.display_veto_dropped tids=%s regions=%d",
        sorted(dropped), len(screen_map.live(now)),
    )
    return obs, dropped
