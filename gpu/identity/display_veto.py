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
import os
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Veto only tracks with at least this many observed frames — a 1-frame
# flicker inside a region is not enough signal to suppress.
_MIN_FRAMES = 2

# Region kinds. A screen shows moving content; a painting or statue
# cannot move. That difference decides whether movement is evidence.
KIND_SCREEN = "screen"
KIND_STATIC = "static"


def _drift(boxes: list[tuple]) -> float:
    """Largest distance the box centre travelled across its frames.

    Centre rather than corners: a mask boundary flickers by a pixel or
    two on a static object and both corners wobble with it, while the
    centre of a genuinely still thing stays put.
    """
    if len(boxes) < 2:
        return 0.0
    cx = [(b[0] + b[2]) / 2.0 for b in boxes]
    cy = [(b[1] + b[3]) / 2.0 for b in boxes]
    return float(max(
        ((x - cx[0]) ** 2 + (y - cy[0]) ** 2) ** 0.5
        for x, y in zip(cx, cy)
    ))


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
    # How far a box may wander and still count as furniture. Measured
    # from the live rooms rather than guessed — see the note in
    # veto_track_ids. Tunable because the right value depends on the
    # camera's noise floor, and erring low only costs a phantom track
    # while erring high can discard a person.
    # MEASURED, not guessed. With the TV Room empty, static painted
    # faces drifted 6.5-9.0px at containment 0.89-0.97 — pure mask
    # jitter, no motion. A 6px threshold was therefore sparing the very
    # things it exists to discard. 15px sits above that noise floor and
    # well under any real step.
    move_px: float = float(os.environ.get("NEMARA_VETO_MOVE_PX", "15"))
    # 0.93, raised from 0.8 on 2026-08-24 and MEASURED rather than
    # picked. Every painted face actually being dropped in the TV Room
    # scored 0.957-1.000 containment — the lowest was 0.957 — so the
    # extra margin costs nothing real while making it markedly harder
    # for a person to be caught by accident. A perfectly still human
    # engulfed 93% inside one painted head is not a case this room can
    # produce; at 0.8 it was at least arguable.
    containment_floor: float = 0.93
    _regions: list[tuple[tuple[float, float, float, float], float, str]] = field(
        default_factory=list,
    )

    def add(self, regions, now: float | None = None,
            kind: str = KIND_STATIC) -> None:
        now = time.monotonic() if now is None else now
        for r in regions:
            box = (float(r[0]), float(r[1]), float(r[2]), float(r[3]))
            # Refresh an existing region if this is essentially the
            # same display seen again; else append.
            for i, (old, _ts, old_kind) in enumerate(self._regions):
                if containment(box, old) > 0.7 and containment(old, box) > 0.7:
                    # A region seen as BOTH a screen and something static
                    # keeps the screen kind: screens are the stricter
                    # rule and the one with an incident behind it.
                    keep = (KIND_SCREEN if KIND_SCREEN in (old_kind, kind)
                            else kind)
                    self._regions[i] = (box, now, keep)
                    break
            else:
                self._regions.append((box, now, kind))
        # Bound the map — a pathological prompt can't grow it forever.
        if len(self._regions) > 16:
            self._regions = sorted(
                self._regions, key=lambda e: e[1], reverse=True,
            )[:16]

    def live(self, now: float | None = None):
        now = time.monotonic() if now is None else now
        return [b for b, ts, _k in self._regions if now - ts <= self.ttl_s]

    def live_kinds(self, now: float | None = None):
        """(box, kind) for every live region."""
        now = time.monotonic() if now is None else now
        return [(b, k) for b, ts, k in self._regions if now - ts <= self.ttl_s]

    def veto_track_ids(
        self, rows_by_frame: dict[int, np.ndarray],
        now: float | None = None,
    ) -> set[int]:
        """Track ids whose EVERY observed bbox is contained in a live
        screen region."""
        kinds = self.live_kinds(now)
        live = [b for b, _k in kinds]
        if not live:
            return set()
        per_tid: dict[int, list[float]] = {}
        boxes: dict[int, list[tuple]] = {}
        for rows in rows_by_frame.values():
            for row in rows:
                tid = int(row[4])
                box = (float(row[0]), float(row[1]),
                       float(row[2]), float(row[3]))
                best = max(containment(box, s) for s in live)
                per_tid.setdefault(tid, []).append(best)
                boxes.setdefault(tid, []).append(box)
        vetoed = set()
        for tid, scores in per_tid.items():
            if len(scores) < _MIN_FRAMES:
                continue
            if min(scores) < self.containment_floor:
                continue
            # WHICH KIND of region is this track sitting in? A SCREEN
            # can show a person who walks about — that is the whole 8:44
            # incident, a face on a laptop minting tracks — so movement
            # proves nothing there and the containment rule stands alone.
            # A painting or a statue cannot move, so movement there is
            # real evidence of a real thing.
            best_kind = KIND_STATIC
            best_c = -1.0
            for box in boxes[tid]:
                for region, kind in kinds:
                    c = containment(box, region)
                    if c > best_c:
                        best_c, best_kind = c, kind
            drift = _drift(boxes[tid])
            if best_kind == KIND_SCREEN:
                vetoed.add(tid)
                continue
            if drift > self.move_px:
                # IT MOVED, SO IT IS NOT ART. The camera has no depth —
                # one lens, no stereo, no sensor — so a person standing
                # against a painted wall and a face painted ON it are
                # the same pixels, and containment alone cannot separate
                # them. Movement can: a painting is nailed to a wall and
                # a person cannot hold still to the pixel.
                #
                # Deliberately the SAFE direction. A false "it moved"
                # keeps a real person and costs one phantom track; a
                # false "it is art" discards a resident standing in
                # front of their own wall, and nothing downstream would
                # ever know they were there.
                log.info(
                    "reid.display_veto_spared tid=%s drift=%.1fpx "
                    "containment=%.2f — moved, so not art",
                    tid, drift, min(scores),
                )
                continue
            log.info(
                "reid.display_veto_drop tid=%s kind=%s containment=%.3f "
                "drift=%.1fpx",
                tid, best_kind, min(scores), drift,
            )
            vetoed.add(tid)
        return vetoed


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
