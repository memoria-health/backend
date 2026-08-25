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

import json
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# Veto only tracks with at least this many observed frames — a 1-frame
# flicker inside a region is not enough signal to suppress.
# Bumped when the on-disk shape changes meaning. v1 held PIXELS; v2
# fractions of the frame; v3 drops the per-region `kind`, which existed
# only to decide who got the movement exemption that no longer exists.
# An older file is DISCARDED rather than reinterpreted: reading pixels
# as fractions vetoes nothing and looks exactly like a room with no
# displays in it.
_REGION_FORMAT = 3

# How close two regions must be to count as one object, and how much to
# grow the result, both as a fraction of the frame.
#
# Concept grounding returns an artwork PANEL BY PANEL, not as one thing.
# The TV Room's canvas came back as seven boxes tiling two rows, and the
# eighth — the bottom-right face — was simply missed, so it fell in the
# gap between the last panel and the frame edge and was tracked as a
# person indefinitely. Asking for a better prompt is a vocabulary
# problem with no end; joining the panels that touch is arithmetic.
#
# The dilation exists because grounding boxes are tight. Clustered but
# undilated, the missed face was 0.900 contained against a 0.93 floor —
# short by a hundredth. At 0.02 it is fully covered.
#
# Clustering deliberately does NOT merge everything: the small framed
# picture by the door sits alone at x 0.50 and stays its own region, so
# this cannot blanket half a room.
_ADJACENT_GAP = 0.01
_DILATE = 0.02

_MIN_FRAMES = 2




def _to_frac(box, frame_wh: tuple[int, int] | None):
    """Pixel box -> fraction of the frame. None means already a fraction.

    EVERYTHING in this module is normalised, because the frame size is
    not a constant. The SFU hands a subscriber whatever simulcast rung
    it judges affordable, so the same wall arrives as 1920x1080 one
    minute and 320x180 the next — measured 866 times in one window on
    the TV Room.

    Comparing a region grounded at one size against a track box measured
    at another gives containment near zero, so the veto silently stops
    vetoing and the painted faces come back as people. That is exactly
    the "churning between identifying those faces and cancelling them
    out" reported 2026-08-25, and it is why the persisted region file
    made things worse rather than better: it carried whatever resolution
    happened to be live when the regions were grounded.

    In fractions the wall is in the same place at every rung.
    """
    if not frame_wh:
        return box
    w, h = frame_wh
    if not w or not h:
        return box
    return (box[0] / w, box[1] / h, box[2] / w, box[3] / h)


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


def _touching(a, b, gap: float) -> bool:
    return (a[0] <= b[2] + gap and b[0] <= a[2] + gap
            and a[1] <= b[3] + gap and b[1] <= a[3] + gap)


def cluster_regions(boxes, gap: float = _ADJACENT_GAP, grow: float = _DILATE):
    """Join regions that touch into one, then grow each a little.

    An artwork is one object and grounding returns it as many. Joining
    the pieces that touch turns seven panel boxes into one canvas, which
    then also covers the panel grounding missed.
    """
    out = [tuple(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                if _touching(out[i], out[j], gap):
                    out[i] = _union(out[i], out[j])
                    del out[j]
                    merged = True
                    break
            if merged:
                break
    return [(max(0.0, b[0] - grow), max(0.0, b[1] - grow),
             min(1.0, b[2] + grow), min(1.0, b[3] + grow)) for b in out]


def containment(inner, outer) -> float:
    """area(inner ∩ outer) / area(inner)."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    # A tiny floor, not 1.0. In fractions a whole-frame box has area
    # 1.0, so the old pixel-era floor made every containment come out as
    # the intersection area and the veto never fired.
    a = max(1e-9, (inner[2] - inner[0]) * (inner[3] - inner[1]))
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
    # A FRACTION OF FRAME WIDTH, not pixels. Everything in this map is
    # normalised (see `add`), so a pixel threshold would mean a
    # different physical distance at every rung the SFU hands us.
    # 15px measured at 2304 wide = 0.0065; the env var stays in pixels
    # against that reference so the tuning history still reads.
    move_frac: float = (
        float(os.environ.get("NEMARA_VETO_MOVE_PX", "15")) / 2304.0)
    # 0.93, raised from 0.8 on 2026-08-24 and MEASURED rather than
    # picked. Every painted face actually being dropped in the TV Room
    # scored 0.957-1.000 containment — the lowest was 0.957 — so the
    # extra margin costs nothing real while making it markedly harder
    # for a person to be caught by accident. A perfectly still human
    # engulfed 93% inside one painted head is not a case this room can
    # produce; at 0.8 it was at least arguable.
    containment_floor: float = 0.93
    _regions: list[tuple[tuple[float, float, float, float], float]] = field(
        default_factory=list,
    )

    def add(self, regions, frame_wh: tuple[int, int] | None = None,
            now: float | None = None) -> None:
        """Remember these regions, in FRACTIONS of the frame.

        `frame_wh` is the size the regions were measured at. None means
        they are already normalised (the restart path, which reloads
        fractions straight off disk).
        """
        now = time.monotonic() if now is None else now
        for r in regions:
            box = _to_frac(
                (float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                frame_wh)
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
        frame_wh: tuple[int, int] | None = None,
        now: float | None = None,
    ) -> set[int]:
        """Track ids whose EVERY observed bbox is contained in a live
        screen region."""
        live = cluster_regions(self.live(now))
        if not live:
            return set()
        per_tid: dict[int, list[float]] = {}
        for rows in rows_by_frame.values():
            for row in rows:
                tid = int(row[4])
                box = _to_frac(
                    (float(row[0]), float(row[1]),
                     float(row[2]), float(row[3])), frame_wh)
                per_tid.setdefault(tid, []).append(
                    max(containment(box, r) for r in live))
        vetoed = set()
        for tid, scores in per_tid.items():
            if len(scores) < _MIN_FRAMES:
                continue
            # ONE RULE: every observed box of this track sat almost
            # entirely inside a region we know is a display or a
            # painting.
            #
            # There used to be a second rule — a movement exemption, so
            # a person standing in front of a painting was spared even
            # when contained. It protected nothing. Measured on the real
            # geometry (2026-08-25): a painting panel is 0.05 x 0.11 of
            # frame and a standing adult is 0.19 x 0.20, six times
            # larger, so the most of a person that can ever fall inside
            # one is 0.16 — against a floor of 0.93. The containment
            # floor alone already makes vetoing a person impossible.
            #
            # What the exemption DID do was spare the paintings: a mask
            # wobble of a couple of pixels, amplified by a downshifted
            # frame, crossed the movement bar often enough to keep the
            # phantoms alive indefinitely.
            #
            # Removing it also removes the reason the two region KINDS
            # existed — screens versus static art differed only in who
            # got the exemption — along with the drift maths and the
            # per-track kind arbitration.
            if min(scores) >= self.containment_floor:
                log.info(
                    "reid.display_veto_drop tid=%s containment=%.3f",
                    tid, min(scores),
                )
                vetoed.add(tid)
        return vetoed


def load_regions(path: str) -> list[tuple[tuple[float, ...], str]]:
    """Regions remembered from a previous run of this process.

    These are FURNITURE — a television and a painting are in the same
    place after a restart as before it — but the map lived only in
    memory, so every restart began blind and re-tracked every painted
    face on the wall until the next search two minutes later. Tracks
    born in that window then live on. Measured 2026-08-24: seven
    phantom tracks, all with the same 17.6s age, seconds after a
    restart.

    Restarts are not rare and are not all deliberate: the pod-progress
    watchdog exits rc=4 on its own several times a day.
    """
    try:
        raw = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out = []
    for row in raw[:16]:
        try:
            if int(row.get("v", 1)) != _REGION_FORMAT:
                continue                 # pixel-era file; see _REGION_FORMAT
            box = tuple(float(v) for v in row["box"][:4])
            if len(box) == 4:
                out.append(box)
        except (TypeError, KeyError, ValueError):
            continue
    if raw and not out:
        log.warning(
            "display_veto.regions_discarded path=%s — wrong format version, "
            "re-grounding from scratch", path)
    return out


def save_regions(path: str, regions) -> None:
    """Best-effort: losing the file costs one blind window, never a tick."""
    try:
        tmp = f"{path}.tmp"
        pathlib.Path(tmp).write_text(json.dumps([
            {"box": list(box), "v": _REGION_FORMAT} for box in regions
        ]), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        log.warning("display_veto.save_failed path=%s", path, exc_info=True)


def apply_veto(obs, screen_map: ScreenMap, *,
               frame_wh: tuple[int, int] | None = None,
               now: float | None = None):
    """Drop screen-contained tracks from a RemoteObservations bundle
    (rows + face obs). Returns (obs, dropped_tids) — obs is mutated in
    place (it is per-tick and not shared)."""
    dropped = screen_map.veto_track_ids(obs.rows_by_frame, frame_wh, now)
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
