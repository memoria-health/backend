"""Track-level face evidence — robust sighting statistics (2026-07-24).

Two live lessons shaped this file in one evening:

1. The AUNT incident: judging each probe independently let ONE 0.46
   outlier (among twenty near-zeros) seat an impostor. Evidence must
   aggregate across a track's sightings.
2. The T11 walk-around: aggregating by MEAN diluted the real patient —
   half a walking track's "face" sightings are back-of-head detections
   scoring ~0.05 with high detector confidence, and averaging them into
   ten genuine 0.42-0.49 sightings left the candidate stuck at 0.23.

So: TOP-K aggregation (garbage lows can't vote, a lone outlier can't
win — it needs K-1 friends), plus a CONSISTENCY WAIVER on the face-size
discount: one small-face cosine is unreliable (that's why px discounts
exist), but N consistent ones shrink the variance — live: the patient
produced ten ≥0.42 sightings in 40s at ~34px, the impostor produced one
≥0.42 all night. Count substitutes for size.

History lives and dies with its track (seat-reference lifecycle).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from identity.scoring import face_conf

WINDOW = 12          # sightings remembered per track
TOP_K = 3            # best-K aggregation (an outlier needs K-1 friends)
RUN_BAR_COS = 0.40   # a sighting this strong counts toward consistency
RUN_BAR_PX = 24.0    # ...if the face isn't tiny
RUN_COUNT = 5        # qualifiers in the window to waive the px discount
MAX_AGE_S = 90.0     # sightings EXPIRE BY TIME, not just by count
                     # (2026-07-25 ghost-evidence incident: the patient
                     # left, his track welded onto a relative, and his
                     # own frozen 0.4+ sightings held the seat for 16
                     # minutes because faceless probes never displaced
                     # them from a count-based window)


@dataclass
class SightingHistory:
    """Recent (ts, cosine, face_px) sightings for one track."""

    _s: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def add(self, cos: float, face_px: float, *, now: float) -> None:
        self._s.append((float(now), float(cos), float(face_px)))

    def _fresh(self, now: float) -> list:
        return [(c, p) for t, c, p in self._s if now - t <= MAX_AGE_S]

    def hits_at(self, now: float) -> int:
        return len(self._fresh(now))

    def evidence(self, *, now: float) -> float:
        """[0,1] face confidence for the TRACK (not a frame).

        max of:
        - top-K mean of per-sighting confs (px-discounted as usual);
        - consistency waiver: ≥RUN_COUNT sightings at cos≥RUN_BAR_COS
          and px≥RUN_BAR_PX → the mean qualifying cosine on the cos
          ramp alone, count standing in for size.

        Only sightings younger than MAX_AGE_S count — evidence is a
        claim about who is on the track NOW, and stale sightings from
        a departed patient must never speak for a welded track.
        """
        fresh = self._fresh(now)
        if not fresh:
            return 0.0
        confs = sorted(
            (face_conf(c, p) for c, p in fresh), reverse=True,
        )
        top = confs[:TOP_K]
        # Pad: fewer than K sightings must not inflate the mean — an
        # outlier alone averages against zeros, not against itself.
        while len(top) < TOP_K:
            top.append(0.0)
        ev = sum(top) / TOP_K

        run = [
            c for c, p in fresh
            if c >= RUN_BAR_COS and p >= RUN_BAR_PX
        ]
        if len(run) >= RUN_COUNT:
            waived = face_conf(sum(run) / len(run), 1e9)  # px waived
            ev = max(ev, waived)
        return ev
