"""Identity service — continuous tracking + sparse identity (2026-07-24 architecture).

The production form of the live-viewer experiment: per-camera continuous
small-batch tracking against the pod's persistent EdgeTAM session, with
the three-rule identity module on top:

  1. Score everyone on one calibrated scale (face weighted by pixel
     size; body is the workhorse) — `scoring.py`.
  2. Highest score takes the single home-wide patient seat, beat-by-δ,
     seat rides the track and dies with it — `seat.py`.
  3. Harvest only face-confirmed solo tracks, liberally — `harvest.py`.

Pure logic lives in batching/state/scoring/seat/harvest (unit-tested);
`loop.py` is the async glue; `tools/run_identity.py` is the entry point.
Design doc: docs/plans/2026-07-24-continuous-identity-architecture.md.
"""
