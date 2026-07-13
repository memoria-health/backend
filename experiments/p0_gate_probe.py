"""Measure rapid_energy on the labeled eval clips across parameter sweeps —
no model calls. Finds whether any (persist_ratio, min_score) separates
convulsion from shivering/negatives on REAL footage, or whether the churn
gate needs a design change.

Usage:  PYTHONPATH=. uv run python experiments/p0_gate_probe.py
"""

from pathlib import Path

from experiments.p0_replay_test import _clip_frames, _person_box
from src.services.detection.rapid_artifact import union_box
from src.services.detection.signals.rapid_energy import rapid_energy

_D = Path.home() / "Desktop" / "eval-set" / "rapid_motion"
RATIOS = (0.15, 0.25, 0.35, 0.5)

for clip in sorted(_D.glob("*.mp4")):
    frames = _clip_frames(clip)
    boxes = _person_box(frames)
    region = union_box(boxes, frames[0].width, frames[0].height) if boxes else None
    row = [f"{clip.stem:10s} region={'person' if region else 'full '}"]
    for pr in RATIOS:
        e = rapid_energy(frames, region=region, persist_ratio=pr)
        row.append(f"pr{pr}: s={e.score:.3f} af={e.active_fraction:.4f}")
    print("  ".join(row))
