"""One calibrated identity scale — face weighted by pixel size, body as
the workhorse.

ArcFace cosines and OSNet pool cosines live on different scales and must
never be compared raw (the old `_TIER` rule existed to forbid exactly
that). Here each side maps to a [0,1] confidence via a piecewise-linear
ramp anchored at the empirically-known operating points, and the face
side is additionally weighted by face pixel height — the load-bearing
fact of this home: a 20px face is noise, a 56px face is trustworthy.

Combined = max(face_side, body_side), tagged with the winning side (the
anchor). Face anchors are what permit harvesting; `max` (not a sum) so
two weak signals can never impersonate one strong one.

All constants are pilot-calibration targets — tune from trace data, not
theory (see plan §6.4).
"""

from __future__ import annotations

from typing import Optional

# CALIBRATED TO THE LEGACY OPERATING POINT (2026-07-24, live miss):
# the legacy brain locked Aryaman all week at cosine 0.45-0.60 with
# ≥36px faces; the first-draft curve scored that same evidence ~0.37 —
# under the floor — so face NEVER locked in the bedroom and the whole
# harvest flywheel stalled. These ramps put the proven legacy locks at
# or above the seat floor (0.40): e.g. 36px @ cos 0.55 → ~0.63.
_FACE_COS_LO, _FACE_COS_HI = 0.35, 0.60
_FACE_PX_LO, _FACE_PX_HI = 16.0, 44.0
# Body pool ramp: pool top-3-mean 0.70 → 0, 0.90 → 1 (the legacy 0.72
# match bar lands at 0.1 — barely admissible, exactly how the eval data
# said marginal far-body matches should count).
_BODY_COS_LO, _BODY_COS_HI = 0.70, 0.90
# NO cap on body confidence (Aryaman, 2026-07-24): a cap flattens the
# top range and erases the margin between similar-clothed people at
# distance (raw 0.95 patient vs 0.85 guest both became 0.75 → tie →
# ambiguity). Body competes at full resolution; the self-match
# inflation problem is handled in seat.py instead — same-track body
# evidence never ratchets the seat's reference score.


def _ramp(x: float, lo: float, hi: float) -> float:
    if x <= lo:
        return 0.0
    if x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def face_conf(cos: Optional[float], face_px: Optional[float]) -> float:
    """[0,1] confidence of a face match, discounted by face size."""
    if cos is None or face_px is None:
        return 0.0
    return _ramp(cos, _FACE_COS_LO, _FACE_COS_HI) * _ramp(
        face_px, _FACE_PX_LO, _FACE_PX_HI,
    )


def body_conf(cos: Optional[float]) -> float:
    """[0,1] confidence of a body-pool match — full resolution, no cap
    (the top of the range is where similar-clothes cases separate)."""
    if cos is None:
        return 0.0
    return _ramp(cos, _BODY_COS_LO, _BODY_COS_HI)


def combine(
    *,
    face_cos: Optional[float],
    face_px: Optional[float],
    body_cos: Optional[float],
) -> tuple[float, Optional[str]]:
    """One combined score + which side won ("face" | "body" | None)."""
    f = face_conf(face_cos, face_px)
    b = body_conf(body_cos)
    if f == 0.0 and b == 0.0:
        return 0.0, None
    if f >= b:
        return f, "face"
    return b, "body"
