"""Combined identity score — one calibrated scale.

Face and body cosines live on different scales (ArcFace ~0.45-0.65 is a
strong match; OSNet pool ≥0.72 is the floor). Each is mapped to a [0,1]
confidence; the face side is additionally weighted by face pixel size
(the empirical fact this whole design serves: a 20px face is noise, a
56px face is trustworthy). Combined = max of the two sides, tagged with
which side won (the anchor — face anchors are what permit harvesting).
"""

from __future__ import annotations

import pytest

from src.services.identity.scoring import body_conf, combine, face_conf


def test_face_conf_zero_below_pixel_floor():
    assert face_conf(cos=0.65, face_px=12.0) == 0.0


def test_face_conf_full_weight_at_large_face():
    strong = face_conf(cos=0.60, face_px=80.0)
    assert strong > 0.7


def test_face_conf_monotone_in_pixels():
    lo = face_conf(cos=0.55, face_px=24.0)
    mid = face_conf(cos=0.55, face_px=40.0)
    hi = face_conf(cos=0.55, face_px=64.0)
    assert lo < mid < hi


def test_face_conf_monotone_in_cosine():
    assert face_conf(cos=0.40, face_px=64.0) < face_conf(cos=0.60, face_px=64.0)


def test_body_conf_zero_below_pool_floor():
    assert body_conf(None) == 0.0
    assert body_conf(0.60) == 0.0


def test_body_conf_ramps():
    assert 0.0 < body_conf(0.75) < body_conf(0.85) <= 1.0


def test_combine_face_anchor_wins_when_face_strong():
    score, anchor = combine(face_cos=0.62, face_px=70.0, body_cos=0.73)
    assert anchor == "face"
    assert score == pytest.approx(face_conf(0.62, 70.0))


def test_combine_body_anchor_when_face_absent():
    score, anchor = combine(face_cos=None, face_px=None, body_cos=0.82)
    assert anchor == "body"
    assert score == pytest.approx(body_conf(0.82))


def test_combine_nothing():
    score, anchor = combine(face_cos=None, face_px=None, body_cos=None)
    assert score == 0.0 and anchor is None


def test_body_conf_full_resolution_no_cap():
    # NO cap (Aryaman 2026-07-24): the top of the range is where
    # similar-clothed people at distance separate — a cap flattened
    # 0.95-patient vs 0.85-guest into a tie. Self-match inflation is
    # handled at the seat reference, not by destroying discrimination.
    assert body_conf(0.95) > body_conf(0.85) > body_conf(0.75)
    assert combine(face_cos=None, face_px=None, body_cos=0.99) == (1.0, "body")
