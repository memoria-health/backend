"""Unit tests for the RAPID recipe pieces (detection v2).

Since Ledger row 5 the standalone lane is gone — the v2 executor
consumes these pieces (overlay prompt swap, describe-then-decide
preamble, box scaling) and its tests cover the runtime behavior
(gate short-circuit, artifact fallback, settings override). Here we
pin the pieces themselves.
"""

from __future__ import annotations

import pytest

from src.services.detection.rapid_call import (
    RAPID_MOTION_SPEC,
    RAPID_SEIZURE_PROMPT,
    rapid_seizure_spec,
    scale_bbox_to_frame,
)


def test_prompt_carries_the_load_bearing_constraints() -> None:
    p = RAPID_SEIZURE_PROMPT.lower()
    assert "never sufficient" in p       # red tint alone can't fire it
    assert "shivering" in p              # the mutual confounder is named
    assert "any position" in p           # not predicated on lying down


def test_overlay_spec_keeps_the_catalog_identifier_and_field_trio() -> None:
    """The overlay variant must land on the SAME seizure_* keys the
    catalog row owns — promotion / severity / dedup / clips downstream
    cannot tell which prompt produced the verdict."""
    spec = rapid_seizure_spec()
    assert spec.identifier == "seizure"
    assert set(spec.schema_fields) == {
        "seizure_detected", "seizure_confidence", "seizure_reasoning",
    }
    assert RAPID_MOTION_SPEC.always_on is True
    assert "rapid_motion_description" in RAPID_MOTION_SPEC.schema_fields


def test_scale_bbox_maps_reid_space_to_native_pixels() -> None:
    """Re-ID boxes live in extracted-frame space (max_width=480). Applied
    to a 1280-wide native frame they must scale by 1280/480; on a source
    already narrower than the sample width they must pass through."""
    scaled = scale_bbox_to_frame((120, 90, 240, 180), 1280, 720, sample_width=480)
    assert scaled == pytest.approx((320.0, 240.0, 640.0, 480.0))
    same = scale_bbox_to_frame((10, 10, 50, 50), 320, 240, sample_width=480)
    assert same == pytest.approx((10.0, 10.0, 50.0, 50.0))
    assert scale_bbox_to_frame((0, 0, 1, 1), 0, 0, sample_width=480) is None
