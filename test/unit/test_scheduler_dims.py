"""Unit: effective-dims resolution (detection v2 P3 T0).

Three-step fallback: row columns → catalog_v2 by identifier → the
legacy default that behaves exactly like today's monolithic call.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.services.detection import catalog_v2
from src.services.detection.scheduling.dims import resolve_dims


def _row(**over) -> SimpleNamespace:
    base = dict(
        identifier="fall", shape="point_in_time",
        evidence_branch=None, subject_scope=None, bucket=None,
        duty_cycle=None, needs_audio=False, severity=None,
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_row_columns_win_over_catalog() -> None:
    row = _row(
        evidence_branch="environment", subject_scope="environment",
        bucket="snapshot", duty_cycle="sampled", needs_audio=True,
        severity=33,
    )
    d = resolve_dims(row)
    assert d.dims_source == "row"
    assert (d.evidence_branch, d.bucket, d.severity, d.needs_audio) == (
        "environment", "snapshot", 33, True,
    )


def test_classified_row_with_null_severity_falls_back_to_catalog() -> None:
    row = _row(
        evidence_branch="person_motion", subject_scope="patient",
        bucket="scene", duty_cycle="every_eligible_tick", severity=None,
    )
    d = resolve_dims(row)
    assert d.dims_source == "row"
    assert d.severity == catalog_v2.dims_for("fall").severity


def test_unbackfilled_builtin_resolves_from_catalog() -> None:
    d = resolve_dims(_row())   # all v2 columns NULL, identifier=fall
    assert d.dims_source == "catalog"
    dims = catalog_v2.dims_for("fall")
    assert (d.evidence_branch, d.subject_scope, d.bucket, d.severity) == (
        dims.evidence_branch, dims.subject_scope, dims.bucket, dims.severity,
    )


def test_pre_p5_custom_gets_legacy_default_like_today() -> None:
    d = resolve_dims(_row(identifier="custom_a1b2", shape="state"))
    assert d.dims_source == "legacy_default"
    # Behaves exactly like today's monolithic call: wide, always due,
    # no subject suppression.
    assert (d.evidence_branch, d.subject_scope, d.bucket, d.duty_cycle) == (
        "person_motion", "anyone", "scene", "every_eligible_tick",
    )
    assert d.shape == "state"
