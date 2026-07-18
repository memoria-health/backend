"""Unit tests for the v2 dimension table (catalog_v2).

Most invariants are enforced at import time by `catalog_v2._validate()`
(fail-loud, same convention as `catalog._validate_catalog`); these tests
make the contract explicit in the suite and pin a few load-bearing values
so a silent remapping shows up as a named failure.
"""

from __future__ import annotations

from src.db.detector import Detector
from src.services.detection import catalog, catalog_v2


def test_every_catalog_key_is_mapped_no_strays() -> None:
    assert set(catalog_v2.DIMS) == {e.key for e in catalog.CATALOG}


def test_severity_lockstep_with_catalog_map() -> None:
    """Promotion falls back to the in-code map for NULL rows — the two
    sources must agree until the map is retired."""
    for key, dims in catalog_v2.DIMS.items():
        assert dims.severity == catalog.severity_for(key), key


def test_load_bearing_dims_pinned() -> None:
    d = catalog_v2.DIMS
    assert d["seizure"].bucket == "rapid"                       # the P0 recipe
    assert d["seizure"].evidence_branch == "person_motion"
    assert d["stove_left_on"].evidence_branch == "environment"  # no person needed
    assert d["stove_left_on"].bucket == "scene"  # snapshot retired 2026-07-18
    assert d["agitation"].needs_audio is True                   # tone matters
    assert d["possible_intruder"].subject_scope == "anyone"     # concealment, not identity
    assert d["fall"].subject_scope == "patient"                 # Aryaman: patient-scoped
    assert d["child_device_usage"].evidence_branch == "person_still"
    # The distance-degradation class carries its perpetual warning.
    assert d["env_hazard_glass"].degradation_warning == "distance_class"


def test_row_kwargs_splat_matches_orm_columns() -> None:
    kw = catalog_v2.row_kwargs("fall")
    assert kw, "built-ins must produce insert kwargs"
    for column in kw:
        assert hasattr(Detector, column), f"no ORM column {column}"
    # Customs are classified by the compiler later (P5) — empty splat.
    assert catalog_v2.row_kwargs("custom_anything") == {}
