"""Patient-focus governor (2026-07-22): the per-tick reid fidelity
decision that keeps the home's total GPU demand inside budget while
never blinding a room the patient could walk into. Born from the live
saturation spiral: both cameras full → 45 shed ticks / 3 min → stitch
gaps → lock churn → lean never engaged."""

from __future__ import annotations

import time
from uuid import uuid4

from src.services.detection.camera_ingest import compute_reid_mode
from src.services.detection.patient_presence import (
    TIER_FACE,
    PatientPresence,
)

CAM_A, CAM_B = uuid4(), uuid4()


# ── the decision table ─────────────────────────────────────────────


def test_lean_when_patient_locked_and_scene_identified():
    assert compute_reid_mode(
        all_bound=True, confirmed_elsewhere=False,
        holder_seen=True, occupied_count=2,
    ) == "lean"


def test_minimal_only_when_patient_seen_elsewhere():
    # Held elsewhere AND freshly in sight → this room coasts.
    assert compute_reid_mode(
        all_bound=False, confirmed_elsewhere=True,
        holder_seen=True, occupied_count=2,
    ) == "minimal"
    # Held elsewhere but sight LOST → escalate (room change coming).
    assert compute_reid_mode(
        all_bound=False, confirmed_elsewhere=True,
        holder_seen=False, occupied_count=2,
    ) == "split"


def test_split_shares_budget_when_nobody_holds():
    assert compute_reid_mode(
        all_bound=False, confirmed_elsewhere=False,
        holder_seen=False, occupied_count=2,
    ) == "split"
    assert compute_reid_mode(
        all_bound=False, confirmed_elsewhere=False,
        holder_seen=False, occupied_count=3,
    ) == "split"


def test_full_when_sole_occupied_room_and_patient_unfound():
    assert compute_reid_mode(
        all_bound=False, confirmed_elsewhere=False,
        holder_seen=False, occupied_count=1,
    ) == "full"
    assert compute_reid_mode(
        all_bound=False, confirmed_elsewhere=False,
        holder_seen=False, occupied_count=0,
    ) == "full"


# ── registry heartbeats ────────────────────────────────────────────


def test_occupied_count_freshness():
    p = PatientPresence()
    p.note_tick(CAM_A, people_present=True)
    p.note_tick(CAM_B, people_present=False)
    assert p.occupied_count() == 1
    p.note_tick(CAM_B, people_present=True)
    assert p.occupied_count() == 2
    # Stale heartbeats age out of the count.
    p._occupancy[CAM_A] = (True, time.monotonic() - 60.0)
    assert p.occupied_count() == 1


def test_holder_seen_fails_open_to_false():
    p = PatientPresence()
    # No holder at all → False (rooms keep looking).
    assert p.holder_seen() is False
    p.claim(CAM_A, tier=TIER_FACE, confidence=0.9)
    # Holder but no sight heartbeat yet → still False.
    assert p.holder_seen() is False
    p.note_tick(CAM_A, people_present=True, holder_seen=True)
    assert p.holder_seen() is True
    # Holder loses sight → False the same tick (escalation trigger).
    p.note_tick(CAM_A, people_present=True, holder_seen=False)
    assert p.holder_seen() is False
    # Sight heartbeat goes stale → False.
    p.note_tick(CAM_A, people_present=True, holder_seen=True)
    p._holder_seen = (True, time.monotonic() - 60.0)
    assert p.holder_seen() is False


def test_room_change_escalation_sequence():
    """The handoff story end-to-end at the decision level: patient
    held+seen in A (B coasts minimal) → A loses sight → B escalates
    to split THAT tick → B can face-bind and take over."""
    p = PatientPresence()
    p.claim(CAM_A, tier=TIER_FACE, confidence=0.9)
    p.note_tick(CAM_A, people_present=True, holder_seen=True)
    p.note_tick(CAM_B, people_present=True)
    assert compute_reid_mode(
        all_bound=False,
        confirmed_elsewhere=p.confirmed_elsewhere(CAM_B),
        holder_seen=p.holder_seen(),
        occupied_count=p.occupied_count(),
    ) == "minimal"
    # Patient walks out of A's view: B is now the only occupied room
    # and the patient is unfound — it escalates past minimal all the
    # way to FULL (sole room gets the whole budget to find him fast).
    p.note_tick(CAM_A, people_present=False, holder_seen=False)
    assert compute_reid_mode(
        all_bound=False,
        confirmed_elsewhere=p.confirmed_elsewhere(CAM_B),
        holder_seen=p.holder_seen(),
        occupied_count=p.occupied_count(),
    ) == "full"
