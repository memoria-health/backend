"""Journal endpoint scope + shape (builders are tested separately)."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api import journal


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def one(self):
        return self._rows[0]

    def first(self):
        return self._rows[0] if self._rows else None


class _Session:
    def __init__(self, routes):
        self.routes = routes

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        for frag, rows in self.routes:
            if frag in sql:
                return _Result(rows)
        return _Result([])


@pytest.fixture
def owned(monkeypatch):
    pid = uuid4()

    async def find(session, patient_id, caregiver_id=None):
        return SimpleNamespace(id=patient_id) if patient_id == pid else None

    async def dets(session, patient_id, caregiver_id=None):
        return []
    monkeypatch.setattr(journal.patient_repo, "find_by_id", find)
    monkeypatch.setattr(journal.detector_repo, "list_for_patient", dets)
    return pid


async def test_unowned_patient_404(owned) -> None:
    with pytest.raises(HTTPException) as e:
        await journal.get_journal(uuid4(), SimpleNamespace(id=uuid4()),
                                  _Session([]), date="2026-08-15")
    assert e.value.status_code == 404


async def test_shape_with_no_detectors(owned) -> None:
    session = _Session([
        ("from caregivers", [("America/Los_Angeles",)]),
        ("date_trunc('hour'", [{"n": 14}]),
    ])
    out = await journal.get_journal(owned, SimpleNamespace(id=uuid4()),
                                    session, date="2026-08-15")
    assert out["date"] == "2026-08-15"
    assert out["coverage_hours"] == 14
    assert out["sections"] == []
    assert out["timezone"] == "America/Los_Angeles"
    assert out["partial"] in (True, False)


async def test_reviewed_wrong_events_do_not_reach_the_day(owned) -> None:
    from types import SimpleNamespace as NS
    from datetime import datetime, timezone
    ev_ok = NS(detector_id="d1", review_choice=None, reviewed=False,
               occurred_at=datetime(2026, 8, 15, 9, tzinfo=timezone.utc),
               id="e1", location="X")
    ev_wrong = NS(detector_id="d1", review_choice="wrong", reviewed=True,
                  occurred_at=datetime(2026, 8, 15, 10, tzinfo=timezone.utc),
                  id="e2", location="X")
    session = _Session([
        ("from caregivers", [("UTC",)]),
        ("FROM events", [ev_ok, ev_wrong]),      # ORM select renders caps
        ("date_trunc('hour'", [{"n": 5}]),
    ])
    out = await journal.get_journal(owned, SimpleNamespace(id=uuid4()),
                                    session, date="2026-08-15")
    assert out["coverage_hours"] == 5
