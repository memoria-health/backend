"""Read endpoints for sleep_nights / activity_days.

The nightly summaries runner has been writing these tables since July
with nothing serving them; the Home tab's sleep_bands and trend_band
widgets are their first reader. Caregiver-scoped like every patient
route: a patient the caller doesn't own is a 404, never a leak.
"""

from datetime import date
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api import summaries


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def execute(self, stmt, params=None):
        self.queries.append((str(stmt), params or {}))
        return _Result(self.rows)


def _auth():
    return SimpleNamespace(id=uuid4())


@pytest.fixture
def owned_patient(monkeypatch):
    pid = uuid4()

    async def find(session, patient_id, caregiver_id=None):
        return SimpleNamespace(id=patient_id) if patient_id == pid else None
    monkeypatch.setattr(summaries.patient_repo, "find_by_id", find)
    return pid


async def test_sleep_nights_returns_rows_ascending(owned_patient) -> None:
    rows = [{"night": "2026-08-13", "duration_minutes": 480,
             "disturbances": 1}]
    session = _Session(rows)
    out = await summaries.sleep_nights(owned_patient, _auth(), session, days=7)
    assert out == {"nights": rows}
    assert session.queries[0][1]["days"] == 7


async def test_activity_days_returns_rows(owned_patient) -> None:
    rows = [{"day": "2026-08-13", "active_fraction": 0.4}]
    out = await summaries.activity_days(owned_patient, _auth(), _Session(rows),
                                        days=30)
    assert out == {"days": rows}


async def test_unowned_patient_is_404(owned_patient) -> None:
    with pytest.raises(HTTPException) as e:
        await summaries.sleep_nights(uuid4(), _auth(), _Session([]), days=7)
    assert e.value.status_code == 404
