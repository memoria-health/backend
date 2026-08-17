"""Per-detector filtering in /v1/insights/widget-data.

The Home tab's widgets are per-detector: `source.detector` narrows every
events query to that detector's rows. Scope rule mirrors `_narrow`: an
unknown or foreign detector id is a 404 (existence is not disclosed),
and the patient set collapses to the detector's own patient.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api import insights_widgets as iw


# ── filter builder (pure) ────────────────────────────────────────────

def test_terms_filter_with_detector_only() -> None:
    det = str(uuid4())
    assert iw._terms_filter([], detector=det) == f"(true and detector_id = '{det}')"


def test_terms_filter_combines_terms_and_detector() -> None:
    det = str(uuid4())
    out = iw._terms_filter(["fall"], detector=det)
    assert out.endswith(f" and detector_id = '{det}')")
    assert "title ilike '%fall%'" in out


def test_terms_filter_alias_applies_to_detector() -> None:
    det = str(uuid4())
    assert f"e.detector_id = '{det}'" in iw._terms_filter([], "e", detector=det)


def test_terms_filter_without_detector_unchanged() -> None:
    assert iw._terms_filter([]) == "true"


# ── endpoint scope checks ────────────────────────────────────────────

class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _Session:
    def __init__(self, row):
        self._row = row

    async def execute(self, *_a, **_k):
        return _Result(self._row)


@pytest.fixture
def scoped_patient(monkeypatch):
    pid = uuid4()

    async def fake_scope(session, auth):
        return [pid]
    monkeypatch.setattr(iw, "_scoped", fake_scope)
    return pid


async def _call(session, detector) -> dict:
    return await iw.widget_data(
        SimpleNamespace(id=uuid4()), session,
        iw.WidgetRequest(source={"metric": "daysSince", "detector": detector}),
    )


@pytest.mark.asyncio
async def test_malformed_detector_id_is_404(scoped_patient) -> None:
    with pytest.raises(HTTPException) as e:
        await _call(_Session(None), "not-a-uuid")
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_unknown_detector_is_404(scoped_patient) -> None:
    with pytest.raises(HTTPException) as e:
        await _call(_Session(None), str(uuid4()))
    assert e.value.status_code == 404


@pytest.mark.asyncio
async def test_foreign_patients_detector_is_404(scoped_patient) -> None:
    with pytest.raises(HTTPException) as e:
        await _call(_Session((uuid4(),)), str(uuid4()))
    assert e.value.status_code == 404
