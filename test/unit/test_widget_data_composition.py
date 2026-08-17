"""Composed widget-data requests (Phase A cross-cut review fix).

The engine's flag dispatch was first-match-wins, so a spec asking for a
chart AND a caption stat (hour_clock: byHour+ratio; dot_calendar:
occurrence rows+daysWith; week_heatmap: heatmap+comparePrev) got one or
the other — ratio even returned rows:[] and blanked the chart. Chart
flags now win and the stat merges into the same response.

Also: metric=sleepNights is served by /v1/patients/{id}/sleep-nights,
NEVER by this engine — it must dead-end explicitly instead of falling
through to an unfiltered events series mislabeled as sleep data.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.api import insights_widgets as iw


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows

    def one(self):
        return self._rows[0]

    def first(self):
        return self._rows[0] if self._rows else None


class _Session:
    """Routes canned rows by recognizable SQL fragments."""

    def __init__(self, routes):
        self.routes = routes  # list of (substring, rows)
        self.sql_log = []

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        self.sql_log.append(sql)
        for frag, rows in self.routes:
            if frag in sql:
                return _Result(rows)
        return _Result([])


@pytest.fixture
def one_patient(monkeypatch):
    pid = uuid4()

    async def fake_scope(session, auth):
        return [pid]
    monkeypatch.setattr(iw, "_scoped", fake_scope)
    return pid


async def _call(session, source):
    return await iw.widget_data(
        SimpleNamespace(id=uuid4()), session, iw.WidgetRequest(source=source))


async def test_by_hour_with_ratio_returns_rows_and_stat(one_patient) -> None:
    session = _Session([
        ("count(distinct occurred_at::date)", [{"n": 2}]),
        ('HH24":00"', [{"t": "02:00", "n": 3}]),
    ])
    out = await _call(session, {"metric": "events", "byHour": True,
                                "days": 14, "ratio": "nightsWith"})
    assert out["rows"][2]["events"] == 3          # chart survived
    assert out["statValue"] == "14%"              # 2 of 14 nights
    assert "nights with a matching event" in out["subtitle"]


async def test_plain_ratio_tile_unchanged(one_patient) -> None:
    session = _Session([("count(distinct occurred_at::date)", [{"n": 7}])])
    out = await _call(session, {"metric": "events", "days": 14,
                                "ratio": "daysWith"})
    assert out["rows"] == []
    assert out["statValue"] == "50%"


async def test_occurrence_mode_with_ratio_keeps_daily_rows(one_patient) -> None:
    session = _Session([
        ("count(distinct occurred_at::date)", [{"n": 4}]),
        ("date_trunc(:gran, occurred_at)", [{"bucket": "2026-08-10", "n": 1}]),
    ])
    out = await _call(session, {"metric": "events", "mode": "occurrence",
                                "days": 14, "ratio": "daysWith"})
    assert len(out["rows"]) == 14                 # filled daily buckets
    assert out["statValue"] == "29%"              # 4 of 14 days


async def test_heatmap_with_compare_prev_merges_delta(one_patient) -> None:
    session = _Session([
        ("extract(isodow from occurred_at)", [{"dow": 5, "h": 17, "n": 3}]),
        ("count(*)::int as n from events", [{"n": 5}]),
    ])
    out = await _call(session, {"metric": "events", "heatmap": True,
                                "days": 14, "comparePrev": True})
    assert out["kind"] == "heatmap"
    assert out["cells"] == [{"dow": 5, "h": 17, "n": 3}]
    assert out["prevValue"] == 5


async def test_sleep_nights_metric_dead_ends(one_patient) -> None:
    out = await _call(_Session([]), {"metric": "sleepNights", "days": 7})
    assert out["rows"] == []
    assert "sleep-nights" in out["unavailable"]
