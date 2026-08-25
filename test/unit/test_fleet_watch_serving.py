"""FleetWatch serving extension (2026-08-14): the lag alarm and the
occupancy duty-cycle sampler — the production tripwire the capacity
rating leans on (docs/architecture/identity-gpu-architecture.md).

Pure unit: DB and HTTP are both faked; what's under test is the
decision logic — streaks, the empty-room exemption, recovery, and what
gets sampled.
"""

from types import SimpleNamespace

import pytest

from src.services.identity import fleet_watch as fw


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalar(self):
        return self._rows


class _FakeSession:
    """Answers the served-homes SELECT with canned rows and records
    every INSERT's params."""

    def __init__(self, served_rows, inserts):
        self._served = served_rows
        self._inserts = inserts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, clause, params=None):
        sql = str(clause)
        if "INSERT INTO public.identity_occupancy_samples" in sql:
            self._inserts.append(params)
            return _FakeResult([])
        if "FROM public.cameras c" in sql and "JOIN" in sql:
            return _FakeResult(self._served)
        # Loudly, not silently. This matched "FROM public.patients p"
        # until the identity shard moved to the sensor, and a fake that
        # answers an unrecognised query with an empty result turns a
        # renamed table into "the fleet is serving nothing" — which reads
        # as a real finding rather than a broken test.
        raise AssertionError(f"unexpected query in fake session: {sql[:120]}")

    async def commit(self):
        pass


class _Pager:
    def __init__(self, monkeypatch):
        self.triggered: list[str] = []
        self.resolved: list[str] = []

        async def _trigger(*, settings, dedup_key, **kw):
            self.triggered.append(dedup_key)
            return True

        async def _resolve(*, settings, dedup_key):
            self.resolved.append(dedup_key)
            return True

        monkeypatch.setattr(fw.pager, "trigger_keyed", _trigger)
        monkeypatch.setattr(fw.pager, "resolve_keyed", _resolve)


def _watch(monkeypatch, states_by_url, inserts):
    settings = SimpleNamespace(
        identity_roster_stale_after_s=90,
        livekit_worker_tick_seconds=10,
        identity_state_token="",
        identity_fleet_watch_seconds=60,
    )
    served = [("pid-1", "node-a", "http://100.1.1.1", 8012)]

    def sessionmaker():
        return _FakeSession(served, inserts)

    w = fw.FleetWatch(settings=settings, sessionmaker=sessionmaker)

    async def _fetch(url):
        return states_by_url.get(url)

    w._fetch_state = _fetch
    return w


def _state(lag, tracks):
    return {"cameras": {"cam-1": {
        "lag_s": lag, "tracks": {"1": {}} if tracks else {},
    }}}


URL = "http://100.1.1.1:8012/state"


@pytest.mark.asyncio
async def test_sustained_lag_pages_on_third_cycle(monkeypatch):
    pager = _Pager(monkeypatch)
    inserts: list = []
    w = _watch(monkeypatch, {URL: _state(lag=25.0, tracks=True)}, inserts)
    for cycle in range(fw.LAG_ALARM_CYCLES):
        counts = await w._check_serving()
    assert pager.triggered == [f"{fw._KEY_LAG}cam-1"]
    assert counts["cameras_lagging"] == 1


@pytest.mark.asyncio
async def test_recovery_resolves_and_resets_streak(monkeypatch):
    pager = _Pager(monkeypatch)
    inserts: list = []
    states = {URL: _state(lag=25.0, tracks=True)}
    w = _watch(monkeypatch, states, inserts)
    for _ in range(fw.LAG_ALARM_CYCLES):
        await w._check_serving()
    states[URL] = _state(lag=0.3, tracks=True)
    await w._check_serving()
    assert pager.resolved == [f"{fw._KEY_LAG}cam-1"]
    assert w._lag_streak == {}


@pytest.mark.asyncio
async def test_empty_room_frozen_lag_never_pages(monkeypatch):
    # An empty room stops batching by design; its last lag_s is a
    # frozen snapshot, not a serving measurement (the 2026-08-13
    # watchdog false-alarm lesson, encoded).
    pager = _Pager(monkeypatch)
    inserts: list = []
    w = _watch(monkeypatch, {URL: _state(lag=99.0, tracks=False)}, inserts)
    for _ in range(fw.LAG_ALARM_CYCLES + 1):
        await w._check_serving()
    assert pager.triggered == []


@pytest.mark.asyncio
async def test_occupancy_sampled_per_node(monkeypatch):
    _Pager(monkeypatch)
    inserts: list = []
    w = _watch(monkeypatch, {URL: _state(lag=0.2, tracks=True)}, inserts)
    counts = await w._check_serving()
    assert counts["cameras_polled"] == 1
    assert counts["cameras_occupied"] == 1
    assert inserts == [{"n": "node-a", "t": 1, "o": 1}]


@pytest.mark.asyncio
async def test_unreachable_node_records_no_sample(monkeypatch):
    _Pager(monkeypatch)
    inserts: list = []
    w = _watch(monkeypatch, {}, inserts)  # fetch returns None
    counts = await w._check_serving()
    assert inserts == []
    assert counts["cameras_polled"] == 0
