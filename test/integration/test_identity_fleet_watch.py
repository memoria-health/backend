"""Integration tests for the GPU fleet dead-man watch.

The whole point of this watchdog is that the failure it reports is INVISIBLE
everywhere else: identity fails open, so a home with no node keeps producing
safety detections and a green /health while its routine detectors quietly
skip. There is no symptom to notice, which is why the 2026-07-26 OOM ran for
hours. So the SQL here is tested against a real database rather than
reasoned about — a miscounted join would restore exactly the silence this
exists to break.

Paging is stubbed out; what is asserted is the counting and the
trigger/resolve decisions.

Assumes a local Supabase with scripts/apply_identity_nodes_delta.sql applied.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_sessionmaker
from src.services.identity import fleet_watch as fw_mod
from src.services.identity.fleet_watch import FleetWatch
import pytest_asyncio

from test.conftest import seed_camera

@pytest_asyncio.fixture(autouse=True)
async def _restore_real_fleet(session: AsyncSession):
    """Snapshot REAL home->node assignments and put them back after
    every test. This suite shares the local database with a live
    production fleet node; its _clear/park helpers detach or re-park
    the real homes, and the node's next roster poll then re-claims
    them with fresh ports from the bottom of the series. That was a
    full day of 'the identity port died again' (2026-08-17) — every
    pytest run was reshuffling production. Real rows = anything not
    named by this suite."""
    rows = [tuple(r) for r in (await session.execute(text(
        "SELECT id, identity_node_id, identity_port FROM public.cameras "
        " WHERE identity_node_id IS NOT NULL "
        "   AND identity_node_id NOT LIKE 'node-%' "
        "   AND identity_node_id <> 'fleet-sentinel'"))).all()]
    yield
    for pid, node, port in rows:
        await session.execute(text(
            "UPDATE public.cameras "
            "   SET identity_node_id = :n, identity_port = :p "
            " WHERE id = :i"), {"n": node, "p": port, "i": pid})
    await session.commit()


pytestmark = pytest.mark.asyncio


class _Pages:
    """Records what would have been paged, instead of paging."""

    def __init__(self) -> None:
        self.triggered: dict[str, dict] = {}
        self.resolved: list[str] = []

    def install(self, monkeypatch) -> None:
        async def _trigger(*, settings, dedup_key, source, **kw):
            self.triggered[dedup_key] = {"source": source, **kw}
            return True

        async def _resolve(*, settings, dedup_key):
            self.resolved.append(dedup_key)
            return True

        monkeypatch.setattr(fw_mod.pager, "trigger_keyed", _trigger)
        monkeypatch.setattr(fw_mod.pager, "resolve_keyed", _resolve)
        monkeypatch.setattr(fw_mod.pager, "is_configured", lambda s: True)


async def _mk_caregiver(session: AsyncSession, cid: UUID, email: str) -> None:
    await session.execute(
        text(
            "INSERT INTO public.caregivers (id, display_name, email) "
            "VALUES (:id, :n, :e) ON CONFLICT (id) DO NOTHING"
        ),
        {"id": cid, "n": "Fleet Test", "e": email},
    )
    await session.commit()


async def _mk_home(session: AsyncSession, cid: UUID, age_mins: int = 0) -> UUID:
    """A patient with one live SENSOR, and the SENSOR's id.

    Returns the sensor because that is what holds an identity slot since
    2026-08-22. It used to return the patient id, which every caller then
    handed to `_assign` — harmless while the slot lived on the patient,
    and a silent no-op afterwards: the UPDATE matched no row, so a home
    the test believed was served read blind.
    """
    pid = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.patients "
            "  (id, caregiver_id, full_name, preferred_name, relation, created_at) "
            "VALUES (:id, :cg, :n, 'Fleet', 'parent', "
            "        now() - make_interval(mins => :m))"
        ),
        {"id": pid, "cg": cid, "n": f"Fleet {str(pid)[:8]}", "m": age_mins},
    )
    # A home is only WATCHED once something is streaming (2026-08-19): a
    # home with no camera online has no identity to be deprived of, and
    # counting those made the critical page fire forever on dormant test
    # rows. Every case in this suite is about a home that should be
    # watched, so the helper gives each one a live camera.
    cam_id = await _mk_online_camera(session, pid, age_mins=age_mins)
    await session.commit()
    return cam_id


async def _mk_online_camera(
    session: AsyncSession, pid: UUID, *, publishes_into: UUID | None = None,
    age_mins: int = 0,
) -> UUID:
    """`livekit_room` is set because every real creation path sets it and
    liveness reads it: a home is watchable when a camera STREAMS INTO its
    room, not when a camera is assigned to its patient. `publishes_into`
    makes the two differ, which is what a shared camera does."""
    return await seed_camera(
        session, patient_id=pid, online=True,
        livekit_room=f"patient_{publishes_into or pid}",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=age_mins),
    )


# One extra live node exists in every test: the sentinel below.
async def _live_baseline(session: AsyncSession, settings) -> int:
    """Live identity_nodes rows this suite did NOT create.

    Historically a constant 1 (the parked-home sentinel) — but the local
    Supabase is ALSO the database a real production fleet node heartbeats
    into every ~10s (2026-08-17), so absolute nodes_live assertions must
    measure the pre-existing live count instead of assuming it. Mirrors
    check()'s exact liveness cutoff.

    Call this AFTER check(), never before: the node keeps beating, so a
    baseline taken first can disagree with the count check() produced a
    moment later. That is a false failure indistinguishable from a real
    regression, which is the worst kind to leave in a suite people are
    about to run on a loop during a soak."""
    return (await session.execute(
        text(
            "SELECT count(*) FROM public.identity_nodes "
            " WHERE node_id NOT LIKE 'node-%' "
            "   AND last_seen_at > now() - make_interval(secs => :s)"
        ),
        {"s": settings.identity_roster_stale_after_s},
    )).scalar() or 0


async def _down_baseline(session: AsyncSession, settings) -> int:
    """Stale identity_nodes rows this suite did NOT create.

    Same reasoning as _live_baseline: the local Supabase is shared with a
    real fleet node, and when that pod is stopped its row goes stale and
    counts as down. Absolute nodes_down assertions have to measure the
    pre-existing count rather than assume zero."""
    return (await session.execute(
        text(
            "SELECT count(*) FROM public.identity_nodes "
            " WHERE node_id NOT LIKE 'node-%' AND node_id <> 'fleet-sentinel' "
            "   AND last_seen_at <= now() - make_interval(secs => :s)"
        ),
        {"s": settings.identity_roster_stale_after_s},
    )).scalar() or 0


async def _clear(session: AsyncSession) -> None:
    """Isolate this test WITHOUT destroying the dev database.

    check() counts across the whole table, so a case needs a known pool.
    The obvious way to get one is to delete every patient — which is what
    an earlier version of this file did, and it cascaded through cameras,
    analyses, events and detectors and wiped the local rig. Never delete
    rows this suite did not create.

    Instead, park every pre-existing home on a freshly-heartbeating
    sentinel node. They are then healthy by definition and contribute
    nothing to the blind/overdue counts, while still being there afterwards.
    Only nodes this suite makes (node-*) are removed.
    """
    await session.execute(
        text(
            "INSERT INTO public.identity_nodes "
            "  (node_id, base_url, capacity, last_seen_at) "
            "VALUES ('fleet-sentinel', 'http://fleet-sentinel', 9999, now()) "
            "ON CONFLICT (node_id) DO UPDATE SET last_seen_at = now()"
        )
    )
    # NO port: a parked home is served by nobody, and the counts under test
    # look at the node, never the port. Handing out ports here instead trips
    # either the per-node uniqueness index (one constant for every home) or
    # the 8012-8111 range check (a counter, once the dev database has
    # accumulated more than a hundred test patients).
    await session.execute(
        text(
            "WITH parked AS ("
            "  SELECT id, NULL::int AS port"
            "    FROM public.cameras"
            # Everything the suite does not own — including homes held by a
            # REAL node that has gone quiet (a stopped dev pod makes its row
            # stale, and its homes then count as blind and fail the
            # healthy-fleet case). _restore_real_fleet puts reality back.
            "   WHERE identity_node_id IS DISTINCT FROM 'fleet-sentinel'"
            ") UPDATE public.cameras p "
            "     SET identity_node_id = 'fleet-sentinel', identity_port = parked.port "
            "    FROM parked WHERE p.id = parked.id"
        )
    )
    await session.execute(
        text("DELETE FROM public.identity_nodes WHERE node_id LIKE 'node-%'")
    )
    await session.commit()


async def _assign(
    session: AsyncSession, camera_id: UUID, node: str, *, seen_mins_ago: int = 0,
) -> None:
    """Put a SENSOR on a node. The slot is the sensor's, not the home's."""
    await session.execute(
        text(
            "INSERT INTO public.identity_nodes "
            "  (node_id, base_url, capacity, last_seen_at) "
            "VALUES (:n, :u, 4, now() - make_interval(mins => :m)) "
            "ON CONFLICT (node_id) DO UPDATE SET "
            "  last_seen_at = excluded.last_seen_at"
        ),
        {"n": node, "u": f"http://{node}", "m": seen_mins_ago},
    )
    await session.execute(
        text(
            "UPDATE public.cameras "
            "   SET identity_node_id = :n, identity_port = 8012 WHERE id = :id"
        ),
        {"n": node, "id": camera_id},
    )
    await session.commit()


def _watch(settings, sessionmaker) -> FleetWatch:
    w = FleetWatch(settings=settings, sessionmaker=sessionmaker)

    # These tests exercise the roster-table logic against fake node
    # names; the serving poll (lag alarm + occupancy, 2026-08-14) would
    # only burn HTTP timeouts against them. Its own coverage lives in
    # test_fleet_watch_serving.py.
    async def _no_state(url):  # noqa: ANN001
        return None

    w._fetch_state = _no_state
    return w


async def test_healthy_fleet_pages_nothing(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)
    pid = await _mk_home(session, cid)
    await _assign(session, pid, "node-ok", seen_mins_ago=0)

    baseline = await _live_baseline(session, settings)
    counts = await _watch(settings, get_sessionmaker()).check()
    assert counts == {"blind": 0, "overdue": 0,
                      "nodes_down": await _down_baseline(session, settings),
                      "nodes_live": 1 + baseline,
                      "cameras_polled": 0, "cameras_occupied": 0,
                      "cameras_lagging": 0}
    assert pages.triggered == {}


async def test_home_with_no_node_pages_critical(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """The end state of every other failure, and the only one with a family
    attached to it."""
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)
    # Older than the grace period, so this is an orphan and not a failover
    # that started a moment ago.
    await _mk_home(session, cid, age_mins=60)

    counts = await _watch(settings, get_sessionmaker()).check()
    assert counts["blind"] == 1
    assert counts["overdue"] == 1
    page = pages.triggered["identity-fleet:unassigned-homes"]
    assert page["severity"] == "critical"


async def test_a_brand_new_home_is_not_paged_immediately(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """A home is unassigned for a moment during every normal failover, and
    for the seconds between being created and the next roster poll. Paging
    on that trains people to ignore the pager."""
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)

    # DELTAS, not absolute counts. check() counts across the whole table, so
    # asserting "blind == 1" silently depends on no other test having left an
    # unassigned home behind — which makes the test order-dependent and its
    # failure look like a product bug. Measure before and after instead; the
    # claim is about what adding one fresh home does, and that is what is
    # actually being tested.
    before = await _watch(settings, get_sessionmaker()).check()
    pages.triggered.clear()
    pages.resolved.clear()

    await _mk_home(session, cid, age_mins=0)
    after = await _watch(settings, get_sessionmaker()).check()

    assert after["blind"] - before["blind"] == 1, "it IS blind right now"
    assert after["overdue"] == before["overdue"], (
        "a home seconds old is mid-failover, not an orphan"
    )
    assert "identity-fleet:unassigned-homes" not in pages.triggered


async def test_dead_node_is_a_warning_not_a_critical(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """A node dying is only an incident if nothing picks its homes up.

    The roster re-places them on the next poll of any node with capacity, so
    this alone is 'capacity vanished', not 'a family is blind'. Paging
    critical for a working failover is how a pager stops being read.
    """
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)
    pid = await _mk_home(session, cid, age_mins=60)
    await _assign(session, pid, "node-dead", seen_mins_ago=30)

    watch = _watch(settings, get_sessionmaker())
    counts = await watch.check()
    # Baselines measured AFTER check(), not before. The dev database is
    # shared with a real fleet node that heartbeats every ~10s, so a node
    # can cross the liveness cutoff between a before-snapshot and the
    # assertion — which failed this test intermittently for a reason that
    # had nothing to do with the behaviour under test.
    assert counts["nodes_down"] == 1 + await _down_baseline(session, settings)
    assert counts["nodes_live"] == await _live_baseline(session, settings)
    # One stale cycle is a heartbeat gap, not an incident (every backend
    # restart produces one) — the node-down page holds for a streak.
    assert "identity-fleet:node-down:node-dead" not in pages.triggered
    counts = await watch.check()
    assert pages.triggered["identity-fleet:node-down:node-dead"]["severity"] == (
        "warning"
    )
    # The home it held is blind too, and THAT is the critical one.
    assert pages.triggered["identity-fleet:unassigned-homes"]["severity"] == (
        "critical"
    )


async def test_recovery_resolves_the_incident(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """A failover that heals must close its own page, or the next real one
    is indistinguishable from the stale one still sitting open."""
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)
    pid = await _mk_home(session, cid, age_mins=60)
    await _assign(session, pid, "node-flap", seen_mins_ago=30)

    watch = _watch(settings, get_sessionmaker())
    await watch.check()
    await watch.check()  # second stale cycle crosses the streak bar
    assert "identity-fleet:node-down:node-flap" in pages.triggered
    assert "identity-fleet:unassigned-homes" in pages.triggered

    # The node comes back and re-claims.
    await _assign(session, pid, "node-flap", seen_mins_ago=0)
    baseline = await _live_baseline(session, settings)
    counts = await watch.check()

    assert counts == {"blind": 0, "overdue": 0,
                      "nodes_down": await _down_baseline(session, settings),
                      "nodes_live": 1 + baseline,
                      "cameras_polled": 0, "cameras_occupied": 0,
                      "cameras_lagging": 0}
    assert "identity-fleet:node-down:node-flap" in pages.resolved
    assert "identity-fleet:unassigned-homes" in pages.resolved


async def test_repeated_checks_do_not_repage(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """PagerDuty dedups on the key, but re-posting every cycle is noise in
    the log and in the incident timeline."""
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)
    await _mk_home(session, cid, age_mins=60)

    watch = _watch(settings, get_sessionmaker())
    await watch.check()
    calls = len(pages.triggered)
    pages.triggered.clear()
    await watch.check()
    assert calls == 1
    assert pages.triggered == {}, "re-paged an incident already open"


async def test_home_with_nothing_streaming_is_not_paged(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """The noise that made the critical page worthless (2026-08-19).

    Prod carried 10 'blind' homes and paged on every cycle; every one was a
    dormant test account with no camera online. A home that is streaming
    nothing has no identity to be deprived of.
    """
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)

    # Old enough to be well past the grace period — the ONLY thing keeping
    # it off the page is that nothing is streaming.
    pid = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.patients "
            "  (id, caregiver_id, full_name, preferred_name, relation, created_at) "
            "VALUES (:id, :cg, 'Fleet Dormant', 'Fleet', 'parent', "
            "        now() - make_interval(mins => 60))"
        ),
        {"id": pid, "cg": cid},
    )
    await session.commit()

    counts = await _watch(settings, get_sessionmaker()).check()

    assert counts["blind"] == 0
    assert counts["overdue"] == 0
    assert "identity-fleet:unassigned-homes" not in pages.triggered


async def test_a_shared_sensor_is_counted_once(
    session: AsyncSession, auth_user, settings, monkeypatch
) -> None:
    """The pager must agree with the roster about what needs a process.

    This asked whether a SECOND RESIDENT on a shared camera was paged as
    blind, because a process was allocated per patient and this module
    kept its own copy of the liveness rule; when the two disagreed, a
    resident read "live but unassigned" and paged CRITICAL every cycle,
    forever, for a system working exactly as designed.

    Sharding on the sensor removes the disagreement at the source — both
    sides now count sensors, and a sensor is one row however many people
    it covers. What is left worth pinning is that coverage cannot inflate
    the count: adding a resident must not turn one unwatched sensor into
    two pages about the same blind camera.
    """
    pages = _Pages()
    pages.install(monkeypatch)
    cid, email = auth_user
    await _clear(session)
    await _mk_caregiver(session, cid, email)
    shared = await _mk_home(session, cid)

    housemate = uuid4()
    await session.execute(
        text(
            "INSERT INTO public.patients "
            "  (id, caregiver_id, full_name, preferred_name, relation) "
            "VALUES (:id, :cg, 'Housemate', 'Housemate', 'other')"
        ),
        {"id": housemate, "cg": cid},
    )
    await session.execute(
        text("INSERT INTO public.camera_patients (camera_id, patient_id) "
             "VALUES (:c, :p) ON CONFLICT DO NOTHING"),
        {"c": shared, "p": housemate},
    )
    await session.commit()

    counts = await _watch(settings, get_sessionmaker()).check()

    assert counts["blind"] == 1, (
        "one unwatched sensor is one page, whether it watches one person "
        "or two"
    )