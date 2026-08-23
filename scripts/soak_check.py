#!/usr/bin/env python3
"""Assert the soak's invariants against the live system. No eyeballing.

Every check here corresponds to something that changed on
feat/camera-ownership and that would otherwise be verified by a human
squinting at a dashboard. Each prints PASS/FAIL/SKIP with the reason, and
the exit code is the number of failures, so it can be run on a loop:

    watch -n 60 'python scripts/soak_check.py'

SKIP is not a pass. It means the precondition for that check is not in
place yet (no second sensor, no identity node) and the check has not run.

Read-only: it opens one DB session and issues SELECTs. It never writes,
so it is safe to run against production as well as the soak.
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field

import httpx
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.session import get_sessionmaker  # noqa: E402
from src.settings import get_settings  # noqa: E402

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Report:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    def render(self) -> int:
        width = max(len(n) for _, n, _ in self.rows) if self.rows else 20
        icon = {PASS: "✅", FAIL: "❌", SKIP: "⏭ "}
        for status, name, detail in self.rows:
            print(f"{icon[status]} {name.ljust(width)}  {detail}")
        fails = sum(1 for s, _, _ in self.rows if s == FAIL)
        skips = sum(1 for s, _, _ in self.rows if s == SKIP)
        print(f"\n{len(self.rows)} checks — "
              f"{len(self.rows) - fails - skips} passed, {fails} failed, "
              f"{skips} skipped")
        return fails


async def _rows(session, sql: str, **params):
    return (await session.execute(text(sql), params)).all()


async def check_schema(session, rep: Report) -> None:
    """The migration's shape, so a half-applied delta is caught here and
    not by a symptom three hours into the soak."""
    cols = {r[0] for r in await _rows(
        session,
        "select table_name || '.' || column_name from information_schema.columns "
        " where table_name in ('cameras','patients','camera_patients')")}

    for want in ("cameras.caregiver_id", "cameras.audio_capture_enabled",
                 "cameras.identity_node_id", "cameras.identity_port"):
        rep.add(PASS if want in cols else FAIL, f"schema: {want}",
                "" if want in cols else "missing — delta not applied")
    for gone in ("camera_patients.is_primary", "patients.audio_capture_enabled",
                 "patients.identity_node_id", "cameras.pi_stream_url"):
        rep.add(PASS if gone not in cols else FAIL, f"schema: no {gone}",
                "" if gone not in cols else "still present — delta not applied")

    nn = await _rows(session,
                     "select is_nullable from information_schema.columns "
                     " where table_name='cameras' and column_name='caregiver_id'")
    ok = bool(nn) and nn[0][0] == "NO"
    rep.add(PASS if ok else FAIL, "schema: caregiver_id NOT NULL",
            "" if ok else "nullable — the RLS owner arm degrades to NULL")

    # CAST to text: asyncpg hands back pg's "char" type as BYTES, so a
    # comparison against "n" is always False and this check reported a
    # correct schema as broken.
    fk = await _rows(session,
                     "select confdeltype::text from pg_constraint "
                     " where conrelid='public.cameras'::regclass "
                     "   and conname='cameras_patient_id_fkey'")
    ok = bool(fk) and fk[0][0] == "n"     # 'n' = SET NULL, 'c' = CASCADE
    rep.add(PASS if ok else FAIL, "schema: patient delete keeps the sensor",
            "" if ok else "still CASCADE — deleting a patient destroys a "
                          "sensor that watches someone else")


async def check_rooms(session, rep: Report) -> None:
    """Account rooms for sensors, patient rooms only for wall phones."""
    bad = await _rows(
        session,
        "select room, livekit_room from public.cameras "
        " where source_type='gateway' and livekit_room not like 'account\\_%'")
    rep.add(PASS if not bad else FAIL, "rooms: sensors are account-scoped",
            "" if not bad else f"{len(bad)} gateway sensor(s) still in a "
                               f"patient room: {[b[0] for b in bad]}")

    orphan = await _rows(
        session,
        "select room from public.cameras where livekit_room is null "
        "   or livekit_room = 'patient_None'")
    rep.add(PASS if not orphan else FAIL, "rooms: no null/garbage room",
            "" if not orphan else f"{[o[0] for o in orphan]}")

    dupes = await _rows(
        session,
        "select caregiver_id, lower(room), count(*) from public.cameras "
        " where room is not null and room <> '' "
        " group by 1,2 having count(*) > 1")
    rep.add(PASS if not dupes else FAIL, "rooms: names unique per account",
            "" if not dupes else f"{len(dupes)} duplicate name(s) — detector "
                                 f"where_rooms will mis-scope")


async def check_coverage(session, rep: Report) -> None:
    """The shim tracks coverage, and coverage is what everything reads."""
    drift = await _rows(
        session,
        "select c.room from public.cameras c "
        " where c.patient_id is not null "
        "   and not exists (select 1 from public.camera_patients cp "
        "                    where cp.camera_id=c.id and cp.patient_id=c.patient_id)")
    rep.add(PASS if not drift else FAIL, "coverage: shim agrees with coverage",
            "" if not drift else f"{[d[0] for d in drift]} — patient_id names "
                                 f"someone who does not cover the sensor")

    shared = await _rows(
        session,
        "select c.room, count(*) from public.cameras c "
        "  join public.camera_patients cp on cp.camera_id=c.id "
        " group by c.room having count(*) > 1")
    if shared:
        rep.add(PASS, "coverage: a shared sensor exists",
                f"{shared[0][0]} covers {shared[0][1]} patients — "
                f"attribution is under test")
    else:
        rep.add(SKIP, "coverage: a shared sensor exists",
                "assign one sensor to 2 patients to exercise attribution")


async def check_fleet(session, rep: Report) -> None:
    """Sensors are streaming, and the bridge is scoped as a bridge."""
    cams = await _rows(
        session,
        "select room, source_type, online, desired_state, last_heartbeat_at "
        "  from public.cameras order by created_at")
    if not cams:
        rep.add(SKIP, "fleet: sensors online", "no cameras provisioned")
    for room, src, online, desired, hb in cams:
        want_on = desired == "connected"
        ok = (online is True) if want_on else True
        rep.add(PASS if ok else FAIL, f"fleet: {room} ({src}) online",
                "" if ok else f"desired={desired} but online={online} — "
                              f"publishing but the backend cannot see it, "
                              f"or the publisher is down")

    mt = await _rows(session,
                     "select multi_tenant, count(*) from public.gateways "
                     " group by 1")
    kinds = {bool(m): n for m, n in mt}
    if kinds.get(True):
        rep.add(PASS, "fleet: a bridge is enrolled multi-tenant",
                f"{kinds[True]} service(s)")
    else:
        rep.add(SKIP, "fleet: a bridge is enrolled multi-tenant",
                "no camera-bridge enrolled yet (Tapo gateway is single-tenant "
                "by design)")


async def check_identity(session, rep: Report) -> None:
    """One slot per SENSOR, and the node actually holds what it claims."""
    settings = get_settings()
    stale_s = settings.identity_roster_stale_after_s

    live = await _rows(
        session,
        "select node_id, base_url, capacity from public.identity_nodes "
        " where last_seen_at > now() - make_interval(secs => :s)", s=stale_s)
    if not live:
        rep.add(SKIP, "identity: a node is heartbeating",
                "no node polled the roster recently — identity is off, "
                "detection still runs (it fails open)")
        return
    rep.add(PASS, "identity: a node is heartbeating",
            f"{[n[0] for n in live]}")

    dangling = await _rows(
        session,
        "select room from public.cameras c where c.identity_node_id is not null "
        "  and not exists (select 1 from public.identity_nodes n "
        "                   where n.node_id = c.identity_node_id)")
    rep.add(PASS if not dangling else FAIL, "identity: no dangling node ref",
            "" if not dangling else f"{[d[0] for d in dangling]} — the roster "
                                    f"can never release these")

    clash = await _rows(
        session,
        "select identity_node_id, identity_port, count(*) from public.cameras "
        " where identity_node_id is not null and identity_port is not null "
        " group by 1,2 having count(*) > 1")
    rep.add(PASS if not clash else FAIL, "identity: one sensor per port",
            "" if not clash else f"{len(clash)} port collision(s)")

    blind = await _rows(
        session,
        "select room from public.cameras "
        " where online and not disabled and identity_node_id is null")
    rep.add(PASS if not blind else FAIL, "identity: every live sensor has a slot",
            "" if not blind else f"{[b[0] for b in blind]} streaming with no "
                                 f"identity process")

    # The node's own view, which is the half that lied in production.
    for node_id, base_url, _cap in live:
        ports = await _rows(
            session,
            "select identity_port from public.cameras "
            " where identity_node_id = :n and identity_port is not null",
            n=node_id)
        if not ports:
            rep.add(SKIP, f"identity: {node_id} serving", "holds no sensors")
            continue
        reachable = 0
        async with httpx.AsyncClient(timeout=5) as client:
            for (port,) in ports:
                try:
                    r = await client.get(f"{base_url.rstrip('/')}:{port}/state")
                    if r.status_code == 200:
                        reachable += 1
                except Exception:  # noqa: BLE001 — unreachable is the finding
                    pass
        ok = reachable == len(ports)
        rep.add(PASS if ok else FAIL, f"identity: {node_id} serving",
                f"{reachable}/{len(ports)} ports answering /state"
                + ("" if ok else " — the roster believes it holds sensors it "
                                 "is not actually running"))


async def main() -> int:
    rep = Report()
    sm = get_sessionmaker()
    async with sm() as session:
        await check_schema(session, rep)
        await check_rooms(session, rep)
        await check_coverage(session, rep)
        await check_fleet(session, rep)
        await check_identity(session, rep)
    return rep.render()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
