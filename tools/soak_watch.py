"""Live soak watcher (detection v2): a terminal ops view over the
analyses stream — every layer's readouts, refreshed in place, while
the soak runs. DB-only; safe to leave running the whole session.

    uv run python tools/soak_watch.py                    # 30s refresh
    uv run python tools/soak_watch.py --interval 10
    uv run python tools/soak_watch.py --since 2026-07-14T14:00:00

Shows, per camera: tick cadence, the LAST tick's full layer readout
(motion source, presence source, track count + strongest track motion,
env motion, scheduler calls + per-call keys, rapid gate readouts, skip
reasons), and running counters since --since (wakes, degraded ticks,
call errors, confirm outcomes, tokens → $ so far → $/day pace).
Everything printed here lives on analyses rows — mid-soak ad-hoc
questions are one SQL query away; this is just the ambient view.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from src.db.analysis import Analysis
from src.db.detector import Detector
from src.db.event import Event
from src.db.session import get_sessionmaker
from src.settings import get_settings

_PRICE_IN = 0.10 / 1e6   # lite $/token in  (2026-07 list)
_PRICE_OUT = 0.40 / 1e6


def _age(ts: datetime, now: datetime) -> str:
    s = int((now - ts).total_seconds())
    return f"{s}s" if s < 120 else f"{s // 60}m"


async def _snapshot(sm, since: datetime):
    async with sm() as session:
        rows = (await session.execute(
            select(Analysis).where(Analysis.captured_at >= since)
            .order_by(Analysis.captured_at)
        )).scalars().all()
        events = (await session.execute(
            select(Event, Detector.identifier)
            .join(Detector, Event.detector_id == Detector.id, isouter=True)
            .where(Event.created_at >= since)
            .order_by(Event.created_at)
        )).all()
    return rows, events


def _render(rows, events, since: datetime, settings) -> str:
    now = datetime.now(timezone.utc)
    out: list[str] = []
    out.append(
        f"SOAK WATCH  {now.strftime('%H:%M:%S')}Z   since {since.strftime('%H:%M')}Z"
        f"   scheduler={settings.detection_v2_scheduler_enabled}"
        f" decision={settings.detection_v2_decision_enabled}"
        f"   floors: person={settings.scheduler_person_motion_floor}"
        f" env={settings.scheduler_env_motion_floor}"
        f" k={settings.scheduler_sample_every_k}"
    )
    by_cam: dict = {}
    for r in rows:
        by_cam.setdefault(str(r.camera_id)[:8], []).append(r)

    tot_in = tot_out = 0
    for cam, crows in sorted(by_cam.items()):
        skips = Counter()
        wakes = degraded = errors = 0
        confirm = Counter()
        for r in crows:
            f = r.detector_fields or {}
            if r.provider == "skipped":
                skips[f.get("skipped_reason", "?")] += 1
            wakes += 1 if f.get("woke_by_backend") else 0
            degraded += 1 if (
                f.get("sched_degraded_audio") or f.get("sched_degraded_fps")
                or f.get("sched_degraded_artifacts")
            ) else 0
            errors += len(f.get("sched_call_errors") or ())
            for _, cf in (f.get("sched_confirm") or {}).items():
                confirm["pass" if cf.get("passed") else "SUPPRESSED"] += 1
            tot_in += int(f.get("vlm_input_tokens", 0))
            tot_out += int(f.get("vlm_output_tokens", 0))

        last = crows[-1]
        f = last.detector_fields or {}
        tracks = f.get("sig_tracks") or []
        amounts = [float(t.get("amount", 0.0)) for t in tracks]
        calls = f.get("sched_calls") or []
        gates = [c.get("gate") for c in calls if c.get("gate")]
        fired = [
            k[:-9] for k, v in f.items()
            if k.endswith("_detected") and v is True
        ]
        amount_txt = f" max_amount={max(amounts):.4f}" if amounts else ""
        out.append(f"\n── camera {cam} — {len(crows)} ticks, last {_age(last.captured_at, now)} ago")
        out.append(
            f"   last tick: motion={f.get('motion_src', '—')}"
            f"  presence={f.get('people_present')}({f.get('people_present_src', '—')})"
            f"  tracks={len(tracks)}{amount_txt}"
        )
        out.append(
            f"   env_motion={f.get('sig_env_motion', '—')}"
            f"  calls={len(calls)} {[c.get('key', '')[:26] for c in calls]}"
        )
        if "rapid_gate_active_fraction" in f:
            out.append(
                f"   rapid_gate af={f['rapid_gate_active_fraction']}"
                f" score={f.get('rapid_gate_score')}  gate_states={gates or '—'}"
            )
        if fired:
            out.append(f"   DETECTED on last tick: {fired}")
        out.append(
            f"   window totals: skips={dict(skips) or '—'} wakes={wakes}"
            f" degraded_ticks={degraded} call_errors={errors}"
            f" confirm={dict(confirm) or '—'}"
        )

    cost = tot_in * _PRICE_IN + tot_out * _PRICE_OUT
    hours = max((now - since).total_seconds() / 3600, 1e-6)
    out.append(
        f"\nTOKENS since start: in={tot_in} out={tot_out}"
        f"  ≈ ${cost:.3f} (${cost / hours * 24:.2f}/day pace)"
    )
    if events:
        out.append("EVENTS:")
        for ev, ident in events[-12:]:
            out.append(
                f"   {ev.created_at.strftime('%H:%M:%S')}Z"
                f"  {ident or '?'}  {ev.title[:60]}"
            )
    else:
        out.append("EVENTS: none")
    return "\n".join(out)


async def run(interval: float, since: datetime) -> None:
    sm = get_sessionmaker()
    settings = get_settings()
    while True:
        try:
            rows, events = await _snapshot(sm, since)
            sys.stdout.write("\x1b[2J\x1b[H")  # clear + home
            print(_render(rows, events, since, settings))
        except Exception as exc:  # noqa: BLE001 — watcher must survive blips
            print(f"[soak_watch] refresh failed: {exc}")
        await asyncio.sleep(interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument(
        "--since", type=str, default=None,
        help="ISO start of the soak (default: now)",
    )
    args = ap.parse_args()
    start = (
        datetime.fromisoformat(args.since).astimezone(timezone.utc)
        if args.since
        else datetime.now(timezone.utc) - timedelta(minutes=1)
    )
    asyncio.run(run(args.interval, start))
