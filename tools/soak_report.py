"""Detection-v2 soak report (P6): aggregate a day of analyses rows
into the numbers the flag-on decision needs.

Reads ONLY the DB (no video, no providers). Run against the local
stack during Aryaman's soak day:

    uv run python tools/soak_report.py                # last 24h
    uv run python tools/soak_report.py --hours 6

Reports, per camera and total:
  ticks         analyzed / skipped (by skipped_reason)
  motion        motion_src histogram (sad / backend / sad+backend /
                none) + closed-state wake count (woke_by_backend) —
                the perpetual-wake watchlist number
  presence      people_present_src histogram — the TV-as-person
                watchlist number is a high tracker_only share on
                ticks you know the room was empty
  signals       sig_env_motion percentiles + per-track amount
                percentiles — the raw material for the P3 floor dials
  scheduler     sched_calls count histogram, degraded markers,
                sched_confirm outcomes (when the decision flag is on)
  spend         vlm/rapid token sums → $/day at current prices
  events        fired per detector, suppressed/deduped counts
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from src.db.analysis import Analysis
from src.db.event import Event
from src.db.session import get_sessionmaker

# $/1M tokens (2026-07 list prices; update alongside provider changes).
_PRICE_IN_PER_M = {"lite": 0.10, "flash": 0.30}
_PRICE_OUT_PER_M = {"lite": 0.40, "flash": 2.50}


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    i = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
    return values[i]


async def run(hours: float) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    sm = get_sessionmaker()
    async with sm() as session:
        rows = (await session.execute(
            select(Analysis).where(Analysis.captured_at >= since)
        )).scalars().all()
        events = (await session.execute(
            select(Event).where(Event.occurred_at >= since)
        )).scalars().all()

    per_cam: dict = defaultdict(lambda: {
        "ticks": 0, "skips": Counter(), "motion_src": Counter(),
        "present_src": Counter(), "wakes": 0, "env": [], "amounts": [],
        "calls": Counter(), "in": 0, "out": 0, "rapid_in": 0,
        "rapid_out": 0, "degraded": Counter(), "confirm": Counter(),
    })

    for r in rows:
        f = r.detector_fields or {}
        c = per_cam[str(r.camera_id)[:8]]
        c["ticks"] += 1
        if r.provider == "skipped":
            c["skips"][f.get("skipped_reason", "?")] += 1
        c["motion_src"][f.get("motion_src", "—")] += 1
        c["present_src"][f.get("people_present_src", "—")] += 1
        if f.get("woke_by_backend"):
            c["wakes"] += 1
        if "sig_env_motion" in f:
            c["env"].append(float(f["sig_env_motion"]))
        for t in f.get("sig_tracks", []):
            c["amounts"].append(float(t.get("amount", 0.0)))
        if "sched_calls" in f:
            c["calls"][len(f["sched_calls"])] += 1
        for k in ("sched_degraded_audio", "sched_degraded_fps"):
            if f.get(k):
                c["degraded"][k] += 1
        c["degraded"]["artifacts"] += int(f.get("sched_degraded_artifacts", 0))
        for ident, cf in (f.get("sched_confirm") or {}).items():
            c["confirm"]["passed" if cf.get("passed") else "suppressed"] += 1
        c["in"] += int(f.get("vlm_input_tokens", 0))
        c["out"] += int(f.get("vlm_output_tokens", 0))
        c["rapid_in"] += int(f.get("rapid_input_tokens", 0))
        c["rapid_out"] += int(f.get("rapid_output_tokens", 0))

    print(f"# Soak report — last {hours:g}h — {len(rows)} rows, {len(events)} events\n")
    tot_in = tot_out = 0
    for cam, c in sorted(per_cam.items()):
        tot_in += c["in"] + c["rapid_in"]
        tot_out += c["out"] + c["rapid_out"]
        print(f"## camera {cam}")
        print(f"  ticks={c['ticks']}  skips={dict(c['skips'])}")
        print(f"  motion_src={dict(c['motion_src'])}  backend_wakes={c['wakes']}")
        print(f"  presence_src={dict(c['present_src'])}")
        print(
            f"  sig_env p50={_pct(c['env'],50):.4f} p90={_pct(c['env'],90):.4f} "
            f"p99={_pct(c['env'],99):.4f}  track_amount p50="
            f"{_pct(c['amounts'],50):.4f} p90={_pct(c['amounts'],90):.4f}"
        )
        if c["calls"]:
            print(f"  sched calls/tick={dict(c['calls'])}  degraded={dict(c['degraded'])}"
                  f"  confirm={dict(c['confirm'])}")
        print(f"  tokens in={c['in']}(+{c['rapid_in']} rapid) out={c['out']}(+{c['rapid_out']})\n")

    # Cost at lite prices for the main path + flash for rapid — an
    # UPPER-ish bound; per-call tier split lives in sched_calls if you
    # need it exact.
    cost = (
        tot_in / 1e6 * _PRICE_IN_PER_M["lite"]
        + tot_out / 1e6 * _PRICE_OUT_PER_M["lite"]
    )
    day = cost * (24 / hours)
    print(f"TOTAL tokens in={tot_in} out={tot_out} ≈ ${cost:.3f} "
          f"(${day:.2f}/day extrapolated, ${day*30:.0f}/mo)")

    fired = Counter(str(e.detector_id)[:8] for e in events)
    print(f"events per detector(id8): {dict(fired)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    asyncio.run(run(args.hours))
