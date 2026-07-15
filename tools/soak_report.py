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
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select

from src.db.analysis import Analysis
from src.db.detector import Detector
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
        "seq": [],
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
        # id-consistency raw material: (when, occupied?, n_tracks)
        c["seq"].append((
            r.captured_at, bool(f.get("people_present")),
            len(f.get("sig_tracks") or []),
        ))
        if "sched_calls" in f:
            c["calls"][len(f["sched_calls"])] += 1
            for call in f["sched_calls"]:
                # RAPID spend since Ledger row 5: overlay-channel calls
                # inside the scheduler ledger (the standalone lane's
                # rapid_*_tokens row fields are gone).
                if "overlay" in str(call.get("key", "")):
                    c["rapid_in"] += int(call.get("input_tokens", 0))
                    c["rapid_out"] += int(call.get("output_tokens", 0))
        for k in ("sched_degraded_audio", "sched_degraded_fps"):
            if f.get(k):
                c["degraded"][k] += 1
        c["degraded"]["artifacts"] += int(f.get("sched_degraded_artifacts", 0))
        for ident, cf in (f.get("sched_confirm") or {}).items():
            c["confirm"]["passed" if cf.get("passed") else "suppressed"] += 1
        c["in"] += int(f.get("vlm_input_tokens", 0))
        c["out"] += int(f.get("vlm_output_tokens", 0))

    print(f"# Soak report — last {hours:g}h — {len(rows)} rows, {len(events)} events\n")
    tot_in = tot_out = rapid_in = rapid_out = 0
    for cam, c in sorted(per_cam.items()):
        # vlm_* row totals already INCLUDE overlay-call spend (executor
        # sums every call) — rapid_* is a view into them, not an addend.
        tot_in += c["in"]
        tot_out += c["out"]
        rapid_in += c["rapid_in"]
        rapid_out += c["rapid_out"]
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
        # Tracking consistency (soak recalibration, 2026-07-14): of the
        # ticks a person was believed present, how many had a live
        # track — and how often tracking dropped to zero while the room
        # stayed occupied (blips). The seated-person blindness and any
        # tracker churn both land here.
        seq = sorted(c["seq"])
        occ = sum(1 for _, p, _ in seq if p)
        trk = sum(1 for _, p, n in seq if p and n)
        losses = sum(
            1 for i in range(1, len(seq))
            if seq[i - 1][2] > 0 and seq[i][2] == 0 and seq[i][1]
        )
        pct = (100 * trk / occ) if occ else 0.0
        print(f"  id-consistency: tracked {trk}/{occ} occupied ticks "
              f"({pct:.0f}%)  losses(track→0 while occupied)={losses}")
        print(f"  tokens in={c['in']}(+{c['rapid_in']} rapid) out={c['out']}(+{c['rapid_out']})\n")

    # Tier-split pricing: overlay (flash) tokens at flash rates, the
    # rest at lite. Confirm votes on flash artifacts land in the lite
    # remainder — a slight floor when k-of-N is configured.
    cost = (
        (tot_in - rapid_in) / 1e6 * _PRICE_IN_PER_M["lite"]
        + (tot_out - rapid_out) / 1e6 * _PRICE_OUT_PER_M["lite"]
        + rapid_in / 1e6 * _PRICE_IN_PER_M["flash"]
        + rapid_out / 1e6 * _PRICE_OUT_PER_M["flash"]
    )
    day = cost * (24 / hours)
    print(f"TOTAL tokens in={tot_in} out={tot_out} ≈ ${cost:.3f} "
          f"(${day:.2f}/day extrapolated, ${day*30:.0f}/mo)")

    fired = Counter(str(e.detector_id)[:8] for e in events)
    print(f"events per detector(id8): {dict(fired)}")


# ── Truth scoring (soak protocol): TP/FN/FP against acted takes ──────
# Truth CSV columns: detector,start,end,note  (ISO timestamps, any tz).
# A truth row is a TP when an event for that detector identifier fires
# inside [start-30s, end+180s] (pre-roll slack + promotion latency).
# Every event not matched by ANY truth row is listed as a candidate FP
# for manual adjudication. For each FN, the row also reports how many
# TICKS inside the window answered detected=true — separating "model
# never saw it" from "seen on ticks but the decision layer held it".

_PRE_SLACK = timedelta(seconds=30)
_POST_SLACK = timedelta(seconds=180)


async def score_truth(truth_path: str, hours: float) -> None:
    rows = list(csv.DictReader(Path(truth_path).open()))
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    sm = get_sessionmaker()
    async with sm() as session:
        pairs = (await session.execute(
            select(Event, Detector.identifier)
            .join(Detector, Event.detector_id == Detector.id, isouter=True)
            .where(Event.created_at >= since)
        )).all()
        analyses = (await session.execute(
            select(Analysis).where(Analysis.captured_at >= since)
        )).scalars().all()

    events = [(ev, ident) for ev, ident in pairs]
    matched: set = set()
    print(f"\n# Truth scoring — {len(rows)} takes vs {len(events)} events\n")
    tally: Counter = Counter()
    for t in rows:
        ident = t["detector"].strip()
        start = datetime.fromisoformat(t["start"]).astimezone(timezone.utc)
        end = datetime.fromisoformat(t["end"]).astimezone(timezone.utc)
        lo, hi = start - _PRE_SLACK, end + _POST_SLACK
        hit = None
        for i, (ev, ev_ident) in enumerate(events):
            when = ev.occurred_at or ev.created_at
            if ev_ident == ident and lo <= when <= hi and i not in matched:
                hit = i
                break
        if hit is not None:
            matched.add(hit)
            tally[f"{ident}:TP"] += 1
            print(f"  TP  {ident:<22} {t.get('note', '')}")
        else:
            tick_hits = sum(
                1 for a in analyses
                if lo <= a.captured_at <= hi
                and (a.detector_fields or {}).get(f"{ident}_detected") is True
            )
            tally[f"{ident}:FN"] += 1
            print(
                f"  FN  {ident:<22} {t.get('note', '')}"
                f"  (ticks detected inside window: {tick_hits}"
                f"{' — decision layer held it' if tick_hits else ' — never seen'})"
            )
    unmatched = [
        (ev, ident) for i, (ev, ident) in enumerate(events) if i not in matched
    ]
    print(f"\n  Candidate FPs ({len(unmatched)}) — adjudicate manually:")
    for ev, ident in unmatched:
        when = (ev.occurred_at or ev.created_at).strftime("%H:%M:%S")
        print(f"  FP? {when}Z  {ident or '?':<22} {ev.title[:60]}")
    print(f"\n  tally: {dict(tally)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument(
        "--truth", type=str, default=None,
        help="CSV of acted takes (detector,start,end,note) to score TP/FN/FP",
    )
    args = ap.parse_args()
    asyncio.run(run(args.hours))
    if args.truth:
        asyncio.run(score_truth(args.truth, args.hours))
