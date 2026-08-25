"""Live local soak dashboard (detection v2) — the genuinely-live view.

A standalone read-only web app: queries the DB directly (never touches
the backend under test), serves one page that polls itself every 10 s —
per-camera layer readouts, scrollable tick log, events, running $ pace,
and the newest identity-evidence frames inline.

    uv run python tools/soak_dashboard.py                    # port 8010
    uv run python tools/soak_dashboard.py --since 2026-07-14T15:45:00-07:00

Open http://localhost:8010 (or http://<mac-lan-ip>:8010 from a phone).
"""

from __future__ import annotations

import asyncio
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse,
)
from sqlalchemy import select, text

from src.db.analysis import Analysis
from src.db.detector import Detector
from src.db.event import Event
from src.db.session import get_sessionmaker
from src.settings import get_settings

# $/token by tier (2026-07 list). The scheduler's per-call ledger
# encodes the tier inside each plan key, so sched rows price EXACTLY;
# tokens outside the ledger (legacy rows, confirm votes) price as lite
# — a slight floor, never an overstatement... except flash confirms,
# noted in the report.
# LIST PRICES, verified 2026-07-28 against the published Gemini API
# pricing (and the 3.5-flash-lite list rate exactly matches the
# 2026-07-19 bill-calibrated effective rate of the old lite tier, which
# validates the calibration method):
#   gemini-3.6-flash       $1.50/M in   $7.50/M out   <- "flash" tier
#   gemini-3.5-flash-lite  $0.30/M in   $2.50/M out   <- "lite" tier
# NOTE the 5x/3x flash-vs-lite gap: with 3.6-flash as PRIMARY the same
# call volume that billed ~$14/day on lite projects to ~$55+/day.
# provider usage["output"] already includes thinking tokens (billed at
# the output rate), so output-token pricing here is bill-accurate.
# Re-verify against AI Studio at the next top-up.
_PRICE = {"lite": (0.30 / 1e6, 2.50 / 1e6), "flash": (1.50 / 1e6, 7.50 / 1e6)}
# Per-MODEL prices — preferred when the ledger entry carries "model"
# (recorded from 2026-07-28 on). The recipe-tier keys above are the
# LEGACY fallback only: recipe tier no longer implies the model (every
# routed call runs settings.gemini_model regardless of tier), so
# tier-priced legacy rows understate whenever the primary is a flash-
# class model.
_MODEL_PRICE = {
    "gemini-3.6-flash":      (1.50 / 1e6, 7.50 / 1e6),
    "gemini-3.5-flash-lite": (0.30 / 1e6, 2.50 / 1e6),
    "gemini-3.1-flash-lite": (0.30 / 1e6, 2.50 / 1e6),
}

app = FastAPI()
_SINCE: datetime = datetime.now(timezone.utc)
_FRAMES_DIR = Path("out/id_frames")
_TRACE_PATH = Path("out/track_trace.jsonl")

# ONE identity process — the home under observation.
#
# In production a node runs one process per room on roster-assigned ports
# from 8012 up, across several nodes. This dashboard follows ONE home; it
# is a local soak tool, not an operations console.
#
# The address is resolved from the ROSTER (cameras.identity_node_id ->
# identity_nodes.base_url + cameras.identity_port), cached 30s, so the
# dashboard follows fleet swaps instead of pointing at dead nodes — on
# 2026-08-13 it sat aimed at a terminated node until restarted by hand.
# SOAK_IDENTITY_BASE remains as an explicit override (rigs with no DB).
#
# EVERY sensor, resolved fresh (2026-08-24). This used to filter on a
# hardcoded patient name and take `LIMIT 1`, so the dashboard showed
# exactly ONE identity service — the lowest port — no matter how many
# were running. Adding the second sensor did not break it loudly; it
# simply never appeared, and the tool whose entire job is telling us
# what the test rig is doing quietly described half of it. Every camera
# added from here would have done the same.
#
# There is no patient filter any more. A soak dashboard should show what
# is ON, and "which resident" is not a property of a sensor that can
# cover several of them.
IDENTITY_BASE = os.environ.get("SOAK_IDENTITY_BASE", "").rstrip("/")
log = logging.getLogger(__name__)

# camera8 -> {"base": "http://host:port", "name": "Aryaman Bedroom"}
_sensors_cache: dict = {"t": 0.0, "v": {}}


async def sensors() -> dict[str, dict]:
    """Every sensor with an identity slot: `{cam8: {base, name}}`.

    The camera's OWN name (`cameras.room`) travels with it, so nothing
    downstream has to fall back to printing a raw uuid fragment when one
    identity service happens not to know a camera another one owns.
    """
    now = time.time()
    if _sensors_cache["v"] and now - _sensors_cache["t"] < 20:
        return _sensors_cache["v"]
    found: dict[str, dict] = {}
    try:
        sm = get_sessionmaker()
        async with sm() as session:
            rows = (await session.execute(text(
                "SELECT substr(c.id::text, 1, 8) AS cam8, "
                "       c.room, n.base_url, c.identity_port "
                "  FROM public.cameras c "
                "  JOIN public.identity_nodes n "
                "    ON n.node_id = c.identity_node_id "
                " WHERE c.identity_port IS NOT NULL "
                "   AND NOT c.disabled "
                " ORDER BY c.room"
            ))).all()
        for cam8, room, base, port in rows:
            if base and port:
                found[cam8] = {
                    "base": f"{base.rstrip('/')}:{port}",
                    "name": room or cam8,
                }
    except Exception:  # noqa: BLE001 — keep serving the last-known set
        # Logged, not swallowed. A silent fallback is how a schema change
        # made this dashboard useless without ever saying so.
        log.warning("soak_dashboard: sensor lookup failed", exc_info=True)
        return _sensors_cache["v"] or {}
    if IDENTITY_BASE:
        # Explicit override: point every sensor at the one address.
        found = {k: {**v, "base": IDENTITY_BASE} for k, v in found.items()}
    _sensors_cache.update(t=now, v=found)
    return found


async def IB(cam_id: str | None = None) -> str:
    """Base URL for ONE camera's identity service.

    Kept so the single-camera endpoints read the same as before. With no
    camera it returns any known service, which is only correct for calls
    that are not about a particular sensor.
    """
    smap = await sensors()
    if cam_id:
        hit = smap.get(cam_id[:8])
        if hit:
            return hit["base"]
    if smap:
        return next(iter(smap.values()))["base"]
    return IDENTITY_BASE or "http://localhost:8012"

# All displayed times are LAPTOP-LOCAL (Aryaman 2026-07-19 — the UTC
# everywhere made cross-referencing painful; evidence-frame filenames
# were already local, so this also unifies the two).
_LOCAL_TZ = datetime.now().astimezone().tzinfo
_TZ_NAME = datetime.now().astimezone().tzname() or "local"
_TZ_OFF = datetime.now().astimezone().utcoffset() or timedelta(0)


def _local_hms(dt: datetime) -> str:
    # DB timestamps arrive NAIVE (timestamp without time zone, UTC by
    # convention). astimezone() on a naive dt assumes it's already
    # local — a silent no-op that displayed UTC digits as if local.
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_LOCAL_TZ).strftime("%H:%M:%S")

# Hourly rollup cache — the lateral jsonb explode over a full soak day
# is too heavy to run on every 10s poll; 55s staleness is invisible on
# an hours-scale chart.
_HOURLY_CACHE: tuple[float, list] | None = None

# camera-id-prefix → room name, for human-readable lanes/log. Loaded
# once — camera rows don't change mid-soak.
_CAM_NAMES: dict[str, str] | None = None

# Per-hour sent-call + token rollup for the whole soak window. A call
# counts as SENT when its ledger entry has no 'gate' key (gated plans
# were never dispatched, $0). Tokens = per-call ledger sums plus the
# row-level fallback fields legacy rows carry.
_HOURLY_SQL = text("""
select to_char(date_trunc('hour', a.created_at + :tzoff), 'MM-DD HH24:00') as h,
       date_trunc('hour', a.created_at + :tzoff)          as hts,
       coalesce(a.detector_fields->>'vlm_model', '')     as m,
       count(*) filter (where a.provider != 'skipped')  as analyzed,
       coalesce(sum(c.calls_sent), 0)                    as calls,
       -- Row totals (vlm_*_tokens) already CONTAIN the per-call ledger
       -- (sched_calls); adding both double-counted every analyzed row.
       -- greatest() takes whichever side is populated, per row.
       coalesce(sum(greatest(
           coalesce(c.led_in, 0),
           coalesce((a.detector_fields->>'vlm_input_tokens')::bigint, 0))), 0)  as tin,
       coalesce(sum(greatest(
           coalesce(c.led_out, 0),
           coalesce((a.detector_fields->>'vlm_output_tokens')::bigint, 0))), 0) as tout,
       count(*) filter (where a.provider = 'gemini')     as g,
       count(*) filter (where a.provider = 'dashscope')  as d,
       count(*) filter (where a.provider = 'openrouter') as o,
       -- Analyzed rows that report NO usage at all (neither ledger nor
       -- row fields). They price as $0, which is a LIE of omission —
       -- OpenRouter/Qwen calls currently land here (59 in the first
       -- 19 h of this soak). Counted so the page can say its cost is a
       -- FLOOR instead of quietly serving a total.
       count(*) filter (
           where a.provider != 'skipped'
             and greatest(
                 coalesce(c.led_in, 0),
                 coalesce((a.detector_fields->>'vlm_input_tokens')::bigint, 0)
             ) = 0
       ) as unpriced
from analyses a
left join lateral (
    select count(*) filter (where not (e ? 'gate'))            as calls_sent,
           sum((e->>'input_tokens')::bigint)                   as led_in,
           sum((e->>'output_tokens')::bigint)                  as led_out
    from jsonb_array_elements(
        coalesce(a.detector_fields->'sched_calls', '[]'::jsonb)) e
) c on true
where a.created_at >= :since
-- 1=hour label, 2=hour timestamp, 3=model. All three, or the model
-- collapses and every row prices at the first model seen.
group by 1, 2, 3 order by 2
""")


async def _hourly(s) -> list:
    global _HOURLY_CACHE
    now = time.monotonic()
    if _HOURLY_CACHE is not None and now - _HOURLY_CACHE[0] < 55:
        return _HOURLY_CACHE[1]
    rows = (await s.execute(
        _HOURLY_SQL, {"since": _SINCE, "tzoff": _TZ_OFF},
    )).all()
    # Cost priced server-side, per RECORDED MODEL (rows are grouped by
    # hour + vlm_model; unattributed legacy rows price as lite). Keeps
    # the chart bill-accurate now that the primary is a flash-class
    # model — the old single-lite pricing would understate bars ~5x.
    #
    # These buckets are also the ONLY source of the page's headline
    # $ figures (2026-07-29): they cover the WHOLE --since window,
    # whereas the row loop in data() is capped at 400 rows for the tick
    # log. Deriving totals from the loop understated the headline by
    # the ratio of (window / 400 rows) — ~17x into a soak's second
    # hour, which read as "$2/day" against a real ~$1.67/HOUR. One
    # aggregate, one price table, no second accumulator.
    by_hour: dict[str, dict] = {}
    for r in rows:
        pin, pout = _MODEL_PRICE.get(r.m, _PRICE["lite"])
        # Flash-CLASS is a price question, not a name question: key off
        # the resolved rate, so a renamed or newly added flash model is
        # counted the moment it lands in _MODEL_PRICE. (The old flag read the
        # recipe tier out of the ledger call key, which this file's
        # header already documents as no longer implying the model.)
        is_flash = (pin, pout) == _PRICE["flash"]
        b = by_hour.setdefault(r.h, {
            # LOCAL-NAIVE, deliberately. The SQL already shifted this into
            # local time (`created_at + :tzoff`), but `created_at` is a
            # timestamptz so the result stays tz-AWARE — carrying that
            # offset through meant the trailing-24h sum compared an aware
            # stamp against a naive cutoff and the whole /data endpoint
            # 500'd with "can't compare offset-naive and offset-aware".
            # Drop the tzinfo here so the JSON contract matches what every
            # consumer already assumes (see cutoff_24h below).
            "h": r.h, "hts": r.hts.replace(tzinfo=None).isoformat(),
            "an": 0, "calls": 0,
            "tin": 0, "tout": 0,
            "cost": 0.0, "ftok": 0, "unp": 0, "g": 0, "d": 0, "o": 0,
        })
        b["an"] += r.analyzed
        b["calls"] += int(r.calls)
        b["tin"] += int(r.tin)
        b["tout"] += int(r.tout)
        b["unp"] += int(r.unpriced)
        b["cost"] = round(
            b["cost"] + int(r.tin) * pin + int(r.tout) * pout, 4)
        if is_flash:
            b["ftok"] += int(r.tin) + int(r.tout)
        b["g"] += r.g; b["d"] += r.d; b["o"] += r.o
    out = list(by_hour.values())
    _HOURLY_CACHE = (now, out)
    return out


async def _cam_names(s=None) -> dict:
    """8-hex camera prefix → room name (cached). Every card shows the
    NAME; ids are hover/debug detail (Aryaman: names all over)."""
    global _CAM_NAMES
    if _CAM_NAMES is None:
        if s is not None:
            cam_rows = (await s.execute(text(
                "select substr(id::text, 1, 8) as c8, room from cameras"
            ))).all()
            _CAM_NAMES = {r.c8: r.room for r in cam_rows}
        else:
            sm = get_sessionmaker()
            async with sm() as s2:
                cam_rows = (await s2.execute(text(
                    "select substr(id::text, 1, 8) as c8, room from cameras"
                ))).all()
                _CAM_NAMES = {r.c8: r.room for r in cam_rows}
    return _CAM_NAMES


@app.get("/data")
async def data() -> JSONResponse:
    settings = get_settings()
    sm = get_sessionmaker()
    async with sm() as s:
        rows = (await s.execute(
            select(Analysis).where(Analysis.captured_at >= _SINCE)
            .order_by(Analysis.captured_at.desc()).limit(400)
        )).scalars().all()
        await _cam_names(s)
        events = (await s.execute(
            select(Event, Detector.identifier)
            .join(Detector, Event.detector_id == Detector.id, isouter=True)
            .where(Event.created_at >= _SINCE)
            .order_by(Event.created_at.desc()).limit(30)
        )).all()
        hourly = await _hourly(s)

    cams: dict[str, dict] = {}
    ticks = []
    idticks = []
    # NOTE: no token/cost accumulation in this loop. `rows` is the
    # newest-400 slice that feeds the tick log and camera cards; summing
    # money over it produced a headline that silently described only the
    # last ~hour while being divided by the full window's hours. The $
    # figures come from _hourly() (whole window) below.
    for r in rows:                       # newest → oldest
        f = r.detector_fields or {}
        cam = str(r.camera_id)[:8]
        c = cams.setdefault(cam, {
            "cam": cam, "ticks": 0, "skips": Counter(), "wakes": 0,
            "degraded": 0, "errors": 0, "confirm": Counter(), "last": None,
        })
        c["ticks"] += 1
        if r.provider == "skipped":
            c["skips"][f.get("skipped_reason", "?")] += 1
        if f.get("woke_by_backend"):
            c["wakes"] += 1
            # Rows iterate newest→oldest, so the first wake seen is
            # the most recent — surfaced on the camera card so "did
            # it wake?" never requires scrolling the log.
            if c.get("lastwake") is None:
                c["lastwake"] = _local_hms(r.captured_at)
        c["degraded"] += 1 if (
            f.get("sched_degraded_audio") or f.get("sched_degraded_fps")
            or f.get("sched_degraded_artifacts")
        ) else 0
        c["errors"] += len(f.get("sched_call_errors") or ())
        for _, cf in (f.get("sched_confirm") or {}).items():
            c["confirm"]["pass" if cf.get("passed") else "SUPPRESSED"] += 1
        tracks = f.get("sig_tracks") or []
        amounts = [float(t.get("amount", 0.0)) for t in tracks]
        calls = f.get("sched_calls") or []
        fired = [k[:-9] for k, v in f.items() if k.endswith("_detected") and v is True]
        tick = {
            "t": _local_hms(r.captured_at),
            # Action→verdict: window end (captured_at) → row save. The
            # per-call ms breakdown rides sched_calls[].ms.
            "lag": round((r.created_at - r.captured_at).total_seconds(), 1)
            if r.created_at else None,
            "cam": cam,
            "skip": f.get("skipped_reason") if r.provider == "skipped" else None,
            "motion": f.get("motion_src"),
            "ppl": bool(f.get("people_present")),
            "psrc": f.get("people_present_src"),
            "tracks": len(tracks),
            "amount": round(max(amounts), 4) if amounts else None,
            "env": f.get("sig_env_motion"),
            "calls": [
                {
                    "k": c2.get("key", ""),
                    "n": len(c2.get("detectors") or []),
                    "g": c2.get("gate"),
                }
                for c2 in calls
            ],
            "gaf": f.get("rapid_gate_active_fraction"),
            # SDK video-delivery health (ingress meter): received fps →
            # kept fps, worst inter-frame gap. Present on skip rows too.
            "rfps": f.get("ingress_recv_fps"),
            "kfps": f.get("ingress_kept_fps"),
            "gap": f.get("ingress_max_gap_ms"),
            # tick_failed rows (any tick error, not just model): class +
            # first line of the message, rendered crit-red in the log.
            "err": f.get("error_class"),
            "emsg": (f.get("error_msg") or "")[:90] or None,
            # Identity verdict for the per-camera timeline lane.
            "pat": bool(f.get("identity_patient_in_clip")),
            "woke": bool(f.get("woke_by_backend")),
            "fired": fired,
        }
        if c["last"] is None:
            c["last"] = tick
        # Identity timeline rides ALL fetched rows (~1h+/camera), not
        # the 120-row log cap — the lane needs the longer horizon.
        # Lane color reads the ACTUAL binding labels (the same source
        # as the box), not the scheduler skip reason. A tick is red
        # ONLY when a track is bound OTHER and no patient is present;
        # everyone-UNKNOWN is amber, never red (2026-07-22: red flapped
        # on unknown people because it keyed off the suppression gate).
        _trklabels = f.get("id_tracks") or {}
        _labelset = (
            set(_trklabels.values()) if isinstance(_trklabels, dict) else set()
        )
        _has_patient = tick["pat"] or f.get("id_binding") == "patient"
        # RED means GENUINELY verified-not-patient — an actual OTHER
        # binding, or a skip the gate justified with
        # has_only_verified_other_tracks. A skip fired only because a
        # face was DETECTED (not matched) while the patient sits in
        # another room is NOT a confirmation: it renders amber/grey,
        # never red (2026-07-22, Aryaman: "are you sure it didn't
        # actually confirm not patient?" — it did not).
        _verified_other = (
            "OTHER" in _labelset
            or f.get("identity_has_only_verified_other_tracks") is True
            or f.get("has_only_verified_other_tracks") is True
        )
        idticks.append({
            "cam": cam, "t": tick["t"], "pat": tick["pat"],
            "trk": tick["tracks"], "skip": tick["skip"],
            "oth": _verified_other and not _has_patient,
            "deg": bool(f.get("quality_degraded")),
        })
        if len(ticks) < 120:
            ticks.append(tick)

    for c in cams.values():
        c["skips"] = dict(c["skips"])
        c["confirm"] = dict(c["confirm"])

    frames = []
    if _FRAMES_DIR.exists():
        for camdir in sorted(_FRAMES_DIR.iterdir()):
            if not camdir.is_dir():
                continue
            newest = sorted(camdir.glob("*.jpg"))[-40:]
            for p in reversed(newest):        # newest first
                stem = p.stem
                # Filenames are YYYYMMDDTHHMMSS[_note], stamped by
                # dump_id_evidence with the CONTAINER's clock — which
                # is UTC in Docker. Parse as UTC, display local.
                # epoch drives the gap chips between thumbnails.
                ts = stem.split("_", 1)[0]
                try:
                    tm = time.strptime(ts, "%Y%m%dT%H%M%S")
                    dtu = datetime(*tm[:6], tzinfo=timezone.utc)
                    epoch = dtu.timestamp()
                    hms = dtu.astimezone(_LOCAL_TZ).strftime("%H:%M:%S")
                except ValueError:
                    epoch, hms = None, ""
                frames.append({
                    "cam": camdir.name,
                    "name": p.name,
                    "hms": hms,
                    "epoch": epoch,
                    # capture reason is encoded in the filename
                    # (…_id-change-1p-unk.jpg) by dump_id_evidence
                    "note": (
                        stem.split("_", 1)[1].replace("-", " ")
                        if "_" in stem else ""
                    ),
                })

    hours = max((datetime.now(timezone.utc) - _SINCE).total_seconds() / 3600, 1e-6)

    # ── money, from the whole-window hourly buckets (never from `rows`)
    tot_in = sum(int(b["tin"]) for b in hourly)
    tot_out = sum(int(b["tout"]) for b in hourly)
    flash_tok = sum(int(b["ftok"]) for b in hourly)
    cost = sum(float(b["cost"]) for b in hourly)
    unpriced = sum(int(b["unp"]) for b in hourly)

    # Rates. All MEASURED — no single-hour figure is ever multiplied out
    # to a day (2026-07-29: quoting the busiest hour × 24 read as a
    # "$53/day pace" against a real ~$26/day; the peak hour is a ceiling,
    # not a cadence, and hours here range $0.02–$2.39).
    #   day_pace  — cost/elapsed×24 over the whole window. The honest
    #               all-in number once the window covers a real day.
    #   hour_rate — the last COMPLETE hour bucket = $/hr at that hour's
    #               occupancy. Judge "is an active hour expensive?" by
    #               this; the in-progress hour is excluded so a bucket
    #               five minutes old never reads as a cheap hour.
    #   day_24h   — actual spend over the trailing 24 wall-clock hours,
    #               once the window is that long. This is the real
    #               day number; it replaces any extrapolation.
    # An hour with NO analyses produces no bucket at all, so the last
    # complete bucket is not necessarily the previous wall-clock hour.
    # It ships with its own label and the UI prints it — an idle-gap
    # rate must never masquerade as the current burn.
    complete = hourly[:-1] if len(hourly) > 1 else []
    hour_rate = float(complete[-1]["cost"]) if complete else None
    hour_label = str(complete[-1]["h"]) if complete else None

    # Trailing 24 h, summed from the buckets (local-naive hour stamps).
    local_now_naive = datetime.now(_LOCAL_TZ).replace(tzinfo=None)
    cutoff_24h = local_now_naive - timedelta(hours=24)
    day_24h = (
        round(sum(
            float(b["cost"]) for b in hourly
            if datetime.fromisoformat(b["hts"]) >= cutoff_24h
        ), 4)
        if hours >= 24 else None
    )
    return JSONResponse({
        "now": _local_hms(datetime.now(timezone.utc)),
        "tz": _TZ_NAME,
        "since": _SINCE.astimezone(_LOCAL_TZ).strftime(
            f"%H:%M {_TZ_NAME}",
        ),
        "hourly": hourly,
        "flags": {
            "scheduler": settings.detection_v2_scheduler_enabled,
            "decision": settings.detection_v2_decision_enabled,
            "id_frames": getattr(settings, "soak_id_frames_enabled", False),
        },
        "cams": sorted(cams.values(), key=lambda c: c["cam"]),
        "names": _CAM_NAMES or {},
        "idt": idticks,
        "ticks": ticks,
        "events": [
            {
                "t": _local_hms(ev.created_at),
                "d": ident or "?",
                "title": ev.title[:70],
            }
            for ev, ident in events
        ],
        "totals": {
            "in": tot_in, "out": tot_out, "flash_tokens": flash_tok,
            "cost": round(cost, 4),
            "hours": round(hours, 2),
            # Whole-window average, not a 400-row sample. None until a
            # full hour has elapsed: extrapolating the first minutes of a
            # soak to a day produced absurd figures (47 seconds of a
            # restart burst read as "$365/day"), which is the same
            # over-extrapolation the removed hour_pace was guilty of.
            "day_pace": round(cost / hours * 24, 2) if hours >= 1 else None,
            "hour_rate": None if hour_rate is None else round(hour_rate, 4),
            "hour_label": hour_label,
            "day_24h": day_24h,
            # >0 means `cost` is a FLOOR: that many analyzed calls
            # reported no usage and priced as $0.
            "unpriced_calls": unpriced,
        },
        "frames": frames,
    })


@app.get("/frames/{cam}/{name}")
async def frame(cam: str, name: str):
    p = (_FRAMES_DIR / Path(cam).name / Path(name).name).resolve()
    if not str(p).startswith(str(_FRAMES_DIR.resolve())) or not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")


_CROPS_DIR = Path(__file__).resolve().parents[1] / "out" / "live-crops"


@app.get("/crops/{kind}")
async def crop_artifact(kind: str):
    f = _CROPS_DIR / f"latest_{kind}.mp4"
    if kind not in ("full", "person") or not f.is_file():
        return JSONResponse({"error": "none"}, status_code=404)
    label = _CROPS_DIR / f"latest_{kind}.txt"
    return FileResponse(f, media_type="video/mp4", headers={
        "Cache-Control": "no-store",
        "X-Age-S": str(round(time.time() - f.stat().st_mtime, 1)),
        "X-Detectors": label.read_text() if label.is_file() else "",
    })


def _pctl(vals: list[float], p: float):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p))]


def _read_trace(limit_bytes: int = 900_000) -> dict:
    """Tail the eval trace JSONL, group by (cam, tick), compute metrics.

    Deliberately NOT computing acquire/release/FP latency here: those need
    the ground-truth 'when did the person enter / is this a guest' that only
    Aryaman supplies by labelling tids. Faking them from the events would be
    wrong. We surface what IS computable — per-stage latency, the body-score
    distribution, event counts, and the reset/contamination flag — and show
    the raw event timeline so the ground-truth overlay is a read, not a guess.
    """
    if not _TRACE_PATH.exists():
        return {"on": False}
    try:
        with open(_TRACE_PATH, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - limit_bytes))
            data = f.read().decode("utf-8", "ignore")
    except OSError:
        return {"on": False}
    lines = data.splitlines()
    if size > limit_bytes and lines:
        lines = lines[1:]  # drop the partial first line
    ticks: dict = {}
    resets: list = []
    lat_e2e, lat_track, lat_face = [], [], []
    body_pat: list = []
    ev = Counter()
    arm = "?"
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        t = r.get("type")
        cam = r.get("cam", "?")
        if t == "reset":
            resets.append(r)
            continue
        # Bucket by (cam, capture_ts) — NOT tick number. The backend's
        # tick counter resets on every restart/hot-reload, so number-
        # keyed buckets merge rows from different runs and freeze the
        # card at the oldest generation's timestamps (live 2026-07-26:
        # "the ticks aren't flowing" while rows appended fine). Every
        # row of one tick inherits the identical capture_ts from the
        # trace context, so ts IS the tick's identity.
        tk = ticks.setdefault((cam, r.get("ts")), {
            "cam": cam, "tick": r.get("tick"), "ts": r.get("ts"),
            "body": [], "idev": [], "harvest": [], "lat": None, "arm": None,
        })
        if t == "body":
            tk["body"].append(r)
            # One score per patient the camera covers (2026-08-20 — the
            # gallery stopped emitting a fixed patient/other pair). The
            # distribution card wants the winning score per attempt, so
            # take the best; a shared camera contributes one sample per
            # probe, same as a private one.
            if r.get("scores"):
                body_pat.append(max(r["scores"].values()))
        elif t == "idevent":
            tk["idev"].append(r)
            ev[r.get("event", "?")] += 1
        elif t == "harvest":
            tk["harvest"].append(r)
            ev["HARVEST"] += 1
        elif t == "latency":
            tk["lat"] = r
            tk["arm"] = arm = r.get("arm", arm)
            if r.get("e2e_ms"):
                lat_e2e.append(r["e2e_ms"])
            if r.get("track_ms"):
                lat_track.append(r["track_ms"])
            if r.get("face_ms"):
                lat_face.append(r["face_ms"])
    ordered = sorted(
        ticks.values(), key=lambda x: (x["ts"] or 0), reverse=True,
    )[:40]
    nz = [v for v in body_pat if v > 0]
    return {
        "on": True,
        "arm": arm,
        "resets": len(resets),
        "metrics": {
            "e2e_p50": _pctl(lat_e2e, 0.5), "e2e_max": max(lat_e2e) if lat_e2e else None,
            "e2e_n": len(lat_e2e),
            "track_p50": _pctl(lat_track, 0.5), "face_p50": _pctl(lat_face, 0.5),
            "body_n": len(body_pat), "body_nz": len(nz),
            "body_min": min(nz) if nz else None, "body_med": _pctl(nz, 0.5),
            "body_max": max(nz) if nz else None,
            "events": dict(ev),
        },
        "ticks": ordered,
    }


@app.get("/trace")
async def trace_data():
    out = _read_trace()
    try:
        names = await _cam_names()
        for tk in out.get("ticks", []):
            c8 = str(tk.get("cam", ""))[:8]
            tk["cam_name"] = names.get(c8, c8)
    except Exception:  # noqa: BLE001 — names are cosmetic, never 500
        pass
    return JSONResponse(out)


def _ident_headers() -> dict[str, str]:
    """Bearer for the identity service, read fresh each call so the
    dashboard picks up a token set after it started."""
    tok = os.environ.get("NEMARA_IDENTITY_STATE_TOKEN", "")
    if not tok:
        # Fall back to Settings, which reads .env. Requiring the token to
        # be exported meant `uv run python tools/soak_dashboard.py` — the
        # obvious way to start it, and the one in this file's own header
        # — produced a page that authenticated with nothing, got 401s,
        # and rendered an empty rig as a healthy one.
        try:
            tok = getattr(get_settings(), "identity_state_token", "") or ""
        except Exception:  # noqa: BLE001 — a dashboard, not the backend
            tok = ""
    return {"Authorization": f"Bearer {tok}"} if tok else {}


@app.get("/identity_stream/{cam_id}")
async def identity_stream(cam_id: str):
    """Proxy the identity service's annotated MJPEG stream.

    The page used to embed :8012/stream directly in an <img>, which
    cannot carry an Authorization header — so token-gating that
    endpoint would have blanked the live view. Proxying keeps the
    token server-side. DEV ONLY, same as the endpoint it fronts: this
    is raw camera video with no room-privacy gating, and prod has no
    live-feed path that bypasses the privacy rules.
    """
    import asyncio

    import httpx

    # FRAME-DROPPING pump (2026-08-12). The identity service moved from
    # localhost onto a GPU node reached over the tailnet; MJPEG has no
    # frame dropping of its own, so when the node produces frames faster
    # than the transocean path carries them, the backlog queues and the
    # view trails wall clock by an ever-growing margin (~30s observed).
    # Fix: drain the upstream as fast as the pipe allows, keep ONLY the
    # newest complete part, forward at a fixed ~6fps. Latency becomes
    # one frame + RTT regardless of upstream fps or path bandwidth.
    async def _pump():
        BOUNDARY = b"--frame"
        state: dict = {}

        async def _reader():
            buf = b""
            try:
                async with httpx.AsyncClient(
                    timeout=None, headers=_ident_headers(),
                ) as c:
                    async with c.stream(
                        "GET", f"{await IB(cam_id)}/stream/{cam_id}",
                    ) as r:
                        if r.status_code != 200:
                            return
                        async for chunk in r.aiter_bytes():
                            buf += chunk
                            while True:
                                nxt = buf.find(BOUNDARY, len(BOUNDARY))
                                if nxt < 0:
                                    break
                                part, buf = buf[:nxt], buf[nxt:]
                                if part.strip():
                                    state["part"] = part
            finally:
                state["done"] = True

        task = asyncio.create_task(_reader())
        last = None
        try:
            while not state.get("done"):
                part = state.get("part")
                if part is not None and part is not last:
                    last = part
                    yield part
                await asyncio.sleep(0.15)
        finally:
            task.cancel()

    return StreamingResponse(
        _pump(), media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/identity_live/{cam_id}")
async def identity_live(cam_id: str):
    """ONE current annotated frame, fetched on demand.

    Continuous MJPEG over a slow path (tailnet to a remote node) queues
    upstream of any client-side dropping — the writer pushes every frame
    and the backlog lives in the node's socket/tailscale buffers, so the
    view drifts behind wall clock without bound. A per-request grab has
    no queue to drift: connect, take the first complete frame (the
    stream always starts at NOW), disconnect. Latency = 1 frame + RTT,
    by construction. The page polls this at ~1.5fps instead of holding
    a stream open.
    """
    import httpx

    BOUNDARY = b"--frame"
    buf = b""
    try:
        async with httpx.AsyncClient(
            timeout=8, headers=_ident_headers(),
        ) as c:
            async with c.stream(
                "GET", f"{await IB(cam_id)}/stream/{cam_id}",
            ) as r:
                if r.status_code != 200:
                    return JSONResponse({"error": "upstream"}, status_code=502)
                async for chunk in r.aiter_bytes():
                    buf += chunk
                    second = buf.find(BOUNDARY, len(BOUNDARY))
                    if second > 0:
                        part = buf[:second]
                        head_end = part.find(b"\r\n\r\n")
                        if head_end > 0:
                            jpeg = part[head_end + 4:].rstrip(b"\r\n")
                            return Response(jpeg, media_type="image/jpeg",
                                            headers={"Cache-Control": "no-store"})
                        break
    except httpx.HTTPError:
        pass
    return JSONResponse({"error": "no frame"}, status_code=504)


@app.get("/identity_snap/{name}")
async def identity_snap(name: str):
    """Same reason as /identity_stream: journal snapshots are <img>
    sources and the endpoint behind them is now token-gated."""
    import httpx

    from fastapi.responses import Response

    try:
        async with httpx.AsyncClient(
            timeout=4.0, headers=_ident_headers(),
        ) as c:
            r = await c.get(f"{await IB()}/snap/{name}")
        if r.status_code != 200:
            return Response(status_code=r.status_code)
        return Response(content=r.content, media_type="image/jpeg")
    except Exception:                        # noqa: BLE001 — service down
        return Response(status_code=502)


# The identity services sample on this period (service.py
# _timeline_sampler). The merge buckets to it.
_TIMELINE_PERIOD_S = 10


async def _fan_out(path: str, timeout: float = 2.5) -> list[tuple[str, dict]]:
    """Ask EVERY identity service the same question, in parallel.

    One process per sensor means there is no single service that knows
    the whole room. Asking one of them and calling that "the identity
    state" is what made the second sensor invisible.

    Returns `(cam8, json)` for the services that answered. A service that
    is down is simply absent — its camera then shows as offline, which is
    the truth, rather than the whole card going blank.
    """
    import httpx

    smap = await sensors()
    if not smap:
        return []

    async def one(cam8: str, meta: dict):
        try:
            async with httpx.AsyncClient(
                timeout=timeout, headers=_ident_headers(),
            ) as c:
                r = await c.get(f"{meta['base']}{path}")
                # STATUS FIRST. A 401 has a perfectly good JSON body
                # (`{"detail": "unauthorized"}`), so parsing without
                # checking reported the service as UP with no cameras —
                # which is what an idle room looks like. Started without
                # NEMARA_IDENTITY_STATE_TOKEN in its environment, this
                # page therefore said everything was fine and showed
                # nothing, for as long as you cared to watch it.
                if r.status_code != 200:
                    log.warning("soak_dashboard: %s%s -> HTTP %d",
                                meta["base"], path, r.status_code)
                    return cam8, None
                return cam8, r.json()
        except Exception:                    # noqa: BLE001 — one service down
            return cam8, None

    results = await asyncio.gather(
        *(one(k, v) for k, v in smap.items()), return_exceptions=True,
    )
    return [r for r in results
            if isinstance(r, tuple) and isinstance(r[1], dict)]


@app.get("/identity_timeline")
async def identity_timeline():
    """Every service's 10s status samples, merged BY TIME.

    Each service only ever reports its own camera, so concatenating
    them produced one row per camera per instant — 172 rows, every one
    holding a single camera. The strip is drawn per camera as
    `row[cam] || 'empty'`, so on the bedroom's strip every row that
    belonged to the TV Room rendered as EMPTY, and vice versa. Both
    strips alternated no matter what the cameras were doing, and it
    read as the system losing the person every other tick.

    Bucketed to the sample period so two services that sampled the same
    instant land in one row. Timestamps drift a little between
    processes, so the bucket is the alignment — not equality.
    """
    buckets: dict[int, dict] = {}
    for _cam8, j in await _fan_out("/timeline?n=720"):
        for sample in (j.get("samples") or []):
            if not isinstance(sample, (list, tuple)) or len(sample) < 2:
                continue
            ts, row = sample[0], sample[1]
            if not isinstance(row, dict):
                continue
            # FLOOR, not round. The services sample on their own
            # clocks: the bedroom lands on ...399, ...409, ...419 and
            # the TV Room on ...395, ...405, ...415 — four seconds
            # apart, which is nothing, except that dividing by 10 puts
            # the TV Room exactly on .5 every time. Python rounds a tie
            # to even, so ...395 and ...405 both became bucket 540 and
            # bucket 541 got no TV Room sample at all. The strip draws
            # `row[cam] || 'empty'`, so the TV Room went empty on every
            # odd bucket — alternating for as long as you watched it,
            # while the camera was ticking cleanly every 10s. The
            # collision also threw one of the two samples away, because
            # the second `update` overwrote the first.
            #
            # Flooring puts any two samples inside the same 10s window
            # in the same bucket regardless of phase, which is what
            # "bucket to the sample period" was always supposed to mean.
            key = int(float(ts) // _TIMELINE_PERIOD_S)
            buckets.setdefault(key, {}).update(row)
    merged = [[k * _TIMELINE_PERIOD_S, row]
              for k, row in sorted(buckets.items())]
    return JSONResponse({"samples": merged})


@app.get("/identity_probes")
async def identity_probes():
    """Per-probe score history from every service, keyed by camera."""
    out: dict = {}
    for _cam8, j in await _fan_out("/probes?n=1600"):
        out.update(j.get("probes") or {})
    return JSONResponse({"probes": out})


@app.get("/identity_events")
async def identity_events():
    """The unsupervised-soak journal from every service, newest last."""
    events: list = []
    stats: list = []
    for _cam8, j in await _fan_out("/events?n=250"):
        events.extend(j.get("events") or [])
        stats.extend(j.get("confirm_stats") or [])
    events.sort(key=lambda e: e.get("t") or e.get("ts") or 0)
    lat = sorted(s["latency_s"] for s in stats if "latency_s" in s)
    return JSONResponse({
        "events": events,
        "confirm_stats": stats,
        "confirm_median_s": (round(lat[len(lat) // 2], 1) if lat else None),
        "confirm_max_s": (round(lat[-1], 1) if lat else None),
    })


@app.get("/identity")
async def identity_state():
    """The whole rig's identity state: every service, merged.

    `on` is true if ANY service answered, and `services` reports each one
    by name so a single dead sensor is visible instead of being averaged
    away. Proxied server-side to avoid CORS; the MJPEG streams embed
    directly as cross-origin <img> tags.
    """
    smap = await sensors()
    answers = await _fan_out("/state")
    cameras: dict = {}
    seats: dict = {}
    services: list = []
    answered = {cam8 for cam8, _ in answers}
    for cam8, j in answers:
        seats.update(j.get("seats") or {})
        for cid, cv in (j.get("cameras") or {}).items():
            # The camera's OWN name from the roster wins. A service that
            # does not own this camera can carry a stale or missing one,
            # and a raw uuid fragment in the UI next to a real room name
            # is the thing that made the page look broken.
            named = dict(cv)
            named["name"] = (smap.get(cid, {}).get("name")
                             or cv.get("name") or cid)
            cameras[cid] = named
    for cam8, meta in smap.items():
        services.append({
            "camera": cam8, "name": meta["name"],
            "base": meta["base"], "up": cam8 in answered,
        })
    services.sort(key=lambda s: s["name"])
    return JSONResponse({
        "on": bool(answers), "cameras": cameras, "seats": seats,
        "services": services,
        "sensors_total": len(smap), "sensors_up": len(answered),
    })


_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Memoria Soak — live</title>
<style>
  :root{--bg:#F7F8F6;--card:#FFF;--ink:#1B1F1C;--mut:#5E665F;--line:#E3E7E1;
    --acc:#0E7C5B;--ok:#0E7C5B;--warn:#A86A14;--crit:#B23E37;--chipbg:#EDF2EE;
    --logbg:#FBFCFA;--zeb:#F2F4F0;}
  @media (prefers-color-scheme:dark){:root{--bg:#111412;--card:#181C19;
    --ink:#E5E9E4;--mut:#98A29A;--line:#272C28;--acc:#43BE92;--ok:#43BE92;
    --warn:#D79A3C;--crit:#E0716A;--chipbg:#20261F;--logbg:#141815;--zeb:#1C211D;}}
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--ink);margin:0;
    font:14px/1.5 ui-sans-serif,system-ui,-apple-system,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:16px 16px 48px}
  header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 12px;margin-bottom:14px}
  h1{font-size:17px;font-weight:650;margin:0}
  h2{font-size:11px;font-weight:650;letter-spacing:.08em;text-transform:uppercase;
     color:var(--mut);margin:0 0 8px}
  .chip{font:600 11px/1 ui-monospace,Menlo,monospace;padding:4px 7px;border-radius:4px;
    background:var(--chipbg);color:var(--mut);border:1px solid var(--line)}
  .chip.on{color:var(--ok)} .chip.warn{color:var(--warn)}
  .card{background:var(--card);border:1px solid var(--line);border-radius:8px;
    padding:13px 14px;margin-bottom:14px}
  .mono{font:12px/1.6 ui-monospace,Menlo,monospace}
  .mut{color:var(--mut)}.ok{color:var(--ok)}.warn{color:var(--warn)}.crit{color:var(--crit)}
  .num{font-variant-numeric:tabular-nums}
  .cams{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
  table{width:100%;border-collapse:collapse;font-size:12.5px}
  th{font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut);
     text-align:left;font-weight:600;padding:3px 8px 5px 0;border-bottom:1px solid var(--line)}
  td{padding:4px 8px 4px 0;border-bottom:1px solid var(--line);vertical-align:top}
  tr:last-child td{border-bottom:none}
  .tblwrap{overflow-x:auto}
  .logwin{background:var(--logbg);border:1px solid var(--line);border-radius:6px;
    max-height:440px;overflow-y:auto;font:11.5px/1.65 ui-monospace,Menlo,monospace}
  .lr{display:flex;gap:9px;padding:2px 9px}
  .logwin .lr:nth-child(even){background:var(--zeb)}
  .logwin .lr.fail{border-left:3px solid var(--crit)}
  .logwin .lr.fail .b{color:var(--crit)}
  .logwin .lr.wake{border-left:3px solid var(--ok)}
  .lr .t{color:var(--mut);flex:0 0 58px}.lr .c{flex:0 0 76px;color:var(--acc);font-weight:600}
  .lr .b{overflow-wrap:anywhere;min-width:0}
  #events>div{overflow-wrap:anywhere;padding:2px 0}
  .skip{color:var(--mut)} .fired{color:var(--crit);font-weight:700}
  .fnote{color:var(--warn)}
  .lane{padding:12px 0;border-top:1px solid var(--line)}
  .lane:first-child{border-top:none;padding-top:2px}
  .lanehead{display:flex;align-items:center;gap:10px;margin-bottom:7px}
  .lanehead b{font-size:13.5px}
  .tlrow{display:flex;align-items:center;gap:8px;margin:0 0 9px}
  .tlt{font-size:10.5px;flex:0 0 auto}
  .tl{display:flex;height:14px;border-radius:3px;overflow:hidden;flex:1;min-width:0;
    border:1px solid var(--line);background:var(--logbg)}
  .tl i{flex:1;min-width:1px}
  .tl i.ok{background:var(--ok)} .tl i.warn{background:var(--warn)}
  .tl i.emp{background:var(--line)} .tl i.skp{background:var(--chipbg)}
  .tl i.oth{background:var(--crit)}
  .tl i.deg{opacity:.45}
  .thumbs{display:flex;gap:8px;overflow-x:auto;padding-bottom:4px}
  /* Fixed thumb height: lazy-loading images used to reflow the page
     on every poll, which read as "the page snapped". */
  .th img{height:104px;display:block;border-radius:4px}
  .th{flex:0 0 190px}
  .th img{width:190px;height:110px;object-fit:cover;display:block;
    border-radius:6px;border:1px solid var(--line)}
  .thcap{font-size:10.5px;white-space:nowrap;overflow:hidden;
    text-overflow:ellipsis;width:190px;margin-bottom:3px}
  .gapchip{flex:0 0 auto;align-self:center;font:600 10px/1 ui-monospace,Menlo,monospace;
    color:var(--mut);border:1px dashed var(--line);border-radius:10px;padding:4px 7px}
</style></head><body><div class="wrap">
<header>
  <h1>Memoria soak — live</h1>
  <span class="chip on" id="live">connecting…</span>
  <span class="chip" id="flags"></span>
  <span class="chip" id="since"></span>
  <span class="chip num" id="pace"></span>
</header>
<div class="card"><h2>Cameras</h2><div class="cams" id="cams"></div></div>
<!-- Legacy "Identity — per camera" card removed (Aryaman 2026-07-24):
     its data source was the stopped batch pipeline. The lane timeline
     lives on, fed by the NEW identity service, above the live views.
     Hidden stub keeps the legacy poll JS harmless until P3 cleanup. -->
<div id="lanes" style="display:none"></div>
<div class="card"><h2>Identity services <span id="idcount" class="mut" style="font-weight:400;font-size:12px"></span> <span id="idfresh" style="font-size:12px;font-weight:400"></span></h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">One process per SENSOR, each on its own port. Every one is polled and merged here — a sensor that is down shows as down rather than being left out.</p>
<div id="idservices" style="display:flex;gap:6px;flex-wrap:wrap;margin:0 0 8px"></div>
<div id="idsvc" class="mut">connecting…</div>
<div id="idtl" style="margin-top:8px"></div>
<div id="idprobes" style="margin-top:10px;max-height:340px;overflow-y:auto"></div>
<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start;margin-top:8px">
<div id="idstreams" style="display:flex;gap:8px;flex-wrap:wrap"></div>
 <div><div class="mono mut" id="croppersonmeta" style="font-size:11px">gemini input · person crop: —</div>
  <video id="cropperson" width="300" muted autoplay loop playsinline></video></div>
</div></div>
<div class="card"><h2>Identity events — unsupervised journal <span class="mut" style="font-weight:400;font-size:12px">(every birth/death/lock, with the picture)</span></h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">The overnight record: one row per identity event, newest first, snapshot inline. <b>lock/moved</b> rows carry confirm-latency (track created → patient confirmed) — the number that must stay under a minute.</p>
<div id="idconfstats" class="mono" style="font-size:12px;margin:0 0 8px"></div>
<div id="idevents" class="logwin" style="max-height:520px;overflow-y:auto"></div></div>
<div class="card"><h2>Tracker eval — live trace <span id="tarm" class="mut" style="font-weight:400;font-size:12px"></span></h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">SAM3 vs SAM3+EdgeTAM rig. Metrics that are computable without ground truth (per-stage latency, body-score distribution, event counts, contamination). The panel is the raw per-tick timeline: <span class="ok">body/BOUND</span> = patient signal · <span class="warn">amber</span> = unmatched/kept · <span class="crit">UNBOUND</span> = lost. Label tids yourself for acquire/FP.</p>
<div id="traceoff" class="mut">tracing off — set <span class="mono">NEMARA_REID_TRACE_ENABLED=true</span> and restart the backend (armed automatically by both identity providers)</div>
<div id="tracebody" style="display:none">
<div id="tresets"></div>
<div id="tmetrics" class="mono" style="font-size:12px;margin:6px 0 10px"></div>
<div class="logwin" id="tpanel" style="max-height:440px"></div>
</div></div>
<div class="card"><h2>Events</h2><div id="events" class="mono mut">none yet</div></div>
<div class="card"><h2>API calls per hour</h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">Each bar = model calls sent that hour. The dollar figure is that hour's estimated cost. Hover any bar for tokens and provider breakdown.</p>
<div id="hourtotals" class="mono" style="margin:0 0 8px"></div>
<div id="hourly" class="mono mut">no calls yet — waiting for the first analyzed tick</div></div>
<div class="card"><h2>Tick log — newest first</h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">How to read a line: every detector in the loadout is asked on every analyzed tick, grouped into the calls shown — <span class="mono">key·Ndet</span> = one Gemini call carrying N detectors. Silence = every detector answered no; <span class="mono">⚑ name</span> = that detector answered YES. <span class="mono">⛔closed</span> = that call was pre-gated and never sent ($0) — today only the convulsion bucket has a pre-gate (<span class="mono">rapid-gate-motion</span> is its motion evidence), which is why gate chatter looks convulsion-specific. Absence/scheduled detectors don't ride calls; they evaluate from coverage at their deadline. <span class="mono">ingest-fps=A→kept B</span> = LiveKit delivered A fps to the backend (stream health; want ≥8) → buffer kept B for analysis (capped at the 4fps target by design; ~3.4 is normal, NOT a problem); <span class="mono">worst-frame-gap</span> = longest frame silence in the window. <span class="mono">track-motion</span> = largest per-track motion fraction this tick; <span class="mono">env-motion</span> = scene motion outside any track.</p>
<div class="logwin" id="log"></div></div>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function tickLine(t){
  const fps=t.rfps!=null?` ingest-fps=${t.rfps}→kept ${t.kfps??"?"} · worst-frame-gap=${t.gap??"?"}ms`:"";
  if(t.skip==="tick_failed")
    return `<span class="crit">⛔ TICK FAILED · ${esc(t.err??"?")}: ${esc(t.emsg??"")}</span>${fps}`;
  if(t.skip) return `skipped · ${esc(t.skip)}${fps}`;
  let b=`${t.woke?'<span class="ok">⚡ WOKE</span> ':""}${t.lag!=null?`<span class="${t.lag>20?"crit":t.lag>12?"warn":"ok"}">lag=${t.lag}s</span> `:""}motion=${esc(t.motion??"—")} people=${t.ppl?"YES":"no"} (via ${esc(t.psrc??"—")}) tracks=${t.tracks}`;
  if(t.amount!=null) b+=` track-motion=${t.amount}`;
  if(t.env!=null) b+=` env-motion=${t.env}`;
  if(t.calls.length) b+=` calls=[${t.calls.map(c=>
    esc(c.k)+"·"+c.n+"det"+(c.g?" ⛔"+esc(c.g):"")).join(" | ")}]`;
  if(t.gaf!=null) b+=` rapid-gate-motion=${t.gaf}`;
  b+=fps;
  if(t.fired.length) b+=` ⚑ ${t.fired.map(esc).join(", ")}`;
  return b;
}
// Scroll preservation across the 10s rebuild (Aryaman 2026-07-19:
// "every 10 seconds it takes me back to the latest"). Two flavors:
//  - #log / #events grow at the TOP (newest first): a reader scrolled
//    into history keeps their place by offsetting the height delta
//    (the chat-history trick). At the top (≤8px) you stay following
//    the latest — that's the live mode.
//  - #hourly's inner strip grows at the RIGHT: plain scrollLeft
//    restore keeps older bars in place.
function snapScroll(){
  const s={};
  for(const id of ["log","events"]){
    const el=$(id);if(!el)continue;
    s[id]={top:el.scrollTop,h:el.scrollHeight};
  }
  const hi=$("hourly")&&$("hourly").firstElementChild;
  if(hi)s.hourly={left:hi.scrollLeft};
  return s;
}
function restoreScroll(s){
  for(const id of ["log","events"]){
    const el=$(id);const o=s[id];if(!el||!o)continue;
    el.scrollTop=o.top>8?o.top+(el.scrollHeight-o.h):o.top;
  }
  const hi=$("hourly")&&$("hourly").firstElementChild;
  if(hi&&s.hourly)hi.scrollLeft=s.hourly.left;
}
async function refresh(){
  try{
    const d=await (await fetch("/data")).json();
    const scrollSnap=snapScroll();
    $("live").textContent=`LIVE · ${d.now} ${d.tz}`;
    $("flags").textContent=`sched=${d.flags.scheduler?"on":"OFF"} · decision=${d.flags.decision?"on":"OFF"} · idframes=${d.flags.id_frames?"on":"OFF"}`;
    $("since").textContent=`since ${d.since}`;
    // Spend, then BOTH rates, explicitly labelled. The header used to
    // read "$X so far · $Y/day pace" where X covered ~the last hour and
    // Y divided it by the whole window — understating by ~17x. X is now
    // whole-window; the live burn is the last-full-hour figure.
    const unp=d.totals.unpriced_calls||0;
    $("pace").textContent=
      `${unp?"≥":""}$${d.totals.cost.toFixed(2)} over ${d.totals.hours}h`
      + (d.totals.day_pace!=null
          ? ` · $${d.totals.day_pace}/day avg incl. idle`
          : ` · /day pace pending 1h`)
      + (d.totals.day_24h!=null? ` · $${d.totals.day_24h.toFixed(2)} last 24h actual`:"")
      + (d.totals.hour_rate!=null
          ? ` · $${d.totals.hour_rate.toFixed(2)} spent in ${d.totals.hour_label} (one hour, not a daily rate)`
          : ` · /hr pending first full hour`)
      + (unp? ` · ${unp} calls UNPRICED (no usage recorded)`:"");
    $("pace").style.color = unp? "var(--warn)" : "";
    // Tracking-consistency rollup over the visible tick window: of the
    // ticks someone was believed present, how many had a live track,
    // and how often a track dropped to zero while the room stayed
    // occupied (a blip = one loss + one id-change capture).
    const byCam={};
    d.ticks.forEach(t=>{(byCam[t.cam]=byCam[t.cam]||[]).unshift(t)});
    const cons={};
    for(const [cam,arr] of Object.entries(byCam)){
      let occ=0,trk=0,loss=0,prev=0;
      for(const t of arr){
        if(t.ppl){occ++; if(t.tracks>0)trk++; if(prev>0&&t.tracks===0)loss++;}
        prev=t.tracks||0;
      }
      cons[cam]={occ,trk,loss};
    }
    const roomOf=c=>(d.names&&d.names[c])?d.names[c]:c;
    $("cams").innerHTML=d.cams.map(c=>{
      const l=c.last||{};
      const k=cons[c.cam];
      const idline=k&&k.occ?`<div class="mono ${k.loss?"warn":"mut"}">id-track ${Math.round(100*k.trk/k.occ)}% of ${k.occ} occupied ticks · losses ${k.loss} (log window)</div>`:"";
      return `<div><div class="mono"><b>${esc(roomOf(c.cam))}</b> <span class="mut">${esc(c.cam)}</span> · ${c.ticks} ticks · wakes ${c.wakes}${c.lastwake?` <span class="ok">(last ⚡ ${esc(c.lastwake)})</span>`:""} · degraded ${c.degraded} · errors ${c.errors}</div>
      <div class="mono mut">skips: ${esc(JSON.stringify(c.skips))}</div>${idline}
      <div class="mono">last: ${tickLine(l)}</div></div>`;
    }).join("") || '<span class="mut">no ticks yet</span>';
    // Identity lanes: per-camera state bar (all fetched ticks, oldest→
    // newest) + fixed-size capture strip (newest left, gap chips).
    const byCamId={};
    (d.idt||[]).forEach(t=>{(byCamId[t.cam]=byCamId[t.cam]||[]).unshift(t)});
    const allCams=[...new Set([...Object.keys(byCamId),...d.frames.map(f=>f.cam)])]
      .sort((a,b)=>roomOf(a).localeCompare(roomOf(b)));
    // Scroll anchoring (Aryaman 2026-07-19): the 10s poll rebuilds the
    // strips, which used to snap a browsing user back to the newest
    // image. Remember which image the viewer was on (per camera) and
    // re-align the strip to that same image after the rebuild.
    const anchors={};
    document.querySelectorAll(".thumbs[data-cam]").forEach(el=>{
      if(el.scrollLeft>8){
        const vis=[...el.querySelectorAll("img[data-name]")]
          .find(i=>i.offsetLeft+i.offsetWidth>el.scrollLeft);
        if(vis)anchors[el.dataset.cam]={name:vis.dataset.name,delta:vis.offsetLeft-el.scrollLeft};
      }
    });
    $("lanes").innerHTML=allCams.length?allCams.map(cam=>{
      const arr=byCamId[cam]||[];
      const cells=arr.map(t=>{
        const cls=(t.pat?"ok":t.oth?"oth":(t.trk>0?"warn":(t.skip?"skp":"emp")))+(t.deg?" deg":"");
        const lbl=(t.pat?"patient confirmed":t.oth?"verified NOT patient (call skipped, $0)":(t.trk>0?"person, not identified":(t.skip?"skipped: "+t.skip:"empty")))+(t.deg?" · degraded input (blur/180p)":"");
        return `<i class="${cls}" title="${esc(t.t)} — ${esc(lbl)}"></i>`;
      }).join("");
      const last=arr[arr.length-1]||{};
      const st=last.pat?["PATIENT","on"]:(last.trk>0?["UNIDENTIFIED","warn"]:["EMPTY",""]);
      const fr=d.frames.filter(f=>f.cam===cam);
      let strip="",prev=null;
      for(const f of fr){
        if(prev!=null&&f.epoch&&(prev-f.epoch)>180)
          strip+=`<div class="gapchip">${Math.round((prev-f.epoch)/60)}m</div>`;
        prev=f.epoch;
        strip+=`<div class="th"><div class="mono mut thcap">${esc(f.hms)}${f.note?` · <span class="fnote">${esc(f.note)}</span>`:""}</div>
          <img loading="lazy" data-name="${esc(f.name)}" src="/frames/${encodeURIComponent(cam)}/${encodeURIComponent(f.name)}"></div>`;
      }
      return `<div class="lane">
        <div class="lanehead"><b>${esc(roomOf(cam))}</b><span class="mono mut">${esc(cam)}</span><span class="chip ${st[1]}">${st[0]}</span></div>
        ${arr.length?`<div class="tlrow"><span class="mono mut tlt">${esc((arr[0]||{}).t||"")}</span><div class="tl">${cells}</div><span class="mono mut tlt">${esc(last.t||"")}</span></div>`:""}
        <div class="thumbs" data-cam="${esc(cam)}">${strip||'<span class="mut">no captures yet</span>'}</div>
      </div>`;
    }).join(""):'<span class="mut">no data yet</span>';
    document.querySelectorAll(".thumbs[data-cam]").forEach(el=>{
      const a=anchors[el.dataset.cam];if(!a)return;
      const img=[...el.querySelectorAll("img[data-name]")].find(i=>i.dataset.name===a.name);
      if(img)el.scrollLeft=Math.max(0,img.offsetLeft-a.delta);
    });
    $("events").innerHTML=d.events.length? d.events.map(e=>
      `<div>${esc(e.t)} &nbsp;<b>${esc(e.d)}</b> &nbsp;${esc(e.title)}</div>`).join(""):"none yet";
    if(d.hourly&&d.hourly.length){
      const cost=x=>x.cost||0;
      const mx=Math.max(...d.hourly.map(x=>x.calls),1);
      const totCalls=d.hourly.reduce((a,x)=>a+x.calls,0);
      const totCost=d.hourly.reduce((a,x)=>a+cost(x),0);
      // This used to divide by hourly.length (buckets WITH data), giving
      // a third pace that disagreed with the header's two. Buckets are
      // now labelled for what they are: spend per hour that actually ran.
      const perActive=totCost/Math.max(1,d.hourly.length);
      $("hourtotals").textContent=
        `Since start: ${totCalls} calls · $${totCost.toFixed(2)} estimated · `+
        `$${perActive.toFixed(2)}/hr averaged over ${d.hourly.length} active hour(s)`;
      $("hourly").innerHTML=`<div style="display:flex;align-items:flex-end;gap:8px;overflow-x:auto;padding:4px 0">`+
        d.hourly.map(x=>{
          const tip=`${esc(x.h)} — ${x.calls} calls from ${x.an} analyzed ticks\n`+
            `${(x.tin/1000).toFixed(1)}k tokens in, ${(x.tout/1000).toFixed(1)}k out\n`+
            `Gemini ${x.g} · DashScope ${x.d} · OpenRouter ${x.o}`+
            (x.unp? `\n${x.unp} call(s) unpriced — no usage recorded, billed as $0 here`:"");
          return `<div style="text-align:center;min-width:56px" title="${tip}">
          <div class="mono" style="font-size:11px">${x.calls}</div>
          <div style="height:${Math.max(3,Math.round(90*x.calls/mx))}px;background:var(--ok);border-radius:3px 3px 0 0;opacity:.85"></div>
          <div class="mono" style="font-size:11px;margin-top:3px">${esc(x.h.slice(6))}</div>
          <div class="mono mut" style="font-size:10.5px">$${cost(x).toFixed(2)}</div>
        </div>`}).join("")+`</div>`;
    }
    $("log").innerHTML=d.ticks.map(t=>
      `<div class="lr ${t.skip==="tick_failed"?"fail":t.skip?"skip":""} ${t.woke?"wake":""} ${t.fired.length?"fired":""}">
        <span class="t">${esc(t.t)}</span><span class="c" title="${esc(t.cam)}">${esc(roomOf(t.cam).slice(0,9))}</span>
        <span class="b">${tickLine(t)}</span></div>`).join("");
    restoreScroll(scrollSnap);
  }catch(e){ $("live").textContent="reconnecting…"; }
}
refresh(); setInterval(refresh, 10000);

// ── Tracker-eval trace panel (independent 5s poll; isolated from /data) ──
function tfmt(v){return v==null?'–':(Math.round(v*10)/10);}
function tf3(v){return v==null?'–':Number(v).toFixed(3);}
async function pollTrace(){
  try{
    const t=await (await fetch('/trace')).json();
    const off=document.getElementById('traceoff'),body=document.getElementById('tracebody');
    if(!t.on){off.style.display='';body.style.display='none';return;}
    off.style.display='none';body.style.display='';
    document.getElementById('tarm').textContent='· identity arm: '+(t.arm||'?');
    const m=t.metrics;
    document.getElementById('tmetrics').innerHTML=
      '<b>latency (measured)</b> capture→processed p50 '+tfmt(m.e2e_p50)+'ms · max '+tfmt(m.e2e_max)+'ms (n='+m.e2e_n+') · pod tracking p50 '+tfmt(m.track_p50)+'ms · face probe p50 '+tfmt(m.face_p50)+'ms<br>'+
      '<b>body scores</b> nonzero '+m.body_nz+'/'+m.body_n+' · min '+tf3(m.body_min)+' · median '+tf3(m.body_med)+' · max '+tf3(m.body_max)+'<br>'+
      '<b>events</b> '+(Object.entries(m.events).map(function(e){return e[0]+'='+e[1];}).join(' · ')||'none');
    document.getElementById('tresets').innerHTML=t.resets>0?
      '<div class="lr fail" style="padding:6px 8px"><b>⚠ '+t.resets+' worker RESET(s)</b> — any run straddling one is CONTAMINATED (ledger wiped; infra, not the tracker)</div>':'';
    var h='';
    (t.ticks||[]).forEach(function(tk){
      var when=new Date((tk.ts||0)*1000).toLocaleTimeString();
      h+='<div class="lr"><b>tick '+(tk.tick==null?'?':tk.tick)+'</b> <span class="mut">'+when+' · '+esc(tk.cam_name||String(tk.cam).slice(0,8))+(tk.arm?' · '+tk.arm:'')+'</span>';
      if(tk.lat)h+=' <span class="mut">capture→processed '+tfmt(tk.lat.e2e_ms)+'ms · pod '+tfmt(tk.lat.track_ms)+'ms · face probe '+tfmt(tk.lat.face_ms)+'ms'+(tk.lat.sidecar_ms!=null?' · backend fold '+tfmt(tk.lat.sidecar_ms)+'ms':'')+' · '+tk.lat.frames+(tk.arm==='sidecar'?' tracks':' frames')+'</span>';
      h+='</div>';
      (tk.body||[]).forEach(function(b){var cls=String(b.verdict).indexOf('PATIENT')===0?'ok':(b.patient>0?'warn':'mut');
        h+='<div class="lr" style="padding-left:18px"><span class="'+cls+'">body '+tf3(b.patient)+'</span> <span class="mut">other '+tf3(b.other)+' → '+b.verdict+'</span></div>';});
      (tk.idev||[]).forEach(function(e){var cls=(e.event==='BOUND'||e.event==='SEAT_STOLEN')?'ok':(String(e.event).indexOf('UNBOUND')===0?'crit':'warn');
        h+='<div class="lr" style="padding-left:18px">⚡ <span class="'+cls+'">'+e.event+'</span> T'+e.tid+' <span class="mut">'+e.cause+(e.score!=null?' score='+e.score:'')+(e.challenger?' vs '+e.challenger:'')+'</span></div>';});
      (tk.harvest||[]).forEach(function(hv){
        h+='<div class="lr" style="padding-left:18px"><span class="mut">↳ harvest '+hv.pool+' score='+hv.score+'</span></div>';});
    });
    document.getElementById('tpanel').innerHTML=h||'<span class="mut">no ticks yet</span>';
  }catch(e){}
}
pollTrace(); setInterval(pollTrace, 5000);
let idStreams="";
async function pollIdentity(){
  try{
    const s=await (await fetch('/identity')).json();
    window._camnames=Object.assign(window._camnames||{},Object.fromEntries(Object.entries(s.cameras||{}).map(([k,v])=>[k.slice(0,8),v.name]).filter(([k,n])=>n)));
    // Per-sensor up/down, ALWAYS rendered — including when everything is
    // down. The old card printed one "not running" line for the single
    // service it knew about, so a rig with one healthy sensor and one
    // dead one looked entirely fine or entirely broken, never mixed.
    const svcs=s.services||[];
    $("idcount").textContent=svcs.length?`· ${s.sensors_up}/${s.sensors_total} up`:'';
    $("idservices").innerHTML=svcs.map(x=>
      `<span class="chip ${x.up?'on':'warn'}">${x.name} · ${x.up?'up':'DOWN'}</span>`
    ).join('') || '<span class="chip warn">no sensor has an identity slot</span>';
    if(!s.on){$("idsvc").innerHTML='<span class="warn">no identity service answered — check the node</span>';return;}
    const nm=c=>{const k=(c||'').slice(0,8);const m=window._camnames||{};return m[k]||m[c]||k;};
    let h='';
    // state-v2: one seat per LOCATED resident, keyed by patient uuid.
    // A camera can watch more than one person, so this is a list —
    // rendering only the first would make a working two-resident lock
    // look like a half-broken one on the exact screen used to verify it.
    const seats=Object.entries(s.seats||{});
    if(seats.length){
      for(const [pid,st] of seats){
        h+=`<div><span class="ok" style="font-weight:700">${pid.slice(0,8)} = T${st.track}</span> on ${nm(st.camera)} · score ${st.score}@${st.anchor}${st.face_confirmed?' <span class="ok">*face-confirmed</span>':''} · held ${Math.max(0,Math.round((Date.now()/1000)-st.since))}s</div>`;
      }
    } else {
      h+='<div><span class="warn">nobody located</span> — searching (walk near a camera to bootstrap)</div>';
    }
    for(const [c,v] of Object.entries(s.cameras||{})){
      const t=Object.keys(v.tracks||{});
      h+=`<div class="mono" style="font-size:12px;margin-top:3px">${nm(c)} · tracks [${t.join(',')||'—'}]${v.solo?' · solo':''} · lag ${v.lag_s}s · pod ${v.pod_ms}ms</div>`;
    }
    $("idsvc").innerHTML=h; window._idLastOk=Date.now();
    const cams=Object.keys(s.cameras||{});
    // Keyed on WHICH cameras, not how many. Swapping one sensor for
    // another leaves the count equal and used to leave the old tiles up.
    const camKey=cams.slice().sort().join(',');
    if(cams.length && idStreams!==camKey){
      // Snapshot polling, NOT a held-open MJPEG stream: over the tailnet
      // to a remote node the stream queues upstream and drifts behind
      // wall clock without bound; a per-request frame cannot lag more
      // than 1 frame + RTT. Two buffers per cam, swapped on load, so
      // the image never flickers white between polls.
      $("idstreams").innerHTML=cams.map(c=>
        `<div style="width:calc(50% - 4px);min-width:280px">`+
        `<div class="mono" style="font-size:11px;display:flex;justify-content:space-between">`+
        `<span>${nm(c)}</span><span id="age_${c}" class="mut"></span></div>`+
        `<img id="live_${c}" src="/identity_live/${c}" style="width:100%;border-radius:6px"></div>`
      ).join('');
      window._frameOk=window._frameOk||{};
      cams.forEach(c=>{
        const el=document.getElementById("live_"+c);
        if(!el || el._poll) return;
        // SELF-CLOCKING, not a fixed timer. Each frame is ~350KB of
        // full-resolution JPEG relayed from a node in another
        // continent and measured at 2.4-4.5s per fetch, against a
        // 700ms interval — so requests piled up faster than they
        // completed, the browser's per-host connection limit
        // saturated, and the picture simply stopped moving. It read as
        // "the stream is down" while the pipeline was perfectly
        // healthy.
        //
        // Asking for the next frame only once the last one has ARRIVED
        // cannot pile up, whatever the round trip costs, and it
        // degrades to "slower" instead of "stuck".
        const tick=()=>{
          const pre=new Image();
          // BOTH handlers. `onload` alone meant a failing fetch never
          // swapped the image and never said so, so the tile held its
          // last good frame indefinitely — the node was restarted for
          // 15 minutes on 2026-08-24 and this card showed a live-looking
          // picture throughout. A frozen frame must announce itself.
          const again=()=>{ el._poll=setTimeout(tick, 250); };
          pre.onload=()=>{
            el.src=pre.src; window._frameOk[c]=Date.now(); again();
          };
          // A failed fetch must re-arm too, or one blip ends the feed
          // for good — which is the same silent stop, arrived at from
          // the other direction.
          pre.onerror=again;
          pre.src=`/identity_live/${c}?t=${Date.now()}`;
        };
        el._poll=true;
        tick();
      });
      idStreams=camKey;
    }
    // Per-tile frame age, so "frozen" is a number on the screen.
    for(const c of cams){
      const lbl=document.getElementById("age_"+c); if(!lbl) continue;
      const last=(window._frameOk||{})[c];
      if(!last){ lbl.innerHTML='<span class="warn">no frame yet</span>'; continue; }
      const age=Math.round((Date.now()-last)/1000);
      lbl.innerHTML = age>10
        ? `<span class="crit" style="font-weight:700">FROZEN ${age}s</span>`
        : `<span class="mut">${age}s</span>`;
    }
  }catch(e){}
}
window._idLastOk=0;
pollIdentity(); setInterval(pollIdentity, 3000);
setInterval(()=>{
  const age=(Date.now()-window._idLastOk)/1000;
  const el=$("idfresh"); if(!el) return;
  el.innerHTML = window._idLastOk===0 ? '' :
    (age>60 ? `<span style="color:#f87171;font-weight:700">STALE — last update ${Math.round(age)}s ago (reload?)</span>`
            : `<span class="mut">updated ${Math.round(age)}s ago</span>`);
}, 5000);
const EVK={track_birth:['#8ab4f8','birth'],track_death:['#9aa0a6','death'],
  seat_locked:['#34d399','PATIENT LOCKED'],seat_moved:['#fbbf24','PATIENT MOVED'],
  seat_died:['#f87171','patient unset'],seat_healed:['#c084fc','SWAP HEALED']};
async function pollIdEvents(){
  try{
    const j=await (await fetch('/identity_events')).json();
    if(j.confirm_median_s!=null){
      $("idconfstats").innerHTML=`confirm latency: median <b>${j.confirm_median_s}s</b> · worst <b>${j.confirm_max_s}s</b> · n=${(j.confirm_stats||[]).length} (last 50: ${(j.confirm_stats||[]).map(s=>s.latency_s+'s@'+s.anchor).join(', ')})`;
    } else { $("idconfstats").innerHTML='<span class="mut">no patient confirmations yet this run</span>'; }
    const evs=(j.events||[]).slice().reverse();
    $("idevents").innerHTML=evs.map(e=>{
      const [col,label]=EVK[e.kind]||['#ddd',e.kind];
      const img=e.img?`<img src="/identity_snap/${e.img}" loading="lazy" style="height:74px;border-radius:4px;vertical-align:middle;margin-right:8px">`:'';
      const lat=e.confirm_latency_s!=null?` · confirm ${e.confirm_latency_s}s`:'';
      const sc=e.score!=null?` · ${e.score}@${e.anchor||''}`:'';
      return `<div style="display:flex;align-items:center;gap:6px;padding:3px 0;border-bottom:1px solid #222">${img}<div class="mono" style="font-size:12px"><span style="color:${col};font-weight:700">${label}</span> T${e.tid??'?'} · ${e.cam_name||((window._camnames||{})[(e.cam||'').slice(0,8)])||(e.cam||'').slice(0,8)}${sc}${lat}<br><span class="mut">${e.iso||''}</span></div></div>`;
    }).join('')||'<span class="mut">no events yet</span>';
  }catch(e){}
}
pollIdEvents(); setInterval(pollIdEvents, 5000);
async function pollIdTimeline(){
  try{
    const j=await (await fetch('/identity_timeline')).json();
    let pj={probes:{}};
    try{pj=await (await fetch('/identity_probes')).json();}catch(e){}
    const probesByCam=pj.probes||{};
    let samples=j.samples||[];
    // No early return. The timeline ring lives in the identity process
    // and is empty for the first ~10s after every restart — returning
    // here blanked the score table too, which has its own source and
    // was fine.
    const hasTl=samples.length>0;
    // NATIVE 10s samples — no bucketing. The old 30s max-priority
    // buckets (patient > people > empty) swallowed real transitions:
    // live 2026-07-26 02:19, a seat death + 35s of empty/people + a
    // face re-lock rendered as solid green because each bucket also
    // contained one patient sample. Every sample is a cell now; a
    // lost lock is VISIBLE as the empty/yellow sliver it really was.
    const cams=[...new Set(samples.flatMap(s=>Object.keys(s[1])))].sort();const tnm=c=>(j.names||{})[c.slice(0,8)]||((window._camnames||{})[c.slice(0,8)])||c.slice(0,8);
    const t0=hasTl?new Date(samples[0][0]*1000):null,t1=hasTl?new Date(samples[samples.length-1][0]*1000):null;
    const hm=d=>`${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
    // Probe → tick correlation: tick #N is the sample's index in this
    // window (labels shared by cell tooltips AND the score table).
    // A probe belongs to the first sample at-or-after its timestamp.
    const hms=d=>`${hm(d)}:${String(d.getSeconds()).padStart(2,'0')}`;
    // EVERY tuple carries the patient it scored AGAINST (loop.py ~1294).
    // A camera can watch several residents, so a score is meaningless
    // without saying whose gallery produced it — that is the whole point
    // of the shared-sensor attribution matrix.
    //
    // This read faces as [tid,cos,px], body as [tid,cos] and cands as
    // [tid,score,anchor,prov] — every one short by the patient added on
    // 2026-08-20. So `cos` was the patient-id STRING and .toFixed threw,
    // inside the one try/catch wrapping this whole function. That is why
    // the timeline and the score table vanished TOGETHER: neither was
    // broken, the render aborted before reaching either.
    const fmtProbe=p=>{
      const f=(p.faces||[]).map(([tid,pid,c,px])=>`t${tid}→${pid} ${(+c).toFixed(2)}@${px}px`).join(' ')||'—';
      const b=(p.body||[]).map(([tid,pid,c])=>`t${tid}→${pid} ${(+c).toFixed(2)}`).join(' ')||'—';
      const cd=(p.cands||[]).map(([tid,pid,s,a,pr])=>`t${tid}→${pid} ${(+s).toFixed(2)} ${a}${pr?'?':''}`).join(' ')||'—';
      return {f,b,cd};
    };
    const tickOf=ts=>{
      for(let i=0;i<samples.length;i++) if(samples[i][0]>=ts) return i;
      return samples.length-1;
    };
    const probeRows=[];
    const byCamTick={};
    for(const cam of Object.keys(probesByCam)){
      for(const p of probesByCam[cam]){
        // NOT dropped when older than the timeline window. That ring
        // lives in the identity process and every restart empties it,
        // so after one the window is minutes long and this quietly
        // emptied the whole score table — which reads as "identity is
        // not probing" when it is probing fine.
        const tick=tickOf(p.ts);
        (byCamTick[cam+'|'+tick]=byCamTick[cam+'|'+tick]||[]).push(p);
        probeRows.push({cam,tick,p});
      }
    }
    $("idtl").innerHTML=!hasTl?'<div class="mut" style="font-size:11px">no timeline samples yet — the ring lives in the identity process and refills ~10s after a restart</div>':cams.map(cam=>{
      const cells=samples.map(([ts,row],i)=>{
        const st=row[cam]||'empty';
        const cls=st==='patient'?'ok':(st==='people'?'warn':'emp');
        const ps=byCamTick[cam.slice(0,8)+'|'+i]||byCamTick[cam+'|'+i]||[];
        const plines=ps.map(p=>{const x=fmtProbe(p);
          return `${hms(new Date(p.ts*1000))} face ${x.f} | body ${x.b} | cand ${x.cd}`;}).join('&#10;');
        return `<i class="${cls}" title="tick #${i} ${hm(new Date(ts*1000))} — ${st}${plines?'&#10;'+plines:''}"></i>`;
      }).join('');
      return `<div style="display:flex;align-items:center;gap:8px;margin:2px 0">
        <span class="mono mut" style="min-width:90px">${tnm(cam)}</span>
        <span class="mono mut" style="font-size:10px">${hm(t0)}</span>
        <div class="tl" style="flex:1">${cells}</div>
        <span class="mono mut" style="font-size:10px">${hm(t1)}</span></div>`;
    }).join('')+`<div class="mut" style="font-size:10.5px;margin-top:2px">last 2h · one cell ≈ 10s = one tick # · <span class="ok">green</span> patient · <span class="warn">amber</span> people, unidentified · light = empty · hover a cell for its probes' scores</div>`;
    // Full probe-score table, newest first — every probe, grouped by
    // tick #, so score trajectories are analyzable without hovering.
    probeRows.sort((a,b)=>b.p.ts-a.p.ts);
    $("idprobes").innerHTML=probeRows.length?`<table style="width:100%">
      <tr><th>time</th><th>tick</th><th>room</th><th>trk</th><th>face raw cos@px</th><th>body raw cos</th><th>fused cand (score anchor, ?=provisional)</th><th>seat</th></tr>`+
      probeRows.slice(0,250).map(({cam,tick,p})=>{
        const x=fmtProbe(p);
        return `<tr><td class="mono">${hms(new Date(p.ts*1000))}</td><td class="mono">#${tick}</td><td>${tnm(cam)}</td><td class="mono">${p.tracks}</td><td class="mono">${x.f}</td><td class="mono">${x.b}</td><td class="mono">${x.cd}</td><td>${p.locked?'locked':'searching'}</td></tr>`;
      }).join('')+`</table>`
      :'<div class="mut" style="font-size:11px">no probe scores — identity is running but returning none. It probes only while a track is live, so an empty room is expected; if somebody IS in frame, check :8012/probes directly.</div>';
  }catch(e){}
}
pollIdTimeline(); setInterval(pollIdTimeline, 10000);
async function pollCrops(){
  // Person crop FIRST, full scene as the fallback. A person crop only
  // exists when a person-bucket detector actually ran, which needs a
  // located body to crop to — so with a scene-only loadout, or nobody
  // seated, none is ever built. Showing an empty box in that case reads
  // as "the dashboard is broken" when the truthful answer is "this tick
  // had no person crop". The label always says which one you are
  // looking at, because confusing the two would misread the evidence.
  const m=document.getElementById("croppersonmeta");
  const v=document.getElementById("cropperson");
  for(const k of ["person","full"]){
    try{
      const r=await fetch("/crops/"+k+"?t="+Date.now());
      if(!r.ok) continue;
      m.textContent="gemini input · "+(k==="person"?"PERSON CROP":"full scene (no person crop this tick)")
        +" · "+r.headers.get("X-Age-S")+"s ago · "+(r.headers.get("X-Detectors")||"");
      const old=v.src;
      v.src=URL.createObjectURL(await r.blob()); if(old)URL.revokeObjectURL(old);
      return;
    }catch(e){}
  }
  m.textContent="gemini input · no artifact this tick";
}
pollCrops(); setInterval(pollCrops, 8000);
</script></div></body></html>"""


@app.get("/")
async def page() -> HTMLResponse:
    return HTMLResponse(_PAGE)


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--since", type=str, default=None,
                    help="ISO soak start (default: now − 10 min)")
    args = ap.parse_args()
    _SINCE = (
        datetime.fromisoformat(args.since).astimezone(timezone.utc)
        if args.since
        else datetime.now(timezone.utc) - timedelta(minutes=10)
    )
    globals()["_SINCE"] = _SINCE
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
