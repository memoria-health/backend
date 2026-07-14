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

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from sqlalchemy import select

from src.db.analysis import Analysis
from src.db.detector import Detector
from src.db.event import Event
from src.db.session import get_sessionmaker
from src.settings import get_settings

_PRICE_IN = 0.10 / 1e6
_PRICE_OUT = 0.40 / 1e6

app = FastAPI()
_SINCE: datetime = datetime.now(timezone.utc)
_FRAMES_DIR = Path("out/id_frames")


@app.get("/data")
async def data() -> JSONResponse:
    settings = get_settings()
    sm = get_sessionmaker()
    async with sm() as s:
        rows = (await s.execute(
            select(Analysis).where(Analysis.captured_at >= _SINCE)
            .order_by(Analysis.captured_at.desc()).limit(400)
        )).scalars().all()
        events = (await s.execute(
            select(Event, Detector.identifier)
            .join(Detector, Event.detector_id == Detector.id, isouter=True)
            .where(Event.created_at >= _SINCE)
            .order_by(Event.created_at.desc()).limit(30)
        )).all()

    cams: dict[str, dict] = {}
    ticks = []
    tot_in = tot_out = 0
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
        c["wakes"] += 1 if f.get("woke_by_backend") else 0
        c["degraded"] += 1 if (
            f.get("sched_degraded_audio") or f.get("sched_degraded_fps")
            or f.get("sched_degraded_artifacts")
        ) else 0
        c["errors"] += len(f.get("sched_call_errors") or ())
        for _, cf in (f.get("sched_confirm") or {}).items():
            c["confirm"]["pass" if cf.get("passed") else "SUPPRESSED"] += 1
        tot_in += int(f.get("vlm_input_tokens", 0))
        tot_out += int(f.get("vlm_output_tokens", 0))

        tracks = f.get("sig_tracks") or []
        amounts = [float(t.get("amount", 0.0)) for t in tracks]
        calls = f.get("sched_calls") or []
        fired = [k[:-9] for k, v in f.items() if k.endswith("_detected") and v is True]
        tick = {
            "t": r.captured_at.strftime("%H:%M:%S"),
            "cam": cam,
            "skip": f.get("skipped_reason") if r.provider == "skipped" else None,
            "motion": f.get("motion_src"),
            "ppl": bool(f.get("people_present")),
            "psrc": f.get("people_present_src"),
            "tracks": len(tracks),
            "amount": round(max(amounts), 4) if amounts else None,
            "env": f.get("sig_env_motion"),
            "calls": [c2.get("key", "")[:26] for c2 in calls],
            "gates": [c2.get("gate") for c2 in calls if c2.get("gate")],
            "gaf": f.get("rapid_gate_active_fraction"),
            "fired": fired,
        }
        if c["last"] is None:
            c["last"] = tick
        if len(ticks) < 120:
            ticks.append(tick)

    for c in cams.values():
        c["skips"] = dict(c["skips"])
        c["confirm"] = dict(c["confirm"])

    frames = []
    if _FRAMES_DIR.exists():
        for camdir in sorted(_FRAMES_DIR.iterdir()):
            if camdir.is_dir():
                newest = sorted(camdir.glob("*.jpg"))[-2:]
                frames += [
                    {"cam": camdir.name, "name": p.name} for p in reversed(newest)
                ]

    cost = tot_in * _PRICE_IN + tot_out * _PRICE_OUT
    hours = max((datetime.now(timezone.utc) - _SINCE).total_seconds() / 3600, 1e-6)
    return JSONResponse({
        "now": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "since": _SINCE.strftime("%H:%M UTC"),
        "flags": {
            "scheduler": settings.detection_v2_scheduler_enabled,
            "decision": settings.detection_v2_decision_enabled,
            "id_frames": getattr(settings, "soak_id_frames_enabled", False),
        },
        "cams": sorted(cams.values(), key=lambda c: c["cam"]),
        "ticks": ticks,
        "events": [
            {
                "t": ev.created_at.strftime("%H:%M:%S"),
                "d": ident or "?",
                "title": ev.title[:70],
            }
            for ev, ident in events
        ],
        "totals": {
            "in": tot_in, "out": tot_out,
            "cost": round(cost, 4), "day_pace": round(cost / hours * 24, 2),
        },
        "frames": frames[:6],
    })


@app.get("/frames/{cam}/{name}")
async def frame(cam: str, name: str):
    p = (_FRAMES_DIR / Path(cam).name / Path(name).name).resolve()
    if not str(p).startswith(str(_FRAMES_DIR.resolve())) or not p.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="image/jpeg")


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
  .logwin{background:var(--logbg);border:1px solid var(--line);border-radius:6px;
    max-height:440px;overflow-y:auto;font:11.5px/1.65 ui-monospace,Menlo,monospace}
  .lr{display:flex;gap:9px;padding:2px 9px;white-space:nowrap}
  .logwin .lr:nth-child(even){background:var(--zeb)}
  .lr .t{color:var(--mut);flex:0 0 58px}.lr .c{flex:0 0 62px;color:var(--acc);font-weight:600}
  .lr .b{overflow:hidden;text-overflow:ellipsis}
  .skip{color:var(--mut)} .fired{color:var(--crit);font-weight:700}
  img.ev{width:100%;border-radius:6px;border:1px solid var(--line);margin-top:6px}
  .frames{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}
</style></head><body><div class="wrap">
<header>
  <h1>Memoria soak — live</h1>
  <span class="chip on" id="live">connecting…</span>
  <span class="chip" id="flags"></span>
  <span class="chip" id="since"></span>
  <span class="chip num" id="pace"></span>
</header>
<div class="card"><h2>Cameras</h2><div class="cams" id="cams"></div></div>
<div class="card"><h2>Identity evidence — latest</h2>
  <div class="frames" id="frames"><span class="mut">no frames yet</span></div></div>
<div class="card"><h2>Events</h2><div id="events" class="mono mut">none yet</div></div>
<div class="card"><h2>Tick log — newest first</h2><div class="logwin" id="log"></div></div>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function tickLine(t){
  if(t.skip) return `skipped · ${esc(t.skip)}`;
  let b=`motion=${esc(t.motion??"—")} ppl=${t.ppl?"Y":"n"}(${esc(t.psrc??"—")}) trk=${t.tracks}`;
  if(t.amount!=null) b+=` amt=${t.amount}`;
  if(t.env!=null) b+=` env=${t.env}`;
  if(t.calls.length) b+=` calls=[${t.calls.map(esc).join(", ")}]`;
  if(t.gates.length) b+=` gate=${t.gates.map(esc).join(",")}`;
  if(t.gaf!=null) b+=` rapidAF=${t.gaf}`;
  if(t.fired.length) b+=` ⚑ ${t.fired.map(esc).join(", ")}`;
  return b;
}
async function refresh(){
  try{
    const d=await (await fetch("/data")).json();
    $("live").textContent=`LIVE · ${d.now}Z`;
    $("flags").textContent=`sched=${d.flags.scheduler?"on":"OFF"} · decision=${d.flags.decision?"on":"OFF"} · idframes=${d.flags.id_frames?"on":"OFF"}`;
    $("since").textContent=`since ${d.since}`;
    $("pace").textContent=`$${d.totals.cost} so far · $${d.totals.day_pace}/day pace`;
    $("cams").innerHTML=d.cams.map(c=>{
      const l=c.last||{};
      return `<div><div class="mono"><b>${esc(c.cam)}</b> · ${c.ticks} ticks · wakes ${c.wakes} · degraded ${c.degraded} · errors ${c.errors}</div>
      <div class="mono mut">skips: ${esc(JSON.stringify(c.skips))}</div>
      <div class="mono">last: ${tickLine(l)}</div></div>`;
    }).join("") || '<span class="mut">no ticks yet</span>';
    $("frames").innerHTML=d.frames.length? d.frames.map(f=>
      `<div><div class="mono mut">${esc(f.cam)} · ${esc(f.name)}</div>
       <img class="ev" src="/frames/${encodeURIComponent(f.cam)}/${encodeURIComponent(f.name)}"></div>`
    ).join(""):'<span class="mut">no frames yet — appear once someone is on camera</span>';
    $("events").innerHTML=d.events.length? d.events.map(e=>
      `<div>${esc(e.t)}Z &nbsp;<b>${esc(e.d)}</b> &nbsp;${esc(e.title)}</div>`).join(""):"none yet";
    $("log").innerHTML=d.ticks.map(t=>
      `<div class="lr ${t.skip?"skip":""} ${t.fired.length?"fired":""}">
        <span class="t">${esc(t.t)}</span><span class="c">${esc(t.cam)}</span>
        <span class="b">${tickLine(t)}</span></div>`).join("");
  }catch(e){ $("live").textContent="reconnecting…"; }
}
refresh(); setInterval(refresh, 10000);
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
