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
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
_PRICE = {"lite": (0.10 / 1e6, 0.40 / 1e6), "flash": (0.30 / 1e6, 2.50 / 1e6)}

app = FastAPI()
_SINCE: datetime = datetime.now(timezone.utc)
_FRAMES_DIR = Path("out/id_frames")

# Hourly rollup cache — the lateral jsonb explode over a full soak day
# is too heavy to run on every 10s poll; 55s staleness is invisible on
# an hours-scale chart.
_HOURLY_CACHE: tuple[float, list] | None = None

# Per-hour sent-call + token rollup for the whole soak window. A call
# counts as SENT when its ledger entry has no 'gate' key (gated plans
# were never dispatched, $0). Tokens = per-call ledger sums plus the
# row-level fallback fields legacy rows carry.
_HOURLY_SQL = text("""
select to_char(date_trunc('hour', a.created_at), 'MM-DD HH24:00') as h,
       count(*) filter (where a.provider != 'skipped')  as analyzed,
       coalesce(sum(c.calls_sent), 0)                    as calls,
       coalesce(sum(c.led_in), 0)
         + coalesce(sum((a.detector_fields->>'vlm_input_tokens')::bigint), 0)  as tin,
       coalesce(sum(c.led_out), 0)
         + coalesce(sum((a.detector_fields->>'vlm_output_tokens')::bigint), 0) as tout,
       count(*) filter (where a.provider = 'gemini')     as g,
       count(*) filter (where a.provider = 'dashscope')  as d,
       count(*) filter (where a.provider = 'openrouter') as o
from analyses a
left join lateral (
    select count(*) filter (where not (e ? 'gate'))            as calls_sent,
           sum((e->>'input_tokens')::bigint)                   as led_in,
           sum((e->>'output_tokens')::bigint)                  as led_out
    from jsonb_array_elements(
        coalesce(a.detector_fields->'sched_calls', '[]'::jsonb)) e
) c on true
where a.created_at >= :since
group by 1 order by 1
""")


async def _hourly(s) -> list:
    global _HOURLY_CACHE
    now = time.monotonic()
    if _HOURLY_CACHE is not None and now - _HOURLY_CACHE[0] < 55:
        return _HOURLY_CACHE[1]
    rows = (await s.execute(_HOURLY_SQL, {"since": _SINCE})).all()
    out = [
        {"h": r.h, "an": r.analyzed, "calls": int(r.calls),
         "tin": int(r.tin), "tout": int(r.tout),
         "g": r.g, "d": r.d, "o": r.o}
        for r in rows
    ]
    _HOURLY_CACHE = (now, out)
    return out


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
        hourly = await _hourly(s)

    cams: dict[str, dict] = {}
    ticks = []
    tot_in = tot_out = 0
    cost = 0.0
    flash_tok = 0
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
        row_in = int(f.get("vlm_input_tokens", 0))
        row_out = int(f.get("vlm_output_tokens", 0))
        tot_in += row_in
        tot_out += row_out
        led_in = led_out = 0
        for c2 in (f.get("sched_calls") or ()):
            ci = int(c2.get("input_tokens", 0))
            co = int(c2.get("output_tokens", 0))
            tier = "flash" if ":flash:" in str(c2.get("key", "")) else "lite"
            pin, pout = _PRICE[tier]
            cost += ci * pin + co * pout
            led_in += ci
            led_out += co
            if tier == "flash":
                flash_tok += ci + co
        # Tokens outside the per-call ledger (legacy rows, confirm
        # votes): price as lite.
        pin, pout = _PRICE["lite"]
        cost += max(0, row_in - led_in) * pin + max(0, row_out - led_out) * pout

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
                newest = sorted(camdir.glob("*.jpg"))[-30:]
                frames += [
                    {
                        "cam": camdir.name,
                        "name": p.name,
                        # capture reason is encoded in the filename
                        # (…_id-change-1p-unk.jpg) by dump_id_evidence
                        "note": (
                            p.stem.split("_", 1)[1].replace("-", " ")
                            if "_" in p.stem else ""
                        ),
                    }
                    for p in reversed(newest)
                ]

    hours = max((datetime.now(timezone.utc) - _SINCE).total_seconds() / 3600, 1e-6)
    return JSONResponse({
        "now": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "since": _SINCE.strftime("%H:%M UTC"),
        "hourly": hourly,
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
            "in": tot_in, "out": tot_out, "flash_tokens": flash_tok,
            "cost": round(cost, 4), "day_pace": round(cost / hours * 24, 2),
        },
        "frames": frames[:60],
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
  .lr .t{color:var(--mut);flex:0 0 58px}.lr .c{flex:0 0 62px;color:var(--acc);font-weight:600}
  .lr .b{overflow-wrap:anywhere;min-width:0}
  #events>div{overflow-wrap:anywhere;padding:2px 0}
  .skip{color:var(--mut)} .fired{color:var(--crit);font-weight:700}
  img.ev{width:100%;border-radius:6px;border:1px solid var(--line);margin-top:6px}
  .frames{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:10px;
    max-height:430px;overflow-y:auto;padding-right:4px}
  .fnote{color:var(--warn)}
</style></head><body><div class="wrap">
<header>
  <h1>Memoria soak — live</h1>
  <span class="chip on" id="live">connecting…</span>
  <span class="chip" id="flags"></span>
  <span class="chip" id="since"></span>
  <span class="chip num" id="pace"></span>
</header>
<div class="card"><h2>Cameras</h2><div class="cams" id="cams"></div></div>
<div class="card"><h2>Test plan — in order; each step earns trust for the next</h2>
<div class="tblwrap"><table>
<tr><th>#</th><th>Do</th><th>Expect</th></tr>
<tr><td>1</td><td>Un-pause both cams, say "go"</td><td>ticks flip to analyzed · exactly one scene call · no events</td></tr>
<tr><td>2</td><td>Walk into view</td><td>tracks=1 on the tick line · evidence photo with an UNKNOWN box</td></tr>
<tr><td>3</td><td>Enroll, then face the camera</td><td>box flips to PATIENT *face — identity chain proven</td></tr>
<tr><td>4</td><td>One fall (lie down abruptly)</td><td>ONE fall event on the phone — full pipeline works</td></tr>
<tr><td>5</td><td>Stay lying a while</td><td>no second fall event — anti-spam holds</td></tr>
<tr><td>6</td><td>Shake hard ~20 s · later, jumping jacks</td><td>shake → convulsion fires (overlay call + gate in log) · jacks → NOTHING</td></tr>
<tr><td>7</td><td>Pill-take gesture</td><td>medication fires · its zoomed call visible in the log</td></tr>
<tr><td>8</td><td>Glance at TV 2 min · later, watch 20 min straight</td><td>glance: nothing · full watch: ONE alert around minute 15</td></tr>
<tr><td>9</td><td>Drink water on camera before 5:45 pm · skip stretching</td><td>Water check silent · ~6:50 pm stretching absence alert with coverage %</td></tr>
<tr><td>10</td><td>Passive: TV on in empty room ~40 min · fan in closed bedroom</td><td>ZERO events — this is the false-alarm + cost dataset</td></tr>
<tr><td>11</td><td>Burner on, walk away 10 min</td><td>stove alert</td></tr>
<tr><td>12</td><td>Privacy on 15 min (ideally mid-recording)</td><td>ticks continue · no clips · in-flight recording dies un-uploaded</td></tr>
</table></div>
<p class="mut" style="font-size:12px;margin:8px 0 0">Steps 1–4 are the "system is alive" gate. Narrate takes in chat ("fall, now") — Claude timestamps <span class="mono">soak_truth.csv</span>; evening scoring: <span class="mono">soak_report.py --hours 4 --truth soak_truth.csv</span></p>
</div>
<div class="card"><h2>Identity evidence — latest</h2>
  <div class="frames" id="frames"><span class="mut">no frames yet</span></div></div>
<div class="card"><h2>Events</h2><div id="events" class="mono mut">none yet</div></div>
<div class="card"><h2>API calls per hour</h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">Each bar = model calls sent that hour. The dollar figure is that hour's estimated cost. Hover any bar for tokens and provider breakdown.</p>
<div id="hourtotals" class="mono" style="margin:0 0 8px"></div>
<div id="hourly" class="mono mut">no calls yet — waiting for the first analyzed tick</div></div>
<div class="card"><h2>Tick log — newest first</h2>
<p class="mut" style="font-size:11.5px;margin:0 0 8px">How to read a line: every detector in the loadout is asked on every analyzed tick, grouped into the calls shown — <span class="mono">key·Ndet</span> = one Gemini call carrying N detectors. Silence = every detector answered no; <span class="mono">⚑ name</span> = that detector answered YES. <span class="mono">⛔closed</span> = that call was pre-gated and never sent ($0) — today only the convulsion bucket has a pre-gate (<span class="mono">rapid_gateAF</span> is its motion evidence), which is why gate chatter looks convulsion-specific. Absence/scheduled detectors don't ride calls; they evaluate from coverage at their deadline. <span class="mono">fps=A→B</span> = LiveKit delivered A fps to the backend (stream health; want ≥8) → buffer kept B for analysis (capped at the 4fps target by design; ~3.4 is normal, NOT a problem); <span class="mono">gap</span> = worst frame silence in the window.</p>
<div class="logwin" id="log"></div></div>
<div class="card"><h2>Compiler acceptance — ran Jul 14 (pure-rules half; pinned by 22 unit goldens)</h2>
<div class="tblwrap"><table>
<tr><th>Ask</th><th>Routed to</th><th>Extras</th></tr>
<tr><td>watching tv &gt;15 min</td><td>still-person · zoom · sampled</td><td>entry 5×-in-15min</td></tr>
<tr><td>stove left on</td><td>room · snapshot</td><td>on sight</td></tr>
<tr><td>dad leaves through the front door</td><td>PERSON, not door</td><td>exit-verb routing ✓</td></tr>
<tr><td>visitor knocks / stranger enters</td><td>anyone</td><td>✓</td></tr>
<tr><td>calls out for help</td><td>patient</td><td>audio: yes ✓</td></tr>
<tr><td>picking at fingers &gt;10 min</td><td>person · wide view</td><td>distance warning · entry 3×-in-5min</td></tr>
<tr><td>door left open 15 min</td><td>room · snapshot</td><td></td></tr>
<tr><td>sleeping over an hour</td><td>still-person · sampled</td><td>entry 8×-in-30min</td></tr>
<tr><td>pacing at night</td><td>person · every tick (safety)</td><td>✓</td></tr>
</table></div>
<p class="mut" style="font-size:12px;margin:8px 0 0">Refusals: eye-twitch / shivering / wearing-a-ring rejected with plain-language reasons · "violent shaking, <i>not just</i> shivering" and the medical bracelet correctly accepted (false-reject class: zero) · reserved field names refused loudly. Notes: "10 minutes" sits between presets and snaps to the sooner one (alerts earlier); fingertip fidgeting rides the wide view with a warning — conservative until soak data justifies the zoom reroute.</p>
</div>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]))}
function tickLine(t){
  const fps=t.rfps!=null?` fps=${t.rfps}→${t.kfps??"?"} gap=${t.gap??"?"}ms`:"";
  if(t.skip==="tick_failed")
    return `<span class="crit">⛔ TICK FAILED · ${esc(t.err??"?")}: ${esc(t.emsg??"")}</span>${fps}`;
  if(t.skip) return `skipped · ${esc(t.skip)}${fps}`;
  let b=`motion=${esc(t.motion??"—")} ppl=${t.ppl?"Y":"n"}(${esc(t.psrc??"—")}) trk=${t.tracks}`;
  if(t.amount!=null) b+=` amt=${t.amount}`;
  if(t.env!=null) b+=` env=${t.env}`;
  if(t.calls.length) b+=` calls=[${t.calls.map(c=>
    esc(c.k)+"·"+c.n+"det"+(c.g?" ⛔"+esc(c.g):"")).join(" | ")}]`;
  if(t.gaf!=null) b+=` rapid_gateAF=${t.gaf}`;
  b+=fps;
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
    $("cams").innerHTML=d.cams.map(c=>{
      const l=c.last||{};
      const k=cons[c.cam];
      const idline=k&&k.occ?`<div class="mono ${k.loss?"warn":"mut"}">id-track ${Math.round(100*k.trk/k.occ)}% of ${k.occ} occupied ticks · losses ${k.loss} (log window)</div>`:"";
      return `<div><div class="mono"><b>${esc(c.cam)}</b> · ${c.ticks} ticks · wakes ${c.wakes} · degraded ${c.degraded} · errors ${c.errors}</div>
      <div class="mono mut">skips: ${esc(JSON.stringify(c.skips))}</div>${idline}
      <div class="mono">last: ${tickLine(l)}</div></div>`;
    }).join("") || '<span class="mut">no ticks yet</span>';
    $("frames").innerHTML=d.frames.length? d.frames.map(f=>
      `<div><div class="mono mut">${esc(f.cam)} · ${esc(f.name.slice(9,11))}:${esc(f.name.slice(11,13))}:${esc(f.name.slice(13,15))}${f.note?` · <span class="fnote">${esc(f.note)}</span>`:""}</div>
       <img class="ev" loading="lazy" src="/frames/${encodeURIComponent(f.cam)}/${encodeURIComponent(f.name)}"></div>`
    ).join(""):'<span class="mut">no frames yet — captured on identity changes (appear/leave/label-flip/face-lock), losses, and every ~2min steady</span>';
    $("events").innerHTML=d.events.length? d.events.map(e=>
      `<div>${esc(e.t)}Z &nbsp;<b>${esc(e.d)}</b> &nbsp;${esc(e.title)}</div>`).join(""):"none yet";
    if(d.hourly&&d.hourly.length){
      const cost=x=>x.tin*0.10/1e6+x.tout*0.40/1e6;
      const mx=Math.max(...d.hourly.map(x=>x.calls),1);
      const totCalls=d.hourly.reduce((a,x)=>a+x.calls,0);
      const totCost=d.hourly.reduce((a,x)=>a+cost(x),0);
      $("hourtotals").textContent=
        `Since start: ${totCalls} calls · $${totCost.toFixed(2)} estimated · `+
        `on pace for $${(totCost/Math.max(1,d.hourly.length)*24).toFixed(2)}/day`;
      $("hourly").innerHTML=`<div style="display:flex;align-items:flex-end;gap:8px;overflow-x:auto;padding:4px 0">`+
        d.hourly.map(x=>{
          const tip=`${esc(x.h)} — ${x.calls} calls from ${x.an} analyzed ticks\n`+
            `${(x.tin/1000).toFixed(1)}k tokens in, ${(x.tout/1000).toFixed(1)}k out\n`+
            `Gemini ${x.g} · DashScope ${x.d} · OpenRouter ${x.o}`;
          return `<div style="text-align:center;min-width:56px" title="${tip}">
          <div class="mono" style="font-size:11px">${x.calls}</div>
          <div style="height:${Math.max(3,Math.round(90*x.calls/mx))}px;background:var(--ok);border-radius:3px 3px 0 0;opacity:.85"></div>
          <div class="mono" style="font-size:11px;margin-top:3px">${esc(x.h.slice(6))}</div>
          <div class="mono mut" style="font-size:10.5px">$${cost(x).toFixed(2)}</div>
        </div>`}).join("")+`</div>`;
    }
    $("log").innerHTML=d.ticks.map(t=>
      `<div class="lr ${t.skip==="tick_failed"?"fail":t.skip?"skip":""} ${t.fired.length?"fired":""}">
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
