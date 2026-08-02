"""LLM-authored Home report — ONE author for the whole Home summary.

The model receives EVIDENCE (cameras, events, presence, per-night monitoring
coverage) plus the caregiver's current local time, reasons over it, and
returns strict JSON: severity + a short highlight line + a short description.
Rules enforced in-prompt: problems first, verified claims only (a "quiet
night" requires the night was actually monitored), natural times, hard
length caps.
"""

import base64
import json
from datetime import datetime

from fastapi import APIRouter, Request
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.api.deps import CaregiverAuth, SessionDep, SettingsDep

router = APIRouter(prefix="/v1", tags=["home-report"])


class HomeReportRequest(BaseModel):
    timezone: str = Field(default="UTC", max_length=64)
    local_time: str = Field(default="", max_length=40)  # client's "now", human form


class HomeReportResponse(BaseModel):
    severity: str      # ok | warn | urgent
    highlight: str
    description: str


_PROMPT = """You are the single author of a home-care app's Home summary.
Caregiver's current local time: {now} ({tz}). Evidence (JSON): {evidence}

Return STRICT JSON: {{"severity":"ok|warn|urgent","highlight":"...","description":"..."}}

RULES:
- highlight: max 60 characters. Priority: unreviewed serious event > cameras
  off / not set up > all clear ("Everyone is alright" / "All quiet for X").
  Be precise about scope: "Some cameras are offline" vs "All cameras are
  offline" vs "Joey's cameras are offline" - never a bare "Cameras offline"
  when only some are. Plain words. No filler like "we can't see".
- description: max 280 characters TOTAL. A NOTABLE summary of each person's
  recent days from the evidence: events and their pattern (what, how often,
  when), when they were last in view, activity trend if it changed.
  NEVER mention camera status (online/offline/paused/not set up) in the
  description - the app shows exact camera status separately. People only.
  Skip people with nothing notable. Wrap names in **bold**. Group identical
  situations.
- NEVER use the phrase "quiet night(s)". Describe only what the evidence
  shows; absence of monitoring is never described as calm.
- PRESENT STATUS: if LIVE frames are attached, lead each person's line with
  what is visible RIGHT NOW - whether someone is in view and their posture
  (standing / sitting / lying down) and where. Observational words only.
  If no one is visible, say so plainly ("no one in the living room right now").
- Data-descriptive ONLY: never health advice, diagnosis, or interpretation.
- Times: use the caregiver's local clock. Today -> "7:21 PM". Yesterday ->
  "yesterday evening". Older -> "July 24". Never print raw timestamps.
- Write COMPLETE natural sentences a person would say aloud. Never
  telegraphic lists, colons, or semicolons - instead: "Over the past week
  she showed agitation eight times and wandered six times." Describe
  activity using the provided daily_activity "observed" label VERBATIM in a
  sentence (e.g. "she has been mostly still or out of view today") - never
  invent your own characterization, never a percentage, never "full". If no
  label is given for a day, say nothing about that day's activity.
- Say each thing once. No advice, no speculation, no meta-commentary,
  apostrophes and normal punctuation, no exclamation marks."""


def _live_frames(request: Request, identities: dict[str, str]) -> list[tuple[str, bytes]]:
    """(label, jpeg) for each live camera — straight from the detection
    worker's in-process frame buffers. Zero extra capture cost."""
    out: list[tuple[str, bytes]] = []
    worker = getattr(request.app.state, "detector_worker", None)
    if worker is None:
        return out
    for room_worker in getattr(worker, "_rooms", {}).values():
        for identity, ingest in getattr(room_worker, "_ingests_by_identity", {}).items():
            label = identities.get(identity)
            if label is None:
                continue
            frames = ingest.frame_buffer.snapshot_last_seconds(4.0)
            if frames:
                out.append((label, frames[-1].data))
    return out


@router.post("/home-report", response_model=HomeReportResponse)
async def home_report(
    body: HomeReportRequest, request: Request, auth: CaregiverAuth,
    session: SessionDep, settings: SettingsDep,
) -> HomeReportResponse:
    ev = []
    identities: dict[str, str] = {}
    pats = (await session.execute(text(
        "select id, coalesce(nullif(preferred_name,''), full_name) from patients where caregiver_id = :cg"),
        {"cg": str(auth.id)})).all()
    for pid, name in pats:
        cams = (await session.execute(text(
            "select room, online, ai_paused, livekit_identity from cameras where patient_id=:p and not disabled"),
            {"p": str(pid)})).all()
        events = (await session.execute(text(
            """select title, occurred_at at time zone :tz, reviewed from events
               where patient_id=:p and occurred_at > now() - interval '48 hours'
               order by occurred_at desc limit 6"""),
            {"p": str(pid), "tz": body.timezone})).all()
        seen = (await session.execute(text(
            """select max(captured_at) filter (where people_present) at time zone :tz,
                      max(captured_at) at time zone :tz
               from analyses where patient_id=:p"""),
            {"p": str(pid), "tz": body.timezone})).first()
        week = (await session.execute(text(
            """select initcap(replace(lower(title),'possible ','')), count(*)::int
               from events where patient_id=:p and occurred_at > now() - interval '7 days'
               group by 1 order by 2 desc limit 6"""), {"p": str(pid)})).all()
        act_rows = (await session.execute(text(
            """select (captured_at at time zone :tz)::date, count(*),
                      round(100.0*avg((motion_in_clip and people_present)::int))
               from analyses where patient_id=:p and captured_at > now() - interval '3 days'
               group by 1 order by 1 desc"""), {"p": str(pid), "tz": body.timezone})).all()
        def _label(ticks: int, pct: float) -> str | None:
            # deterministic, honest label — the model must not interpret numbers
            if ticks < 30:
                return None  # too little observation to say anything
            if pct < 10:
                return "mostly still or out of view"
            if pct < 35:
                return "some movement"
            if pct < 65:
                return "fairly active"
            return "very active"
        act = [(d, lab) for d, t, p in act_rows if (lab := _label(t, float(p or 0)))]
        for rm, o, a, ident in cams:
            if o and not a and ident:
                identities[ident] = f"{name} — {rm or 'camera'}"
        ev.append({
            "name": name,
            "cameras": [{"room": r, "online": o, "ai_paused": a} for r, o, a, _ in cams],
            "recent_events": [{"what": t, "local_time": str(w), "reviewed": rv} for t, w, rv in events],
            "last_person_seen_local": str(seen[0]) if seen and seen[0] else None,
            "last_analysis_local": str(seen[1]) if seen and seen[1] else None,
            "event_pattern_7d": [{"what": t, "times": n} for t, n in week],
            "daily_activity": [{"day": str(d), "observed": lab} for d, lab in act],
        })

    frames = _live_frames(request, identities)
    content: list[dict] = [{"type": "text", "text": _PROMPT.format(
        now=body.local_time or datetime.utcnow().isoformat(),
        tz=body.timezone, evidence=json.dumps(ev))}]
    for label, jpeg in frames[:4]:
        content.append({"type": "text", "text": f"LIVE frame, captured seconds ago: {label}"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}})
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model="gpt-5.4-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    out = json.loads(resp.choices[0].message.content or "{}")
    return HomeReportResponse(
        severity=out.get("severity", "ok") if out.get("severity") in ("ok", "warn", "urgent") else "ok",
        highlight=str(out.get("highlight", ""))[:80],
        description=str(out.get("description", ""))[:300],
    )
