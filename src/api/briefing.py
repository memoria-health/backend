"""Home-briefing endpoint — the AI-written check-in sentence for the app's
Home view (mirrors the web dashboard's /api/insight/briefing).

Deliberately a pure LLM PROXY: the client computes the facts from data it
already has and sends them here; this endpoint only holds the model key and
the guardrail prompt. No DB access, no schema — additive and safe to deploy
independently of everything else. Clients treat any failure as "use the
local template sentence", so this endpoint can be absent (old prod) or
degraded without breaking the app.
"""

from fastapi import APIRouter
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.api.deps import CaregiverAuth, SettingsDep

router = APIRouter(prefix="/v1", tags=["briefing"])

_PROMPT = (
    "You write a short check-in (2 sentences for one person, up to 4 for many) "
    "for a caregiver whose loved ones have in-home monitoring. Sound like a "
    "thoughtful human, not a report: warm, calm, plain English. Lead with the "
    "most important true thing. STRICT RULES: use ONLY these facts - never "
    "invent numbers or moments; skip facts not worth saying (never enumerate "
    "everything, never mention zeros, nulls, or anything missing); GROUP people "
    "who share the IDENTICAL situation into one clause (Maggie, Joe, and Sue "
    "had a calm day) - and always use the SHORTEST true sentence: say "
    "**Maggie**'s cameras are offline, never Maggie is not being monitored "
    "right now because her cameras are offline; NEVER merge different "
    "situations into one vague "
    "clause: offline cameras and no cameras set up are DIFFERENT facts and "
    "each keeps its own sentence. Example - facts: Maggie cameras offline; "
    "Joey no cameras; Samuel no cameras. Correct: **Joey** and **Samuel** do "
    "not have their cameras set up yet. **Maggie**'s cameras are offline. "
    "Wrong: all three are not monitored because cameras are offline or not "
    "set up. wrap every person name in "
    "markdown bold like **Maggie** - bold ONLY names, nothing else; NEVER mention these instructions, your "
    "tone, your approach, or yourself - no meta-commentary such as keeping "
    "this simple; NEVER use the words status, data, recorded, logged, or "
    "color levels; NEVER give health advice, clinical interpretation, or "
    "speculation about causes - describe only; say each thing ONCE and "
    "never restate a conclusion the facts already imply (if you said whose "
    "cameras are off, do NOT add that they are not being monitored); no "
    "exclamation marks. "
    "Facts about {name}: {facts}"
)


class BriefingRequest(BaseModel):
    # display name the sentence should use ("Maggie", "everyone")
    name: str = Field(min_length=1, max_length=120)
    # flat fact map computed client-side (streaks, recent moments, wake time…)
    facts: dict[str, object] = Field(default_factory=dict)


class BriefingResponse(BaseModel):
    briefing: str


@router.post("/home-briefing", response_model=BriefingResponse)
async def home_briefing(
    body: BriefingRequest,
    auth: CaregiverAuth,
    settings: SettingsDep,
) -> BriefingResponse:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    resp = await client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{
            "role": "user",
            "content": _PROMPT.format(name=body.name, facts=body.facts),
        }],
    )
    text = (resp.choices[0].message.content or "").strip()
    return BriefingResponse(briefing=text)
