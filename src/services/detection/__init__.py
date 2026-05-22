"""Detection service.

Lives under `src/services/` alongside `onboarding.py` and `auth.py` —
the orchestration layer above `src/repos/` (DB CRUD) and `src/db/`
(ORM models). Sub-package shape mirrors the dashboard's
`nemara-temp-dashboard/lib/{detectors,providers,camera-state,extract-frames}.ts`
1:1 so the port is reviewable side-by-side.

Phase 0 wires this synchronously into `POST /v1/clips/ingest`. Phase 2
will run it as a long-running asyncio worker that subscribes to
LiveKit rooms — the same modules just driven by a different input
loop.

Module layout:
    prompt.py            — BASE_PROMPT + people_present (always-on) +
                            `build_prompt_and_schema(detectors)`
    registry.py          — load enabled detectors for a camera
    providers/types.py   — ProviderResult / Provider protocol
    providers/_parse.py  — shared JSON validation + clamping
    providers/gemini.py  — Gemini path (MP4 inline)
    providers/openai.py  — OpenAI fallback (extracted frames @ 5fps)
    frame_extraction.py  — ffmpeg subprocess for OpenAI frames
    state_machine.py     — open/closed transitions per camera
    analyses_store.py    — INSERT into `analyses`
    event_promotion.py   — analysis → event row when something fires
"""
