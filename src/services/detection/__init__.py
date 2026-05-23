"""Detection service.

Lives under `src/services/` alongside `onboarding.py` and `auth.py` —
the orchestration layer above `src/repos/` (DB CRUD) and `src/db/`
(ORM models). Sub-package shape mirrors the dashboard's
`nemara-temp-dashboard/lib/{detectors,providers,camera-state,extract-frames}.ts`
1:1 so the port is reviewable side-by-side.

Driven by an in-process asyncio task tree (`DetectorWorker`) that
joins LiveKit rooms as a viewer participant and runs the pipeline
on a per-camera 10s tick. The provider / state-machine / analyses /
event-promotion modules are transport-agnostic — `camera_ingest.py`
is the orchestrator that ties them together.

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
    ── Subscriber-bot ──
    worker.py            — DetectorWorker (lifespan entry)
    room_worker.py       — one RoomWorker per LiveKit room
    camera_ingest.py     — one CameraIngest per camera; the tick body
    frame_buffer.py      — I420 ring of recent video frames
    mp4_remux.py         — frames → MP4 via ffmpeg stdin pipe
    motion_cache.py      — per-camera motion_state heartbeat cache
"""
