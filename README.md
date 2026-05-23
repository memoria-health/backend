# nemara-backend

FastAPI + Supabase backend for the **Nemara caregiver iOS app** and the **detector subscriber-bot**.

Identity, patients, cameras, detectors, events, plus the Phase 0–3 detector pipeline: 6-digit camera pairing, per-camera api-keys, LiveKit token minting, an in-process **LiveKit subscriber-bot** (Phase 2) that pulls frames + audio off the SFU and runs the same Gemini → OpenAI provider fallback ported from `nemara-temp-dashboard`, and Phase 3 caregiver Talk mode. APNs push and event-clip archival via LiveKit Track Egress (Phase 4) are out of scope here.

## Quick start

```bash
brew install uv supabase
make install                  # uv sync
cp .env.example .env          # fill in values from `supabase status -o env`
supabase start                # one-time local Postgres + Supabase services
make db-reset                 # apply supabase/schema.sql + seed demo data
make dev                      # uvicorn on :8002 with --reload
```

Health check: `curl http://localhost:8002/health` → `{"status":"ok"}`.

## Where things live

```
nemara-backend/
├── src/
│   ├── main.py              # @app entry, lifespan, router registration
│   ├── settings.py          # pydantic-settings, NEMARA_* prefix
│   ├── errors.py            # WolfError-style exception hierarchy
│   ├── api/                 # FastAPI routers (one file per aggregate)
│   │   ├── deps.py          #   CaregiverAuth (JWT) + CameraAuth (api-key)
│   │   ├── caregivers.py    #   /v1/me
│   │   ├── patients.py      #   /v1/patients...
│   │   ├── cameras.py       #   list/delete + /v1/patients/{id}/cameras/pairing-code
│   │   ├── detectors.py     #   /v1/detectors...
│   │   ├── events.py        #   /v1/events...
│   │   ├── pairing.py       #   /v1/cameras/pair  (Phase 0)
│   │   ├── livekit.py       #   /v1/livekit/token (Phase 0/1/3)
│   │   ├── livekit_webhooks.py #   /v1/livekit/webhooks (Phase 2 — subscriber-bot discovery)
│   │   └── talk.py          #   /v1/patients/{id}/active-camera + /talk/{start,end} (Phase 3)
│   ├── db/                  # SQLAlchemy ORM models (one per table)
│   ├── repos/               # pure async queries; no business logic
│   ├── schemas/             # pydantic v2 request/response models
│   ├── services/            # business logic, owns transactions
│   │   ├── auth.py          #   JWT verification (HS256 dev / ES256 prod)
│   │   ├── onboarding.py    #   provision caregiver + seed presets
│   │   └── detection/       #   Phase 0 detector pipeline (port of dashboard's lib/)
│   │       ├── prompt.py       # build_prompt_and_schema + BASE_PROMPT + always-on people_present
│   │       ├── registry.py     # load enabled detectors for a camera
│   │       ├── providers/      # gemini.py, openai.py, _parse.py, types.py
│   │       ├── frame_extraction.py  # ffmpeg subprocess (used by OpenAI fallback)
│   │       ├── state_machine.py # open↔closed per-camera state
│   │       ├── analyses_store.py # INSERT into analyses
│   │       ├── event_promotion.py # analysis → events row promotion
│   │       │   ── Phase 2 subscriber-bot ──
│   │       ├── worker.py          # DetectorWorker — lifespan entry, owns the task tree
│   │       ├── room_worker.py     # one RoomWorker per LiveKit room
│   │       ├── camera_ingest.py   # one CameraIngest per camera; 10s tick loop
│   │       ├── frame_buffer.py    # I420 ring of recent video frames
│   │       ├── mp4_remux.py       # frames → MP4 via ffmpeg stdin pipe
│   │       └── motion_cache.py    # per-camera latest motion_state heartbeat
│   └── utils/
├── supabase/
│   ├── config.toml          # local Supabase config
│   ├── schema.sql           # single-file schema (edit-in-place; no migrations)
│   └── seed.sql             # demo data
├── scripts/
│   ├── seed.py              # programmatic seed (calls services, not raw SQL)
│   └── e2e/                 # scenario CLI (`uv run e2e <command>`)
│       └── cli.py           #   fire-event, mint-pairing-code, simulate-clip, etc.
└── test/
    ├── unit/
    └── integration/         # FastAPI TestClient against the local Supabase
```

## Conventions

- **Single-file schema** — `supabase/schema.sql`. Edit in place, run `make db-reset`. No migrations until we're past MVP.
- **Repository pattern** — `repos/*` hold pure SQLAlchemy queries. Services own transactions and call `session.commit()`.
- **Two auth paths** — both presented as `Authorization: Bearer …`:
  - **Caregiver JWT** (Supabase) on every iOS-facing route. `verify_supabase_jwt` dispatches on the token's `alg` header — ES256 via JWKS in prod, HS256 with `NEMARA_SUPABASE_JWT_SECRET` for local dev. Exposed via `CaregiverAuth` dep in `api/deps.py`.
  - **Camera api-key** (Phase 0) on wall-phone routes (`/v1/clips/ingest`, `/v1/livekit/token` for camera role). The key is bcrypt-hashed on the `cameras` row and verified against every active camera in a linear scan (O(N), small N). Exposed via `CameraAuth` dep.
- **RLS is on for every table.** Belt-and-braces against the FastAPI repo layer, **and** the filter Supabase Realtime applies when iOS subscribes to table changes. `analyses` uses `caregiver_id = auth.uid()` directly (denormalized for cheap reads); other tables filter via the patients-ownership subselect.
- **Snake_case JSON** on the wire (FastAPI / Pydantic default). iOS Codable models map to camelCase via explicit `CodingKeys`.
- **Pinned to Python 3.13** via `.python-version`. Python 3.14 silently skips `.pth` files starting with `_`, which breaks hatchling's editable install (`_editable_impl_*`). Don't bump until upstream fixes this.

## Detector pipeline

The dashboard's detector pipeline lives here, in Python, behind an
in-process **LiveKit subscriber-bot** (Phase 2). The wall phone
publishes to a `patient_<id>` LiveKit room; the backend joins the
same room as a viewer-role participant and consumes video frames +
the camera's `motion_state` data-packet heartbeats.

1. **Discovery:** `/v1/livekit/webhooks` receives LiveKit Cloud's
   `participant_joined` / `participant_left` events. When a
   `cam_<...>` identity joins a `patient_<...>` room, the worker
   spawns a `RoomWorker` for that room (or attaches if already joined).
2. **Subscription:** `RoomWorker` joins the room and listens for
   `track_subscribed`. The first track from a new participant resolves
   to a `CameraIngest` via DB lookup on `cameras.livekit_identity`.
3. **Frame buffer:** `CameraIngest` consumes `rtc.VideoStream` frames,
   converts to I420, copies into bytes, and pushes into a
   time-bounded `FrameBuffer` (window 15s, target 5fps).
4. **Motion state:** `RoomWorker` also listens for `motion_state`
   data packets the wall phone publishes every ~5s; the latest value
   per camera lands in a shared `MotionCache` (TTL 15s).
5. **Tick:** every 10s per camera, the tick body — which **calls the
   same registry / state-machine / analyses-store / event-promotion
   functions** the Phase 0 HTTP route used to — runs:
   - Detector load from `services/detection/registry.py`.
   - Closed-state short-circuit when no motion is cached.
   - Frames → MP4 via `mp4_remux.py` (ffmpeg stdin pipe).
   - `gemini.py` first, falls back to `openai.py` on any exception.
     `_parse.py` validates the JSON output and clamps
     `*_confidence` to `[0, 1]`.
   - `state_machine.py` flips `cameras.state` between `open` and
     `closed`; `event_promotion.py` writes `events` rows with
     30-second dedup per `(patient, detector)`.

The worker is webhook-driven and zero-cost when idle: until a wall
phone is paired and publishing, no rooms are joined and no SDK
connections are open. Configure LiveKit Cloud's webhook URL to point
at `{backend}/v1/livekit/webhooks` to wire it up.

## Caregiver Talk mode (Phase 3)

The iOS Live tab is a phone-call metaphor: the caregiver swipes a
carousel of camera tiles and taps a call button overlaid on the
currently-displayed tile. Audio routes to **the camera the caregiver
picked** — server-side best-guess is a suggestion for the carousel's
initial page, not a routing decision.

- `GET /v1/patients/{id}/active-camera` — caregiver-auth. Returns
  `{"camera_id": UUID | null}` — the camera with the freshest
  `analyses.people_present=true` row in the last 30s (see
  `src/services/active_camera.py`). iOS uses this to default the
  Live-tab carousel; `null` falls back to the first online camera
  client-side.
- `POST /v1/patients/{id}/talk/start` — caregiver-auth. Body
  `{"camera_id": UUID}`. Server sends `is_active=true` to the
  specified camera and `is_active=false` to every other camera in the
  patient's room. Re-POSTing with a different `camera_id` transfers
  an active call atomically (no `/end` needed between transfers).
- `POST /v1/patients/{id}/talk/end` — sends `is_active=false` to every
  paired camera. Idempotent; safe to call twice.
- Packet shape (sent on the `set_talk_active` LiveKit topic):
  `{"type": "set_talk_active", "is_active": bool}`. Cameras gate
  local audio playback on the most recent flag — the SFU never mutes
  tracks, so routing changes are O(ms).

## Out of scope (later)

- **Phase 4:** LiveKit Track Egress → S3 event clips. APNs push.
  Multi-replica horizontal scaling (the current subscriber-bot is
  single-process; horizontal scale would need a Redis claim or
  Postgres advisory lock on `camera_id`).
- Photo upload via Supabase Storage.
- Custom-detector LLM prompt compilation (caregiver's free-text →
  runtime config).
- Daily summary email job.

## Sister repos

- [`nemara-caregiver-app`](../nemara-caregiver-app) — the iOS client this backend serves.
- [`elephantLabs-wolf`](../elephantLabs-wolf) — the architectural source for this scaffold (auth, settings, repo pattern).
