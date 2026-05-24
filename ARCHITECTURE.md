# nemara-backend · architecture

How the backend is laid out, what conventions to follow when adding to it, and how the major pipelines wire together.

## File layout

```
nemara-backend/
├── pyproject.toml                # uv + ruff + pytest config; pinned to py 3.13
├── Makefile                      # fresh, fresh-seed, dev, down, db-reset
├── .env                          # tracked; per-machine LAN IP overridden by local-start.sh
├── scripts/
│   ├── local-start.sh            # one-command stack-up (Supabase + buckets + uvicorn + ngrok)
│   ├── seed.py                   # programmatic demo seed (calls services, not raw SQL)
│   └── e2e/cli.py                # `uv run e2e <command>` — fire-event, mint-pairing-code, simulate-clip
├── supabase/
│   ├── config.toml               # local Supabase config; [db.seed] wires schema.sql + seed.sql
│   ├── schema.sql                # SINGLE-FILE schema (edit in place; no migrations until post-MVP)
│   └── seed.sql                  # currently empty; Python seeder runs in scripts/seed.py
├── src/
│   ├── main.py                   # @app entry, lifespan, router registration
│   ├── settings.py               # pydantic-settings, NEMARA_* prefix
│   ├── errors.py                 # WolfError-style exception hierarchy
│   ├── api/                      # FastAPI routers (one file per aggregate)
│   │   ├── deps.py               #   CaregiverAuth (JWT) + CameraAuth (api-key)
│   │   ├── caregivers.py         #   /v1/me
│   │   ├── patients.py           #   /v1/patients...
│   │   ├── cameras.py            #   list/delete + /v1/patients/{id}/cameras/pairing-code
│   │   ├── detectors.py          #   /v1/detectors...
│   │   ├── events.py             #   /v1/events... + /v1/events/{id}/clip-url
│   │   ├── pairing.py            #   /v1/cameras/pair
│   │   ├── livekit.py            #   /v1/livekit/token
│   │   ├── livekit_webhooks.py   #   /v1/livekit/webhooks (subscriber-bot discovery + egress completion)
│   │   ├── talk.py               #   /v1/patients/{id}/active-camera + /talk/{start,end}
│   │   └── devices.py            #   /v1/me/devices (APNs token registration)
│   ├── db/                       # SQLAlchemy ORM models (one per table)
│   ├── repos/                    # pure async queries; no business logic
│   ├── schemas/                  # pydantic v2 request/response models
│   ├── services/
│   │   ├── auth.py               # JWT verification (HS256 dev / ES256 prod)
│   │   ├── onboarding.py         # provision caregiver + seed preset detectors
│   │   ├── storage.py            # Supabase Storage uploader (photos + clips); signed URLs
│   │   ├── push.py               # APNs dispatch (aioapns, ES256 .p8)
│   │   ├── clip_archival.py      # event → LiveKit Egress → S3-compat Supabase Storage → ffmpeg concat
│   │   └── detection/            # the detector pipeline
│   │       ├── prompt.py             # BASE_PROMPT + build_prompt_and_schema
│   │       ├── registry.py           # load enabled detectors for a camera
│   │       ├── providers/            # gemini.py, openai.py, _parse.py, types.py
│   │       ├── frame_extraction.py   # ffmpeg subprocess (OpenAI fallback path)
│   │       ├── state_machine.py      # open↔closed per-camera state
│   │       ├── analyses_store.py     # INSERT into analyses
│   │       ├── event_promotion.py    # analysis → events row + APNs + clip kickoff
│   │       ├── worker.py             # DetectorWorker — lifespan entry, owns the task tree
│   │       ├── room_worker.py        # one RoomWorker per LiveKit room
│   │       ├── camera_ingest.py      # one CameraIngest per camera; 10s tick loop
│   │       ├── frame_buffer.py       # I420 ring of recent video frames
│   │       ├── audio_buffer.py       # int16 mono PCM ring; sibling to frame_buffer
│   │       ├── mp4_remux.py          # frames + optional PCM → MP4 via ffmpeg pipe
│   │       └── motion_cache.py       # per-camera latest motion_state heartbeat
│   └── utils/
└── test/
    ├── unit/
    └── integration/              # FastAPI TestClient against the local Supabase
```

## Conventions

- **Single-file schema** — `supabase/schema.sql`. Edit in place, run `make db-reset`. No migration files until we're past MVP.
- **Repository pattern** — `repos/*` hold pure SQLAlchemy queries. Services own transactions and call `session.commit()`.
- **Two auth paths** — both presented as `Authorization: Bearer …`:
  - **Caregiver JWT** (Supabase) on every iOS-facing route. `verify_supabase_jwt` dispatches on the token's `alg` header — ES256 via JWKS in prod, HS256 with `NEMARA_SUPABASE_JWT_SECRET` for local dev. Exposed via `CaregiverAuth` dep in `api/deps.py`.
  - **Camera api-key** on wall-phone routes (`/v1/livekit/token` camera role). The key is bcrypt-hashed on the `cameras` row and verified against every active camera in a linear scan (O(N), small N). Exposed via `CameraAuth` dep.
- **RLS is on for every table.** Belt-and-braces against the FastAPI repo layer, **and** the filter Supabase Realtime applies when iOS subscribes to table changes. `analyses` uses `caregiver_id = auth.uid()` directly (denormalized for cheap reads); other tables filter via the patients-ownership subselect.
- **Snake_case JSON** on the wire (FastAPI / Pydantic default). iOS Codable models map to camelCase via explicit `CodingKeys`.
- **Module-level imports.** No lazy imports inside function bodies.
- **No trivial tests.** Tests must catch real bugs the type system doesn't already.
- **Pinned to Python 3.13** via `.python-version`. Python 3.14 silently skips `.pth` files starting with `_`, which breaks hatchling's editable install. Don't bump until upstream fixes this.

## Detector pipeline (LiveKit subscriber-bot)

The dashboard's detector pipeline lives here, in Python, behind an in-process **LiveKit subscriber-bot**. The wall phone publishes to a `patient_<id>` LiveKit room; the backend joins the same room as a viewer-role participant and consumes video frames + the camera's `motion_state` data-packet heartbeats.

1. **Discovery:** `/v1/livekit/webhooks` receives LiveKit Cloud's `participant_joined` / `participant_left` events. When a `cam_<...>` identity joins a `patient_<...>` room, the worker spawns a `RoomWorker` for that room (or attaches if already joined). The handler also flips `cameras.online` on join / off on leave, and bumps `last_seen_at`.
2. **Subscription:** `RoomWorker` joins the room and listens for `track_subscribed`. The first track from a new participant resolves to a `CameraIngest` via DB lookup on `cameras.livekit_identity`.
3. **Frame buffer:** `CameraIngest` consumes `rtc.VideoStream` frames, converts to I420, copies into bytes, and pushes into a time-bounded `FrameBuffer` (window 15s, target 5fps).
4. **Audio buffer:** Sibling `AudioBuffer` consumes `rtc.AudioStream` int16 mono PCM at 16kHz. Same TTL-ring semantics. Used by the MP4 remux step so the VLM sees audio cues, and by clip archival's pre-event clip.
5. **Motion state:** `RoomWorker` also listens for `motion_state` data packets the wall phone publishes every ~5s; the latest value per camera lands in a shared `MotionCache` (TTL 15s).
6. **Tick:** every 10s per camera, the tick body runs:
   - Detector load from `services/detection/registry.py`.
   - Closed-state short-circuit when no motion is cached.
   - Frames (+ PCM) → MP4 via `mp4_remux.py` (ffmpeg stdin pipe).
   - `gemini.py` first, falls back to `openai.py` on any exception. `_parse.py` validates the JSON output and clamps `*_confidence` to `[0, 1]`.
   - `state_machine.py` flips `cameras.state` between `open` and `closed`; `event_promotion.py` writes `events` rows with 30-second dedup per `(patient, detector)`.
7. **Event side-effects** (in `event_promotion.maybe_promote_analysis`):
   - APNs push to all of the patient's caregiver's registered devices (fire-and-forget, doesn't block tick).
   - Clip archival kickoff: pre-event clip uploaded immediately from `FrameBuffer` + `AudioBuffer`; LiveKit Track Composite Egress fired for the next 15s; concat happens in the `egress_ended` webhook.

The worker is webhook-driven and zero-cost when idle: until a wall phone is paired and publishing, no rooms are joined and no SDK connections are open.

## Caregiver Talk mode

The iOS Live tab is a phone-call metaphor: the caregiver swipes a carousel of camera tiles and taps a call button overlaid on the currently-displayed tile. Audio routes to **the camera the caregiver picked** — server-side best-guess is a suggestion for the carousel's initial page, not a routing decision.

- `GET /v1/patients/{id}/active-camera` — caregiver-auth. Returns `{"camera_id": UUID | null}` — the camera with the freshest `analyses.people_present=true` row in the last 30s.
- `POST /v1/patients/{id}/talk/start` — caregiver-auth. Body `{"camera_id": UUID}`. Server sends `is_active=true` to the specified camera and `is_active=false` to every other camera in the patient's room. Re-POSTing with a different `camera_id` transfers an active call atomically.
- `POST /v1/patients/{id}/talk/end` — sends `is_active=false` to every paired camera. Idempotent.
- Packet shape (sent on the `set_talk_active` LiveKit topic): `{"type": "set_talk_active", "is_active": bool}`. Cameras gate local audio playback on the most recent flag — the SFU never mutes tracks, so routing changes are O(ms).

## Clip archival

After an event fires:

1. **Pre-event clip** (15s before the fire): assembled in-process from the existing `FrameBuffer` + `AudioBuffer` via `mp4_remux.frames_to_mp4`. Uploaded to `incident-clips/events/{event_id}/pre.mp4` via the Supabase Storage S3-compat API.
2. **Post-event clip** (15s after the fire): a LiveKit Track Composite Egress starts immediately, configured to write to the same bucket via LiveKit's `EncodedFileOutput.s3` pointing at Supabase's S3-compat endpoint. Stops 15s later via a scheduled `_stop_after_delay` task.
3. **Concat**: `egress_ended` webhook fires `_on_egress_ended` → downloads both clips, runs `ffmpeg -f concat -c copy`, uploads the result as `events/{event_id}.mp4`, deletes the intermediates, flips `events.clip_storage_key` + `has_clip=true`.
4. Caregiver app: when `has_clip=true`, `GET /v1/events/{id}/clip-url` mints a short-lived signed URL the iOS `VideoPlayer` plays inline.

## Out of scope (later)

- Multi-replica horizontal scaling — the current subscriber-bot is single-process. Horizontal scale would need a Redis claim or Postgres advisory lock on `camera_id`.
- Custom-detector LLM prompt compilation pipeline (caregiver free-text → runtime config).
- Daily summary email job.
- Sensitive-content gate on the Live view.
- Admin pause/resume endpoint for the worker.

## Sister repos

- [`nemara-camera-app`](../nemara-camera-app)
- [`nemara-caregiver-app`](../nemara-caregiver-app)
