# nemara-backend · architecture

How the backend is laid out, what conventions to follow when adding to it, and how the major pipelines wire together.

## File layout

```
nemara-backend/
├── pyproject.toml                # uv + ruff + pytest config; py >=3.12,<3.14 (.python-version pins 3.13)
├── Makefile                      # install, dev, test[-unit|-integration], lint, format, db-*, e2e, fresh[-seed], down
├── docker/                       # Dockerfile (CPU torch; weights pre-cached) + Dockerfile.dev + docker-compose.local.yml
├── railway.toml                  # Railway deploy config
├── requirements-gpu.txt          # GPU-host deps (ultralytics SAM3, boxmot/OSNet, insightface, onnxruntime-gpu)
├── yolo11n.pt / yolo11s.pt       # YOLO11 person-detector weights
├── scripts/
│   ├── local-start.sh            # one-command stack-up (Supabase + buckets + uvicorn :8002 + ngrok)
│   ├── seed.py                   # programmatic demo seed (calls services, not raw SQL)
│   ├── seed_demo_camera.py       # App Review demo camera (persistent reusable pairing code)
│   ├── reseed_*_prompt.py        # refresh a preset detector's prompt in a live DB
│   ├── apply_*_delta.sql         # idempotent prod schema patches (see Conventions)
│   ├── backfill_detector_dims.py # backfill detection-v2 dimension columns from catalog_v2
│   ├── pod.sh / pod_bootstrap.sh / pod_tunnel.sh / gpu_setup.sh   # GPU-pod ops (perception server)
│   ├── e2e/cli.py                # `uv run e2e <command>` — fire-event, add-patient, flip-detector…
│   └── pi/                       # Pi device-side: install.sh, captive portal, announce daemon, MJPEG pusher
├── supabase/
│   ├── config.toml               # local Supabase config
│   ├── schema.sql                # SINGLE-FILE canonical schema (local); prod patched via delta scripts
│   └── seed.sql                  # empty; Python seeder runs in scripts/seed.py
├── tools/
│   ├── lk_room_audit.py          # LiveKit room snapshot; flags zombie/idle detector workers
│   ├── download_models.py        # pre-fetch model weights
│   ├── live_tracker_view.py      # legacy live-tracker debug viewer (superseded by gpu/identity/service.py)
│   └── soak_dashboard.py / soak_report.py / soak_watch.py   # soak monitoring + analysis
├── gpu/                          # GPU-host deployables (run on the pod / GPU box, NOT in the backend container)
│   ├── perception/               # SAM3 + EdgeTAM continuous-tracking server (docs/architecture/continuous-tracking.md)
│   │   ├── server.py             #   FastAPI: POST /observe (frames → stable tracks) + POST /faces
│   │   ├── continuous.py         #   persistent per-camera CameraSession — SAM3 grounds, EdgeTAM propagates
│   │   └── sam_track.py / face_worker.py / person_trigger.py
│   ├── identity/                 # the consolidated identity pipeline (see gpu/identity/README.md)
│   │   ├── service.py            #   entry point — live MJPEG overlay :8012 + authenticated /state
│   │   └── loop.py / seat.py / scoring.py / harvest.py / gallery.py / matching.py / …
│   └── deploy/identity_supervisor.sh   # restart-on-exit supervisor for the identity service
├── src/
│   ├── main.py                   # app factory; lifespan owns the 3 background runtimes
│   ├── settings.py               # pydantic-settings, NEMARA_* prefix
│   ├── errors.py                 # exception hierarchy
│   ├── api/                      # FastAPI routers (one file per aggregate)
│   │   ├── deps.py               #   CaregiverAuth (JWT) + CameraAuth (api-key) deps
│   │   ├── health.py             #   /health (liveness) + /ready (DB reachability)
│   │   ├── caregivers.py         #   /v1/me CRUD + DELETE (full account deletion)
│   │   ├── patients.py           #   /v1/patients… + photo upload/url + test-email
│   │   ├── patient_members.py    #   patient sharing — owner/member roles, share-by-email
│   │   ├── detectors.py          #   detector CRUD + /detectors/library + /detectors/infer
│   │   ├── events.py             #   list/get/review + /v1/events/{id}/clip-url
│   │   ├── cameras.py            #   cameras CRUD, pairing-code, /v1/cameras/me/* (wall phone)
│   │   ├── cameras_ingest.py     #   Pi push: /v1/cameras/me/frames + /ingest-status
│   │   ├── pairing.py            #   /v1/cameras/pair (6-digit code → api key)
│   │   ├── livekit.py            #   /v1/livekit/token (camera api-key OR caregiver JWT)
│   │   ├── livekit_webhooks.py   #   participant joined/left + room_finished → worker dispatch
│   │   ├── face_embeddings.py    #   identity selfie enrollment + status/summary
│   │   ├── talk.py               #   /active-camera + /talk/{start,end}
│   │   ├── devices.py            #   /v1/me/devices (APNs token registration)
│   │   ├── support.py            #   Contact Support → Resend to founders
│   │   ├── feedback.py           #   product feedback: persist row + best-effort email
│   │   └── internal/             #   internal-route scaffold
│   ├── db/                       # SQLAlchemy ORM models (one per table; 12 tables)
│   ├── repos/                    # pure async queries; no business logic
│   ├── schemas/                  # pydantic v2 request/response models
│   ├── services/
│   │   ├── auth.py               # JWT verification (HS256 dev / ES256 prod)
│   │   ├── auth_admin.py         # Supabase GoTrue admin (delete auth user, service-role key)
│   │   ├── onboarding.py         # provision caregiver + age-based preset detector seeding
│   │   ├── detector_compiler.py  # caregiver free text → compiled custom detector (LLM)
│   │   ├── account_deletion.py   # storage + workers + auth-user cascade delete
│   │   ├── active_camera.py      # freshest people_present camera for Talk carousel
│   │   ├── storage.py            # Supabase Storage (photos + clips); signed URLs
│   │   ├── region.py             # region-aware storage routing (US/IN data-residency partition)
│   │   ├── push.py               # APNs dispatch (HTTP/2, ES256 .p8)
│   │   ├── email.py              # Resend: event alerts + support + feedback
│   │   ├── clip_archival.py      # in-process pre+post event clip → concat → upload
│   │   ├── livekit_privacy.py    # live SFU enforcement when a patient flips privacy mode
│   │   ├── scheduling/scheduler.py  # ScheduledDetectorRunner — absence + trailing detectors
│   │   ├── detection/            # realtime detector pipeline (see section below)
│   │   ├── identity/             # backend half of the identity seam (see section below; was services/reid)
│   │   ├── data_collection/      # opt-in per-patient session recorder → Backblaze B2
│   │   └── ingest/pi/            # Pi MJPEG push → LiveKit relay (see section below)
│   └── internals/                # experimental routes; gated by NEMARA_ENABLE_INTERNALS (default off)
│       ├── record_clip.py        #   POST …/record-clip — on-demand server-side clip recording
│       └── clip_recorder.py      #   standalone LiveKit subscriber → MP4 to local disk
└── test/
    ├── unit/                     # no DB dependency
    └── integration/              # FastAPI TestClient against the local Supabase
```

## Conventions

- **Schema changes are hybrid.** `supabase/schema.sql` is the canonical single-file schema for local dev (`make db-reset` re-applies it). Production is patched live with idempotent delta scripts in `scripts/apply_*_delta.sql` — new features land in schema.sql first, then get a delta backport. No migration framework yet.
- **Repository pattern** — `repos/*` hold pure SQLAlchemy queries. Services own transactions and call `session.commit()`.
- **RLS is on for every table.** Belt-and-braces against the FastAPI repo layer, **and** the filter Supabase Realtime applies when iOS subscribes to table changes. `analyses` uses `caregiver_id = auth.uid()` directly (denormalized for cheap reads); other tables filter via the patients-ownership subselect.
- **Snake_case JSON** on the wire (FastAPI / Pydantic default). iOS Codable models map to camelCase via explicit `CodingKeys`.
- **Module-level imports.** No lazy imports inside function bodies (exception: the `internals` router import is guarded behind its settings flag).
- **No trivial tests.** Tests must catch real bugs the type system doesn't already.
- **Python 3.13 via `.python-version`** (pyproject allows `>=3.12,<3.14`). Python 3.14 silently skips `.pth` files starting with `_`, which breaks hatchling's editable install. Don't bump until upstream fixes this.
- **`.env` is NOT tracked.** Secrets stay out of git (rotated after the early tracked-`.env` period); share dev secrets out-of-band, prod secrets live in the deploy platform. `scripts/local-start.sh` auto-detects the LAN IP so iOS devices on the same network reach local Supabase.
- **Heavy ML deps are split by host.** The backend image ships torch (CPU) + insightface/onnxruntime (SCRFD/ArcFace) for face enrollment; the GPU host installs `requirements-gpu.txt` (ultralytics for SAM3, boxmot/OSNet body embeddings, onnxruntime-gpu) for the perception + identity services under `gpu/`.

## Auth paths

All presented as `Authorization: Bearer …`; four distinct verifiers:

1. **Caregiver JWT** (Supabase) on every iOS-facing route. `verify_supabase_jwt` dispatches on the token's `alg` header — ES256 via JWKS in prod, HS256 with `NEMARA_SUPABASE_JWT_SECRET` for local dev. Exposed via `CaregiverAuth` dep in `api/deps.py`.
2. **Camera api-key** on wall-phone routes (`/v1/cameras/me/*`, `/v1/livekit/token` camera role). bcrypt-hashed on the `cameras` row, verified by linear scan over active cameras (O(N), small N). A well-formed key with no match returns **404** (not 401) so the wall phone knows its row was deleted and resets to pairing. Exposed via `CameraAuth`.
3. **Fleet ingest token** on the gateway/bridge endpoints (`/v1/fleet/poll`, `/v1/gateway/roster`, `/v1/gateway/enroll`, `/v1/livekit/token`). High-entropy `token_urlsafe(32)`, stored as an **unsalted SHA-256** digest (`cameras.ingest_token_hash` / `gateways.token_hash`) so auth is an indexed equality lookup instead of a bcrypt scan.
4. **LiveKit webhook signature** on `/v1/livekit/webhooks` — JWT verified against the LiveKit API key/secret.

`POST /v1/cameras/pair` is deliberately unauthenticated (possession of the pairing code is the trust anchor).

## App lifecycle (main.py)

The lifespan starts three independent background runtimes, all on `app.state`:

- **`DetectorWorker`** — webhook-driven LiveKit subscriber-bot (realtime pipeline below). Zero-cost idle: no rooms joined until a camera publishes.
- **`ScheduledDetectorRunner`** — standalone loop for "absence" detectors (below), ticks every `scheduler_tick_seconds` (default 60s).
- **`PresenceCache` + `PresenceSweeper`** — `cameras.app_present` ownership (below).

Shutdown stops them in reverse, then closes push (APNs) and email (Resend) clients and disposes the engine.

## Detector pipeline (LiveKit subscriber-bot)

The realtime pipeline lives in `src/services/detection/`, behind an in-process LiveKit subscriber-bot. A camera (iPhone wall phone, or the backend's own Pi relay) publishes to a `patient_<id>` room; the backend joins as a viewer and consumes video + audio + `motion_state` data packets.

1. **Discovery:** `/v1/livekit/webhooks` receives `participant_joined` / `participant_left` / `room_finished`. A `cam_<…>` join spawns (or attaches to) a `RoomWorker` for that room and nudges its reconciler; a leave marks the camera offline immediately.
2. **RoomWorker** (`room_worker.py`): one per patient room, with a per-instance unique LiveKit identity (prevents DuplicateIdentity evict storms), token rotation, reconnect supervision with backoff, and a connect-time backfill walk over already-present participants. `track_subscribed` resolves the publisher identity to a `cameras` row and spawns one `CameraIngest` per camera.
3. **Room reconciler** (`room_reconciler.py`): polls LiveKit's REST participant list every ~10s and is the **sole steady-state writer of `cameras.online`**. Two-signal rule: identity present in REST **and** fresh frames at the ingest → online; either drops → offline. Also backfills missed track subscriptions and reaps ingests absent from REST for 2 consecutive polls. On false→true it resets the camera state machine to `open` (fresh start).
4. **Buffers:** `FrameBuffer` (I420, 11s window, ~4fps target — trimmed from 15s/5fps to cut frames per LLM call) and `AudioBuffer` (int16 mono PCM 16kHz). `MotionCache` holds the latest `motion_state` data-packet heartbeat per camera (published every ~5s by the camera, TTL 15s).
5. **Tick** (`camera_ingest.py`, every 10s per camera, fire-and-forget — a slow LLM call doesn't delay the next tick). Gates in order:
   - Camera row gone → raise, reconciler reaps the ingest.
   - `disabled` / `ai_paused` → skip LLM, write a synthetic "skipped" analysis (state machine deliberately not advanced).
   - `state == closed` and no cached motion → skip LLM, advance state with `people_present=false`.
   - **Identity gate:** if the patient is enrolled and the identity service's verdict (read via `services/identity/identity_client.py`, the `/state` seam) positively places the patient elsewhere with no track held as PATIENT → skip patient-scoped work (saves the provider call when e.g. a helper is in frame).
   - Otherwise: snapshot buffers → MP4 via `mp4_remux.py` (ffmpeg pipe) → provider chain per `vlm_routing.py`: the caregiver-account `vlm_provider` picks the primary; the Gemini path runs `gemini_model` (default `gemini-3.6-flash`, native video + audio) with a same-path Gemini model fallback (`gemini_fallback_model`), then DashScope (Alibaba Qwen3-VL, native video), then OpenRouter (frame-sampled Qwen) as the last-resort net. The OpenAI path (`openai_model`, default `gpt-5.4`) uses 4fps JPEG frames via `frame_extraction.py`. `_parse.py` validates the JSON and clamps `*_confidence` to `[0, 1]`.
6. **Presence verdict:** the `people_present` fed to the state machine is `LLM OR identity-patient-match` when enrollment exists; LLM-only otherwise. `state_machine.py` flips `cameras.state` open↔closed atomically (single row-locked UPDATE; `close_threshold` consecutive no-people ticks to close).
7. **Event promotion** (`event_promotion.py`): a detector fires when `<id>_detected == true`, `<id>_confidence >= threshold`, and no event for the same `(patient, detector)` exists in the last 30s. Thresholds: base 0.70 for every catalog detector, shifted by the caregiver's per-detector sensitivity (low +0.15, standard +0, high −0.30; clamped to [0.05, 0.95]).
8. **Side-effects** (fire-and-forget after commit): APNs push to all registered caregiver devices, email alerts (if `patient.email_alerts_enabled`), and clip archival kickoff.

Every tick writes an `analyses` row (~10s cadence; debugging surface, not user-facing).

## Camera liveness: `online` vs `app_present`

Two independent signals with different owners:

- **`cameras.online`** — "is video actually flowing right now?" Owned by the room reconciler (steady state) + webhooks (authoritative immediate offline on leave / room_finished) + a startup sweep.
- **`cameras.app_present`** — "is the Memoria Sensor app open on the iPhone?" (Pi rows default true.) The wall phone's 1s desired-state poll touches the in-memory `PresenceCache` (no DB write on the hot path); `PresenceSweeper` flips the column only on transitions. **Single replica only.**

UI mapping: `online` → LIVE; `!online && app_present` → reconnectable "Disconnected"; neither → "App closed".

## Detection library & custom detectors

- **`detection/catalog.py` is the single source of truth** for every built-in detector: 24 entries (seeded presets + library add-ons + the hidden `setup_test`), each a frozen `CatalogEntry` carrying key, prompt fragment, category (`safety`/`routine`), the alerting spine (`alert_trigger` ∈ `leading`/`trailing`/`duration`/`absence` + `is_sustained` + optional `duration_seconds` — this replaced the old `shape` column), cohort (`child`/`adult`/`both`), and promotion threshold. `detection/catalog_v2.py` holds the built-ins' detection-v2 dimensions (`requires_motion`, `subject_scope`, `bucket`, `duty_cycle`, …). Consumers: onboarding seeding, the `/detectors/library` endpoints, and event promotion's thresholds.
- **Age-based seeding** (`onboarding.py`): minors get Falls + Climbing + Distress; adults get Falls + Agitation + Wandering + Medication.
- **Custom detectors** (`detector_compiler.py`): `POST /v1/detectors/infer` classifies free text into `alert_trigger`/`is_sustained`/category/sensitivity/schedule synchronously; `POST /v1/patients/{id}/detectors` returns 201 immediately and compiles in a background task — the LLM (OpenAI `openai_compile_model`, default `gpt-5.4`; Gemini fallback) writes the prompt fragment, while the `custom_<8hex>` identifier and JSON schema are templated server-side. `scheduling/compiler_v2.py` derives the v2 dimension columns, including the internal M-of-N entry-confidence gate by detection difficulty. Room scope is the `where_rooms` JSONB set (the stored `where_scope` scalar was dropped; it's derived on serialize). Inclusivity is tuned by stakes: reactive-safety detectors compile inclusive (missed emergency > false alarm); routine and scheduled detectors compile conservative.

## Scheduled ("absence") and trailing detectors

`scheduling/scheduler.py` runs `ScheduledDetectorRunner`, independent of the webhook-driven worker. It owns two `alert_trigger` values that fire outside the per-tick path:

- **`absence`** (rename of the old `shape='scheduled'`): a recurring window in `detectors.schedule_config` (JSONB: cadence, days, window, deadline). Each tick (60s), for any detector whose deadline has passed today and isn't yet evaluated (`last_scheduled_eval_date` single-fire guard), it converts the caregiver-timezone window to UTC, queries `analyses` for the watched signal, and — if the signal was **never seen** — fires a patient-level absence event (`camera_id=NULL`, narrative includes the window and camera-uptime %).
- **`trailing`**: fires on the FALLING edge of a sustained activity — the runner watches for an episode that has ended and emits one event per episode.

Both reuse the same push + email dispatchers. The reactive path skips `absence`/`trailing` detectors entirely; the same prompt fragment runs in every tick either way, this runner only changes when the firing happens.

## Patient identity (continuous tracking + identity service)

Identity answers "is the person in frame the *patient*, or someone else?" so detections aren't suppressed (or fired) on the wrong person. Since the 2026-07 consolidation there is **one identity pipeline, and it lives on the GPU host** (`gpu/identity/`, see `gpu/identity/README.md`); the legacy in-process per-tick brain (`inference.py` / `body_pipeline.py` / BoT-SORT tracker in `services/reid/`) was deleted — the sidecar is the only identity provider.

- **Perception substrate** (`gpu/perception/`): a persistent per-camera SAM3 + EdgeTAM continuous-tracking session on the pod produces **stable track ids across ticks** (`POST /observe`) plus a face pass (`POST /faces`). Deep-dive: `docs/architecture/continuous-tracking.md`.
- **Identity service** (`gpu/identity/service.py`, supervised by `gpu/deploy/identity_supervisor.sh`, port 8012): per-camera continuous batches drive the pod session; a three-rule identity layer sits on top — calibrated scoring (face weighted by pixel size, OSNet body pool as the workhorse), a single home-wide patient **seat**, and liberal body-pool **harvest** only when a face-confirmed track is alone in frame. Serves a live MJPEG overlay and an authenticated `/state` JSON (`identity.contract`).
- **Backend consumer** (`src/services/identity/`, renamed from `services/reid` 2026-07-27): `identity_client.py` reads `/state` (~ms), `result.py`/`observations.py` hold the verdict shapes, and `track_ledger.py` is the bind-once-hold sticky lock — every stable track is born UNKNOWN, binds PATIENT on a face/pool match, and is then held until the track dies (retire on grace / bounded scene-extension, `reid_scene_extension_max_s`). Two labels only: PATIENT or UNKNOWN.
- **Enrollment** (stays in the backend): caregiver uploads 1–N selfies (`POST /v1/patients/{id}/face-embeddings`, 202 + background processing). `enrollment_processor.py` downloads from Supabase Storage, runs SCRFD face detection + ArcFace embedding (InsightFace), quality-gates on det score, and marks rows `ready`/`failed`. Embeddings (512-dim float32) live in Postgres `patient_face_embeddings`; the GPU-side gallery reads them back through the DB.
- If identity can't run (no enrollment, sidecar down, timeout), the pipeline degrades to LLM-only presence — identity failure never blocks detection.

## Caregiver Talk mode

The iOS Live tab is a phone-call metaphor: the caregiver swipes a carousel of camera tiles and taps a call button. Audio routes to **the camera the caregiver picked** — the server-side best-guess only seeds the carousel's initial page.

- `GET /v1/patients/{id}/active-camera` — the camera with the freshest `analyses.people_present=true` row in the last 30s (`active_camera.py`), or null.
- `POST /v1/patients/{id}/talk/start` — body `{"camera_id": UUID}`. Sends `is_active=true` to that camera and `is_active=false` to every other; re-POSTing with a different camera transfers the call atomically.
- `POST /v1/patients/{id}/talk/end` — `is_active=false` to every paired camera. Idempotent.
- Packet shape (LiveKit `set_talk_active` topic): `{"type": "set_talk_active", "is_active": bool}`. Cameras gate local audio playback on the latest flag — the SFU never mutes tracks, so routing changes are O(ms).

## Clip archival

Fully in-process — **no LiveKit Egress** (it's documented in `clip_archival.py` as the future alternative if reliability demands it).

1. **Pre-event clip:** the ~11s MP4 the tick already built for the LLM is reused as-is.
2. **Post-event clip:** sleep `clip_post_event_wait_seconds` (15s), then snapshot the live `FrameBuffer` + `AudioBuffer` anchored at the pre-clip's end timestamp (so the LLM's think-time gap doesn't leak into the clip).
3. **Concat + upload:** ffmpeg concat → `incident-clips/events/{event_id}.mp4` (Supabase Storage), then `events.clip_storage_key` + `has_clip=true`; Realtime notifies the app, and `GET /v1/events/{id}/clip-url` mints a short-lived signed URL for inline playback.

(The old `clip_annotate.py` step — burning a green patient bbox onto archived clips — was removed along with the legacy in-process re-ID pipeline.)

If the worker dies during the post-event sleep, the clip is lost (event text survives) — accepted for v0.

## Support, feedback, account deletion

- **`POST /v1/support`** — forwards the message via Resend to the founder inbox(es), reply-to set to the caregiver, caregiver phone included (E.164-normalized via a Pydantic validator on the caregiver schema).
- **`POST /v1/feedback`** — persists a `feedback` row first (source of truth), then best-effort Resend email; email failure never fails the request.
- **`DELETE /v1/me`** (`account_deletion.py`) — collects storage keys (selfies, clips, avatars), best-effort deletes objects and stops detector workers, then deletes the Supabase auth user via the GoTrue admin API — FK cascade from `auth.users` erases caregivers → patients → cameras/detectors/events/analyses/face-embeddings/devices.

## Out of scope (later)

- Multi-replica horizontal scaling — the subscriber-bot, presence cache, and Pi relay registry are all single-process. Horizontal scale needs a Redis claim or Postgres advisory lock per camera.
- Pi push audio (`POST /v1/cameras/me/audio`).
- Per-detector threshold tuning (uniform 0.70 base today).
- Daily summary email job.
- Sensitive-content gate on the Live view.
- Admin pause/resume endpoint for the worker.

## Sister repos

- [`nemara-camera-app`](../../../nemara-camera-app) — iPhone wall-phone publisher (NemaraCam)
- [`nemara-caregiver-app`](../../../nemara-caregiver-app) — caregiver iOS app
- [`nemara-web-app`](../../../nemara-web-app) — caregiver web dashboard
- [`nemara-gateway`](../../../nemara-gateway) — edge appliance: RTSP cams → LiveKit `cam_*` participants
