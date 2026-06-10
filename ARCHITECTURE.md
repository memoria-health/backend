# nemara-backend · architecture

How the backend is laid out, what conventions to follow when adding to it, and how the major pipelines wire together.

## File layout

```
nemara-backend/
├── pyproject.toml                # uv + ruff + pytest config; py >=3.12,<3.14 (.python-version pins 3.13)
├── Makefile                      # install, dev, test[-unit|-integration], lint, format, db-*, e2e, fresh[-seed], down
├── Dockerfile                    # CPU torch; YOLO + InsightFace weights pre-cached
├── yolo11n.pt / yolo11s.pt       # YOLO11 person-detector weights (re-ID; 11s is the default)
├── scripts/
│   ├── local-start.sh            # one-command stack-up (Supabase + buckets + uvicorn :8002 + ngrok)
│   ├── seed.py                   # programmatic demo seed (calls services, not raw SQL)
│   ├── seed_demo_camera.py       # App Review demo camera (persistent reusable pairing code)
│   ├── reseed_*_prompt.py        # refresh a preset detector's prompt in a live DB
│   ├── apply_*_delta.sql         # idempotent prod schema patches (see Conventions)
│   ├── e2e/cli.py                # `uv run e2e <command>` — fire-event, add-patient, flip-detector…
│   └── pi/                       # Pi device-side: install.sh, captive portal, announce daemon, MJPEG pusher
├── supabase/
│   ├── config.toml               # local Supabase config
│   ├── schema.sql                # SINGLE-FILE canonical schema (local); prod patched via delta scripts
│   └── seed.sql                  # empty; Python seeder runs in scripts/seed.py
├── tools/
│   ├── lk_room_audit.py          # LiveKit room snapshot; flags zombie/idle detector workers
│   └── provision_pi.sh           # SSH into a Pi and run scripts/pi/install.sh
├── src/
│   ├── main.py                   # app factory; lifespan owns the 3 background runtimes
│   ├── settings.py               # pydantic-settings, NEMARA_* prefix
│   ├── errors.py                 # exception hierarchy
│   ├── api/                      # FastAPI routers (one file per aggregate)
│   │   ├── deps.py               #   CaregiverAuth (JWT) + CameraAuth (api-key) deps
│   │   ├── health.py             #   /health (liveness) + /ready (DB reachability)
│   │   ├── caregivers.py         #   /v1/me CRUD + DELETE (full account deletion)
│   │   ├── patients.py           #   /v1/patients… + photo upload/url + test-email
│   │   ├── detectors.py          #   detector CRUD + /detectors/library + /detectors/infer
│   │   ├── events.py             #   list/get/review + /v1/events/{id}/clip-url
│   │   ├── cameras.py            #   cameras CRUD, pairing-code, pi-provision, /v1/cameras/me/* (wall phone)
│   │   ├── cameras_ingest.py     #   Pi push: /v1/cameras/me/frames + /ingest-status
│   │   ├── pairing.py            #   /v1/cameras/pair (6-digit code → api key)
│   │   ├── livekit.py            #   /v1/livekit/token (camera api-key OR caregiver JWT)
│   │   ├── livekit_webhooks.py   #   participant joined/left + room_finished → worker dispatch
│   │   ├── pi_discovery.py       #   /v1/cameras/pi-announce (unauthenticated) + /discovered
│   │   ├── face_embeddings.py    #   re-ID selfie enrollment + status/summary
│   │   ├── talk.py               #   /active-camera + /talk/{start,end}
│   │   ├── devices.py            #   /v1/me/devices (APNs token registration)
│   │   ├── support.py            #   Contact Support → Resend to founders
│   │   └── feedback.py           #   product feedback: persist row + best-effort email
│   ├── db/                       # SQLAlchemy ORM models (one per table; 11 tables)
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
│   │   ├── push.py               # APNs dispatch (HTTP/2, ES256 .p8)
│   │   ├── email.py              # Resend: event alerts + support + feedback
│   │   ├── clip_archival.py      # in-process pre+post event clip → annotate → concat → upload
│   │   ├── clip_annotate.py      # burn green patient bbox onto archived clips (best-effort)
│   │   ├── scheduling/scheduler.py  # ScheduledDetectorRunner — "absence" detectors
│   │   ├── detection/            # realtime detector pipeline (see section below)
│   │   ├── reid/                 # patient re-identification (see section below)
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
- **Heavy ML deps are real.** torch (CPU), ultralytics (YOLO11), boxmot (BoT-SORT + OSNet), insightface + onnxruntime (SCRFD/ArcFace) all ship in the image for re-ID; weights are pre-cached in the Dockerfile.

## Auth paths

All presented as `Authorization: Bearer …`; four distinct verifiers:

1. **Caregiver JWT** (Supabase) on every iOS-facing route. `verify_supabase_jwt` dispatches on the token's `alg` header — ES256 via JWKS in prod, HS256 with `NEMARA_SUPABASE_JWT_SECRET` for local dev. Exposed via `CaregiverAuth` dep in `api/deps.py`.
2. **Camera api-key** on wall-phone routes (`/v1/cameras/me/*`, `/v1/livekit/token` camera role). bcrypt-hashed on the `cameras` row, verified by linear scan over active cameras (O(N), small N). A well-formed key with no match returns **404** (not 401) so the wall phone knows its row was deleted and resets to pairing. Exposed via `CameraAuth`.
3. **Pi ingest token** on `/v1/cameras/me/frames` + `/ingest-status`. High-entropy `token_urlsafe(32)`, stored as an **unsalted SHA-256** digest (`cameras.ingest_token_hash`) so auth is an indexed equality lookup instead of a bcrypt scan. Delivered exactly once via the pi-announce response.
4. **LiveKit webhook signature** on `/v1/livekit/webhooks` — JWT verified against the LiveKit API key/secret.

`POST /v1/cameras/pair` and `POST /v1/cameras/pi-announce` are deliberately unauthenticated (pairing-code possession / physical-proximity provisioning are the trust anchors).

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
   - **Re-ID identity gate:** if the patient is enrolled, re-ID ran, no track locked as PATIENT, and there is positive not-the-patient evidence (≥1 face was checked, or every body track locked as OTHER) → skip LLM entirely (saves the provider call when e.g. a helper is in frame).
   - Otherwise: snapshot buffers → MP4 via `mp4_remux.py` (ffmpeg pipe) → **Gemini first** (`gemini_model`, default `gemini-3.5-flash`, native video + audio), fall back to **OpenAI** (`openai_model`, default `gpt-5.4`, 4fps JPEG frames via `frame_extraction.py`; patient bbox burned in by `frame_annotate.py` when re-ID located them). `_parse.py` validates the JSON and clamps `*_confidence` to `[0, 1]`.
6. **Presence verdict:** the `people_present` fed to the state machine is `LLM OR re-ID-patient-match` when enrollment exists; LLM-only otherwise. `state_machine.py` flips `cameras.state` open↔closed atomically (single row-locked UPDATE; `close_threshold` consecutive no-people ticks to close).
7. **Event promotion** (`event_promotion.py`): a detector fires when `<id>_detected == true`, `<id>_confidence >= threshold`, and no event for the same `(patient, detector)` exists in the last 30s. Thresholds: base 0.70 for every catalog detector, shifted by the caregiver's per-detector sensitivity (low +0.15, standard +0, high −0.30; clamped to [0.05, 0.95]).
8. **Side-effects** (fire-and-forget after commit): APNs push to all registered caregiver devices, email alerts (if `patient.email_alerts_enabled`), and clip archival kickoff.

Every tick writes an `analyses` row (~10s cadence; debugging surface, not user-facing).

## Camera liveness: `online` vs `app_present`

Two independent signals with different owners:

- **`cameras.online`** — "is video actually flowing right now?" Owned by the room reconciler (steady state) + webhooks (authoritative immediate offline on leave / room_finished) + a startup sweep.
- **`cameras.app_present`** — "is the Memoria Sensor app open on the iPhone?" (Pi rows default true.) The wall phone's 1s desired-state poll touches the in-memory `PresenceCache` (no DB write on the hot path); `PresenceSweeper` flips the column only on transitions. **Single replica only.**

UI mapping: `online` → LIVE; `!online && app_present` → reconnectable "Disconnected"; neither → "App closed".

## Detection library & custom detectors

- **`detection/catalog.py` is the single source of truth** for every built-in detector: 22 entries (6 seeded presets + 16 library add-ons), each a frozen `CatalogEntry` carrying key, prompt fragment, category (`safety`/`routine`), shape (`point_in_time`/`state`/`scheduled`), cohort (`child`/`adult`/`both`), and promotion threshold. Consumers: onboarding seeding, the `/detectors/library` endpoints, and event promotion's thresholds.
- **Age-based seeding** (`onboarding.py`): minors get Falls + Climbing + Distress; adults get Falls + Agitation + Wandering + Medication.
- **Custom detectors** (`detector_compiler.py`): `POST /v1/detectors/infer` classifies free text into shape/category/sensitivity/schedule synchronously; `POST /v1/patients/{id}/detectors` returns 201 immediately and compiles in a background task — the LLM (OpenAI `openai_compile_model`, default `gpt-5.4`; Gemini fallback) writes the prompt fragment, while the `custom_<8hex>` identifier and JSON schema are templated server-side. Inclusivity is tuned by stakes: reactive-safety detectors compile inclusive (missed emergency > false alarm); routine and scheduled detectors compile conservative.

## Scheduled ("absence") detectors

`scheduling/scheduler.py` runs `ScheduledDetectorRunner`, independent of the webhook-driven worker. Detectors with `shape='scheduled'` describe a recurring window in `detectors.schedule_config` (JSONB: cadence, days, window, deadline). Each tick (60s), for any detector whose deadline has passed today and isn't yet evaluated (`last_scheduled_eval_date` single-fire guard), it converts the caregiver-timezone window to UTC, queries `analyses` for the watched signal, and — if the signal was **never seen** — fires a patient-level absence event (`camera_id=NULL`, narrative includes the window and camera-uptime %). Reuses the same push + email dispatchers. The reactive path skips `scheduled`-shape detectors entirely; the same prompt fragment runs in every tick either way, the scheduler only inverts and windows the firing.

## Patient re-identification (re-ID)

`src/services/reid/` answers "is the person in frame the *patient*, or someone else?" so detections aren't suppressed (or fired) on the wrong person.

- **Enrollment:** caregiver uploads 1–N selfies (`POST /v1/patients/{id}/face-embeddings`, 202 + background processing). `enrollment_processor.py` downloads from Supabase Storage, runs SCRFD face detection + ArcFace embedding (InsightFace), quality-gates on det score, and marks rows `ready`/`failed`. Embeddings (512-dim float32) live in Postgres `patient_face_embeddings`; only selfie photos persist in storage.
- **Inference per tick** (`inference.py`, bounded by `reid_timeout_seconds`): sample frames at ~5fps → YOLO11 person detection (`body_detector.py`) → BoT-SORT tracking with OSNet body embeddings (`tracker.py`) → periodic face checks per track (`embeddings.py`, quality-adaptive thresholds) → tracks lock as PATIENT / OTHER / UNKNOWN (`body_pipeline.py`). Output `IdentityResult` feeds the tick's identity gate and presence verdict, and shares its decoded frames with the OpenAI fallback path.
- **Cross-clip body gallery** (`cross_clip_body_gallery.py`): in-memory, clustered OSNet embeddings per patient (TTL hours) so a person identified in clip N is pre-labeled in clip N+1 even when their face is hidden.
- **Adaptive harvest** (`harvester.py`): high-confidence face matches insert `source='harvested'` embedding rows (FIFO-capped) to close the selfie→wall-cam domain gap.
- If re-ID can't run (no enrollment, timeout, crash), the pipeline degrades to LLM-only presence — re-ID failure never blocks detection.

## Pi camera ingest

`src/services/ingest/pi/` makes a Raspberry Pi camera look identical to an iPhone from everything downstream. The Pi cannot be a LiveKit participant (CPU + NAT), so the backend acts as its publisher proxy:

1. **Provisioning:** `tools/provision_pi.sh` → `scripts/pi/install.sh` sets up systemd units (`memoria-*` services, `/etc/memoria/` env files). Customer setup is a Wi-Fi captive portal (`Memoria-Setup-<MAC6>` AP at 192.168.4.1); once on home Wi-Fi, `announce_daemon.py` heartbeats `POST /v1/cameras/pi-announce` every 60s with its MAC-derived `hardware_id`.
2. **Pairing:** caregiver sees unclaimed Pis via `GET /v1/cameras/discovered` (5-min freshness window) and provisions via `POST /v1/patients/{id}/cameras/pi-provision` — creates the `cameras` row (`source_type='pi'`), mints the ingest token, and stashes the plaintext on `pi_discoveries.pending_ingest_token`. The next announce response delivers it **exactly once** (`SELECT … FOR UPDATE`, column nulled in the same transaction); the Pi writes `/etc/memoria/ingest.env` and starts streaming.
3. **Streaming:** the Pi holds one long-lived `POST /v1/cameras/me/frames` with `multipart/x-mixed-replace` MJPEG (outbound dial — home routers block inbound). `mjpeg_reader.py` parses the body, `jpeg_decoder.py` produces the Y-plane + I420 per frame, and `relay_publisher.py` republishes into LiveKit as a regular `cam_<…>` participant — plus the same `motion_state` heartbeat packets the iPhone sends, computed server-side by `motion_detector.py` (a port of the iOS SAD-tile algorithm with a Pi-specific threshold of 8.0 vs iOS 1.5, tuned for MJPEG re-compression noise). Most-recent-wins: a new connection for the same camera cancels the old relay task.
4. **Pause:** caregiver disconnect flips `desired_state`; the push endpoint returns 423 LOCKED and `cancel_active_relay()` kills the in-flight relay. The Pi polls `GET /v1/cameras/me/ingest-status` (~5s) while paused.

Audio for the Pi path is not wired yet (`audio_reader.py` is pull-mode vestigial; push audio is a planned follow-up endpoint).

## Caregiver Talk mode

The iOS Live tab is a phone-call metaphor: the caregiver swipes a carousel of camera tiles and taps a call button. Audio routes to **the camera the caregiver picked** — the server-side best-guess only seeds the carousel's initial page.

- `GET /v1/patients/{id}/active-camera` — the camera with the freshest `analyses.people_present=true` row in the last 30s (`active_camera.py`), or null.
- `POST /v1/patients/{id}/talk/start` — body `{"camera_id": UUID}`. Sends `is_active=true` to that camera and `is_active=false` to every other; re-POSTing with a different camera transfers the call atomically.
- `POST /v1/patients/{id}/talk/end` — `is_active=false` to every paired camera. Idempotent.
- Packet shape (LiveKit `set_talk_active` topic): `{"type": "set_talk_active", "is_active": bool}`. Cameras gate local audio playback on the latest flag — the SFU never mutes tracks, so routing changes are O(ms).

## Clip archival & annotation

Fully in-process — **no LiveKit Egress** (it's documented in `clip_archival.py` as the future alternative if reliability demands it).

1. **Pre-event clip:** the ~11s MP4 the tick already built for the LLM is reused as-is.
2. **Post-event clip:** sleep `clip_post_event_wait_seconds` (15s), then snapshot the live `FrameBuffer` + `AudioBuffer` anchored at the pre-clip's end timestamp (so the LLM's think-time gap doesn't leak into the clip).
3. **Annotation:** `clip_annotate.py` re-runs the re-ID body pipeline over the clip and burns a green box around the patient (best-effort — any failure uploads the original clip).
4. **Concat + upload:** ffmpeg concat → `incident-clips/events/{event_id}.mp4` (Supabase Storage), then `events.clip_storage_key` + `has_clip=true`; Realtime notifies the app, and `GET /v1/events/{id}/clip-url` mints a short-lived signed URL for inline playback.

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

- [`nemara-camera-app`](../nemara-camera-app) — iPhone wall-phone publisher (NemaraCam)
- [`nemara-caregiver-app`](../nemara-caregiver-app) — caregiver iOS app
- [`nemara-web-app`](../nemara-web-app) — caregiver web dashboard
