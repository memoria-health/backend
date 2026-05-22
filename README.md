# nemara-backend

FastAPI + Supabase backend for the **Nemara caregiver iOS app** and **wall-phone clip ingest**.

Identity, patients, cameras, detectors, events, plus the Phase 0 detector pipeline: 6-digit camera pairing, per-camera api-keys, LiveKit token minting, and clip ingest (`POST /v1/clips/ingest`) with the Gemini → OpenAI provider fallback ported from `nemara-temp-dashboard`. The detector worker that subscribes to LiveKit egress (Phase 2) and APNs push (Phase 4) are out of scope here.

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
│   │   ├── clips.py         #   /v1/clips/ingest  (Phase 0)
│   │   └── livekit.py       #   /v1/livekit/token (Phase 0)
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
│   │       ├── frame_extraction.py
│   │       ├── state_machine.py # open↔closed per-camera state
│   │       ├── analyses_store.py # INSERT into analyses
│   │       └── event_promotion.py # analysis → events row promotion
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

## Detector pipeline (Phase 0)

The dashboard's detector pipeline lives here, in Python, behind
`POST /v1/clips/ingest`:

1. **Auth:** wall phone presents `Bearer <api_key>` (bcrypt-hashed on
   the camera row; minted via `POST /v1/cameras/pair` after a
   caregiver-generated 6-digit code).
2. **Detector load:** `services/detection/registry.py` picks up that
   camera's enabled detectors (the `people_present` always-on block
   is implicit — see `prompt.py`).
3. **Provider call:** `gemini.py` first (MP4 inline, env-tunable fps),
   falls back to `openai.py` (ffmpeg-extracted frames @ 5fps × 480px).
   `_parse.py` validates the JSON output and clamps `*_confidence` to
   `[0, 1]`.
4. **State machine:** `state_machine.py` flips `cameras.state` between
   `open` and `closed` based on `cameras.close_threshold` consecutive
   ticks with no people + no motion. Closed cameras short-circuit the
   LLM call.
5. **Analyses row:** every tick (or skipped tick) lands in `analyses`
   for Studio / detector-worker (Phase 2) consumption.
6. **Event promotion:** `event_promotion.py` watches for
   `<id>_detected: true` (+ confidence gate for built-ins) and inserts
   a row into `events`; Realtime broadcasts to the iOS caregiver app.
   30-second dedup per `(patient, detector)`.

Phase 2 will swap the HTTP-clip transport for a long-running
LiveKit-subscribing detector worker — same modules, different input
loop.

## Out of scope (later)

- **Phase 2:** Detector worker as a LiveKit subscriber-bot (kills the
  HTTP-clip transport).
- **Phase 3:** Caregiver Talk mode (full-duplex audio routed to the
  patient's active camera).
- **Phase 4:** LiveKit Track Egress → S3 event clips. APNs push.
  Webhook receiver (`/v1/livekit/webhooks`) for online/offline state.
- Photo upload via Supabase Storage.
- Custom-detector LLM prompt compilation (caregiver's free-text →
  runtime config).
- Daily summary email job.

## Sister repos

- [`nemara-caregiver-app`](../nemara-caregiver-app) — the iOS client this backend serves.
- [`elephantLabs-wolf`](../elephantLabs-wolf) — the architectural source for this scaffold (auth, settings, repo pattern).
