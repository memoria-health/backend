# nemara-backend

FastAPI + Supabase backend for the **Nemara caregiver iOS app**.

This is the v0 caregiver-app-facing skeleton: identity, patients, cameras, detectors, events — everything that flows between iOS and the backend. Wall-phone ingestion, LiveKit token endpoints, APNs push, and photo uploads are deliberately out of scope and will land in follow-on plans.

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
│   ├── db/                  # SQLAlchemy ORM models
│   ├── repos/               # pure async queries; no business logic
│   ├── schemas/             # pydantic v2 request/response models
│   ├── services/            # business logic, owns transactions
│   └── utils/
├── supabase/
│   ├── config.toml          # local Supabase config
│   ├── schema.sql           # single-file schema (edit-in-place; no migrations)
│   └── seed.sql             # demo data
├── scripts/
│   └── seed.py              # programmatic seed (calls services, not raw SQL)
└── test/
    ├── unit/
    └── integration/         # FastAPI TestClient against the local Supabase
```

## Conventions

- **Single-file schema** — `supabase/schema.sql`. Edit in place, run `make db-reset`. No migrations until we're past MVP.
- **Repository pattern** — `repos/*` hold pure SQLAlchemy queries. Services own transactions and call `session.commit()`.
- **`Bearer <supabase_jwt>` on every protected route.** `verify_supabase_jwt` dispatches on the token's `alg` header — ES256 via JWKS in prod, HS256 with `NEMARA_SUPABASE_JWT_SECRET` for local dev.
- **RLS is on for every table.** Belt-and-braces against the FastAPI repo layer, **and** the filter Supabase Realtime applies when iOS subscribes to table changes.
- **Snake_case JSON** on the wire (FastAPI / Pydantic default). iOS Codable models map to camelCase via explicit `CodingKeys`.

## Out of scope (later)

- LiveKit room management + token endpoints.
- Wall-phone pairing + device JWT + event ingest.
- APNs push registration + fan-out.
- Photo / clip upload via Supabase Storage.
- Custom-detector LLM prompt compilation.
- Daily summary email job.

## Sister repos

- [`nemara-caregiver-app`](../nemara-caregiver-app) — the iOS client this backend serves.
- [`elephantLabs-wolf`](../elephantLabs-wolf) — the architectural source for this scaffold (auth, settings, repo pattern).
