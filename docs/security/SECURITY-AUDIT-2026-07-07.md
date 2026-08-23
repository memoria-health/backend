# Security Audit — nemara-backend

**Date:** 2026-07-07
**Auditor:** Automated static security review (vibe-security methodology)
**Commit audited:** `2f46308` (HEAD, detached)
**Scope:** Static source analysis only. No live systems, databases, or secrets were accessed. Findings are drawn from code, SQL migrations, configs, and git history.

---

## Executive Summary

nemara-backend is a FastAPI service for an eldercare camera-monitoring product (patients, caregivers, Raspberry-Pi / iPhone / gateway cameras, LiveKit video, Supabase auth + storage, and a VLM detection pipeline). The application-layer authorization is generally **well built**: every caregiver-facing endpoint requires a verified Supabase JWT, repositories consistently scope queries by `caregiver_id`, cross-account access returns 404, LiveKit token minting verifies room ownership, JWT verification uses `jwt.verify()` (not `decode`) with audience checks, LiveKit webhooks verify signatures, and storage object keys are server-derived (no path traversal). Mass-assignment on `PATCH /v1/me` is correctly allowlisted at the schema layer.

However, the audit found **two Critical and one High** issues that undermine that otherwise-solid perimeter, plus several Medium/Low issues:

1. **Live production/development secrets are committed to git history** — including a Supabase `service_role` key, the JWT signing secret, the LiveKit API secret, Gemini/OpenAI/Resend API keys, and an **APNs EC private key**. Deleting the file (which was done) does not remove them from history.
2. **Two `public` tables ship without Row-Level Security** — `pi_discoveries` and `gateways` — so anyone holding the public Supabase anon/publishable key can read/write them directly via PostgREST. `pi_discoveries` exposes **plaintext one-shot Pi ingest tokens** and the MJPEG stream URLs of cameras inside customers' homes.
3. **`POST /v1/gateway/provision-camera` trusts a client-asserted email with a self-generated token** — anyone who knows a caregiver's email can bind their own gateway token to that caregiver's account, create cameras under all their patients, obtain valid ingest tokens, inject fake video into the patient's live room, and enumerate the victim's camera roster.

For a product whose entire purpose is trustworthy remote observation of vulnerable people, #2 and #3 are especially serious because they allow an attacker to **inject or observe camera feeds**, not merely read tabular data.

The remediation priorities, in order: **rotate every secret in git history and purge the history**, **enable RLS on `pi_discoveries` and `gateways`**, and **replace the email-only gateway provisioning trust model**.

### Findings by severity

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | Critical | Live secrets (incl. service_role key + APNs private key) committed to git history | git history (`.env`, removed in `cf36938`) |
| 2 | Critical | `pi_discoveries` and `gateways` tables have no RLS; anon key can read plaintext ingest tokens & stream URLs | `supabase/schema.sql`, `scripts/deltas/apply_pi_schema_delta.sql`, `scripts/deltas/apply_gateway_delta.sql` |
| 3 | High | Email-only gateway provisioning — self-asserted identity, no proof of email ownership | `src/api/cameras_ingest.py:460` |
| 4 | Medium | `caregivers` RLS `FOR ALL` lets a user write privileged columns via PostgREST, bypassing the API allowlist | `supabase/schema.sql:705` |
| 5 | Medium | No rate limiting anywhere; unauthenticated endpoints + linear-bcrypt camera auth enable CPU/abuse DoS | `src/api/deps.py:111`, `src/api/pairing.py`, `src/api/pi_discovery.py` |
| 6 | Medium | No application-level per-user AI/LLM spend caps | `src/services/detection/*`, `src/settings.py` |
| 7 | Low | Unauthenticated `pi-announce` hands the ingest token to whoever announces a known hardware_id first | `src/api/pi_discovery.py:43` |
| 8 | Low | Interactive API docs / OpenAPI schema served in every environment | `src/main.py:166` |

---

## Critical

### 1. Live secrets committed to git history (service_role key, JWT secret, APNs private key, LLM & LiveKit keys)

**Location:** git history — `.env`, tracked until commit `cf36938` ("chore(security): stop tracking .env"). Recoverable via `git show cf36938~1:.env`.

**What was exposed.** The removed `.env` (generated 2026-05-23) contains real values for:

- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` — **bypasses all RLS on the entire database**
- `NEMARA_SUPABASE_JWT_SECRET` — the HS256 signing secret; whoever holds it can **forge caregiver JWTs** for any account
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` — mint arbitrary LiveKit tokens (join/publish/subscribe any room)
- `NEMARA_GEMINI_API_KEY`, `NEMARA_OPENAI_API_KEY`, `NEMARA_RESEND_API_KEY` — billable third-party keys
- `NEMARA_APNS_AUTH_KEY` — a full **EC private key** (`-----BEGIN PRIVATE KEY----- MIGTAgEAMBMGByqGSM49...`) plus `APNS_TEAM_ID` / `APNS_KEY_ID`, enabling forged push notifications to the app's users

**Why it is exploitable.** Removing a file from the working tree does not remove it from history. Anyone with clone access (the repo, a fork, a CI cache, a backup, or a laptop that ever pulled it) can recover every one of these secrets. The JWT secret and service_role key together are a complete authentication + database bypass: forge a caregiver token *and* read/write any row regardless of RLS.

The `.gitignore` comment claims "its leaked secrets rotated," but the audit cannot verify rotation from source, and the material (especially the APNs private key and any keys still matching production) remains retrievable from history. **Treat every value above as compromised.**

**Remediation.**
1. **Rotate all of them now**, regardless of prior rotation claims: Supabase service_role + JWT secret (Supabase dashboard → API → roll keys), LiveKit key/secret, Gemini/OpenAI/Resend keys, and generate a **new APNs key** (revoke the old Key ID in the Apple Developer portal).
2. **Purge history** with `git filter-repo --path .env --invert-paths` (or BFG), force-push, and have all collaborators re-clone. Note this only limits future exposure — rotation is the load-bearing fix.
3. Add a `gitleaks`/`trufflehog` pre-commit hook and CI gate so a secret can never be committed again.
4. Keep prod secrets only in Railway's environment config (already the stated model) and dev secrets in a secret manager, never in a tracked file.

---

### 2. `pi_discoveries` and `gateways` tables have no Row-Level Security — anon key can read plaintext ingest tokens and home camera stream URLs

**Location:**
- `supabase/schema.sql:334` (table `pi_discoveries`) — no `enable row level security`, no policy anywhere in `schema.sql` or `scripts/deltas/apply_pi_schema_delta.sql`.
- `scripts/deltas/apply_gateway_delta.sql:38` (table `gateways`) — no RLS in the delta that creates it.
- Contrast: `supabase/schema.sql:696-786` enables RLS on all 11 other public tables, and the header comment even asserts *"Every table is RLS-on."* These two are the exceptions.

**Why it is exploitable.** In Supabase, every table in the `public` schema is exposed through PostgREST and reachable with the **anon/publishable key**, which is public by design (it ships in the iOS app and web client). RLS is the only gate. A table with RLS disabled is fully readable *and writable* by anyone with that key.

`pi_discoveries` (schema `:334-369`) stores:
- `pending_ingest_token` — a **plaintext, one-shot bearer token** a Pi uses to authenticate its frame-push ingest (`/v1/cameras/me/frames`). Reading it lets an attacker push arbitrary video into a patient's LiveKit room as that camera.
- `current_url` / `current_audio_url` — the live MJPEG/audio endpoints of cameras **inside customers' homes**, plus `hardware_id` (MAC) and `device_label`.

An attacker with the anon key can therefore:
```
GET  {supabase}/rest/v1/pi_discoveries?select=*      # dump every home's stream URLs + pending ingest tokens
POST {supabase}/rest/v1/pi_discoveries               # inject fake discoveries
DELETE ...                                           # wipe the discovery table
```

`gateways` (`apply_gateway_delta.sql:38`) stores `token_hash` + `caregiver_id`. With no RLS, an attacker can `INSERT` a row binding a **token hash they control** to any `caregiver_id`, then use that token against `/v1/gateway/roster` and `/v1/gateway/wait` — the anon-key equivalent of Finding #3, and can also `DELETE` legitimate gateways to break real camera fleets.

**Remediation.** Enable RLS on both tables and add restrictive policies (or, since both are written only by the trusted backend via a direct Postgres connection, deny all PostgREST/anon access):

```sql
-- pi_discoveries: backend-only; no caregiver/anon access via PostgREST.
alter table public.pi_discoveries enable row level security;
-- (no policy for anon/authenticated = deny-all; the backend uses a direct
--  DB connection as the table owner and is unaffected by RLS.)

-- gateways: same — backend-only.
alter table public.gateways enable row level security;
```
Additionally, move `pending_ingest_token` out of a client-exposed table entirely (a `private` schema table, unreachable via PostgREST) as defense-in-depth, and add the "enable RLS on every public table" loop from the vibe-security database-security reference to the migration so future tables can't ship ungated.

---

## High

### 3. Email-only gateway provisioning — self-asserted identity binds an attacker's token to any caregiver

**Location:** `src/api/cameras_ingest.py:460` — `POST /v1/gateway/provision-camera`.

**What the code does.** The endpoint authenticates with `Authorization: Bearer <box_token>`, where `box_token` is (per the docstring) *"generated on first boot"* by the device itself — i.e. an arbitrary, attacker-choosable string, not a secret issued or verified by the server. The body carries a plaintext `email`. The handler then (`:488-515`):
1. Looks up the caregiver **by `body.email`**,
2. Binds `hash_ingest_token(box_token)` → that caregiver in `public.gateways`,
3. Creates a `source_type='gateway'` camera under **every one of that caregiver's patients** (`:528-564`), and
4. Returns valid `ingest_token`s for each new camera (`:573`).

**Why it is exploitable.** The only thing linking the caller to the caregiver is the email in the request body. Email addresses are **not secrets** — they are shared, guessable, and often public. There is no password, no proof of email control, no invite/claim code, and no shared secret validated against anything the server issued. An attacker who knows or guesses a caregiver's email can:
- Bind a gateway token they control to the victim's account, then poll `/v1/gateway/roster` and `/v1/gateway/wait` to **enumerate the victim's entire camera topology** (`livekit_identity`, `livekit_room`, `desired_state`).
- Obtain valid ingest tokens and **push attacker-controlled video/audio into the patient's live LiveKit room** (`patient_<id>`), which the caregiver's app renders as a real in-home camera. In a fall/agitation-monitoring product, injecting a benign-looking feed while a real emergency occurs is a safety-critical outcome.
- Spam-create cameras across all the victim's patients (also amplifying LLM/detection cost — see Finding #6).

**Remediation.** Provisioning must require proof that the caller is authorized for the target account. Options, best first:
- Require an **authenticated caregiver** to mint a short-lived, single-use **claim code** in-app; the box submits that code (not an email) to bind itself. The code is the proof of authorization, exactly like the existing camera `pairing_code` flow (`src/api/pairing.py`).
- Alternatively, pre-register each box's token server-side (issued by the backend, not self-generated) and have the caregiver associate a specific pre-registered `box_id` from within their authenticated session.
- In all cases, reject binding to an account the caller hasn't proven control of, and rate-limit + audit-log this endpoint.

---

## Medium

### 4. `caregivers` RLS uses `FOR ALL`, letting a user write privileged columns via PostgREST and bypass the API's field allowlist

**Location:** `supabase/schema.sql:705-707`
```sql
alter table public.caregivers enable row level security;
create policy caregivers_self on public.caregivers
  for all using (id = auth.uid()) with check (id = auth.uid());
```

**Why it is exploitable.** The API layer correctly restricts `PATCH /v1/me` to `display_name` and `phone` only (`src/services/onboarding.py`, `UpdateCaregiverRequest`). But the `FOR ALL` RLS policy also grants **direct `UPDATE` via PostgREST** using the anon key + the caregiver's own JWT — bypassing the API entirely. A caregiver can therefore write any column on their own row, including:
- `vlm_provider` (`schema.sql:99`) — routes their account's detection to a chosen paid model, read on the detection hot path. A user can silently move themselves onto the most expensive provider (cost abuse) or set an invalid value that breaks their own detection.
- `show_live_analyses`, `timezone`, `onboarded_web` — product-state flags the app assumes only the server sets.

This is the "sensitive fields on a user-updatable table" pattern the vibe-security database-security reference calls out. There are no `is_admin`/`credits`/`subscription_tier` columns today, so blast radius is limited to the user's own account — hence Medium, not High — but the gap will silently widen the moment a privileged column is added to this table.

**Remediation.** Restrict which columns the `authenticated` role may update, so RLS can't be used to write server-owned fields:
```sql
revoke update on public.caregivers from authenticated;
grant  update (display_name, phone) on public.caregivers to authenticated;
```
Keep server-owned columns (`vlm_provider`, `show_live_analyses`, `onboarded_web`, `timezone`, `email`) writable only by the backend's direct DB connection.

### 5. No rate limiting anywhere; unauthenticated endpoints + linear-bcrypt camera auth enable CPU / abuse DoS

**Location:** entire `src/` (no `slowapi`/`ratelimit`/`upstash`/limiter dependency or middleware exists); notably `src/api/deps.py:111` (`camera_auth`), `src/api/pairing.py:40`, `src/api/pi_discovery.py:43`, `src/api/cameras_ingest.py`.

**Why it is exploitable.**
- **Algorithmic-complexity DoS in `camera_auth` (`deps.py:137-156`).** Camera api-key auth loads *every* camera row with a non-null `api_key_hash` and runs `bcrypt.checkpw` (cost factor 10, ~50–100 ms each) against them until a match. An unauthenticated attacker sending a bogus bearer to any `CameraAuth` endpoint (e.g. `GET /v1/cameras/me/desired-state`, polled every 1 s) forces a **full N-row bcrypt scan per request**. Cost grows linearly with the customer fleet; a modest request flood saturates the executor pool and the CPU. The code comment acknowledges the linear scan but treats it as a perf note, not a DoS vector.
- **Unauthenticated abuse surface.** `POST /v1/cameras/pair`, `POST /v1/cameras/pi-announce`, and the gateway endpoints have no throttle. Pairing codes are 6 digits (`schemas/pairing.py`) — brute-forceable in ~10⁶ attempts with no lockout — and `pi-announce` can be flooded to churn `pi_discoveries`.

**Remediation.** Add per-IP **and** per-principal rate limiting (e.g. `slowapi` or an edge/API-gateway limiter) to all unauthenticated and auth endpoints. For `camera_auth`, add an indexed `api_key_prefix` lookup (the code's own "Phase 4" TODO) so verification is O(1) instead of O(N) bcrypts, and cap pairing-code attempts per camera/IP with exponential backoff.

### 6. No application-level per-user AI/LLM spend caps

**Location:** `src/services/detection/` (Gemini/OpenAI/OpenRouter/DashScope providers), `src/settings.py` (provider keys and model config; no per-user quota setting).

**Why it is exploitable.** The detection pipeline calls paid VLM providers per camera per tick. There is no per-user/per-tier usage accounting or hard cap in the application — cost control relies solely on the providers' own caps, which the vibe-security ai-integration reference warns against (they lag and are global, not per-tenant). Combined with Finding #3 (attacker-created cameras) or Finding #4 (self-routing to the most expensive provider), an attacker can drive detection volume/cost with no in-app ceiling.

**Remediation.** Track token/`call` usage per caregiver in the DB, enforce daily/monthly hard caps per account or tier, and set hard spending caps + billing alerts in each provider console. Return a clear error when a user exceeds their quota rather than continuing to bill.

---

## Low

### 7. Unauthenticated `pi-announce` returns the ingest token to whoever announces a known hardware_id first

**Location:** `src/api/pi_discovery.py:43-90`.

`POST /v1/cameras/pi-announce` is intentionally unauthenticated (a Pi has no credentials pre-pairing). On announce it `consume_pending_ingest_token`s and returns the plaintext token to the caller. Because `hardware_id` is a MAC address (partly leaked in `device_label`, and fully readable from `pi_discoveries` per Finding #2), an attacker who announces a provisioned device's `hardware_id` **before the real Pi does** receives that device's one-shot ingest token, then pushes frames as that camera. The module docstring already flags this ("v2 will add a per-device secret"). Severity is Low on its own (requires a timing race and a known hardware_id) but rises sharply when combined with #2, which removes the "known hardware_id" barrier.

**Remediation.** Introduce the planned per-device secret negotiated at physical setup so announcements are attributable, and never return a bearer token to an unauthenticated caller identified only by a guessable hardware_id. Until then, at minimum enable RLS on `pi_discoveries` (Finding #2) so hardware_ids/tokens aren't directly dumpable.

### 8. Interactive API docs and OpenAPI schema served in every environment

**Location:** `src/main.py:166-171` — `FastAPI(...)` is created without `docs_url=None` / `redoc_url=None` / `openapi_url=None`, so `/docs`, `/redoc`, and `/openapi.json` are publicly served in production. The access-log filter at `:56-65` even special-cases these paths, confirming they're live.

**Why it matters.** This publishes the full API surface (every route, schema, and internal field name) to anonymous users, easing reconnaissance. Low impact on its own for an authenticated API, but unnecessary attack-surface disclosure.

**Remediation.** Disable the docs/schema endpoints in production (gate on `settings.env == "local"`), e.g. `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` when not local.

---

## What was checked and found sound

To keep the report honest about scope, these areas were reviewed and did **not** yield findings:

- **JWT verification** (`src/services/auth.py`): uses `pyjwt.decode` with a signing key and `algorithms=["ES256","HS256"]`, enforces `audience="authenticated"`, validates `sub`/`email`. No `alg=none` acceptance, no unsigned `decode` trust. Failures are flattened to an opaque 401 (`deps.py:39`).
- **API-layer authorization / IDOR**: every caregiver endpoint uses `CaregiverAuth`; repos (`patient_repo`, `event_repo`, etc.) scope by `caregiver_id`; cross-account access returns 404. Event clip-url and patient photo-url verify ownership before signing (`events.py:144`, `patients.py`).
- **LiveKit token minting** (`src/api/livekit.py`): camera tokens are constrained to the camera's own room/identity; caregiver tokens verify patient ownership by room name; identities are force-prefixed to survive privacy cuts.
- **LiveKit webhooks** (`src/api/livekit_webhooks.py`): signature verified via `WebhookReceiver`/`TokenVerifier` against the raw body before any action.
- **SQL injection**: raw `text()` queries in `cameras_ingest.py` use bound parameters throughout; no string-concatenated user input. ORM usage is parameterized.
- **Storage path traversal** (`src/services/storage.py`): object keys are built from server-side UUIDs (`{patient_id}.{ext}`, `events/{event_id}.mp4`), not client filenames; content-type allowlisted; size-capped.
- **Mass assignment**: `PATCH /v1/me` and patient updates go through Pydantic schemas with explicit writable fields (the RLS gap in Finding #4 is the exception, and is a database-layer issue).
- **CORS**: no `CORSMiddleware` / wildcard origin is configured (native app + server-to-server), so there is no permissive-CORS exposure.
- **Email service** (`src/services/email.py`): fixed Resend endpoint (no SSRF), recipients from server config, `reply_to` from the authenticated caregiver's stored email.
- **Dependencies** (`pyproject.toml`): pins are current (fastapi ≥0.115, pyjwt[crypto] ≥2.10, sqlalchemy ≥2.0.36, etc.); no obviously known-vulnerable versions. Recommend adding `pip-audit`/`uv`-based dependency scanning to CI to keep this true over time (the `uv.lock` was not exhaustively CVE-checked in this static pass).

---

## Prioritized remediation checklist

1. **(Critical #1)** Rotate every secret recoverable from git history — service_role key, JWT secret, LiveKit secret, Gemini/OpenAI/Resend keys, and the APNs key — then purge history and add a secret-scanning gate.
2. **(Critical #2)** Enable RLS on `pi_discoveries` and `gateways`; move `pending_ingest_token` to a non-PostgREST-exposed schema. Add the "RLS on every public table" migration guard.
3. **(High #3)** Replace email-only gateway provisioning with an authenticated caregiver-minted claim code (or server-issued box tokens).
4. **(Medium #4)** Convert the `caregivers` RLS to column-scoped `GRANT UPDATE (display_name, phone)`.
5. **(Medium #5)** Add per-IP + per-user rate limiting; make `camera_auth` O(1) via an api-key prefix index; cap pairing-code attempts.
6. **(Medium #6)** Enforce per-user LLM usage quotas in-app + hard provider spend caps.
7. **(Low #7, #8)** Add per-device secrets to `pi-announce`; disable `/docs`,`/redoc`,`/openapi.json` in production.
