# Security Audit — nemara-backend

**Date:** 2026-08-01
**Scope:** Static source audit of the `nemara-backend` FastAPI service (repo HEAD `a53ec6e`, branch `main`).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill framework — secrets/env, database access control (Supabase RLS), authentication/authorization, injection, rate limiting, payments, deployment, AI/LLM, mobile.
**Constraints:** Static only. No live systems, databases, or secrets were accessed. Findings are substantiated by source; speculative issues were excluded.

---

## Executive Summary

nemara-backend is a FastAPI backend for a patient-monitoring product (elderly-care camera/fall detection). It uses Supabase for auth (JWT) and storage, connects to Postgres **directly** (bypassing RLS), mints LiveKit tokens, ingests camera frames from iPhone/Pi/gateway devices, and runs VLM (Gemini/OpenAI/OpenRouter/DashScope) detectors over video.

The codebase shows real security maturity in several places: JWT verification is correct (`jwt.decode` with signature + audience checks, opaque 401s), LiveKit token minting enforces room/identity/membership, camera and Pi ingest tokens are hashed with constant-time compare, RLS is broadly enabled with a membership-based `has_patient_access()` helper, and write schemas use `extra="forbid"` (no mass-assignment). The API route layer's membership scoping via `accessible_patient_ids()` is applied consistently across the caregiver-facing endpoints.

However, the audit found **two Critical** and **two High** issues that undermine that baseline:

1. **Live third-party secrets are exposed in git history** (`.env.bak.1783536884`) — Gemini, OpenAI, LiveKit key+secret, Resend, OpenRouter API keys and an APNs signing key. Untracking the file did not remove the blob from history.
2. **The gateway-provisioning endpoint has no real authentication** — an attacker who knows a caregiver's email can bind an attacker-chosen credential to that account, fabricate cameras under every one of the victim's patients, and inject arbitrary video into the patient's live monitoring feed.
3. **Two `public` tables ship without RLS** (`pi_discoveries`, `gateways`) — reachable via the exposed Supabase publishable key; `pi_discoveries` leaks plaintext Pi ingest tokens.
4. **Deleting a patient is not owner-guarded** — any shared member (including a read-only `viewer`) can permanently destroy a patient and all cascaded cameras/events/clips.

There is also **no rate limiting anywhere** in the application (pairing-code brute force, auth, unbounded VLM spend) and the interactive API docs are exposed in all environments.

### Severity tally

| Severity | Count | Findings |
|----------|-------|----------|
| Critical | 2 | C1 secrets in git history; C2 unauthenticated gateway provisioning |
| High     | 2 | H1 tables without RLS; H2 patient delete not owner-guarded |
| Medium   | 2 | M1 no rate limiting (pairing brute force + AI cost); M2 API docs exposed in prod |
| Low      | 3 | L1 reusable demo pairing code; L2 missing security headers; L3 unvalidated email inputs / authz inconsistency |

### Top remediation priorities

1. **Rotate every third-party key** in `.env.bak.1783536884` now (C1), and purge the blob from history (BFG/`git filter-repo`).
2. **Require a real caregiver credential** (JWT) or a pre-registered box-token allowlist for gateway provisioning (C2).
3. **Enable RLS** on `pi_discoveries` and `gateways`; stop storing the Pi ingest token in plaintext (H1).
4. **Owner-guard patient deletion** with `access.is_owner()` (H2).

---

## Critical

### C1 — Live production/third-party secrets committed to git history

**File:** `.env.bak.1783536884` (removed from HEAD in commit `a53ec6e`, but **present in history** at commit `f1c0564` and reachable via `git show f1c0564:.env.bak.1783536884`).

The file was committed with the patient-sharing work (`f1c0564`) and later untracked (`a53ec6e`). Untracking does **not** remove the blob from git history — anyone with repo access (or a leaked clone) can recover every secret. The commit message itself acknowledges this and says to rotate.

Secrets exposed in the blob:

| Key | Value (prefix) | Status |
|-----|----------------|--------|
| `NEMARA_GEMINI_API_KEY` | `AIzaSyDW3quQpxcp288g…` | **Live — rotate** |
| `NEMARA_OPENAI_API_KEY` | `sk-proj-YbEoTJIgpWhubz4Ox…` | **Live — rotate** |
| `NEMARA_LIVEKIT_API_KEY` / `_SECRET` | `APITxi4QucycnEW` / `254B2sRXlByHCPe1…` | **Live — rotate** (mints room tokens) |
| `NEMARA_OPENROUTER_API_KEY` | `sk-or-v1-2e601e2b1fa70…` | **Live — rotate** |
| `NEMARA_RESEND_API_KEY` | `re_N3Fo9qHR_PVtmFSWz…` | **Live — rotate** |
| `NEMARA_APNS_AUTH_KEY` | `-----BEGIN PRIVATE KEY-----` (ES256 .p8) + `APNS_KEY_ID=P34L5354FB`, `APNS_TEAM_ID=2RM3SKJN35` | **Live — revoke key in Apple Developer** |
| `NEMARA_DEMO_PAIRING_CODE` | `424242` | Enables the demo-pair path (see L1) |
| `NEMARA_SUPABASE_JWT_SECRET` / `_SERVICE_ROLE_KEY` | `super-secret-jwt-token-…` / `eyJ…supabase-demo…` | Local Supabase **defaults** (lower risk), but confirm prod never reused them |

**Why it's exploitable:** Each live key grants third-party access billed to the project. The LiveKit API key+secret let an attacker mint arbitrary room-join/publish tokens for any `patient_<id>` room (bypassing the backend's membership checks entirely — the LiveKit server only validates the signature). The AI keys allow unbounded spend. The Resend key allows sending mail as the verified domain. The APNs `.p8` allows forging push notifications to the caregiver app.

**Remediation:**
- Rotate/revoke **all** keys above immediately (LiveKit, Gemini, OpenAI, OpenRouter, Resend, and revoke the APNs auth key + generate a new one).
- Purge the blob from history: `git filter-repo --path .env.bak.1783536884 --invert-paths` (or BFG), then force-push and have all collaborators re-clone.
- Confirm production Supabase JWT secret / service-role key are not the local-default values shown.
- `.gitignore` already covers `.env.bak.*` going forward — good. Add a pre-commit secret scan (`gitleaks`) to CI to prevent recurrence.

---

### C2 — Unauthenticated gateway provisioning → account takeover + live-feed injection

**File:** `src/api/cameras_ingest.py:472-589` (`POST /v1/gateway/provision-camera`). Router mounted unconditionally in `src/main.py:180`.

The endpoint's only credential is a bearer token the **caller itself generates** ("box's UNIQUE token (generated on first boot)"), which is never checked against any pre-existing registry. The victim account is selected purely by an attacker-supplied `body.email`:

```python
box_token = authorization[len("bearer "):].strip()          # attacker-chosen, unverified
cg = (await session.execute(
    text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
    {"e": body.email.strip()})).first()                       # victim chosen by email
...
token_hash = hash_ingest_token(box_token)
# INSERT INTO public.gateways (caregiver_id, token_hash, ...)  -> binds attacker token to victim
...
# creates a source_type='gateway' Camera under EVERY one of the victim's patients
# and RETURNS the plaintext ingest_token for each (line 585)
```

**Exploit chain** (attacker only needs the target's email, which is not secret):
1. `POST /v1/gateway/provision-camera` with `{"email":"victim@example.com"}` and any `Authorization: Bearer <attacker-chosen>`.
2. The attacker's token is now a **persistent credential** tied to the victim in `public.gateways`.
3. A phantom camera is created under **every** victim patient; the response returns each camera's plaintext `ingest_token`.
4. The attacker `POST /v1/cameras/me/frames` (`cameras_ingest.py:99`) with that ingest token to **publish arbitrary attacker-controlled MJPEG video** into the victim's `patient_<id>` LiveKit room as a `cam_*` participant — footage the caregiver watches live and that the VLM detectors analyze.
5. The bound gateway token also enables `GET /v1/gateway/roster` (`:324`) and `/gateway/wait` (`:402`) to enumerate the victim's cameras (identities, rooms, desired_state).

For a patient-safety monitoring product this is a data-integrity **and** safety issue: an attacker can inject fabricated "all clear" footage, drown the event stream with false detections, and burn VLM budget. No password, JWT, or ownership proof is required.

**Remediation:**
- Require a genuine caregiver credential to associate a box with an account: gate provisioning behind `CaregiverAuth` and use `auth.id` instead of trusting `body.email`; **or**
- Pre-register legitimate box tokens out-of-band (manufacturing/ops allowlist table) and reject any `box_token` not already provisioned, before looking up the caregiver.
- Either way, never return ingest tokens for an account the caller has not proven they control.

---

## High

### H1 — `public` tables shipped without Row-Level Security (`pi_discoveries`, `gateways`)

**Files:** `supabase/schema.sql:435` (`pi_discoveries` — RLS never enabled); `scripts/apply_gateway_delta.sql:38` (`gateways` — RLS never enabled). Every other `public` table has `enable row level security` + a `has_patient_access`/owner policy (`schema.sql:855-922`); these two do not.

Supabase exposes all `public` tables via PostgREST, and the anon/publishable key (`sb_publishable_…`) is shipped to the iOS client. A table with RLS disabled is fully readable/writable by anyone holding that public key — independent of the backend's own code paths.

- **`pi_discoveries`** stores `pending_ingest_token` in **plaintext** (`schema.sql:458-460`), plus `hardware_id` (device MAC), `device_label`, and `current_url`/`current_audio_url` (internal LAN endpoints). An attacker with the publishable key can `SELECT` rows and steal a Pi's one-shot ingest bearer token before the Pi picks it up, then impersonate that camera on `/v1/cameras/me/frames`. They can also `INSERT`/`UPDATE` fake discoveries or overwrite `current_url` to redirect where the pull-mode reader connects.
- **`gateways`** stores `caregiver_id`, `token_hash`, and `tunnel_port` (a reverse-SSH ops port). Readable rows leak the caregiver→gateway→ops-port mapping; writable rows compound C2.

**Why it's exploitable:** The publishable key is, by design, embedded in the mobile app and therefore public. RLS is the only thing standing between that key and these tables.

**Remediation:**
```sql
ALTER TABLE public.pi_discoveries ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.gateways       ENABLE ROW LEVEL SECURITY;
-- Add no policies (or scope tightly). With RLS on and no policy, PostgREST
-- access is denied while the backend's direct-Postgres connection (which
-- owns these rows) is unaffected.
```
Add the sweep from the skill's `database-security.md` to every migration to catch missed tables. Additionally, stop persisting the Pi ingest token in plaintext — store only its hash (the code already hashes ingest tokens elsewhere in `services/ingest/pi/auth.py`).

---

### H2 — Deleting a patient is not owner-guarded; any shared member can destroy all data

**Files:** `src/api/patients.py:249-266` (`delete_patient`) → `src/repos/patient_repo.py:86-103` (`delete_by_id`).

`delete_by_id` scopes the delete by `accessible_patient_ids(caregiver_id)`, which matches **any** `patient_members` row regardless of `role`:

```python
delete(Patient).where(
    Patient.id == patient_id,
    Patient.id.in_(accessible_patient_ids(caregiver_id)),  # any member, incl. viewer
)
```

But an owner can share a patient with role `viewer` or `caregiver` (`patient_members.py:26,66-103`), and the authorization contract in `src/repos/access.py:5-7` explicitly states role is meant to gate "owner-guarded actions (share/unshare, **delete**)". Share and member-removal are correctly owner-gated (`patient_members.py:72,110`); **delete is not**.

**Exploit:** An owner shares a patient as `viewer`. That viewer calls `DELETE /v1/patients/{id}` and permanently destroys the patient — cascading via `ON DELETE CASCADE` to all cameras, detectors, events, face embeddings, and clips. A read-only collaborator should never hold destructive rights.

**Remediation:** Gate the handler on ownership before deleting:
```python
if not await access.is_owner(session, patient_id=patient_id, caregiver_id=auth.id):
    raise HTTPException(status.HTTP_403_FORBIDDEN, "Only an owner can delete a patient")
```

---

## Medium

### M1 — No rate limiting anywhere (pairing-code brute force + unbounded AI spend)

**Evidence:** No rate-limiting middleware, dependency, or per-user quota exists in `src/` (no `slowapi`/`limiter`/`ratelimit` usage; no limiter in `src/main.py`). The skill's `rate-limiting.md` flags exactly these surfaces.

Concrete impacts:
- **Pairing-code brute force** — `POST /v1/cameras/pair` (`pairing.py:45`) is unauthenticated and matches a **6-digit** `pairing_code` valid for 600s (`settings.pairing_code_ttl_seconds`). With ~10⁶ combinations and no throttle, an attacker can brute-force an active code within the window, obtain the camera's `api_key`, and take over the camera (mint LiveKit tokens, read desired-state). Same brute-force exposure on any camera whose detail screen is open (the app rotates the code on open).
- **Unbounded VLM cost** — every detector tick and the compile-on-create path call Gemini/OpenAI/OpenRouter/DashScope (`services/detection/*`, `services/detector_compiler.py`) with no per-user/per-account usage cap in the app. Combined with C2 (attacker can spin up phantom cameras that publish frames), an attacker can drive detector ticks and drain the AI budget.
- **Auth / email surfaces** — no throttle on token-verified endpoints, support/feedback email sends (`support.py`, `feedback.py`), or ingest.

**Remediation:** Add per-IP **and** per-identity rate limiting (e.g. `slowapi` + Redis, or at the Railway/edge layer) — strictest on `/v1/cameras/pair` (a few attempts/min per IP, with lockout). Add hard per-account VLM usage quotas in the app and hard spending caps in each provider console. Store counters outside client-writable tables.

### M2 — Interactive API docs exposed in all environments

**File:** `src/main.py:167-172` — `FastAPI(...)` is constructed with defaults, so `/docs`, `/redoc`, and `/openapi.json` are served unconditionally, including in `prod`. (The access-log filter at `:56-65` even special-cases these paths, confirming they're live.)

**Why it matters:** The full API schema — every route, parameter, and internal detail (including the gateway/ingest surfaces in C2/H1) — is published to anyone, lowering the cost of attack reconnaissance. This is information disclosure, not a direct breach.

**Remediation:** Disable docs in production:
```python
_s = get_settings()
app = FastAPI(
    title="nemara-backend", version="0.0.1", lifespan=lifespan,
    docs_url=None if _s.env == "prod" else "/docs",
    redoc_url=None if _s.env == "prod" else "/redoc",
    openapi_url=None if _s.env == "prod" else "/openapi.json",
)
```

---

## Low

### L1 — Reusable, never-expiring demo pairing code

**File:** `src/api/pairing.py:53-61,101-145`. When `NEMARA_DEMO_PAIRING_CODE` is set (production sets it to `424242`, exposed in C1), `POST /v1/cameras/pair` special-cases that exact code: it pairs the demo camera **without expiry or single-use clearing** and re-mints a fresh `api_key` on every call. Anyone who knows the code (now public via the leaked env) can repeatedly pair and control the seeded demo camera. Impact is limited to the demo camera, but the value should be rotated and the path should ideally be disabled outside a dedicated review build. **Remediation:** rotate the demo code, scope it to a non-prod/review environment, and consider IP/attempt limits even on the demo path.

### L2 — Missing HTTP security headers

**Evidence:** No security-header middleware in `src/main.py` and no CORS config. For a mobile-only API, absent CORS is acceptable, but standard hardening headers are missing. **Remediation:** add `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy` (e.g. via a small ASGI middleware), and if any browser client is ever added, configure CORS to an explicit allowlist (never `*` with credentials).

### L3 — Unvalidated email inputs & authorization-model inconsistency

- **Unvalidated email fields:** `ShareRequest.email` (`patient_members.py:30`) and `GatewayProvisionRequest.email` (`cameras_ingest.py:467`) are plain `str`, not `EmailStr`. Low risk (used only in parameterized lookups), but validating with `pydantic.EmailStr` tightens input handling. **Remediation:** use `EmailStr`.
- **Membership-model inconsistency:** `talk.py:164,187`, `detectors.py:119` (wandering-door), and the `face_embeddings.py:314` delete pre-read scope by `Patient.caregiver_id == auth.id` (creator-only) rather than `accessible_patient_ids`. This is **more** restrictive than membership, so it is *not* an access-control breach — it merely locks *shared* caregivers out of those specific features while the rest of the app treats them as members. Flagged only because divergence from the stated membership model could mask a future regression. **Remediation:** standardize on `accessible_patient_ids`/`access.is_member` (or `is_owner` where destructive).

---

## Areas checked and found sound (no findings)

- **JWT verification** (`services/auth.py`) — uses `jwt.decode` with signature verification, `algorithms` allowlist (ES256/HS256), and `audience="authenticated"`; opaque `invalid_auth` 401s (`api/deps.py`). No `alg:none` acceptance, no unverified `decode`.
- **LiveKit token minting** (`api/livekit.py`) — camera path enforces room+identity match; caregiver path enforces patient membership and forces a `caregiver_`-prefixed identity; privacy-mode grants baked into the token.
- **LiveKit webhook** (`api/livekit_webhooks.py`) — verifies the signed `Authorization` against the API secret via `WebhookReceiver`/`TokenVerifier` before acting.
- **Camera api-key & Pi ingest-token auth** — hashed at rest (bcrypt / SHA-256), constant-time compare for the ingest token (`services/ingest/pi/auth.py`).
- **RLS core** — enabled on all patient-scoped tables with a `SECURITY DEFINER` `has_patient_access()` that correctly sets `search_path = public` (`schema.sql:847-848`); `WITH CHECK` present on write policies.
- **Injection** — all DB access uses SQLAlchemy Core/ORM or parameterized `text(...)` bindings; no f-string/`%`-built SQL, no `eval`/`exec`/`pickle`/`os.system`/`shell=True`.
- **Mass assignment** — write schemas (`Update*Request`, `Create*Request`) use `extra="forbid"` and expose only safe columns; `share_patient` validates `role` against an allowlist.
- **Storage** — private buckets; reads go through short-lived (600s default) signed URLs minted only after a membership-scoped `event_repo.find_by_id` check (`api/events.py:126-161`); service-role key used server-side only.
- **Dependencies** — `pyproject.toml`/`uv.lock` pin current releases (fastapi 0.136, starlette 1.0.1, pyjwt 2.13, cryptography 48.0, urllib3 2.7, pillow 12.2, sqlalchemy 2.0.49); no known-vulnerable pins were identified.

---

*Generated by [Claude Code](https://claude.ai/code) using the vibe-security audit framework. Static analysis only — findings should be confirmed against the running system and current key state.*
