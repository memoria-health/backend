# Security Audit — nemara-backend

**Date:** 2026-06-25
**Scope:** Static source review of the `nemara-backend` FastAPI service (API routes, auth, DB/Supabase schema + RLS, storage, AI/LLM pipeline, deployment config, git history).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill framework — secrets & env, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment, and data-access/input-validation.
**Constraint:** Static only. No live systems, databases, or secrets were accessed. Findings are substantiated by source; where exploitability depends on runtime configuration that can't be confirmed statically, this is stated explicitly.

---

## Executive Summary

nemara-backend is a generally well-constructed FastAPI backend. Authorization is applied consistently at the API layer (every caregiver-facing route filters by `caregiver_id`, returns 404-not-403 to avoid resource enumeration), Supabase RLS is enabled on **11 of 12** tables as defense-in-depth, all SQL uses bound parameters (`text(":k")` / ORM), and all subprocess calls use argument lists (no shell) — so there is **no SQL or command injection** and **no hardcoded secret in the current source tree**.

However, the audit found two **Critical** issues that undermine the otherwise-solid model, plus several High/Medium gaps:

1. **Real third-party credentials were committed to git history** (`.env`, commit `eb8eba7`). Untracking the file later (`cf36938`) does not remove it from history; the OpenAI, Gemini, LiveKit, and APNs keys remain retrievable from any clone and must be treated as compromised.
2. **The `pi_discoveries` table has no Row-Level Security.** Every other public table is RLS-on (the schema's own comment calls RLS "the load-bearing check"), but `pi_discoveries` is omitted entirely — and it stores a plaintext one-shot camera ingest bearer token plus camera LAN URLs. With Supabase's default anon/authenticated grants it is readable and writable via PostgREST by anyone holding the (public) publishable key.
3. **There is no rate limiting anywhere in the codebase.** This makes the unauthenticated 6-digit pairing endpoint brute-forceable (camera takeover), turns the O(N)-bcrypt camera-auth path into a CPU-exhaustion DoS amplifier, allows discovery-table flooding, and leaves the LLM endpoints with no abuse ceiling.

Priority actions: **(1)** rotate every credential that ever lived in the committed `.env` and scrub git history; **(2)** enable RLS on `pi_discoveries`; **(3)** add rate limiting (per-IP + per-user) to the unauthenticated pairing/announce endpoints and the auth/LLM paths.

### Findings by severity

| # | Severity | Title |
|---|----------|-------|
| 1 | Critical | Real API credentials committed to git history (`.env`) |
| 2 | Critical | `pi_discoveries` table missing RLS — token + LAN URLs exposed via anon key |
| 3 | High | No rate limiting on unauthenticated pairing endpoint → 6-digit code brute-force / camera takeover |
| 4 | High | Camera auth performs O(N) bcrypt full-table scan → CPU-exhaustion DoS |
| 5 | Medium | Unbounded AI/LLM cost — no per-user usage cap or rate limit on inference/compile |
| 6 | Medium | Cross-caregiver Pi discovery exposure & hijack (no per-caregiver scoping) |
| 7 | Medium | Unauthenticated `pi-announce` enables discovery-table flooding |
| 8 | Low | HS256 JWTs accepted in production alongside ES256 |
| 9 | Low | Guessable demo pairing code (`424242`) is reusable & never-expiring in prod |
| 10 | Low | No HTTP security headers (HSTS, X-Content-Type-Options, etc.) |
| 11 | Low | Hardcoded personal email addresses as recipient defaults |

---

## Critical

### 1. Real third-party API credentials committed to git history

**Evidence:** `.env` was first committed in `eb8eba7` and only untracked later in `cf36938` ("chore(security): stop tracking .env"). The historical blob (`git show eb8eba7:.env` / `cf36938^:.env`) contains real cloud-provider secrets, including:

- `NEMARA_OPENAI_API_KEY=sk-pro…` (OpenAI secret key)
- `NEMARA_GEMINI_API_KEY=AIzaSy…` (Google Gemini key)
- `NEMARA_LIVEKIT_API_KEY=APItud…` / `NEMARA_LIVEKIT_API_SECRET=eQslXj…` (LiveKit Cloud)
- `NEMARA_APNS_AUTH_KEY="-----BEGIN …"` (Apple APNs ES256 `.p8` private key)
- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` / `NEMARA_SUPABASE_JWT_SECRET` (these point at `http://127.0.0.1` local Supabase and may be the well-known local defaults, but must still be verified)

**Why it's exploitable:** Removing a file from the working tree does **not** remove it from git history. Anyone with read access to the repository (or any existing clone/fork/CI cache) can run `git show eb8eba7:.env` and recover the keys. The leaked OpenAI/Gemini keys allow unbounded billed LLM usage at the company's expense; the LiveKit keys allow minting tokens to join/publish into any patient's live video room; the APNs key allows sending push notifications as the app. The `.gitignore` claims the secrets were "rotated," but rotation cannot be verified statically and the keys remain in history regardless.

**Remediation:**
- **Rotate every credential** that ever appeared in the committed `.env` — OpenAI, Gemini, LiveKit (key+secret), APNs `.p8`, and (if not the local default) the Supabase service-role key and JWT secret. Treat all as compromised.
- Scrub history with `git filter-repo --invert-paths --path .env` (or BFG) and force-push, then have all collaborators re-clone. Note this rewrites SHAs.
- Run `gitleaks detect` in CI as a pre-merge gate so a secret can never be committed again.
- Confirm `.env` is in `.gitignore` (it now is) and keep `.env.example` placeholder-only (it is).

---

### 2. `pi_discoveries` table missing Row-Level Security

**Location:** `supabase/schema.sql:319` (table definition) — and its **absence** from the RLS block at `supabase/schema.sql:682-763`.

**Evidence:** The schema enables RLS on `caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`, `feedback`, and `ui_events` (11 tables). The 12th table, `pi_discoveries`, has **no `ALTER TABLE … ENABLE ROW LEVEL SECURITY` and no policy**. The schema comment at line 676 states: *"Every table is RLS-on … RLS is the load-bearing check"* — so the codebase explicitly relies on RLS to gate anon/authenticated PostgREST access. `pi_discoveries` stores:

```
pending_ingest_token text   -- one-shot PLAINTEXT camera ingest bearer token (schema.sql:345)
current_url           text  -- camera's LAN MJPEG endpoint (schema.sql:331)
hardware_id           text  -- device MAC address
```

**Why it's exploitable:** Supabase grants the `anon` and `authenticated` roles default privileges on `public` tables; RLS is what restricts row visibility for the publishable-key (anon) PostgREST API the iOS app uses. With RLS off, anyone holding the **public** publishable key can `SELECT`, `INSERT`, `UPDATE`, and `DELETE` `pi_discoveries` rows directly via `…/rest/v1/pi_discoveries`. Concretely an attacker can:
- **Steal a camera's ingest token** during the provisioning window — `pending_ingest_token` is the plaintext bearer accepted by `POST /v1/cameras/me/frames` (`src/api/cameras_ingest.py:127`). With it, the attacker can push arbitrary JPEG frames into the patient's LiveKit room, spoofing the caregiver's live feed of a vulnerable person.
- **Enumerate** every camera's internal LAN URL and MAC across all customers.
- **Delete/tamper** discovery rows to disrupt pairing or redirect provisioning.

The plaintext token is transient (nulled on first announce read, `src/api/pi_discovery.py:74`), which narrows but does not close the window.

**Remediation:** Enable RLS on the table and add an owner-scoped (or no-public-access) policy. Because the Pi has no caregiver identity pre-pairing and the announce/provision paths already go through the authenticated FastAPI backend (which uses a privileged DB connection that bypasses RLS), the safest policy is to expose **no** rows to the anon/authenticated PostgREST roles at all:

```sql
alter table public.pi_discoveries enable row level security;
-- No policy for anon/authenticated → PostgREST returns zero rows and
-- rejects writes. The FastAPI backend's direct connection is unaffected.
-- (Optionally scope reads to a caregiver who has claimed the row.)
revoke all on public.pi_discoveries from anon, authenticated;
```
Also add the catch-all loop from the skill (`references/database-security.md`) to every future migration to auto-enable RLS on any new table, and store `pending_ingest_token` hashed rather than in plaintext.

---

## High

### 3. No rate limiting on the unauthenticated pairing endpoint → brute-force camera takeover

**Location:** `src/api/pairing.py:45` (`POST /v1/cameras/pair`); code space defined at `src/api/cameras.py:184`.

**Evidence:** `pair_camera` is intentionally unauthenticated ("the code IS the proof of authorization"). The pairing code is a 6-digit number restricted to `100000–999999` (`secrets.randbelow(900_000) + 100_000`), valid for `pairing_code_ttl_seconds` (default **600s**). There is **no rate limiting** in the codebase (a repo-wide search for `ratelimit`/`slowapi`/`limiter`/`throttle` returns only unrelated MJPEG `delimiter` matches). On a successful guess the endpoint returns a long-lived `api_key` plus the LiveKit room/identity (`src/api/pairing.py:90-98`).

**Why it's exploitable:** The keyspace is only 900,000 values. With no per-IP/per-code throttling an attacker can brute-force all candidates within the 10-minute window (~1,500 req/s) — and the probability rises with each concurrently-active pairing window. A successful guess yields a valid camera `api_key`, granting the attacker control of that camera's `desired_state`, the ability to mint LiveKit `camera`-role tokens (`src/api/livekit.py:76`), and to publish into the patient's room.

**Remediation:** Add strict rate limiting (e.g. Upstash Redis or an API-gateway limiter) keyed on both IP and pairing-code prefix, with a low ceiling and exponential backoff/lockout on repeated failures. Increase code entropy (alphanumeric, ≥8 chars) and shorten the TTL. Consider single-IP attempt caps that trip well before the keyspace can be meaningfully sampled.

### 4. Camera authentication does an O(N) bcrypt full-table scan → CPU-exhaustion DoS

**Location:** `src/api/deps.py:111-159` (`camera_auth`).

**Evidence:** To authenticate a camera api-key the dependency selects **all** rows with a non-null `api_key_hash` and runs `bcrypt.checkpw` against each until a match (or exhausts the table on a bad token). bcrypt at `rounds=10` is ~50–100 ms per check by design. This dep guards `GET /v1/cameras/me/desired-state`, which the wall phone polls **every 1 second** (`src/api/cameras.py:670`). The code comments acknowledge this is a "Phase 0" linear scan.

**Why it's exploitable:** Combined with the absence of rate limiting (Finding 3), a single unauthenticated request bearing a bogus token forces a bcrypt comparison against *every* camera row (no early exit, since nothing matches). An attacker sending modest request volume multiplies CPU cost by N (number of paired cameras), exhausting the event-loop executor pool and denying service to legitimate cameras and caregivers. Cost grows with customer scale.

**Remediation:** Stop scanning. Move cameras to a token shape that supports indexed lookup — e.g. a `key_id` prefix stored unhashed (the code's own TODO at `deps.py:104`) so a single row is fetched, then one bcrypt verify; or switch camera api-keys to the same SHA-256-hashed-high-entropy-token scheme already used for Pi ingest tokens (`src/services/ingest/pi/auth.py`), which does an indexed equality match. Add rate limiting to the auth paths regardless.

---

## Medium

### 5. Unbounded AI/LLM cost — no per-user usage cap or rate limit

**Location:** `src/api/detectors.py:217` (`POST /v1/detectors/infer`) and `:257` (`create_detector` → background Gemini/OpenAI compile, `src/services/detector_compiler.py`).

**Evidence:** `/v1/detectors/infer` synchronously calls an LLM on caregiver-supplied text, and `create_detector` fires a background LLM compile per call. Both require only `CaregiverAuth` (any signed-in caregiver). There is no per-user/day token budget, no request throttle, and no application-level spending cap. The skill's `references/rate-limiting.md` and `references/ai-integration.md` flag exactly this: *"A single user can drain your entire monthly budget in minutes."*

**Why it's exploitable:** An authenticated user (Supabase signup is open) can loop these endpoints to run up arbitrary OpenAI/Gemini spend. Even absent malice, a buggy client can do the same.

**Remediation:** Enforce per-user/per-tier quotas tracked server-side with hard limits, rate-limit the LLM endpoints, and set hard spending caps in the OpenAI and Google consoles. Return a clear 429 when a user's quota is exceeded.

### 6. Cross-caregiver Pi discovery exposure & hijack

**Location:** `src/api/pi_discovery.py:97` (`GET /v1/cameras/discovered`) and `src/api/cameras.py:265` (`provision_pi_camera`).

**Evidence:** `list_discovered` returns **all** unclaimed, recently-seen Pi devices to **any** authenticated caregiver — there is no per-caregiver scoping (acknowledged in the module docstring, lines 14-18). The response includes each device's `current_url` (LAN endpoint), `hardware_id` (MAC), and label. `provision_pi_camera` then lets a caregiver claim any unclaimed `discovery_id` (409 only if already claimed).

**Why it's exploitable:** During the discovery window a malicious caregiver can see another customer's Pi and **claim it to their own account**, binding the victim's physical camera to the attacker's patient and obtaining its ingest token — i.e. hijacking the live feed of someone else's home. It also leaks home-network topology across tenants.

**Remediation:** Scope discovery and provisioning to the requesting caregiver. Implement the planned per-device secret negotiated at BLE setup (referenced in the docstring) so a discovery can be attributed to, and claimed only by, the caregiver who physically provisioned the device. Until then, minimize exposed fields and require a proof-of-possession token in the provision call.

### 7. Unauthenticated `pi-announce` enables discovery-table flooding

**Location:** `src/api/pi_discovery.py:48` (`POST /v1/cameras/pi-announce`).

**Evidence:** The announce endpoint is unauthenticated by design (the Pi has no pre-pairing credential) and upserts a `pi_discoveries` row keyed on a client-supplied `hardware_id`, with no rate limiting.

**Why it's exploitable:** An attacker can submit unlimited announcements with arbitrary `hardware_id`s, flooding `pi_discoveries` and polluting the cross-tenant `/discovered` picker every caregiver sees (compounding Finding 6), and consuming storage/DB resources. Combined with Finding 2 (no RLS), the same rows are also directly writable via the anon PostgREST API.

**Remediation:** Rate-limit by IP, cap the number of distinct unclaimed discoveries, expire stale rows aggressively, and move to the BLE-negotiated per-device secret so announcements are attributable.

---

## Low / Informational

### 8. HS256 JWTs accepted in production alongside ES256
`src/services/auth.py:26` sets `JWT_ALGORITHMS = ["ES256", "HS256"]`. Production Supabase uses ES256 (asymmetric, verified via JWKS); HS256 is the legacy/local path using the shared symmetric `supabase_jwt_secret`. The dispatch selects the key by the token's `alg` header, so the classic algorithm-confusion attack is **not** present. However, continuing to accept HS256 in production means a leak of `supabase_jwt_secret` (see Finding 1) lets an attacker forge any caregiver token. **Remediation:** restrict to `["ES256"]` in non-local environments.

### 9. Guessable demo pairing code, reusable and never-expiring in prod
`src/settings.py:153` (`demo_pairing_code`) plus `scripts/seed_demo_camera.py:15,129` default the App-Review demo code to `424242`. When set in production, `pair_camera` special-cases it to pair the demo camera **without expiry or single-use clearing** (`src/api/pairing.py:59-61`). Anyone guessing `424242` can repeatedly mint an api_key for the demo camera and publish into its room. Impact is limited to the demo camera (no real patient). **Remediation:** use a high-entropy demo code, rotate it, or gate the demo path behind an additional check.

### 10. No HTTP security headers
`src/main.py` adds no `add_middleware` for security headers; no CORS middleware is configured either (which is fine — the API is consumed by native clients, so no cross-origin browser surface). Responses lack `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options`, etc. **Remediation:** add a middleware setting HSTS and `X-Content-Type-Options: nosniff` at minimum (per `references/deployment.md`). Low priority given no browser client.

### 11. Hardcoded personal email addresses as recipient defaults
`src/settings.py:355-369` default support/feedback recipients to personal Gmail and company addresses, and `scripts/seed.py:47-48` contains a demo email/password (`jane@patel.co` / `Mar9aret!2024`, labeled local-dev-only). These are recipient/seed values rather than secrets, but hardcoding personal addresses in source is poor hygiene. **Remediation:** source recipient lists exclusively from environment configuration.

---

## What was checked and found clean

- **SQL injection:** none. All raw SQL uses `text()` with bound parameters (`worker.py:247`, `presence_sweeper.py`, `camera_presence.py`, etc.); all other access is via SQLAlchemy ORM. No f-string/`%`-formatted SQL.
- **Command injection:** none. `clip_archival.py`, `clip_annotate.py`, `frame_extraction.py`, and `mp4_remux.py` invoke ffmpeg via `create_subprocess_exec`/`subprocess.run` with argument lists; no `shell=True`, no `os.system`, no `eval`/`exec`.
- **Mass assignment:** API write models are explicit Pydantic schemas with `extra="forbid"` on settings and field-restricted update models (`patch_patient` notes that `state`/`status_line` are rejected as 422); request bodies are not spread into ORM writes.
- **IDOR / authorization:** every caregiver route resolves ownership via `caregiver_id`-scoped repo lookups and returns 404 (not 403) on cross-account access (`patients.py`, `events.py`, `detectors.py`, `cameras.py`, `talk.py`). LiveKit token minting validates camera-room match and patient ownership (`livekit.py:148-193`). Internal record-clip route reuses the Talk ownership check and is gated off by default (`enable_internals=False`).
- **Webhook verification:** the LiveKit webhook verifies signatures against the API key/secret on the raw body before acting (`livekit_webhooks.py:65-75`).
- **Token storage:** camera api-keys and Pi ingest tokens are stored hashed (bcrypt / SHA-256 of high-entropy tokens); plaintext is returned once. Ingest-token compare is constant-time (`auth.py:78`).
- **Upload safety:** photo uploads are streamed with a running size cap (413 on overflow), content-type allow-listed, and stored under a `{patient_id}.{ext}` key — no path traversal from client filenames (`patients.py:43-61`, `storage.py:52-73`).
- **Prompt injection / output handling:** user text goes into the user turn, not the system prompt; LLM JSON output is schema-constrained and never executed or rendered as raw HTML (email bodies HTML-escape user/LLM text, `email.py:252,366`).
- **No Stripe/payments** integration is present, so the payments category does not apply.

---

*Generated by an automated static security review using the vibe-security methodology. Findings should be triaged and validated against runtime configuration (especially Supabase role grants and whether the leaked credentials were in fact rotated).*
