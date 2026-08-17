# Security Audit — nemara-backend — 2026-06-16

Static source review of the `nemara-backend` FastAPI + Supabase service.
Methodology: the **vibe-security** skill (secrets, database/RLS, auth, data
access, rate-limiting, payments, AI integration, deployment, mobile). No live
systems, databases, or secrets were accessed — this is a code/config/git-history
review only. Findings are limited to issues substantiated by the code.

---

## Executive Summary

The application code is, on the whole, carefully written: API route handlers
consistently authenticate via a verified Supabase JWT and authorize by
`caregiver_id`-scoped repository queries; Pydantic request models use
`extra="forbid"` (blocking mass assignment); LiveKit token minting and the Pi
ingest-token flow are well constrained; the Stripe/payment surface does not
exist; and Row-Level Security is enabled on almost every table. There is no
SQL-injection, command-injection, or reachable SSRF surface.

The serious problems are concentrated in **secrets management** and a
**single missing RLS policy**, amplified by the **complete absence of rate
limiting**.

| # | Severity | Title |
|---|----------|-------|
| 1 | **Critical** | Production secrets committed to git history (service_role key, JWT secret, APNs private key, LLM + LiveKit + Resend keys) |
| 2 | **Critical** | `pi_discoveries` table has RLS disabled — anon key can read plaintext Pi ingest tokens and tamper with the device picker |
| 3 | **High** | JWT verifier accepts HS256 signed with the (leaked) shared secret → forge any caregiver's session |
| 4 | **High** | Unauthenticated, unthrottled 6-digit camera-pairing code is brute-forceable → camera takeover |
| 5 | **Medium** | No rate limiting anywhere (auth, email relay, AI, uploads) |
| 6 | **Medium** | Over-broad `FOR ALL` RLS policies let clients write server-managed columns directly via PostgREST |
| 7 | **Medium** | No server-side AI usage caps / per-user quotas |
| 8 | **Medium** | Reusable, never-expiring demo pairing code usable in production |
| 9 | **Medium** | Prompt injection via caregiver-authored custom detector prompts |
| 10 | **Low** | Latent SSRF in Pi pull-mode ingest (not currently wired) |
| 11 | **Low** | Container runs as root; DEBUG default; no security headers |
| 12 | **Low** | APNs device-token row keyed solely on a guessable-unique token |

**Act first on #1 and #2.** They are independently exploitable and #1 makes #3
trivially exploitable.

---

## Critical

### 1. Production secrets committed to git history

**Evidence:** `.env` was tracked until commit `cf36938`
(`chore(security): stop tracking .env`). The full secret-bearing file is still
recoverable from history at `cf36938^:.env`. It contained, with **real
values**:

- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` — bypasses **all** Row-Level Security; full
  read/write/delete on every table in the project.
- `NEMARA_SUPABASE_JWT_SECRET` — the symmetric secret the backend accepts for
  HS256 token verification (see finding #3).
- `NEMARA_APNS_AUTH_KEY` — the full ES256 `.p8` APNs private key
  (`-----BEGIN PRIVATE KEY----- … -----END PRIVATE KEY-----`), plus
  `NEMARA_APNS_TEAM_ID` / `NEMARA_APNS_KEY_ID`.
- `NEMARA_GEMINI_API_KEY`, `NEMARA_OPENAI_API_KEY` — billable LLM keys.
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` /
  `NEMARA_LIVEKIT_WEBHOOK_API_KEY` (+ the project URL
  `wss://memoria-dev-…livekit.cloud`).
- `NEMARA_RESEND_API_KEY` — outbound email.

`README.md:82-92` still actively instructs teammates that these secrets are
"committed in `.env`" ("The repo ships with a working `.env` (it's a private
repo)").

**Why it's exploitable:** Deleting a file from the tip does not remove it from
history — anyone who has ever cloned the repo, or who gains read access, can
run `git show cf36938^:.env` and recover every key. The service_role key alone
is game-over: it bypasses RLS and grants full access to all patient PII,
clips, and face embeddings. The APNs private key allows forging push to every
device. The `.gitignore` comment claims the secrets were "rotated," but
rotation cannot be confirmed from source and the README contradicts it.

**Remediation:**
1. **Rotate every secret listed above immediately** — Supabase keys (incl.
   service_role and JWT secret), the APNs `.p8`, Gemini, OpenAI, LiveKit, and
   Resend. Assume all are compromised.
2. Purge them from history (`git filter-repo --invert-paths --path .env`, or
   BFG) and force-push; have all collaborators re-clone.
3. Fix `README.md` — never commit `.env`; distribute dev secrets via a secret
   manager. Add `gitleaks` to CI as a pre-merge gate.
4. Keep production secrets only in the deployment platform's secret store
   (Railway), never in the repo.

### 2. `pi_discoveries` table has RLS disabled

**File:** `supabase/schema.sql:310-353` (table) — the RLS block at
`supabase/schema.sql:664-754` enables `row level security` on **every other**
table (`caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`,
`events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`,
`feedback`, `ui_events`) but **omits `pi_discoveries`**.

**Why it's exploitable:** In Supabase, every table in the `public` schema is
exposed through PostgREST and granted to the public `anon` / `authenticated`
roles. With RLS disabled, the table's contents are fully readable and writable
by anyone holding the Supabase **anon key**, which is public by design (it
ships inside the iOS app bundle). `pi_discoveries` holds
`pending_ingest_token` (`schema.sql:336`) — the **plaintext** Pi ingest bearer
token left for a freshly-provisioned Pi to pick up. An attacker can:

- `SELECT pending_ingest_token` and use it against `POST /v1/cameras/me/frames`
  to **push arbitrary video** into a patient's LiveKit room, impersonating the
  camera (the caregiver sees attacker-controlled footage / a frozen scene).
- Read every device's `hardware_id`, `device_label`, and home-network
  `current_url`.
- `INSERT` spoofed discovery rows so bogus devices appear in every caregiver's
  "Add a camera" picker, or `UPDATE`/`DELETE` legitimate rows.

This is the skill's #1 critical category ("a table without RLS is fully
readable and writable by anyone with the anon key").

**Remediation:** Enable RLS and add policies (or no policies, to block all
PostgREST access since only the backend — connecting as the `postgres`
superuser — needs this table):

```sql
alter table public.pi_discoveries enable row level security;
-- The Pi announce path and the caregiver "discovered" list both go through
-- the backend (which uses the service role / direct superuser connection and
-- bypasses RLS), so no anon/authenticated policy is required. RLS-on with no
-- policy blocks all direct PostgREST access while leaving the backend working.
```
Also stop storing the ingest token in plaintext at rest (store only a hash, as
is already done for `cameras.ingest_token_hash`).

---

## High

### 3. JWT verifier accepts HS256 signed with the shared secret

**File:** `src/services/auth.py:26,72-88` — `JWT_ALGORITHMS = ["ES256",
"HS256"]`; the verifier dispatches on the token's own `alg` header and, for
`HS256`, verifies with `settings.supabase_jwt_secret`.

**Why it's exploitable:** Production Supabase tokens are ES256 (verified via
JWKS), but the verifier *also* accepts any HS256 token signed with the
symmetric `supabase_jwt_secret`. That secret was leaked in git history
(finding #1). An attacker can mint a token `{ "alg": "HS256", "sub":
"<victim-uuid>", "email": "...", "aud": "authenticated", "exp": <future> }`,
sign it with the leaked secret, and the backend will accept it as **any
caregiver** — full account takeover across all tenants, reachable on every
authenticated endpoint. Accepting a caller-selected symmetric algorithm in
production is dangerous regardless of the leak: it makes one shared secret a
universal forgery key.

**Remediation:** In production, restrict acceptance to `ES256` and the JWKS
path only; do not accept HS256 unless `env in {local, test}`. After rotating
the JWT secret (finding #1):

```python
algorithms = ["ES256"] if settings.env == "prod" else ["ES256", "HS256"]
```
and refuse to fall back to the symmetric secret in prod.

### 4. Unauthenticated, unthrottled camera-pairing code is brute-forceable

**Files:** `src/api/pairing.py:45-86`; `src/settings.py:130`
(`pairing_code_ttl_seconds=600`); pairing code is 6 digits
(`schema.sql:170`).

**Why it's exploitable:** `POST /v1/cameras/pair` is unauthenticated ("the code
IS the proof of authorization") and there is **no rate limiting anywhere** in
the app (finding #5). The lookup is a single indexed `SELECT … WHERE
pairing_code = ?` *before* any bcrypt work, so each guess is cheap. The code
space is only 10^6 and the window is 10 minutes. An attacker who knows a
caregiver is mid-pairing can brute-force the active code and call `pair`
first, receiving the camera's long-lived `api_key` — which then lets them mint
LiveKit tokens for the patient's room, upload clips, and poll desired-state
(i.e. take over the camera identity / view the room).

**Remediation:** Add strict per-IP and global rate limiting + temporary
lockout on `/v1/cameras/pair`; increase the pairing-code entropy (≥8
alphanumeric chars) and/or cap total guesses per active code; consider
single-attempt invalidation on repeated misses.

---

## Medium

### 5. No rate limiting on any endpoint

**Evidence:** No `slowapi`, `Ratelimit`, Redis limiter, or middleware exists
(grep across `src/` returns nothing). The skill requires limits on auth,
AI-calling, and email/SMS endpoints.

**Impact:**
- Pairing-code brute force (finding #4).
- `POST /v1/support` and `POST /v1/feedback` forward via Resend on each
  authenticated call — usable as a low-rate spam/relay amplifier and to burn
  Resend quota (`src/api/support.py`, `src/api/feedback.py`,
  `src/services/email.py`).
- Photo / feedback-screenshot uploads and detector-triggering paths have no
  abuse ceiling beyond per-request size caps.

**Remediation:** Add a shared limiter (Upstash Redis or equivalent) combining
per-IP and per-user windows; apply tight limits to pairing, support/feedback,
uploads, and any LLM-triggering route.

### 6. Over-broad `FOR ALL` RLS policies expose server-managed columns

**File:** `supabase/schema.sql:677-754` — `patients_owner`, `cameras_owner`,
`detectors_owner`, `caregivers_self`, etc. are all `FOR ALL … USING/WITH CHECK
(ownership)` with no column restriction.

**Why it matters:** The iOS app holds a real *authenticated* Supabase session
and talks to PostgREST/Realtime directly (the schema notes `ui_events` are
inserted "directly … via the authed Supabase client (no backend endpoint)").
Because the policies are `FOR ALL` over all columns, a caregiver can bypass the
backend's validation and directly `UPDATE` columns the API treats as
server-managed — e.g. `patients.state` / `status_line` / `last_event_at`, and
`cameras.api_key_hash` / `ingest_token_hash` / `online` / `desired_state` /
`state`. RLS still blocks cross-tenant access, so the impact is integrity /
self-tampering *within the caller's own tenant* (e.g. forging their own
patient's status, or rewriting their own camera's `api_key_hash`), not
cross-account data theft — hence Medium, not Critical.

**Remediation:** Use column-level grants so clients can only write the fields
they own, keeping server-managed columns to the service role:

```sql
revoke update on public.cameras from authenticated;
grant update (room, icon_name, disabled, ai_paused) on public.cameras to authenticated;
-- repeat for patients (display fields only); leave state/api_key_hash/etc. to the backend.
```
Or split the catch-all `FOR ALL` policies into per-operation policies that
exclude sensitive columns.

### 7. No server-side AI usage caps or per-user quotas

**Files:** `src/settings.py` (no usage-cap/quota settings);
`src/services/detection/` runs Gemini→OpenAI per camera per tick.

**Why it matters:** The skill requires hard per-user/per-tier quotas in
addition to provider caps. Here there is no application-level token/clip
budget. With no rate limiting (finding #5) and caregiver-creatable
patients/cameras/custom detectors, a user can scale up the number of active
detection loops and drive LLM spend with no enforced ceiling.

**Remediation:** Track per-caregiver (or per-tier) token/clip usage in the DB,
enforce daily/monthly hard caps server-side, and set hard spending caps in the
Gemini and OpenAI consoles.

### 8. Reusable, never-expiring demo pairing code usable in production

**Files:** `src/settings.py:131-137` (`demo_pairing_code`);
`src/api/pairing.py:59-61,101-145` (`_pair_demo_camera`).

**Why it matters:** When `NEMARA_DEMO_PAIRING_CODE` is set (the comment says
"production is the only env that sets it"), `POST /v1/cameras/pair` special-cases
that exact code to pair the seeded demo camera **without expiry or single-use
clearing**, re-minting a fresh `api_key` each time. Anyone who learns the
static code (it ships to Apple reviewers) can repeatedly obtain the demo
camera's `api_key` and mint LiveKit viewer/broadcaster tokens for its room —
viewing video and pushing talk audio. Scope is limited to the demo
patient/camera, hence Medium.

**Remediation:** Don't enable the demo code in the production project; host the
review demo in an isolated Supabase/LiveKit project, or bind the demo path to a
sandbox tenant with no real patient data.

### 9. Prompt injection via caregiver-authored custom detector prompts

**Files:** `src/services/detection/prompt.py:114-123` (detector
`prompt_fragment`s are concatenated into the per-tick LLM prompt);
custom detectors originate from caregiver `custom_prompt`
(`schema.sql:379`) compiled via `src/services/detector_compiler.py`.

**Why it matters:** A caregiver controls free text that ends up in the model
prompt, so they can steer detector verdicts and the generated `narrative`/
`description` for **their own** patients (self-scoped — no cross-tenant
reach). The generated narrative is later emailed; the email path HTML-escapes
output (`email.py:_support_html`/`_feedback_html`/`_event_html` use `&lt;`
escaping), which limits HTML/script injection. Main risk is detector
reliability and using the model as an arbitrary generator.

**Remediation:** Keep untrusted user text strictly inside a delimited
user-role message rather than blended into the system/base prompt; continue to
constrain output to the JSON schema (already enforced); treat custom prompts
as untrusted input.

---

## Low

### 10. Latent SSRF in Pi pull-mode ingest (not currently reachable)

**Files:** `src/api/cameras.py:319-323` (the `pi-provision` direct-URL mode
stores a caregiver-supplied `pi_stream_url` / `pi_audio_url` with no
host/IP validation); `src/services/ingest/pi/relay_publisher.py:239-244,435`
+ `mjpeg_reader.read_mjpeg` would fetch that URL server-side.

**Why it's only Low:** The running backend wires **only** the authenticated
push path (`cameras_ingest.py` → `run_from_iterator`, `pi_stream_url=""`); the
pull-mode `RelayPublisher.run()` helper (`relay_publisher.py:588`) is not
started anywhere in `src/main.py` or any background worker, so the fetch is not
reachable. The provision endpoint is also caregiver-authenticated.

**Remediation:** If pull mode is ever re-enabled, validate the URL against a
public-IP allowlist (reject RFC1918, link-local `169.254.169.254`, loopback)
before fetching. Drop the vestigial direct-URL mode if unused.

### 11. Container hardening: root user, DEBUG default, no security headers

**Files:** `Dockerfile` (no `USER` directive → runs as root); `.env.example:5`
(`NEMARA_LOG_LEVEL=DEBUG`); `src/main.py` (no security-header middleware; no
CORS middleware — correctly absent, so cross-origin browser calls are blocked
by default, but HSTS/X-Content-Type-Options/etc. are also absent).

**Remediation:** Add a non-root `USER` to the image; default prod log level to
`INFO`; add `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`,
`X-Frame-Options: DENY`, and a `Referrer-Policy` via middleware.

### 12. APNs device-token row keyed solely on a unique token

**Files:** `src/api/devices.py`; `schema.sql:580-593`
(`caregiver_devices_token_uniq`); `upsert_by_token`.

**Why it matters:** Registration upserts on the globally-unique `apns_token`,
reassigning `caregiver_id`. A caregiver who obtains another user's APNs token
could re-register it under their own account, redirecting that device's push
delivery. APNs tokens are not trivially guessable, hence Low.

**Remediation:** Don't move an existing token to a new `caregiver_id` on
upsert without re-validating device ownership; or scope uniqueness to
`(caregiver_id, apns_token)`.

---

## Dependencies

Lockfile (`uv.lock`) pins are current with no known-vulnerable versions
identified: `pyjwt 2.13.0`, `cryptography 48.0.0`, `fastapi 0.136.1`,
`starlette 1.0.1`, `pillow 12.2.0`, `aiohttp 3.13.5`, `requests 2.34.2`,
`bcrypt 5.0.0`. No action required beyond routine updates.

---

## What was checked and found clean

- **API authz / IDOR:** route handlers consistently require `CaregiverAuth`
  and scope reads/writes by `caregiver_id` in the repo layer; cross-tenant
  attempts return 404 (no ID disclosure). LiveKit token minting enforces
  room/identity ownership (`src/api/livekit.py`). Talk-mode and internals
  reuse the ownership helper.
- **Mass assignment:** request models use `extra="forbid"`; server-managed
  columns are not exposed in the API schemas (`patients.py`, `caregivers.py`).
- **Injection:** all DB access is via SQLAlchemy ORM / parameterized queries;
  no string-built SQL, no `os.system`/`shell=True` with user input.
- **Webhooks:** LiveKit webhook signatures are verified before dispatch
  (`src/api/livekit_webhooks.py`).
- **Uploads:** photo/clip uploads are size-capped while streaming
  (`patients.py:_read_capped_upload`) and stored in private buckets served via
  short-lived signed URLs.
- **Payments:** no Stripe / payment surface exists.
- **JWT basics:** signature + audience (`authenticated`) + expiry are verified;
  failures collapse to an opaque 401. (The HS256-acceptance issue is finding
  #3.)

---

*Audit performed statically against commit `741a653`. No live systems were
accessed. Real secret values discovered in git history are intentionally not
reproduced in this report.*
