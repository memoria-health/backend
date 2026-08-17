# Security Audit — nemara-backend

- **Date (UTC):** 2026-08-04
- **Commit audited:** `7f325e0` (branch `main`)
- **Scope:** Static source review of `nemara-backend` (FastAPI + Supabase + LiveKit). No live systems, databases, or secrets were accessed.
- **Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill — secrets & env, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment config, and data-access/input validation, plus injection / SSRF / IDOR / mass-assignment review.

---

## Executive Summary

The codebase is, on the whole, **well-engineered from a security standpoint.** Supabase Row-Level Security is enabled on every tenant table with membership-scoped policies; the backend connects to Postgres directly (bypassing RLS) and consistently re-enforces authorization in the repo layer by scoping every query to `caregiver_id`/patient membership. JWT verification is correct (signature-verified, `alg` allow-list, audience checked, `"none"` rejected). Camera/gateway ingest tokens are high-entropy and stored hashed; the internal ops route uses a constant-time shared-secret check that fails closed; LiveKit webhooks are signature-verified; the LiveKit token minter properly binds cameras to their own room and caregivers to rooms they can access. Uploads are size-capped and streamed. IDOR was specifically probed across events, patients, cameras, detectors, and face-embeddings and **none was found** — cross-account access returns 404 by design.

However, the audit found **two issues that require immediate action**, plus several lower-severity hardening items:

1. **(Critical) Production secrets are committed in git history** — including an **APNs signing private key (`.p8`)**, the **LiveKit API secret**, the **Supabase `service_role` key**, and **Gemini / OpenAI / Resend / OpenRouter API keys** — inside `.env` and a `.env.bak.1783536884` backup. The files were untracked from `HEAD` but remain fully recoverable in history. Every one of these credentials must be rotated and the history purged.
2. **(Critical) `POST /v1/gateway/provision-camera` has no real authentication.** It trusts a self-asserted, never-registered bearer token and identifies the target account by a **non-secret email address**. Anyone who knows a caregiver's email can create cameras under *all* of that caregiver's patients and receive plaintext ingest tokens that let them **publish arbitrary video/audio into the patient's private LiveKit rooms** — directly undermining a safety-monitoring product for a vulnerable population.

Ranked findings follow.

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | Critical | Production secrets (APNs key, LiveKit secret, service_role, API keys) in git history | `.env`, `.env.bak.1783536884` (history) |
| 2 | Critical | Gateway camera provisioning has no authentication (account takeover via email) | `src/api/cameras_ingest.py:527` |
| 3 | Medium  | No rate limiting anywhere — pairing-code brute force + unbounded LLM cost | `src/api/pairing.py:40`, `src/api/detectors.py:448`, app-wide |
| 4 | Medium  | Unauthenticated Pi-announce: device injection into every caregiver, no scoping | `src/api/pi_discovery.py:43` |
| 5 | Medium  | Reusable, non-expiring demo pairing code can be enabled in production | `src/api/pairing.py:59`, `src/settings.py:386` |
| 6 | Low     | Event-alert email interpolates LLM text into HTML without escaping | `src/services/email.py:124` |
| 7 | Low     | Interactive API docs / OpenAPI schema exposed unauthenticated | `src/main.py:204` |
| 8 | Info    | Dependency advisory scan not run (offline); no vulnerable pins found by inspection | `uv.lock` |

---

## Critical

### 1. Production secrets committed to git history (`.env`, `.env.bak.1783536884`)

**Files:** `.env` and `.env.bak.1783536884` — removed from `HEAD` but present in history at (at least) commits `cf36938` ("stop tracking .env"), `a53ec6e` ("untrack .env.bak…contains production keys"), and their predecessors `1c886c3`, `f1c0564`, `59f62fc`.

**What leaked.** The backup `.env.bak.1783536884` contained a full environment with real values for:

- `NEMARA_APNS_AUTH_KEY` — the **APNs `.p8` ECDSA P-256 private key** (the full `-----BEGIN PRIVATE KEY----- … -----END PRIVATE KEY-----` block is in the diff), plus `NEMARA_APNS_KEY_ID` and `NEMARA_APNS_TEAM_ID`.
- `NEMARA_LIVEKIT_API_KEY` + `NEMARA_LIVEKIT_API_SECRET`.
- `NEMARA_SUPABASE_SERVICE_ROLE_KEY`, `NEMARA_SUPABASE_JWT_SECRET`, `NEMARA_SUPABASE_PUBLISHABLE_KEY`.
- `NEMARA_GEMINI_API_KEY` (`AIzaSyDW3q…` — Google API key format), `NEMARA_OPENAI_API_KEY`, `NEMARA_OPENROUTER_API_KEY`, `NEMARA_RESEND_API_KEY`.
- `NEMARA_DEMO_PAIRING_CODE=424242` (see Finding 5).

*(Secrets are intentionally not reproduced in full here — they are readable directly from `git log -p` on the commits above.)*

**Why it's exploitable.**
- **APNs private key** → forge push notifications to any Nemara caregiver device (spoof safety alerts, or silence real ones by exhausting the token) for the lifetime of that key.
- **LiveKit API secret** → mint LiveKit tokens for *any* room, i.e. subscribe to any patient's live camera feed or publish into it, entirely outside the backend.
- **Supabase `service_role` key** → bypass **all** RLS and read/modify/delete every row in the database (all patients, caregivers, events, embeddings). This is the "game over" key the skill flags as the top secret to never expose.
- **Gemini / OpenAI / OpenRouter / Resend keys** → drain the company's LLM/email billing and send mail from a verified domain.

Deleting the files from `HEAD` does **not** remediate this: the skill's rule is explicit — *"If a secret was ever committed to Git history, consider it compromised."* The README states the earlier `.env` leak's secrets were rotated, but (a) that claim is unverifiable from the repo, and (b) the `.env.bak.1783536884` backup is a *separate* artifact whose rotation status is not documented.

**Remediation.**
1. **Rotate every credential listed above, now**, treating all as compromised: revoke & re-issue the APNs auth key in Apple Developer; roll the LiveKit API key/secret; rotate the Supabase `service_role` key and JWT secret (project-level); regenerate Gemini, OpenAI, OpenRouter, and Resend keys; change the demo pairing code.
2. **Purge the files from history** (`git filter-repo --path .env --path .env.bak.1783536884 --invert-paths`, or BFG), then force-push and have all collaborators re-clone. The `.gitignore` already covers `.env`, `.env.bak*`, `.pod.env` — good, but ignore rules don't scrub existing history.
3. Add a pre-commit secret scanner (`gitleaks`) and a CI gate so a real key can never be committed again.
4. Confirm production actually runs on the rotated values (Railway env), not any of the leaked ones.

---

### 2. `POST /v1/gateway/provision-camera` — no authentication; account takeover via known email

**File:** `src/api/cameras_ingest.py:527-648`

```python
async def gateway_provision_camera(body, session, authorization=Header(default="")):
    box_token = authorization[len("bearer "):].strip()      # attacker-chosen, never verified
    ...
    cg = (await session.execute(
        text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
        {"e": body.email.strip()})).first()                  # target selected by non-secret email
    ...
    # inserts a gateways row tying THIS attacker token -> the victim caregiver,
    # then creates a camera under EVERY one of the victim's patients and
    # returns each camera's PLAINTEXT ingest_token to the caller.
```

**Why it's exploitable.** The only "credential" is `Authorization: Bearer <anything>`: the handler never checks the token against any pre-registered allow-list — if no `gateways` row has that token's hash it simply **inserts a new one bound to the victim**. The victim is chosen purely by `body.email`, which is not a secret. So an unauthenticated attacker who knows (or guesses) a caregiver's email can:

1. Create `source_type='gateway'` cameras under **every patient** owned by that caregiver.
2. Receive the **plaintext `ingest_token`** for each new camera in the response.
3. Use those tokens against `POST /v1/cameras/me/frames` (`cameras_ingest.py:101`) to **publish attacker-controlled MJPEG frames into the patient's LiveKit room** as a `cam_*` participant — indistinguishable downstream from a real camera. The detector pipeline then analyzes the attacker's frames, and any caregiver viewing the room sees them.
4. Poll `GET /v1/gateway/roster` / `/gateway/wait` with the same token to enumerate the victim's cameras (identities, rooms, privacy state).

In an elder/dementia-care safety product this is severe: an attacker can **inject a benign loop to mask a real fall/wandering event**, or inject frames that **trigger false alerts**, and can attach spoofed cameras to a stranger's account. This is a textbook "never trust the client" broken-access-control failure.

**Remediation.**
- Do not accept a self-asserted box token. Provisioning must be authorized by something the attacker cannot supply:
  - Require a **caregiver JWT** (`CaregiverAuth`) on the provisioning call so the account is proven, not asserted by email; or
  - Issue each gateway box a **pre-registered, backend-minted claim token** (created by an authenticated caregiver/ops action) and reject any token not already in `gateways`; or
  - Bind provisioning to a short-lived, single-use, caregiver-generated pairing code (mirror the camera pairing flow), never to a plaintext email.
- Until fixed, treat any `source_type='gateway'` camera and its ingest token as untrusted and consider disabling the endpoint in production.
- Independently, verify publisher-side LiveKit track permissions so a `cam_*` identity cannot subscribe to caregiver Talk audio beyond what's intended.

---

## Medium

### 3. No rate limiting anywhere (pairing-code brute force + unbounded LLM spend)

**Files:** app-wide — no rate-limiting dependency or middleware exists (`src/main.py` registers routers with no throttle; `pyproject.toml` has no limiter). Concrete hotspots: `src/api/pairing.py:40`, `src/api/detectors.py:448` (`/detectors/infer`) and the compile path at `src/api/detectors.py`, `src/api/pi_discovery.py:43`.

**Why it matters.**
- **Pairing brute force:** `POST /v1/cameras/pair` is unauthenticated and matches a **6-digit numeric** `pairing_code` (`schema.sql:331`) valid for 10 minutes (`pairing_code_ttl_seconds`). With no rate limit, the ~10⁶ space is brute-forceable within the window; a hit lets the attacker pair *their* phone as the camera and receive a long-lived `api_key`. The opaque 401 helps but does not prevent enumeration.
- **Unbounded LLM cost:** `/v1/detectors/infer` and the detector-compile path each make VLM/LLM calls per authenticated request with no per-user quota or hard spend cap in code. The skill calls out that AI endpoints "almost never" have limits and a single user "can drain your entire monthly budget in minutes." Provider-side caps are not visible in the repo.
- **Announce/ingest spam:** the unauthenticated `pi-announce` and the token endpoints have no throttle (see Finding 4).

**Remediation.** Add per-IP **and** per-principal rate limiting (e.g. Upstash Redis / `slowapi`) to: `/v1/cameras/pair` (tight — e.g. 5/min/IP), `/v1/cameras/pi-announce`, `/v1/detectors/infer` and detector create/compile, and the ingest endpoints. Widen the pairing code (alphanumeric, ≥8 chars) or add attempt lockout. Set hard per-account LLM usage quotas in the DB and hard spend caps in the Gemini/OpenAI/OpenRouter dashboards.

### 4. Unauthenticated Pi-announce — device injection into every caregiver, no per-caregiver scoping

**File:** `src/api/pi_discovery.py:43` (`POST /v1/cameras/pi-announce`), and `/v1/cameras/discovered` at `:93`.

**Why it matters.** `pi-announce` is unauthenticated (acknowledged in the code as a v1 trade-off) and upserts rows keyed by an attacker-suppliable `hardware_id` with an attacker-controlled `current_url`. `/v1/cameras/discovered` returns **all** unclaimed recent discoveries to **every** caregiver (no per-caregiver scoping). An attacker can therefore (a) flood `pi_discoveries` (no rate limit), and (b) inject fake devices with attacker-chosen labels/URLs into every caregiver's "Add a camera" picker; a caregiver who provisions one wires their camera setup to attacker-named infrastructure. The one-shot `pending_ingest_token` design and RLS deny-all on the table limit direct data exposure, so this is Medium, not Critical.

**Remediation.** Rate-limit and abuse-monitor the announce path; scope `/cameras/discovered` to the caregiver's own network/context; bring forward the planned per-device secret (BLE-negotiated) so announcements are attributable. Sanitize/validate `current_url` scheme (https-only) and `device_label` before display.

### 5. Reusable, non-expiring demo pairing code enablable in production

**Files:** `src/api/pairing.py:59-61` + `_pair_demo_camera:101`; setting `src/settings.py:386` (`demo_pairing_code`).

**Why it matters.** When `NEMARA_DEMO_PAIRING_CODE` is set, `POST /v1/cameras/pair` special-cases that exact code and pairs the demo camera **without expiry and without clearing the code**, re-minting an `api_key` on every call. The leaked `.env.bak` (Finding 1) shows this was set to the trivially guessable `424242`. If that (or any short code) is configured in a real environment, anyone can repeatedly pair the demo camera and obtain a working camera `api_key` for it. Scoped to the single demo camera, hence Medium.

**Remediation.** Keep this strictly out of production (assert `env == 'prod'` never enables it, or gate it behind an allow-list). If a demo path is needed for App Review, scope it to a dedicated non-prod project and use a long random code rotated per review cycle.

---

## Low

### 6. Event-alert email interpolates LLM/detector text into HTML without escaping

**File:** `src/services/email.py:124-135` (`_event_html`), called from `_dispatch_one:189` and `send_test_email:411`.

`title`, `narrative`, and `location` are formatted directly into the email HTML with no escaping, unlike the support/feedback builders (`_support_html:252`, `_feedback_html:366`) which correctly `esc()`. These fields originate from VLM/detector output (influenced by scene content and a caregiver's `custom_prompt`), so a crafted value could inject markup into the caregiver's alert email. Impact is limited (recipients are the caregiver's own configured addresses; no script execution in most mail clients), hence Low.

**Remediation.** Apply the same HTML-escaping helper used by the support/feedback paths to `title`, `narrative`, and `location`.

### 7. Interactive API docs / OpenAPI schema exposed unauthenticated in production

**File:** `src/main.py:204` (`create_app` — `docs_url`/`redoc_url`/`openapi_url` left at FastAPI defaults; `main.py:57` only *quiets the access logs* for `/docs`, `/redoc`, `/openapi.json`, it does not disable them).

`/docs`, `/redoc`, and `/openapi.json` are reachable without auth in production, disclosing the full API surface (every route, params, schemas) to anyone — useful reconnaissance for the issues above. Low.

**Remediation.** Disable the docs/OpenAPI routes when `env == 'prod'` (`FastAPI(docs_url=None, redoc_url=None, openapi_url=None)`), or gate them behind auth.

---

## Informational

### 8. Dependency advisory scan not performed (offline environment)

`uv.lock` pins recent versions (e.g. `fastapi 0.136.1`, `starlette`, `cryptography 48.0.0`, `pillow 12.2.0`, `python-multipart 0.0.29`, `pyjwt 2.13.0`, `aiohttp 3.13.5`, `jinja2 3.1.6`, `requests 2.34.2`, `torch 2.13.0`, `onnx 1.18.0`). No known-vulnerable version was identified by inspection, but a live advisory database was not reachable in this static run. **Run `pip-audit` / `uv`'s advisory check (or Dependabot) in CI** to catch CVEs as they are published — especially for the media stack (pillow, opencv, onnx/onnxruntime, ffmpeg-adjacent) that processes untrusted camera bytes.

---

## What was reviewed and found sound (no action needed)

- **Supabase RLS** (`supabase/schema.sql:922-1082`): RLS enabled on every table; tenant tables use the membership function `has_patient_access(...)`; `caregivers`/`caregiver_devices`/`feedback`/`ui_events` scope to `auth.uid()`; `gateways` and `pi_discoveries` are `REVOKE`d from `anon, authenticated` with deny-all RLS; `INSERT/UPDATE` policies carry `WITH CHECK`; region/country columns are made immutable from the client path by triggers; `SECURITY DEFINER` functions set `search_path`.
- **API-layer authorization**: the backend uses a direct Postgres connection (service credentials, RLS-bypassing — `src/db/session.py`), and every caregiver-facing route re-enforces ownership by scoping to `caregiver_id`/membership (`src/repos/access.py`). IDOR probing across events, patients, cameras, detectors, and face-embeddings found consistent 404-on-cross-account behavior. Mass-assignment is blocked by `extra="forbid"` Pydantic models exposing only writable fields.
- **JWT verification** (`src/services/auth.py`): signature-verified, `algorithms` allow-list (`ES256`/`HS256`), audience checked, expiry enforced, `"none"`/unknown alg rejected; failures collapse to an opaque 401.
- **Secrets in current tree**: only `.env.example` / `.pod.env.example` are tracked, and they contain placeholders. No hardcoded keys in source. No client-exposed AI keys (all server-side).
- **LiveKit**: webhook receiver verifies the signed `Authorization` against the API key/secret (`src/api/livekit_webhooks.py:72`); token minting binds a camera to its own room/identity and a caregiver to rooms they can access, and forces a `caregiver_`-prefixed identity (`src/api/livekit.py`).
- **Ingest & internal auth**: ingest tokens are 32-byte random, SHA-256-hashed at rest, looked up by indexed hash; the `/internal/*` route uses `secrets.compare_digest` and fails closed outside local/test (`src/api/internal/cameras.py:48`).
- **Storage**: private buckets, service-role-signed short-TTL URLs minted only after an ownership check; object keys are server-constructed from UUIDs (no path traversal); uploads are streamed and size-capped (`src/api/patients.py:44`, `src/services/storage.py`).
- **Injection**: SQLAlchemy ORM / bound parameters throughout; the raw-`text()` statements in `cameras_ingest.py` are all parameterized. No `f`-string SQL, no shell string interpolation of user input observed.
- **SSRF**: the Pi pull-mode fetch is vestigial; the push architecture passes `pi_stream_url=""` and does not fetch caregiver/Pi-supplied URLs server-side.
