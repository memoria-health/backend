# Security Audit — nemara-backend

**Date:** 2026-07-25
**Auditor:** Automated static security review (vibe-security methodology)
**Scope:** Static source review of `nemara-backend` (FastAPI + Supabase + LiveKit backend). No live systems, databases, or secrets were accessed. Findings are drawn from source, SQL schema, configuration, and the dependency lockfile only.
**Framework:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) — secrets & env, database access control (Supabase RLS), auth/authz, injection, rate-limiting, payments, AI/LLM, deployment.

---

## Executive Summary

nemara-backend is a well-structured FastAPI service and, in most respects, security-conscious: JWT verification is done correctly (signature + audience + algorithm allow-list, `"none"` rejected), every REST endpoint is scoped to the authenticated caregiver at the repository layer (strong IDOR defense via `accessible_patient_ids`), Supabase RLS is enabled on 12 of 13 tables with ownership-scoped policies, SQL is uniformly parameterized, subprocesses use `exec` argument lists (no shell injection), uploads are size-capped and content-type-validated, and dependencies are current with no known-vulnerable versions. CORS is not wildcarded (no middleware at all).

However, the audit found one **Critical** and two **High** severity issues that require prompt action:

1. **Critical — Secrets committed to the repository.** `/.env.bak.1783536884` is tracked in git and contains a full credential set: LiveKit API secret, OpenAI, Gemini, OpenRouter, and Resend API keys, an APNs signing key, and Supabase auth values. Anyone with read access to the repo (or its history) has live third-party credentials. These allow LiveKit room-token minting (access to patients' live video/audio), AI-provider billing abuse, and sending mail/push as the company. **Rotate every key immediately** and purge the file from history.

2. **High — One database table has no Row-Level Security.** `public.pi_discoveries` is created but never `ALTER TABLE ... ENABLE ROW LEVEL SECURITY`'d (the other 12 tables are). It stores `pending_ingest_token` — a **plaintext ingest bearer token** — plus device MAC addresses and URLs. With Supabase's default grants, this table is readable and writable by anyone holding the public (publishable/anon) key. An attacker can read a Pi's ingest token and publish arbitrary video into a patient's LiveKit room.

3. **High — Unauthenticated gateway camera provisioning.** `POST /v1/gateway/provision-camera` authenticates with a bearer token that the caller **chooses themselves** (it is stored on first use, not pre-issued), and identifies the target account solely by a non-secret `email`. Any anonymous attacker who knows a caregiver's email can create cameras under all of that caregiver's patients and receive working ingest tokens, enabling fake camera feeds and spurious safety alerts on a stranger's account.

The remaining findings (pairing-code brute-force, absent rate-limiting / bcrypt scan amplification, exposed API docs, the fixed demo-pairing code) are Medium/Low and defense-in-depth.

### Severity tally

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High     | 2 |
| Medium   | 3 |
| Low      | 3 |

---

## Critical

### C1 — Live credentials committed to the repository (`.env.bak.1783536884`)

**File:** `/.env.bak.1783536884` (tracked; confirmed via `git ls-files`)

`.gitignore` excludes `.env` and `.env.local`, but **not** `.env.bak.*`, so this backup — created "for e2e walkthrough" and headed "Do not commit" — was committed anyway. It contains:

| Line | Credential | Impact if leaked |
|------|-----------|------------------|
| `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` | LiveKit Cloud API secret (`254B2sRX...`) | Mint tokens for **any** `patient_<id>` room → **watch any patient's live video/audio**, publish into rooms. This is the most severe: it defeats the entire privacy model. |
| `NEMARA_OPENAI_API_KEY` (`sk-proj-…`) | OpenAI key | Unbounded billing on your account |
| `NEMARA_GEMINI_API_KEY` (`AIzaSy…`) | Google Gemini key | Unbounded billing |
| `NEMARA_OPENROUTER_API_KEY` (`sk-or-v1-…`) | OpenRouter key | Unbounded billing |
| `NEMARA_RESEND_API_KEY` (`re_…`) | Resend email key | Send email as `@memoria-health.com` (phishing) |
| `NEMARA_APNS_AUTH_KEY` (PEM `.p8`) + team/key IDs | Apple push signing key | Forge push notifications to caregivers |
| `NEMARA_SUPABASE_*` | Supabase URL / publishable / JWT secret / service-role | See note below |

**Note on the Supabase values:** the committed `NEMARA_SUPABASE_JWT_SECRET` (`super-secret-jwt-token-with-at-least-32-characters-long`) and service-role key (JWT `iss: "supabase-demo"`) are the **well-known Supabase local-dev defaults**, not production secrets — so those specific strings are low-risk *as printed*. The danger is that this file demonstrates production third-party keys (LiveKit/OpenAI/Gemini/OpenRouter/Resend/APNs above) are managed the same way and were committed. Treat the JWT secret as a template for the real risk: **if the production `NEMARA_SUPABASE_JWT_SECRET` ever lands in a committed file, an attacker can forge HS256 caregiver tokens** — the verifier (`src/services/auth.py`) accepts HS256 signed with that secret — and impersonate any account.

**Why exploitable:** Git history is permanent; deleting the file in a new commit does not remove it. Anyone with repo access (current or historical, including any fork/clone/CI artifact) reads these keys.

**Remediation:**
1. **Rotate every key listed above right now** (LiveKit, OpenAI, Gemini, OpenRouter, Resend, APNs; regenerate the Supabase project's real keys if they were ever the ones used here).
2. Remove the file and purge it from history: `git rm .env.bak.1783536884`, then `git filter-repo --path .env.bak.1783536884 --invert-paths` (or BFG), and force-push.
3. Broaden `.gitignore` to cover backups: add `.env*` and `*.bak*`.
4. Add a pre-commit secret scan (`gitleaks protect --staged`) so this cannot recur.

Note: `.dockerignore` *does* exclude `.env.*`, so this file is not baked into the production image — good, but the git exposure stands.

---

## High

### H1 — `public.pi_discoveries` has no RLS; plaintext ingest token is exposed via the public key

**File:** `supabase/schema.sql:435` (table), RLS section `:855–922` (table omitted)

Every table in the schema is given `enable row level security` **except `pi_discoveries`** (the section comment at `:815` even claims "Every table is RLS-on"). That table holds:

- `pending_ingest_token` (`:461`) — a **one-shot plaintext ingest bearer token** handed to a Pi so it can push frames.
- `hardware_id` (MAC address), `current_url`, `current_audio_url`.

In Supabase, the `anon`/`authenticated` roles are granted access to `public` tables by default; **without RLS the table is fully readable and writable by anyone holding the publishable/anon key** (which is, by design, public — it ships in the iOS app and is in the committed env). An attacker can:

- `SELECT pending_ingest_token, current_url FROM pi_discoveries` → obtain a live ingest token, then present it as `Authorization: Bearer <token>` to `POST /v1/cameras/me/frames` (`src/api/cameras_ingest.py:99`) and **publish attacker-controlled video into the patient's `patient_<id>` LiveKit room** — which the caregiver watches live and which the detector pipeline analyzes (fake or suppressed safety alerts).
- `UPDATE`/`DELETE` discovery rows: tamper with `claimed_camera_id`, `current_url`, or wipe the picker.

**Remediation:** Enable RLS and add a policy that denies the anon/authenticated client path entirely (this table is only ever written by the backend's service-role connection, which bypasses RLS):

```sql
alter table public.pi_discoveries enable row level security;
-- No policy for anon/authenticated ⇒ PostgREST access is fully blocked;
-- the backend's direct service-role connection is unaffected.
```

Additionally, stop returning secrets through an unauthenticated read surface — see M1.

### H2 — Unauthenticated gateway provisioning binds a self-chosen token to any account by email

**File:** `src/api/cameras_ingest.py:472` (`gateway_provision_camera`), auth block `:491–508`

The endpoint's "authentication" is `Authorization: Bearer <box's UNIQUE token>` — but the token is **whatever the caller sends**: on first use the handler simply `INSERT`s it into `public.gateways` (`:517`), tying that attacker-chosen token to the caregiver named by `body.email`. Email is not a secret.

**Attack:** An anonymous attacker POSTs `{"email": "victim@known.com"}` with `Authorization: Bearer <attacker-picked>`. The backend:
- creates a `gateways` row linking the attacker's token to the victim caregiver, then
- creates a camera under **every one of the victim's patients** (`:540–579`) and returns a working plaintext `ingest_token` for each (`:585`).

The attacker can now poll `/v1/gateway/roster` (`:324`) for the victim's camera identities/rooms and `POST /v1/cameras/me/frames` with the returned tokens to **inject video into the victim's patient rooms** and spawn phantom cameras that appear in the victim's app — driving false safety events/alerts and eroding trust in a medical-monitoring product. (Live *viewing* of the real feed still requires a caregiver JWT, so confidentiality of the genuine stream is not directly broken; integrity and availability of the monitoring signal are.)

**Remediation:** Do not accept a self-asserted box token bound by email. Options:
- Pre-provision each gateway with a server-generated secret out-of-band and verify it against a stored hash *before* trusting any email.
- Require the provisioning call to carry the owning caregiver's authenticated JWT (or a signed, single-use claim code the caregiver generated in-app), not a plaintext email.
- At minimum, reject provisioning when the presented token is already unknown *and* no proof of account ownership is supplied.

---

## Medium

### M1 — Camera pairing code is brute-forceable (6 digits, 10-minute window, no rate limit)

**File:** `src/api/pairing.py:40` (`pair_camera`), lookup `:63–75`

`POST /v1/cameras/pair` is unauthenticated ("the code IS the proof of authorization") and matches a **6-digit** `pairing_code` valid for 600 s. There is no rate limiting anywhere in the service (see M2), so the 10⁶ code space is guessable within the window at even modest request rates. A correct guess returns a long-lived camera `api_key` (`:90–98`) that grants publish rights into the patient's LiveKit room and camera-role token minting. Exploitable only while a caregiver has an open pairing window, which narrows it — hence Medium, not High.

**Remediation:** Add strict per-IP + per-code rate limiting on this endpoint (e.g. lock after 5 failed attempts against the current window), lengthen the code, and/or add exponential backoff. Consider a longer alphanumeric code.

### M2 — No rate limiting; camera-auth performs an O(N) bcrypt table scan per request

**Files:** `src/api/deps.py:111` (`camera_auth`), `src/api/pi_discovery.py:43` (`announce_pi`), detection LLM calls throughout `src/services/detection/`

The vibe-security rate-limiting checklist requires limits on auth, AI, and expensive endpoints; none exist in the codebase. Two concrete amplifiers:

- **bcrypt scan (DoS):** `camera_auth` (`deps.py:137–159`) loads **every** camera row with a non-null `api_key_hash` and runs `bcrypt.checkpw` (cost 2¹⁰) against each until a match. A single unauthenticated request with a bogus bearer to any `CameraAuth` endpoint forces a full-fleet bcrypt scan. As the camera count grows this is a cheap-request → heavy-CPU amplifier. (The code acknowledges the linear scan and defers an index to "Phase 4".)
- **Unauthenticated write amplification:** `POST /v1/cameras/pi-announce` upserts a DB row per call with no auth or throttle — a flood tool inflates `pi_discoveries` unboundedly.
- **AI cost:** detector/compile paths call Gemini/OpenAI with no per-account usage cap.

**Remediation:** Introduce a rate-limiting layer (e.g. `slowapi` + Redis, or gateway/edge limits) on unauthenticated and expensive routes. Replace the bcrypt linear scan with the already-planned indexed `api_key_prefix` lookup so at most one bcrypt runs per request. Add per-account AI usage quotas and provider-side spending caps.

### M3 — `pi-announce` returns a one-shot ingest token to whoever presents a (spoofable) MAC

**File:** `src/api/pi_discovery.py:48` (`announce_pi`), token consume `:74–82`

The announce endpoint is intentionally unauthenticated (the Pi has no pre-pairing credentials — a documented v1 trade-off). But it upserts keyed on the **client-supplied `hardware_id`** (a MAC address, not a secret) and, if a `pending_ingest_token` exists for that id, returns it to the caller (`consume_pending_ingest_token`). Combined with H1 (an attacker can read `hardware_id` values straight out of the un-RLS'd table), an attacker can race a legitimate Pi's announce — or announce right after a caregiver provisions — and **steal the one-shot ingest token**, then publish into the patient room. This is the exploit chain that makes H1 concrete.

**Remediation:** Ship the v2 per-device secret (negotiated over BLE at setup) sooner; until then, at minimum bind token delivery to a value the attacker cannot read/guess, close H1 so `hardware_id`/tokens aren't enumerable, and alert on multiple announces for one `hardware_id` from different sources.

---

## Low

### L1 — Interactive API docs exposed unconditionally

**File:** `src/main.py:167` (`create_app` — no `docs_url=None`), confirmed by the access-log filter at `:56–57` that silences `/openapi.json`, `/docs`, `/redoc`

FastAPI mounts `/docs`, `/redoc`, and `/openapi.json` by default, and they are never disabled for production. This publishes the full API surface (every route, schema, and auth shape) to anyone. Low impact (it aids reconnaissance; it is not itself a breach) but unnecessary in prod.

**Remediation:** Disable in non-local envs: `FastAPI(..., docs_url=None, redoc_url=None, openapi_url=None)` when `settings.env == "prod"`.

### L2 — Fixed, never-expiring demo pairing code in production

**File:** `src/settings.py:198` (`demo_pairing_code`), `src/api/pairing.py:59–61`, `_pair_demo_camera`

`NEMARA_DEMO_PAIRING_CODE` (`424242` in the leaked env) is special-cased to bypass expiry and single-use so an App Review tester can repeatedly pair the seeded demo camera. The setting docstring states "production is the only env that sets it." This is a permanent fixed-credential pairing path in prod. Blast radius is limited to the seeded `cam_demo` camera, so it is Low — but it is a standing backdoor.

**Remediation:** Scope the demo camera to a dedicated demo account with no real patient data, rotate/retire the code after review, or gate it behind a staging-only environment.

### L3 — Verbose logging defaults and root-logger capture

**Files:** `src/main.py:20–23` (`logging.basicConfig` on the root logger from `NEMARA_LOG_LEVEL`), committed env `NEMARA_LOG_LEVEL=DEBUG`

The service configures the root logger at the env-supplied level and the sample/backup envs set `DEBUG`. At DEBUG, `deps.py` logs accepted `caregiver_id`/`camera_id` and auth-reject reasons. Not a leak of secrets, but DEBUG in production risks over-logging identifiers and third-party library internals.

**Remediation:** Force `INFO` or higher in prod; ensure request bodies/tokens are never logged. (Auth failures are already flattened to an opaque `invalid_auth` for clients — good.)

---

## Areas Reviewed With No Findings

- **JWT verification** (`src/services/auth.py`): dispatches on the token `alg`, verifies signature with an explicit `algorithms` allow-list (`ES256`/`HS256`), enforces `aud="authenticated"`, rejects unsupported algs (including `"none"`) and missing claims. No `decode`-without-verify. HS256 uses the shared secret (not the public key), so no algorithm-confusion.
- **IDOR / authorization** (all of `src/api/*` + `src/repos/*`): every patient-scoped read/write funnels through `accessible_patient_ids(caregiver_id)` (`src/repos/access.py`), returning 404 (not 403) on cross-account access. `patch_patient`/`patch_event` restrict writable fields via Pydantic models (no mass assignment); `state`/`region`/`reviewed_at` are server-controlled. LiveKit token minting (`src/api/livekit.py`) enforces camera↔room match and caregiver membership, and bakes privacy-mode grant restrictions into the token.
- **Supabase RLS** (`supabase/schema.sql`): 12 tables have ownership-scoped `USING`/`WITH CHECK` policies keyed to `auth.uid()` via `has_patient_access`; `region` columns are immutable from the client path via triggers; `SECURITY DEFINER` helpers set `search_path`. (Sole gap: `pi_discoveries` — see H1.)
- **SQL injection:** all raw `text()` statements (`cameras_ingest.py`) use bound parameters; the ORM is used elsewhere. No string-built SQL.
- **Command injection:** every subprocess (`frame_extraction`, `mp4_remux`, `clip_archival`, `streaming_encoder`, `mux`) uses `asyncio.create_subprocess_exec` with argument lists — no shell.
- **Webhook verification** (`src/api/livekit_webhooks.py`): LiveKit signatures verified via `TokenVerifier`/`WebhookReceiver` against the raw body before dispatch; 401 on failure.
- **Uploads** (`src/services/storage.py`, `src/api/patients.py`): content-type allow-list, streamed size caps (413 before buffering), private buckets, short-lived signed URLs, service-role key used only server-side.
- **CORS:** no CORS middleware is registered — no wildcard/credentialed cross-origin exposure.
- **Dependencies** (`uv.lock`): fastapi 0.136.1, starlette 1.0.1, pyjwt 2.13.0, cryptography 48.0.0, urllib3 2.7.0, requests 2.34.2, pillow 12.2.0, jinja2 3.1.6, sqlalchemy 2.0.49 — all current with no known-vulnerable versions identified.
- **Payments:** no payment/Stripe integration present — section not applicable.

---

## Prioritized Remediation Summary

1. **(C1) Rotate all committed third-party keys now** (LiveKit, OpenAI, Gemini, OpenRouter, Resend, APNs), delete `.env.bak.1783536884`, purge it from git history, and broaden `.gitignore` + add a secret-scanning pre-commit hook.
2. **(H1) Enable RLS on `public.pi_discoveries`** with no anon/authenticated policy, so its plaintext ingest tokens stop being world-readable via the public key.
3. **(H2) Replace self-asserted gateway provisioning** — require a pre-issued device secret or an authenticated caregiver claim instead of a caller-chosen token + plaintext email.
4. **(M1/M2) Add rate limiting** on pairing, announce, and camera-auth endpoints; replace the O(N) bcrypt scan with the indexed prefix lookup; add per-account AI usage caps.
5. **(M3)** Bind Pi ingest-token delivery to a non-guessable secret; accelerate the v2 per-device credential.
6. **(L1–L3)** Disable API docs in prod, retire the fixed demo pairing code, and force non-DEBUG logging in prod.
