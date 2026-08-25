# Security Audit — nemara-backend

**Date:** 2026-06-07
**Scope:** Static source audit of the `nemara-backend` FastAPI service (237 tracked files, `src/`, `supabase/`, configs, lockfile).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) audit framework — secrets, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment, data access/input validation.
**Constraints:** Static analysis only. No live systems, databases, or secrets were accessed. Findings are substantiated by file/line references in the committed source.

---

## Executive Summary

nemara-backend is a well-structured FastAPI backend for a caregiver patient-monitoring product (LiveKit video, Gemini/OpenAI detector pipeline, Supabase auth + storage, APNs/Resend notifications). The core authorization model is **solid**: every caregiver-facing endpoint enforces a verified Supabase JWT, the repo layer filters by `caregiver_id` as defense-in-depth, JWT verification correctly uses signature validation with audience/algorithm pinning, Pydantic schemas with `extra="forbid"` prevent mass assignment, and all subprocess calls use `exec`-form (no shell injection). RLS is enabled and correctly owner-scoped on every PHI-bearing table.

However, the audit found **one Critical and three High** issues that materially undermine that posture:

1. **(Critical) Live third-party secrets are committed to the repository.** The tracked `.env` contains real-looking OpenAI, Gemini, LiveKit (key + secret), and Resend API keys plus an APNs signing private key. The project's own `.gitignore` comment confirms these are real and "need rotation." They are recoverable from git history forever.
2. **(High) The `pi_discoveries` table is the one table with RLS disabled.** It is exposed via PostgREST in the `public` schema, so anyone holding the public anon/publishable key (itself committed) can read plaintext device ingest tokens and tamper with discovery rows.
3. **(High) Camera pairing is brute-forceable.** The unauthenticated `POST /v1/cameras/pair` accepts a 6-digit code with no rate limiting anywhere in the application; a successful guess yields a camera credential that can join the patient's live LiveKit room (video + audio).
4. **(High) Symmetric-JWT forgery primitive.** The verifier accepts HS256 tokens signed with `NEMARA_SUPABASE_JWT_SECRET`; that secret is committed, so any party who holds it can forge a token for any caregiver — total account takeover / arbitrary IDOR.

The remaining issues (no rate limiting / usage caps, unauthenticated Pi announce token-theft race, unescaped LLM text in alert emails, DEBUG logging default, missing security headers) are Medium/Low.

**Top priorities:** (1) rotate **every** secret in `.env`, purge it from history, and untrack it; (2) enable RLS on `pi_discoveries`; (3) add rate limiting to the pairing flow; (4) ensure production never reuses the committed JWT secret.

---

## Findings

### CRITICAL

#### C-1 — Live third-party API keys and an APNs private key committed in `.env`
**File:** `.env` (tracked — confirmed via `git ls-files`); lines `18`, `19`, `20`, `23`, `28`, `34`, `35`, `60–66`, `80`
**Category:** Secrets & Environment Variables

The repository tracks `.env` (the `.gitignore` only ignores `.env.local`, and includes an explicit comment acknowledging `.env` is committed). It contains:

| Line | Secret | Notes |
|------|--------|-------|
| `.env:28` | `NEMARA_OPENAI_API_KEY=sk-proj-…` | Live-format OpenAI key |
| `.env:23` | `NEMARA_GEMINI_API_KEY=AIza…` | Live-format Google API key |
| `.env:34-35` | `NEMARA_LIVEKIT_API_KEY` + `NEMARA_LIVEKIT_API_SECRET` | LiveKit Cloud server credentials |
| `.env:80` | `NEMARA_RESEND_API_KEY=re_…` | Resend email API key |
| `.env:60-66` | `NEMARA_APNS_AUTH_KEY` (`-----BEGIN PRIVATE KEY-----`) | APNs `.p8` signing key |
| `.env:18-20` | Supabase publishable key, JWT secret, service-role key | These two latter values are the *well-known Supabase local demo defaults* (`issuer: supabase-demo`), but see C-1 note and H-3 |

**Why it's exploitable:** Anyone with read access to the repo (or any clone/fork, or anyone who ever sees git history) obtains working credentials. Concretely: drain OpenAI/Gemini billing, mint LiveKit access tokens to join customers' rooms and watch patient video/audio (LiveKit key+secret is all the token minter in `src/api/livekit.py` needs), send email from the product's domain via Resend, and sign APNs push payloads to caregiver devices. The project's own `.gitignore` comment states all of these "need rotation." Deleting the file does **not** remove the secrets from history.

**Remediation:**
1. **Rotate every secret listed above immediately** (OpenAI, Gemini, LiveKit, Resend, APNs key — and the Supabase service-role/JWT secret for any non-local project).
2. `git rm --cached .env`, add `.env` to `.gitignore`, and purge it from history (`git filter-repo --path .env --invert-paths` or BFG), then force-push and have collaborators re-clone.
3. Move runtime secrets to the deployment platform's secret manager (Railway variables); keep only `.env.example` (which already uses placeholders correctly) in the repo.
4. Add a pre-commit secret scanner (`gitleaks`) to prevent recurrence.

---

### HIGH

#### H-1 — `pi_discoveries` table has Row-Level Security disabled
**File:** `supabase/schema.sql:310` (table created), RLS block `supabase/schema.sql:627–700` (every other table enabled; `pi_discoveries` omitted). Confirmed: no `enable row level security` and no policy for `pi_discoveries`.
**Category:** Database Access Control (the framework's #1 critical class)

Every other table in the schema gets `alter table … enable row level security` with an owner-scoped policy. `public.pi_discoveries` does **not**. The Supabase API exposes the entire `public` schema (`supabase/config.toml`: `[api] schemas = ["public"]`), and Supabase grants the `anon`/`authenticated` roles table access by default — RLS is the only gate. With RLS off, the table is fully readable and writable by anyone holding the public anon/publishable key (which is itself committed at `.env:18` and is designed to be public).

**Why it's exploitable:** The table holds `pending_ingest_token` — a **plaintext** one-shot bearer token (`schema.sql:336`) that authorizes a Pi to publish frames into a patient's LiveKit room (`src/api/cameras_ingest.py`). An attacker with the anon key can:
- `SELECT` all rows and read `pending_ingest_token` values, hijacking a freshly-provisioned camera's ingest stream (publish arbitrary video into a patient room).
- `INSERT`/`UPDATE` rows to poison `current_url` (shown to every caregiver in the "Add a camera" picker) or flood the discovery list.

**Remediation:**
- Enable RLS: `alter table public.pi_discoveries enable row level security;`. Because Pis are unauthenticated and all legitimate access is via the service-role backend (which bypasses RLS), a deny-all default (RLS on, no anon policy) is correct and sufficient.
- Move `pending_ingest_token` out of any PostgREST-exposed table, or never store the plaintext (the cameras table already stores only a SHA-256 hash — apply the same pattern here).
- Add a regression guard: run the "enable RLS on every public table" loop from the framework's `database-security.md` at the end of `schema.sql`.

#### H-2 — Unauthenticated camera pairing is brute-forceable (no rate limiting)
**File:** `src/api/pairing.py:45` (`POST /v1/cameras/pair`, unauthenticated); code generated at `src/api/cameras.py:184` (`secrets.randbelow(900_000) + 100_000` → 6 digits, ~9×10⁵ space); TTL `settings.pairing_code_ttl_seconds` default 600s (`src/settings.py`). No rate limiting exists anywhere in the codebase (grep for rate-limit/throttle/slowapi returns only comments).
**Category:** Authentication / Rate Limiting

The pairing code is the sole proof of authorization for an unauthenticated endpoint, drawn from a ~900,000-value space, valid for 10 minutes, with **no attempt throttling, lockout, or per-IP limit**. The code entropy source (`secrets`) is fine, but the keyspace is small and unguarded.

**Why it's exploitable:** While any caregiver has an active pairing window open, an attacker can brute-force the live code (≈900k requests within 600s is achievable at modest concurrency). A successful guess returns a long-lived camera `api_key` (`pairing.py:90`). That key lets the attacker mint a LiveKit `role='camera'` token for the camera's room (`src/api/livekit.py:76-86`), and the camera grant includes `can_subscribe=true` — so the attacker joins the patient's room and receives the **live patient video and audio** streams, in addition to hijacking the camera slot.

**Remediation:**
- Add per-IP and global rate limiting + exponential backoff to `/v1/cameras/pair` (and a hard lockout after N failed codes against a given window).
- Increase code entropy (e.g., 8–10 alphanumeric chars) and/or bind the code to a device-presented nonce.
- Invalidate the entire window after a small number of failed attempts.

#### H-3 — HS256 verification path enables full account takeover if the JWT secret is exposed
**File:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), dispatch `auth.py:76-88`; secret wired at `src/api/deps.py:80-84` from `settings.supabase_jwt_secret`; secret committed at `.env:19`.
**Category:** Authentication

The verifier accepts whichever algorithm the *token* declares: ES256 (asymmetric, via JWKS) **or** HS256 (symmetric, verified with `NEMARA_SUPABASE_JWT_SECRET`). HS256 verification means the secret is, in effect, a token-*signing* key — anyone who holds it can forge a valid caregiver JWT with any `sub` (caregiver_id) and `email`. That secret is committed in `.env:19` (currently the public local default), and nothing in the configuration prevents the same `.env`-commit pattern from carrying a real secret to production (see C-1).

**Why it's exploitable:** With the symmetric secret, an attacker forges a token for *any* caregiver and passes `caregiver_auth` (`deps.py:66`), defeating every per-caregiver ownership check at once — read/modify/delete any patient, camera, event, or face embedding. This is the highest-leverage consequence of the committed-secrets issue.

**Remediation:**
- In production, prefer asymmetric ES256/JWKS exclusively: drop `HS256` from `JWT_ALGORITHMS` once Supabase signing keys are migrated, so the symmetric secret can never forge a token.
- Treat `NEMARA_SUPABASE_JWT_SECRET` as a signing key: never commit it, rotate it (it is currently exposed), and store it only in the secret manager.

---

### MEDIUM

#### M-1 — No rate limiting or usage caps on abuse-prone and cost-bearing endpoints
**File:** application-wide (no limiter middleware; `src/main.py` registers routers with no throttling). Affected surfaces include LLM-invoking detector endpoints (`src/api/detectors.py:218` `infer_detector`, `:257` `create_detector` → `src/services/detector_compiler.py` calls Gemini/OpenAI), email-sending endpoints (`src/api/patients.py:199` `test-email`, `src/api/support.py`, `src/api/feedback.py`), and the unauthenticated `pi-announce`/`pair` endpoints.
**Category:** Rate Limiting / AI Integration

There are no per-user or per-IP rate limits and no per-user LLM usage quotas/spend caps in the application. The detector pipeline calls Gemini/OpenAI on every tick of every live camera, and the `infer`/`create` detector endpoints invoke a "compile" LLM call on demand. Authenticated abuse (a malicious or compromised caregiver) or the pairing brute-force (H-2) can drive unbounded model spend, and the email endpoints can be used as a low-volume spam/relay amplifier (attacker-controlled `subject`/`message` forwarded via Resend).

**Remediation:** Add per-IP + per-user rate limiting (e.g., a Redis/Upstash sliding window) on auth-adjacent, LLM-invoking, and email-sending endpoints; enforce hard per-user/per-tier LLM usage quotas in the app; set hard spending caps in the OpenAI/Anthropic/Google consoles as a backstop.

#### M-2 — Unauthenticated `pi-announce` allows ingest-token theft race and picker poisoning
**File:** `src/api/pi_discovery.py:43-90` (`POST /v1/cameras/pi-announce`, unauthenticated); token handoff `pi_discovery.py:74-82`; repo `src/repos/pi_discovery_repo.py:180` `consume_pending_ingest_token`.
**Category:** Authentication / Database Access Control

The announce endpoint is intentionally unauthenticated (the Pi has no pre-pairing credential — acknowledged in the module docstring). But `announce_pi` returns the plaintext `pending_ingest_token` to *whoever announces with a given `hardware_id`* and then nulls it (one-shot). An attacker who knows or guesses a target device's `hardware_id` (a MAC address — low entropy, sometimes physically observable) can poll `pi-announce` and win the race to receive the ingest token the moment a caregiver provisions that device, then publish frames into the patient room. The same endpoint accepts arbitrary `current_url`/`device_label`, which surface to every caregiver in the discovered-devices picker (`/v1/cameras/discovered`).

**Why it's Medium (not High):** Exploitation requires knowing the victim `hardware_id` and winning a timing race against the legitimate Pi. This compounds H-1 (RLS off on the same table makes the tokens directly readable without the race).

**Remediation:** Implement the planned per-device secret negotiated over BLE so announcements are attributable; rate-limit the announce path; never return a plaintext long-lived token over an unauthenticated channel (deliver a short-lived, single-use exchange token bound to the provisioning caregiver's action).

#### M-3 — LLM/event-generated text injected unescaped into alert email HTML
**File:** `src/services/email.py:115-126` (`_event_html` interpolates `title`, `narrative`, `location`, `patient_name` directly into an HTML string with no escaping). Contrast with `_support_html` (`email.py:235`) and `_feedback_html` (`email.py:298`), which **do** escape.
**Category:** AI Integration / Output Handling (untrusted LLM output)

Event `title` and `narrative` are produced by the Gemini/OpenAI detector pipeline (`src/services/detection/…`), i.e. untrusted model output influenced by whatever appears on camera (visual prompt-injection is a known class). They are interpolated into the alert email body without HTML-escaping, so model output containing markup is rendered as HTML in recipients' mail clients (content spoofing, misleading links). The framework explicitly treats LLM output as untrusted and requires sanitization before rendering as HTML.

**Remediation:** Apply the same `esc()` helper already used by `_feedback_html`/`_support_html` to `title`, `narrative`, `location`, and `patient_name` in `_event_html` (and `send_test_email`).

#### M-4 — Verbose `DEBUG` logging committed as the default
**File:** `.env:6` and `.env.example:5` (`NEMARA_LOG_LEVEL=DEBUG`); consumed at `src/main.py:20-23`.
**Category:** Deployment Configuration

`DEBUG` is the shipped default in both `.env` and `.env.example`. In production, DEBUG-level logs across the detector worker, webhook handlers, and auth path (e.g. `auth.accept caregiver_id=…` at `auth.py`/`deps.py`) increase the chance of sensitive data landing in logs and add noise that hampers incident response. The framework calls for disabling debug/verbose logging in production.

**Remediation:** Default to `INFO` (or `WARNING`) for non-local environments; gate `DEBUG` to local dev only. Audit log statements to ensure no tokens/secrets are emitted at any level.

---

### LOW

#### L-1 — No security-header or CORS middleware configured
**File:** `src/main.py:145-186` (`create_app` registers routers only; no `add_middleware`). Grep for CORS/headers middleware returns nothing.
**Category:** Deployment Configuration

No `Strict-Transport-Security`, `X-Content-Type-Options`, `X-Frame-Options`, `Content-Security-Policy`, etc. are set. Impact is limited because the API serves native clients (iOS app, Pi) rather than a browser app, and the *absence* of CORS middleware is fail-safe (no permissive `Access-Control-Allow-Origin: *`). Still worth adding HSTS and `X-Content-Type-Options: nosniff` at the edge/proxy. If a browser-based dashboard consumes this API, configure CORS to an explicit allowlist (never `*` with credentials).

#### L-2 — Discovered-Pi list is not scoped per caregiver
**File:** `src/api/pi_discovery.py:93-122` (`GET /v1/cameras/discovered` returns all unclaimed recent Pis to any authenticated caregiver). Acknowledged in the module docstring.
**Category:** Data Access

Every caregiver sees every unclaimed device's `hardware_id`/`device_label`/`current_url` (and could attempt to claim it). At larger scale this is a cross-customer information leak and a claim-jacking vector. Add per-caregiver scoping (e.g., via the planned BLE-negotiated device secret) before scaling beyond the current small deployment.

---

## What was checked and found clean

- **Authorization model.** Every caregiver endpoint requires a verified JWT (`src/api/deps.py:66`); the repo layer additionally filters by `caregiver_id` (e.g. `event_repo.find_by_id(..., caregiver_id=auth.id)` at `src/api/events.py:144`), giving defense-in-depth against the service-role RLS bypass. Cross-account access returns 404, not 403.
- **JWT verification** (`src/services/auth.py`) uses `pyjwt.decode` with signature verification, explicit `algorithms` allowlist, and `audience="authenticated"` — not the unsafe `decode`-without-verify pattern. (See H-3 for the HS256 caveat.)
- **LiveKit token minting** (`src/api/livekit.py`) correctly verifies room ownership for caregivers (`_ensure_caregiver_can_join`) and room/identity match for cameras (`_ensure_camera_room_match`).
- **Webhook signature verification** (`src/api/livekit_webhooks.py:65-75`) verifies the LiveKit signature over the raw body before acting.
- **RLS** is enabled and correctly owner-scoped (`= auth.uid()` / patient-ownership subquery, with `WITH CHECK`) on all 10 PHI-bearing tables (`supabase/schema.sql:627-700`). Only `pi_discoveries` is missing (H-1).
- **SQL/command injection.** No string-built SQL; all queries use SQLAlchemy expression language. All subprocess calls (`src/services/clip_archival.py`, `clip_annotate.py`, `detection/frame_extraction.py`, `mp4_remux.py`) use `create_subprocess_exec`/list-form `subprocess.run` — no `shell=True`, no `os.system`, no `eval`/`exec`/`pickle`/`yaml.load`.
- **Mass assignment.** Update handlers use typed Pydantic models with `extra="forbid"` and explicit field allowlists (`src/schemas/patient.py:97`), so server-managed columns (`state`, `status_line`, `last_event_at`) cannot be set by clients.
- **Password/secret hashing.** Camera api-keys are bcrypt-hashed (`src/api/pairing.py:78`); Pi ingest tokens are SHA-256 hashed server-side (`src/api/cameras_ingest.py:121-127`). Camera api-key verification uses `bcrypt.checkpw` (constant-time).
- **Dependencies.** Spot-checked lockfile versions (`fastapi 0.136.1`, `starlette 1.0.1`, `pyjwt 2.13.0`, `python-multipart 0.0.29`, `pillow 12.2.0`, `cryptography 48.0.0`, `urllib3 2.7.0`, `aiohttp 3.13.5`, `jinja2 3.1.6`, `requests 2.34.2`) — all current; the vulnerable `python-jose` is **not** used. No known-vulnerable pins identified.

---

## Prioritized Remediation Summary

1. **C-1 — Rotate all committed secrets, purge `.env` from history, untrack it, move to a secret manager, add `gitleaks`.** Highest priority; every other credential-dependent risk flows from this.
2. **H-1 — Enable RLS on `pi_discoveries`** (deny-all default) and stop storing plaintext ingest tokens in an API-exposed table.
3. **H-2 — Rate-limit and harden the pairing flow** (per-IP/global limits, lockout, larger code entropy).
4. **H-3 — Eliminate the HS256 forgery path in production** (asymmetric-only verification) and ensure the JWT secret never lives in committed config.
5. **M-1 — Add rate limiting + per-user LLM usage quotas and provider spend caps.**
6. **M-2 — Authenticate/limit `pi-announce`; never hand a plaintext long-lived token over an unauthenticated channel.**
7. **M-3 — HTML-escape LLM/event text in alert emails.**
8. **M-4 / L-1 / L-2 —** Default logs to `INFO` in prod, add security headers at the edge, scope the discovered-Pi list per caregiver.
