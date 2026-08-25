# Security Audit — nemara-backend

**Date:** 2026-06-19
**Auditor:** Automated security review (vibe-security methodology)
**Scope:** Static source audit of the `nemara-backend` FastAPI service (commit on `main`, HEAD `4873148`). No live systems, databases, or secrets were accessed.
**Methodology:** [vibe-security skill](https://github.com/raroque/vibe-security-skill) — secrets, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment, and data-access/input-validation.

---

## Executive Summary

`nemara-backend` is a generally **well-architected** codebase from a security standpoint. Authorization is enforced consistently and correctly: every caregiver-facing endpoint scopes its database reads and writes by the authenticated caregiver's id (`caregiver_id=auth.id`), Supabase Row-Level Security is enabled with owner-scoped `USING` **and** `WITH CHECK` clauses on every table, JWTs are verified by signature and audience (never `decode()`-only), LiveKit tokens and webhooks are properly authorized/signature-verified, secrets at rest are hashed (bcrypt for camera API keys, SHA-256 for Pi ingest tokens), and no SQL injection or IDOR was found. The Supabase `service_role` key is used strictly server-side.

The audit nonetheless found **one Critical** and several lower-severity issues. The single most urgent item is that a real `.env` file containing live third-party credentials — including an **Apple APNs signing key, and OpenAI, Gemini, LiveKit, and Resend API keys** — was committed to git history and remains retrievable from any clone, even though the file was later untracked.

| # | Severity | Title |
|---|----------|-------|
| 1 | **Critical** | Live third-party secrets committed to git history (`.env`) |
| 2 | **High** | No rate limiting anywhere — unauthenticated pairing-code brute force |
| 3 | **Medium** | `camera_auth` linear bcrypt scan — CPU-exhaustion DoS amplifier |
| 4 | **Medium** | JWT verifier accepts symmetric HS256 alongside ES256 |
| 5 | **Medium** | Unauthenticated, un-scoped Pi discovery (`pi-announce` / `discovered`) |
| 6 | **Low** | LLM scene description rendered into alert-email HTML without escaping |
| 7 | **Low** | Static, reusable, never-expiring demo pairing code |
| 8 | **Low** | Caregiver-/peer-supplied Pi URLs stored (latent SSRF) |

### Top priorities
1. **Rotate every key that was ever in the committed `.env` and purge it from git history** (Critical).
2. **Add rate limiting** to the unauthenticated pairing/ingest/announce endpoints and to authenticated AI/LLM-cost paths (High).
3. **Harden the camera-auth path** against algorithmic-complexity DoS and **drop HS256 acceptance in production** (Medium).

---

## Critical

### 1. Live third-party secrets committed to git history (`.env`)

**Location:** git history — `.env` at commits `59f62fc` and earlier; removed from tracking in `cf36938` ("chore(security): stop tracking .env"). Still fully retrievable via `git show 59f62fc:.env`.

The `.gitignore` (lines 11–14) states the leaked secrets were rotated, but the file remains in history and **a static audit cannot confirm rotation** — every secret below must be treated as compromised until rotation is independently verified.

Secrets present in the historical `.env`:

| Key | Value (prefix) | Sensitivity |
|-----|----------------|-------------|
| `NEMARA_APNS_AUTH_KEY` | `-----BEGIN PRIVATE KEY----- MIGTAgEAMBMG…` | **Apple APNs ES256 push signing key (.p8)** — can mint push notifications to all users impersonating the app |
| `NEMARA_OPENAI_API_KEY` | `sk-proj-YbEoTJIg…` | Live OpenAI key — unbounded spend |
| `NEMARA_GEMINI_API_KEY` | `AIzaSyDW3quQ…` | Live Google Gemini key — unbounded spend |
| `NEMARA_LIVEKIT_API_KEY` / `_SECRET` | `APItudo6ecfroAu` / `eQslXjZ7OHy…` | LiveKit Cloud creds — mint tokens, join/publish to any room, read all camera streams |
| `NEMARA_RESEND_API_KEY` | `re_N3Fo9qHR_…` | Resend key — send mail from the verified domain (phishing) |

> Note: `NEMARA_SUPABASE_JWT_SECRET` (`super-secret-jwt-token-with-at-least-32-characters-long`) and `NEMARA_SUPABASE_SERVICE_ROLE_KEY` in that file are the **well-known Supabase local-dev default values**, not production secrets — but see Finding 4 for why HS256 acceptance makes the JWT secret class especially dangerous if a real one ever leaks.

**Why it's exploitable:** Anyone with read access to the repository (or any past clone/fork/CI cache) can extract these from history with a single `git` command. The APNs key and LiveKit credentials are the most damaging: the LiveKit secret allows minting tokens to subscribe to **any patient's live camera/audio room** — a direct privacy breach of vulnerable monitored individuals.

**Remediation:**
- **Rotate all five credential sets now** (Apple APNs key, OpenAI, Gemini, LiveKit key+secret, Resend) and confirm the old values are revoked at each provider.
- Purge `.env` from history (`git filter-repo --path .env --invert-paths` or BFG), force-push, and have all collaborators re-clone. Removing it from tracking alone (commit `cf36938`) does **not** remove it from history.
- Run `gitleaks detect --no-git=false` in CI to prevent regressions.
- Keep production secrets only in the Railway environment (already the stated practice — enforce it).

---

## High

### 2. No rate limiting anywhere — unauthenticated pairing-code brute force

**Location:** `src/api/pairing.py:45` (`POST /v1/cameras/pair`); systemic — no rate-limiting middleware or dependency exists anywhere in `src/`.

`pair_camera` is **unauthenticated by design** (the 6-digit pairing code is the proof of authorization). A code is drawn from `100000–999999` (`src/api/cameras.py:184`) and is valid for 10 minutes (`pairing_code_ttl_seconds=600`). There is no per-IP, per-code, or global throttle on attempts, and failures return an opaque 401 that does not slow the attacker.

**Why it's exploitable:** While a caregiver has an active pairing window open, an attacker who can reach the endpoint can submit codes at network speed against the ~900k space. With no throttle, a meaningful fraction of the keyspace can be swept inside the 600-second window, racing the legitimate phone. A successful guess returns a real camera `api_key` (`pairing.py:90`), which grants the attacker the camera's identity and the ability to mint **LiveKit camera tokens** for that patient's room (`src/api/livekit.py:76`) and publish into it. The same total absence of rate limiting also exposes:
- The detector-create path (`/v1/patients/{id}/detectors`) which triggers an **LLM compile call** per request — unbounded model cost.
- The JWT/camera auth paths (account/credential probing).
- `pi-announce` (Finding 5).

This maps directly to the skill's rate-limiting guidance: auth endpoints, AI calls, and expensive operations all require limits, and AI assistants "almost never add it."

**Remediation:**
- Put a rate limiter in front of `POST /v1/cameras/pair` (e.g. per-IP + per-code attempt budget; lock a code after N failures). Consider shrinking the active-window count and increasing code entropy (e.g. 8+ digits) for the same UX cost.
- Add per-user/per-IP limits on detector-create (LLM compile) and other expensive endpoints; set hard provider spend caps on OpenAI/Gemini.
- Use a shared store (Upstash Redis / private-schema table), not a public table, so counters can't be reset by clients.

---

## Medium

### 3. `camera_auth` linear bcrypt scan — CPU-exhaustion DoS amplifier

**Location:** `src/api/deps.py:111-159` (`camera_auth`); also reused by `src/api/livekit.py:77` and polled every ~1s by `GET /v1/cameras/me/desired-state` (`src/api/cameras.py:561`).

For every camera-authenticated request, the handler loads **all** cameras with a non-null `api_key_hash` and runs `bcrypt.checkpw` (cost factor 10) against each until a match is found (or never, on a bad token). The code comments acknowledge this "linear-scan all rows" design.

**Why it's exploitable:** bcrypt at rounds=10 is intentionally ~50–100 ms of CPU. An attacker sending requests with an arbitrary bearer token to any `CameraAuth`-protected route (e.g. `/v1/cameras/me/desired-state`, or `/v1/livekit/token` with `role=camera`) forces a full scan = `N × bcrypt` per request, with **no rate limit** (Finding 2) to cap concurrency. As the fleet grows, a small number of concurrent bad-token requests can saturate the executor thread pool and stall the event loop, degrading service for all cameras and caregivers. This is an algorithmic-complexity DoS amplified by the missing rate limiting.

**Remediation:**
- Implement the `api_key_prefix` index the code's own comment proposes (store an unhashed short prefix, look up by it, then bcrypt-verify the single candidate) so each request does **one** bcrypt, not N.
- Bound executor concurrency and apply per-IP rate limiting to camera-auth routes.

### 4. JWT verifier accepts symmetric HS256 alongside ES256

**Location:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), dispatch at lines 72–88.

The verifier dispatches on the token's `alg` header: ES256 tokens are checked against Supabase JWKS public keys; HS256 tokens are checked against the shared symmetric `supabase_jwt_secret`. The implementation is **not** vulnerable to the classic algorithm-confusion attack (for HS256 it uses `jwt_secret`, never the EC public key as an HMAC key — lines 76–79), so this is a hardening finding, not an immediate bypass.

**Why it matters:** Production Supabase issues ES256 tokens verified via public JWKS; the HS256 path exists only for "legacy / local-dev." Continuing to accept HS256 in production means the symmetric `supabase_jwt_secret` becomes a **forge-any-user master key**: anyone who obtains it can mint a valid JWT for **any** `sub`/caregiver and impersonate them. Symmetric secrets are far easier to leak than they look (a default value of this exact secret class was already committed to history — Finding 1), and a single leak is full account takeover for every user.

**Remediation:**
- In production, restrict `JWT_ALGORITHMS` to `["ES256"]` and verify only via JWKS. Gate HS256 acceptance behind `settings.env in {"local","test"}` so it can never be used against a real deployment.

### 5. Unauthenticated, un-scoped Pi discovery (`pi-announce` / `discovered`)

**Location:** `src/api/pi_discovery.py:48` (`POST /v1/cameras/pi-announce`, unauthenticated) and `:97` (`GET /v1/cameras/discovered`, authenticated but not per-caregiver scoped). The module docstring and `supabase/schema.sql:303-309` explicitly accept this as a v1 risk.

`pi-announce` accepts attacker-controlled `hardware_id`, `device_label`, `current_url`, and `current_audio_url` with no authentication and no rate limit, upserting a `pi_discoveries` row. `discovered` returns **all** recent unclaimed discoveries to **every** caregiver, with no tenant scoping.

**Why it's exploitable:** Any client that can reach the backend can (a) flood `pi_discoveries` with junk (no rate limit), and (b) inject a spoofed "device" (`device_label`/`current_url` of their choosing) that appears in **every** caregiver's "Add a camera" picker — a cross-tenant injection usable for social engineering (e.g. luring a caregiver into provisioning an attacker-named device) and for poisoning the picker UX. A provisioned spoof also stores the attacker's URL on the camera row (see Finding 8).

**Remediation:**
- Rate-limit `pi-announce` per IP / `hardware_id`.
- Bind discoveries to a tenant as planned (the docstring's "per-device secret negotiated over BLE"), and scope `discovered` to that binding so a caregiver only sees devices on their own network/account.
- Validate/normalize `current_url` (reject non-LAN schemes/hosts) at announce time.

---

## Low

### 6. LLM scene description rendered into alert-email HTML without escaping

**Location:** `src/services/email.py:124-135` (`_event_html`) — `narrative`, `title`, and `location` are interpolated into HTML with no escaping. `narrative` is the VLM's free-text scene description (`src/services/detection/event_promotion.py:281-300`, `narrative = description`).

The sibling builders `_support_html` (line 252) and `_feedback_html` (line 366) **do** escape `<`/`>`/newlines; `_event_html` does not. Title is from a fixed allow-list dict (safe) and location is the caregiver-set room name, but `narrative` is model-generated, untrusted output rendered as raw HTML in an email.

**Why it matters:** The skill treats LLM output as untrusted and requires sanitizing before HTML rendering. A scene description influenced by what the camera sees (or, in edge cases, by a caregiver's own custom-detector prompt) could embed markup/links into the alert email. Impact is limited (most mail clients strip scripts; recipients are the caregiver's own configured addresses), hence Low — but it is an inconsistency with the codebase's own escaping pattern.

**Remediation:** Apply the same `esc()` helper used by `_feedback_html` to `narrative` (and defensively to `location`) in `_event_html`.

### 7. Static, reusable, never-expiring demo pairing code

**Location:** `src/api/pairing.py:59-61` and `_pair_demo_camera` (`:101`); setting `demo_pairing_code` (`src/settings.py:137`).

When `NEMARA_DEMO_PAIRING_CODE` is set (production only, for App Review), `POST /v1/cameras/pair` special-cases that exact code to pair the seeded demo camera **without expiry or single-use clearing**, re-minting a fresh `api_key` on every call.

**Why it matters:** Anyone who learns the demo code can repeatedly pair to the demo camera and obtain LiveKit tokens for the demo patient's room. Blast radius is limited to the demo/review fixture (which should hold no real patient data), so Low — provided the demo patient is never connected to a real home.

**Remediation:** Keep the demo code high-entropy and rotate it after each review cycle; ensure the demo camera/patient is fully isolated from production data; consider disabling the path outside active review windows.

### 8. Caregiver-/peer-supplied Pi URLs stored (latent SSRF)

**Location:** `src/api/cameras.py:265` (`provision_pi_camera`, direct-URL mode stores `pi_stream_url`/`pi_audio_url`) and the announce-supplied `current_url` (Finding 5).

These URLs were originally fetched server-side in the old MJPEG pull design. The current architecture is push-based (`src/api/cameras_ingest.py`), so the stored URLs are vestigial and **not** actively fetched by the relay — no live SSRF today.

**Why it matters:** The values are externally influenced and persisted with only a length cap (`schema.sql:266`), no scheme/host validation. If any pull-mode or preview/health-check feature is reintroduced that dereferences them, this becomes a server-side request forgery vector against internal services.

**Remediation:** Validate these URLs at write time (allowed scheme, reject internal/link-local/metadata addresses) even while unused, so a future fetch path can't silently turn them into SSRF. Drop the columns if they are truly dead.

---

## Areas Reviewed With No Findings (positive controls)

- **Supabase RLS** (`supabase/schema.sql:664-754`): every table has RLS enabled with owner-scoped `USING` **and** `WITH CHECK`; no `USING (true)` or `auth.uid() IS NOT NULL` policies; sensitive server-managed columns (`patients.state`, etc.) are excluded from the caregiver-writable Pydantic models, preventing privilege/field tampering. The `SECURITY DEFINER` email-sync trigger sets a fixed `search_path` and only mirrors email.
- **Application-layer authorization (IDOR):** every caregiver endpoint resolves resources through `*_repo.find_by_id(..., caregiver_id=auth.id)` / caregiver-scoped `UPDATE`/`DELETE` (`src/repos/patient_repo.py`, and uniformly across patients/events/cameras/detectors/devices). Cross-account access returns 404, not 403. Account deletion uses the caller's own `auth.id` (no IDOR). No mass-assignment: request bodies are constrained Pydantic models, not spread into ORM writes.
- **Authentication:** JWTs verified with signature + audience (`aud="authenticated"`) and `exp` (`src/services/auth.py`); no `jwt.decode()`-without-verify; opaque 401s avoid oracle leakage.
- **LiveKit:** token minting enforces room/identity match for cameras and patient-ownership for caregivers (`src/api/livekit.py`); webhook receiver verifies the signed `Authorization` JWT over the raw body before acting (`src/api/livekit_webhooks.py`).
- **Secrets handling:** camera API keys stored bcrypt-hashed; Pi ingest tokens stored SHA-256; Supabase `service_role` key used **only** server-side (storage uploads, GoTrue admin) — never exposed to clients; no `NEXT_PUBLIC_`/`VITE_`/`EXPO_PUBLIC_` client-prefix leakage; current working tree contains no `.env`.
- **Injection / input validation:** all DB access via SQLAlchemy Core/ORM with bound parameters — no string-built SQL, no `$queryRawUnsafe` equivalent. Uploads are size-capped via streaming reads (`patients.py:_read_capped_upload`) and content-type allow-listed (`storage.py`). Feedback screenshot base64 is validated before forwarding.
- **AI/LLM:** provider keys are server-side only; user input goes into structured prompt fields, not concatenated into system instructions; LLM output is consumed as structured JSON against a fixed schema (the only gap is HTML-escaping at the email boundary — Finding 6). Custom-detector prompt injection is self-scoped (a caregiver can only affect their own patient's detection).
- **Deployment:** production `CMD` runs uvicorn without `--reload`/debug (`--reload` appears only in the dev `Makefile`); `service_role` confined to backend; `.gitignore` covers `.env*` and model weights. (CORS/security headers are not configured, but the service is consumed by native iOS/Pi clients rather than a browser SPA on this origin, so their absence is low-impact — recommend adding HSTS and a restrictive CORS allow-list if any browser client is introduced.)
- **Payments:** no Stripe/payment code present — section not applicable.
- **Dependencies:** locked versions are current (fastapi 0.136.1, starlette 1.0.1, pillow 12.2.0, python-multipart 0.0.29, urllib3 2.7.0, cryptography 48.0.0, pyjwt via crypto extra). No known-vulnerable pinned versions were identified in `uv.lock`.

---

## Prioritized Remediation Summary

1. **(Critical)** Rotate the APNs key + OpenAI/Gemini/LiveKit/Resend keys that were committed in `.env`, purge the file from git history, and add `gitleaks` to CI.
2. **(High)** Add rate limiting — start with `POST /v1/cameras/pair`, `pi-announce`, and the LLM-cost detector-create path; set hard AI-provider spend caps.
3. **(Medium)** Replace the `camera_auth` linear bcrypt scan with a prefix-indexed single-verify; bound executor concurrency.
4. **(Medium)** Restrict JWT verification to ES256/JWKS in production; gate HS256 to local/test only.
5. **(Medium)** Authenticate/scope Pi discovery and validate announced URLs.
6. **(Low)** HTML-escape `narrative` in alert emails; rotate/isolate the demo pairing code; validate stored Pi URLs.

---

*This report is based solely on static analysis of the source at audit time. It does not assess runtime configuration, the production Supabase project's actual RLS state, deployed environment variables, or whether the historically-committed secrets were in fact rotated — those require live verification.*
