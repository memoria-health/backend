# Security Audit — nemara-backend

**Date:** 2026-07-22
**Scope:** Static source audit of `nemara-backend` (FastAPI + Supabase backend for the Memoria/Nemara caregiver patient-monitoring product).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill — secrets & env, database access control (Supabase RLS), authentication/authorization, injection/SSRF, rate-limiting, payments, deployment, AI/LLM integration.
**Basis:** Source code only. No live systems, databases, or secrets were accessed. Findings are limited to what is provable from the committed code.

---

## Executive summary

The application's core authorization design is **good**: patient-scoped data access is enforced through a membership model (`patient_members` + `has_patient_access()`), every user-facing REST endpoint requires `CaregiverAuth`, DB queries are consistently scoped by `caregiver_id`/`accessible_patient_ids(...)`, Supabase RLS is enabled on every table with owner/membership policies, all SQL goes through SQLAlchemy (parameterized), LiveKit tokens are minted with per-room ownership checks, webhook signatures are verified, and camera/Pi bearer tokens are hashed with constant-time comparison. This is not a typical "no auth anywhere" vibe-coded backend.

The audit nonetheless found one **Critical** issue that dominates risk: a credentials file (`.env.bak.1783536884`) containing real-looking third‑party API keys and an Apple push private key was committed to the repository. Because this is a healthcare product handling live audio/video and PHI of vulnerable patients, credential exposure and any authentication weakness carry elevated impact.

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | **Critical** | Production credentials committed in `.env.bak` (OpenAI, Gemini, OpenRouter, LiveKit secret, Resend, APNs private key) | `.env.bak.1783536884` |
| 2 | **High** | JWT verifier accepts HS256 (shared symmetric secret) alongside ES256 — forgeable session if the HS256 secret is weak/known | `src/services/auth.py:26,83` |
| 3 | **Medium** | Unauthenticated `pi-announce` write + globally-unscoped discovery list (poisoning, provisioning hijack, cross-customer disclosure) | `src/api/pi_discovery.py:43`, `src/api/cameras.py:316` |
| 4 | **Medium** | No rate limiting anywhere; `camera_auth` does a per-request bcrypt scan of all camera rows (CPU-DoS amplifier); email endpoints allow authenticated spam | `src/api/deps.py:137`, `src/api/support.py`, `src/api/patients.py:220` |
| 5 | **Low** | Alert/test emails render untrusted VLM/DB text as unescaped HTML | `src/services/email.py:124` |
| 6 | **Low** | Debug logging + `NEMARA_LOG_LEVEL=DEBUG` shipped in committed env; auth logs caregiver IDs | `.env.bak.1783536884:6`, `src/api/deps.py:89` |
| 7 | **Low** | No HTTP security headers (HSTS, X-Content-Type-Options, etc.) | `src/main.py:167` |

**Top priorities:** (1) Purge `.env.bak` from the working tree **and git history**, and rotate every third-party key it contains. (2) Restrict production JWT verification to ES256 only.

---

## Critical

### 1. Production credentials committed in `.env.bak.1783536884`

**File:** `.env.bak.1783536884` (tracked in git — confirmed via `git ls-files`; added in commit `f1c0564`, "feat(sharing): membership-based backend authorization + share endpoints").

A full environment file with real-format secrets is committed to the repo:

| Line | Secret | Assessment |
|------|--------|------------|
| 23 | `NEMARA_GEMINI_API_KEY=AIzaSy…` | Live Google API key format |
| 28 | `NEMARA_OPENAI_API_KEY=sk-proj-…` | Live OpenAI project key format |
| 90 | `NEMARA_OPENROUTER_API_KEY=sk-or-v1-…` | Live OpenRouter key format |
| 34–35 | `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` | LiveKit Cloud API key + secret (paired with real host `nemara-pdx0uelu.livekit.cloud`) |
| 80 | `NEMARA_RESEND_API_KEY=re_…` | Live Resend key format |
| 60–65 | `NEMARA_APNS_AUTH_KEY=-----BEGIN PRIVATE KEY-----…` | Apple APNs ES256 (.p8) **private key** |
| 19–20 | `NEMARA_SUPABASE_JWT_SECRET`, `NEMARA_SUPABASE_SERVICE_ROLE_KEY` | These two are the **well-known Supabase local-demo defaults** (`iss: supabase-demo`, the canonical `super-secret-jwt-token-…` string) — not secret in themselves, but see finding #2 for why their *pattern of use* is dangerous |

**Why it is exploitable.** Anyone who obtains the repository (a leaked clone, a former contributor, a future public/mirror push, a compromised laptop) gets working credentials to:
- **Drain billing** on OpenAI, Gemini, and OpenRouter — LLM keys allow unlimited spend; a leaked key can burn thousands of dollars quickly.
- **Access patient live media.** The LiveKit API key + secret let an attacker mint tokens for any `patient_<id>` room and subscribe to the live audio/video of monitored patients (elderly/dementia care) — a severe privacy breach.
- **Forge push notifications** to caregiver devices using the APNs private key (spoofed "fall detected"/"all clear" alerts).
- **Send mail as the org** via the Resend key.

`.gitignore` even documents that `.env` was *previously* tracked and "its leaked secrets rotated" — the `.env.bak` copy re-introduced the exact exposure the team already had to remediate once. Deleting the file in a new commit is **not sufficient**: it remains in history.

**Remediation.**
1. **Rotate now**, in the provider dashboards: OpenAI, Gemini, OpenRouter, LiveKit (API key/secret), Resend, and the APNs auth key (`P34L5354FB` — revoke in the Apple Developer portal and generate a new `.p8`).
2. Remove the file and purge it from history:
   ```bash
   git rm --cached .env.bak.1783536884
   # then rewrite history (git filter-repo recommended) and force-push:
   git filter-repo --path .env.bak.1783536884 --invert-paths
   ```
3. Broaden `.gitignore` to cover backups: add `.env.bak*` and `.env.*` (keeping `!.env.example`).
4. Run `gitleaks detect` in CI to block future credential commits.
5. Store real secrets only in Railway's environment settings (never in the repo), as the team's own gitignore note already intends.

---

## High

### 2. JWT verification accepts HS256 with a shared symmetric secret

**File:** `src/services/auth.py:26`, `:76–88`

```python
JWT_ALGORITHMS = ["ES256", "HS256"]
...
elif alg == "HS256":
    signing_key = jwt_secret          # settings.supabase_jwt_secret
...
claims = pyjwt.decode(token, signing_key, algorithms=JWT_ALGORITHMS, audience=EXPECTED_AUDIENCE)
```

The verifier dispatches on the token's own `alg` header and, for `HS256`, verifies against the **symmetric** `supabase_jwt_secret`. Supabase 2.x issues **ES256** (asymmetric, JWKS) tokens in production, so the HS256 branch is legacy/local-dev support that remains enabled everywhere.

**Why it is exploitable.** HS256 uses the *same* secret to sign and verify. Any party that knows `supabase_jwt_secret` can mint a token with an arbitrary `sub` (caregiver ID) and `aud="authenticated"`, and the backend will accept it — a complete authentication bypass allowing impersonation of **any** caregiver and access to **any** patient's data, cameras, and clips. The committed `.env.bak` (finding #1, line 19) shows the team runs with the **publicly-known default** Supabase JWT secret; if that same value (or any weak value) is ever used for the deployed backend, forgery is trivial. Even absent a known secret, keeping a symmetric fallback alongside JWKS needlessly widens the attack surface.

*Note:* This is not classic "alg confusion" (the code correctly does not feed the ES256 public key into HMAC verification), so it is not unconditionally exploitable — the risk is conditional on the HS256 secret's strength/exposure. Given #1, that condition is realistic.

**Remediation.** In non-local environments, verify **ES256 only**:

```python
# Before
JWT_ALGORITHMS = ["ES256", "HS256"]

# After — gate HS256 to local/test; production accepts only asymmetric tokens
JWT_ALGORITHMS = ["ES256"] if settings.env in ("staging", "prod") else ["ES256", "HS256"]
```

and reject any token whose header `alg` is not in the allowed list before selecting a key. Ensure the production `supabase_jwt_secret` is never the demo default.

---

## Medium

### 3. Unauthenticated Pi announce + globally-unscoped discovery list

**Files:** `src/api/pi_discovery.py:43` (`POST /v1/cameras/pi-announce`), `src/api/pi_discovery.py:93` (`GET /v1/cameras/discovered`), `src/api/cameras.py:316`

`pi-announce` is intentionally unauthenticated (a Pi has no credentials pre-pairing — documented in `supabase/schema.sql:327`). It upserts a `pi_discoveries` row keyed by attacker-chosen `hardware_id`, with attacker-controlled `device_label` and `current_url`. `GET /v1/cameras/discovered` then returns **all** unclaimed recent discoveries to **every** authenticated caregiver with no per-caregiver scoping.

**Why it matters.**
- **Discovery poisoning / DoS of onboarding:** anyone who can reach the backend can flood the "Add a camera" picker with bogus devices, or overwrite a legitimate device's `hardware_id` row (`current_url`, `device_label`) since the upsert key is caller-supplied.
- **Cross-customer information disclosure:** every caregiver sees every other customer's device labels and home LAN URLs.
- **Provisioning hijack / latent SSRF seed:** on provision, `pi_stream_url = discovery.current_url` (`cameras.py:316`) — an attacker-supplied URL is copied onto the camera row. The current architecture is push-mode so this URL is not actively fetched today (the pull path in `relay_publisher.py` is not wired from provisioning), but the value is attacker-controlled and any future re-activation of pull mode would become an unauthenticated-seeded SSRF against internal metadata endpoints.

**Remediation.** Bring forward the planned per-device secret (schema comment references a v2 BLE-negotiated secret). Short-term: rate-limit `pi-announce` per IP; validate `current_url` against an allowlist of RFC1918/link-local **denies** (block `169.254.0.0/16`, loopback, cloud metadata IPs) before storing; and scope `discovered` results so a caregiver only sees devices on their own network/tenant once tenancy exists.

### 4. No rate limiting; `camera_auth` bcrypt table-scan is a CPU-DoS amplifier

**Files:** `src/api/deps.py:111–159`, `src/api/support.py`, `src/api/patients.py:220`

There is no rate limiting anywhere in the service (no middleware, no Redis/Upstash, no per-IP/per-user throttle). Two concrete consequences:

- **bcrypt fan-out (`deps.py:137`):** `camera_auth` runs `SELECT * FROM cameras WHERE api_key_hash IS NOT NULL` and then `bcrypt.checkpw` against **every** row until a match. A single request with a bogus bearer token forces a bcrypt comparison (deliberately expensive, `rounds=10`) against *all* cameras. As the fleet grows, an unauthenticated attacker hammering any camera-authed route (`/v1/cameras/me/*`, `/v1/livekit/token`) multiplies CPU cost per request — a cheap asymmetric DoS. The code comments acknowledge the linear scan but ship no limit in front of it.
- **Authenticated email spam (`support.py`, `patients.py:220` test-email, `feedback.py`):** a logged-in caregiver can call `/v1/support` and `/v1/patients/{id}/test-email` in a loop; each triggers a Resend send with no throttle.

**Remediation.** Add per-IP + per-user rate limiting (Upstash Redis or an ASGI middleware) on all auth, token-mint, ingest, and email endpoints. For camera auth specifically, add an indexed `api_key_prefix` lookup (the code already anticipates this) so verification is O(1) instead of a full bcrypt scan.

---

## Low

### 5. Untrusted text rendered as unescaped HTML in alert emails

**File:** `src/services/email.py:124–135` (`_event_html`)

`title`, `narrative`, and `location` are interpolated into an HTML email body without escaping. These originate from the VLM detector pipeline (LLM output) and DB rows. Per the skill's guidance, **LLM output is untrusted**; it can contain markup. The support/feedback paths correctly escape (`_support_html`, `_feedback_html`), but the event path does not. Impact is limited (recipients are the caregivers themselves and mail clients sandbox HTML), hence Low — but it is an inconsistency worth closing.

**Remediation.** Apply the same `esc()` helper used in `_feedback_html` to `title`/`narrative`/`location` in `_event_html`.

### 6. Debug logging shipped in env; auth logs identifiers

**Files:** `.env.bak.1783536884:6` (`NEMARA_LOG_LEVEL=DEBUG`), `src/api/deps.py:89`

The committed env sets `DEBUG` logging, and `caregiver_auth` logs `caregiver_id` at debug level. Verbose logging in production risks leaking identifiers and internal detail into logs. Ensure production runs at `INFO`/`WARNING` and avoid logging principal IDs at accept time.

### 7. No HTTP security headers

**File:** `src/main.py:167` (`create_app`)

No middleware sets `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options`, `Referrer-Policy`, or a CSP. (Note: the absence of any CORS middleware is *safe-by-default* for a native-app API — browsers block cross-origin reads — so CORS is not a finding here.) For a JSON API the headers matter less than for a browser app, but HSTS and `nosniff` are cheap wins. Add a small headers middleware.

---

## Areas reviewed and found sound (no action needed)

These were checked against the skill's methodology and are implemented correctly — recorded so the team knows they were covered:

- **Supabase RLS** (`supabase/schema.sql:711–821`): RLS enabled on every table; policies scope to `has_patient_access()` (membership) or `auth.uid()` (owner); `INSERT/UPDATE` policies carry `WITH CHECK`; the `patients` write-check pins `caregiver_id = auth.uid()` so ownership can't be reassigned from the authed client; `has_patient_access` is `SECURITY DEFINER` with `SET search_path = public`. Sensitive server-managed columns (`state`, `status_line`) are excluded from the caregiver-writable Pydantic model (`extra="forbid"`), preventing mass-assignment.
- **Endpoint authorization / IDOR:** every read/write is scoped by `caregiver_id`; cross-account access returns 404 (no existence oracle). LiveKit token minting (`api/livekit.py`) enforces room-ownership for viewers and room/identity match for cameras. Owner-guarded sharing (`api/patient_members.py`) blocks non-owners and protects the last owner.
- **Injection:** all DB access uses SQLAlchemy Core/ORM with bound parameters; no string-concatenated SQL, no `$queryRawUnsafe`-equivalent. The one raw-SQL heartbeat uses a bound `:digest` parameter.
- **Payments:** no Stripe/payment surface exists in this backend — section not applicable.
- **Webhooks:** LiveKit webhook signatures are verified against the API secret before dispatch (`api/livekit_webhooks.py:65`).
- **Secrets in DB access:** the Supabase `service_role` key is used **server-side only** (`services/storage.py`, `services/auth_admin.py`) — correct; it never reaches a client. Storage object keys are derived from server-side UUIDs (no path traversal), and reads are gated behind short-lived signed URLs minted only after an ownership check.
- **Token handling:** camera and Pi ingest tokens are high-entropy, stored only as hashes (bcrypt / SHA-256), and compared in constant time (`hmac.compare_digest`).

---

## Dependencies

`pyproject.toml` pins reasonable **lower** bounds that already exclude some known CVEs — e.g. `python-multipart>=0.0.18` (>= the fix for the CVE-2024-53981 multipart DoS) and `pyjwt[crypto]>=2.10`. Exact resolved versions live in `uv.lock`; no specifically known-vulnerable pinned version was identified in this static pass. Recommendation: run `pip-audit`/`uv pip audit` (or Dependabot) in CI against the lockfile so newly-disclosed CVEs in the large ML dependency graph (torch, opencv, pillow, onnxruntime, ultralytics) surface automatically.

---

*Generated by an automated static security audit using the vibe-security methodology. Every finding cites a concrete file/line and is derived solely from committed source — no live systems were accessed. Validate fixes against your production configuration, particularly the values behind the leaked keys (finding #1) and the production JWT algorithm/secret (finding #2).*
