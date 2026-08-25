# Security Audit — nemara-backend

**Date:** 2026-08-07
**Scope:** Static source audit of the `nemara-backend` FastAPI + Supabase service (caregiver monitoring backend: patients, cameras, face embeddings, LiveKit A/V, LLM detection pipeline).
**Method:** Static analysis only — no live systems, databases, or secrets were accessed. Audit framework: the `vibe-security` skill (secrets & env, database/RLS access control, authentication, data access, rate limiting, payments, deployment, AI/LLM, mobile).
**Commit audited:** `7f325e0` (branch `main`).

---

## Executive Summary

The application-layer security of this backend is, on the whole, **well built**. Every patient-scoped API route enforces authorization at the repository layer via membership-scoped queries (`accessible_patient_ids`), cross-account access consistently returns `404` rather than `403`, all 14 Postgres tables have Row-Level Security enabled, JWT verification uses `jwt.verify` with audience checks, LiveKit token minting validates room ownership, raw SQL is uniformly parameterized, subprocess calls avoid the shell, and no unsafe deserialization or path traversal was found. Dependencies are pinned at non-vulnerable versions.

The serious problems are **not** in the request-handling logic — they are in **secret hygiene** and the **absence of any rate limiting**.

- **CRITICAL:** Live production and service credentials — including the **production LiveKit API secret** (which mints tokens into patients' live audio/video rooms), **two Apple APNs push-signing private keys**, and OpenAI / Gemini / OpenRouter / Resend API keys — remain **recoverable from git history**. Two `.env`-family files were committed and later untracked, but their blobs are still reachable from `refs/heads/main`, so every clone carries them. **These must be rotated now and purged from history.**
- **HIGH:** A **working production caregiver account password** is hardcoded in tracked docs/scripts/README.
- **HIGH:** The unauthenticated 6-digit **camera-pairing endpoint is brute-forceable** with zero rate limiting — a path to hijacking a stranger's home monitoring camera and their patient's LiveKit room.
- **MEDIUM:** An unauthenticated, cross-tenant `pi-announce` endpoint (device-picker injection, one-time ingest-token theft, latent SSRF); and no rate limits or cost caps on authenticated LLM endpoints (billing drain).

Immediate priorities, in order: **(1)** rotate every credential listed in Finding 1 and purge the blobs from git history; **(2)** rotate/disable the seeded production account in Finding 2; **(3)** add rate limiting to the pairing and LLM/ingest surfaces (Findings 3–5).

### Severity Tally

| Severity | Count | Findings |
|----------|-------|----------|
| Critical | 1 | Secrets in git history |
| High | 2 | Hardcoded prod password; pairing-code brute force |
| Medium | 2 | Unauthenticated `pi-announce` (injection/token-theft/latent SSRF); no LLM rate limits |
| Low | 5 | HS256 accepted in prod; docs exposed; unauth freshness probe; unenforced share roles; self-scoped prompt injection |

---

## Critical

### 1. Live production & service secrets recoverable from git history

**Where:** Blobs reachable from `refs/heads/main`:
- `.env` — blobs `d3fadf5` / `98a893e` (added in `59f62fc`, later untracked)
- `.env.bak.1783536884` — blob `5aa7d5a` (added in `f1c0564`, later untracked)

Confirmed reachable: `git rev-list --objects refs/heads/main | grep -E 'd3fadf5|98a893e|5aa7d5a'` returns all three.

**Exposed credentials (real-format, high-entropy, live):**

From the `.env` blob:
- `NEMARA_GEMINI_API_KEY` (Google AI Studio)
- `NEMARA_OPENAI_API_KEY` (`sk-proj-…` OpenAI project key)
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` (dev LiveKit project)
- `NEMARA_APNS_AUTH_KEY` — full `-----BEGIN PRIVATE KEY-----` Apple push-signing key
- `NEMARA_RESEND_API_KEY` (transactional email)

From the `.env.bak.1783536884` blob:
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` — **production** LiveKit project (`nemara-pdx0uelu`, confirmed production in `docs/guides/setup-hosted-stack.md`)
- `NEMARA_APNS_AUTH_KEY` — a **second** EC push-signing private key
- `NEMARA_OPENROUTER_API_KEY`
- Same Gemini / OpenAI / Resend keys.

**Why it's exploitable:**
- The **production LiveKit secret** allows an attacker to mint arbitrary access tokens and **join patients' live audio/video rooms** (identity/grant of their choosing) — a direct, severe privacy breach of a vulnerable-person monitoring system.
- The **APNs private keys** allow forging push notifications to the caregiver app under this Apple team.
- The **LLM keys** (OpenAI/Gemini/OpenRouter) allow unlimited inference billed to the owner's accounts.
- Deleting a file from the working tree does **not** remove it from history; `.gitignore:12` asserts the keys were "rotated," but that is unverifiable statically and does not remove them from clones.

> Note: the Supabase values in the same blobs (`NEMARA_SUPABASE_JWT_SECRET=super-secret-jwt-token-with-at-least-32-characters-long` and the `iss:"supabase-demo"` service_role JWT) are the **well-known Supabase CLI local defaults** — not real secrets. Loopback DB DSNs (`postgres:postgres@127.0.0.1`) are dev defaults, also not findings.

**Remediation:**
1. **Rotate now** — treat as compromised and reissue: both LiveKit projects (esp. production), **both** APNs signing keys, OpenAI, Gemini, OpenRouter, and Resend keys.
2. **Purge from history** — `git filter-repo --invert-paths --path .env --path .env.bak.1783536884` (or BFG), then force-push and have all collaborators re-clone. Rotation without purge still leaves the old keys retrievable (and re-usable if any rotation was missed).
3. Add a pre-commit secret scanner (`gitleaks`) to prevent recurrence.

---

## High

### 2. Working production caregiver password hardcoded in tracked files

**Where:**
- `scripts/seed/seed.py:48` — `DEMO_PASSWORD = "Mar9aret!2024"` (comment claims "local-dev only; never used in prod")
- `docs/guides/setup-hosted-stack.md:74` — `password: Mar9aret!2024` for `jane@patel.co`
- `README.md:26`, `README.md:106` — same credential pair for the web dashboard and iOS app

**Why it's exploitable:** The setup guide states this account was **seeded into production Supabase** and is the working sign-in for the production web dashboard and caregiver iOS app, owning a real patient record. This contradicts the `seed.py` "local-only" comment. The credential is in the **current tree** (not just history) and grants access to production caregiver/patient data.

**Remediation:** Change (or disable/delete) the production `jane@patel.co` account immediately. Stop committing the plaintext — read `DEMO_PASSWORD` from an environment variable in `seed.py`, and scrub the value from the docs and README (reference an env var instead).

### 3. Unauthenticated pairing-code brute force → camera & patient hijack

**Where:** `src/api/pairing.py:45` (`POST /v1/cameras/pair`); code generation `src/api/cameras.py:189`; TTL `src/settings.py:385` (`pairing_code_ttl_seconds`, default **600 s**).

**Why it's exploitable:** The endpoint is **unauthenticated** ("the code IS the proof of authorization"). A valid 6-digit code is exchanged for a long-lived camera `api_key` plus the camera's `patient_id`, `livekit_room`, and `livekit_identity`. The code space is only **900,000** values (`secrets.randbelow(900_000) + 100_000`), each code lives **10 minutes**, and there is **no rate limiting anywhere in the codebase** (confirmed: no limiter/throttle in `src/main.py` or `src/api/`). An attacker can pound `/v1/cameras/pair` at high volume; against the pool of codes active at any moment, the probability of landing a valid pairing is real and grows with concurrency. A single success yields:
- the camera `api_key` → authenticate as that camera on every camera-scoped endpoint;
- the `patient_id` + `livekit_room` → with a `role='camera'` token mint (`src/api/livekit.py`), join the patient's live A/V room.

This directly compromises the **live video/audio feed of a monitored patient** (elderly / at-risk individuals) — a severe confidentiality and safety breach.

Related: `src/settings.py:392` `demo_pairing_code` — when set, `POST /v1/cameras/pair` special-cases a **permanent, reusable, never-expiring** code (`src/api/pairing.py:101`). It is off by default and only pairs the seeded demo camera, but if ever set in production it is a standing unauthenticated pairing credential. Keep it unset in production.

**Remediation:**
- Add strict rate limiting to `/v1/cameras/pair` — per-IP **and** a global cap on failed attempts, plus exponential backoff / temporary lockout after a small number of misses.
- Consider widening the code (e.g. 8–10 digits or alphanumeric) and/or single-attempt-per-code semantics with a short lockout, so the code space cannot be meaningfully swept within the TTL window.
- Ensure `NEMARA_DEMO_PAIRING_CODE` is never set in production.

---

## Medium

### 4. Unauthenticated, cross-tenant `pi-announce` — picker injection, ingest-token theft, latent SSRF

**Where:**
- Write (no auth): `src/api/pi_discovery.py:44` (`POST /v1/cameras/pi-announce`). Body fields `hardware_id`, `device_label`, `current_url`, `current_audio_url` are attacker-controlled.
- URL validation is **length-only**: `src/schemas/pi_discovery.py:37` and `src/schemas/camera.py:36-41` (`min_length=8, max_length=2048`, no scheme/host allow-list; `camera.py:32-35` explicitly defers validation to "connect time").
- Read with **no per-caregiver scoping**: `src/repos/pi_discovery_repo.py:77-106` (`list_unclaimed_recent`) returns all unclaimed discoveries; `src/api/pi_discovery.py:97-104` documents "no per-caregiver filtering at v1."

**Why it's exploitable:**
1. **Cross-tenant picker injection (present today).** Any anonymous internet client can flood **every** caregiver's "Add a camera" list with fake devices bearing arbitrary `device_label`/`current_url` — a phishing / UI-spoofing vector across all tenants.
2. **One-time ingest-token theft (present today).** `announce_pi` calls `consume_pending_ingest_token` keyed **only** by `hardware_id` (a MAC address, guessable/observable). An attacker who announces a victim Pi's `hardware_id` just after the caregiver provisions it can **consume the plaintext ingest token** intended for the real device (`src/api/pi_discovery.py:74`), then push arbitrary frames into the patient's camera stream — or deny the real Pi its token.
3. **Latent SSRF.** The fetch path is fully present — `src/services/ingest/pi/relay_publisher.py:239` → `read_mjpeg(pi_stream_url)` (`mjpeg_reader.py`, plain `httpx` GET on the raw URL) and `:435` → `read_pcm_audio(pi_audio_url)`. Today the only live `RelayConfig` uses push mode with `pi_stream_url=""` (`src/api/cameras_ingest.py:177`), so the server does not currently fetch these URLs. The moment pull-mode ingest is re-enabled, an attacker-chosen URL stored via provision becomes a full **unauthenticated SSRF** (cloud metadata `169.254.169.254`, `localhost` services, RFC1918).

**Remediation:**
- Authenticate/attribute `pi-announce` (per-device secret negotiated at provisioning, as the module docstring's "v2" already anticipates), or at minimum rate-limit it and scope `/discovered` to the caregiver's own network/devices.
- Bind the pending ingest token to more than a guessable `hardware_id` (e.g. a provisioning nonce).
- Validate `current_url`/`pi_stream_url`/`pi_audio_url` to `http(s)` + **public** IPs only (block loopback, link-local `169.254.0.0/16`, and RFC1918) at the schema **and** again at connect time — before pull mode is ever restored.

### 5. No rate limit or cost cap on authenticated LLM endpoints (billing DoS)

**Where:**
- `src/api/briefing.py:61` (`POST /v1/home-briefing`) — one OpenAI call per request.
- `src/api/home_report.py:92` (`POST /v1/home-report`) — one OpenAI call + up to 4 base64 images per request.
- `src/api/detectors.py` — custom-detector create (`:614+`), edit/recompile (`:856+`), plus `/infer` and `/preview` each fire paid Gemini + OpenAI compile (`src/services/detector_compiler.py`) and a Gemini summary.

**Why it's exploitable:** All require `CaregiverAuth`, but there is **no rate-limiting middleware and no per-user token/cost quota anywhere**. A single authenticated caregiver (or a compromised account) can loop any of these endpoints to drive **unbounded paid inference**, draining the OpenAI/Gemini/OpenRouter billing accounts. Per the methodology, AI endpoints "almost never" get rate limits and a single user can exhaust a monthly budget in minutes.

**Remediation:** Add a per-caregiver rate limit and a daily/monthly token budget on these endpoints (e.g. Upstash Redis or a private-schema counter, not a public table). Debounce detector recompiles. Set hard spending caps in the OpenAI/Anthropic/Google consoles as a backstop.

> **Systemic root cause (Findings 3–5):** the service ships **no rate limiting of any kind**. Auth endpoints are covered because login is delegated to Supabase, but every locally-exposed sensitive surface (pairing, ingest/announce, LLM proxy, expensive image uploads) is unthrottled. A shared rate-limiting layer (edge/middleware + per-user quotas) addresses all three.

---

## Low

### 6. HS256 JWTs accepted in production alongside ES256

**Where:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), dispatched at `:72-79`.

**Why it matters:** Verification correctly selects the key by the token's `alg` (ES256 → JWKS public key; HS256 → `supabase_jwt_secret`), so there is **no algorithm-confusion vulnerability**. However, HS256 is accepted in **all** environments (the code comments call it "legacy / local-dev"). Accepting a shared-secret algorithm in production widens the forgery surface: anyone who obtains `NEMARA_SUPABASE_JWT_SECRET` can forge any caregiver token, whereas the ES256/JWKS path requires the (non-exportable) Supabase signing key.

**Remediation:** Restrict `algorithms` to `["ES256"]` when `env in ("staging","prod")` and keep HS256 only for `local`/`test`.

### 7. Interactive API docs and OpenAPI schema publicly exposed

**Where:** `src/main.py:205` — `FastAPI(title=..., version=..., lifespan=...)` with no `docs_url`/`redoc_url`/`openapi_url` overrides, so `/docs`, `/redoc`, and `/openapi.json` are served unauthenticated. (The access-log filter at `src/main.py` only *silences logging* for these paths; it does not disable them.)

**Why it matters:** Publishes the complete API surface (every route, schema, and auth shape) to anonymous clients, easing reconnaissance.

**Remediation:** Disable docs in production (`docs_url=None, redoc_url=None, openapi_url=None` when `env in ("staging","prod")`), or gate them behind the internal-key / an authenticated proxy.

### 8. Unauthenticated data-freshness probe leaks camera inventory

**Where:** `src/api/health.py:192` (`GET /health/freshness`).

**Why it matters:** Returns per-camera id prefixes plus each online camera's analysis-freshness/age with **no authentication**, while its sibling `/ready` (`:53`) is `X-Internal-Key`-gated. Leaks camera inventory and operational timing to anonymous callers.

**Remediation:** Apply the same `X-Internal-Key` gate as `/ready`, or strip identifiers from the unauthenticated response.

### 9. Shared "viewer"/"caregiver" roles grant full access

**Where:** `src/api/patient_members.py` (share/list), `src/repos/access.py` (`has_patient_access` / `accessible_patient_ids`).

**Why it matters:** `role` is stored but **unenforced** — every membership row grants full read/write access (documented as an MVP gap). A caregiver who shares a patient as `viewer` believing it is read-only actually grants full control, including the ability to modify or delete the patient's data.

**Remediation:** Either enforce `viewer` as read-only in the repo/authorization layer, or remove `viewer` from `_SHAREABLE_ROLES` until per-role enforcement lands, so the granted access matches the label.

### 10. Self-scoped prompt injection (hardening)

**Where:** Caregiver free-text into LLM prompts — custom-detector `custom_prompt` → `src/services/detection/prompt.py:160-161`; `src/api/briefing.py:20-47`; `src/api/home_report.py:36-71`. Camera video/audio is likewise untrusted VLM input.

**Why it's low:** Injection stays within a single caregiver's own data producing their own output — no privilege boundary is crossed. Output is **not** executed, **not** rendered as HTML, and **not** used to build SQL/shell; it is schema-validated (`providers/_parse.py`), length/enum-capped, and detection identifiers are matched against a known `firing_ids` allow-list (`prompt.py:180-193`). The `_PROMPT.format(...)` calls use a constant template with user values as arguments (no `{}`/format-string injection).

**Remediation (defense-in-depth):** Wrap untrusted fields in explicit delimiters and instruct the model to treat them strictly as data; continue validating all model output before persistence.

---

## Areas Reviewed and Found Clean

- **Row-Level Security:** all 14 `public` tables have RLS enabled; patient-scoped policies use membership via `has_patient_access` (a `SECURITY DEFINER` function with `search_path` pinned); `gateways`/`pi_discoveries` additionally `REVOKE ALL` from `anon, authenticated`. Policies include `WITH CHECK`; no `USING (true)` / `USING (auth.uid() IS NOT NULL)` on user tables. Note: the backend connects via a direct asyncpg DSN (RLS-bypassing), so RLS protects the iOS Supabase-Realtime path; the load-bearing check for the API is the repo layer — which is consistently membership-scoped.
- **Authorization (IDOR):** every patient/camera/event/embedding route filters by `caregiver_id` / `accessible_patient_ids` and returns `404` on cross-account access; `patient_members` enforces owner-only share/remove and a last-owner guard; LiveKit token minting validates camera room/identity match and caregiver room membership; Talk/record-clip verify camera ownership.
- **JWT:** `jwt.decode` with signature verification, `audience="authenticated"`, expiry enforced, `sub`/`email` claim validation; opaque uniform `401` on all failures.
- **SQL injection:** all raw `text()` SQL uses bound parameters; the one f-string (`src/api/cameras_ingest.py:393`) interpolates a static local literal, with request data still passed as bound params. ORM queries parameterized throughout.
- **Command/shell injection:** no `shell=True`, `os.system`, `os.popen`, `eval`, or `exec`; all subprocesses use argv lists (`create_subprocess_exec` / `Popen` with lists); ffmpeg args are internal temp paths and numeric geometry.
- **Path traversal:** storage keys built from typed UUIDs + allow-listed content-type extensions; temp files via `tempfile` with fixed suffixes; `UploadFile.filename` never used to build a path; uploads are size/content-type validated and streamed with a byte cap.
- **Unsafe deserialization:** no `pickle`, `yaml.load`, or `torch.load` of untrusted data; ML models load from fixed ids/paths.
- **AI keys:** Gemini/OpenAI/OpenRouter/DashScope keys are server-side (`Settings`) only; never returned to clients; images passed as inline base64, not URLs. Outbound HTTP targets are all settings-derived hosts (no user-controlled fetch in live paths).
- **Dependencies:** resolved pins in `uv.lock` sit at/above the fixed version for every relevant CVE checked — `python-multipart 0.0.29` (CVE-2024-53981), `jinja2 3.1.6`, `pyjwt 2.13.0`, `requests 2.34.2`, `urllib3 2.7.0`, `starlette 1.0.1`, `torch 2.13.0` (CVE-2025-32434), `onnx 1.18.0`, `pillow 12.2.0`. No confident CVE match. (Several pins post-date the Jan 2026 knowledge cutoff and could not be checked against newer advisories.)
- **`.gitignore` / current tree:** correctly ignores `.env`, `.env.local`, `.pod.env`, `.env.bak*`; no real env file is currently tracked; `.env.example` / `.pod.env.example` contain only placeholders. (The exposure in Finding 1 is entirely historical.)

---

## Prioritized Remediation Checklist

1. **[Critical]** Rotate all credentials in Finding 1 (production LiveKit secret, both APNs keys, OpenAI, Gemini, OpenRouter, Resend); purge `.env` and `.env.bak.1783536884` from git history and force-push; add `gitleaks` pre-commit.
2. **[High]** Change/disable the seeded production account `jane@patel.co`; move `DEMO_PASSWORD` to an env var; scrub it from docs/README.
3. **[High]** Rate-limit `/v1/cameras/pair` (per-IP + global failed-attempt cap + lockout); widen the code space and/or shorten reuse; keep `NEMARA_DEMO_PAIRING_CODE` unset in prod.
4. **[Medium]** Authenticate/attribute `pi-announce`, scope `/discovered` per caregiver, bind the ingest token to a provisioning nonce, and SSRF-filter Pi URLs before pull mode is re-enabled.
5. **[Medium]** Add per-caregiver rate limits and token budgets to the LLM endpoints (`/home-briefing`, `/home-report`, detector compile/infer/preview); set provider-side spending caps.
6. **[Low]** Restrict JWT `algorithms` to ES256 in prod; disable `/docs` `/redoc` `/openapi.json` in prod; gate `/health/freshness`; enforce or remove the `viewer` share role; add delimiters around untrusted LLM inputs.

---

*Audit performed statically against commit `7f325e0`. No live systems were accessed. Line references are to the state of the tree at that commit.*
