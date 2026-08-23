# Security Audit — nemara-backend

**Date:** 2026-06-28
**Scope:** Static source audit of the `nemara-backend` FastAPI service (Supabase Postgres + Auth, LiveKit, Gemini/OpenAI VLM detection, Raspberry-Pi/iPhone camera ingest, Resend email, APNs push).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill — secrets, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM, deployment, data access.
**Constraint:** STATIC ONLY. No live systems, databases, or secrets were accessed. Findings are derived from source, SQL migrations, and config.

---

## Executive Summary

nemara-backend is a generally **well-built** codebase by vibe-coded standards: every caregiver-facing API endpoint enforces `CaregiverAuth`, repository queries are consistently scoped by `caregiver_id` (defense-in-depth beyond RLS), cross-account access returns `404` (no existence leak), all DB access uses the SQLAlchemy ORM (no string-built SQL), every `ffmpeg`/`ffprobe` call uses `create_subprocess_exec` with an argument list (no shell), JWTs are verified with signature + audience checks (no bare `decode`), the Stripe-style price-trust class of bug is absent (no payment surface present), LiveKit webhooks verify signatures, and no hardcoded secrets are committed.

However, the audit found **one Critical issue** that breaks the database's primary access-control guarantee, plus a **High** safety-relevant authentication gap on the camera-ingest trust chain:

1. **CRITICAL — `pi_discoveries` table has Row-Level Security disabled.** It is the only `public`-schema table without RLS, contradicting the schema's own "Every table is RLS-on" comment. With no RLS and Supabase's default grants, anyone holding the public anon/publishable key can read, write, and delete every row — including the **plaintext one-shot Pi ingest bearer tokens** and every deployed device's MAC/label/URL across all customers.

2. **HIGH — Unauthenticated `pi-announce` discloses the plaintext ingest token with no device attribution.** An attacker who announces a *claimed* device's `hardware_id` (low-entropy MAC, also leaked via #1) harvests its ingest token and can then push arbitrary video/audio frames into a patient's LiveKit room — i.e. feed fabricated footage to, or suppress real alerts from, a life-safety dementia-monitoring system.

The remaining findings (Medium/Low) concern missing rate limiting and per-user AI spend caps, a guessable reusable demo pairing code, acceptance of HS256 JWTs in production, and missing security headers.

**Top priorities:** (1) Enable RLS on `pi_discoveries` (and stop storing the ingest token in plaintext); (2) add device-bound authentication to the Pi announce/ingest trust chain; (3) rotate any Pi ingest tokens issued while #1/#2 were live.

| # | Severity | Title |
|---|----------|-------|
| 1 | **Critical** | `pi_discoveries` table missing RLS — full read/write via public anon key |
| 2 | **High** | Unauthenticated `pi-announce` leaks ingest token / camera impersonation |
| 3 | Medium | No rate limiting or per-user AI spend cap on LLM + email endpoints |
| 4 | Medium | Guessable, reusable, unauthenticated demo pairing code mints valid api_keys |
| 5 | Low | JWT verifier accepts HS256 in production (shared-secret forgery surface) |
| 6 | Low | No security headers (HSTS, X-Content-Type-Options, etc.) set by the app |
| 7 | Low | `.env` was previously committed — confirm history scrub + key rotation |
| 8 | Low | Dependencies floor-pinned; no scheduled vulnerability scan of `uv.lock` |

---

## Critical

### 1. `pi_discoveries` table has RLS disabled — full read/write to anyone with the public anon key

**File:** `supabase/schema.sql:334` (table definition), absent from the RLS section `supabase/schema.sql:693`–`782`.

The schema enables RLS and defines an owner policy for all 11 other public tables (`caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`, `feedback`, `ui_events`). **`pi_discoveries` is the lone exception** — there is no `alter table public.pi_discoveries enable row level security;` and no policy anywhere in the file. The comment at `schema.sql:695` even asserts "Every table is RLS-on," which is false.

Why it's exploitable:
- `supabase/config.toml` exposes the `public` schema via PostgREST (`schemas = ["public", "graphql_public"]`).
- There are **no `GRANT`/`REVOKE` statements** in `schema.sql`, so Supabase's default grants to the `anon` and `authenticated` roles apply. With those grants present and RLS *off*, the table is fully reachable at `<SUPABASE_URL>/rest/v1/pi_discoveries` using the **publishable/anon key, which is public by design** (it ships in the iOS app and web bundle).
- The table holds, per `schema.sql:357`–`360`, `pending_ingest_token` — the **plaintext, one-shot ingest bearer token** a Pi uses to authenticate to `POST /v1/cameras/me/frames`. (Note the `cameras` table correctly stores only the SHA-256 hash in `ingest_token_hash`; this table leaks the plaintext.)

Concrete attacker actions with only the public anon key:
1. `GET /rest/v1/pi_discoveries?select=*` → dump every deployed device across all customers: `hardware_id` (MAC), `device_label`, `current_url` (MJPEG endpoint on the home LAN), `claimed_camera_id`, and any non-null `pending_ingest_token`.
2. Use a leaked `pending_ingest_token` to authenticate to `/v1/cameras/me/frames` (`src/api/cameras_ingest.py:121`–`133`) and **publish attacker-controlled video/audio as that camera** into the patient's LiveKit room, or poll `/v1/cameras/me/ingest-status`.
3. `PATCH`/`DELETE` rows: rewrite `current_url`/`device_label`, or wipe discovery records to disrupt pairing.

This is the single most common critical class of vibe-coded vulnerability (`references/database-security.md`: "A table without RLS is fully readable and writable by anyone with the anon key").

**Remediation:**
```sql
-- Enable RLS and lock the table down. Pi discovery is a backend-only
-- concern: the Pi reaches it via the unauthenticated FastAPI endpoint
-- (which uses the service-role key), and caregivers read the curated
-- list via GET /v1/cameras/discovered. No client needs direct table
-- access, so the safe policy is "no anon/authenticated access at all."
alter table public.pi_discoveries enable row level security;
-- With RLS on and no policy, anon/authenticated are denied; the
-- service-role key (used by the backend) bypasses RLS as intended.
revoke all on public.pi_discoveries from anon, authenticated;
```
Additionally: stop persisting the ingest token in plaintext. Store only its hash in `pi_discoveries` (mirroring `cameras.ingest_token_hash`) and deliver the plaintext exactly once in the announce response without retaining it. Run the catch-all loop from `references/database-security.md` in CI to assert RLS is enabled on every `public` table so a future table can't repeat this.

---

## High

### 2. Unauthenticated `pi-announce` returns the plaintext ingest token with no device attribution

**File:** `src/api/pi_discovery.py:43`–`90` (`POST /v1/cameras/pi-announce`); token consumed at `src/api/pi_discovery.py:74`–`82`.

`announce_pi` is intentionally unauthenticated (`pi_discovery.py:5`–`12`: the Pi has no credentials before pairing). On every announce it calls `consume_pending_ingest_token(...)` keyed solely by the client-supplied `hardware_id` and, if a token is pending, returns `ingest_token` (plaintext) + `ingest_url` in the response body. There is **no proof the caller is the real Pi** — the only identifier is the `hardware_id`, which is a MAC address: low-entropy, enumerable, surfaced to every caregiver via `GET /v1/cameras/discovered` (`pi_discovery.py:111`–`120`), and directly readable via Finding #1.

Exploit: after a caregiver provisions a Pi (which stages `pending_ingest_token`), an attacker who races the real device — or simply replays a known `hardware_id` — calls `pi-announce` and receives the one-shot token. They then authenticate to `POST /v1/cameras/me/frames` and inject arbitrary frames as that camera. Because this is a **safety-monitoring product for dementia patients**, the impact is not just data exposure: an attacker can feed fabricated "all calm" video to suppress real fall/wandering alerts, or inject content to trigger false alerts. The module docstring acknowledges the missing attribution as a "v1" tradeoff (`pi_discovery.py:8`–`12`), but the consequence (silent camera takeover on a life-safety system) is High.

**Remediation:** Establish a per-device secret at physical provisioning time (the captive-portal WiFi flow in `scripts/pi/wifi_ap_provision.py` is already the trust anchor) and require the Pi to present it (e.g. HMAC over the announce body) before the backend will release an ingest token. Bind `pending_ingest_token` issuance to that secret, and rate-limit/alert on announce attempts for already-claimed `hardware_id`s. Until then, treat any ingest token issued while this was live as compromised and rotate.

---

## Medium

### 3. No rate limiting and no per-user AI spend cap on LLM and email endpoints

**Files:** `src/api/detectors.py:217` (`POST /v1/detectors/infer`), `src/api/detectors.py:252`/`349` (`POST /v1/patients/{id}/detectors` → background `compile_custom_detector`), `src/api/support.py:19`, `src/api/feedback.py:25`. No rate-limiting middleware exists anywhere in `src/`.

`references/rate-limiting.md` and `references/ai-integration.md` require both transport rate limits and **application-level per-user usage caps** on any endpoint that calls an LLM, because "a single user can drain your entire monthly budget in minutes." Here:
- `infer_detector` makes a synchronous Gemini/OpenAI call per request (`detector_compiler.infer_detector_config`) with no per-caregiver quota or throttle. An authenticated caregiver can loop it to run up arbitrary VLM cost.
- `create_detector` fires a background `compile_custom_detector` LLM call per request, same exposure.
- `/v1/support` and `/v1/feedback` dispatch Resend emails on every authenticated call with no throttle (abuse/spam; blast radius is limited because recipients and `reply_to` are server-fixed — see #note below).

There is no per-account AI usage tracking or hard spend cap in the application (the detection worker has windowing/pause controls for the streaming path, but the user-triggered LLM endpoints are uncapped).

**Remediation:** Add per-IP **and** per-user rate limiting (e.g. a sliding-window limiter backed by Redis, not a Supabase public table — see `references/rate-limiting.md`) to the LLM and email endpoints, plus a per-caregiver daily/monthly LLM-call quota with a hard ceiling. Set hard spend caps in the Gemini/OpenAI/OpenRouter consoles as a backstop.

*Note:* The email bodies themselves are safe from injection — `src/services/email.py` HTML-escapes all caregiver-supplied content (`_support_html` `:252`, `_feedback_html` `esc()` `:366`) and Resend takes the subject as a JSON field (no SMTP header injection). The only gap is volume/abuse.

### 4. Guessable, reusable, unauthenticated demo pairing code mints valid api_keys

**File:** `src/api/pairing.py:59`–`61` / `_pair_demo_camera` `:101`–`145`; default value `424242` in `scripts/seed/seed_demo_camera.py:129` and `src/api/pairing.py:35`.

When `NEMARA_DEMO_PAIRING_CODE` is set (per `src/settings.py:147`–`153`, "production is the only env that sets it"), `POST /v1/cameras/pair` special-cases that exact code: it pairs the seeded demo camera **without expiry and without single-use clearing**, re-minting a fresh `api_key` on every call. The code is unauthenticated (the code *is* the authorization) and the documented value is `424242` — trivially guessable. Anyone who submits it obtains a working camera `api_key` plus the LiveKit room/identity for the demo patient, granting LiveKit token-minting and room access (`src/api/livekit.py`). Impact is bounded to demo data (no real patient PII *if* the demo account is isolated), but it is a permanent, unauthenticated, low-entropy credential-minting endpoint in production.

**Remediation:** Use a long random demo code (not `424242`), rotate it after each App Review cycle, scope the demo camera/patient to a throwaway account with no real data, and disable the path entirely outside active review windows.

---

## Low

### 5. JWT verifier accepts HS256 in production

**File:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), `:83`–`88`.

The verifier dispatches on the token's `alg` header and selects the matching key (ES256 → JWKS public key; HS256 → shared `supabase_jwt_secret`). This is **not** the classic algorithm-confusion bug — the public key is never used as an HMAC secret, and the chosen key is correctly tied to the algorithm. However, production Supabase issues ES256 tokens; continuing to accept HS256 (a legacy/local-dev convenience) means that if the shared `supabase_jwt_secret` ever leaks, an attacker can forge fully valid caregiver JWTs for any account. `references/authentication.md` advises validating the algorithm strictly.

**Remediation:** Gate the accepted algorithm on environment — accept only `ES256` when `settings.env == "prod"`; keep HS256 for `local`/`test` only.

### 6. No security headers set by the application

**File:** `src/main.py:145`–`186` (`create_app` registers routers only; no middleware).

The app sets none of the headers recommended in `references/deployment.md` (`Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`). For a JSON API behind a TLS-terminating proxy (Railway) the risk is modest, but HSTS and `nosniff` are cheap defense-in-depth.

**Remediation:** Add a small middleware (or configure the proxy) to set the standard security headers on all responses. *(No CORS finding: the app adds no CORS middleware, which is the safe default for a non-browser API — browsers cannot make credentialed cross-origin calls. Add an explicit allowlist only if the web dashboard calls this backend directly.)*

### 7. `.env` was previously committed to git history

**File:** `.gitignore` (comment: *".env was previously tracked for partner onboarding; it has been untracked and its leaked secrets rotated."*).

`.env`/`.env.local` are now correctly ignored and `.env.example` contains only placeholders (`<from-...>`). The in-repo note states the leaked secrets were rotated. This is a Low/verification item: confirm (a) every secret that ever sat in the committed `.env` (DB URL, `supabase_jwt_secret`, `supabase_service_role_key`, Gemini/OpenAI/LiveKit/Resend/B2/APNs keys) was actually rotated, and (b) consider scrubbing history (`git filter-repo`) or accept that anyone with a pre-rotation clone still holds the old values.

**Remediation:** Verify rotation of all formerly-committed secrets; run `gitleaks detect` over full history as a backstop.

### 8. Dependencies are floor-pinned; no scheduled vulnerability scan

**File:** `pyproject.toml` (all deps use `>=`), with `uv.lock` present (exact resolution pinned).

The lockfile pins exact transitive versions (good for reproducibility), but `pyproject.toml` uses floors and there is no evidence of a scheduled `pip-audit`/`uv`-based vulnerability scan in CI. No statically-obvious known-vulnerable pin was found (`pyjwt[crypto]>=2.10`, `fastapi>=0.115` are current), but the ML stack (`torch`, `ultralytics`, `onnx`, `opencv-python-headless`, `insightface`) is large and a frequent source of CVEs.

**Remediation:** Add `pip-audit` (or `uv pip audit`) to CI against `uv.lock` and review on a schedule.

---

## Areas Reviewed With No Findings

- **Authorization / IDOR:** Every caregiver endpoint resolves resources through repo methods scoped by `caregiver_id` (e.g. `event_repo.find_by_id(..., caregiver_id=auth.id)` at `events.py:144`; `patient_repo.find_by_id(..., caregiver_id=auth.id)` at `detectors.py:288`). Cross-account access returns `404`. LiveKit token minting validates camera room/identity match (`livekit.py:180`–`197`) and patient ownership (`livekit.py:200`–`230`); Talk and the internals record-clip endpoint reuse `_resolve_owned_cameras_or_403` (`talk.py:175`, `record_clip.py:80`).
- **SQL injection:** No raw/string-built SQL; all access via SQLAlchemy ORM with bound parameters.
- **Command injection:** All `ffmpeg`/`ffprobe` invocations use `create_subprocess_exec`/`subprocess.run` with argument lists and operate on server-generated tempfile paths (`clip_annotate.py`, `frame_extraction.py`, `mp4_remux.py`, `data_collection/mux.py`); no `shell=True`.
- **SSRF:** No server-side fetch of client-controlled URLs. The pull-mode `pi_stream_url` is vestigial and empty in the current push architecture (`relay_publisher.py:149`).
- **Unsafe deserialization:** No `eval`/`exec`/`pickle.load`/`yaml.load`.
- **Mass assignment:** PATCH handlers use Pydantic request schemas + `model_dump(exclude_unset=True)` as an allowlist before reaching `update_partial` (`caregivers.py:53`, `events.py:84`, `detectors.py:415`); the `caregivers` table has no privilege/credit columns to escalate into.
- **Secrets:** No hardcoded API keys/tokens in tracked source; `service_role` key is used only server-side (`storage.py`, `auth_admin.py`, `scripts/seed/seed.py`) and never exposed to clients. `.env` is gitignored; `.env.example` is placeholder-only.
- **JWT handling:** Signature + audience (`authenticated`) + required-claim validation; opaque `401` to prevent oracle behavior (`deps.py:39`). No bare `jwt.decode`.
- **Payments:** No payment/Stripe surface present — the entire client-price-trust class is N/A.
- **Webhooks:** LiveKit webhook verifies the signed `Authorization` JWT against the body via `WebhookReceiver`/`TokenVerifier` and returns `401` on failure (`livekit_webhooks.py:65`–`75`).
- **Storage:** Object keys are server-derived UUIDs (`{patient_id}.{ext}`, `events/{event_id}.mp4`) — no client filename → no path traversal; uploads validate content-type/size; signed URLs are minted only after an ownership check (`events.py:144`, `storage.py:287`).
- **Camera/ingest token hashing:** `cameras.ingest_token_hash` stores unsalted SHA-256 of a 32-byte random token (acceptable for high-entropy secrets) with constant-time compare (`services/ingest/pi/auth.py`). *(The plaintext copy in `pi_discoveries` is the issue — see Finding #1.)*

---

*Generated by an automated static security review using the vibe-security methodology. Findings are based solely on source inspection; no runtime testing was performed. Validate each item against the live configuration before and after remediation.*
