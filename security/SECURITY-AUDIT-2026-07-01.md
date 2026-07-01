# Security Audit — nemara-backend

**Date:** 2026-07-01
**Auditor:** Automated security review (vibe-security methodology)
**Scope:** Static source audit of the `nemara-backend` FastAPI service — application code, SQL migrations/schema, configuration, dependency lockfile, and git history. No live systems, databases, or secrets were accessed.
**Commit audited:** `e79cc76` (main)

---

## Executive Summary

nemara-backend is a FastAPI service for a caregiver eldercare-monitoring product (patient cameras, LLM-based incident detection, LiveKit video, Supabase auth/storage). The codebase is, on the whole, **unusually disciplined about security fundamentals**: Supabase Row-Level Security is enabled and correctly owner-scoped on every table, application-layer authorization is consistent and load-bearing (every caregiver-facing repo joins through `patients.caregiver_id`), all SQL uses bound parameters, all `ffmpeg` subprocess calls use argument lists over internally-generated paths (no shell), request schemas use `extra="forbid"` (mass assignment is structurally prevented), JWTs are verified (signature + `aud` + `exp`), bearer/ingest tokens are stored only as hashes, and third-party dependencies are current with no known-vulnerable pins.

That said, there is **one Critical issue that requires immediate action**: real production secrets were committed to git history and remain retrievable, including the Supabase JWT signing secret and service-role key. Because the backend accepts legacy HS256 caregiver tokens signed with that JWT secret, the leaked secret is a direct, forge-any-caregiver authentication-bypass primitive; the service-role key independently bypasses all RLS on Supabase Storage/Realtime. These must be rotated now, regardless of the `.gitignore` note claiming prior rotation.

Beyond that, the notable gaps are the **complete absence of rate limiting / brute-force protection** (High) — no throttle on the unauthenticated pairing exchange, auth surfaces, LLM-cost operations, or the email relay — and a **cross-tenant Pi-camera discovery/claim gap** (Medium) in a privacy-critical product. The remaining findings are Low-severity hardening items.

### Findings at a glance

| # | Severity | Title |
|---|----------|-------|
| 1 | **Critical** | Production secrets (JWT signing secret, service-role key, API keys) committed to git history |
| 2 | **High** | No rate limiting anywhere — pairing-code brute force, auth abuse, unbounded LLM cost, email relay |
| 3 | **Medium** | Cross-tenant Pi-camera discovery: any caregiver can claim/redirect another household's camera |
| 4 | **Low** | Interactive API docs + OpenAPI schema publicly exposed in production |
| 5 | **Low** | Latent SSRF via unauthenticated `pi-announce` `current_url` (currently dead pull path) |
| 6 | **Low** | Unescaped LLM-derived event narrative rendered as HTML in alert emails |
| 7 | **Low** | `/ready` leaks raw database exception strings to unauthenticated callers |
| 8 | **Low** | Permanent, reusable demo pairing code re-mints credentials for a live camera |
| 9 | **Low** | JWT `decode()` accepts both algorithms instead of the single resolved one (hardening) |

---

## Critical

### 1. Production secrets committed to git history

**Files / evidence:** `.env` present in 6 commits (`git log --all -- .env`): `cf36938`, `59f62fc`, `217e7ff`, `6065268`, `cd363ff`, `b393c5d`. The file is no longer tracked (removed in `cf36938`), but every prior blob is still retrievable, e.g. `git show cf36938~1:.env`.

The historical `.env` contained real, secret-bearing values (not placeholders):

- `NEMARA_SUPABASE_JWT_SECRET` — the HMAC secret used to verify HS256 caregiver JWTs
- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` — the Supabase service-role key (bypasses all RLS)
- `NEMARA_DB_URL` — Postgres connection string with embedded credentials
- `NEMARA_GEMINI_API_KEY`, `NEMARA_OPENAI_API_KEY`, `NEMARA_OPENROUTER_API_KEY` — LLM provider keys (direct billing)
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` — video infrastructure
- `NEMARA_APNS_AUTH_KEY` — Apple push private key (PEM `.p8`)
- `NEMARA_RESEND_API_KEY` — transactional email

**Why it is exploitable (concrete impact):**

1. **Forge any caregiver session.** `src/services/auth.py:26` accepts `HS256` tokens (`JWT_ALGORITHMS = ["ES256", "HS256"]`) verified with `settings.supabase_jwt_secret` (`auth.py:76-88`). Anyone with the leaked JWT secret can mint a token with `aud="authenticated"` and an arbitrary `sub`/`email`. `caregiver_auth` (`src/api/deps.py:66`) will accept it, and the app-layer authz then scopes to that forged `sub` — i.e. full impersonation of *any* caregiver, reading every patient, camera feed, and incident clip of that account.
2. **Bypass all database security.** The service-role key bypasses RLS entirely; with it an attacker can read/modify/delete every row via Supabase's REST/Storage/Realtime APIs and mint signed URLs for any stored clip or patient photo.
3. **Financial / infra abuse.** The Gemini/OpenAI/OpenRouter/LiveKit/Resend/APNs credentials allow direct billing drain and outbound abuse under the company's identity.

Deleting the file from the working tree (as `cf36938` did) does **not** remove it from history — the secrets remain readable by anyone who ever cloned or forks the repo. Per the vibe-security methodology: *"If a secret was ever committed to Git history, consider it compromised."*

**Remediation:**
- **Rotate every credential listed above right now**, in the respective provider consoles (Supabase JWT secret + service-role key, Gemini, OpenAI, OpenRouter, LiveKit key/secret, Resend, APNs `.p8`, and the Postgres password in `NEMARA_DB_URL`). The `.gitignore` comment claims secrets were rotated after untracking — confirm this is true for *all* of the above, and treat any unrotated value as live-compromised.
- Purge the blobs from history (`git filter-repo --path .env --invalid-refs` or BFG) and force-push, then have all collaborators re-clone. Note that on GitHub the old blobs may persist in forks/caches, so rotation — not just history rewrite — is the load-bearing fix.
- Add a pre-commit secret scanner (`gitleaks protect --staged`) and run `gitleaks detect` in CI to prevent recurrence.
- Consider dropping the legacy `HS256` acceptance in `auth.py` in production (see Finding 9) so a future JWT-secret leak is not a full auth bypass on its own.

---

## High

### 2. No rate limiting or brute-force protection anywhere

**Evidence:** A repo-wide search for `ratelimit`/`slowapi`/`limiter`/`throttle` returns no rate-limiting middleware or per-route guard. `src/main.py` registers routers with no throttling layer. The vibe-security methodology flags rate limiting as required on auth, AI-cost, and email/abuse surfaces — none are present. Concrete exposures:

- **Pairing-code brute force.** `POST /v1/cameras/pair` (`src/api/pairing.py:45`) is unauthenticated and unthrottled. Codes are 6-digit numeric (`str(secrets.randbelow(900_000) + 100_000)`, `src/api/cameras.py:184`) → ~900k space, valid for a 10-minute window. With unlimited attempts and no lockout, a targeted attacker can enumerate an active pairing code; success returns a freshly-minted `api_key` plus the victim's `patient_id`, LiveKit room, and identity (`pairing.py:90-98`) — i.e. pairing an attacker device to a victim's camera. The opaque 401 slows enumeration but does not prevent it.
- **Unbounded LLM cost.** Caregiver-triggered operations that call paid LLMs (detector creation/compile via `src/services/detector_compiler.py`) have no per-user quota or cost cap. There is no application-level per-caregiver usage limit anywhere.
- **Email relay abuse.** `POST /v1/support` and `POST /v1/feedback` (`src/api/support.py`, `src/api/feedback.py`) trigger Resend sends per request with no throttle — an authenticated caregiver can drive unbounded outbound email (spam/cost) through the company's verified domain.
- **Unauthenticated announce flooding.** `POST /v1/cameras/pi-announce` (`src/api/pi_discovery.py:48`) is unauthenticated and unthrottled; it upserts DB rows on every call (see also Finding 3).

**Remediation:** Introduce a shared limiter (e.g. `slowapi`, or an Upstash/Redis token bucket) applied at least to: `/v1/cameras/pair` (aggressive per-IP limit + short lockout after N failures), `/v1/cameras/pi-announce`, the LLM-cost routes (per-caregiver quota + a hard billing cap in each provider console), and the support/feedback email routes (per-caregiver daily cap). Combine per-IP and per-user limits so neither IP rotation nor new accounts defeats it. Set hard spend caps on Gemini/OpenAI/OpenRouter/LiveKit/Resend.

---

## Medium

### 3. Cross-tenant Pi-camera discovery and claim

**Files:** `src/api/pi_discovery.py:48-122`, `src/api/cameras.py:305-401` (`provision_pi_camera` → `pi_discovery_repo.find_by_id`/`claim`), `src/repos/pi_discovery_repo.py:34,77,109`.

The `pi_discoveries` aggregate has **no tenant/caregiver column**. Three consequences compound:

1. `GET /v1/cameras/discovered` (`pi_discovery.py:97`) returns **every** unclaimed Pi seen network-wide — including raw `current_url`/`current_audio_url` LAN/stream endpoints — to **any** authenticated caregiver (`list_unclaimed_recent`, no caregiver filter).
2. `provision_pi_camera` ownership-checks the target *patient* (`caregiver_id=auth.id`) but resolves `discovery_id` with **no caregiver scoping** (`cameras.py:316`, `find_by_id`). Caregiver A can therefore claim a Pi announced on Caregiver B's network, binding that physical camera's stream to A's patient and consuming its one-shot ingest token.
3. `POST /v1/cameras/pi-announce` is unauthenticated (the Pi has no pre-pairing credential) and accepts an arbitrary `current_url` (`schemas/pi_discovery.py`, length-only validation). An attacker can seed or overwrite a discovery row's URL for any `hardware_id`.

Together these allow an attacker to enumerate another household's to-be-paired eldercare camera, redirect its advertised stream URL, or race to claim it — a meaningful privacy/integrity impact for this product class. This is documented as an accepted v1 tradeoff (`pi_discovery.py:14-18`) pending a "v2 per-device secret," and exploitation of the claim path requires an authenticated caregiver, which is why it is rated Medium rather than High.

**Remediation:** Add a device-attested secret negotiated during WiFi provisioning so `pi-announce` is authenticated and discoveries are attributable to a caregiver; scope `/discovered` and `provision_pi_camera` to discoveries the requesting caregiver can claim; and omit raw stream URLs from the picker payload until a device is attributable to the caller.

---

## Low

### 4. Interactive API docs and OpenAPI schema exposed in production

**File:** `src/main.py:167` — `FastAPI(...)` is constructed without `docs_url=None`/`redoc_url=None`/`openapi_url=None`, so `/docs`, `/redoc`, and `/openapi.json` are publicly served in every environment. The access-log filter (`main.py:56-65`) even notes these paths are "hit constantly by browsers + scanners." This discloses the full API surface (every route, schema, and auth requirement) to anonymous clients, easing reconnaissance.

**Remediation:** Disable the docs/schema endpoints in `prod` (gate on `settings.env`), or place them behind authentication. Example: `FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` when `settings.env == "prod"`.

### 5. Latent SSRF via unauthenticated `pi-announce` `current_url`

**Files:** `src/schemas/pi_discovery.py:37` (`current_url`, length-only), `src/api/cameras.py:316` (copied to `cameras.pi_stream_url`), sink at `src/services/ingest/pi/mjpeg_reader.py:364` via `src/services/ingest/pi/relay_publisher.py:239` (`RelayPublisher.run`).

An attacker-supplied `current_url` (from the unauthenticated announce endpoint, no scheme/host validation) would be fetched server-side by the pull-mode relay — classic SSRF toward internal/metadata endpoints. **Currently not reachable:** the only live server path uses push mode (`src/api/cameras_ingest.py:168` hardcodes `pi_stream_url=""` and drives `run_from_iterator`); `RelayPublisher.run()`/`read_mjpeg` is dead code ("now-disabled reconciliation worker"). Rated Low/latent because no live route reaches the sink today.

**Remediation:** Validate `current_url`/`pi_stream_url` at the schema boundary — require `http`/`https` and reject loopback/link-local/RFC-1918 hosts (block `169.254.0.0/16`, `127.0.0.0/8`, `10/8`, `172.16/12`, `192.168/16`, `::1`, `fc00::/7`). If the pull path is retired, delete `RelayPublisher.run()`/`read_mjpeg` to remove the latent sink.

### 6. Unescaped LLM-derived event narrative rendered as HTML in alert emails

**File:** `src/services/email.py:124-135` (`_event_html`) interpolates `title`, `narrative`, and `location` directly into email HTML with no escaping. The support/feedback paths escape their inputs (`email.py:252`, `_feedback_html`), but the event path does not. `narrative` originates from the VLM's free-text `description` output (`src/services/detection/event_promotion.py:486`), and LLM output must be treated as untrusted. Someone in a camera's view could prompt-inject the detector to emit HTML/anchor markup, which lands unescaped in caregivers' alert emails (phishing links, spoofed content). Most mail clients strip active scripts, limiting this to HTML/link injection — hence Low.

**Remediation:** HTML-escape `title`, `narrative`, and `location` before interpolation, mirroring the `esc()` helper already used in `_feedback_html`.

### 7. `/ready` leaks raw database exception strings

**File:** `src/api/health.py:48-54` — on a DB error the unauthenticated `/ready` endpoint returns `str(exc)` inside the 503 body (`{"error": str(exc)}`). SQLAlchemy/asyncpg exception text can include the connection target, driver, and internal details, aiding reconnaissance.

**Remediation:** Return a generic `{"status": "fail", "checks": {"db": "fail"}}` to the client and log the exception detail server-side only.

### 8. Permanent reusable demo pairing code re-mints live-camera credentials

**Files:** `src/settings.py:159` (`demo_pairing_code`), `src/api/pairing.py:59-98` / `_pair_demo_camera` (`pairing.py:101`), `scripts/seed_demo_camera.py` (documents `424242`). When `NEMARA_DEMO_PAIRING_CODE` is set (production only), `POST /v1/cameras/pair` special-cases it: the code never expires and is not single-use, and each call re-mints an `api_key` for the seeded demo camera (invalidating the prior device's key). Anyone who learns the code can repeatedly pair to — and receive the LiveKit credentials for — that live demo camera. Impact is bounded to the demo camera, hence Low.

**Remediation:** Keep the demo camera pointed at non-sensitive content, rotate/retire the code after App Review, and prefer a long random value over a memorable one; ideally scope the demo path to a build/flag that is off in the general-availability production config.

### 9. JWT verification passes both algorithms to `decode()` (hardening)

**File:** `src/services/auth.py:83-88` — after dispatching on the token's `alg` and selecting the matching key, `pyjwt.decode(..., algorithms=JWT_ALGORITHMS)` still permits *both* `ES256` and `HS256`. Because the key is chosen to match the branch, this is not exploitable as an algorithm-confusion bug today, but passing the single resolved algorithm is the defensive norm and removes any future footgun. This finding is closely tied to Finding 1: as long as `HS256` is accepted in production, the (leaked) `supabase_jwt_secret` alone forges valid caregiver tokens.

**Remediation:** Pass only the resolved algorithm to `decode()` (`algorithms=["ES256"]` or `["HS256"]` per branch), and consider disabling `HS256` acceptance entirely in `prod` (local/dev only), so production trusts only asymmetric JWKS-verified tokens.

---

## Verified safe (checked, no issue found)

The following were specifically examined and found sound — recorded so the audit's coverage is auditable:

- **Supabase RLS** (`supabase/schema.sql:705-785`): RLS is enabled on every public table; all policies are owner-scoped to `auth.uid()` (via `caregiver_id`, or a subquery through `patients`/`cameras`). No `USING (true)` or `USING (auth.uid() IS NOT NULL)` policies; INSERT/UPDATE policies include `WITH CHECK`. Realtime publication relies on these policies to filter per-tenant.
- **Application-layer authorization**: consistent and load-bearing (the backend's direct SQLAlchemy connection bypasses RLS). Every caregiver-facing repo method takes a keyword `caregiver_id` and joins the resource through `patients.caregiver_id` (or matches `caregivers.id`), returning None/empty for cross-account access → 404. Confirmed across `patient_repo`, `camera_repo`, `detector_repo`, `event_repo`, `face_embedding_repo`, `camera_detector_repo`, `caregiver_device_repo`. No IDOR found on patients/cameras/detectors/events/devices/face-embeddings.
- **Mass assignment**: all request schemas use `extra="forbid"` with whitelisted fields; ownership columns (`caregiver_id`, `patient_id`, state/review fields) are set server-side from the authenticated principal, never from the body. Updates use `model_dump(exclude_unset=True)` over whitelisted models.
- **SQL injection**: all raw `text()` statements use bound parameters only (`camera_presence.py`, `state_machine.py`, `presence_sweeper.py`, `worker.py` advisory locks, `health.py`); no f-string/`%`/`.format()` SQL; ORM everywhere else.
- **Command injection**: all `ffmpeg`/`ffprobe` calls use `create_subprocess_exec`/`subprocess.run` with list args (no `shell=True`) over internally-generated tempfile/`pipe:` paths, never client-named files (`mp4_remux.py`, `frame_extraction.py`, `clip_annotate.py`, `streaming_encoder.py`, `mux.py`, `clip_archival.py`).
- **Path traversal**: storage object keys are built from `UUID`-typed components with whitelisted extensions (`storage.py:46`, `reid/storage.py`); download/delete keys come from DB columns, not raw request input.
- **Unsafe deserialization**: no `pickle`, unsafe `yaml.load`, `eval`/`exec`, `torch.load`, or `jsonpickle` on untrusted data; LLM JSON is parsed with `json.loads` and coerced against enum whitelists (`detector_compiler.py`).
- **Auth token handling**: LiveKit webhook signature is verified before any state change (`livekit_webhooks.py:70`); LiveKit token minting enforces room/identity ownership and privacy-mode grants (`livekit.py`); camera api-keys and Pi ingest tokens are stored only as bcrypt/SHA-256 hashes; auth failures return opaque errors.
- **Secrets in current tree**: `.env` is gitignored and no longer tracked; `.env.example` contains only placeholders; scripts/docs read credentials from the environment (no hardcoded keys in the working tree). The client-side env-prefix risks (`NEXT_PUBLIC_`/`VITE_`/`EXPO_PUBLIC_`) do not apply — this is a server-only backend.
- **Dependencies**: resolved versions in `uv.lock` are current (e.g. `cryptography 48.0.0`, `pyjwt 2.13.0`, `pillow 12.2.0`, `python-multipart 0.0.29`, `fastapi 0.136.1`, `urllib3 2.7.0`); no known-vulnerable pins identified.
- **CORS**: no CORS middleware is registered — appropriate for a native-mobile-facing API (no permissive `Access-Control-Allow-Origin: *`).

---

## Prioritized remediation summary

1. **(Critical) Rotate every credential exposed in git history now** — Supabase JWT secret + service-role key first (auth-bypass and RLS-bypass), then DB password and all provider keys. Purge history and add secret scanning. (Finding 1)
2. **(High) Add rate limiting / brute-force protection** to the pairing exchange, auth surfaces, LLM-cost operations, and email relay; set hard provider spend caps. (Finding 2)
3. **(Medium) Close the cross-tenant Pi-discovery gap** — device-attested announce, caregiver-scoped discovery/claim, and drop raw URLs from the picker. (Finding 3)
4. **(Low) Hardening** — disable public docs in prod (4), validate/retire the SSRF sink (5), escape LLM narrative in emails (6), stop leaking DB errors from `/ready` (7), tighten the demo pairing code (8), and pin the JWT algorithm / drop HS256 in prod (9).
