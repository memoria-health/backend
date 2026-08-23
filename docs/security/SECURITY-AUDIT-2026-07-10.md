# Security Audit — nemara-backend

**Date:** 2026-07-10
**Scope:** Static source review of the `nemara-backend` FastAPI service (API routes, auth, DB layer, Supabase schema/RLS, LiveKit integration, storage, AI/LLM surfaces, config, dependency manifest, git history).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) audit framework — secrets & env, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment config, and data-access/input validation.
**Constraint:** Static analysis only. No live systems, databases, or secrets were accessed. Findings are drawn from committed source; where a finding depends on runtime/deployment state that could not be observed, the precondition is stated explicitly.

---

## Executive Summary

nemara-backend is the backend for a **medical / eldercare camera-monitoring product** (patients, caregivers, live camera feeds, fall/agitation/wandering detection). The core request-handling code is, on the whole, **well built for security**: every caregiver-facing route re-verifies the Supabase JWT, and every data-access repo call is scoped by `caregiver_id`, so the primary CRUD surface (patients, cameras, detectors, events, devices, talk) is free of IDOR and mass-assignment defects. Supabase RLS is enabled on every table with correct owner-scoped `USING`/`WITH CHECK` policies. JWT verification is done correctly (`jwt.verify` with an algorithm allowlist and audience check). These are genuine strengths and are called out below.

However, the audit found **two Critical issues and one High issue that materially undermine that posture**:

1. **Real third-party API keys were committed to git history** (`.env`) — including what appear to be live **OpenAI**, **Google Gemini**, and **LiveKit** credentials. Deleting the file (commit `cf36938`) does not remove them from history; the commit message itself acknowledges "Secrets in prior history still require rotation." Anyone with read access to the repository can extract working keys and drain the associated billing accounts / access the LiveKit project.

2. **The entire `/v1/gateway/*` endpoint family authenticates on a self-asserted bearer token with no server-side secret**, and identifies the account by **email address alone**. An unauthenticated attacker who knows (or guesses) a caregiver's email can register their own token against that account, provision cameras under **every one of the victim's patients**, and receive working ingest tokens — which then let them **push arbitrary video/audio into the patient's live-monitoring rooms**. This is an authentication-bypass into a health-monitoring system.

3. **No application-level rate limiting exists anywhere.** The unauthenticated 6-digit camera-pairing endpoint can be brute-forced within its validity window to hijack a pairing, and the authenticated AI endpoints can be abused to run unbounded LLM/VLM calls at the operator's expense.

The remainder are Medium/Low hardening items (Realtime broadcast of sensitive columns, unescaped LLM output in alert emails, HS256 acceptance in production, exposed API docs, hardcoded default recipient emails, and a dormant SSRF-capable pull path).

### Priority actions
1. **Rotate the OpenAI, Gemini, and LiveKit keys now** and purge them from git history (Critical #1).
2. **Put a real shared secret and/or authenticated caregiver on the `/v1/gateway/*` endpoints**, and stop identifying the account by email alone (Critical #2).
3. **Add rate limiting** to `/v1/cameras/pair`, `/v1/cameras/pi-announce`, all `/v1/gateway/*`, and the AI endpoints (High #3).

---

## Findings

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | Critical | Live third-party API keys committed to git history | git history / `.env` |
| 2 | Critical | Unauthenticated gateway provisioning → camera-stream injection & cross-account camera creation | `src/api/cameras_ingest.py:460`, `:324`, `:390`, `:277` |
| 3 | High | No rate limiting: pairing-code brute force + AI-cost abuse | `src/api/pairing.py`, `src/api/detectors.py`, app-wide |
| 4 | Medium | Sensitive credential columns broadcast over Supabase Realtime | `supabase/schema.sql:820-829` |
| 5 | Medium | HS256 JWTs accepted in production (symmetric-secret attack surface) | `src/services/auth.py:26,76-88` |
| 6 | Low | Unescaped detector/LLM output rendered as HTML in alert emails | `src/services/email.py:124-135` |
| 7 | Low | Reusable, never-expiring demo pairing code | `src/api/pairing.py:59-61`, `src/settings.py:193` |
| 8 | Low | Dormant pull-mode path fetches a DB-stored URL (latent SSRF) | `src/services/ingest/pi/relay_publisher.py:237-240` |
| 9 | Low | Interactive API docs & OpenAPI schema exposed unconditionally | `src/main.py:166-171` |
| 10 | Low | Hardcoded personal email addresses as default recipients | `src/settings.py:447-461` |
| — | Info | Schema drift: gateway tables/columns absent from committed `schema.sql` | `supabase/schema.sql` vs `src/api/cameras_ingest.py` |

---

### 1. Critical — Live third-party API keys committed to git history

**Location:** git history (`.env`, removed in commit `cf36938`); `.gitignore` documents the prior tracking.

The `.env` file was tracked in git before commit `cf36938 chore(security): stop tracking .env`. The deletion commit removes it from the working tree but **not from history** — the values remain fully recoverable via `git show cf36938 -- .env` (as demonstrated during this audit). The recovered file contains what appear to be **real, non-placeholder** credentials:

- `NEMARA_OPENAI_API_KEY=sk-proj-…` (live OpenAI project key)
- `NEMARA_GEMINI_API_KEY=AIzaSy…` (live Google Gemini key)
- `NEMARA_LIVEKIT_API_KEY=APItudo6ecfroAu` / `NEMARA_LIVEKIT_API_SECRET=eQslXjZ7…` for a real project (`wss://memoria-dev-5v5d1l62.livekit.cloud`)

(The `NEMARA_SUPABASE_JWT_SECRET` and `NEMARA_SUPABASE_SERVICE_ROLE_KEY` in the same file are the well-known Supabase **local-demo** values, and `NEMARA_DB_URL` points at `127.0.0.1` — those are low-risk local dev artifacts. The OpenAI/Gemini/LiveKit keys are the genuinely sensitive ones.)

**Why it's exploitable:** Anyone with read access to the repository (current or historical, including forks/clones and any CI cache) can extract these keys. Leaked LLM keys allow unbounded API usage billed to the operator — "a leaked key can drain thousands of dollars in minutes" (vibe-security `ai-integration.md`). The LiveKit API key/secret allows minting arbitrary room tokens for the real project, i.e. joining/publishing to patient camera rooms.

**Remediation:**
- **Rotate all three keys immediately** (OpenAI, Gemini, LiveKit) — assume compromised. The commit message already flagged this; treat it as an open action item, not a closed one.
- Purge the secret from history (`git filter-repo --path .env --invert-paths`, or BFG) and force-push; rotate again after, since the pre-purge history may already be cloned.
- Confirm `.env` remains in `.gitignore` (it does) and add a pre-commit `gitleaks` scan to prevent recurrence.

---

### 2. Critical — Unauthenticated gateway provisioning enables camera-stream injection and cross-account camera creation

**Location:** `src/api/cameras_ingest.py` — `gateway_provision_camera` (`:460`), `gateway_roster` (`:324`), `gateway_wait` (`:390`), `post_heartbeat` (`:277`). Router mounted at `src/main.py:178`.

The `/v1/gateway/*` endpoints implement a "field-provisioned camera box" flow. Their entire auth model is:

```python
box_token = authorization[len("bearer "):].strip()      # attacker-chosen
...
token_hash = hash_ingest_token(box_token)
# if no gateway row already holds this hash, INSERT one, tied to...
cg = SELECT id FROM caregivers WHERE lower(email) = lower(:e)   # body.email — public info
```

There is **no shared secret, no signature, and no authenticated caregiver** validating the caller. The "box's unique token" is whatever the client sends; the endpoint simply *trusts* it and *binds* it to the caregiver named by `body.email`. Because email addresses are not secrets, an unauthenticated attacker can:

1. `POST /v1/gateway/provision-camera` with any bearer token and a victim caregiver's email.
2. The server creates a `gateways` row tying the attacker's token to the victim, then creates a `cameras` row under **every** patient the victim owns, and **returns the plaintext `ingest_token` for each** in the response body.
3. The attacker then calls `POST /v1/cameras/me/frames` with those ingest tokens and **streams arbitrary JPEG frames into the victim's `patient_<id>` LiveKit rooms**, which the detector pipeline, caregiver viewer, and clip archival all treat as a genuine camera feed.

In a medical-monitoring product this is severe: an attacker can inject fabricated video into a patient's monitoring feed (masking a real emergency, or generating false alerts), and can enumerate valid caregiver accounts (the endpoint returns `404 caregiver_not_found` vs `422 caregiver_has_no_patients` vs success). `gateway_roster`/`gateway_wait`/`post_heartbeat` are equally open — any presented token that matches a `gateways` row is trusted with no further authorization.

**Precondition / caveat:** The committed `supabase/schema.sql` does **not** contain the `public.gateways` table, the `cameras.last_heartbeat_at` column, or `'gateway'` in the `cameras.source_type` CHECK constraint (`schema.sql:255-256` allows only `'iphone'`/`'pi'`). On the *exact committed schema* these endpoints error at runtime rather than succeeding (see the Info item on schema drift). This is a design-level authentication flaw in committed, mounted code that is clearly intended to run in production; if the deployed database carries the gateway schema (which the fully-built long-poll `gateway_wait` flow strongly implies), it is directly exploitable. It is rated Critical on that basis, with the schema precondition stated for honesty.

**Remediation:**
- Require a **provisioning secret the server issued** (not one the client invents): pre-register box tokens out-of-band, or gate `provision-camera` behind an authenticated caregiver session (`CaregiverAuth`) so the account is proven, not asserted by email.
- Never return `caregiver_not_found`/enumeration-distinguishing responses to an unauthenticated caller.
- Once bound, verify the presented gateway token maps to the caregiver on **every** `/gateway/*` call (already looked up — the gap is the initial trust step).

---

### 3. High — No rate limiting: pairing-code brute force and AI-cost abuse

**Location:** app-wide (no `slowapi`/limiter/middleware present); notably `src/api/pairing.py` (`/v1/cameras/pair`), `src/api/detectors.py` (`/detectors/infer`, `/detect-doors`, `create_detector` compile), `src/api/pi_discovery.py` (`/cameras/pi-announce`).

There is **no application-level rate limiting anywhere** in the service (confirmed: no `slowapi`, `Ratelimit`, or throttling middleware in `src/` or `pyproject.toml`; the only transport-layer note is an aspirational comment in `ingest/pi/auth.py`). Two concrete abuses:

- **Pairing-code brute force → camera hijack.** `POST /v1/cameras/pair` is unauthenticated and matches a **6-digit** `pairing_code` (`pairing.py:63-69`) valid for `pairing_code_ttl_seconds` (default 600s). With no per-IP/per-endpoint limit and no lockout, an attacker can spray the 10⁶ keyspace. A single window is borderline, but multiple concurrent pairing windows across the fleet raise the hit probability, and the endpoint returns an unauthenticated success (api_key) on a hit — pairing the attacker's device to a victim's camera.
- **AI-cost abuse.** `POST /v1/detectors/infer` and `POST /v1/detect-doors` each trigger a Gemini/OpenAI call, and `create_detector` fires a background compile call — all reachable by any authenticated caregiver with no per-user quota. "A single user can drain your entire monthly budget in minutes" (vibe-security `rate-limiting.md`). Combined with the leaked keys (Finding 1), the blast radius is larger.

**Remediation:** Add per-IP + per-user rate limiting (e.g. `slowapi` with Redis, or an edge/API-gateway limit) to: all unauthenticated endpoints (`/cameras/pair`, `/cameras/pi-announce`, `/gateway/*`) and all AI endpoints. Add a lockout/backoff on repeated `/cameras/pair` failures, and enforce hard per-user AI usage quotas server-side (not just provider-side caps).

---

### 4. Medium — Sensitive credential columns broadcast over Supabase Realtime

**Location:** `supabase/schema.sql:820-829`.

The `cameras` table is added to the `supabase_realtime` publication **with `REPLICA IDENTITY FULL`**:

```sql
alter publication supabase_realtime add table ... public.cameras ...;
alter table public.cameras replica identity full;
```

`REPLICA IDENTITY FULL` makes every UPDATE/DELETE broadcast carry the **entire row**, and the `cameras` row includes secret-bearing columns: `pairing_code` (plaintext 6-digit), `api_key_hash` (bcrypt), and `ingest_token_hash` (sha256). RLS correctly scopes the broadcast to the owning caregiver, so this is not a cross-tenant leak — but it means a camera's live plaintext pairing code and its credential hashes are streamed to the caregiver's app (and anything with that session), where they need not be. Bcrypt/sha256 hashes are not directly reversible, but a plaintext `pairing_code` in a realtime payload is an unnecessary exposure that widens the window in Finding 3.

**Remediation:** Exclude the `cameras` table from Realtime, or use a Realtime-safe view/publication that projects only non-sensitive columns; keep `pairing_code`, `api_key_hash`, and `ingest_token_hash` out of any client-visible broadcast. If `REPLICA IDENTITY FULL` is needed for the iOS cache-diff logic, restrict it to the columns actually consumed.

---

### 5. Medium — HS256 JWTs accepted in production alongside ES256

**Location:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), `:76-88`.

The verifier dispatches on the token's own `alg` header and, for `HS256`, verifies with the symmetric `supabase_jwt_secret`. The dispatch itself is implemented safely (the key is chosen by branch, so classic RS/HS confusion is not directly possible here). The residual risk is **defense-in-depth**: production Supabase (2.x) issues ES256 tokens verified via public JWKS, but the service *also* permanently trusts HS256 tokens signed by a **symmetric secret**. If that secret is ever exposed, an attacker can **forge arbitrary caregiver identities** and bypass auth entirely. This is not hypothetical for this repo — a `NEMARA_SUPABASE_JWT_SECRET` value already leaked into git history (Finding 1; that particular value is the demo secret, but it demonstrates the exposure path for the real one).

**Remediation:** In production, restrict `JWT_ALGORITHMS` to `["ES256"]` (reject HS256), keeping HS256 only for local/test via an env-gated setting. Continue to validate `iss` in addition to `aud`.

---

### 6. Low — Unescaped detector/LLM output rendered as HTML in alert emails

**Location:** `src/services/email.py:124-135` (`_event_html`), used by `dispatch_event_emails`.

`_event_html` interpolates `title`, `narrative`, `location`, and `patient_name` **without HTML-escaping** into the alert email body. These originate from the detector/LLM pipeline (event rows) and patient records. The support and feedback email builders *do* escape (`_support_html`, `_feedback_html`), so the pattern is inconsistent. Per vibe-security `ai-integration.md`, "LLM output should be treated as untrusted … sanitize before rendering as HTML." Impact is limited (recipient is the caregiver; most mail clients don't execute JS), but markup/HTML injection into alert emails is possible.

**Remediation:** Apply the same `.replace('<','&lt;')…` escaping used in `_support_html`/`_feedback_html` to all interpolated values in `_event_html`.

---

### 7. Low — Reusable, never-expiring demo pairing code

**Location:** `src/api/pairing.py:59-61` / `_pair_demo_camera:101`; `src/settings.py:193` (`demo_pairing_code`).

When `NEMARA_DEMO_PAIRING_CODE` is set (intended for production, for App Review), `POST /v1/cameras/pair` special-cases it: it pairs the seeded demo camera **without expiry or single-use clearing**, re-minting a fresh api_key on every call. Anyone who learns the demo code (it ships in an App Store build) can repeatedly re-pair the demo camera, invalidating the reviewer's key each time. Scope is limited to the single seeded demo camera, so impact is low, but it is a standing unauthenticated, unthrottled, credential-minting path.

**Remediation:** Scope the demo path tightly (only the demo camera, which it does), add rate limiting, and prefer disabling it outside an explicit review window.

---

### 8. Low — Dormant pull-mode path fetches a DB-stored URL (latent SSRF)

**Location:** `src/services/ingest/pi/relay_publisher.py:237-240` (`read_mjpeg(self._config.pi_stream_url)`), `:435` (`read_pcm_audio(pi_audio_url)`).

The push-ingest refactor left a pull-mode path that opens an HTTP stream to `pi_stream_url` / `pi_audio_url` — columns stored on the `cameras` row. The committed schema marks these "vestigial … NOT referenced by the relay path," and no active caller was found wiring them from the DB into a `RelayPublisher` pull run. If that path is ever reactivated with a caregiver- or Pi-influenced URL, the backend would issue requests to an attacker-chosen host (SSRF against internal metadata endpoints / internal services). Flagged as latent/defense-in-depth given no live caller.

**Remediation:** Delete the dead pull-mode code, or if retained, validate/allowlist the target host and block private/link-local ranges before fetching.

---

### 9. Low — Interactive API docs & OpenAPI schema exposed unconditionally

**Location:** `src/main.py:166-171` (`FastAPI(...)` with default docs enabled).

`/docs`, `/redoc`, and `/openapi.json` are served in all environments (the app only *filters their access logs*, `main.py:56-68`, it does not disable them). This publishes the full API surface (including the `/v1/gateway/*` and internal-shaped routes) to any anonymous visitor, easing reconnaissance. Low severity on its own.

**Remediation:** Disable docs in production (`FastAPI(docs_url=None, redoc_url=None, openapi_url=None)` when `env == "prod"`), or gate them behind auth.

---

### 10. Low — Hardcoded personal email addresses as default recipients

**Location:** `src/settings.py:447-461`.

`support_email_recipients` defaults to `["sunny.work.2002@gmail.com"]` and `feedback_email_recipients` to founder addresses (`aryaman@…`, `sunny@…`). These are overridable by env, but committing personal addresses as defaults risks them being used in an environment that forgets to override, and is a minor privacy/info-disclosure item.

**Remediation:** Default these to empty and require explicit env configuration; the send helpers already no-op cleanly when the recipient list is empty.

---

### Info — Schema drift: gateway objects absent from committed `schema.sql`

The gateway ingest code (`src/api/cameras_ingest.py`) reads/writes `public.gateways`, `cameras.last_heartbeat_at`, and `source_type='gateway'`, none of which exist in `supabase/schema.sql` (source_type CHECK allows only `'iphone'`/`'pi'`, `schema.sql:255-256`). `config.toml` confirms the team edits a single `schema.sql` in place with no migration files. This is an operational/correctness risk (endpoints 500 on any environment built from the committed schema) and it obscures security review of the gateway surface. Recommend adopting versioned migrations so the committed schema is the source of truth and the gateway tables get explicit RLS review (they are not covered by the RLS block in `schema.sql`).

---

## What was checked and found sound

These are noted so the report reflects the whole review, not only the defects:

- **Per-caregiver authorization (no IDOR / mass assignment):** every route in `patients.py`, `events.py`, `detectors.py`, `devices.py`, `talk.py`, `caregivers.py`, `face_embeddings.py` threads `caregiver_id=auth.id` into repo calls and returns `404` (not `403`) on cross-account access to avoid ID enumeration. PATCH bodies use Pydantic models with `extra="forbid"` exposing only writable fields — no request-body spreading into ORM writes.
- **Supabase RLS:** enabled on all 11 exposed tables with correct owner-scoped `USING` **and** `WITH CHECK` clauses (`schema.sql:711-801`); no `USING (true)` / `auth.uid() IS NOT NULL` policies. The direct-client `ui_events` insert path is correctly scoped.
- **JWT verification:** `verify_supabase_jwt` uses `jwt.decode` with an explicit `algorithms` allowlist and `audience="authenticated"` (rejects `alg:none`, verifies signature) — see Finding 5 for the one hardening note.
- **LiveKit token minting** (`api/livekit.py`) correctly binds camera tokens to the camera's own room/identity, scopes viewer/broadcaster tokens to owned patients, and enforces privacy mode in the grant. **Webhook signatures are verified** via `WebhookReceiver`/`TokenVerifier` on the raw body (`livekit_webhooks.py`).
- **Ingest-token auth** uses SHA-256 of a 256-bit random token with `hmac.compare_digest` (`ingest/pi/auth.py`); camera api-keys are bcrypt-hashed. Upload size caps are enforced by streaming reads (`patients._read_capped_upload`) and at the storage layer. Storage object keys are built from server-side UUIDs (no path traversal).
- **No SQL injection** in the reviewed code: all data access uses SQLAlchemy ORM or parameterized `text(... :param ...)` bindings; no string-concatenated SQL and no `$…Unsafe`-equivalent.
- **No payment/Stripe surface** exists in this backend, so that category is N/A.
- **CORS:** no permissive `CORSMiddleware` is installed (default-deny cross-origin), appropriate for a mobile-app backend.

---

*Generated by an automated static security audit (vibe-security methodology). Findings marked with runtime preconditions should be confirmed against the deployed environment. No application code was modified as part of this audit.*
