# Security Audit — nemara-backend

**Date:** 2026-07-16
**Scope:** Static source review of the `nemara-backend` FastAPI service (repo `aryamandarda/nemara-backend`).
**Methodology:** [vibe-security](https://github.com/raroque/vibe-security-skill) skill — secrets & env, Supabase RLS / database access control, authentication & authorization, injection/SSRF/deserialization, dependencies, deployment config, and AI/LLM surfaces.
**Nature:** Static only. No live systems, databases, or secrets were accessed.

---

## Executive Summary

nemara-backend is a caregiver-monitoring backend (FastAPI + SQLAlchemy/asyncpg against Supabase Postgres, LiveKit for media, multiple VLM providers, APNs/Resend). The codebase is, on the whole, **well-built from a security standpoint**: every caregiver-facing route consistently enforces per-caregiver ownership at the repository layer (`caregiver_id=auth.id`), returns 404-not-403 to avoid ID enumeration, uses parameterized SQL throughout, validates JWT signatures properly (no `decode()`-without-verify), keeps the Supabase `service_role` key server-side, and defines correct, owner-scoped RLS policies on every table. Dependencies are current with no known-vulnerable versions.

Two findings are serious and should be addressed before further production exposure:

1. **Critical — Real secrets committed to git history.** A tracked `.env` leaked live OpenAI, Gemini, LiveKit, and Resend API keys plus an APNs `.p8` private key. Removing the file (commit `cf36938`) does **not** remove them from history. All must be rotated.
2. **Critical — Authentication bypass in the gateway-provisioning API.** `POST /v1/gateway/provision-camera` and the sibling `/gateway/*` endpoints treat a fully **client-chosen** bearer token as proof of identity and bind it to any caregiver **by email address alone**. An attacker who knows a caregiver's email can provision cameras under all of that caregiver's patients, receive plaintext ingest tokens, and inject arbitrary video/audio into the victim's live patient-monitoring rooms.

The remaining findings (JWT algorithm downgrade, total absence of rate limiting, pairing-code brute-force, schema drift, minor email HTML) are lower severity but meaningful for a product handling vulnerable-adult video/audio and PHI-adjacent data.

| # | Severity | Title |
|---|----------|-------|
| 1 | Critical | Live API keys & APNs private key committed to git history |
| 2 | Critical | Auth bypass: self-asserted gateway token + email-only caregiver binding |
| 3 | High | Gateway provisioning returns plaintext ingest tokens → live-room media injection |
| 4 | Medium | Production JWT verifier accepts HS256 (symmetric) alongside ES256 |
| 5 | Medium | No rate limiting on auth, pairing, LLM, or email endpoints |
| 6 | Medium | Camera pairing-code brute-force (6-digit, no throttle) |
| 7 | Low | Alert-email HTML does not escape event/room fields |
| 8 | Low | Unauthenticated Pi announce + cross-tenant discovery visibility |
| 9 | Info | Schema drift: gateway code references tables/columns absent from `schema.sql` |

---

## Critical

### 1. Live API keys and an APNs private key are committed to git history

**Files:** `.env` (history — present at `cf36938~1`, removed in `cf36938`)

A `.env` file containing **real** production/staging credentials was tracked in git. It was later untracked and added to `.gitignore` (commit `cf36938 "chore(security): stop tracking .env"`), and the commit message itself acknowledges *"Secrets in prior history still require rotation."* Deleting a file does not purge it from history — anyone with repo (or fork/clone) access can recover it with `git show cf36938~1:.env`.

Secrets exposed in history include:

- **OpenAI API key** — `NEMARA_OPENAI_API_KEY=sk-proj-YbEoTJIg…` (live project key)
- **Google Gemini API key** — `NEMARA_GEMINI_API_KEY=AIzaSyDW3quQ…`
- **LiveKit API key + secret** — `APItudo6ecfroAu` / `eQslXjZ7OHyVJF4LV4aODMeI1nilYU3xpd2bEIqQGNo` (two rotations both present in history)
- **Resend API key** — `NEMARA_RESEND_API_KEY=re_N3Fo9qHR_…`
- **APNs auth key** — a full `-----BEGIN PRIVATE KEY-----` ES256 `.p8` (`NEMARA_APNS_AUTH_KEY`), with `apns_team_id=V8N59CF74G`, `apns_key_id=TTTD2823D2`

(The `NEMARA_SUPABASE_JWT_SECRET` and `NEMARA_SUPABASE_SERVICE_ROLE_KEY` values in history are the well-known Supabase *local-dev demo* defaults, not real project secrets — but see Finding 4 for why the JWT secret still matters in prod.)

**Why it's exploitable:** The LiveKit API secret mints room tokens for **any** patient room — a holder can subscribe to any family's live camera feed or publish into it. The OpenAI/Gemini/Resend keys allow unlimited spend on the company's accounts (billing drain) and, for Resend, sending mail as the verified domain. The APNs private key lets an attacker forge push notifications to the caregiver app.

**Remediation:**
- **Rotate every key above immediately**, regardless of the `.gitignore` fix. Treat all as compromised.
- Set hard spending caps + billing alerts on OpenAI, Gemini, and LiveKit.
- Purge the blobs from history (`git filter-repo --path .env --invert-paths`, or BFG) and force-push; then invalidate any long-lived clones. `.env` is now correctly gitignored — keep it that way.
- Add a pre-commit secret scanner (`gitleaks`) to CI so this cannot recur.

---

### 2. Authentication bypass — self-asserted gateway token bound to any caregiver by email

**File:** `src/api/cameras_ingest.py:460-577` (`POST /v1/gateway/provision-camera`); companions `/gateway/roster` (`:324`), `/gateway/wait` (`:390`), `/cameras/me/heartbeat` (`:277`).

`gateway_provision_camera` is intended to let a field-installed mini-PC ("box") register itself. Its authentication is:

```python
box_token = authorization[len("bearer "):].strip()     # line 483 — WHATEVER the client sends
...
cg = await session.execute(
    text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
    {"e": body.email.strip()})                          # line 488 — caregiver chosen by EMAIL
...
INSERT INTO public.gateways (caregiver_id, token_hash, ...)  # line 507 — bind attacker token → caregiver
```

The bearer token is **never validated against any allow-list, registry, or pre-shared secret** — it is whatever the caller chooses, and the endpoint simply hashes it and stores it as a valid gateway credential. The caregiver whose account the box is attached to is selected **solely by `body.email`**, which is not a secret (it is user-facing, guessable, and often known).

**Why it's exploitable:** An unauthenticated attacker sends:

```
POST /v1/gateway/provision-camera
Authorization: Bearer attacker-picks-this-string
{ "email": "victim@example.com", "room": "Bedroom" }
```

The server binds the attacker's chosen token to the victim's `caregiver_id`, creates a camera under **every one of the victim's patients**, and (Finding 3) returns a plaintext ingest token for each. The attacker now holds a valid gateway credential for someone else's account and can poll `/gateway/roster` / `/gateway/wait` to enumerate the victim's cameras (`livekit_identity`, `livekit_room`, `desired_state`). This is a cross-tenant account compromise gated on nothing but knowing an email address.

**Caveat (see Finding 9):** the committed `supabase/schema.sql` does not define the `public.gateways` table, and its `cameras.source_type` CHECK allows only `('iphone','pi')` — so against the *committed* schema these INSERTs error and roll back. The elaborate, live gateway code path (roster long-poll, heartbeat, provisioning) strongly implies production runs a newer, uncommitted schema where the table/enum exist and the endpoint functions. The auth-model defect is unconditional in code and the router is always mounted (`main.py:178`).

**Remediation:**
- Do **not** derive identity from a client-supplied email. The box token must be a **pre-provisioned secret** minted server-side and delivered out-of-band; the endpoint should look the box up by its token and derive `caregiver_id` from that record — never accept an arbitrary token + email.
- Require an authenticated caregiver session (or an admin-scoped provisioning key) to *create* a gateway↔caregiver binding in the first place. Note the parallel, correctly-secured `POST /v1/patients/{id}/cameras/gateway-provision` in `src/api/cameras.py:418` already does this behind `CaregiverAuth` — consolidate on that model and remove the unauthenticated path.
- Reject the request if the presented token is unknown, rather than upserting it.

---

## High

### 3. Gateway/Pi provisioning returns plaintext ingest tokens enabling live-room media injection

**Files:** `src/api/cameras_ingest.py:573` (gateway), `src/api/cameras.py:489` (gateway-provision), `src/api/cameras_ingest.py:99-208` (`POST /v1/cameras/me/frames`).

Any holder of a camera's `ingest_token` can open `POST /v1/cameras/me/frames` and stream a `multipart/x-mixed-replace` JPEG body that the `RelayPublisher` decodes and **publishes into the LiveKit room `patient_<patient_id>` as a `cam_*` participant** (`cameras_ingest.py:171-185`). That is exactly the feed the caregiver watches live and the detector pipeline analyzes.

Chained with Finding 2, an attacker who provisions cameras under a victim's patients receives these tokens (returned in the provision response — "the only place it lands") and can therefore **inject arbitrary video/audio into a vulnerable adult's live monitoring feed**: show a fake "all quiet" scene to mask a real emergency, or feed frames that trigger false fall/agitation events. Even absent Finding 2, tokens surfaced in an HTTP response body are prone to logging/leakage, and there is no per-token scoping beyond the single camera.

**Remediation:**
- Fix Finding 2 first (that is the token-issuance vector).
- Prefer delivering ingest tokens over the same out-of-band channel as device provisioning, not in a synchronous HTTP response to a self-asserted caller.
- Consider short-lived, rotating ingest tokens and verifying the publishing participant against expected device identity at the SFU.

---

## Medium

### 4. Production JWT verifier accepts HS256 (symmetric) alongside ES256

**File:** `src/services/auth.py:26,72-88`

```python
JWT_ALGORITHMS = ["ES256", "HS256"]
...
if alg == "ES256":   signing_key = <JWKS public key>
elif alg == "HS256": signing_key = jwt_secret
claims = pyjwt.decode(token, signing_key, algorithms=JWT_ALGORITHMS, audience="authenticated")
```

Production Supabase tokens are ES256 (asymmetric, verified via JWKS). The verifier *also* accepts HS256 tokens signed with the shared symmetric `supabase_jwt_secret`. The key is correctly selected by the token's declared `alg`, so this is **not** the classic RS/HS confusion attack — but it does mean the symmetric secret is a **full authentication bypass key**: anyone who learns `supabase_jwt_secret` can forge a valid token for *any* caregiver (`sub` = victim's UUID) and take over their account and all patient data. Given secrets have already leaked to git history once (Finding 1), keeping a symmetric fallback in prod meaningfully widens the blast radius. The verifier also does not validate the `iss` (issuer) claim.

**Remediation:**
- In non-local environments, restrict `algorithms` to `["ES256"]` only (gate HS256 behind `env == "local"`).
- Validate `issuer` against the expected Supabase project URL in addition to `audience`.
- Ensure `supabase_jwt_secret` is a strong, rotated value and stored only in the secrets manager.

### 5. No rate limiting on authentication, pairing, LLM, or email endpoints

**Files:** app-wide — no limiter middleware is registered (`src/main.py`), and none of the routers apply per-IP/per-user limits.

Per the vibe-security rate-limiting guidance, several endpoints need throttling and have none:

- **LLM/VLM cost:** `POST /v1/detect-doors` (`detectors.py:88`) accepts an arbitrary base64 image and calls a VLM on every request; custom-detector compile/infer call a "stronger frontier model." Any authenticated caregiver can loop these to drain the OpenAI/Gemini budget. There are no per-user usage quotas.
- **Email relay:** `POST /v1/support` and `POST /v1/feedback` fan out to Resend on each call — usable as a spam/abuse relay against the configured recipients.
- **Auth-adjacent:** `POST /v1/cameras/pair` (see Finding 6) and unauthenticated `POST /v1/cameras/pi-announce` accept unlimited attempts.

**Remediation:** Add per-IP **and** per-user rate limiting (e.g. a Redis/Upstash sliding window or an API-gateway limit) on auth, pairing, all VLM-invoking routes, and outbound-email routes. Enforce hard per-account/per-tier VLM usage quotas server-side in addition to provider-level caps.

### 6. Camera pairing-code brute-force

**File:** `src/api/pairing.py:63-98`

`POST /v1/cameras/pair` looks a camera up by a **6-digit numeric** `pairing_code` (900,000-value space) with a ~10-minute validity window, an opaque 401 on miss, and **no attempt throttling or lockout**. An attacker enumerating codes during any open pairing window (≈1,500 req/s to cover the space in 10 min — feasible) can redeem another user's pending pairing and receive a long-lived camera `api_key`, becoming that "wall phone": minting LiveKit camera tokens for the patient room and reading the camera's `desired_state`.

**Remediation:** Rate-limit `/cameras/pair` per IP and globally; add exponential backoff/lockout after repeated failures; consider a longer/alphanumeric code or binding attempts to a device. The single-use + TTL design is good but insufficient without throttling.

---

## Low

### 7. Alert-email HTML does not escape event/room fields

**File:** `src/services/email.py:124-135` (`_event_html`)

`title`, `narrative`, and `location` are interpolated into alert-email HTML without escaping (the support/feedback templates *do* escape — `_support_html:252`, `_feedback_html:366`). These fields are largely VLM/detector-generated, but `location`/room names originate from caregiver input and flow into the email unescaped, allowing limited HTML/style injection into the message body.

**Remediation:** Apply the same `esc()` helper used by `_feedback_html` to all interpolated values in `_event_html`.

### 8. Unauthenticated Pi announce + cross-tenant discovery visibility

**File:** `src/api/pi_discovery.py:43-122`

`POST /v1/cameras/pi-announce` is unauthenticated (acknowledged: the Pi has no pre-pairing credentials), and `GET /v1/cameras/discovered` shows **all** unclaimed recently-seen Pis to **every** caregiver (no per-caregiver scoping). A malicious actor on-path can announce spoofed devices (arbitrary `hardware_id`/`current_url`) that surface in every caregiver's discovery picker; a caregiver could then provision a hostile stream URL. The module documents this as an accepted v1 risk mitigated by physical/BLE provisioning being the real trust anchor.

**Remediation:** As planned for v2, carry a per-device secret negotiated at BLE setup so announcements are attributable, and scope `/discovered` per caregiver. Validate/normalize `current_url` (reject non-LAN / SSRF-prone targets) before it can be provisioned.

---

## Informational

### 9. Schema drift — gateway code references undefined tables/columns

`supabase/schema.sql` is declared the single source of truth ("edited in place, applied via `make db-reset`"), yet `src/api/cameras_ingest.py` reads/writes `public.gateways` and `cameras.last_heartbeat_at`, and both gateway-provision paths set `cameras.source_type = 'gateway'` — **none of which exist in the committed schema** (the `source_type` CHECK allows only `iphone`/`pi`). Either production runs an uncommitted schema (a governance/audit gap — the committed schema no longer describes the deployed system) or these endpoints error at runtime. Reconcile `schema.sql` with the deployed database and add the `gateways` table's RLS policies before relying on it.

### Positive observations (defense-in-depth working as intended)

- **RLS is correctly configured** on every table with owner-scoped `USING`/`WITH CHECK` policies (`schema.sql:711-801`); `SECURITY DEFINER` trigger fns pin `search_path`. The client-writable `ui_events` table is protected by an insert `WITH CHECK (caregiver_id = auth.uid())`.
- The FastAPI service connects as the `postgres` superuser (bypasses RLS), but **every** repo method independently filters by `caregiver_id`, so RLS and the app layer are true belt-and-braces.
- **No SQL injection:** all raw `text()` queries use bound parameters; ORM filters use typed columns. No `f-string` SQL with user input.
- LiveKit token minting (`api/livekit.py`) verifies room ownership and forces `caregiver_`-prefixed identities; the webhook receiver verifies signatures.
- Supabase `service_role` key is used only server-side (storage uploads/signing); the client is handed short-lived signed URLs.
- JWT verification uses `jwt.verify` semantics (signature + audience + expiry), never bare `decode`.
- Dependencies (`uv.lock`) are current — fastapi 0.136, starlette 1.0.1, cryptography 48, pyjwt 2.13, pillow 12.2, requests 2.34, urllib3 2.7, jinja2 3.1.6 — no known-vulnerable versions identified.

---

## Prioritized Remediation Checklist

1. **Rotate all leaked credentials** (OpenAI, Gemini, LiveKit key+secret, Resend, APNs `.p8`) and purge `.env` from git history. *(Finding 1)*
2. **Rebuild gateway provisioning auth** — never trust a client-supplied token or identify caregivers by email; require a pre-issued secret or an authenticated caregiver session; remove the unauthenticated `/v1/gateway/provision-camera` path in favor of the `CaregiverAuth`-gated one. *(Findings 2, 3)*
3. **Restrict production JWT to ES256** and validate `iss`. *(Finding 4)*
4. **Introduce rate limiting** (per-IP + per-user) on auth, pairing, VLM, and email endpoints, plus hard per-account VLM spend quotas. *(Findings 5, 6)*
5. Escape all values in alert-email HTML; scope Pi discovery per caregiver and add SSRF validation of announced URLs. *(Findings 7, 8)*
6. Reconcile `schema.sql` with the deployed database (gateways table + RLS). *(Finding 9)*
