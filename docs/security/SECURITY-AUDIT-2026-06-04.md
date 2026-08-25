# Security Audit — nemara-backend

**Date:** 2026-06-04
**Scope:** Static source review of `nemara-backend` (FastAPI + Supabase/Postgres backend for the Nemara/Memoria caregiver monitoring product).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) audit framework — secrets & env, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment configuration, and data-access/input-validation.
**Constraint:** Static analysis only. No live systems, databases, or secrets were accessed. Findings are substantiated by source; no dynamic confirmation was performed.

---

## Executive Summary

The application is, on the whole, **well-architected from an authorization standpoint**. Every caregiver-facing endpoint funnels through a verified Supabase JWT (`CaregiverAuth`) and a repository layer that consistently scopes queries by `caregiver_id`, giving solid defense against IDOR. JWT verification is implemented correctly (signature checked, audience enforced, algorithm-confusion avoided by selecting the key from the token's declared `alg`). RLS is enabled on almost every table as a second line of defense, and the LiveKit token-minting and webhook-verification paths properly validate room ownership and signatures.

However, the audit found one **Critical** and several **High**-severity issues that must be addressed before this repository is made public or relied on in production:

1. **CRITICAL — Live third-party API secrets are committed to git.** The tracked `.env` file contains real OpenAI, Google Gemini, LiveKit Cloud, Resend, and Apple APNs (`.p8` private key) credentials, present throughout git history. Anyone with repository read access (now or after a leak/open-sourcing) can drain billing accounts and send push/email as the product.
2. **HIGH — `pi_discoveries` table has no Row-Level Security.** It is the only public table missing RLS and is reachable through the public Supabase anon/publishable key via PostgREST. It stores **plaintext one-time ingest bearer tokens** and device network locations.
3. **HIGH — Unauthenticated camera pairing has no rate limiting.** The 6-digit pairing code (≈10⁶ space, 10-minute window) on the unauthenticated `POST /v1/cameras/pair` is brute-forceable, allowing an attacker to pair their own device to a victim's camera.
4. **MEDIUM — No rate limiting anywhere in the application,** including authenticated LLM-invoking endpoints (cost-drain) and email-sending endpoints (spam relay).

A handful of Medium/Low items (unauthenticated device-announce writes, unescaped LLM output in alert emails, missing security headers, push-token reassignment) round out the report.

### Severity tally

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High     | 2 |
| Medium   | 3 |
| Low      | 4 |

---

## Critical

### C1 — Live third-party API credentials committed to the repository (`.env` tracked in git)

**Files:** `.env:23, 28, 33–35, 60–66, 80` (and `.env` is present throughout git history — `git log -- .env` shows it across multiple commits). `.gitignore:11–15` documents that `.env` is *intentionally* tracked.

The `.env` file is checked into the repo and contains genuine, billable production credentials:

| Line | Secret | Risk |
|------|--------|------|
| `.env:28` | `NEMARA_OPENAI_API_KEY=sk-proj-…` | Unlimited OpenAI spend on the project's account |
| `.env:23` | `NEMARA_GEMINI_API_KEY=AIza…` | Unlimited Gemini spend |
| `.env:34–35` | `NEMARA_LIVEKIT_API_KEY` / `…_API_SECRET` | Mint LiveKit tokens for *any* room → join/publish to any camera feed, forge webhooks |
| `.env:80` | `NEMARA_RESEND_API_KEY=re_…` | Send arbitrary email from the product's verified domains |
| `.env:60–66` | `NEMARA_APNS_AUTH_KEY` (full `.p8` ECDSA private key) + Team/Key ID | Forge APNs auth tokens → send push notifications to all caregiver devices |

**Why it's exploitable:** Anyone with read access to the repo — including its full history — obtains working keys. The `.gitignore` comment itself acknowledges these "all… need rotation" before the repo goes public. Secrets in git history are *not* removed by deleting the file later; they remain in every clone and on every fork.

**Note on the Supabase keys (`.env:19–20`):** the `NEMARA_SUPABASE_SERVICE_ROLE_KEY` and `NEMARA_SUPABASE_JWT_SECRET` in this file are the **well-known public Supabase local-dev demo values** (the JWT decodes to `iss: "supabase-demo"`, and the secret is the documented `super-secret-jwt-token-with-at-least-32-characters-long`). Those particular values are not themselves sensitive — but their presence shows the file pattern that production secrets follow, and if this `.env` is ever swapped for the production one and committed, the same exposure applies to the load-bearing keys.

**Remediation:**
1. **Rotate every key listed above immediately** (OpenAI, Gemini, LiveKit, Resend, APNs). Treat them as compromised.
2. Remove `.env` from tracking: `git rm --cached .env`, add `.env` (not just `.env.local`) to `.gitignore`, commit.
3. Purge `.env` from history (`git filter-repo --path .env --invert-paths`) before the repo is ever made public, and force-push.
4. Inject secrets at deploy time via the platform's secret store (Railway variables), never a committed file. Keep only `.env.example` with placeholders (already present and correct).
5. Add a CI secret-scanner (`gitleaks detect`) as a pre-merge gate.

---

## High

### H1 — `pi_discoveries` table is missing Row-Level Security (exposed via public anon key)

**Files:** `supabase/schema.sql:310` (table definition) — and notably **absent** from the RLS section at `supabase/schema.sql:618–700`, despite the section's comment at line 621 asserting *"Every table is RLS-on."* `scripts/deltas/apply_pi_schema_delta.sql:85` creates the same table with no `enable row level security`.

Every other public table (`caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`, `feedback`) has `alter table … enable row level security` plus an owner-scoped policy. `pi_discoveries` has neither. In a standard Supabase project the `anon` and `authenticated` roles hold default table grants, and the publishable/anon key is public (`.env:18`), so PostgREST exposes this table for read **and write** to anyone:

```
GET  <supabase_url>/rest/v1/pi_discoveries     apikey: <publishable key>
POST <supabase_url>/rest/v1/pi_discoveries     apikey: <publishable key>
```

**Why it's exploitable:** The table holds `pending_ingest_token` (`schema.sql:336`) — the **plaintext** one-time bearer token a Pi uses to authenticate frame ingest (`POST /v1/cameras/me/frames`) — plus `current_url`/`current_audio_url` (device network endpoints) and `hardware_id` (MAC). An attacker with the public key can:
- **Read** pending ingest tokens and replay them to push attacker-controlled frames into a paired camera's LiveKit room (impersonating the Pi).
- **Read** every discovered device's local network URL and MAC across all customers.
- **Insert/Update/Delete** rows: flood the caregiver "Add a camera" picker with spoofed devices, or overwrite `current_url` to redirect a caregiver who picks that device.

**Remediation:** Enable RLS and add a policy. Because the announce path is intentionally unauthenticated (the Pi has no JWT) and the consumer is the backend service-role connection (which bypasses RLS), the safest posture is **RLS on with no anon/authenticated policy** (deny-all to PostgREST), keeping all access via the backend's privileged DB connection:

```sql
alter table public.pi_discoveries enable row level security;
-- No policy for anon/authenticated → PostgREST sees zero rows and cannot write.
-- The FastAPI backend connects as the postgres/service role and is unaffected.
revoke all on public.pi_discoveries from anon, authenticated;
```

Additionally, do not store the ingest token in plaintext — store only a SHA-256 hash (as `cameras.ingest_token_hash` already does) and hand the plaintext to the Pi exactly once in the announce response.

### H2 — Unauthenticated camera pairing is brute-forceable (no rate limiting)

**Files:** `src/api/pairing.py:40–98` (`POST /v1/cameras/pair`); pairing code is 6 digits (`supabase/schema.sql:170` comment "6 digits"), TTL 600 s (`src/settings.py:130`). No rate-limiting middleware exists anywhere in the app (confirmed: no `slowapi`/limiter/throttle in `src/`).

The endpoint is unauthenticated by design — "the code IS the proof of authorization" (`pairing.py:13`). It looks up a `cameras` row by `pairing_code` within the non-expired window and, on match, mints and returns a long-lived camera `api_key`.

**Why it's exploitable:** A 6-digit numeric code is a ~10⁶ keyspace. With no per-IP/per-code rate limit, an attacker can spray codes for the full 10-minute window of any active pairing. A hit returns a valid `api_key` bound to the victim's camera, which (via `POST /v1/livekit/token` with `role=camera`, `src/api/livekit.py:76–86`) lets the attacker mint a LiveKit token with `can_publish=true` for that camera's room — i.e. publish arbitrary video/audio into the caregiver's live feed, or deny service by racing the legitimate device. The opaque-401 design (`pairing.py:70–75`) prevents code enumeration *content*, but not brute-force throughput.

**Remediation:**
- Add strict rate limiting to `POST /v1/cameras/pair` (e.g. per-IP and global: a few attempts/minute), returning `429` and exponentially backing off.
- Lengthen/strengthen the pairing code (8+ alphanumeric, `secrets`-generated) to push the keyspace well beyond brute-force reach within the TTL.
- Lock out a code after N failed attempts and consider a global cap on concurrent failed pair attempts.

---

## Medium

### M1 — No rate limiting on authenticated LLM and email endpoints (cost-drain & spam relay)

**Files:** `src/api/detectors.py:61` (`POST /v1/detectors/infer` — synchronous Gemini/OpenAI call on caller text), `src/api/detectors.py:96` (`POST /v1/patients/{id}/detectors` → background compile LLM call), `src/api/support.py:19` (`POST /v1/support` → Resend send), `src/api/feedback.py` (`POST /v1/feedback` → Resend send), `src/api/patients.py:198` (`POST /v1/patients/{id}/test-email` → Resend send). No limiter anywhere.

The skill's rate-limiting guidance flags AI calls ("a single user can drain your entire monthly budget in minutes") and email senders ("attackers can use your app as a spam relay") as mandatory rate-limit points. All of these are reachable by any authenticated caregiver (account signup is open — `supabase/config.toml:34 enable_signup = true`), with no per-user quota.

**Why it's exploitable:** A single signed-up account can loop `POST /v1/detectors/infer` to run unbounded paid LLM inferences (cost amplification against the OpenAI/Gemini budget), or loop the support/feedback/test-email endpoints to emit attacker-chosen email content from the product's verified Resend domain (the `replyTo` is the attacker's own account email, but the *From* is the trusted domain).

**Remediation:** Add per-user and per-IP rate limits (e.g. Upstash Redis / `slowapi`) on every LLM- and email-invoking route. Enforce hard per-user/per-tier daily quotas on LLM calls, and set hard spend caps in the OpenAI/Gemini/Resend dashboards as a backstop.

### M2 — Unauthenticated `pi-announce` permits arbitrary writes to a security-relevant table

**Files:** `src/api/pi_discovery.py:43–90` (`POST /v1/cameras/pi-announce`), backed by `pi_discovery_repo.upsert_announcement`. No authentication (documented as intentional, `pi_discovery.py:5–12`).

Any unauthenticated caller can upsert `pi_discoveries` rows keyed by an attacker-chosen `hardware_id`, setting `device_label` and `current_url`/`current_audio_url` to arbitrary values (length-capped at 2048, `schema.sql:338–341`). Combined with **H1** (no RLS) this is writable both via PostgREST and via this endpoint.

**Why it's exploitable:** An attacker can (a) flood the caregiver-facing "discovered devices" picker (`GET /v1/cameras/discovered`) with spoofed entries, and (b) plant a `current_url` pointing at an internal/attacker host that a caregiver may select during provisioning. While the current push-mode relay does not have the *backend* fetch `current_url`, the value is surfaced to clients and used in the provisioning decision, making it a content-spoofing / potential SSRF seed if any consumer dereferences it.

**Remediation:** Bind announcements to a per-device secret established out-of-band (the BLE-provisioning secret the code's own comments anticipate, `schema.sql:303–309`). Until then: rate-limit the announce endpoint per IP, validate/normalize `current_url` against an allowlist of private-LAN ranges expected for cameras, and scope the discovered-device list per caregiver so cross-tenant spoofed rows can't appear in another customer's picker.

### M3 — LLM-generated alert content is injected into email HTML without escaping

**Files:** `src/services/email.py:115–126` (`_event_html`) interpolates `title`, `narrative`, and `location` directly into HTML with no escaping; called from `dispatch_event_emails` / `send_test_email` (`email.py:180–190, 352–364`). Contrast with `_support_html` (`email.py:236`) and `_feedback_html` (`email.py:307`), which *do* escape `<`/`>`.

Per the AI-integration guidance, LLM output must be treated as untrusted and sanitized before rendering as HTML. The `title`/`narrative`/`location` fields originate from the detector LLM's structured output over camera frames.

**Why it's exploitable:** A crafted scene (or a future prompt-injection path through the LLM) could yield event text containing HTML/markup that renders in the caregiver's mail client (content/markup injection; most modern clients strip scripts, limiting it to spoofed formatting/links rather than active XSS). Lower impact than a web-context XSS, but inconsistent with the escaping already applied on the other two templates.

**Remediation:** Apply the same HTML-escaping helper used in `_feedback_html`/`_support_html` to `title`, `narrative`, and `location` in `_event_html`.

---

## Low

### L1 — No security/transport headers configured

**Files:** `src/main.py:145–186` (`create_app`) registers routers but adds no middleware for HSTS, `X-Content-Type-Options`, `X-Frame-Options`, or a Content-Security-Policy. (CORS is intentionally absent, which is a safe default for a native-app-only API — browsers will block cross-origin reads.)

**Remediation:** Add a small middleware setting `Strict-Transport-Security`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy` on all responses, per the deployment reference.

### L2 — Push-notification token can be reassigned to another account if the APNs token is known

**Files:** `src/repos/caregiver_device_repo.py:19–59` (`upsert_by_token`) — `on_conflict_do_update` overwrites `caregiver_id` for an existing `apns_token`, so whoever last registers a given token "owns" it.

The APNs token is high-entropy and Apple-issued, so practical exploitation requires already knowing a victim's token; impact is limited to redirecting that device's push alerts. Documented as intentional (handles sign-out/sign-in on a shared device). Flagged for awareness.

**Remediation:** If a device can be re-bound across accounts, ensure the prior owner's alerts stop and consider requiring the token to be presented by the device itself (it is) plus logging the reassignment for audit.

### L3 — Reusable, never-expiring demo pairing code path in production

**Files:** `src/api/pairing.py:53–61, 101–145` (`_pair_demo_camera`); gated by `settings.demo_pairing_code` (`src/settings.py:131–137`, "production is the only env that sets it").

When set, the demo code bypasses expiry and single-use, re-minting an `api_key` for the seeded demo camera on every call. If the configured demo code is short/guessable, anyone can repeatedly pair against the demo camera (App-Review fixture). Impact is confined to the demo camera, but it is a standing unauthenticated credential-minting path in production.

**Remediation:** Use a long, random demo code; scope it tightly to the demo patient/camera; and disable it once App Review no longer needs it.

### L4 — Verbose DEBUG logging and principal identifiers in logs

**Files:** `.env:6 NEMARA_LOG_LEVEL=DEBUG`; `src/main.py:20–23` honors it; `src/api/deps.py:89` logs `caregiver_id` at debug, and auth-reject reasons are logged throughout. Running production at `DEBUG` increases the chance of sensitive data (tokens, ids, request internals) reaching logs.

**Remediation:** Ensure production runs at `INFO` or higher (the `Settings` default is already `INFO`; the committed `.env` overrides it to `DEBUG`). Confirm the deploy environment does not carry `DEBUG`.

---

## Areas Reviewed and Found Sound

These were examined specifically and did **not** yield findings — noted so the review's coverage is explicit:

- **JWT verification** (`src/services/auth.py`): uses `jwt.decode` with an explicit `algorithms` list and enforced `audience="authenticated"`. The ES256/HS256 dual-path selects the verification key from the token's declared `alg` *before* calling `decode`, so the classic algorithm-confusion attack (signing an HS256 token with the public ES256 key) does not apply. Claims are validated for presence/type.
- **IDOR / authorization**: caregiver endpoints (`patients`, `events`, `detectors`, `talk`, `caregivers`, `devices`, `face_embeddings`) uniformly pass `caregiver_id=auth.id` into repository functions that filter by ownership; cross-account access returns `404` (no existence leak). RLS provides defense-in-depth on all but `pi_discoveries` (H1).
- **LiveKit token minting** (`src/api/livekit.py`): camera tokens are constrained to the camera's own room+identity; caregiver viewer/broadcaster tokens are validated against patient ownership via the `patient_<uuid>` room name.
- **LiveKit webhooks** (`src/api/livekit_webhooks.py`): signature verified via `WebhookReceiver`/`TokenVerifier` against the raw body before any dispatch.
- **SQL injection**: all queries use SQLAlchemy Core/ORM with bound parameters; no string-concatenated SQL or `text()` with user input was found. The `.sql` files are static migrations.
- **Mass assignment**: PATCH/POST bodies use Pydantic models with `extra="forbid"` semantics and explicit field allowlists (`settings.py:19` global config; e.g. `patch_patient` notes server-only columns are rejected as 422). Caregiver `id` comes from the JWT, never the body.
- **File upload**: patient-photo upload is size-capped via chunked draining (`patients.py:43–61`) and content-type allowlisted to JPEG/PNG (`storage.py:46–73`).
- **Pi ingest token**: 32-byte `secrets.token_urlsafe`, stored as SHA-256, looked up by indexed equality (`services/ingest/pi/auth.py`). High entropy; appropriate.
- **Dependencies** (`uv.lock`): resolved versions are current (e.g. `python-multipart 0.0.29` — past the CVE-2024-53981 fix, `fastapi 0.136.1`, `cryptography 48.0.0`, `pillow 12.2.0`, `pyjwt 2.13.0`). No known-vulnerable pinned versions were identified.
- **Docker image** (`Dockerfile`, `.dockerignore`): `.env`/`.env.*` are excluded from the build context, so the secrets in C1 are **not** baked into the production image (they remain in git, which is the C1 problem).

---

## Prioritized Remediation Summary

1. **(C1) Rotate all committed third-party keys now** (OpenAI, Gemini, LiveKit, Resend, APNs); untrack and purge `.env` from history; move secrets to the deploy secret store; add `gitleaks` to CI.
2. **(H1) Enable RLS on `pi_discoveries`** (deny-all to PostgREST, access only via the backend role) and stop storing the ingest token in plaintext.
3. **(H2) Rate-limit `POST /v1/cameras/pair`** and strengthen the pairing code to defeat brute force.
4. **(M1) Add per-user/per-IP rate limiting** to all LLM- and email-invoking endpoints, plus provider-side hard spend caps.
5. **(M2) Authenticate or tightly rate-limit `pi-announce`** and validate `current_url`; scope discovered devices per caregiver.
6. **(M3) HTML-escape LLM-derived fields** in alert emails.
7. **(L1–L4)** Add security headers; review push-token reassignment; harden/retire the demo pairing code; run production at `INFO`.
