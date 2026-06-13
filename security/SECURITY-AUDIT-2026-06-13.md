# Security Audit — nemara-backend

**Date:** 2026-06-13
**Auditor:** Automated security review (scheduled agent) using the `vibe-security` skill methodology
**Commit audited:** `aa55a0d` (detached HEAD on `main`)
**Scope:** Static source review only. No live systems, databases, or secrets were accessed.
**Stack:** FastAPI (Python 3.12) · SQLAlchemy async (asyncpg) · Supabase (Postgres + Auth + Storage + Realtime) · LiveKit Cloud · Gemini/OpenAI detector pipeline · APNs push · Resend email · Raspberry-Pi camera ingest.

---

## Executive Summary

nemara-backend is a caregiver-facing monitoring backend. Its application-layer authorization is, on the whole, **well built**: every API handler requires a verified Supabase JWT, every repository query is scoped by `caregiver_id` (verified in `event_repo`, `patient_repo`, etc.), cross-account access returns opaque 404s, JWT verification correctly uses `jwt.decode(..., algorithms=[...])` with audience checks (no `alg:none`/algorithm-confusion exposure), and Supabase Row-Level Security is enabled on **almost** every table. No SQL injection, command injection, unsafe deserialization, or path-traversal sinks were found in application code.

However, the audit identified two issues that warrant immediate attention plus several hardening gaps:

1. **CRITICAL — Live secrets were committed to git history**, including the Supabase `service_role` key, the Supabase JWT signing secret, OpenAI and Gemini API keys, LiveKit API key/secret, and a **complete APNs `.p8` ECDSA private key** (with its Team ID and Key ID). The working tree no longer tracks `.env`, and a repo comment claims the secrets were rotated — but they remain fully reconstructable from history and rotation cannot be verified statically.
2. **HIGH — The `pi_discoveries` table has no Row-Level Security**, while every other table in the schema does. It is reachable through Supabase's auto-exposed PostgREST API with the public anon key (shipped in the mobile app). The table stores a **plaintext one-shot camera ingest bearer token**, so an anon-key holder can read tokens and inject spoofed video into a victim's camera feed, and can insert/modify rows to poison every caregiver's "Add a camera" device picker.

The remaining findings concern the **complete absence of rate limiting** (no limiter anywhere in the codebase), which exposes the unauthenticated 6-digit pairing-code exchange to brute force and leaves AI/email cost paths unbounded, plus a few lower-severity hardening items.

| # | Severity | Title |
|---|----------|-------|
| 1 | Critical | Live secrets committed to git history (Supabase service_role, JWT secret, OpenAI/Gemini/LiveKit keys, APNs private key) |
| 2 | High | `public.pi_discoveries` missing RLS — plaintext ingest token & picker exposed via anon key |
| 3 | Medium | No rate limiting on the unauthenticated 6-digit pairing-code exchange (camera takeover via brute force) |
| 4 | Medium | Unauthenticated `pi-announce` write path, unbounded + unauthenticated (combines with #2) |
| 5 | Medium | No per-user AI usage caps and no rate limiting on email (support/feedback) endpoints |
| 6 | Low | LiveKit token `identity` is client-controlled and unvalidated |
| 7 | Low | Reusable, never-expiring demo pairing code matched in production |
| 8 | Info | Backend connects to Postgres as superuser (bypasses RLS by design) — app-layer authz is the load-bearing control |
| 9 | Info | No automated dependency vulnerability scanning performed/observed |

---

## Critical

### 1. Live secrets committed to git history

**Location:** `.env` as committed in history (e.g. `git show df0863d:.env`); removal commit `cf36938` ("chore(security): stop tracking .env"). Referenced by `.gitignore:9-14`.

**Evidence (values redacted; lengths/identifiers are real):** the historical `.env` contains real, non-placeholder values:

- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` — 164-char JWT. **The `service_role` key bypasses all RLS and grants full read/write to every table in the database.**
- `NEMARA_SUPABASE_JWT_SECRET` — 55-char HS256 signing secret. Anyone holding it can **forge caregiver auth tokens** for any account (the backend accepts HS256 — `src/services/auth.py:26`).
- `NEMARA_OPENAI_API_KEY` (164 chars) and `NEMARA_GEMINI_API_KEY` (39 chars) — billable LLM keys.
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` — allow minting tokens to join/publish in any LiveKit room (i.e. view or inject any patient's live camera).
- `NEMARA_APNS_AUTH_KEY` — the **full APNs `.p8` ECDSA P-256 private key is printed in plaintext** (`-----BEGIN PRIVATE KEY----- … -----END PRIVATE KEY-----`), alongside `NEMARA_APNS_TEAM_ID=V8N59CF74G` and `NEMARA_APNS_KEY_ID=TTTD2823D2`. This key can send arbitrary push notifications to every user of bundle `com.elephantlabs.nemara.caregiver`.

**Why it is exploitable:** Deleting a file from the tree does not remove it from history — `git show df0863d:.env` returns the secrets verbatim. Anyone with read access to the repository (or any historical clone/fork/CI cache) can recover every credential. The Supabase service_role key and JWT secret alone are a full database compromise and an authentication-bypass primitive for this PHI-adjacent (patient monitoring) application.

**Remediation:**
1. **Rotate every secret above** at its provider (Supabase service_role + JWT secret, OpenAI, Gemini, LiveKit, and especially revoke/replace the APNs key in the Apple Developer portal). The `.gitignore` note asserts rotation already happened — confirm this explicitly for **each** key, particularly the APNs `.p8`.
2. **Purge the secrets from history** using `git filter-repo` (or BFG) and force-push, then have all collaborators re-clone. History rewriting is required because rotation notes are not verifiable and old clones persist.
3. Add a pre-commit secret scanner (`gitleaks`/`trufflehog`) and a CI gate so this cannot recur.
4. Keep production secrets only in the deploy platform's secret store (Railway), never in a tracked file.

---

## High

### 2. `public.pi_discoveries` is missing Row-Level Security — plaintext ingest token and device picker exposed via the public anon key

**Location:** Table defined `supabase/schema.sql:310-345`. The RLS section (`supabase/schema.sql:619-700`) enables RLS and adds owner policies for `caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`, and `feedback` — but **omits `pi_discoveries` entirely**. PostgREST exposure is on: `supabase/config.toml` → `[api] enabled = true`, `schemas = ["public", ...]`, `max_rows = 1000`.

**Why it is exploitable:** In Supabase, every table in the `public` schema is served by PostgREST and reachable with the **anon key, which is public** (it is shipped inside the iOS app bundle). A table with RLS disabled is fully readable and writable by anyone holding that key. `pi_discoveries` stores sensitive data:

- `pending_ingest_token` (`schema.sql:336`) — a **plaintext, one-shot bearer token** stashed between caregiver provisioning and the Pi's next announce (`src/repos/pi_discovery_repo.py:121-143`, `consume_pending_ingest_token` at `:180`). That exact token authenticates `POST /v1/cameras/me/frames` (`src/api/cameras_ingest.py:96-133`). An attacker who reads it before the Pi consumes it can **push spoofed MJPEG frames into a victim's camera**, which the backend relays to LiveKit and treats as the real camera — corrupting the live feed and the detector/event pipeline for a monitored patient.
- `hardware_id` (MAC addresses) and device URLs for every customer's cameras.

Because the table is also writable via the anon key, an attacker can **insert fake discovery rows or rewrite `current_url`**. The discovered-device list (`GET /v1/cameras/discovered`, `src/api/pi_discovery.py:93-122`) has **no per-caregiver scoping** ("every caregiver sees all unclaimed Pis", module docstring `:16-18`), so injected rows appear in every caregiver's "Add a camera" picker — enabling social-engineering of a victim into pairing an attacker-controlled device.

**Remediation:**
- Enable RLS on the table and add an explicit policy. The backend reaches this table as the Postgres superuser (see Finding 8), so it is unaffected by RLS; only the public PostgREST surface needs locking down:
  ```sql
  alter table public.pi_discoveries enable row level security;
  -- No anon/authenticated client ever needs direct access — all reads/writes
  -- go through the backend. Add no permissive policy (RLS-on with no policy
  -- denies all PostgREST access), and revoke the default grants:
  revoke all on public.pi_discoveries from anon, authenticated;
  ```
- Stop persisting the ingest token in plaintext: store only a hash (mirroring `cameras.ingest_token_hash`) and deliver the plaintext solely in the announce HTTP response.
- Scope `GET /v1/cameras/discovered` per-caregiver (or behind device attestation) as already planned for v2.

---

## Medium

### 3. No rate limiting on the unauthenticated 6-digit pairing-code exchange (camera takeover)

**Location:** `src/api/pairing.py:40-98` (`POST /v1/cameras/pair`). Pairing code is 6 digits (`supabase/schema.sql:170`), TTL 600s (`settings.pairing_code_ttl_seconds`). No rate limiter exists anywhere in the codebase (grep for `ratelimit`/`slowapi`/`limiter` returns nothing in `src/`).

**Why it is exploitable:** The endpoint is unauthenticated by design — "the code IS the proof of authorization." A 6-digit code is a 10^6 keyspace, and there is no per-IP limit, no global limit, and no per-attempt lockout. An attacker who guesses an active code during its 10-minute window receives a long-lived `api_key` plus the camera's `patient_id`, `livekit_room`, and `livekit_identity` (`PairResponse`), letting them mint camera LiveKit tokens (`src/api/livekit.py`) and **publish arbitrary video into the patient's room**. The active-code window is small at low scale, but the absence of any throttling makes targeted brute force against a known pairing moment practical and undetectable.

**Remediation:** Add per-IP **and** global rate limiting to `/cameras/pair`; lock out a code after N failed attempts (and invalidate it); consider widening the code (8+ digits or alphanumeric) and emitting an alert on repeated failures. A shared limiter (e.g. Upstash Redis or a private-schema counter — not a public table) should front all unauthenticated endpoints.

### 4. Unauthenticated, unbounded `pi-announce` write path

**Location:** `src/api/pi_discovery.py:43-90` (`POST /v1/cameras/pi-announce`).

**Why it is exploitable:** The endpoint is intentionally unauthenticated (the Pi has no credentials pre-pairing) and upserts into `pi_discoveries` keyed by attacker-chosen `hardware_id` with no rate limit. An attacker can flood the table with arbitrary rows and arbitrary `device_label`/`current_url` values. Combined with Finding 2 (no RLS) and the unscoped discovered-list (Finding 3's note), this enables device-picker poisoning and storage-growth abuse. The risk is acknowledged in the module docstring as a v1 trade-off, but no compensating rate limit exists.

**Remediation:** Rate-limit by source IP; cap rows per IP/time window; implement the planned per-device secret (BLE-negotiated attestation) so announcements are attributable; enable RLS per Finding 2.

### 5. No per-user AI usage caps; no rate limiting on email (support/feedback) endpoints

**Location:** Detector pipeline `src/services/detection/providers/{gemini,openai}.py` (no per-caregiver token/cost accounting); `src/api/support.py:19-37` and `src/api/feedback.py` (Resend send on every authenticated call, no throttle).

**Why it is exploitable:** Per the `rate-limiting` and `ai-integration` guidance, expensive operations need hard, server-side per-user quotas — not just provider-side caps (which lag). There is no per-caregiver LLM spend ceiling in the application, and the support/feedback endpoints will dispatch an email to the founders' inboxes on every request, so a single authenticated account can be used for inbox flooding / Resend cost abuse. (Both are gated by auth, which bounds the blast radius relative to Findings 3–4.)

**Remediation:** Track per-caregiver/per-tier usage with hard limits and clear 429s; set hard spending caps in the OpenAI/Anthropic/Gemini consoles plus billing alerts; rate-limit the support/feedback endpoints per user and per IP.

---

## Low

### 6. LiveKit token `identity` is client-controlled and unvalidated

**Location:** `src/api/livekit.py:108` (viewer/broadcaster path sets `identity = body.identity`).

**Why it is exploitable:** A caregiver can request a token with any `identity` string. Room membership is correctly constrained to patients they own (`_ensure_caregiver_can_join`, `:168-193`), so the impact is limited to impersonating an arbitrary participant identity **within their own room**, but it weakens room-level identity guarantees and could confuse per-identity gating (e.g., the Talk-mode `set_talk_active` routing keyed on camera identities).

**Remediation:** Derive the viewer/broadcaster identity server-side from the authenticated caregiver id (e.g. `viewer_<caregiver_id>`), ignoring the client-supplied value.

### 7. Reusable, never-expiring demo pairing code matched in production

**Location:** `src/api/pairing.py:53-61` and `_pair_demo_camera:101-145`; `settings.demo_pairing_code` (`src/settings.py:131-137`).

**Why it is exploitable:** When `NEMARA_DEMO_PAIRING_CODE` is set (only in production, per the comment), the exact code bypasses expiry and single-use and is checked **before** the normal pairing path. Anyone who knows or guesses this static, long-lived code can repeatedly pair the seeded demo camera. Impact is low because it targets only the demo camera (no real patient data), but a short/guessable value combined with the lack of rate limiting (Finding 3) makes it a free, persistent foothold into the pairing surface.

**Remediation:** Use a long random value, rotate it, scope it to a non-production review build, or gate it behind an additional reviewer-only header. Ensure it can never collide with a real camera's data.

---

## Informational / Notes

### 8. Backend connects to Postgres as superuser — RLS is bypassed for backend queries by design

`.env.example:10` uses `postgresql+asyncpg://postgres:postgres@...`. The Postgres `postgres` role bypasses RLS, so the backend's queries are **not** protected by the policies in `schema.sql`; RLS only guards the direct Supabase client path (iOS Realtime/PostgREST). This is an intentional, common design — but it means the **app-layer `caregiver_id` scoping is the load-bearing access control** for everything the backend serves. That scoping was reviewed and is consistently applied (e.g. `src/repos/event_repo.py`, `patient_repo.py`; handlers pass `caregiver_id=auth.id` everywhere). Keep it that way: any new repo method that omits the `caregiver_id` filter becomes an IDOR. Consider running the backend under a non-superuser role that respects RLS as defense-in-depth.

### 9. Dependency vulnerability scanning not performed

This static review did not run a lockfile vulnerability scan (no offline advisory DB available in the audit environment). `uv.lock`/`pyproject.toml` are present and pinned. Add `pip-audit` (or `uv`'s audit tooling) and a CI job (e.g. Dependabot/Renovate) to catch known-vulnerable dependency versions.

### Positives observed (no action required)

- JWT verification uses signature-verifying `jwt.decode` with `algorithms=["ES256","HS256"]` and `audience="authenticated"`; the HS256 path uses the shared secret (not a JWKS public key), so the classic RS/HS algorithm-confusion attack does not apply (`src/services/auth.py`).
- All caregiver-facing endpoints scope by owner and return opaque 404s on cross-account access (`src/api/patients.py`, `events.py`, `talk.py`).
- LiveKit webhook signatures are verified before dispatch (`src/api/livekit_webhooks.py:65-75`).
- Stripe-style price manipulation is not applicable (no payment flow present).
- Upload size is capped while streaming to avoid memory-exhaustion (`src/api/patients.py:43-61`).
- ffmpeg/subprocess calls use argument lists via `create_subprocess_exec` (no shell), so no command injection (`src/services/detection/mp4_remux.py`, `frame_extraction.py`).
- Auth failures are flattened to a single opaque `invalid_auth` 401 to prevent token-state enumeration (`src/api/deps.py`).

---

## Prioritized Remediation Plan

1. **Rotate and purge all leaked secrets (Finding 1).** Confirm rotation of each provider credential — especially the APNs `.p8` private key — and rewrite git history. *Critical, do first.*
2. **Enable RLS on `pi_discoveries` and stop storing the ingest token in plaintext (Finding 2).** One migration; closes a remote, anon-key-only data-exposure and feed-spoofing path.
3. **Introduce rate limiting** across unauthenticated endpoints, starting with `/cameras/pair` and `/cameras/pi-announce` (Findings 3, 4), then AI/email cost paths (Finding 5).
4. **Harden the remaining items:** server-side LiveKit identities (6), demo-code hygiene (7), run the backend as a non-superuser DB role (8), and add dependency + secret scanning to CI (1, 9).
