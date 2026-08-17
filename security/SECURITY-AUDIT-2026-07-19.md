# Security Audit — nemara-backend

**Date:** 2026-07-19
**Auditor:** Automated static review (scheduled agent) using the `vibe-security` methodology
**Scope:** Static analysis of the `nemara-backend` repository (FastAPI + Supabase/Postgres, LiveKit, VLM detector pipeline). No live systems, databases, or secrets were accessed.
**Commit audited:** `c159356` on `main`

---

## Executive Summary

nemara-backend is a caregiver-monitoring backend: caregivers authenticate with Supabase JWTs, cameras (iPhone / Raspberry Pi / gateway) stream video that a VLM pipeline analyzes, and incident clips + patient data are stored in Supabase Storage. The core auth and data-access design is mostly sound — JWTs are verified correctly (signature, audience, algorithm allow-list), the primary caregiver-facing REST endpoints authorize by owner, and Row-Level Security is enabled with owner-scoped policies on the main application tables. Signed-URL endpoints (clip / photo) correctly scope by owner, so no IDOR was found there.

However, the audit found **two Critical** and **three High** issues that expose customer data and allow an attacker to inject rogue camera feeds into a victim's account:

1. **Real third-party API keys were committed to git history** (Gemini, OpenAI, LiveKit, Resend, and an APNs signing key) and there is no evidence they were rotated. Anyone with repo/history access can drain those billing accounts and forge push notifications.
2. **`POST /v1/gateway/provision-camera` has no real authentication.** It accepts an attacker-chosen bearer token as a valid "gateway box" and identifies the target account by a plaintext email in the request body. An attacker who knows a caregiver's email can provision cameras under **all** of that caregiver's patients and receive working ingest tokens — then push arbitrary video/audio that the caregiver sees as a legitimate camera and that feeds the alerting pipeline.
3. **Two Supabase tables (`pi_discoveries`, `gateways`) are missing Row-Level Security**, unlike every other table. Because clients talk to Supabase's REST API directly with the public anon/publishable key, these tables are world-readable/writable — leaking plaintext Pi ingest tokens and gateway↔caregiver mappings, and allowing tampering.
4. **No rate limiting exists anywhere**, including on the unauthenticated `POST /v1/cameras/pair` endpoint whose 6-digit pairing code is brute-forceable, and on the expensive VLM pipeline (no per-account cost cap).

The Critical items should be remediated immediately (rotate keys, put the gateway provisioning behind a real pre-shared secret handshake, add RLS to the two tables). Detailed findings, exact locations, exploitability, and fixes follow.

### Severity tally

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High     | 3 |
| Medium   | 3 |
| Low      | 3 |

---

## Critical

### C-1 — Real third-party API keys committed to git history (and likely not rotated)

**Where:** git history, commit `cf36938` ("chore(security): stop tracking .env"), plus the note in `.gitignore`.

**What:** The `.env` file was tracked before commit `cf36938` removed it. The removed file contained what appear to be **live** credentials:

- `NEMARA_GEMINI_API_KEY=AIza…RZ4c` (Google AI Studio key)
- `NEMARA_OPENAI_API_KEY=sk-proj-…jB1XzWmXFGWmblzSR4IwYQeou4YA`
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` (two rotations visible in history)
- `NEMARA_RESEND_API_KEY=re_N3Fo9qHR_…`
- `NEMARA_APNS_AUTH_KEY` (a full `-----BEGIN PRIVATE KEY-----` ES256 `.p8` used to sign Apple push tokens)
- `NEMARA_APNS_KEY_ID=TTTD2823D2`, plus the Supabase publishable key.

(The `NEMARA_SUPABASE_SERVICE_ROLE_KEY` and `NEMARA_SUPABASE_JWT_SECRET` in that same file are the well-known Supabase **local-dev demo** values — `iss: "supabase-demo"`, the literal `super-secret-jwt-token-with-at-least-32-characters-long` — so those two are not real secrets. The five listed above are the concern.)

**Why it's exploitable:** Deleting a file does not remove it from history — `git log -p -- .env` recovers every value. Anyone who ever cloned the repo, any CI cache, or anyone who gains read access to the repo has these keys. The commit message itself says *"Secrets in prior history still require rotation,"* while the current `.gitignore` claims they were *"rotated"* — a direct contradiction, so rotation is unverified. Impact of live keys:
- Gemini / OpenAI keys → unlimited model spend billed to the company (thousands of dollars possible in hours).
- LiveKit key/secret → an attacker can mint their own room tokens and join/publish to any patient room, bypassing this backend entirely.
- Resend key → send email as the company's verified domain (phishing).
- APNs `.p8` → forge push notifications to every caregiver device (fake alerts / social engineering). This key is per-Apple-developer-account and cannot be scoped down.

**Fix:**
1. **Rotate all five credentials now** (Gemini, OpenAI, LiveKit key+secret, Resend, APNs `.p8` — generate a new key ID). Treat them as compromised.
2. Reconcile the contradictory documentation and confirm rotation date.
3. Optionally purge history (`git filter-repo`) — but rotation is the real fix; purging does not help keys already scraped.
4. Add a pre-commit secret scan (`gitleaks`) so this cannot recur.

---

### C-2 — `POST /v1/gateway/provision-camera`: unauthenticated, account-targeted camera injection

**Where:** `src/api/cameras_ingest.py:472-589` (`gateway_provision_camera`); router mounted unconditionally in `src/main.py:178`.

**What:** The endpoint's only "auth" is a bearer token that the *caller* invents. There is no server-side registry of legitimate box tokens checked **before** provisioning — the handler simply hashes whatever bearer is presented and, if that hash isn't already in `public.gateways`, **inserts a new gateway row** tying that attacker-chosen token to the caregiver named by `body.email`:

```python
box_token = authorization[len("bearer "):].strip()      # attacker-chosen, line 495
cg = (await session.execute(                              # target chosen by email, lines 500-503
    text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
    {"e": body.email.strip()},
)).first()
...
if existing is None:                                      # self-registers the token, lines 517-522
    await session.execute(
        text("INSERT INTO public.gateways (caregiver_id, token_hash, created_at) ..."),
        {"c": caregiver_id, "h": token_hash, "n": now})
```

It then creates a camera under **every** patient of that caregiver and returns each camera's **plaintext ingest token** (lines 539-589).

**Why it's exploitable:** Email is an identifier, not a secret. An attacker who knows/guesses a caregiver's email (`aryaman@memoria-health.com` and `sunny@memoria-health.com` are even hard-coded in `src/settings.py:456`) can:
1. `POST /v1/gateway/provision-camera` with `Authorization: Bearer attacker-picked-token` and `{"email": "victim@example.com"}`.
2. Receive `ingest_token` for a new camera on each of the victim's patients.
3. `POST /v1/cameras/me/frames` with that token to publish **attacker-controlled video/audio** into the victim's LiveKit patient rooms — the caregiver's app renders it as a real camera and the VLM pipeline runs on it (fabricated "all quiet," or fake incidents).
4. Poll `GET /v1/gateway/roster` / `/gateway/wait` with the self-registered token to enumerate the victim's cameras (`livekit_identity`, `livekit_room`, `desired_state`).

This is a full broken-access-control / account-targeting flaw against a system that streams live audio/video of vulnerable people in their homes.

**Fix:**
- Do **not** let the box self-assert its token. Provision boxes out-of-band: pre-generate a box token/secret, store its hash server-side, and reject provisioning for any token not already registered (`existing is None` → `401`, not "insert new gateway").
- Do not identify the account by an email in the body. Require the caregiver to authorize the pairing (e.g. a short-lived, caregiver-authenticated pairing code the box presents), the same trust model already used for camera pairing.
- Rate-limit and log provisioning attempts.

---

## High

### H-1 — `public.pi_discoveries` has no Row-Level Security (leaks plaintext Pi ingest tokens)

**Where:** table defined in `supabase/schema.sql:334-377` and `scripts/apply_pi_schema_delta.sql:85`; **absent** from the RLS block (`supabase/schema.sql:720-801`). Confirmed: no `enable row level security` / `create policy` for this table in either file.

**What / why exploitable:** `supabase/config.toml` exposes the `public` schema via the Supabase REST API, and clients (iOS) hit that API directly with the **publishable/anon key**, which is public by design (extractable from the app bundle). Every other table has RLS enabled; `pi_discoveries` does not, so it is fully readable and writable by anyone holding the anon key. The row contains **`pending_ingest_token`** — the *plaintext, one-shot* bearer a Pi uses to authenticate to `POST /v1/cameras/me/frames`. An attacker can:
- `GET` all discovery rows and read `pending_ingest_token` before the Pi consumes it → steal a camera's ingest credential and push frames (same impact as C-2).
- Read `hardware_id` + `current_url` (customers' local-network MJPEG camera URLs) — a device/location inventory.
- `INSERT`/`UPDATE`/`DELETE` rows: spoof announcements, overwrite tokens, or unclaim cameras.

**Fix:** enable RLS and add a policy (there is no caregiver column, so the correct policy is "no direct client access" — deny all and let only the service-role backend touch it):
```sql
alter table public.pi_discoveries enable row level security;
-- no policy => PostgREST (anon/authenticated) is blocked; the backend uses service_role, which bypasses RLS.
```
Also stop storing the ingest token in plaintext in an API-reachable table.

---

### H-2 — `public.gateways` has no Row-Level Security

**Where:** `scripts/apply_gateway_delta.sql:38-60` creates the table with **no** `enable row level security` and no policy (every other delta — `caregiver_devices`, `patient_face_embeddings`, `feedback`, `ui_events` — does enable RLS).

**What / why exploitable:** Same exposure vector as H-1 (public schema + anon key). `gateways` holds `token_hash` and `caregiver_id`. With no RLS an anon client can:
- Read the full caregiver↔gateway mapping (which accounts have gateways).
- `INSERT` a gateway row binding an attacker-chosen `token_hash` to any `caregiver_id`, then drive `GET /v1/gateway/roster` with that token to enumerate the victim's cameras — a database-side variant of C-2 that doesn't even need the provisioning endpoint.

**Fix:**
```sql
alter table public.gateways enable row level security;
-- deny all direct client access; only the service-role backend reads/writes this table.
```

---

### H-3 — No rate limiting anywhere (auth, pairing, and expensive VLM paths)

**Where:** entire `src/api/` surface. No `slowapi`, no limiter middleware, no per-IP/per-user counters (grep for rate limiting found nothing). Notably:
- `POST /v1/cameras/pair` (`src/api/pairing.py:40-98`) — **unauthenticated**, and the credential is a **6-digit** `pairing_code` valid for 10 minutes (`settings.pairing_code_ttl_seconds=600`).
- `POST /v1/gateway/provision-camera`, `POST /v1/cameras/pi-announce`, `POST /v1/support`, `POST /v1/feedback` — all reachable without limits.
- The VLM detector pipeline (Gemini/OpenAI/OpenRouter/DashScope) has **no per-account usage cap** — cost scales with the number of cameras/patients/detectors a caregiver creates.

**Why exploitable:** With the whole 6-digit space (1,000,000 codes) and a live 10-minute pairing window, an attacker can brute-force an in-flight pairing code and hijack the pairing (receiving the camera's long-lived `api_key`) with no lockout. More broadly, unlimited hits on the email/support/announce/provision endpoints enable spam relaying, DB flooding, and (via the model calls) billing exhaustion — the classic "drain the AI budget" abuse.

**Fix:**
- Add rate limiting (per-IP **and** per-account) via `slowapi`/Redis on all unauthenticated and expensive endpoints. On `/v1/cameras/pair`, cap attempts per code and per IP and lock after a handful of failures; consider a longer, higher-entropy pairing code.
- Enforce per-account VLM quotas with hard caps in addition to provider-side spend caps.

---

## Medium

### M-1 — 6-digit pairing code is low-entropy (compounds H-3)

**Where:** `src/api/pairing.py:63-75`, `src/db/camera.py` (`pairing_code`).
A 6-digit numeric code is ~20 bits. Even with rate limiting it is a weak bearer for handing out a long-lived camera `api_key`. **Fix:** use a longer alphanumeric code (or a signed, single-use token), keep the short TTL, and enforce attempt caps.

### M-2 — Unbounded VLM model cost per account (no billing protection)

**Where:** detector pipeline (`src/services/detection/worker.py`, `camera_ingest.py`) driven by per-camera ticks; provider keys in `src/settings.py:62-129`.
There is no per-caregiver ceiling on how many cameras/detectors run or how many model calls they generate. A single account (or the C-2/H-1 injection paths) can multiply model spend without limit. The `vibe-security` billing-protection guidance calls this out explicitly. **Fix:** per-account concurrency/quota limits and hard provider spend caps with alerting.

### M-3 — Schema drift: canonical `schema.sql` does not match production, hiding RLS gaps

**Where:** `supabase/schema.sql` vs `scripts/apply_*_delta.sql`. Production tables/columns (`gateways`, `cameras.last_heartbeat_at`, `source_type='gateway'`) exist only in ad-hoc delta scripts, and `schema.sql`'s `cameras` CHECK constraint still forbids `'gateway'`. Because there is no single authoritative migrated schema, RLS coverage cannot be verified from one file — and this is exactly how H-1/H-2 slipped through. **Fix:** adopt real migrations (or regenerate `schema.sql` from prod) and add a CI check asserting every `public` table has RLS enabled (the `pg_tables` loop in the vibe-security database reference).

---

## Low

### L-1 — Interactive API docs and OpenAPI schema are public in production

**Where:** `src/main.py:166-171` — `FastAPI(...)` is created without `docs_url=None`/`redoc_url=None`/`openapi_url=None`; `main.py:56-68` even adds a log filter to quiet `/docs`, `/openapi.json`, `/redoc`, implying they are live in prod. This publishes the full endpoint map (including `/v1/gateway/provision-camera`) to anyone. **Fix:** disable docs in non-local envs (gate on `settings.env`).

### L-2 — Caregiver-supplied detector prompts are concatenated into VLM prompts (prompt injection)

**Where:** `src/services/detection/prompt.py` (fragments assembled into `BASE_PROMPT`); `detectors.custom_prompt` / `prompt_fragment` are caregiver-authored (`supabase/schema.sql:403,426`). A caregiver can inject instructions into the model prompt. Impact is low today: it is self-scoped (only affects that caregiver's own patient's analysis), and the model output is consumed as structured JSON internally, not rendered as HTML or executed. **Fix:** keep user text in a clearly delimited user block, validate/limit length, and treat model output as untrusted (already effectively the case).

### L-3 — Persistent, reusable demo pairing code in production

**Where:** `src/settings.py:187-193` (`demo_pairing_code`), `src/api/pairing.py:59-61,101-145`. When set (prod only), this code never expires and is reusable, and pairing against the demo camera re-mints an `api_key` on every call. If the value leaks, anyone can repeatedly pair against the seeded demo camera. Impact is limited to the single demo camera/row. **Fix:** scope tightly, rotate periodically, and ensure the demo camera is never attached to real patient data.

---

## What was checked and found OK

- **JWT verification** (`src/services/auth.py`) — uses `jwt.decode` **with** signature verification, an explicit `algorithms` allow-list (`ES256`/`HS256`, so `alg:none` is rejected), and audience (`authenticated`) + expiry validation. Correct.
- **RLS on core tables** — `caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`, `feedback`, `ui_events` all have RLS enabled with owner-scoped `USING` **and** `WITH CHECK` clauses. Good.
- **Signed-URL endpoints** — `GET /v1/events/{id}/clip-url` and `GET /v1/patients/{id}/photo-url` scope the lookup by `caregiver_id=auth.id` (`src/api/events.py:144`, `src/api/patients.py:340-362`); no IDOR.
- **LiveKit token minting** (`src/api/livekit.py`) — camera tokens are constrained to the camera's own room/identity; caregiver tokens are constrained to `patient_<id>` rooms they own; privacy-mode is baked into grants. Sound.
- **LiveKit webhook** (`src/api/livekit_webhooks.py`) — verifies the signed body via `TokenVerifier` before acting.
- **SQL** — raw `text()` queries in `cameras_ingest.py` use bound parameters (`:c`, `:d`, `:h`); no string-concatenated user input. No SQL injection found.
- **Service-role key** — used only server-side in `src/services/storage.py`; never returned to clients.
- **CORS** — no permissive `*` CORS middleware is configured; for a mobile/device backend this is appropriately restrictive.
- **Mass assignment** — caregiver provisioning takes `id`/`email` from the verified JWT, not the body (`src/services/onboarding.py:74-82`); PATCH uses an allow-listed field set.

---

## Prioritized remediation checklist

1. **(C-1)** Rotate the Gemini, OpenAI, LiveKit, Resend, and APNs credentials now; confirm and document rotation; add `gitleaks` to CI.
2. **(C-2)** Rework `POST /v1/gateway/provision-camera` to require a pre-registered box secret and caregiver-authorized pairing; stop trusting a body email + self-asserted token.
3. **(H-1, H-2)** `ALTER TABLE ... ENABLE ROW LEVEL SECURITY` on `pi_discoveries` and `gateways`; stop storing plaintext ingest tokens in an API-reachable table.
4. **(H-3, M-1, M-2)** Add per-IP + per-account rate limiting (pairing, announce, provision, support/feedback, model calls); strengthen the pairing code; enforce per-account VLM quotas + hard provider spend caps.
5. **(M-3)** Move to real migrations and add a CI assertion that every `public` table has RLS enabled.
6. **(L-1, L-3)** Disable `/docs` + OpenAPI in production; tighten the demo pairing code.
