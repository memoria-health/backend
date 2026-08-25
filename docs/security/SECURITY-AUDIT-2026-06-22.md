# Security Audit — nemara-backend

**Date:** 2026-06-22
**Auditor:** Automated static security review (vibe-security methodology)
**Scope:** Static source review of the `nemara-backend` FastAPI service (API routes, auth, DB/RLS schema, AI integration, deployment config, git history). No live systems, databases, or secrets were accessed.
**Commit audited:** `4873148` (branch `main`)

---

## Executive Summary

`nemara-backend` is a well-structured FastAPI backend for a caregiver monitoring product (Supabase Postgres + Auth, LiveKit, APNs, Gemini/OpenAI detector pipeline, Re-ID). The application-layer security posture is, in most respects, **good**: every Supabase table has Row-Level Security enabled with correctly owner-scoped `USING` **and** `WITH CHECK` clauses; the API layer adds defense-in-depth `caregiver_id` filtering on every query (no IDOR found); update schemas use Pydantic `extra="forbid"` (no mass assignment found); database access goes through the SQLAlchemy ORM with parameterized queries (no SQL injection found); LiveKit token minting and webhook signatures are properly verified; uploads are size-capped and streamed; and signed Storage URLs are short-lived.

However, the audit found **one Critical issue that must be remediated immediately**: a `.env` file containing **real production-grade secrets** (Supabase JWT signing secret, Supabase `service_role` key, OpenAI/Gemini/LiveKit API keys, Resend key, and a full APNs `.p8` private key) was committed to git history and remains retrievable from history today, even though the file has since been untracked. Because the backend accepts HS256 caregiver JWTs signed with the leaked `supabase_jwt_secret`, and because the `service_role` key bypasses all RLS, this single leak is a **complete authentication and database-authorization bypass** for anyone with read access to the repository history.

Beyond that, the most material weaknesses are: **no rate limiting anywhere in the service** (the unauthenticated 6-digit pairing-code exchange is brute-forceable to take over a camera), an **unauthenticated Pi "announce" endpoint that hands out one-time ingest tokens keyed only on a (non-secret) MAC address**, and a **bcrypt linear-scan camera-auth path invoked on a 1-second poll loop** that is a denial-of-service amplifier at scale.

### Findings by severity

| # | Severity | Title |
|---|----------|-------|
| 1 | **Critical** | Real secrets committed to git history (`.env`): JWT secret, `service_role` key, API keys, APNs private key |
| 2 | **High** | Unauthenticated Pi `pi-announce` discloses one-time ingest token keyed on a guessable hardware_id |
| 3 | **High** | No rate limiting on the unauthenticated pairing-code exchange → camera-takeover brute force |
| 4 | **Medium** | Camera-auth bcrypt linear scan executed on a 1 s poll → CPU exhaustion / DoS amplifier |
| 5 | **Medium** | Reusable, never-expiring demo pairing code bypasses the pairing window |
| 6 | **Medium** | No per-user quota / rate limit on LLM-backed endpoints → unbounded provider cost |
| 7 | **Low** | LLM-generated event text rendered as unescaped HTML in alert emails |
| 8 | **Low** | `/ready` leaks raw database exception strings to unauthenticated callers |
| 9 | **Low** | Production container runs as `root` |
| 10 | **Low** | JWT verifier accepts HS256 alongside ES256, widening the forgery surface |

---

## Critical

### 1. Real secrets committed to git history (`.env`)

**Where:** `.env` (tracked from commit `e482e90`, untracked in `cf36938`); the file and its secrets are still present in git history. Confirmed via `git show cf36938~1:.env`.

**What was leaked** (verified present in history; values redacted here):

- `NEMARA_SUPABASE_JWT_SECRET` — the HS256 signing secret for caregiver auth tokens (`super-…`).
- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` — the Supabase service-role JWT (`eyJhbG…`), which **bypasses all RLS**.
- `NEMARA_SUPABASE_PUBLISHABLE_KEY` (`sb_pub…`).
- `NEMARA_GEMINI_API_KEY` (`AIzaSy…`) and `NEMARA_OPENAI_API_KEY` (`sk-pro…`).
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET`.
- `NEMARA_APNS_AUTH_KEY` — a complete ES256 `-----BEGIN PRIVATE KEY----- … -----END PRIVATE KEY-----` `.p8` body, printed in full in history, plus `NEMARA_APNS_TEAM_ID` / `NEMARA_APNS_KEY_ID`.
- `NEMARA_RESEND_API_KEY` (`re_…`).

**Why it is exploitable:**

- **Full auth bypass.** `src/services/auth.py` (`verify_supabase_jwt`) accepts `HS256` tokens verified with `supabase_jwt_secret`. An attacker with the leaked secret can forge a valid caregiver JWT for **any** `sub`/`email` (`{"sub": "<any-uuid>", "email": "x@y.com", "aud": "authenticated"}`), passing `caregiver_auth` in `src/api/deps.py:66` and impersonating any caregiver.
- **Full database compromise.** The `service_role` key bypasses every RLS policy in `supabase/schema.sql`. Anyone holding it can read/modify/delete every caregiver's patients, cameras, events, face embeddings, and storage objects directly against the Supabase REST/Storage API.
- **Spend / abuse.** The OpenAI, Gemini, LiveKit and Resend keys allow unbounded billing on the company's accounts and outbound email spoofed as the product.
- **Push impersonation.** The APNs `.p8` private key allows forging push notifications to all caregivers.

The `.gitignore` comment claims the leaked secrets were "rotated." This audit **cannot verify rotation** and the secrets remain in history regardless; treat all of the above as compromised until each is independently confirmed rotated.

**Remediation:**

1. **Rotate every secret listed above, now** — Supabase JWT secret and `service_role` key (Supabase dashboard → API → roll keys), OpenAI/Gemini/LiveKit/Resend keys, and the APNs auth key (revoke the key ID in the Apple Developer portal and issue a new `.p8`).
2. **Purge the secrets from git history** (`git filter-repo --invert-paths --path .env`, or BFG), then force-push and have all collaborators re-clone. Removing the file in a later commit is **not** sufficient.
3. Add a pre-commit secret scanner (`gitleaks`) and CI gate so this cannot recur. `.env` is now correctly gitignored — keep it that way.
4. Confirm production secrets live only in the Railway environment, never in any committed file.

---

## High

### 2. Unauthenticated `pi-announce` discloses a one-time ingest token keyed on a guessable hardware_id

**Where:** `src/api/pi_discovery.py:48-90` (`POST /v1/cameras/pi-announce`), `src/api/cameras_ingest.py:101-205` (`POST /v1/cameras/me/frames`), and `pi_discovery_repo.consume_pending_ingest_token`.

**Why it is exploitable:** The announce endpoint is intentionally unauthenticated and identifies a device solely by `body.hardware_id` (a MAC address). When a caregiver provisions a Pi, a plaintext `pending_ingest_token` is stashed on the discovery row and returned in the **next announce response** for that `hardware_id`, then nulled (one-time read). An attacker who knows or guesses a target's `hardware_id` — it is a MAC address, and the human-readable `device_label` even embeds "the last 6 of MAC" — can poll `pi-announce` with that `hardware_id`. If their request wins the race against the legitimate Pi (or simply arrives in the window between provisioning and the Pi's next 60 s announce), they receive the plaintext ingest token. With it they can call `POST /v1/cameras/me/frames` and **publish arbitrary video into a patient's live monitoring feed** as that camera — directly defeating a safety product (e.g. masking a fall by injecting a benign clip). The unauthenticated insert path also allows unbounded `pi_discoveries` row creation (storage/DoS) and flooding the caregiver's "discovered devices" picker with spoofed entries.

**Remediation:**

- Bind announcement/claim to a per-device secret established out-of-band during physical provisioning (the module docstring already plans this for "v2" — it is needed now). Do not return a bearer credential to a caller authenticated only by a MAC address.
- At minimum, require the ingest token to be delivered over the BLE/captive-portal provisioning channel rather than the open announce response, and rate-limit + size-cap the announce endpoint per IP.
- Treat `hardware_id` as an identifier, never as an authenticator.

### 3. No rate limiting on the unauthenticated pairing-code exchange (camera-takeover brute force)

**Where:** `src/api/pairing.py:40-98` (`POST /v1/cameras/pair`); no rate-limiting middleware exists anywhere in `src/` (confirmed — no `slowapi`/`Ratelimit`/`429` usage).

**Why it is exploitable:** `pair_camera` looks up a camera by a **6-digit** `pairing_code` (10⁶ keyspace) valid for 10 minutes, with no per-IP or global throttle. On success the caller receives a long-lived `api_key` plus everything needed to publish to the camera's LiveKit room and mint viewer/broadcaster tokens. With zero rate limiting an attacker can hammer the endpoint to guess an in-flight pairing code and **hijack the pairing of a camera being set up**, gaining publish access to a patient's room and a durable credential. The same absence of rate limiting exposes every other sensitive surface (token mint, ingest, account endpoints) to brute force and abuse.

**Remediation:**

- Add rate limiting (per-IP and global) to `POST /v1/cameras/pair` — e.g. a few attempts per IP per minute and a hard cap on failed attempts per code. Lock/rotate a pairing code after N failures.
- Lengthen the pairing code (e.g. 8+ alphanumeric, ~10¹² keyspace) so the 10-minute window is not brute-forceable even with throttling.
- Introduce a shared rate-limit layer (Upstash Redis / middleware) and apply it to auth, pairing, ingest, and LLM-backed routes (see #6).

---

## Medium

### 4. Camera-auth bcrypt linear scan executed on a 1-second poll loop

**Where:** `src/api/deps.py:111-159` (`camera_auth`); consumed by `GET /v1/cameras/me/desired-state` (`src/api/cameras.py:557-595`), which the docstring notes the wall phone polls **every 1 second**, and by `src/api/livekit.py`.

**Why it is exploitable:** `camera_auth` selects **all** camera rows with a non-null `api_key_hash` and runs `bcrypt.checkpw` (cost `2^10`) against each until a match. A presented invalid token forces a full scan of every camera in the system on every request. Because the legitimate wall phone polls `desired-state` once per second, steady-state cost already grows as O(cameras²) bcrypt operations per second across the fleet; an unauthenticated attacker sending invalid bearer tokens can amplify this into CPU exhaustion / denial of service with very few requests. The code comments acknowledge the linear scan is "fine for small N" — it is a latent scaling and DoS hazard.

**Remediation:** Store an indexed, non-secret `api_key_id`/prefix on the camera row and look the row up by that in O(1), then run a single `bcrypt.checkpw` against that one row's hash. Combine with rate limiting (#3). This removes the per-request fan-out entirely.

### 5. Reusable, never-expiring demo pairing code bypasses the pairing window

**Where:** `src/settings.py:137` (`demo_pairing_code`), `src/api/pairing.py:53-61,101-145` (`_pair_demo_camera`).

**Why it is exploitable:** When `NEMARA_DEMO_PAIRING_CODE` is set (the comment states **production is the only env that sets it**), `POST /v1/cameras/pair` special-cases that exact code to pair the demo camera with **no expiry and no single-use clearing**, re-minting a fresh `api_key` on every call. Anyone who learns or guesses this static code (e.g. from an App Review build, support docs, or a short value) can repeatedly pair the demo camera, obtain valid api_keys, and join its LiveKit room. Impact is confined to the seeded demo camera, but it is a permanent unauthenticated credential issuer in production.

**Remediation:** Make the demo code long and high-entropy, scope it to a non-production-data demo patient only, and prefer gating the demo-pair path behind a dedicated review build rather than a production setting. Rotate the value periodically.

### 6. No per-user quota or rate limit on LLM-backed endpoints

**Where:** `src/services/detector_compiler.py` (`compile_custom_detector`, `infer_detector_config`) invoked from the detector-create API; the detector pipeline providers in `src/services/detection/providers/`.

**Why it matters:** Custom-detector create/infer triggers OpenAI and Gemini calls with no per-caregiver usage cap and no rate limiting. An authenticated caregiver (or an attacker using a forged token per #1) can spam these to drive provider cost. The skill's guidance is to enforce hard per-user quotas in-app rather than relying solely on provider-side caps.

**Remediation:** Add per-user/day quotas on detector compile/infer and any other LLM-invoking route, set hard spending caps in the OpenAI/Anthropic/Gemini dashboards, and apply the shared rate-limit layer from #3.

---

## Low

### 7. LLM-generated event text rendered as unescaped HTML in alert emails

**Where:** `src/services/email.py:124-135` (`_event_html`) — interpolates `title`, `narrative`, `location` directly into HTML. (Note: the support and feedback paths *do* escape via `_support_html`/`_feedback_html`; the event path does not.)

**Why it matters:** `narrative`/`title` originate from the Gemini/OpenAI detector output (untrusted per the skill's "LLM output is untrusted" rule). Rendered unescaped, model output containing HTML could distort or inject markup into alert emails. Impact is limited by the email client's HTML sandbox and that the content is not direct attacker input, but it is inconsistent with the escaping already applied elsewhere.

**Remediation:** HTML-escape `title`, `narrative`, and `location` in `_event_html` exactly as `_support_html`/`_feedback_html` already do.

### 8. `/ready` leaks raw database exception strings

**Where:** `src/api/health.py:39-49` — on DB failure returns `detail={..., "error": str(exc)}` to an unauthenticated caller.

**Why it matters:** SQLAlchemy/asyncpg exception strings can disclose database host, driver, schema, or connection details useful for reconnaissance.

**Remediation:** Return a generic `{"status": "fail", "checks": {"db": "fail"}}` to the client and log the detailed exception server-side only.

### 9. Production container runs as root

**Where:** `Dockerfile` — no `USER` directive; the app runs as `root`.

**Why it matters:** A code-execution or container-escape bug runs with maximum in-container privileges. Standard hardening is to run as an unprivileged user.

**Remediation:** Create and switch to a non-root user (`RUN useradd -m app && USER app`) before `CMD`, ensuring the venv and any model cache paths are owned/readable by that user.

### 10. JWT verifier accepts HS256 alongside ES256

**Where:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`) and the alg-dispatch in `verify_supabase_jwt`.

**Why it matters:** Production Supabase tokens are ES256 (asymmetric, verified via JWKS). Continuing to accept HS256 (symmetric, verified with the shared `supabase_jwt_secret`) keeps a symmetric-secret forgery path alive — which is exactly what made finding #1 a full auth bypass. Narrowing the accepted algorithms reduces the blast radius of any future secret leak.

**Remediation:** In production, restrict accepted algorithms to `["ES256"]` and verify only via JWKS; reserve HS256 for local/test environments via configuration. Continue to validate `aud`, and add `iss`/expiry checks as available.

---

## Areas Reviewed With No Findings

- **Supabase RLS (`supabase/schema.sql`):** every table has `ENABLE ROW LEVEL SECURITY` with owner-scoped policies using `auth.uid()` and matching `WITH CHECK` clauses (no `USING (true)`, no `auth.uid() IS NOT NULL`-only policies). Sensitive caregiver/patient state columns are not caregiver-writable (enforced by `extra="forbid"` schemas). The Realtime publication is correctly limited to owner-scoped tables.
- **IDOR / authorization:** every caregiver-facing query filters by `caregiver_id=auth.id` (repos + RLS, defense-in-depth); cross-account access returns 404, not 403. Camera/LiveKit/Talk/record-clip paths verify patient ownership before acting.
- **Mass assignment:** PATCH/PUT handlers use Pydantic models with `extra="forbid"`; request bodies are never spread into ORM writes.
- **SQL injection:** all DB access via SQLAlchemy ORM / parameterized constructs; no string-concatenated SQL or `text()` with user input found.
- **LiveKit:** token minting validates camera↔room/identity match and caregiver↔patient ownership; webhook signatures verified via `WebhookReceiver`/`TokenVerifier` against the raw body.
- **Payments:** none present (no Stripe/billing code).
- **Dependencies:** resolved versions in `uv.lock` (FastAPI 0.136.1, Starlette 1.0.1, Pillow 12.2.0, cryptography 48.0.0, PyJWT 2.13.0, urllib3 2.7.0, requests 2.34.2, aiohttp 3.13.5) are current; no specific known-vulnerable pin was substantiated. Recommend wiring `pip-audit`/`uv`-based vulnerability scanning into CI as ongoing hygiene.
- **CORS:** no CORS middleware is configured, which is appropriate for a native-iOS + Pi backend (no browser cross-origin surface). No `Access-Control-Allow-Origin: *` exposure.
- **Debug/config:** no debug flags or `--reload` in production; experimental `src/internals/` routes are gated behind `NEMARA_ENABLE_INTERNALS` (default off) and still enforce caregiver ownership.
- **Uploads:** photo/clip uploads are streamed and size-capped (413 on overflow) with a content-type allowlist.

---

## Prioritized Remediation Plan

1. **(Critical #1)** Rotate **all** leaked secrets immediately; purge `.env` from git history; add `gitleaks` to CI.
2. **(High #2)** Stop returning ingest tokens to MAC-only-identified callers; bind provisioning to a physically-established per-device secret; rate-limit/size-cap `pi-announce`.
3. **(High #3)** Add rate limiting (per-IP + global) and lengthen the pairing code.
4. **(Medium #4)** Replace the bcrypt linear scan with an indexed api-key-prefix lookup.
5. **(Medium #5, #6)** Harden the demo pairing code; add per-user LLM quotas and provider spend caps.
6. **(Low #7–#10)** Escape LLM text in emails; stop leaking DB errors from `/ready`; run the container as non-root; pin JWT verification to ES256 in production.
