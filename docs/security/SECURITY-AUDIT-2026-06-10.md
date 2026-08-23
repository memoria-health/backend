# Security Audit — nemara-backend

**Date:** 2026-06-10
**Scope:** Static source audit of the `nemara-backend` FastAPI service (API routes, auth, DB layer, Supabase schema/RLS, detector/LLM pipeline, ingest, deployment config, dependencies).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill — secrets, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment, data access/input validation.
**Constraint:** Static only. No live systems, databases, or secrets were accessed. Findings are substantiated against the checked-out source.

---

## Executive Summary

The backend is, on the whole, **well-engineered for authorization**: every caregiver-facing endpoint goes through a verified Supabase JWT dependency, repository queries are consistently scoped by `caregiver_id` (defense-in-depth on top of RLS), update payloads are constrained by Pydantic schemas with `extra="forbid"` (no mass-assignment), LiveKit token minting enforces room/identity ownership, the Stripe-style webhook (LiveKit) verifies signatures, and dependencies are on current, patched versions. There are **no payment flows** in scope and **no client-trusted pricing**.

However, two **Critical** issues undermine that posture, and both are textbook "vibe-coded" mistakes:

1. **A tracked `.env` with live third-party secrets** (OpenAI, Gemini, LiveKit API secret, Resend, APNs signing key) committed to git history. These are exposed to everyone with repo access and persist in history even if deleted.
2. **One Supabase table (`pi_discoveries`) is missing Row-Level Security** while the schema comment asserts "Every table is RLS-on." Because Supabase exposes every `public` table through PostgREST under the public anon/publishable key, this table is world-readable and world-writable — leaking per-device plaintext ingest bearer tokens and letting anyone poison, hijack, or delete camera-discovery records across all customers.

Two **High** issues follow from a complete absence of rate limiting: the 6-digit camera pairing code is brute-forceable, and the per-request linear bcrypt scan in camera auth is a cheap CPU-exhaustion DoS.

### Severity-ranked findings

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | **Critical** | Live API secrets committed in tracked `.env` | `.env`, `.gitignore` |
| 2 | **Critical** | `pi_discoveries` table missing RLS — exposed via public anon key | `supabase/schema.sql:310`, RLS block `:619-700` |
| 3 | **High** | No rate limiting → 6-digit pairing code brute-forceable | `src/api/pairing.py:63`, app-wide |
| 4 | **High** | Per-request linear bcrypt scan = CPU-exhaustion DoS | `src/api/deps.py:111-159` |
| 5 | **Medium** | Unauthenticated, unscoped Pi discovery spoofing/poisoning | `src/api/pi_discovery.py:43-122` |
| 6 | **Low** | `/ready` leaks raw DB exception string | `src/api/health.py:50` |
| 7 | **Low** | Unescaped LLM/event content in alert email HTML | `src/services/email.py:115-126` |
| 8 | **Low** | `SECURITY DEFINER` trigger uses `search_path = public` not `''` | `supabase/schema.sql:53-68` |
| 9 | **Low** | No security headers / no rate limit on AI-driving paths; DEBUG logging in `.env` | `src/main.py`, `.env` |

---

## Critical

### 1. Live API secrets committed in a tracked `.env` file

**`.env` (tracked in git, present across history), `.gitignore:11-15`**

`.env` is **intentionally tracked** (`.gitignore` comments confirm this and `git log -- .env` shows it across many commits). It contains what appear to be **real, production third-party credentials**:

- `NEMARA_OPENAI_API_KEY=sk-proj-YbEoTJIg...` (OpenAI project key)
- `NEMARA_GEMINI_API_KEY=AIzaSyDW3q...` (Google Gemini key)
- `NEMARA_LIVEKIT_API_KEY=APItudo6ecfroAu` / `NEMARA_LIVEKIT_API_SECRET=eQslXjZ7OH...` (LiveKit Cloud — can mint join/publish tokens for any room)
- `NEMARA_RESEND_API_KEY=re_N3Fo9qHR_...` (Resend — can send mail as your domain)
- `NEMARA_APNS_AUTH_KEY="-----BEGIN PRIVATE KEY----- ..."` (APNs ES256 .p8 signing key — can push notifications to all users)

**Why it's exploitable:** Anyone with read access to the repository (or any fork/clone, or anyone who obtains a copy of git history) extracts these keys directly. The OpenAI and Gemini keys allow unlimited model usage billed to you; the LiveKit secret lets an attacker mint tokens to join/publish into any patient's live video room; the Resend key enables phishing from your verified domain; the APNs key forges push notifications to every caregiver. The `.gitignore` comment ("Move it back to gitignored before flipping the repo public — all secrets need rotation") shows the team already knows these need rotation — but they are exposed **now**, to everyone with repo access, and deleting the file will not purge them from history.

> Note: `NEMARA_SUPABASE_JWT_SECRET` (`super-secret-jwt-token-with-at-least-32-characters-long`) and `NEMARA_SUPABASE_SERVICE_ROLE_KEY` (`...supabase-demo...`) are the **well-known Supabase local-dev defaults**, not production values — but if any environment is ever started with the HS256 local secret, anyone can forge a valid caregiver JWT (`verify_supabase_jwt` accepts HS256 with this secret). Ensure production uses ES256/JWKS only (it does today) and never ships these defaults.

**Remediation:**
1. **Rotate every real key now** (OpenAI, Gemini, LiveKit API key+secret, Resend, APNs .p8) — treat all as compromised.
2. Remove `.env` from tracking: `git rm --cached .env`, add `.env` to `.gitignore`, commit. Provide a `.env.example` with placeholders only (already present — verify it has no real values).
3. Purge from history before any visibility change (e.g. `git filter-repo --path .env --invert-paths`) and force-push, coordinating with the team.
4. Load real secrets from the deployment platform's secret store (Railway/Vercel env vars), never from a tracked file.
5. Add a CI secret scan (`gitleaks detect`) as a pre-merge gate.

---

### 2. `pi_discoveries` table is missing Row-Level Security — fully exposed via the public anon key

**`supabase/schema.sql:310` (table definition); RLS block `:619-700` (table omitted); comment `:621` claims "Every table is RLS-on")**

The RLS section enables row-level security and adds owner-scoped policies for **10 of the 11** public tables (`caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `patient_face_embeddings`, `analyses`, `caregiver_devices`, `feedback`). **`public.pi_discoveries` is never given `enable row level security`** and has no policy — despite the section's own comment asserting every table is covered. This is exactly the "forgotten related table" failure mode the audit framework calls out.

Supabase grants the `anon` and `authenticated` roles access to every table in the `public` schema and exposes them through PostgREST. **RLS is the only gate.** With RLS off, `pi_discoveries` is fully **readable, insertable, updatable, and deletable by anyone holding the publishable/anon key** — which is, by design, public (it ships inside the iOS app bundle).

The table holds (`schema.sql:310-345`):
- `pending_ingest_token` — a **plaintext, one-shot ingest bearer token** the Pi uses to authenticate its frame-push stream (`src/api/cameras_ingest.py` matches `sha256(token)` against `cameras.ingest_token_hash`).
- `current_url` / `current_audio_url` — internal home-network MJPEG/audio endpoints for every camera, across every customer.
- `hardware_id` (device MAC), `claimed_camera_id`, etc.

**Why it's exploitable (concrete impact):**
- **Read** → an attacker dumps `pending_ingest_token` for any freshly-provisioned Pi and uses it to **push arbitrary frames into a victim's detection pipeline** — fabricating or suppressing fall / medication / agitation alerts in a medical-monitoring product. They also enumerate every customer's internal camera URLs (privacy breach of in-home video endpoints).
- **Write/Update** → inject fake discovery rows (poison every caregiver's "Add a camera" picker, since the picker is **not per-caregiver scoped** — see Finding 5), or rewrite `current_url` to an attacker-controlled or internal address.
- **Delete** → wipe discoveries to break the pairing flow (DoS).

**Remediation:** Enable RLS and add the missing policy. Because pre-pairing devices have no `auth.uid()`, the safest fix is to enable RLS with **no anon/authenticated policy at all** (the table is only ever touched by the backend, which connects as the `postgres`/service role and bypasses RLS) — this denies all PostgREST access while leaving the backend fully functional:

```sql
-- Add to supabase/schema.sql alongside the other RLS statements
alter table public.pi_discoveries enable row level security;
-- No policy for anon/authenticated: PostgREST access is denied entirely.
-- The FastAPI backend uses the service role and is unaffected.
```

Then add the loop from the audit framework as a guardrail so a future new table can't silently ship without RLS:

```sql
DO $$ DECLARE r RECORD;
BEGIN
  FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public'
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY;', r.tablename);
  END LOOP;
END $$;
```

Separately, stop storing the ingest token in plaintext even transiently — store only its hash and hand the plaintext to the Pi in the announce response without persisting it.

---

## High

### 3. No rate limiting anywhere — 6-digit pairing code is brute-forceable

**`src/api/pairing.py:63-75` (and app-wide: no limiter in `src/main.py`)**

`POST /v1/cameras/pair` is unauthenticated by design ("the code IS the proof of authorization") and matches an active camera by a **6-digit numeric `pairing_code`** within a 10-minute (`pairing_code_ttl_seconds=600`) window. A search across the codebase finds **no rate limiting** (no `slowapi`, no Redis limiter, no middleware) on this or any other endpoint.

**Why it's exploitable:** The pairing code space is `10^6 = 1,000,000` combinations. With no rate limit, an attacker can brute-force the entire space in minutes against any camera currently in its pairing window. A successful guess returns a long-lived `api_key` plus the LiveKit room/identity for the patient — letting the attacker authenticate as that camera, publish into the patient's live room, and drive ingest. The same gap lets the LiveKit token endpoint, ingest endpoints, and support/feedback mail endpoints be hammered without limit.

**Remediation:**
- Add per-IP **and** per-target rate limiting on `/v1/cameras/pair` (e.g. 5 attempts/min, then exponential backoff/lockout). Use Upstash Redis or an API-gateway limiter, not a public DB table.
- Widen the pairing code to high-entropy (≥ 8 alphanumeric chars, `secrets`-generated) so brute force is infeasible even with limits.
- Apply rate limits to all auth-adjacent and mail-sending endpoints.

---

### 4. Per-request linear bcrypt scan enables CPU-exhaustion DoS

**`src/api/deps.py:111-159` (`camera_auth`)**

Camera authentication has no token→row index. Every request to a camera-authenticated endpoint does:

```python
stmt = select(Camera).where(Camera.api_key_hash.is_not(None))
rows = (await session.execute(stmt)).scalars().all()
for cam in rows:
    matched = await loop.run_in_executor(None, bcrypt.checkpw, token, hash_bytes)  # cost 2^10 each
```

i.e. it loads **every** camera row and runs `bcrypt.checkpw` (deliberately expensive, `bcrypt_rounds=10`) against each one. A non-matching token runs the full `O(N)` set of bcrypt verifications.

**Why it's exploitable:** Combined with the absence of rate limiting (Finding 3), an unauthenticated attacker sends a stream of requests with a bogus bearer to any camera endpoint. Each request burns `N × bcrypt(2^10)` CPU across the thread-pool executor. As the fleet grows, a handful of concurrent attackers can saturate CPU and starve the event loop, denying service to real cameras and caregivers — a cheap, asymmetric DoS. (The code comment already acknowledges the linear scan as a known shortcut.)

**Remediation:**
- Store an indexed, non-secret lookup key alongside the hash (e.g. a random `api_key_id` prefix presented as `Bearer <id>.<secret>`), look the row up by that index in `O(1)`, then run a single `bcrypt.checkpw`. (The code's own comment already proposes a prefix index — implement it.)
- Add rate limiting (Finding 3) as a backstop.

---

## Medium

### 5. Unauthenticated, un-scoped Pi discovery spoofing and list poisoning

**`src/api/pi_discovery.py:43-122`**

`POST /v1/cameras/pi-announce` is unauthenticated (documented: a Pi has no credentials pre-pairing) and upserts a discovery row keyed by attacker-suppliable `hardware_id`, storing attacker-supplied `current_url` / `device_label`. `GET /v1/cameras/discovered` is caregiver-authenticated but has **no per-caregiver scoping** — every caregiver sees every unclaimed Pi (`:97-122`, confirmed in the module docstring).

**Why it's exploitable:** An attacker who can reach the announce endpoint can plant arbitrary discovery rows (fake `device_label`, attacker-chosen `current_url`) that surface in **every** caregiver's "Add a camera" picker. A caregiver could be socially-engineered into provisioning an attacker's "camera." The stored `current_url` is not validated; if the pull-mode relay (`src/services/ingest/pi/mjpeg_reader.read_mjpeg`, used by `RelayPublisher`) is ever activated against a provisioned camera's stored URL, the unvalidated URL becomes a **server-side request forgery (SSRF)** sink (e.g. `http://169.254.169.254/…`, internal services). Today the live path runs in push mode (`pi_stream_url=""`), so SSRF is latent rather than active — but the unvalidated, unauthenticated, cross-tenant write path is real. This finding is amplified by the missing RLS in Finding 2 (the same rows are also writable directly via PostgREST).

**Remediation:**
- Bring forward the planned per-device shared secret (BLE-negotiated) so announcements are attributable, and scope `/cameras/discovered` to the requesting caregiver (e.g. by LAN/subnet association or claim token).
- Validate `current_url` against an allowlist (scheme `http`/`https`, private-LAN ranges only as appropriate) and **block link-local/metadata/loopback ranges** before the backend ever connects to it.
- Cap announce frequency per source IP (Finding 3).

---

## Low

### 6. `/ready` leaks raw database exception strings

**`src/api/health.py:50`**

```python
except SQLAlchemyError as exc:
    raise HTTPException(status_code=503,
        detail={"status": "fail", "checks": {"db": "fail"}, "error": str(exc)})
```

`str(exc)` on a driver error can include connection details (host, port, database, role) in the public `/ready` response. **Remediation:** return a static failure body; log the detail server-side only.

### 7. Unescaped LLM/event content rendered as HTML in alert emails

**`src/services/email.py:115-126` (`_event_html`)**

`title`, `narrative`, and `location` are interpolated into the email HTML **without escaping**, unlike the support/feedback paths which use an `esc()` helper. These values originate from the detector LLM output / event rows. Impact is limited (modern mail clients sandbox HTML and the content is not directly third-party-attacker-controlled), but it is inconsistent with the codebase's own escaping elsewhere. **Remediation:** route these through the same `esc()` helper used by `_support_html`/`_feedback_html`.

### 8. `SECURITY DEFINER` trigger uses `search_path = public` instead of `''`

**`supabase/schema.sql:53-68` (`sync_caregiver_email_from_auth`)**

The framework recommends `SET search_path = ''` on `SECURITY DEFINER` functions to prevent search-path hijacking. This function is a **trigger on `auth.users`** (not PostgREST-callable) and its body is a simple keyed `UPDATE`, so real-world risk is low — but hardening to `search_path = ''` (with fully-qualified names, which it already uses) closes the gap. **Remediation:** change to `set search_path = ''`.

### 9. No security headers; AI-driving paths and DEBUG logging

**`src/main.py`, `.env`**

- No response security headers (HSTS, CSP, `X-Content-Type-Options`, `X-Frame-Options`). For a native-app API this is lower priority, but add them (or a middleware) for defense-in-depth. CORS is not configured at all (no permissive `*` — good).
- `.env` sets `NEMARA_LOG_LEVEL=DEBUG`; `src/main.py:20` honors it via `logging.basicConfig`. Debug logging in any non-local environment risks leaking tokens/PII into logs (the auth code logs at `debug` with principal/camera IDs). Ensure production runs at `INFO`+.
- The detector/LLM pipeline has provider-side model calls but no application-level **per-tenant AI usage cap**. Cost is currently bounded by the number of physically-paired cameras (webhook-driven, not user-triggerable at scale) and the experimental record-clip endpoint is gated off in prod (`enable_internals=false`), so this is low risk today — but add per-camera/per-tenant usage quotas and hard provider spend caps before opening any user-triggerable inference path.

---

## What was checked and found clean

- **Authorization / IDOR:** Every caregiver endpoint resolves rows through `*_repo` helpers scoped by `caregiver_id` (e.g. `event_repo.find_by_id(..., caregiver_id=auth.id)`); no object is fetched by ID alone. LiveKit token minting enforces camera room/identity match and caregiver→patient ownership (`src/api/livekit.py:148-194`).
- **Mass assignment:** `PATCH` handlers use Pydantic request models with `extra="forbid"`; `body.model_dump(exclude_unset=True)` only yields schema-declared, caregiver-writable fields (`src/api/patients.py:128-150`, `detectors.py`, `events.py`). Sensitive columns (`state`, `is_builtin`, `caregiver_id`) are not in the writable schemas.
- **JWT verification:** Uses `jwt.decode` with signature verification, `algorithms=["ES256","HS256"]`, audience pinned to `authenticated`, and dispatches the key by the token's `alg` (no algorithm-confusion gap; `alg:none` rejected). `decode`-without-verify is not used (`src/services/auth.py`).
- **Webhook signature:** LiveKit webhook verifies the signed `Authorization` over the raw body via `WebhookReceiver`/`TokenVerifier` before acting; rejects with 401 on failure (`src/api/livekit_webhooks.py`).
- **SQL injection:** All DB access is via SQLAlchemy ORM with bound parameters; no string-concatenated SQL or raw-unsafe execution found. `/ready` uses a static `text("SELECT 1")`.
- **Ingest token:** SHA-256 of a high-entropy random, matched against an indexed hash column — appropriate (no need for bcrypt on a 256-bit random).
- **Dependencies:** All current/patched — `fastapi 0.136.1`, `starlette 1.0.1`, `pydantic ≥2.10`, `pyjwt ≥2.10`, `cryptography 48.0.0`, `httpx 0.28.1`, `aiohttp 3.13.5`, `jinja2 3.1.6`, `pillow 12.2.0`, `python-multipart 0.0.29`. No known-vulnerable versions identified in `uv.lock`.
- **Payments:** No payment/Stripe code in scope — no client-trusted pricing to exploit.
- **Internals route:** The experimental `record-clip` endpoint is gated behind `enable_internals` (default off) and reuses the owned-camera authorization helper.

---

## Prioritized remediation order

1. **Rotate all real third-party keys and untrack `.env`** (Finding 1) — do this first; the keys are exposed now.
2. **Enable RLS on `pi_discoveries`** and add the all-tables RLS guard loop (Finding 2).
3. **Add rate limiting** to pairing/auth/mail endpoints and **widen the pairing code** (Finding 3).
4. **Index camera-auth lookups** to remove the linear bcrypt scan (Finding 4).
5. **Scope and validate Pi discovery** (per-caregiver list, `current_url` allowlist) (Finding 5).
6. Address the Low-severity hardening items (6–9).
