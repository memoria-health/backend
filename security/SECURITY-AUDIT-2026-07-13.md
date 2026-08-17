# Security Audit — nemara-backend

**Date:** 2026-07-13
**Scope:** Static source review of the `nemara-backend` FastAPI service (`src/`, `supabase/schema.sql`, configs, dependency manifests).
**Methodology:** `vibe-security` skill (Chris Raroque) — secrets & env, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment, data access.
**Constraint:** Static only. No live systems, databases, or secrets were accessed.

---

## Executive Summary

nemara-backend is a caregiver‑monitoring backend for elderly / dementia patients: cameras (iPhone, Raspberry‑Pi, and mini‑PC "gateway") stream live video into per‑patient LiveKit rooms, a VLM detector pipeline analyzes it, and caregivers view feeds and incident clips. The blast radius of an access‑control flaw here is unusually high — it is **live video of vulnerable people in their homes**.

The core caregiver‑facing API (`/v1/patients`, `/v1/events`, `/v1/cameras`, `/v1/detectors`, LiveKit token minting) is generally **well built**: every endpoint requires a verified Supabase JWT, ownership is enforced with `caregiver_id`‑scoped repo queries (returning 404, not 403), the JWT verifier is correct (signature + audience + expiry, no `alg` confusion), LiveKit grants are scoped to owned rooms, storage buckets are private with server‑derived keys, and SQL is parameterized throughout. Dependencies are current with no known‑vulnerable pins.

The serious problems are concentrated in the **device‑provisioning and Pi/gateway ingest surface**, which was bolted on with a weaker trust model than the rest of the app:

- **CRITICAL — Unauthenticated account takeover / camera injection** via `POST /v1/gateway/provision-camera`: the gateway "box token" is self‑asserted (trust‑on‑first‑use, no pre‑registration) and the account is identified by a plaintext **email**. Anyone who knows a caregiver's email can bind their own token to that account, get valid ingest tokens, and push arbitrary video into that patient's room.
- **CRITICAL — `pi_discoveries` table has no Row‑Level Security** while every other table does. It is exposed to the public anon key via PostgREST and stores a **plaintext ingest token** plus device stream URLs for every camera.
- **HIGH — Brute‑forceable 6‑digit pairing code** with no rate limiting, and a **systemic absence of rate limiting / AI cost caps** across the whole service.

Two Critical and three High issues should be remediated before this handles real patient traffic.

### Severity tally

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 3 |
| Medium | 3 |
| Low | 4 |

---

## Critical

### C1 — Unauthenticated account takeover & camera injection via gateway provisioning
**`src/api/cameras_ingest.py:460` (`POST /v1/gateway/provision-camera`)**

The endpoint authenticates with `Authorization: Bearer <box token>`, but the token is **whatever the caller sends** — there is no pre‑registered device allowlist. On first use it *creates* a `gateways` row binding that attacker‑chosen token to a caregiver, and the caregiver is selected **only by `body.email`**:

```python
box_token = authorization[len("bearer "):].strip()          # attacker-chosen
cg = (await session.execute(
    text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
    {"e": body.email.strip()})).first()                      # victim by email
...
if existing is None:
    await session.execute(text(
        "INSERT INTO public.gateways (caregiver_id, token_hash, created_at) ..."))
# then: create a source_type='gateway' camera under EVERY one of the caregiver's
# patients and RETURN each camera's plaintext ingest_token
```

**Exploit:** An attacker who knows or guesses a target caregiver's email (emails are not secret) sends one unauthenticated‑equivalent request with an arbitrary bearer token. The server:
1. Ties the attacker's token to the victim's account (`public.gateways`).
2. Creates a camera under **every patient** the caregiver owns.
3. Returns a valid **plaintext ingest token** for each.

With those ingest tokens the attacker can:
- **Push arbitrary MJPEG video into the patient's LiveKit room** via `POST /v1/cameras/me/frames` (`cameras_ingest.py:99`) — injecting fake footage into a life‑safety monitoring feed, triggering or suppressing fall/seizure/wandering alerts.
- **Enumerate the account's cameras** (identities, rooms, desired state) via `GET /v1/gateway/roster` and `/v1/gateway/wait` (`cameras_ingest.py:324`, `:390`), which trust the same self‑asserted token.

There is no rate limit, so accounts can also be enumerated/mass‑provisioned. Note the committed `schema.sql` has **no `gateways` table** and its `cameras.source_type` CHECK only allows `'iphone'|'pi'` (`schema.sql:255`), so the code is ahead of the audited schema — meaning the `gateways` table's own RLS/grants were not reviewable here and should be assumed unaudited.

**Remediation:**
- Do **not** trust a self‑asserted box token. Pre‑provision gateway devices out‑of‑band (mint the token server‑side, ship it to the device); reject unknown tokens instead of creating a binding.
- Never bind a device to an account by email alone. Require the *caregiver* to authorize the binding from an authenticated session (e.g. caregiver generates a short‑lived claim code in the app, the box presents it).
- Add rate limiting and audit logging to all `/gateway/*` endpoints.
- Confirm `public.gateways` has RLS enabled (it is not in the audited schema).

---

### C2 — `pi_discoveries` table ships without Row‑Level Security (plaintext ingest token exposed to the anon key)
**`supabase/schema.sql:334` (table), RLS block `schema.sql:711–801` (omits it)**

Every table in the schema enables RLS **except `public.pi_discoveries`**:

```
alter table public.caregivers          enable row level security;
alter table public.patients            enable row level security;
alter table public.cameras             enable row level security;
... (9 more) ...
-- public.pi_discoveries: NONE
```

`pi_discoveries` lives in the `public` schema (exposed by Supabase PostgREST) and stores, per device:

```sql
hardware_id            text not null,   -- MAC address
current_url            text not null,   -- live MJPEG stream endpoint
current_audio_url      text,
pending_ingest_token   text             -- ONE-SHOT PLAINTEXT INGEST BEARER TOKEN
```

No `REVOKE`/`GRANT` statements appear anywhere in the schema, so the Supabase default grants to `anon`/`authenticated` stand. With RLS disabled, **anyone holding the anon/publishable key** (which is shipped inside the iOS/web clients and is public by design) can:

```
GET https://<project>.supabase.co/rest/v1/pi_discoveries?select=*
Authorization: Bearer <anon key>
apikey: <anon key>
```

and read **every** Pi device's hardware ID, live stream URL, and any **plaintext `pending_ingest_token`**. That token authenticates frame‑push into a patient's room (see C1's push path), so this is both an information disclosure (enumerate all cameras + stream URLs) and a credential leak. The realtime‑subscription rationale in the schema comment (`schema.sql:714`) confirms RLS is the intended load‑bearing control — it was simply never applied here.

**Remediation:**
- Enable RLS immediately and add a policy (or, since the table is only touched by the backend via the service role, deny the anon/authenticated roles entirely):
  ```sql
  alter table public.pi_discoveries enable row level security;
  -- no policy for anon/authenticated => PostgREST returns nothing
  revoke all on public.pi_discoveries from anon, authenticated;
  ```
- Stop storing the ingest token in plaintext — store only a hash (as `cameras.ingest_token_hash` already does) and deliver the plaintext once over the announce response only.
- Rotate any ingest tokens currently sitting in this table.

---

## High

### H1 — Brute‑forceable pairing code, no throttling
**`src/api/pairing.py:40` (`POST /v1/cameras/pair`), codes minted in `src/api/cameras.py:184`**

Pairing codes are 6 numeric digits (`secrets.randbelow(900_000) + 100_000` → 900,000 values), valid for 10 minutes (`pairing_code_ttl_seconds=600`), and redeemed at an **unauthenticated** endpoint whose query matches **any** camera holding the presented code:

```python
select(Camera).where(Camera.pairing_code == body.pairing_code)
               .where(Camera.pairing_expires_at > now)
```

There is **no rate limiting and no per‑attempt lockout**. An attacker can spray codes against `/v1/cameras/pair`; each hit against any caregiver's outstanding pairing window yields a valid `api_key` for that camera, which grants LiveKit `viewer`/`camera` tokens (live video of the patient) and control of the camera's `desired_state`. With ~900k values and a 10‑minute window, an unthrottled attacker has a realistic hit rate — and the odds compound across all simultaneously‑open windows.

**Remediation:** Rate‑limit `/v1/cameras/pair` per‑IP and globally; lock a camera's pairing after N failed attempts; increase entropy (alphanumeric / longer code) and/or bind the code to the specific camera the caregiver is pairing rather than a global lookup.

### H2 — No rate limiting anywhere in the service
**Global (no `slowapi`/limiter middleware present; confirmed absent across `src/`)**

No endpoint has rate limiting. Beyond H1, this exposes:
- **Auth/pairing brute force** (H1).
- **AI cost abuse** (H3) — `/v1/detectors/infer` and detector compile.
- **Email/SMS spam relay** — `POST /v1/support`, `POST /v1/feedback`, and `POST /v1/patients/{id}/test-email` all trigger Resend sends; an authenticated user can loop them to spam the founders' inboxes or burn Resend quota.
- **Unauthenticated write amplification** — `/v1/cameras/pi-announce` and the `/gateway/*` endpoints accept unauthenticated/weakly‑authenticated writes.

**Remediation:** Introduce a shared limiter (Upstash Redis or equivalent) keyed per‑IP **and** per‑principal on every auth, ingest, provisioning, AI, and email endpoint. Store counters off the public PostgREST surface.

### H3 — Unbounded AI/LLM cost (no per‑account usage or spend cap)
**`src/api/detectors.py:407` (`/v1/detectors/infer`), `:448` (compile on create), detector tick loop in `src/services/detection/`**

`/v1/detectors/infer` makes a synchronous LLM call on arbitrary caregiver‑supplied text on every request (`infer_detector_config`), and `create_detector` fires a background LLM "compile". Neither has a per‑user quota, and the settings expose no application‑level spend cap — the VLM detector loop otherwise runs continuously per active camera. A single authenticated account can spam `infer`/`create` (no rate limit — H2) to drive **unbounded model spend**. The `vibe-security` AI‑integration guidance is explicit: *"A single user can drain your entire monthly budget in minutes."*

**Remediation:** Enforce per‑account daily/monthly token quotas with hard limits; set provider‑side hard spending caps (Gemini/OpenAI/OpenRouter/DashScope); alert on anomalous usage.

---

## Medium

### M1 — Unauthenticated Pi announce enables discovery spoofing & one‑shot token theft
**`src/api/pi_discovery.py:43` (`POST /v1/cameras/pi-announce`)**

The announce endpoint is unauthenticated by design (a pre‑pairing Pi has no credentials) and is keyed only by `hardware_id` (a MAC address — guessable, not secret). Because `upsert_announcement` overwrites `current_url` for a given `hardware_id`, an attacker who learns a target Pi's MAC can:
- **Poison the discovery record** — set `current_url` to an attacker‑controlled MJPEG endpoint, so a caregiver who later provisions that discovery ingests the attacker's video (feeds fabricated footage) — and pivots into M2 (SSRF).
- **Steal the one‑shot ingest token** — if the device has been provisioned, the announce response returns the plaintext `ingest_token` and nulls it (`consume_pending_ingest_token`); an attacker racing the real Pi captures it and gains frame‑push access.

The module docstring acknowledges this as a v1 tradeoff, but returning credentials on an unauthenticated, MAC‑keyed endpoint is a real exposure and compounds C2.

**Remediation:** Add a per‑device secret negotiated at physical setup (as the docstring anticipates) so announcements are attributable; never return an ingest token to an unauthenticated caller; deliver it over a channel bound to the provisioning caregiver.

### M2 — Latent SSRF via caregiver/discovery‑supplied stream URLs (dormant in push mode)
**`src/api/cameras.py:316–322` → `src/services/ingest/pi/relay_publisher.py:239,435` (pull path)**

`cameras.pi_stream_url` / `pi_audio_url` come either from a caregiver‑supplied direct URL or from a discovery record (attacker‑poisonable per M1). Their schemas (`schemas/pi_discovery.py:37`, `schemas/camera.py:32,37`) are **length‑only strings — no scheme allowlist, no private/link‑local/metadata blocking.** The pull‑mode consumers `RelayPublisher.run()` → `read_mjpeg(pi_stream_url)` and `_audio_publish_loop` → `read_pcm_audio(pi_audio_url)` issue a raw outbound `httpx` GET on that URL with no SSRF guard (`mjpeg_reader.py:364`, `audio_reader.py:138`), so a URL like `http://169.254.169.254/…` or `http://localhost:…` would be fetched server‑side.

**Important caveat:** production runs in **push mode**, where `push_frames` constructs `RelayConfig(pi_stream_url="", pi_audio_url=None)` (`cameras_ingest.py:171–176`) and the pull path `RelayPublisher.run()` / `default_relay_coroutine` is **defined but invoked nowhere in `src`**. So today the attacker URL is *stored but never fetched* — the SSRF is **latent**. It becomes directly exploitable the moment pull mode is (re)enabled. Rated Medium on that basis.

**Remediation:** Validate stream URLs at announce **and** provision — allow only `http`/`https`, reject RFC1918 / loopback / link‑local / `169.254.169.254` (resolve the host, don't just parse it) — and re‑assert the check inside `read_mjpeg`/`read_pcm_audio` before any pull‑mode reintroduction.

### M3 — Prior `.env` exposure in git history
**`.gitignore:11–14` (comment)**

`.gitignore` states `.env` "was previously tracked… it has been untracked and its leaked secrets rotated." Rotation cannot be verified statically, and the secrets remain in git history. The `.env` referenced production‑adjacent values (DB URL, Supabase JWT secret/service‑role key, Gemini/OpenAI/LiveKit/Resend/B2 keys).

**Remediation:** Confirm every secret that was ever in `.env` has been rotated; consider scrubbing history (`git filter-repo`) and running `gitleaks detect` over the full history; keep prod secrets only in Railway.

---

## Low

### L1 — `/ready` leaks raw database error strings
**`src/api/health.py:48`** — On a DB failure the handler returns `str(exc)` in the response body, which can disclose connection details / internal hostnames. Return a generic failure body; log the detail server‑side.

### L2 — Caregiver custom‑detector prompt injection (self‑scoped)
**`src/api/detectors.py:502` → detector compile → VLM prompt** — Caregiver‑authored `custom_prompt` is compiled and concatenated into the VLM system prompt. Impact is limited to the attacker's *own* patient analyses (they cannot reach other tenants), so this is defense‑in‑depth rather than cross‑tenant, but malformed prompts can degrade detection reliability. Constrain/segment untrusted prompt fragments and validate compiled output.

### L3 — JWT issuer not validated
**`src/services/auth.py:83`** — `pyjwt.decode` verifies signature, audience (`authenticated`), and expiry, but not `iss`. Signature already binds the token to your Supabase project keys, so impact is minimal; add `issuer=<supabase_url>/auth/v1` for defense in depth.

### L4 — Static, never‑expiring demo pairing code in production
**`src/settings.py:193`, `src/api/pairing.py:59`** — When `NEMARA_DEMO_PAIRING_CODE` is set (prod only, for App Review), the code is reusable and never expires and re‑mints an api_key for the demo camera on each call. Acceptable for a seeded demo camera, but ensure the demo camera never carries real patient data and unset the code once review completes.

---

## Areas reviewed and found sound

- **JWT verification** (`services/auth.py`): correct signature/audience/expiry checks; ES256 (JWKS) vs HS256 (shared secret) dispatched by header `alg` with the correct key each way — no algorithm‑confusion.
- **Caregiver API authorization**: `patients`, `events`, `cameras`, `detectors`, `devices`, `face_embeddings`, `talk`, `caregivers` all require `CaregiverAuth` and scope every query by `caregiver_id` (404, not 403, on cross‑tenant access). No IDOR found on these.
- **LiveKit token minting** (`api/livekit.py`): camera tokens pinned to the camera's own room/identity; caregiver tokens gated by patient ownership; privacy‑mode strips subscribe/publish grants.
- **Mass assignment**: `PATCH /v1/me` and `PATCH /v1/cameras/{id}` update only whitelisted fields; the `caregivers` table has no privilege/subscription/credit columns to escalate.
- **SQL injection**: all queries parameterized (`:param` / ORM); the only f‑string in raw SQL (`cameras_ingest.py:374`) interpolates a constant, not user input.
- **Command injection**: subprocess calls use `create_subprocess_exec`/`subprocess.run` with argument vectors (no `shell=True`, no interpolated user data).
- **Secrets**: no hardcoded credentials in tracked source; `.env` is gitignored; `.env.example` holds only placeholders; the Supabase `service_role` key is used server‑side only (storage upload, admin account deletion) and never returned to clients.
- **Storage**: buckets are private; object keys are server‑derived (`{patient_id}.ext`, DB `clip_storage_key`) behind ownership checks; signed URLs are short‑lived — no path traversal / IDOR.
- **Payments**: no payment/Stripe integration present — section not applicable.
- **Dependencies** (`pyproject.toml`, `uv.lock`): current, no known‑vulnerable pins — pyjwt 2.13.0, cryptography 48.0.0, fastapi 0.136.1, starlette 1.0.1, aiohttp 3.13.5, pillow 12.2.0, urllib3 2.7.0, python-multipart 0.0.29, h11 0.16.0 (request‑smuggling CVE patched), requests 2.34.2, httpx 0.28.1. `python-jose` not used.

---

## Prioritized remediation

1. **C1 — Gateway provisioning auth bypass:** stop trusting self‑asserted box tokens and email‑only account binding; require caregiver‑authenticated device claim. *(highest impact: unauthenticated injection of video into any patient's room)*
2. **C2 — Enable RLS / revoke anon access on `pi_discoveries` and stop storing plaintext ingest tokens; rotate exposed tokens.**
3. **H1/H2 — Add rate limiting** (per‑IP + per‑principal) starting with `/v1/cameras/pair`, `/gateway/*`, `/cameras/pi-announce`, and all email endpoints; lock pairing after repeated failures.
4. **H3 — Enforce per‑account AI usage quotas and provider hard spend caps.**
5. **M1/M2 — Authenticate Pi announce (per‑device secret); validate stream URLs against private/metadata ranges before the relay connects.**
6. **M3/L1–L4 — Confirm secret rotation + scrub history; genericize `/ready` errors; constrain custom‑prompt handling; validate JWT issuer; retire the demo pairing code post‑review.**
