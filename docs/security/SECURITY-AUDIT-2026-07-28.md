# Security Audit — nemara-backend

**Date:** 2026-07-28
**Scope:** Static source audit of `nemara-backend` (FastAPI + Supabase/Postgres backend for the Nemara caregiver + camera iOS apps and the in-process LiveKit detector pipeline).
**Methodology:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) skill — secrets & env, database access control (Supabase RLS), authentication/authorization, rate limiting, payments, AI/LLM integration, deployment, and data-access/input-validation.
**Constraints:** Static only. No live systems, databases, or secrets were accessed. Findings are drawn from committed source and git history.

---

## Executive Summary

The backend is, in most respects, carefully built: application-layer authorization is consistently scoped through a single membership helper (`accessible_patient_ids`), JWT verification is done correctly with `jwt.verify` semantics (signature + audience enforced, algorithm allow-list, opaque 401s), SQL is parameterized throughout, subprocesses are spawned with `exec` (no shell), and there is no permissive CORS. Those fundamentals are good.

However, the audit found **three Critical** and **two High** issues that undermine that foundation. The dominant theme is **broken database access control and secret exposure**, exactly the class the methodology flags as the #1 source of breaches in this kind of app:

1. **Production secrets are committed to git history** — OpenAI, Gemini, LiveKit API secret, Resend, and an APNs signing key are all recoverable from prior commits despite the working-tree `.env` being untracked. (Critical)
2. **`pi_discoveries` has no Row-Level Security** — a public-schema table holding **plaintext one-shot ingest bearer tokens** and in-home camera stream URLs is fully readable and writable by anyone holding the (public, app-embedded) Supabase anon key. (Critical)
3. **`gateways` has no Row-Level Security** — same PostgREST exposure for the gateway auth table (caregiver bindings + token hashes). (Critical)
4. **`POST /v1/gateway/provision-camera` authenticates a device to an account by email alone** — a public identifier — letting an attacker attach a rogue camera to any caregiver's patients and receive live ingest tokens, injecting or suppressing frames in a safety-critical monitoring feed. (High)
5. **`POST /v1/cameras/pi-announce` is unauthenticated and hands back a claimed camera's plaintext ingest token keyed only by MAC address**, which combines with finding #2 into a clean token-theft chain. (High)

Because this system watches vulnerable/elderly patients for falls, wandering, and agitation, integrity of the camera feed is a safety property, not just a confidentiality one — the ability to forge or mute a stream is weighted accordingly below.

**Immediate actions:** (1) Rotate every credential ever committed to `.env`/`.env.bak` and purge them from history; (2) enable RLS with owner-scoped policies on `pi_discoveries` and `gateways` (and add a catch-all "RLS on every public table" guard to the schema); (3) replace email-only gateway provisioning with an authenticated, owner-approved binding.

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | Critical | Production API keys & APNs signing key committed to git history | git history (`.env`, `.env.bak.1783536884`) |
| 2 | Critical | `pi_discoveries` table has no RLS — plaintext ingest tokens + stream URLs exposed via anon key | `supabase/schema.sql:435`, RLS block `:855-922` |
| 3 | Critical | `gateways` table has no RLS — gateway auth records exposed via anon key | `scripts/deltas/apply_gateway_delta.sql:38` |
| 4 | High | Email-only gateway provisioning attaches a rogue camera to any account | `src/api/cameras_ingest.py:472-589` |
| 5 | High | Unauthenticated `pi-announce` returns plaintext ingest token keyed by MAC | `src/api/pi_discovery.py:43-90` |
| 6 | Medium | SSRF: attacker-influenced camera stream URL fetched server-side without validation | `src/services/ingest/pi/mjpeg_reader.py:322-366` |
| 7 | Medium | JWT verifier accepts symmetric HS256 alongside ES256 in production | `src/services/auth.py:26,72-88` |
| 8 | Low | No rate limiting on unauthenticated / ingest / AI-driving endpoints | app-wide (`src/main.py`) |
| 9 | Low | Interactive API docs (`/docs`, `/openapi.json`) exposed unauthenticated | `src/main.py:167-172` |
| 10 | Low | Reusable, non-expiring demo pairing code path | `src/api/pairing.py:59-61`, `src/settings.py:192-198` |

---

## Critical

### 1. Production API keys and an APNs signing key are committed to git history

**Location:** git history — `.env` (removed in commit `cf36938`) and `.env.bak.1783536884` (removed in commit `a53ec6e`, HEAD).

The working-tree `.env` is now correctly gitignored and the `.gitignore` comment claims "its leaked secrets rotated." **The secrets are still fully recoverable from history.** Retrieving the parent of the removal commit yields live-looking credentials:

```
$ git show cf36938~1:.env
NEMARA_GEMINI_API_KEY=AIzaSyDW3quQpxcp288gbQ9PkTwy9fEBQQPRZ4c
NEMARA_OPENAI_API_KEY=sk-proj-YbEoTJIgpWhubz4OxQCG0UD_OIjOmFIeQVwG4boXlxO...
NEMARA_LIVEKIT_API_KEY=APItudo6ecfroAu
NEMARA_LIVEKIT_API_SECRET=eQslXjZ7OHyVJF4LV4aODMeI1nilYU3xpd2bEIqQGNo
NEMARA_RESEND_API_KEY=re_N3Fo9qHR_PVtmFSWzLbnF4Kej9k1o3AMv
NEMARA_APNS_AUTH_KEY="-----BEGIN PRIVATE KEY-----
MIGTAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBHkwdwIBAQQgK5fX9OpNONgFYpKr...
-----END PRIVATE KEY-----"
```

The most recent commit is literally `chore: untrack .env.bak.1783536884 (contains production keys)` — confirming these are production values, and confirming they remain in the immutable history that anyone with repo (or fork/clone) access can read.

**Why it's exploitable.** These are server-side secrets that grant direct billing and impersonation power:
- **OpenAI / Gemini keys** → unlimited model spend on the project's account (the methodology's "drain thousands in minutes"). No per-key cap is visible in-repo.
- **LiveKit API key + secret** → mint arbitrary room tokens for *any* room, i.e. subscribe to any patient's live camera/audio and publish into it. This is the same secret the backend itself uses to authorize camera and caregiver access (`src/api/livekit.py:146`); possessing it defeats the entire LiveKit access model.
- **Resend key** → send mail as the org's verified domain (phishing caregivers).
- **APNs `.p8` signing key** → mint APNs auth JWTs and send push notifications to the caregiver app as Nemara (fake "all clear" / fake alerts).

The Supabase `service_role` key and `jwt_secret` in the same file are the well-known Supabase *local-demo* defaults (harmless), but every other value above is real.

**Remediation.**
1. **Rotate now**, independent of history cleanup: new OpenAI, Gemini, LiveKit (key+secret), Resend keys, and generate a new APNs key in Apple Developer → Keys (revoke the old one). Treat all as compromised.
2. Purge the blobs from history (`git filter-repo --path .env --path .env.bak.1783536884 --invert-paths`) and force-push; rotate again after, since forks/clones may retain the old history.
3. Move production secrets to the deployment secret store (Railway) only; keep the working-tree `.gitignore` rules (already present).
4. Add a pre-commit `gitleaks` / secret-scan hook and enable GitHub secret scanning + push protection so this can't recur.

---

### 2. `pi_discoveries` table has no Row-Level Security — plaintext ingest tokens exposed to the anon key

**Location:** `supabase/schema.sql:435` (table), `supabase/schema.sql:855-922` (RLS block — table absent). Also created RLS-less in `scripts/deltas/apply_pi_schema_delta.sql:85` (prod path).

Every other public table in the schema gets `alter table ... enable row level security` plus an owner/membership policy in the RLS block (`schema.sql:855-922`). The schema even asserts at `:815` *"Every table is RLS-on."* **`pi_discoveries` is the exception — it has neither `enable row level security` nor any policy.** In Supabase, tables in the `public` schema are exposed through PostgREST and granted to the `anon`/`authenticated` roles by default; a table with RLS disabled is fully readable and writable by anyone holding the anon (publishable) key — which is embedded in the shipped iOS apps and therefore public.

The table's columns make this severe (`schema.sql:435-470`):

```sql
create table public.pi_discoveries (
  hardware_id          text not null,       -- device MAC (the "ID")
  current_url          text not null,       -- in-home MJPEG stream endpoint
  current_audio_url    text,                -- in-home audio endpoint
  claimed_camera_id    uuid ...,            -- links a discovery to a live camera
  pending_ingest_token text,                -- ONE-SHOT PLAINTEXT INGEST BEARER TOKEN
  ...
);
```

**Why it's exploitable.** With only the app-extractable anon key, an attacker can call `GET /rest/v1/pi_discoveries?select=*` against the project's Supabase REST endpoint and:
- **Read `pending_ingest_token` in plaintext.** That token is the bearer credential accepted at `POST /v1/cameras/me/frames` (`src/api/cameras_ingest.py:99-208`) — holding it lets the attacker push arbitrary JPEG frames into the patient's LiveKit room, i.e. **forge the monitoring feed** (inject a benign scene to mask a real fall, or a fake incident). It is only "one-shot" against the announce endpoint; read directly from the table it is simply a live secret.
- **Read `current_url` / `current_audio_url`** — enumerate the live camera/audio endpoints inside customers' homes.
- **Write/DELETE rows** (no RLS = no write restriction either): tamper with `current_url` to redirect a not-yet-claimed device, or delete discoveries to break onboarding (DoS).

**Remediation.** Enable RLS and lock the table down. Discovery rows are written by the unauthenticated Pi path and read by the backend (superuser connection, bypasses RLS), so no direct anon access is required at all:

```sql
alter table public.pi_discoveries enable row level security;
-- No policy for anon/authenticated => PostgREST denies all direct access,
-- while the backend's postgres/service connection is unaffected.
```

Additionally, **stop storing the ingest token in plaintext** — store only a hash (as `cameras.ingest_token_hash` already does) and return the plaintext exactly once at provision time. Add the schema's own "enable RLS on every public table" `DO $$ ... $$` guard (from the methodology) at the end of `schema.sql` so a future forgotten table fails safe.

---

### 3. `gateways` table has no Row-Level Security

**Location:** `scripts/deltas/apply_gateway_delta.sql:38-46` (definition; no `enable row level security`, no policy). Not present in `supabase/schema.sql` at all — it exists only via this prod delta.

```sql
CREATE TABLE IF NOT EXISTS public.gateways (
    id           uuid ...,
    caregiver_id uuid,          -- which account this box serves
    patient_id   uuid,
    token_hash   text NOT NULL, -- sha256 of the box's site token
    ...
);
```

This is the auth table for the mini-PC "gateway" roster path (`src/api/cameras_ingest.py:329-462`). Like `pi_discoveries`, it lives in `public` with no RLS, so the anon key grants full CRUD via PostgREST.

**Why it's exploitable.** An attacker with the anon key can:
- **`SELECT`** all gateway rows — map which `caregiver_id` each deployed box serves.
- **`INSERT`** a row binding an attacker-chosen `token_hash` to a victim `caregiver_id`. That token then authenticates against `GET /v1/gateway/roster` / `/gateway/wait` (`cameras_ingest.py:329,407`), returning the victim's cameras' `livekit_identity`/`livekit_room`/`desired_state` and the ops SSH `tunnel_port`, and stamping liveness. It also composes with finding #4 below.
- **`DELETE`** rows to knock deployed gateways offline (DoS).

The stored token is only a SHA-256 hash (good), but table-level exposure of the binding and the write path is the problem.

**Remediation.** `alter table public.gateways enable row level security;` with no anon/authenticated policy (backend uses the superuser connection). Fold the `gateways` definition into the canonical `schema.sql` alongside its RLS so it isn't a delta-only, easily-missed table.

---

## High

### 4. Email-only gateway provisioning attaches a rogue camera to any caregiver's patients

**Location:** `src/api/cameras_ingest.py:472-589` (`POST /v1/gateway/provision-camera`).

The endpoint authenticates the *device* with a bearer token the device generates itself on first boot, and identifies the *account* purely by `body.email`:

```python
box_token = authorization[len("bearer "):].strip()        # attacker-chosen
cg = (await session.execute(
    text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
    {"e": body.email.strip()},
)).first()                                                 # any email, no proof
...
# ties THIS box's token -> that caregiver, then creates a camera under
# EVERY one of the caregiver's patients and returns plaintext ingest tokens
cameras.append({..., "ingest_token": ingest_token_plaintext})
```

There is **no verification that the caller controls that email or that account** — email is a public identifier, not a secret, and caregiver emails are further readable via the members list (`src/api/patient_members.py:63`) and the RLS-less tables above.

**Why it's exploitable.** With just a victim caregiver's email, an unauthenticated attacker (no account needed) can:
1. Bind an attacker-held token to the victim caregiver in `public.gateways`.
2. Cause a `gateway`-source camera to be created **under every one of the victim's patients**, and receive a **plaintext `ingest_token` for each**.
3. Use those tokens at `POST /v1/cameras/me/frames` to publish attacker-controlled video into the patients' LiveKit rooms.

In a fall/wandering/agitation monitoring product this is a **safety-integrity** compromise: the attacker can inject a calm empty-room stream that suppresses real detections, or fabricate incidents. It also creates phantom cameras and enrolls preset detectors on them. The only precondition — knowing an email address — is trivially met.

**Remediation.** Provisioning must be authorized by the account owner, not asserted by the device:
- Require a caregiver-authenticated action (valid Supabase JWT) that mints a **short-lived, single-use gateway enrollment code**; the box presents that code (not an email) to bind. Or
- Have the caregiver app perform the binding server-side (`CaregiverAuth`) and hand the box its token out-of-band.
- Never return per-camera ingest tokens to a request that isn't proven to belong to the account owner. Rate-limit and log provisioning attempts per email/IP.

---

### 5. Unauthenticated `pi-announce` returns a claimed camera's plaintext ingest token keyed only by MAC

**Location:** `src/api/pi_discovery.py:43-90` (`POST /v1/cameras/pi-announce`).

The announce endpoint is intentionally unauthenticated (the module docstring acknowledges this and defers a per-device secret to "v2"). It upserts a discovery keyed by `hardware_id` (the device MAC) and, if that discovery has been claimed, returns the pending ingest token to *whoever announced with that MAC*:

```python
discovery = await pi_discovery_repo.upsert_announcement(session, hardware_id=body.hardware_id, ...)
pending = await pi_discovery_repo.consume_pending_ingest_token(session, hardware_id=body.hardware_id)
if pending is not None:
    _camera_id, ingest_token = pending
    ingest_url = f"{settings.public_base_url}/v1/cameras/me/frames"
return PiAnnounceResponse(..., ingest_token=ingest_token)   # plaintext, to any caller
```

**Why it's exploitable.** `hardware_id` is a MAC address — not a secret. It is discoverable on the LAN, guessable within a vendor OUI range, and (per finding #2) **directly readable from the RLS-less `pi_discoveries` table** along with each row's `claimed_at`. An attacker who announces with a target's MAC at the moment a caregiver provisions it wins the token race and receives the plaintext ingest token, then pushes frames as that camera. Findings #2 and #5 chain: read `hardware_id` from the open table → announce → collect token.

**Remediation.** Introduce the per-device secret the docstring anticipates: provision the Pi with a random secret at setup (captive-portal step) and require it (in addition to `hardware_id`) on announce before any token is returned. Until then, deliver the ingest token only over a channel bound to something the legitimate device proves it holds, and hash-at-rest as noted in #2.

---

## Medium

### 6. SSRF: attacker-influenced stream URL fetched server-side without host validation

**Location:** `src/services/ingest/pi/mjpeg_reader.py:322-366` (`read_mjpeg`), consuming `camera.pi_stream_url` / discovery `current_url`.

`read_mjpeg(url)` opens `httpx.AsyncClient().stream("GET", url)` on a URL that ultimately originates from device-supplied values — `current_url` is set through the **unauthenticated** `pi-announce` path (`pi_discovery.py:67`) and, given finding #2, is even directly writable in `pi_discoveries` via the anon key. There is no allow-list or private-IP/metadata-endpoint filtering before the fetch.

**Why it's exploitable.** An attacker who gets a malicious `current_url` (e.g. `http://169.254.169.254/…`, `http://10.x…`, `http://localhost:…`) into a discovery that the backend then pulls can make the server issue requests to internal/cloud-metadata endpoints. Exfiltration is constrained because the client expects a `multipart/x-mixed-replace` response and errors otherwise, so this is primarily blind SSRF / internal port-probing rather than a clean read — hence Medium. The precondition (a claimed/pulled malicious URL) keeps it below the token findings.

**Remediation.** Before fetching, resolve the host and reject private, loopback, link-local, and metadata ranges (RFC 1918, `169.254.0.0/16`, `::1`, etc.); require `https` to known camera hosts where possible; disable redirects; and validate the URL scheme/host against an allow-list. Apply the same validation wherever `pi_stream_url` is consumed.

### 7. JWT verifier accepts symmetric HS256 alongside ES256 in production

**Location:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), dispatch at `:72-88`.

Verification itself is done correctly (signature + `aud="authenticated"` enforced, algorithm allow-list passed to `pyjwt.decode`, opaque 401s). The residual risk is that production accepts **HS256**, whose key (`supabase_jwt_secret`) is a symmetric shared secret. Anyone who learns that secret can *forge* caregiver tokens for arbitrary `sub`/`email`. A JWT secret was among the values committed to `.env` history (finding #1) — in that instance the well-known Supabase demo default, but the pattern means a real secret leak is catastrophic rather than merely confidential, and the dual-accept widens the surface.

**Remediation.** In production, verify **ES256 only** (Supabase's asymmetric default, keys fetched via JWKS); accept HS256 solely in `env in {local, test}`. Gate `JWT_ALGORITHMS` on `settings.env`.

## Low

### 8. No rate limiting on unauthenticated / ingest / AI-driving endpoints

No rate-limiting middleware or per-user quota is present (`src/main.py` registers routers with none; no Redis/limiter dependency in `pyproject.toml`). The methodology calls for limits on exactly the surfaces this app exposes: the unauthenticated `pi-announce` (`pi_discovery.py:43`) and gateway `provision-camera` (`cameras_ingest.py:472`), the frame-ingest path that drives paid Gemini/OpenAI detector calls, and the support/feedback email senders (`src/api/support.py`, `src/api/feedback.py`) which could be used as a spam relay. Combined with the auth gaps above, the missing limits raise the ceiling on abuse (cost-drain via forged frame streams, provisioning floods, email relay). **Remediation:** add per-IP + per-user limiting (e.g. Upstash/Redis or an API-gateway limiter) on auth, provisioning, ingest, AI-driving, and email endpoints; set hard provider spend caps on OpenAI/Gemini and per-caregiver usage quotas.

### 9. Interactive API docs exposed unauthenticated

`create_app()` (`src/main.py:167-172`) leaves FastAPI's defaults, so `/docs`, `/redoc`, and `/openapi.json` are publicly served in every environment (the code even adds a log filter for them at `:56-64`, confirming they're live). This discloses the full internal API surface to anyone, easing reconnaissance of the endpoints above. **Remediation:** set `docs_url=None, redoc_url=None, openapi_url=None` when `settings.env == "prod"` (or gate them behind auth).

### 10. Reusable, non-expiring demo pairing code

`pair_camera` (`src/api/pairing.py:59-61`) special-cases `settings.demo_pairing_code` (`src/settings.py:192-198`): when the code matches, it re-mints an api-key for the demo camera **without expiry or single-use invalidation**. It is scoped to the single seeded demo camera and only active when the env var is set (prod-only, for App Review), so impact is limited — but a leaked demo code allows unbounded re-pairing of that camera. **Remediation:** restrict the demo path to a non-production/reviewer environment, or bind it to a dedicated throwaway patient; ensure it can never resolve to a real customer camera.

---

## What was checked and found sound

- **Application-layer authorization** — every caregiver-facing repo query is scoped through `accessible_patient_ids` / `has_patient_access` (`src/repos/access.py`, used across `event_repo`, `camera_repo`, `detector_repo`, `face_embedding_repo`, etc.); patient-member endpoints correctly enforce `owner` role and 404-not-403 for non-members (`src/api/patient_members.py`). No backend IDOR found.
- **RLS on the core tables** — `caregivers`, `patients`, `cameras`, `detectors`, `cameras_detectors`, `events`, `analyses`, `patient_face_embeddings`, `caregiver_devices`, `feedback`, `ui_events`, `patient_members` all enable RLS with membership/owner-scoped `using` + `with check` clauses, and use a `security definer` membership function with `set search_path` (`schema.sql:847-922`). Region-change is blocked on the client path by trigger (`schema.sql:126-128`). This is well done — findings #2/#3 are the two tables that missed the pattern.
- **JWT verification** — `pyjwt.decode` with signature verification, audience `authenticated`, and an algorithm allow-list; unverified header is read only to *select* the algorithm and re-validated (`src/services/auth.py`). No `alg:none` acceptance, no `decode`-without-verify. (See #7 for the HS256-in-prod caveat.)
- **LiveKit token minting** — camera tokens are constrained to the camera's own room+identity; caregiver tokens are membership-checked and forced to a `caregiver_`-prefixed identity; privacy mode is baked into grants so a reconnect can't wash it out (`src/api/livekit.py:181-233,127-141`).
- **Injection** — all raw SQL uses bound parameters (`:name`); the one f-string in `text()` (`cameras_ingest.py:386`) interpolates a module constant, not user input. No `os.system`/`shell=True`/`eval`/`pickle.load`; subprocesses use `create_subprocess_exec` with arg lists. Object keys are slugified to prevent path injection (`data_collection/keys.py`).
- **Mass assignment** — caregiver/patient updates go through typed Pydantic request schemas restricting writable fields (`src/api/caregivers.py:53-70`), not raw-body spreads.
- **Payments** — no Stripe/payment integration in this repo; section N/A.
- **CORS** — no `CORSMiddleware` registered; no wildcard-origin exposure (the API is consumed by native apps).
- **AI/LLM** — provider API keys are server-side settings, not client-exposed; detector prompts are assembled server-side. The gaps here are cost-control (no per-user AI quota / spend cap — see #8), not key exposure or prompt-injection-to-tool paths (the LLM has no tool/exec access).

---

## Prioritized remediation summary

1. **Rotate all committed secrets and purge git history (#1).** Independent of everything else, do this first — OpenAI, Gemini, LiveKit key+secret, Resend, and the APNs signing key must be assumed compromised.
2. **Enable RLS on `pi_discoveries` and `gateways` (#2, #3)** and add the "RLS on every public table" guard to `schema.sql`; stop storing `pending_ingest_token` in plaintext.
3. **Replace email-only gateway provisioning with an owner-authenticated, single-use enrollment binding (#4)**, and require a per-device secret on `pi-announce` before returning any token (#5).
4. **Add SSRF host-validation before server-side stream fetches (#6)** and pin production JWT verification to ES256 (#7).
5. **Add rate limiting + provider spend caps (#8)**, disable public docs in prod (#9), and constrain the demo pairing path (#10).

---

*Static audit only; findings are based on committed source and git history as of 2026-07-28. Dynamic testing against a running deployment (PostgREST reachability of the RLS-less tables, live token acceptance) is recommended to confirm exploitability in the production configuration.*
