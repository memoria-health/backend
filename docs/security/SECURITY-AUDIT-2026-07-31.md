# Security Audit — nemara-backend

**Date:** 2026-07-31
**Auditor:** Automated static security review (scheduled agent), using the `vibe-security` methodology
**Scope:** `nemara-backend` (FastAPI + Supabase + LiveKit + multi-provider VLM) — static source review only. No live systems, databases, or secrets were accessed.
**Commit audited:** `a53ec6e`

---

## Executive Summary

nemara-backend is the caregiver-app backend for a home safety-monitoring product: it ingests live camera video, runs LLM/VLM detectors over it (falls, wandering, agitation, etc.), stores incident clips + patient face embeddings, and lets caregivers watch/talk to cameras via LiveKit. This is a **life-safety system handling sensitive health and biometric data of vulnerable people**, so the bar for access control and data protection is high.

The good news: the core caregiver API surface is **well built**. Row-Level Security is enabled with membership-scoped policies on every table in `schema.sql`; the application layer independently scopes every patient/camera/event/detector query by caregiver membership (defense-in-depth against the fact that the backend connects as the Postgres superuser and bypasses RLS); JWTs are properly *verified* (not merely decoded); the LiveKit webhook signature is checked; SQL is parameterized throughout; and error responses are deliberately opaque (404-not-403) to avoid enumeration.

However, the audit found **two Critical issues** that undermine that foundation, plus a High and several Medium/Low issues clustered around the newer, less-polished "gateway" camera subsystem and operational hygiene.

### Critical (act today)

1. **Live production secrets committed to git history.** Two secret-bearing files remain in the repository's git history — `.env.bak.1783536884` (blob `5aa7d5a`, committed **2026-07-26**) and an earlier tracked `.env` (blobs `d3fadf`/`98a893`). Between them they expose a **Supabase `service_role` key** (bypasses ALL Row-Level Security → full read/write/delete of every patient's data), **OpenAI / Gemini / OpenRouter API keys**, a **Resend API key**, the **LiveKit API key + secret** (mint tokens into any patient's room), the **Supabase JWT signing secret** (forge caregiver auth tokens), the **DB connection string**, and an **APNs `.p8` push private key**. Deleting the files does not remove them from history. The untracking commit's own message says the keys must be rotated; the repository contains no evidence they were. **Treat all of them as compromised.**

2. **Unauthenticated gateway camera provisioning + live-feed injection.** `POST /v1/gateway/provision-camera` (`src/api/cameras_ingest.py:472`) is guarded only by a "box token" that the caller **generates itself and self-registers on first use** — there is no allowlist of legitimate tokens. An attacker who knows a target caregiver's email (a low-secret identifier — it's the handle used for patient sharing) can call this endpoint with any bearer string and provision cameras under **every one of that caregiver's patients**, receiving back **plaintext ingest tokens**. Those tokens let the attacker publish arbitrary video/audio into the patient's live LiveKit monitoring room (`POST /v1/cameras/me/frames`) — e.g. loop a benign "all calm" feed to mask a real fall, or inject disturbing content — and enumerate which emails are registered accounts.

### High

3. **`public.gateways` table has no Row-Level Security** (`scripts/deltas/apply_gateway_delta.sql:38`), unlike every table in `schema.sql`. If the Supabase REST API (PostgREST) is reachable with the public anon key, the table is fully readable/writable by anyone: read the caregiver↔gateway map, delete rows (DoS the gateway fleet), or insert a token binding pointed at a victim caregiver.

### Medium / Low

No global rate limiting (LLM endpoints and open signup enable cost/DoS abuse); a per-request O(N) bcrypt scan in camera auth enables unauthenticated CPU exhaustion; unauthenticated `pi-announce` allows discovery-list pollution and a provisioning-gated SSRF; legacy HS256 JWTs accepted in production; API docs exposed unauthenticated; verbose upstream error text leaked to clients.

**Dependency scan:** the pinned versions in `uv.lock` (e.g. `pyjwt 2.13.0`, `cryptography 48.0.0`, `starlette 1.0.1`, `fastapi 0.136.1`, `pillow 12.2.0`, `urllib3 2.7.0`, `aiohttp 3.13.5`) did **not** match any known-vulnerable version this review could substantiate. No dependency findings are asserted (static, version-based check only).

---

## Findings

Severity ranking: **Critical → High → Medium → Low**. Each finding lists the exact location, why it is exploitable, and a concrete remediation.

---

### CRITICAL

---

#### C1 — Production secrets committed to git history (Supabase service_role, LLM keys, LiveKit secret, APNs key, JWT secret)

**Location:** git history blobs `5aa7d5a` (`.env.bak.1783536884`) and `d3fadf5`/`98a893e` (`.env`). Untracked in commits `a53ec6e` and `cf36938`; **still present in history**.

**What is exposed** (values redacted in this report; recovered from history for verification):

| Credential | Env var | Impact if abused |
|---|---|---|
| Supabase **service_role** key (JWT) | `NEMARA_SUPABASE_SERVICE_ROLE_KEY` | **Bypasses ALL RLS.** Full read/write/delete of every caregiver, patient, camera, event, face embedding, and Storage object (patient photos, incident clips, enrollment selfies) via the Supabase REST/Storage API. |
| Supabase **JWT signing secret** | `NEMARA_SUPABASE_JWT_SECRET` | Forge HS256 caregiver JWTs (accepted by `verify_supabase_jwt`, see L1) → impersonate any caregiver. |
| **LiveKit** API key + secret | `NEMARA_LIVEKIT_API_KEY` / `_API_SECRET` | Mint tokens for any `patient_<id>` room → watch/inject live audio-video for any patient; forge webhook signatures. |
| **OpenAI / Gemini / OpenRouter** API keys | `NEMARA_OPENAI_API_KEY`, `NEMARA_GEMINI_API_KEY`, `NEMARA_OPENROUTER_API_KEY` | Unbounded third-party billing at the org's expense; drain quota. |
| **Resend** API key | `NEMARA_RESEND_API_KEY` | Send email as the org's verified domain (phishing caregivers). |
| **APNs** auth key (`.p8` EC private key, full PEM) | `NEMARA_APNS_AUTH_KEY` | Forge push notifications to the caregiver app (fake/suppress safety alerts). |
| DB connection string | `NEMARA_DB_URL` | Direct Postgres access if network-reachable. |

**Why it's exploitable:** Anyone with clone/history access to the repo (current or former collaborators, CI logs, forks, a future leak) can extract these from `git log`/`git cat-file`. The `service_role` key and JWT secret alone are game-over: they defeat the otherwise-solid RLS + membership model entirely. The `.env.bak` file was committed on 2026-07-26 and its untracking commit message explicitly warns "the blob remains in git HISTORY — rotate the exposed keys or purge history"; nothing in the repo indicates rotation happened.

**Remediation (do all, in order):**
1. **Rotate every credential now**, treating each as public: Supabase service_role **and** anon/publishable keys and JWT secret (Supabase dashboard → API → "Reset"/rotate; rotating the JWT secret invalidates outstanding tokens — coordinate), LiveKit key/secret, OpenAI/Gemini/OpenRouter/Resend keys, APNs auth key (revoke the Key ID in the Apple Developer portal and issue a new one), and the DB password.
2. **Purge the blobs from history** with `git filter-repo` (or BFG), then force-push and have all collaborators re-clone. Untracking is not enough.
3. Add a pre-commit secret scanner (`gitleaks`, `trufflehog`) and a CI gate so a real secret can never be committed again. Verify `.gitignore` covers `.env`, `.env.*`, and `.env.bak*` (it now does).
4. Review provider audit logs for anomalous usage since 2026-07-26.

---

#### C2 — Unauthenticated gateway provisioning → cross-tenant camera creation & live-feed injection

**Location:** `src/api/cameras_ingest.py:472` (`gateway_provision_camera`); trust boundary at lines `491`–`527`. Downstream abuse via `src/api/cameras_ingest.py:99` (`push_frames`).

**Vulnerability:** The endpoint "authenticates" a `Authorization: Bearer <box_token>`, but the token is not validated against any pre-issued list. If no `gateways` row matches the token hash, the handler simply **inserts one**, binding the attacker-chosen token to whatever caregiver `body.email` names:

```python
existing = (await session.execute(
    text("SELECT id FROM public.gateways WHERE token_hash = :h"), {"h": token_hash},
)).first()
if existing is None:
    await session.execute(
        text("INSERT INTO public.gateways (caregiver_id, token_hash, created_at) "
             "VALUES (:c, :h, :n)"), {"c": caregiver_id, "h": token_hash, "n": now})
```

It then creates a `source_type='gateway'` camera under **every** patient of that caregiver and returns each camera's **plaintext `ingest_token`**.

**What an attacker can do:**
- Send `Authorization: Bearer <any-random-string>` and `{"email": "victim@example.com"}`.
- **Account/email enumeration:** `404 caregiver_not_found` vs `201`/`422` distinguishes registered caregivers.
- **Provision rogue cameras** on all the victim's patients (spurious tiles, detector assignments).
- **Obtain ingest tokens** and then `POST /v1/cameras/me/frames` to publish attacker-controlled JPEG frames as a `cam_*` participant in the patient's `patient_<id>` LiveKit room — the exact feed the caregiver watches and the detector pipeline analyzes. In a fall/wandering monitor, feeding a fake calm scene to hide a real emergency, or injecting distressing footage, is a direct safety/integrity attack.

The sibling caregiver-authenticated endpoint `POST /v1/patients/{id}/cameras/gateway-provision` (`src/api/cameras.py:418`) is correctly gated by `CaregiverAuth` + patient ownership — proof the unauthenticated variant is the anomaly.

**Remediation:**
- Do **not** auto-create gateway identities from unauthenticated input. Pre-register gateway boxes out-of-band (issue a signed, high-entropy token per box stored server-side) and reject any token that isn't already provisioned.
- Bind provisioning to an **authenticated caregiver action** (caregiver claims a box via the app), not to `body.email`. Never derive account ownership from an email supplied by an anonymous caller.
- Rate-limit and log this endpoint; alert on new gateway registrations.
- Since the endpoint has been live, audit `public.gateways` and `cameras (source_type='gateway')` for rows not created by a legitimate caregiver flow, and rotate ingest tokens.

---

### HIGH

---

#### H1 — `public.gateways` table has no Row-Level Security (exposed via anon key)

**Location:** `scripts/deltas/apply_gateway_delta.sql:38` (also `apply_gateway_tunnel_port_delta.sql`). No `ENABLE ROW LEVEL SECURITY` / policy is defined for `gateways` anywhere.

**Why it's exploitable:** Every table in `supabase/schema.sql` explicitly enables RLS and defines a membership/owner policy. The `gateways` table is created **only** by the prod delta scripts and never gets RLS. Supabase's PostgREST exposes the `public` schema to the `anon` and `authenticated` roles (`config.toml` → `schemas = ["public", ...]`), and the anon/publishable key is public by design (it ships in clients). A table with RLS disabled is, per Supabase semantics, fully accessible to those roles. An attacker with the public anon key can therefore hit `…/rest/v1/gateways` to:
- **Read** the full caregiver↔gateway↔`tunnel_port` mapping (recon; `tunnel_port` reveals the ops reverse-SSH scheme).
- **Insert** a row binding an attacker token to a victim `caregiver_id` (feeds C2 without even calling the API).
- **Delete/Update** rows to knock gateways off the roster (fleet DoS) or hijack tunnel-port assignment.

**Remediation:**
- Enable RLS on `public.gateways` immediately and add a restrictive policy (deny all to `anon`/`authenticated`; the backend uses the superuser/service connection and is unaffected):
  ```sql
  ALTER TABLE public.gateways ENABLE ROW LEVEL SECURITY;
  REVOKE ALL ON public.gateways FROM anon, authenticated;
  -- no policy for anon/authenticated => no PostgREST access
  ```
- Audit **all** delta-script-created tables for the same omission; add a CI check that fails if any `public` table lacks RLS (the loop in the `vibe-security` DB reference is a good template).
- Add a `UNIQUE` constraint on `gateways.token_hash` (currently absent — allows duplicate/ambiguous token rows).

---

### MEDIUM

---

#### M1 — No rate limiting anywhere (LLM cost abuse, brute force, DoS); open signup

**Location:** app-wide — no `slowapi`/limiter/middleware exists (`src/main.py` registers no rate-limiting middleware). Hot spots:
- `POST /v1/detectors/infer` (`src/api/detectors.py:407`) — calls Gemini/OpenAI, auth-only, **no patient scope**.
- `POST /v1/detect-doors` (`src/api/detectors.py:88`) — base64 image → VLM, auth-only.
- `POST /v1/patients/{id}/detectors` (`src/api/detectors.py:448`) — fires a background LLM compile per call.
- `POST /v1/patients/{id}/face-embeddings` (`src/api/face_embeddings.py:85`) — up to 40 image uploads + InsightFace passes per call.
- `supabase/config.toml:34` — `enable_signup = true` (account creation is open).

**Why it's exploitable:** Any signed-in caregiver (accounts are free to create) can loop these endpoints to run up unbounded OpenAI/Gemini/OpenRouter/Resend/compute bills or exhaust worker capacity. The `vibe-security` rate-limiting reference calls out AI endpoints specifically: "a single user can drain your entire monthly budget in minutes."

**Remediation:** Add per-user + per-IP rate limits (e.g. Upstash Redis / API-gateway limits) on all AI, upload, email, and auth-adjacent endpoints; set hard provider-side spending caps and per-user/per-tier usage quotas; add billing alerts.

---

#### M2 — Camera auth performs an O(N) bcrypt scan → unauthenticated CPU-exhaustion DoS

**Location:** `src/api/deps.py:111`–`159` (`camera_auth`), reachable pre-authentication on every `CameraAuth` route, e.g. `GET /v1/cameras/me/desired-state` (`src/api/cameras.py:745`, polled ~1/sec per phone) and the camera path of `POST /v1/livekit/token`.

**Why it's exploitable:** To resolve a bearer, the handler loads **all** cameras with a non-null `api_key_hash` and runs `bcrypt.checkpw` (cost 10) against each in the default executor:

```python
rows = (await session.execute(select(Camera).where(Camera.api_key_hash.is_not(None)))).scalars().all()
for cam in rows:
    matched = await loop.run_in_executor(None, bcrypt.checkpw, token, hash_bytes)
```

A single request with a bogus bearer forces N bcrypt computations. An unauthenticated attacker sending a flood of bogus-bearer requests multiplies CPU cost by N (the fleet size), saturating the executor thread pool and stalling the event loop for all requests. The code comments acknowledge the linear scan but not its DoS amplification.

**Remediation:** Store an indexed, unguessable **key id / prefix** (already contemplated in the code comment) and look the row up by that in O(1), running exactly one bcrypt against the single candidate. Cap the executor pool and add per-IP rate limiting on the auth path as a stopgap.

---

#### M3 — Unauthenticated `pi-announce`: discovery-list pollution + provisioning-gated SSRF

**Location:** `src/api/pi_discovery.py:43` (`announce_pi`, intentionally unauthenticated, no per-caregiver scoping); the injected `current_url` flows to `pi_stream_url` at `src/api/cameras.py:316` and is later pulled by the relay in pull mode.

**Why it's exploitable:** Anyone on the internet can POST discovery rows with an arbitrary `hardware_id`, `device_label`, and `current_url`. Because `GET /v1/cameras/discovered` shows **all** unclaimed devices to **every** caregiver (module docstring confirms no scoping), an attacker can flood every caregiver's device picker with spoofed/garbage entries (annoyance/DoS) or a convincingly-labeled lure. If a caregiver provisions the lure, the attacker-chosen `current_url` becomes the camera's `pi_stream_url`, and the backend relay will connect to it — an **SSRF** primitive (e.g. `http://169.254.169.254/…`, internal services) executed from the backend's network position. The module docstring notes a v2 per-device secret is planned but not implemented.

**Remediation:** Add a device-setup secret (negotiated during the physical captive-portal provisioning step) that the announce must carry, and scope `/discovered` to the announcing caregiver. Validate/allowlist `current_url` (scheme, and block private/link-local/loopback IP ranges) before the relay ever connects. Rate-limit `pi-announce` per IP/hardware_id.

---

### LOW

---

#### L1 — Legacy HS256 JWTs accepted in production alongside ES256

**Location:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), used by `verify_supabase_jwt`.

**Why it matters:** The verifier accepts HS256 tokens signed with the shared symmetric `supabase_jwt_secret`. Key selection is correctly tied to the token's `alg` (so this is *not* an alg-confusion bug), but keeping HS256 enabled means anyone who obtains the symmetric secret can forge caregiver tokens — and that very secret is exposed in git history (see **C1**). Prod Supabase issues ES256 tokens.

**Remediation:** After rotating the JWT secret (C1), pin production verification to **ES256 only** (JWKS); keep HS256 solely for local dev, gated by `settings.env`.

---

#### L2 — API docs exposed unauthenticated in production

**Location:** `src/main.py:167` (`FastAPI(...)` created with default `docs_url`/`redoc_url`/`openapi_url`). The access-log filter at `main.py:56` even references `/openapi.json`, `/docs`, `/redoc`, confirming they're live.

**Why it matters:** `/docs`, `/redoc`, and `/openapi.json` publish the complete API surface (including the gateway endpoints in C2/H1) to any unauthenticated visitor, easing reconnaissance. The `vibe-security` deployment reference recommends disabling schema/debug exposure in production.

**Remediation:** Set `docs_url=None, redoc_url=None, openapi_url=None` when `settings.env == "prod"` (or place them behind auth).

---

#### L3 — Demo pairing code is a permanent, reusable pairing credential

**Location:** `src/settings.py:198` (`demo_pairing_code`), `src/api/pairing.py:59`–`61`, `_pair_demo_camera` at `pairing.py:101`.

**Why it matters:** When `NEMARA_DEMO_PAIRING_CODE` is set (production only, per the comment), that exact code pairs the demo camera **without expiry or single-use**, re-minting a fresh api_key each call. Anyone who learns the code (the value `424242` appears in the leaked `.env.bak`) can repeatedly pair/hijack the demo camera and drive its LiveKit room.

**Remediation:** Scope the demo camera to a throwaway demo patient with no real data; rotate the demo code (it's in the leaked history); consider disabling it outside active App Review windows.

---

#### L4 — Upstream provider error text leaked to clients

**Location:** `src/services/storage.py:133`–`136`/`254`/`290`/`334` (StorageError includes `response.text`), surfaced to clients as 502 detail in `src/api/patients.py:325` and `src/api/face_embeddings.py:166`.

**Why it matters:** Raw Supabase Storage error bodies (which can include internal identifiers/paths) are propagated to API clients, a minor information disclosure.

**Remediation:** Log the upstream body server-side; return a generic client-facing message.

---

#### L5 — APNs device-token upsert allows cross-account rebinding

**Location:** `src/api/devices.py:32` (`upsert_by_token`, keyed on `apns_token`).

**Why it matters:** Registration upserts by token, reassigning `caregiver_id`. An attacker who learns a victim's APNs token could register it to their own account, causing the victim's device to stop receiving push alerts (safety-alert suppression). APNs tokens are not trivially obtainable, so impact is limited.

**Remediation:** Require re-verification before moving a token between accounts, or key device rows on `(caregiver_id, apns_token)` and let each account manage only its own bindings.

---

## What was reviewed and found solid (no action needed)

These are called out so the areas are known-covered:

- **RLS in `schema.sql`** — every table (`caregivers`, `patients`, `patient_members`, `cameras`, `detectors`, `cameras_detectors`, `events`, `analyses`, `patient_face_embeddings`, `caregiver_devices`, `feedback`, `ui_events`) enables RLS with membership/owner-scoped policies incl. `WITH CHECK`; the `has_patient_access` helper is `SECURITY DEFINER` with `SET search_path` (schema.sql:844–922). **Exception: `gateways` — see H1.**
- **Application-layer authorization** — `repos/access.accessible_patient_ids` + per-repo `caregiver_id` scoping give consistent 404-not-403 IDOR protection across patients/cameras/events/detectors/face-embeddings (e.g. `patient_repo.py`, `cameras.py`, `events.py`).
- **JWT handling** — `jwt.decode` with signature verification, explicit audience (`authenticated`), expiry enforced; never `jwt.decode(..., verify=False)` (`services/auth.py`).
- **LiveKit token minting** — camera tokens locked to the camera's own room/identity; caregiver tokens scoped to member-accessible `patient_<id>` rooms; privacy-mode strips subscribe/publish and forces `caregiver_` identities (`api/livekit.py`).
- **LiveKit webhook** — HMAC signature verified via `WebhookReceiver`/`TokenVerifier` before any dispatch (`api/livekit_webhooks.py`).
- **SQL** — all raw `text()` statements in `cameras_ingest.py` are bind-parameterized; no string-concatenated SQL found; ORM used elsewhere.
- **Uploads** — chunked size caps (413) before buffering; image content-type allowlist (`patients.py`, `face_embeddings.py`, `storage.py`).
- **Secrets in current tree** — no live secrets in tracked files; `.env`/`.env.bak*` are gitignored; `.env.example` holds only placeholders. (The exposure is in **history** — C1.)
- **Storage** — private buckets, short-TTL signed URLs, service_role key used server-side only (`services/storage.py`).

## Prioritized remediation summary

1. **C1 — Rotate every leaked credential now and purge git history.** The service_role key + JWT secret defeat all other controls. (Add `gitleaks` to CI.)
2. **C2 — Lock down `POST /v1/gateway/provision-camera`.** Pre-register gateway tokens; never mint identity/ownership from an unauthenticated email. Audit existing `gateways`/gateway cameras.
3. **H1 — Enable RLS on `public.gateways`** (and audit all delta-created tables); add a CI check that every `public` table has RLS.
4. **M1 — Add rate limiting + hard spending caps** on all LLM/upload/email/auth endpoints; reconsider open signup.
5. **M2 — Replace the O(N) bcrypt camera-auth scan** with an indexed key-id lookup.
6. **M3 — Authenticate/scope `pi-announce`** and validate `current_url` against SSRF (block private/link-local ranges).
7. **L1–L5 — Pin ES256 in prod, disable docs in prod, rotate/limit the demo code, stop leaking upstream error text, harden device-token rebinding.**

---

*Static review only — findings are based on source inspection and should be validated against the running environment (especially whether the leaked C1 credentials have already been rotated and whether PostgREST/anon access to `public.gateways` is reachable in the current deployment).*
