# Security Audit — nemara-backend

**Date:** 2026-07-04
**Auditor:** Automated security review (vibe-security methodology)
**Scope:** Static source audit of the `nemara-backend` FastAPI + Supabase backend at commit `e79cc76`. No live systems, databases, or secrets were accessed.
**Framework:** `vibe-security` skill (secrets, database/RLS, authentication, rate-limiting, payments, data-access, deployment, AI-integration).

---

## Executive Summary

Nemara is a caregiver safety-monitoring backend: it ingests live video/audio from in-home cameras (iPhone wall phones and Raspberry Pi devices), runs it through VLM/LLM detectors to flag falls, wandering, agitation, etc., and alerts caregivers. The subjects are vulnerable people (elderly, minors) being watched in their homes, so the confidentiality and integrity of the video pipeline is unusually safety-critical.

The codebase is, on the whole, carefully written: JWT verification is correct (signature-checked, audience-pinned, `alg` allow-listed), almost every application endpoint scopes its DB queries to the authenticated `caregiver_id` and returns `404` (not `403`) on cross-tenant access, LiveKit token minting enforces room/identity ownership, Stripe is not used, dependency versions in `uv.lock` are current (no known-vulnerable pins found), and there are no SQL-injection sinks (all raw SQL uses bound parameters).

However, the audit found two **Critical** issues and several **High** issues that materially undermine that posture:

1. **Real production API keys were committed to git history** in a tracked `.env` file (OpenAI, Gemini, LiveKit API key + secret, Resend). The file was later untracked, but the secrets remain in **6 commits** in history and must be treated as compromised.
2. **The `pi_discoveries` table is missing Row-Level Security** — the only public table without it — while it holds plaintext camera ingest bearer tokens and the in-home network URLs of cameras. If reachable via PostgREST with the public anon/publishable key, this is an anonymous read/write of the entire camera-discovery surface.
3. **The unauthenticated Pi-announce endpoint** lets an attacker who knows a device's hardware MAC steal its one-shot ingest token and push arbitrary frames as that camera, and poison the discovery list.
4. **There is no rate limiting anywhere** — enabling pairing-code brute force, unbounded LLM spend, and using the support/feedback endpoints as an email relay.

**Immediate actions:** (a) rotate every credential that ever appeared in `.env` and purge it from history; (b) enable RLS on `pi_discoveries`; (c) authenticate or bind the Pi-announce flow and stop handing out plaintext ingest tokens to unauthenticated callers; (d) add rate limiting to auth, pairing, AI-compile, and email endpoints.

| # | Severity | Title |
|---|----------|-------|
| 1 | Critical | Production API keys committed to git history (`.env`) |
| 2 | Critical | `pi_discoveries` table missing RLS (leaks ingest tokens + camera URLs) |
| 3 | High | Unauthenticated Pi-announce enables ingest-token theft & discovery poisoning |
| 4 | High | No rate limiting: pairing brute-force, LLM cost abuse, email relay |
| 5 | Medium | SSRF via caregiver-claimed malicious discovery `current_url` |
| 6 | Medium | Cross-tenant Pi discovery list leaks all in-home camera URLs |
| 7 | Medium | Reusable, never-expiring demo pairing code mints camera keys unauthenticated |
| 8 | Low | Camera api-key auth: full-table bcrypt linear scan (auth-time DoS amplifier) |
| 9 | Low | `backend_base_url` derived from client-controlled Host header |
| 10 | Low | Prompt injection into custom-detector LLM compile (no input cap) |

---

## Critical

### 1. Production API keys committed to git history (`.env`)

**Files:** `.gitignore:11-14` (admits prior tracking); git history commits `59f62fc`, `217e7ff`, `cd363ff`, `6065268`, `54daf1c`, `b393c5d` (removed in `cf36938`).

The repository tracked a real `.env` for "partner onboarding." It is no longer in the working tree, but `git log --all -- .env` shows it in **6 commits**. Reconstructing the blob (`git show 59f62fc:.env`) reveals live, non-placeholder secrets:

```
NEMARA_OPENAI_API_KEY=sk-proj-YbEoTJIgpWhubz4OxQCG0UD_OIjOmFIeQVwG4boXlxO1m62UJd_jN…  (real OpenAI project key)
NEMARA_GEMINI_API_KEY=AIzaSyDW3quQpxcp288gbQ9PkTwy9fEBQQPRZ4c                          (real Google API key)
NEMARA_LIVEKIT_API_KEY=APItudo6ecfroAu
NEMARA_LIVEKIT_API_SECRET=eQslXjZ7OHyVJF4LV4aODMeI1nilYU3xpd2bEIqQGNo                  (LiveKit Cloud creds)
NEMARA_LIVEKIT_URL=wss://memoria-dev-5v5d1l62.livekit.cloud
NEMARA_RESEND_API_KEY=re_N3Fo9qHR_PVtmFSWzLbnF4Kej9k1o3AMv                             (real Resend key)
NEMARA_APNS_KEY_ID=TTTD2823D2
```

(The Supabase JWT secret / service-role key / DB URL in that same file are the well-known Supabase *local-dev demo* values — `iss=supabase-demo`, `postgres:postgres@127.0.0.1` — and are not themselves sensitive, but their presence shows the file was a real working config, not a template.)

**Why it's exploitable:** Anyone with read access to the repository or any clone/fork/backup can extract these from history even though the file was deleted — deletion does not rewrite history. With them an attacker can:
- **Drain AI billing** — the OpenAI and Gemini keys allow unlimited paid inference.
- **Mint LiveKit access tokens** — the LiveKit API key+secret sign room tokens, granting subscribe/publish into patient rooms, i.e. **live video and audio of vulnerable people inside their homes**, plus the ability to forge webhook signatures.
- **Send email as the org** via the Resend key (phishing caregivers).

The `.gitignore` comment claims the leaked secrets were "rotated," but that cannot be verified from source and the LiveKit URL still points at a live cloud instance.

**Remediation:**
1. **Rotate every credential** that ever appeared in `.env`: OpenAI, Gemini, LiveKit (key+secret), Resend, APNs key, and any Supabase prod keys — regardless of the rotation claim, and confirm the old ones are revoked at each provider.
2. **Purge `.env` from history** with `git filter-repo --path .env --invert-paths` (or BFG), then force-push and have all collaborators re-clone. Rotation is mandatory even after purging, because clones already exist.
3. Add pre-commit secret scanning (`gitleaks`, `trufflehog`) to CI so this cannot recur.
4. Keep `.env` gitignored (already done) and distribute dev secrets via a secret manager.

---

### 2. `pi_discoveries` table is missing Row-Level Security

**File:** `supabase/schema.sql:334-368` (table), `:696-786` (RLS block — `pi_discoveries` absent).

Every table in `public` has RLS enabled and an owner-scoped policy — the schema header even states *"Every table is RLS-on"* (`schema.sql:699`). **`pi_discoveries` is the sole exception**: there is no `alter table public.pi_discoveries enable row level security` and no policy for it. The table stores:

```sql
current_url            text not null,   -- in-home camera MJPEG endpoint (LAN/public URL)
current_audio_url      text,            -- in-home camera audio endpoint
pending_ingest_token   text,            -- ONE-SHOT PLAINTEXT ingest bearer token (schema.sql:357-360)
hardware_id            text not null,   -- device MAC
```

**Why it's exploitable:** In Supabase, tables in the `public` schema are exposed through PostgREST to the `anon` and `authenticated` roles, and the anon/publishable key is public (it ships in the iOS/web clients — see `NEMARA_SUPABASE_PUBLISHABLE_KEY`). With RLS disabled, PostgREST applies no row filter, so **any anonymous holder of the publishable key can `SELECT * FROM pi_discoveries`** — harvesting live plaintext `pending_ingest_token` values (each of which authenticates the camera-frame push endpoint, `cameras_ingest.py:127`) and the private network URLs of every camera in every customer's home. They can also `INSERT`/`UPDATE`/`DELETE` rows to poison the discovery list (see #3, #5). This is the single most damaging class of vibe-coded bug — an exposed table — and it holds exactly the credentials that let an attacker take over the video pipeline.

**Remediation:** Enable RLS and lock the table down. It has no per-caregiver owner column, and it is written by the unauthenticated announce flow via the backend's service-role connection (which bypasses RLS), so the correct policy is **deny-all to anon/authenticated** and let only the backend touch it:

```sql
alter table public.pi_discoveries enable row level security;
-- No policy for anon/authenticated ⇒ PostgREST returns zero rows and rejects writes.
-- The FastAPI backend connects with the service-role/postgres role, which bypasses RLS.
revoke all on public.pi_discoveries from anon, authenticated;
```

Additionally, stop storing the ingest token in plaintext — store only a hash (as `cameras.ingest_token_hash` already does) and deliver the plaintext solely in the provision/announce HTTP response.

---

## High

### 3. Unauthenticated Pi-announce enables ingest-token theft & discovery poisoning

**File:** `src/api/pi_discovery.py:43-90` (`POST /v1/cameras/pi-announce`), token hand-off at `:74-82`; consumed from `pi_discovery_repo.consume_pending_ingest_token`.

The announce endpoint is intentionally unauthenticated ("the Pi has no credentials before pairing"). It upserts a discovery keyed by the client-supplied `hardware_id` and, critically, **returns the one-shot plaintext ingest token to whoever calls with a claimed device's `hardware_id`**:

```python
pending = await pi_discovery_repo.consume_pending_ingest_token(session, hardware_id=body.hardware_id)
...
if pending is not None:
    _camera_id, ingest_token = pending
    ingest_url = f"{settings.public_base_url}/v1/cameras/me/frames"
```

**Why it's exploitable:** `hardware_id` is a device MAC address — low-entropy, enumerable, and often observable (on-network, from device labels like *"Nemara Camera A4B7C2"* which embed MAC bytes, or via the cross-tenant discovery list in #6). After a caregiver provisions a Pi, the plaintext token sits in `pending_ingest_token` until the *first* announce for that `hardware_id` reads it. An attacker who races or simply guesses the MAC receives the token, then authenticates to `POST /v1/cameras/me/frames` (`cameras_ingest.py:96`) and **publishes arbitrary frames as that camera** — injecting a static "all calm" scene to mask a real fall, or bogus footage — into a life-safety monitoring pipeline. They can also freely rewrite `current_url`/`current_audio_url` for any `hardware_id`.

**Remediation:** Introduce a per-device secret established out-of-band at provisioning (the module docstring already anticipates this as "v2"). Until then: (a) deliver the ingest token only over the provisioning channel, never in an unauthenticated announce response; (b) bind announce writes to a device secret so an attacker cannot overwrite another device's URL; (c) rate-limit and alert on announce calls per source IP. Combine with #2 so discovery rows aren't independently readable.

---

### 4. No rate limiting anywhere (auth brute-force, LLM cost, email relay)

**Files:** entire `src/` — no `slowapi`/limiter/middleware present (confirmed by grep); notable sinks: `src/api/pairing.py:45`, `src/api/detectors.py` (compile/infer), `src/api/support.py:19`, `src/api/feedback.py:25`.

There is no per-IP or per-user rate limiting on any endpoint. Concrete abuse paths:

- **Pairing-code brute force** (`pairing.py:63-75`): `POST /v1/cameras/pair` accepts a 6-digit `pairing_code` (10⁶ space) valid for a 10-minute window, and returns an opaque `401`. With no throttle, an attacker can exhaust the space in the window and **pair their own phone to a victim's camera**, obtaining a `livekit_room`/`identity` + api_key for that patient's room.
- **Unbounded LLM spend**: `POST /v1/patients/{id}/detectors` (custom compile) and the detector-infer endpoint each call OpenAI/Gemini (`services/detector_compiler.py`). There is **no per-user usage cap anywhere** in settings. An authenticated caregiver can loop these to run up the provider bill (the skill flags this as "drain your entire monthly budget in minutes").
- **Email relay**: `POST /v1/support` and `POST /v1/feedback` trigger Resend sends per request with attacker-controlled subject/body, usable to spam or phish through the org's verified domain.

**Remediation:** Add rate limiting (e.g. `slowapi` + Redis, or an API-gateway limit) combining per-IP and per-user buckets. Prioritize: pairing (`/cameras/pair`), LLM compile/infer, and the email endpoints. Add hard per-user daily/monthly LLM-usage quotas and provider-side spend caps as defense in depth.

---

## Medium

### 5. SSRF via caregiver-claimed malicious discovery `current_url`

**Files:** `src/api/cameras.py:304-323` (provision resolves URL from discovery), `src/services/ingest/pi/relay_publisher.py:239-241` (`read_mjpeg(pi_stream_url…)`), `src/services/ingest/pi/audio_reader.py:129-140` (opens `httpx` stream to `pi_audio_url`). URL only length-validated (`schema.sql:362-365`); no scheme/host allowlist.

`pi_stream_url`/`pi_audio_url` originate from a `pi_discoveries.current_url` that any unauthenticated caller can set (#3/#2). When a caregiver provisions that discovery, the backend later opens an outbound `httpx` stream to the URL with no validation of scheme or destination host. An attacker who seeds a discovery pointing at `http://169.254.169.254/…` (cloud metadata), `http://localhost:…`, or an internal service can turn the backend into an SSRF proxy once the row is claimed. The global discovery list (#6) makes a caregiver claiming an attacker's row plausible.

**Remediation:** Validate `current_url`/`pi_stream_url`/`pi_audio_url` against an allowlist (scheme `http(s)` only; reject private/link-local/loopback ranges after DNS resolution; pin to expected LAN or egress ranges). Reject on provision, not just at connect time.

### 6. Cross-tenant Pi discovery list leaks all in-home camera URLs

**File:** `src/api/pi_discovery.py:93-122` (`GET /v1/cameras/discovered`) — docstring acknowledges "no per-caregiver filtering."

Any authenticated caregiver receives **every** unclaimed Pi's `device_label`, `current_url`, and `current_audio_url` across all tenants. This discloses other households' in-home camera network endpoints and device identifiers, and lets one caregiver claim a physically-nearby camera belonging to another household (there is no proximity/ownership binding). Compounds #3 and #5 by supplying the `hardware_id`/URLs needed.

**Remediation:** Scope discovery to the requesting caregiver via a device-secret negotiated at physical setup, or a claim-code the caregiver enters from the device. At minimum, stop returning `current_url`/`current_audio_url` in the list.

### 7. Reusable, never-expiring demo pairing code mints camera keys unauthenticated

**Files:** `src/settings.py:153-159` (`demo_pairing_code`), `src/api/pairing.py:53-61,101-145`.

When `NEMARA_DEMO_PAIRING_CODE` is set, `POST /v1/cameras/pair` special-cases it to pair the demo camera **without expiry or single-use consumption**, minting a fresh api_key on every call, unauthenticated. Currently constrained to prod and matched to a single seeded camera by `livekit_identity == "cam_demo"`, so blast radius is one demo device. The risk is operational: if the same code is ever reused for a non-demo row, or the setting leaks, it becomes a permanent unauthenticated key-mint.

**Remediation:** Keep it strictly bound to the demo identity (already done), never log the value, and consider gating the whole path behind `env == "prod"` plus an explicit demo-camera flag so it can never attach to a real patient camera.

---

## Low

### 8. Camera api-key auth does a full-table bcrypt linear scan

**File:** `src/api/deps.py:111-159` (`camera_auth`).

Every camera-authenticated request `SELECT`s all rows with a non-null `api_key_hash` and runs `bcrypt.checkpw` against each until a match (the code comments acknowledge this). Bcrypt is deliberately slow; as the fleet grows, each auth costs O(N) bcrypt verifications, which an attacker can amplify into CPU exhaustion by spamming requests with a bogus bearer (forcing a full-table scan every time). Migrate to the planned `api_key_prefix` index lookup so at most one bcrypt runs per request.

### 9. `backend_base_url` derived from client-controlled Host header

**File:** `src/api/pairing.py:148-153` (`_backend_base_url` uses `request.url`).

The pairing response tells the freshly-paired phone which base URL to use for all follow-up calls (token refresh, ingest), derived from the request's Host. If the reverse proxy forwards an attacker-supplied Host, a phone could be steered to an attacker origin for subsequent traffic. Low because it requires the proxy to pass an unvalidated Host. Pin the base URL to `settings.public_base_url` instead of deriving it from the request.

### 10. Prompt injection into custom-detector LLM compile

**Files:** `src/api/detectors.py` (custom create / infer), `src/services/detector_compiler.py`.

Free-text detector descriptions are compiled through an LLM. No explicit length cap or sanitization was observed on the free-text field, so a caregiver can attempt prompt injection. Impact is low: output is structured detection JSON scoped to the caller's own patient, is not rendered as HTML, and does not drive tool/shell/SQL execution. Add a length bound and treat compiled output as untrusted (schema-validate before persisting), primarily to bound cost (ties to #4).

---

## Prioritized Remediation Summary

1. **Rotate all secrets that ever touched `.env` and purge it from git history (#1)** — assume compromise; this exposes patient video via LiveKit and the paid AI/email accounts.
2. **Enable RLS + revoke anon/authenticated on `pi_discoveries` (#2)** — closes anonymous read of plaintext ingest tokens and in-home camera URLs.
3. **Fix the Pi-announce trust model (#3)** — per-device secret; never return the ingest token to an unauthenticated caller; store the token hashed.
4. **Add rate limiting and per-user LLM quotas (#4)** — pairing, AI compile/infer, and email endpoints first.
5. **Validate outbound Pi URLs against a private-range denylist (#5)** and **scope the discovery list per caregiver (#6)**.
6. Harden the demo pairing path (#7); replace the bcrypt full-scan with a prefix index (#8); pin `backend_base_url` (#9); bound detector-compile input (#10).

---

*Static audit only — findings are substantiated by the source at commit `e79cc76`. Runtime confirmation (e.g. whether `pi_discoveries` is actually reachable through the deployed PostgREST instance, and whether the leaked keys were truly rotated) should be verified operationally.*
