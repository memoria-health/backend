# Security Audit — nemara-backend

**Date:** 2026-08-16
**Scope:** Static source audit of the `nemara-backend` FastAPI service (caregiver/patient monitoring backend: Supabase Postgres + RLS, LiveKit video, Supabase Storage, APNs push, multi-provider VLM/LLM detection pipeline, Pi/gateway camera ingest).
**Method:** Manual review of auth, DB/RLS, and config surfaces, plus focused sub-audits of access-control/IDOR, storage/SSRF/AI-cost, and dependencies/deployment. Framework: the `vibe-security` skill methodology.
**Constraint:** Static only — no live systems, databases, or secrets were accessed. No application code was modified; this report is the only file added.

---

## Executive Summary

The codebase is, on the whole, **well-engineered for security**: Row-Level Security is enabled on every tenant table with membership-scoped policies, backend repositories independently scope queries by caregiver access (defense in depth), all PATCH/POST bodies use Pydantic `extra="forbid"` (no mass-assignment was found), JWT verification is signature-checked with an audience claim, the LiveKit webhook verifies signatures, Supabase Storage keys are server-derived with the service-role key kept server-side, and all pinned dependencies are on current, patched versions. Those areas are documented as verified-clean below so coverage is visible.

However, the audit found **one Critical** and **three High** issues that need prompt attention, plus several Medium hardening gaps:

- **Critical:** Real, billing-capable third-party API keys and an Apple APNs signing key were committed to git history in a `.env.bak` file that is still reachable from `main`.
- **High:** An unauthenticated gateway-provisioning endpoint lets anyone create cameras inside an arbitrary account (selected by email) and receive working ingest credentials for it.
- **High:** Deleting a patient is not owner-guarded — any shared member, including a read-only `viewer`, can permanently delete a patient and cascade-delete all of their cameras, events, clips, and biometric data.
- **High:** There is no rate limiting anywhere in the service; the unauthenticated 6-digit camera-pairing code is brute-forceable within its 10-minute window.

The most urgent action is **credential rotation and history purge** (Critical), followed by adding a real authentication anchor to gateway provisioning and an owner check on patient deletion.

### Findings at a glance

| # | Severity | Title | Location |
|---|----------|-------|----------|
| C1 | **Critical** | Live third-party API keys + APNs private key committed to git history | `.env.bak.1783536884` (commit `22fbbad`, reachable from `main`) |
| H1 | **High** | Unauthenticated gateway provisioning → cross-account camera creation + ingest credential issuance | `src/api/cameras_ingest.py:527` |
| H2 | **High** | `DELETE /v1/patients/{id}` not owner-guarded → any member can cascade-delete a patient | `src/api/patients.py:251`, `src/repos/patient_repo.py:86` |
| H3 | **High** | No rate limiting; brute-forceable 6-digit unauthenticated pairing code | `src/api/pairing.py:40` (+ service-wide) |
| M1 | Medium | HS256 caregiver JWTs accepted in all envs on a shared symmetric secret | `src/services/auth.py:26` |
| M2 | Medium | `pi-announce` releases one-shot ingest token to any caller who knows the MAC-based `hardware_id` | `src/api/pi_discovery.py:48` |
| M3 | Medium | No per-account caps on LLM/VLM usage → unbounded spend | `src/api/detectors.py`, `src/services/detector_compiler.py` |
| M4 | Medium | Production container runs as root | `docker/Dockerfile` |
| M5 | Medium | `/internal/*` and deep `/health` fail **open** if `NEMARA_ENV` is not set to prod | `src/settings.py:29`; `src/api/internal/cameras.py:54`; `railway.toml` |
| M6 | Medium | Hardcoded reviewer/admin allowlist trusting the JWT `email` claim | `src/api/review.py:18` |
| L1 | Low | Latent SSRF in disabled pull-mode ingest (unvalidated, unauth-seeded `pi_stream_url`) | `src/services/ingest/pi/relay_publisher.py` |
| L2 | Low | OpenAPI `/docs`, `/redoc`, `/openapi.json` exposed in production | `src/main.py:205` |
| L3 | Low | Prompt injection into detector-compile meta-prompts (same-tenant blast radius) | `src/services/detector_compiler.py:224` |
| L4 | Low | `/v1/cameras/discovered` leaks all unclaimed Pis (LAN URLs, labels) cross-tenant | `src/api/pi_discovery.py:97` |
| L5 | Low | No security response headers / no TrustedHost | `src/main.py:204` |
| L6 | Info | Over-strict-but-inconsistent scoping (ownership vs membership) — correctness, not a hole | `talk.py`, `detectors.py`, `face_embeddings.py` |

---

## Critical

### C1 — Live third-party API keys and an APNs private key are committed in git history

**Location:** `.env.bak.1783536884`, added in commit `22fbbad` ("fix(clips): remove green-box annotation … (#26)", 2026-07-23). The file is **not** in the current working tree, but the blob is reachable from `main`/`HEAD` — anyone with clone access can recover it with `git cat-file` or `git log -p`.

**What leaked:** The backup contains real, spend-/impersonation-capable credentials (values intentionally not reproduced here):

- `NEMARA_GEMINI_API_KEY` (`AIza…`) — Google Generative AI
- `NEMARA_OPENAI_API_KEY` (`sk-pro…`)
- `NEMARA_OPENROUTER_API_KEY` (`sk-or-…`)
- `NEMARA_RESEND_API_KEY` (`re_…`) — can send email as the company's verified domain
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` — mint arbitrary room tokens
- `NEMARA_APNS_AUTH_KEY` — the Apple `.p8` ES256 signing key (PEM), plus `APNS_TEAM_ID` / `APNS_KEY_ID` — sign push as the app
- `NEMARA_SUPABASE_PUBLISHABLE_KEY`, `NEMARA_DEMO_PAIRING_CODE`

**Why it's exploitable:** These are bearer credentials. Anyone with repo history (current/former contractors, partners onboarded onto the repo, or anyone who obtains a clone) can extract them and:
- Drain the OpenAI/Gemini/OpenRouter billing accounts (no spend cap is enforced app-side — see M3).
- Send email from the verified Resend domain (phishing caregivers).
- Mint LiveKit tokens to join patient video rooms.
- Sign APNs pushes impersonating the app.

Deleting the file from the tree does **not** remove it from history; the credentials must be treated as compromised.

> **Note on the Supabase auth secrets in the same file:** `NEMARA_SUPABASE_JWT_SECRET` in this backup is the *well-known public Supabase local default* (`super-secret-jwt-token-…`) and `NEMARA_SUPABASE_URL` is a local/tunnel host, so this particular backup looks like a developer's local env. That does **not** reduce the severity: the third-party keys above are shared across environments for a small team and are almost certainly live. It does, however, connect to finding **M1** — because the backend accepts HS256 tokens signed with that secret in every environment.

**Remediation (do in order):**
1. **Rotate every credential above now** — regenerate the OpenAI, Gemini, OpenRouter, Resend, LiveKit, and Supabase publishable keys, and revoke + reissue the APNs `.p8` key in the Apple Developer portal. Rotate the demo pairing code.
2. **Purge the blob from history** (`git filter-repo --path .env.bak.1783536884 --invert-paths`, or BFG), then force-push and have all collaborators re-clone. Coordinate because history rewrite is disruptive.
3. Confirm the earlier `.env` untracking (noted in `.gitignore`) also had *its* keys rotated — the `.gitignore` claims rotation for `.env` but this is a separate `.env.bak` file.
4. Add a pre-commit secret scanner (`gitleaks`) and a CI gate so a future `.env*` can never be committed. `.gitignore` already lists `.env.bak*`, so this file predates that rule — the CI gate closes the gap for good.

---

## High

### H1 — `POST /v1/gateway/provision-camera` has no real authentication; any caller can create cameras in an arbitrary account and mint ingest credentials

**Location:** `src/api/cameras_ingest.py:527-648` (`gateway_provision_camera`)

**Why it's exploitable:** The only "auth" is `Authorization: Bearer <box_token>`, where `box_token` is any non-empty string the *caller* chooses. It is never checked against a pre-registered allowlist: if the token hash isn't already in `public.gateways`, the handler simply **INSERTs a new gateways row for it** (`:568-583`). The target account is chosen entirely by the client-supplied `body.email` (`:555-563`). An attacker can therefore:

1. `POST /v1/gateway/provision-camera` with `Authorization: Bearer <anything>` and `{"email": "<victim>@example.com"}`.
2. The handler binds the attacker-chosen token to the victim caregiver, creates a `source_type='gateway'` camera under **every one of the victim's patients** (`:585-646`), and returns the **plaintext ingest token** for each created camera in the response (`:644`).
3. The attacker's token is now bound to the victim, so it can call `GET /v1/gateway/roster` and `GET /v1/gateway/wait` to continuously read the victim's cameras' `livekit_room` (`patient_<uuid>` — leaks patient IDs), `livekit_identity`, `desired_state`, `privacy_enabled`, and `online`.

This is unauthenticated cross-account **write** + credential issuance + information disclosure. The `404 caregiver_not_found` vs `201` distinction (`:559`) is also an account-enumeration oracle. *Partial mitigation:* created cameras start `desired_state='disconnected'`, and `push_frames` returns `423` while disconnected — so *immediate* live-media injection into a patient room is gated behind the caregiver later connecting the phantom camera. The durable ingest credential, cross-account row creation, and roster read are not gated.

**Remediation:** Pre-provision gateway boxes out-of-band (create the `public.gateways` row with a server-issued secret at manufacturing/enrollment) and **reject any `token_hash` not already present** — never upsert one on the fly. Derive `caregiver_id` from the already-registered gateway row, not from a client-supplied `email`. Do not return minted ingest tokens to a caller whose authority over the account hasn't been proven. Add rate limiting + generic errors to remove the enumeration oracle.

### H2 — `DELETE /v1/patients/{id}` is not owner-guarded; any shared member can cascade-delete a patient

**Location:** `src/api/patients.py:251-265` (`delete_patient`) → `src/repos/patient_repo.py:86-103` (`delete_by_id`)

**Why it's exploitable:** `delete_by_id` scopes with `Patient.id.in_(accessible_patient_ids(caregiver_id))` — i.e. *any membership row*, not ownership. `src/api/patient_members.py` lets an owner share a patient with role `caregiver` **or `viewer`** (`_SHAREABLE_ROLES`, `:26`). A caregiver added as a read-oriented `viewer` can then call `DELETE /v1/patients/{id}` and permanently delete the patient, cascading (DB `ON DELETE CASCADE`) to all cameras, detectors, events, analyses, incident clips, and face embeddings. This is a **missing check, not intended behavior**: `src/repos/access.py:6-7` explicitly states role is meant to guard *"owner-guarded actions (share/unshare, delete)"*, and the share/remove-member handlers in `patient_members.py` *do* correctly enforce `role == "owner"`.

**Remediation:** In `delete_patient`, require `await access.is_owner(session, patient_id=patient_id, caregiver_id=auth.id)` before deleting (return 403/404 otherwise), matching the pattern already used in `patient_members.remove_patient_member`.

### H3 — No rate limiting anywhere; the 6-digit unauthenticated pairing code is brute-forceable

**Location:** `src/api/pairing.py:40-98` (`POST /v1/cameras/pair`); service-wide (confirmed: no rate-limit library or middleware exists — `grep` for `ratelimit`/`slowapi`/`limits`/`429` across `src/` returns nothing, and `pyproject.toml` declares no such dependency).

**Why it's exploitable:** `POST /v1/cameras/pair` is unauthenticated (the code is the proof of authorization) and matches a **6-digit** `pairing_code` (10⁶ = 1,000,000 possibilities) valid for `pairing_code_ttl_seconds` (default 600s). With no rate limiting, an attacker can brute-force the active code space during any open 10-minute pairing window; a successful guess returns a long-lived camera `api_key`, which authorizes clip ingest and LiveKit token minting for that camera's patient room. The absence of rate limiting also leaves every other sensitive surface open to automated abuse: JWT-auth guessing, the ingest/gateway token endpoints, the account-enumeration oracle in H1, and the AI-cost endpoints in M3.

**Remediation:** Add per-IP **and** per-target rate limiting (e.g. `slowapi` + Redis, or an edge/API-gateway limit) on `/v1/cameras/pair`, the auth surfaces, and the AI endpoints. Independently, raise the pairing-code entropy (longer alphanumeric code) and/or add a per-camera attempt lockout so the 10-minute window can't be exhaustively searched. Combine per-IP and per-account limits (per the rate-limiting guidance) since either alone is trivially bypassed.

---

## Medium

### M1 — HS256 caregiver JWTs are accepted in every environment using a shared symmetric secret

**Location:** `src/services/auth.py:26` (`JWT_ALGORITHMS = ["ES256", "HS256"]`), dispatched in `verify_supabase_jwt` (`:64-94`); `src/api/deps.py:80-84` always passes `settings.supabase_jwt_secret`.

**Why it matters:** The verifier accepts both ES256 (production, verified via Supabase JWKS public keys) *and* HS256 (verified with the symmetric `supabase_jwt_secret`) in all environments, dispatching purely on the token's `alg` header. HS256 uses a **shared secret that both the server and the token signer hold**. If that secret is ever weak, leaked, or the well-known default, an attacker can forge an HS256 token with any `sub`/`email` and `aud="authenticated"` and authenticate as **any caregiver** — full account takeover, bypassing Supabase entirely. Finding C1 shows this exact secret set to the *public Supabase default* in a committed backup; even if production uses a strong secret, keeping the HS256 path enabled in prod is an unnecessary account-takeover vector.

**Remediation:** In staging/prod, restrict `JWT_ALGORITHMS` to `["ES256"]` only (gate on `settings.env`), so production tokens must be JWKS-verified and the symmetric secret can never authenticate a caregiver. Keep HS256 for `local`/`test` only. Ensure the production `supabase_jwt_secret` is a strong, unique value regardless.

### M2 — `pi-announce` hands a claimed camera's one-shot ingest token to any caller who knows the `hardware_id`

**Location:** `src/api/pi_discovery.py:43-90` (`announce_pi`) → `src/repos/pi_discovery_repo.py` (`consume_pending_ingest_token`)

**Why it matters:** The announce endpoint is unauthenticated by design (a Pi has no credentials before pairing). After a caregiver provisions a Pi, the plaintext ingest token is stashed on the discovery row and delivered to whoever next announces that `hardware_id` — keyed only by MAC address, with no device secret. An attacker who learns/guesses a target Pi's `hardware_id` (MAC addresses are not secret; partially predictable by OUI, visible on-LAN) and races the real device to announce first receives the one-shot ingest token, which authorizes pushing frames into that camera's `patient_<id>` LiveKit room. Bounded (one-shot, only between provision and the real Pi's next announce) and requires knowing the MAC — hence Medium. The module docstring already acknowledges this and plans a v2 per-device secret.

**Remediation:** Bind announcements to a per-device secret negotiated during the physical Wi-Fi provisioning step and require it on `pi-announce` before releasing a pending token; or deliver the token over a channel tied to the provisioning session rather than to the client-asserted `hardware_id`.

### M3 — No per-account caps or rate limits on LLM/VLM usage → unbounded spend

**Location:** `src/api/detectors.py` (`/infer`, create, `/refine`, `/detect-doors`), `src/services/detector_compiler.py`

**Why it matters:** No endpoint enforces per-caregiver usage caps or call-frequency limits. Several caregiver-triggered endpoints each fire model calls: `POST /v1/detectors/infer` (one synchronous Gemini+OpenAI call per press), detector **create** (background OpenAI *then* Gemini compile + a summarize call), `PATCH`/`refine` (re-compile + summarize, refine also runs a synchronous infer), and `POST /v1/detect-doors` (one VLM call per request on a caregiver-supplied image). There is **no limit on detectors-per-patient or cameras-per-patient**, and each compiled detector then runs a VLM call **every tick** (default 10s) on every assigned camera — so detector/camera fan-out is an uncapped steady-state cost multiplier. Combined with C1's leaked provider keys and H3's absent rate limiting, cost-drain risk is real. (Positives: `gemini_video_fps`/tick cadence are env-fixed, so per-user fps abuse isn't possible, and `compiler_v2.reject_reason` blocks some no-op asks before spend.)

**Remediation:** Add per-caregiver rate limits on the infer/create/refine/detect-doors endpoints; enforce hard caps on detectors-per-patient and cameras-per-patient; add a per-account daily VLM-spend budget at the routing choke point; and set **hard provider-side spending caps** on OpenAI/Gemini/OpenRouter as a backstop.

### M4 — Production container runs as root

**Location:** `docker/Dockerfile` (and `docker/Dockerfile.dev`)

**Why it matters:** The prod image has no `USER` directive, so uvicorn runs as UID 0. Any RCE or container-escape primitive then executes with root inside the container, and baked caches live under `/root`. Least-privilege hardening.

**Remediation:** Add a non-root user before `CMD` (`RUN useradd -m app && chown -R app /app`, then `USER app`), and ensure pre-baked model-weight caches are readable by that user (they currently sit under `/root`, so move the warmup or set `HOME`).

### M5 — `/internal/*` and the deep `/health` check fail **open** when `NEMARA_ENV` is not set to prod

**Location:** `src/settings.py:29` (`env` defaults to `"local"`); `src/api/internal/cameras.py:54-59`; `src/api/health.py` (`/ready` guard); `railway.toml` (does not set `NEMARA_ENV`); `.env.example:1` sets `NEMARA_ENV=local`.

**Why it matters:** The fail-closed guarantee for the support-only `/internal/cameras/{id}/reclassify` privacy-override route (and the authenticated deep-health probe) depends on `settings.env` being `staging`/`prod`: when `internal_service_key` is empty, the guard only raises 403 `if settings.env not in ("local","test")`. Since `env` **defaults to `local`** and `railway.toml` does not export `NEMARA_ENV`, a deploy that forgets to set `NEMARA_ENV=prod` (and leaves `NEMARA_INTERNAL_SERVICE_KEY` unset) exposes the privacy-override admin route **unauthenticated**. The guard logic itself is correct (constant-time compare, documented fail-closed) — the risk is the environment default.

**Remediation:** Guarantee `NEMARA_ENV=prod` and a non-empty `NEMARA_INTERNAL_SERVICE_KEY` in the Railway environment. Defensively, fail startup if `env` is unset in a container, or change the default to a non-privileged sentinel, so a missing env var can never open the internal surface.

### M6 — Hardcoded reviewer/admin allowlist trusts the JWT `email` claim

**Location:** `src/api/review.py:18` (`REVIEWER_EMAILS = {"admin@gmail.com"}`), used by `_is_admin`

**Why it matters:** `_is_admin` trusts the JWT `email` claim against a hardcoded set. An admin sees and acts on **every** patient's held events (`/v1/review/queue`, `/v1/review/{event_id}`) and can toggle `require_review` on any patient. If an attacker can register/control `admin@gmail.com` in the Supabase project (it is a real, ownable Gmail address), they gain cross-tenant admin access to held events across all accounts. Impact is entirely contingent on whether that email is registerable in the project's auth settings; the code marks it dev-only ("Move to settings/DB when this leaves local dev"). Flagged so it is not shipped as-is.

**Remediation:** Remove the hardcoded allowlist before production. Drive admin authorization from a server-controlled source (a `role`/`is_admin` column or a Supabase custom claim that the user cannot self-set), never from the self-asserted `email` claim.

---

## Low / Informational

### L1 — Latent SSRF in disabled pull-mode ingest
`src/services/ingest/pi/relay_publisher.py` (`run()` → `read_mjpeg`/`read_pcm_audio` in `mjpeg_reader.py`/`audio_reader.py`). The backend can fetch an arbitrary URL server-side from `cameras.pi_stream_url`/`pi_audio_url`, which trace back to the **unauthenticated** `pi-announce` `current_url` (validated only for length, `max 2048`, no scheme/host allow-list). This would be a High SSRF (metadata endpoints, internal services, DNS rebinding) — but the live path is push-mode (`cameras_ingest.py:177` hard-codes `pi_stream_url=""`), and the pull entrypoint has **no production caller** (only tests/docstrings). **Low today, High if pull mode is ever re-enabled.** Before re-enabling, validate the URL at provision *and* fetch time: require `http(s)`, reject private/link-local/loopback/metadata ranges, pin to the resolved IP (anti-rebind), and disable redirects.

### L2 — OpenAPI docs exposed in production
`src/main.py:205` never sets `docs_url=None`/`openapi_url=None`, so `/docs`, `/redoc`, `/openapi.json` are publicly reachable and disclose the full API surface to anonymous users (endpoints remain auth-gated — information disclosure only). Gate these to `None` in non-local envs.

### L3 — Prompt injection into detector-compile meta-prompts
`src/services/detector_compiler.py:224-251, 509-589`. Caregiver-supplied `name`/`description`/`custom_prompt` are interpolated into compile/infer meta-prompts, so a caregiver can attempt to override the hardcoded safety `_HARD_RULES`. Blast radius is limited: the output JSON schema is templated server-side, reserved keys are validated, and LLM output is **never** used to build SQL/shell/tool calls — and all text is same-tenant, rendered back only to the same caregiver. Residual risk is a caregiver degrading their *own* detector's safety bias. Keep the hard rules as the final instruction block (they are) and add length caps / light sanitization on `custom_prompt`/`name`.

### L4 — Cross-tenant device disclosure in `/v1/cameras/discovered`
`src/api/pi_discovery.py:97-122` returns *all* unclaimed Pis (LAN `current_url`, `device_label`, MAC-derived `hardware_id`) to *any* authenticated caregiver. Acknowledged in-code as acceptable at v1 scale; leaks other homes' device labels and LAN URLs. Scope the list per caregiver once the provisioning trust anchor (M2) exists.

### L5 — Missing security response headers / no TrustedHost
`src/main.py:204-255` registers no `CORSMiddleware`, security headers (HSTS, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`), or `TrustedHostMiddleware`. For a JSON/bearer-token API consumed by an iOS app this is low value (no HTML rendered, TLS/HSTS terminated at the edge, no cookie auth), but adding the standard header set is cheap hardening.

### L6 — Inconsistent (over-strict) scoping — correctness, not a hole
`talk.py:_ensure_patient_owned`, `detectors.py:_owned_camera`, `face_embeddings.py` delete, and `review.py` toggle/act use `patients.caregiver_id == caregiver_id` (ownership) instead of `accessible_patient_ids` (membership). This is *stricter* than the rest of the app — shared caregivers can't use Talk / wandering-door config / delete embeddings — so it's a correctness inconsistency, not an access-control gap. Align intentionally one way or the other.

---

## Verified Clean (coverage notes)

These were examined and found sound; recorded so the audit's coverage is explicit.

- **Row-Level Security** (`supabase/schema.sql`): RLS enabled on every tenant table; policies are membership-scoped via a `SECURITY DEFINER` `has_patient_access()` with `search_path` pinned; backend-internal tables (`gateways`, `pi_discoveries`) are RLS-on with **no policy** *plus* `REVOKE ALL … FROM anon, authenticated` (deny-all, defense in depth); `region`/`country` residency columns are immutable from the client path via triggers; `WITH CHECK` clauses prevent ownership reassignment.
- **Mass assignment:** none found — every PATCH/POST body model sets `extra="forbid"` and exposes only non-privileged fields; repos receive `model_dump(exclude_unset=True)` from those models; server-managed keys (`region`, `state`, `caregiver_id`, `tos_accepted_at`, …) are set server-side only.
- **Storage & signed URLs** (`src/services/storage.py`, `identity/storage.py`): object keys are derived from server-side UUIDs (never client filenames; extensions from a content-type whitelist), buckets are private, signed URLs are minted server-side with a short TTL, and the `service_role` key appears only in outbound `Authorization` headers — never in any response model. Photo-URL/clip-URL handlers scope the lookup to the caller's access before signing (no IDOR).
- **JWT verification** (`src/services/auth.py`): uses `jwt.decode()` with an algorithm allowlist and `audience="authenticated"`; no `alg:none` acceptance; no RS/HS confusion (HS256 uses the symmetric secret, not a public key). See M1 for the HS256-in-prod hardening.
- **LiveKit token minting** (`src/api/livekit.py`): dual-auth enforced — camera tokens require the api-key and must match the camera's own room+identity; caregiver tokens require membership in the `patient_<id>` room and are forced to a `caregiver_`-prefixed identity.
- **LiveKit webhook** (`src/api/livekit_webhooks.py`): signature verified via `WebhookReceiver`/`TokenVerifier` before any dispatch; fails closed.
- **Injection:** all raw `text()` SQL in `cameras_ingest.py` uses bound parameters; subprocess calls (`censor.py`, `mux.py`, `frame_extraction.py`, `mp4_remux.py`) use `create_subprocess_exec` with argv arrays (no shell, no user-string interpolation). No `eval`/`exec`/`pickle`/`yaml.load` on untrusted input.
- **Route authorization:** the access-control sweep confirmed every state-changing/data-returning route requires auth and scopes by caregiver access, except the issues in H1/H2/M2/M6 above.
- **Dependencies** (`uv.lock`): all security-sensitive packages are on current, patched versions — `fastapi 0.136.1`, `starlette 1.0.1`, `pyjwt 2.13.0` (no `python-jose`), `cryptography 48.0.0`, `python-multipart 0.0.29` (past CVE-2024-53981), `urllib3 2.7.0`, `pillow 12.2.0`, `aiohttp 3.13.5`, `requests 2.34.2`, `jinja2 3.1.6`. No known-vulnerable pins found.
- **Debug/error exposure:** FastAPI is not in debug mode; no custom exception handler leaks tracebacks; auth failures are flattened to an opaque `{"detail":"invalid_auth"}`.
- **Docker secret handling:** `.dockerignore`/`.railwayignore` exclude `.env*` and `.git`; the prod Dockerfile COPYs an explicit allow-list (no `COPY . .`), so no secret is baked into the image; experimental `/internals` routes default off.

---

*Generated by an automated Claude Code security-audit routine using the `vibe-security` methodology. Findings are static-analysis based; validate fixes against a running environment. Credential values in C1 were intentionally not reproduced.*
