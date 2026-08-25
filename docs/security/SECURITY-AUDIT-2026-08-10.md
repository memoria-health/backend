# Security Audit — nemara-backend

**Date:** 2026-08-10
**Auditor:** Automated static security review (vibe-security methodology)
**Scope:** Static source review of the `nemara-backend` FastAPI + Supabase repository at commit `ae7b553`. No live systems, databases, or secrets were accessed.
**Methodology:** `vibe-security` skill — secrets/env, database access control (Supabase RLS), authn/authz, rate limiting, payments, AI/LLM, deployment, data access/input validation.

---

## Executive Summary

nemara-backend is a **security-conscious codebase**. Authentication and authorization are, on the whole, done correctly: JWTs are verified (not merely decoded) with the algorithm and audience pinned; Supabase Row-Level Security is enabled on **every** tenant table with membership-scoped policies; the FastAPI repo layer independently re-scopes queries by `caregiver_id` as defense-in-depth; API-key, ingest-token, and internal-service-key checks use constant-time comparison and fail closed. LiveKit token minting and Stripe-style client-trust pitfalls are handled well. These are genuinely above-average fundamentals for a rapidly-built application.

The report is dominated by **one Critical issue that outweighs everything else**: live production credentials are present in git history — and, worse, a second credential dump (`.env.bak.1783536884`) was committed on **2026-07-22, six weeks after** the June 9 "stop tracking .env / rotate secrets" cleanup, carrying a *different, apparently un-rotated* set of keys. Because one of those secrets is the Supabase `service_role` key, anyone who can read the repository history can bypass all Row-Level Security and read or modify every patient's data. This must be remediated (rotate + purge history) before anything else.

There is also one genuine **High** access-control flaw beyond the secrets: `POST /v1/gateway/provision-camera` performs an authentication *gesture* (it requires a bearer token) but no actual *authorization* — the token is attacker-generated and the target account is selected by a non-secret email — letting an anonymous caller create spoofed cameras under any caregiver's patients and receive working ingest tokens for them.

The remaining findings are hardening gaps typical of pre-MVP backends: **no rate limiting anywhere** (brute-force / cost-abuse exposure), a latent SSRF via client-supplied Pi stream URLs, a hardcoded reviewer-admin email allow-list, HS256 JWTs accepted alongside ES256 in production, interactive API docs exposed in production, and a container that runs as root.

### Findings by severity

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | **Critical** | Live secrets in git history, incl. Supabase `service_role` key; re-leaked after "rotation" | git history (`.env`, `.env.bak.1783536884`) |
| 2 | **High** | Unauthenticated cross-account camera provisioning (auth check establishes no authorization) | `src/api/cameras_ingest.py:531` |
| 3 | **High** | No rate limiting on any endpoint (auth/pairing brute-force, AI cost, ingest DoS) | entire API surface |
| 4 | **Medium** | SSRF: client-supplied `pi_stream_url` / `current_url` fetched server-side | `src/api/cameras.py:325`, `src/api/pi_discovery.py:67`, sink `relay_publisher.py:239` |
| 5 | **Medium** | Cross-tenant admin access keyed on a hardcoded gmail email claim | `src/api/review.py:17` |
| 6 | **Medium** | HS256 JWTs accepted in production alongside ES256 (forgeable if shared secret leaks) | `src/services/auth.py:26` |
| 7 | **Medium** | No in-app per-user AI usage cap (unbounded model spend) | `src/services/detection/*` |
| 8 | **Low** | Interactive API docs / OpenAPI schema served in production | `src/main.py:204` |
| 9 | **Low** | Container runs as root (no `USER` directive) | `docker/Dockerfile:78` |
| 10 | **Low** | Authz keyed on ownership (`patients.caregiver_id`) instead of membership in some handlers | `detectors.py:115`, `talk.py:176`, `presence.py:24` |
| 11 | **Low** | `SECURITY DEFINER` helpers use `search_path = public`, not `''` | `supabase/schema.sql:957` |
| 12 | **Info** | Floating (`>=`) dependency version constraints | `pyproject.toml` |

---

## Critical

### 1. Live production secrets committed to git history — including the Supabase `service_role` key — and re-leaked after the documented rotation

**Where:** git history. `.env` was tracked and later removed in `cf36938` (2026-06-09, *"chore(security): stop tracking .env … Secrets in prior history still require rotation"*). A second file, **`.env.bak.1783536884`**, was then added in commit **`a10e682` (2026-07-22, `feat(db): add caregivers.region`)** and remains reachable from `main`.

**What is exposed** (both files contain real, non-placeholder values — verified by inspecting the blobs, values redacted here):

- `NEMARA_SUPABASE_SERVICE_ROLE_KEY` — a Supabase `service_role` JWT (`eyJ…`). **This bypasses ALL Row-Level Security.**
- `NEMARA_SUPABASE_JWT_SECRET` — the HS256 signing secret. Anyone holding it can **forge caregiver auth tokens** for any account (see Finding 3).
- `NEMARA_GEMINI_API_KEY` (`AIza…`), `NEMARA_OPENAI_API_KEY` (`sk-p…`) — billable LLM keys.
- `NEMARA_LIVEKIT_API_KEY` / `NEMARA_LIVEKIT_API_SECRET` — mint room tokens for any patient's live audio/video.
- `NEMARA_APNS_AUTH_KEY` — the APNs `.p8` ES256 private key (push-notification signing).

**Why it is exploitable:** The two committed files carry **different values** (e.g. distinct LiveKit API keys and distinct APNs team IDs), so `.env.bak.1783536884` is a *separate* credential set. Its filename timestamp (`1783536884` = 2026-07-08) and its commit date (2026-07-22) both post-date the 2026-06-09 rotation commit — meaning the June rotation did **not** cover these keys, and they are very likely still live. With the `service_role` key alone, an attacker with read access to the repo (a leaked clone, a former collaborator, a compromised laptop, or a future public/open-source push) can connect to Supabase and read/modify/delete **every patient's** medical-monitoring data, camera rows, face embeddings, and incident clips — RLS provides zero protection against `service_role`. This is patient health / biometric data, so the blast radius includes a reportable data breach.

**Concrete impact:** Full tenant-wide data compromise (read + write + delete), forged authentication as any caregiver, drained LLM/LiveKit billing, and the ability to sign push notifications to patients' caregivers.

**Remediation (do all, in order):**
1. **Rotate every credential now**, treating all values in both files as compromised: Supabase `service_role` **and** JWT secret (rotate in the Supabase dashboard → this invalidates existing sessions, which is the point), Gemini key, OpenAI key, LiveKit API key/secret, and the APNs auth key. Rotating the `service_role`/JWT secret is the load-bearing step.
2. **Purge the files from history** (`git filter-repo --path .env --path .env.bak.1783536884 --invert-paths`, or BFG), then force-push and have all collaborators re-clone. Note that GitHub may retain unreachable blobs — rotation, not just history rewrite, is what actually protects you.
3. **Prevent recurrence:** the current `.gitignore` covers `.env`, `.env.bak*`, and `.pod.env`. Add a pre-commit secret scanner (`gitleaks protect --staged`) and a CI `gitleaks detect` gate so a future `*.bak` with secrets is blocked at commit time. `.env.bak.1783536884` matched none of the ignore rules *at the time it was committed*, which is exactly how it slipped in.

```bash
# Confirm nothing sensitive is tracked now (should show only *.example):
git ls-files | grep -iE '\.env|secret|\.p8|\.pem'
# Rewrite history to drop both leaked files:
git filter-repo --path .env --path .env.bak.1783536884 --invert-paths
```

---

## High

### 2. Unauthenticated cross-account camera provisioning — the auth check establishes no authorization

**Where:** `src/api/cameras_ingest.py:531-648` — `POST /v1/gateway/provision-camera`.

**What the code does:**
```python
if not authorization.lower().startswith("bearer "):
    raise HTTPException(401, "missing_bearer_token")
box_token = authorization[len("bearer "):].strip()      # attacker-chosen, "generated on first boot"
...
cg = (await session.execute(
    text("SELECT id FROM public.caregivers WHERE lower(email) = lower(:e)"),
    {"e": body.email.strip()})).first()                 # victim selected by EMAIL (not a secret)
...
# binds box_token -> that caregiver in public.gateways, then creates a
# source_type='gateway' Camera under EVERY one of the caregiver's patients
# and RETURNS the plaintext ingest_token for each (line 604, 644)
```

**Why it is exploitable:** The endpoint requires a bearer token but never verifies it against anything — the token is generated by the caller ("on first boot") and is simply *bound* to whichever caregiver the request names via `body.email`. The account is chosen purely by email, which is not secret. So an **anonymous** attacker who knows (or guesses) a caregiver's email can:

1. Register their own token as a gateway for the victim (`gateways` insert, line 573),
2. Cause a `source_type='gateway'` camera to be created under **every one of the victim's patients**, inside the victim's `patient_<id>` LiveKit room (lines 596–634),
3. Receive the **plaintext `ingest_token`** for each of those cameras in the HTTP response (line 644),
4. Then call `/v1/cameras/me/frames` with that token to **push attacker-controlled video/audio frames into the patient's room** — which drives the detection pipeline, fabricated events, incident clips, and real push/email alerts to the genuine caregiver — and enumerate the victim's gateway cameras via `/v1/gateway/roster` and `/v1/gateway/wait`.

The `body.email` is parameterized (no SQL injection), and provisioning requires the victim to already have ≥1 patient (else `422`). But none of that limits the core flaw: a caller with no valid credential of any kind creates resources in another tenant and walks away with working ingest tokens for that tenant's patient rooms. For a patient-safety monitoring product, injected/fabricated telemetry and false alerts are a direct integrity and trust harm.

**Impact:** Unauthenticated cross-account resource creation, camera spoofing, and frame injection into arbitrary patients' rooms (given only the caregiver's email).

**Remediation:** Provisioning must be authorized, not merely token-shaped. Either:
- Require a `CaregiverAuth` session so the account is proven by JWT (the caregiver provisions their own box from the app), **or**
- Issue each gateway box a **pre-registered, server-generated** secret at manufacture/enrollment and verify the presented token against a stored hash **before** binding it to any account — never trust a self-asserted first-boot token plus a plaintext email as proof of ownership.
Also stop returning plaintext ingest tokens to an unauthenticated caller.

### 3. No rate limiting anywhere on the API

**Where:** entire API surface. A repository-wide search for `ratelimit` / `slowapi` / `limiter` / `upstash` / `429` returns **no** rate-limiting implementation. There is no CORS middleware and no per-endpoint throttle.

**Why it matters (per the vibe-security rate-limiting checklist, these endpoints all need limits and have none):**

- **`POST /v1/cameras/pair`** (`src/api/pairing.py:40`) — unauthenticated pairing-code exchange. Codes are 6-digit numeric (`secrets.randbelow(900_000)+100_000`, `src/api/cameras.py:189`), cryptographically generated, 10-minute TTL, single-use. The short window and ~900k space make single-window brute-force impractical but not impossible (≈6–7% success at ~100 req/s for a known-open window), and there is **no lockout** — an attacker can hammer the endpoint freely across every open pairing window. A successful guess pairs a rogue wall-phone to a patient and yields a long-lived camera `api_key`.
- **AI/LLM tick path** — every detector tick calls Gemini/OpenAI (billable). No per-account request ceiling exists in the application (see Finding 4).
- **Pi/gateway ingest** (`POST /v1/cameras/me/frames`, `/v1/gateway/*`) — CPU-intensive frame decode with no application-level limit. The ingest-token module even comments that it is "rate-limited at the transport layer," but no such limit is present in this repo.
- **Support / feedback email** (`POST /v1/support`, `POST /v1/feedback`) — trigger outbound Resend emails; without limits they can be abused as a spam/cost relay.

**Impact:** Credential brute-forcing without lockout, unbounded LLM/infra cost, and denial-of-service against the ingest and email paths.

**Remediation:** Add rate limiting at the edge (Railway/reverse proxy) and/or in-app (`slowapi`, or a small Redis/Upstash sliding-window). Combine per-IP **and** per-principal (per-camera / per-caregiver) limits. Prioritize `POST /v1/cameras/pair` (tight per-IP limit + a per-camera attempt lockout that invalidates the code after N failures), the AI tick path, and the ingest endpoints.

---

## Medium

### 4. SSRF: client-supplied stream URLs fetched server-side

**Where:** stored at `src/api/cameras.py:325` (`provision_pi_camera`, direct-URL mode) and `src/api/pi_discovery.py:67-73` (`announce_pi`, **unauthenticated**); the schema explicitly declines to validate (`src/schemas/camera.py:32` — *"We do not parse / validate"*); server-side fetch sink at `src/services/ingest/pi/relay_publisher.py:239-243` (`read_mjpeg(pi_stream_url)`).

**Why it matters:** An arbitrary caller-supplied URL (`pi_stream_url` / `current_url`) is persisted and later opened server-side by the pull-mode relay with **no host allowlist and no block on private/link-local ranges** (`127.0.0.1`, RFC-1918, `169.254.169.254` cloud metadata). `announce_pi` is unauthenticated, so anyone can seed a `current_url`; a caregiver can then adopt it by claiming the discovery. Additionally, `GET /v1/cameras/discovered` (`pi_discovery.py:93-122`) returns **every** unclaimed Pi's `current_url` to any authenticated caregiver with no per-caregiver scoping (acknowledged in the docstring) — a minor cross-tenant info leak.

**Why Medium, not High:** the pull-mode relay (`relay_publisher.run()`) is documented as backwards-compatibility for a now-disabled reconciliation worker; the active ingest path is push-mode. This is a **latent** SSRF — the untrusted URL is stored and reachable by any code path that calls `run()`.

**Impact:** Server-side requests to internal/metadata endpoints if the pull path is ever re-enabled; unauthenticated write of attacker URLs; cross-tenant disclosure of discovered-device URLs.

**Remediation:** Before any server-side fetch of a client-supplied URL, resolve the host and reject private/link-local/loopback/metadata ranges, and apply a scheme + host allowlist. Authenticate `announce_pi` (or accept its unauthenticated nature only with the SSRF guard in place), and scope `/cameras/discovered` per caregiver.

### 5. Cross-tenant admin access keyed on a hardcoded gmail email claim

**Where:** `src/api/review.py:17` — `REVIEWER_EMAILS = {"admin@gmail.com"}`; `_is_admin` (`:20`) grants global access used by `queue` (`:59`), `toggle` (`:95`), and `act` (`:108`): `where (:admin or p.caregiver_id = :cg)`. The router is mounted unconditionally in `src/main.py:215`.

**Why it matters:** A caregiver whose Supabase JWT carries `email == "admin@gmail.com"` gets **cross-tenant read/write** over every patient's held events and the per-patient `require_review` toggle. Authorization rests entirely on an email-claim string match against a hardcoded, public gmail address, and it ships to production (the code comment concedes it is a local-dev artifact: *"Move to settings/DB when this leaves local dev"*). Email is a weak authorization key — if the JWT issuer ever admits an unverified email, or that address is otherwise obtainable, it becomes global data access.

**Impact:** Potential tenant-wide read/write of patient events via a spoofable/guessable identity axis.

**Remediation:** Replace the email allow-list with an explicit role claim (Supabase custom claim / app-role table) checked server-side, remove the hardcoded address, and ensure the reviewer surface is disabled or role-gated in production.

### 6. HS256 JWTs accepted in production alongside ES256

**Where:** `src/services/auth.py:26` — `JWT_ALGORITHMS = ["ES256", "HS256"]`, dispatched on the token's own `alg` header (`:67`, `:72–88`).

**Why it matters:** Production Supabase tokens are ES256 (asymmetric, verified via JWKS public keys — good). The verifier *also* accepts HS256 signed with the shared `supabase_jwt_secret`. This is intended for local dev, but it is enabled unconditionally in every environment. The risk is directly compounded by Finding 1: the HS256 secret is one of the leaked values. Anyone holding that secret can **forge a valid HS256 caregiver token for any `sub`/`email`** and the backend will accept it. The key-selection logic is otherwise sound (it does not feed the ES256 public key into HMAC verification, so the classic algorithm-confusion swap is not directly exploitable) — the exposure is specifically "HS256 accepted + shared secret compromised."

**Impact:** Full authentication bypass / impersonation of any caregiver if the JWT secret is (or ever becomes) known.

**Remediation:** In production, restrict accepted algorithms to `["ES256"]` and require JWKS verification; gate HS256 behind `env in ("local","test")`. Independently, rotate the JWT secret (Finding 1).

```python
# src/services/auth.py — env-gated algorithm allowlist
algorithms = ["ES256"] if settings.env == "prod" else ["ES256", "HS256"]
```

### 7. No in-application per-user cap on AI/LLM spend

**Where:** the detection pipeline (`src/services/detection/*`, `providers/*`). Model choice, fps, frame count, and thinking level are richly configurable in `src/settings.py`, but there is no per-caregiver / per-patient **usage quota** enforced in the app. Cost is bounded only by tick cadence and the number of active cameras.

**Why it matters:** Per the AI-integration guidance, provider-side caps lag and should not be the only defense. A compromised or abusive account (e.g. one that pairs many cameras, or a forged token per Finding 3) can drive LLM spend with no application-side ceiling or anomaly alert.

**Impact:** Unbounded/abusive LLM billing.

**Remediation:** Track token/request usage per account in the DB and enforce daily/monthly hard caps that short-circuit the tick when exceeded; set hard spending caps in the Gemini/OpenAI consoles; add anomaly alerting on sudden spikes. This pairs with Finding 2.

---

## Low

### 8. Interactive API docs and OpenAPI schema exposed in production

**Where:** `src/main.py:204` — `FastAPI(...)` is constructed without `docs_url=None` / `redoc_url=None` / `openapi_url=None`. `main.py:57–69` even installs an access-log filter for `/openapi.json`, `/docs`, `/redoc`, confirming they are live in all environments.

**Why it matters:** `/docs`, `/redoc`, and `/openapi.json` publish the complete route inventory, request/response schemas, and auth scheme to any unauthenticated visitor in production — a reconnaissance aid. Not a direct compromise, but the deployment checklist calls for disabling non-essential surface in prod.

**Remediation:** Disable the docs routes when `env == "prod"`:
```python
_prod = get_settings().env == "prod"
app = FastAPI(title="nemara-backend", version="0.0.1", lifespan=lifespan,
              docs_url=None if _prod else "/docs",
              redoc_url=None if _prod else "/redoc",
              openapi_url=None if _prod else "/openapi.json")
```

### 9. Container runs as root

**Where:** `docker/Dockerfile:78` — `CMD ["sh","-c","uv run uvicorn …"]` with no `USER` directive anywhere in the image.

**Why it matters:** The application process runs as UID 0. Any RCE or container escape has root inside the container, widening blast radius. (No `--reload`/debug flag is set — good.)

**Remediation:** Add a non-root user and drop to it before `CMD`:
```dockerfile
RUN useradd -r -u 10001 appuser
USER appuser
```

### 10. Authorization keyed on ownership instead of membership in several handlers

**Where:** `src/api/detectors.py:115-124` (`_owned_camera`) and `:229`, `src/api/talk.py:176-213`, `src/api/presence.py:24-31`, `src/api/home_report.py:99-124`, `src/api/face_embeddings.py:308-315` — all scope by `Patient.caregiver_id == caregiver_id` (ownership) rather than the `accessible_patient_ids()` membership subquery used everywhere else.

**Why it matters:** This is **stricter** than the membership model, so it is **not** an IDOR/leak — a non-owner shared caregiver is *denied*, not over-granted. The concern is correctness and consistency: the patient-sharing feature (`patient_members`) will silently fail on these routes (talk mode, wandering-door config, presence, home report, face-embedding delete) for shared caregivers. All use bound parameters (no injection). Flagged so the divergence is fixed before sharing ships broadly.

**Remediation:** Route these checks through `accessible_patient_ids()` / `has_patient_access` so authorization is uniform with the rest of the API.

### 11. `SECURITY DEFINER` functions use `search_path = public` instead of `''`

**Where:** `supabase/schema.sql:957` (`has_patient_access`), and the `sync_caregiver_email_from_auth` / `region_of` helpers.

**Why it matters:** `has_patient_access` is `SECURITY DEFINER` and lives in `public` (callable via PostgREST). It correctly pins access to `auth.uid()`, so it is not an authz bypass. However, the database-security guidance recommends `SET search_path = ''` (fully-qualified names) on `SECURITY DEFINER` functions to eliminate search-path–based function/table shadowing. These use `search_path = public`. Low risk because inputs are trivial and the bodies reference `public.` explicitly, but worth hardening.

**Remediation:** Set `search_path = ''` and fully-qualify all object references inside these functions.

---

## Informational

### 12. Floating dependency version constraints

`pyproject.toml` pins dependencies with `>=` (e.g. `fastapi>=0.115`, `livekit-api>=1.0`). `uv.lock` provides reproducibility for locked installs, but the manifest permits drift. No known-vulnerable pinned versions were identified in static review. Recommend periodic `pip-audit` / `uv` advisory scanning in CI.

---

## What was checked and found sound (no action needed)

These are called out so the team knows they were reviewed, not skipped:

- **Supabase RLS** (`supabase/schema.sql`): RLS is `ENABLE`d on every tenant table; policies are membership-scoped via `has_patient_access(patient_id)` (not `USING (true)` / `USING (auth.uid() IS NOT NULL)`); `WITH CHECK` clauses are present on writes; backend-internal tables (`gateways`, `pi_discoveries`) are RLS-on with **no policy plus an explicit `REVOKE … FROM anon, authenticated`** (deny-all, defense-in-depth); data-residency `region`/`country` are immutable from the client path via triggers. This is notably well done.
- **JWT verification** (`src/services/auth.py`): uses `jwt.decode()` with signature verification, explicit `algorithms=`, and `audience="authenticated"` — no `jwt.decode(..., verify=False)`, no `alg:none` acceptance. (The only caveat is the HS256-in-prod breadth, Finding 3.)
- **Authorization / IDOR (the common path):** the vast majority of handlers scope by the authenticated principal via the centralized `accessible_patient_ids()` membership subquery in the repo layer — `patch_patient` / `update_partial` pass `caregiver_id=auth.id`; LiveKit token minting verifies camera↔room match and caregiver↔patient membership (`src/api/livekit.py:175`, `:195`); the `/internal` route uses a constant-time shared-secret guard that fails closed outside local/test (`src/api/internal/cameras.py:48`); the camera/ingest/webhook auth schemes are all sound. A full route inventory was built and only the deviations are reported above (Findings 2, 5, 10). No route was found missing an auth check *unintentionally* — the three unauthenticated routes (`/health`, `/v1/cameras/pair`, `/v1/cameras/pi-announce`) are unauthenticated by design; `gateway/provision-camera` (Finding 2) is the one whose auth check does not actually establish authorization.
- **Mass assignment:** update bodies are Pydantic models with `extra="forbid"` exposing only caregiver-writable fields; server-owned columns (`state`, `region`, `caregiver_id`, `privacy_enabled`) are not client-writable.
- **SQL injection:** all raw `text()` usage (`src/services/review_gate.py`, `src/services/detection/worker.py`) uses **bound parameters** (`:ids`, `:id`, `:k`); no string-formatted SQL. ORM access is parameterized.
- **Command injection:** `subprocess` / `create_subprocess_exec` calls (censor, mux, encoder) use argument lists (no `shell=True`) over internal temp paths, not user-controlled strings.
- **Secrets handling at runtime:** the Supabase `service_role` key is used only server-side (`src/services/storage.py`, `src/services/auth_admin.py`); storage buckets are private with short-lived signed URLs; camera `api_key` is bcrypt-hashed, ingest/gateway tokens are SHA-256 with constant-time compare.
- **Payments:** no Stripe/payment flow present — section not applicable.
- **Prompt injection / AI output trust:** caregiver "custom detector" text is length-bounded (≤2000 chars, `src/schemas/detector.py:68`), passes a compile-time reject/reserved-key gate (`compiler_v2.reject_reason` / `validate_schema_fields`), and LLM output is parsed as structured JSON — not executed and not exposed to tools. Injection risk is low and self-scoped (author == victim).

---

## Prioritized remediation summary

1. **(Critical) Rotate every leaked credential and purge git history.** The Supabase `service_role` key in `.env.bak.1783536884` (committed 2026-07-22, after the documented rotation) gives full RLS-bypassing access to all patient data. Rotate first, rewrite history second.
2. **(High) Fix `POST /v1/gateway/provision-camera`** so provisioning is authorized by a `CaregiverAuth` session or a pre-registered, server-verified box secret — not a self-asserted token plus a public email. Stop returning ingest tokens to unauthenticated callers.
3. **(High) Add rate limiting** — per-IP + per-principal — starting with `POST /v1/cameras/pair` (add failure lockout), the AI tick path, and ingest endpoints.
4. **(Medium) Add an SSRF guard** (private/link-local/metadata range block + host allowlist) before any server-side fetch of client-supplied stream URLs; authenticate `announce_pi`; scope `/cameras/discovered`.
5. **(Medium) Replace the hardcoded reviewer-admin email allow-list** with a role claim; restrict JWT algorithms to ES256 in production (gate HS256 to local/test); enforce per-account AI usage quotas and provider spending caps.
6. **(Low) Disable `/docs` `/redoc` `/openapi.json` in prod; run the container as a non-root user; align ownership-scoped handlers on membership; set `search_path=''` on `SECURITY DEFINER` functions; add a `gitleaks` CI gate.**
