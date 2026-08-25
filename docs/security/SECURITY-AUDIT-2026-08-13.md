# Security Audit — nemara-backend

**Date:** 2026-08-13
**Scope:** Static source review of the `nemara-backend` FastAPI service (auth, database access control, API endpoints, secrets, deployment config, AI/LLM surfaces).
**Method:** [`vibe-security`](https://github.com/raroque/vibe-security-skill) audit methodology — secrets, database RLS, authentication/authorization, rate-limiting/abuse, payments, AI integration, deployment, data-access/injection.
**Constraint:** Static only. No live systems, databases, or secrets were accessed.

---

## Executive Summary

nemara-backend is a well-engineered FastAPI backend for an in-home elderly/patient monitoring product. The security fundamentals are, on the whole, **notably strong for a fast-moving codebase**:

- **Supabase RLS is correctly implemented** — every tenant table is `ENABLE ROW LEVEL SECURITY` with membership-scoped `USING` *and* `WITH CHECK` clauses, backend-internal tables are deny-all (`REVOKE ALL ... FROM anon, authenticated`), and ownership cannot be reassigned client-side. This is the #1 source of critical vibe-coded vulnerabilities and it was handled with care.
- **JWT verification is correct** — signatures are verified (not `decode`-only), the `alg` is constrained to an allowlist, and `aud` is checked. Algorithm-confusion is not exploitable because the verification key is selected by `alg`.
- **No secrets are committed** in the current tree; `.env` is gitignored and `.env.example` holds only placeholders.
- **No SQL injection** — raw `text()` statements consistently use bound parameters.
- **No permissive CORS**, ingest/pi tokens use high entropy + constant-time comparison, and per-camera privacy severs exist as defense-in-depth.

The findings below are therefore **not** systemic collapses but specific, exploitable gaps. The two most important are a **hardcoded admin email allow-list that grants cross-tenant access to every patient's incident video** and an **unauthenticated, un-rate-limited 6-digit camera-pairing flow** that permits takeover of a patient's live camera during a pairing window. The recurring theme underneath several findings is the **complete absence of rate limiting** anywhere in the application, which the underlying product's sensitivity (protected health information, live video of vulnerable adults) makes materially riskier than for a typical app.

### Findings by severity

| # | Severity | Title | Location |
|---|----------|-------|----------|
| 1 | **High** | Hardcoded admin allow-list grants cross-tenant access to all patients' events & clips | `src/api/review.py:18` |
| 2 | **High** | Unauthenticated pairing with 6-digit code and no rate limiting → camera/feed takeover | `src/api/pairing.py:45`, `src/api/cameras.py:189` |
| 3 | **Medium** | No rate limiting or per-user cost cap on LLM endpoints → billing drain / open LLM proxy | `src/api/briefing.py:66`, `src/api/home_report.py:158` |
| 4 | **Medium** | Camera auth does a linear bcrypt scan → CPU-amplification DoS | `src/api/deps.py:137` |
| 5 | **Medium** | Unauthenticated `/health/freshness` discloses fleet-wide camera metadata | `src/api/health.py:191` |
| 6 | **Low** | Interactive API docs / OpenAPI schema exposed in all environments | `src/main.py:205` |
| 7 | **Low** | Historical secret exposure in git history (rotation unverified) | `.gitignore` (note) |
| 8 | **Low** | HS256 JWTs accepted in production (shared-secret forgery surface) | `src/services/auth.py:26` |

---

## High

### 1. Hardcoded admin allow-list grants cross-tenant access to every patient's events and incident clips

**`src/api/review.py:18-22`**

```python
REVIEWER_EMAILS = {"admin@gmail.com"}

def _is_admin(auth) -> bool:
    return str(auth.email).lower() in REVIEWER_EMAILS
```

`_is_admin` is consulted by all three reviewer endpoints, and when true it removes the per-caregiver ownership filter entirely:

- `GET /v1/review/queue` (`review.py:59`) — `where (:admin or p.caregiver_id = :cg)` returns **every patient's** held events across all tenants, including title, narrative, patient name, room, and a **freshly minted signed URL to the incident video clip** (`review.py:68`).
- `POST /v1/review/toggle` (`review.py:95`) — `where id = :pid and (:admin or caregiver_id = :cg)` lets an admin flip `require_review` on **any** patient.
- `POST /v1/review/{event_id}` (`review.py:108`) — approve/release **any** event.

**Why it is exploitable.** The admin decision is made purely from the `email` claim of the Supabase JWT. `admin@gmail.com` is a generic, third-party-controlled address, not a domain the company owns. Any principal whose token carries that email — obtained by registering a Supabase account with it (feasible if email confirmation is disabled, which is a common vibe-coded default) or by any future compromise of that mailbox — receives read access to the incident video and event stream of **every patient in the system**. For a product recording vulnerable adults in bathrooms and bedrooms, this is a direct PHI/privacy breach, not an abstract risk. The in-code comment "*Move to settings/DB when this leaves local dev*" indicates this local-dev shortcut is at risk of shipping to production.

**Remediation.**
- Remove the hardcoded email allow-list from source entirely.
- Drive privilege from a server-controlled signal that a user cannot self-assign: a dedicated `admin`/`reviewer` table keyed by caregiver UUID, or a Supabase custom claim (`app_metadata.role`) set only by the backend service role — never a plain `email` claim.
- Require a **verified** email and, ideally, a distinct authentication realm for reviewers rather than a normal caregiver account.
- Fail closed: no admin bypass unless the environment explicitly configures a reviewer identity.

```python
# Before
REVIEWER_EMAILS = {"admin@gmail.com"}
def _is_admin(auth) -> bool:
    return str(auth.email).lower() in REVIEWER_EMAILS

# After — privilege comes from a server-set claim / table, not an email string
async def _is_admin(auth, session) -> bool:
    return await reviewer_repo.is_reviewer(session, caregiver_id=auth.id)
```

---

### 2. Unauthenticated camera pairing with a 6-digit code and no rate limiting → live-feed takeover

**`src/api/pairing.py:45-98`** (endpoint) and **`src/api/cameras.py:189`** (code generation)

```python
# cameras.py:189 — the pairing code is a 6-digit number
candidate = str(secrets.randbelow(900_000) + 100_000)   # 900,000 possibilities

# pairing.py — unauthenticated; the code IS the proof of authorization
stmt = (select(Camera)
        .where(Camera.pairing_code == body.pairing_code)
        .where(Camera.pairing_expires_at.is_not(None))
        .where(Camera.pairing_expires_at > now))
cam = (await session.execute(stmt)).scalar_one_or_none()
...
api_key = secrets.token_hex(32)
cam.api_key_hash = bcrypt.hashpw(...)
return PairResponse(api_key=api_key, camera_id=..., livekit_identity=..., livekit_room=f"patient_{cam.patient_id}", ...)
```

**Why it is exploitable.** `POST /v1/cameras/pair` is unauthenticated by design (the code is the credential), and there is **no rate limiting, throttling, or lockout anywhere in the application** (confirmed: a repo-wide search for rate-limit primitives returns none). The code space is only 900,000 values and the lookup is **not scoped to any tenant** — a guess matches whichever camera anywhere in the system currently holds that code with an unexpired 10-minute window. A successful guess returns the camera's `api_key` plus its `livekit_room` (`patient_<id>`) and `livekit_identity`, i.e. full credentials to:

- publish media into a specific patient's LiveKit room, and
- act as that camera on every subsequent authenticated call.

During any active pairing window an attacker sending high-volume guesses (expected ~450k requests to hit a live code, well within a 10-minute window at a few thousand req/s, with nothing to slow them down) can hijack the pairing and bind their own device to a stranger's patient. This is exactly the OTP/pairing brute-force the methodology warns about.

**Remediation.**
- Add strict rate limiting and lockout on `/v1/cameras/pair` — per source IP *and* a global ceiling — plus exponential backoff and alerting on repeated failures.
- Enforce a small per-camera attempt budget: after N failed attempts against a pairing window, invalidate the code and force regeneration.
- Increase entropy (e.g. 8+ digits or alphanumeric) if UX allows; the current 6 digits only stay safe *with* aggressive rate limiting.
- Consider binding the pairing exchange to the initiating caregiver session so a code is only redeemable in the context that created it.

---

## Medium

### 3. No rate limiting or per-user cost cap on LLM endpoints → billing drain and open LLM proxy

**`src/api/briefing.py:66-73`** and **`src/api/home_report.py:158-163`**

```python
# briefing.py — module docstring: "Deliberately a pure LLM PROXY"
client = AsyncOpenAI(api_key=settings.openai_api_key)
resp = await client.chat.completions.create(
    model="gpt-5.4-mini",
    messages=[{"role": "user",
               "content": _PROMPT.format(name=body.name, facts=body.facts)}],
)
```

`BriefingRequest.facts` is `dict[str, object]` with **no size or depth bound** (`briefing.py`/`schemas`), formatted directly into the prompt. `POST /v1/home-report` similarly issues a multimodal GPT call (up to 4 images) on every request.

**Why it is exploitable.** Both endpoints require a valid caregiver JWT, but there is no per-user quota, no request-size cap, and no rate limit. A single authenticated caregiver — or anyone with a stolen/replayed token — can call these in a tight loop and/or submit very large `facts` payloads, driving unbounded OpenAI spend. Because `briefing` embeds fully client-controlled content into the prompt and returns the model's raw text, it doubles as a **free general-purpose LLM proxy** billed to the company, and is trivially prompt-injectable (low direct harm here since output returns only to the same caller, but it defeats the guardrail prompt's intent).

**Remediation.**
- Enforce per-user (and per-IP) rate limits and daily/monthly token quotas with hard ceilings, returning `429` when exceeded.
- Bound input size: cap `facts` serialized length and reject oversized bodies before calling the model.
- Set hard spending caps in the OpenAI/Gemini dashboards as a backstop (provider caps lag, so app-level quotas are primary).
- Prefer sending client facts as a structured `user` message distinct from the system prompt rather than string-formatting into one prompt.

---

### 4. Camera authentication performs a linear bcrypt scan → CPU-amplification DoS

**`src/api/deps.py:137-159`**

```python
stmt = select(Camera).where(Camera.api_key_hash.is_not(None))
rows = (await session.execute(stmt)).scalars().all()
for cam in rows:
    ...
    matched = await loop.run_in_executor(None, bcrypt.checkpw, token, hash_bytes)
    if matched:
        return cam
```

**Why it is exploitable.** Every request to a `CameraAuth`-protected route verifies the presented bearer against **every** camera row's bcrypt hash (cost `2^bcrypt_rounds`, default `2^10`) until a match or exhaustion. A non-matching token therefore forces `O(N)` deliberately-expensive hash operations. With no rate limiting, an attacker submitting garbage bearer tokens amplifies each cheap HTTP request into N×bcrypt of server CPU; as the fleet grows this becomes a low-effort denial-of-service that also starves the executor pool shared by the event loop. The code's own comments acknowledge the linear scan as a "Phase 0" shortcut.

**Remediation.**
- Move camera api-keys to the same model already used for **ingest** tokens (`src/services/ingest/pi/auth.py`): store a SHA-256 digest of a high-entropy random key and do a single **indexed equality** lookup (`WHERE api_key_hash = :digest`), eliminating the scan and the per-request bcrypt cost. High-entropy random tokens do not need bcrypt's slowness — the entropy already defeats offline guessing.
- Alternatively, add the `api_key_prefix` index the comment describes so at most one bcrypt runs per request.
- Add rate limiting on the camera-facing routes regardless.

---

### 5. Unauthenticated `/health/freshness` discloses fleet-wide camera metadata

**`src/api/health.py:191-236`**

```python
@router.get("/health/freshness", response_model=FreshnessResponse)
async def freshness(session: SessionDep) -> FreshnessResponse:   # <-- no auth dependency
    rows = (await session.execute(text(
        "select substr(c.id::text, 1, 8) as cam, ... age ..."
        " where c.online and not c.disabled and not c.ai_paused"))).all()
    ...
    raise HTTPException(status_code=503, detail={"status": "stale", "ages_s": ages, ...})
```

**Why it is exploitable.** Unlike the sibling `/ready` endpoint — which was deliberately placed behind the `X-Internal-Key` shared secret and fails closed outside local/test — `/health/freshness` has no auth. Any unauthenticated caller learns, for **every online camera in the system**, a partial camera ID and the age of its most recent analysis (both in the `200` body and the `503` detail). That is operational and PHI-adjacent metadata (which patients are actively monitored, and whether their feed is currently flowing) leaked to the internet.

**Remediation.** Gate `/health/freshness` behind the same `X-Internal-Key` guard as `/ready`, or reduce its public form to a bare `ok`/`stale` boolean with no per-camera identifiers or ages. Keep the unauthenticated `/health` (pure liveness) for platform probes.

---

## Low

### 6. Interactive API docs and OpenAPI schema exposed in all environments

**`src/main.py:205-209`**

```python
app = FastAPI(title="nemara-backend", version="0.0.1", lifespan=lifespan)
```

FastAPI mounts `/docs`, `/redoc`, and `/openapi.json` by default, and `main.py:57-66` confirms they are live (it adds an access-log filter to quiet them). This publishes the complete API surface — every route, parameter, and schema — to unauthenticated scanners, easing reconnaissance of the endpoints above.

**Remediation.** Disable the docs/schema outside local/test:

```python
_s = get_settings()
_expose_docs = _s.env in ("local", "test")
app = FastAPI(
    title="nemara-backend", version="0.0.1", lifespan=lifespan,
    docs_url="/docs" if _expose_docs else None,
    redoc_url="/redoc" if _expose_docs else None,
    openapi_url="/openapi.json" if _expose_docs else None,
)
```

### 7. Historical secret exposure in git history (rotation unverified)

**`.gitignore` (Env section)**

The gitignore notes: "*`.env` was previously tracked for partner onboarding; it has been untracked and its leaked secrets rotated*," and similarly that `out/` id-frame JPEGs and a `dash.log` were once committed. Deleting a tracked `.env` does not remove it from history — anyone with an old clone or fork can recover whatever was in those commits.

**Why it matters.** If any of the once-committed values (Supabase `service_role` key, LiveKit API secret, OpenAI/Gemini/OpenRouter/DashScope keys, Resend key, Backblaze B2 keys, APNs auth key) were *not* fully rotated, they remain usable. Static review cannot confirm the rotation actually covered every key (git history was not deep-cloned here).

**Remediation.** Verify rotation of **all** credentials that ever touched a committed `.env`. Run `gitleaks detect --log-opts="--all"` over full history. Consider purging the secrets from history (`git filter-repo`) and confirm the id-frame JPEGs contained no patient imagery of concern.

### 8. HS256 JWTs accepted in production (shared-secret forgery surface)

**`src/services/auth.py:26`**

```python
JWT_ALGORITHMS = ["ES256", "HS256"]
```

The verifier accepts both ES256 (JWKS public-key, production) and HS256 (shared `supabase_jwt_secret`, local/legacy). This is *correctly* implemented — the key is chosen by `alg`, so the classic "public key used as HMAC secret" confusion is not possible. The residual risk is only that, as long as HS256 is accepted in production, knowledge of `supabase_jwt_secret` allows forging a token for **any** user. That secret is a required prod setting and a high-value single point of failure.

**Remediation (hardening).** Once all production tokens are ES256, restrict the accepted algorithms to `["ES256"]` in non-local environments so a leaked symmetric secret cannot mint valid caregiver tokens.

---

## What was checked and found clean

- **Supabase RLS** (`supabase/schema.sql:922-1049`) — all tenant tables RLS-on with membership-scoped `USING` + `WITH CHECK`; `gateways`/`pi_discoveries` are deny-all with an explicit `REVOKE`; `patients` write-check prevents ownership reassignment; `has_patient_access` is `SECURITY DEFINER` with `SET search_path`.
- **JWT auth** (`src/services/auth.py`) — signature verified, `aud="authenticated"` enforced, `alg` allowlisted, opaque 401s.
- **Endpoint authorization** — patient/event/detector/camera routes are consistently scoped by `accessible_patient_ids(...)` (`src/repos/access.py`) and return 404 (not 403) cross-account; `livekit.py` token minting verifies room membership and forces a `caregiver_`-prefixed identity; `/internal/*` fails closed on the shared key.
- **SQL injection** — raw `text()` statements use bound parameters throughout; the one f-string in `cameras_ingest.py:393` interpolates a hardcoded constant with bound params, not user input.
- **Secrets** — none in the tracked tree; `.env` gitignored; `.env.example`/`.pod.env.example` are placeholders.
- **Ingest/Pi tokens** (`src/services/ingest/pi/auth.py`) — 32-byte random, SHA-256, `hmac.compare_digest`.
- **CORS** — no middleware configured (no cross-origin exposure); no `debug=True`; no client-side secret prefixes (this is a pure backend).
- **Payments** — no payment/Stripe surface present in this service; section not applicable.

---

## Prioritized remediation summary

1. **(High) Remove the `admin@gmail.com` allow-list** — replace with a DB/claim-based reviewer privilege that a user cannot self-assign; it currently exposes all patients' incident video. *(`review.py:18`)*
2. **(High) Rate-limit and harden camera pairing** — throttle/lock `POST /v1/cameras/pair`, cap per-code attempts, raise entropy; today a 6-digit code with no throttle enables live-feed takeover. *(`pairing.py`, `cameras.py:189`)*
3. **(Medium) Add app-level rate limits + token quotas** across the app, starting with the LLM endpoints (`briefing`, `home-report`) and the camera-auth path; set provider spend caps.
4. **(Medium) Replace the camera-auth bcrypt scan** with an indexed SHA-256 lookup (mirror the ingest-token design).
5. **(Medium) Authenticate `/health/freshness`** behind `X-Internal-Key`.
6. **(Low) Disable `/docs` `/redoc` `/openapi.json` in production**; verify full rotation of any once-committed secrets; restrict prod JWTs to ES256.

*This report is a static, best-effort review and does not guarantee the absence of other vulnerabilities. No application code was modified.*
