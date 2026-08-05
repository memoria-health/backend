# Memoria — Security Remediation, 2026-08-05

**Supersedes but does not replace** `SECURITY-AUDIT-2026-08-03-CROSS-SYSTEM.md`.
That document stays as the point-in-time record. This one says what was
re-verified against **production**, what turned out to be wrong, what was
fixed, and what is knowingly left open.

**Method:** live probes against the deployed backend, plus source verification
on the current branches. Where the 08-03 audit carried a finding forward
without re-testing, it was re-tested here.

**Scope:** `nemara-backend`, `nemara-web-app`, `nemara-caregiver-app`,
`nemara-caregiver-app-android`.

---

## Corrections to the 08-03 audit

Four findings were wrong or mis-stated. Recording them because a security
document that overstates is as harmful as one that misses — it spends
attention in the wrong place.

### C3 "unauthenticated gateway provisioning → account takeover" — REFUTED AS WRITTEN

The claimed mechanism was that an attacker-supplied `current_url` from the
anonymous `/v1/cameras/pi-announce` endpoint became a camera's video source,
giving feed injection and SSRF.

`pi_stream_url` is **dead code**. `src/db/camera.py:126` states it: the push
refactor (the Pi POSTs frames to `/v1/cameras/me/frames`) made the column
unused, and it is "NOT referenced by the relay path". The only `RelayConfig`
construction passes `pi_stream_url=""`. The backend never fetches that URL.
No SSRF, no pull-mode injection.

**But the endpoint was still dangerous, by a different mechanism.**
`consume_pending_ingest_token` keyed only on a caller-supplied `hardware_id`
and returned a **plaintext bearer token**. An attacker who guessed or observed
a Pi's MAC during provisioning could race the real device, take the ingest
credential, and push arbitrary frames as that camera. Feed injection via
*push*, not pull.

**Fixed by deletion** — see SEC-1 below.

### H1 "Insights tenancy is hand-written WHERE clauses across ~19 queries" — OVERSTATED

It is not 19 chances to forget. `nemara-web-app/lib/insight/scope.ts` is a
single chokepoint: `scopedPatients()` resolves the caregiver's patients from
the **backend** using their own token (so the backend stays the one authority),
and `scopeIds()` **throws** on a foreign id rather than returning empty.

All 12 exported query functions in `lib/insight/db.ts` were checked
individually: every one is patient-scoped and fully parameterised
(`= any($1::uuid[])`). No string-interpolated SQL.

The underlying concern is still real and still open — `nemara_readonly`
bypasses RLS, so the database is not a backstop and the 13th query that
forgets its `tenant` argument leaks silently. But the risk is "one future
mistake", not "19 existing ones".

### H3 "deleting a patient is not owner-guarded" — CONFIRMED (was unverified)

Real, and it read as safe. `patients.py` passes `caregiver_id=auth.id`, which
looks owner-scoped, but `patient_repo.delete_by_id` resolved it through
`accessible_patient_ids` — which matches **any** membership row. Any caregiver
the owner invited could delete the patient and cascade away every camera,
detector and event.

`src/repos/access.py` had promised the opposite in its own docstring:
"`role` is used only for owner-guarded actions (share/unshare, delete)".
Share and unshare did guard correctly. Delete was the single gap.

### M4 "password reset over a custom URL scheme" — LIVE, but mitigated

`project.yml` claimed the `nemara://` scheme was "reserved for future" and
"not handled in v0". Stale by months: `NemaraCaregiverApp.onOpenURL` routes
`nemara://reset-password?code=…` into `ResetPasswordView`. The comment made
the scheme read as inert during review, which is when it most needed to read
as live. Corrected in the file.

Severity is lower than the audit implied. Custom schemes are claimable by any
installed app, so interception is possible — but the exchange goes through
`client.auth.exchangeCodeForSession(authCode:)`, the PKCE path, which needs a
code verifier held only by the app that started the flow. An intercepted code
does not exchange. Universal Links remain the correct fix and are post-pilot
work (they need an `apple-app-site-association` file on the domain plus the
associated-domains entitlement).

---

## Confirmed live against production

Probed the deployed backend directly on 2026-08-05.

| Finding | Result |
|---|---|
| M2 — public API docs | **CONFIRMED.** `/openapi.json`, `/docs`, `/redoc` all returned HTTP 200 unauthenticated: 53 paths, 68 operations, with request/response schemas |
| M1 — rate limiting | **CONFIRMED ABSENT.** No limiter, no 429 path, nothing, in either backend or web app |
| Security headers | **CONFIRMED ABSENT.** No HSTS, CSP, X-Content-Type-Options, X-Frame-Options |
| TLS | **SOUND.** TLS 1.3, `TLS_AES_256_GCM_SHA384`, chain verifies |
| M6 — `internal_service_key` | **CONFIRMED ABSENT** from prod's 65 Railway variables. Fails closed, so safe rather than exposed, but the diagnostics layer is dead in production |
| H4 — prod linked to working dir | **CONFIRMED.** `supabase/.temp/linked-project.json` held the live project ref |

### New findings from this pass

- **`NEMARA_DEMO_PAIRING_CODE` is set in production.** A never-expiring,
  reusable six-digit code that pairs against one seeded row
  (`DEMO_CAMERA_IDENTITY = "cam_demo"`). Needed for Apple App Review, which
  cannot produce a live 10-minute pairing code. Blast radius is the demo row
  only — **not** a path into real patient data. Accepted as Low; the new rate
  limit closes the brute-force angle. Removing it risks failing App Review
  right before the pilot.
- **`NEMARA_DASHSCOPE_API_KEY` exists in production** and was missing from the
  key-rotation list. Rotation list is now six keys, not five.
- **`NEMARA_LIVEKIT_WEBHOOK_API_KEY` is absent in production** but present
  locally. Worth confirming prod verifies webhook signatures at all.
- **Railway ownership.** Production `memoria` sits in `potaton00b`'s workspace.
  Aryaman has full read/write (all 65 variables readable) but not ownership,
  so rotation was never blocked — only the ownership transfer is outstanding.
- **`nemara-caregiver-app-android` still has no git remote.** 79 commits on
  `phase-2-networking-auth`, existing on one laptop. Not a vulnerability; a
  total-loss risk.

---

## Fixed

### Backend (`feat/data-collection-timing-truth`)

| ID | Fix |
|---|---|
| SEC-1 | **Pi discovery deleted.** `src/api/pi_discovery.py` and its test removed, router unregistered. Deleted rather than env-gated: a gated route is one config mistake from live, and a missing module fails loudly instead of silently re-mounting. Regression guard asserts absence in all four envs |
| SEC-2 | **H3 closed.** New `owned_patient_ids()` filters `role='owner'`; `delete_by_id` uses it. A non-owner member now gets the same 404 as a stranger, which also avoids confirming the patient exists |
| SEC-3 | **Docs unmounted** outside local/test. Passing `None` removes the routes, so it cannot be defeated by requesting the path |
| SEC-4 | **`SecurityHeadersMiddleware`** — HSTS (HTTPS only, read via `X-Forwarded-Proto` since Railway terminates TLS), CSP `default-src 'none'`, nosniff, frame-deny, no-referrer, permissions. CSP skipped for `/docs` so Swagger still renders locally |
| SEC-5 | **Rate limiting** on pairing (10/min) and detector creation (30/min) |
| SEC-6 | **Prod unlinked** from the working directory |

`src/api/rate_limit.py` is built as a seam, not a helper: `RateLimitPolicy`
(what) / `RateLimitBackend` protocol (where) / `InMemorySlidingWindow` (the one
implementation) / `RateLimit` dependency (how). Any endpoint opts in with
`dependencies=[Depends(RateLimit(POLICY))]`. A Redis backend drops in without
touching a route.

Two non-obvious choices, both deliberate:
- **True sliding window, not a fixed bucket.** Fixed buckets allow `limit`
  requests at 0:59 and `limit` again at 1:01 — 2x the intended rate straight
  through the boundary, which is exactly what a brute-forcer wants.
- **Idle keys are swept.** Otherwise a caller rotating source addresses grows
  the store forever, turning the limiter into a memory-exhaustion vector.
- **`client_key` reads the RIGHT-most `X-Forwarded-For` entry.** The proxy
  appends what it actually saw; the usual left-most idiom would let a caller
  mint a fresh identity per request and never hit the limit.

1049 unit tests pass, including 14 new guards in
`test/unit/test_security_surface.py`.

### Web app (`fix/dashboard-local-dev`, commit `2e61235`)

- **Rate limiting** on the two paid-model routes (`briefing` 20/min, `chat`
  30/min) via `lib/rateLimit.ts`, mirroring the backend's shape.
- **`briefing` now checks `scopedPatients()` itself** rather than relying only
  on `proxy.ts`. It went straight from request body to a paid model call, so
  widening `PUBLIC_PATHS` or the proxy matcher would have silently turned it
  into an open LLM proxy billed to us.

Honest caveat recorded in the code: on Vercel each lambda instance holds its
own counters, so the ceiling is (instances x limit). That bounds spend without
being an exact quota; exactness needs shared state (Upstash / Vercel KV), which
the interface accommodates.

### iOS (`fix/door-picker-private-camera`, commit `dba5504`)

- **Analytics buffer encrypted at rest** — `.completeFileProtection` on the
  write. It previously inherited the iOS default (readable after first unlock)
  while carrying `caregiver_id` and a screen-by-screen trail of what that
  caregiver viewed and when.
- **Stale `project.yml` deep-link comment corrected** (see M4 above).

---

## Knowingly open

Ordered by what I would do next.

1. **Key rotation** — Gemini, OpenAI (x2), OpenRouter, Resend, Dashscope,
   Supabase service role + JWT. Deferred by choice to a call with Sunny. The
   exposure being closed is the git-history leak (`f1c0564`), which is
   independent of workspace ownership, so this does not need to wait for the
   Railway transfer.
2. **`apply_rls_sleep_activity_review_delta.sql` must ship in the same batch**
   as the sleep/review deltas. Non-negotiable before the prod migration.
   Production is safe today only because those deltas are still pending.
3. **Insights BYPASSRLS** (H1). App-layer scoping is sound; the database is
   still not a backstop. Fix is `set local role authenticated` + JWT claims per
   transaction, preserving the existing SQL.
4. **Platform ownership** — Railway workspace and 3 GitHub repos under
   `potaton00b`.
5. **Android git remote.** 79 commits on one laptop.
6. **Universal Links** to replace the custom URL scheme (M4).
7. **TLS pinning** (M3). Deliberately NOT done. Pinning against Supabase,
   LiveKit and Railway certificates — all of which rotate outside our
   control — risks bricking the app in the field for every pilot family. The
   threat it stops (active MITM with a CA-trusted certificate) is far less
   likely at this stage than the outage it can cause. Revisit with proper
   backup-pin and kill-switch machinery.
8. **`#if DEBUG` HTTP body logging** in `NemaraAPIClient`. Left as-is: it is
   compiled out of Release builds, and redacting it would degrade the only
   debugging path for decode failures. Accepted, documented.
9. **`cam_demo` isolation** not yet verified — needs a local Supabase running.
10. **Backend integration suite** not run this pass (needs `supabase start`);
    only the 1049 unit tests were exercised.

---

## Three loose ends worth knowing about

### The backend security fixes live in a commit labelled for annotations

SEC-1 through SEC-6 are in **`daef424 feat(annotations): sidecar spans`**, not
in a commit named for them. Two sessions were working this branch at once; the
other one ran `git commit` between this session staging its files and
committing them, so a shared index carried nine security files into its commit.

Nothing was lost and the tests pass, but two consequences:

- The work is unfindable by reading the log for anything security-shaped.
- That session had already produced one revert (`1c86034`). A revert of
  `daef424` would take the security fixes with it.

Left as-is deliberately: rewriting history while another session holds the
branch would corrupt its working state. Worth a follow-up commit or a note in
the PR description when the branch lands.

### `/v1/cameras/me/frames` is orphaned

Deleting the Pi announce route removed the only caller of
`consume_pending_ingest_token`. `pi-provision` can still *create* a pending
ingest token, but nothing can retrieve it, so no valid bearer for the push
endpoint can ever be obtained.

This is safe — an unreachable endpoint is strictly less attack surface than a
reachable one — but it is dead weight, and dead weight that looks live is how
`pi_stream_url` ended up being flagged as a Critical that did not exist. Worth
removing as a clean-up, along with the `pi_discoveries` table and the
`pi_stream_url` / `pi_audio_url` columns.

### `seed_demo_camera.py` now refuses non-local databases without `--remote`

Added after running it against the local dev database during this pass left a
`cam_demo` row behind that broke three tests. The failure was trivial to undo;
the shape of it was not trivial — the script writes wherever `NEMARA_DB_URL`
points, with no signal about where that is. Exactly the
`supabase/.temp/project-ref` problem from H4: a routine command that silently
targets whichever database the environment happens to name.

It now prints the host and requires `--remote` for anything that is not
localhost. Re-seeding production during App Review therefore needs one extra
flag, which is the point.

## Note on the 08-03 meta-finding

That audit's central point — 24 reports, 18 open PRs, nothing ever fixed —
is addressed here for the first time: this pass fixed nine findings across
three repos rather than producing a tenth report. `nemara-web-app` has now
been audited. The Android repo still has no remote and remains unreviewable
in any durable way.
