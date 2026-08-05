# Memoria — Cross-System Security Audit, 2026-08-03

> **SUPERSEDED IN PART — see `SECURITY-AUDIT-2026-08-05-REMEDIATION.md`.**
> That pass re-tested this document against production and found four items
> here wrong or mis-stated: **C3 is refuted as written** (the mechanism named
> does not exist — though the endpoint was dangerous for a different reason and
> is now deleted), **H1 is overstated**, **H3 is confirmed** (it was carried
> forward unverified), and **M4 is live** despite a source comment claiming
> otherwise. Nine findings are now fixed. Read the newer document first; this
> one is kept unedited as the point-in-time record.

**Scope:** `nemara-backend`, `nemara-caregiver-app`, `nemara-web-app`,
`nemara-caregiver-app-android`, and the seams between them (Supabase, LiveKit,
APNs, Resend, Railway, Vercel). `nemara-gateway` excluded by request — it was
remediated separately today.

**Method:** `vibe-security` framework, but **live** rather than static — this
audit queried the running database, its grants, and the deployed API. That
distinction produced the Critical below, which none of the 24 existing static
reports could have found.

**Status of the existing programme:** 24 audit reports across 18 open PRs
(backend 3, caregiver-app 15). None merged. None of the findings fixed.

---

## The meta-finding: the audit programme is not reducing risk

This matters more than any single item below.

- **`nemara-web-app` has never been audited.** Zero audit branches, ever. It is
  the repo that leaked live production credentials to `origin/main` and that
  holds the BYPASSRLS database credential. The one system never looked at is
  the one with the worst findings.
- **`nemara-caregiver-app-android` has no git remote at all.** Never pushed. It
  exists only on one laptop — unaudited, unbacked-up, and unreviewable.
- **Nothing is ever fixed.** In `nemara-caregiver-app`, the same four issues
  recur across audits on 06-04, 06-19, 07-10 and 07-31, reworded each time: no
  TLS pinning, password reset over a custom URL scheme, unverifiable
  server-side authz, unencrypted analytics buffer. Fifteen open PRs, four
  months, zero remediation.
- **Static audits are blind to the class of bug that actually bit.** Every
  report states "no live systems, databases, or secrets were accessed."
  Today's Critical is a *grant* + *RLS flag* interaction that exists only at
  runtime. No amount of source reading finds it.

**Recommendation:** stop generating new reports until the backlog is triaged.
Add a live-database check to the routine. Put `nemara-web-app` in scope and
give the Android repo a remote.

---

## Critical

### C1 — Patient health tables world-writable via the public app key

**FOUND AND FIXED TODAY** — `scripts/apply_rls_sleep_activity_review_delta.sql`,
commit `cda6364`.

`apply_sleep_activity_tables_delta.sql` and `apply_review_gate_delta.sql` create
their tables with no `enable row level security`. Supabase grants `anon` and
`authenticated` full `SELECT, INSERT, UPDATE, DELETE, TRUNCATE` on everything in
`public`, and PostgREST exposes it. With RLS off, those grants are the only
control.

`anon` is the **publishable** key. It ships inside the iOS binary and the web
bundle; it is public by design.

Verified on the live DB before the fix — `relrowsecurity = false`, zero
policies, `anon` holding all seven privileges on:

| Table | Contents |
|---|---|
| `sleep_nights` | per-patient sleep records (health data) |
| `activity_days` | per-patient activity records (health data) |
| `review_devices` | APNs push tokens + user ids |

**Attacker impact:** extract the key from the app bundle, then read every
patient's sleep and activity history, forge records, or `TRUNCATE` the tables.
No account required.

**Why production was spared:** both source deltas are still in the 23 pending
for prod. **The fix delta must be applied in the same batch — never after.**

### C2 — Live production secrets in git history, three separate times

Three distinct leaks, three different shapes, each with a written justification
that made it look deliberate:

1. `nemara-backend/.env.bak.1783536884` — swept in by `git add -A` (2026-07-19).
2. `nemara-web-app/.env.local` — `.gitignore` carried an explicit
   `!.env.local`, commented *"small private repo … useful for new developers"*
   (2026-07-28). Live `DATABASE_URL`, OpenAI and Google keys reached
   `origin/main` and six branches.
3. `nemara-gateway/GATEWAY_ACCESS.md` — a Markdown file, invisible to every
   `.gitignore` rule, stating *"credentials are committed on purpose (project
   policy)"*. Held the SSH/sudo password, the camera password, a home Wi-Fi
   password, and the **production** LiveKit key and secret.

**Remediated today:** all three untracked or scrubbed; gateway history rewritten
and force-pushed; `.gitignore` rules added to five repos; a credential-hygiene
rule added to the workspace `CLAUDE.md`.

**Still outstanding:** rotation of Gemini, OpenAI (×2, backend and web),
OpenRouter, and Resend. LiveKit, APNs and `DATABASE_URL` are done.
`refs/pull/1/head` in `nemara-gateway` still carries the old values and cannot
be deleted client-side — only GitHub Support or repo recreation removes it.

### C3 — Unauthenticated gateway provisioning → account takeover

Carried forward from the 2026-08-01 backend audit, **not independently
re-verified today**. Flagged as leading to account takeover and live-feed
injection. Given it concerns the same provisioning path reworked today, it
should be re-tested against current `main` before the pilot.

---

## High

### H1 — Insights holds a BYPASSRLS credential; tenancy is hand-written SQL

`nemara-web-app/lib/insight/db.ts` connects to production Postgres as
`nemara_readonly`, which **bypasses every RLS policy**. Tenant isolation exists
only in hand-written `WHERE` clauses across ~19 analytical queries. One missing
filter is a full cross-tenant leak of patient data, and nothing downstream would
catch it. The file's own header concedes: *"In a production build these become
caregiver-scoped queries behind RLS."*

**Fix (recommended, cheapest correct option):** keep `pg`, stop bypassing RLS —
wrap each query in a transaction that assumes the caregiver's identity:

```sql
set local role authenticated;
set local request.jwt.claims = '{"sub":"<caregiver-uuid>"}';
```

All existing SQL is preserved and the database enforces tenancy underneath it.
PostgREST is not a viable alternative here — the queries use `string_agg`,
`array_agg(... order by ...)`, `date_trunc` and `count(*) filter (…)`, which it
cannot express.

### H2 — Platform ownership sits with a departed employee

Not a code defect, but the highest-consequence item on this list.

| Platform | State |
|---|---|
| GitHub `nemara-caregiver-web`, `NemaraCam`, `nemara-gateway` | owned by `potaton00b`; Aryaman has `pull, push, triage` — **not admin** |
| Vercel | transfer in progress |
| Supabase | **resolved today** — project moved to the `memoria` org |
| Railway | ownership unconfirmed |

Without admin, the repos cannot be transferred, reconfigured, or protected from
deletion. Full mirrors were taken today as insurance, but they do not preserve
issues, PRs, or the canonical URLs deploys point at.

### H3 — Deleting a patient is not owner-guarded

From the 2026-08-01 backend audit: any shared member can destroy all data for a
patient. Not re-verified today. Given `patient_members` exists and sharing is
live, this deserves a test before the pilot.

### H4 — Production database linked to the local working directory

`nemara-backend/supabase/.temp/project-ref` contains the production project ref.
`supabase db reset` or `db push` run from that directory targets **production**,
while local development uses a separate `supabase start` stack. A single
muscle-memory command destroys prod. Sharpened by agents running commands in
this repo.

**Fix:** `rm supabase/.temp/project-ref` when not actively linked. It is
gitignored, so it was never shared.

---

## Medium

### M1 — No rate limiting anywhere

Confirmed by search: no limiter, no 429 path, nothing. Two concrete
consequences:

- **Pairing-code brute force.** Codes are six digits (100000–999999). With no
  limit, the space is exhaustible.
- **Unbounded AI spend.** Every detection tick is a paid VLM call and detectors
  are user-creatable. There is no per-account cap, so a compromised or hostile
  account can run up the Gemini bill without limit.

### M2 — OpenAPI docs public in every environment

`create_app()` sets no `docs_url=None` / `redoc_url=None` / `openapi_url=None`.
**Demonstrated during this audit:** production `/openapi.json` was fetched with
no credentials, yielding the complete API surface including request and response
schemas. That is reconnaissance handed to an attacker for free.

**Fix:** disable outside local/test, mirroring the `internal_service_key`
pattern already used for `/health/freshness`.

### M3 — No TLS pinning on a PHI-bearing iOS app

Flagged in four consecutive caregiver-app audits (06-04 → 07-31), never
actioned. Connections to the backend, Supabase and LiveKit all carry patient
data.

### M4 — Password reset over a custom URL scheme

`nemara://reset-password?code=…`. Custom schemes are claimable by any installed
app; Universal Links are not. Also flagged four times, never actioned.

### M5 — `/cameras/discovered` has no per-caregiver scoping

Every caregiver sees all unclaimed Pi devices. Documented in-source as
acceptable at 10–20 devices; revisit before scaling.

### M6 — `internal_service_key` unset in production

Fails **closed**, so this is safe rather than exposed — but `/health/freshness`
and every `/internal` route returns 403 in prod, meaning the diagnostics layer
is dead exactly when it is needed.

---

## Low

- `#if DEBUG` builds log full HTTP response bodies (potential PHI) and transmit
  auth tokens in cleartext over the LAN.
- Analytics buffer persisted to disk without an explicit Data Protection class.
- No jailbreak / runtime-integrity signal on an in-home camera controller.
- Missing HTTP security headers (HSTS, CSP, X-Content-Type-Options).
- Reusable, never-expiring demo pairing code.

---

## Checked and found sound

- **RLS policy quality** — 12 policies, all tenant-scoped via
  `has_patient_access(patient_id)`. No `using (true)`, no missing predicates.
- **Storage buckets** — `incident-clips` and `patient-photos` both private.
- **Auth coverage** — 74 routes; only `/health` and `/cameras/pi-announce` are
  unauthenticated, both deliberately and documented.
- **Client secret hygiene (web)** — no `service_role` reachable from client
  code; every `NEXT_PUBLIC_*` var is legitimately public.
- **iOS credential storage** — Keychain, not `UserDefaults`; no ATS exceptions.
- **Prompt injection surface** — `custom_prompt` bounded to 2000 chars and
  compiled rather than interpolated raw.
- **SQL injection** — Insights queries are parameterised (`$1::uuid[]`).

---

## Prioritised remediation

1. **Apply `apply_rls_sleep_activity_review_delta.sql` in the same batch as the
   sleep/review deltas.** Non-negotiable before the prod migration. (C1)
2. **Secure repo ownership from `potaton00b`** while goodwill holds. (H2)
3. **Finish rotation:** Gemini, OpenAI ×2, OpenRouter, Resend. (C2)
4. **Unlink prod from the working directory.** One command. (H4)
5. **Disable public API docs outside local/test.** Small, high value. (M2)
6. **Add rate limiting** to pairing and detector-creation paths. (M1)
7. **Fix Insights tenancy** via `set local role` + JWT claims. (H1)
8. **Re-verify C3 and H3** against current `main` — both are carried forward
   unverified.
9. **Triage the 18 open audit PRs**, then put web-app in scope and give Android
   a remote. (meta-finding)
