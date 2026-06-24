# Prod Migration Runbook — `feature/camera-unpair-state` → production

_Generated 2026-06-23. Verified against the live prod Supabase (`zgifmvjnggmgzvtnosnu`)
and the live Railway project (`nemara-backend` / `production`) — not from memory._

---

## TL;DR — what's actually new on prod (outside the repos)

| Target | Change | Required? | Destructive? |
|---|---|---|---|
| **Supabase prod (DB)** | `apply_vlm_provider_delta.sql` (add `caregivers.vlm_provider` col + widen `analyses_provider_check` to allow `openrouter`) | **YES** | No — purely additive |
| **Railway (backend env)** | Add `NEMARA_OPENROUTER_API_KEY` | **YES** (for Qwen) | No |
| Railway (backend env) | `NEMARA_OPENROUTER_MODEL`, `NEMARA_OPENROUTER_BASE_URL`, `NEMARA_DEFAULT_VLM_PROVIDER` | No — safe defaults in code | No |
| **Vercel (web)** | `SHOW_LIVE_ANALYSES = true` (code flag, in the commit) — **product decision** | Decision | No |
| Vercel (web env) | — none new — | n/a | n/a |

**Already on prod, NO action needed** (probed read-only and confirmed present):
`cameras.app_present` ✅, `caregiver_devices` table ✅, `ui_events` table ✅,
DB pool size already `20 + 20 = 40` ✅ (the incident mitigation persisted).

---

## ⚠️ The one ordering rule that matters

**Run the DB migration BEFORE deploying the backend.**

The new backend's ORM (`src/db/caregiver.py`) maps `vlm_provider`, so every
`SELECT ... FROM caregivers` will request that column. If the column doesn't
exist yet, **every caregiver query 500s** (same failure mode as the
`app_present` lesson). So: **DB first, then `railway up`.**

Everything here is non-destructive, but get the order right:
**① DB migrate → ② add Railway env var → ③ merge to main → ④ `railway up` → ⑤ Vercel web → ⑥ iOS builds.**

---

## This session's changes (5 backend + 3 client commits, all local-only)

### nemara-backend (`feature/camera-unpair-state`)
- `2421009` camera soft-unpair endpoints + per-device push de-register
- `feacd0a` OpenRouter (Qwen3-VL) provider + per-account VLM routing
- `4def68c` allow `openrouter` in `analyses.provider` CHECK
- `67f5bdb` re-paired camera starts DISCONNECTED (UI shows "Reconnect")
- `9ddec22` keep LiveKit identity/room on unpair (fixes 403 on first Start)

New files: `providers/openrouter.py`, `vlm_routing.py`, `apply_vlm_provider_delta.sql`.
New settings: `openrouter_api_key/model/base_url`, `default_vlm_provider` (default `gemini`).

### nemara-caregiver-app — Memoria Care (`feature/camera-unpair-state`)
- `3cda71e` move detector "Add" buttons above the list
- `f293ab7` camera unpair action + re-bind APNs token on sign-in/out
- `7b91d0f` UNPAIRED camera status (status line + chip)

### nemara-caregiver-web — dashboard (`feature/camera-unpair-state`)
- `3e9d89f` per-camera VLM badge (Gemini / OpenAI / **Qwen**); flips `SHOW_LIVE_ANALYSES=true`
- ⚠️ `.env.local` in this commit points at **localhost** (dev). It does NOT affect
  Vercel prod (Vercel uses dashboard env vars), but consider not merging that file.

### NemaraCam — Memoria Sensor (`feature/camera-unpair-state`)
- `1af5efc` "Unpair this phone" now soft-unpairs via backend (real-time) instead of no-op

---

## Step-by-step

### ① Supabase prod DB migration (do this FIRST)
Non-destructive, idempotent. Current prod constraint is
`provider IN ('gemini','openai','skipped')`; prod has **21,626** analyses rows
(all already valid for the widened set, so re-validation is trivial).

Run `scripts/apply_vlm_provider_delta.sql` against prod via the Supabase SQL editor
(project `zgifmvjnggmgzvtnosnu`) or `psql "<prod url>" -f`. It does:
```sql
alter table public.caregivers add column if not exists vlm_provider text;
alter table public.analyses drop constraint if exists analyses_provider_check;
alter table public.analyses add constraint analyses_provider_check
  check (provider in ('gemini','openai','openrouter','skipped'));
```
**Lock note:** the `add constraint` briefly takes an ACCESS EXCLUSIVE lock while it
scans 21k rows (sub-second). If you want zero stall, split into
`... add constraint ... NOT VALID;` then `validate constraint ...;` (validate
takes only a SHARE UPDATE lock). For 21k rows it's not worth it.

Verify after: `select pg_get_constraintdef(oid) from pg_constraint where conname='analyses_provider_check';`

### ② Railway env var
Add (value is in local `nemara-backend/.env` as `NEMARA_OPENROUTER_API_KEY=sk-or-v1…`):
```
NEMARA_OPENROUTER_API_KEY = <copy from local .env>
```
Leave `NEMARA_DEFAULT_VLM_PROVIDER` UNSET → defaults to `gemini`, so **no existing
prod account changes behavior**. Model/base_url also default correctly.

### ③ + ④ Merge backend to main & deploy
After ① is confirmed:
```
git checkout main && git merge --no-ff feature/camera-unpair-state   # backend
railway up
```
On boot, nothing routes to OpenRouter yet (default = gemini, all caregivers
`vlm_provider = NULL`), so the new code is inert until you opt an account in.

### ⑤ Vercel web
Merge the web branch and deploy (`vercel --prod --yes`). **Decision:** the commit
flips `SHOW_LIVE_ANALYSES = true`, which makes the Live analyses feed visible in
prod (it's currently hidden). Confirm you want that on. No new Vercel env vars.

### ⑥ iOS (Memoria Care + Memoria Sensor)
These are **client builds, not server deploys** — merging to main does NOT push
anything to devices. They reach users only via a new TestFlight/App Store build.
The unpair/push features depend on the new backend, so ship iOS **after** ④.

---

## Optional — actually turn Qwen on for a prod account
After ①+④, route one caregiver (e.g. the App-Review demo `jane@patel.co`) to Qwen.
One-row, non-destructive, takes effect within ~60s (resolver cache TTL):
```sql
update public.caregivers set vlm_provider = 'openrouter' where email = 'jane@patel.co';
```
Revert any time with `set vlm_provider = NULL` (→ falls back to gemini).

---

## Safety summary
- **No destructive ops.** One added nullable column, one widened CHECK constraint,
  one new env var, one feature-flag flip. No drops, no data rewrites, no resets.
- **Never reset prod** (standing rule). Apply deltas only.
- **DB-before-deploy** is the only hard ordering constraint.
- Rollback: backend = redeploy previous main; web = Vercel instant rollback;
  DB delta is forward-only but harmless to leave (column nullable, constraint wider).
