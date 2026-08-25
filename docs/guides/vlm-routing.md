# How to switch a patient/account between Qwen and Gemini

_A practical guide. Written for someone (or their Claude Code) who has access to
the production Supabase project and the Railway backend, and wants to control
which vision model (VLM) analyzes a given family's camera feed._

---

## 1. The mental model (read this first)

Every detection "tick" (a ~10s clip from a camera) is sent to **one vision model**
to be analyzed. There are four possible providers:

| Code value     | Model that actually runs                       | Provider / how it's billed |
|----------------|------------------------------------------------|----------------------------|
| `gemini`       | Google Gemini **model ladder** (native video): primary **`gemini-3.6-flash`**, with an in-provider retry on the fallback tier (`NEMARA_GEMINI_FALLBACK_MODEL` — see the ⚠️ in §8) | Google API key |
| `openai`       | **GPT-5.4** by default (`NEMARA_OPENAI_MODEL`; dev `.env` currently pins `gpt-5.4-mini`) — frame sampling | OpenAI API key |
| `openrouter`   | **Qwen3-VL-32B-Instruct** (frame sampling)     | OpenRouter API key → "Qwen" in the UI |
| `dashscope`    | **Qwen3-VL-32B-Instruct** via Alibaba DashScope (**native video + fps** — sandbox arm) | DashScope API key |

> ⚠️ **`dashscope` persistence gap:** `analyses.provider` has a CHECK constraint
> that currently allows only `('gemini','openai','openrouter','skipped')` —
> `'dashscope'` is **not** in it (`supabase/schema.sql`, and no delta script adds
> it). A tick that actually runs on DashScope will analyze fine but **fail to
> save its row**, exactly the failure mode §8 warns about for `openrouter`.
> Widen the constraint before routing anyone to `dashscope`.

> **`openrouter` === Qwen.** In the web dashboard the badge says **"Qwen"**, but in
> the database / code / env the value is the literal string **`openrouter`**. Don't
> type "qwen" anywhere — it won't match and will silently fall back to the default.

### Which model a tick uses is decided in two layers

1. **Per-account override** — the `caregivers.vlm_provider` column.
   - If it's set to `gemini` / `openai` / `openrouter` / `dashscope`, **that**
     provider is used for **every camera owned by that caregiver**.
   - If it's `NULL` (the default for everyone), the account falls through to…
2. **Global default** — the backend env var `NEMARA_DEFAULT_VLM_PROVIDER`
   (currently **`gemini`**). Used by any account whose column is `NULL`.

So the resolution is literally: **`caregivers.vlm_provider` → else → global default → else → `gemini`.**
(The final `gemini` is a hard safety net if someone sets a garbage value.)

### ⚠️ "Per patient" is really "per caregiver account"

There is **no per-patient or per-camera** setting. The override lives on the
**caregiver** (the login/account). A caregiver owns one or more patients, each with
one or more cameras — **all of them** use the caregiver's `vlm_provider`. If you
need two patients on different models, they'd have to be under different caregiver
accounts. (This is by design — billing/routing is per-account.)

---

## 2. Prerequisite (already done in this prod, but verify for any new deploy)

For `openrouter` (Qwen) to work at all, the backend must have an OpenRouter API key:

```
NEMARA_OPENROUTER_API_KEY = sk-or-v1-…   # set on Railway
```

Optional (both have sane defaults, don't set unless you're changing models):
```
NEMARA_OPENROUTER_MODEL    = qwen/qwen3-vl-32b-instruct   # the default
NEMARA_OPENROUTER_BASE_URL = https://openrouter.ai/api/v1 # the default
```

Check it's present:
```bash
railway variables | grep -i openrouter      # run from the linked backend repo
```

If the key is missing and you route an account to `openrouter`, the tick will
**fall back** through dashscope→gemini automatically (see §6), so you won't get
errors — you just won't get Qwen. So always confirm the key first.

---

## 3. Toggle ONE account to Qwen (the common case)

This is a single SQL `UPDATE` on the `caregivers` table. It is **hot** — no deploy,
no restart. It takes effect on the **next tick within ~60 seconds** (there's a 60s
in-memory cache per caregiver; see §5).

### The statements

```sql
-- Put this family on QWEN:
update public.caregivers set vlm_provider = 'openrouter' where email = 'friend@example.com';

-- Put them back on GEMINI explicitly:
update public.caregivers set vlm_provider = 'gemini'     where email = 'friend@example.com';

-- Or hand them back to the global default (whatever NEMARA_DEFAULT_VLM_PROVIDER is):
update public.caregivers set vlm_provider = null         where email = 'friend@example.com';
```

That's the entire toggle. `openrouter` = Qwen, `gemini` = Gemini, `null` = default.

### Where to run the SQL — pick one

**Option A — Supabase Dashboard (easiest, no secrets):**
1. Go to `https://supabase.com/dashboard/project/<PROJECT_REF>/sql/new`
   (this prod's ref is **`zgifmvjnggmgzvtnosnu`**).
2. Paste the `update …` statement, hit **Run**.

**Option B — Supabase Management API (good for Claude Code / scripts):**
You need a Management API token (a `sbp_…` personal access token from
https://supabase.com/dashboard/account/tokens). Then:
```bash
curl -s -X POST \
  "https://api.supabase.com/v1/projects/zgifmvjnggmgzvtnosnu/database/query" \
  -H "Authorization: Bearer $SUPABASE_PAT" \
  -H "Content-Type: application/json" \
  -H "User-Agent: curl/8.4.0" \
  -d '{"query": "update public.caregivers set vlm_provider = '\''openrouter'\'' where email = '\''friend@example.com'\'';"}'
```
> **Gotcha:** the Management API is behind Cloudflare, which **403s the default
> Python/urllib User-Agent**. Always send a `User-Agent` header (curl sets one
> automatically; if you script it in Python, set it manually, e.g. `curl/8.4.0`).

**Option C — direct Postgres** (if you have the DB connection string, e.g. from
`railway variables | grep NEMARA_DB_URL`): `psql "<conn>" -c "update …"`.

### See the current state of everyone
```sql
select email, coalesce(vlm_provider, '(default → gemini)') as provider
from public.caregivers
order by email;
```

---

## 4. Flip the GLOBAL DEFAULT to Qwen (everyone at once)

If you want **every account that hasn't been explicitly overridden** to use Qwen,
change the global default instead of touching rows one by one. This is an **env var**,
so unlike §3 it **requires a redeploy/restart** (the backend reads settings at boot).

```bash
# from the linked backend repo:
railway variables --set "NEMARA_DEFAULT_VLM_PROVIDER=openrouter"
# then restart so the new value is read:
railway redeploy        # or: railway up   (redeploys current main)
```

Now the logic becomes: any caregiver with `vlm_provider = NULL` → **Qwen**.
Accounts with an explicit `gemini`/`openai` in their column still win — the column
always overrides the default.

To go back: `railway variables --set "NEMARA_DEFAULT_VLM_PROVIDER=gemini"` + redeploy.

> Valid values for this env var are exactly `gemini` | `openai` | `openrouter` |
> `dashscope`. Anything else is treated as invalid and the code falls back to
> `gemini`. (`dashscope` is a sandbox arm — see the ⚠️ in §1 before using it.)

### Per-account vs. global — which should you use?

| You want… | Do this | Restart needed? |
|---|---|---|
| One family on Qwen, rest on Gemini | §3 — set that one row to `openrouter` | No (≤60s) |
| Everyone on Qwen by default | §4 — set env default to `openrouter` | Yes (redeploy) |
| Everyone on Qwen EXCEPT a few | §4 default `openrouter`, then §3 set the exceptions to `gemini` | Yes for the default change |

---

## 5. How fast does a change take effect?

- **Per-account column (§3):** ≤ **60 seconds**. The backend caches each caregiver's
  choice for 60s (`_TTL_SECONDS` in `vlm_routing.py`) to avoid a DB hit on every
  tick. So after your `UPDATE`, wait up to a minute and the next clip uses the new
  model. No restart.
- **Global default env var (§4):** only after the backend **restarts** (env is read
  once at startup). `railway redeploy` / `railway up` does that.

---

## 6. What happens if a model fails (so you're not surprised)

Each choice has a **fallback chain** — if the chosen provider errors (outage, rate
limit, bad key), the tick automatically tries the next one so detection never goes
dark. The chains as of 2026-07-27 (`_CHAINS` in `vlm_routing.py`):

```
gemini      →  gemini-fallback-tier  →  dashscope  →  openrouter
openrouter  →  dashscope  →  gemini
openai      →  openrouter
dashscope   →  openrouter
```

Notes:
- The **gemini fallback tier** is a second Gemini model on the same native-video
  path (`NEMARA_GEMINI_FALLBACK_MODEL`) — it is chain-internal, not a routable
  choice, and its analyses rows still say `provider = 'gemini'`.
- A keyless provider in the chain (e.g. DashScope with no
  `NEMARA_DASHSCOPE_API_KEY`) raises immediately and the chain moves on, so
  keyless environments fail through cleanly.

Practical consequence: if you set someone to `openrouter` but the OpenRouter key is
missing/invalid, their ticks quietly run on **gemini** instead (via the dashscope
hop, which is skipped when keyless). So if you toggle to Qwen and the dashboard
still says "Gemini," check the OpenRouter key (§2) and the backend logs.

---

## 7. How to verify it actually switched

Three ways, in order of convenience:

1. **Web dashboard** — open the patient's **Live analyses** feed. Each row has a
   provider badge: **Gemini / OpenAI / Qwen**. New rows should say **Qwen**.
   (Requires `SHOW_LIVE_ANALYSES=true` in the web build — it is, in this prod.)

2. **Database** — every analysis row records which model ran:
   ```sql
   select a.provider, a.captured_at, left(a.description, 50) as preview
   from public.analyses a
   join public.cameras  c  on c.id = a.camera_id
   join public.patients p  on p.id = c.patient_id
   join public.caregivers cg on cg.id = p.caregiver_id
   where cg.email = 'friend@example.com'
   order by a.captured_at desc
   limit 5;
   ```
   After the switch, the newest rows' `provider` should be `openrouter`.

3. **Backend logs** — `railway logs` shows lines like
   `worker cam=… DONE provider=openrouter …` once Qwen is running.

---

## 8. Reference — where this lives in the code

- `src/services/detection/vlm_routing.py`
  - `resolve_vlm_provider()` — reads `caregivers.vlm_provider`, falls back to
    `settings.default_vlm_provider`, then to `gemini`. 60s cache.
  - `run_vlm()` — runs the chosen provider with the fallback chains above
    (`_CHAINS`).
  - `_VALID = ("gemini", "openai", "openrouter", "dashscope")` — the only
    accepted values.
- `src/services/detection/providers/` — `gemini.py` (native video, model
  ladder), `openai.py`, `openrouter.py`, `dashscope.py`.
- `src/settings.py` — `default_vlm_provider` (env `NEMARA_DEFAULT_VLM_PROVIDER`),
  `gemini_model` / `gemini_fallback_model`,
  `openrouter_api_key` / `openrouter_model` / `openrouter_base_url`,
  `dashscope_api_key` / `dashscope_model` / `dashscope_base_url`.

  > ⚠️ **`gemini_fallback_model` default is currently ambiguous** (as of commit
  > `cd32111`, 2026-07-27): `src/settings.py` defines the field **twice** —
  > once with default `gemini-3.1-flash-lite` (the intended fallback per the
  > commit) and, a few lines below, a leftover older definition with default
  > `gemini-3.5-flash`. In Python the **later definition wins**, so the
  > effective in-code default is `gemini-3.5-flash` until the duplicate is
  > removed. If the exact fallback model matters to you, set
  > `NEMARA_GEMINI_FALLBACK_MODEL` explicitly in the environment.
- `src/db/caregiver.py` — the `vlm_provider` column on the ORM.
- Schema: `caregivers.vlm_provider text` (nullable) and the
  `analyses_provider_check` constraint must include `'openrouter'`
  (see `scripts/deltas/apply_vlm_provider_delta.sql` if migrating a fresh DB —
  **without that constraint value, Qwen ticks run fine but fail to save**).
  The constraint does **not** yet include `'dashscope'` — see the ⚠️ in §1.

---

## 9. Copy-paste cheat sheet

```sql
-- ONE account → Qwen
update public.caregivers set vlm_provider='openrouter' where email='X';
-- ONE account → Gemini
update public.caregivers set vlm_provider='gemini'     where email='X';
-- ONE account → use global default
update public.caregivers set vlm_provider=null         where email='X';
-- See everyone
select email, coalesce(vlm_provider,'(default)') from public.caregivers order by email;
```
```bash
# EVERYONE → Qwen by default (then redeploy)
railway variables --set "NEMARA_DEFAULT_VLM_PROVIDER=openrouter" && railway redeploy
# EVERYONE → Gemini by default (then redeploy)
railway variables --set "NEMARA_DEFAULT_VLM_PROVIDER=gemini" && railway redeploy
```

**Remember:** `openrouter` = Qwen everywhere in code/DB/env. The word "qwen" only
appears in the UI badge and the model id — never as a routing value.
```
