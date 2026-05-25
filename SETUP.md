# SETUP.md — Hosted production stack

This doc describes the **deployed** Nemara stack: where each piece lives, how to access it, and how to operate it. The local-dev path is documented in [README.md](./README.md); this doc is for the cloud version we're running for the demo.

> **TL;DR for a teammate cloning this repo today**: there is nothing to install or seed. The whole stack is already live in the cloud. You need:
> 1. dashboard access on the four platforms (one-time invites, listed below),
> 2. the credentials in `.env` — already committed in this repo,
> 3. the CLI tools listed in "[Day-to-day operating tools](#day-to-day-operating-tools)" if you want to deploy / read logs from a terminal.
>
> The web dashboard, both iOS apps, and the backend talk to the cloud stack out of the box. No `make fresh-seed` step needed.

---

## Topology

```
                                       ┌────────────────────────┐
                                       │ web dashboard          │
                                       │ Vercel                 │
                                       │ nemara-caregiver-web   │
                                       └──────────┬─────────────┘
                                                  │
                                                  │ HTTPS + Realtime WS
                                                  ▼
   ┌──────────────────┐       ┌────────────────────────────────┐       ┌──────────────────┐
   │ NemaraCam        │──────▶│ nemara-backend (FastAPI)       │──────▶│ Supabase Cloud    │
   │ wall phones      │ HTTPS │ Railway                        │  pg   │ Postgres + Auth  │
   │ (publishers)     │       │ project f2f8606e-9ecd…         │       │ + Realtime + S3  │
   └────────┬─────────┘       │ container: ffmpeg + uv         │       │ zgifmvjnggm…     │
            │                 └──────────────┬─────────────────┘       └──────────────────┘
            │                                │
            │  publishes audio/video         │ webhook (participant_joined / _left)
            ▼                                ▼
   ┌────────────────────────────────────────────────────────┐
   │ LiveKit Cloud                                          │
   │ wss://nemara-pdx0uelu.livekit.cloud                    │
   └───────────────────┬────────────────────────────────────┘
                       │ tracks (audio + video + data packets)
                       ▼
            ┌──────────────────────┐
            │ NemaraCaregiver iOS  │
            │ (subscriber)         │
            └──────────────────────┘
```

The backend is the only component running our code; everything else is a managed service.

## What lives where

| Component | Service | Identifier | Public URL |
|-----------|---------|------------|------------|
| Backend (this repo) | Railway | project `f2f8606e-9ecd-448b-b127-9cafb8926b5f`, service `fb2955d5-…` | `https://nemara-backend-production.up.railway.app` |
| Web dashboard | Vercel | project `prj_FUTkPl0C5Yc77EnAEtyZ5xYjzGfA`, team `sunnys-projects-8be52dc6` | `https://nemara-caregiver-web.vercel.app` |
| Database / Auth / Realtime / Storage | Supabase Cloud | project ref `zgifmvjnggmgzvtnosnu`, org `gtpeceoncuvbtimhkvnc` | `https://zgifmvjnggmgzvtnosnu.supabase.co` |
| Video / audio / data packets | LiveKit Cloud | project `nemara-pdx0uelu` (Aryaman's account) | `wss://nemara-pdx0uelu.livekit.cloud` |
| LLM (default) | Google Gemini API | model `gemini-2.5-flash` | (key in `.env`) |
| LLM (fallback) | OpenAI | model in `.env` | (key in `.env`) |
| Push notifications | Apple APNs | Team `V8N59CF74G`, bundle `co.elephantlabs.nemara.caregiver` | (key in `.env`) |

All four hosted services are on **free tiers** at the moment. Limits worth knowing:
- Supabase free: max 2 active projects per personal org. Pausing/deleting older projects frees the slot.
- Vercel Hobby: no team members — see [collaboration](#getting-a-teammate-access).
- Railway: no free-tier project sharing limits, but resource usage is metered.
- LiveKit free: 100 monthly meeting hours, 50 concurrent.

## Demo login

The web dashboard and caregiver iOS app accept this user (seeded once into prod Supabase):

```
email:    jane@patel.co
password: Mar9aret!2024
```

She owns the patient "Maggie O'Brien" with four preset detectors (Falls, Agitation, Wandering, Medication adherence). Cameras need to be paired in-app — there are no seeded cameras in the prod DB.

## Getting a teammate access

None of the platforms expose member invites via CLI / API. All must be done in dashboards:

1. **Supabase**: https://supabase.com/dashboard/org/gtpeceoncuvbtimhkvnc/team → Invite → email → role Developer (or Owner if they'll need to delete projects).
2. **Railway**: https://railway.com/project/f2f8606e-9ecd-448b-b127-9cafb8926b5f/settings/members → invite by email.
3. **Vercel**: Hobby plan blocks sharing entirely. Either upgrade to Pro Team ($20/mo) or transfer the `nemara-caregiver-web` project to a team they already own.
4. **LiveKit**: https://cloud.livekit.io/projects/nemara-pdx0uelu/team → invite. (Note: the LiveKit project lives under Sunny's `sunny@menora-health.com` account, which Aryaman shared at the start.)
5. **GitHub**: add to the three private repos
   - https://github.com/aryamandarda/nemara-backend
   - https://github.com/aryamandarda/nemara-caregiver-app
   - https://github.com/potaton00b/NemaraCam
   - https://github.com/potaton00b/nemara-caregiver-web
6. **Apple Developer Team** `V8N59CF74G`: needed to sign the iOS apps with the existing bundle IDs (`co.elephantlabs.nemara.*`). Apple Developer → People → Invite. If a teammate signs with their own Apple Team they'll have to swap the bundle ID in `project.yml` + `Defaults.xcconfig` + `Info.plist` — keep those changes local and uncommitted (don't push them).

## Day-to-day operating tools

Install on macOS (one-time):

```bash
brew install supabase/tap/supabase
brew install railway
brew install vercel-cli
brew install livekit-cli      # provides `lk`
brew install gh               # GitHub CLI (optional)
brew install uv               # Python package manager — only needed if you edit/run backend locally
```

Auth each one:

```bash
supabase login                # opens browser
railway login                 # browser
vercel login                  # browser
gh auth login                 # browser

# LiveKit: `lk` uses LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET env vars from .env
# Sample: list rooms
LIVEKIT_URL=$(grep ^NEMARA_LIVEKIT_URL .env | cut -d= -f2-) \
LIVEKIT_API_KEY=$(grep ^NEMARA_LIVEKIT_API_KEY .env | cut -d= -f2-) \
LIVEKIT_API_SECRET=$(grep ^NEMARA_LIVEKIT_API_SECRET .env | cut -d= -f2-) \
lk room list
```

Once `railway login` is done, link this repo to the project:

```bash
cd nemara-backend
railway link --project f2f8606e-9ecd-448b-b127-9cafb8926b5f
```

After that, `railway logs`, `railway variables`, `railway up` etc. just work.

## Reading logs

| What you want | Where |
|---------------|-------|
| Live backend stdout (uvicorn, detector worker, webhook handler) | `railway logs` or dashboard → Logs |
| Last successful Docker build | `railway logs --build` |
| Supabase auth attempts, DB queries, Realtime channels | https://supabase.com/dashboard/project/zgifmvjnggmgzvtnosnu/logs/explorer |
| Vercel build + edge function logs | https://vercel.com/sunnys-projects-8be52dc6/nemara-caregiver-web → Deployments → click one |
| LiveKit room / participant events | https://cloud.livekit.io/projects/nemara-pdx0uelu → Sessions / Webhooks |

The single most useful log filter when LLM analyses go silent:
```bash
railway logs | grep -E "mp4_remux|worker cam=|signature_failed|participant_joined"
```

## Configuration: `.env`

`.env` is **intentionally tracked in this private repo** so a teammate clone is fully runnable without copying secrets around. All Supabase/LiveKit/Gemini/OpenAI/APNs values inside it are live and match the deployed Railway env. If you rotate a credential, update both `.env` and the matching variable on Railway.

The currently-tracked keys (full values in `.env`):

```
# Environment + DB
NEMARA_ENV                  prod (was local in old README)
NEMARA_DB_URL               Supabase Cloud pooler (Supavisor) connection string

# Supabase auth + storage
NEMARA_SUPABASE_URL
NEMARA_SUPABASE_PUBLISHABLE_KEY
NEMARA_SUPABASE_JWT_SECRET
NEMARA_SUPABASE_SERVICE_ROLE_KEY

# LLM providers
NEMARA_GEMINI_API_KEY / _MODEL / _VIDEO_FPS / _MEDIA_RESOLUTION
NEMARA_OPENAI_API_KEY / _MODEL / _FRAMES_FPS

# LiveKit
NEMARA_LIVEKIT_URL                          wss://nemara-pdx0uelu.livekit.cloud
NEMARA_LIVEKIT_API_KEY                      APITxi4Q…
NEMARA_LIVEKIT_API_SECRET                   254B2sR…
NEMARA_LIVEKIT_WEBHOOK_API_KEY              (empty — falls back to API_KEY; do not set to anything else without re-checking livekit_webhooks.py)
NEMARA_LIVEKIT_WORKER_TICK_SECONDS          10
NEMARA_LIVEKIT_WORKER_WINDOW_SECONDS        15
NEMARA_LIVEKIT_WORKER_TARGET_FPS            5
NEMARA_LIVEKIT_MOTION_STATE_TTL_SECONDS     15

# Pairing + storage tunables
NEMARA_PAIRING_CODE_TTL_SECONDS             600
NEMARA_BCRYPT_ROUNDS                        12
NEMARA_STORAGE_BUCKET_PATIENT_PHOTOS / _INCIDENT_CLIPS
NEMARA_STORAGE_MAX_*_BYTES
NEMARA_STORAGE_*_TIMEOUT_SECONDS
NEMARA_STORAGE_SIGNED_URL_DEFAULT_TTL_SECONDS

# APNs
NEMARA_APNS_TEAM_ID                         V8N59CF74G
NEMARA_APNS_KEY_ID
NEMARA_APNS_AUTH_KEY                        multi-line .p8 contents, BEGIN/END preserved
NEMARA_APNS_BUNDLE_ID                       co.elephantlabs.nemara.caregiver
NEMARA_APNS_ENVIRONMENT                     sandbox | production
NEMARA_APNS_JWT_LIFETIME_SECONDS

# Event promotion
NEMARA_CLIP_POST_EVENT_WAIT_SECONDS
```

Railway has the same set, set via `railway variables set …` during the original deploy. To sync local `.env` → Railway after editing one value:

```bash
# replace a single var on Railway
railway variable set NEMARA_LIVEKIT_WORKER_TICK_SECONDS=5
```

(Editing `.env` alone doesn't redeploy; Railway pulls from its own env store.)

## Deploying changes

**Backend (Railway)**

```bash
cd nemara-backend
railway up --service nemara-backend --detach
# build + deploy takes ~2 min; tail with:
railway logs --build
```

The `Dockerfile` installs `ffmpeg` + `uv` + the project's dependencies; the `CMD` reads `$PORT` at start time so Railway can inject any port.

If a deploy seems live but `/ready` still 500s, check `railway logs` for stack traces — usually a missing or malformed env var. The webhook fix from the deploy session covers the most-painful failure mode (`NEMARA_LIVEKIT_WEBHOOK_API_KEY` accidentally bound to a comment block — see the corresponding commit message).

**Web dashboard (Vercel)**

```bash
cd nemara-caregiver-web
vercel --prod --yes
```

Vercel re-enables SSO protection by default on each project; if a fresh deploy returns 401 with `_vercel_sso_nonce`, hit the project's `ssoProtection: null` endpoint via the management API (see commit history for an example). The stable alias `https://nemara-caregiver-web.vercel.app` always points at the latest production deploy.

**Database migrations**

Schema changes go in `supabase/schema.sql` for local dev AND in a timestamped file under `supabase/migrations/` for cloud. To push a migration:

```bash
cd nemara-backend
supabase link --project-ref zgifmvjnggmgzvtnosnu     # one-time per machine
supabase db push                                      # applies any new migrations
```

The cloud DB's connection string is in `.env` as `NEMARA_DB_URL`; you can also connect directly with any postgres client.

**iOS apps**: not deployed — built and side-loaded onto wall phones / your caregiver phone via Xcode. The Hardcoded backend URL in `NemaraCam/AppSettings.swift` and the fallback URLs in `nemara-caregiver-app/Networking/AppConfig.swift` both point at the Railway URL, so fresh installs hit prod with zero configuration.

## Request flow (end-to-end fall detection)

For when something feels off and you need to trace where things stop:

1. Wall phone (NemaraCam) captures frames at 15fps, runs on-device motion detection (max-tile SAD on Y plane), and publishes a `motion_state` LiveKit data packet every ~5s with the boolean "was there motion in the last 15s".
2. Wall phone also publishes its audio + video tracks continuously to `wss://nemara-pdx0uelu.livekit.cloud` room `patient_<patient_id>`.
3. On `participant_joined`, LiveKit Cloud fires a signed webhook to `https://nemara-backend-production.up.railway.app/v1/livekit/webhooks`. The backend verifies the JWT, looks up the camera row, resets `state='open'` + `no_people_count=0`, and starts a `RoomWorker` that subscribes to the new participant's tracks.
4. Inside the `RoomWorker`, a `CameraIngest` task runs every `NEMARA_LIVEKIT_WORKER_TICK_SECONDS` (10s). On each tick:
   - If the camera is `closed` and there's no recent motion → skip (write a synthetic "skipped" row to `public.analyses` so the dashboard knows it ticked).
   - Otherwise: snapshot the last 15s of frames + audio from the in-memory ring buffers, `ffmpeg`-remux into MP4, hand the MP4 to Gemini (or OpenAI as fallback).
5. Gemini returns `people_present`, plus per-detector verdicts (`fall_detected`, `agitation_detected`, …). A row lands in `public.analyses` with the result.
6. `event_promotion.py` decides whether any detector hit warrants creating an `events` row (typically requires `*_detected=true` AND `*_confidence>=threshold`). If yes, a `public.events` insert happens.
7. Supabase Realtime broadcasts the new `analyses` and `events` rows on the `supabase_realtime` publication. The web dashboard's `AnalysesFeed` and `EventsFeed` components are subscribed and update in real time.
8. If an `events` row was created AND the caregiver has a registered APNs token, the backend also sends a push notification.

If LLM analyses stop flowing, the breakage is almost always at step 3 (webhook signature) or step 4 (ffmpeg missing / motion-cache stale / state stuck closed). The grep filter in [Reading logs](#reading-logs) covers all three.

## Common operations cheat sheet

```bash
# Inspect prod camera state machine (asyncpg directly)
cd nemara-backend
PROD_URL=$(railway variables --service nemara-backend --json | \
  python3 -c "import json,sys; print(json.load(sys.stdin)['NEMARA_DB_URL'].replace('postgresql+asyncpg://', 'postgresql://'))")
psql "$PROD_URL" -c "select id, room, state, no_people_count, close_threshold, last_seen_at from public.cameras;"

# Force a camera back to 'open' state (e.g. demo just started and it's stuck closed)
psql "$PROD_URL" -c "update public.cameras set state='open', no_people_count=0, state_changed_at=now();"

# List active LiveKit rooms + publishers
lk --url "$NEMARA_LIVEKIT_URL" --api-key "$NEMARA_LIVEKIT_API_KEY" --api-secret "$NEMARA_LIVEKIT_API_SECRET" room list

# Verify webhook routing — manually create a room to trigger participant_joined
lk room create webhook-probe-$(date +%s)
```

## Demo-day pre-flight

A short checklist to run before powering up wall phones in front of investors:

1. `curl https://nemara-backend-production.up.railway.app/ready` → expect HTTP 200.
2. Hit `https://nemara-caregiver-web.vercel.app/login` in a private tab → log in as `jane@patel.co` → confirm the patient view loads without errors.
3. In Supabase SQL editor: `select count(*) from public.cameras;` — should match the number of physical wall phones you've paired in (no seeded ghost cameras).
4. `railway logs | tail` — confirm no stack traces in the last few minutes.
5. Bring one wall phone online; in `railway logs` you should see `livekit_webhook participant_joined` followed by `room_worker.connected` within 2s, then `worker cam=… DONE provider=gemini` on the next tick.
