# nemara-backend

FastAPI + Supabase backend for the Nemara caregiver iOS app and the in-process LiveKit subscriber-bot detector pipeline.

See [docs/architecture/backend-architecture.md](./docs/architecture/backend-architecture.md) for the file tree, conventions, detector pipeline, and Talk-mode wiring. For the **deployed cloud stack** (Railway / Supabase Cloud / Vercel / LiveKit Cloud), see [docs/guides/setup-hosted-stack.md](./docs/guides/setup-hosted-stack.md). For per-account VLM routing (Gemini / OpenAI / Qwen), see [docs/guides/vlm-routing.md](./docs/guides/vlm-routing.md).

> **Start here.** This repo is the entry point — bring it up first, then build the two iOS apps ([`nemara-camera-app`](../nemara-camera-app), [`nemara-caregiver-app`](../nemara-caregiver-app)).

## Before a new teammate can run this

Three admin invites that only the project owner can do — not steps the teammate themselves runs:

1. **LiveKit Cloud** → Project Settings → Members → invite by email. Your local `.env` points at *your* LiveKit project; without project membership, the teammate can't update the webhook URL if their ngrok domain changes.
2. **Apple Developer** → People → invite. The bundle IDs (defined in each iOS repo's `project.yml` — currently `co.sunny.*` prefixed) and the APNs key live under Team `V8N59CF74G`. Either invite the teammate to the team (simpler), or they'll fork to their own team + bundle IDs + APNs key — see each iOS README for that branch.
3. **GitHub** → Settings → Collaborators → invite to the sibling repos (`nemara-backend`, `nemara-camera-app`, `nemara-caregiver-app`, `nemara-web-app`, `nemara-gateway`). Send the clone URLs once they accept.

The teammate also needs a free [ngrok](https://dashboard.ngrok.com/signup) account for their own authtoken (free tier is fine, one tunnel at a time).

## What you're getting

After `make fresh-seed` completes you'll have:

- **Supabase** (Postgres + Auth + Realtime + Storage) on local ports (API `:54341`, Studio `:54343`).
- **Backend** uvicorn on `:8000`, serving the iOS apps + the detector pipeline. (`make dev` and the Dockerised `make backend-up` run on `:8002` instead.)
- **ngrok tunnel** in front of `:8000` so LiveKit Cloud can reach `/v1/livekit/webhooks` from the public internet.
- A demo caregiver (`jane@patel.co` / `Mar9aret!2024`) with patient Maggie + seeded detectors + a few historical events.

## Prereqs (one-time, on a fresh Mac)

```bash
brew install uv supabase ngrok docker
ngrok config add-authtoken <token>          # https://dashboard.ngrok.com → Your Authtoken
```

You also need Docker running (Supabase local uses it). The first `supabase start` will pull a few images — give it a few minutes.

## Bring the stack up

```bash
git clone <this repo>
cd nemara-backend
cp .env.example .env    # then fill in values — get the shared dev secrets from a teammate
make install            # uv sync
make fresh-seed         # full reset + demo data + backend + ngrok
```

> `.env` is **gitignored** (it used to be committed; it was removed after a key-leak incident). `.env.example` documents the shape; the actual dev secrets are shared out-of-band. Production secrets live only in Railway.

What `make fresh-seed` does:

1. Stops any stale uvicorn / ngrok / Supabase from a prior run.
2. `supabase start` + `supabase db reset` (applies `supabase/schema.sql` + `supabase/seed.sql`).
3. Creates the two private Storage buckets (`patient-photos`, `incident-clips`) via REST.
4. Seeds demo data via `scripts/seed.py` — caregiver + patient + detectors + events.
5. Auto-detects this Mac's LAN IP and exports `NEMARA_SUPABASE_URL=http://<lan>:54341` so iOS devices on the same Wi-Fi can hit local Supabase. (Falls back to the value in `.env` if no Wi-Fi.)
6. Starts uvicorn `:8000` in the background (PID + log under `.dev/`).
7. Starts ngrok in the background and prints the public webhook URL.

Other Make targets:

```bash
make fresh              # same as fresh-seed but skip demo data
make dev                # foreground uvicorn on :8002 (auto-starts Supabase; no ngrok)
make backend-up         # backend in Docker on :8002 against local Supabase (docker/docker-compose.local.yml)
make stack-up           # Supabase + backend + web dashboard (../nemara-web-app) on :3000
make down               # stop everything cleanly
make db-reset           # wipe + reapply schema.sql, re-seed demo data, recreate buckets
```

Smoke test: `curl http://localhost:8000/health` → `{"status":"ok"}` (use `:8002` for `make dev` / `make backend-up`).

## After it's up — the one thing you must do

The script prints, in yellow:

```
Action needed: paste this webhook URL into LiveKit Cloud →
  https://<your-ngrok-domain>/v1/livekit/webhooks
```

Open [LiveKit Cloud](https://cloud.livekit.io) → **your project** → **Settings → Webhooks** → either edit the existing entry or add a new one with that URL. Signing key auth = the project's API key (LiveKit auto-fills). Without this, the subscriber-bot never gets discovered when a wall phone joins a room.

> `scripts/local-start.sh` pins the tunnel to a static free ngrok domain by default (override with the `NGROK_DOMAIN` env var — get your own free reserved domain in ngrok's dashboard), so the URL survives `make fresh` cycles and you only paste it into LiveKit once.

## Configuration (`.env`)

`.env` is **not** committed (see `.env.example` for the shape; secrets are shared out-of-band). Per-machine values are auto-handled:

| Setting | Per-machine? | How it's handled |
|---|---|---|
| `NEMARA_SUPABASE_URL` | yes (LAN IP) | `local-start.sh` exports the current LAN IP on every run |
| `NEMARA_LIVEKIT_*` | shared (single dev project) | in the shared dev `.env` |
| `NEMARA_GEMINI_API_KEY` | shared (one key, shared quota) | in the shared dev `.env` |
| `NEMARA_OPENAI_API_KEY` | shared | in the shared dev `.env` |
| `NEMARA_APNS_*` | shared if same Apple Team (`V8N59CF74G`) | in the shared dev `.env` |

The full settings catalog (VLM routing, OpenRouter/DashScope, re-ID, storage, B2, email, …) lives in `src/settings.py` — every field maps to an env var with the `NEMARA_` prefix.

If you're on a different Apple Developer Team, generate your own APNs key in Apple Developer → Keys → Push, and replace `NEMARA_APNS_TEAM_ID` + `NEMARA_APNS_KEY_ID` + `NEMARA_APNS_AUTH_KEY` (the latter being the contents of the `.p8` file, BEGIN/END lines preserved, between double quotes — the existing entry shows the format).

## The iOS apps

Both iOS apps (`nemara-camera-app`, `nemara-caregiver-app`) live in sibling repos. Once the backend is up:

- **Wall phone** (`nemara-camera-app`): tap into the pairing code the caregiver app shows. The Mac's LAN IP is what the iPhone connects to — same network required.
- **Caregiver phone** (`nemara-caregiver-app`): sign in with `jane@patel.co` / `Mar9aret!2024` (or create your own), pair a new wall phone from the Settings tab.

See each repo's README for the iOS build flow.

## Sister repos

- [`nemara-camera-app`](../nemara-camera-app) — wall-phone iOS app (Memoria Sensor).
- [`nemara-caregiver-app`](../nemara-caregiver-app) — caregiver iOS app (Memoria Care).
- [`nemara-web-app`](../nemara-web-app) — caregiver web dashboard (Next.js; `make stack-up` runs it in Docker).
- [`nemara-gateway`](../nemara-gateway) — edge appliance: Tapo RTSP cams → LiveKit participants.

## Troubleshooting

- **Supabase won't start** → `supabase stop --no-backup && docker system prune -f && make fresh-seed`. Usually a dangling container from a previous unclean exit.
- **Backend health check fails** → `tail .dev/backend.log`. 90% of failures are missing creds in `.env` or Docker not running.
- **ngrok exits immediately** → did you run `ngrok config add-authtoken`? Free accounts allow one tunnel at a time — kill any other ngrok instance.
- **iPhone can't reach backend** → make sure iPhone is on the same Wi-Fi. Verify with `curl http://<mac-lan-ip>:8002/health` (Docker / `make dev`) or `:8000` (`make fresh-seed`) from another device on the network.
- **LiveKit subscriber-bot never wakes up** → check that the webhook URL in LiveKit Cloud matches the current ngrok URL. Webhooks are silently dropped if the URL is stale.
