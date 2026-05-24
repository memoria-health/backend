# nemara-backend

FastAPI + Supabase backend for the Nemara caregiver iOS app and the in-process LiveKit subscriber-bot detector pipeline.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the file tree, conventions, detector pipeline, and Talk-mode wiring.

## What you're getting

After `make fresh-seed` completes you'll have:

- **Supabase** (Postgres + Auth + Realtime + Storage) on local ports.
- **Backend** uvicorn on `:8002`, serving the iOS apps + the detector pipeline.
- **ngrok tunnel** in front of `:8002` so LiveKit Cloud can reach `/v1/livekit/webhooks` from the public internet.
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
make install            # uv sync
make fresh-seed         # full reset + demo data + backend + ngrok
```

What `make fresh-seed` does:

1. Stops any stale uvicorn / ngrok / Supabase from a prior run.
2. `supabase start` + `supabase db reset` (applies `supabase/schema.sql` + `supabase/seed.sql`).
3. Creates the two private Storage buckets (`patient-photos`, `incident-clips`) via REST.
4. Seeds demo data via `scripts/seed.py` — caregiver + patient + detectors + events.
5. Auto-detects this Mac's LAN IP and exports `NEMARA_SUPABASE_URL=http://<lan>:54341` so iOS devices on the same Wi-Fi can hit local Supabase. (Falls back to the value in `.env` if no Wi-Fi.)
6. Starts uvicorn `:8002` in the background (PID + log under `.dev/`).
7. Starts ngrok in the background and prints the public webhook URL.

Other Make targets:

```bash
make fresh              # same as fresh-seed but skip demo data
make dev                # foreground uvicorn, no Supabase / ngrok touched
make down               # stop everything cleanly
make db-reset           # reapply schema.sql without restarting the rest
```

Smoke test: `curl http://localhost:8002/health` → `{"status":"ok"}`.

## After it's up — the one thing you must do

The script prints, in yellow:

```
Action needed: paste this webhook URL into LiveKit Cloud →
  https://<your-ngrok>.ngrok-free.app/v1/livekit/webhooks
```

Open [LiveKit Cloud](https://cloud.livekit.io) → **your project** → **Settings → Webhooks** → either edit the existing entry or add a new one with that URL. Signing key auth = the project's API key (LiveKit auto-fills). Without this, the subscriber-bot never gets discovered when a wall phone joins a room.

> ngrok URLs change every time the tunnel restarts unless you have a reserved domain. If you're going to iterate, get a free reserved domain in ngrok's dashboard and add `--domain=<your>.ngrok.app` in `scripts/local-start.sh` step 6.

## Configuration (`.env`)

The repo ships with a working `.env` (it's a private repo). Per-machine values are auto-handled:

| Setting | Per-machine? | How it's handled |
|---|---|---|
| `NEMARA_SUPABASE_URL` | yes (LAN IP) | `local-start.sh` exports the current LAN IP on every run |
| `NEMARA_LIVEKIT_*` | shared (single project) | committed in `.env` |
| `NEMARA_GEMINI_API_KEY` | shared (one key, shared quota) | committed |
| `NEMARA_OPENAI_API_KEY` | shared | committed |
| `NEMARA_APNS_*` | shared if same Apple Team (`V8N59CF74G`) | committed |

If you're on a different Apple Developer Team, generate your own APNs key in Apple Developer → Keys → Push, and replace `NEMARA_APNS_TEAM_ID` + `NEMARA_APNS_KEY_ID` + `NEMARA_APNS_AUTH_KEY` (the latter being the contents of the `.p8` file, BEGIN/END lines preserved, between double quotes — the existing entry shows the format).

## The iOS apps

Both iOS apps (`nemara-camera-app`, `nemara-caregiver-app`) live in sibling repos. Once the backend is up:

- **Wall phone** (`nemara-camera-app`): tap into the pairing code the caregiver app shows. The Mac's LAN IP is what the iPhone connects to — same network required.
- **Caregiver phone** (`nemara-caregiver-app`): sign in with `jane@patel.co` / `Mar9aret!2024` (or create your own), pair a new wall phone from the Settings tab.

See each repo's README for the iOS build flow.

## Sister repos

- [`nemara-camera-app`](../nemara-camera-app) — wall-phone iOS app.
- [`nemara-caregiver-app`](../nemara-caregiver-app) — caregiver iOS app.

## Troubleshooting

- **Supabase won't start** → `supabase stop --no-backup && docker system prune -f && make fresh-seed`. Usually a dangling container from a previous unclean exit.
- **Backend health check fails** → `tail .dev/backend.log`. 90% of failures are missing creds in `.env` or Docker not running.
- **ngrok exits immediately** → did you run `ngrok config add-authtoken`? Free accounts allow one tunnel at a time — kill any other ngrok instance.
- **iPhone can't reach backend** → make sure iPhone is on the same Wi-Fi. Verify with `curl http://<mac-lan-ip>:8002/health` from another device on the network.
- **LiveKit subscriber-bot never wakes up** → check that the webhook URL in LiveKit Cloud matches the current ngrok URL. Webhooks are silently dropped if the URL is stale.
