#!/usr/bin/env bash
# scripts/local-start.sh — bring the full local stack up to a clean,
# reproducible, runnable state in one shot. Idempotent: re-runnable.
#
# What runs after this script exits successfully:
#   • Supabase local (Postgres + Auth + Storage + Studio)
#   • Backend (uvicorn on :8002, backgrounded)
#   • ngrok tunnel pointing at :8002 (backgrounded)
#
# What's left for you afterwards:
#   • Paste the printed ngrok URL into LiveKit Cloud's webhook config.
#   • Open the two iOS .xcodeproj's and ⌘R to your phones.
#
# Process state under .dev/ (gitignored):
#   .dev/backend.pid, .dev/backend.log
#   .dev/ngrok.pid,   .dev/ngrok.log
#
# Usage:
#   ./scripts/local-start.sh              # reset + start everything, no seed
#   ./scripts/local-start.sh --seed       # reset + start + seed demo data
#   ./scripts/local-start.sh --down       # stop everything cleanly
#   ./scripts/local-start.sh --no-ngrok   # skip ngrok (no webhooks)
#   ./scripts/local-start.sh -h | --help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DEV_DIR=".dev"

# ── flags ───────────────────────────────────────────────────────
SEED=false
DOWN_ONLY=false
WITH_NGROK=true

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seed) SEED=true; shift ;;
    --down|--teardown|--teardown-only) DOWN_ONLY=true; shift ;;
    --no-ngrok) WITH_NGROK=false; shift ;;
    -h|--help)
      sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# //; s/^#//'
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      echo "Try --help" >&2
      exit 2
      ;;
  esac
done

# ── pretty output ───────────────────────────────────────────────
if [[ -t 1 ]]; then
  CB=$'\033[34m'; CG=$'\033[32m'; CY=$'\033[33m'; CD=$'\033[2m'; CR=$'\033[0m'
else
  CB=""; CG=""; CY=""; CD=""; CR=""
fi
step() { printf "%s▸%s %s\n"  "$CB" "$CR" "$*"; }
ok()   { printf "%s✓%s %s\n"  "$CG" "$CR" "$*"; }
warn() { printf "%s!%s %s\n"  "$CY" "$CR" "$*"; }
dim()  { printf "  %s%s%s\n"  "$CD" "$*"  "$CR"; }

# ── prereqs ─────────────────────────────────────────────────────
missing=()
for tool in supabase uv lsof curl python3; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if $WITH_NGROK && ! command -v ngrok >/dev/null 2>&1; then
  warn "ngrok not installed — falling back to --no-ngrok"
  warn "  install: brew install ngrok && ngrok config add-authtoken <token>"
  WITH_NGROK=false
fi
if (( ${#missing[@]} > 0 )); then
  echo "Missing required tool(s): ${missing[*]}" >&2
  echo "Install via: brew install ${missing[*]}" >&2
  exit 1
fi

mkdir -p "$DEV_DIR"

# ── helpers ─────────────────────────────────────────────────────
kill_by_pidfile() {
  local pidfile="$1" label="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid=$(cat "$pidfile" 2>/dev/null || true)
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      # Give it a second to drain, then SIGKILL if still around.
      sleep 1
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
      ok "$label stopped (pid $pid)"
    else
      dim "$label not running (stale pidfile)"
    fi
    rm -f "$pidfile"
  else
    dim "$label not running"
  fi
}

# ── 1. Stop stale processes ─────────────────────────────────────
step "Stopping backend"
kill_by_pidfile "$DEV_DIR/backend.pid" "backend"
# Sweep stray uvicorn on :8002 (a prior `make dev` foreground run).
if PIDS=$(lsof -ti:8002 2>/dev/null) && [[ -n "$PIDS" ]]; then
  echo "$PIDS" | xargs kill -9 2>/dev/null || true
  ok "swept stray process(es) on :8002"
fi

step "Stopping ngrok"
kill_by_pidfile "$DEV_DIR/ngrok.pid" "ngrok"

step "Stopping Supabase"
if supabase status >/dev/null 2>&1; then
  # --no-backup wipes data volumes (Storage objects + DB state). Falls
  # back to bare `stop` on older CLIs that don't know the flag.
  supabase stop --no-backup >/dev/null 2>&1 \
    || supabase stop >/dev/null 2>&1 \
    || true
  ok "stopped"
else
  dim "not running"
fi

if $DOWN_ONLY; then
  echo
  ok "Teardown complete."
  exit 0
fi

# ── 2. Start Supabase ───────────────────────────────────────────
step "Starting Supabase"
supabase start >/dev/null
ok "started"

step "Applying supabase/schema.sql"
# `[db.seed]` in supabase/config.toml wires schema.sql + seed.sql into
# the "seed" step, so --no-seed would skip the whole schema apply.
# Plain `db reset` runs both; the SQL seed.sql is currently empty so
# only schema.sql lands. The optional Python demo seed runs in step 4.
supabase db reset >/dev/null 2>&1
ok "schema applied"

# ── 3. Storage buckets ──────────────────────────────────────────
step "Ensuring Storage buckets"
ENV_FILE=$(mktemp)
trap 'rm -f "$ENV_FILE"' EXIT
supabase status -o env > "$ENV_FILE"
API_URL=$(grep '^API_URL=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')
SR=$(grep '^SERVICE_ROLE_KEY=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"')

if [[ -z "$API_URL" || -z "$SR" ]]; then
  warn "couldn't read API_URL / SERVICE_ROLE_KEY — skipping bucket setup"
else
  for BUCKET in patient-photos incident-clips; do
    CODE=$(curl -sS -o /dev/null -w "%{http_code}" \
      -X POST "$API_URL/storage/v1/bucket" \
      -H "Authorization: Bearer $SR" \
      -H "Content-Type: application/json" \
      -d "{\"id\":\"$BUCKET\",\"name\":\"$BUCKET\",\"public\":false}")
    case "$CODE" in
      200|201) ok "$BUCKET created" ;;
      409)     dim "$BUCKET already exists" ;;
      400)     dim "$BUCKET exists or rejected (HTTP 400 — check Studio)" ;;
      *)       warn "$BUCKET unexpected status: HTTP $CODE" ;;
    esac
  done
fi

# ── 4. Optional seed ────────────────────────────────────────────
if $SEED; then
  step "Seeding demo data"
  PYTHONPATH=. uv run python scripts/seed.py
  ok "demo data inserted"
else
  dim "skipping seed (--seed to include demo data)"
fi

# ── 5. Start backend (background) ───────────────────────────────
step "Starting backend on :8002"
# Plain uvicorn (not `make dev`) so we control PID + log redirection.
# Logs go to .dev/backend.log; PID to .dev/backend.pid.
#
# Auto-detect this Mac's LAN IP and override NEMARA_SUPABASE_URL so
# iOS devices on the same Wi-Fi reach the local Supabase by IP, not
# the loopback baked into .env. Pydantic-settings reads env vars
# before .env values, so this overrides without touching the committed
# file. Falls back to whatever's in .env if no Wi-Fi interface is up.
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
if [[ -n "${LAN_IP:-}" ]]; then
  export NEMARA_SUPABASE_URL="http://${LAN_IP}:54341"
  ok "Supabase URL pinned to LAN: $NEMARA_SUPABASE_URL"
fi
PYTHONPATH=. nohup uv run uvicorn src.main:app \
  --host 0.0.0.0 --port 8002 --no-access-log \
  > "$DEV_DIR/backend.log" 2>&1 &
echo $! > "$DEV_DIR/backend.pid"
BACKEND_PID=$(cat "$DEV_DIR/backend.pid")

# Wait for /health to return 200. Lifespan does engine init + worker
# startup, so the first response can take 5-10s on a cold env.
HEALTHY=false
for i in $(seq 1 30); do
  if curl -fsS -m 1 http://127.0.0.1:8002/health >/dev/null 2>&1; then
    HEALTHY=true
    break
  fi
  # If the process already died, fail fast.
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    warn "backend exited before going healthy — tail of .dev/backend.log:"
    tail -20 "$DEV_DIR/backend.log" | sed 's/^/    /'
    rm -f "$DEV_DIR/backend.pid"
    exit 1
  fi
  sleep 1
done
if $HEALTHY; then
  ok "backend healthy (pid $BACKEND_PID, log: $DEV_DIR/backend.log)"
else
  warn "backend didn't return /health after 30s — check $DEV_DIR/backend.log"
fi

# ── 6. Start ngrok (background) ─────────────────────────────────
NGROK_URL=""
if $WITH_NGROK; then
  step "Starting ngrok tunnel"
  nohup ngrok http 8002 --log=stdout > "$DEV_DIR/ngrok.log" 2>&1 &
  echo $! > "$DEV_DIR/ngrok.pid"
  NGROK_PID=$(cat "$DEV_DIR/ngrok.pid")

  # ngrok exposes a local API on :4040 once the tunnel is up.
  for i in $(seq 1 15); do
    NGROK_URL=$(curl -fsS -m 1 http://127.0.0.1:4040/api/tunnels 2>/dev/null \
      | python3 -c 'import sys, json
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for t in data.get("tunnels", []):
    if t.get("public_url", "").startswith("https://"):
        print(t["public_url"]); break' 2>/dev/null || true)
    [[ -n "$NGROK_URL" ]] && break
    if ! kill -0 "$NGROK_PID" 2>/dev/null; then
      warn "ngrok exited — tail of .dev/ngrok.log:"
      tail -10 "$DEV_DIR/ngrok.log" | sed 's/^/    /'
      rm -f "$DEV_DIR/ngrok.pid"
      break
    fi
    sleep 1
  done

  if [[ -n "$NGROK_URL" ]]; then
    ok "tunnel: $NGROK_URL"
  else
    warn "ngrok didn't expose a tunnel — check $DEV_DIR/ngrok.log"
  fi
fi

# ── 7. Done ─────────────────────────────────────────────────────
echo
ok "Local stack is up."
echo
echo "Health:  http://127.0.0.1:8002/health"
echo "Studio:  http://127.0.0.1:54343"
echo "Logs:    tail -f $DEV_DIR/backend.log   # backend"
if $WITH_NGROK && [[ -n "$NGROK_URL" ]]; then
  echo "         tail -f $DEV_DIR/ngrok.log     # ngrok"
fi
echo "Stop:    make down"
echo

if $WITH_NGROK && [[ -n "$NGROK_URL" ]]; then
  echo "${CY}Action needed:${CR} paste this webhook URL into LiveKit Cloud →"
  echo "  ${CB}${NGROK_URL}/v1/livekit/webhooks${CR}"
  echo
  echo "(ngrok URLs change per session unless you have a reserved domain.)"
fi
