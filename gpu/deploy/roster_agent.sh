#!/usr/bin/env bash
# Roster agent — asks the backend which homes this node should run, and makes
# reality match the answer.
#
# Replaces the static NEMARA_IDENTITY_ROOMS list. With a fixed list, adding a
# node under load or recovering from a dead one both mean hand-editing two
# nodes' env and restarting them. Polling means: start a node and it drains
# the unassigned queue; lose a node and a survivor picks its homes up.
#
# The poll IS the heartbeat — if this stops, the backend releases this node's
# homes to whoever polls next after identity_roster_stale_after_s (90s).
#
# RECONCILIATION is the fiddly part and is deliberately dumb:
#   * one child process per room, pid recorded in $RUN/<room>.pid
#   * a room in the roster with no live pid  -> start it
#   * a live pid for a room NOT in the roster -> stop it
#   * everything else                         -> leave alone
# No restart-on-every-poll, because a needless restart costs that home a
# re-ground and ~30-60s of silently-skipped routine detectors.
#
# REQUIRED ENV
#   NEMARA_BACKEND_URL           e.g. https://api.memoria-health.com
#   NEMARA_INTERNAL_SERVICE_KEY  the roster's shared secret (X-Internal-Key)
#   NEMARA_NODE_ID               stable node identity; RunPod pod id
#   NEMARA_NODE_BASE_URL         where the backend reaches THIS node's
#                                identity processes (tailnet name in prod)
# OPTIONAL
#   NEMARA_NODE_CAPACITY         homes this node accepts (default 4)
#   NEMARA_ROSTER_POLL_S         seconds between polls (default 30)
set -uo pipefail

: "${NEMARA_BACKEND_URL:?set NEMARA_BACKEND_URL}"
: "${NEMARA_INTERNAL_SERVICE_KEY:?set NEMARA_INTERNAL_SERVICE_KEY (the roster rejects unauthenticated polls outside local/test)}"
: "${NEMARA_NODE_ID:?set NEMARA_NODE_ID}"
: "${NEMARA_NODE_BASE_URL:?set NEMARA_NODE_BASE_URL}"

CAPACITY="${NEMARA_NODE_CAPACITY:-4}"
POLL_S="${NEMARA_ROSTER_POLL_S:-30}"
STATE_DIR="${NEMARA_STATE_DIR:-/persist/state}"
LOGDIR="$STATE_DIR/logs"
RUN="$STATE_DIR/run"
PERCEPTION_PORT="${PERCEPTION_PORT:-8000}"
mkdir -p "$LOGDIR" "$RUN"

log() { echo "$(date -u +%FT%TZ) roster: $*" | tee -a "$LOGDIR/roster.log"; }

# Ports are derived from the room id, not from arrival order. Order-based
# assignment would move a room to a different port whenever the roster
# changed, and a room that changed port looks to the backend like a brand new,
# empty home. A hash is stable for the life of the room.
_port_for() {
  python3 - "$1" <<'PY'
import hashlib, sys
# 8012..8111 — 100 slots, far above any node's capacity.
print(8012 + int(hashlib.sha256(sys.argv[1].encode()).hexdigest(), 16) % 100)
PY
}

_start_room() {
  local room="$1" port patient
  port="$(_port_for "$room")"
  patient="${room#patient_}"
  mkdir -p "$STATE_DIR/$room"
  (
    while true; do
      cd "$STATE_DIR/$room" || exit 1
      # --patient explicitly: its argparse default is a hardcoded pilot UUID,
      # which on any other home silently binds the wrong patient.
      python3 -m identity.service \
        --room "$room" --patient "$patient" --port "$port" \
        --pod-url "http://127.0.0.1:${PERCEPTION_PORT}" \
        >>"$LOGDIR/identity-$room.log" 2>&1
      rc=$?
      # 4 = pod wedged, 5 = camera walk exhausted; both are deliberate
      # self-exits asking to be restarted.
      echo "$(date -u +%FT%TZ) identity room=$room exited rc=$rc — restart in 3s" \
        >>"$LOGDIR/identity-$room.log"
      sleep 3
    done
  ) &
  echo $! > "$RUN/$room.pid"
  log "started room=$room port=$port pid=$(cat "$RUN/$room.pid")"

  # Userspace tailscaled cannot reach a local listener on its own; `serve` is
  # what publishes it onto the tailnet.
  if [ -n "${TS_AUTHKEY:-}" ]; then
    tailscale --socket=/var/run/tailscale/tailscaled.sock \
      serve --bg --tcp "$port" "tcp://127.0.0.1:$port" \
      >>"$LOGDIR/tailscale.log" 2>&1 || \
      log "WARN could not publish port $port on the tailnet"
  fi
}

_stop_room() {
  local room="$1" pid
  pid="$(cat "$RUN/$room.pid" 2>/dev/null)"
  [ -n "$pid" ] || return 0
  # Kill the supervising subshell AND its current child, or the loop simply
  # respawns identity and the room never actually stops.
  pkill -P "$pid" 2>/dev/null
  kill "$pid" 2>/dev/null
  rm -f "$RUN/$room.pid"
  log "stopped room=$room (reassigned or released)"
}

_alive() {
  local pid; pid="$(cat "$RUN/$1.pid" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

log "agent up node=$NEMARA_NODE_ID capacity=$CAPACITY poll=${POLL_S}s"

while true; do
  RESP="$(curl -sS -m 20 -X POST "$NEMARA_BACKEND_URL/v1/identity/roster" \
      -H "Content-Type: application/json" \
      -H "X-Internal-Key: $NEMARA_INTERNAL_SERVICE_KEY" \
      -d "{\"node_id\":\"$NEMARA_NODE_ID\",\"base_url\":\"$NEMARA_NODE_BASE_URL\",\"capacity\":$CAPACITY}" \
      2>>"$LOGDIR/roster.log")" || RESP=""

  WANT="$(printf '%s' "$RESP" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)          # non-JSON (5xx, proxy error) -> treat as unreachable
for r in d.get("rooms", []):
    print(r)
' 2>/dev/null)" || WANT="__UNREACHABLE__"

  if [ "$WANT" = "__UNREACHABLE__" ]; then
    # DO NOT stop rooms on a failed poll. A backend blip or a network hiccup
    # would otherwise blind every home on this node, and the backend has its
    # own 90s staleness window before it reassigns anything. Keep serving.
    log "WARN roster unreachable — keeping current rooms"
    sleep "$POLL_S"
    continue
  fi

  # start what is newly assigned (or has died and lost its pid)
  while IFS= read -r room; do
    [ -z "$room" ] && continue
    _alive "$room" || _start_room "$room"
  done <<< "$WANT"

  # stop what is no longer ours
  for pidf in "$RUN"/*.pid; do
    [ -e "$pidf" ] || continue
    room="$(basename "$pidf" .pid)"
    grep -qxF "$room" <<< "$WANT" || _stop_room "$room"
  done

  sleep "$POLL_S"
done
