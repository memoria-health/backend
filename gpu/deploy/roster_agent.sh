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
# RUN is CONTAINER-LOCAL, never on /persist (node F melt, 2026-08-13).
# Pid/port files on the shared network volume outlive the container that
# wrote them — and are visible to OTHER nodes mounting the same volume —
# so a fresh boot reconciled against dead containers' pids, and PID reuse
# made `kill -0` vouch for unrelated processes. Run-state must die with
# the container that owns it.
RUN="${NEMARA_RUN_DIR:-/run/nemara}"
PERCEPTION_PORT="${PERCEPTION_PORT:-8000}"
mkdir -p "$LOGDIR" "$RUN"

log() { echo "$(date -u +%FT%TZ) roster: $*" | tee -a "$LOGDIR/roster.log"; }

# The PORT COMES FROM THE ROSTER. It is not derived here.
#
# An earlier version hashed the room id to a port. That is wrong twice over:
# the backend has to REACH this process, so it cannot be a number the node
# picks unilaterally, and a hash into 100 slots collides (~6% at capacity 4)
# with the loser wedged in an EADDRINUSE restart loop while looking healthy.
# The backend allocates the lowest free port per node and tells us, the same
# way it allocates gateways.tunnel_port.
_start_room() {
  local room="$1" port="$2" patient
  patient="${room#patient_}"
  mkdir -p "$STATE_DIR/$room"
  # Singleton per room, enforced, not assumed (node F melt, 2026-08-13):
  # a stop/kill race once orphaned a live identity process, and the
  # duplicate pair kicked each other off LiveKit (same participant
  # identity) in an endless churn. Before starting, kill any straggler
  # serving this room — match on the exact --room argument.
  pkill -f -- "identity.service --room $room " 2>/dev/null && sleep 1
  (
    backoff=3
    while true; do
      cd "$STATE_DIR/$room" || exit 1
      t0=$(date +%s)
      # --patient explicitly: its argparse default is a hardcoded pilot UUID,
      # which on any other home silently binds the wrong patient.
      python3 -m identity.service \
        --room "$room" --patient "$patient" --port "$port" \
        --pod-url "http://127.0.0.1:${PERCEPTION_PORT}" \
        >>"$LOGDIR/identity-$room.log" 2>&1
      rc=$?
      dt=$(( $(date +%s) - t0 ))
      # 4 = pod wedged, 5 = camera walk exhausted; both are deliberate
      # self-exits asking to be restarted. Backoff doubles to 60s on
      # rapid deaths and resets after a 300s healthy run — a service
      # dying at import must never become a fork storm again.
      [ "$dt" -gt 300 ] && backoff=3
      echo "$(date -u +%FT%TZ) identity room=$room exited rc=$rc after ${dt}s — restart in ${backoff}s" \
        >>"$LOGDIR/identity-$room.log"
      sleep "$backoff"
      backoff=$(( backoff * 2 ))
      [ "$backoff" -gt 60 ] && backoff=60
    done
  ) &
  local sup=$!
  echo "$sup" > "$RUN/$room.pid"
  # The port is recorded too, so the next poll can tell "already running here"
  # from "running on a port the backend has since moved".
  echo "$port" > "$RUN/$room.port"
  log "started room=$room port=$port pid=$sup"

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
  # ORDER MATTERS (node F melt, 2026-08-13). The old sequence killed the
  # child first, then the supervisor — if the supervisor was between
  # `sleep` and respawn, the fresh python it spawned in that window was
  # orphaned alive, and the next _start_room made a duplicate. Kill the
  # supervisor FIRST so nothing can respawn, then pattern-kill any
  # identity process still serving this room (matches the exact --room
  # argument), then verify the pid is actually gone before declaring
  # the room stopped.
  kill "$pid" 2>/dev/null
  pkill -f -- "identity.service --room $room " 2>/dev/null
  for _ in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null
    pkill -9 -f -- "identity.service --room $room " 2>/dev/null
    log "stopped room=$room (FORCED — did not exit on TERM)"
  else
    log "stopped room=$room (reassigned or released)"
  fi
  rm -f "$RUN/$room.pid" "$RUN/$room.port"
}

_alive() {
  local pid; pid="$(cat "$RUN/$1.pid" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

log "agent up node=$NEMARA_NODE_ID capacity=$CAPACITY poll=${POLL_S}s"

# Userspace tailscale routes NOTHING outbound by itself — a tailnet
# (100.64.0.0/10) backend is reachable only through tailscaled's HTTP
# proxy on :1055. Production backends are public and go direct; adding
# the proxy there would make roster liveness depend on tailscaled for
# no reason. Found on the first real node boot (2026-08-12): every
# poll to the local rig's tailnet backend died with "unreachable".
case "$NEMARA_BACKEND_URL" in
  http://100.*|https://100.*) CURL_PROXY=(--proxy "http://127.0.0.1:1055") ;;
  *)                          CURL_PROXY=() ;;
esac

while true; do
  # Body to a file, status code to stdout. Keeping them apart is what lets
  # the two failure shapes below be told apart at all.
  CODE="$(curl -sS -m 20 "${CURL_PROXY[@]}" -X POST "$NEMARA_BACKEND_URL/v1/identity/roster" \
      -H "Content-Type: application/json" \
      -H "X-Internal-Key: $NEMARA_INTERNAL_SERVICE_KEY" \
      -d "{\"node_id\":\"$NEMARA_NODE_ID\",\"base_url\":\"$NEMARA_NODE_BASE_URL\",\"capacity\":$CAPACITY}" \
      -o "$RUN/.roster.json" -w '%{http_code}' \
      2>>"$LOGDIR/roster.log")" || CODE="000"
  RESP="$(cat "$RUN/.roster.json" 2>/dev/null)"

  # A 4xx is CONFIGURATION, and it will never fix itself: a rotated internal
  # key (403), or a base_url the roster refuses as unaddressable (422). Both
  # otherwise look identical to a network blip — the node keeps polling,
  # keeps warning, and never claims a home. Say plainly that this one needs
  # a human, and include the body, which names the offending field.
  case "$CODE" in
    4*)
      log "ERROR roster REJECTED this node (HTTP $CODE) — this will NOT recover on its own"
      log "      node=$NEMARA_NODE_ID base_url=$NEMARA_NODE_BASE_URL"
      log "      $(printf '%s' "$RESP" | head -c 300)"
      ;;
  esac

  # Emits "<room> <port>" per line. Exits non-zero on anything that is not a
  # well-formed roster reply.
  #
  # An error BODY is still valid JSON — a 403 from a rotated key parses fine
  # and has no "rooms" key, which an earlier version read as "you have been
  # assigned nothing" and dutifully stopped every room on the node. The
  # presence of the key, not the truthiness of its value, is the signal.
  WANT="$(printf '%s' "$RESP" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)                 # non-JSON: 5xx, proxy error, empty body
if not isinstance(d, dict) or "rooms" not in d:
    sys.exit(1)                 # error body (403/422/500), NOT an empty roster
for r in d["rooms"]:
    print(r["room"], r["port"])
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
  while IFS=' ' read -r room port; do
    [ -z "$room" ] && continue
    if _alive "$room"; then
      # Already running — but if the backend moved it to a different port,
      # restart it there. Otherwise the process keeps serving a port nobody
      # is calling and the home looks alive while getting no traffic.
      if [ "$(cat "$RUN/$room.port" 2>/dev/null)" != "$port" ]; then
        log "room=$room port changed -> $port, restarting"
        _stop_room "$room"
        _start_room "$room" "$port"
      fi
    else
      _start_room "$room" "$port"
    fi
  done <<< "$WANT"

  # stop what is no longer ours
  for pidf in "$RUN"/*.pid; do
    [ -e "$pidf" ] || continue
    room="$(basename "$pidf" .pid)"
    awk '{print $1}' <<< "$WANT" | grep -qxF "$room" || _stop_room "$room"
  done

  sleep "$POLL_S"
done
