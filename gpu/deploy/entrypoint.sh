#!/usr/bin/env bash
# GPU node entrypoint — brings up the whole deployable in one container:
#
#   fetch checkpoints  (first boot only, onto the network volume)
#   tailscaled         (userspace; how Railway reaches /state)
#   perception :8000   LOOPBACK ONLY
#   identity   :8012+i one supervised process per room
#
# This is the docker-compose from the productionization plan's task 2.2
# ("perception + N identity instances, one per home/room env") as an image
# entrypoint — same shape, one artifact.
#
# WHY ONE CONTAINER: identity sends ~13MB batches to perception every cycle.
# Over a network that is ~2.2 TB/month/camera, and RunPod's inter-pod link is
# 100 Mbps (~1s per batch). Colocated it is a localhost copy — and perception
# never binds a network interface.
#
# WHY ONE PROCESS PER ROOM: each identity instance holds its own EdgeTAM
# sessions, seat and track ledger, and joins exactly one LiveKit room.
# Isolation is the point — the 2026-07-26 OOM killed identity and it "simply
# stayed dead — the seat provider for the entire backend". Per-room means one
# blind home, not all of them.
#
# REQUIRED ENV
#   NEMARA_IDENTITY_STATE_TOKEN  bearer for /state — MUST be set
#   NEMARA_DB_URL                face-enrollment gallery
#   NEMARA_LIVEKIT_URL / _API_KEY / _API_SECRET
#   HUGGINGFACE_TOKEN            first boot only, gated facebook/sam3.1
# ROSTER MODE (production) — set NEMARA_BACKEND_URL to select it
#   NEMARA_BACKEND_URL           e.g. https://api.memoria-health.com
#   NEMARA_INTERNAL_SERVICE_KEY  shared secret for the roster poll
#   TS_AUTHKEY                   required in practice: the tailnet name is
#                                what the backend dials, and this node
#                                DERIVES both its id and that name below
# STATIC MODE (bring-up / local docker run) — used when the above is unset
#   NEMARA_IDENTITY_ROOMS        comma-separated room ids
# OPTIONAL
#   NEMARA_NODE_ID               overrides the RunPod pod id
#   NEMARA_NODE_BASE_URL         overrides the derived tailnet address
#   NEMARA_NODE_CAPACITY         homes this node accepts (default 4)
#   NEMARA_PERCEPTION_CONCURRENCY  passes admitted at once (default 1)
#   NEMARA_STATE_DIR             writable state, default /persist/state
set -uo pipefail

STATE_DIR="${NEMARA_STATE_DIR:-/persist/state}"
LOGDIR="$STATE_DIR/logs"
mkdir -p "$LOGDIR"

log() { echo "$(date -u +%FT%TZ) entrypoint: $*"; }

# ── Config guards, before anything expensive ───────────────────────────
# An empty token is the dangerous one: contract.auth_ok() treats "" as
# "accept every caller", so /state, /probes and /stream (live annotated
# patient video) would be open to anything that reaches the port. Refuse.
# Rooms come from the roster in production. Only required in the static
# fallback, so the check moved down to where that path begins.
: "${NEMARA_IDENTITY_STATE_TOKEN:?refusing to start: an empty token disables auth on /state and /stream}"
: "${NEMARA_DB_URL:?set NEMARA_DB_URL — identity reads the face gallery from Postgres}"

# src.settings.Settings is `extra="forbid"` and REQUIRES these four, even
# though identity never reads them — it imports the backend's settings module
# wholesale (the documented packaging exception in gpu/identity/README.md).
# Obviously-fake placeholders, never real credentials: if any code path ever
# actually uses one, it must fail visibly rather than half-work.
export NEMARA_SUPABASE_URL="${NEMARA_SUPABASE_URL:-https://unused-on-gpu-node.invalid}"
export NEMARA_SUPABASE_PUBLISHABLE_KEY="${NEMARA_SUPABASE_PUBLISHABLE_KEY:-unused-on-gpu-node}"
export NEMARA_SUPABASE_JWT_SECRET="${NEMARA_SUPABASE_JWT_SECRET:-unused-on-gpu-node}"
export NEMARA_SUPABASE_SERVICE_ROLE_KEY="${NEMARA_SUPABASE_SERVICE_ROLE_KEY:-unused-on-gpu-node}"

# ── 1. Checkpoints (first boot only) ───────────────────────────────────
log "checking model weights in ${NEMARA_MODEL_DIR:-/persist/models}"
/workspace/nemara/deploy/fetch_checkpoints.sh 2>&1 | tee -a "$LOGDIR/checkpoints.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  log "FATAL: checkpoint fetch failed — see $LOGDIR/checkpoints.log"
  exit 1
fi

# ── 2. tailscale ───────────────────────────────────────────────────────
# Optional so the image still runs under a plain `docker run` for testing.
if [ -n "${TS_AUTHKEY:-}" ]; then
  log "starting tailscale supervisor"
  TS_HOSTNAME="${TS_HOSTNAME:-nemara-gpu-$(hostname | tail -c 7)}" \
    setsid --fork /workspace/nemara/deploy/tailscale_supervisor.sh \
    </dev/null >>"$LOGDIR/tailscale.log" 2>&1
else
  log "TS_AUTHKEY unset — skipping tailscale (Railway will have no route in)"
fi

# ── 3. perception ──────────────────────────────────────────────────────
# Loopback by default. PERCEPTION_HOST exists only so exposing it is a
# deliberate, reviewable act rather than a default nobody noticed.
(
  while true; do
    cd /workspace/edgetam || exit 1
    python3 -m uvicorn perception.server:app \
      --host "${PERCEPTION_HOST:-127.0.0.1}" \
      --port "${PERCEPTION_PORT:-8000}" \
      --timeout-keep-alive 75 >>"$LOGDIR/perception.log" 2>&1
    echo "$(date -u +%FT%TZ) perception exited rc=$? — restart in 3s" >>"$LOGDIR/perception.log"
    sleep 3
  done
) &

log "waiting for perception on 127.0.0.1:${PERCEPTION_PORT:-8000}"
for _ in $(seq 1 90); do
  curl -sf -m 2 -o /dev/null "http://127.0.0.1:${PERCEPTION_PORT:-8000}/health" && break
  sleep 2
done
curl -sf -m 2 -o /dev/null "http://127.0.0.1:${PERCEPTION_PORT:-8000}/health" \
  || log "WARNING: perception not answering yet — identity will retry"

# ── 4. identity ────────────────────────────────────────────────────────
# Two modes. ROSTER is production: the node asks the backend which homes it
# should run and reconciles continuously, so adding capacity is "start
# another node" and losing one is recovered by whoever polls next.
#
# The static list is the fallback for a node with no backend to talk to
# (bring-up, a local docker run, debugging a single home).
if [ -n "${NEMARA_BACKEND_URL:-}" ]; then
  # ── who am I, and where can the backend reach me? ────────────────────
  # Both are DERIVED HERE rather than passed in, because neither exists at
  # the moment the node is created: `pod.sh release` calls the RunPod API
  # and only learns the pod id from the response, and the tailnet name is
  # not assigned until tailscaled has authenticated. Anything passed in at
  # create time would be a guess.

  # RunPod injects RUNPOD_POD_ID into every container. It is the right node
  # identity because it is stable across container restarts of the SAME pod
  # — so a restarting node re-claims its own homes instead of appearing as
  # a stranger and waiting out the staleness window while they sit blind.
  export NEMARA_NODE_ID="${NEMARA_NODE_ID:-${RUNPOD_POD_ID:-}}"
  if [ -z "$NEMARA_NODE_ID" ]; then
    log "FATAL: roster mode needs NEMARA_NODE_ID and RUNPOD_POD_ID is unset"
    log "       (not on RunPod? pass NEMARA_NODE_ID explicitly)"
    exit 1
  fi

  # base_url is what the backend will dial. Ask tailscaled for the name it
  # was actually GIVEN rather than assuming it matches TS_HOSTNAME —
  # Tailscale appends a suffix (`-1`, `-2`) when the name is already taken,
  # which happens routinely when a replaced node's old record is still
  # present. Guessing there would register an address that resolves to the
  # DEAD node, and every home on this one would look assigned and be blind.
  if [ -z "${NEMARA_NODE_BASE_URL:-}" ] && [ -n "${TS_AUTHKEY:-}" ]; then
    log "waiting for a tailnet name"
    for _ in $(seq 1 60); do
      TS_NAME="$(tailscale --socket=/var/run/tailscale/tailscaled.sock status --json 2>/dev/null \
        | python3 -c 'import json,sys
try: d = json.load(sys.stdin)
except Exception: sys.exit(0)
# Only trust the name once the node is actually Online — DNSName is
# populated during login, before the connection is usable.
s = (d.get("Self") or {})
if s.get("Online"):
    print((s.get("DNSName") or "").rstrip("."))' 2>/dev/null)"
      [ -n "$TS_NAME" ] && break
      sleep 2
    done
    if [ -n "${TS_NAME:-}" ]; then
      export NEMARA_NODE_BASE_URL="http://$TS_NAME"
      log "tailnet name: $TS_NAME"
    fi
  fi
  if [ -z "${NEMARA_NODE_BASE_URL:-}" ]; then
    # Refuse rather than register an unreachable address. A node in the
    # roster with a base_url nobody can dial is WORSE than an absent one:
    # it claims homes up to its capacity and then serves none of them, and
    # the backend fails open, so those homes lose identity silently.
    log "FATAL: roster mode has no NEMARA_NODE_BASE_URL and could not derive one"
    log "       set TS_AUTHKEY (production), or pass NEMARA_NODE_BASE_URL explicitly"
    exit 1
  fi

  log "identity: roster mode (backend=$NEMARA_BACKEND_URL node=$NEMARA_NODE_ID at $NEMARA_NODE_BASE_URL)"
  exec /workspace/nemara/deploy/roster_agent.sh
fi

log "identity: static mode — NEMARA_BACKEND_URL unset, using NEMARA_IDENTITY_ROOMS"
: "${NEMARA_IDENTITY_ROOMS:?static mode needs NEMARA_IDENTITY_ROOMS, or set NEMARA_BACKEND_URL for roster mode}"

# ── 4b. static fallback: one process per room from the env list ────────
# Ports are 8012, 8013, ... in ROOMS order. Order is stable across restarts
# because it comes from the env string — a room that changed port would look
# to the backend like a new, empty home.
#
# cwd is the state dir on purpose: service.py hardcodes
# EventJournal(Path("out/identity_soak")), which is relative, so this puts the
# journal on the network volume instead of ephemeral container disk.
i=0
IFS=',' read -ra ROOMS <<< "$NEMARA_IDENTITY_ROOMS"
for room in "${ROOMS[@]}"; do
  room="$(echo "$room" | tr -d '[:space:]')"
  [ -z "$room" ] && continue
  port=$((8012 + i))
  patient="${room#patient_}"
  log "starting identity room=$room patient=$patient port=$port"
  mkdir -p "$STATE_DIR/$room"
  (
    while true; do
      cd "$STATE_DIR/$room" || exit 1
      # --patient explicitly: its argparse default is a hardcoded pilot UUID,
      # which on any other home would silently bind the wrong patient.
      python3 -m identity.service \
        --room "$room" --patient "$patient" --port "$port" \
        --pod-url "http://127.0.0.1:${PERCEPTION_PORT:-8000}" \
        >>"$LOGDIR/identity-$room.log" 2>&1
      rc=$?
      # 4 = pod wedged, 5 = camera walk exhausted. Both are deliberate
      # self-exits asking to be restarted (service.py:571,653-671).
      echo "$(date -u +%FT%TZ) identity room=$room exited rc=$rc — restart in 3s" \
        >>"$LOGDIR/identity-$room.log"
      sleep 3
    done
  ) &

  # In userspace mode tailscaled cannot reach a local listener by itself —
  # `serve` is what publishes it onto the tailnet.
  #
  # Run SYNCHRONOUSLY, not as a backgrounded `(sleep 15; ...) &`. That job
  # would be a child of this shell, and the `wait -n` below returns as soon
  # as ANY child exits — so the short-lived helper finished ~15s into every
  # boot, `wait -n` returned, and the container exited 1 and crash-looped.
  # `tailscale serve` is idempotent and cheap; there is nothing to defer.
  if [ -n "${TS_AUTHKEY:-}" ]; then
    tailscale --socket=/var/run/tailscale/tailscaled.sock \
      serve --bg --tcp "$port" "tcp://127.0.0.1:$port" \
      >>"$LOGDIR/tailscale.log" 2>&1 \
      || log "WARN could not publish port $port on the tailnet"
  fi

  i=$((i + 1))
done

log "up: perception + $i identity room(s), state=$STATE_DIR"

# If a SUPERVISED loop exits, fail the container so the platform restarts it
# rather than leaving a healthy-looking empty shell. Every remaining child is
# an infinite restart loop, so any exit here is genuinely abnormal.
wait -n
log "a supervised loop exited unexpectedly — failing the container"
exit 1
