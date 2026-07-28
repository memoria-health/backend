#!/usr/bin/env bash
# Identity-service supervisor — restart-on-exit, same file-based
# discipline as the perception pod's supervisor.
#
# WHY (2026-07-26): the identity service was OOM-killed at 11:57 after
# ~10h and simply stayed dead — the seat provider for the entire
# backend, gone silently. A supervised service turns that class of
# failure into a ~3s blip plus a log line. (The OOM cause itself —
# the unbounded experiment queue — is also fixed, STREAM_CAPACITY=2.)
#
# Usage (dev Mac or GPU host):
#   NEMARA_IDENTITY_ROOM=patient_<uuid> gpu/deploy/identity_supervisor.sh
# Optional: NEMARA_IDENTITY_LOG (default /tmp/identity_service.log),
#           NEMARA_IDENTITY_PORT (default 8012),
#           NEMARA_IDENTITY_TOKEN (enables /state auth — see
#           gpu/identity/contract.py).
set -u

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOM="${NEMARA_IDENTITY_ROOM:?set NEMARA_IDENTITY_ROOM (e.g. patient_<uuid>)}"
LOG="${NEMARA_IDENTITY_LOG:-/tmp/identity_service.log}"
PORT="${NEMARA_IDENTITY_PORT:-8012}"

echo "$(date -u +%FT%TZ) identity supervisor: start room=$ROOM port=$PORT" >> "$LOG"
while true; do
  uv run --directory "$REPO" python gpu/identity/service.py \
    --room "$ROOM" --port "$PORT" >> "$LOG" 2>&1
  rc=$?  # capture BEFORE the echo — $(date) inside it resets $? to 0
  echo "$(date -u +%FT%TZ) identity supervisor: service exited rc=$rc — restart in 3s" >> "$LOG"
  sleep 3
done
