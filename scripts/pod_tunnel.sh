#!/usr/bin/env bash
# Self-healing pod tunnel (2026-07-25 — a bare `ssh -N -L` died silently
# mid-soak and the identity service ran blind against localhost:8000).
# Reads .pod.env; reconnects forever with backoff. Run detached:
#
#   nohup scripts/pod_tunnel.sh >/tmp/pod_tunnel.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .pod.env
while true; do
  echo "$(date '+%F %T') tunnel: connecting to ${POD_HOST}:${POD_PORT}"
  ssh -i .pod/runpod_key -p "${POD_PORT}" \
    -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=.pod/pod_known_hosts \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
    -N -L 127.0.0.1:8000:localhost:8000 "root@${POD_HOST}"
  echo "$(date '+%F %T') tunnel: dropped (rc=$?) — retry in 5s"
  sleep 5
done
