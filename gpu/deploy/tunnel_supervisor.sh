#!/usr/bin/env bash
# Pod-tunnel supervisor (2026-07-28): the ssh -N -L tunnel is the stack's
# last unsupervised link and it zombies — process alive, forward dead
# (ServerAlive catches a dead TCP link, NOT a broken forward). Same
# philosophy as identity_supervisor: a dumb loop + an ACTIVE health probe;
# 3 consecutive probe failures = kill and relaunch. Colocation deletes
# this entire class in production.
# Usage: gpu/deploy/tunnel_supervisor.sh   (from nemara-backend; reads .pod.env)
set -u
cd "$(dirname "$0")/../.."
POD_HOST=$(grep '^POD_HOST=' .pod.env | cut -d= -f2)
POD_PORT=$(grep '^POD_PORT=' .pod.env | cut -d= -f2)
LOG="${TUNNEL_LOG:-/tmp/pod_tunnel.log}"
echo "$(date -u +%FT%TZ) tunnel supervisor: start host=$POD_HOST" >> "$LOG"
while true; do
  ssh -i .pod/runpod_key -p "$POD_PORT" \
      -o StrictHostKeyChecking=yes -o UserKnownHostsFile=.pod/pod_known_hosts \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
      -o ExitOnForwardFailure=yes -o ConnectTimeout=10 \
      -N -L 127.0.0.1:8000:localhost:8000 "root@$POD_HOST" >> "$LOG" 2>&1 &
  SSH_PID=$!
  fails=0
  while kill -0 "$SSH_PID" 2>/dev/null; do
    sleep 30
    if curl -s -m 5 -o /dev/null http://localhost:8000/health; then
      fails=0
    else
      fails=$((fails+1))
      if [ "$fails" -ge 3 ]; then
        echo "$(date -u +%FT%TZ) tunnel supervisor: forward dead (3 probe fails) — killing ssh $SSH_PID" >> "$LOG"
        kill "$SSH_PID" 2>/dev/null
        break
      fi
    fi
  done
  wait "$SSH_PID" 2>/dev/null
  echo "$(date -u +%FT%TZ) tunnel supervisor: ssh exited — relaunch in 3s" >> "$LOG"
  sleep 3
done
