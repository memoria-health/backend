#!/usr/bin/env bash
# Backend container entrypoint (Railway).
#
# Brings up the tailnet leg, then execs uvicorn. Everything here is
# CONDITIONAL on TS_AUTHKEY, so a plain `docker run` and the local compose
# stack behave exactly as they did before this file existed.
#
# WHY THE BACKEND NEEDS A TAILNET AT ALL
# Identity runs on GPU nodes whose only stable address is a tailnet name —
# stable precisely because RunPod reassigns a pod's direct-TCP endpoint on
# every restart, and because those ports must not be exposed to the public
# internet. The backend is the client, so it has to be on the tailnet too.
#
# USERSPACE NETWORKING: a Railway container has no /dev/net/tun, so the
# kernel path is unavailable. Userspace is sufficient because this side only
# ever dials OUT; nothing on the tailnet connects to the backend.
#
# ENV
#   TS_AUTHKEY   tagged, reusable auth key. Unset = tailscale is skipped.
#   TS_HOSTNAME  tailnet name for this container (default nemara-backend)
#   PORT         injected by Railway; 8002 locally
set -uo pipefail

log() { echo "$(date -u +%FT%TZ) backend-entrypoint: $*"; }

SOCK=/var/run/tailscale/tailscaled.sock
PROXY_PORT=1055

if [ -n "${TS_AUTHKEY:-}" ]; then
  mkdir -p /var/run/tailscale /var/lib/tailscale

  # --outbound-http-proxy-listen, not just --socks5-server. httpx needs the
  # `socksio` package to speak socks5:// and we do not ship it, so a SOCKS
  # proxy would raise ImportError on the first identity call — at runtime,
  # in production, on a path that otherwise fails open and stays quiet.
  # An HTTP proxy is native to httpx.
  #
  # It also solves DNS: MagicDNS names do not resolve through the
  # container's resolver, but a CONNECT carries the hostname and tailscaled
  # resolves it on the other side.
  tailscaled \
    --tun=userspace-networking \
    --socks5-server=localhost:$((PROXY_PORT + 1)) \
    --outbound-http-proxy-listen=localhost:${PROXY_PORT} \
    --state=/var/lib/tailscale/tailscaled.state \
    --socket="$SOCK" &
  TS_PID=$!

  sleep 2
  # --advertise-tags is what makes infra/tailscale-acl.hujson apply. Without
  # it the node joins under a human identity and every rule there misses it,
  # so the grant to tag:gpu:8012-8111 would not cover this container.
  tailscale --socket="$SOCK" up \
    --authkey="$TS_AUTHKEY" \
    --hostname="${TS_HOSTNAME:-nemara-backend}" \
    --advertise-tags=tag:backend \
    --accept-dns=false \
    || log "WARN tailscale up failed — identity will be unreachable"

  # Wait for the connection to actually be usable. Booting uvicorn first
  # would mean the first ticks after every deploy find identity unreachable
  # and fail open, which reads downstream as a patient who is briefly
  # unidentifiable rather than as a backend that is not ready yet.
  for _ in $(seq 1 30); do
    if tailscale --socket="$SOCK" status --json 2>/dev/null \
         | grep -q '"Online": *true'; then
      log "tailnet up as ${TS_HOSTNAME:-nemara-backend}"
      break
    fi
    sleep 1
  done

  # Default the identity proxy rather than requiring it to be set alongside
  # TS_AUTHKEY. Forgetting it would leave identity calls going out directly,
  # failing, and failing OPEN — homes silently unidentified with a healthy
  # looking backend. Explicit config still wins.
  if [ -z "${NEMARA_IDENTITY_PROXY_URL:-}" ]; then
    export NEMARA_IDENTITY_PROXY_URL="http://localhost:${PROXY_PORT}"
    log "identity proxy defaulted to $NEMARA_IDENTITY_PROXY_URL"
  fi

  # Supervise it: if tailscaled dies, the backend loses identity for every
  # home at once and nothing else would notice. Exiting lets the platform
  # restart the container, which is the behaviour we actually want.
  (
    while kill -0 "$TS_PID" 2>/dev/null; do sleep 15; done
    log "FATAL tailscaled exited — killing the container so it is restarted"
    kill -TERM 1
  ) &
else
  log "TS_AUTHKEY unset — no tailnet; identity reachable only if directly routable"
fi

exec uv run uvicorn src.main:app --host 0.0.0.0 --port "${PORT:-8002}"
