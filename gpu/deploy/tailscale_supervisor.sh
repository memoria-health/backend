#!/usr/bin/env bash
# Tailscale supervisor for the GPU host — restart-on-exit, same
# file-based discipline as identity_supervisor.sh and the perception
# supervisor.
#
# WHY THIS EXISTS (2026-08-11): the backend reaches identity over the
# tailnet, so tailscaled is now a production dependency. An unsupervised
# daemon that dies takes /state with it and the backend silently loses
# every seat — the same failure shape as the 11:57 identity OOM, which
# is what motivated supervising that one.
#
# USERSPACE NETWORKING, deliberately. RunPod pods are containers and
# generally have no /dev/net/tun, so the kernel path is unavailable.
# Userspace mode is enough here because the GPU host only ever accepts
# INBOUND connections (Railway calls /state, CI ssh's in); it never
# dials out over the tailnet. If a future caller needs outbound, it goes
# through the SOCKS5 proxy on :1055, not through this file.
#
# Usage (on the pod):
#   TS_AUTHKEY=tskey-auth-... gpu/deploy/tailscale_supervisor.sh
# Optional:
#   TS_HOSTNAME  (default nemara-pod)   the MagicDNS name Railway targets
#   TS_LOG       (default /tmp/tailscaled.log)
#   TS_STATE     (default /var/lib/tailscale/tailscaled.state)
set -u

: "${TS_AUTHKEY:?set TS_AUTHKEY (tagged, reusable) — see infra/tailscale-acl.hujson}"
HOSTNAME_="${TS_HOSTNAME:-nemara-pod}"
LOG="${TS_LOG:-/tmp/tailscaled.log}"
STATE="${TS_STATE:-/var/lib/tailscale/tailscaled.state}"
SOCK=/var/run/tailscale/tailscaled.sock

mkdir -p "$(dirname "$STATE")" "$(dirname "$SOCK")"

echo "$(date -u +%FT%TZ) tailscale supervisor: start host=$HOSTNAME_" >> "$LOG"

while true; do
  # --tun=userspace-networking: no /dev/net/tun in a RunPod container.
  # The two outbound proxies are only used if something on this host ever
  # needs to dial OUT over the tailnet; harmless otherwise and cheaper than
  # rebuilding the daemon later to add them.
  #
  # DIFFERENT PORTS. Both flags previously named localhost:1055, which is
  # two listeners on one address — the second bind fails, and it fails
  # inside a supervised restart loop where it reads as a daemon that will
  # not stay up rather than as a flag conflict.
  tailscaled \
    --tun=userspace-networking \
    --socks5-server=localhost:1056 \
    --outbound-http-proxy-listen=localhost:1055 \
    --state="$STATE" \
    --socket="$SOCK" >> "$LOG" 2>&1 &
  DAEMON=$!

  # `tailscale up` is idempotent: on a restart with existing state it
  # re-uses the node identity rather than burning another key slot.
  # --advertise-tags is what makes the ACL apply; without it the node
  # lands under a human identity and every rule above misses it.
  sleep 2
  tailscale --socket="$SOCK" up \
    --authkey="$TS_AUTHKEY" \
    --hostname="$HOSTNAME_" \
    --advertise-tags=tag:gpu \
    --accept-dns=false \
    --reset >> "$LOG" 2>&1 || \
    echo "$(date -u +%FT%TZ) tailscale up FAILED — see $LOG" >> "$LOG"

  # ACTIVE probe, not just "is the process alive". A tailnet node can be
  # up as a process while its connection is dead — the same zombie the
  # ssh tunnel supervisor was written for. `tailscale status` reporting
  # a self address is the cheapest real liveness signal.
  fails=0
  while kill -0 "$DAEMON" 2>/dev/null; do
    sleep 30
    if tailscale --socket="$SOCK" status --json 2>/dev/null | grep -q '"Online": *true'; then
      fails=0
    else
      fails=$((fails + 1))
      if [ "$fails" -ge 3 ]; then
        echo "$(date -u +%FT%TZ) tailscale supervisor: offline x3 — restarting daemon" >> "$LOG"
        kill "$DAEMON" 2>/dev/null
        break
      fi
    fi
  done

  wait "$DAEMON" 2>/dev/null
  echo "$(date -u +%FT%TZ) tailscale supervisor: daemon exited — relaunch in 3s" >> "$LOG"
  sleep 3
done
