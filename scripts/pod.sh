#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# pod.sh — RunPod GPU lifecycle for the EdgeTAM-continuous sam_service.
#
# ONE controlled entry point for spinning the perception pod up/down and
# shipping code to it, so another engineer can operate it without
# re-deriving the RunPod API or the SSH hardening. Secrets are read from
# a gitignored .pod.env (copy .pod.env.example) — nothing sensitive lives
# in this script or in git.
#
# Usage:
#   scripts/pod.sh status            # RunPod desiredStatus + $/hr
#   scripts/pod.sh start             # resume a STOPPED pod (billing on)
#   scripts/pod.sh stop              # stop the pod (billing off, disk kept)
#   scripts/pod.sh ssh "<cmd>"       # run a command on the pod
#   scripts/pod.sh deploy            # ship sam_service/ + relaunch server
#   scripts/pod.sh logs [N]          # tail N lines of the sam server log
#   scripts/pod.sh health            # recent continuous.tick cadence
#
# Cost note: the pod has NO persistent network volume (volumeInGb=0), so
# everything lives on the container disk. `stop` retains that disk on
# RunPod Secure Cloud and `start` resumes it; if the disk is ever
# reclaimed, rebuild with docs/POD_RUNBOOK.md (setup + deploy, ~1h).
# ─────────────────────────────────────────────────────────────────────
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${POD_ENV_FILE:-$HERE/.pod.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "pod.sh: missing $ENV_FILE — copy .pod.env.example and fill it in." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${RUNPOD_API_KEY:?set in .pod.env}"
: "${POD_ID:?set in .pod.env}"
: "${POD_HOST:?set in .pod.env}"
: "${POD_PORT:?set in .pod.env}"
: "${POD_SSH_KEY:?set in .pod.env}"
: "${POD_KNOWN_HOSTS:?set in .pod.env}"

GQL="https://api.runpod.io/graphql"
# Host key PINNED (security review 2026-07-20): RunPod recycles direct-TCP
# endpoints across tenants, so StrictHostKeyChecking=no would hand patient
# frames to whoever owns this IP:port next. POD_KNOWN_HOSTS is captured
# once with ssh-keyscan at pod setup; a NEW pod means re-scan.
SSHC=(ssh -i "$POD_SSH_KEY" -p "$POD_PORT"
      -o StrictHostKeyChecking=yes
      -o UserKnownHostsFile="$POD_KNOWN_HOSTS"
      -o ConnectTimeout=25 "root@$POD_HOST")

_gql() {  # $1 = graphql query/mutation string
  curl -s --max-time 30 -X POST "$GQL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -d "{\"query\":\"$1\"}"
}

case "${1:-}" in
  status)
    _gql "query { pod(input:{podId:\\\"$POD_ID\\\"}) { id name desiredStatus volumeInGb containerDiskInGb costPerHr machine { gpuDisplayName } } }"
    echo ;;

  start|resume)
    echo "resuming pod $POD_ID (billing resumes) ..."
    _gql "mutation { podResume(input:{podId:\\\"$POD_ID\\\"}) { id desiredStatus } }"
    echo ;;

  stop)
    echo "stopping pod $POD_ID (billing stops, container disk retained) ..."
    _gql "mutation { podStop(input:{podId:\\\"$POD_ID\\\"}) { id desiredStatus } }"
    echo ;;

  ssh)
    "${SSHC[@]}" "${2:?usage: pod.sh ssh \"<cmd>\"}" 2>&1 \
      | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  deploy)
    # Ship the live perception code and relaunch the supervisor on the
    # edgetam venv (one env hosts sam2 + sam3 + faces). Mirrors
    # scripts/deploy_continuous_remote.sh, kept here so deploy is one cmd.
    echo "shipping sam_service/ -> pod ..."
    tar czf - -C "$HERE" sam_service 2>/dev/null \
      | "${SSHC[@]}" "cd /workspace/nemara && tar xzf - && echo shipped" \
      2>&1 | grep -viE "Warning: Permanently added|post-quantum|store now"
    echo "relaunching continuous supervisor ..."
    "${SSHC[@]}" 'bash -s' < "$HERE/scripts/deploy_continuous_remote.sh" \
      2>&1 | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  logs)
    "${SSHC[@]}" "tail -n ${2:-60} /workspace/sam_server.log" 2>&1 \
      | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  health)
    "${SSHC[@]}" "date -u; echo '--- last 12 ticks ---'; grep continuous.tick /workspace/sam_server.log | tail -12" \
      2>&1 | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  *)
    echo "usage: pod.sh {status|start|stop|ssh <cmd>|deploy|logs [N]|health}" >&2
    exit 2 ;;
esac
