#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# pod.sh — RunPod GPU lifecycle for the EdgeTAM-continuous perception.
#
# ONE controlled entry point for spinning the perception pod up/down and
# shipping code to it, so another engineer can operate it without
# re-deriving the RunPod API or the SSH hardening. Secrets are read from
# a gitignored .pod.env (copy .pod.env.example) — nothing sensitive lives
# in this script or in git.
#
# Usage:
#   scripts/pod.sh up                # start + rediscover endpoint + repin + wait healthy
#   scripts/pod.sh down              # alias for stop (billing off)
#   scripts/pod.sh status            # RunPod desiredStatus + $/hr
#   scripts/pod.sh start             # resume a STOPPED pod (billing on)
#   scripts/pod.sh stop              # stop the pod (billing off, disk kept)
#   scripts/pod.sh ssh "<cmd>"       # run a command on the pod
#   scripts/pod.sh deploy            # ship perception/ + relaunch server
#   scripts/pod.sh logs [N]          # tail N lines of the sam server log
#   scripts/pod.sh health            # recent continuous.tick cadence
#
# USE `up`, NOT `start`. RunPod assigns the direct-TCP endpoint only when
# the pod reaches RUNNING, and reassigns it across restarts — while the pod
# is stopped the API returns `runtime: null`, so POD_HOST/POD_PORT in
# .pod.env are stale the moment you stop. The pinned host key then no
# longer matches and ssh refuses to connect. That is the pinning working
# (RunPod recycles endpoints across tenants), not a bug, but it is why
# every manual bring-up "misconfigures something". `up` discovers the
# endpoint from the API, rewrites .pod.env and re-pins the host key, so
# the endpoint is never something a human has to remember.
#
# Cost note: `stop` halts billing and retains the container disk on RunPod
# Secure Cloud; `start`/`up` resumes it. As of 2026-08-10 the pod also has
# a 20 GB network volume (volumeInGb=20), which survives independently of
# the container disk — the older "volumeInGb=0, everything on the container
# disk" note in docs/runbooks/pod.md is stale. If the disk is ever reclaimed
# anyway, rebuild per that runbook §3 (~40 min, fully scripted).
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

# Rebuild SSHC after `up` rewrites the endpoint — the array above was
# built from the values sourced at startup, which are exactly the stale
# ones we just replaced.
_rebuild_sshc() {
  SSHC=(ssh -i "$POD_SSH_KEY" -p "$POD_PORT"
        -o StrictHostKeyChecking=yes
        -o UserKnownHostsFile="$POD_KNOWN_HOSTS"
        -o ConnectTimeout=25 "root@$POD_HOST")
}

# Ask RunPod for the CURRENT ssh endpoint. Empty until desiredStatus is
# RUNNING and the container has actually been placed, which is why the
# caller polls rather than asking once.
_discover_endpoint() {
  _gql "query { pod(input:{podId:\\\"$POD_ID\\\"}) { runtime { ports { ip publicPort privatePort type isIpPublic } } } }" \
  | python3 -c '
import json,sys
try: rt = (json.load(sys.stdin)["data"]["pod"] or {}).get("runtime")
except Exception: sys.exit(0)
for p in ((rt or {}).get("ports") or []):
    if p.get("privatePort") == 22 and p.get("type") == "tcp" and p.get("isIpPublic"):
        print(p["ip"], p["publicPort"]); break
'
}

# Rewrite one KEY=value in .pod.env, in place, keeping everything else.
_set_env_var() {  # $1=key $2=value
  python3 - "$ENV_FILE" "$1" "$2" <<'PY'
import pathlib, re, sys
p, key, val = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
lines = p.read_text().splitlines(keepends=True)
out, seen = [], False
for ln in lines:
    if re.match(rf'^{re.escape(key)}=', ln):
        out.append(f"{key}={val}\n"); seen = True
    else:
        out.append(ln)
if not seen:
    if out and not out[-1].endswith("\n"): out.append("\n")
    out.append(f"{key}={val}\n")
p.write_text("".join(out))
PY
}

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
    echo "NOTE: bare 'start' does not re-pin the endpoint — prefer 'up'." >&2
    _gql "mutation { podResume(input:{podId:\\\"$POD_ID\\\"}) { id desiredStatus } }"
    echo ;;

  up)
    # Resume can legitimately fail for a while. Stopping a Secure Cloud pod
    # RELEASES its GPU to the host; if the host reallocated it, podResume
    # returns "not enough free GPUs on the host machine" and the pod stays
    # EXITED until one frees up on that SAME machine. It is not a config
    # error and retrying is the only remedy, so poll rather than die — but
    # print the reason, because swallowing it is what makes this look like
    # a misconfiguration. RESUME_TRIES x 20s (default ~10 min).
    echo "1/4 resuming pod $POD_ID (billing resumes) ..."
    resumed=""; last_err=""
    for _ in $(seq 1 "${RESUME_TRIES:-30}"); do
      resp="$(_gql "mutation { podResume(input:{podId:\\\"$POD_ID\\\"}) { id desiredStatus } }")"
      if ! grep -q '"errors"' <<<"$resp"; then resumed=1; break; fi
      last_err="$(python3 -c '
import json,sys
try: print(json.load(sys.stdin)["errors"][0]["message"])
except Exception: print("unknown error")' <<<"$resp")"
      echo "    waiting: $last_err"
      sleep 20
    done
    if [[ -z "$resumed" ]]; then
      echo >&2
      echo "pod.sh up: could not resume. RunPod says: $last_err" >&2
      if grep -qi "free GPUs" <<<"$last_err"; then
        echo >&2
        echo "  This is GPU contention on the host, NOT a broken config." >&2
        echo "  A stopped Secure Cloud pod holds no GPU reservation. Options:" >&2
        echo "    - keep retrying ('pod.sh up' again, or RESUME_TRIES=90)" >&2
        echo "    - rebuild on a different host: docs/runbooks/pod.md §3 (~40 min)" >&2
        echo "    - to avoid this entirely, stop stopping it (see the cost note)" >&2
      fi
      exit 1
    fi
    echo "2/4 waiting for RunPod to publish an ssh endpoint (up to 180s) ..."
    ep=""
    for _ in $(seq 1 36); do
      ep="$(_discover_endpoint)"
      [[ -n "$ep" ]] && break
      sleep 5
    done
    if [[ -z "$ep" ]]; then
      echo "pod.sh up: no ssh endpoint after 180s. 'pod.sh status' then retry;" >&2
      echo "  if desiredStatus is RUNNING but no port appears, the pod may be" >&2
      echo "  stuck placing — stop it and start again." >&2
      exit 1
    fi
    NEW_HOST="${ep%% *}"; NEW_PORT="${ep##* }"
    if [[ "$NEW_HOST" == "$POD_HOST" && "$NEW_PORT" == "$POD_PORT" ]]; then
      echo "    endpoint unchanged ($NEW_HOST:$NEW_PORT) — keeping the existing pin"
    else
      echo "    endpoint moved: $POD_HOST:$POD_PORT -> $NEW_HOST:$NEW_PORT"
      cp "$ENV_FILE" "$ENV_FILE.bak-$(date -u +%Y%m%d%H%M%S)"
      _set_env_var POD_HOST "$NEW_HOST"
      _set_env_var POD_PORT "$NEW_PORT"
      POD_HOST="$NEW_HOST"; POD_PORT="$NEW_PORT"
      echo "3/4 re-pinning host key (RunPod recycles endpoints across tenants) ..."
      # Back up rather than append: a stale pin for a REUSED ip:port is
      # exactly the collision StrictHostKeyChecking is meant to catch, and
      # appending would leave both keys trusted.
      [[ -f "$POD_KNOWN_HOSTS" ]] && cp "$POD_KNOWN_HOSTS" "$POD_KNOWN_HOSTS.bak-$(date -u +%Y%m%d%H%M%S)"
      for _ in $(seq 1 12); do
        if ssh-keyscan -T 10 -p "$NEW_PORT" "$NEW_HOST" 2>/dev/null > "$POD_KNOWN_HOSTS.tmp" \
           && [[ -s "$POD_KNOWN_HOSTS.tmp" ]]; then
          mv "$POD_KNOWN_HOSTS.tmp" "$POD_KNOWN_HOSTS"; break
        fi
        sleep 5
      done
      rm -f "$POD_KNOWN_HOSTS.tmp"
      if [[ ! -s "$POD_KNOWN_HOSTS" ]]; then
        echo "pod.sh up: ssh-keyscan produced nothing — sshd not up yet. Re-run 'pod.sh up'." >&2
        exit 1
      fi
    fi
    _rebuild_sshc
    echo "4/4 waiting for sshd + perception /health (up to 180s) ..."
    ok=""
    for _ in $(seq 1 36); do
      if "${SSHC[@]}" 'curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:8000/health' 2>/dev/null | grep -q 200; then
        ok=1; break
      fi
      sleep 5
    done
    echo
    if [[ -n "$ok" ]]; then
      echo "pod is UP and perception is answering on :8000"
    else
      echo "pod is reachable but perception did not answer /health in 180s."
      echo "  the supervisor may still be loading checkpoints — 'pod.sh logs 100'."
      echo "  if it never comes up, 'pod.sh deploy' relaunches the supervisor."
    fi
    echo
    echo "  endpoint : $POD_HOST:$POD_PORT   (written to .pod.env, host key re-pinned)"
    echo "  next     : gpu/deploy/tunnel_supervisor.sh        # backend -> pod :8000"
    echo "             NEMARA_IDENTITY_ROOM=patient_<uuid> \\"
    echo "               gpu/deploy/identity_supervisor.sh    # identity :8012"
    echo "             scripts/pod.sh health                  # confirm ticks"
    ;;

  stop|down)
    echo "stopping pod $POD_ID (billing stops, container disk retained) ..."
    _gql "mutation { podStop(input:{podId:\\\"$POD_ID\\\"}) { id desiredStatus } }"
    echo
    echo "note: POD_HOST/POD_PORT in .pod.env are now stale by definition."
    echo "      bring it back with 'pod.sh up', which rediscovers and re-pins."
    ;;

  ssh)
    "${SSHC[@]}" "${2:?usage: pod.sh ssh \"<cmd>\"}" 2>&1 \
      | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  deploy)
    # Ship the live perception code and relaunch the supervisor on the
    # edgetam venv (one env hosts sam2 + sam3 + faces). Mirrors
    # scripts/deploy_continuous_remote.sh, kept here so deploy is one cmd.
    echo "shipping perception/ -> pod ..."
    tar czf - -C "$HERE/gpu" perception 2>/dev/null \
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
    echo "usage: pod.sh {up|down|status|start|stop|ssh <cmd>|deploy|logs [N]|health}" >&2
    exit 2 ;;
esac
