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
#   scripts/pod.sh release <tag>     # create a NEW node from a GHCR image tag
#   scripts/pod.sh up                # start + rediscover endpoint + repin + wait healthy
#   scripts/pod.sh down              # alias for stop (billing off)
#   scripts/pod.sh status            # RunPod desiredStatus + $/hr
#   scripts/pod.sh start             # resume a STOPPED pod (billing on)
#   scripts/pod.sh stop              # stop the pod (billing off, disk kept)
#   scripts/pod.sh ssh "<cmd>"       # run a command on the pod
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
# set -a: source EXPORTS everything, because release's request body is
# built by a python3 child that reads os.environ — plain `source` leaves
# the vars shell-local and the child silently sees none of them. The
# first real release (2026-08-12) shipped a pod whose only env var was
# RunPod's own PUBLIC_KEY this way; the shell-side guards all passed
# because the shell could see what the child could not.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

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

# REST (not GraphQL) — pod CREATION lives on rest.runpod.io/v1.
_rest() {  # $1=method $2=path $3=json-body-or-empty
  local m="$1" p="$2" b="${3:-}"
  if [[ -n "$b" ]]; then
    # Body goes in on STDIN (`@-`), not as `-d "$b"`. The create body carries
    # the node's whole environment — DB URL, LiveKit secret, HF token — and
    # an argument is visible in `ps` to every other process on this machine
    # for as long as the request runs. This is the same rule the release
    # command follows for its own inputs a few lines below.
    printf '%s' "$b" | curl -s --max-time 60 -X "$m" "https://rest.runpod.io/v1$p" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H "Content-Type: application/json" --data-binary @-
  else
    curl -s --max-time 60 -X "$m" "https://rest.runpod.io/v1$p" \
      -H "Authorization: Bearer $RUNPOD_API_KEY"
  fi
}

case "${1:-}" in
  release)
    # Create a NEW node from an image tag. This is the `railway up` of the
    # GPU side: the artifact is the tag, and a lost node is replaced rather
    # than repaired. It does NOT touch the existing pod — terminating that is
    # a separate, deliberate act (see the note it prints at the end).
    TAG="${2:?usage: pod.sh release <image-tag>   e.g. pod.sh release gpu-v1}"
    : "${GHCR_IMAGE:?set GHCR_IMAGE in .pod.env, e.g. ghcr.io/<owner>/memoria-gpu}"
    : "${RUNPOD_REGISTRY_AUTH_ID:?set RUNPOD_REGISTRY_AUTH_ID in .pod.env — the image is private; save GHCR creds in RunPod (Settings > Container Registry Auth) and put the id here}"
    : "${RUNPOD_NETWORK_VOLUME_ID:?set RUNPOD_NETWORK_VOLUME_ID in .pod.env — without a network volume RunPod TERMINATES the pod (not stops it) at \$0 balance and the data is unrecoverable; this is how pod #1 died}"
    : "${RUNPOD_DATACENTER_ID:?set RUNPOD_DATACENTER_ID in .pod.env — a network volume is pinned to its datacenter, so the node must be created there}"

    IMAGE="$GHCR_IMAGE:$TAG"
    NAME="${RUNPOD_POD_NAME:-nemara-gpu}-$TAG"
    GPU_TYPES="${RUNPOD_GPU_TYPES:-\"NVIDIA A40\"}"

    # Node env. Deliberately assembled here rather than baked: these are
    # per-deployment secrets and per-node placement, and the image must stay
    # free of both. Sourced from .pod.env so nothing lands on a command line
    # that shows up in shell history or `ps`.
    for v in NEMARA_DB_URL NEMARA_LIVEKIT_URL NEMARA_LIVEKIT_API_KEY \
             NEMARA_LIVEKIT_API_SECRET NEMARA_IDENTITY_STATE_TOKEN \
             HUGGINGFACE_TOKEN; do
      [[ -n "${!v:-}" ]] || { echo "pod.sh release: $v not set in $ENV_FILE" >&2; exit 2; }
    done

    # ── roster mode is the only thing worth releasing ──────────────────
    # The entrypoint falls back to a STATIC room list when NEMARA_BACKEND_URL
    # is unset, and that fallback is for a local `docker run`, not for a node
    # on real homes: a statically-pinned node cannot be scaled, cannot be
    # failed over, and holds its rooms even after the backend has reassigned
    # them elsewhere — two nodes serving one home, both writing seats.
    #
    # Releasing that by accident is easy and silent, so it is refused here.
    # NEMARA_IDENTITY_ROOMS is no longer required (and is not forwarded):
    # in roster mode the backend decides, and shipping a stale list alongside
    # it only invites the two to disagree.
    for v in NEMARA_BACKEND_URL NEMARA_INTERNAL_SERVICE_KEY; do
      [[ -n "${!v:-}" ]] || {
        echo "pod.sh release: $v not set in $ENV_FILE" >&2
        echo "  a node released without it boots the STATIC fallback: it never polls" >&2
        echo "  the roster, so it cannot be scaled or failed over." >&2
        exit 2; }
    done
    # TS_AUTHKEY is not optional in roster mode either — the node derives the
    # base_url the backend dials from its tailnet name, and without a tailnet
    # it has no reachable address to register and refuses to start.
    [[ -n "${TS_AUTHKEY:-}" ]] || {
      echo "pod.sh release: TS_AUTHKEY not set in $ENV_FILE" >&2
      echo "  the backend reaches identity over the tailnet; without it the node" >&2
      echo "  has no address to register and will exit on boot." >&2
      exit 2; }

    # NEMARA_NODE_ID and NEMARA_NODE_BASE_URL are deliberately NOT sent: the
    # pod id does not exist until the API call below returns, and the tailnet
    # name is not assigned until tailscaled authenticates. The entrypoint
    # derives both at boot, when they are actually knowable.

    # Exported so the heredoc's python can read them; they are shell locals
    # otherwise. Secrets go via the environment, never argv — argv is visible
    # in `ps` to anything else on the machine.
    export NAME IMAGE GPU_TYPES RUNPOD_NETWORK_VOLUME_ID \
           RUNPOD_DATACENTER_ID RUNPOD_REGISTRY_AUTH_ID

    BODY="$(python3 - <<'PY'
import json, os
env = {k: os.environ[k] for k in (
    "NEMARA_DB_URL", "NEMARA_LIVEKIT_URL", "NEMARA_LIVEKIT_API_KEY",
    "NEMARA_LIVEKIT_API_SECRET", "NEMARA_IDENTITY_STATE_TOKEN",
    "NEMARA_BACKEND_URL", "NEMARA_INTERNAL_SERVICE_KEY", "TS_AUTHKEY",
    "HUGGINGFACE_TOKEN",
) if os.environ.get(k)}
for opt in ("NEMARA_PERCEPTION_TOKEN", "NEMARA_PERCEPTION_CONCURRENCY",
            "NEMARA_NODE_CAPACITY", "NEMARA_ROSTER_POLL_S", "TS_HOSTNAME",
            # BLAS/OMP thread caps, forwardable for experiments. A/B
            # tested 2026-08-13 (=2, 16 vCPUs, ~1260 threads): NOT the
            # CPU wall — no change at 1-2 cams, worse at 3. Leave unset
            # normally; the wall was codec+CPU-face work, fixed in
            # gpu-v2 (raw wire + cuDNN). NO apostrophes in this file:
            # macOS bash 3.2 breaks on them inside $(heredoc) blocks.
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            # gpu-v2 revert knobs: frame wire back to JPEG, and the
            # faces-must-be-on-GPU boot guard override.
            "NEMARA_FRAME_WIRE", "NEMARA_FACE_REQUIRE_GPU",
            # SSH pubkey for the entrypoint sshd. RunPod injects
            # PUBLIC_KEY only into console-created pods; API-created
            # pods must carry it themselves (learned on node G,
            # 2026-08-13 - no key, no sshd, no shell when it mattered).
            "PUBLIC_KEY",
            # Capacity-experiment knobs (2026-08-14): SAM3 re-ground
            # cadence and tracking fps. Unset = proven defaults (5, 3.0).
            "NEMARA_EDGE_REGROUND", "NEMARA_TARGET_FPS"):
    if os.environ.get(opt):
        env[opt] = os.environ[opt]
print(json.dumps({
    "name": os.environ["NAME"],
    "imageName": os.environ["IMAGE"],
    "cloudType": "SECURE",          # Community/Spot can be reclaimed mid-soak
    "computeType": "GPU",
    "gpuTypeIds": json.loads("[" + os.environ["GPU_TYPES"] + "]"),
    "gpuCount": 1,
    "gpuTypePriority": "availability",
    # Hosts must run a driver for our CUDA 12.4 base. Two community
    # 3090s placed without this filter came up cuda:false (torch saw no
    # GPU at all, 2026-08-14) - old-driver hosts. Harmless on secure
    # cloud, essential on community.
    "allowedCudaVersions": ["12.4", "12.5", "12.6", "12.7", "12.8"],
    "containerDiskInGb": 70,
    # Mounted at /persist, NOT /workspace — the image keeps sam3, edgetam and
    # our code under /workspace and a volume there would shadow all of it.
    "volumeMountPath": "/persist",
    "networkVolumeId": os.environ["RUNPOD_NETWORK_VOLUME_ID"],
    "dataCenterIds": [os.environ["RUNPOD_DATACENTER_ID"]],
    "containerRegistryAuthId": os.environ["RUNPOD_REGISTRY_AUTH_ID"],
    "ports": ["22/tcp"],
    "interruptible": False,
    "env": env,
}))
PY
)" || { echo "pod.sh release: failed to build request body" >&2; exit 1; }

    echo "creating node from $IMAGE ..."
    RESP="$(_rest POST /pods "$BODY")"
    NEW_ID="$(python3 -c '
import json,sys
try:
    d = json.loads(sys.stdin.read())
except Exception:
    sys.exit(0)
print(d.get("id") or "")
' <<<"$RESP")"
    if [[ -z "$NEW_ID" ]]; then
      echo "pod.sh release: create failed" >&2
      echo "$RESP" | head -c 800 >&2; echo >&2
      exit 1
    fi
    echo "  new pod id: $NEW_ID"
    echo
    echo "next:"
    echo "  1. set POD_ID=$NEW_ID in $ENV_FILE"
    echo "  2. scripts/pod.sh up          # discovers the endpoint, re-pins the host key"
    echo "  3. scripts/pod.sh health      # confirm ticks"
    echo
    echo "the OLD pod ($POD_ID) is untouched and still billing if RUNNING."
    echo "terminate it deliberately once the new node is verified."
    ;;

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
      echo "  the node may still be fetching checkpoints — 'pod.sh logs 100'."
      echo "  first boot pulls ~3.8GB onto the network volume and is slow."
    fi
    echo
    echo "  endpoint : $POD_HOST:$POD_PORT   (written to .pod.env, host key re-pinned)"
    echo "  next     : nothing to launch by hand. The node polls the backend"
    echo "             for its homes and starts one identity process per room;"
    echo "             the backend reaches them over the tailnet, not a tunnel."
    echo "             scripts/pod.sh health                  # confirm ticks"
    echo "             curl \$NEMARA_BACKEND_URL/ready         # confirm the fleet"
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
    # REMOVED 2026-08-11 along with scripts/deploy_continuous_remote.sh.
    #
    # It rsynced perception/ onto a live pod and restarted the supervisor in
    # place, which meant the running code was whatever the last person had
    # pushed from their laptop — no tag, no record, and not reproducible on
    # a replacement node. That is precisely how the pod became a pet that
    # could not be rebuilt when its GPU was reallocated.
    #
    # The node is now an immutable image: build a tag, release a node from
    # it, and replace rather than patch.
    echo "pod.sh deploy: removed — the node is built from an image now." >&2
    echo >&2
    echo "  1. git tag gpu-vN && git push origin gpu-vN   # CI builds + pushes" >&2
    echo "  2. scripts/pod.sh release gpu-vN              # new node from that tag" >&2
    echo "  3. scripts/pod.sh up                          # endpoint + host key" >&2
    echo >&2
    echo "to iterate WITHOUT a release, ssh in and edit under /workspace —" >&2
    echo "but know that the next node will not have your change." >&2
    exit 2 ;;

  logs)
    "${SSHC[@]}" "tail -n ${2:-60} /workspace/sam_server.log" 2>&1 \
      | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  health)
    "${SSHC[@]}" "date -u; echo '--- last 12 ticks ---'; grep continuous.tick /workspace/sam_server.log | tail -12" \
      2>&1 | grep -viE "Warning: Permanently added|post-quantum|store now" ;;

  *)
    echo "usage: pod.sh {release <tag>|up|down|status|start|stop|ssh <cmd>|logs [N]|health}" >&2
    exit 2 ;;
esac
