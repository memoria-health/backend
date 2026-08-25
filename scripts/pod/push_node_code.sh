#!/bin/bash
# Push the node's code to a fleet node's dev-code overlay and restart it.
#
# WHAT GETS SHIPPED: gpu/{identity,perception,deploy} AND src/. Both,
# always. gpu/identity/gallery.py imports src.services.identity.trace,
# so shipping one without the other produces a node running two
# different commits — which is exactly what happened on 2026-08-23 and
# broke every identity probe while /state still reported `ready`.
#
# Layout matches the IMAGE (Dockerfile 224-233 + entrypoint
# dev_code_overlay): /persist/dev-code/{nemara/{identity,perception,
# deploy},src}. An older scratchpad script nested a gpu/ dir the overlay
# never read, so every "push" since the v3 image was a silent no-op.
#
# tar, NOT rsync: the fleet nodes have no rsync (the image is CUDA +
# python, nothing else), and the previous version's rsync failed with
# "command not found" on the far side.
#
# The staged tree is MIRRORED — each directory is removed before extract
# — so a file deleted locally is deleted on the node. Additive copies
# leave dead modules importable, which is how the deleted Pi path
# survived on a node for a week.
#
# Usage: scripts/pod/push_node_code.sh <ssh_host> <ssh_port> <node_tailnet_ip>
#        (ssh_host/port: any pod mounting the volume; the node itself is
#         fine. node_tailnet_ip is where the identity ports are reached.)
set -euo pipefail
# /../.. — this lives in scripts/pod/, so one level up is scripts/.
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HOST=$1; PORT=$2; NODE=$3
SSH=(ssh -o StrictHostKeyChecking=no -p "$PORT")

# COPYFILE_DISABLE: macOS otherwise packs AppleDouble "._name" siblings,
# which land next to every module on the node.
COPYFILE_DISABLE=1 tar czf - --no-xattrs \
  --exclude='__pycache__' --exclude='*.pyc' \
  -C "$REPO/gpu" identity perception deploy \
  | "${SSH[@]}" "root@$HOST" '
      set -e
      mkdir -p /persist/dev-code/nemara
      rm -rf /persist/dev-code/nemara/identity \
             /persist/dev-code/nemara/perception \
             /persist/dev-code/nemara/deploy
      tar xzf - --no-same-owner -C /persist/dev-code/nemara
      find /persist/dev-code/nemara -name "._*" -delete'

COPYFILE_DISABLE=1 tar czf - --no-xattrs \
  --exclude='__pycache__' --exclude='*.pyc' \
  -C "$REPO" src \
  | "${SSH[@]}" "root@$HOST" '
      set -e
      mkdir -p /persist/dev-code
      rm -rf /persist/dev-code/src
      tar xzf - --no-same-owner -C /persist/dev-code
      find /persist/dev-code/src -name "._*" -delete'

echo "staged: /persist/dev-code/{nemara/{identity,perception,deploy},src}"

# The overlay is applied by the SERVICE on (re)launch, so restarting the
# identity processes is what makes the push take effect. deploy/ and the
# src/ mirror need a container restart — the roster agent is the
# container's foreground process and bash reads its own script lazily,
# so replacing it under a running agent is not safe.
STOKEN=$(grep '^NEMARA_IDENTITY_STATE_TOKEN' "$REPO/.env" | cut -d= -f2 | tr -d "'\"")
for p in 8012 8013; do
  curl -s -m 5 -X POST -H "Authorization: Bearer $STOKEN" \
    "http://$NODE:$p/admin/exit" -o /dev/null -w "identity :$p %{http_code} "
done
echo
echo "NOTE: identity code is live now. Changes to gpu/deploy/ or a DELETED"
echo "      src/ module need 'scripts/pod/pod.sh up' (container restart)."
