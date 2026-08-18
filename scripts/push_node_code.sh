#!/bin/bash
# Push gpu/ code to the fleet node's dev-code overlay and restart its
# services. Layout matches the IMAGE (Dockerfile 224-227 + entrypoint
# dev_code_overlay): /persist/dev-code/nemara/{identity,perception,deploy}
# — the old scratchpad script nested a gpu/ dir the overlay never read,
# so every "push" since the v3 image was a silent no-op (2026-08-18).
# src/ is NOT overlaid (entrypoint only covers nemara/); src changes
# need an image build.
#
# Usage: scripts/push_node_code.sh <reader_host> <reader_port> <node_tailnet_ip>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST=$1; PORT=$2; NODE=$3
SSH="ssh -o StrictHostKeyChecking=no -p $PORT"
$SSH "root@$HOST" 'mkdir -p /persist/dev-code/nemara'
rsync -az --delete -e "$SSH" --exclude '__pycache__' --exclude '*.pyc' \
  "$REPO/gpu/identity" "$REPO/gpu/perception" "$REPO/gpu/deploy" \
  "root@$HOST:/persist/dev-code/nemara/"
echo "pushed (image-layout)"
STOKEN=$(grep '^NEMARA_IDENTITY_STATE_TOKEN' "$REPO/.env" | cut -d= -f2 | tr -d "'\"")
for p in 8012 8013; do
  curl -s -m 5 -X POST -H "Authorization: Bearer $STOKEN" \
    "http://$NODE:$p/admin/exit" -o /dev/null -w "identity :$p %{http_code} "
done
echo
