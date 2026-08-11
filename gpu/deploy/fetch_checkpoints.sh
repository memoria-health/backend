#!/usr/bin/env bash
# First-boot checkpoint fetch — the part of scripts/pod_bootstrap.sh that
# cannot live in the image.
#
# WHY NOT BAKED:
#   * `facebook/sam3.1` is a GATED repo. Baking those weights into an image
#     redistributes them, which is what the gate exists to prevent. The image
#     would have to stay private forever and still be a licence question.
#   * sam3.1_multiplex.pt alone is 3.34GB. Baking takes the image from ~9-10GB
#     to ~14GB, and image size IS node cold-start time — the thing this whole
#     design is buying.
#   * It matches how the node works today: pod_bootstrap.sh downloads onto the
#     pod's disk and they persist there.
#
# Idempotent. Writes into NEMARA_MODEL_DIR, which should be on the RunPod
# NETWORK VOLUME so the ~3.8GB survives the node being recreated. A node in a
# datacenter with no volume simply re-downloads — slower, still correct.
set -uo pipefail

MODEL_DIR="${NEMARA_MODEL_DIR:-/workspace/models}"
EDGETAM_CKPT_DIR=/workspace/edgetam/checkpoints
mkdir -p "$MODEL_DIR" "$EDGETAM_CKPT_DIR"

log() { echo "$(date -u +%FT%TZ) checkpoints: $*"; }

need() { [ ! -s "$1" ]; }

# ── EdgeTAM carry checkpoint (54MB, ungated) ───────────────────────────
if need "$EDGETAM_CKPT_DIR/edgetam.pt"; then
  log "fetching edgetam.pt"
  python3 - <<'PY' || exit 1
import shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download("facebook/EdgeTAM", "edgetam.pt")
shutil.copy(p, "/workspace/edgetam/checkpoints/edgetam.pt")
PY
else
  log "edgetam.pt present"
fi

# ── SAM3 multiplex grounder (3.34GB, GATED) ────────────────────────────
# Needs HUGGINGFACE_TOKEN. Fail loudly and early: a node that comes up
# without a grounder looks alive (perception answers /health) but returns
# nothing useful, which is the worst kind of broken.
if need "$MODEL_DIR/sam3.1_multiplex.pt"; then
  if [ -z "${HUGGINGFACE_TOKEN:-}" ]; then
    log "FATAL: sam3.1_multiplex.pt missing and HUGGINGFACE_TOKEN unset."
    log "       facebook/sam3.1 is gated — the node cannot ground without it."
    exit 1
  fi
  log "fetching sam3.1_multiplex.pt (3.3GB, gated) ..."
  MODEL_DIR="$MODEL_DIR" python3 - <<'PY' || exit 1
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    "facebook/sam3.1", "sam3.1_multiplex.pt",
    token=os.environ["HUGGINGFACE_TOKEN"],
)
dst = os.path.join(os.environ["MODEL_DIR"], "sam3.1_multiplex.pt")
shutil.copy(p, dst)
print("  ->", dst, os.path.getsize(dst) // 2**20, "MB")
PY
else
  log "sam3.1_multiplex.pt present"
fi

# ── YOLO: one model serves BOTH the reground trigger (counts) and the
# yoloseg grounding path (masks) in the same ~140ms pass. yolo26x is kept
# because NEMARA_GROUNDER=yoloseg remains switchable.
for w in yolo26x.pt yolo26x-seg.pt; do
  if need "$MODEL_DIR/$w"; then
    log "fetching $w"
    (cd "$MODEL_DIR" && python3 -c "from ultralytics import YOLO; YOLO('$w')") || exit 1
  else
    log "$w present"
  fi
done

log "ready — $(ls -1 "$MODEL_DIR" | tr '\n' ' ')"
