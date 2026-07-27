#!/usr/bin/env bash
# Full perception-pod rebuild from NOTHING (2026-07-25 — written the
# night the original pod was terminated by RunPod when credits ran out
# and its disk vanished with it; the old bootstrap lived only on that
# disk. This one is committed.)
#
# Target: RunPod "PyTorch 2.4 / CUDA 12.4" template (torch preinstalled),
# A40-class GPU, ~70GB container disk. Runs ON THE POD:
#
#   scripts/pod.sh ssh "bash -s" < scripts/pod_bootstrap.sh
#
# Then from the Mac: scripts/pod.sh deploy   (ships perception + launches)
#
# Layout it builds (PYTHONPATH order matters — see deploy script):
#   /workspace/sam3      Meta SAM3 repo   (concept grounding)
#   /workspace/edgetam   EdgeTAM repo     (vendored sam2, carry tracking)
#   /workspace/nemara    our code (shipped separately) + models/
set -euo pipefail
cd /workspace

echo "== GPU =="
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

echo "== repos =="
[ -d sam3 ]    || git clone --depth 1 https://github.com/facebookresearch/sam3
[ -d edgetam ] || git clone --depth 1 https://github.com/facebookresearch/EdgeTAM edgetam

echo "== EdgeTAM perceiver patch (multi-object crash; POD_RUNBOOK §3.4) =="
sed -i \
  -e 's/\.expand(B, -1, -1)\.view(-1, 1, C)/.expand(B, -1, -1).reshape(-1, 1, C)/' \
  -e 's/latents_2d\.view(B, num_window, num_window, C)/latents_2d.reshape(B, num_window, num_window, C)/' \
  edgetam/sam2/modeling/perceiver.py
grep -c reshape edgetam/sam2/modeling/perceiver.py

echo "== SAM3 bool-argsort patch (torch 2.4: no bool sort on CUDA) =="
sed -i 's/pos_pred_mask_idx = pos_pred_mask.argsort(descending=True)/pos_pred_mask_idx = pos_pred_mask.int().argsort(descending=True)/' \
  sam3/sam3/model/sam3_multiplex_base.py
grep -c "int().argsort" sam3/sam3/model/sam3_multiplex_base.py

echo "== pip layer (NEVER let pip touch the template torch) =="
pip install -q --no-deps ultralytics ultralytics-thop
# ORT pinned to the CUDA-12 line (1.28+ wants CUDA 13 — template is
# 12.4); cuDNN comes from torch's bundled nvidia wheels via
# LD_LIBRARY_PATH (set in the deploy script's supervisor env).
pip uninstall -q -y onnxruntime 2>/dev/null || true
pip install -q "onnxruntime-gpu==1.20.1"
pip install -q fastapi "uvicorn[standard]" python-multipart httpx \
  opencv-python-headless insightface \
  hydra-core omegaconf iopath py-cpuinfo pandas polars matplotlib tqdm psutil \
  einops timm pycocotools decord scikit-image ftfy regex

echo "== checkpoints =="
mkdir -p /workspace/nemara/models
# EdgeTAM carry checkpoint (in-repo pointer or HF)
if [ ! -f edgetam/checkpoints/edgetam.pt ]; then
  (cd edgetam/checkpoints 2>/dev/null && bash download_ckpts.sh) || \
  python3 -c "from huggingface_hub import hf_hub_download; import shutil, os; os.makedirs('edgetam/checkpoints', exist_ok=True); p=hf_hub_download('facebook/EdgeTAM','edgetam.pt'); shutil.copy(p,'edgetam/checkpoints/edgetam.pt')"
fi
ls -la edgetam/checkpoints/ | head -3
# SAM3 multiplex — hosted at HF facebook/sam3.1 (verified 2026-07-25;
# ultralytics does NOT auto-fetch this name). HUGGINGFACE_TOKEN env
# used if the repo is gated for the account.
if [ ! -f /workspace/nemara/models/sam3.1_multiplex.pt ]; then
  python3 - << 'PY'
import os, shutil
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    "facebook/sam3.1", "sam3.1_multiplex.pt",
    token=os.environ.get("HUGGINGFACE_TOKEN") or None,
)
shutil.copy(p, "/workspace/nemara/models/sam3.1_multiplex.pt")
print("sam3.1_multiplex.pt ->", os.path.getsize(p) // 2**20, "MB")
PY
fi
# YOLO26x (reground trigger)
[ -f /workspace/nemara/models/yolo26x.pt ] || \
  (cd /workspace/nemara/models && python3 -c "from ultralytics import YOLO; YOLO('yolo26x.pt')")
# Seg variant: trigger counts AND yoloseg grounding masks (2026-07-25)
[ -f /workspace/nemara/models/yolo26x-seg.pt ] || \
  (cd /workspace/nemara/models && python3 -c "from ultralytics import YOLO; YOLO('yolo26x-seg.pt')")
ls -la /workspace/nemara/models/

echo "== smoke imports (the deploy will fail loudly if these do) =="
PYTHONPATH=/workspace/sam3:/workspace/nemara:/workspace/edgetam python3 - << 'PY'
import torch, cv2, numpy
from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa
from sam2.build_sam import build_sam2_video_predictor  # noqa
import insightface, ultralytics, fastapi  # noqa
print("smoke imports OK")
PY

echo "== bootstrap complete — now run scripts/pod.sh deploy from the Mac =="
