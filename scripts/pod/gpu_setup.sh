#!/usr/bin/env bash
# One-shot GPU-box setup for the SAM identity A/B (2026-07-20).
# Target: a fresh RunPod/Vast NVIDIA box with a "PyTorch 2.4 / CUDA
# 12.4" template (torch+CUDA already present). Idempotent-ish; safe to
# re-run.
#
#   bash scripts/pod/gpu_setup.sh
#
# Needs env: HUGGINGFACE_TOKEN (for gated SAM 3.1 / AdaFace weights).
set -euo pipefail

echo "== GPU check =="
python -c "import torch; print('cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo "== deps =="
# If the template did NOT ship CUDA torch, uncomment:
# pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt

echo "== AdaFace source (embed net not pip-installable) =="
mkdir -p src/services/reid/_adaface
if [ ! -f src/services/reid/_adaface/net.py ]; then
  git clone --depth 1 https://github.com/mk-minchul/AdaFace /tmp/adaface
  cp /tmp/adaface/net.py src/services/reid/_adaface/net.py
  cp -r /tmp/adaface/backbones src/services/reid/_adaface/ 2>/dev/null || true
  touch src/services/reid/_adaface/__init__.py
fi

echo "== models =="
python tools/download_models.py

echo "== done. Run the A/B: =="
echo "  python tools/pipeline_ab.py --clips out/ab_clips --patient <PID> --backends legacy,sam --labels out/ab_labels.csv"
