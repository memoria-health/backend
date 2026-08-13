# GPU node image — perception + identity, the deployable form of
# the hand-run bootstrap script this replaced (scripts/pod_bootstrap.sh,
# deleted 2026-08-11 — see git history if you need what it did).
#
# WHY THIS EXISTS. The pod has been a hand-built pet: bootstrap mutates a
# running container over ssh (~40 min) and `pod.sh deploy` rsyncs six files
# onto it. That has cost us a machine twice — RunPod terminated the original
# when credits hit zero and the disk went with it, and on 2026-08-10 a stopped
# pod could not resume because the host had reallocated its GPU. Neither is
# recoverable by rsync.
#
# An image makes the node disposable: the artifact is a tag, a lost node is
# replaced by creating another from it, and "what is running?" has an answer
# that is not "whatever we last scp'd".
#
# BOTH SERVICES SHIP HERE. Identity sends ~13MB batches to perception every
# cycle. Over a network that is ~2.2 TB/month/camera, and RunPod's inter-pod
# link is 100 Mbps (~1s per batch). Colocated it is a localhost copy — and
# perception never has to bind a network interface.
#
# BUILD — on an amd64 runner. Do NOT build on an Apple-silicon Mac: QEMU
# emulation of the torch/CUDA layers takes hours.
#
#   docker build -f gpu/Dockerfile -t ghcr.io/<org>/memoria-gpu:<tag> .
#
# Build context is the REPO ROOT (we copy both gpu/ and src/).

# ── Tailscale binaries ─────────────────────────────────────────────────
# Taken from the official image rather than installed from apt: it is one
# COPY of two static binaries, it needs no apt repo/keyring dance, and the
# version is a tag we control instead of whatever the repo serves today.
#
# NOT OPTIONAL on a production node. The backend reaches identity over the
# tailnet, and the entrypoint derives the address it registers from the name
# tailscaled reports — so without these binaries a roster-mode node cannot
# name itself and refuses to boot. (Until now the image shipped without them
# while the entrypoint called `tailscaled`, so tailscale silently never ran.)
# Both ARGs are declared HERE, above the first FROM. An ARG used in a FROM
# must be in the global scope; declared after one, it belongs to that stage
# and the next FROM expands it to an empty string.
ARG TAILSCALE_TAG=stable
ARG BASE_IMAGE=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

FROM tailscale/tailscale:${TAILSCALE_TAG} AS tailscale

FROM ${BASE_IMAGE}

COPY --from=tailscale /usr/local/bin/tailscaled /usr/local/bin/tailscaled
COPY --from=tailscale /usr/local/bin/tailscale  /usr/local/bin/tailscale
# Fail the BUILD if they are not runnable, in keeping with the assertions
# further down. The alternative is discovering it on a live node, where the
# symptom is a node that boots, finds no tailnet name, and exits — twenty
# minutes and one RunPod bill after the mistake was made.
RUN tailscaled --version && tailscale --version

# PINNED, not `main`. Two builds a week apart off a branch get different Meta
# code, which is how a working node becomes a broken one with no diff to
# blame — and is exactly how the two patches below came to be needed. Bumping
# either is a change that deserves a soak.
# Captured 2026-08-11 from each repo's default branch.
ARG SAM3_REF=96914d2425f90a64f45ca977c2b5165418099543
ARG EDGETAM_REF=7711e012a30a2402c4eaab637bdb00a521302c91

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates socat \
 && rm -rf /var/lib/apt/lists/*
# socat: the local-rig tailnet DB bridge (entrypoint §3.5) — asyncpg
# cannot ride a proxy, so a tailnet DB URL gets a TCP bridge through
# tailscaled's HTTP CONNECT proxy. Unused in prod (public DB).

# ── Upstream model code ────────────────────────────────────────────────
RUN git clone https://github.com/facebookresearch/sam3 sam3 \
 && git -C sam3 checkout "${SAM3_REF}" \
 && git clone https://github.com/facebookresearch/EdgeTAM edgetam \
 && git -C edgetam checkout "${EDGETAM_REF}"

# ── Patch 1: EdgeTAM Perceiver reshape ─────────────────────────────────
# `.expand().view()` on a non-contiguous tensor throws the moment a SECOND
# person is tracked — invisible to solo testing, 500s every /observe in a real
# room. The grep asserts the patch applied: silently shipping the unpatched
# file is how this bites again.
RUN sed -i \
      -e 's/\.expand(B, -1, -1)\.view(-1, 1, C)/.expand(B, -1, -1).reshape(-1, 1, C)/' \
      -e 's/latents_2d\.view(B, num_window, num_window, C)/latents_2d.reshape(B, num_window, num_window, C)/' \
      edgetam/sam2/modeling/perceiver.py \
 && grep -q 'expand(B, -1, -1).reshape(-1, 1, C)' edgetam/sam2/modeling/perceiver.py \
 && grep -q 'latents_2d.reshape(B, num_window, num_window, C)' edgetam/sam2/modeling/perceiver.py \
 || (echo "EdgeTAM perceiver patch did not apply — upstream moved" && exit 1)

# ── Patch 2: SAM3 bool argsort ─────────────────────────────────────────
# torch 2.4 cannot sort a bool tensor on CUDA. The original pod ran a 2.8-dev
# nightly where it works, which is why this only surfaced on rebuild.
RUN sed -i \
      's/pos_pred_mask_idx = pos_pred_mask.argsort(descending=True)/pos_pred_mask_idx = pos_pred_mask.int().argsort(descending=True)/' \
      sam3/sam3/model/sam3_multiplex_base.py \
 && grep -q "int().argsort" sam3/sam3/model/sam3_multiplex_base.py \
 || (echo "SAM3 bool-argsort patch did not apply — upstream moved" && exit 1)

# ── ML deps ────────────────────────────────────────────────────────────
# --no-deps throughout this block: every one of these declares torch, and
# letting pip resolve it will happily swap the base image's CUDA build for
# something else. A mismatched torch/CUDA pair costs an evening.
# Pinned to the versions the LOCAL suite proves daily (2026-08-12) —
# unpinned latest broke at import inside the container the first time a
# node actually booted. NOTE local proves these against torch 2.13; the
# image runs the base's CUDA torch 2.4 — if import still fails here,
# suspect the torch gap before anything else.
RUN pip install -q --no-deps ultralytics==8.4.55 ultralytics-thop==2.0.19 boxmot==19.0.0
# gdown is boxmot's weight downloader. --no-deps above means it never
# installed, and the base image ships one too old for the fuzzy= kwarg
# boxmot passes (OSNet prefetch died on it, first CI build 2026-08-12).
# torch-free, so installing it WITH deps is safe. <6 because gdown 6.0
# (2026) removed the fuzzy= kwarg again — 5.x is the only line whose
# download() signature matches what boxmot calls.
RUN pip install -q "gdown>=5,<6"
# filelock: the base image's copy predates AsyncFileLock; --no-deps
# blocks the upgrade ultralytics would have requested, and the newest
# ultralytics dies importing it (first node boot, 2026-08-12). The pin
# above avoids that import today; upgrading anyway costs nothing and
# removes the tripwire. torch-free.
RUN pip install -q -U filelock

# onnxruntime-gpu pinned to the CUDA-12 line. 1.28+ needs CUDA 13, and the
# failure mode is not an error but a silent fall back to CPUExecutionProvider
# (~10x slower face probes). NOTE pyproject.toml declares plain
# `onnxruntime>=1.16` for the backend — installing that here would shadow this
# with the CPU build, which is why src/'s deps are curated below rather than
# installed wholesale. Check /health's face_provider after any bump.
# 1.20.1 was never published for -gpu (PyPI jumps 1.20.0 -> 1.20.2);
# found by the first real CI build 2026-08-12.
RUN pip uninstall -q -y onnxruntime || true \
 && pip install -q "onnxruntime-gpu==1.20.2"

# ORT 1.20.x dlopens libcudnn.so.9 when creating the CUDA EP, and this base
# tag ships no cuDNN 9 — the failure mode is the same silent CPU fallback as
# above (found live 2026-08-13: /health face_provider=CPUExecutionProvider,
# every face probe on CPU, GPU idle under load). The previous fix — an ENV
# LD_LIBRARY_PATH pointing at a dist-packages path — failed silently because
# the path never existed here. So: install the exact cuDNN torch 2.4+cu124
# was built against (--no-deps: its cublas dep comes from the CUDA toolkit
# already in the image), register it with the SYSTEM loader via ldconfig
# (not LD_LIBRARY_PATH — see face_worker.py's history with env-var lib
# surgery), and prove at build time that (a) cuDNN 9 actually loads and
# (b) every library the ORT CUDA provider links resolves. Only libcuda.so.1
# (the driver, absent on a GPU-less builder) is allowed to be missing.
RUN pip install -q --no-deps "nvidia-cudnn-cu12==9.1.0.70" \
 && python3 -c "import nvidia.cudnn, pathlib; \
lib = pathlib.Path(nvidia.cudnn.__file__).parent / 'lib'; \
assert list(lib.glob('libcudnn.so.9*')), f'no libcudnn.so.9 under {lib}'; \
pathlib.Path('/etc/ld.so.conf.d/nemara-cudnn.conf').write_text(str(lib) + '\n')" \
 && ldconfig \
 && python3 -c "import ctypes; ctypes.CDLL('libcudnn.so.9'); print('cudnn9 loads')"
RUN ORT_CUDA=$(python3 -c "import onnxruntime, pathlib; \
print(pathlib.Path(onnxruntime.__file__).parent / 'capi' / 'libonnxruntime_providers_cuda.so')") \
 && MISSING=$(ldd "$ORT_CUDA" | grep 'not found' | grep -v libcuda.so.1 || true) \
 && { [ -z "$MISSING" ] || { echo "ORT CUDA EP unresolved:"; echo "$MISSING"; exit 1; }; } \
 && echo "ORT CUDA provider fully resolved"

RUN pip install -q \
      opencv-python-headless insightface \
      hydra-core omegaconf iopath py-cpuinfo pandas polars matplotlib tqdm psutil \
      einops timm pycocotools decord scikit-image ftfy regex \
      gdown lap filterpy

# ── Backend deps, curated ──────────────────────────────────────────────
# identity imports src.settings / src.db.session / src.repos, which drag in
# most of src/ (src/repos/__init__ eagerly imports nine repos;
# src/services/__init__ pulls auth + onboarding). So the backend's runtime
# deps are needed — MINUS everything ML, which is pinned above.
#
# Deliberately excluded from pyproject.toml's list: torch, torchvision,
# onnxruntime, onnx, numpy, opencv-python-headless, ultralytics, boxmot,
# insightface. Installing those here would fight the pins above.
RUN pip install -q \
      fastapi "uvicorn[standard]" python-multipart httpx \
      "sqlalchemy[asyncio]" asyncpg \
      pydantic pydantic-settings python-dotenv \
      "pyjwt[crypto]" bcrypt typer \
      google-genai openai \
      livekit "livekit-api" \
      h2 pillow aioboto3 webrtcvad-wheels

# ── OSNet body-embedding weights, baked ────────────────────────────────
# gpu/identity/body_embeddings.py lazy-loads `osnet_x0_25_msmt17.pt` through
# boxmot, which downloads it to ~/.boxmot on first use. Baking it means the
# first body embedding after a cold start does not depend on the network.
# Runs on CPU — CI has no GPU, and the download is device-independent.
# Uses the SAME import path and constructor as body_embeddings.py:_get_backend,
# so if boxmot moves it the build breaks here rather than at the first body
# embedding on a live node.
RUN python3 -c "\
from boxmot.reid.backends.pytorch_backend import PyTorchBackend;\
import torch, pathlib;\
PyTorchBackend(weights=pathlib.Path('osnet_x0_25_msmt17.pt'), device=torch.device('cpu'), half=False);\
print('osnet weights cached')" \
 || (echo "OSNet prefetch failed — body embeddings would hit the network on first use" && exit 1)

# ── Our code ───────────────────────────────────────────────────────────
# Last, so editing our code invalidates only these layers; everything above
# stays cached.
#
# src/ is 7.3MB of source. It is here because gpu/identity/service.py imports
# src.db.session, src.repos and src.settings, and gpu/identity/gallery.py
# imports src.services.identity.trace. Vendoring those is not viable —
# src/repos/__init__ alone pulls nine repos. The proper fix is a GPU-host
# config surface (see gpu/identity/README.md); this is the documented
# packaging exception until then.
COPY src            /workspace/src
COPY gpu/perception /workspace/nemara/perception
COPY gpu/identity   /workspace/nemara/identity
COPY gpu/deploy     /workspace/nemara/deploy
RUN chmod +x /workspace/nemara/deploy/*.sh

# ── Runtime env ────────────────────────────────────────────────────────
# cwd must be /workspace/edgetam — it resolves configs/edgetam.yaml relatively.
WORKDIR /workspace/edgetam

# PYTHONPATH order matters. /workspace/sam3 must precede /workspace so
# `import sam3` finds the inner package, not the clone directory. /workspace
# is last and exists so `src.*` resolves.
#
# This also makes service.py's own sys.path bootstrap correct rather than
# merely harmless: it inserts Path(__file__).parents[2], which in-container is
# /workspace — exactly where src/ lives.
# Persistent state lives under /persist, NOT under /workspace. RunPod mounts
# a network volume at volumeMountPath, and the obvious choice — /workspace —
# would SHADOW everything the image put there (sam3, edgetam, our code), so
# the node would boot with no application at all. /persist is empty in the
# image and exists precisely to be mounted over.
RUN mkdir -p /persist/models /persist/state

ENV PYTHONPATH=/workspace/sam3:/workspace/nemara:/workspace/edgetam:/workspace \
    NEMARA_MODEL_DIR=/persist/models \
    NEMARA_STATE_DIR=/persist/state \
    NEMARA_SAM_CKPT=/persist/models/sam3.1_multiplex.pt \
    NEMARA_TRIGGER_WEIGHTS=/persist/models/yolo26x-seg.pt \
    NEMARA_GROUNDER=sam3 \
    PERCEPTION_HOST=127.0.0.1 \
    PERCEPTION_PORT=8000

# Fail the BUILD on a broken environment rather than finding out on a node at
# 3am. Covers both services: the perception model builders AND the src/ import
# chain that identity needs (this is the one most likely to regress, since a
# backend refactor can break it from a distance).
RUN python3 -c "\
import torch, cv2, numpy;\
from sam3.model_builder import build_sam3_multiplex_video_predictor;\
from sam2.build_sam import build_sam2_video_predictor;\
import insightface, ultralytics, fastapi, boxmot;\
import src.settings, src.db.session, src.repos;\
import src.services.identity.trace;\
print('smoke imports OK — perception + identity + src')"

EXPOSE 8000

# Checkpoints are NOT baked. They are fetched on first boot into the network
# volume by deploy/fetch_checkpoints.sh — same as the old bootstrap did,
# it keeps the image ~9-10GB instead of ~14GB, and it avoids redistributing
# the GATED facebook/sam3.1 weights inside an image.
ENTRYPOINT ["/workspace/nemara/deploy/entrypoint.sh"]
