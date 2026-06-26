# Production image for the nemara-backend FastAPI service.
#
# Two-stage layout: first install dependencies into a venv (cacheable),
# then copy the source. Layer ordering puts the rarely-changing
# pyproject.toml + uv.lock above the source so source-only changes
# don't bust the dep cache.
#
# Built and run by Railway. The CMD reads $PORT at runtime (Railway
# injects it; defaults to 8002 for local docker runs).

FROM python:3.12-slim

# System deps:
#   - ffmpeg: mp4_remux.py + frame_extraction.py (Re-ID inference and
#     LLM frame extraction both shell out to ffmpeg)
#   - libgomp1: OpenMP runtime, required by onnxruntime when running
#     InsightFace under the CPU execution provider
#   - libopenblas-dev: NumPy linear-algebra accelerator (matters at
#     gallery-match scan time once a patient has 100+ embeddings)
#   - libgl1, libglib2.0-0: required by opencv-python-headless for its
#     non-GUI image decode path
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 libopenblas-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# uv is the package manager this project uses; pre-installing it via
# pip avoids fetching it on every build.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Dependencies layer — only pyproject.toml + uv.lock. Touching any
# .py file invalidates only the layers below this one.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Application source.
COPY src ./src
COPY supabase ./supabase
COPY scripts ./scripts
# README.md is referenced as the project's `readme` field in
# pyproject.toml — hatchling fails the build step below if it's
# absent. Cheap to include.
COPY README.md ./

# Install the project itself into the venv.
RUN uv sync --frozen --no-dev

# Pre-download InsightFace's buffalo_l model bundle at image build
# time. ~280 MB lands in /root/.insightface/models/buffalo_l and is
# baked into the image so the first tick after deploy doesn't pay
# the cold-download cost (which can be 30-60s on Railway).
RUN uv run python -c "from insightface.app import FaceAnalysis; \
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
    app.prepare(ctx_id=0, det_size=(640, 640)); \
    print('insightface buffalo_l warmed')"

# Pre-download YOLO11n and OSNet weights for the body Re-ID pipeline.
# YOLO11n: ~6 MB, lands at /app/yolo11n.pt. OSNet x0_25 MSMT17:
# ~10 MB, lands wherever boxmot's PyTorchBackend caches by default
# (typically /root/.cache/boxmot/ or the CWD). Same rationale as the
# buffalo_l pre-cache: first tick after deploy shouldn't pay a cold
# download.
RUN uv run python -c "from ultralytics import YOLO; \
    m = YOLO('yolo11n.pt'); \
    print('yolo11n warmed')"
RUN uv run python -c "import torch; \
    from pathlib import Path; \
    from boxmot.reid.backends.pytorch_backend import PyTorchBackend; \
    PyTorchBackend(weights=Path('osnet_x0_25_msmt17.pt'), device=torch.device('cpu'), half=False); \
    print('osnet x0_25 msmt17 warmed')"

# Match the local-dev port for parity. Railway overrides via $PORT.
EXPOSE 8002

# `sh -c` form so $PORT is interpolated at container start.
CMD ["sh", "-c", "uv run uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8002}"]
