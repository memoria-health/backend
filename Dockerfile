# Production image for the nemara-backend FastAPI service.
#
# Two-stage layout: first install dependencies into a venv (cacheable),
# then copy the source. Layer ordering puts the rarely-changing
# pyproject.toml + uv.lock above the source so source-only changes
# don't bust the dep cache.
#
# Built and run by Railway. The CMD reads $PORT at runtime (Railway
# injects it; defaults to 8002 for local docker runs).

FROM python:3.13-slim

# ffmpeg is required by src/services/detection/mp4_remux.py to remux
# buffered LiveKit frames + audio into MP4 before the Gemini call.
# Without it, the camera_ingest worker logs "mp4_remux failed: ffmpeg
# binary not found on PATH" every tick and no analyses rows are written.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
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

# Match the local-dev port for parity. Railway overrides via $PORT.
EXPOSE 8002

# `sh -c` form so $PORT is interpolated at container start.
CMD ["sh", "-c", "uv run uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8002}"]
