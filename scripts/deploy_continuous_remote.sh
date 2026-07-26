#!/usr/bin/env bash
# Runs ON THE POD — piped in by `scripts/pod.sh deploy`. Stops the old
# SAM-only server + supervisor and relaunches the supervisor on the
# edgetam venv (one env hosts sam2 + sam3 + insightface). server.py +
# continuous.py + sam_track.py must already be shipped to
# /workspace/nemara/sam_service/ (pod.sh deploy ships them first).
set -uo pipefail

# Stop old supervisor + server — INCLUDING the legacy
# start_continuous.sh supervisor (2026-07-24: two supervisors raced
# for port 8000 after every deploy; the legacy one always won with
# stale env, so env changes silently never applied). Disable it for
# good so exactly one supervisor exists.
kill "$(cat /workspace/sam_supervisor.pid 2>/dev/null)" 2>/dev/null || true
pkill -f 'start_continuous.sh' 2>/dev/null || true
[ -f /workspace/start_continuous.sh ] \
  && mv /workspace/start_continuous.sh /workspace/start_continuous.sh.disabled
pkill -f 'sam_supervisor' 2>/dev/null || true
pkill -f 'uvicorn sam_servic[e]' 2>/dev/null || true
sleep 3

# Relaunch the supervisor (cwd=/workspace/edgetam for configs/edgetam.yaml;
# PYTHONPATH spans sam3 + nemara + edgetam). NOTE: `python -m uvicorn`, not
# the `uvicorn` console script — the edgetam venv exposes the module but
# not the entrypoint binary.
nohup bash -c '
  echo $$ > /workspace/sam_supervisor.pid
  cd /workspace/edgetam
  export NEMARA_SAM_CKPT=/workspace/nemara/models/sam3.1_multiplex.pt
  export PYTHONPATH=/workspace/sam3:/workspace/nemara:/workspace/edgetam
  # YOLO26x reground trigger (person counts on carry ticks; fail-open)
  export NEMARA_TRIGGER_WEIGHTS=/workspace/nemara/models/yolo26x.pt
  # Grounder: sam3 (Aryaman, 2026-07-24 eval verdict) — YOLO box-IoU
  # remap swapped ids between ADJACENT people (overlapping boxes on a
  # couch traded track numbers ~19:01 live; masks overlap far less).
  # YOLO grounding stays available (NEMARA_GROUNDER=yolo) pending a
  # globally-optimal, keep-biased assignment; the YOLO trigger still
  # runs either way (new-person latency).
  export NEMARA_GROUNDER=sam3
  # onnxruntime-gpu needs cuDNN 9 — torch's bundled nvidia wheels
  # provide it (2026-07-25 pod rebuild; without this, faces fall back
  # to CPUExecutionProvider and probes run ~10x slower)
  export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
  # ── Perception tuning knobs (defaults are production) ──────────────
  #   NEMARA_EDGE_REGROUND=5            SAM3 re-ground cadence (ticks)
  #   NEMARA_EDGE_TRACK_W=640           EdgeTAM track-resolution width
  #   NEMARA_EDGE_DRIFT_BALLOON=1.8     drift-guard banded balloon
  #   NEMARA_EDGE_DRIFT_BIG_BALLOON=2.5 drift-guard hard balloon (any shape)
  #   NEMARA_EDGE_DRIFT_WIDE_ASPECT=1.15  banded "went wide" gate
  #   NEMARA_EDGE_DRIFT_LCC_MIN=0.55    banded fragmentation gate
  # Visual verification ONLY (fills disk, slows ticks) — uncomment for a
  # drift-guard check, then redeploy with it OFF:
  #   export NEMARA_EDGE_DEBUG_DIR=/workspace/edge_debug
  while true; do
    # SYSTEM python — /workspace/edgetam/.venv no longer exists; the
    # phantom path made this loop rc=127 forever while the legacy
    # supervisor actually served (2026-07-24).
    /usr/bin/python3 -m uvicorn sam_service.server:app \
      --host 127.0.0.1 --port 8000 --timeout-keep-alive 75 \
      >> /workspace/sam_server.log 2>&1
    echo "$(date -u +%FT%TZ) supervisor: exited rc=$? restart 3s" \
      >> /workspace/sam_server.log
    sleep 3
  done
' >/dev/null 2>&1 </dev/null &
disown 2>/dev/null || true
echo "continuous supervisor launched (edgetam venv)"
