#!/usr/bin/env bash
# Runs ON THE POD — piped in by `scripts/pod.sh deploy`. Stops the old
# supervisor + server, (re)writes the FILE-BASED supervisor, and
# relaunches it via setsid. perception/ must already be shipped to
# /workspace/nemara/perception/ (pod.sh deploy ships it first).
#
# WHY a supervisor FILE + setsid (runbook "operational traps"): the
# previous inline `nohup bash -c '...'` launch lost its exported env
# when piped through `ssh bash -s` (2026-07-26: uvicorn came up with
# no PYTHONPATH → ModuleNotFoundError → dead loop). A file on disk is
# quote-mangling-proof, inspectable, and identical however launched.
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
pkill -f 'start_sam_superviso[r]' 2>/dev/null || true
pkill -f 'sam_supervisor.pi[d]' 2>/dev/null || true
pkill -f 'uvicorn perceptio[n]' 2>/dev/null || true
# Transitional (2026-07-26 sam_service→perception rename): kill a server
# still running under the OLD module name; remove after one clean deploy.
pkill -f 'uvicorn sam_servic[e]' 2>/dev/null || true
rm -rf /workspace/nemara/sam_service
sleep 3

# (Re)write the supervisor file — the single source of runtime env.
cat > /workspace/start_sam_supervisor.sh <<'SUP'
#!/usr/bin/env bash
echo $$ > /workspace/sam_supervisor.pid
# cwd=/workspace/edgetam for configs/edgetam.yaml
cd /workspace/edgetam
export NEMARA_SAM_CKPT=/workspace/nemara/models/sam3.1_multiplex.pt
export PYTHONPATH=/workspace/sam3:/workspace/nemara:/workspace/edgetam
# ONE YOLO model serves trigger (counts) AND grounding (seg masks):
# yolo26x-seg returns boxes + retina masks in the same ~140ms pass.
export NEMARA_TRIGGER_WEIGHTS=/workspace/nemara/models/yolo26x-seg.pt
# Grounder: sam3 (Aryaman, 2026-07-26 verdict) — the yoloseg trial
# fragmented a recumbent/occluded person into two instances (lying on
# bed with laptop → torso cut as its own mask), flapping the trigger
# and churning track ids. SAM3 never fragmented. yoloseg stays
# switchable (NEMARA_GROUNDER=yoloseg) pending a containment-dedup
# pass; the YOLO trigger (counts) still runs either way.
export NEMARA_GROUNDER=sam3
# onnxruntime-gpu needs cuDNN 9 — torch's bundled nvidia wheels
# provide it (2026-07-25 pod rebuild; without this, faces fall back
# to CPUExecutionProvider and probes run ~10x slower)
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
# ── Perception tuning knobs (defaults are production) ──────────────
#   NEMARA_EDGE_REGROUND=5            re-ground cadence (ticks)
#   NEMARA_EDGE_TRACK_W=640           EdgeTAM track-resolution width
# Visual verification ONLY (fills disk, slows ticks):
#   export NEMARA_EDGE_DEBUG_DIR=/workspace/edge_debug
while true; do
  # SYSTEM python — /workspace/edgetam/.venv no longer exists; the
  # phantom path made this loop rc=127 forever while the legacy
  # supervisor actually served (2026-07-24).
  /usr/bin/python3 -m uvicorn perception.server:app \
    --host 127.0.0.1 --port 8000 --timeout-keep-alive 75 \
    >> /workspace/sam_server.log 2>&1
  echo "$(date -u +%FT%TZ) supervisor: exited rc=$? restart 3s" \
    >> /workspace/sam_server.log
  sleep 3
done
SUP
chmod +x /workspace/start_sam_supervisor.sh

# Launch detached in its OWN session — env comes from the file, and the
# session contains no kill-pattern strings to self-match later.
setsid --fork /workspace/start_sam_supervisor.sh </dev/null >/dev/null 2>&1
sleep 2
pgrep -f 'uvicorn perceptio[n]' >/dev/null \
  && echo "continuous supervisor launched (file-based)" \
  || echo "WARNING: uvicorn not up yet — check /workspace/sam_server.log"
