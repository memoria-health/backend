#!/bin/bash
# v2 pod start script — SUPERVISED (2026-07-21 VRAM-leak incident).
# Install: cp to /workspace/start_sam.sh on the pod and run it.
#
# Hardening lineage (2026-07-20 security review): 127.0.0.1 bind
# (access is via SSH -L only), stale /dev/shm patient clips swept,
# waits for the port to actually free before starting.
#
# Supervision (2026-07-21): ~9h of session create/destroy fragmented
# the CUDA allocator to 25.5GB resident and 3x tick latency. Three
# independent belts:
#   1. expandable_segments allocator mode (prevents the fragmentation
#      pattern at the source)
#   2. the server self-exits once old (>6h) AND idle (>5min) — see
#      SAM_RECYCLE_* in server.py
#   3. the supervisor loop below restarts it (also catches crashes),
#      so recycles are invisible to the backend
PIDFILE=/workspace/sam_supervisor.pid
if [ -f "$PIDFILE" ]; then
  kill "$(cat "$PIDFILE")" 2>/dev/null
  rm -f "$PIDFILE"
  sleep 1
fi
pkill -f 'uvicorn perceptio[n]' 2>/dev/null
for i in $(seq 1 20); do
  ss -tlnp 2>/dev/null | grep -q ':8000 ' || break
  sleep 1
done
rm -f /dev/shm/tmp*.mp4 2>/dev/null
cd /workspace/nemara
export NEMARA_SAM_CKPT=/workspace/nemara/models/sam3.1_multiplex.pt
export PYTHONPATH=/workspace/sam3:/workspace/nemara
# FACE_DEBUG stays ARMED through the multi-person gauntlet (fitting
# data for the deep-mismatch bar); disarm after — see REVERT ledger.
export NEMARA_FACE_DEBUG_DIR=/workspace/face_debug
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SAM_RECYCLE_UPTIME_S=${SAM_RECYCLE_UPTIME_S:-21600}
export SAM_RECYCLE_IDLE_S=${SAM_RECYCLE_IDLE_S:-300}
setsid nohup bash -c '
  echo $$ > /workspace/sam_supervisor.pid
  while true; do
    uvicorn perception.server:app --host 127.0.0.1 --port 8000 \
      --timeout-keep-alive 75 \
      >> /workspace/sam_server.log 2>&1
    echo "$(date -u +%FT%TZ) supervisor: server exited rc=$? — restart in 3s" \
      >> /workspace/sam_server.log
    sleep 3
  done
' > /dev/null 2>&1 < /dev/null &
echo started
