# GPU node load test + optimization — 2026-08-13

Question under test (Aryaman): how many concurrently-occupied cameras can
one GPU node serve while every home's identity lag stays under the 7-10s
SLA (always shorter than a tick), and what breaks first as load grows.

Method: `stage_profile.py` (session scratchpad) drives the REAL bedroom
camera plus 15fps H.264 replay publishers (2.5Mbps 720p re-encodes of a
real clip) into synthetic homes on the same node — levels 1→2→3 occupied
cameras, ~4 min per level. Per-batch stage timings come from the
service's own instrumentation (`/state.stage_ms`); `lag_s` = wall time
from a frame's intake to its published verdict, sampled per batch (not
probe cadence — probe intervals are state-dependent and invalid as an
SLA metric). CPU/GPU from RunPod telemetry (coarse ~60-90s sticky
windows; treat single samples as plateaus, not instants).

## Baseline — gpu-v1 (node C, 2026-08-13 morning)

| stage p50 (ms/batch) | 1 cam | 2 cams | 3 cams | growth |
|---|---|---|---|---|
| encode (JPEG, identity) | 14 | 183 | 390 | ×27 |
| wire (pod_ms − pod compute) | 32 | 946 | 5,692 | ×178 |
| pod_compute (perception) | 139 | 1,586 | 4,302 | ×31 |
| publish | 4 | 4 | 9 | ~flat |
| queue (admission) | 0 | 0 | 0 | never waits |
| lag_s p50 / p95 (s) | 0.22 / 0.70 | 3.6 / 7.1 | 10.9 / 19.2 | SLA dead at 3 |
| CPU / GPU util | 3% / bursts | 81-98% / ~0 | 96-100% / 5-13% | inverted |

Thread-cap A/B (OMP/MKL/OPENBLAS=2, node D): falsified the
thread-thrash theory — no change at 1-2 cams, WORSE at 3 (4 batches in
4 min vs 59). The ~1,260 threads were idle pools, not contention.

## Root causes (both proven, not inferred)

1. **Face embedding ran on CPU on every node since gpu-v1.** The
   Dockerfile installed `onnxruntime-gpu` early; insightface's pip layer
   declares plain `onnxruntime`, pip does not treat `-gpu` as satisfying
   it, and the CPU wheel overwrote the module. Proof: node E console —
   `Available providers: AzureExecutionProvider, CPUExecutionProvider`
   on an RTX 4090. ORT's failure mode is a warning + silent fallback.
2. **JPEG loopback hop.** Identity and perception share the box, yet
   every batch paid JPEG encode (identity) → starlette multipart parse
   (>1MB parts spool through DISK) → JPEG decode (perception), all out
   of the same starved cores. Probe loop additionally JPEG-encoded
   NATIVE frames every 1.5s while searching.
3. Once CPU pegged, everything queued behind it — "wire" ×178 is
   client-side waiting, not network. GPU starved at 0-13% because
   kernel dispatch itself was starved of CPU.

## Fixes (gpu-v2 → v2.2, commits c73ab35, 193a494, +roster agent)

- onnxruntime-gpu (+ cuDNN 9 via ldconfig) installs LAST, after every
  layer that could shadow it; build FAILS unless the CUDA provider is
  present in the installed module, cuDNN loads, and every linked lib
  resolves. Runtime: perception refuses to serve if faces land on CPU
  with a GPU present (`NEMARA_FACE_REQUIRE_GPU=0` overrides).
- Raw frame wire (`perception/rawwire.py`, one module owns pack+parse):
  8-byte length + JSON header + BGR pixels via `/observe_raw` and
  `/faces_raw`. Multipart JPEG routes remain for off-box callers.
  `NEMARA_FRAME_WIRE=jpeg` is the one-env revert.

## Result — gpu-v2.2 (node G, same 4090 class, same clips, levels 1-5)

Two runs: 1-3 (first pass), then a full 1-5 ladder with a per-camera
breakdown proving every synthetic home ran the REAL identity path (own
service process, own LiveKit room, own tracking session — synthetics
are HARSHER than a locked real home: no gallery, so face probes run at
the 1.5s search cadence, not 20s).

| stage p50 (ms/batch) | 1 cam | 2 cams | 3 cams | 4 cams | 5 cams |
|---|---|---|---|---|---|
| encode | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| wire | 42 | 40 | 64 | 86 | 209 |
| pod_compute | 99 | 148 | 202 | 270 | 321 |
| queue | 0 | 0 | 0 | 0 | 0 |
| lag_s p50 / p95 (s) | 0.19/0.64 | 0.25/0.86 | 0.34/1.04 | 0.52/1.49 | 0.79/1.89 |
| CPU util | 5-8% | 11-18% | 20-26% | 25-33% | 30-43% |
| GPU util | 0-34% | 51-82% | 79-87% | 62-81% | **91-100%** |

Per-camera balance held at every level (L5: 31/32/32/32/31 batches per
camera; real camera lag p50 0.78s vs synthetics 0.57-0.94s).

**Level-5 saturation event.** At t≈169-177s into level 5, ALL five
cameras stopped batching within one 8s window and GPU fell 100%→1% —
a shared-resource death at full saturation, consistent with the
perception process dying (most plausibly VRAM exhaustion: 5 concurrent
EdgeTAM sessions + prime caches on a 24GB card; same signature as the
node-E OOM). The supervisor restarted it and the node self-recovered to
healthy (verified ~25 min later: /health ok, faces on CUDA, roster
fresh). Level-5 stats cover the 158 batches over ~3 min of full load
before the halt. Note: the real room emptied (Aryaman left) ~4.5 min
before the halt, yet the real camera batched until the shared halt —
so the halt was not caused by the room emptying.

## Capacity statement (MEASURED)

- **4 concurrently-occupied cameras per 4090 node is the comfortable
  operating point**: lag p50 0.52s / p95 1.49s (SLA margin ~14×), GPU
  62-81%, CPU ~30%. Stable for the full window.
- **5 is the redline, not the rating**: latency still fine (0.79s p50)
  but GPU pins at 91-100% and the perception process died at full
  saturation once. Do not plan capacity at 5 without the VRAM work
  below.
- Idle homes cost ~0 (idle node: 0% CPU). Homes-per-node therefore
  multiplexes well above occupied-camera count; with typical occupancy
  duty cycles the 8-home capacity setting is consistent with the
  4-occupied-camera rating, but FleetWatch must alarm on sustained
  lag_s > tick length as the real-world guard.
- Follow-ups to raise the ceiling: VRAM budget per session (disable
  feature-cache priming when >4 sessions are live — priming is worth
  ~14% speed but ~2.5GB/session), and an admission cap on concurrent
  occupied sessions per node.
- Separate ceiling (unchanged today): backend/recording sharding needed
  past ~5 homes — see TESTING-2026-08-12 scale note.

## Fleet post-mortems from the same day (each now guarded)

| # | Failure | Guard now in place |
|---|---|---|
| 1 | CPU onnxruntime silently shadowing -gpu (v1, v2) | build-time CUDA-provider assert + boot refuse-to-serve |
| 2 | Node E: guard crash-loop orphaned ~90 CUDA procs → GPU OOM | face child reaped before exit; perception supervisor backoff 3s→120s |
| 3 | No shell on any image node (sshd never started; RunPod injects PUBLIC_KEY only for console pods) | entrypoint starts sshd when PUBLIC_KEY present; pod.sh forwards PUBLIC_KEY |
| 4 | Perception log file-only + shared-volume interleave | tee to console; per-container filenames |
| 5 | Node F melt: roster run-state on shared /persist (stale pids, PID reuse), stop-race orphans, duplicate services kicking each other off LiveKit, 3s no-backoff respawn → fork exhaustion (took down shell access too) | run-state container-local (/run/nemara); stop kills supervisor first + verifies death (escalates -9); start pattern-kills stragglers; respawn backoff 3s→60s |
| 6 | Local backend (Mac) wedged silently under the churn storm — zero logs, /health timeout — blocking all roster polls | recovery = docker restart (existing runbook). OPEN: needs a liveness alarm; FleetWatch ledger |

## Open items

- FleetWatch alarm on sustained lag_s > tick length (ledger).
- Confirm the L5 perception death was VRAM OOM (perception log on node
  G's console has the traceback — tee'd since v2.2) and add the VRAM
  guards: per-session budget, prime-cache off above 4 sessions,
  occupied-session admission cap.
- Soak dashboard must point at the CURRENT node (roster-driven lookup,
  not a hardcoded base URL) — it was left aimed at dead node C today.
- Backend liveness alarm (post-mortem #6).
- identity_nodes dead rows accumulate (C/D/E/F now) — reaper or cleanup
  in the M3 deploy batch.
- RunPod account: SSH key added to Settings; balance top-up (was $11.68
  mid-day with two nodes billing).
