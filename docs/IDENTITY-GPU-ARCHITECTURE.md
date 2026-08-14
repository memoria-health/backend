# Identity service — architecture, deployment, and measured capacity

The identity system answers one question continuously, per home: **is
the person the cameras see the patient** — so detections can be
attributed (patient vs visitor vs staff) and patient-scoped detectors
can skip rooms the patient is not in. It must answer within the tick
(SLA: always under 7-10s), fail OPEN (an identity outage must never
silence detection), and cost nothing in empty rooms.

Two services share each GPU node, colocated on purpose (identity ships
multi-MB frame batches to perception every cycle; on one box that hop
is a memcpy, across RunPod's 100Mbps inter-pod link it would be ~1s per
batch):

```
 Tapo cams ──RTSP──> gateway box ──WebRTC──> LiveKit room patient_<id>
                                              │  participants cam_<id>
                                              ▼
   ┌──────────────────────────── GPU node (one container) ─────────────┐
   │  identity service (one PROCESS per home, port 8012+)              │
   │    room join → CameraLoop per cam_* participant                   │
   │    ring buffer → batch → raw wire → verdict → seat                │
   │        │ /observe_raw, /faces_raw (loopback :8000)                │
   │        ▼                                                          │
   │  perception service (one process, shared by all homes)            │
   │    SAM3 grounding + EdgeTAM continuous session per camera         │
   │    face worker (SCRFD+ArcFace, isolated child process, CUDA)      │
   │  roster agent (poll = heartbeat; spawns/stops home processes)     │
   │  tailscaled (userspace) · sshd · socat DB bridge (local rig)      │
   └───────────────────────────────────────────────────────────────────┘
              ▲ roster poll /v1/identity/roster (30s)
        backend (assigns homes+ports to nodes; reads /state for seat)
```

## 1. Fleet layer — nodes are cattle

- A node is a RunPod GPU pod created from an IMAGE TAG
  (`ghcr.io/…/memoria-gpu:gpu-vX`, built by CI on tag push). The tag is
  the artifact: a lost node is replaced by `pod.sh release <tag>`,
  never repaired. Model checkpoints live on a shared network volume
  (`/persist`), fetched once, mounted per node.
- **Roster protocol**: the node POSTs
  `{node_id, base_url, capacity}` to the backend every 30s. The poll IS
  the registration AND the heartbeat — 90s of silence and the backend
  releases the node's homes to whoever polls next (measured failover:
  ~82s + camera re-attach). The response is the authoritative list of
  `{room, port}` this node must run; the roster agent reconciles
  reality to it: start what is newly assigned, stop what left, touch
  nothing else. Ports come FROM the backend (it must dial them), never
  derived on the node.
- Run-state (`<room>.pid/.port`) is CONTAINER-LOCAL (`/run/nemara`) —
  never on the shared volume. Stops kill the supervisor first, then
  pattern-kill the room's service, then VERIFY death (escalating to
  -9). Respawns back off 3s→60s. Each rule is a scar from the node F
  melt (post-mortem table, §6).
- The backend reaches identity over the tailnet
  (`base_url = http://<tailnet-ip>`, `:8012+` published via
  `tailscale serve`). Userspace tailscaled (no /dev/net/tun on RunPod)
  means every outbound dial into the tailnet must ride its local
  proxies (:1055 HTTP CONNECT; asyncpg gets a socat TCP bridge for the
  local-rig DB). Production backend/DB/LiveKit are public — no proxy.

## 2. Identity service — one process per home

Entry: `python3 -m identity.service --room patient_<id> --patient <id>
--port <port> --pod-url http://127.0.0.1:8000`. Joins the LiveKit room
with `auto_subscribe`; every `cam_*` video track gets a `CameraLoop`.
A 10s "missed subscription" walk re-toggles any unsubscribed or quiet
camera (gateway restarts can re-publish without a callback); 6 failed
walks → deliberate self-exit → supervisor rebuild. Room disconnect →
same doctrine: die loudly, be rebuilt fresh.

**Intake** (per camera): rtc frames at 15fps into a ~20s ring, each
stamped at ARRIVAL (`pipeline_ts` — the timestamp `lag_s` is measured
from). Ships NATIVE resolution (decision 2026-07-26: downscale cut
wire ~9x but GPU only ~10%; frame COUNT is the cost lever, resolution
is detection recall).

**Batching**: `plan_batch` decimates the ring to TARGET_FPS=3 from the
last shipped frame (10-19 frames typical, MAX_BATCH=40 ceiling). Under
load the stride STRETCHES instead of lag compounding. Per-home
`pod_lock` bounds admission (cameras within a home serialize; homes
compete at perception's own gate).

**Empty-room economy**: with no tracks, the loop ships NOTHING. A
64×36 grayscale thumbnail diff (mean abs ≥ 3.0, sampled 0.5s) latches
motion; only then does an existence-check batch go to the pod. Zero
GPU, zero wire for an empty house — this is why homes-per-node
multiplexes far above the occupied-camera rating (§5).

**Raw frame wire** (2026-08-13): batches go to perception as raw BGR
pixels — 8-byte length + JSON header + concatenated frames — via
`/observe_raw`. One module (`perception/rawwire.py`) owns pack AND
parse so the services cannot drift. The JPEG multipart routes remain
for off-box callers; `NEMARA_FRAME_WIRE=jpeg` is the one-env revert.
(The JPEG hop was a top CPU stage: encode alone p50 390ms at 3 cams,
plus starlette spooling >1MB parts through disk, plus decode — §5.)

**Identity is OUT-OF-BAND** (2026-07-24 decoupling): tracking batches
carry `face_cap=0` — the tracking cadence never pays for identity.
A separate probe loop calls `/faces_raw` on the newest native frame:
every 1.5s while SEARCHING (plus a full-frame pass and a coverage scan
over ring frames, so a brief walk-past can never fall between checks),
every 20s once LOCKED. Face evidence fuses with body embeddings
(OSNet, local) into the SEAT — single patient holder per home, sticky
lock, quality-gated harvesting into the DB gallery (the gallery never
leaves the backend/DB; the pod makes zero identity decisions).

**Guards**: stale intake (10s without frames → clear tracks — a dead
reader must not hold the seat); pod epoch change → hard reset (a pod
restart recycles track ids; inheriting one would donate the seat);
screen-region veto (SAM screen concept every 120s; faces inside TV/
laptop regions are discounted — the screen-face incident).

**Observability**: `/state` (bearer-auth) exposes per-camera tracks,
seat, `lag_s` (intake→published verdict — THE SLA metric), `pod_ms`,
and `stage_ms` {encode, queue, wire, pod_compute, publish}; `/stream`
serves the annotated debug view. The soak dashboard reads these.

## 3. Perception service — one process per node, perception ONLY

FastAPI on loopback :8000 (never a public bind; bearer-auth shared
token). Returns RAW observations — no labels, no gallery, no
thresholds. Endpoints: `/observe(_raw)` (track people across a batch),
`/faces(_raw)` (detect+embed on known boxes / full frame), `/health`
(open; exposes `face_provider` — see guard below).

- **Tracking**: SAM3 concept grounding + EdgeTAM CONTINUOUS session
  per camera — one persistent session yields stable track ids across
  ticks (periodic SAM3 re-ground + EdgeTAM carry-propagate). Frames
  are staged straight from memory into the model tensor
  (`frame_tensor.py` — the disk round-trip was 68% of every call), and
  a feature-cache "prime" batches the encoder pass (~14% speed,
  ~2.5GB/session at cap 24 frames — the VRAM lever in §5).
- **Faces**: SCRFD + ArcFace (buffalo_l) in an ISOLATED child process
  — onnxruntime-gpu and torch fight over cuDNN in one address space.
  Strict pipe protocol (one in-flight request, lock-guarded; a desync
  would attach one person's embedding to another's crop), circuit
  breaker on repeated failure, auto-restart.
- **Admission**: N-way gate (`NEMARA_PERCEPTION_CONCURRENCY`, node
  default 2) with a bounded acquire (10s) → instant 429 → identity
  fails open. Queue depth is deliberately ~zero: when service time
  exceeds tick period, queueing converts 429s into timeouts while
  burning GPU on abandoned work.
- **Hygiene**: `torch.cuda.empty_cache()` per pass and an
  uptime+idle self-recycle (allocator fragmentation once stretched
  session-create from <1s to 9.5s); the entrypoint supervisor restarts
  it (backoff 3s→120s) — a perception death costs one recovery window,
  never the node.
- **Refuse-to-serve guard**: at startup, if a GPU is present and the
  face provider is not CUDA, perception logs CRITICAL, reaps the face
  child, and exits — the node never becomes ready. ORT's native
  failure mode is a silent CPU fallback that cost us every node until
  2026-08-13 (§6 #1). Build-time asserts back this up: the installed
  onnxruntime must LIST the CUDA provider, cuDNN 9 must load, and
  every lib the provider links must resolve, or the image does not
  build. `NEMARA_FACE_REQUIRE_GPU=0` overrides for deliberate CPU
  sessions.

## 4. Image, CI, and boot

`gpu/Dockerfile` (context = repo root): runpod/pytorch 2.4/cu124 base;
pinned SAM3+EdgeTAM checkouts with build-asserted patches; pinned ML
deps (`--no-deps` so pip cannot swap the CUDA torch); onnxruntime-gpu
+ cuDNN installed LAST (nothing may shadow them — §6 #1); OSNet
weights baked; smoke-imports both services at build. CI builds on tag
push (`gpu-v*`), runs the unit suite first, pushes with
`provenance: false` (attestation manifests break older dockerd pulls).

Boot (`entrypoint.sh`): sshd (when PUBLIC_KEY provided — pod.sh
forwards it; RunPod only injects it for console-created pods) →
checkpoints → tailscale supervisor → perception (log TEES to console
with per-container filename — a file-only log on a shell-less node is
a black box) → DB bridge (local rig) → tailnet name → roster agent.

## 5. Measured performance and capacity (2026-08-13)

Question (Aryaman): how many concurrently-occupied cameras can one
node serve with lag always under the 7-10s SLA, and what breaks first.

Method: `stage_profile.py` drives the REAL bedroom camera plus 15fps
H.264 replay publishers (2.5Mbps 720p re-encodes of a real clip) into
synthetic homes on the same node. Per-batch stage timings from
`/state.stage_ms`; `lag_s` = intake→published verdict (probe cadence
is state-dependent and invalid as an SLA metric). RunPod telemetry is
coarse (~60-90s sticky windows) — treat samples as plateaus.
Synthetic homes run the REAL path (own service process, own room, own
session) and are HARSHER than a locked real home: no gallery → face
probes at the 1.5s search cadence, not 20s.

### Baseline — gpu-v1 (node C)

| stage p50 (ms/batch) | 1 cam | 2 cams | 3 cams | growth |
|---|---|---|---|---|
| encode (JPEG, identity) | 14 | 183 | 390 | ×27 |
| wire (pod_ms − pod compute) | 32 | 946 | 5,692 | ×178 |
| pod_compute (perception) | 139 | 1,586 | 4,302 | ×31 |
| lag_s p50 / p95 (s) | 0.22 / 0.70 | 3.6 / 7.1 | 10.9 / 19.2 | SLA dead at 3 |
| CPU / GPU util | 3% / bursts | 81-98% / ~0 | 96-100% / 5-13% | inverted |

Thread-cap A/B (OMP/MKL/OPENBLAS=2): falsified — no change at 1-2
cams, WORSE at 3. Root causes (both proven live): (1) the CPU
onnxruntime wheel had silently shadowed onnxruntime-gpu since gpu-v1 —
every face probe on CPU; (2) the JPEG loopback hop. Once CPU pegged,
"wire" (client-side waiting) exploded while the GPU starved.

### After — gpu-v2.2 (node G, same 4090 class, same clips)

| stage p50 (ms/batch) | 1 | 2 | 3 | 4 | 5 cams |
|---|---|---|---|---|---|
| encode | 0.1 | 0.1 | 0.1 | 0.1 | 0.1 |
| wire | 42 | 40 | 64 | 86 | 209 |
| pod_compute | 99 | 148 | 202 | 270 | 321 |
| lag_s p50 / p95 (s) | 0.19/0.64 | 0.25/0.86 | 0.34/1.04 | 0.52/1.49 | 0.79/1.89 |
| CPU util | 5-8% | 11-18% | 20-26% | 25-33% | 30-43% |
| GPU util | 0-34% | 51-82% | 79-87% | 62-81% | **91-100%** |

Per-camera balance held at every level (L5: 31/32/32/32/31 batches;
real camera lag within the synthetic band). At 3 cams vs baseline:
lag p50 ×33 better, throughput ×2.8, CPU freed 96%→20%.

**Level-5 saturation event**: at t≈170s all five cameras halted within
8s and GPU fell 100%→1% — perception died at full saturation (most
plausibly VRAM: 5 EdgeTAM sessions + prime caches on 24GB; same
signature as the node-E OOM), then self-recovered via the supervisor.
Not caused by the real room emptying (it had emptied 4.5 min earlier
and the camera kept batching until the shared halt).

### Capacity rating

- **RATED: 4 concurrently-occupied cameras per 4090 node** — lag p50
  0.52s / p95 1.49s (~14× SLA margin), GPU 62-81%, stable.
- **5 is the redline, not the rating**: latency fine (0.79s) but GPU
  pins and perception died once. Raising the ceiling needs the VRAM
  work: prime-cache off above 4 live sessions, per-session VRAM
  budget, occupied-session admission cap.
- Idle homes cost ~0 (idle node: 0% CPU) — homes-per-node multiplexes
  well above the occupied rating; capacity=8 homes/node is consistent
  with realistic occupancy duty cycles, with FleetWatch alarming on
  sustained lag_s > tick as the production guard.
- 6-10 camera synthetic run: see the addendum at the end of this doc.
- Separate ceiling: backend/recording sharding needed past ~5 homes
  (TESTING-2026-08-12 scale note).

## 6. Fleet post-mortems (2026-08-13, each now guarded)

| # | Failure | Guard now in place |
|---|---|---|
| 1 | CPU onnxruntime silently shadowing -gpu (v1, v2) — faces on CPU on every node | ORT installs last + build-time CUDA-provider assert + boot refuse-to-serve |
| 2 | Node E: guard crash-loop orphaned ~90 CUDA procs → GPU OOM | face child reaped before exit; perception supervisor backoff 3s→120s |
| 3 | No shell on any image node (sshd never started; RunPod injects PUBLIC_KEY only for console pods) | entrypoint starts sshd when PUBLIC_KEY present; pod.sh forwards PUBLIC_KEY |
| 4 | Perception log file-only + shared-volume interleave | tee to console; per-container filenames |
| 5 | Node F melt: roster run-state on shared /persist (stale pids, PID reuse), stop-race orphans, duplicate services kicking each other off LiveKit, 3s no-backoff respawn → fork exhaustion (took down shell access too) | run-state container-local; stop kills supervisor first + verifies death; start pattern-kills stragglers; respawn backoff |
| 6 | Local backend (Mac) wedged silently under the churn storm — zero logs, /health timeout — blocking all roster polls | recovery = docker restart (runbook). OPEN: liveness alarm |

## 7. Open items

- **/health is too shallow (found 2026-08-13 evening).** After the L5
  OOM, perception restarted, could not load SAM (orphaned CUDA
  children still held VRAM), tolerated it ("will warm lazily"), and
  reported `status: ok` for ~5 hours while every /observe returned
  500. Health must exercise the tracker (fail when get_sam fails) so
  the roster/FleetWatch can see a node that cannot actually serve.
- **Supervisor must reap orphaned CUDA children before relaunching
  perception** — the guard-path reap (post-mortem #2) does not cover
  an OOM/abort death; the orphans blocked the successor's SAM load.
  Interim recovery: container restart.
- FleetWatch alarm on sustained lag_s > tick length (ledger).
- Confirm the L5 perception death was VRAM OOM (node G console log)
  and add the VRAM guards: per-session budget, prime-cache off above 4
  sessions, occupied-session admission cap.
- Soak dashboard must follow the roster for the current node (was left
  aimed at a dead node today).
- Backend liveness alarm (post-mortem #6).
- identity_nodes dead rows accumulate — reaper or cleanup in the M3
  deploy batch.
- RunPod account: SSH key in Settings; balance top-up cadence.

## Addendum — 6-10 camera synthetic run (2026-08-13 evening)

All-synthetic (real room empty): 10 cameras across the 5 synthetic
homes, two publishers per room at full load — one CameraLoop per
camera, per-home pod admission, i.e. the multi-camera-house shape.
Run AFTER a clean container restart (the earlier attempt found the
shallow-/health outage, §7). ~3 min holds at 6, 8, 10.

| | 6 cams | 8 cams | 10 cams |
|---|---|---|---|
| lag_s p50 / p95 | 2.09 / 6.59 | 4.04 / 6.84 | 7.38 / 13.24 |
| pod round-trip p50 | 2.0s | 4.0s | 7.2s |
| pod_compute p50 | 788ms | 1,001ms | 1,127ms |
| wire (≈ queueing) p50 | 1.1s | 2.8s | 5.7s |
| CPU / GPU | 48-98% / 91-92% | 53-85% / sat. | 66-69% / thrash |
| batches | 163 | 206 | 34 (stalled) |

Reading: past GPU saturation (~5 cams) added cameras stretch QUEUEING
(wire ≈ waiting for the busy pod), not compute — pod_compute grows
gently while wire explodes. 6-8 cameras stay INSIDE the 7-10s SLA
(p95 <7s) — a legitimate degraded surge envelope. At 10, lag p50
breaches the SLA and throughput collapses into admission-timeout
thrash — but the node survives: no crash, health honest, fail-open
shedding as designed (contrast the L5 OOM death, which the pending
VRAM/orphan/health guards address). Per-camera balance held at every
level.

**Operating guidance**: rate 4 concurrently-occupied cameras
(comfort, ~14× margin) · tolerate 6-8 as a surge envelope (SLA holds,
no headroom) · never plan for 10 (broken). Caveat: this run used 2
cameras/home across 5 homes; 10 single-camera homes would contend
slightly differently at the same perception gate.

## Addendum 2 — GPU class comparison (L4 ladder, 2026-08-13 night)

Same 1-10 ladder on an NVIDIA L4 ($0.49/hr, 24GB, ~300GB/s) under
gpu-v2.3.1, synthetic homes moved to the test node via their DB
assignment (real home undisturbed on the 4090), node terminated after.

| | 4090 (1008 GB/s, $0.74) | L4 (300 GB/s, $0.49) |
|---|---|---|
| lag p50 @1/2/3 cams | 0.19 / 0.25 / 0.34s | 0.53 / 0.77 / 0.77s |
| collapse point | 10 cams | 4 cams |
| rated (comfort) | 4 | 2 |
| $/occupied camera | $0.185 | $0.245 |

The ~3× throughput gap matches the memory-bandwidth ratio almost
exactly — bandwidth (with FP16 tensor rate) is the workload's scaling
predictor, so consumer flagships beat cut-down datacenter inference
cards for dense video tracking. v2.3.1 hardening validated under
saturation on the weak card: prime-skip fired at 81% VRAM, zero
perception deaths, deep health honest, collapse was graceful shed.

**3090 ladders (community cloud, $0.22/hr, 2026-08-14).** After two
infra lemons (EU-CZ-1 corrupt image cache — `exec entrypoint.sh: No
such file or directory` from an image that runs everywhere else; and
two old-driver community hosts exposing NO GPU — fixed permanently by
the `allowedCudaVersions` filter in pod.sh), a healthy CUDA-filtered
host ran the full ladder on v2.2:

| 3090 (healthy, v2.2) | 1 | 2 | 4 | 6+ |
|---|---|---|---|---|
| lag p50 / p95 | 0.39/1.22s | 0.53/1.67s | 2.21/3.51s | hard stall |
| batches | 34 | 66 | 86 | 2 |

Rated **3 comfortable / 4 strained-in-SLA**, collapse at 6. At $0.22 →
**$0.055-0.073/occupied camera** — but even healthy community 3090s ran
~2.5× slower per batch than the 4090 (systematic community throttling
suspected; secure-3090 unmeasured), and community tier is non-prod for
patient data (custody). Fleet stays on the secure 4090.

**The collapse model (replaces the earlier "v2.3.1 level-4 stall
regression" framing, which was confounded).** Every card hard-stalls —
throughput collapses into admission-timeout thrash, health honest, no
crash — at roughly **2-2.5× its rated capacity**: L4 rated 2 → stalled
at 4; 3090 rated 3 → stalled at 6; 4090 rated 4 → stalled at 10. The
production guard is the FleetWatch lag alarm, which fires at 1× (lag >
tick), far below the cliff.

**The real v2.3 regression, convicted and fixed (gpu-v2.3.2).**
Same-host A/B/C, one variable per run, rung-4 batches:
v2.2 → **86** · v2.3.1 → **13** · v2.3.1 with
`PYTORCH_CUDA_ALLOC_CONF` blanked → **79**. The expandable_segments
allocator conf (added in v2.3 as fragmentation insurance) collapses
concurrent multi-session throughput ~6× under torch 2.4 and had pulled
the thrash cliff down to the rated capacity itself. v2.3.2 deletes the
ENV line; every other guard in the hardening bundle is exonerated, and
the C run doubles as v2.3.2's load validation (identical code path).
Fragmentation stays covered by empty_cache-per-pass + self-recycle.

GPU procurement guidance: 4090 stays the fleet default; RTX 3090
(936 GB/s at ~⅓-½ the price → est. $0.07-0.14/cam) and RTX A5000
(768 GB/s → est. $0.10-0.14/cam) are the two candidates worth one
ladder each when EU-RO-1 stock appears; skip L-series small cards.
Community-cloud 4090s (~half price) are a deliberate experiment for
non-anchor nodes — replace-don't-repair + 82s failover makes
interruptible capacity tolerable by design. None of this changes the
cost curve's SHAPE — that is the detector-cascade epic (SAM demoted
to episodic) — GPU choice is the coefficient.
