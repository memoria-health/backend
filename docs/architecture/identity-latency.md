# Identity service latency — measured facts

> Written 2026-07-29 after an afternoon spent re-deriving the same
> ground repeatedly and getting it wrong five times. Everything here is
> MEASURED unless marked otherwise. The disproven-hypotheses section is
> the most valuable part: do not re-litigate those without new evidence.

## How to measure (do this first, always)

Every batch logs its full split since 2026-07-29 15:00:

```
identity.batch cam=X frames=N span=Ss stride=Nms |
  encode=E queue=Q pod=P (compute=C sam=S decode=D wire=W) publish=U
```

* `queue` — waiting for the CLIENT-side single-flight pod lock (i.e. the
  other camera's turn). Not the pod's queue.
* `pod` — lock acquired → response received. **Does NOT include queue.**
  (Getting this wrong cost hours: `pod - queue` is meaningless.)
* `compute` — the pod's OWN reported work (`timings_ms.total`).
* `wire` — `pod - compute`. Tunnel round trip + upload + response.

Report tool, takes any window:

```bash
uv run python tools/identity_timing_report.py --from 19:30 --to 20:15
uv run python tools/identity_timing_report.py --last 45
```

It reports per camera and combined, splits by CONCURRENCY (minutes with
one camera tracking vs both), and prints the sample count on every line
so a window straddling the instrumentation change can't mislead.

Raw log: `/tmp/identity_service.log`, appends across service restarts.
Pod-side log: `/workspace/sam_server.log` on the RunPod box.

## The cost model (measured)

```
tick = interval(0.6s) + encode + queue + pod + publish
pod  = compute + wire
```

| term | measured | notes |
|---|---|---|
| encode | 55-180ms | ~2% of a tick. Not worth touching. |
| publish | 7-17ms | <1%. |
| **wire** | **600-730ms** | **FLAT.** See below — not bandwidth. |
| **compute** | **79-81% of pod** | The whole game. |
| `sam` within compute | ≈ **720ms fixed + ~96-117ms/frame, LINEAR** | 110 batches, 8 buckets, 12→43 frames |
| decode (pod-side JPEG) | 50-170ms | scales with frames, small |

At 1 track and 12.8 frames: `sam` = 2364ms → **185 ms/frame**.

**The headline discrepancy:** `continuous.py` documents EdgeTAM
propagation at **~36ms/frame**. We measure **110-185ms/frame**, i.e.
**3-5x the design assumption**. Every capacity surprise follows from
this one number. Cause still UNKNOWN (see open questions).

### Grounding is NOT the problem

* SAM3 grounds on `frames[: min(6, len(frames))]` — **only the first 6
  frames**, never the whole batch. It is a fixed cost.
* `reground_every = 5`, and the pod log confirms it: **4 `mode=ground`
  to 14 `mode=carry`**, and **0 forced regrounds**. Working as designed.

## Capacity and the bistable behaviour

Per-camera GPU duty at TARGET_FPS=3 and the REAL per-frame cost is
~80%, not the ~11% the design assumed. Consequences:

* **One camera: always fine.** `queue=0`, cycle ~3.8-4.1s.
* **Two cameras: sits at ~98% utilisation — right on the edge.** Queue
  amplification near saturation goes as `1/(1-utilisation)`, so a ~2%
  perturbation flips it.
* **It is BISTABLE, not linear.** Same concurrency, two outcomes:
  * good equilibrium: ~16 frames/batch, queue ~1.1s, cycle ~5.5s
  * runaway: 41 frames (the `MAX_BATCH` clamp), pod 7-15s, lag 12-16s
* Nothing restores it except a room emptying. That is why it "only goes
  up" within an occupancy episode.
* `MAX_BATCH=40` is what converts an unbounded spiral into the ~15s
  plateau. It is the safety net, not the cause.

Historical, same measurement (12-frame batches, zero queue): Jul 26
2853ms · Jul 27 ~3100ms · Jul 28 ~3100ms · Jul 29 2695-3171ms.

### Empty rooms are free

A still empty room makes **ZERO pod calls** (publishes a frame and
returns). Only a motion-latched empty room polls, at ~2313ms for 4
frames. So N cameras cost nothing when idle — **the constraint is
concurrent OCCUPANCY, not camera count.** Five cameras with one
occupied room is fine; two occupied rooms is the problem.

## DISPROVEN — do not re-derive these

Each was tested and failed. Re-opening any of them needs NEW evidence.

1. **"Pod time is flat regardless of batch size."** False, and the
   error was arithmetic: `pod` already excludes `queue`, so `pod-queue`
   subtracted a term that wasn't there.
2. **"Wire/bandwidth is the bottleneck."** False. Wire is ~20% of the
   round trip and **did not change when frames were shrunk ~4x**
   (native 1080p ≈334KB/frame vs 960px ≈85KB/frame → wire stayed
   600-730ms). It is fixed per-request overhead, not bytes in flight.
3. **"Give each camera its own SSH tunnel."** Tested, 3 alternating
   rounds: shared 0.82 MB/s vs split 0.54 MB/s. No benefit, slightly
   worse. SSH multiplexing is not the constraint.
4. **"SAM3 is grounding on every tick."** False — 4:14 ground:carry,
   zero forced regrounds.
5. **"Cost scales with track count."** False. 1 track gives the same
   ~185 ms/frame. (Aryaman flagged this from experience before the
   data confirmed it.)
6. **"It degrades over time / session or memory leak."** False. At
   fixed batch size and zero queue, pod cost is flat across FOUR days.
   Pod GPU memory 5.6GB of 46GB. What rises within an episode is batch
   size, which is an OUTPUT of the loop, not a cause.
7. **"Shrinking image size will fix it."** FALSE, and it is slightly
   WORSE. Settled by a live A/B on 2026-07-29, ~4x smaller frames
   (960px vs native 1920px), 110 batches across 8 batch-size buckets.
   Compared at MATCHED frame counts (the only fair way — see the
   ms/frame trap below):

   | frames | native `sam` | 960px `sam` |
   |---|---|---|
   | 13.5 | 2074ms | **2735ms** |
   | 17.5 | 2507ms | **3023ms** |
   | 21.5 | 2786ms | **3320ms** |

   Wire also unchanged (~700ms vs 744ms) despite ~4x fewer bytes.
   Downscaling is 20-30% SLOWER at the same batch size — consistent
   with EdgeTAM upsampling back to its fixed internal resolution, so
   we pay a resize for nothing. The `TRACK_SEND_W` comment ("EdgeTAM
   encodes at its own fixed internal res") is CORRECT and Aryaman's
   2026-07-26 decision to ship native was right. Reverted.
   `NEMARA_TRACK_SEND_W` stays as an env override, but the answer is
   known: do not spend time here again.

## Measurement traps that produced wrong conclusions

* **`lag_s` in an EMPTY room is ~0 by construction** — the empty-room
  path publishes `ring[-1]`, the newest frame. A camera reading 0.03s
  means it stopped tracking, NOT that the pipeline sped up.
* **ms/frame across different batch sizes is not comparable** — `sam`
  is fixed + slope, so the ratio moves with batch size alone. Compare
  MATCHED BUCKETS, never a single ms/frame ratio. This trap nearly
  reversed the 960px verdict: the first sample (29.7 frames) read
  "160 ms/frame vs 155 native — no change", when the matched-bucket
  comparison showed 960px is 20-30% SLOWER.
* **`pod` is superlinear in frames but `sam` is LINEAR.** Do not quote
  a quadratic model for the tracker; the earlier "pod ≈ 2313 + 7.9N²"
  fit was on `pod` (compute + wire + contention effects), not on the
  pod's own compute. Measured on `sam` directly, the curve is a
  straight line at ~96-117 ms/frame from 12 to 43 frames.
* **Synthetic test frames produce `det_count=0`**, so EdgeTAM has
  nothing to propagate and `sam` goes flat (~1.9s for 1..30 frames).
  Any resolution/frame-count test built on synthetic frames measures an
  empty code path and will mislead.
* **Direct `/observe` calls bypass the client-side pod lock**, so they
  contend with live traffic and inflate each other. Never benchmark
  against a live occupied house without saying so.

## What this costs the product

Identity is ASYNC by design ("STABILITY IS THE PRODUCT; LATENCY IS
SACRIFICIAL", Aryaman 2026-07-24) — detection ticks read the published
seat and never wait on it. So:

* **Safety detectors are unaffected.** Fall / choking / calling-for-help
  are fail-loud and fire without identity confirmation; the emergency
  exemption bypasses the patient-elsewhere skip.
* **Attribution lags** the tracking lag plus the probe cadence (20s when
  the seat is held elsewhere) → realistically **30-60s** before the seat
  moves to a room the patient just entered.
* **Patient-scoped ROUTINE detectors can be MISSED, not delayed**
  (medication, drinking water, watching TV) — they fail quiet, and a
  tick landing inside the lag window skips with `patient_elsewhere`
  and does not retry.
* Sleep/activity minutes drop for that window (both require a
  confirmed patient); shows up honestly as reduced `labeled_minutes`.

## THE ROOT CAUSE (settled 2026-07-29 ~16:00)

**The pod serialises all inference behind its own global lock** —
`_perception_lock = threading.Lock()` in `perception/server.py:80`,
held at line 199 around the inference path. One request at a time,
by design, for the whole box.

Everything else follows from that:

* The client-side `pod_lock` was never the constraint — it merely
  SHADOWED the pod's lock. Removing it does not create parallelism; it
  moves the queue from our process to theirs.
* And it moves it somewhere INVISIBLE: the pod's `timings_ms` measures
  only work INSIDE the handler, so time spent waiting to enter the
  handler is charged to `wire` by our arithmetic (`pod - compute`).
  That is why "wire" appeared to explode to ~3.4s under concurrency
  and stayed there with 4x fewer bytes. **Wire under concurrency is
  pod-side queueing, not network.**

Measured, both rooms occupied:

| config | frames | queue | pod | compute | wire |
|---|---|---|---|---|---|
| single-flight (default) | 31.4 | ~1500 | ~7000 | ~5000 | **740** |
| concurrency=2 | 33 | 0 | 8942 | 5495 | **3447** |
| concurrency=2 + 960px | 26.9 | 0 | 7459 | 4329 | **3129** |

Same total either way. Both experiments REVERTED.

**The GPU is not the bottleneck and a bigger one is worthless.**
Measured under live two-camera load: GPU **5-8% mean, max 45%**, idle
88% of samples; **3 of 96 CPU cores**; **5.6GB of 46GB** GPU memory.
With concurrency=2 it rose only to 13% mean. The machine is ~97% idle
while the pipeline runs 15s behind. The cost is per-frame
orchestration (kernel launches, Python, tensor marshalling) inside a
serialised handler — not tensor math.

**Therefore the ONLY lever is less work: fewer frames.** Which is
exactly what Aryaman found empirically when dropping to 3fps fixed an
earlier version of this. Concretely: `TARGET_FPS` (arrival rate) or
`MAX_BATCH` (bounds the runaway; 40 frames permits a ~15s call).

If the pod's lock could be narrowed to only the GPU section — or the
pod run with several worker processes — real concurrency becomes
available and two cameras stop competing. That is a pod-side change
and the highest-value one remaining.

### PROVEN 2026-07-29 ~18:00 — the GPU DOES do real concurrency

A second, independent pod process was launched on port 8001 (its own
model copy, its own `_perception_lock`) and driven alongside the one
on 8000. Three alternating rounds, both rooms occupied:

| mode | wall | call A (pod compute) | call B (pod compute) | speedup |
|---|---|---|---|---|
| sequential | 18905ms | 8204 (2000) | 10691 (2029) | — |
| parallel | 10655ms | 8548 (2349) | 10644 (2006) | **1.77x** |
| sequential | 17139ms | 8376 (2005) | 8752 (2040) | — |
| parallel | 10638ms | 10632 (1871) | 8605 (1962) | **1.61x** |
| sequential | 18367ms | 10346 (1951) | 8012 (2003) | — |
| parallel | 10307ms | 8374 (1954) | 10299 (1879) | **1.78x** |

**Pod-side compute stayed 1.9-2.3s in BOTH modes.** The two processes
did not contend for the GPU at all. So `_perception_lock` is a
software constraint protecting a shared predictor object, not a
hardware limit — Aryaman's hypothesis, confirmed.

The shortfall from 2x is the TUNNEL, not the GPU: each call showed
8-10.6s wall against ~2s of compute because the probe traffic shared
the uplink with the live cameras' ~13MB batches.

**The bottlenecks are stacked in series — three of them.** This is
why every single-variable experiment today read as "no change":
remove one and the next binds immediately, so the total does not move.

1. client-side `pod_lock` → removed it, time relocated to (2)
2. pod-side `_perception_lock` → two processes fix it (proven above)
3. the uplink → binds only once (1) and (2) are gone

Corollary: finding 7 above ("shrinking images won't help") is true
**at the current layer** and stays true until (2) is fixed. Revisit
downscaling ONLY after per-camera pod processes ship, and only for the
uplink — never for `sam` compute, where it measured 20-30% slower.

Production shape (NOT yet built, needs its own review): one pod
process per camera on its own port, cameras routed to their own
process. Session affinity comes free — `_SESSIONS` is process-global,
so each camera's EdgeTAM state lives with its own process instead of
round-robining. Two model copies cost ~11GB of 46GB. Touches the
client (per-camera pod URL) and the supervisor (N processes).

## Open questions (the ones actually worth time)

1. **Why is propagation 3-5x its documented 36ms/frame?** Not
   resolution, not track count, not grounding, not a leak — all
   disproven above. Remaining candidates: the 36ms figure was measured
   on different hardware or a different EdgeTAM build; or the A40 is
   simply slower than whatever produced it; or something in the
   propagate path regressed before instrumentation existed.
2. **The ~580ms fixed floor in `sam` and the ~650ms fixed floor in
   `wire`** — together ~1.2s of pure per-call overhead, paid on every
   batch regardless of size. Worth attacking before frame counts.
3. If (1) can't be closed, the honest options are fewer frames
   (`TARGET_FPS` below ~1.9 for two cameras), a lower `MAX_BATCH` to
   bound the runaway, or a second GPU. Colocation helps LATENCY, not
   bandwidth — do not sell it as a bandwidth fix.
