# Cross-camera batching — future enhancement plan

Status: **deferred by design** (2026-08-16, Aryaman's call). Build it only
if the cheaper capacity levers — 4 workers on a 48GB card, stateless
workers with a shared queue, the SDP-pin and no-gc speedups — measurably
fail to reach ~15-20 occupied cameras per node inside the 10s SLA.
This document exists so that whoever picks it up later can build it from
first principles without re-deriving the constraints.

## 1. The constraint every design here must respect

The 2026-08-15/16 corruption campaign established one hard fact:

> **Two inference passes running at the same time inside one Python
> process silently corrupt EdgeTAM's tracking** — a present, visible
> person vanishes from the carry with no error raised.

Every reachable software cause was falsified by experiment: private
model instances per pass (17 drops/362 ticks), stubbing gc + CUDA cache
flushes (35/618), pinning the SDPA kernel-selection flags (85/2165),
serializing the shared YOLO (27/448), and all of those combined
(52/1462). Only two configurations ever produced zero drops across
~5,000 scored ticks: full serialization inside a process, and separate
worker processes. The fault therefore lives below Python — in the CUDA
context / torch substrate that threads share — and cannot be patched
from our side.

**Consequence:** capacity can never come from letting passes overlap in
one process. It can only come from (a) more worker processes, or
(b) making a single pass serve more than one camera. Cross-camera
batching is option (b). It is attractive precisely because it is
concurrency-safe *by construction*: one pass, one thread, one model —
just more frames in the tensor.

## 2. Where the time actually goes (the cost model)

A perception pass on one camera's batch (measured on A40/4090, 640-wide
track resolution, 2fps, ~12-24 frames per tick):

| Stage | Share of pass | Batchable across cameras? |
|---|---|---|
| Frame decode + tensor prep (CPU) | ~15% | yes — trivially |
| EdgeTAM image encoder (per frame) | ~31% of propagation | **yes — this is the target** |
| EdgeTAM memory attention + mask decode | ~69% of propagation | only with model surgery (§5) |
| SAM3 ground (every ~5 ticks) | ~0.7s when it fires | separate problem, see §6 |
| Face crops / embedding | small, async | already off the critical path |

The encoder is a pure function of pixels: it never touches tracking
state. That is why it batches safely. Memory attention and mask
decoding are stateful per camera (each camera has its own tracked
objects, and the tensors are shaped by that camera's object count),
which is why they do not batch without restructuring.

## 3. Tier 1 — encoder-stage batching (bounded, ~3-5 days)

**Idea in one sentence:** when several cameras' batches are waiting at
the same worker, run ONE image-encoder forward over all their frames
stacked into a single tensor, cache each camera's features, then run
each camera's stateful propagation serially off its cached features.

**Why the plumbing already 80% exists:** `gpu/perception/
frame_tensor.py::prime_feature_cache` already does exactly this shape
for ONE camera — it encodes all frames of a batch in one forward and
pre-fills the predictor's feature cache so propagation never calls the
encoder. The extension is stacking frames from k cameras instead of 1
and splitting the resulting features back per camera.

**Implementation sketch:**
1. Give each worker a tiny admission window (~100-200ms). Passes that
   arrive within the window are grouped instead of queued serially.
   (The worker stays single-flight: the *group* is one pass.)
2. Concatenate the groups' track-res frame tensors along the batch
   dimension; run `predictor.forward_image` once (respect
   `_PRIME_MAX_FRAMES` and the VRAM floor already in
   `prime_feature_cache` — those guards exist because of a real OOM,
   2026-08-13).
3. Slice `backbone_fpn` / `vision_pos_enc` per camera and populate each
   camera's `state["cached_features"]` exactly as `prime_feature_cache`
   does today.
4. Run each camera's `propagate_in_video` serially, in arrival order.
5. Return per-camera results as separate responses (the HTTP layer
   needs a small change: either hold k requests open and answer them
   from the group, or an internal batch endpoint the identity service
   calls).

**Expected gain:** the encoder is ~31% of propagation; batching k=3-4
cameras cuts most of its per-camera cost, so **~15-25% per-pass**, plus
better GPU occupancy from larger kernels. Worthwhile, not
transformative — which is exactly why it is deferred behind the
cheaper levers.

**Risks/care:** mixed resolutions must not share a stack (group by
(H, W)); the admission window adds up to ~200ms latency at low load
(disable the window when only one camera is waiting); VRAM guard math
must use the GROUP's frame count.

## 4. What NOT to do (falsified shortcuts)

- Do not "just run two passes in threads" — that is the corruption
  (§1). No amount of locking short of full serialization fixed it.
- Do not tile multiple cameras into one mosaic frame for a single
  session. Masks, track ids, and SAM3's instance decisions are
  resolution- and geometry-sensitive (see `_SAM3_GROUND_W` history:
  moving ground input from 1280 to 1920 alone caused phantom splits).
  A mosaic changes every geometric prior at once.
- Do not share one EdgeTAM inference session across cameras. Track ids
  and memory features are per-scene; the model has no notion of
  "which camera is this object from".

## 5. Tier 2 — full cross-session batching (research-grade, weeks)

Batching the *stateful* 69% (memory attention + mask decode) across
cameras means giving EdgeTAM a camera dimension it does not have:
each camera has a different number of tracked objects, different
memory-bank lengths, and different mask shapes. Doing it for real
requires restructuring the vendored predictor internals — padding or
nested tensors across sessions, per-camera attention masking, and a
re-validation of mask parity against the single-camera path.

The campaign's lesson applies squarely: this is exactly the class of
vendored-model surgery where plausible-looking changes fail silently.
If it is ever attempted, it must be validated the same way the
corruption was diagnosed — synthetic multi-camera load, per-camera
drop scoring (`score_drops.py` metric: live-count decrease while YOLO
still sees more people than the tracker kept), against the known-clean
baseline of zero drops.

**Decision gate:** attempt Tier 2 only if Tier 1 + 4 workers +
stateless dispatch measurably cannot hold ~20 occupied cameras in SLA,
AND the product actually needs >20 per node (check GPU rental
economics first — a second node may simply be cheaper than weeks of
model surgery).

## 6. Adjacent lever the reader should know about

SAM3 grounding (~0.7s, every ~5 ticks per camera) does not batch across
cameras in this plan — its "multiplex" session machinery is not safe
under overlap (it dominated the corruption matrix: serializing only
SAM3 removed ~90% of drops). Its cost is instead controlled by:
occupancy-gated re-ground cadence (`NEMARA_EDGE_REGROUND`), the
1280-wide ground input cap, and the no-gc lever
(`NEMARA_EXP_SAM3_NO_GC`) which removes the gc + device-wide CUDA
cache flush SAM3 otherwise runs after every single ground.

## 7. Current validated baseline (what "better" is measured against)

- Architecture: N single-flight worker processes per node (2 on 24GB,
  3-4 on 48GB), cameras dispatched to workers; identity service per
  home on the same box.
- 4090, 2 workers, BLAS pins: 10 occupied cameras, p50 6.11s / p95
  9.28s, 0 drops in 2240 ticks.
- Speed levers already in the image behind env flags, safe per-process:
  `NEMARA_EXP_SDP_PIN=1` (removes per-attention-call global flag churn;
  showed up to ~3x tick cadence in the corruption arms) and
  `NEMARA_EXP_SAM3_NO_GC=1` (removes the per-ground stall; needs a
  memory-growth soak before fleet default).
- Next planned (not this doc): stateless workers + shared-queue
  dispatch, so any free worker serves any camera (session state — a
  few masks + counters — travels with the request).
