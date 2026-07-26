# Continuous tracking (SAM3 + EdgeTAM) — architecture, failure modes, tuning

This is the perception core that produces stable person tracks for the
re-identification brain. It replaced the per-clip track-ledger stitching
(the "ghost deadlock" era) with a **persistent per-camera session** that
holds identity across ticks. Read this before touching `sam_service/continuous.py`.

---

## 1. How it works

One `CameraSession` per camera lives in a process-global store on the pod.
Each backend tick sends a short window of frames to `POST /observe`; the
session returns `{frame_pos: {track_id: (bbox_xyxy, mask)}}` with **stable
track ids across ticks**.

Two model roles, cascaded:

- **SAM3 grounds** ("person" concept) **periodically** — every
  `NEMARA_EDGE_REGROUND` ticks and on the first tick. SAM3 multiplex needs
  ~6 frames of context to ground, so we ground over the tick's first
  frames and take frame 0's masks. **SAM3 is the authority on existence:
  if it grounds nobody, nobody is there.**
- **EdgeTAM propagates** the mask between grounds at ~36 ms/frame.

Ids stay stable two ways:

1. **Mask-carry** — a non-ground ("carry") tick seeds EdgeTAM from the
   *previous* tick's final masks under the same ids (0 lost, ~0 px drift
   when the tick windows overlap by a frame).
2. **IoU re-map** — a ground tick maps SAM3's fresh masks back onto
   existing persistent ids by IoU (`_REMAP_IOU`), so a re-ground refreshes
   geometry without renumbering anyone.

**Downscale optimization:** EdgeTAM re-encodes every frame in
`init_state`, which dominates cost, so we track at `NEMARA_EDGE_TRACK_W`
(640) and scale bboxes back to full-res on output. The face pass still
uses the full-res crop, so identity quality is untouched. This cut carry
ticks ~2800 ms → ~745 ms.

```
observe(frames):
  reground = first tick OR (tick - last_ground) >= reground_every OR last_masks empty
  seed = SAM3 ground (reground)  |  last_masks (carry)
  out  = EdgeTAM propagate(frames, seed)
  drift-guard(out)        # §2.2
  last_masks = out[last_frame] (cleared to {} if empty)   # §2.1
```

---

## 2. Failure modes and the fixes (2026-07-22)

All four were found from live soak frames; the audit notes matter because
the fixes are calibrated to measured numbers, not guesses.

### 2.1 The "green box on the empty sofa" — clear-on-empty
EdgeTAM is **sticky**: when its target leaves, it does not report "gone,"
it keeps a mask. Fix: `last_masks` is **cleared to empty** whenever a tick
produces no tracks — a ground that found nobody, or a carry that lost
everyone. A ground is authoritative on existence.

### 2.2 Drift smear (the couch phantom) — mask-coherence drift guard
On **carry** ticks, a lost/departing target makes EdgeTAM **smear** the
mask onto the nearest high-contrast background (the couch). Audited from
the leave sequence (camera 3c6570af): the smear **balloons 2.2–2.9× the
last SAM3 ground mask** and turns wide / fragments, while legit tracking
of a present moving person stays **≤ ~1.6×** and coherent.

So the signal is mask **area + shape vs the ground reference** — *not*
centroid teleport (measured centroid drift was ~0.12 for stable AND
drifting tracks alike; teleport cannot tell them apart, and the smear
balloons *in place*). `_is_drift`:

- `balloon ≥ 2.5×` (any shape) → drift, or
- `1.8× ≤ balloon < 2.5×` **and** (bbox went wide `w/h ≥ 1.15` **or**
  fragmented `largest-component/total < 0.55`) → drift.

A flagged track is dropped and a **re-ground is forced** so SAM3
re-adjudicates existence next tick — a real person is re-acquired cheaply,
a phantom is cleared. This is why false positives are safe: the worst case
is one extra ~2 s ground tick.

### 2.3 The 890-second phantom lock — lives in the BACKEND
The pod drift fed a backend bug: a phantom track keeps `num_tracks > 0` →
`people_present` → `scene_present`, and the ledger's scene-extension reset
`last_seen = now` **every tick with no cap**, holding a departed patient
for ~15 minutes. Fixed in `nemara-v2-soak` (`track_ledger.py`): the
scene-extension is **bounded** by `reid_scene_extension_max_s` (150 s)
since the last *real* body sighting. A still patient in bed is re-grounded
by SAM3 within the window; a phantom is not. §2.2 removes the drift at the
source; this is the backend backstop.

### 2.4 Re-entry blindness — motion calibration (BACKEND)
Separate from tracking: a returning patient at couch distance (~5–6 m) was
never even waking the camera. A block's mean-luma delta is an **edge-sweep
signal that decays ~44/metres**, so the old wake floor of 12 went blind
past ~3.7 m. Fixed in `nemara-v2-soak` (`motion.py`): floor 12 → 7,
adaptive gate 4× → 3×, and a **spatial-adjacency wake rule** (≥2
orthogonally-adjacent tripped blocks) that rejects grain by *shape*, which
is what makes the lower floor safe. Plus an opt-in **warm-room keepalive**
(`reid_warm_keepalive_s`, default off) for a patient who returns and
immediately settles.

---

## 3. Tuning knobs

Pod / perception (env, read in `continuous.py`):

| var | default | meaning |
|---|---|---|
| `NEMARA_EDGE_REGROUND` | 5 | ticks between SAM3 re-grounds |
| `NEMARA_EDGE_TRACK_W` | 640 | EdgeTAM track-resolution width (0 = full-res) |
| `NEMARA_EDGE_REMAP_IOU` | 0.30 | IoU floor to keep an id across a re-ground |
| `NEMARA_EDGE_DRIFT_BIG_BALLOON` | 2.5 | drift: balloon that drops a track regardless of shape |
| `NEMARA_EDGE_DRIFT_BALLOON` | 1.8 | drift: balloon that drops a track *if* wide/fragmented |
| `NEMARA_EDGE_DRIFT_WIDE_ASPECT` | 1.15 | drift: banded "went wide" bbox w/h |
| `NEMARA_EDGE_DRIFT_LCC_MIN` | 0.55 | drift: banded fragmentation floor |
| `NEMARA_EDGE_DEBUG_DIR` | "" | if set, dump per-tick mask overlays (OFF in prod) |

Backend (pydantic `Settings`, env-tunable, in `nemara-v2-soak`):

| setting | default | meaning |
|---|---|---|
| `reid_binding_retire_after_s` | 45 | base grace before a patient binding retires |
| `reid_scene_extension_max_s` | 150 | cap on scene-presence extension without a body re-sighting |
| `reid_warm_keepalive_s` | 0 (off) | keep a recently-occupied room scanning for a quick return |
| `NEMARA_MOTION_BLOCK_DELTA_TRIP` | 7 | per-block wake floor (luma) |
| `NEMARA_MOTION_BASELINE_MULT` | 3 | adaptive-baseline multiplier |
| `NEMARA_MOTION_WAKE_MIN_ADJ_PAIRS` | 1 | adjacent tripped-pairs required to wake |

---

## 4. Verifying it

- **Tick cadence & mode:** `scripts/pod.sh health` (or grep
  `continuous.tick` in `/workspace/sam_server.log`). Each line reports
  `mode=ground|carry seeded=N live=N drift=N`.
- **See the mask:** set `NEMARA_EDGE_DEBUG_DIR=/workspace/edge_debug`,
  redeploy, and per-tick overlays land there (green mask + id + bbox).
  Pull them and eyeball the person vs a smear. **Turn this OFF for
  production** — it fills disk and slows ticks.
- **Drift-guard unit tests:** `tests/test_drift_guard.py` (numpy + cv2,
  runs off-GPU), pinned to the measured 2.2–2.9× smear.

---

## 5. Deferred / open

- **Adaptive fps when the patient is confirmed** — hypothesis (higher fps
  keeps the mask coherent under motion) was **NOT confirmed** by the fps
  audit (1.5 vs ~4.5 fps showed no balloon reduction; the couch phantom is
  an *exit* smear with no target to hold, so no fps helps it). Needs a
  real high-fps walk-out clip before it's worth building.
- **Stricter / subject-scoped SAM3 grounding** — the couch phantom was
  *not* SAM3's fault (it grounded nobody correctly), so this is lower
  priority; still useful for general seed quality.
- **Backend bind-once + hold-with-audit ledger rewrite** — the continuous
  session now gives stable ids, so the ledger can eventually be simplified
  to bind-once. Deferred.
