# Continuous tracking (SAM3 + EdgeTAM) — architecture, failure modes, tuning

This is the perception core that produces stable person tracks for the
identity layer (`gpu/identity/`, consumed by the backend via
`src/services/identity/`). It replaced the per-clip track-ledger stitching
(the "ghost deadlock" era) with a **persistent per-camera session** that
holds identity across ticks. Read this before touching
`gpu/perception/continuous.py` (served by `gpu/perception/server.py` —
`POST /observe` + `POST /faces`).

Two runtime toggles added since this doc was first written:
`NEMARA_TRACKER` (`edgetam` default = SAM3-ground + EdgeTAM-carry as
described below; `sam3` = SAM3's own tracker over the whole clip, with
`NEMARA_SAM3_STRIDE` frame striding) and `NEMARA_GROUNDER` (`sam3`
production default — chosen 2026-07-26 after `yoloseg` fragmented a
recumbent person into two instances).

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
  last_masks = out[last_frame] (cleared to {} if empty)   # §2.1
```

(The drift-guard step that used to sit after propagate was removed —
see §2.2.)

---

## 2. Failure modes and the fixes (2026-07-22)

All four were found from live soak frames; the audit notes matter because
the fixes are calibrated to measured numbers, not guesses.

### 2.1 The "green box on the empty sofa" — clear-on-empty
EdgeTAM is **sticky**: when its target leaves, it does not report "gone,"
it keeps a mask. Fix: `last_masks` is **cleared to empty** whenever a tick
produces no tracks — a ground that found nobody, or a carry that lost
everyone. A ground is authoritative on existence.

### 2.2 Drift smear (the couch phantom) — drift guard **[SUPERSEDED — removed 2026-07-24]**
On **carry** ticks, a lost/departing target makes EdgeTAM **smear** the
mask onto the nearest high-contrast background (the couch). Audited from
the leave sequence (camera 3c6570af): the smear **balloons 2.2–2.9× the
last SAM3 ground mask** and turns wide / fragments, while legit tracking
of a present moving person stays **≤ ~1.6×** and coherent.

A mask-coherence drift guard (`_is_drift`: balloon ≥ 2.5× any shape, or
1.8–2.5× when wide/fragmented → drop the track + force a re-ground) was
built on those numbers — and then **PURGED on 2026-07-24**: in live use
it mistook a person walking TOWARD the camera (whose mask legitimately
balloons) for a smear and killed live tracks. The `NEMARA_EDGE_DRIFT_*`
knobs and `tests/test_drift_guard.py` are gone with it.

The current defense is simpler: **re-grounds are the sole existence
authority**. A sticky-mask phantom dies at the next SAM3 ground
(≤ `NEMARA_EDGE_REGROUND` ticks) when the grounder finds nobody there,
and §2.1's clear-on-empty prevents it from re-seeding. The backend
scene-extension cap (§2.3) is the backstop for the window in between.

### 2.3 The 890-second phantom lock — lives in the BACKEND
The pod drift fed a backend bug: a phantom track keeps `num_tracks > 0` →
`people_present` → `scene_present`, and the ledger's scene-extension reset
`last_seen = now` **every tick with no cap**, holding a departed patient
for ~15 minutes. Fixed and now on this branch
(`src/services/identity/track_ledger.py` — since rewritten as the
bind-once-hold ledger keyed directly by the pod's stable `track_id`,
labels PATIENT/UNKNOWN only): the scene-extension is **bounded** by
`reid_scene_extension_max_s` (150 s) since the last *real* body sighting.
A still patient in bed is re-grounded by SAM3 within the window; a
phantom is not. Regrounds remove the drift at the source (§2.2); this is
the backend backstop.

### 2.4 Re-entry blindness — motion calibration (BACKEND) **[NOT MERGED — staged in the `nemara-v2-soak` worktree only]**
Separate from tracking: a returning patient at couch distance (~5–6 m) was
never even waking the camera. A block's mean-luma delta is an **edge-sweep
signal that decays ~44/metres**, so the old wake floor of 12 went blind
past ~3.7 m. The staged fix (in `nemara-v2-soak`, `motion.py`): floor
12 → 7, adaptive gate 4× → 3×, a **spatial-adjacency wake rule** (≥2
orthogonally-adjacent tripped blocks) that rejects grain by *shape*, and
an opt-in **warm-room keepalive** (`reid_warm_keepalive_s`, default off).
**On this branch (`feat/gateway-camera-ownership`) none of that has
landed**: `src/services/detection/motion.py` still has
`BLOCK_DELTA_TRIP = 12.0`, `BASELINE_MULT = 4.0`, no adjacency rule, and
there is no `reid_warm_keepalive_s` setting.

---

## 3. Tuning knobs

Pod / perception (env, read in `gpu/perception/continuous.py`):

| var | default | meaning |
|---|---|---|
| `NEMARA_EDGE_REGROUND` | 5 | ticks between SAM3 re-grounds |
| `NEMARA_EDGE_TRACK_W` | 640 | EdgeTAM track-resolution width (0 = full-res) |
| `NEMARA_EDGE_REMAP_IOU` | 0.30 | IoU floor to keep an id across a re-ground |
| `NEMARA_EDGE_CKPT` / `_CFG` / `_ROOT` | /workspace/edgetam/… | EdgeTAM checkpoint / config / repo root |
| `NEMARA_TRACKER` | edgetam | tracker backend: `edgetam` (ground+carry, this doc) or `sam3` (SAM3's own tracker per clip) |
| `NEMARA_SAM3_STRIDE` | 1 | sam3 mode only: track every Nth clip frame (~halves cost at 2) |
| `NEMARA_GROUNDER` | sam3 | grounder: `sam3` (production) or `yoloseg` (rejected 2026-07-26 — fragmented a recumbent person) |
| `NEMARA_EDGE_DEBUG_DIR` | "" | if set, dump per-tick mask overlays (OFF in prod) |

(The `NEMARA_EDGE_DRIFT_*` knobs are gone — the drift guard was removed, §2.2.)

Backend (pydantic `Settings`, env-tunable via `NEMARA_*`, `src/settings.py`):

| setting | default | meaning |
|---|---|---|
| `reid_binding_retire_after_s` | 15 | base grace before a patient binding retires (was 45 in the soak-era doc) |
| `reid_scene_extension_max_s` | 150 | cap on scene-presence extension without a body re-sighting |

The §2.4 motion knobs (`reid_warm_keepalive_s`,
`NEMARA_MOTION_BLOCK_DELTA_TRIP` 7, `BASELINE_MULT` 3, adjacency pairs)
exist only in the unmerged `nemara-v2-soak` staging — not on this branch.

---

## 4. Verifying it

- **Tick cadence & mode:** `scripts/pod/pod.sh health` (or grep
  `continuous.tick` in `/workspace/sam_server.log`). Each line reports
  `mode=ground|carry seeded=N live=N drift=N`.
- **See the mask:** set `NEMARA_EDGE_DEBUG_DIR=/workspace/edge_debug`,
  redeploy, and per-tick overlays land there (green mask + id + bbox).
  Pull them and eyeball the person vs a smear. **Turn this OFF for
  production** — it fills disk and slows ticks.
- **Unit tests:** the drift-guard tests were deleted with the guard
  (§2.2). The pure identity-layer modules that consume these tracks are
  tested in `test/unit/test_identity_*.py` (off-GPU).

---

## 5. Deferred / open

- **Adaptive fps when the patient is confirmed** — hypothesis (higher fps
  keeps the mask coherent under motion) was **NOT confirmed** by the fps
  audit (1.5 vs ~4.5 fps showed no balloon reduction; the couch phantom is
  an *exit* smear with no target to hold, so no fps helps it). Needs a
  real high-fps walk-out clip before it's worth building.
- **Stricter / subject-scoped SAM3 grounding** — the couch phantom was
  *not* SAM3's fault (it grounded nobody correctly), so this is lower
  priority; still useful for general seed quality. (A YOLO26-seg grounder
  was tried and rejected 2026-07-26 — see `NEMARA_GROUNDER` in §3.)
- ~~**Backend bind-once ledger rewrite**~~ — **DONE**:
  `src/services/identity/track_ledger.py` is now the bind-once-hold
  ledger keyed by the pod's stable `track_id` (PATIENT/UNKNOWN only,
  no geometric stitcher).
