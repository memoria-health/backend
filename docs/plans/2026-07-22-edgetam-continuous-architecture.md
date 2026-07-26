# EdgeTAM-continuous identity — architecture + build plan

> **North star (the design axiom):** once the patient is found with sufficient
> confidence, the lock is **sacrosanct** — nothing interferes with it until the
> patient **actually leaves view**. Every design choice below is judged against
> this one sentence.

## Why we pivoted

The track-ledger ran identity on **discrete ~10s SAM3 clips**. SAM3 renumbers
the subject across each tick boundary; the geometric stitcher can't always
bridge it, so the binding strands on a bodyless "ghost" that then deadlocks the
live track (see `../ghost-deadlock-known-fix.md`). It also ran SAM3 at
~250ms/frame → dropped ticks + cross-camera desync. Both failures come from the
**same root**: identity re-decided every tick because tracking isn't continuous
across ticks. Continuous tracking removes the root.

## The model

**Pod = continuous perception (one persistent session per camera).** SAM3
grounds "person" *periodically* (text → masks); **EdgeTAM propagates between
grounds** at ~36ms/frame. Output: tracks with **stable ids across ticks** +
clean per-person masks. The tick boundary stops existing as far as identity is
concerned.

**Backend = bind-once, hold-with-audit.** Each stable track gets an identity
bound **once** (face or pool), then **held** across ticks — but *not blindly*.
The binding is **re-verified intermittently** (face/pool) at a cadence that
**relaxes toward zero only when the hold is >95% confident** (a strong recent
face lock) and **tightens** when it rests on weaker evidence (pool-only, or a
stale lock). A contradicting re-verification **re-decides**. This is the safety
valve against a *silent tracker swap* — a stable id that quietly starts
pointing at the wrong person. **The lock is sacrosanct against noise, never
against evidence.** The binding releases when the pod drops the track (patient
left view). The **governor stays** — it sets both the SAM3 re-ground cadence
and this audit cadence.

**Continuity moves into the pod, but we don't blindly trust it.** The pod's
persistent session spans ticks and hands us stable ids, so the *heavy* stitcher
(velocity extrapolation, ambiguity resolution, fragment-merge) is retired. But
because we still run on clips/ticks, a **lightweight boundary reconciliation**
(the cheap one-frame id sanity check) stays as a permanent safety net for pod
hiccups — and stays *especially* until the pod is proven to deliver stable ids
reliably in the soak.

This is the north star structurally: stable track = *found patient persists*;
hold-with-audit = *sacrosanct against noise, correctable by evidence*; track
end = *left view*.

## Validated (2026-07-22 pod probes)

- EdgeTAM: **36ms/frame** (~7× SAM3), **0 lost frames / 30** on 2-person + the
  lying-down case that normally breaks trackers.
- SAM3 + EdgeTAM import in **one env** → one process hosts the cascade.
- **Open risk:** seed quality on cluttered / same-color scenes (green couch
  merged person+couch mask). It's a SAM3 *grounding* issue, not EdgeTAM. Fix =
  subject-scoped grounding + a sane re-ground cadence.

## Pod build (`sam_service`)

1. **Session store:** per-camera `{edgetam inference_state, obj_id→track_id map,
   last_ground_tick}`. Create on first frame from a camera; evict on idle
   (reuse the existing VRAM watchdog).
2. **`/observe(camera_id, frames)`:**
   - **Re-ground when due** (first tick / every N ticks / EdgeTAM confidence
     drop / new-person signal): SAM3 grounds "person"; map new masks to
     existing `obj_id`s by IoU so **ids stay stable**; genuinely-new persons
     get new ids (`add_new_mask`).
   - **Propagate** the tick's frames through EdgeTAM from the persistent state →
     per-frame masks per **stable** id.
   - **Faces** on mask-clean crops (unchanged: SCRFD + ArcFace).
   - Return observe-v2 with **stable track ids**.
3. **Re-ground cadence:** governed — default every ~5–10 ticks; immediate on a
   lost/low-confidence track or rising occupancy.
4. **First spike in the build:** EdgeTAM/SAM2 streaming — does `init_state`
   accept incremental frame append across `/observe` calls, or do we re-init on
   a sliding window carrying the object memory? `edge_propagate.py` used a fixed
   `init_state(framedir)`; the streaming variant is the one unknown left.

## Backend build (bind-once-hold)

Replace `track_ledger.TrackLedger` with a thin **per-stable-track binding
registry**:
- `track_id → {label, provenance, confidence, bound_at}`.
- Bind PATIENT once when a track's face locks (≥0.45) or pool matches (≥0.72),
  respecting single-holder (`PatientPresence`).
- **Hold** across ticks — the pod keeps the id stable, so nothing re-decides.
- **Retire** when the pod drops the track (left view). Re-entry = new id =
  re-bind.
- **Audit** (kept, simplified): sustained verified-OTHER on a PATIENT track →
  unbind to UNKNOWN.

**KEEP (identity signals + safety valves — these are NOT cruft):**
- **Face rec** (SCRFD + ArcFace) — primary identity signal.
- **OSNet / pool** (`cross_clip_body_gallery`) — now *more* important: the
  **re-entry** signal (match a fresh track to a recent patient by outfit) and a
  binding signal when the face isn't visible.
- **Harvester** — keeps closing the wall-cam domain gap on the face gallery.
- **Intermittent audit** — the confidence-scaled face/pool re-verification of a
  held track (the hold-with-audit safety valve; relaxes only at >95%).
- **Lightweight boundary reconciliation** — the cheap one-frame id sanity check
  across ticks (pod-hiccup safety net; the *thin* survivor of the stitcher).
- **`PatientPresence`** — single-holder + cross-camera suppression.
- **Governor** — `compute_reid_mode`, heartbeats, `all_bound`,
  `wants_face_evidence` (now confidence-scaled).

## Rip-out list — HEAVY cross-tick plumbing only

> Signals (face / pool / harvester) and a **thin one-frame reconciliation** stay
> (see KEEP above). Only the machinery that *reconstructed* continuity from
> nothing is ripped — and only once the pod is proven to deliver stable ids in
> the soak. We do not rip the safety net before we trust the replacement.

| Component | File | Reason it dies |
|---|---|---|
| **Heavy** stitcher: velocity extrapolation + ambiguity resolution (`_stitch` guts, `_refresh_geometry` vel, `ambiguous_stitch`) | `track_ledger.py` | pod owns continuity; a *thin* one-frame id reconciliation survives as the safety net |
| `_merge_fragments`, `_merge_priority`, `_absorb` | `track_ledger.py` | EdgeTAM masks are one-per-object + stable — no per-tick fragment churn |
| ghost handling: retire-grace + `_scene_present` extension + conflict deadlock | `track_ledger.py` | no ghosts — a track ends when the pod drops it |
| two-way accumulators (`mismatch_total`, `smallface_total`, cross-tick smallface) | `track_ledger.py` + `body_pipeline` inputs | cross-tick accumulation was a discrete-clip crutch |
| stitcher settings (`reid_binding_iou_floor`, `center_*_px`, `ambig_margin`, `max_stitch_gap_s`, `min_birth_frames`) | `settings.py` | no stitcher |

## Build order

1. **Pod persistent session** — spike the streaming API, build it, validate
   stable ids across ticks on live footage.
2. **Backend bind-once-hold** — replace the ledger; execute the rip-out.
3. **Wire + soak** — point the soak at `nemara-v2-edgetam`, re-run the
   walk-to-sofa (M3): the lock **must hold across the renumber** — the exact
   failure that motivated this.
4. **Seed quality** — subject-scoped grounding so the couch never merges in.

## Definition of done

The M3 test: walk in, face-lock, walk to the sofa — **identity holds every
tick, no ghost, no UNKNOWN**, until you actually leave the room. The north star,
by construction rather than by luck.
