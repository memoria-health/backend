# Identity persistence — implementation plan

> Drafted 2026-07-19 ~02:00Z during soak #2, from live failures (bedroom
> PATIENT flicker, seated/lying track death at 41% of occupied ticks,
> lunch green-box FPs, aunt/uncle unidentified, blur↔identity r=0.52,
> 320×180 rung demotions) + research survey
> (`docs/research/2026-07-19-person-locking-identity-persistence.md`).
>
> Design decisions locked with Aryaman tonight:
> - Sticky lock: absence of evidence NEVER demotes; only contradiction
>   or verified emptiness on GOOD ticks does.
> - ONE patient per home (not per camera) — the presence registry
>   becomes the single-holder arbiter. Facilities later: N holders.
> - Greedy takeover: higher-quality claim wins the lock; loser →
>   UNKNOWN (never OTHER).
> - Dim faces may VOTE, only quality faces may LOCK; harvest needs a
>   hard bar (green-box + gallery-poisoning fix).
> - Backlog: bounded queue (depth 1, max age) replaces skip — a
>   safety verdict 15s late beats a dropped action window.
> - Media classes / when-where / duty / fire rules / severity≥80
>   emergency exemption: UNCHANGED. This plan touches identity and
>   tick plumbing only.

## The algorithm, before → after (one tick, occupied room)

TODAY: every tick rebuilds tracker+labels from scratch → blurry/180p
tick ⇒ zero detections ⇒ tracks die, labels vanish, empty-room
counters advance; two tracks can both lock PATIENT; 15px faces can
lock at a 0.24 adaptive bar and then HARVEST body embeddings into the
patient pool; gallery fragments into singleton clusters (merge@0.85
unreachable); overflow ticks are dropped (tick_backlog).

AFTER: tick arrives → (W2) input-quality check: a mushy tick still
runs the scene call for safety but is barred from testifying about
identity/presence state; (W6) detector feeds low-conf boxes so a
still/lying person keeps their track; (W3) the camera's lock holds
PATIENT between good ticks and the HOME registry holds exactly one
PATIENT with greedy takeover; (W4) each real face lock banks 4-6
poses so body matching clears 0.72 solidly at range; (W7) tracks that
die anyway leave a 2-clip ghost (position+label+embedding) that a
reappearing person re-inherits; (W1) overflow clips queue ≤20s
instead of being dropped.

## Workstreams

### W0 (external, gateway repo — Aryaman/partner)
Single 360p publish layer (or min-bitrate floor) so the SFU cannot
demote to 320×180, + tripwire hardening delta 30→45 (already written
there). Ships as the one gateway update already planned. Backend needs
nothing for this.

### W1 — Bounded tick queue (replaces tick_backlog skip)
- `src/services/detection/camera_ingest.py` `_tick_loop`: on in-flight
  cap hit, enqueue the tick (depth 1; newest wins if a second arrives)
  with its capture timestamp; run it when a slot frees IF age ≤
  `scheduler_tick_queue_max_age_s` (new setting, default 20); else
  drop with a loud `tick_dropped_stale` row (crit on dashboard).
- Tests: queued tick runs after slot frees; stale one drops with the
  new skip reason; depth stays 1.

### W2 — Quality-aware tick gating (evidence admissibility)
- Compute per-tick input quality during reid frame extraction
  (`src/services/reid/inference.py` — frames already decoded):
  `frame_width`, `frames_extracted`, mean Laplacian variance over ≤3
  sampled frames. Stash on `IdentityResult` and on
  `analyses.detector_fields` (`quality_width`, `quality_sharpness`,
  `quality_frames`).
- Degraded when: width < 640, OR sharpness < floor (calibrate from
  tonight: LOST median 2436 vs TRACKED 4745 → start ~3200,
  setting `reid_quality_sharpness_floor`), OR frames < 6.
- Consumers of degraded=true:
  - identity lock (W3): tick cannot confirm, contradict, or empty-out.
  - state machine: empty-room close counter does NOT advance
    (`camera_ingest` apply_tick_update guard) — a blur tick can't
    close an occupied room.
  - dashboard: mark the tick cell (distinct hatch) so quality loss is
    visible.
- Scene/LLM calls still run (safety unaffected — the LLM handles blur
  far better than YOLO, tonight's llm_only=28/34 proves it).
- Tests: degraded tick leaves lock+state untouched; good tick flows.

### W3 — Identity lock (camera) + single-holder registry (home)
- NEW `src/services/reid/identity_lock.py` — per-camera lock:
  - ENTER: face lock passing W5 quality floor, OR body-gallery match
    ≥ `reid_body_match_patient_threshold` (0.72, unchanged).
  - HOLD: while exactly one person continuously present on GOOD
    ticks; ≤ `identity_lock_empty_grace_s` (45) of no-person GOOD
    ticks tolerated; degraded ticks frozen (W2).
  - RELEASE: verified-OTHER quality face on the held person; a second
    person appears (fall back to per-tick verdicts, lock parked, not
    poisoned); >45s genuinely empty on good ticks; home-registry
    takeover by a better claim elsewhere.
- `src/services/detection/patient_presence.py` → grows into the
  home single-holder: `claim(camera_id, confidence, quality_tier,
  ts)`; exactly one holder per patient/home; greedy takeover
  (face-quality claim > body claim > held/stale claim; then higher
  confidence); demoted camera's lock releases to UNKNOWN.
  Cross-camera detector muting and the severity≥80 emergency
  exemption keep their exact current semantics on top of it.
- Wire in `camera_ingest` after `label_clip`: lock adjusts
  `IdentityResult` (patient_in_clip, relabeled track, held
  confidence) + stashes `id_lock_state/age/source` fields.
- Tests: enter/hold-through-blur/hold-through-45s-gap; second-person
  parks lock; takeover moves the home lock camera A→B; emergency
  exemption regression test unchanged.

### W4 — Gallery v2 (pool + best-of-pool + multi-pose harvest)
- `src/services/reid/cross_clip_body_gallery.py`: replace cluster
  list with a flat per-patient pool: cap
  `reid_body_max_embeddings` (64), FIFO, dedup 0.95 (keep), TTL 4h
  (keep — outfit-bound). MATCH = mean of top-3 cosines against the
  pool (robust max; DeepStream-style many-vector gallery). Thresholds
  0.72 / 0.80 unchanged. OTHER embeddings pool per-label same way.
- Multi-pose harvest (`inference.py` harvest loop): from each
  face-locked track, harvest up to 4 embeddings spread across its
  bbox_series (first/best/middle/last frames), each passing the W5
  harvest bar. One living-room walk-by ⇒ tonight's outfit from
  several angles.
- Tests: pool caps/dedup/TTL; top-3 matching beats singleton on the
  synthetic cross-pose fixture; harvest count/diversity; existing
  gallery tests updated.

### W5 — Face-lock quality floor + in-clip single PATIENT
- `src/services/reid/body_pipeline.py`:
  - LOCK requires `face_px ≥ reid_face_lock_min_px` (new; calibrate
    from tonight's labeled `out/id_frames` set — start 36) AND
    `det_score ≥ reid_face_lock_min_det` (0.6). Below floors: the
    match still VOTES (score recorded, prompt addendum unaffected)
    but cannot flip the label. Kills 15-30px green-box locks.
  - At most one PATIENT track per clip: on second qualifying lock,
    higher score holds, other stays UNKNOWN.
- HARVEST bar (used by W4): face-locked at quality floor AND match
  score ≥ 0.45. Nothing below feeds the pool — ends silent gallery
  poisoning by marginal locks.
- Tests: sub-floor face cannot lock/harvest; conflict resolution
  keeps best; existing green-box repro fixture (lunch params) now
  yields UNKNOWN.

### W6 — Low-confidence track continuation
- `src/services/reid/body_detector.py`: detect at floor 0.10
  (`reid_yolo_detect_floor`), return all boxes with confidences.
- `src/services/reid/tracker.py`: configure BoT-SORT two-stage
  association explicitly — `track_high_thresh` ≈ current 0.30/0.40
  (birth + first-pass), low-threshold second pass consumes the
  0.10-0.30 boxes for CONTINUATION only, `new_track_thresh` keeps
  birth at today's bar (no ghost tracks from shadows).
- Tests: synthetic still-person sequence (conf decaying 0.8→0.15)
  keeps one continuous track; a 0.2-conf-only sequence with no
  high-conf birth never creates a track.

### W7 — Tracklet carry (nobody teleports)
- Per-camera carry buffer (in `identity_lock.py` or sibling): on clip
  end, store each substantial track's {label, last bbox, best OSNet
  embedding, ts}; TTL `tracklet_carry_ttl_s` (25 ≈ 2 clips).
- On next clip, before gallery pre-match: a new track whose first box
  center falls within a radius of a carried tracklet's last box and
  whose embedding does not contradict (cosine ≥ ~0.5) inherits the
  carried label (marked `carried=True`, same override rules as
  pre-labels — in-clip face evidence wins).
- Tests: label survives a 1-2 clip track death at same position;
  contradiction blocks inheritance; TTL expiry.

## Order & rollout

1. **Batch A (tonight-capable): W1 + W2 + W6** — small, independent,
   attack the live bleeding (dropped windows, blur poisoning state,
   still-person death). One backend restart.
2. **Batch B: W3 + W5** — the lock + quality floors (single-holder
   registry included). One restart.
3. **Batch C: W4** then **W7**. One restart each (or fold into B's if
   same session).
- Each batch: unit tests green → restart container → watch 10 min of
  soak rows + lanes before next batch.
- All on the current worktree branch; commits local-only per standing
  rule; no schema/app changes required (all backend + settings).

## Cost note (explicit)
Restoring tracking on the ~41% blind occupied ticks returns the crop
call on those ticks: occupied-room spend rises from ~1.6 to ~2.0 lite
calls/tick — the accuracy mode returning, bounded by existing
duty/thinning/k-of-N. The soak prices it exactly; no other cost paths
change. Blur ticks still run 1 scene call (unchanged).

## Calibration TBDs (from tonight's data, before Batch B)
- `reid_quality_sharpness_floor`: from the 240-frame labeled set
  (start 3200; verify false-degrade rate on tracked frames).
- `reid_face_lock_min_px` / `min_det`: from out/id_frames labeled
  you-vs-relatives set (start 36px / 0.6; must keep tonight's real
  bedroom locks while excluding lunch green-boxes — verify both).
