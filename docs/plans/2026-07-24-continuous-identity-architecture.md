# Continuous tracking + parallel identity — architecture & migration plan

> Status (2026-07-27): implemented (P0–P3), with one deliberate deviation — identity did NOT land as the co-located backend module §2b recommends: Aryaman locked a separate identity service on the GPU host (2026-07-26), now `gpu/identity/service.py` (port 8012, supervised via `gpu/deploy/identity_supervisor.sh`), feeding the backend through the versioned `/state` contract (`gpu/identity/contract.py`) consumed by `src/services/identity/identity_client.py` — the sidecar is the only identity provider (dfb9b4e). Key commits: c1ee735 (continuous identity service), 77d546c (backend sidecar provider), dfb9b4e (legacy brain out). P4's SSE/push dashboard was not built (dashboards still poll); P5 edge endgame remains future. File refs herein to `src/services/reid/` and `sam_service/` are stale — see `src/services/identity/`, `gpu/identity/`, `gpu/perception/`.

2026-07-24. Synthesized from a full-pipeline scour (5 mapped subsystems, file:line
anchors verified against this worktree). Companion: the e2e flow sketch shared in
session; `docs/architecture/continuous-tracking.md` (pod engine);
`docs/research/2026-07-23-tracker-eval-findings.md`.

**Goal:** near-real-time, high-accuracy patient tracking+identity, FP-averse,
scaling with activity (not camera count), feeding the detection pipeline through
one clean seam. EdgeTAM continuous spine (validated live 2026-07-23: ids held
through crossing/swap/partial-occlusion) + sparse greedy identity + patient
state as the single source of truth.

---

## 1. Why the current pipeline is slow (measured, not guessed)

The 10–40s dashboard lag decomposes into four stacked causes:

1. **Tick cadence + blind gap.** `.env` runs tick=40s, window=11s → an 11s slice
   is analyzed, then **29s are never seen at all** (`camera_ingest.py:426`,
   `_tick_loop`). At defaults (tick=10/window=11) coverage is contiguous; the
   40s value was the eval config, not the design.
2. **End-of-tick DB row.** The dashboard can only show a tick after
   `add_analysis` + commit (`camera_ingest.py:1761-1782`), which happens AFTER
   the full reid + VLM tail (~5–20s inside the tick).
3. **Batch pod round-trip.** One blocking POST of ~22–66 frames per tick
   (`sam_client.py:97`); the pod is single-flight (`server.py:73`), so cameras
   queue. Pod cost is ~linear in frame count (init_state re-encode dominates,
   `continuous.py:47-51`).
4. **Poll cadence.** Dashboard polls /data every 10s, /trace every 5s
   (`soak_dashboard.py:741,776`).

None of these are inherent. The ingest path itself is real-time (frames land in
the buffer within ~1s; `room_worker.py:961-1034`).

## 2. Target architecture (two loops + one seam)

```
LiveKit frames → FrameBuffer (existing, real-time)
   │  occupied-camera gate (presence-cheap; empty rooms = idle)
   ▼
LOOP A — SPINE (continuous, per occupied camera)
   contiguous small batches (6-12 frames, ≥1-frame overlap between batches,
   pre-downscaled to track width) → pod /observe (persistent CameraSession,
   EdgeTAM carry + periodic SAM3 ground) → stable {track_id, box[, mask]}
   → LIVE TRACK STATE (in-memory, per camera)  → dashboard (push)
   ▼
LOOP B — IDENTITY (sparse, event-driven, off the hot path)
   triggers: new unlocked track · patient lost (all-camera burst) · slow
   re-verify heartbeat while locked
   evidence: ArcFace vs DB gallery + OSNet vs body pool, accumulated per track,
   reliability-weighted (face weight = smooth calibrated fn of face px)
   decision: greedy SINGLE-SEAT lock (floor + margin over runner-up, else
   UNKNOWN) → PATIENT STATE {camera, bound_track, box/mask, confidence}
   ▼
DETECTION reads PATIENT STATE (no inline identity)
   patient detectors → patient crop on winning camera; auto-off when no patient
   scene detectors → full frame; presence/location ← patient state
```

### 2a. Spine specifics (from the pod map)
- The pod already provides everything Loop A needs: persistent per-camera
  session with stable ids across calls (`continuous.py:36,478-484`), mask-carry
  + IoU re-map (`:421`, `:258-269`), drift guard (`:155-175`).
- **Batching rules** (why the live view lagged): cost ≈ N × (encode + 36ms) at
  640px; ship 6–12 frames, overlapping by ≥1 frame (carry seed requires it —
  probes use overlap 2–4), pre-downscaled to `NEMARA_EDGE_TRACK_W` when the
  batch is tracking-only. `face_cap=0` and `detect_screens=0` on pure-tracking
  calls kills the +1.5-2.5s countersign and +0.5s screen spikes; screens
  refresh every ~120s as today.
- **Masks on the wire (small pod change):** masks exist at track-res at
  serialization time (`server.py:273`) and are simply dropped; add an RLE field
  per row. Enables mask crops + the live mask overlay.
- **Multi-camera fairness:** single-flight A40 serializes cameras. Scheduler
  weights pod slots by patient-presence (patient camera streams; other occupied
  rooms get periodic glances; empty rooms none). GPU-per-N-cameras scales it;
  the edge gateway (endgame) removes contention entirely.

### 2b. Identity specifics (from the brain map)
- **Replace the categorical `_TIER` face-vs-pool rule** (`track_ledger.py:66-68`,
  applied `:320-332`) with one reliability-weighted accumulated score per track.
  Precedent for pixel-size weighting already exists:
  `effective_match_threshold` scales the face bar by face pixel area
  (`embeddings.py:38-68`) — generalize that curve into a weight, calibrated
  empirically (measure ArcFace accuracy vs face px on our own crops).
- **Lock rule:** highest accumulated score, above floor, AND margin over
  runner-up; two plausible patients → lock neither (wrong-lock is the failure
  mode that scares us; UNKNOWN is recoverable). Evidence accumulates over the
  track's life — a real patient racks up matches, a guest doesn't.
- **Adaptive resampling seams already exist:** fps by mode at
  `inference.py:451-465`; face cadence at `wants_face_evidence`
  (`track_ledger.py:143-156`). Add the "just lost → all-camera burst" state
  (ledger has `left_view`/takeover events to key from).
- **Harvest (anti-poison):** keep the no-pool→pool-self-feed invariant
  (`track_ledger.py:61`). Tighten: require face-locked THIS clip (drop or
  face-px-gate the 60s afterglow hint at `inference.py:810-841`, which today
  bypasses the 36px gate), raise `reid_harvest_min_confidence` from 0.40,
  keep burst-cap/dedup/golden-FIFO/TTL (`cross_clip_body_gallery.py:256-304`).
- **Galleries:** face = persistent DB (`patient_face_embeddings`, load via
  `face_embedding_repo.list_ready_for_patient_unscoped`); body pool = in-memory
  per-process — if identity becomes a separate process, the pool needs
  persistence or stays co-located. **Recommend co-located module first** (same
  process as ingest), extracted to a service only when the edge move happens.

### 2c. The detection seam (from the seam map — the keystone)
- Everything downstream reads one object: the `IdentityResult` produced at
  `camera_ingest.py:1080-1122` and rewritten by `TrackLedger.apply` (`:1122`).
  The scheduler (`scheduler.py:117-134,257-271`), crops (`executor.py:241-260`),
  event gate (`camera_ingest.py:1326-1395`), presence (`:1669-1692` +
  `patient_presence.py`), and the analyses row (`:1602-1698`) consume only it.
- **Integration = implement the same `apply(identity, tick_mono) → IdentityResult`
  surface** from the new patient state at `camera_ingest.py:1122`. Scheduler,
  packer, executor, promotion: untouched. `confirmed_elsewhere` is supplied
  directly by patient state (replacing the registry election).
- What gets deleted when this lands: inline `label_clip` heavy path, TrackLedger
  bind/hold/retire internals, PatientPresence election — replaced by thin
  adapters over patient state.

### 2d. Dashboard (from the dashboard map)
- The near-real-time pattern already exists in embryo: the trace tap emits
  mid-pipeline (`trace.py:86-94`) — promote it: Loop A publishes track-state
  deltas to an SSE/WebSocket channel; the dashboard subscribes instead of
  polling end-of-tick rows. The analyses row remains the historical record.

## 3. Latency budget (targets)

| Path | Today | Target (remote pod) | Endgame (edge) |
|---|---|---|---|
| frame → live track state | 10–40s | **≤1.5s** (batch fill ~0.5-1s + pod ~0.5s) | ≤150ms |
| track state → dashboard | +10s poll | **≤0.5s** (push) | same |
| new track → identity verdict | n/a (inline) | ≤3s (event-driven) | ≤1s |
| patient lost → re-lock (visible) | ~40-120s | ≤10s (burst mode) | ≤5s |
| detection tick (VLM) | 40s eval / 10s design | 10s, reads state (no added identity latency) | same |

## 4. Migration phases (each shippable, each with rollback)

- **P0 — config restore + quick wins (hours).** Revert eval `.env` (tick 40→10;
  window 11 stays → contiguous). Small-batch + downscale in the viewer (done in
  spirit; finish `--target-fps`/batch tuning). Keep lean-eval rig available
  behind its flag. *Rollback: env only.*
- **P1 — spine service (1-2 days).** A `ContinuousTracker` per occupied camera
  (module beside ingest): drains FrameBuffer in contiguous overlapped batches →
  pod → maintains LiveTrackState. Publishes SSE for the dashboard + the viewer.
  Pod: add RLE masks + a tracking-only fast path (`face_cap=0` honored — already
  is). Presence gate: only occupied cameras run. *Rollback: flag
  `NEMARA_SPINE_ENABLED=false` → today's batch path.*
- **P2 — identity module (2-3 days).** Greedy reliability-weighted lock over
  LiveTrackState (calibrate the face-px weight curve from logged crops first);
  adaptive triggers (new-track / lost-burst / heartbeat); tightened harvest.
  Runs beside the spine, decisions logged + traced before they gate anything.
  *Rollback: shadow mode (log-only) vs enforce flag.*
- **P3 — detection integration (1-2 days).** PatientState → `IdentityResult`
  adapter at `camera_ingest.py:1122`; scheduler/executor untouched; delete the
  inline label_clip heavy path + ledger internals once parity is verified
  (side-by-side on the soak dashboard). *Rollback: adapter flag flips back to
  inline path.*
- **P4 — dashboard live view (1 day).** SSE panel: live boxes + patient state +
  identity events; keep /data for history. Retire the 10s-poll dependency for
  "what's happening now".
- **P5 — edge endgame (later, sized separately).** Tracker (+ identity) on the
  home gateway; cloud keeps gallery, VLM, fusion. Kills wire latency, pod
  contention, and the per-home GPU cost curve; aligns with DPDP (frames stay
  home).

## 5. Validation per phase
- P1: id-stability scenario ladder (the S0-S8 protocol) on the live view; spine
  e2e ≤1.5s p95; zero blind gaps (frame-index continuity assert in the driver).
- P2: shadow-mode confusion matrix vs ground truth (you label tracks in the
  viewer); wrong-lock rate target 0 on guest scenarios; re-lock ≤10s p90.
- P3: side-by-side analyses parity (identity fields old vs new) for a soak day;
  detector routing unchanged on scene detectors; patient-crop detectors receive
  crops only when state confident.
- Contamination guards stay: worker-restart reset markers; screen veto on.

## 6. Open decisions (need Aryaman)
1. P2 identity co-located module (recommended now) vs separate process from day
   one (needed only for edge; adds body-pool persistence work).
2. Masks on the wire: RLE now (small pod deploy) or boxes-only until P4.
3. Re-verify heartbeat while locked: interval + what evidence (face-only vs
   combined) — cost vs drift-catch tradeoff.
4. The calibrated face-px weight curve: collect from the existing trace/crops
   (fast, rough) vs a short dedicated capture session (cleaner).
5. P0 timing: restore tick=10 now or after the current eval wraps.
