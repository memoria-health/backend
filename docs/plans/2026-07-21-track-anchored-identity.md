# Track-anchored identity (binding ledger) — implementation plan

> Status (2026-07-27): implemented, then superseded by the EdgeTAM-continuous pivot (2026-07-22 plan). Built and soaked (checkpoint f2c5115); the geometric stitcher stranded bindings on ghost tracks and deadlocked (`docs/runbooks/ghost-deadlock-known-fix.md`, 837da2b — fix deliberately not applied because the continuous pivot dissolves the bug class). The `TrackLedger` name and camera_ingest seam survive but the module was rewritten as the bind-once-hold binding brain keyed by pod stable ids (`src/services/identity/track_ledger.py`); the stitcher, fragment merge, mismatch accumulators, and verified-OTHER audit specified here were all deleted.

**Date:** 2026-07-21 · **Status:** approved by Aryaman, building
**Goal:** "Confirm the patient once and stay locked no matter where they
go or what they do" — for ALL identities (PATIENT / OTHER / UNKNOWN
symmetric), while simplifying the per-tick guardrail stack.

## Diagnosis (why)

Every load-bearing knob of the current brain (carry cosine floor, pool
bars, pool-margin, strike counts, release hysteresis, park-on-crowd)
compensates for the same root defect: identity is re-litigated from
scratch every ~10s clip because SAM sessions are per-clip and track ids
reset at clip boundaries. The park-on-crowd rule makes the system
identity-blind in exactly the product's core environment (dinner,
caregivers). Measured on 2026-07-21 (probe_continuity.py):

- sam3 instance seeds (add_sam2_new_points, caller obj_ids) round-trip
  ids but DO NOT propagate under VG propagation; text+seed mixing
  corrupts the session → SAM-side seeded handoff rejected.
- Concept tracking within-clip is rock solid (0.93 stable).
- Display-veto prompt "television, laptop, or phone screen" finds
  screens reliably (temporal union = complete); phantom screen-person
  containment 1.0 vs patient max 0.13 → binary veto.

## Architecture

Pixels (SAM, within-clip) → **geometry (NEW stitcher, clip-to-clip)** →
appearance (OSNet pool: births/gaps/cross-camera) → face (truth:
confirm + audit). Each layer covers the failure modes of the one above.

New component: `src/services/identity/track_ledger.py` (path at the time: `src/services/reid/`) — per-camera
`TrackLedger` replacing `IdentityLock` at the exact same seam
(`apply(identity, tick_mono)` + `stash_fields()` in camera_ingest),
selected by `settings.reid_binding_mode: "legacy" | "track"` (env
`NEMARA_REID_BINDING_MODE`, default legacy until validated; rollback =
env flip + recreate).

### TrackLedger semantics

- **Stitch:** match each clip track's first bbox against live ledger
  entries' last bbox (velocity-extrapolated, dt-aware IoU + center
  gate; greedy one-to-one; ambiguity → no inherit, conservative).
  Windows overlap (tick=10s, window=11s) so boundary dt ≈ one sample
  interval.
- **Birth:** unmatched track → new persistent id; binding from this
  clip's evidence: face lock → PATIENT (single-patient rules), verified
  OTHER → OTHER, pool pre-label → per pre-label, else UNKNOWN.
- **Persist:** binding follows the persistent id. UNKNOWN-labeled clips
  (back turned, no face) change NOTHING for a bound track.
- **Audit:** on a PATIENT-bound track, a quality verified-OTHER clip
  (non-degraded) increments a contra streak; 2 consecutive → unbind to
  UNKNOWN (swap suspicion — loud log; NEVER direct to OTHER). Any
  supporting/neutral clip resets the streak. Face lock on an
  OTHER-bound track (guest mislabel) rebinds by score.
- **Upgrade:** UNKNOWN-bound + any positive evidence → bind now; the
  track's history retroactively belongs to that identity.
- **Retire:** unseen for retire_after_s (ran ticks only) → entry dies;
  a retired PATIENT = "left view" (release analogue). Re-entry = new
  id, pool/face re-binds (existing machinery).
- **Resilience rails:** reid_ran=False ticks freeze the ledger;
  quality_degraded ticks stitch geometry but are inadmissible for
  audit/retire (mirrors W2). Single-patient invariant at bind time
  (higher face score steals, loser → UNKNOWN).

### What this deletes / demotes (after validation, physical prune)

- IdentityLock enter/hold/park/release + park-on-crowd + release
  hysteresis (intent survives as the audit contra streak)
- TrackletCarry (binding mode passes carry=None; geometry + pool
  covers it)
- Per-tick pool-margin adjudication (pool consulted at birth only)
- Per-tick strikes against bound tracks (strikes now = birth labeling
  + audit input)

Kept unchanged: face countersign, sharpness/quality floors, lock
quality bars (confirm-time), pool + harvest (harvest gated by binding),
scheduler/crop semantics, IdentityResult contract (ledger rewrites the
same fields the lock did; track_id becomes the persistent id).

### Display-veto (rides along)

Backend asks the pod for a screen pass every N ticks
(`detect_screens=true` form flag — pod stays stateless); pod returns
screen regions from the compound prompt. Brain keeps a per-camera
temporal-union region map (decay ~10 min); person tracks with bbox
containment ≥ 0.8 in a screen region are dropped before the ledger and
never harvest. Separation measured: 1.0 vs 0.13.

## Order

1. `track_ledger.py` + `test/unit/test_track_ledger.py` (pure logic,
   TDD) ← this commit
2. Settings flag + camera_ingest seam + carry=None in binding mode
3. Pod screen pass + client + region map + drop
4. Ship + recreate + live solo re-validation (walk-in, back-turn,
   phone-face, screen, exit/re-entry) → then the multi-person gauntlet
   on the new architecture

Uncommitted soak-worktree work; branch + PR + review before any deploy
(per repo rules).
