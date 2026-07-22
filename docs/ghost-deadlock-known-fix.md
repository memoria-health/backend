# Track-ledger ghost deadlock — diagnosis + known easy fix

**Status:** KNOWN BUG, fix documented below, **NOT applied**. The architecture
is pivoting to EdgeTAM-continuous tracking (see memory
`nemara-edgetam-continuous-pivot`), which dissolves the whole bug class by
giving the pod stable track ids across ticks. Apply this fix **only** if the
ledger architecture must be kept as the live path.

## Symptom

Patient walks up, face-locks cleanly, is tracked well for ~30s, then walks to
the sofa and shows **UNKNOWN despite being physically present**. The ledger
still "holds" the binding internally (`patient_conflict_kept` fires for
minutes) but the output shows `lock=None` / `patient_in_clip=False`.

## Mechanism (live evidence 2026-07-22, cam 3c6570af)

1. Face lock at 0.57 binds `pid=1019` (via face). Tracks well while SAM3 keeps
   that within-clip track number alive.
2. Subject walks; SAM3 **renumbers** the body track across the ~10s tick
   boundary (discrete-clip renumbering). The geometric stitch (`_stitch`)
   can't bridge the jump, so a NEW entry (`1020`/`1012`) births instead of
   continuing `1019`.
3. `1019` loses its live track → bodyless **"ghost"**. But `_retire`
   (`track_ledger.py` ~:795) extends a PATIENT ghost's `last_seen` every tick
   the scene sees *anyone* → the ghost **never retires** (immortalized).
4. The live re-numbered track (`1020`) matches the patient pool (0.72) and
   tries to bind, but `_bind_patient` keeps the incumbent on an equal-tier tie
   (`score > holder.confidence` is false when 0.72 == 0.72, `track_ledger.py`
   :707) → `conflict_kept` → the live track is **refused every tick**.
5. **Deadlock:** the ghost holds the seat with no pixels to render; the live
   subject is refused → output shows no patient.

## Easy fix — the ghost-yield invariant (general, ~15 lines)

*A binding lives on a LIVE body. A bodyless ghost may coast a tick or two but
must never (a) block a live patient-grade body, nor (b) be immortalized.*

- **`_bind_patient`**: if the current holder was NOT seen this tick
  (`holder.last_seen < now` ⇒ a ghost) and the challenger IS live and
  patient-grade (it reached `_bind_patient`, so it already passed pool ≥ 0.72
  or a face lock), the **challenger wins** — hand the seat over instead of
  keeping the ghost. Log `patient_handoff`.
- **`_retire`**: the scene-present extension of a bodyless PATIENT ghost
  applies **only** when no live tracks explain the scene (`not seen`) — the
  genuine lying-in-bed / tracker-blink case. If other tracks are live this
  tick, the scene person is them → let the ghost retire on grace.

**Accuracy:** the challenger's patient-grade bar (pool 0.72 / face lock) is the
existing precision guard — no threshold is lowered, and a similar-outfit
relative is gated exactly as it is today. The spike that motivated this was
purely a stale-incumbent problem, not a recall problem.

## Why it's not applied

The root cause is the discrete ~10s SAM3 clips renumbering the subject. The
continuous-tracking pivot removes that renumbering at the source, so the ghost
stops forming. The ghost-yield invariant still survives as **insurance for
genuine occlusion re-entry** in the new architecture, but it is no longer the
hot path — so it's carried forward as a design principle, not shipped here.
