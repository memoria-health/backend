# Tracker Eval — Findings, Weaknesses & Coverage

Living record for the SAM3 vs SAM3+EdgeTAM capability eval. Started 2026-07-23.
Companion to `docs/runbooks/tracker-eval.md` (how to run) — this is what we've
learned. Outcome: EdgeTAM won (2026-07-24); the continuous identity service
(`gpu/identity/service.py`) is built on it.

---

## 1. How SAM3's tracker actually works (research)

SAM3 is **concept-promptable** ("person") detection + tracking. It inherits SAM2's
video memory:

- **Memory bank + masklet propagation.** Each frame, the detector finds new
  concept instances; the tracker propagates "masklets" (temporal mask segments)
  from prior frames via self/cross-attention, over a shared Perception Encoder.
- **Detection↔track matching is IoU-based.** A match/update stage merges
  propagated masks with newly detected ones — "ensuring consistency under
  occlusion or re-appearance." IoU decides whether a detection continues an
  existing identity or starts a new one.
- **Occlusion handling** is memory + **detection-score decay + re-prompting**,
  plus periodic memory resets from high-confidence detections to stop drift.
- **It does NOT identify people.** SAM3 gives *track continuity* on the "person"
  concept. WHO the track is (patient vs unknown) is entirely our job (face +
  body pool). SAM3 renumbering ≠ misidentification; it's loss of continuity.

**Undocumented (matters to us):** max occlusion duration, and the frame-gap
threshold beyond which re-appearance is treated as a NEW instance. Empirically,
our ~30s unobserved gap with real movement is past that threshold → renumber.

Sources: Roboflow "What is SAM 3", SAMURAI (motion-aware memory for SAM tracking).

---

## 2. How OUR pipeline uses it (the crux)

The pod `observe()` takes a **batch of ~66 frames (one ~8s window at 8fps)** per
tick and returns tracks with ids. Then there's a **~30s blind gap** before the
next window.

- **Within a tick (the 8s window): continuous.** SAM3's memory propagates ids
  across all 66 frames → in-window movement and brief occlusion are handled.
  *This is why "SAM sees every frame so movement shouldn't break it" is right —
  within a window.*
- **Across ticks (the ~30s blind gap): fragile.** The pod (`gpu/perception/continuous.py`)
  re-maps each new window's tracks onto the previous window's ids **by IoU**.
  - Stationary → boxes overlap → same id (stable). Observed: T4 held across
    ticks 16–26 while seated.
  - Moved during the gap → new position doesn't overlap the old → IoU fails →
    **new id (renumber).** Observed: T2→T4→T5→T6 on each locomotion.
- **The ledger has no safety net on the SAM3 arm.** `track_ledger.py`
  (`src/services/identity/` — legacy, slated for deletion at P3) keys
  entries DIRECTLY by pod track_id and "never re-associates: no geometric
  stitcher, no fragment merge." Those SAM3-renumber *compensations were deleted*
  in favor of EdgeTAM's stable ids. So a renumber = a dead entry = re-acquire
  from scratch.

**Therefore:** the entire identity design — bind-once, then "stay locked no
matter where they go or what they do" (the couch-at-20ft northstar) — is
**predicated on stable track ids**. SAM3 doesn't provide them across our gaps.
EdgeTAM (persistent session, mask carry across ticks) is the *foundation the
design rests on*, not a tuner. That is the whole thesis of this eval.

---

## 3. Observations — solo, living room, SAM3 arm (2026-07-23, cam 3c6570af)

8fps (~66f/observe), 40s tick, e2e ~20s (track ~14s = SAM on 66 frames @210ms).

| phase | ticks | what happened |
|---|---|---|
| Acquire | 11–12 | Face lock 0.538 at ~1m frontal → harvest body → body 0.741 next tick |
| Hold (seated) | 16–26 | T4 held GREEN even as raw body dipped to 0.67–0.72 (kept on live track) |
| Move: behind chairs | 13 | T2 renumbered → UNBOUND(left_view) → ~2 min body-only re-acquire → T4 @16 |
| Move: along couch | 27–28 | T4→T5 → SEAT_KEPT then UNBOUND(left_view) → 1-tick gap → T5 rebind @28 |
| Move: up to camera | 32–33 | T5→T6 → SEAT_STOLEN via face 0.531 + harvest → body healed to 0.849 |
| Release (left room) | 34 | UNBOUND(left_view) at first observe after exit; trk→0; **no linger, no phantom** |

**Asymmetry:** stripped hysteresis → **flickery acquire** (renumber churn) but
**crisp release** (1 tick, low FP). Prod's scene-extension trades the other way
(smoother hold, ~150s FP linger on exit).

---

## 4. Weaknesses identified (ranked)

1. **SAM3 renumbers the track on any real movement** (across the blind gap).
   Cross-tick IoU stitch fails when position changes; ledger has no stitcher →
   every renumber costs a re-acquire. *The headline SAM3 weakness.*
2. **Bind-once-hold requires stable ids** — defeated on SAM3. "Stay locked no
   matter where they go" cannot hold if the id keeps changing under the lock.
3. **Harvest is face-gated, face only locks close+frontal** (≥36px, ~1m in a big
   room). So the body pool is *exclusively close-frontal samples* → far / angled
   / seated body match is marginal (grazes the 0.72 bar). (Aryaman, 2026-07-23.)
4. **Distance hurts twice:** bigger position delta → renumber more likely; AND
   re-acquire is body-only (face won't re-lock far) against a pool with no far
   samples → 1–3 tick flicker, up to ~2 min behind the chairs.
5. **Our 40s-tick / ~8s-window config is the worst case for cross-tick IoU** —
   huge blind gap. Amplifies #1. Partly a config artifact, not pure SAM3 fault;
   denser sampling would help the SAM3 stitch (EdgeTAM removes the need).
6. **Observability trap:** the trace card `no_match` (raw per-tick body re-match)
   looks like an unbind, but the ledger may KEEP the binding on a live track
   (timeline stays green). Consider a "kept→Tn" marker so amber ≠ lost.
7. **Ledger is in-memory; a worker re-subscribe wipes it** (reset markers catch
   it). A clothes change wipes the body pool (OSNet is appearance-keyed).

**Strengths (be fair):** stationary solo hold is solid; release is clean/fast
(low FP); close re-acquire is instant (face re-lock heals the pool → 0.85).

---

## 5. Test coverage matrix

Dimensions: **arm** × **room** × **single vs cross-room** × **people**. Each cell
exercises: acquire · hold-through-movement · occlusion · break/leave-return ·
release (+ for multi: crossing-swap, FP-on-guest).

| arm | room | config | people | status | notes |
|---|---|---|---|---|---|
| sam3 | living | single | 1 | **DONE** | §3 — flickery acquire on move, clean release |
| sam3 | living | single | 2+ | pending | crossing/swap + FP-on-guest |
| sam3 | bedroom | single | 1 | pending | expect worse: far, face rarely locks |
| sam3 | bedroom | single | 2+ | pending | |
| sam3 | cross-room | living→bedroom | 1 | pending | hand-off between cameras |
| sam3 | cross-room | living→bedroom | 2+ | pending | |
| edgetam | living | single | 1 | **next** | same script as §3 — the head-to-head |
| edgetam | living | single | 2+ | pending | |
| edgetam | bedroom | single | 1 | pending | |
| edgetam | bedroom | single | 2+ | pending | |
| edgetam | cross-room | living→bedroom | 1 | pending | |
| edgetam | cross-room | living→bedroom | 2+ | pending | |

**Decisive next run:** edgetam / living / solo — repeat §3's exact script. If the
id survives the walk-behind-chairs + couch moves (no T4/T5/T6), EdgeTAM wins solo
and the renumber churn is confirmed as the SAM3 ceiling.
