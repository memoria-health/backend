# Physical measurement — grounding, calibration, and the experiment

Scope for gait speed and sit-to-stand. Design only; nothing implemented.
Companion to `2026-08-27-product-spec.md` (Layer 2).

## 0. The principle

> **Every measurement is validated by a physical invariant that is
> independent of the model producing it.**

A confidence score from a model is the model marking its own homework. A
physical invariant is not. Four invariants do all the work here:

| invariant | validates |
|---|---|
| a person's **height** does not change | calibration, and pose |
| **limb lengths** do not change | pose |
| a person cannot **teleport** | tracking |
| a person cannot be in **two places** | identity |

Bounding-box aspect ratio has no such check — it assumes the tracker is
correct and offers no way to know when it isn't. That is why it is not an
acceptable basis for a clinical measurement, and Aryaman is right to
reject it.

## 1. Calibrating a Tapo / Warner camera

**Method A — known-object homography (the ground truth).**
A printed checkerboard or A4 sheet placed flat on the floor at 3-4
positions across the frame. Solve the ground-plane homography from the
known dimensions. Two minutes, done once at install, by the person who is
already in the room hanging the camera.

This is grounded in a *physical measurement*, not in any model working. It
is the only method whose error we can state.

**Method B — pedestrian-height auto-calibration (the monitor).**
Given head and ankle keypoints and the patient's known standing height,
the ground plane can be recovered from observations at multiple positions.
Weaker than A, but it runs **continuously and for free**.

**Method C — DeepStream 9.1 AutoMagicCalib.** Unknown accuracy in a
domestic room. Evaluate it *against* Method A, never instead of it.

### The design

Calibrate with **A**. Monitor with **B**. If recovered person height drifts
more than a few percent from the enrolled height, the calibration is
**stale** — the camera has been bumped — and every metric measurement from
that camera is quarantined until recalibration.

That is what makes calibration a grounded factor rather than a
set-and-forget assumption. A camera that moved and nobody noticed is the
silent failure that would corrupt months of baseline.

**Required at enrolment:** the patient's standing height. One form field,
and it is the anchor for everything downstream.

## 2. The four grounded factors

| factor | grounded by | continuously validated by | silent failure if ungrounded |
|---|---|---|---|
| **Calibration** | physical object at install | recovered height vs enrolled height | every metric measurement wrong, no error signal |
| **Pose** | per-keypoint confidence | **limb-length constancy in metric space** — a shin does not change length | hallucinated joints, wrong timing |
| **Tracking** | track continuity | no-teleport bound + height constant along the track | a merged or split track silently attributes one person's motion to another |
| **Identity** | face / body match confidence | single-occupancy windows as ground truth | **baseline poisoned permanently** |

**Rule: nothing enters the baseline unless every validator passes.** Fail
closed. These are statistics, not events — discarding half the samples
costs precision; admitting one wrong sample costs the baseline.

## 3. Acceptance thresholds, declared before the experiment

Set from the clinical literature, so "accurate enough" is not decided
after seeing the data.

**Gait speed.** Perera 2006: small meaningful change **0.05 m/s**,
substantial **0.10 m/s**. To detect the smallest meaningful change, our
measurement error must be materially below it.
- target bias ≤ **0.03 m/s**
- target 95% limits of agreement within **±0.05 m/s**

**Sit-to-stand.** SPPB chair-stand bands are ~2.5 s wide over five reps,
so ~0.5 s per rep. Single-rise timing error must sit well inside that.
- target bias ≤ **0.15 s**
- target 95% limits of agreement within **±0.3 s**

Reference values for context — 5xSTS healthy: 60-69 **7.8 ± 2.4 s**,
70-79 **9.3 ± 2.1 s**, 80-85 **10.8 ± 2.6 s**; >12 s discriminates stroke;
~10.9 s sarcopenia cutoff. SPPB: 4 pts ≤11.19 s · 3 pts 11.20-13.69 ·
2 pts 13.70-16.69 · 1 pt ≥16.70 · 0 if unable or >60 s.

**Honesty constraint:** the validated test is *five consecutive* rises.
Real life gives us one at a time. We measure **single-rise duration**,
trend it against his own baseline, and **never report an SPPB score**.

## 4. The experiment

Ground truth is a **stopwatch and a tape measure** — which is not a
compromise, it is the clinical gold standard for both tests.

**Phase 0 — calibrate.** Method A on both cameras. Record the residual
reprojection error; that is the calibration's own accuracy figure.

**Phase 1 — gait speed.** A tape-measured 4 m course.
- 10 walks × 3 distances from camera (≈3 m, 6 m, 9 m) × 3 headings
  (toward, across, diagonal) = **90 walks**
- each timed by stopwatch over the marked course

**Phase 2 — sit-to-stand.**
- 10 rises × 3 chair positions × 2 lighting conditions (day, IR) = **60 rises**
- plus 10 rises using armrests, and 10 with a walker if available

**Phase 3 — stress the conditions.** Loose vs fitted clothing (pose
degrades on loose), partial occlusion by furniture, a second person in
frame.

**Phase 4 — analysis.** **Bland-Altman**: bias and 95% limits of
agreement, per condition. *Not correlation* — correlation can be excellent
while the measurement is systematically wrong, which is exactly the
failure we need to exclude.

### Kill criteria, also declared in advance

- Gait speed limits of agreement exceed **±0.05 m/s** at a given distance
  → that distance is **out of scope**, and the camera moves or the
  measurement is dropped.
- Sit-to-stand error exceeds **±0.3 s** → not clinically usable.
- Pose keypoint confidence below threshold on **>40%** of frames in IR →
  IR is out of scope for that measurement, and we say so to the family.

Publishing the kill criteria before the run is what makes this an
experiment rather than a demo.

## 5. Yield — the separate question that decides everything

Accuracy without yield is useless. Independently of Phase 1-4, measure on
ordinary days, storing nothing:

- **qualifying rises per day** — a rise at a known furniture region, with
  identity confident and pose confident
- **qualifying walking bouts per day** — ≥3 s, ≥2-3 m, roughly straight,
  unconfounded, identity confident

If a senior living alone yields three clean bouts a day, gait speed cannot
carry a weekly statistic and sit-to-stand has to carry the product. That
is knowable from tracks we already compute, before any storage schema
exists.

## 6. Implementation shape (not a plan, a shape)

- **Node computes, database stores summaries.** One row per *event* — a
  rise, a bout — not per tick. ~100 rows/day/person, not 15.8M/year/home.
- **Every stored row carries its validators**: calibration age and
  residual, pose confidence, identity confidence, track continuity flag.
- **Two-stage admission.** Store everything measured, with validators.
  Admit to the *baseline* only rows where all validators pass. Then a
  later change to the admission rule can be re-applied retrospectively
  without losing data.
- **Pixel-space is recoverable.** Calibration is a static per-camera
  transform, so trajectories stored before calibration convert afterwards
  — provided we know the camera has not moved, which is what Method B
  monitors.

## Sources

Perera et al., *Meaningful Change and Responsiveness in Common Physical
Performance Measures in Older Adults*, JAGS 2006 · SPPB chair-stand
scoring · Five Times Sit-to-Stand normative values and sarcopenia cutoffs ·
DeepStream 9.1 AutoMagicCalib.
