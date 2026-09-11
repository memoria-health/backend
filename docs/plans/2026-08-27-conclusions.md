# Conclusions — 2026-08-26/27

Supersedes §2 and §5 of `2026-08-26-soak-requirements.md`; those numbers
assumed 2 calls/tick and an invented gate rate. Everything below is
measured unless marked otherwise.

## Cost, measured

- **Today: 1 VLM call per tick.** 7 days: 35,927 scene calls, 20 person
  calls. The person lane is idle because only `fall`, `setup_test`,
  `sleep` are on and all three are scene-bucket. Not a bug.
- **~$1,150–1,360 / patient / month** at 3 detectors. Model reproduces the
  real bill within 9%.
- **~$3,400 / patient / month** at 15 detectors, because the catalog's
  11:5 split turns on a second call per tick.
- **Video is ~78% of input.** Detector count barely matters; the *call*
  is what costs. 15 detectors can be cheaper than 10 (cache floor).
- **Nothing is suppressed when a person is present.** 34,505 of 34,505
  occupied ticks made a call. The motion floor is unreachable dead code
  (`scheduler_person_requires_track=False`); it caught 15 ticks in 50,627.

## Two bugs, both fixed on `fix/vlm-cost-part-order-single-confirm`

- **Caching had never worked, once.** Video was the first Part, so every
  request began with a unique MP4 and the prompt behind it could never be
  a shared prefix. Prompt now leads. Only prefixes ≥4,096 tokens cache,
  so the scene call caches and the person call does not.
- **k-of-N removed → single pass.** Measured: 5 of 6 confirm rounds
  returned 0 of 3 — unanimous disagreement, not split decisions. Extra
  votes bought nothing. Confirm costs ~$1.44/month; it was never a cost
  problem, it was a correctness one. 1,621 tests pass.

## What does NOT work

- **fps 4→2** — a fall is sampled ~once. Below the floor.
- **Cutting the window without the tick** — W < T leaves seconds
  unwatched. W=5/T=10 is a 50% blind spot, not a saving.
- **A gate OR'd across detectors** — `activity` (person_present) and
  `fire_smoke` (environment, no motion needed) each fire ~100% of ticks.
  The gate does nothing.
- **Emergency exemption pins both lanes.** `fall` 95, `choking` 100,
  `fire_smoke` 98 are exempt from every cost gate and sit in both
  buckets. No gate reaches them without a policy change.

## What does work

- **Cadence split.** `record_only` continuous states (`activity`,
  `sleep`) do not need 10s sampling. Config only (`duty_cycle`,
  `sample_every_k`). ~$3,400 → ~$2,000.
- **Longer tick, overlap kept.** T=20/W=21 is −13% to −30% (output
  scaling unmeasured). Costs latency, not evidence quality.
- **Output trimmed on "no"** — reasoning only when a detector is true;
  never touch `heard`/`audio_source`/`seen`, which fixed the 08-06 FPs.

## The architecture conclusion (unproven, the real prize)

CV facts as the substrate; VLM only where evidence is object-semantic.

    Layer 1  pose + EdgeTAM + open-vocab grounding, on the node we own   $0
    Layer 2  TEXT call over the fact stream, no video uploaded         ~$84/mo
    Layer 3  VIDEO, only when semantics are needed (~5% of ticks)      ~$88/mo

~$172/month mid-case against ~$1,665 for the scene lane today. Both rates
are estimates and are what the shadow run must measure.

## Product

- **We are a witness, not an alarm.** Up to 70% of falls in older adults
  are unwitnessed, and syncope-vs-mechanical sends the patient down two
  different clinical pathways. Nair asked for a fall checklist because
  the missing witness is the central problem in fall assessment. Alarms
  are commoditised; the account of what happened is not.
- **SafelyYou/Inspiren can copy the checklist.** The moat is the
  combination: home (not facility, where staff already witness) + cost
  structure that reaches India + accumulated personal baseline. Only the
  third compounds, and only with retention.
- **The clinical framing:** functional decline is graded in ADLs/IADLs.
  A camera observes them directly; a wrist observes none of them.
  *The only continuous, passive, in-home measurement of ADLs.*
- **Split the surfaces.** Family-selectable alerts (sparse, semantic,
  VLM) vs always-on clinical tracking (continuous, CV, free, never a
  checkbox). That split is also exactly the cost architecture.
- **Gait speed** is the sixth vital sign (<0.8 m/s risk, <0.6 m/s ~1.9x
  frail) and its *variability* tracks MCI (ORCATECH/ISAAC, on ceiling PIR
  sensors — we have video). It is arithmetic on data we already compute
  and then discard.

## Next, in Aryaman's order

1. **Decide what families should be able to track**, and what we surface
   as events vs markers. Everything downstream depends on this.
2. **Falls:** CV detector → 3.7-flash confirm → structured fall
   assessment as the output. Take Nair's checklist literally.
3. **In parallel:** build gait. Start by **persisting track position over
   time** — it unlocks gait speed, life-space, transfers and circadian
   rhythm at once, needs no new model, and every day we don't is corpus
   we cannot recover.

## Open, needs a decision

Whether a *geometric* precondition may gate an emergency detector. The
exemption exists because a *semantic* gate (one bad face match) once
suppressed choking on the real patient. Everything below ~$2,000/month
depends on this answer, and it wants staged positives, not more analysis.
