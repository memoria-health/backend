# Soak — tiered inference and cost, as an ADDITION to the existing plan

Written 2026-08-26. **This is not a new soak plan.** The soak plan is
`docs/runbooks/soak-plan.md` (phases, hardware, attribution matrix) and the
methodology is `docs/runbooks/soak-protocol.md` (truth-CSV, scoring, pass
bars). Both stand. This document adds **one new phase** and **one missing
meter**, and nothing else.

Deleted in the same commit as superseded: `2026-08-25-overnight-state.md`,
`2026-08-25-verification-owed.md`, `2026-08-25-motionless-object-veto.md`.
Phase 1 of `2026-07-28-pilot-master-worklist.md` is stale — the runbooks
replaced it.

---

## 0. What is already true (verified 2026-08-26, do not re-derive)

`scripts/soak_check.py`: **23 checks, 23 passed, 0 failed, 0 skipped.**

- `coverage: a shared sensor exists` — **TV Room covers 2 patients.**
  Phase 3 attribution is already armed and under test.
- `fix/analyses-patient-keying` is **merged into main** (`1f5eb10`). The
  Phase 3 prerequisite is satisfied — observations resolve through coverage,
  not the `cameras.patient_id` shim.
- Both sensors online, node `7yettlzalhkkl0` serving 2/2 ports.

**Assert, do not watch.** `watch -n 60 '.venv/bin/python scripts/soak_check.py'`

---

## 1. Corrections to my own earlier framing

Recorded so the mistakes are not repeated, per the standing rule to read the
conclusions docs before proposing anything.

**The unidentified-subject rule was already decided**, in soak-plan Phase 3.
I re-opened it as an open question. It is not:

> nobody identifiable → **no** care/behaviour event for either resident;
> a **safety** detector fires → event for **both**.
> *"A fall nobody can attribute is still a fall; a care record filed against
> the wrong resident is not recoverable."*

**k-of-N was already ruled on**, in soak-protocol: confirm is a per-detector
opt-in for miss-prone or false-fire-prone detectors, never a blanket 3x.
Dropping it for tiering is consistent with a position already held, not new.

**FN measurement already has tooling.** I proposed inventing it. It exists:
`soak_truth.csv` + `tools/soak_report.py --truth`, and its FN rows already
print how many ticks inside the window answered `detected=true` — which
separates *"the model never saw it"* from *"the decision layer held it"*.
That is precisely the number the tiering design needs, and it predates me.

**My taxonomy map was incomplete.** I read `alert_trigger` values from the
current DB rows and got `leading` / `duration` / `record_only`. The actual
enum (`src/db/detector.py:79`) is **`leading` / `trailing` / `duration` /
`absence`**. `trailing` and `absence` are exactly the two firing paths
soak-protocol flags as an **OPEN gate** — never soak-validated, and required
before detection is re-enabled. Tiering must map onto them too (§3).

**The third camera is not an open question.** The Warner bridge sensor is
provisioned and publishing as `cam_c51733e6`, for Aryaman's father, and
soak-plan Phase 2 covers its enrolment. I asked as though it were undecided.

---

## 2. The two cost levers — memorise, there are no others

1. **The VLM call.** Detector count in the prompt, frame count (`TARGET_FPS`),
   `media_res`, and call frequency (suppression).
2. **LiveKit ingestion by the consumer.** What backend + identity download
   and decode. `livekit_subscribe_quality` / `livekit_idle_subscribe_quality`.

Independent. Lowering the subscribe layer does **not** cut VLM tokens —
Gemini re-encodes at `media_res` regardless. Measured 2026-08-26: TV Room
dropped to MEDIUM 29 times overnight; the VLM bill did not move.

**Lever 2 has no meter at all today.** That is the single blocking gap.

---

## 3. NEW — Phase 6: tiered inference

Slots after Phase 5 in soak-plan.md. Replaces k-of-N.

**Tier-1** answers everything cheaply. Anything it marks true goes to
**tier-2**, which is asked to *refute*, scoped to only the firing detectors,
split by bucket.

### Model facts, verified 2026-08-26

| Model | $/M in | $/M out | Temperature | Note |
|---|---|---|---|---|
| gemini-2.5-flash-lite | 0.10 | 0.40 | **honoured** | **RETIRES 2026-10-16** |
| gemini-3.1-flash-lite | 0.125 | 0.75 | ignored | |
| gemini-3.5-flash-lite | 0.30 | 2.50 | ignored | **running today** |
| gemini-3.7-flash | 0.75 | 3.75 | ignored | intro to 2026-12-31, then 1.50/7.50 |

Temperature is deprecated across the whole 3.x Flash line — accepted,
silently ignored, HTTP 200; future generations will 400. We set it nowhere
today, so everything runs at default 1.0. That is the mechanism behind "the
same benign clip fired 4/10".

**Consequence:** temp-0 tier-1 works only on 2.5-flash-lite, which dies in
~7 weeks. Prompt-level determinism (explicit system-instruction rules) is
the only version that survives October. Build it now.

### Mapping onto the taxonomy

| Axis | Values | What it decides for tiering |
|---|---|---|
| `bucket` | scene / person | **Prices tier-2** — one call per bucket, not per detector |
| `is_sustained` | point / state | **What triggers tier-2** (below) |
| `alert_trigger` | leading / trailing / duration / absence / record_only | **Whether tier-2 runs at all** (below) |
| `evidence_unit` | verdict / sample | verdict → "refute this claim". **`sample` has no claim to refute — likely sits out entirely. OPEN.** |
| `category` | safety / routine / behaviour / care | Refusal bar: safety tier-2 defaults to NOT suppressing under uncertainty |

**`is_sustained` is the trap.** Point detectors trigger on the verdict —
measured non-state fire rate 2026-08-26 was **0.02%** (2 in 8,293 ticks).
State detectors must trigger on the **transition**, never the level:
`sleep_detected` was true on 1,100/8,293 ticks, so level-triggering would
cost more than today. That is the 08-23 failure verbatim (33M tokens
re-asking "is this person asleep" 905 times). The field-and-value scoping
shipped 2026-08-25 is the mechanism and took sleep from ~245 confirms/night
to 3. Tier-2 inherits it unchanged.

**`alert_trigger` decides participation:**

- `leading` — tier-2 gates the alert; latency budget applies (Aryaman
  accepts ~10s, 2026-08-26; still measure against the <=90s SLA).
- `duration` — the state must persist anyway; tier-2 rides the entry edge
  with slack.
- `trailing` — fires on the FALLING edge. **Tier-2 semantics here are
  undefined and the path is itself unvalidated.** Resolve before arming.
- `absence` — makes **zero** VLM calls by construction. Tier-2 is
  inapplicable. Confirms the design does not break it.
- `record_only` — **tier-2 must not run.** Nothing is delivered, so
  refutation buys nothing a nightly re-read could not.

### The acknowledged weakness

Tier-2 only sees what tier-1 marked true, so it **cannot fix false
negatives**. Aryaman's position (2026-08-26) is that our empirical FN rate
is very low and this is an acceptable trade — recorded as a **decision, not
a measurement**. `detection-accuracy-knobs.md` holds the opposing evidence:
the 08-06 falls failure was tier-1 *narrating* rather than seeing.

**The existing truth-CSV scoring already produces the number.** Run it
against tier-1 on 2.5-flash-lite before committing.

### Pass bars

- Per taxonomy cell: tier-1 TP >= 90%; tier-2 kills >= 70% of tier-1 FPs;
  tier-2 kills **zero** true positives.
- **Any cell where tier-2 suppresses a real safety event fails the phase.**
- Run **logging-only first** — both verdicts recorded, suppression OFF.
  Turn suppression on only after the above holds. Measuring before
  suppressing is the whole lesson of the 08-05/06 night: 24 false alerts,
  attributable to no single change, because no FP rate was recorded across
  any of them.

---

## 4. NEW — the missing meters

Neither exists today. Both block reporting.

- [ ] **LiveKit bytes ingested, per camera per hour.** Lever 2 is
      unmeasured, so we cannot report savings on it at all. This is what
      answers Aryaman's direct question: *when both residents are located,
      do the other cameras actually get cheaper?* Log `layer_request`
      transitions with the **reason** (idle / no-presence / occupied) and
      the resident location that drove it.
- [ ] **`tier` and `bucket` tags on `sched_calls`.** The ledger already
      carries model, tokens, detectors, ms. Without a tier tag the §5 fire
      rate stays an assumption instead of a measurement.
- [ ] Dashboard reconciliation at hour 6 and hour 24: recorded tokens x list
      price vs the provider bill. **Flag >10% drift.** Both known dashboard
      bugs (double-count, stale price) are fixed; this re-verifies.

Scenario to run once both meters exist: both residents in the TV Room for
30 min — the bedroom must step down **and** suppress, confirmed on both
meters, within 60s of the last resident leaving. Inverse: one resident per
camera, neither steps down.

---

## 5. Cost model — the model swap is the SMALL lever

### The unit is the occupied-room-hour, not the camera

Aryaman, 2026-08-26: *"you are taking this 2 camera example but we are
monitoring 2 patients. The second camera does not come into play when it is
1 patient and they are in 1 room."*

Correct, and it invalidates every per-camera figure I quoted earlier. A
camera with nobody in it already costs ~$0 (empty-room suppression works —
TV Room ran five 100%-empty hours at ~$0). So cost tracks **occupied-room-
hours**, and a single patient can only occupy one room at a time.

Measured 2026-08-26, 20h, 3 detectors, 3.5-flash-lite:

| Camera | ticks | VLM calls | % called | occupied-hrs | $ / 20h |
|---|---|---|---|---|---|
| Aryaman Bedroom | 7,184 | 6,386 | 89% | 17.7 | $31.41 |
| TV Room | 6,565 | 3,422 | 52% | 9.5 | $15.32 |
| **Total** | | **9,808** | | **27.2** | **$46.73** |

**$1.72 per occupied-room-hour** at 3 detectors — which independently
reproduces the "~$2/occupied-room-hr at media_res=high" figure already
recorded in the cost-truth notes. The model is sound; my *unit* was wrong.

**The marginal cost of the second patient is not 2x.** Two patients
generated 32.6 occupied-room-hours/day against ~18 for one — **1.8x**,
because they are sometimes in the same room and one camera covers both.
Pinning that multiplier is one of the soak's primary economic outputs
(Part 2, §7).

### Basis

Measured 2026-08-26, 2 cameras, 20h, 3.5-flash-lite: **9,776 calls,
112.6M input tokens**, 11,516 in / 525 out per call.

### Where the tokens actually are

Decomposing 11,516 input tokens at 3 detectors (`detection-inference-architecture.md`
§7: video ~221 tok/frame @ high, prompt ~540 + ~390/detector):

    video    ~9,806   85%      <-- 11s @ 4fps @ media_res=high
    prompt    ~1,710  15%

**Video is 85% of the input bill.** At 15 detectors the prompt grows to
6,390, so the split becomes video 61% / prompt 39% — and 16,196 in,
~1,620 out per tick, at ~11,731 calls/day.

**Changing the model changes the PRICE. It does nothing to the VOLUME.**
That is the whole answer to "how come it is still expensive": at $0.10/M we
are still pushing **190M input tokens per day across two cameras**.

### The ladder — priced per occupied-room-hour

15 detectors, tier-2 at the 2% assumption throughout. 360 calls per
occupied-hour at today's cadence.

| | $ / occ-hr | **1 patient/mo** | **2 patients/mo** | What changed |
|---|---|---|---|---|
| **A. Today** — 3.5-flash-lite, no tiering | $3.21 | **$1,733** | **$3,120** | baseline |
| **B. + tiering** — 2.5-lite tier-1, 3.7 tier-2 | $0.985 | **$532** | **$958** | price only |
| **C. + prompt caching** | $0.778 | **$420** | **$756** | prompt is byte-identical every call |
| **D. + fps 4→2 on non-rapid** | $0.543 | **$293** | **$528** | video halves; resolution untouched |
| **E. + occupied-but-static suppression** | $0.380 | **$205** | **$369** | assumes 30% fewer calls |

Single patient assumed at **18 occupied-room-hours/day** (540/month);
two patients at the **measured 1.8x**, not 2x. Both assumptions are soak
outputs, not constants — see §7.

**A single patient today costs $1,733/month in inference. That is the
number that has to come down, and it is one patient, not two.**

### The three levers I had left out

1. **Prompt caching.** Cached input costs ~90% less. Our system prompt +
   detector fragments are byte-identical across all ~11,700 daily calls —
   75M tokens/day of pure repetition at 15 detectors. On Flash, lean on
   *implicit* caching; explicit adds a per-token-per-hour storage charge
   ($1.00/M/hr on Flash) that only pays off at very high reuse.
   **Caveat: implicit caching reportedly stopped applying its discount for
   `gemini-3.1-flash-lite` on some projects around 2026-06-30. Verify the
   discount actually lands on our project before banking it.**
2. **Frame count, not resolution.** `detection-inference-architecture.md` §7:
   video scales linearly with fps, and *"slow branches can drop to 1-2fps,
   rapid needs 8fps"*. 4→2 fps on non-rapid buckets halves the dominant term
   **without touching `media_res`** — so it does not collide with the 08-06
   finding that resolution is already marginal for distant subjects.
3. **Occupied-but-static suppression.** The bedroom made **360 VLM calls per
   hour at motion 0.33** while a person slept motionless. Today suppression
   asks *"is the room empty?"*, never *"has anything changed?"*. This needs
   real design — a still sleeper must not become an invisible fallen person —
   which is why it sits last.

At step D, **output tokens are 54% of the remaining cost** (1,620/tick x
11,731 calls). Trimming per-detector output verbosity — short verdicts,
reasoning emitted only when true — becomes the next lever after these.

### Tier-2 sensitivity (at step B)

| Tick fire rate | Tier-2 $/day | Total $/day | $/mo (2 cam) |
|---|---|---|---|
| 0.02% (measured, quiet) | $0.05 | $26.65 | $800 |
| 2% (planning assumption, 100x measured) | $4.67 | $31.27 | $938 |
| 10% | $23.35 | $49.95 | $1,499 |
| 25% (implausible) | $58.38 | $84.98 | $2,549 |

Caveats: 2.5-flash-lite dies 2026-10-16 (on 3.1-flash-lite tier-1 rises
~25%); 3.7-flash intro pricing doubles 2027-01-01; assumes empty-room
suppression keeps working, which it does — TV Room cost ~$0 across five
100%-empty hours.

---

## 6. Deferred, with a home

**Nighttime IR and sleep.** Substantial work landed (CLAHE, IR person
threshold, remap IoU, ground grace, seat parking, carry grace,
transition-scoped confirm, nightly summariser). Good enough to keep
collecting. **Not a soak gate.** Three open items:

1. **Our IR detector is wrong.** `is_ir()` thresholds channel spread at 2.0
   on a subsampled frame and it chatters — 74 mind-changes in 5h39m on
   2026-08-26, two of them **1 second apart**. A mechanical IR-cut filter
   cannot move that fast, so an unknown fraction are our own threshold
   noise, not camera behaviour. Needs hysteresis on the *detector* before
   any conclusion about the camera is drawn, and before the 6-tick grace
   shipped 08-26 can be evaluated at all.
2. **Identity is 0% through IR sleep** — ~2,100 consecutive ticks with
   `people_count = 1` and `identity_patient_in_clip = false`. At 25-30 feet
   this may be unsolvable; two people in one bed under IR certainly is
   today. The 08-25 night summarised only because the identity-confirmed
   `entering_bed` landed at 03:16, minutes before identity went dark. Luck.
3. **Body pool does not persist.** `body_pool_embeddings` has 0 rows;
   in-memory only, dies on restart. Likely feeds directly into (2).

**Phase 5 masked-capture test stays exactly as written in soak-plan.md** —
it is unverified in both directions and I had omitted it entirely. Its
standing constraint holds: **do not delete the 496 existing B2 objects
until it runs.**

---

## 7. The soak splits in two

Aryaman, 2026-08-26: *"the soak needs to be split into 2 where 1 section is
just testing whether single patient is working in a cost effective and
accurate manner and whether multi-patient is working in the way we
intended across our entire system."*

The split is real, not cosmetic: **Part 1 has one variable (does the
pipeline work, cheaply and correctly, for one person) and Part 2 has a
different one (does everything that branches on *which* person work).**
Running them together is how a cost regression and an attribution bug
become indistinguishable.

### Part 1 — single patient: cost-effective AND accurate

**One patient covered, other patients' coverage removed.** Not "ignore the
second patient" — actually uncovered, so nothing multi-patient can
contaminate the result.

*Cost questions:*
- [ ] What is the true **occupied-room-hours/day** for one person living
      normally? (I assumed 18. Measure it.)
- [ ] $ per occupied-room-hour at each rung of the §5 ladder, measured, not
      modelled. Every rung lands as a real number or it does not count.
- [ ] Does prompt caching actually apply on our project? Verify the discount
      lands before banking it — implicit caching reportedly regressed for
      `3.1-flash-lite` around 2026-06-30.
- [ ] LiveKit bytes ingested (lever 2), which has no meter today.

*Accuracy questions:*
- [ ] TP/FP per taxonomy cell via `soak_truth.csv` + `soak_report.py --truth`,
      with the confounders soak-protocol already specifies.
- [ ] **The FN number** for tier-1 on 2.5-flash-lite. This is the one that
      decides whether the tiering design's known weakness is acceptable.
- [ ] `trailing` and `absence` firing paths — the standing OPEN gate.
- [ ] Tier-1 vs tier-2 verdicts logged side by side, **suppression off**.

**Pass bar:** a $/patient/month we would quote to a customer, and a TP/FP
table with no cell where tier-2 kills a true safety positive.

### Part 2 — multi-patient: does the system do what we intended

Only starts once Part 1 passes. Everything here branches on **identity**,
which is why it cannot be mixed into Part 1.

- [ ] soak-plan **Phase 3 attribution matrix** as written — already armed
      (`coverage: a shared sensor exists`, TV Room covers 2 patients). The
      unidentified rule is already decided: no care/behaviour event for
      either, safety fires for **both**.
- [ ] **The inverse safety test**, never run live: both residents in frame,
      one falls, the alert must still fire. Highest-stakes cell.
- [ ] `SOLO_TICKS_BEFORE_SUPPRESSING = 3` — chosen, never measured. Pull the
      solo-run-length distribution. If churn breaks most runs before 3, the
      gate silently never engages.
- [ ] Cross-camera: resident A in the bedroom, event in the TV Room — must
      not attribute to A.
- [ ] **The 1.8x multiplier.** Measure occupied-room-hours for two patients
      against Part 1's single-patient baseline. This is the number that says
      whether a second resident is nearly free or nearly double.
- [ ] Does the *other* camera get cheaper when both residents are located in
      one room? Confirm on both meters, within 60s of the last resident
      leaving.
- [ ] Phase 2 (Warner sensor / third camera) and Phase 5 (masked capture)
      ride here.

**Pass bar:** zero misattributions; every safety cell fires; a measured
marginal cost for patient #2.

### Sequencing

1. Revert the two soak overrides in `.env`
   (`NEMARA_SENSITIVITY_SCREENING_ENABLED=false`, `NEMARA_SLEEP_CLIPS_DIR`)
   before this stack points at anyone but its operator.
2. Add the two meters (§4). Without them Part 1's cost questions cannot be
   answered at all.
3. Land tiered inference behind a flag, **logging only, suppression off**.
4. **Part 1.** Score with `soak_report.py --truth`.
5. Turn tier-2 suppression on only after Part 1's pass bars hold.
6. **Part 2** — re-cover the second patient, run soak-plan Phases 2-5.

**Node deploy hazard** (soak-plan, still live): `push_node_code.sh` silently
no-ops when the running agent predates the overlay block. After any change
under `gpu/deploy/`, restart the container (`pod.sh up`) before trusting a
push. There is no log line for a push that did nothing.
