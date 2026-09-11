# Product spec — what Memoria is, after 2026-08-26/27

Single source for what we build. **Supersedes and replaces**
`2026-08-27-safety-detector-research.md` (deleted; its epidemiology is in
Appendix A). Cost and inference architecture live in
`2026-08-27-conclusions.md`; the soak lives in `docs/runbooks/soak-plan.md`.

## 0. What changed

**Before:** a flexible VLM watching for a long list of things, alerting on
each. Detectors were the product.

**After:** a **continuous measurement instrument** that establishes one
person's baseline, with a **small set of well-built emergency detectors**
on top, whose output is a **structured clinical account** rather than a
boolean.

Detectors are not the product. They are the sensors. The product is the
baseline, the deviation from it, and the account of what happened.

Three reasons this shift is forced, all measured this week:

- A single detection is worth almost nothing to a family. A fall happens
  once or twice a year; 363 days the emergency layer produces nothing.
- Emergency detection is commoditised (Life Alert, SafelyYou, Inspiren,
  Ring). The account and the baseline are not.
- Sixteen general detectors at 10s cadence costs ~$3,400/patient/month.
  Six focused ones on a CV substrate is a business.

---

## 1. Layer 1 — Emergency (7)

Configurable: the family chooses what buzzes their phone.

| detector | notes |
|---|---|
| **Fall** | with **time-down escalation** — 10s is a trend entry, 20min is an ambulance. Output is the structured account, not a boolean |
| **Unresponsiveness** | subsumes syncope, post-ictal, overdose, stroke, down-without-a-fall. "A person is in a state they should not be in, for this long" |
| **Choking** | gated on *within 20 min of an eating episode*. Actionable in our markets: India multigen households, live-in attendants, facilities with staff. Known miss: silent choking, person turned away |
| **Left home alone** | unaccompanied exit, out of character for them. Renamed from `wandering` — that is a diagnosis we cannot make; once out the door we do not know if they are disoriented or buying milk |
| **Unattended stove** | 10 min of lit burner, nobody present. Prevention, not detection — a smoke alarm fires when it is already a fire |
| **Calling for help** | audio. The only channel left when immobile |
| **Fire / smoke** | cheap, table stakes, do not sell on it |

**Last-known-well** is a field on every one of these, not a detector.
20-30% of ischemic strokes are wake-up or unwitnessed and were historically
excluded from thrombolysis for want of an onset time. We know it to the
minute, for free, from data we already store.

**Cut, with reasons:** entrapment and failure-to-appear (subsumed by
unresponsiveness), post-ictal seizure (is unresponsiveness), long lie (is
the fall escalation ladder), intruder (commodity, not senior-specific),
blood/vomiting (subsumed), wet flooring (a risk, not an event), potential
heart issue (**we cannot see cardiac function — remove from the deck**).

## 1b. Routine and behaviour (7, family-facing)

Measured always; only the alert is configurable.

**Sleep** — bedtime, sleep onset, wake time, night wakings.
**Meals** — whether they ate, when, how long it took.
**Medication** — whether taken, and within the expected window. Flags a second dose shortly after a first (2/3 of adverse-drug-event hospitalisations are unintentional overdoses).
**Activity** — how the day is spent across lying, sitting, standing, moving.
**Bathroom usage** — how often they go. A change is often the first UTI sign.
**Agitation** — pacing, restlessness, raised voice, and when in the day.
**Repetitive speech** — the same thing said repeatedly in a short window. Common in dementia; a *sudden increase* indicates delirium, which in an older adult means infection until proven otherwise.

## 2. Physical measurements (4) — always on, never a checkbox

Continuous, CV-derived, ~free, and not switchable. Family sees trends only.

**Sit-to-stand time** — how long to stand up, whether armrests are needed, whether an attempt fails. Anchors: 5xSTS norms 7.8/9.3/10.8 s by decade; >12 s discriminates stroke; ~10.9 s sarcopenia cutoff. Measures *weakness*, the Fried criterion that predicts inability to get off the floor.
**Gait speed** — walking speed and its day-to-day consistency. Sixth vital sign: <0.8 m/s elevated risk, <0.6 m/s ~1.9x frail. **Variability tracks MCI** (ORCATECH/ISAAC) and moves before the mean does.
**Time out of home** — hours away from the house. A real Life-Space Assessment level boundary; contraction is an early mobility-decline sign.
**Rooms visited** *(exploratory)* — distinct camera-covered rooms per day. Plausible proxy for the innermost LSA zone but **unvalidated, and confounded by camera placement.** Do not attach LSA's predictive evidence to it.

**Instrument anchors:** Fried frailty (weight loss, exhaustion, weakness, slow gait, low activity) — we observe **3 of 5**. SPPB (gait speed, chair stand, balance) — we observe **2 of 3**. No proprietary Memoria score.

**Derivations, not tracked things:** sedentary time, nighttime activity, transfer counts, meal-duration trend, circadian regularity, failed-rise counts, bathroom-frequency change. Each is a query over the above. **Store base events, derive on read** — so a change in definition never costs us history.

## 3. Layer 3 — Outputs

| output | cadence | audience |
|---|---|---|
| **Reassurance** | daily | family. He's up, he ate, he moved |
| **Trend** | weekly | family. What changed against *his own* normal |
| **The account** | per event | clinician. Structured, travels into the record |

Deviation is always **against his own baseline**, never population norms —
the range across healthy 75-year-olds is enormous. Establish 4-6 weeks of
normal plus its variance; report **sustained** departure, never a single
day.

### The line on advice

- **Safe:** an observation. *"He's walking less this month than usual."*
- **Safe:** established public guidance, clearly attributed (WHO/CDC
  physical activity guidance for older adults).
- **Not safe:** generated advice about a specific patient. That is a
  clinical recommendation from an unregulated system.

**Families get observations. Clinicians get interpretations.** We are the
witness and the instrument, not the physician. (Aryaman 2026-08-27:
condition-specific healthy-aging guidance is a later phase; general-purpose
for now.)

## 4. User-facing vs not

| | family-facing | configurable | notes |
|---|---|---|---|
| Emergency alerts (7) | yes | **yes** | alert fatigue is what churns customers |
| Daily reassurance | yes | no | the reason to open the app |
| Weekly trend | yes | no | |
| Event account | via clinician | no | the document that travels |
| Measurement layer | as trends only | **no** | switchable = no baseline = no product |
| Identity, track quality, gate decisions, cost/tick | **no** | no | internal telemetry |

**Rule:** measurement is automatic, notification is configurable.

## 5. Care provision — a separate product, not a detector

*"He was assisted to the bathroom at 3am"* is a fact about the **patient**.
*"The attendant assisted 4 times in 8 hours"* is a performance record about
an **employee** — who is a data subject in their own right, and whose
consent the patient cannot give.

Ship the first now. Treat the second as a deliberate B2B decision with
counsel, not something that arrives via a custom detector written in 30
seconds. **The compiler should flag any custom detector whose subject is a
non-patient person, at creation time.**

## 6. Where we actually are

**Built and running:** VLM detectors (fall, sleep, activity, medication,
meal, wandering, stove, agitation, calling-for-help, choking, fire,
hazards, setup-test), identity/seats, EdgeTAM tracking, YOLO26x, face and
body embeddings, nightly sleep summariser, scheduler v2, compiler v2 for
customs, privacy/censoring, consent gate.

**Fixed 2026-08-27:** prompt-before-video (caching had **never** worked),
k-of-N replaced by single-pass confirm. On
`fix/vlm-cost-part-order-single-confirm`, 1,621 tests pass, **needs a PR**.

**Not built — this is the gap:** persisted track position over time, gait
speed, pose, camera calibration, baseline + deviation, the structured fall
account, last-known-well, embedding-diff gate, respiratory rate.

Nothing in Layer 2 exists yet. That is the whole product.

## 7. Game plan

**Now, this week**
1. Merge the two cost fixes (PR open).
2. **Persist track position over time.** Already computed on the node and
   thrown away every tick. Unlocks gait speed, life-space, transfers and
   circadian rhythm at once, needs no new model, and every day we don't do
   it is corpus we can never recover.
3. Evaluate **DeepStream 9.1** on the node beside the existing pipeline —
   AutoMagicCalib for the ground plane (gait speed in m/s), pose, ReID.
   Fed from the Tapo RTSP the gateway already has.

**Next**
4. Structured fall account — take Nair's checklist literally.
5. Baseline + deviation on whatever measurements exist.
6. Cadence split and the embedding-diff gate (see `conclusions.md`).

**Not available to us (Aryaman 2026-08-27)**
There is no live-customer panel to ask. Build something **generally
useful** on the clinical evidence above, ship it, and treat feature
requests as they arrive. The literature is the substitute for customer
discovery here — gait speed, circadian regularity, frailty criteria and
ADLs are validated regardless of what any one family asks for.

**Standing rule:** check the ecosystem before any build over a week.
Twice this week the answer was "someone shipped it" — Roboflow for the
pipeline, DeepStream for perception.

---

## Appendix A — the epidemiology

| cause | scale (US, 65+) |
|---|---|
| **Falls** | **41,400 deaths, 3,850,612 ED visits** (2023); death rate **up 21%** 2018→2024 |
| Motor vehicle | 9,123 deaths — outside the home, not addressable |
| Poisoning | 7,803 deaths |
| Suffocation | 4,283 deaths (includes choking) |
| Fire | ~930 deaths, ~3,000 injured/yr; 65+ die at **2.5x** population rate, 85+ at **4.4x** |
| Adverse drug events | **14.3%** of older-adult ED visits, **76% preventable**; 2/3 are unintentional overdoses |
| Wandering | **>34,000 elope/yr** aging in place; **32%** of missing-person incidents end in death |
| Long lie (>1hr down) | ~50% of *uninjured* fallers cannot rise unaided; **half of those down >1hr died within 6 months** |

Sources: CDC falls data & MMWR; CDC/NCHS unintentional injury 65+;
*Inability to get up after falling* (90+ cohort) and Bloch, *Critical
Falls*, JAGS 2012; NEJM *Emergency Hospitalizations for Adverse Drug Events
in Older Americans*; elopement in community-dwelling dementia; USFA/NFPA
older adults and fire; AHA wake-up stroke and 2026 thrombolysis guidelines.
