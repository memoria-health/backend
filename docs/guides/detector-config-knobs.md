# Detector config knobs, by lifecycle stage

Every knob answers a question at exactly one stage. If you know which stage
you are asking about, only a few fields matter.

    1. GATE     does it run this tick, and what does it look at?
    2. ASK      what is the model asked?
    3. PROMOTE  does the answer become an event?
    4. SUSTAIN  while it is still happening, do we re-fire?

---

## 1. GATE — does it run this tick?

| knob | what it decides |
|---|---|
| `is_on` | on/off |
| `always_on` | rides every call regardless of loadout (people_present) |
| `when_scope` | time-of-day window; outside it, skipped |
| `duty_cycle` | `every_eligible_tick` / `thinned` (every k-th) / `scheduled_only` |
| `requires_motion` | needs motion above the floor, else skipped |
| `subject_scope` | `patient` / `anyone` / `environment` — who it is about |
| `where_rooms` | which cameras it is assigned to |
| `bucket` | **what it looks at**: `scene` (full frame) / `person` (crop) / `face` / `rapid` |
| `needs_audio` | audio muxed into the clip |
| `severity` | ≥80 exempts it from every cost gate above |

**The one that bites:** `bucket` decides cropping. A `person` detector that
runs is cropped, always, unless >4 people home-wide. If it cannot be cropped
it does not run — except severity ≥80, which falls open to full frame.

---

## 2. ASK — what is the model asked?

| knob | what it decides |
|---|---|
| `prompt_fragment` | the detector's question, appended to the base prompt |
| `schema_json` | the fields it must return |
| `custom_prompt` / `user_criteria` / `refinements` | caregiver's words, compiled into the fragment |
| `sensitivity` | how eager the wording is |
| `evidence_unit` | `scene` vs `sample` — whole clip or per-sample |

Always emitted first, before any detector: `heard` / `audio_source` / `seen`.
A detector may not claim speech that does not appear in `heard`.

---

## 3. PROMOTE — does it become an event?

`alert_trigger` picks the shape, and **it decides which other knobs apply**:

| `alert_trigger` | fires when | who runs it |
|---|---|---|
| `leading` | the moment it starts | reactive promotion |
| `trailing` | the moment it ends | scheduled runner |
| `duration` | it has lasted `duration_seconds` | scheduled runner |
| `absence` | it did NOT happen by a deadline | scheduled runner |
| `record_only` | **never** — fields feed the summarizers | nobody |

`record_only` short-circuits before everything in this section and section 4.
No entry rule, no confirm, no cooldown, no episode. Sleep and Activity use it.

For `leading` only, `eventing` holds the entry rule:

| field | meaning |
|---|---|
| `entry_m` | qualifying ticks needed to enter |
| `entry_t_seconds` | window they must land in (0 = fire on sight) |
| `confirm_k` | re-checks that must agree before firing |
| `confirm_n` | re-check calls budgeted |

So `{"entry": {"m": 3, "t_seconds": 330}}` = three positive ticks within 5½
minutes. `fire_rule` is just the name for this shape: `instant` (m=1) vs
accumulating (m>1).

Also here: `category` splits fail-quiet vs fail-loud. Routine detectors fire
only when identity confirms the patient; safety detectors fire unless the
patient is confirmed elsewhere.

---

## 4. SUSTAIN — while it is still happening

Only for `is_sustained = true`.

| knob | what it decides |
|---|---|
| `is_sustained` | this is a STATE, not a moment |
| exit gap (constant) | 60s safety / 300s routine — evidence must stop this long to end the episode |
| `cooldown_seconds` | minimum spacing between re-fires of a point event |
| `duration_seconds` | the "too long" threshold, `alert_trigger=duration` only |
| `episode_gap_seconds` | **DEAD.** Replaced 2026-07-29; never read |

**What the exit gap is for:** stopping a re-fire during a brief interruption —
watching TV, going to the bathroom, coming back. One episode, one event.

**Where it breaks:** the gap is wall-clock, but a `thinned` detector produces
evidence sparsely, so an ordinary session can exceed it while nothing changed.
Watching TV fired 3 events in 19 minutes this way (gaps of 380s and 310s
against a 300s window) while the resident sat still — because the compiled
prompt treated *eating* as a competing activity, so the boolean flapped.

---

## The rule that matters most

**A state the caregiver wants to SEE is `record_only`. A state they want to be
TOLD about is `duration`. Only moments are `leading`.**

The compiler cannot currently emit `record_only` — its prompt offers four
options and anything unrecognised falls back to `leading`, so every custom
detector alerts. That is why `Watching TV` was compiled as
`alert_trigger=leading, is_sustained=true`: correctly identified as a state,
with nowhere to put it.

---

## Simplest fix for the state class

Not transitions. Reuse what is already configured:

1. **Entry unchanged** — `eventing.entry` (m ticks in t seconds) turns the
   state ON. Already set on every sustained detector.
2. **Ask the model to maintain it** — while ON, the question becomes "is this
   still happening?" and only a confident NO ends it. Absence of evidence
   never ends it, so a bathroom break, a duty-cycle gap, or a flap cannot.
3. **`record_only` for states nobody needs alerting about** — no event, no
   episode, no exit gap. It lands in the timeline instead.

This deletes the exit gap for state detectors rather than tuning it, which is
the point: the gap is the only knob that must scale with how long the
behaviour lasts, and for a custom detector that duration is unknowable. Entry
counts do not scale — 3 ticks means the same thing for a 2-minute activity and
an 8-hour one.
