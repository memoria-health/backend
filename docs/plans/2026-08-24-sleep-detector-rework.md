# Sleep does not fit the detector architecture — rework needed

**Status: OPEN. Nothing here is fixed.** Raised by Aryaman 2026-08-24
after the overnight run: the sleep detector needs reworking *and* so does
the way its results reach a caregiver. The generic detector pipeline was
built for events, and sleep is not an event.

This is a scoping note, not a design. It records what we measured and
which questions are still open, so the rework starts from evidence rather
than from the first plausible fix.

---

## The mismatch, in one line

Every other detector answers **"did a thing happen?"** Sleep answers
**"what state is this person in, continuously, for eight hours?"** The
pipeline treats the second as a special case of the first, and it is not.

Three places where that shows up.

### 1. `<ident>_detected` means the wrong thing

For `fall`, `fall_detected = true` is the event. For sleep,
`sleep_detected = true` means **asleep** — the normal, quiet, desired
state that holds for most of the night. The interesting moments are in a
*different* field, `sleep_activity` (`entering_bed`, `leaving_bed`,
`stirring`, `sitting_up`).

Everything generic keys off `_detected`. So anything we attach to the
sleep detector attaches to "is asleep", not to "something happened".

**Measured cost of exactly that.** Aryaman asked for a confirmer on
sleep *disturbances*. `detectors.eventing = {"confirm": {"k":2,"n":3}}`
is the only lever, and `executor.py` votes whenever `_detected` is true —
so it re-checked **"is this person asleep"**, three times, on every tick
where they were:

```
03:26 -> 10:14 IST     905 ticks voted      confirmed 852   vetoed 53
                       unanimous 3/3 818    unanimous 0/3 41
                       33,258,129 input tokens
                        1,546,171 output tokens
```

~33M input tokens in under seven hours, on one camera, for one detector,
to confirm the quiet state. Disarmed 2026-08-24 (`eventing = null`).

There is no way to say "confirm only when this FIELD takes these
values". That is the missing primitive.

### 2. `record_only` has no delivery path of its own

Sleep fires no event, so nothing downstream runs for it: no
`clip_archival`, no notification, no Home surface. The nightly record
comes from `tools/run_daily_summaries.py`, **a script somebody runs by
hand**. `sleep_nights` last had a row for 2026-08-18 — six days of data
collected and never summarised, not because anything failed but because
nobody ran it.

`src/services/detection/sleep_summary.py` itself is good and was written
against real nights. The gap is around it: nothing schedules it, and a
`record_only` detector has no route to a caregiver except a summary
table two API endpoints read.

### 3. Per-tick verdicts are the wrong unit for a state

The night of 2026-08-23 (19:00 -> 10:52 IST, 4,762 labelled ticks):

- the `asleep` verdict changed **739 times**
- longest unbroken `asleep` run: **15 ticks (~2.5 min)**
- 856 ticks asleep, 3,906 awake

**Do not read that as "the detector is noisy" yet.** The confirmer was
armed at 03:26 and vetoes flip `_detected` to false, so an unknown share
of those changes is the confirmer, not the model. The two are entangled
in this data and the run needs repeating with the confirmer off before
anyone concludes anything about model stability.

---

## Open questions — evidence needed, not opinions

**Q1. Does identity block the night, and should it?**
`sleep_summary` requires the first `entering_bed` to be
identity-confirmed, so a visitor asleep in the bed cannot become the
patient's night. On 2026-08-23 all 30 bed-crossing ticks carried
`identity_patient_in_clip = false` while 1,771 other ticks that night
carried `true`, and the summariser returned `no_confirmed_bed_entry`.
A crossing WAS confirmed once on 08-19 at 02:03, so it is not
structurally impossible.
*Aryaman does not agree with the reading that this is what blocked the
night.* Resolve against clips before changing the rule.

**Q2. What is the right confirm trigger for a state detector?**
Probably field-and-value scoped ("confirm when `sleep_activity` is a
disturbance"), which nothing supports today. Cheaper shapes worth
costing: one vote escalating to three only on disagreement; a cheaper
confirm model; no confirm at all on `record_only`, since nothing acts on
the verdict.

**Q3. What does a caregiver actually see?**
`sleep_nights` feeds two widgets. Whether that is the product for sleep,
or whether it needs its own surface, is unanswered.

**Q4. What is the right cadence?**
Every 10s all night for a still sleeper is a lot of clips. The
sleep-watch forced look already exists for the "still sleeper hides a
wake" case; the base cadence may be far too fast.

---

## Also on the list, smaller

- **Schedule `run_daily_summaries.py`.** Nothing else in this document
  matters if the summary only exists when someone remembers.
- **`sleep_clips.py`** (the disturbance-clip sidecar) is a dev
  instrument writing unscreened frames. It earned its keep, but it is
  not a product path and must not become one.
- The clips from the night are in `out/sleep-clips/650ca94d/` — 48
  files, 290 MB, 26 flagged moments with their fields beside them.
