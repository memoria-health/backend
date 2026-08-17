# Behaviour + Care categories, and folding away the Sensors tab

**Status:** scoping only. No code written. Decisions marked **[DECIDE]** need
Aryaman before implementation.

**Why now:** the flyer pitches four things Memoria watches for — Safety,
Routine, Behaviour, Care. The product models two. Closing that gap is mostly
honest labelling, but it forces a question the current design has been able to
avoid.

---

## Part 1 — What `category` actually does today

This is the crux, and it is smaller than it looks.

`detector_category` is a Postgres ENUM with exactly two values
(`supabase/schema.sql:17`), mirrored in `src/db/enums.py:17` and
`src/schemas/detector.py:16`.

It drives **exactly one behaviour**: the precision/recall bias in the compiled
VLM prompt (`src/services/detector_compiler.py:204`).

| Condition | Bias |
|---|---|
| `absence` trigger | CONSERVATIVE (a false positive would *suppress* a real alert) |
| `reactive` + `safety` | **INCLUSIVE** — "a missed emergency is far worse than a false alarm" |
| `reactive` + anything else | CONSERVATIVE — "a false alarm is just noise" |

It does **not** drive alerting. Push (`services/push.py`) and email
(`services/email.py`) never read it. It does not drive scheduling.

Two further couplings worth knowing:
- The compiler asks the LLM to classify new customs into `safety|routine`
  (`detector_compiler.py:549-553`) and **defaults to `safety` on anything
  unrecognised** (`:607-608`). Adding enum values without touching this means
  new categories are never auto-assigned, and any typo silently becomes safety.
- In the clients, category is a **boolean in disguise**: every call site is
  `category == .safety ? terra : sky`. There is no four-way mapping anywhere.

### `severity` already exists, and already governs treatment

`detectors.severity` is a nullable integer (`src/db/detector.py:129`), defaulted
from the catalog (`dims.py:74`, `severity=50`). It is live and load-bearing:

- `scheduling/scheduler.py:192,245` — `d.severity < _EMERGENCY_SEVERITY` is the
  emergency exemption that lets a high-severity detector run even when the
  patient is elsewhere.
- `scheduling/packer.py:156,184` — sorts and prioritises the pack by severity.

**So the axis Aryaman was reaching for ("does it depend on criticality or
severity, if that is a thing?") is a thing, it exists, and it is already wired.**

---

## Part 2 — The treatment-policy question

Aryaman framed two options:

> *"for behavior and care it depends on criticality or severity"* — or —
> *"anything that is truly safety only goes in safety and the other 3 are purely
> semantic rigor"*

**Recommendation: the second option, with one correction — and the correction is
the whole point of the change.**

Category becomes **purely semantic**: what kind of thing is this, for labelling,
colour, filtering and the caregiver's mental model. Four values, no behavioural
meaning.

Treatment moves entirely onto **severity**, which already carries it.

The correction: today the compiler bias keys off `category == "safety"`. If
category becomes semantic, that line must key off **severity** instead, or the
bias quietly follows the label and a high-stakes Behaviour detector (a fall
risk, say, or wandering at 3am) gets the conservative prompt and misses things.

```
# today
elif category == "safety":       -> inclusive

# proposed
elif severity >= INCLUSIVE_BIAS_THRESHOLD:   -> inclusive
```

That single change is what makes four categories safe. Without it, adding
Behaviour and Care is a downgrade: detectors that would have been filed under
Safety (and got the generous prompt) become Behaviour (and get the strict one).
**This is the highest-risk part of the whole feature and it is a one-line
condition.**

**[DECIDE 1]** Threshold value. `dims.py` defaults to 50 and
`_EMERGENCY_SEVERITY` already exists as a constant — reusing it keeps one
concept rather than inventing a second. Recommend: reuse `_EMERGENCY_SEVERITY`.

**[DECIDE 2]** The compiler currently asks the LLM to pick a category. With four
values it should pick **category AND severity** independently, and the prompt
needs rewriting so it does not treat them as the same judgement. This is the
largest single piece of work in Part 1.

---

## Part 3 — Migration

The ENUM is the constraint that shapes everything.

**Postgres `ALTER TYPE ... ADD VALUE` is additive and non-destructive** — it fits
the workspace's apply-deltas-only rule with no rewrite. Caveat: pre-PG12 it
cannot run inside a transaction block; on current Supabase it can, but the delta
should be written to be safe either way.

```sql
alter type detector_category add value if not exists 'behaviour';
alter type detector_category add value if not exists 'care';
```

**[DECIDE 3] Spelling: `behaviour` or `behavior`?** The flyer says "Behaviour".
The codebase is mixed — `detector_compiler.py:551` writes "behaviour", other
places use "behavior". Once this is an enum value it is permanent-ish (removing
an enum value requires a type rewrite). Recommend: **`behaviour`** to match the
flyer, since the value leaks into API responses.

**[DECIDE 4] Backfill.** Every existing detector is `safety` or `routine`. Do we
reclassify the built-in catalog into four buckets, or leave existing rows and
only let new ones use the new values? Recommend: **reclassify the built-in
catalog** (it is a fixed list in `catalog_v2.py`, so it is a code change plus a
data delta), and leave user-written customs alone. A caregiver's own detector
silently changing category is worse than an inconsistent catalogue.

Note `catalog_v2.py:144` already asserts row severity matches the catalog map,
so any catalog reclassification needs the delta to match or that check fires.

---

## Part 4 — Colour

Today every client is binary (`terra` = safety, `sky` = everything else). Four
categories need four hues, and the tokens do not exist.

**Web** (`app/globals.css`) has `terra`, `sky`, `success`, `warning`, `danger`,
`action`, each with `-soft` variants. **No fourth or fifth decorative hue.**
`success`/`warning`/`danger` are semantic status colours and must NOT be reused
for category — a Care event is not a "warning", and overloading them makes the
Insights status dots ambiguous.

**iOS** has `NemaraColor.terra/terraSoft/terraDeep`, `sky/skySoft`. Same gap.

The flyer already implies a palette: Safety coral/red, Routine blue, Behaviour
teal, Care teal-green. **Behaviour and Care read as the same family there**,
which is fine on a flyer and not fine in a dense event feed where the two must
be told apart at a glance.

**[DECIDE 5]** Two new hues, distinct from each other and from the existing
status colours. Suggest Behaviour = violet/indigo (distinct from `sky`), Care =
green/sage (already in the Insights palette as `sage`, so partially precedented).
Needs a design call, not an engineering one.

Accessibility constraint worth stating: category must never be **colour-only**.
Every surface already pairs it with an icon or label — that must hold at four
values, because four hues at small sizes is past what colour alone can carry.

### Client call sites to update (all currently binary)

| Surface | File |
|---|---|
| iOS detector row | `Features/Patient/Settings/DetectorRow.swift:32,35` |
| iOS add-detector cards | `Features/Patient/Settings/AddDetectorView.swift:287` |
| iOS event detail | `Features/Patient/Events/EventDetailView.swift:45,48` |
| iOS event card | `Features/Patient/Events/EventCard.swift:11,14` |
| Web events feed | `app/_components/EventsFeed.tsx:160` |
| Web alert toast | `app/_components/alerts/AlertToast.tsx` |
| Android | equivalent surfaces (not yet enumerated) |

Recommend extracting a single `color(for: category)` helper per client rather
than fixing six ternaries in place — otherwise the fifth category repeats this
exercise.

---

## Part 5 — Removing the Sensors tab

**iOS.** `Features/Patient/PatientShell.swift:29` — `PatientTab` is
`live, events, cameras, detectors, settings`. The "Sensors" tab is `cameras`
(label at `:70`). The file's own comment notes five is the iOS maximum before
the bar collapses into "More", so removing one buys real headroom.

Uniquely reachable from that tab today: `PatientCamerasView` → the paired-sensor
list, `CameraDetailView`, `SensorSetupWizard`, `AddCameraSheet`. Live already
has a sensors section (`PatientLiveView.swift:129` handles the "no sensors
paired at all" empty state), which is what makes the fold plausible.

**Android.** `ui/patient/PatientShell.kt` with `SensorsTabGlyph`,
`ui/cameras/CameraListScreen.kt`, `ui/cameras/CameraHosts.kt`. Note
`ui/live/LiveHost.kt:23`: a destination is owned by the shell *"because it is
reachable from the Sensors tab too"* — removing the tab lets that ownership
move down into Live, which is a simplification rather than a workaround.

**Web is different and should probably not change.** It has no Sensors tab —
`/patients/[id]/cameras`, `/cameras/new`, `/cameras/[cameraId]` are routes, and
`app/(app)/settings/page.tsx` frames sensors as setup ("care setup lives here").
A dashboard on a large screen has room for a real management surface; a phone
does not. **Recommend: mobile-only change. Leave web alone.**

**[DECIDE 6]** Agreed?

### What the fold must preserve

The Sensors tab is where a caregiver goes when something is *wrong* — a sensor
offline, needing re-pair. Folding it into Live is right for the common case
(everything works, you want to see the feed) and needs care for the failure
case. The Live sensors section must carry:

- per-sensor connection state, legible without tapping in
- an obvious **Add a sensor** entry point (today's tab has one; a section can
  lose it)
- room to tap each sensor comfortably — Aryaman's point about spacing is the
  crux, since a cramped list is exactly why a separate tab felt necessary
- the offline case surfaced as prominently as the tab did

**Risk:** this is the third UX overhaul of this area (per the detector-UX-v2
work). Deleting a tab is easy to do and expensive to undo once muscle memory
forms. Worth shipping behind the pilot cohort and watching, rather than
assuming.

---

## Rough sequencing

1. **Backend, no client change:** add enum values, decouple compiler bias from
   category onto severity, update the classification prompt. Ship-able alone —
   clients keep rendering two colours because no row uses the new values yet.
2. **Catalog reclassification** + delta.
3. **Colour tokens + per-client helper**, then the six call sites.
4. **Sensors fold**, iOS and Android, mobile only.

Steps 1 and 4 are independent and can run in parallel.

## Branches

Per the standing rule, work stays on each repo's current branch:
`nemara-backend` → `feat/data-collection-timing-truth` (or `feat/event-gate`,
whichever is current when this starts), `nemara-caregiver-app` →
`fix/door-picker-private-camera`, `nemara-web-app` → `fix/dashboard-local-dev`,
`nemara-caregiver-app-android` → `phase-2-networking-auth`.
