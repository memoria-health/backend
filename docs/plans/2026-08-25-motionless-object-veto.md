# Vetoing a motionless object that is not a person

**Status: DESIGNED, NOT BUILT.** Aryaman asked for this at ~03:50 on
2026-08-25 ("implement it if it is a straight path"). It is not a
straight path, and the reason is worth writing down rather than
discovering at 4am on a night the sleep detector was being shipped.

## What was asked

The TV Room's statue catches the light and gets picked up as a person.
It never moves. So: veto tracks that do not move.

## Why the obvious version is dangerous

"A box that has not moved is not a person" also describes **a sleeping
person**. This veto lives in the identity service, which is what
produces tracks; if it drops a sleeper's track, identity reports an
empty room, `people_present` goes false, and `nightly.py` discards the
tick as having no sleep evidence. The sleep record would come back
blank, and the failure would look exactly like "he did not sleep".

That is the one thing the same night's work was fixing. Shipping the
two together would have quietly cancelled one with the other.

## Why the existing veto does not already cover it

`display_veto.py` has precisely this idea and it works: a `KIND_STATIC`
region (a painting) vetoes a contained track UNLESS the track moved,
because a painting cannot move and a person can. The reason it misses
the statue is upstream — the veto only acts on regions the perception
pass GROUNDED, and "statue" never grounded a region in any wording
tried. It was dropped from the concept rotation on 2026-08-25 for that
reason: it spent a third of every rotation asking a question with no
answer.

So the machinery is right and the input is missing.

## The design that is safe

Learn the region from the tracks instead of from a concept.

1. A track that stays within a few pixels of its birth box for a long
   time (minutes, not seconds) and was NEVER identity- or
   face-confirmed is a candidate.
2. When a candidate dies and a new track is born at the same rectangle
   and does the same thing, that rectangle becomes a `KIND_STATIC`
   region — persisted alongside the grounded ones in
   `/persist/state/screen_regions_<cam>.json`, which already exists.
3. From then on the EXISTING logic applies unchanged, including the
   movement exemption that spares a real person who later stands in
   front of the statue.

The repetition across track deaths is what makes it safe. A statue is
in the same place every time it is rediscovered. A sleeping person is
not rediscovered in the same rectangle night after night, and more
importantly a bedroom camera's sleeper would have to survive rule 1,
which excludes anyone identity has ever confirmed.

## What must be true before it ships

- **Never on a camera with a sleep detector assigned**, at least for v1.
  Cheap, and removes the whole class of failure.
- The candidate window measured, not guessed. The TV Room churned
  tracks six times in three minutes on 2026-08-24, so "a track that
  lives for minutes" may be rarer than it sounds and the rule could
  simply never fire.
- A log line for every region learned this way, because a region learned
  wrongly is invisible until someone stops being detected.

## Cheaper thing to try first

The statue is caught because it is SHINY, not because it is
person-shaped. Before building region learning, try grounding it as what
it looks like to the model — reflective object, ornament, vase, figurine
on a table — and measure. `_SCREEN_CONCEPT_FILE` on the node takes one
concept per line and is re-read every pass, so each attempt is one
`echo` and two minutes, with no rebuild. If one of them grounds, the
existing static-region path handles the rest and none of the above is
needed.
