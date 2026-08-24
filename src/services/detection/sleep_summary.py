"""Nightly sleep record — three answers.

REWRITTEN 2026-08-03 (Aryaman). The night has exactly three outputs:

  1. when they fell asleep              -> sleep_onset
  2. when they woke UP during the night -> awakenings[] (windows)
  3. when they finally woke up          -> wake_time

## The shape

Bed occupancy is a STATE WE REMEMBER, entered and left by two crossing
EVENTS, and never re-asked in between:

    AWAY --entering_bed (+ identity confirmed once)--> IN BED
    IN BED --leaving_bed--> AWAY

That is the whole reason this works. A sleeper under a duvet cannot be
re-detected, so any design that keeps asking "is someone in the bed?"
fails exactly when it matters. Asking instead "did someone cross into or
out of the bed?" is a motion question over a 15-26s video clip, which is
what the model is good at. Measured on the 2026-08-02 night: three
`leaving_bed` ticks in ~1350, all correct, matching the resident's own
account to the minute — while the static question produced NINE phantom
out-of-bed events on a night he never once left the bed.

## Memory: silence never moves the state

A tick where nobody is visible is NOT evidence of absence. It is usually
evidence of the opposite — someone under the covers. State-dependent
missingness: the gap is a symptom of the hidden state, so treating it as
neutral biases the answer. Only a positive `leaving_bed` crossing ends
occupancy, and only positive wakefulness ends sleep. Nothing is ever
inferred from an absent observation.

## Identity: asked once, around the crossing

A visitor asleep in the bed must not become the patient's night, so the
first `entering_bed` must be identity-confirmed. After that it is never
consulted again — identity needs a visible upright person and a sleeper
is neither (08-02: 110 confirmed ticks while awake, then zero across
three hours of solid sleep).

"Around", not "on". The crossing tick itself is already too late:
somebody climbing into bed is bent, turned away and half under a duvet,
so identity fails there for the same reason it fails afterwards. The
confirmation may come from any tick in the preceding
`IDENTITY_LOOKBACK_SECONDS` — the person walking in and getting ready,
while they are still upright and visible. Nothing is relaxed about WHO
must be confirmed; only when we are allowed to have seen them.

Pure function — no I/O, no clock reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

# Both thresholds come from ACTIGRAPHY, the clinical standard for
# scoring sleep from the outside, validated against polysomnography.
# They were 600/180 — invented by me, and roughly inverted.
#
# Onset: of the 4, 5, 6, 10 and 15-minute immobility criteria tested in
# the actigraphy literature, FIVE minutes gave the most accurate sleep
# latencies. 15 was evaluated and lost.
ONSET_MIN_SECONDS = 300          # 5 min
# Waking is deliberately harder to trigger than falling asleep — the
# standard rule pairs a 5-minute onset with a 10-minute wake criterion.
# The asymmetry is the point: brief restlessness is not waking up.
WAKE_MIN_SECONDS = 600           # 10 min
# ...but the 10 minutes buys CERTAINTY, and only when the evidence is
# ambiguous. Actigraphy waits because all it has is movement, and a roll
# looks like a wake. We often have better: the model reports WHAT it saw.
# When it can see a person on a phone with the lamp on, waiting another
# two minutes adds nothing — on the 08-06 night that rule discarded a real
# 8-minute awakening backed by 30 consecutive device ticks.
#
# So an awakening is real if EITHER the observed awake time reaches
# WAKE_MIN_SECONDS (unchanged, the ambiguous case) OR the model positively
# SAW wakefulness for this many consecutive ticks.
WAKE_MIN_EVIDENCED_TICKS = 3
# Activities that are a SIGHTING of wakefulness rather than a failure to
# see sleep. `none` is deliberately excluded: `asleep=False, activity=none`
# is an absence of a sleep call, not evidence of anything, and the prompt
# tells the model to answer False when it cannot tell.
_AWAKE_ACTIVITIES = frozenset({"awake_in_bed", "sitting_up", "leaving_bed"})
# Below this the tick is too uncertain to use at all.
#
# NOTE this is a VISIBILITY floor, not a sleep-probability. The prompt
# defines the field as "0.7+ = what you reported is clearly visible", and
# the 08-06 night bears that out: conf=1.0 ticks were only 44.8% asleep
# while conf=0.95 ticks were 97.2% asleep, because 1.0 is used equally for
# a confident ASLEEP and a confident AWAKE. It filters degenerate ticks
# (empty description, conf 0.0) and nothing else — do not read it as
# "how sure are we that they were asleep".
MIN_CONFIDENCE = 0.6
# Consecutive in-bed ticks that re-open occupancy when the return
# crossing was missed. Small: a single tick could be a misread, but the
# 08-02 return produced hundreds in a row.
REENTRY_MIN_TICKS = 3

# How far back the admissibility gate may look for an identity
# confirmation IN THIS ROOM before accepting a bed entry.
#
# The gate demands the patient be identity-confirmed before their night
# can open, and it used to demand it ON the crossing tick. Measured on
# the night of 2026-08-23, bedroom camera, ticks with a person visible:
#
#     entering_bed     9    identity true 1    false 8
#     leaving_bed     17    identity true 0    false 17
#     awake_in_bed   807    identity true 1    false 806
#
# Identity answered "not the patient" for 806 of 807 ticks of the
# resident lying in his own bed. That is not a visitor. It is this
# module's own premise — "identity needs a visible upright person and a
# sleeper is neither" — arriving one step earlier than expected, because
# somebody climbing into bed is already bent, turned away and half under
# a duvet. So the gate asked the one question identity cannot answer at
# the one moment it cannot answer it, and the night was thrown away: the
# only confirmed crossing landed near dawn, leaving all 856 asleep ticks
# outside any occupancy span.
#
# Asking earlier changes nothing about WHO must be confirmed. A visitor
# is never identity-confirmed as the patient in this room, so they still
# cannot open the night. The patient walks in, is confirmed while
# upright, gets ready, and lies down — and stops being recognisable as
# themselves at exactly the point this gate used to start looking.
#
# One hour, chosen from the same night. The real bed entry was at
# 21:56:30; the last identity confirmation was 21:07:27, 49 minutes
# earlier, because he was in his room reading in between. Every window
# from one hour to twenty-four gives the IDENTICAL answer — 11 spans, 4
# runs, 306 minutes of sleep — so this is not a threshold balanced on an
# edge, it is the difference between allowing an earlier sighting or
# not. Ten and thirty minutes both recover nothing at all:
#
#     lookback    600s   spans 1    runs 0    sleep   0m
#     lookback   1800s   spans 1    runs 0    sleep   0m
#     lookback   3600s   spans 11   runs 4    sleep 306m
#     lookback  86400s   spans 11   runs 4    sleep 306m
#
# The smallest window that works is the one to keep: it still requires
# the patient to have been seen in that room within the hour.
IDENTITY_LOOKBACK_SECONDS = 3600  # 1 h

_ENTER = "entering_bed"
_LEAVE = "leaving_bed"


@dataclass(frozen=True)
class SleepTick:
    """One labeled tick (from analyses detector_fields)."""

    at: datetime
    asleep: bool                 # sleep_detected — genuinely asleep
    activity: str                # entering_bed|leaving_bed|stirring|sitting_up|awake_in_bed|none
    confidence: float            # sleep_confidence
    patient_confirmed: bool = True


def _in_bed(t: SleepTick) -> bool:
    """Does this tick, on its own, prove the person is IN the bed?

    Being asleep proves it. So do the in-bed activities — all three name
    a person who is in the bed. `none` while awake proves nothing either
    way and must not re-open a span.
    """
    return t.asleep or t.activity in ("stirring", "sitting_up", "awake_in_bed")


def _identity_lookback(ts: list[SleepTick]) -> list[bool]:
    """For each tick: was the patient identity-confirmed on this camera
    within the last `IDENTITY_LOOKBACK_SECONDS`?

    One backward pass, so the gate costs nothing to widen. A tick that is
    itself confirmed answers True for itself, which keeps the original
    behaviour as a special case of this one.
    """
    out = [False] * len(ts)
    last_ok: datetime | None = None
    for i, t in enumerate(ts):
        if t.patient_confirmed:
            last_ok = t.at
        out[i] = (
            last_ok is not None
            and (t.at - last_ok).total_seconds() <= IDENTITY_LOOKBACK_SECONDS
        )
    return out


def _in_bed_spans(ts: list[SleepTick]) -> list[tuple[int, int]]:
    """(start_idx, end_idx) index spans during which the patient is in bed.

    The FIRST span must open on an identity-confirmed `entering_bed` —
    that is the admissibility gate, and a visitor must never open the
    patient's night.

    After a `leaving_bed`, a span may re-open EITHER on another crossing
    or on `REENTRY_MIN_TICKS` of direct in-bed evidence. Crossings alone
    are too brittle to carry the night: on the 08-02 night the resident
    got out of bed at 01:48 and back in minutes later, and the return
    crossing was never emitted — so a crossings-only rule discarded the
    following five hours of sleep entirely. Exits are reliable; entries
    are not, and the night cannot hang on the unreliable one. Identity
    is NOT re-checked on a re-entry — it was established at the gate,
    and a sleeper cannot be face-confirmed anyway.
    """
    confirmed_recently = _identity_lookback(ts)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    opened_once = False
    run = 0
    for i, t in enumerate(ts):
        if t.activity == _LEAVE:
            if start is not None:
                spans.append((start, i))
                start = None
            run = 0
            continue
        if start is not None:
            continue                     # already in bed; nothing to do
        if t.activity == _ENTER and (opened_once or confirmed_recently[i]):
            start, run, opened_once = i, 0, True
            continue
        # Evidence-based opening. Before the gate has ever been passed
        # the evidence must ALSO be identity-confirmed — that is what
        # keeps a visitor from opening the patient's night. Afterwards
        # in-bed evidence alone is enough, because identity cannot see a
        # sleeper and re-demanding it is asking for the impossible.
        #
        # Needed because `entering_bed` is not guaranteed: across ~1350
        # ticks on the 08-02 night it fired ZERO times while
        # `leaving_bed` fired three times, all correct. Hanging the whole
        # night on the crossing that does not fire yields a blank record.
        ok = _in_bed(t) and (opened_once or confirmed_recently[i])
        run = run + 1 if ok else 0
        if run >= REENTRY_MIN_TICKS:
            start = i - run + 1
            run = 0
            opened_once = True
    if start is not None:
        spans.append((start, len(ts) - 1))
    return spans


def _sleep_runs(ts: list[SleepTick], lo: int, hi: int) -> list[tuple[int, int]]:
    """Sustained sleep runs inside [lo, hi].

    A break between two sleep runs is measured by the AWAKE TICKS ACTUALLY
    OBSERVED in it — never by elapsed clock time. Two distinct things
    otherwise masquerade as a wake-up:

      * unobserved time. Ticks where nobody is visible are dropped
        upstream, so a sleeper who slips under the duvet leaves a hole
        with no ticks at all. Timing that hole reported a 26-minute
        awakening at 06:26 on the 08-02 night, when the log plainly reads
        "lying in bed under covers ... head on pillow" on both sides of
        it. Silence must never become wakefulness.
      * a single flicker tick, which spans zero seconds of observation.

    Both collapse out naturally: with no awake ticks between two sleep
    ticks, or with awake ticks spanning less than `WAKE_MIN_SECONDS`,
    the runs are one run.
    """
    asleep_idx = [i for i in range(lo, hi + 1) if ts[i].asleep]
    if not asleep_idx:
        return []
    runs: list[list[int]] = [[asleep_idx[0], asleep_idx[0]]]
    for i in asleep_idx[1:]:
        if _break_is_an_awakening(ts, runs[-1][1], i):
            runs.append([i, i])
        else:
            runs[-1][1] = i
    # NOT filtered by ONSET_MIN_SECONDS here. Runs from every span are
    # merged across span boundaries first (see `summarize_night`), because
    # a single false crossing splits a span and would otherwise turn one
    # long sleep into two short ones — dropping BOTH.
    return [(a, b) for a, b in runs]


def _break_is_an_awakening(ts: list[SleepTick], prev_end: int, next_start: int) -> bool:
    """Is the gap between two sleep runs a real wake-up?

    Every index strictly between two asleep ticks is an OBSERVED awake
    tick — invisible ones were dropped upstream and never reach this list,
    which is what keeps silence from becoming wakefulness.
    """
    first, last = prev_end + 1, next_start - 1
    if last < first:
        return False                                 # contiguous sleep
    if _gap(ts, first, last) >= WAKE_MIN_SECONDS:
        return True                                  # long enough on its own
    # Shorter than the duration rule — accept it only on positive sighting,
    # and only if sustained. A lone tick is exactly the shape of the 05:49
    # hallucination ("got out of bed and abruptly collapsed to the floor"
    # between two ticks of lying asleep under the covers), so the same
    # reasoning as REENTRY_MIN_TICKS applies: one tick can be a misread.
    streak = 0
    for i in range(first, last + 1):
        t = ts[i]
        if not t.asleep and t.activity in _AWAKE_ACTIVITIES:
            streak += 1
            if streak >= WAKE_MIN_EVIDENCED_TICKS:
                return True
        else:
            streak = 0
    return False


def _merge_across_spans(
    ts: list[SleepTick], runs: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Join runs whose separating break is not a real awakening.

    `_sleep_runs` only ever sees one span, so a break created by a crossing
    is invisible to it. This applies the identical test to the seams.
    """
    if not runs:
        return []
    merged: list[list[int]] = [list(runs[0])]
    for a, b in runs[1:]:
        if _break_is_an_awakening(ts, merged[-1][1], a):
            merged.append([a, b])
        else:
            merged[-1][1] = b
    return [(a, b) for a, b in merged]


def _iso_utc(dt: datetime) -> str:
    """ISO 8601 with an explicit +00:00.

    This module works in NAIVE UTC, and `.isoformat()` on a naive value
    produces a string with no offset — which every reader is then free to
    interpret in its own zone. The `timestamptz` COLUMNS are stamped
    through `_as_utc` for exactly this reason; the jsonb windows were
    not, so they went out naked (2026-08-25). An iOS decoder that
    requires an offset rejects them outright, and one that does not
    silently shifts the whole night by the phone's zone.
    """
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).isoformat()


def _gap(ts: list[SleepTick], a: int, b: int) -> float:
    return (ts[b].at - ts[a].at).total_seconds()


def summarize_night(ticks: list[SleepTick], *, window_end: datetime) -> dict:
    """Fold one night's ticks (any order) into the nightly record."""
    ts = sorted(
        (t for t in ticks if t.confidence >= MIN_CONFIDENCE),
        key=lambda t: t.at,
    )
    out: dict = {
        "bedtime": None, "sleep_onset": None, "wake_time": None,
        "duration_minutes": None, "disturbances": 0,
        "awakenings": [], "out_of_bed_events": [], "flags": [],
    }
    if not ts:
        out["flags"].append("no_ticks")
        return out

    spans = _in_bed_spans(ts)
    if not spans:
        # Either nobody was seen getting into bed, or the crossing was
        # never identity-confirmed. Both mean the night is inadmissible.
        out["flags"].append("no_confirmed_bed_entry")
        return out
    out["bedtime"] = ts[spans[0][0]].at

    # Spans are cut by crossings, and a crossing can be wrong. Merging
    # across span boundaries with the SAME test used inside a span means a
    # lone false `leaving_bed` can no longer manufacture an awakening it
    # could never have produced mid-span. The onset filter runs last, on
    # the merged runs, so a split night is not discarded as two short ones.
    runs = _merge_across_spans(ts, [
        r for lo, hi in spans for r in _sleep_runs(ts, lo, hi)
    ])
    runs = [(a, b) for a, b in runs if _gap(ts, a, b) >= ONSET_MIN_SECONDS]
    if not runs:
        out["flags"].append(
            "incomplete_night" if ts[-1].at >= window_end else "no_sleep_onset"
        )
        return out

    # 1. when they fell asleep
    out["sleep_onset"] = ts[runs[0][0]].at

    # 3. when they finally woke up — end of the last sustained sleep run.
    # If sleep was still running when observation ended, we did not see
    # them wake and must not invent a time.
    last_end = ts[runs[-1][1]].at
    if runs[-1][1] == len(ts) - 1 and last_end >= window_end:
        out["flags"].append("wake_unobserved")
    else:
        out["wake_time"] = last_end

    # 2. wake-ups DURING the night, as WINDOWS (never a bare count —
    # a count cannot be checked against what the person remembers).
    awakenings = [
        {
            "from": _iso_utc(ts[prev[1]].at),
            "to": _iso_utc(ts[nxt[0]].at),
            "duration_minutes": round(_gap(ts, prev[1], nxt[0]) / 60),
        }
        for prev, nxt in zip(runs, runs[1:])
    ]
    out["awakenings"] = awakenings
    out["disturbances"] = len(awakenings)
    # Existing jsonb column, now holding NIGHT WAKE-UP windows. This
    # record no longer claims to know whether anyone left the bed.
    out["out_of_bed_events"] = awakenings

    out["sleep_windows"] = [
        {
            "from": _iso_utc(ts[a].at),
            "to": _iso_utc(ts[b].at),
            "duration_minutes": round(_gap(ts, a, b) / 60),
        }
        for a, b in runs
    ]
    out["duration_minutes"] = round(sum(_gap(ts, a, b) for a, b in runs) / 60)
    return out
