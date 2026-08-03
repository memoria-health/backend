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

## Identity: asked once, at the crossing

A visitor asleep in the bed must not become the patient's night, so the
first `entering_bed` must be identity-confirmed. After that it is never
consulted again — identity needs a visible upright person and a sleeper
is neither (08-02: 110 confirmed ticks while awake, then zero across
three hours of solid sleep).

Pure function — no I/O, no clock reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

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
# Below this the tick is too uncertain to use at all.
MIN_CONFIDENCE = 0.6
# Consecutive in-bed ticks that re-open occupancy when the return
# crossing was missed. Small: a single tick could be a misread, but the
# 08-02 return produced hundreds in a row.
REENTRY_MIN_TICKS = 3

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
        if t.activity == _ENTER and (opened_once or t.patient_confirmed):
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
        ok = _in_bed(t) and (opened_once or t.patient_confirmed)
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
        prev = runs[-1][1]
        # Every index strictly between two asleep ticks is an OBSERVED
        # awake tick (invisible ones never reach this list).
        first_awake, last_awake = prev + 1, i - 1
        if last_awake < first_awake:
            runs[-1][1] = i                          # contiguous sleep
            continue
        if _gap(ts, first_awake, last_awake) < WAKE_MIN_SECONDS:
            runs[-1][1] = i                          # flicker / too brief
            continue
        runs.append([i, i])
    return [
        (a, b) for a, b in runs
        if _gap(ts, a, b) >= ONSET_MIN_SECONDS
    ]


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

    runs = [r for lo, hi in spans for r in _sleep_runs(ts, lo, hi)]
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
            "from": ts[prev[1]].at.isoformat(),
            "to": ts[nxt[0]].at.isoformat(),
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
            "from": ts[a].at.isoformat(),
            "to": ts[b].at.isoformat(),
            "duration_minutes": round(_gap(ts, a, b) / 60),
        }
        for a, b in runs
    ]
    out["duration_minutes"] = round(sum(_gap(ts, a, b) for a, b in runs) / 60)
    return out
