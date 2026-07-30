"""Nightly sleep summarizer — label-based state machine (v2, 2026-07-29).

v1 assumed ticks were motion-gated, so silence-between-ticks meant
"asleep". The first real night disproved that: with a person visible in
bed, the room state stays open and the bedroom ticks every ~10s ALL
NIGHT — tick silence never happens in bed, so v1 never found an onset.
v2 reads what the ticks actually SAY (the VLM's per-tick labels), not
whether they exist.

Confirmation-streak rule (Aryaman 2026-07-29): sleep-start and
sleep-end are only believed after the evidence PERSISTS — and once a
streak meets its threshold, the recorded timestamp is the FIRST
positive tick of that streak, not the moment confidence was reached.

  bedtime     = first confident in-bed tick (identity-confirmed)
  sleep onset = first in-bed tick of a streak with ONSET_CONFIRM of
                sustained in-bed evidence: no awake/out-of-bed tick in
                the window and >= ONSET_MIN_QUIET quiet ticks
  disturbance = cluster of stirring/sitting_up ticks while asleep
                (clusters split when > DISTURB_GAP apart)
  out-of-bed  = awake/out evidence followed by sleep RESUMING within
                RESUME_WINDOW (bathroom trips, 3am wanders, and the
                nap-within-30-min-counts-as-the-night rule)
  wake        = first tick of an awake streak (>= WAKE_MIN_TICKS within
                WAKE_STREAK_SPAN) after which sleep does NOT resume
                within RESUME_WINDOW. "Awake" is VISIBLE wakefulness:
                sleep_detected false while the person is still seen,
                'awake_in_bed' (eyes open / on the phone / talking),
                or 'leaving_bed'.
  blind gaps  = ticks stopping entirely while asleep (under-blanket:
                the VLM sees no person, ticks skip). Sleep continues
                through a gap; whatever evidence follows the gap
                decides what it was. A night that ENDS in a gap gets
                wake = last in-bed tick + a 'wake_unobserved' flag.

Honesty rules unchanged: a window ending mid-sleep yields
"incomplete_night" (the runner re-runs later); zero qualifying ticks
yields "no_bed_entry"; in-bed ticks that are never identity-confirmed
yield "no_confirmed_bed_entry" (a visitor napping must not open the
patient's night). Pure function — no I/O, no clock reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

ONSET_CONFIRM = timedelta(minutes=15)   # in-bed evidence must persist
ONSET_MIN_QUIET = 3                     # quiet ticks required in the window
WAKE_MIN_TICKS = 2                      # awake ticks to form a streak
WAKE_STREAK_SPAN = timedelta(minutes=10)
RESUME_WINDOW = timedelta(minutes=30)   # sleep resuming within this merges
DISTURB_GAP = timedelta(minutes=5)
MIN_CONFIDENCE = 0.6
WALK_BACK_MAX_GAP = timedelta(minutes=5)     # see the onset walk-back
WALK_BACK_MAX_SPAN = timedelta(minutes=60)

# Tick classes, from (in_bed, activity):
_QUIET, _RESTLESS, _SETTLING, _AWAKE, _OUT = range(5)


@dataclass(frozen=True)
class SleepTick:
    """One labeled tick (from analyses detector_fields)."""

    at: datetime
    in_bed: bool                 # sleep_detected
    activity: str                # settling|stirring|sitting_up|leaving_bed|awake_in_bed|none
    confidence: float            # sleep_confidence
    # Identity gate: the night may only START on a streak containing a
    # tick where identity POSITIVELY confirmed the patient — a visitor
    # napping in the bed must never open the patient's night. Once
    # open, unconfirmed ticks count (a person under covers can't
    # face-confirm; persistence carries identity).
    patient_confirmed: bool = True


def _classify(t: SleepTick) -> int:
    if t.activity == "awake_in_bed":
        return _AWAKE
    if t.activity == "leaving_bed":
        return _OUT
    if t.in_bed:
        if t.activity in ("stirring", "sitting_up"):
            return _RESTLESS
        if t.activity == "settling":
            return _SETTLING
        return _QUIET
    # Not in bed, not leaving: a visible person who is not asleep —
    # pre-bed phone time, sitting on the edge, standing in the room.
    return _AWAKE


def summarize_night(
    ticks: list[SleepTick],
    *,
    window_end: datetime,
) -> dict:
    """Fold one night's ticks (any order) into the nightly record.

    `window_end` bounds how far observation extends — wake and
    incomplete-night judgments are made relative to it.
    """
    ts = sorted(
        (t for t in ticks if t.confidence >= MIN_CONFIDENCE),
        key=lambda t: t.at,
    )
    out: dict = {
        "bedtime": None, "sleep_onset": None, "wake_time": None,
        "duration_minutes": None, "disturbances": 0,
        "out_of_bed_events": [], "flags": [],
    }
    cls = [_classify(t) for t in ts]
    in_bed_idx = [i for i, c in enumerate(cls) if c in (_QUIET, _RESTLESS, _SETTLING)]
    if not in_bed_idx:
        out["flags"].append("no_bed_entry")
        return out
    confirmed_bed = [i for i in in_bed_idx if ts[i].patient_confirmed]
    if not confirmed_bed:
        out["flags"].append("no_confirmed_bed_entry")
        return out
    # Bedtime is set HERE, before the onset scan can bail out: a night
    # where we plainly saw the patient get into bed but never found a
    # sustained sleep run should still report when they went to bed,
    # not a blank row. It is also anchored on the first IDENTITY-
    # CONFIRMED in-bed tick — the first in-bed tick alone could be a
    # visitor sitting on the bed earlier in the evening.
    out["bedtime"] = ts[confirmed_bed[0]].at

    # ── onset: first in-bed tick of a sustained in-bed streak ────────
    onset_i: int | None = None
    for i in in_bed_idx:
        w_end = ts[i].at + ONSET_CONFIRM
        if w_end > window_end:
            break                      # window can't confirm any later start
        quiet = 0
        broken = False
        confirmed = False
        for j in range(i, len(ts)):
            if ts[j].at > w_end:
                break
            if cls[j] in (_AWAKE, _OUT):
                broken = True
                break
            if cls[j] == _QUIET:
                quiet += 1
            if ts[j].patient_confirmed:
                confirmed = True
        if not broken and quiet >= ONSET_MIN_QUIET and confirmed:
            # First-positive anchor (Aryaman): the threshold was met at
            # i's window, but the onset is the FIRST tick of the
            # unbroken in-bed run containing it — walk back through
            # contiguous in-bed evidence (settling/stirring included).
            onset_i = i
            anchor_at = ts[i].at
            while onset_i > 0:
                prev = onset_i - 1
                if cls[prev] not in (_QUIET, _RESTLESS, _SETTLING):
                    break
                # Two bounds, both needed. A big time gap between
                # adjacent ticks is a blind stretch, not evidence of
                # lying there — walking across it invented hours of
                # "sleep" nobody observed. And a hard span cap stops
                # the walk reaching an unrelated earlier bed visit
                # (e.g. a visitor's evening nap) and attributing this
                # night's onset to it.
                if ts[onset_i].at - ts[prev].at > WALK_BACK_MAX_GAP:
                    break
                if anchor_at - ts[prev].at > WALK_BACK_MAX_SPAN:
                    break
                onset_i = prev
            break
    if onset_i is None:
        out["flags"].append(
            "no_sleep_onset" if ts[in_bed_idx[0]].at + ONSET_CONFIRM <= window_end
            else "incomplete_night"
        )
        return out
    out["sleep_onset"] = ts[onset_i].at

    # ── walk the night: disturbances, out-of-bed, wake ──────────────
    wake: datetime | None = None
    wake_unobserved = False
    last_disturb: datetime | None = None
    asleep_yet = False           # settling churn before the first quiet
    i = onset_i + 1              # tick is falling asleep, not a disturbance
    while i < len(ts):
        c = cls[i]
        if c == _QUIET:
            asleep_yet = True
        if c == _RESTLESS:
            if asleep_yet and (
                last_disturb is None or ts[i].at - last_disturb > DISTURB_GAP
            ):
                out["disturbances"] += 1
            if asleep_yet:
                last_disturb = ts[i].at
            i += 1
            continue
        if c in (_AWAKE, _OUT):
            # Candidate sleep-end streak starting here. Confirmation
            # (Aryaman): the evidence must repeat — >= WAKE_MIN_TICKS
            # awake/out ticks within WAKE_STREAK_SPAN — AND sleep must
            # not resume within RESUME_WINDOW. Timestamp = FIRST tick.
            streak = [j for j in range(i, len(ts))
                      if ts[j].at - ts[i].at <= WAKE_STREAK_SPAN
                      and cls[j] in (_AWAKE, _OUT)]
            # Return scan is UNBOUNDED on purpose. A 3am wander that
            # lasts hours is the clinically important event, and
            # capping the scan turns it into a false wake that silently
            # discards the rest of the night (v1 capped at 60min and
            # was fixed for exactly this; the first v2 draft
            # reintroduced the bug at 30min). If sleep resumes at ANY
            # later point, this was an absence, not the end of the
            # night. RESUME_WINDOW is only used below, to decide
            # whether we have watched long enough to call a wake.
            resume = next(
                (j for j in range(i + 1, len(ts))
                 if cls[j] in (_QUIET, _RESTLESS, _SETTLING)),
                None,
            )
            # An absence needs the same confirmation as a wake: one
            # stray misread tick (or a moment of open eyes) must not
            # manufacture an out-of-bed event, and at 10s ticks the
            # unguarded version produced hundreds of 1-minute entries.
            if resume is not None and len(streak) < WAKE_MIN_TICKS:
                i += 1
                continue
            if resume is not None:
                # Sleep resumed: bathroom trip / wander / merged nap.
                out["out_of_bed_events"].append({
                    "at": ts[i].at.isoformat(),
                    "duration_minutes": round(
                        (ts[resume].at - ts[i].at).total_seconds() / 60),
                })
                i = resume + 1
                continue
            if len(streak) < WAKE_MIN_TICKS:
                # A single stray awake tick with nothing after it —
                # not enough evidence to end the night on. Skip it.
                i += 1
                continue
            if window_end - ts[i].at < RESUME_WINDOW:
                # Can't yet rule out a resume — night still in progress.
                out["flags"].append("incomplete_night")
                return out
            wake = ts[i].at
            break
        i += 1                        # QUIET / SETTLING — sleep continues

    if wake is None:
        # Ticks ran out while asleep. If the window has moved well past
        # the last tick, the person left the camera's view without an
        # observed wake (or the VLM misread the morning): close the
        # night at the last in-bed evidence and say so.
        last_bed = ts[max(j for j in in_bed_idx)]
        if window_end - last_bed.at >= RESUME_WINDOW:
            wake = last_bed.at
            wake_unobserved = True
        else:
            out["flags"].append("incomplete_night")
            return out

    out["wake_time"] = wake
    if wake_unobserved:
        out["flags"].append("wake_unobserved")
    # Observation-gap honesty (2026-07-29): ticks stopping can mean
    # under-blanket invisibility OR a pipeline/network outage (the
    # first real night hid a 107-min full outage inside "quiet"). The
    # night still folds — gaps count as sleep — but any substantial
    # blind time is surfaced so nobody mistakes it for observation.
    gap_min = 0
    prior = out["sleep_onset"]
    for t in ts:
        if t.at <= prior:
            continue
        if wake is not None and t.at > wake:
            break
        d = (t.at - prior).total_seconds() / 60
        if d >= 10:
            gap_min += round(d)
        prior = t.at
    if gap_min >= 30:
        out["flags"].append(f"observation_gap_{gap_min}m")
    away = sum(e["duration_minutes"] for e in out["out_of_bed_events"])
    out["duration_minutes"] = max(
        0, round((wake - out["sleep_onset"]).total_seconds() / 60) - away)
    return out
