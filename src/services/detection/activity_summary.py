"""Daily activity summarizer v2 — VLM state timeline (Aryaman 2026-07-28).

v1 counted raw motion ticks, which mislabels hand-fidgeting-while-seated
as activity. v2 consumes the Activity detector's per-tick STATE labels
(lying / sitting / standing / active — seated exercise IS active) and
applies the same persistence principle as sleep: a label holds until the
next label, because a state change is physically a motion event and
motion fires a tick. The daily record is therefore TIME-WEIGHTED state
minutes, not tick counts — strictly more accurate than any motion proxy.

Coverage honesty: time before the first label of the day is UNOBSERVED
(not attributed to any state), and `labeled_minutes` rides the record so
a low-observation day can never masquerade as a low-activity day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

MIN_CONFIDENCE = 0.6
# `sitting` split into still/moving on 2026-08-07. The classifier reads
# POSTURE reliably and judges PURPOSE badly — measured live, continuous
# overhead arm raises returned `sitting` at confidence 1.0 while `seen`
# correctly read "arms are raised above their head for the duration".
# So the detector is now asked only what it gets right ("is the body
# moving?") and the purpose question is retired.
#
# `sitting` is KEPT as a legacy value: every tick recorded before this
# change carries it, and dropping it would silently delete those
# minutes from history rather than degrade them. It folds in with
# `sitting_still`, which is the honest reading — the old label meant
# "seated and not judged to be exercising".
STATES = (
    "lying", "sitting", "sitting_still", "sitting_moving",
    "standing", "active",
)
LEGACY_SITTING = "sitting"

# Sedentary behaviour is defined by POSTURE plus low energy expenditure
# — sitting, reclining or lying — not by what the person intends. So the
# split below is a posture split, and it is deliberately NOT the same
# question as `active_fraction`.
#
# This matters because the intent judgement is the unreliable half.
# Measured live 2026-08-07: continuous overhead arm raises came back
# `sitting` at confidence 1.0, with `seen` correctly reading "arms are
# raised above their head for the duration". The model reads POSTURE
# reliably and judges PURPOSE badly. Every measure derived here uses
# only the half it gets right:
#
#   - upright_minutes      standing + active — a posture reading
#   - sedentary_minutes    lying + sitting
#   - sedentary_breaks     sedentary -> upright transitions
#   - longest_sedentary_bout_minutes
#
# `sedentary_breaks` and the longest bout are not garnish. In older
# adults, how often sedentary time is BROKEN predicts outcomes
# independently of how much sedentary time there is: the same ten hours
# seated is a different picture in one block versus twenty. That makes
# them the measures to lead a customer-facing view with, because they
# survive the classifier's weakest judgement.
#
# `sitting_moving` is in NEITHER set, on purpose. It is not sedentary
# (the body is working) and it is not upright (they are in a chair), and
# collapsing it into either one throws away the distinction the split
# was made to capture. For a frail patient it may be the whole of their
# exercise; for a healthy one it is mostly gesturing. Which way it
# counts is a product decision per cohort, so it rides as its own band
# and the caller chooses.
SEDENTARY_STATES = frozenset({"lying", "sitting", "sitting_still"})
UPRIGHT_STATES = frozenset({"standing", "active"})


@dataclass(frozen=True)
class StateTick:
    """One labeled tick from the Activity detector."""

    at: datetime
    camera: str
    state: str                   # activity_state
    confidence: float            # activity_confidence


def summarize_day(
    ticks: list[StateTick],
    *,
    window_end: datetime,
) -> dict:
    """Fold one day's state labels into time-weighted state minutes.

    Each label persists until the next label (or `window_end` for the
    last one — bounded by how far observation actually extends).
    """
    ts = sorted(
        (t for t in ticks
         if t.confidence >= MIN_CONFIDENCE and t.state in STATES),
        key=lambda t: t.at,
    )
    minutes = {s: 0.0 for s in STATES}
    transitions = 0
    last_cam: str | None = None
    # Sedentary bouts: a run of consecutive sedentary labels. `breaks`
    # counts sedentary -> upright crossings, so a day that starts
    # upright contributes no break, and consecutive sedentary labels
    # extend the current bout rather than opening a new one.
    sedentary_breaks = 0
    longest_bout = 0.0
    current_bout = 0.0
    prev_sedentary: bool | None = None
    for i, t in enumerate(ts):
        until = ts[i + 1].at if i + 1 < len(ts) else window_end
        span = max(0.0, (until - t.at).total_seconds() / 60)
        minutes[t.state] += span
        if last_cam is not None and t.camera != last_cam:
            transitions += 1
        last_cam = t.camera

        is_sedentary = t.state in SEDENTARY_STATES
        if is_sedentary:
            current_bout += span
        else:
            # A BREAK requires getting UPRIGHT. `sitting_moving` ends the
            # bout — the body is working, so the sedentary run is over —
            # but it is not a break, because a break in sedentary time
            # means interrupting the POSTURE, and they are still in the
            # chair. Counting chair exercise as a sit-to-stand would
            # inflate the one measure that is supposed to be the honest
            # one. Flagged as a product call rather than a silent
            # default: if the customer view ever wants "interruptions of
            # any kind", that is a second counter, not a redefinition of
            # this one.
            if prev_sedentary and t.state in UPRIGHT_STATES:
                sedentary_breaks += 1
            longest_bout = max(longest_bout, current_bout)
            current_bout = 0.0
        prev_sedentary = is_sedentary
    longest_bout = max(longest_bout, current_bout)   # day ended seated

    labeled = round(sum(minutes.values()), 1)
    sedentary = sum(minutes[s] for s in SEDENTARY_STATES if s in minutes)
    upright = sum(minutes[s] for s in UPRIGHT_STATES if s in minutes)
    return {
        "state_minutes": {s: round(m, 1) for s, m in minutes.items()},
        "active_fraction": (
            round(minutes["active"] / labeled, 3) if labeled else None
        ),
        # Posture-derived, so independent of the intent judgement the
        # classifier gets wrong (see the note above SEDENTARY_STATES).
        "sedentary_minutes": round(sedentary, 1),
        "upright_minutes": round(upright, 1),
        "upright_fraction": (
            round(upright / labeled, 3) if labeled else None
        ),
        # Its own band — counted as neither sedentary nor upright.
        "sitting_moving_minutes": round(minutes.get("sitting_moving", 0.0), 1),
        "sedentary_breaks": sedentary_breaks,
        "longest_sedentary_bout_minutes": round(longest_bout, 1),
        "room_transitions": transitions,
        "labeled_minutes": labeled,      # observation coverage, not a score
        "label_ticks": len(ts),
    }
