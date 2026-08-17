"""Nightly sleep record — three outputs, driven by bed crossings.

Every sleep test below encodes something that actually went wrong on the
2026-08-02 night, found by tracing the stored ticks. Ground truth for
that night, from the resident: in bed ~01:45, brief trip out at 01:48,
asleep ~02:50-03:00, slept straight through, woke 07:20-07:45, did not
get out of bed until ~08:12.
"""

from datetime import datetime, timedelta

from src.services.detection.activity_summary import StateTick, summarize_day
from src.services.detection.sleep_summary import (
    ONSET_MIN_SECONDS,
    SleepTick,
    summarize_night,
)

_N = datetime(2026, 8, 2, 23, 0)           # naive-UTC convention
_END = _N + timedelta(hours=14)


def _t(mins, asleep, activity="none", pc=True, conf=0.9) -> SleepTick:
    return SleepTick(
        at=_N + timedelta(minutes=mins), asleep=asleep,
        activity=activity, confidence=conf, patient_confirmed=pc,
    )


def _run(a, b, asleep, activity="none", step=0.5, pc=True) -> list[SleepTick]:
    """A run of ticks every `step` minutes — the real 10-30s cadence."""
    out, m = [], a
    while m < b:
        out.append(_t(m, asleep, activity, pc=pc))
        m += step
    return out


def _enter(at=0.0, pc=True):
    return [_t(at, False, "entering_bed", pc=pc)]


def _leave(at):
    return [_t(at, False, "leaving_bed")]


# ── the three outputs ────────────────────────────────────────────────

def test_reports_onset_wake_and_no_awakenings_on_a_clean_night() -> None:
    ticks = _enter() + _run(1, 300, True) + _leave(300)
    out = summarize_night(ticks, window_end=_END)
    assert out["flags"] == []
    assert out["sleep_onset"] == _N + timedelta(minutes=1)
    assert out["wake_time"] == _N + timedelta(minutes=299.5)
    assert out["disturbances"] == 0
    assert out["awakenings"] == []


def test_a_real_awakening_is_reported_as_a_window() -> None:
    ticks = (
        _enter()
        + _run(1, 120, True)
        + _run(120, 135, False, "awake_in_bed")      # 15 min awake, real
        + _run(135, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 1
    w = out["awakenings"][0]
    # The window is bounded by what we OBSERVED: last tick seen asleep ->
    # first tick seen asleep again. We cannot know the exact instant the
    # eyes opened, so the window is deliberately the outer bound rather
    # than the span of awake ticks.
    assert w["from"] == (_N + timedelta(minutes=119.5)).isoformat()
    assert w["to"] == (_N + timedelta(minutes=135)).isoformat()
    assert w["duration_minutes"] == 16          # 15.5 min, rounded


# ── bug 1: silence is not wakefulness (the 06:26 case) ───────────────

def test_an_unobserved_gap_is_not_an_awakening() -> None:
    """A sleeper slips under the duvet, `people_present` goes false, those
    ticks are dropped upstream — leaving a hole with NO ticks. Timing that
    hole with a clock reported a 26-minute awakening on a night whose log
    reads 'lying in bed under covers' on both sides of it."""
    ticks = (
        _enter()
        + _run(1, 120, True)
        # 40 minutes with no ticks at all — nobody visible
        + _run(160, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 0, "unobserved time must never become a wake-up"
    assert out["awakenings"] == []
    assert out["sleep_onset"] == _N + timedelta(minutes=1)


# ── bug 2: a single flicker tick is not an awakening (05:15) ─────────

def test_single_awake_tick_is_absorbed() -> None:
    """Five isolated not-asleep ticks appeared across 5.5 hours on the
    08-02 night, against two genuine awakenings of 5 and 11 minutes."""
    ticks = (
        _enter()
        + _run(1, 120, True)
        + [_t(120, False, "awake_in_bed")]           # one tick
        + _run(120.5, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 0
    assert out["sleep_onset"] == _N + timedelta(minutes=1)


# ── bug 4: an awakening needs EVIDENCE, not just elapsed time (08-06) ─
#
# Ground truth for the 2026-08-06 night: asleep ~01:56, woke ~07:30 and
# used a phone in bed for ~8 minutes, dozed again, got out of bed 07:55.
# The model saw all of it — 30 consecutive ticks of "looking at a glowing
# smartphone", plus a bedside lamp being switched on. The summarizer threw
# every one of them away because 8 minutes < WAKE_MIN_SECONDS, and instead
# reported a 0-minute awakening invented by one hallucinated crossing.

def test_a_short_awakening_with_visible_evidence_is_reported() -> None:
    """8 minutes of `awake_in_bed` is under the 10-minute wake criterion,
    but the criterion exists to absorb AMBIGUOUS movement. A person the
    model can see is on a device is not ambiguous, and no amount of
    waiting makes it more certain."""
    ticks = (
        _enter()
        + _run(1, 120, True)
        + _run(120, 128, False, "awake_in_bed")      # 8 min < WAKE_MIN_SECONDS
        + _run(128, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 1
    assert out["awakenings"][0]["duration_minutes"] == 8


def test_a_lone_hallucinated_leaving_bed_does_not_fabricate_an_awakening() -> None:
    """05:49:27 on the 08-06 night: one tick claimed "got out of bed and
    abruptly collapsed to the floor" while he lay asleep under the covers,
    with normal sleep ticks either side. Because a crossing CLOSES the
    occupancy span, the runs either side became two runs and the flicker
    rule — which lives inside a span — never saw the break at all."""
    ticks = (
        _enter()
        + _run(1, 120, True)
        + _leave(120)                                # single false crossing
        + _run(120.5, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 0
    assert out["duration_minutes"] > 290


def test_two_evidenced_awake_ticks_are_still_absorbed() -> None:
    """The evidence rule must not make a two-tick misread into a wake-up;
    a hallucinated tick is exactly what it has to survive."""
    ticks = (
        _enter()
        + _run(1, 120, True)
        + _run(120, 121, False, "awake_in_bed")      # 2 ticks
        + _run(121, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 0


def test_unevidenced_awake_ticks_still_need_the_full_duration() -> None:
    """`asleep=False, activity=none` is an ABSENCE of a sleep call, not a
    sighting of wakefulness. It keeps the conservative 10-minute rule."""
    ticks = (
        _enter()
        + _run(1, 120, True)
        + _run(120, 128, False, "none")              # 8 min, no evidence
        + _run(128, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 0


# ── bug 3: a missed re-entry crossing must not kill the night ────────

def test_night_survives_a_missed_reentry_crossing() -> None:
    """The 01:48 defect. He got out of bed and back in; only the EXIT was
    emitted. A crossings-only rule discarded the following five hours."""
    ticks = (
        _enter()
        + _run(1, 3, False, "awake_in_bed")
        + _leave(3)
        + _run(4, 6, False, "none")                  # out of the room
        # back in bed — NO entering_bed event was produced
        + _run(6, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["flags"] == []
    assert out["sleep_onset"] is not None
    assert out["duration_minutes"] > 280


# ── occupancy: only leaving_bed ends it ──────────────────────────────

def test_awake_in_bed_never_ends_occupancy() -> None:
    """`awake_in_bed` was classified as out-of-room and produced nine
    phantom exits on a night with zero `leaving_bed` ticks."""
    ticks = (
        _enter()
        + _run(1, 100, True)
        + _run(100, 130, False, "awake_in_bed")      # awake, still in bed
        + _run(130, 300, True)
        + _leave(300)
    )
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 1                  # a wake-up, not an exit
    assert out["sleep_onset"] == _N + timedelta(minutes=1)


def test_leaving_bed_ends_the_last_sleep_run() -> None:
    ticks = _enter() + _run(1, 200, True) + _leave(200) + _run(201, 260, False)
    out = summarize_night(ticks, window_end=_END)
    assert out["wake_time"] <= _N + timedelta(minutes=200)


# ── identity: once, at the gate ──────────────────────────────────────

def test_unconfirmed_entry_cannot_open_the_night() -> None:
    """A visitor asleep in the bed must not become the patient's night."""
    ticks = _enter(pc=False) + _run(1, 300, True, pc=False)
    out = summarize_night(ticks, window_end=_END)
    assert out["flags"] == ["no_confirmed_bed_entry"]
    assert out["sleep_onset"] is None


def test_identity_is_not_rechecked_after_the_gate() -> None:
    """Identity needs a visible upright person; a sleeper is neither. On
    08-02: 110 confirmed ticks while awake, then zero across three hours
    of solid sleep."""
    ticks = _enter(pc=True) + _run(1, 300, True, pc=False) + _leave(300)
    out = summarize_night(ticks, window_end=_END)
    assert out["flags"] == []
    assert out["sleep_onset"] is not None


# ── edges ────────────────────────────────────────────────────────────

def test_still_asleep_at_window_end_reports_no_wake_time() -> None:
    ticks = _enter() + _run(1, 300, True)
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=299))
    assert out["wake_time"] is None
    assert "wake_unobserved" in out["flags"]


def test_dozing_shorter_than_the_onset_threshold_is_not_sleep() -> None:
    brief = ONSET_MIN_SECONDS / 60 - 1
    ticks = _enter() + _run(1, 1 + brief, True) + _leave(30)
    out = summarize_night(ticks, window_end=_END)
    assert out["sleep_onset"] is None
    assert "no_sleep_onset" in out["flags"]


def test_low_confidence_ticks_are_ignored() -> None:
    ticks = _enter() + _run(1, 300, True) + _leave(300)
    ticks += [_t(150, False, "leaving_bed", conf=0.1)]    # junk exit
    out = summarize_night(ticks, window_end=_END)
    assert out["disturbances"] == 0
    assert out["duration_minutes"] > 280


def test_no_ticks_at_all() -> None:
    out = summarize_night([], window_end=_END)
    assert out["flags"] == ["no_ticks"]


# ── activity summarizer (unchanged) ──────────────────────────────────

def test_activity_state_timeline_time_weighted() -> None:
    base = datetime(2026, 7, 28, 9, 0)

    def s(mins, cam, state, conf=0.9):
        return StateTick(base + timedelta(minutes=mins), cam, state, conf)

    ticks = [
        s(0, "LR", "sitting"),
        s(60, "LR", "active"),
        s(90, "BR", "lying"),
        s(95, "BR", "junk", 0.9),
        s(120, "LR", "sitting", 0.2),
    ]
    out = summarize_day(ticks, window_end=base + timedelta(minutes=150))
    # `sitting` is the LEGACY label, kept alongside the 2026-08-07
    # still/moving split so historical ticks keep their minutes rather
    # than silently vanishing from the record.
    assert out["state_minutes"] == {
        "lying": 60.0, "sitting": 60.0, "sitting_still": 0.0,
        "sitting_moving": 0.0, "standing": 0.0, "active": 30.0,
    }
    assert out["active_fraction"] == 0.2
    assert out["room_transitions"] == 1
    assert out["labeled_minutes"] == 150.0
    assert out["label_ticks"] == 3
    # Posture-derived measures: legacy `sitting` counts as sedentary.
    assert out["sedentary_minutes"] == 120.0
    assert out["upright_minutes"] == 30.0
