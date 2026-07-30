"""Unit tests for the sleep state machine v2 (label-based, pure function).

The fixtures mirror the FIRST REAL NIGHT (2026-07-28→29): ticks arrive
every ~10s all night while a person is visible (the room state stays
open), so the machine must read the labels, never tick-silence. The
confirmation-streak rule: start/end are believed only after evidence
persists, and the recorded time is the FIRST tick of the streak.
"""

from datetime import datetime, timedelta

from src.services.detection.activity_summary import StateTick, summarize_day
from src.services.detection.sleep_summary import SleepTick, summarize_night

_N = datetime(2026, 7, 28, 23, 30)         # naive-UTC convention


def _t(mins, in_bed, activity="none", conf=0.9, pc=True):
    return SleepTick(
        at=_N + timedelta(minutes=mins), in_bed=in_bed,
        activity=activity, confidence=conf, patient_confirmed=pc,
    )


def _run(mins_from, mins_to, in_bed, activity="none", step=0.5, pc=True):
    """A run of ticks every `step` minutes — the real 10s-30s cadence."""
    out, m = [], mins_from
    while m < mins_to:
        out.append(_t(m, in_bed, activity, pc=pc))
        m += step
    return out


def test_real_night_shape_continuous_ticks() -> None:
    """Phone-in-bed pre-sleep, settling+stirring, quiet night with a
    disturbance cluster and a bathroom trip, morning awake streak."""
    ticks = (
        _run(0, 60, False, "none")                # 23:30-00:30 phone, awake
        + [_t(60, True, "settling")]              # 00:30 into sleep position
        + _run(60.5, 75, True, "stirring", 2)     # settling churn
        + _run(75, 210, True, "none")             # asleep
        + _run(210, 212, True, "stirring", 0.5)   # 03:00 disturbance cluster
        + _run(212, 240, True, "none")
        + [_t(240, False, "leaving_bed")]         # 03:30 bathroom
        + _run(241, 250, False, "none", 2)
        + [_t(250, True, "settling")]             # 03:40 back
        + _run(250.5, 420, True, "none")
        + _run(420, 440, False, "awake_in_bed", 1)  # 06:30 awake on phone
        + _run(440, 445, False, "leaving_bed", 2)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["bedtime"] == _t(60, True).at
    # First-positive anchor: onset = 00:30 (start of the sustained
    # in-bed streak), even though quiet confirmation lands later.
    assert out["sleep_onset"] == _t(60, True).at
    assert out["disturbances"] == 1               # one cluster, not N ticks
    assert len(out["out_of_bed_events"]) == 1
    assert out["out_of_bed_events"][0]["duration_minutes"] == 10
    assert out["wake_time"] == _t(420, False).at  # first awake tick
    # onset 00:30 -> wake 06:30 = 360m, minus 10 away
    assert out["duration_minutes"] == 350
    assert out["flags"] == []


def test_blind_gap_under_blanket_continues_the_night() -> None:
    """Ticks stop entirely for hours (VLM can't see the sleeper), then
    in-bed quiet resumes: the gap is sleep, not a wake."""
    ticks = (
        [_t(0, True, "settling")]
        + _run(0.5, 30, True, "none")
        + _run(300, 330, True, "none")            # resumes 4.5h later
        + _run(330, 340, False, "awake_in_bed", 1)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["wake_time"] == _t(330, False).at
    assert out["out_of_bed_events"] == []
    assert out["duration_minutes"] == 330
    # The 4.5h blind stretch counts as sleep but must be DISCLOSED —
    # a gap can also be a pipeline outage (learned 2026-07-29).
    assert out["flags"] == ["observation_gap_270m"]


def test_night_ending_in_blind_gap_flags_unobserved_wake() -> None:
    ticks = [_t(0, True, "settling")] + _run(0.5, 60, True, "none")
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["wake_time"] == ticks[-1].at
    assert "wake_unobserved" in out["flags"]


def test_nap_within_resume_window_merges() -> None:
    """Awake 20 min then back asleep: the night continues (Aryaman's
    nap-within-30-min rule); the interlude records as out-of-bed."""
    ticks = (
        [_t(0, True, "settling")] + _run(0.5, 300, True, "none")
        + _run(300, 320, False, "awake_in_bed", 2)   # 20 min awake
        + _run(320, 380, True, "none")               # resumes < 30 min
        + _run(380, 395, False, "leaving_bed", 2)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert len(out["out_of_bed_events"]) == 1
    assert out["out_of_bed_events"][0]["duration_minutes"] == 20
    assert out["wake_time"] == _t(380, False).at
    assert out["duration_minutes"] == 380 - 20


def test_single_stray_awake_tick_does_not_end_the_night() -> None:
    ticks = (
        [_t(0, True, "settling")] + _run(0.5, 200, True, "none")
        + [_t(200, False, "none")]                   # one VLM blip
        + _run(200.5, 300, True, "none")
        + _run(300, 315, False, "leaving_bed", 2)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["wake_time"] == _t(300, False).at
    assert out["flags"] == []


def test_unconfirmed_visitor_never_opens_the_night() -> None:
    ticks = [_t(m, True, "none", pc=False) for m in range(0, 60, 5)]
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["flags"] == ["no_confirmed_bed_entry"]
    assert out["bedtime"] is None


def test_confirmed_start_then_unconfirmed_night_accepted() -> None:
    ticks = (
        [_t(0, True, "settling", pc=True)]
        + _run(0.5, 300, True, "none", pc=False)
        + _run(300, 315, False, "leaving_bed", 2, pc=False)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["wake_time"] is not None
    assert out["flags"] == []


def test_window_ending_mid_sleep_is_incomplete() -> None:
    ticks = [_t(0, True, "settling")] + _run(0.5, 100, True, "none")
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=110))
    assert "incomplete_night" in out["flags"]
    assert out["wake_time"] is None


def test_low_confidence_ticks_ignored() -> None:
    out = summarize_night(
        [_t(0, True, "settling", conf=0.3)],
        window_end=_N + timedelta(minutes=600))
    assert out["flags"] == ["no_bed_entry"]


def test_awake_in_bed_breaks_onset_streak() -> None:
    """Lying in bed on the phone (awake_in_bed) must not count toward
    sleep onset even if the VLM also marks in_bed=true."""
    ticks = (
        _run(0, 30, True, "awake_in_bed", 1)
        + [_t(30, True, "settling")]
        + _run(30.5, 60, True, "none")
        + _run(60, 75, False, "leaving_bed", 2)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=600))
    assert out["sleep_onset"] == _t(30, True).at


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
    assert out["state_minutes"] == {
        "lying": 60.0, "sitting": 60.0, "standing": 0.0, "active": 30.0,
    }
    assert out["active_fraction"] == 0.2
    assert out["room_transitions"] == 1
    assert out["labeled_minutes"] == 150.0
    assert out["label_ticks"] == 3


def test_night_wander_over_an_hour_is_out_of_bed_not_wake() -> None:
    """REGRESSION (deleted in the v2 rewrite, restored 2026-07-29): a
    multi-hour 3am wander is the clinically important event. It must
    record as out-of-bed and the night must continue — never a false
    wake that silently discards the rest of the night."""
    ticks = (
        [_t(0, True, "settling")] + _run(0.5, 200, True, "none")
        + _run(200, 210, False, "leaving_bed", 2)   # 02:50 wander out
        + _run(320, 500, True, "none")              # back 2 hours later
        + _run(500, 515, False, "leaving_bed", 2)   # real wake
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=700))
    assert len(out["out_of_bed_events"]) == 1
    assert out["out_of_bed_events"][0]["duration_minutes"] == 120
    assert out["wake_time"] == _t(500, False).at
    assert out["duration_minutes"] == 500 - 120


def test_single_awake_misread_makes_no_out_of_bed_event() -> None:
    """One stray awake tick mid-night is noise, not a bathroom trip."""
    ticks = (
        [_t(0, True, "settling")] + _run(0.5, 200, True, "none")
        + [_t(200, False, "awake_in_bed")]          # single misread
        + _run(200.5, 400, True, "none")
        + _run(400, 415, False, "leaving_bed", 2)
    )
    out = summarize_night(ticks, window_end=_N + timedelta(minutes=700))
    assert out["out_of_bed_events"] == []
    assert out["wake_time"] == _t(400, False).at
