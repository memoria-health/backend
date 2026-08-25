"""Every track death says why, in one line.

Three theories about the churn were argued on 2026-08-25 and two were
wrong — the identity layer re-assert and the display filter — because
nothing recorded the death itself. Every argument was inference from
second-order symptoms: seat lifetimes, event counts, commit timings.

The fields exist to rule things IN or OUT rather than to be pretty:

    gap_s        sub-second is a detector blink; multi-second is a
                 person who left
    vetoed       we threw it away vs the detector lost it
    rows a->b    the detector returned nothing vs we discarded it
    reborn_iou   a renumbering vs a genuine disappearance
    seat         whether this death cost a face match
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

for _n in ("cv2", "numpy", "torch"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

from identity.state import TrackEvents


class _Loop:
    """Just enough of CameraLoop to exercise the report."""

    camera_id = "cam1"

    def __init__(self, tracks, seats=None):
        self.state = types.SimpleNamespace(tracks=tracks)
        self.home = types.SimpleNamespace(seats=seats or {})

    from identity.loop import CameraLoop as _C
    _report_deaths = _C._report_deaths


def _tv(box):
    return types.SimpleNamespace(bbox=box)


def test_a_blink_and_a_departure_are_distinguishable(caplog) -> None:
    """The whole point: two deaths that look identical in the old log."""
    caplog.set_level(logging.INFO)
    loop = _Loop(tracks={})
    prev = {
        1: ((0.0, 0.0, 10.0, 20.0), 99.8, 90.0, 40),   # gone 0.2s
        2: ((50.0, 0.0, 60.0, 20.0), 92.0, 80.0, 30),  # gone 8s
    }
    loop._report_deaths(
        TrackEvents(births=[], deaths=[1, 2]), prev,
        vetoed=set(), rows_before=3, rows_after=3, now=100.0)
    text = caplog.text
    assert "tid=1" in text and "gap_s=0.20" in text
    assert "tid=2" in text and "gap_s=8.00" in text


def test_a_vetoed_death_is_marked_as_ours(caplog) -> None:
    """'We threw it away' must never look like 'the detector lost it'."""
    caplog.set_level(logging.INFO)
    loop = _Loop(tracks={})
    loop._report_deaths(
        TrackEvents(births=[], deaths=[7]),
        {7: ((0.0, 0.0, 10.0, 20.0), 99.0, 50.0, 100)},
        vetoed={7}, rows_before=4, rows_after=0, now=100.0)
    assert "vetoed=True" in caplog.text
    assert "rows=4->0" in caplog.text


def test_a_renumbering_shows_up_as_overlap(caplog) -> None:
    """A track born in the same batch, in the same place, is the same
    person — and that is the seat-carry case."""
    caplog.set_level(logging.INFO)
    box = (100.0, 100.0, 200.0, 300.0)
    loop = _Loop(tracks={23: _tv((101.0, 101.0, 199.0, 299.0))})
    loop._report_deaths(
        TrackEvents(births=[23], deaths=[22]),
        {22: (box, 99.9, 40.0, 200)},
        vetoed=set(), rows_before=2, rows_after=2, now=100.0)
    assert "reborn_iou=0.9" in caplog.text


def test_a_death_with_no_replacement_reads_zero(caplog) -> None:
    caplog.set_level(logging.INFO)
    loop = _Loop(tracks={})
    loop._report_deaths(
        TrackEvents(births=[], deaths=[22]),
        {22: ((100.0, 100.0, 200.0, 300.0), 99.0, 40.0, 200)},
        vetoed=set(), rows_before=0, rows_after=0, now=100.0)
    assert "reborn_iou=0.00" in caplog.text
    assert "rows=0->0" in caplog.text


def test_a_death_that_cost_an_identity_is_flagged(caplog) -> None:
    """The deaths that matter are the ones holding a seat."""
    caplog.set_level(logging.INFO)
    seat = types.SimpleNamespace(track_id=22)
    loop = _Loop(tracks={}, seats={"p1": seat})
    loop._report_deaths(
        TrackEvents(births=[], deaths=[22]),
        {22: ((0.0, 0.0, 10.0, 20.0), 99.0, 40.0, 200)},
        vetoed=set(), rows_before=1, rows_after=1, now=100.0)
    assert "seat=True" in caplog.text


def test_the_report_can_never_end_a_tick(caplog) -> None:
    """Instrumentation is not a dependency."""
    caplog.set_level(logging.INFO)
    loop = _Loop(tracks={})
    loop.home = None          # force an AttributeError inside
    loop._report_deaths(
        TrackEvents(births=[], deaths=[1]),
        {1: ((0, 0, 1, 1), 99.0, 40.0, 2)},
        vetoed=set(), rows_before=1, rows_after=1, now=100.0)
    # No exception escaped; that is the assertion.
