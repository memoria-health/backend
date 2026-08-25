"""The veto must survive the SFU changing the frame size under it.

A subscriber does not choose its resolution. The SFU hands it whatever
simulcast rung it judges affordable, and on 2026-08-25 the TV Room
arrived as 2304x1296 on 628 samples, 640x360 on 355 and 320x180 on 511
— inside a single window.

The map used to hold PIXELS. A painting grounded at 2304 wide, compared
against a track box measured at 320 wide, gives containment near zero,
so the veto stops vetoing and the painted faces come back as people.
That is the reported symptom: "churning between identifying those faces
and cancelling them out", then "just identifying them even though there
is nobody in the room".
"""

from __future__ import annotations

import sys
import types

import pytest

for _n in ("cv2",):
    sys.modules.setdefault(_n, types.ModuleType(_n))

import numpy as np

from identity.display_veto import (
    KIND_STATIC, ScreenMap, _REGION_FORMAT, load_regions, save_regions,
)

BIG = (2304, 1296)
SMALL = (320, 180)


def _rows(box, tid=7, frames=3):
    """The same physical box, as the tracker would report it."""
    return {
        i: np.array([[box[0], box[1], box[2], box[3], tid]], dtype=float)
        for i in range(frames)
    }


def _painting_at(w, h):
    """A painting occupying the same physical quarter of the wall."""
    return (0.25 * w, 0.25 * h, 0.45 * w, 0.55 * h)


def test_a_region_grounded_big_still_vetoes_a_small_frame() -> None:
    """Ground at 2304x1296, then have the SFU drop us to 320x180."""
    m = ScreenMap()
    m.add([_painting_at(*BIG)], frame_wh=BIG, kind=KIND_STATIC)

    # The same painted face, now measured in a 320x180 frame.
    dropped = m.veto_track_ids(_rows(_painting_at(*SMALL)), SMALL)
    assert dropped == {7}, (
        "the painting was grounded at one rung and the track measured at "
        "another — in pixels this vetoes nothing")


def test_it_still_vetoes_when_the_size_never_changes() -> None:
    """The ordinary case must not regress."""
    m = ScreenMap()
    m.add([_painting_at(*BIG)], frame_wh=BIG, kind=KIND_STATIC)
    assert m.veto_track_ids(_rows(_painting_at(*BIG)), BIG) == {7}


def test_a_person_elsewhere_is_never_vetoed_at_any_size() -> None:
    """The failure that matters: discarding a real person.

    Someone standing well away from the painting must survive at every
    rung, because a wrong veto is invisible downstream — the room simply
    reports nobody there.
    """
    m = ScreenMap()
    m.add([_painting_at(*BIG)], frame_wh=BIG, kind=KIND_STATIC)
    for w, h in (BIG, SMALL, (640, 360), (1920, 1080)):
        person = (0.70 * w, 0.40 * h, 0.85 * w, 0.95 * h)
        assert m.veto_track_ids(_rows(person), (w, h)) == set(), (
            f"a person was vetoed at {w}x{h}")


def test_movement_is_judged_in_frame_fractions() -> None:
    """A step across the room is a step at every resolution.

    With a pixel threshold, the same physical movement measured in a
    320-wide frame is 7x smaller than in a 2304-wide one — so a moving
    person would be spared on the big rung and discarded on the small.
    """
    m = ScreenMap()
    for w, h in (BIG, SMALL):
        m2 = ScreenMap()
        m2.add([(0.0, 0.0, 1.0 * w, 1.0 * h)], frame_wh=(w, h),
               kind=KIND_STATIC)
        # A box that moves 10% of the frame width across three frames.
        rows = {
            i: np.array(
                [[0.10 * w + i * 0.05 * w, 0.10 * h,
                  0.20 * w + i * 0.05 * w, 0.30 * h, 3]], dtype=float)
            for i in range(3)
        }
        assert m2.veto_track_ids(rows, (w, h)) == set(), (
            f"a clearly moving track was vetoed at {w}x{h}")


def test_a_pixel_era_region_file_is_discarded(tmp_path) -> None:
    """Old files hold pixels. Reading them as fractions vetoes nothing
    and looks identical to a camera with no displays — so they must be
    thrown away and re-grounded, not trusted."""
    path = str(tmp_path / "regions.json")
    import json
    # What the previous format wrote: pixels, no version.
    (tmp_path / "regions.json").write_text(
        json.dumps([{"box": [576.0, 324.0, 1036.0, 712.0], "kind": "static"}]),
        encoding="utf-8")
    assert load_regions(path) == []


def test_a_saved_region_round_trips(tmp_path) -> None:
    path = str(tmp_path / "regions.json")
    save_regions(path, [((0.25, 0.25, 0.45, 0.55), KIND_STATIC)])
    got = load_regions(path)
    assert len(got) == 1
    assert got[0][1] == KIND_STATIC
    assert got[0][0] == pytest.approx((0.25, 0.25, 0.45, 0.55))


def test_regions_are_scaled_by_the_frame_they_were_measured_in() -> None:
    """The denominator must come from the pod, not from intake.

    The regions are measured in `imgs[-1]` of whatever batch the pod
    grounded on. Identity used to normalise them with the size its
    INTAKE reader last saw — a different frame whenever the SFU had
    changed rung in between.

    Observed 2026-08-25 in the TV Room: a painting occupying the right
    edge of frame, grounded on a 640-wide frame, divided by 2304. Every
    one of the 16 regions landed at x 0.17-0.33 — the empty ceiling —
    so the veto had regions and covered nothing, and the painted faces
    kept being tracked as people.
    """
    m = ScreenMap()
    W_GROUND, H_GROUND = 640, 360          # what the pod actually saw
    W_INTAKE, H_INTAKE = 2304, 1296        # what intake last reported

    # A painting on the right edge, in the GROUNDING frame's pixels.
    painting_ground = (0.82 * W_GROUND, 0.20 * H_GROUND,
                       1.00 * W_GROUND, 0.44 * H_GROUND)
    m.add([painting_ground], frame_wh=(W_GROUND, H_GROUND), kind=KIND_STATIC)

    # The same painting, as the tracker reports it in a full-size frame.
    painting_full = (0.82 * W_INTAKE, 0.20 * H_INTAKE,
                     1.00 * W_INTAKE, 0.44 * H_INTAKE)
    assert m.veto_track_ids(_rows(painting_full), (W_INTAKE, H_INTAKE)) == {7}

    # And the wrong denominator must NOT accidentally work.
    wrong = ScreenMap()
    wrong.add([painting_ground], frame_wh=(W_INTAKE, H_INTAKE),
              kind=KIND_STATIC)
    assert wrong.veto_track_ids(_rows(painting_full), (W_INTAKE, H_INTAKE)) == set(), (
        "scaling by the intake size happened to work — the test cannot "
        "distinguish the bug from the fix")
