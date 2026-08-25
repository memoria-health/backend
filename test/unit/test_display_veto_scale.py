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
    ScreenMap, _REGION_FORMAT, load_regions, save_regions,
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
    m.add([_painting_at(*BIG)], frame_wh=BIG)

    # The same painted face, now measured in a 320x180 frame.
    dropped = m.veto_track_ids(_rows(_painting_at(*SMALL)), SMALL)
    assert dropped == {7}, (
        "the painting was grounded at one rung and the track measured at "
        "another — in pixels this vetoes nothing")


def test_it_still_vetoes_when_the_size_never_changes() -> None:
    """The ordinary case must not regress."""
    m = ScreenMap()
    m.add([_painting_at(*BIG)], frame_wh=BIG)
    assert m.veto_track_ids(_rows(_painting_at(*BIG)), BIG) == {7}


def test_a_person_elsewhere_is_never_vetoed_at_any_size() -> None:
    """The failure that matters: discarding a real person.

    Someone standing well away from the painting must survive at every
    rung, because a wrong veto is invisible downstream — the room simply
    reports nobody there.
    """
    m = ScreenMap()
    m.add([_painting_at(*BIG)], frame_wh=BIG)
    for w, h in (BIG, SMALL, (640, 360), (1920, 1080)):
        person = (0.70 * w, 0.40 * h, 0.85 * w, 0.95 * h)
        assert m.veto_track_ids(_rows(person), (w, h)) == set(), (
            f"a person was vetoed at {w}x{h}")


def test_the_same_wall_vetoes_at_every_rung() -> None:
    """One region, four resolutions, same verdict.

    This replaces a test of the movement exemption, which is gone: it
    protected nothing a 0.93 containment floor did not already protect,
    and it was what spared the paintings. What still matters is that the
    verdict does not depend on which rung the SFU happens to be sending.
    """
    m = ScreenMap()
    m.add([_painting_at(*BIG)], frame_wh=BIG)
    for w, h in (BIG, SMALL, (640, 360), (1920, 1080), (1280, 720)):
        assert m.veto_track_ids(_rows(_painting_at(w, h)), (w, h)) == {7}, (
            f"the same painting escaped the veto at {w}x{h}")

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
    save_regions(path, [(0.25, 0.25, 0.45, 0.55)])
    got = load_regions(path)
    assert len(got) == 1
    assert got[0] == pytest.approx((0.25, 0.25, 0.45, 0.55))


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
    m.add([painting_ground], frame_wh=(W_GROUND, H_GROUND))

    # The same painting, as the tracker reports it in a full-size frame.
    painting_full = (0.82 * W_INTAKE, 0.20 * H_INTAKE,
                     1.00 * W_INTAKE, 0.44 * H_INTAKE)
    assert m.veto_track_ids(_rows(painting_full), (W_INTAKE, H_INTAKE)) == {7}

    # And the wrong denominator must NOT accidentally work.
    wrong = ScreenMap()
    wrong.add([painting_ground], frame_wh=(W_INTAKE, H_INTAKE))
    assert wrong.veto_track_ids(_rows(painting_full), (W_INTAKE, H_INTAKE)) == set(), (
        "scaling by the intake size happened to work — the test cannot "
        "distinguish the bug from the fix")


# ── the tiled artwork ────────────────────────────────────────────────

_CANVAS_PANELS = [
    (0.819, 0.208, 0.864, 0.314), (0.820, 0.312, 0.866, 0.416),
    (0.861, 0.203, 0.906, 0.310), (0.864, 0.308, 0.910, 0.421),
    (0.900, 0.201, 0.954, 0.309), (0.907, 0.309, 0.960, 0.429),
    (0.948, 0.200, 1.000, 0.310),
]
_FRAMED_PICTURE = (0.500, 0.231, 0.528, 0.348)   # separate object, far away
_MISSED_PANEL = (0.96, 0.33, 1.00, 0.44)         # grounding never found it


def test_the_panel_grounding_missed_is_still_covered() -> None:
    """The real failure, with the real boxes.

    Concept grounding returned the TV Room's canvas as seven panels
    across two rows and missed the eighth — the bottom-right face. It
    fell in the gap between the last panel and the frame edge and was
    tracked as a person for hours. Joining panels that touch turns them
    into one canvas, which covers the miss.
    """
    from identity.display_veto import cluster_regions

    clustered = cluster_regions(_CANVAS_PANELS)
    assert len(clustered) == 1, f"the canvas did not become one object: {clustered}"
    c = clustered[0]
    assert c[0] <= _MISSED_PANEL[0] and c[2] >= _MISSED_PANEL[2]
    assert c[1] <= _MISSED_PANEL[1] and c[3] >= _MISSED_PANEL[3]


def test_a_separate_object_is_not_swallowed() -> None:
    """Clustering must not blanket the room.

    The small framed picture by the door is a real, separate region. If
    everything merged, the resulting box would span half the frame and
    could contain a person.
    """
    from identity.display_veto import cluster_regions

    clustered = cluster_regions(_CANVAS_PANELS + [_FRAMED_PICTURE])
    assert len(clustered) == 2, f"expected canvas + picture, got {clustered}"
    assert min(b[2] - b[0] for b in clustered) < 0.1


def test_a_person_cannot_be_vetoed_by_the_clustered_canvas() -> None:
    """The clustered region is person-sized, so this has to be checked.

    It is safe for a reason worth stating: the art hangs high on the
    wall, and a standing or seated person's box always extends well
    below it. Measured against the floor of 0.93, the worst case here is
    0.25.
    """
    m = ScreenMap()
    for panel in _CANVAS_PANELS:
        m.add([panel])
    for who, person in (
        ("standing in front of the wall", (0.86, 0.30, 0.95, 0.90)),
        ("seated under the art", (0.86, 0.40, 0.95, 0.70)),
        ("across the room", (724 / 2304, 518 / 1296, 1156 / 2304, 778 / 1296)),
    ):
        rows = {i: np.array([[*person, 11]]) for i in range(3)}
        assert m.veto_track_ids(rows, now=1.0) == set(), f"vetoed a person {who}"


def test_a_person_in_front_of_a_low_hung_screen_is_not_vetoed() -> None:
    """The case that does NOT depend on how this living room is arranged.

    A containment floor is safe only while a room's art hangs higher
    than a person's box reaches. That is true of the TV Room and is not
    a property of the system — a large television at eye level would
    contain someone standing in front of it, and vetoing them is silent:
    nothing downstream ever learns they were there.

    What generalises is that a person WALKED IN. Wherever they settle is
    far from where their track was born; a face painted on the screen is
    born where it dies.
    """
    m = ScreenMap()
    # A big screen occupying the middle of the wall, person height.
    m.add([(0.30, 0.25, 0.75, 0.85)])
    person = (0.40, 0.30, 0.60, 0.80)          # squarely inside it
    rows = {i: np.array([[*person, 4]]) for i in range(3)}

    # Without the history it is contained, and would be vetoed.
    assert m.veto_track_ids(rows, now=1.0) == {4}
    # Knowing the track walked here, it never is.
    assert m.veto_track_ids(rows, now=1.0, moved={4}) == set()


def test_a_painted_face_that_never_moved_is_still_vetoed() -> None:
    """The exemption must not become a hole.

    Only tracks that actually travelled are spared. A phantom is born
    where it dies, so it is never in `moved` and the veto still applies.
    """
    m = ScreenMap()
    m.add([(0.30, 0.25, 0.75, 0.85)])
    rows = {i: np.array([[0.40, 0.30, 0.60, 0.80, 9]]) for i in range(3)}
    assert m.veto_track_ids(rows, now=1.0, moved=set()) == {9}
