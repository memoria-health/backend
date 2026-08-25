"""One blink is not an absence — the carry-tick version.

Identity has had `MISSES_BEFORE_DEATH` for a while: a track missing from
one batch is a detector blink, not a departure. Perception had the same
rule only for the ALL-EMPTY tick. A tick that kept 2 of 3 masks dropped
the third with no grace whatever.

That gap is what renumbers people. A track missing from `last_masks`
cannot be matched by `_remap_masks` at the next ground, so it returns as
a NEW ID — and the seat, holding a face match that took 20-60s to earn,
was keyed on the old number.

Measured on the node 2026-08-25: EdgeTAM lost a seeded track on 2% of
ticks on BOTH cameras (46 Bedroom, 77 TV Room). It is the only churn
source common to both — the painted faces and the resolution flips are
TV-Room-only and cannot explain the Bedroom at all.
"""

from __future__ import annotations

from perception.continuous import carry_grace

A, B = "mask-a", "mask-b"


def test_a_track_that_propagated_is_untouched():
    out, misses, held = carry_grace({1: A}, {1: B}, {})
    assert out == {1: B}          # the FRESH mask, never the stale one
    assert misses == {}
    assert held == []


def test_one_blink_holds_the_id():
    """The case the old code dropped: 2 of 3 survived."""
    out, misses, held = carry_grace({1: A, 2: A, 3: A}, {1: B, 2: B}, {})
    assert out[3] == A            # held at its last known mask
    assert misses == {3: 1}
    assert held == [(3, 1)]


def test_a_second_blink_lets_it_go():
    """Grace is 2, so the second consecutive miss is a real absence.
    Without this a departed person's mask would be immortal."""
    out, misses, held = carry_grace({1: A}, {}, {1: 1})
    assert 1 not in out
    assert misses == {}
    assert held == []


def test_coming_back_clears_the_debt():
    """A track that flickers every other tick must not accumulate its
    way to a death — the same reasoning as state.apply resetting
    `misses` on every sighting."""
    _out, misses, _held = carry_grace({1: A}, {1: B}, {1: 1})
    assert misses == {}


def test_a_track_that_was_never_seeded_is_not_invented():
    """Grace can only ever preserve something that already existed."""
    out, _misses, _held = carry_grace({}, {9: B}, {})
    assert out == {9: B}


# ── The paintings must not be COUNTED as people ────────────────────────
#
# Identity sends the pod the ids it has judged to be a display or a
# painting. The pod keeps TRACKING them — it must, or a wrong judgement
# would blind it permanently — but stops counting them when asking
# whether its view of the room matches YOLO's.


def test_wire_ids_that_make_no_sense_are_dropped_individually():
    """A bad value costs one unnecessary re-look, never a lost person,
    so each is skipped rather than failing the whole list."""
    from perception.server import _int_set
    assert _int_set([1, "2", None, "x", 3.7]) == {1, 2, 3}


def test_a_missing_or_wrong_shaped_list_is_simply_empty():
    """An older identity service sends nothing. The pod must then behave
    exactly as it did before this existed."""
    from perception.server import _int_set
    assert _int_set(None) == set()
    assert _int_set("7") == set()
    assert _int_set({"a": 1}) == set()


def test_the_list_is_bounded():
    """A hostile or corrupt response must not allocate without limit."""
    from perception.server import _int_set
    assert len(_int_set(list(range(500)))) == 64
