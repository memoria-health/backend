"""Every camera gets asked about every concept.

The screen pass grounds ONE concept per pass and rotates. The rotation
counter used to be a single global, which hands out concepts by arrival
order across all cameras — so when the camera count shares a factor with
the concept count, every camera locks onto one index and stays there.

That is not hypothetical. On 2026-08-25 the list was
("television or computer screen", "painting", "statue") with two
cameras. 3 and 2 are coprime, so both cameras saw all three over time.
Dropping "statue", because it never grounded anything, left 2 and 2:

    bedroom  kind=static  every pass    (painting, forever)
    TV Room  kind=screen  every pass    (never painting, forever)

The TV Room's painted faces could not be re-grounded, its regions aged
out (live 12 -> 11 -> 0), and the phantom people came back — from a
change that only removed a concept that did nothing.
"""

from __future__ import annotations

import importlib
import sys
import types


def _server():
    """Import the perception module without its GPU dependencies."""
    for name in ("cv2", "numpy", "torch"):
        sys.modules.setdefault(name, types.ModuleType(name))
    return importlib.import_module("perception.server")


def _rotation(mod, camera_id: str, n: int) -> list[str]:
    return [mod._screen_params(camera_id)[0] for _ in range(n)]


def test_two_cameras_and_two_concepts_do_not_lock(monkeypatch) -> None:
    """The exact shape that regressed."""
    mod = _server()
    monkeypatch.setattr(mod, "_screen_turn", {})
    monkeypatch.setattr(
        mod, "_screen_config",
        lambda: (("television or computer screen", "painting"), 0.05))

    # Interleaved, the way two cameras actually poll.
    a, b = [], []
    for _ in range(4):
        a.append(mod._screen_params("cam_a")[0])
        b.append(mod._screen_params("cam_b")[0])

    assert len(set(a)) == 2, f"cam_a locked onto {set(a)}"
    assert len(set(b)) == 2, f"cam_b locked onto {set(b)}"


def test_every_camera_walks_the_whole_rotation(monkeypatch) -> None:
    """However many cameras, however many concepts."""
    mod = _server()
    concepts = ("screen", "painting", "statue", "mirror")
    monkeypatch.setattr(mod, "_screen_turn", {})
    monkeypatch.setattr(mod, "_screen_config", lambda: (concepts, 0.05))

    for cams in (1, 2, 3, 4, 8):
        monkeypatch.setattr(mod, "_screen_turn", {})
        for c in range(cams):
            seen = set(_rotation(mod, f"cam_{c}", len(concepts)))
            assert seen == set(concepts), (
                f"with {cams} cameras, cam_{c} only ever saw {seen}")


def test_a_camera_advances_only_its_own_turn(monkeypatch) -> None:
    """One busy camera must not skip another camera through the list."""
    mod = _server()
    monkeypatch.setattr(mod, "_screen_turn", {})
    monkeypatch.setattr(mod, "_screen_config", lambda: (("a", "b", "c"), 0.05))

    for _ in range(10):
        mod._screen_params("busy")
    # The quiet camera has not moved: its first pass is still concept 0.
    assert mod._screen_params("quiet")[0] == "a"
