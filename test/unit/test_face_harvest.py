"""Learning a face from a lock, without poisoning the gallery.

The enrolled gallery is 25 selfies from one session (`source='selfie'`)
and the camera is on a wall looking down across a room. Measured on a
two-minute walk: 55 face sightings, best cosine 0.39 against a ~0.40
bar — and size did not help, a 132px face scored 0.20 while a 48px one
scored 0.39. That is a viewpoint gap, not a recognition failure, and no
number of extra frames or pixels closes it.

A face-confirmed lock is the one moment we KNOW who someone is and have
their face in front of us at this camera's angle. Learning there widens
the gallery toward the walking case.

Learned from a FACE lock, deliberately, not the body lock that would
bootstrap faster: face is the decisive signal and its gallery must never
be taught by a weaker one. `harvest_eligible` — already hardened for the
body pool — is the gate, unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import numpy as np

from identity import loop as L

PID = uuid4()


def _loop(gallery: np.ndarray | None = None):
    """A CameraLoop with just enough on it to run the harvest gate."""
    c = L.CameraLoop.__new__(L.CameraLoop)
    c.camera_id = "c51733e6"
    c._face_harvested_at = 0.0
    c.home = SimpleNamespace(face_galleries={PID: gallery} if gallery is not None else {})
    # The decision is sync and queues; the async probe loop drains it.
    # That split is deliberate — `_fold_probe` runs inside
    # asyncio.to_thread, where there is no event loop to schedule on.
    c._pending_face = []
    return c


def _emb(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_a_good_look_is_learned():
    """THE point: a confirmed lock teaches the gallery this camera's
    view of the person."""
    c = _loop()
    c._harvest_face(PID, (_emb(1), 0.9, 100.0))
    assert len(c._pending_face) == 1


def test_nothing_to_learn_from_is_skipped():
    c = _loop()
    c._harvest_face(PID, None)
    assert c._pending_face == []


def test_a_weak_detection_is_refused():
    """det_score is the detector's own confidence. A poor one is as
    likely to be a hairline as a face, and this gallery is decisive."""
    c = _loop()
    c._harvest_face(PID, (_emb(2), 0.3, 100.0))
    assert c._pending_face == []


def test_a_tiny_face_is_refused():
    c = _loop()
    c._harvest_face(PID, (_emb(3), 0.9, 20.0))
    assert c._pending_face == []


def test_a_view_we_already_hold_is_not_stored_twice():
    """DIVERSITY, not repetition. A lock that lingers would otherwise
    deposit a hundred near-identical rows and crowd out every other
    view — the body pool learned this first."""
    same = _emb(4)
    c = _loop(gallery=same.reshape(1, -1))
    c._harvest_face(PID, (same, 0.9, 100.0))
    assert c._pending_face == []


def test_a_genuinely_new_view_IS_stored():
    """The whole value: a different angle is exactly what the selfie
    gallery lacks."""
    c = _loop(gallery=_emb(5).reshape(1, -1))
    c._harvest_face(PID, (_emb(6), 0.9, 100.0))
    assert len(c._pending_face) == 1


def test_learned_rows_cannot_swamp_the_enrolled_ones():
    """`source='selfie'` is ground truth — a human said this is them.
    Harvested rows are the system's own opinion and stay capped."""
    big = np.stack([_emb(i) for i in range(L.FACE_HARVEST_MAX_ROWS + 5)])
    c = _loop(gallery=big)
    c._harvest_face(PID, (_emb(999), 0.9, 100.0))
    assert c._pending_face == []


def test_one_harvest_per_window():
    """A held lock probes every 1.5s. Without this the gallery would
    take a hundred looks at one moment."""
    c = _loop()
    c._harvest_face(PID, (_emb(7), 0.9, 100.0))
    c._harvest_face(PID, (_emb(8), 0.9, 100.0))
    assert len(c._pending_face) == 1


def test_the_whole_thing_is_switchable_off(monkeypatch):
    monkeypatch.setattr(L, "FACE_HARVEST_ENABLED", False)
    c = _loop()
    c._harvest_face(PID, (_emb(9), 0.9, 100.0))
    assert c._pending_face == []
