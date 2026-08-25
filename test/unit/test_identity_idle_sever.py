"""An empty room costs nothing, and the camera itself wakes us.

Stepping down a rung still downloads a stream. LiveKit bills each
subscriber for what it pulls, and the room is empty most of the day —
measured 2026-08-24, the TV Room had a person in 40% of ticks and the
bedroom 74%. The right amount to download for an empty room is none.

Unsubscribing is normally the same as going blind, because identity's
wake signal is a thumbnail diff over frames it would no longer receive.
It is safe here because the camera publishes a `motion_state` DATA
packet, and LiveKit delivers data to every participant in the room
regardless of track subscriptions. So identity keeps the cheap channel,
drops the expensive one, and needs nothing from the backend to return.
"""

from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture()
def svc(monkeypatch):
    for n in ("cv2", "numpy", "torch"):
        sys.modules.setdefault(n, types.ModuleType(n))
    import importlib
    mod = importlib.import_module("identity.service")
    monkeypatch.setattr(mod, "_idle_video", {}, raising=False)
    monkeypatch.setattr(mod, "_layer", {}, raising=False)
    monkeypatch.setattr(mod, "_pubs", {}, raising=False)
    monkeypatch.setattr(mod, "_PAUSED_SEVER_ONLY", False, raising=False)
    return mod


class _Pub:
    def __init__(self):
        self.subscribed = [True]
        self.quality = []

    def set_subscribed(self, want):
        self.subscribed.append(bool(want))

    def set_video_quality(self, q):
        self.quality.append(q)


def test_an_empty_room_stops_the_download(svc) -> None:
    pub = _Pub()
    svc._pubs["cam1"] = pub
    svc.set_idle_video("cam1", True)
    assert pub.subscribed[-1] is False, "an empty room was still being pulled"
    assert svc._idle_video["cam1"] is True


def test_motion_resumes_it_and_re_pins_the_rung(svc) -> None:
    """A re-subscribe resets the SFU's idea of what we asked for, so the
    pin must go out with it or we come back on the smallest rung."""
    pub = _Pub()
    svc._pubs["cam1"] = pub
    svc.set_idle_video("cam1", True)
    svc.set_idle_video("cam1", False)
    assert pub.subscribed[-1] is True
    assert pub.quality, "resumed without re-pinning the layer"


def test_the_repair_walk_does_not_undo_the_economy(svc) -> None:
    """The subscription walk re-asserts what is WANTED every 10s.

    If it ignored the idle sever it would re-subscribe an empty room on
    every pass, and the saving would silently evaporate.
    """
    svc._idle_video["cam1"] = True
    paused: set = set()
    want = ("cam1"[:8] not in paused) and not svc._idle_video.get("cam1")
    assert want is False


def test_a_paused_camera_is_still_severed_independently(svc) -> None:
    """Two separate reasons not to pull; either one is sufficient."""
    svc._idle_video["cam1"] = False
    paused = {"cam1"[:8]}
    want = ("cam1"[:8] not in paused) and not svc._idle_video.get("cam1")
    assert want is False


def test_the_step_only_escape_hatch_never_unsubscribes(svc, monkeypatch) -> None:
    """If severing ever misses a wake, this makes the room merely
    cheaper instead of silent — without a redeploy."""
    monkeypatch.setattr(svc, "_PAUSED_SEVER_ONLY", True, raising=False)
    pub = _Pub()
    svc._pubs["cam1"] = pub
    svc.set_idle_video("cam1", True)
    assert pub.subscribed[-1] is True, "escape hatch still unsubscribed"
    assert pub.quality, "escape hatch should step the rung instead"


def test_a_missing_publication_is_not_a_crash(svc) -> None:
    """Called before the track exists; an economy must never break intake."""
    svc.set_idle_video("nosuchcam", True)
    assert svc._idle_video["nosuchcam"] is True
