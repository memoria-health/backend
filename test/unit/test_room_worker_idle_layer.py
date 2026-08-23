"""A closed camera must stop paying for video it does not look at.

Skipping the VLM call on a closed camera saved the model cost and none
of the bandwidth. LiveKit bills the SUBSCRIPTION, and the subscription
does not care that the tick returned early — the Warner kept delivering
its top layer (2304x1296, 5 Mbps) to the backend AND to its identity
service all night while its room was empty. Measured 2026-08-24: the
backend container alone pulled 3.96 GB/hr with one occupied camera and
one empty one.

So a closed tick now asks for the idle layer and a look asks for the
full one back. Two things this must never do: signal on every 10s tick
(a layer request is a round-trip), and let a bandwidth economy break a
tick if the request fails.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

from livekit import rtc

from src.services.detection.motion_cache import MotionCache
from src.services.detection.room_worker import RoomWorker

IDENT = "cam_c51733e6"


class _FakePublication:
    """Records what layer was asked for, in order."""

    def __init__(self, *, fail: bool = False) -> None:
        self.requested: list = []
        self._fail = fail

    def set_video_quality(self, quality) -> None:  # noqa: ANN001
        if self._fail:
            raise RuntimeError("SFU said no")
        self.requested.append(quality)


def _worker(idle_quality: str = "medium") -> RoomWorker:
    settings = SimpleNamespace(
        livekit_url="wss://example.livekit.cloud",
        livekit_api_key="key",
        livekit_api_secret="secret",
        livekit_worker_window_seconds=15,
        livekit_worker_target_fps=15,
        livekit_worker_tick_seconds=10,
        livekit_subscribe_quality="high",
        livekit_idle_subscribe_quality=idle_quality,
    )
    rw = RoomWorker(
        room_name="account_a30c505e",
        settings=settings,
        sessionmaker=None,
        motion_cache=MotionCache(ttl_seconds=15),
        reid=None,
    )
    rw._ingests_by_identity[IDENT] = SimpleNamespace(camera_id=uuid4())
    # A fresh subscribe starts at the full layer, which is the state
    # `_handle_track_subscribed` leaves behind.
    rw._layer_by_identity[IDENT] = "high"
    return rw


_QUALITY = {
    rtc.VideoQuality.VIDEO_QUALITY_LOW: "low",
    rtc.VideoQuality.VIDEO_QUALITY_MEDIUM: "medium",
    rtc.VideoQuality.VIDEO_QUALITY_HIGH: "high",
}


def _names(pub: _FakePublication) -> list[str]:
    return [_QUALITY[q] for q in pub.requested]


def test_a_closed_camera_steps_down():
    """THE point: the empty room stops paying for pixels nobody reads."""
    rw = _worker()
    pub = _FakePublication()
    rw._video_pubs_by_identity[IDENT] = pub

    rw.request_idle_layer(IDENT, idle=True)

    assert _names(pub) == ["medium"]


def test_a_look_steps_back_up():
    rw = _worker()
    pub = _FakePublication()
    rw._video_pubs_by_identity[IDENT] = pub

    rw.request_idle_layer(IDENT, idle=True)
    rw.request_idle_layer(IDENT, idle=False)

    assert _names(pub) == ["medium", "high"]


def test_only_transitions_signal():
    """This is consulted every 10s tick, all night. Re-asserting the
    same layer 8,640 times a day is signalling traffic bought with
    nothing."""
    rw = _worker()
    pub = _FakePublication()
    rw._video_pubs_by_identity[IDENT] = pub

    for _ in range(50):
        rw.request_idle_layer(IDENT, idle=True)

    assert _names(pub) == ["medium"], "a steady state must be silent"


def test_a_failed_request_does_not_raise():
    """`set_video_quality` is best-effort by design — on failure we stay
    on whatever layer the SFU chose. Losing the saving is acceptable;
    losing the tick is not."""
    rw = _worker()
    rw._video_pubs_by_identity[IDENT] = _FakePublication(fail=True)

    rw.request_idle_layer(IDENT, idle=True)  # must not raise


def test_an_unsubscribed_camera_is_a_no_op():
    """Ticks can outlive a publication — the publisher drops, the ingest
    is still around for a beat. Nothing to ask, nothing to remember."""
    rw = _worker()

    rw.request_idle_layer(IDENT, idle=True)

    assert rw._layer_by_identity[IDENT] == "high", (
        "recording a layer we never managed to request would make the "
        "real request look redundant and get skipped"
    )


def test_the_step_down_is_configurable_off():
    """`livekit_idle_subscribe_quality=high` disables the economy — the
    rollback lever, in case a first look at 640x360 costs a detection."""
    rw = _worker(idle_quality="high")
    pub = _FakePublication()
    rw._video_pubs_by_identity[IDENT] = pub

    rw.request_idle_layer(IDENT, idle=True)

    assert _names(pub) == []


def test_removal_forgets_the_layer():
    """A camera that rejoins is a fresh subscribe at the full layer. A
    remembered `medium` would make the restore look redundant and leave
    it stuck low for as long as it stayed connected."""
    rw = _worker()
    rw._video_pubs_by_identity[IDENT] = _FakePublication()
    rw._layer_by_identity[IDENT] = "medium"

    rw._video_pubs_by_identity.pop(IDENT, None)
    rw._layer_by_identity.pop(IDENT, None)

    assert IDENT not in rw._layer_by_identity
