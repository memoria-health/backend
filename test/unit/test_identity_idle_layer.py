"""Identity steps to the BOTTOM rung when the room is empty.

Low rather than the backend's medium, and the difference is deliberate:
the backend's waking tick analyses its buffered window, so the rung
arriving during the quiet period is what a fall gets judged on. Identity
keeps no such window, and its wake authority is a thumbnail pixel diff
rather than person detection, so it still notices someone walking in at
320x180 and asks for full resolution before the next batch.
The identity service pays for its own copy of every camera.

Each camera in a room is subscribed TWICE — once by the backend's
detector worker and once by this camera's identity service on the GPU
node. LiveKit bills the subscription, so an empty room delivered the
Warner's 5 Mbps top layer to both of them all night for nothing.

Two jobs here, and the first is not the economy.

PINNING is the safety property. With no explicit request the SFU picks
the rung and re-picks it whenever its congestion estimate twitches,
which reads downstream as sharp<->blurry flapping that blinds face
matching (soak #2: blur<->identity r=0.52). That flapping is the whole
reason the gateway ships `.simulcast-off` and the Tapo publishes a
single layer nobody can step down from. Pinning takes the choice away
from the SFU, which is what makes restoring the ladder safe.

Then the step-down, on the same signal the loop already uses to skip
the pod: nothing tracked and nothing moved.
"""

from __future__ import annotations

import pytest
from livekit import rtc

from identity import service

CAM = "c51733e6"
NAMES = {
    rtc.VideoQuality.VIDEO_QUALITY_LOW: "low",
    rtc.VideoQuality.VIDEO_QUALITY_MEDIUM: "medium",
    rtc.VideoQuality.VIDEO_QUALITY_HIGH: "high",
}


class _Pub:
    def __init__(self, *, fail: bool = False) -> None:
        self.asked: list[str] = []
        self._fail = fail

    def set_video_quality(self, quality) -> None:  # noqa: ANN001
        if self._fail:
            raise RuntimeError("SFU said no")
        self.asked.append(NAMES[quality])


@pytest.fixture(autouse=True)
def _clean():
    service._pubs.clear()
    service._layer.clear()
    yield
    service._pubs.clear()
    service._layer.clear()


def test_an_empty_room_steps_down():
    pub = _Pub()
    service._pubs[CAM] = pub
    service.request_layer(CAM, service._LAYER_ACTIVE)
    service.request_layer(CAM, service._LAYER_IDLE)
    assert pub.asked == ["high", "low"]


def test_motion_steps_back_up():
    pub = _Pub()
    service._pubs[CAM] = pub
    service.request_layer(CAM, service._LAYER_IDLE)
    service.request_layer(CAM, service._LAYER_ACTIVE)
    assert pub.asked == ["low", "high"]


def test_only_transitions_signal():
    """This rides the loop's per-step decision, which runs continuously.
    Re-asserting a layer that has not changed is a signalling round-trip
    bought with nothing."""
    pub = _Pub()
    service._pubs[CAM] = pub
    for _ in range(40):
        service.request_layer(CAM, service._LAYER_IDLE)
    assert pub.asked == ["low"]


def test_a_failed_request_does_not_raise():
    """Best-effort by design: on failure we keep whatever rung we were
    already getting. Losing the saving is fine; losing intake is not."""
    service._pubs[CAM] = _Pub(fail=True)
    service.request_layer(CAM, service._LAYER_IDLE)


def test_an_unsubscribed_camera_remembers_nothing():
    """A request we never managed to make must not be recorded, or the
    real one later looks redundant and gets skipped."""
    service.request_layer(CAM, service._LAYER_IDLE)
    assert CAM not in service._layer


def test_the_step_down_is_configurable_off():
    """NEMARA_IDENTITY_IDLE_QUALITY=high disables it — the rollback if a
    step-down ever costs a lock."""
    pub = _Pub()
    service._pubs[CAM] = pub
    service.request_layer(CAM, "high")
    before = len(pub.asked)
    service.request_layer(CAM, "high")   # what idle would ask for
    assert len(pub.asked) == before
