"""A disturbance keeps its tick, and the tick after it.

Sleep is `record_only` — it fires no event, so `clip_archival` never
runs for it and a night of sleep data leaves nothing to look at. This
sidecar is the evidence: it saves the mp4 the tick was judged on, next
to the fields that judged it, so a real disturbance can be told from a
hallucinated one in the morning.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from src.services.detection import sleep_clips

CAM = uuid4()
MP4 = b"\x00\x00\x00\x18ftypmp42fake"


def _settings(tmp_path):
    return SimpleNamespace(sleep_clips_dir=str(tmp_path))


def _clips(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("*.mp4"))


@pytest.fixture(autouse=True)
def _clear_state():
    sleep_clips._follow_up.clear()
    yield
    sleep_clips._follow_up.clear()


@pytest.mark.asyncio
async def test_a_quiet_tick_saves_nothing(tmp_path):
    """The steady state is most of the night. Saving it would bury the
    handful of moments worth reading."""
    await sleep_clips.save_if_disturbance(
        camera_id=CAM, detector_fields={"sleep_activity": "none"},
        mp4_bytes=MP4, settings=_settings(tmp_path))
    assert _clips(tmp_path) == []


@pytest.mark.asyncio
async def test_a_disturbance_saves_that_tick_and_the_next(tmp_path):
    """THE point. The mark lands on the tick where movement was visible;
    the tick AFTER is what says whether they settled or stayed up."""
    s = _settings(tmp_path)
    await sleep_clips.save_if_disturbance(
        camera_id=CAM, detector_fields={"sleep_activity": "sitting_up"},
        mp4_bytes=MP4, settings=s)
    await sleep_clips.save_if_disturbance(
        camera_id=CAM, detector_fields={"sleep_activity": "none"},
        mp4_bytes=MP4, settings=s)

    names = _clips(tmp_path)
    assert len(names) == 2, "a disturbance must keep its follow-up tick"
    assert any(n.endswith("_sitting_up.mp4") for n in names)
    assert any(n.endswith("_after.mp4") for n in names)


@pytest.mark.asyncio
async def test_the_third_quiet_tick_stops(tmp_path):
    """One follow-up, not a stuck recorder."""
    s = _settings(tmp_path)
    for activity in ("stirring", "none", "none", "none"):
        await sleep_clips.save_if_disturbance(
            camera_id=CAM, detector_fields={"sleep_activity": activity},
            mp4_bytes=MP4, settings=s)
    assert len(_clips(tmp_path)) == 2


@pytest.mark.asyncio
async def test_a_run_of_movement_keeps_saving(tmp_path):
    """A long episode must not capture only its first frame."""
    s = _settings(tmp_path)
    for activity in ("stirring", "sitting_up", "leaving_bed", "none"):
        await sleep_clips.save_if_disturbance(
            camera_id=CAM, detector_fields={"sleep_activity": activity},
            mp4_bytes=MP4, settings=s)
    assert len(_clips(tmp_path)) == 4


@pytest.mark.asyncio
async def test_the_fields_are_written_beside_the_footage(tmp_path):
    """A clip alone cannot say what the model claimed while watching it."""
    await sleep_clips.save_if_disturbance(
        camera_id=CAM,
        detector_fields={"sleep_activity": "leaving_bed",
                         "sleep_detected": False, "sleep_confidence": 0.8},
        mp4_bytes=MP4, settings=_settings(tmp_path))
    notes = list(tmp_path.rglob("*.txt"))
    assert len(notes) == 1
    body = notes[0].read_text()
    assert "sleep_activity=leaving_bed" in body
    assert "sleep_confidence=0.8" in body


@pytest.mark.asyncio
async def test_off_by_default(tmp_path):
    """No directory configured means this never touches disk — it is a
    dev instrument writing unscreened frames."""
    await sleep_clips.save_if_disturbance(
        camera_id=CAM, detector_fields={"sleep_activity": "sitting_up"},
        mp4_bytes=MP4, settings=SimpleNamespace(sleep_clips_dir=""))
    assert _clips(tmp_path) == []


@pytest.mark.asyncio
async def test_a_tick_with_no_mp4_is_survivable(tmp_path):
    """Evidence must never cost the tick that produced it."""
    await sleep_clips.save_if_disturbance(
        camera_id=CAM, detector_fields={"sleep_activity": "stirring"},
        mp4_bytes=None, settings=_settings(tmp_path))
    assert _clips(tmp_path) == []


@pytest.mark.asyncio
async def test_two_cameras_do_not_arm_each_other(tmp_path):
    """A disturbance in one room must not save a quiet tick in another."""
    s = _settings(tmp_path)
    other = uuid4()
    await sleep_clips.save_if_disturbance(
        camera_id=CAM, detector_fields={"sleep_activity": "stirring"},
        mp4_bytes=MP4, settings=s)
    await sleep_clips.save_if_disturbance(
        camera_id=other, detector_fields={"sleep_activity": "none"},
        mp4_bytes=MP4, settings=s)
    assert len(_clips(tmp_path)) == 1
