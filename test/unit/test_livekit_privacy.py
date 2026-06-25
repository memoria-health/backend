"""Live SFU severance for privacy mode (Phase 4.2).

When privacy flips on, caregivers ALREADY connected still hold a
can_subscribe=true token. ``sever_caregiver_media`` walks the room and
revokes subscribe+publish on every ``caregiver_*`` participant right
now — leaving the detector bot (``detector_worker_*``) and the camera
(``cam_*``) untouched so analysis keeps running.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.services.livekit_privacy import sever_caregiver_media


class _FakeRoomService:
    def __init__(self, identities: list[str]) -> None:
        self._identities = identities
        self.updated: list = []

    async def list_participants(self, req):
        return SimpleNamespace(
            participants=[SimpleNamespace(identity=i) for i in self._identities]
        )

    async def update_participant(self, req):
        self.updated.append(req)


class _FakeLK:
    def __init__(self, identities: list[str]) -> None:
        self.room = _FakeRoomService(identities)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_severs_only_caregivers_not_bot_or_camera() -> None:
    lk = _FakeLK(
        [
            "caregiver_aaa_ios",
            "caregiver_bbb_web",
            "detector_worker_deadbeef",  # bot — must stay
            "cam_1234",  # camera — must stay
        ]
    )

    severed = await sever_caregiver_media(lk, "patient_x")

    assert severed == 2
    touched = {req.identity for req in lk.room.updated}
    assert touched == {"caregiver_aaa_ios", "caregiver_bbb_web"}
    for req in lk.room.updated:
        assert req.permission.can_subscribe is False
        assert req.permission.can_publish is False


@pytest.mark.asyncio
async def test_no_caregivers_is_noop() -> None:
    lk = _FakeLK(["detector_worker_x", "cam_y"])
    severed = await sever_caregiver_media(lk, "patient_x")
    assert severed == 0
    assert lk.room.updated == []
