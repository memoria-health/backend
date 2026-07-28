"""Live SFU severance for per-camera privacy (room-derived, 2026-07-27).

When a camera's ``privacy_enabled`` flips true (classifier
finalization, internal reclassify override), caregivers already in the
room are still subscribed to that camera's tracks.
``sever_caregiver_media`` force-unsubscribes every ``caregiver_*``
participant from THAT camera's tracks via ``UpdateSubscriptions`` —
leaving the detector bot (``detector_worker_*``), the cameras
(``cam_*``), and every OTHER camera's tracks untouched, so analysis and
non-private live views keep flowing. Best-effort defense-in-depth: the
durable cut is publisher-side track subscription permissions.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.services.livekit_privacy as livekit_privacy
from src.services.livekit_privacy import (
    enforce_privacy_for_camera,
    enforce_privacy_for_room,
    sever_caregiver_media,
)

_SETTINGS = SimpleNamespace(
    livekit_url="ws://x", livekit_api_key="k", livekit_api_secret="s",
)


def _participant(identity: str, track_sids: list[str] | None = None):
    return SimpleNamespace(
        identity=identity,
        tracks=[SimpleNamespace(sid=sid) for sid in (track_sids or [])],
    )


class _FakeRoomService:
    def __init__(self, participants: list) -> None:
        self._participants = participants
        self.unsubscribed: list = []

    async def list_participants(self, req):
        return SimpleNamespace(participants=self._participants)

    async def update_subscriptions(self, req):
        self.unsubscribed.append(req)


class _FakeLK:
    def __init__(self, participants: list) -> None:
        self.room = _FakeRoomService(participants)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_unsubscribes_only_caregivers_from_only_this_cameras_tracks() -> None:
    lk = _FakeLK(
        [
            _participant("caregiver_aaa_ios"),
            _participant("caregiver_bbb_web"),
            _participant("detector_worker_deadbeef"),  # bot — must stay
            _participant("cam_1234", ["TR_video1", "TR_audio1"]),  # private cam
            _participant("cam_5678", ["TR_video2"]),  # other camera — untouched
        ]
    )

    severed = await sever_caregiver_media(lk, "patient_x", "cam_1234")

    assert severed == 2
    touched = {req.identity for req in lk.room.unsubscribed}
    assert touched == {"caregiver_aaa_ios", "caregiver_bbb_web"}
    for req in lk.room.unsubscribed:
        # Per-camera cut: exactly the private camera's tracks, sub OFF.
        assert list(req.track_sids) == ["TR_video1", "TR_audio1"]
        assert req.subscribe is False
        assert req.room == "patient_x"


@pytest.mark.asyncio
async def test_no_caregivers_is_noop() -> None:
    lk = _FakeLK(
        [
            _participant("detector_worker_x"),
            _participant("cam_y", ["TR_v"]),
        ]
    )
    severed = await sever_caregiver_media(lk, "patient_x", "cam_y")
    assert severed == 0
    assert lk.room.unsubscribed == []


@pytest.mark.asyncio
async def test_camera_not_publishing_is_noop() -> None:
    """Camera offline / not yet joined → nothing to unsubscribe from
    (publisher-side permissions cover the eventual join)."""
    lk = _FakeLK([_participant("caregiver_aaa_ios")])
    severed = await sever_caregiver_media(lk, "patient_x", "cam_gone")
    assert severed == 0
    assert lk.room.unsubscribed == []


@pytest.mark.asyncio
async def test_camera_with_no_tracks_is_noop() -> None:
    lk = _FakeLK(
        [
            _participant("caregiver_aaa_ios"),
            _participant("cam_1234"),  # joined, zero published tracks
        ]
    )
    severed = await sever_caregiver_media(lk, "patient_x", "cam_1234")
    assert severed == 0
    assert lk.room.unsubscribed == []


@pytest.mark.asyncio
async def test_enforce_swallows_client_construction_error(monkeypatch) -> None:
    """Best-effort: if constructing the LiveKit client throws, the error
    must NOT propagate. The classifier finalization / reclassify request
    already committed privacy_enabled=true; a raise here would fail an
    operation whose durable enforcement (publisher-side permissions) is
    unaffected."""

    def _boom(*a, **k):
        raise RuntimeError("livekit down")

    monkeypatch.setattr(livekit_privacy.lkapi, "LiveKitAPI", _boom)
    # Must return normally, not raise.
    await enforce_privacy_for_camera(
        _SETTINGS, room="patient_x", camera_identity="cam_1234",
    )


@pytest.mark.asyncio
async def test_enforce_swallows_sever_error_and_still_closes(monkeypatch) -> None:
    """A transient error mid-sever is swallowed AND aclose() still runs
    (no leaked client)."""
    closed = {"v": False}

    class _RaisingRoom:
        async def list_participants(self, req):
            raise RuntimeError("transient")

    class _LK:
        def __init__(self, *a, **k):
            self.room = _RaisingRoom()

        async def aclose(self):
            closed["v"] = True

    monkeypatch.setattr(livekit_privacy.lkapi, "LiveKitAPI", _LK)
    await enforce_privacy_for_camera(
        _SETTINGS, room="patient_x", camera_identity="cam_1234",
    )  # must not raise
    assert closed["v"] is True


# ── enforce_privacy_for_room (caregiver-join self-heal) ──────────────


class _FakeSession:
    """Answers the private-cameras-in-room SELECT with canned identity
    rows (or raises)."""

    def __init__(self, rows: list[tuple], *, raises: bool = False) -> None:
        self._rows = rows
        self._raises = raises
        self.statements: list = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *args) -> None:
        return None

    async def execute(self, stmt):
        self.statements.append(stmt)
        if self._raises:
            raise RuntimeError("db down")
        return iter(self._rows)


@pytest.mark.asyncio
async def test_room_resever_hits_only_private_cameras(monkeypatch) -> None:
    """A caregiver JOIN must re-apply the per-camera sever for every
    camera the DB says is private — and only those. (The SELECT itself
    filters privacy_enabled=true; the fake returns its result rows.)"""
    session = _FakeSession([("cam_private1",), ("cam_private2",)])
    severed: list = []

    async def _record(settings, *, room, camera_identity):
        severed.append((room, camera_identity))

    monkeypatch.setattr(
        livekit_privacy, "enforce_privacy_for_camera", _record,
    )
    await enforce_privacy_for_room(_SETTINGS, lambda: session, room="patient_x")

    assert severed == [
        ("patient_x", "cam_private1"),
        ("patient_x", "cam_private2"),
    ]
    # The lookup is scoped: room match + privacy_enabled=true + a
    # LiveKit identity to sever.
    sql = str(session.statements[0])
    assert "privacy_enabled" in sql
    assert "livekit_room" in sql
    assert "livekit_identity" in sql


@pytest.mark.asyncio
async def test_room_resever_no_private_cameras_is_noop(monkeypatch) -> None:
    session = _FakeSession([])

    async def _must_not_run(settings, *, room, camera_identity):
        raise AssertionError("no private cameras → no sever calls")

    monkeypatch.setattr(
        livekit_privacy, "enforce_privacy_for_camera", _must_not_run,
    )
    await enforce_privacy_for_room(_SETTINGS, lambda: session, room="patient_x")


@pytest.mark.asyncio
async def test_room_resever_swallows_db_error(monkeypatch) -> None:
    """Best-effort: a DB failure during the lookup must not raise into
    the webhook handler."""
    session = _FakeSession([], raises=True)

    async def _must_not_run(settings, *, room, camera_identity):
        raise AssertionError("lookup failed → no sever calls")

    monkeypatch.setattr(
        livekit_privacy, "enforce_privacy_for_camera", _must_not_run,
    )
    # Must return normally, not raise.
    await enforce_privacy_for_room(_SETTINGS, lambda: session, room="patient_x")


@pytest.mark.asyncio
async def test_enforce_noop_without_livekit_wiring(monkeypatch) -> None:
    """An unpaired camera row has no livekit_room/identity yet — the
    enforce helper must no-op instead of constructing a client."""

    def _boom(*a, **k):  # would fail loudly if the client were built
        raise AssertionError("client must not be constructed")

    monkeypatch.setattr(livekit_privacy.lkapi, "LiveKitAPI", _boom)
    await enforce_privacy_for_camera(_SETTINGS, room=None, camera_identity=None)
    await enforce_privacy_for_camera(
        _SETTINGS, room="patient_x", camera_identity=None,
    )
