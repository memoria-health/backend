"""Integration tests for `src.services.clip_archival.archive_event`.

Per `feedback_no_mocking_async_wiring`: the supporting objects are
all real — real Event/Patient/Caregiver rows, real FrameBuffer /
AudioBuffer instances pre-populated with synthetic frames. Only the
storage HTTP layer is swapped out (via the `fake_httpx` pattern from
test_storage.py) so we can assert the exact wire shape and avoid
touching real Supabase Storage.

What we assert:

  - The full pipeline lands `events.has_clip=true` and
    `events.clip_storage_key=events/{event_id}.mp4`.
  - The Supabase upload is a single POST to the right
    `incident-clips/events/{id}.mp4` URL with the combined MP4 body.
  - If the post-event frame snapshot is empty, the row stays as-is
    (the events row is the user-facing surface; the clip is optional).
  - `concat_mp4s` actually produces an MP4 from two real ones (uses
    ffmpeg; skipped if not on PATH).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.caregiver import Caregiver
from src.db.event import Event
from src.db.patient import Patient
from src.db.session import get_sessionmaker
from src.services import clip_archival, storage as storage_module
from src.services.clip_archival import archive_event, concat_mp4s
from src.services.detection.audio_buffer import AudioBuffer
from src.services.detection.frame_buffer import FrameBuffer
from src.services.detection.mp4_remux import frames_to_mp4
from src.settings import Settings, get_settings

_pytestmark_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)


# ── HTTP fake (mirrors test_storage.py's pattern) ─────────────────────


class _FakeAsyncClient:
    state: dict[str, Any] = {"responses": [], "captured": []}

    def __init__(self, *, timeout: float | None = None, **_kw: Any) -> None:
        self._timeout = timeout

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    async def post(self, url: str, **kw: Any) -> httpx.Response:
        return self._record("POST", url, kw)

    async def get(self, url: str, **kw: Any) -> httpx.Response:
        return self._record("GET", url, kw)

    async def delete(self, url: str, **kw: Any) -> httpx.Response:
        return self._record("DELETE", url, kw)

    @classmethod
    def _record(cls, method: str, url: str, kw: dict[str, Any]) -> httpx.Response:
        cls.state["captured"].append({"method": method, "url": url, "kw": kw})
        if not cls.state["responses"]:
            raise RuntimeError(f"no canned response for {method} {url}")
        return cls.state["responses"].pop(0)


def _install_fake_httpx(monkeypatch) -> dict[str, Any]:
    """Install the fake AFTER any seeding step that hits real
    Supabase. The fake replaces httpx.AsyncClient globally, so any
    other code path that constructs one (e.g. conftest's
    `auth_user_factory` hitting `/auth/v1/admin/users`) gets the
    fake too — which would deadlock seeding."""
    state: dict[str, Any] = {"responses": [], "captured": []}
    _FakeAsyncClient.state = state
    monkeypatch.setattr(storage_module.httpx, "AsyncClient", _FakeAsyncClient)
    return state


# ── seeding ───────────────────────────────────────────────────────────


async def _seed_event(session: AsyncSession, auth_user_factory) -> tuple[UUID, UUID]:
    uid, email = await auth_user_factory()
    now = datetime.now(timezone.utc)
    session.add(
        Caregiver(
            id=uid, display_name="Clip Tester", email=email,
            created_at=now, updated_at=now,
        )
    )
    patient_id = uuid4()
    session.add(
        Patient(
            id=patient_id, caregiver_id=uid,
            full_name="Margaret O'Brien", preferred_name="Maggie",
            relation="parent", state="calm", status_line="OK",
            cameras_paused=False, created_at=now, updated_at=now,
        )
    )
    await session.flush()
    event_id = uuid4()
    session.add(
        Event(
            id=event_id, patient_id=patient_id,
            detector_id=None, camera_id=None,
            category="safety", source="preset", icon_name="fall",
            title="Fall", location="Bedroom",
            occurred_at=now, has_clip=False,
            narrative="Maggie fell.", opened=False, reviewed=False,
            review_choice=None, review_notes=None, reviewed_at=None,
            clip_storage_key=None, created_at=now,
        )
    )
    await session.commit()
    return event_id, patient_id


def _i420_blob(width: int, height: int) -> bytes:
    return b"\x80" * (width * height * 3 // 2)


async def _build_pre_event_mp4() -> bytes:
    """A real MP4 stream — concat_mp4s needs both inputs to share
    codec params for `-c copy`, which is guaranteed only when both
    come out of the same frames_to_mp4 call. So we build the same
    way the production code does."""
    from src.services.detection.frame_buffer import FrameSnapshot
    frames = [
        FrameSnapshot(
            width=320, height=240,
            data=_i420_blob(320, 240),
            captured_at=100.0 + i / 5,
        )
        for i in range(15)  # 3s @ 5 fps
    ]
    return await frames_to_mp4(frames, fps=5)


@pytest_asyncio.fixture
async def filled_buffers() -> AsyncIterator[tuple[FrameBuffer, AudioBuffer]]:
    """A pre-populated post-event frame + audio buffer.

    archive_event() sleeps for clip_post_event_wait_seconds and then
    calls snapshot_last_seconds(clip_post_event_wait_seconds) against
    time.monotonic(). In production the room-worker is continuously
    pushing fresh frames during that sleep, so the snapshot always
    finds recent ones. In the test we have no such background pusher,
    so we future-date the fixture's captured_at values to outlive any
    setup delay between fixture build and archive_event call (ffmpeg
    pre-mp4 build can take a few hundred ms).
    """
    import time
    base = time.monotonic() + 2.0
    frames = FrameBuffer(window_seconds=15.0, target_fps=5)
    audio = AudioBuffer(window_seconds=15.0)
    for i in range(15):
        frames.push(
            width=320, height=240, data=_i420_blob(320, 240),
            captured_at=base + i / 5,
        )
    yield frames, audio


# ── tests ─────────────────────────────────────────────────────────────


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_uploads_and_flips_has_clip(
    auth_user_factory, filled_buffers, monkeypatch,
):
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    frame_buffer, audio_buffer = filled_buffers
    pre_mp4 = await _build_pre_event_mp4()
    # Override the wait so the test doesn't pause for 15s.
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(httpx.Response(200, text="ok"))

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        pre_event_mp4=pre_mp4,
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # One upload, to the final key.
    assert len(fake_httpx["captured"]) == 1
    call = fake_httpx["captured"][0]
    assert call["method"] == "POST"
    expected_key = f"events/{event_id}.mp4"
    assert call["url"].endswith(f"/incident-clips/{expected_key}")
    assert call["kw"]["headers"]["Content-Type"] == "video/mp4"

    # Body is a valid MP4 (starts with ftyp box past the size prefix).
    body = call["kw"]["content"]
    assert isinstance(body, bytes)
    assert body[4:8] == b"ftyp"

    # Row updated.
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is True
    assert row.clip_storage_key == expected_key


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_skips_upload_under_privacy(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """Privacy mode strips the event clip. The archiver's re-check before
    upload drops it: no storage POST, has_clip stays False. The text
    event/insight (created upstream) is unaffected — only the video is
    suppressed."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id = await _seed_event(s, auth_user_factory)
        # Patient is in privacy mode by the time the clip would upload.
        p = await s.get(Patient, patient_id)
        assert p is not None
        p.privacy_mode_enabled = True
        await s.commit()

    fake_httpx = _install_fake_httpx(monkeypatch)
    frame_buffer, audio_buffer = filled_buffers
    pre_mp4 = await _build_pre_event_mp4()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        pre_event_mp4=pre_mp4,
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # Privacy dropped the clip — no upload was ever attempted.
    assert fake_httpx["captured"] == []

    # Row stays without a clip (text insight already lives in the row),
    # and is stamped clip_suppressed so the apps can say WHY (privacy flip
    # happened DURING the post-roll, so it's stamped here, not at creation).
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is False
    assert row.clip_storage_key is None
    assert row.clip_suppressed is True


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_noops_when_post_frames_missing(
    auth_user_factory, monkeypatch,
):
    """If no frames have arrived in the post-event window the
    archiver doesn't upload (and doesn't crash). The events row
    stays without a clip — caregiver still sees the alert text."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    # Empty buffers (no frames pushed).
    frame_buffer = FrameBuffer(window_seconds=15.0, target_fps=5)
    audio_buffer = AudioBuffer(window_seconds=15.0)
    pre_mp4 = await _build_pre_event_mp4()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        pre_event_mp4=pre_mp4,
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    # No upload attempted.
    assert fake_httpx["captured"] == []

    # Row stays without a clip.
    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is False
    assert row.clip_storage_key is None


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_concat_mp4s_produces_a_valid_mp4_with_combined_duration() -> None:
    """End-to-end ffmpeg concat. Two 3s clips → ~6s combined.
    Asserts via ffprobe so the actual stream is well-formed."""
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not on PATH")
    pre = await _build_pre_event_mp4()
    post = await _build_pre_event_mp4()
    combined = await concat_mp4s(pre, post)

    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", "-i", "pipe:0"],
        input=combined,
        capture_output=True,
        check=True,
    )
    info = json.loads(proc.stdout)
    duration = float(info["format"]["duration"])
    # Each input is 15 frames @ 5fps = 3s. Combined should be ~6s.
    # Allow ±0.2s for muxer rounding.
    assert 5.6 <= duration <= 6.5
    streams = info["streams"]
    assert any(s["codec_type"] == "video" for s in streams)


@_pytestmark_ffmpeg
@pytest.mark.asyncio
async def test_archive_event_swallows_upload_failure(
    auth_user_factory, filled_buffers, monkeypatch,
):
    """Upload errors must NOT crash the camera tick. The archiver is
    a fire-and-forget task; we log and move on."""
    sm = get_sessionmaker()
    async with sm() as s:
        event_id, patient_id = await _seed_event(s, auth_user_factory)

    fake_httpx = _install_fake_httpx(monkeypatch)
    frame_buffer, audio_buffer = filled_buffers
    pre_mp4 = await _build_pre_event_mp4()
    base = get_settings()
    overrides = base.model_dump()
    overrides["clip_post_event_wait_seconds"] = 0.05
    settings = Settings(**overrides)

    fake_httpx["responses"].append(
        httpx.Response(503, text="boom", request=httpx.Request("POST", "https://x")),
    )

    # Must not raise.
    await archive_event(
        event_id=event_id,
        patient_id=patient_id,
        pre_event_mp4=pre_mp4,
        pre_end_captured_at=time.monotonic(),
        frame_buffer=frame_buffer,
        audio_buffer=audio_buffer,
        sessionmaker=sm,
        settings=settings,
    )

    async with sm() as s:
        row = await s.get(Event, event_id)
    assert row is not None
    assert row.has_clip is False
