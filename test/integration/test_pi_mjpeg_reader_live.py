"""Integration test for the MJPEG reader against a real Raspberry Pi.

Gated on `NEMARA_TEST_PI_STREAM_URL` so CI does not require a
physical camera. Set the env var to the running Pi's stream URL
(e.g. `http://10.0.0.44:8000/stream.mjpg`) to exercise this test
locally.

Validates the end-to-end happy path: TCP connect, MJPEG protocol
parsing, and that the JPEG payloads coming off the Pi are real JPEGs
(SOI marker present, finite size). It runs for at most 10 seconds
of wall time and bails as soon as enough frames have been collected.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from src.services.ingest.pi.mjpeg_reader import MjpegStreamError, read_mjpeg

_pi_url = os.environ.get("NEMARA_TEST_PI_STREAM_URL")

pytestmark = pytest.mark.skipif(
    _pi_url is None,
    reason="Set NEMARA_TEST_PI_STREAM_URL to enable the live Pi integration test",
)


_TARGET_FRAMES = 10
_DEADLINE_SECONDS = 10.0


async def test_pulls_real_jpeg_frames_from_live_pi() -> None:
    assert _pi_url is not None  # placate type checker after skipif

    collected: list[bytes] = []

    async def _consume() -> None:
        async with read_mjpeg(
            _pi_url, connect_timeout=5.0, read_timeout=5.0
        ) as frames:
            async for jpeg in frames:
                collected.append(jpeg)
                if len(collected) >= _TARGET_FRAMES:
                    return

    try:
        await asyncio.wait_for(_consume(), timeout=_DEADLINE_SECONDS)
    except TimeoutError:
        pytest.fail(
            f"Only got {len(collected)} frames in {_DEADLINE_SECONDS}s; "
            f"either the Pi is not streaming or it is slower than 1 fps"
        )
    except MjpegStreamError as exc:
        pytest.fail(f"MJPEG reader failed against live Pi: {exc}")

    assert len(collected) == _TARGET_FRAMES, (
        f"Expected exactly {_TARGET_FRAMES} frames, got {len(collected)}"
    )

    # Per-frame sanity: every frame is a non-trivial JPEG.
    for i, jpeg in enumerate(collected):
        assert isinstance(jpeg, bytes), (
            f"Frame {i}: expected bytes, got {type(jpeg).__name__}"
        )
        assert jpeg.startswith(b"\xff\xd8"), (
            f"Frame {i}: missing JPEG SOI marker; first bytes are {jpeg[:4]!r}"
        )
        # 720p JPEG quality 75 is typically 30-80 KB; 480p is smaller.
        # We accept anything in [1 KB, 1 MB] as a sane payload.
        assert 1024 <= len(jpeg) <= 1024 * 1024, (
            f"Frame {i}: size {len(jpeg)} bytes is outside [1 KB, 1 MB]"
        )
