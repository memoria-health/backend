"""Unit tests for `services/ingest/pi/audio_reader.py`.

Same two-layer pattern as the MJPEG reader's tests:

  - `_frame_chunks` is the pure re-framer. Hand-craft byte streams
    chunked at every interesting boundary; verify fixed-size frames
    are produced regardless of arrival pattern.
  - `read_pcm_audio` is the transport wrapper. Tested against a tiny
    local asyncio HTTP server so we exercise the real httpx path
    end-to-end.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.services.ingest.pi.audio_reader import (
    FRAME_BYTES,
    AudioConnectError,
    AudioStreamError,
    _frame_chunks,
    _validate_content_type,
    read_pcm_audio,
)

# ─── Content-Type validation ─────────────────────────────────────


def test_validate_accepts_audio_pcm() -> None:
    _validate_content_type("audio/pcm")
    _validate_content_type("audio/pcm; rate=16000; channels=1")


def test_validate_accepts_audio_l16() -> None:
    _validate_content_type("audio/L16; rate=16000")


def test_validate_accepts_x_pcm_variant() -> None:
    _validate_content_type("audio/x-pcm")


def test_validate_rejects_wrong_mime() -> None:
    with pytest.raises(AudioConnectError, match="Unexpected Content-Type"):
        _validate_content_type("audio/mpeg")


def test_validate_rejects_empty() -> None:
    with pytest.raises(AudioConnectError):
        _validate_content_type("")


# ─── _frame_chunks reframer ──────────────────────────────────────


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for c in chunks:
        yield c


async def test_reframe_yields_one_frame_when_input_is_exact_frame_size() -> None:
    payload = b"\x00\x01" * (FRAME_BYTES // 2)
    out = [
        f async for f in _frame_chunks(_chunks(payload), frame_bytes=FRAME_BYTES)
    ]
    assert out == [payload]


async def test_reframe_emits_multiple_frames_from_one_chunk() -> None:
    payload = b"\xaa" * (FRAME_BYTES * 3)
    out = [
        f async for f in _frame_chunks(_chunks(payload), frame_bytes=FRAME_BYTES)
    ]
    assert len(out) == 3
    assert all(len(f) == FRAME_BYTES for f in out)
    assert b"".join(out) == payload


async def test_reframe_reassembles_across_chunk_boundaries() -> None:
    payload = bytes(range(256)) * 8  # 2048 bytes
    # Split arbitrarily into uneven pieces.
    a, b, c = payload[:100], payload[100:1200], payload[1200:]
    frames = [
        f async for f in _frame_chunks(_chunks(a, b, c), frame_bytes=320)
    ]
    # 2048 // 320 = 6 full frames, 128 bytes remainder dropped.
    assert len(frames) == 6
    # Concatenated frames must equal the first 6*320=1920 bytes of payload.
    assert b"".join(frames) == payload[: 6 * 320]


async def test_reframe_drops_partial_trailing_frame() -> None:
    """A partial frame at stream end cannot be sent to LiveKit's
    audio encoder, so it must be dropped silently.
    """
    payload = b"\x01" * (FRAME_BYTES + 5)  # one full frame + 5 trailing bytes
    out = [
        f async for f in _frame_chunks(_chunks(payload), frame_bytes=FRAME_BYTES)
    ]
    assert len(out) == 1
    assert len(out[0]) == FRAME_BYTES


async def test_reframe_byte_at_a_time_input_still_works() -> None:
    payload = b"\xff" * (FRAME_BYTES * 2)
    one_byte_chunks = [bytes([b]) for b in payload]
    frames = [
        f async for f in _frame_chunks(_chunks(*one_byte_chunks), frame_bytes=FRAME_BYTES)
    ]
    assert len(frames) == 2
    assert b"".join(frames) == payload


async def test_reframe_rejects_zero_frame_bytes() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        async for _ in _frame_chunks(_chunks(), frame_bytes=0):
            pass


# ─── read_pcm_audio end-to-end via local server ──────────────────


async def _spawn_pcm_server(
    body_chunks: list[bytes],
    *,
    status: int = 200,
    content_type: str | None = "audio/pcm; rate=16000; channels=1",
) -> tuple[str, int, asyncio.Server]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
            status_line = f"HTTP/1.1 {status} OK\r\n".encode()
            headers = b""
            if content_type is not None:
                headers += f"Content-Type: {content_type}\r\n".encode()
            headers += b"\r\n"
            writer.write(status_line + headers)
            await writer.drain()
            for chunk in body_chunks:
                writer.write(chunk)
                await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return host, port, server


async def test_read_pcm_audio_yields_frames_against_real_server() -> None:
    payload = b"\x42" * (FRAME_BYTES * 5)
    host, port, server = await _spawn_pcm_server([payload])
    try:
        url = f"http://{host}:{port}/audio.pcm"
        frames: list[bytes] = []
        async with read_pcm_audio(url) as stream:
            async for frame in stream:
                frames.append(frame)
                if len(frames) == 5:
                    break
        assert len(frames) == 5
        assert all(len(f) == FRAME_BYTES for f in frames)
        assert b"".join(frames) == payload
    finally:
        server.close()
        await server.wait_closed()


async def test_read_pcm_audio_raises_connect_error_on_refused() -> None:
    with pytest.raises(AudioConnectError, match="Failed to connect"):
        async with read_pcm_audio(
            "http://127.0.0.1:1/", connect_timeout=0.5, read_timeout=0.5
        ) as frames:
            async for _ in frames:
                pass


async def test_read_pcm_audio_raises_connect_error_on_non_200() -> None:
    host, port, server = await _spawn_pcm_server(
        [], status=404, content_type="text/plain"
    )
    try:
        url = f"http://{host}:{port}/missing"
        with pytest.raises(AudioConnectError, match="HTTP 404"):
            async with read_pcm_audio(url) as frames:
                async for _ in frames:
                    pass
    finally:
        server.close()
        await server.wait_closed()


async def test_read_pcm_audio_raises_connect_error_on_wrong_content_type() -> None:
    host, port, server = await _spawn_pcm_server(
        [b"some bytes"], content_type="audio/mpeg"
    )
    try:
        url = f"http://{host}:{port}/stream"
        with pytest.raises(AudioConnectError, match="Unexpected Content-Type"):
            async with read_pcm_audio(url) as frames:
                async for _ in frames:
                    pass
    finally:
        server.close()
        await server.wait_closed()


# ─── error hierarchy ─────────────────────────────────────────────


def test_connect_error_inherits_audio_stream_error() -> None:
    """Manager-side reconnect logic catches AudioStreamError; both
    error subclasses must be reachable through it.
    """
    assert issubclass(AudioConnectError, AudioStreamError)
