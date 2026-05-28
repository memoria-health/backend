"""Unit tests for `services/ingest/pi/mjpeg_reader.py`.

The reader has two layers and we exercise both:

  1. `parse_mjpeg` — pure parser. Most tests live here. We hand-craft
     byte streams chunked at every interesting boundary (mid-header,
     mid-frame, one-byte-at-a-time) to verify the state machine
     re-assembles frames correctly regardless of how the bytes arrive
     from the transport.

  2. `read_mjpeg` + `parse_boundary` — transport wrapper. Tested with
     a tiny local asyncio HTTP server so we exercise the real httpx
     `aiter_bytes` path and the Content-Type → boundary extraction in
     a single end-to-end test. Connection-error and protocol-error
     paths are exercised by hitting unreachable hosts / serving
     malformed responses.

Why a real server rather than `httpx.MockTransport`: MockTransport
streaming responses with arbitrary custom byte chunks is fiddly and
the parser logic that matters most lives below the transport. A real
server costs ~10ms per test and gives us confidence that we are
parsing what the real transport actually delivers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.services.ingest.pi.mjpeg_reader import (
    DEFAULT_MAX_FRAME_BYTES,
    MjpegConnectError,
    MjpegProtocolError,
    MjpegStreamInterrupted,
    parse_boundary,
    parse_mjpeg,
    read_mjpeg,
)

# ─── parse_boundary ──────────────────────────────────────────────


def test_parse_boundary_extracts_bare_value() -> None:
    assert parse_boundary("multipart/x-mixed-replace; boundary=FRAME") == b"FRAME"


def test_parse_boundary_extracts_quoted_value() -> None:
    assert (
        parse_boundary('multipart/x-mixed-replace; boundary="my boundary"')
        == b"my boundary"
    )


def test_parse_boundary_is_case_insensitive_on_mime() -> None:
    assert (
        parse_boundary("Multipart/X-Mixed-Replace; boundary=X") == b"X"
    )


def test_parse_boundary_tolerates_extra_whitespace() -> None:
    assert (
        parse_boundary("  multipart/x-mixed-replace ;   boundary = TOKEN  ")
        == b"TOKEN"
    )


def test_parse_boundary_tolerates_trailing_parameter() -> None:
    """Some servers append `; charset=...` after the boundary. We must
    not reject these as malformed.
    """
    assert (
        parse_boundary("multipart/x-mixed-replace; boundary=FRAME; charset=utf-8")
        == b"FRAME"
    )


def test_parse_boundary_tolerates_leading_parameter() -> None:
    """Parameter order is not fixed by RFC 2046; tolerate boundary
    appearing after another parameter as well.
    """
    assert (
        parse_boundary("multipart/x-mixed-replace; charset=utf-8; boundary=FRAME")
        == b"FRAME"
    )


def test_parse_boundary_rejects_wrong_mime() -> None:
    with pytest.raises(MjpegConnectError, match="Unexpected Content-Type"):
        parse_boundary("image/jpeg")


def test_parse_boundary_rejects_missing_boundary() -> None:
    with pytest.raises(MjpegConnectError, match="Unexpected Content-Type"):
        parse_boundary("multipart/x-mixed-replace")


def test_parse_boundary_rejects_empty_string() -> None:
    with pytest.raises(MjpegConnectError):
        parse_boundary("")


# ─── parse_mjpeg helpers ─────────────────────────────────────────


def _frame_bytes(jpeg: bytes) -> bytes:
    """Build one MJPEG part for a given JPEG payload."""
    return (
        b"--FRAME\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n"
        b"\r\n"
        + jpeg
        + b"\r\n"
    )


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    """Build an async iterator over the given chunks. Each `yield` is
    distinct so the parser sees real chunk boundaries between them.
    """
    for c in chunks:
        yield c


# A minimal but plausible JPEG: SOI marker, an empty APP0, EOI marker.
# Decoders may reject it but the parser only cares about byte counts.
_JPEG_A = b"\xff\xd8" + b"PAYLOAD-A" + b"\xff\xd9"
_JPEG_B = b"\xff\xd8" + b"PAYLOAD-BB-LONGER" + b"\xff\xd9"


# ─── parse_mjpeg: single-chunk happy paths ───────────────────────


async def test_yields_one_frame_when_input_is_one_complete_part() -> None:
    stream = _chunks(_frame_bytes(_JPEG_A))
    frames = [f async for f in parse_mjpeg(stream, boundary=b"FRAME")]
    assert frames == [_JPEG_A]


async def test_yields_multiple_frames_from_a_single_chunk() -> None:
    blob = _frame_bytes(_JPEG_A) + _frame_bytes(_JPEG_B) + _frame_bytes(_JPEG_A)
    frames = [f async for f in parse_mjpeg(_chunks(blob), boundary=b"FRAME")]
    assert frames == [_JPEG_A, _JPEG_B, _JPEG_A]


async def test_yields_frames_when_each_part_arrives_in_its_own_chunk() -> None:
    stream = _chunks(_frame_bytes(_JPEG_A), _frame_bytes(_JPEG_B))
    frames = [f async for f in parse_mjpeg(stream, boundary=b"FRAME")]
    assert frames == [_JPEG_A, _JPEG_B]


# ─── parse_mjpeg: split-chunk reassembly ─────────────────────────


async def test_chunk_split_inside_delimiter_reassembles() -> None:
    full = _frame_bytes(_JPEG_A) + _frame_bytes(_JPEG_B)
    # Split right in the middle of the second delimiter.
    boundary_idx = full.find(b"--FRAME", 5)
    a, b = full[: boundary_idx + 3], full[boundary_idx + 3 :]
    frames = [f async for f in parse_mjpeg(_chunks(a, b), boundary=b"FRAME")]
    assert frames == [_JPEG_A, _JPEG_B]


async def test_chunk_split_inside_headers_reassembles() -> None:
    full = _frame_bytes(_JPEG_A)
    # Split inside the header block (after "Content-Type:").
    cut = full.index(b"image/jpeg")
    frames = [
        f async for f in parse_mjpeg(_chunks(full[:cut], full[cut:]), boundary=b"FRAME")
    ]
    assert frames == [_JPEG_A]


async def test_chunk_split_inside_frame_payload_reassembles() -> None:
    full = _frame_bytes(_JPEG_A)
    # Split somewhere inside the JPEG body.
    body_start = full.index(_JPEG_A)
    cut = body_start + 3
    frames = [
        f async for f in parse_mjpeg(_chunks(full[:cut], full[cut:]), boundary=b"FRAME")
    ]
    assert frames == [_JPEG_A]


async def test_byte_at_a_time_input_still_yields_frames() -> None:
    """Worst-case fragmentation: every chunk is exactly one byte."""
    blob = _frame_bytes(_JPEG_A) + _frame_bytes(_JPEG_B)
    one_byte_chunks = [bytes([byte]) for byte in blob]
    stream = _chunks(*one_byte_chunks)
    frames = [f async for f in parse_mjpeg(stream, boundary=b"FRAME")]
    assert frames == [_JPEG_A, _JPEG_B]


async def test_yields_exact_jpeg_bytes_with_no_mutation() -> None:
    """The JPEG payload bytes must be returned verbatim. We verify both
    the byte sequence and the length so a silent off-by-one corruption
    would be caught (the existing equality check on _JPEG_A/B is the
    primary assertion, but length is explicitly cross-checked here).
    """
    stream = _chunks(_frame_bytes(_JPEG_A))
    frames = [f async for f in parse_mjpeg(stream, boundary=b"FRAME")]
    assert len(frames) == 1
    assert frames[0] == _JPEG_A
    assert len(frames[0]) == len(_JPEG_A)
    # Returned value should be `bytes`, not `bytearray`, so downstream
    # consumers cannot accidentally mutate the parser's internal buffer.
    assert isinstance(frames[0], bytes)


# ─── parse_mjpeg: protocol-error paths ───────────────────────────


async def test_missing_content_length_raises() -> None:
    bad = (
        b"--FRAME\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"\r\n"
        + _JPEG_A
        + b"\r\n"
    )
    with pytest.raises(MjpegProtocolError, match="Missing Content-Length"):
        async for _ in parse_mjpeg(_chunks(bad), boundary=b"FRAME"):
            pass


async def test_invalid_content_length_raises() -> None:
    bad = (
        b"--FRAME\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: not-a-number\r\n"
        b"\r\n"
        + _JPEG_A
        + b"\r\n"
    )
    with pytest.raises(MjpegProtocolError, match="Invalid Content-Length"):
        async for _ in parse_mjpeg(_chunks(bad), boundary=b"FRAME"):
            pass


async def test_zero_content_length_raises() -> None:
    bad = (
        b"--FRAME\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: 0\r\n"
        b"\r\n\r\n"
    )
    with pytest.raises(MjpegProtocolError, match="must be positive"):
        async for _ in parse_mjpeg(_chunks(bad), boundary=b"FRAME"):
            pass


async def test_oversize_frame_raises_before_buffering() -> None:
    """The Content-Length check must fire before we accumulate the
    body, otherwise a malicious server could exhaust memory.
    """
    huge = (
        b"--FRAME\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"Content-Length: 99999999\r\n"
        b"\r\n"
    )
    with pytest.raises(MjpegProtocolError, match="exceeds max_frame_bytes"):
        async for _ in parse_mjpeg(
            _chunks(huge), boundary=b"FRAME", max_frame_bytes=1024
        ):
            pass


async def test_stream_ends_mid_frame_raises() -> None:
    full = _frame_bytes(_JPEG_A)
    truncated = full[: -10]  # Cut the JPEG short.
    with pytest.raises(MjpegProtocolError, match="ended mid-frame"):
        async for _ in parse_mjpeg(_chunks(truncated), boundary=b"FRAME"):
            pass


async def test_stream_ends_in_header_block_raises() -> None:
    truncated = b"--FRAME\r\nContent-Type: image/jpeg"
    with pytest.raises(MjpegProtocolError, match="middle of a header block"):
        async for _ in parse_mjpeg(_chunks(truncated), boundary=b"FRAME"):
            pass


async def test_oversize_header_block_raises() -> None:
    junk = b"X-Filler: " + b"a" * (8 * 1024 + 1) + b"\r\n"
    bad = b"--FRAME\r\n" + junk  # No header terminator, ever.
    with pytest.raises(MjpegProtocolError, match="Header block exceeded"):
        async for _ in parse_mjpeg(_chunks(bad), boundary=b"FRAME"):
            pass


# ─── parse_mjpeg: graceful termination ───────────────────────────


async def test_end_of_stream_terminator_completes_cleanly() -> None:
    """`--FRAME--` after a part is the multipart terminator. The parser
    treats it as a graceful end-of-stream, not an error.
    """
    blob = _frame_bytes(_JPEG_A) + b"--FRAME--\r\n"
    frames = [f async for f in parse_mjpeg(_chunks(blob), boundary=b"FRAME")]
    assert frames == [_JPEG_A]


async def test_empty_stream_yields_nothing_and_does_not_raise() -> None:
    frames = [f async for f in parse_mjpeg(_chunks(), boundary=b"FRAME")]
    assert frames == []


# ─── parse_mjpeg: input validation ───────────────────────────────


def test_zero_max_frame_bytes_rejected_at_call_time() -> None:
    """Validation must happen synchronously when `parse_mjpeg` is
    called, not deferred to the first iteration. This protects callers
    that build the generator object eagerly and pass it elsewhere
    (e.g. into the relay's transport-error wrapper).
    """
    with pytest.raises(ValueError, match="max_frame_bytes must be positive"):
        parse_mjpeg(_chunks(), boundary=b"FRAME", max_frame_bytes=0)


def test_negative_max_frame_bytes_rejected_at_call_time() -> None:
    with pytest.raises(ValueError, match="max_frame_bytes must be positive"):
        parse_mjpeg(_chunks(), boundary=b"FRAME", max_frame_bytes=-1)


def test_empty_boundary_rejected_at_call_time() -> None:
    with pytest.raises(ValueError, match="boundary must be non-empty"):
        parse_mjpeg(_chunks(), boundary=b"")


# ─── parse_mjpeg: tolerance for non-strict servers ───────────────


async def test_lf_only_line_endings_are_tolerated() -> None:
    """Some embedded servers ship LF-only line breaks. Tolerate them
    so we don't fail against well-meaning but non-strict servers.
    """
    bad = (
        b"--FRAME\n"
        b"Content-Type: image/jpeg\n"
        b"Content-Length: " + str(len(_JPEG_A)).encode() + b"\n"
        b"\n"
        + _JPEG_A
        + b"\n"
    )
    frames = [f async for f in parse_mjpeg(_chunks(bad), boundary=b"FRAME")]
    assert frames == [_JPEG_A]


# ─── read_mjpeg: end-to-end via local server ─────────────────────


async def _spawn_http_server(
    body_chunks: list[bytes],
    *,
    status: int = 200,
    content_type: str | None = "multipart/x-mixed-replace; boundary=FRAME",
    chunk_delay_s: float = 0.0,
) -> tuple[str, int, asyncio.Server]:
    """Start a one-shot HTTP server that responds to the first request
    with the given body chunks. Caller is responsible for closing the
    returned server. Returns (host, port, server).
    """

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            # Drain the request line + headers. We do not parse them.
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
                if chunk_delay_s > 0:
                    await asyncio.sleep(chunk_delay_s)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    return host, port, server


async def test_read_mjpeg_yields_frames_against_real_local_server() -> None:
    body = _frame_bytes(_JPEG_A) + _frame_bytes(_JPEG_B) + _frame_bytes(_JPEG_A)
    host, port, server = await _spawn_http_server([body])
    try:
        url = f"http://{host}:{port}/stream.mjpg"
        collected: list[bytes] = []
        async with read_mjpeg(url) as frames:
            async for jpeg in frames:
                collected.append(jpeg)
                if len(collected) == 3:
                    break
        assert collected == [_JPEG_A, _JPEG_B, _JPEG_A]
    finally:
        server.close()
        await server.wait_closed()


async def test_read_mjpeg_raises_connect_error_on_refused_connection() -> None:
    # 127.0.0.1:1 is reliably refused on every platform.
    with pytest.raises(MjpegConnectError, match="Failed to connect"):
        async with read_mjpeg(
            "http://127.0.0.1:1/", connect_timeout=0.5, read_timeout=0.5
        ) as frames:
            async for _ in frames:
                pass


async def test_read_mjpeg_raises_connect_error_on_non_200() -> None:
    host, port, server = await _spawn_http_server(
        [], status=404, content_type="text/plain"
    )
    try:
        url = f"http://{host}:{port}/missing"
        with pytest.raises(MjpegConnectError, match="HTTP 404"):
            async with read_mjpeg(url) as frames:
                async for _ in frames:
                    pass
    finally:
        server.close()
        await server.wait_closed()


async def test_read_mjpeg_raises_connect_error_on_wrong_content_type() -> None:
    host, port, server = await _spawn_http_server(
        [b"not an mjpeg stream"], content_type="text/plain"
    )
    try:
        url = f"http://{host}:{port}/stream"
        with pytest.raises(MjpegConnectError, match="Unexpected Content-Type"):
            async with read_mjpeg(url) as frames:
                async for _ in frames:
                    pass
    finally:
        server.close()
        await server.wait_closed()


async def test_read_mjpeg_raises_connect_error_on_missing_content_type() -> None:
    host, port, server = await _spawn_http_server([b""], content_type=None)
    try:
        url = f"http://{host}:{port}/stream"
        with pytest.raises(MjpegConnectError, match="Unexpected Content-Type"):
            async with read_mjpeg(url) as frames:
                async for _ in frames:
                    pass
    finally:
        server.close()
        await server.wait_closed()


async def test_read_mjpeg_protocol_error_surfaces_through_generator() -> None:
    """Protocol errors raised by the inner parser must propagate when
    the caller iterates the yielded async iterator, not silently
    swallowed.
    """
    # A part missing Content-Length triggers MjpegProtocolError.
    bad = (
        b"--FRAME\r\n"
        b"Content-Type: image/jpeg\r\n"
        b"\r\n"
        + _JPEG_A
        + b"\r\n"
    )
    host, port, server = await _spawn_http_server([bad])
    try:
        url = f"http://{host}:{port}/stream.mjpg"
        with pytest.raises(MjpegProtocolError, match="Missing Content-Length"):
            async with read_mjpeg(url) as frames:
                async for _ in frames:
                    pass
    finally:
        server.close()
        await server.wait_closed()


async def test_read_mjpeg_exits_promptly_when_caller_breaks_early() -> None:
    """Cancellation safety: when the caller breaks out of the
    `async for` loop, the surrounding `async with read_mjpeg(...)`
    context manager must run its cleanup (response.aclose +
    client.aclose) without blocking on remaining server output.

    Each test chunk is padded above httpx's 64 KiB internal chunk
    threshold so the consumer actually receives the first chunk
    promptly. A real Pi sends one ~50 KB JPEG per write, well above
    that threshold; tiny synthetic chunks would not surface to the
    consumer until httpx had accumulated 64 KiB, which would mask
    the cancellation property we want to verify.

    The server then drip-feeds 30 padded chunks at 100ms each = 3s
    total if iterated. If cleanup were waiting for the server to
    finish, the test would take ~3s. We assert it completes in well
    under 1s, which proves cleanup is not blocking on the server.
    """
    bulky_frame = _frame_bytes(_JPEG_A) + b"X" * (96 * 1024)
    body_chunks = [bulky_frame for _ in range(30)]
    host, port, server = await _spawn_http_server(
        body_chunks, chunk_delay_s=0.1
    )
    try:
        url = f"http://{host}:{port}/stream.mjpg"

        async def _consume_once() -> None:
            async with read_mjpeg(url) as frames:
                async for _ in frames:
                    break

        await asyncio.wait_for(_consume_once(), timeout=1.0)
    except TimeoutError:
        pytest.fail(
            "read_mjpeg context manager did not exit within 1s of "
            "caller break; cleanup is hanging or connection is leaking"
        )
    finally:
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=5.0)
        except TimeoutError:
            pass


async def test_read_mjpeg_wraps_read_timeout_as_stream_interrupted() -> None:
    """A Pi that stops emitting frames mid-stream must surface as
    `MjpegStreamInterrupted`, not as a raw `httpx.ReadTimeout`.
    This keeps the relay layer's reconnect logic simple: catch
    `MjpegStreamError`, back off, reconnect.
    """
    # Send one valid frame, then stall indefinitely. With read_timeout
    # set to 200ms the second iteration will raise.
    one_frame = _frame_bytes(_JPEG_A)
    host, port, server = await _spawn_http_server(
        [one_frame],
        # After the first chunk write, the server sleeps far longer
        # than read_timeout, simulating a wedged Pi.
        chunk_delay_s=5.0,
    )
    try:
        url = f"http://{host}:{port}/stream.mjpg"
        with pytest.raises(MjpegStreamInterrupted, match="interrupted mid-body"):
            async with read_mjpeg(
                url, connect_timeout=1.0, read_timeout=0.2
            ) as frames:
                async for _ in frames:
                    # Will get the first frame, then block waiting for
                    # more bytes and time out.
                    continue
    finally:
        server.close()
        try:
            await asyncio.wait_for(server.wait_closed(), timeout=2.0)
        except TimeoutError:
            pass


# ─── DEFAULT_MAX_FRAME_BYTES sanity ──────────────────────────────


def test_default_max_frame_bytes_is_reasonable() -> None:
    """Sanity-check the published default so a future edit that
    accidentally zeros or sextuples it gets caught.
    """
    # 2 MiB is comfortably more than any realistic JPEG (1080p high
    # quality is well under 1 MiB) and small enough that a malformed
    # stream cannot exhaust process memory per frame.
    assert 1 * 1024 * 1024 <= DEFAULT_MAX_FRAME_BYTES <= 4 * 1024 * 1024
