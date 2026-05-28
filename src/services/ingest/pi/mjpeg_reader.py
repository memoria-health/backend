"""Async MJPEG-over-HTTP stream reader.

Pulls JPEG frames from a `multipart/x-mixed-replace` endpoint and
yields each frame as `bytes`. The Pi-side daemon
(`/home/pi/stream.py` on the camera Pi) emits one JPEG per part,
delimited by a CRLF + `--<boundary>` and headers carrying a
`Content-Length`. Content-Length is authoritative: we never scan
the JPEG payload for a delimiter, so a coincidental boundary
sequence inside the JPEG bytes is safe.

Separation of concerns:

  - `parse_mjpeg(chunks, *, boundary, max_frame_bytes)` is pure: it
    consumes an async byte iterator and yields JPEG frame bytes.
    Has no transport dependency. Unit-tested with hand-crafted byte
    streams.

  - `read_mjpeg(url, *, connect_timeout, read_timeout, max_frame_bytes)`
    is the transport wrapper: opens an `httpx.AsyncClient` stream,
    extracts the boundary from the response's Content-Type header,
    and delegates the body to `parse_mjpeg`. Integration-tested
    against a real Pi (gated on env var).

Error types:

  - `MjpegConnectError`: TCP refused, DNS failure, non-2xx response,
    wrong/missing Content-Type, missing boundary parameter.
  - `MjpegProtocolError`: stream body did not conform to multipart
    MJPEG (missing Content-Length, oversize frame, truncated body).

Both inherit from `MjpegStreamError` so callers can catch the
category they care about.

Cancellation: both layers are cancellation-safe. `httpx` closes the
underlying connection when its `stream(...)` context manager exits,
which happens automatically when the calling task is cancelled.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

# Default per-frame ceiling. A 720p JPEG is typically 30-80 KB, a 1080p
# JPEG 80-200 KB. 2 MB leaves comfortable headroom for any reasonable
# resolution and quality, and prevents a malicious or malformed server
# from inducing unbounded buffer growth.
DEFAULT_MAX_FRAME_BYTES: int = 2 * 1024 * 1024

# Default per-chunk read size. Larger reads reduce syscall count at
# the cost of latency-to-first-frame in pathological cases. 64 KiB
# is the conventional sweet spot for MJPEG and also matches httpx's
# default `aiter_bytes` chunk size, so we never request smaller chunks
# than the transport already produces.
_HTTP_CHUNK_SIZE: int = 64 * 1024


class MjpegStreamError(Exception):
    """Base class for all MJPEG reader failures."""


class MjpegConnectError(MjpegStreamError):
    """Failed to establish the stream.

    Raised for: TCP/DNS failures, HTTP non-2xx responses, missing or
    wrong Content-Type, missing boundary parameter.
    """


class MjpegProtocolError(MjpegStreamError):
    """Stream body did not conform to multipart MJPEG.

    Raised for: missing Content-Length on a part, Content-Length above
    `max_frame_bytes`, truncated body mid-frame, malformed header
    block.
    """


class MjpegStreamInterrupted(MjpegStreamError):
    """The connection was established and a stream began, then the
    underlying transport failed (read timeout, RST, abrupt EOF).

    Distinct from `MjpegConnectError` (never connected) and
    `MjpegProtocolError` (bytes arrived but were malformed). The
    relay layer translates this into a reconnect with backoff.
    """


# `multipart/x-mixed-replace` content type with a boundary parameter.
# Boundary values per RFC 2046 may be quoted or unquoted; both forms
# appear in the wild. We accept either.
# Content-Type may contain trailing parameters after `boundary=...`
# (e.g. `; charset=utf-8`). We require `multipart/x-mixed-replace` and
# a boundary parameter; additional parameters before or after the
# boundary are tolerated. We do not parse those extra parameters.
_CONTENT_TYPE_RE = re.compile(
    r"""
    ^\s*multipart/x-mixed-replace\s*
    (?:;\s*[^=;\s]+\s*=\s*(?:"[^"]*"|[^\s;]*)\s*)*
    ;\s*boundary\s*=\s*
    (?:"(?P<quoted>[^"]+)"|(?P<bare>[^\s;]+))
    \s*(?:;\s*[^=;\s]+\s*=\s*(?:"[^"]*"|[^\s;]*)\s*)*
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Header lines may end with CRLF (strict HTTP) or LF (lenient embedded
# servers). Split on either so a mixed or LF-only header block still
# yields one logical line per header.
_LINE_END_RE = re.compile(rb"\r\n|\n")


def parse_boundary(content_type: str) -> bytes:
    """Extract the multipart boundary from a Content-Type header value.

    Returns the boundary as bytes (without the leading `--`).
    Raises `MjpegConnectError` if the header is missing the
    `multipart/x-mixed-replace` mime type or the boundary parameter.
    """
    m = _CONTENT_TYPE_RE.match(content_type)
    if m is None:
        raise MjpegConnectError(
            f"Unexpected Content-Type for MJPEG stream: {content_type!r}"
        )
    raw = m.group("quoted") or m.group("bare")
    return raw.encode("ascii")


def parse_mjpeg(
    chunks: AsyncIterator[bytes],
    *,
    boundary: bytes,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> AsyncIterator[bytes]:
    """Parse a multipart MJPEG byte stream and yield each JPEG frame.

    `chunks` is an async iterator of arbitrary-size byte chunks
    (typically from `httpx.Response.aiter_bytes`). The function
    re-assembles frames across chunk boundaries using an internal
    `bytearray`.

    Frames are sized by their Content-Length header. The Content-Length
    is authoritative; we never scan the JPEG payload for a delimiter.

    Argument validation happens at call time (this function is a
    sync wrapper around the underlying async generator). Protocol
    errors are raised by the returned generator on iteration.

    Raises:
        ValueError: invalid `boundary` or `max_frame_bytes`, raised
            synchronously by the call itself.
        MjpegProtocolError: malformed body, raised when iterating the
            returned generator.
    """
    if max_frame_bytes <= 0:
        raise ValueError("max_frame_bytes must be positive")
    if not boundary:
        raise ValueError("boundary must be non-empty")
    return _parse_mjpeg_impl(
        chunks, boundary=boundary, max_frame_bytes=max_frame_bytes
    )


async def _parse_mjpeg_impl(
    chunks: AsyncIterator[bytes],
    *,
    boundary: bytes,
    max_frame_bytes: int,
) -> AsyncIterator[bytes]:
    """Real async generator. Caller via `parse_mjpeg` has already
    validated inputs.
    """
    delimiter = b"--" + boundary
    buf = bytearray()
    state: str = "SEEK_DELIMITER"
    frame_length: int = 0

    async for chunk in chunks:
        buf.extend(chunk)

        # Drive the state machine until we cannot make progress with
        # the bytes currently buffered, then fall back to waiting for
        # more chunks.
        while True:
            if state == "SEEK_DELIMITER":
                idx = buf.find(delimiter)
                if idx == -1:
                    # Keep a small tail to handle the case where the
                    # delimiter is split across the chunk boundary.
                    if len(buf) > len(delimiter):
                        del buf[: len(buf) - len(delimiter)]
                    break
                # Discard everything up to and including the delimiter.
                del buf[: idx + len(delimiter)]
                state = "AFTER_DELIMITER"

            elif state == "AFTER_DELIMITER":
                # After `--<boundary>`, the server sends either CRLF
                # (next part begins) or `--` (terminator: no more
                # parts). We need at least 2 bytes to decide.
                if len(buf) < 2:
                    break
                if buf[:2] == b"--":
                    # End-of-stream terminator. Treat as graceful end.
                    return
                if buf[:2] == b"\r\n":
                    del buf[:2]
                    state = "READ_HEADERS"
                    continue
                # Some servers emit only LF. Tolerate it.
                if buf[:1] == b"\n":
                    del buf[:1]
                    state = "READ_HEADERS"
                    continue
                # Anything else is malformed.
                raise MjpegProtocolError(
                    f"Unexpected bytes after boundary delimiter: "
                    f"{bytes(buf[:8])!r}"
                )

            elif state == "READ_HEADERS":
                # Find the blank line that terminates the header block.
                # Prefer CRLF CRLF, fall back to LF LF for tolerance.
                end = buf.find(b"\r\n\r\n")
                header_sep_len = 4
                if end == -1:
                    end = buf.find(b"\n\n")
                    header_sep_len = 2
                if end == -1:
                    # Headers can be large in pathological cases. Cap
                    # the header block size so a server that never
                    # sends the terminator cannot fill memory.
                    if len(buf) > 8 * 1024:
                        raise MjpegProtocolError(
                            "Header block exceeded 8 KiB without terminator"
                        )
                    break
                headers_raw = bytes(buf[:end])
                del buf[: end + header_sep_len]
                frame_length = _content_length_from_headers(headers_raw)
                if frame_length <= 0:
                    raise MjpegProtocolError(
                        f"Content-Length must be positive, got {frame_length}"
                    )
                if frame_length > max_frame_bytes:
                    raise MjpegProtocolError(
                        f"Frame size {frame_length} exceeds max_frame_bytes "
                        f"{max_frame_bytes}"
                    )
                state = "READ_FRAME"

            elif state == "READ_FRAME":
                if len(buf) < frame_length:
                    break
                frame = bytes(buf[:frame_length])
                del buf[:frame_length]
                state = "SEEK_DELIMITER"
                yield frame

            else:  # pragma: no cover -- defensive
                raise AssertionError(f"Unknown state {state!r}")

    # Stream ended. If we were mid-frame, that is a protocol error;
    # if we were waiting for the next delimiter, treat as graceful end.
    if state == "READ_FRAME":
        raise MjpegProtocolError(
            f"Stream ended mid-frame: had {len(buf)} of {frame_length} bytes"
        )
    if state == "READ_HEADERS":
        raise MjpegProtocolError(
            "Stream ended in the middle of a header block"
        )


def _content_length_from_headers(headers_raw: bytes) -> int:
    """Extract the Content-Length value from a header block.

    Header names are case-insensitive (RFC 7230). Tolerates both CRLF
    and LF-only line endings within the same block so the parser works
    against non-strict embedded servers.
    """
    for line in _LINE_END_RE.split(headers_raw):
        if b":" not in line:
            continue
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            try:
                return int(value.strip())
            except ValueError as exc:
                raise MjpegProtocolError(
                    f"Invalid Content-Length value: {value!r}"
                ) from exc
    raise MjpegProtocolError("Missing Content-Length header on MJPEG part")


async def _wrap_transport_errors(
    inner: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Re-raise httpx transport failures during iteration as
    `MjpegStreamInterrupted` so callers can catch a single error
    hierarchy (`MjpegStreamError`) for reconnect logic.

    `parse_mjpeg` itself raises `MjpegProtocolError` for malformed
    bytes. This wrapper catches transport-level failures that arise
    inside `aiter_bytes` (read timeout, abrupt EOF, RST) and that
    would otherwise leak through as raw httpx types.
    """
    try:
        async for frame in inner:
            yield frame
    except httpx.HTTPError as exc:
        raise MjpegStreamInterrupted(
            f"Stream interrupted mid-body: {exc!r}"
        ) from exc


@asynccontextmanager
async def read_mjpeg(
    url: str,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 10.0,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> AsyncIterator[AsyncIterator[bytes]]:
    """Open an MJPEG stream over HTTP and yield an async iterator of JPEG frames.

    Use as an async context manager so the underlying connection is
    closed on exit (including exceptions and task cancellation):

        async with read_mjpeg(url) as frames:
            async for jpeg_bytes in frames:
                ...

    `connect_timeout` bounds the initial TCP/TLS connect; `read_timeout`
    bounds the maximum gap between body chunks. A Pi that stops emitting
    frames for longer than `read_timeout` raises
    `MjpegStreamInterrupted` (NOT a raw httpx exception) so the caller's
    reconnect logic can catch the unified `MjpegStreamError` hierarchy.

    Raises:
        MjpegConnectError: connection failed, non-2xx response, or the
            server's Content-Type is not multipart MJPEG. Raised before
            the async context manager yields.
        MjpegProtocolError: body did not conform to multipart MJPEG.
            Raised by the inner generator; surfaces when the caller
            iterates the yielded async iterator.
        MjpegStreamInterrupted: established stream dropped mid-body
            (read timeout, RST, abrupt EOF). Also surfaces during
            iteration of the yielded generator.
    """
    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=5.0,
        pool=5.0,
    )
    client = httpx.AsyncClient(timeout=timeout)
    try:
        try:
            req = client.build_request("GET", url)
            response = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise MjpegConnectError(f"Failed to connect to {url}: {exc}") from exc

        wrapped: AsyncIterator[bytes] | None = None
        try:
            if response.status_code != 200:
                # Drain so the connection can be released cleanly.
                await response.aclose()
                raise MjpegConnectError(
                    f"MJPEG endpoint returned HTTP {response.status_code}"
                )
            content_type = response.headers.get("content-type", "")
            boundary = parse_boundary(content_type)
            chunks = response.aiter_bytes(chunk_size=_HTTP_CHUNK_SIZE)
            parsed = parse_mjpeg(
                chunks, boundary=boundary, max_frame_bytes=max_frame_bytes
            )
            wrapped = _wrap_transport_errors(parsed)
            yield wrapped
        finally:
            # Async generators rely on the event loop's finalizer to
            # run their cleanup, which can be delayed by several
            # seconds. Explicitly closing them here unblocks the
            # response close below promptly when the consumer breaks
            # early. `aclose()` is idempotent and safe to call on
            # already-closed generators.
            if wrapped is not None:
                try:
                    await wrapped.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass
            await response.aclose()
    finally:
        await client.aclose()
