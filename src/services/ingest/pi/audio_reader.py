"""Async PCM-over-HTTP audio stream reader.

Counterpart to `mjpeg_reader` for audio. The Pi's audio daemon
serves raw PCM bytes via HTTP chunked transfer. We pull them and
yield them in frame-sized chunks suitable for direct hand-off to
`livekit.rtc.AudioSource.capture_frame`.

Audio format contract (Pi side must produce, backend assumes):

  - 16-bit signed little-endian PCM
  - Mono (1 channel)
  - 16 kHz sample rate
  - Served as `Content-Type: audio/pcm; rate=16000; channels=1`
    (the boundary marker is the rate parameter; we validate)

These match LiveKit's default audio track expectations and the
sample rate the existing iPhone-path RoomWorker already consumes
via `rtc.AudioStream(sample_rate=16000, num_channels=1)`. Matching
exactly means the detection pipeline's AudioBuffer treats Pi-source
audio identically to iPhone-source audio.

Frame size: 320 samples per frame (~20 ms at 16 kHz) = 640 bytes.
LiveKit's WebRTC opus encoder accepts 10/20/40 ms frames; 20 ms is
the conventional choice.

Failure model parallels `mjpeg_reader`: connect failures raise
`AudioConnectError`, stream interruptions (read timeout, RST)
raise `AudioStreamInterrupted`. Both inherit from
`AudioStreamError` so the relay can catch the unified category and
trigger reconnect.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

# 16-bit mono PCM at 16 kHz: each second is 16000 * 2 = 32 KB.
# 20 ms frame = 320 samples = 640 bytes.
SAMPLE_RATE_HZ: int = 16000
CHANNELS: int = 1
BYTES_PER_SAMPLE: int = 2
FRAME_DURATION_MS: int = 20
FRAME_SAMPLES: int = SAMPLE_RATE_HZ * FRAME_DURATION_MS // 1000
FRAME_BYTES: int = FRAME_SAMPLES * BYTES_PER_SAMPLE * CHANNELS

_HTTP_CHUNK_SIZE: int = 16 * 1024  # 0.5 second of audio at the chosen rate


class AudioStreamError(Exception):
    """Base for all audio reader failures."""


class AudioConnectError(AudioStreamError):
    """Failed to establish the audio stream.

    Raised for TCP/DNS failures, non-2xx responses, or a server that
    returns a Content-Type that does not parse as the expected PCM
    format.
    """


class AudioStreamInterrupted(AudioStreamError):
    """Established stream dropped mid-body (read timeout, RST,
    abrupt EOF). Triggers the manager's reconnect path.
    """


# Content-Type: audio/pcm; rate=16000; channels=1
# We accept additional parameters anywhere after the mime type, in
# any order, mirroring the leniency of the MJPEG reader's boundary
# parser.
_CONTENT_TYPE_RE = re.compile(
    r"""
    ^\s*audio/(?:pcm|L16|x-pcm)\s*
    (?:;\s*[^=;\s]+\s*=\s*(?:"[^"]*"|[^\s;]*)\s*)*
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _validate_content_type(content_type: str) -> None:
    """Raise if the Content-Type does not look like a PCM stream.
    We accept `audio/pcm`, `audio/L16`, and `audio/x-pcm` since
    different ALSA-based servers prefer different mime strings.
    The actual format (rate / channels / bit depth) is assumed to
    match `SAMPLE_RATE_HZ`, `CHANNELS`, `BYTES_PER_SAMPLE`; we
    document the requirement but do not enforce it at the wire
    level since most servers do not set those parameters reliably.
    """
    if _CONTENT_TYPE_RE.match(content_type) is None:
        raise AudioConnectError(
            f"Unexpected Content-Type for PCM audio: {content_type!r}"
        )


@asynccontextmanager
async def read_pcm_audio(
    url: str,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 5.0,
    frame_bytes: int = FRAME_BYTES,
) -> AsyncIterator[AsyncIterator[bytes]]:
    """Open a PCM-over-HTTP stream and yield fixed-size frames.

    Used as an async context manager so the underlying connection
    closes on exit:

        async with read_pcm_audio(url) as frames:
            async for pcm_frame in frames:
                ...  # 640 bytes by default

    Each yielded chunk is exactly `frame_bytes` bytes. Partial
    chunks at the end of the stream are dropped: a partial frame
    cannot be safely passed to LiveKit's audio encoder anyway.

    Raises:
        AudioConnectError: connect failed, non-2xx response, or the
            server's Content-Type is not PCM. Raised before the
            async context manager yields.
        AudioStreamInterrupted: established stream dropped mid-body.
            Surfaces during iteration of the yielded generator.
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
            raise AudioConnectError(
                f"Failed to connect to audio stream {url}: {exc}"
            ) from exc

        wrapped: AsyncIterator[bytes] | None = None
        try:
            if response.status_code != 200:
                await response.aclose()
                raise AudioConnectError(
                    f"Audio endpoint returned HTTP {response.status_code}"
                )
            content_type = response.headers.get("content-type", "")
            _validate_content_type(content_type)
            chunks = response.aiter_bytes(chunk_size=_HTTP_CHUNK_SIZE)
            wrapped = _frame_chunks(chunks, frame_bytes=frame_bytes)
            yield wrapped
        finally:
            if wrapped is not None:
                try:
                    await wrapped.aclose()  # type: ignore[attr-defined]
                except Exception:
                    pass
            await response.aclose()
    finally:
        await client.aclose()


async def _frame_chunks(
    chunks: AsyncIterator[bytes], *, frame_bytes: int,
) -> AsyncIterator[bytes]:
    """Re-frame arbitrary-size HTTP chunks into fixed `frame_bytes`
    chunks. Wraps `httpx` transport errors as `AudioStreamInterrupted`
    so callers see a single error hierarchy for reconnect logic.

    A partial trailing frame at stream end is silently dropped:
    LiveKit's audio encoder requires whole frames.
    """
    if frame_bytes <= 0:
        raise ValueError("frame_bytes must be positive")
    buf = bytearray()
    try:
        async for chunk in chunks:
            buf.extend(chunk)
            while len(buf) >= frame_bytes:
                yield bytes(buf[:frame_bytes])
                del buf[:frame_bytes]
    except httpx.HTTPError as exc:
        raise AudioStreamInterrupted(
            f"Audio stream interrupted mid-body: {exc!r}"
        ) from exc
