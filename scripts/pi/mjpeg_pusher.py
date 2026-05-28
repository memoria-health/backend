#!/usr/bin/env python3
"""Nemara camera MJPEG pusher. Runs on the Pi.

Opens a long-lived HTTPS POST to the backend's
`/v1/cameras/me/frames` endpoint and streams JPEG frames captured
from the Pi camera. Authenticates with a per-camera bearer token
written to /etc/nemara/ingest.env by the announce daemon once
pairing completes.

Why outbound POST (not inbound MJPEG serving):
  Home routers NAT outbound but block inbound by default. A
  backend running in the cloud cannot dial the Pi at its LAN IP;
  the Pi has to dial the backend. Pi-side opens the TCP, server
  receives request body as the frame stream, no port forwarding
  needed.

Configuration via env vars (loaded by systemd unit from
/etc/nemara/camera-stream.env + /etc/nemara/ingest.env):

  NEMARA_INGEST_URL          required, e.g. https://api.nemara.app/v1/cameras/me/frames
  NEMARA_INGEST_TOKEN        required, plaintext bearer token
  NEMARA_CAMERA_WIDTH        optional, default 1280
  NEMARA_CAMERA_HEIGHT       optional, default 720
  NEMARA_CAMERA_FPS          optional, default 15
  NEMARA_CAMERA_BITRATE_BPS  MJPEGEncoder target bitrate, default 4000000
                             (4 Mbps; rough guide for 720p: 4M clean,
                             6M sharp, 8M visually lossless)
  NEMARA_PUSH_RECONNECT_S    optional backoff base seconds, default 2.0
  NEMARA_LOG_LEVEL           optional, default INFO

Failure model:
  - Network drop / 5xx -> sleep `reconnect_s` then reconnect (with
    multiplicative backoff capped at 30 s).
  - 401 -> the stored token is stale (Pi was unpaired, or the
    caregiver re-provisioned). Exit; systemd Restart=on-failure
    will retry, and the announce daemon will eventually write a
    fresh token.
  - Camera open failure -> exit non-zero; systemd restarts.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    # picamera2 is a Pi-only dep; importing it here keeps mypy /
    # local-dev type-checking working on Linux/macOS without it.
    from picamera2 import Picamera2

log = logging.getLogger("nemara.pusher")

# MIME boundary string used in the multipart body. Arbitrary token
# that does not appear in JPEG payloads (JPEGs always begin with
# 0xFF 0xD8 0xFF and never contain the literal --frame--).
_BOUNDARY = b"frame"

# Cap for the reconnect-backoff multiplier. 30 s keeps us responsive
# when a transient backend issue resolves while not hammering the
# server during a longer outage.
_MAX_BACKOFF_S = 30.0


class _JpegRing(io.BufferedIOBase):
    """Thread-safe single-slot buffer that holds the most-recent JPEG
    frame produced by picamera2's hardware JpegEncoder.

    picamera2's `FileOutput.fileoutput` setter does
    `isinstance(file, io.BufferedIOBase)`, so this MUST inherit from
    that class (a plain class with the same methods raises
    `Must pass io.BufferedIOBase`). Only `write` is actually called
    by FileOutput; the rest of the ABC is satisfied by the default
    raise-UnsupportedOperation implementations.

    picamera2 calls `write(buf)` from the encoder thread for each
    encoded frame. The async push loop calls `get()` from the asyncio
    thread when it is ready to send a frame. The single-slot design
    drops older frames if the consumer is slower than the producer
    (real-time streaming, not buffered playback). This is exactly the
    behavior we want for a live camera: never queue up stale frames.
    """

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: bytes | None = None
        self._closed = False

    def writable(self) -> bool:
        return True

    def write(self, buf: bytes) -> int:  # type: ignore[override]
        """File-like sink for picamera2's FileOutput. Called on the
        encoder thread.
        """
        with self._cond:
            self._latest = bytes(buf)
            self._cond.notify_all()
        return len(buf)

    def flush(self) -> None:  # FileOutput contract
        pass

    def close(self) -> None:  # type: ignore[override]
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        super().close()

    def get(self, timeout_s: float = 2.0) -> bytes | None:
        """Block (with timeout) until a frame is available; return it
        and clear the slot so the next call waits for a fresh frame.
        Returns None on timeout or after close().
        """
        with self._cond:
            if self._latest is None and not self._closed:
                self._cond.wait(timeout=timeout_s)
            if self._closed:
                return None
            frame = self._latest
            self._latest = None
            return frame


def _build_camera_and_encoder(
    width: int, height: int, fps: int, bitrate_bps: int,
):
    """Configure picamera2 with the hardware JpegEncoder feeding a
    `_JpegRing`. The hardware encoder uses the SoC's ISP for the
    JPEG step, dropping per-frame CPU cost from "saturate the core"
    (Pillow) to "single-digit percent" (hardware).

    Returns `(camera, encoder, output, ring)`. Caller starts +
    stops via `cam.start_recording(encoder, output)` /
    `cam.stop_recording()`.
    """
    from picamera2 import Picamera2
    from picamera2.encoders import MJPEGEncoder
    from picamera2.outputs import FileOutput

    cam = Picamera2()
    # CRITICAL: if anything below this line raises, we must close the
    # Picamera2 instance, or the camera device stays bound to this
    # process and subsequent retries fail with `Device or resource
    # busy` until the whole pusher process exits. The outer try/
    # finally in push_once only runs if this function returns.
    try:
        # MJPEGEncoder uses the SoC's V4L2 hardware encoder
        # (single-digit CPU%). picamera2 also has `JpegEncoder` whose
        # name suggests hardware too, but is actually software
        # libjpeg-turbo on the CPU and cannot sustain 15 fps at 720p
        # on a Zero W. We hit ~3 fps with JpegEncoder before this
        # swap; MJPEGEncoder gets back to the 15 fps the old serve-
        # mode script achieved.
        #
        # `FrameRate` control pins the sensor to the requested fps
        # (matches the old stream.py's working config). The encoder
        # then emits at the sensor rate.
        cfg = cam.create_video_configuration(
            main={"size": (width, height)},
            controls={"FrameRate": float(fps)},
        )
        cam.configure(cfg)
        # MJPEGEncoder() with no args is what the original
        # nemara-cam-viewer/stream.py used and produced clean
        # ~50 KB JPEGs at 720p15 (well within Zero W WiFi budget).
        # DO NOT pass `bitrate=...`. Passing bitrate makes picamera2
        # translate it to the V4L2 driver's `compression_quality`
        # control via a low-value path, producing ~16 KB blurry
        # frames at quality~20. Bare MJPEGEncoder() uses the
        # picamera2 default mapping which sets compression_quality
        # to a sensible value (~60-70).
        encoder = MJPEGEncoder()
        ring = _JpegRing()
        output = FileOutput(ring)
    except Exception:
        try:
            cam.close()
        except Exception:
            pass
        raise
    return cam, encoder, output, ring


async def _capture_frames(
    ring: "_JpegRing",
    *,
    stop: asyncio.Event,
) -> AsyncIterator[bytes]:
    """Pull most-recent JPEG frames from the encoder ring. The
    hardware encoder runs at the camera's fps; this loop only yields
    when a fresh frame is in the slot. If the consumer falls behind
    (slow network upload), frames are dropped silently inside the
    ring rather than queued.

    Each ring.get() runs in a worker thread because it blocks on a
    threading.Condition; doing it inline would block the event loop.

    Logs throughput (frames/sec + average JPEG size) every 5s so we
    can see if the bottleneck is the encoder rate, the network, or
    the backend's consumer side independently.
    """
    loop = asyncio.get_event_loop()
    bucket_count = 0
    bucket_bytes = 0
    bucket_start = loop.time()
    log_interval_s = 5.0
    while not stop.is_set():
        frame = await asyncio.to_thread(ring.get, 2.0)
        if frame is None:
            # Timeout or ring closed. Loop back; outer stop check
            # will exit cleanly on shutdown.
            continue
        bucket_count += 1
        bucket_bytes += len(frame)
        now = loop.time()
        elapsed = now - bucket_start
        if elapsed >= log_interval_s:
            avg_size_kb = (bucket_bytes / bucket_count) / 1024.0
            log.info(
                "fps source=%.2f avg_jpeg_kb=%.1f bytes_per_s_kb=%.1f",
                bucket_count / elapsed,
                avg_size_kb,
                (bucket_bytes / elapsed) / 1024.0,
            )
            bucket_count = 0
            bucket_bytes = 0
            bucket_start = now
        yield frame


async def _stream_body(
    frame_source: AsyncIterator[bytes],
) -> AsyncIterator[bytes]:
    """Wrap an async iterator of JPEG bytes into an async iterator
    of multipart-MJPEG body chunks. Each yielded chunk is one full
    multipart part:

        --frame\r\n
        Content-Type: image/jpeg\r\n
        Content-Length: <N>\r\n
        \r\n
        <N bytes of JPEG>\r\n
    """
    async for jpeg in frame_source:
        header = (
            b"--" + _BOUNDARY + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n"
            b"\r\n"
        )
        yield header + jpeg + b"\r\n"
    # Closing boundary so the server sees a clean end if/when the
    # frame source exits cleanly (vs the connection being dropped).
    yield b"--" + _BOUNDARY + b"--\r\n"


async def push_once(
    *,
    ingest_url: str,
    ingest_token: str,
    width: int,
    height: int,
    fps: int,
    bitrate_bps: int,
) -> None:
    """Open one push connection and stream frames until the camera
    or the server closes it. Returns on clean exit (server closed,
    Pi shutdown); raises on transport error so the caller can back
    off and reconnect.
    """
    cam, encoder, output, ring = _build_camera_and_encoder(
        width, height, fps, bitrate_bps,
    )
    stop = asyncio.Event()
    cam.start_recording(encoder, output)
    try:
        frames = _capture_frames(ring, stop=stop)
        body = _stream_body(frames)
        headers = {
            "Authorization": f"Bearer {ingest_token}",
            "Content-Type": f'multipart/x-mixed-replace; boundary="{_BOUNDARY.decode()}"',
            "User-Agent": "nemara-camera-pusher/1",
        }
        # No total timeout: this is a long-lived stream. Read/write
        # timeouts are also disabled because the connection stays
        # idle between frames at low fps. We rely on TCP-level
        # disconnect detection from httpx.
        timeout = httpx.Timeout(connect=10.0, read=None, write=None, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            log.info("opening push to %s", ingest_url)
            resp = await client.post(
                ingest_url, content=body, headers=headers,
            )
            if resp.status_code == 401:
                log.error("server rejected token (401); exiting for re-provision")
                # SystemExit so systemd's Restart=on-failure does NOT
                # kick us; we wait for the announce daemon to write a
                # fresh ingest.env.
                raise SystemExit(2)
            if resp.status_code >= 500:
                log.warning(
                    "server returned %d; will reconnect", resp.status_code,
                )
                return
            log.info("push closed cleanly status=%d", resp.status_code)
    finally:
        # picamera2's start_recording / stop_recording pair owns the
        # camera lifecycle: stop_recording calls stop() + releases
        # the encoder internally. Calling cam.stop() again leaves
        # libcamera in a `Configured` state instead of releasing
        # back to `Available`, and the next acquire() then fails with
        # "Device or resource busy". Just close() after
        # stop_recording. Wrap each step so a failure in one doesn't
        # leak the next.
        stop.set()
        ring.close()
        try:
            cam.stop_recording()
        except Exception:
            log.exception("stop_recording failed during cleanup")
        try:
            cam.close()
        except Exception:
            log.exception("close failed during cleanup")


async def main_loop() -> None:
    ingest_url = os.environ["NEMARA_INGEST_URL"]
    ingest_token = os.environ["NEMARA_INGEST_TOKEN"]
    width = int(os.environ.get("NEMARA_CAMERA_WIDTH", "1280"))
    height = int(os.environ.get("NEMARA_CAMERA_HEIGHT", "720"))
    fps = int(os.environ.get("NEMARA_CAMERA_FPS", "15"))
    # Target bitrate for MJPEGEncoder. Default 4 Mbps gives clean
    # 720p; raise for sharper image at higher bandwidth cost.
    bitrate_bps = int(os.environ.get("NEMARA_CAMERA_BITRATE_BPS", "4000000"))
    base_backoff_s = float(os.environ.get("NEMARA_PUSH_RECONNECT_S", "2.0"))

    log.info(
        "starting pusher url=%s res=%dx%d fps=%d bitrate_mbps=%.1f",
        ingest_url, width, height, fps, bitrate_bps / 1_000_000,
    )

    backoff = base_backoff_s
    while True:
        try:
            await push_once(
                ingest_url=ingest_url,
                ingest_token=ingest_token,
                width=width, height=height,
                fps=fps, bitrate_bps=bitrate_bps,
            )
            # Clean exit: server closed gracefully. Reset backoff
            # and reconnect immediately.
            backoff = base_backoff_s
            await asyncio.sleep(0.5)
        except SystemExit:
            raise
        except Exception:
            log.exception(
                "push errored; sleeping %.1fs before reconnect", backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, _MAX_BACKOFF_S)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("NEMARA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        log.info("stopped by signal")
        sys.exit(0)


if __name__ == "__main__":
    main()
