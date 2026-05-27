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
  NEMARA_CAMERA_QUALITY      optional JPEG quality 1..100, default 80
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
import logging
import os
import sys
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


def _build_camera(width: int, height: int) -> Picamera2:
    """Configure picamera2 for an MJPEG capture loop at the requested
    resolution. Returns the started camera; caller must `.stop()` /
    `.close()` it. Raises if libcamera/picamera2 are not installed.
    """
    from picamera2 import Picamera2

    cam = Picamera2()
    cfg = cam.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"},
    )
    cam.configure(cfg)
    cam.start()
    return cam


async def _capture_frames(
    cam: Picamera2,
    *,
    fps: int,
    quality: int,
    stop: asyncio.Event,
) -> AsyncIterator[bytes]:
    """Yield JPEG bytes from `cam` at the requested fps. Stops when
    `stop` is set.

    The capture runs in a worker thread so the asyncio event loop
    keeps servicing the HTTPS push. JPEG encoding happens in the
    same thread (Pillow); CPU cost is the dominant cost on a Zero W
    so we cap fps to keep the loop sustainable.
    """
    import io

    from PIL import Image

    def _encode_jpeg(buf_in, quality_in: int) -> bytes:
        """Run in a worker thread; encodes a captured buffer to JPEG
        bytes. Despite picamera2's "RGB888" format label, the buffer
        is actually laid out as BGR in memory (libcamera convention
        on little-endian platforms; picamera2's examples use cv2
        which expects BGR so this works for them). Pillow's "RGB"
        mode would treat the buffer as RGB, swapping R and B and
        producing an all-blue image of red objects. Reverse the last
        axis (channels) so what Pillow encodes is genuinely R, G, B.
        """
        out = io.BytesIO()
        Image.fromarray(buf_in[..., ::-1], "RGB").save(
            out, format="JPEG", quality=quality_in,
        )
        return out.getvalue()

    interval_s = 1.0 / max(1, fps)
    loop = asyncio.get_event_loop()

    while not stop.is_set():
        tick_start = loop.time()
        # capture_array returns a BGR888 numpy array under picamera2's
        # misleading "RGB888" format label; see _encode_jpeg for the
        # channel-swap rationale.
        buf = await asyncio.to_thread(cam.capture_array)
        jpeg_bytes = await asyncio.to_thread(_encode_jpeg, buf, quality)
        yield jpeg_bytes

        # Pace to the target fps. Skip if we already ran long.
        elapsed = loop.time() - tick_start
        slack = interval_s - elapsed
        if slack > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=slack)
                # stop was set during the sleep; exit on next iter
            except TimeoutError:
                pass


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
    quality: int,
) -> None:
    """Open one push connection and stream frames until the camera
    or the server closes it. Returns on clean exit (server closed,
    Pi shutdown); raises on transport error so the caller can back
    off and reconnect.
    """
    cam = _build_camera(width, height)
    stop = asyncio.Event()
    try:
        frames = _capture_frames(cam, fps=fps, quality=quality, stop=stop)
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
        stop.set()
        try:
            cam.stop()
        finally:
            cam.close()


async def main_loop() -> None:
    ingest_url = os.environ["NEMARA_INGEST_URL"]
    ingest_token = os.environ["NEMARA_INGEST_TOKEN"]
    width = int(os.environ.get("NEMARA_CAMERA_WIDTH", "1280"))
    height = int(os.environ.get("NEMARA_CAMERA_HEIGHT", "720"))
    fps = int(os.environ.get("NEMARA_CAMERA_FPS", "15"))
    quality = int(os.environ.get("NEMARA_CAMERA_QUALITY", "80"))
    base_backoff_s = float(os.environ.get("NEMARA_PUSH_RECONNECT_S", "2.0"))

    log.info(
        "starting pusher url=%s res=%dx%d fps=%d quality=%d",
        ingest_url, width, height, fps, quality,
    )

    backoff = base_backoff_s
    while True:
        try:
            await push_once(
                ingest_url=ingest_url,
                ingest_token=ingest_token,
                width=width, height=height,
                fps=fps, quality=quality,
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
