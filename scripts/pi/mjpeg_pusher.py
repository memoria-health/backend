#!/usr/bin/env python3
"""Memoria camera MJPEG pusher. Runs on the Pi.

Opens a long-lived HTTPS POST to the backend's
`/v1/cameras/me/frames` endpoint and streams JPEG frames captured
from the Pi camera. Authenticates with a per-camera bearer token
written to /etc/memoria/ingest.env by the announce daemon once
pairing completes.

Why outbound POST (not inbound MJPEG serving):
  Home routers NAT outbound but block inbound by default. A
  backend running in the cloud cannot dial the Pi at its LAN IP;
  the Pi has to dial the backend. Pi-side opens the TCP, server
  receives request body as the frame stream, no port forwarding
  needed.

Configuration via env vars (loaded by systemd unit from
/etc/memoria/camera-stream.env + /etc/memoria/ingest.env):

  MEMORIA_INGEST_URL          required, e.g. https://api.nemara.app/v1/cameras/me/frames
  MEMORIA_INGEST_TOKEN        required, plaintext bearer token
  MEMORIA_CAMERA_WIDTH        optional, default 1280
  MEMORIA_CAMERA_HEIGHT       optional, default 720
  MEMORIA_CAMERA_FPS          optional, default 15
  MEMORIA_CAMERA_SENSOR_WIDTH  optional; with _HEIGHT pins the raw sensor
  MEMORIA_CAMERA_SENSOR_HEIGHT mode to this resolution and lets the ISP
                             downscale to WIDTH x HEIGHT. Use the full
                             sensor size (IMX219: 3280x2464) to keep the
                             full field of view at 1080p — the sensor's
                             native 1080p readout is a center CROP. Unset
                             -> picamera2 picks the mode (binned full-FOV
                             for <=720p).
  MEMORIA_CAMERA_QUALITY_LEVEL optional MJPEG quality: VERY_LOW | LOW |
                             MEDIUM | HIGH | VERY_HIGH. Unset -> encoder
                             default (the validated 720p behavior).
  MEMORIA_CAMERA_BITRATE_BPS  read but currently INERT — bare
                             MJPEGEncoder() ignores it (passing bitrate=
                             yields blurry frames). Use
                             MEMORIA_CAMERA_QUALITY_LEVEL instead.
  MEMORIA_PUSH_RECONNECT_S    optional backoff base seconds, default 2.0
  MEMORIA_LOG_LEVEL           optional, default INFO

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

log = logging.getLogger("memoria.pusher")

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
    sensor_width: int = 0, sensor_height: int = 0,
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
        # When sensor_width/height are provided, pin the raw sensor mode
        # to that resolution and let the ISP downscale to the `main`
        # size. On the IMX219 the native 1920x1080 readout is a center
        # CROP (narrower FOV); requesting the full 3280x2464 sensor mode
        # and downscaling preserves the whole-room field of view. Full
        # resolution caps the IMX219 at ~15 fps. Left at 0, picamera2
        # picks its own mode (the binned 1640x1232 full-FOV mode for
        # <=720p), preserving the original behavior.
        config_kwargs: dict = {
            "main": {"size": (width, height)},
            "controls": {"FrameRate": float(fps)},
        }
        # Optional manual white balance. The stock AWB drifts green on the
        # Arducam 105° M12 lens; MEMORIA_CAMERA_AWB_GAINS="<red>,<blue>"
        # pins fixed ColourGains (and disables auto-AWB) — appropriate for
        # a fixed-mount camera and avoids AWB hunting. Unset -> auto AWB.
        _gains = os.environ.get("MEMORIA_CAMERA_AWB_GAINS", "").strip()
        if _gains:
            try:
                _r, _b = (float(x) for x in _gains.split(","))
                config_kwargs["controls"]["AwbEnable"] = False
                config_kwargs["controls"]["ColourGains"] = (_r, _b)
            except ValueError:
                log.warning("ignoring bad MEMORIA_CAMERA_AWB_GAINS=%r", _gains)
        # Optional ISP denoise strength. The stock "Fast" denoise is far too
        # aggressive for this low-gain (~2.0) Arducam IMX219 — it smears fine
        # detail (measured: lap-var 260 -> 71, ~73% of detail gone) and only
        # trades against grain that barely exists in a lit room. "Minimal"
        # behaves identically to "Fast" on the VC4 ISP, so the useful knob is
        # off vs a tuned profile. MEMORIA_CAMERA_DENOISE=off|minimal|fast|hq.
        # Unset -> camera default (Fast). Note: in genuine low light the gain
        # climbs and "off" gets noisy — prefer a tuned rpi.sdn for night use.
        _denoise = os.environ.get("MEMORIA_CAMERA_DENOISE", "").strip().lower()
        if _denoise:
            from libcamera import controls as _lc
            _nr = {
                "off": _lc.draft.NoiseReductionModeEnum.Off,
                "minimal": _lc.draft.NoiseReductionModeEnum.Minimal,
                "fast": _lc.draft.NoiseReductionModeEnum.Fast,
                "hq": _lc.draft.NoiseReductionModeEnum.HighQuality,
            }.get(_denoise)
            if _nr is not None:
                config_kwargs["controls"]["NoiseReductionMode"] = _nr
            else:
                log.warning("ignoring bad MEMORIA_CAMERA_DENOISE=%r", _denoise)
        if sensor_width > 0 and sensor_height > 0:
            config_kwargs["raw"] = {"size": (sensor_width, sensor_height)}
        cfg = cam.create_video_configuration(**config_kwargs)
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


_PUSH_RESULT_CLOSED = "closed"
_PUSH_RESULT_PAUSED = "paused"


def _resolve_quality(level: str | None):
    """Map a MEMORIA_CAMERA_QUALITY_LEVEL string to a picamera2 `Quality`
    enum, or None to use the encoder default.

    `Quality` is the SUPPORTED knob for trading MJPEG frame size against
    sharpness on the hardware encoder. Passing `bitrate=` to
    `MJPEGEncoder()` is the trap that maps to a very low
    compression_quality and produces blurry ~16 KB frames (see
    `_build_camera_and_encoder`). Returns None for unset/invalid so the
    caller falls back to `start_recording` without a quality override.
    """
    if not level:
        return None
    from picamera2.encoders import Quality
    try:
        return Quality[level.upper()]
    except KeyError:
        log.warning(
            "ignoring invalid MEMORIA_CAMERA_QUALITY_LEVEL=%r "
            "(want VERY_LOW|LOW|MEDIUM|HIGH|VERY_HIGH)", level,
        )
        return None


async def push_once(
    *,
    ingest_url: str,
    ingest_token: str,
    width: int,
    height: int,
    fps: int,
    bitrate_bps: int,
    sensor_width: int = 0,
    sensor_height: int = 0,
    quality_level: str | None = None,
) -> str:
    """Open one push connection and stream frames until the camera
    or the server closes it.

    Returns one of:
      - `_PUSH_RESULT_CLOSED`: server hung up normally; the caller
        should immediately reconnect with the existing backoff.
      - `_PUSH_RESULT_PAUSED`: server returned 423 indicating the
        caregiver hit Disconnect, the camera is hidden, or any
        other "stop pushing" signal. Caller MUST switch to paused
        mode (poll ingest-status until it flips back to connected)
        rather than reconnecting at full rate.

    Raises on transport error so the caller can back off and reconnect.
    """
    cam, encoder, output, ring = _build_camera_and_encoder(
        width, height, fps, bitrate_bps,
        sensor_width=sensor_width, sensor_height=sensor_height,
    )
    stop = asyncio.Event()
    quality = _resolve_quality(quality_level)
    if quality is not None:
        cam.start_recording(encoder, output, quality=quality)
    else:
        cam.start_recording(encoder, output)
    try:
        frames = _capture_frames(ring, stop=stop)
        body = _stream_body(frames)
        headers = {
            "Authorization": f"Bearer {ingest_token}",
            "Content-Type": f'multipart/x-mixed-replace; boundary="{_BOUNDARY.decode()}"',
            "User-Agent": "memoria-camera-pusher/1",
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
            if resp.status_code == 423:
                # The caregiver disconnected this camera (or some other
                # "do not relay" signal). Switch the outer loop to
                # paused mode so we stop hammering the push endpoint
                # at 15 Hz; the paused loop polls ingest-status at
                # ~5 s instead and resumes the moment the server
                # flips back to connected.
                log.info(
                    "server returned 423 paused (body=%s); switching to paused mode",
                    resp.text[:200],
                )
                return _PUSH_RESULT_PAUSED
            if resp.status_code >= 500:
                log.warning(
                    "server returned %d; will reconnect", resp.status_code,
                )
                return _PUSH_RESULT_CLOSED
            log.info("push closed cleanly status=%d", resp.status_code)
            return _PUSH_RESULT_CLOSED
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


def _derive_status_url(ingest_url: str) -> str:
    """Map the push endpoint URL to its sibling status endpoint.

    `ingest_url` is the full `https://.../v1/cameras/me/frames` value
    the backend handed us in ingest.env. The status endpoint shares
    everything except the trailing path segment. Keeping the
    derivation literal (string replace) avoids a second env var and
    a second source of truth — install.sh / announce_daemon only
    have to write the one URL.
    """
    if ingest_url.endswith("/frames"):
        return ingest_url[: -len("/frames")] + "/ingest-status"
    # Fall back to appending the new segment so we never crash if the
    # endpoint shape ever shifts; the backend will 404 us into the
    # main-loop reconnect cycle in that case.
    return ingest_url.rstrip("/") + "/ingest-status"


async def fetch_desired_state(
    *,
    status_url: str,
    ingest_token: str,
) -> str:
    """One-shot fetch of the camera's current desired_state via the
    lightweight ingest-status endpoint.

    Used by main_loop as a pre-flight gate before opening the
    streaming push. Without this gate the Pi opens a long-running
    multipart POST whose body httpx is still flushing while the
    server has already returned 423; httpx surfaces the early stream
    close as `httpx.ReadError` *before* it can parse the actual
    423 response, so the Pi never sees the paused signal and loops
    forever on connect+ReadError instead of switching modes. The
    cheap GET sidesteps that races by checking the state before we
    ever start streaming.

    Returns the string the backend returned ('connected' or
    'disconnected'). Raises on transport / auth / parse errors so
    main_loop can apply its normal back-off and try again.
    """
    headers = {
        "Authorization": f"Bearer {ingest_token}",
        "User-Agent": "memoria-camera-pusher/1",
    }
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(status_url, headers=headers)
        if resp.status_code == 401:
            log.error(
                "ingest-status returned 401; exiting for re-provision",
            )
            raise SystemExit(2)
        resp.raise_for_status()
        data = resp.json()
        state = data.get("desired_state")
        if state not in ("connected", "disconnected"):
            raise ValueError(f"unexpected desired_state: {state!r}")
        return state


async def wait_until_active(
    *,
    status_url: str,
    ingest_token: str,
    poll_interval_s: float,
) -> None:
    """Paused-mode loop. Polls the ingest-status endpoint at a low
    fixed cadence (default 5 s) and returns the moment the server
    reports `desired_state == 'connected'`.

    No camera is open here, no frames are encoded, no LiveKit
    egress. This is the zero-bandwidth idle state the user wants
    when they hit Disconnect on a Pi camera.
    """
    headers = {
        "Authorization": f"Bearer {ingest_token}",
        "User-Agent": "memoria-camera-pusher/1",
    }
    timeout = httpx.Timeout(connect=5.0, read=5.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            try:
                resp = await client.get(status_url, headers=headers)
            except httpx.HTTPError as exc:
                log.warning(
                    "paused-mode poll failed (%s); will retry in %.1fs",
                    exc, poll_interval_s,
                )
                await asyncio.sleep(poll_interval_s)
                continue
            if resp.status_code == 401:
                log.error(
                    "paused-mode poll got 401; exiting for re-provision",
                )
                raise SystemExit(2)
            if resp.status_code != 200:
                log.warning(
                    "paused-mode poll http=%d body=%s; will retry in %.1fs",
                    resp.status_code, resp.text[:200], poll_interval_s,
                )
                await asyncio.sleep(poll_interval_s)
                continue
            try:
                state = resp.json().get("desired_state")
            except Exception:
                log.warning(
                    "paused-mode poll body not JSON: %s; will retry",
                    resp.text[:200],
                )
                await asyncio.sleep(poll_interval_s)
                continue
            if state == "connected":
                log.info("paused-mode poll saw desired_state=connected; resuming")
                return
            # Still disconnected — quiet log to avoid spam, then sleep.
            log.debug("paused-mode poll desired_state=%s; sleeping", state)
            await asyncio.sleep(poll_interval_s)


async def main_loop() -> None:
    ingest_url = os.environ["MEMORIA_INGEST_URL"]
    ingest_token = os.environ["MEMORIA_INGEST_TOKEN"]
    status_url = _derive_status_url(ingest_url)
    width = int(os.environ.get("MEMORIA_CAMERA_WIDTH", "1280"))
    height = int(os.environ.get("MEMORIA_CAMERA_HEIGHT", "720"))
    fps = int(os.environ.get("MEMORIA_CAMERA_FPS", "15"))
    # Target bitrate for MJPEGEncoder. Default 4 Mbps gives clean
    # 720p; raise for sharper image at higher bandwidth cost.
    bitrate_bps = int(os.environ.get("MEMORIA_CAMERA_BITRATE_BPS", "4000000"))
    # Optional raw sensor mode (see _build_camera_and_encoder): set both
    # to the full sensor resolution to keep the full FOV at 1080p.
    sensor_width = int(os.environ.get("MEMORIA_CAMERA_SENSOR_WIDTH", "0"))
    sensor_height = int(os.environ.get("MEMORIA_CAMERA_SENSOR_HEIGHT", "0"))
    quality_level = os.environ.get("MEMORIA_CAMERA_QUALITY_LEVEL") or None
    base_backoff_s = float(os.environ.get("MEMORIA_PUSH_RECONNECT_S", "2.0"))
    paused_poll_s = float(os.environ.get("MEMORIA_PAUSED_POLL_S", "5.0"))

    log.info(
        "starting pusher url=%s status_url=%s res=%dx%d fps=%d "
        "sensor=%dx%d quality=%s",
        ingest_url, status_url, width, height, fps,
        sensor_width, sensor_height, quality_level or "default",
    )

    backoff = base_backoff_s
    while True:
        # Pre-flight gate: read desired_state before opening the
        # streaming push. The push endpoint also returns 423 when
        # disconnected, but the Pi can't reliably read that
        # response — see the docstring on `fetch_desired_state` for
        # the httpx-on-streaming-POST race that loses the 423.
        # Doing a cheap GET first lets us route into paused mode
        # cleanly without paying any frame-encoding cost.
        try:
            state = await fetch_desired_state(
                status_url=status_url, ingest_token=ingest_token,
            )
        except SystemExit:
            raise
        except Exception:
            log.exception(
                "desired-state pre-flight failed; sleeping %.1fs", backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
            continue

        if state == "disconnected":
            log.info("desired_state=disconnected; entering paused mode")
            backoff = base_backoff_s
            await wait_until_active(
                status_url=status_url,
                ingest_token=ingest_token,
                poll_interval_s=paused_poll_s,
            )
            # wait_until_active returned -> desired_state flipped to
            # connected. Loop back round, re-pre-check, then push.
            continue

        # state == "connected"
        try:
            result = await push_once(
                ingest_url=ingest_url,
                ingest_token=ingest_token,
                width=width, height=height,
                fps=fps, bitrate_bps=bitrate_bps,
                sensor_width=sensor_width, sensor_height=sensor_height,
                quality_level=quality_level,
            )
            backoff = base_backoff_s
            if result == _PUSH_RESULT_PAUSED:
                # Rare: the streaming POST DID surface 423 cleanly.
                # Drop straight into paused mode. The pre-flight will
                # also catch this on the next iteration, but cheaper
                # to react now.
                await wait_until_active(
                    status_url=status_url,
                    ingest_token=ingest_token,
                    poll_interval_s=paused_poll_s,
                )
            else:
                # Clean close. Reconnect immediately.
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
        level=os.environ.get("MEMORIA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        log.info("stopped by signal")
        sys.exit(0)


if __name__ == "__main__":
    main()
