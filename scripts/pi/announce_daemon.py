#!/usr/bin/env python3
"""Nemara camera announce daemon. Runs on the Pi.

Sends a heartbeat to the backend's /v1/cameras/pi-announce endpoint
every `INTERVAL_SECONDS` seconds with this device's stable hardware
ID (WiFi NIC MAC) plus its current MJPEG URL (and optional audio
URL). The backend upserts a `pi_discoveries` row keyed by the
hardware ID; caregivers see the device in their "Add a camera"
picker until they bind it to a patient.

Configuration via env vars (loaded by systemd unit from
/etc/nemara/announce.env):

  NEMARA_BACKEND_URL     required, e.g. https://api.nemara.app
  NEMARA_CAMERA_STREAM_URL  required (Pi-local path served by stream.py)
                            e.g. http://<ip>:8000/stream.mjpg
  NEMARA_CAMERA_AUDIO_URL   optional, set when a USB mic is attached
                            e.g. http://<ip>:8001/audio.pcm
  NEMARA_CAMERA_LABEL_PREFIX optional, default "Nemara Camera"
                            label becomes "<prefix> <last-6-of-MAC>"
  NEMARA_ANNOUNCE_INTERVAL_S optional, default 60
  NEMARA_WIFI_IFACE      optional, default "wlan0"

The daemon resolves its own IP at every iteration so a DHCP lease
change pushes the new URL within one interval.

Failure model: a backend that returns non-2xx or is unreachable
logs a warning and tries again on the next interval. Never crashes
the process; systemd would restart anyway.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import sys
from pathlib import Path

import httpx

log = logging.getLogger("nemara.announce")


def _read_mac(iface: str) -> str:
    """Read the MAC address of `iface` from /sys. The MAC is stable
    across reboots (modulo MAC randomization, which we do NOT enable
    on Nemara cameras), so it is the right identity key.
    """
    path = Path(f"/sys/class/net/{iface}/address")
    return path.read_text().strip()


def _device_label(mac: str, prefix: str) -> str:
    """e.g. 'Nemara Camera A4B7C2' from MAC 'b8:27:eb:a4:b7:c2'."""
    suffix = mac.replace(":", "").upper()[-6:]
    return f"{prefix} {suffix}"


def _current_ip() -> str:
    """Best-effort own-IP discovery. Opens a UDP socket to a public
    address (without actually sending packets) and reads the local
    endpoint the kernel chose. Works offline because no DNS or
    routing happens, just the OS routing-table lookup.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("1.1.1.1", 80))  # routing target; nothing is sent
        return s.getsockname()[0]
    finally:
        s.close()


def _substitute_ip(template_url: str, ip: str) -> str:
    """If the configured URL contains `{ip}`, fill in the current
    address. This lets the operator write
    NEMARA_CAMERA_STREAM_URL=http://{ip}:8000/stream.mjpg and have
    the daemon refresh it on every announcement, surviving DHCP
    lease changes.
    """
    return template_url.replace("{ip}", ip)


async def announce_loop() -> None:
    backend_url = os.environ["NEMARA_BACKEND_URL"].rstrip("/")
    stream_url_tpl = os.environ["NEMARA_CAMERA_STREAM_URL"]
    audio_url_tpl = os.environ.get("NEMARA_CAMERA_AUDIO_URL")
    label_prefix = os.environ.get("NEMARA_CAMERA_LABEL_PREFIX", "Nemara Camera")
    interval_s = int(os.environ.get("NEMARA_ANNOUNCE_INTERVAL_S", "60"))
    iface = os.environ.get("NEMARA_WIFI_IFACE", "wlan0")

    mac = _read_mac(iface)
    label = _device_label(mac, label_prefix)
    log.info(
        "starting hardware_id=%s label=%r interval=%ds backend=%s",
        mac, label, interval_s, backend_url,
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        while True:
            try:
                ip = _current_ip()
                stream_url = _substitute_ip(stream_url_tpl, ip)
                audio_url = (
                    _substitute_ip(audio_url_tpl, ip) if audio_url_tpl else None
                )
                body = {
                    "hardware_id": mac,
                    "device_label": label,
                    "current_url": stream_url,
                }
                if audio_url is not None:
                    body["current_audio_url"] = audio_url

                r = await client.post(
                    f"{backend_url}/v1/cameras/pi-announce", json=body,
                )
                if r.status_code == 200:
                    log.info(
                        "announced ip=%s url=%s discovery_id=%s",
                        ip, stream_url, r.json().get("discovery_id"),
                    )
                else:
                    log.warning(
                        "announce non-200 status=%d body=%r",
                        r.status_code, r.text[:200],
                    )
            except Exception:
                log.exception("announce failed; will retry")

            await asyncio.sleep(interval_s)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("NEMARA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(announce_loop())
    except KeyboardInterrupt:
        log.info("stopped by signal")
        sys.exit(0)


if __name__ == "__main__":
    main()
