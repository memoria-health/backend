#!/usr/bin/env python3
"""Nemara camera announce daemon. Runs on the Pi.

Sends a heartbeat to the backend's /v1/cameras/pi-announce endpoint
every `INTERVAL_SECONDS` seconds with this device's stable hardware
ID (WiFi NIC MAC). The backend upserts a `pi_discoveries` row
keyed by the hardware ID; caregivers see the device in their "Add
a camera" picker until they bind it to a patient.

Once paired, the backend's announce response includes the
`ingest_url` + `ingest_token` (delivered exactly once). This daemon
writes those values to /etc/nemara/ingest.env atomically and
triggers the `nemara-camera-stream.service` unit, which then opens
the long-lived push connection.

Configuration via env vars (loaded by systemd unit from
/etc/nemara/announce.env):

  NEMARA_BACKEND_URL           required, e.g. https://api.nemara.app
  NEMARA_CAMERA_STREAM_URL     optional (vestigial; only used by the
                               legacy pull-mode path)
  NEMARA_CAMERA_AUDIO_URL      optional, ditto
  NEMARA_CAMERA_LABEL_PREFIX   optional, default "Nemara Camera"
                               label becomes "<prefix> <last-6-of-MAC>"
  NEMARA_ANNOUNCE_INTERVAL_S   optional, default 60
  NEMARA_WIFI_IFACE            optional, default "wlan0"
  NEMARA_INGEST_ENV_PATH       optional, default /etc/nemara/ingest.env
  NEMARA_STREAM_SERVICE_NAME   optional, default nemara-camera-stream.service

Failure model: a backend that returns non-2xx or is unreachable
logs a warning and tries again on the next interval. Never crashes
the process; systemd would restart anyway.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

log = logging.getLogger("nemara.announce")

# Default file the announce daemon writes when it receives a freshly-
# minted ingest token in the announce response. The camera-stream
# systemd unit reads this file at start to learn its bearer + URL.
DEFAULT_INGEST_ENV_PATH = "/etc/nemara/ingest.env"

# Default systemd unit name that consumes the ingest.env. Triggering
# it after writing the env file is what flips a paired Pi from idle
# (just announcing) to actively pushing frames.
DEFAULT_STREAM_SERVICE = "nemara-camera-stream.service"


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


def _write_ingest_env_atomic(
    path: str, ingest_url: str, ingest_token: str,
) -> None:
    """Write ingest_url + ingest_token to `path` atomically.

    The camera-stream systemd unit re-reads this file every time it
    (re)starts. Atomic write via temp-file + os.replace prevents the
    stream unit from observing a half-written file if it happens to
    start mid-write.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # We escape both values; the systemd EnvironmentFile parser
    # tolerates quoted values, and the token uses URL-safe-base64
    # which is shell-safe anyway. Quoting is defense-in-depth.
    contents = (
        f"NEMARA_INGEST_URL={shlex.quote(ingest_url)}\n"
        f"NEMARA_INGEST_TOKEN={shlex.quote(ingest_token)}\n"
    )
    fd, tmp_path = tempfile.mkstemp(
        prefix=".ingest.env.", dir=str(target.parent), text=True,
    )
    try:
        with os.fdopen(fd, "w") as f:
            f.write(contents)
        os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, target)
    except Exception:
        # If we crashed before os.replace, the tmp file is dangling.
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def _systemctl_restart(unit: str) -> None:
    """Best-effort `systemctl restart <unit>`. Logs and swallows any
    failure (e.g. the unit isn't installed yet, or systemd isn't
    available in dev). The next announce will try again.
    """
    try:
        subprocess.run(
            ["systemctl", "restart", unit],
            check=True, timeout=10,
            capture_output=True, text=True,
        )
        log.info("triggered %s", unit)
    except subprocess.CalledProcessError as exc:
        log.warning(
            "systemctl restart %s failed rc=%s stderr=%r",
            unit, exc.returncode, exc.stderr[:200] if exc.stderr else "",
        )
    except Exception:
        log.exception("systemctl restart %s failed", unit)


async def announce_loop() -> None:
    backend_url = os.environ["NEMARA_BACKEND_URL"].rstrip("/")
    label_prefix = os.environ.get("NEMARA_CAMERA_LABEL_PREFIX", "Nemara Camera")
    interval_s = int(os.environ.get("NEMARA_ANNOUNCE_INTERVAL_S", "60"))
    iface = os.environ.get("NEMARA_WIFI_IFACE", "wlan0")
    # Vestigial pull-mode values: kept in env for backwards compat
    # but the backend ignores them after the push refactor.
    stream_url_tpl = os.environ.get("NEMARA_CAMERA_STREAM_URL", "")
    audio_url_tpl = os.environ.get("NEMARA_CAMERA_AUDIO_URL")
    # New push-mode handoff: where to write the ingest token and
    # which systemd unit to kick after writing it.
    ingest_env_path = os.environ.get(
        "NEMARA_INGEST_ENV_PATH", DEFAULT_INGEST_ENV_PATH,
    )
    stream_service = os.environ.get(
        "NEMARA_STREAM_SERVICE_NAME", DEFAULT_STREAM_SERVICE,
    )

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
                # current_url is required by the announce schema even
                # though it is vestigial. Use a placeholder if no
                # stream URL is configured locally.
                if stream_url_tpl:
                    current_url = _substitute_ip(stream_url_tpl, ip)
                else:
                    current_url = f"http://{ip}/"
                audio_url = (
                    _substitute_ip(audio_url_tpl, ip) if audio_url_tpl else None
                )
                body = {
                    "hardware_id": mac,
                    "device_label": label,
                    "current_url": current_url,
                }
                if audio_url is not None:
                    body["current_audio_url"] = audio_url

                r = await client.post(
                    f"{backend_url}/v1/cameras/pi-announce", json=body,
                )
                if r.status_code == 200:
                    data = r.json()
                    log.info(
                        "announced ip=%s discovery_id=%s paired=%s "
                        "has_token=%s",
                        ip, data.get("discovery_id"),
                        data.get("paired"),
                        bool(data.get("ingest_token")),
                    )
                    # One-shot token delivery: backend returns the
                    # plaintext exactly once when the camera was just
                    # paired. Persist it and (re)start the stream
                    # unit so the push connection picks it up.
                    ingest_token = data.get("ingest_token")
                    ingest_url = data.get("ingest_url")
                    if ingest_token and ingest_url:
                        try:
                            _write_ingest_env_atomic(
                                ingest_env_path, ingest_url, ingest_token,
                            )
                            log.info(
                                "wrote ingest token to %s; restarting %s",
                                ingest_env_path, stream_service,
                            )
                            _systemctl_restart(stream_service)
                        except Exception:
                            # If we cannot persist the token, log
                            # loudly. The token is now lost (server
                            # already nulled its copy). Next step is
                            # to re-provision from the caregiver app
                            # which mints a fresh token.
                            log.exception(
                                "failed to persist ingest token; "
                                "re-provision required",
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
