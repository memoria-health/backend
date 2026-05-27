#!/usr/bin/env python3
"""Nemara camera BLE WiFi-provisioning daemon. Runs on the Pi.

When the Pi has no saved WiFi connection, this daemon advertises a
BLE GATT service. The caregiver app scans for nearby
"Nemara Camera *" devices, connects, writes the home WiFi SSID and
password to two characteristics, then writes a third "submit"
characteristic to commit. The Pi attempts to join WiFi via
`nmcli`; on success it disables this daemon (so future boots skip
straight to the announce daemon).

The trust anchor is physical proximity: BLE pairing requires the
caregiver's phone to be within a few meters of the Pi. v1 does
not use a per-device secret because the proximity requirement plus
the small customer scale (10-20 devices) makes a cross-customer
collision extremely unlikely. v2 will add a printed setup code
that the caregiver scans / enters to authenticate the BLE write.

Service UUID and characteristic UUIDs are fixed; the caregiver app
hardcodes them. They are in the experimental UUID space
(no service-discovery profile registered) but stable across
firmware versions.

  Service:  6e400001-b5a3-f393-e0a9-e50e24dcca9e (Nemara setup)
  SSID:     6e400002-b5a3-f393-e0a9-e50e24dcca9e (write)
  PSK:      6e400003-b5a3-f393-e0a9-e50e24dcca9e (write, encrypted)
  Submit:   6e400004-b5a3-f393-e0a9-e50e24dcca9e (write any 1 byte
            to trigger nmcli join; reads back the result code:
              0x00 = idle (initial)
              0x01 = joining
              0x02 = success
              0xFF = failure
            )

Configuration via env vars (loaded by systemd unit from
/etc/nemara/ble.env):

  NEMARA_WIFI_IFACE       optional, default "wlan0"
  NEMARA_BLE_LABEL_PREFIX optional, default "Nemara Camera"
                          (BLE advertised name uses this + MAC suffix)
  NEMARA_PROVISIONED_FILE optional, default "/var/lib/nemara/provisioned"
                          (touch on success; presence skips this
                          daemon on subsequent boots)

Requires `bless` (BLE peripheral library) installed via pip and
NetworkManager (`nmcli`) installed at the system level.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

# `bless` is an async BLE peripheral library wrapping bluez via dbus.
# Install on the Pi with `pip install bless`. The import is at module
# top per project guideline; if the Pi is missing the dep, the systemd
# unit log surfaces the ImportError immediately on boot.
from bless import (  # type: ignore[import-not-found]
    BlessGATTCharacteristic,
    BlessServer,
    GATTAttributePermissions,
    GATTCharacteristicProperties,
)

log = logging.getLogger("nemara.ble_provision")

_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
_SSID_CHAR_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
_PSK_CHAR_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
_SUBMIT_CHAR_UUID = "6e400004-b5a3-f393-e0a9-e50e24dcca9e"

_STATUS_IDLE = bytes([0x00])
_STATUS_JOINING = bytes([0x01])
_STATUS_SUCCESS = bytes([0x02])
_STATUS_FAILURE = bytes([0xFF])


class ProvisionState:
    """Per-process state shared between the BLE write handlers and
    the join-attempt coroutine. Single asyncio task accesses each
    attribute so no lock is required.
    """

    def __init__(self) -> None:
        self.ssid: str | None = None
        self.psk: str | None = None
        self.status: bytes = _STATUS_IDLE
        self.done_event = asyncio.Event()


def _read_mac(iface: str) -> str:
    return Path(f"/sys/class/net/{iface}/address").read_text().strip()


def _device_label(mac: str, prefix: str) -> str:
    suffix = mac.replace(":", "").upper()[-6:]
    return f"{prefix} {suffix}"


def _wifi_already_configured() -> bool:
    """Skip the BLE flow entirely if NetworkManager has any saved
    WiFi connection. This means a Pi that has been provisioned
    before will not start advertising on subsequent boots even if
    the marker file is missing.
    """
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE", "connection", "show"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        return any(line.strip() == "802-11-wireless" for line in out.splitlines())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


async def _attempt_wifi_join(
    ssid: str, psk: str, iface: str,
) -> bool:
    """Run `nmcli device wifi connect ...` in a subprocess. Returns
    True on success. Logs (but does not raise) the nmcli stderr on
    failure so the BLE characteristic can report status accurately.
    """
    cmd = [
        "nmcli", "device", "wifi", "connect", ssid,
        "password", psk, "ifname", iface,
    ]
    log.info("attempting WiFi join ssid=%r iface=%s", ssid, iface)
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=45.0)
        if proc.returncode == 0:
            log.info("WiFi join success: %s", stdout.decode().strip())
            return True
        log.warning(
            "WiFi join failed rc=%d stderr=%r",
            proc.returncode, stderr.decode().strip(),
        )
        return False
    except asyncio.TimeoutError:
        log.warning("nmcli join timed out after 45s")
        return False


def _mark_provisioned(marker_path: Path) -> None:
    """Touch the marker file so subsequent boots skip this daemon."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()


async def serve() -> None:
    iface = os.environ.get("NEMARA_WIFI_IFACE", "wlan0")
    label_prefix = os.environ.get("NEMARA_BLE_LABEL_PREFIX", "Nemara Camera")
    marker_path = Path(
        os.environ.get("NEMARA_PROVISIONED_FILE", "/var/lib/nemara/provisioned")
    )

    if marker_path.exists() or _wifi_already_configured():
        log.info(
            "WiFi already configured (marker=%s, nmcli=%s); skipping BLE",
            marker_path.exists(), _wifi_already_configured(),
        )
        return

    mac = _read_mac(iface)
    ble_name = _device_label(mac, label_prefix)
    log.info("advertising as %r mac=%s", ble_name, mac)

    state = ProvisionState()

    def _on_ssid_write(_char: BlessGATTCharacteristic, value: bytes) -> None:
        state.ssid = value.decode("utf-8", errors="replace")
        log.info("received SSID len=%d", len(state.ssid))

    def _on_psk_write(_char: BlessGATTCharacteristic, value: bytes) -> None:
        state.psk = value.decode("utf-8", errors="replace")
        log.info("received PSK len=%d", len(state.psk))

    async def _on_submit_write(
        char: BlessGATTCharacteristic, _value: bytes,
    ) -> None:
        if state.ssid is None or state.psk is None:
            log.warning("submit before SSID/PSK; ignoring")
            state.status = _STATUS_FAILURE
            char.value = state.status
            return
        state.status = _STATUS_JOINING
        char.value = state.status
        ok = await _attempt_wifi_join(state.ssid, state.psk, iface)
        if ok:
            state.status = _STATUS_SUCCESS
            char.value = state.status
            _mark_provisioned(marker_path)
            log.info("provisioning complete; shutting down BLE")
            state.done_event.set()
        else:
            state.status = _STATUS_FAILURE
            char.value = state.status
            # Allow re-attempts: do NOT set done_event; the caregiver
            # can retype the password and re-submit.

    server = BlessServer(name=ble_name)
    await server.add_new_service(_SERVICE_UUID)
    write_perm = (
        GATTCharacteristicProperties.write
        | GATTCharacteristicProperties.write_without_response
    )
    read_perm = GATTCharacteristicProperties.read
    await server.add_new_characteristic(
        _SERVICE_UUID,
        _SSID_CHAR_UUID,
        write_perm,
        b"",
        GATTAttributePermissions.writeable,
    )
    server.get_characteristic(_SSID_CHAR_UUID).value = b""
    server.get_characteristic(_SSID_CHAR_UUID).set_write_callback(_on_ssid_write)

    await server.add_new_characteristic(
        _SERVICE_UUID,
        _PSK_CHAR_UUID,
        write_perm,
        b"",
        GATTAttributePermissions.writeable,
    )
    server.get_characteristic(_PSK_CHAR_UUID).set_write_callback(_on_psk_write)

    await server.add_new_characteristic(
        _SERVICE_UUID,
        _SUBMIT_CHAR_UUID,
        write_perm | read_perm,
        _STATUS_IDLE,
        GATTAttributePermissions.readable | GATTAttributePermissions.writeable,
    )
    server.get_characteristic(_SUBMIT_CHAR_UUID).set_write_callback(_on_submit_write)

    await server.start()
    log.info("BLE GATT server running; waiting for provisioning")
    try:
        await state.done_event.wait()
    finally:
        await server.stop()
        log.info("BLE GATT server stopped")


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("NEMARA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        log.info("stopped by signal")
        sys.exit(0)


if __name__ == "__main__":
    main()
