#!/usr/bin/env python3
"""Nemara camera WiFi-AP provisioning daemon. Runs on the Pi.

When the Pi has no saved WiFi connection, this daemon brings up an
open WiFi access point (named "Memoria-Setup-XXXXXX" using the last
6 of the WiFi MAC) and serves a captive portal at 192.168.4.1.

Customer flow, identical on iOS / Android / Windows / macOS:

  1. Customer plugs in the camera. Pi boots, this daemon notices
     there is no saved WiFi connection, and starts the AP.
  2. Customer connects their phone or laptop to the
     "Memoria-Setup-XXXXXX" WiFi network. On iOS the captive portal
     detector pops the setup page automatically; on other platforms
     the customer opens any URL and gets redirected.
  3. Customer picks their home WiFi from a dropdown (pre-scanned
     before AP-mode came up so the cached list is available) and
     enters the password.
  4. POST /provision triggers `nmcli device wifi connect ssid
     password ...` against the home network. On success the daemon
     tears down the AP, touches a marker file, exits. NetworkManager
     auto-connects to the saved home WiFi from then on.

The trust anchor is physical proximity: the customer must be close
enough to see the AP, and only one Pi at a time is in setup mode
per home. There is no per-device secret; v2 would add a printed
setup code that the form requires for additional confirmation.

Why nmcli throughout (vs raw hostapd / dnsmasq config files):
NetworkManager handles the AP / STA mode juggling, DHCP server
for clients connected to the AP, and the home-WiFi connect with
a single CLI. Pi OS Bookworm + Trixie ship with NetworkManager
by default.

Environment configuration (loaded by systemd from
/etc/nemara/wifi-ap.env):

  NEMARA_WIFI_IFACE        optional, default "wlan0"
  NEMARA_AP_LABEL_PREFIX   optional, default "Memoria-Setup"
                           (full SSID = "<prefix>-<last 6 of MAC>")
  NEMARA_AP_CHANNEL        optional, default "6" (2.4 GHz)
  NEMARA_AP_BIND_ADDR      optional, default "192.168.4.1"
  NEMARA_AP_HTTP_PORT      optional, default "80"
  NEMARA_PROVISIONED_FILE  optional, default "/var/lib/nemara/provisioned"
  NEMARA_LOG_LEVEL         optional, default "INFO"
"""

from __future__ import annotations

import logging
import os
import socketserver
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs

log = logging.getLogger("nemara.wifi_ap")

# How long the daemon stays up before giving up on a customer. Long
# enough that someone can take their time finding the WiFi network
# and entering credentials; short enough that a Pi that nobody is
# touching reboots and tries again rather than hogging the radio.
_SETUP_TIMEOUT_S: int = 30 * 60

# After nmcli successfully joins home WiFi, give the OS a moment to
# settle before tearing down the AP. Otherwise the page that the
# customer is reading may not load its final state.
_TEARDOWN_DELAY_S: float = 5.0

# Maximum length of password input. WPA2/3 cap at 63 bytes; we
# enforce 8-63 since shorter values are not valid WPA passwords.
_MIN_PSK_LEN: int = 8
_MAX_PSK_LEN: int = 63
_MAX_SSID_LEN: int = 32


@dataclass(frozen=True)
class Config:
    wifi_iface: str
    ap_ssid: str
    ap_channel: str
    bind_addr: str
    http_port: int
    marker_path: Path


def load_config() -> Config:
    iface = os.environ.get("NEMARA_WIFI_IFACE", "wlan0")
    mac = _read_mac(iface)
    suffix = mac.replace(":", "").upper()[-6:]
    prefix = os.environ.get("NEMARA_AP_LABEL_PREFIX", "Memoria-Setup")
    return Config(
        wifi_iface=iface,
        ap_ssid=f"{prefix}-{suffix}",
        ap_channel=os.environ.get("NEMARA_AP_CHANNEL", "6"),
        bind_addr=os.environ.get("NEMARA_AP_BIND_ADDR", "192.168.4.1"),
        http_port=int(os.environ.get("NEMARA_AP_HTTP_PORT", "80")),
        marker_path=Path(
            os.environ.get(
                "NEMARA_PROVISIONED_FILE", "/var/lib/nemara/provisioned"
            )
        ),
    )


def _read_mac(iface: str) -> str:
    return Path(f"/sys/class/net/{iface}/address").read_text().strip()


# ─── NetworkManager helpers ──────────────────────────────────────


def already_provisioned(marker: Path) -> bool:
    """The marker file is the canonical "this Pi is on the customer's
    home wifi, don't enter AP mode" signal.

    Self-healing semantics so the operator never has to touch the
    marker manually:

      - Marker present                          → True (idle path).
      - Marker absent, but nmcli has an active
        wifi connection (Pi joined home wifi
        outside our captive portal — Imager
        pre-config, netplan auto-regen, manual
        nmcli)                                  → touch marker, True.
      - Marker absent, no active wifi           → False (enter AP).

    Why the self-heal: without it, a Pi flashed with Imager wifi creds
    has the marker absent forever. The supervisor's `serve()` waits
    1800 s for a successful captive-portal join, times out, tears the
    AP down, re-enters AP — on a 30-minute cycle. Looks like the Pi
    spontaneously broadcasts the setup SSID every half hour. The
    only thing that breaks the cycle is the marker existing, so let
    the supervisor write it the moment it observes the wifi is
    already up.

    Distinction from saved-but-inactive: we filter on STATE!=
    'deactivated' (any of activated, activating, deactivating counts
    as "wifi is doing something"). A connection profile that exists
    but hasn't tried to come up doesn't qualify — that's the
    customer-reset flow where the profile may be cached but the
    wipe semantics intended for AP mode to take over.
    """
    if marker.exists():
        return True
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE", "connection", "show", "--active"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    has_active_wifi = any(
        line.strip().startswith("802-11-wireless:") for line in out.splitlines()
    )
    if has_active_wifi:
        log.info(
            "already_provisioned: marker absent but active wifi found, "
            "self-healing by touching %s",
            marker,
        )
        try:
            mark_provisioned(marker)
        except Exception:
            log.exception("already_provisioned: marker self-heal failed (continuing)")
        return True
    return False


def scan_nearby_networks(iface: str) -> list[str]:
    """One-shot scan before we enter AP mode. nmcli prints the SSIDs
    of all visible WiFi networks; we de-duplicate and return them.
    Returning an empty list is fine, the customer can still type the
    SSID manually.
    """
    try:
        out = subprocess.run(
            [
                "nmcli", "-t", "-f", "SSID", "device", "wifi", "list",
                "ifname", iface, "--rescan", "yes",
            ],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        log.warning("Network scan failed: %s", exc)
        return []
    seen: list[str] = []
    for line in out.splitlines():
        ssid = line.strip()
        if ssid and ssid not in seen:
            seen.append(ssid)
    return seen


# Default per-step nmcli timeout. Each individual nmcli invocation
# should complete in seconds; anything longer is a hung D-Bus call
# (the class of failure that wedged us repeatedly). 30 s is generous
# but bounded so the supervisor loop can back off and retry.
#
# `connection up` for an AP is the exception: on Pi OS Trixie, adding
# a new wifi connection triggers NM -> netplan-regenerate -> systemctl
# daemon-reload (~13s each) and the chain runs 2-3 times during a
# single `connection up`. Confirmed 2026-05-28 from the FAT32 debug
# dump: AP was successfully ACTIVATED at 23:23:26 -- 46s after the
# `nmcli connection up` was issued at 23:22:40. The supervisor had
# already given up at 23:23:10 (30s) and torn the AP down 16s before
# NM finished activating it. So `up-ap-connection` needs ~120s headroom.
_NMCLI_TIMEOUT_S: int = 30
_NMCLI_UP_TIMEOUT_S: int = 120


def _nmcli(
    args: list[str],
    *,
    check: bool,
    label: str,
    timeout_s: int | None = None,
) -> subprocess.CompletedProcess:
    """Wrapped nmcli call with hard timeout + structured logging.

    Each call logs the command, exit code, and (when non-zero or
    `check=True`) stdout/stderr. This is the debug path that surfaces
    *why* AP setup fails without us having to attach to the device.

    `timeout_s` overrides the default `_NMCLI_TIMEOUT_S` for slow ops
    like `connection up` where NM's internal netplan/daemon-reload
    cascade legitimately takes 40-60s on Pi OS Trixie.
    """
    effective_timeout = timeout_s if timeout_s is not None else _NMCLI_TIMEOUT_S
    log.info("nmcli.run [%s]: %s", label, " ".join(args))
    try:
        result = subprocess.run(
            args,
            capture_output=True, text=True,
            timeout=effective_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        log.error(
            "nmcli.timeout [%s] after %ds: %s",
            label, effective_timeout, " ".join(args),
        )
        raise RuntimeError(f"nmcli {label} hung past {effective_timeout}s") from exc

    if result.returncode != 0:
        log.warning(
            "nmcli.nonzero [%s] rc=%d stdout=%r stderr=%r",
            label, result.returncode,
            result.stdout.strip()[:300], result.stderr.strip()[:300],
        )
        if check:
            raise RuntimeError(
                f"nmcli {label} failed rc={result.returncode}: "
                f"{result.stderr.strip()[:300]}"
            )
    else:
        log.info("nmcli.ok [%s] rc=0", label)
    return result


def _ap_connection_exists(ssid: str) -> bool:
    """True if NM already has a connection profile with this name.
    install.sh pre-bakes the AP keyfile (since 2026-05-28) so the fast
    path can skip the expensive `connection add` step entirely."""
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "NAME", "connection", "show"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return any(line.strip() == ssid for line in out.splitlines())


def start_ap(cfg: Config) -> None:
    """Bring up the WiFi access point via NetworkManager.

    Fast path (install.sh has pre-baked the keyfile): just `connection up`.
    On Pi OS Trixie, every `nmcli connection add` cascades into
    NM -> netplan-regenerate -> systemctl daemon-reload (~13s each), and
    the chain runs 2-3 times per `up`. With the keyfile already present
    NM has nothing to mirror to netplan and the cascade does not fire,
    so AP activation drops from ~50s to ~3s. Confirmed 2026-05-28 from
    the FAT32 debug dump.

    Slow fallback (legacy install or corrupted keyfile): delete any
    stale profile, add a fresh one, then up. This preserves working
    behavior for Pis installed before the pre-bake was added.
    """
    log.info(
        "start_ap.begin ssid=%r iface=%s channel=%s bind=%s",
        cfg.ap_ssid, cfg.wifi_iface, cfg.ap_channel, cfg.bind_addr,
    )

    # Pre-flight: confirm NM is responsive at all (this also catches
    # the case where NM hasn't finished starting yet).
    _nmcli(
        ["nmcli", "-t", "-f", "STATE", "general"],
        check=True, label="nm-state",
    )

    # Pre-flight: confirm the wifi interface exists and isn't blocked.
    _nmcli(
        ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "device"],
        check=True, label="device-status",
    )

    if _ap_connection_exists(cfg.ap_ssid):
        log.info("start_ap.fast_path: pre-baked %r exists, skipping add", cfg.ap_ssid)
    else:
        log.info("start_ap.slow_path: pre-bake missing, falling back to add")
        _nmcli(
            ["nmcli", "connection", "delete", cfg.ap_ssid],
            check=False, label="delete-stale-ap",
        )
        _nmcli(
            [
                "nmcli", "connection", "add",
                "type", "wifi",
                "ifname", cfg.wifi_iface,
                "con-name", cfg.ap_ssid,
                "autoconnect", "no",
                "ssid", cfg.ap_ssid,
                "mode", "ap",
                "ipv4.method", "shared",
                "ipv4.addresses", f"{cfg.bind_addr}/24",
                "wifi.band", "bg",
                "wifi.channel", cfg.ap_channel,
            ],
            check=True, label="add-ap-connection",
        )
    # NB: we intentionally do NOT set wifi-sec.key-mgmt. In NetworkManager,
    # `key-mgmt=none` means WEP (legacy open auth, still requires a wep-key0),
    # which makes `connection up` fail with "Secrets were required". The
    # absence of any [802-11-wireless-security] section is what gives us a
    # truly open network.
    _nmcli(
        ["nmcli", "connection", "up", cfg.ap_ssid],
        check=True, label="up-ap-connection",
        timeout_s=_NMCLI_UP_TIMEOUT_S,
    )
    log.info("start_ap.complete ssid=%r is now broadcasting", cfg.ap_ssid)


def stop_ap(cfg: Config) -> None:
    """Tear down the AP. Best effort; if NM is already shutting it
    down (e.g. because we joined another network) the down call is
    harmless.

    We deliberately do NOT `connection delete` here. The pre-baked
    keyfile from install.sh is the fast path for the NEXT start_ap
    (skips the ~13s netplan cascade). Deleting it would force the
    next AP cycle into the slow path. `connection down` is enough to
    free wlan0 for join_wifi to use as a client.
    """
    log.info("tearing down AP")
    subprocess.run(
        ["nmcli", "connection", "down", cfg.ap_ssid],
        capture_output=True, check=False,
    )


def join_wifi(cfg: Config, ssid: str, psk: str) -> tuple[bool, str]:
    """Attempt to connect to the customer's home WiFi. Returns
    (success, message). On success NetworkManager saves the
    connection so subsequent boots auto-connect.

    The high-level `nmcli device wifi connect` shortcut silently
    inherits properties from any pre-existing connection profile
    that matches the SSID. On a Pi flashed with wifi creds via
    Raspberry Pi Imager, netplan auto-generates an NM keyfile for
    the imager-configured network — that keyfile may be missing
    `wifi-sec.key-mgmt` (saw `Error: 802-11-wireless-security.key-mgmt:
    property is missing` on 2026-05-28 16:57 when the captive portal
    tried to rejoin Suburban Living). The shortcut command then
    fails to merge our fresh creds onto the malformed profile, the
    supervisor loop times out after 30 minutes, and the Pi falls
    back into AP mode on the next iteration — looking, from the
    outside, like the Pi spontaneously decided to start broadcasting
    the setup SSID again.

    Fix: delete any pre-existing profile with the same SSID first
    (best-effort, ignore errors when nothing matched), then create a
    fresh connection with all WPA2 fields explicitly set. The
    explicit add bypasses the shortcut's merge logic entirely so a
    malformed pre-existing keyfile can't break us.
    """
    log.info("joining home WiFi ssid=%r", ssid)
    # Clear any pre-existing connection with this SSID. Both the
    # SSID-keyed `con-name` and netplan's `netplan-wlan0-<ssid>` form
    # need scrubbing — we hit the latter on a fresh imager flash.
    for con_name in (ssid, f"netplan-{cfg.wifi_iface}-{ssid}"):
        subprocess.run(
            ["nmcli", "connection", "delete", con_name],
            capture_output=True, check=False,
        )
    try:
        proc = subprocess.run(
            [
                "nmcli", "connection", "add",
                "type", "wifi",
                "ifname", cfg.wifi_iface,
                "con-name", ssid,
                "ssid", ssid,
                "wifi-sec.key-mgmt", "wpa-psk",
                "wifi-sec.psk", psk,
                "connection.autoconnect", "yes",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip()
            log.warning("nmcli connection add failed: %s", msg)
            return False, msg
        # Same netplan/daemon-reload cascade as up-ap-connection — see
        # _NMCLI_UP_TIMEOUT_S docstring. Customer-side join must outlast it.
        up_proc = subprocess.run(
            ["nmcli", "connection", "up", ssid, "ifname", cfg.wifi_iface],
            capture_output=True, text=True, timeout=_NMCLI_UP_TIMEOUT_S,
        )
        if up_proc.returncode == 0:
            return True, up_proc.stdout.strip()
        msg = (up_proc.stderr or up_proc.stdout).strip()
        log.warning("nmcli connection up failed: %s", msg)
        return False, msg
    except subprocess.TimeoutExpired:
        return False, "Connection attempt timed out."


def mark_provisioned(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


# ─── HTTP server (the captive portal) ────────────────────────────


_PROVISION_RESULT: dict[str, str] = {}  # mutated by the handler
_DONE = threading.Event()


_SETUP_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#EFE6DA">
<title>Nemara camera setup</title>
<style>
  :root {{
    --paper: #EFE6DA;
    --card: #FBF8F0;
    --ink: #1C1A14;
    --ink-soft: #4A4539;
    --line: #C9C0AB;
    --terra: #A04331;
    --terra-soft: #F2DDD3;
    --terra-deep: #7A2F1F;
    --action-ink: #FBF8F0;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
    background: var(--paper);
    color: var(--ink);
    min-height: 100vh;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 32px 16px 48px;
    -webkit-font-smoothing: antialiased;
  }}
  .card {{
    width: 100%;
    max-width: 420px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 28px 24px 24px;
  }}
  .eyebrow {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-soft);
    margin: 0 0 10px;
  }}
  h1 {{
    font-family: ui-serif, "Iowan Old Style", "Apple Garamond", Georgia,
                 "Times New Roman", serif;
    font-weight: 600;
    font-size: 28px;
    line-height: 1.15;
    margin: 0 0 10px;
    color: var(--ink);
  }}
  p.lede {{
    font-size: 15px;
    line-height: 1.45;
    color: var(--ink-soft);
    margin: 0 0 24px;
  }}
  label {{
    display: block;
    margin-top: 18px;
    margin-bottom: 8px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--ink-soft);
  }}
  input, select {{
    width: 100%;
    padding: 12px 14px;
    font-size: 16px;
    font-family: inherit;
    color: var(--ink);
    background: var(--card);
    border: 1.5px solid var(--line);
    border-radius: 10px;
    -webkit-appearance: none;
    appearance: none;
    transition: border-color 0.15s;
  }}
  select {{
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%234A4539' d='M6 8L0 0h12z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 14px center;
    background-size: 10px;
    padding-right: 36px;
  }}
  input:focus, select:focus {{
    outline: none;
    border-color: var(--ink);
    caret-color: var(--terra);
  }}
  button {{
    width: 100%;
    margin-top: 28px;
    padding: 16px 20px;
    font-size: 16px;
    font-family: inherit;
    font-weight: 600;
    background: var(--ink);
    color: var(--action-ink);
    border: 0;
    border-radius: 10px;
    cursor: pointer;
    -webkit-appearance: none;
    appearance: none;
  }}
  button:active {{ background: #000; }}
  .err {{
    background: var(--terra-soft);
    color: var(--terra-deep);
    padding: 12px 14px;
    border-radius: 10px;
    margin-top: 20px;
    font-size: 14px;
    line-height: 1.4;
  }}
</style>
</head><body>
<main class="card">
  <p class="eyebrow">Nemara &middot; Camera setup</p>
  <h1>Set up your camera</h1>
  <p class="lede">Pick your home WiFi and enter the password. Your camera
  will join your network and this setup network will disappear in about
  a minute.</p>
  {error_block}
  <form method="POST" action="/provision">
    <label for="ssid">Home WiFi</label>
    {ssid_input}
    <label for="psk">Password</label>
    <input type="password" id="psk" name="psk" autocomplete="off"
           minlength="8" maxlength="63" required>
    <button type="submit">Connect</button>
  </form>
</main>
</body></html>"""


_SUCCESS_PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="theme-color" content="#EFE6DA">
<title>Connecting your camera</title>
<style>
  :root {{
    --paper: #EFE6DA;
    --card: #FBF8F0;
    --ink: #1C1A14;
    --ink-soft: #4A4539;
    --line: #C9C0AB;
    --terra: #A04331;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, system-ui, "Segoe UI", Roboto, sans-serif;
    background: var(--paper);
    color: var(--ink);
    min-height: 100vh;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding: 48px 16px;
    -webkit-font-smoothing: antialiased;
  }}
  .card {{
    width: 100%;
    max-width: 420px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 36px 28px 32px;
    text-align: center;
  }}
  .spinner {{
    width: 44px;
    height: 44px;
    margin: 0 auto 24px;
    border: 3px solid var(--line);
    border-top-color: var(--terra);
    border-radius: 50%;
    animation: spin 0.9s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .eyebrow {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--terra);
    margin: 0 0 10px;
  }}
  h1 {{
    font-family: ui-serif, "Iowan Old Style", "Apple Garamond", Georgia,
                 "Times New Roman", serif;
    font-weight: 600;
    font-size: 28px;
    line-height: 1.2;
    margin: 0 0 14px;
    color: var(--ink);
  }}
  p {{
    font-size: 15px;
    line-height: 1.5;
    color: var(--ink-soft);
    margin: 0 0 10px;
  }}
  .ssid {{
    font-weight: 600;
    color: var(--ink);
  }}
  .hint {{
    margin-top: 22px;
    font-size: 13px;
    color: var(--ink-soft);
    opacity: 0.85;
  }}
</style>
</head><body>
<main class="card" id="card">
  <div class="spinner" aria-hidden="true" id="spinner"></div>
  <p class="eyebrow" id="eyebrow">Connecting</p>
  <h1 id="heading">Connecting your camera</h1>
  <p id="body">Your camera is joining <span class="ssid">{ssid}</span>.
  It will appear in the Nemara app in about a minute.</p>
  <p class="hint" id="hint">This tab will close automatically.</p>
</main>
<script>
  // Polling-based "AP went down" detector. The Pi serves /ping with
  // 204 while the AP is still up; the moment join_wifi succeeds and
  // the radio tears down to switch into client mode, every fetch to
  // 192.168.4.1 starts failing (either the iPhone loses the
  // association or its own captive-portal probe times out). Two
  // consecutive failures = AP gone = join succeeded. Morph the UI to
  // a final state and best-effort close the tab (window.close only
  // works when the tab was opened via window.open / captive portal
  // sheet; otherwise the user still sees the success state).
  (function () {{
    var POLL_MS = 2000;
    var FAILS_BEFORE_DONE = 2;
    var MAX_POLL_S = 60;
    var failures = 0;
    var startedAt = Date.now();
    var finished = false;
    function done() {{
      if (finished) return;
      finished = true;
      var spinner = document.getElementById('spinner');
      if (spinner) spinner.style.display = 'none';
      var eb = document.getElementById('eyebrow');
      if (eb) eb.textContent = 'Connected';
      var h = document.getElementById('heading');
      if (h) h.textContent = 'Your camera is connected';
      var b = document.getElementById('body');
      if (b) b.innerHTML = 'Open the Nemara app to add this camera. It will appear under <b>Add a camera</b> within about a minute.';
      var hint = document.getElementById('hint');
      if (hint) hint.textContent = 'You can close this tab.';
      // Try to close the tab (works only when opened via window.open
      // or via iOS captive-portal sheet which auto-dismisses).
      setTimeout(function () {{
        try {{ window.close(); }} catch (e) {{}}
      }}, 1200);
    }}
    function poll() {{
      if (finished) return;
      if ((Date.now() - startedAt) / 1000 > MAX_POLL_S) {{
        // Safety: give up after MAX_POLL_S and morph to final state
        // anyway so the user isn't stuck on a spinner.
        done();
        return;
      }}
      var ctrl = new AbortController();
      var t = setTimeout(function () {{ ctrl.abort(); }}, 1500);
      fetch('/ping', {{ cache: 'no-store', signal: ctrl.signal }})
        .then(function (r) {{
          clearTimeout(t);
          if (r.ok || r.status === 204) failures = 0;
          else failures++;
          if (failures >= FAILS_BEFORE_DONE) done();
          else setTimeout(poll, POLL_MS);
        }})
        .catch(function () {{
          clearTimeout(t);
          failures++;
          if (failures >= FAILS_BEFORE_DONE) done();
          else setTimeout(poll, POLL_MS);
        }});
    }}
    // Wait one POLL_MS so the page has a chance to render before the
    // first ping fires (and so the initial pings reliably succeed
    // while the AP is still up, avoiding a false-positive "done").
    setTimeout(poll, POLL_MS);
  }})();
</script>
</body></html>"""


def _render_success_page(ssid: str) -> bytes:
    return _SUCCESS_PAGE_TEMPLATE.format(ssid=_html_escape(ssid)).encode("utf-8")


# Path where the last failed join_wifi error is persisted, so the
# next AP-mode session can show the customer what went wrong
# instead of looking like an unexplained 30-min retry. Written by
# `_join_then_signal` on failure, consumed by `serve()` on startup,
# unconditionally cleared after a successful provision. Living on the
# durable /var partition means a power cycle between failure and
# the user's retry doesn't lose the message.
_LAST_JOIN_ERROR_FILE: Path = Path("/var/lib/nemara/last_join_error")


def _persist_join_error(message: str) -> None:
    """Write the latest failed-join error to disk so the next AP
    session can render it. Best-effort: failure to write is logged
    but never raises, because we don't want a disk-full / readonly
    rootfs to break the captive portal lifecycle itself.
    """
    try:
        _LAST_JOIN_ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
        _LAST_JOIN_ERROR_FILE.write_text(message, encoding="utf-8")
    except Exception:
        log.exception("failed to persist last_join_error")


def _consume_join_error() -> str | None:
    """Read + delete the persisted error. We delete eagerly so the
    error only ever shows on the FIRST captive-portal load after a
    failed join; subsequent reloads in the same session won't keep
    re-displaying a stale message after the customer fixed the
    issue. Missing file is the happy path and returns None.
    """
    try:
        msg = _LAST_JOIN_ERROR_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except Exception:
        log.exception("failed to read last_join_error")
        return None
    try:
        _LAST_JOIN_ERROR_FILE.unlink()
    except Exception:
        log.exception("failed to unlink last_join_error after consume")
    return msg or None


def _join_then_signal(cfg: "Config", ssid: str, psk: str) -> None:
    """Run after the success page has been flushed to the client.
    Sleep briefly so the phone has time to render the page, then tear
    down the AP and join the customer's home WiFi. On success, mark
    provisioned and signal the daemon to exit cleanly; on failure,
    persist the error and let the supervisor restart the AP so the
    next captive-portal load shows the customer what went wrong.

    Critical: stop_ap MUST run before join_wifi. The Pi has a single
    radio. While wlan0 is broadcasting the AP, `nmcli device wifi
    connect` cannot see the target SSID and fails with
    "No network with SSID '<name>' found." stop_ap brings the radio
    out of AP mode so it can scan for and join the client network.
    """
    time.sleep(_TEARDOWN_DELAY_S)
    stop_ap(cfg)
    # Brief settle so the radio fully releases AP mode before we ask
    # NM to scan in client mode. Without this, the next nmcli call
    # can race the AP teardown and still fail.
    time.sleep(2.0)
    ok, msg = join_wifi(cfg, ssid, psk)
    if ok:
        # Clear any stale error from a previous attempt — a fresh
        # success should never carry forward an old "wrong password"
        # message into the post-recovery UI.
        try:
            _LAST_JOIN_ERROR_FILE.unlink(missing_ok=True)
        except Exception:
            log.exception("failed to clear last_join_error on success")
        _PROVISION_RESULT["ssid"] = ssid
        _DONE.set()
    else:
        log.error("background join failed for ssid=%r: %s", ssid, msg)
        _persist_join_error(f"Couldn't join {ssid!r}: {msg}")


def _render_setup_page(networks: list[str], error: str | None = None) -> bytes:
    if networks:
        opts = "\n".join(
            f'<option value="{_html_escape(s)}">{_html_escape(s)}</option>'
            for s in networks
        )
        ssid_input = (
            f'<select id="ssid" name="ssid" required>{opts}</select>'
        )
    else:
        ssid_input = (
            '<input type="text" id="ssid" name="ssid" required '
            f'maxlength="{_MAX_SSID_LEN}" autocomplete="off">'
        )
    error_block = (
        f'<div class="err">{_html_escape(error)}</div>' if error else ""
    )
    page = _SETUP_PAGE_TEMPLATE.format(
        ssid_input=ssid_input, error_block=error_block,
    )
    return page.encode("utf-8")


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


class _Handler(BaseHTTPRequestHandler):
    cfg: Config
    nearby_networks: list[str]
    last_error: str | None = None

    def log_message(self, format: str, *args: object) -> None:
        log.info("http %s %s", self.address_string(), format % args)

    # iOS / Android sometimes hit a generic detection URL before they
    # show the captive portal. We redirect any unknown path to the
    # setup root so the user always lands on the form.
    def _is_captive_probe(self) -> bool:
        return self.path not in ("/", "/provision", "/ping")

    def do_GET(self) -> None:
        # /ping is the success-page poll target. Returns 204 No Content
        # with no caching so the client-side JS can detect "AP is gone"
        # the moment the radio tears down to switch into client mode.
        # No body, no logging, intentionally trivial.
        if self.path == "/ping":
            self.send_response(204)
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        if self._is_captive_probe():
            self.send_response(302)
            self.send_header(
                "Location", f"http://{self.cfg.bind_addr}/"
            )
            self.end_headers()
            return
        body = _render_setup_page(self.nearby_networks, self.last_error)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/provision":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw)
        ssid = (form.get("ssid", [""])[0] or "").strip()
        psk = (form.get("psk", [""])[0] or "").strip()

        if not ssid or len(ssid) > _MAX_SSID_LEN:
            return self._reshow_with_error("Please pick a network.")
        if len(psk) < _MIN_PSK_LEN or len(psk) > _MAX_PSK_LEN:
            return self._reshow_with_error(
                "Password must be between 8 and 63 characters."
            )

        # Send the "Connecting..." page FIRST, while the phone is still
        # on the AP and able to receive bytes. We can't wait for
        # join_wifi to confirm before showing a result page because
        # join_wifi swaps wlan0 from AP-mode to client-mode, tearing
        # down the AP and stranding the phone before it can render
        # anything. So the page reads as pending ("Connecting your
        # camera...") rather than past-tense success. The real
        # confirmation is the camera appearing in the Nemara app.
        body = _render_success_page(ssid)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

        threading.Thread(
            target=_join_then_signal,
            args=(self.cfg, ssid, psk),
            daemon=True,
        ).start()

    def _reshow_with_error(self, error: str) -> None:
        type(self).last_error = error
        body = _render_setup_page(self.nearby_networks, error)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ─── main ────────────────────────────────────────────────────────


def serve(cfg: Config) -> None:
    networks = scan_nearby_networks(cfg.wifi_iface)
    log.info("scanned %d nearby networks", len(networks))
    start_ap(cfg)
    # If the previous AP session ended in a failed join_wifi (wrong
    # password, malformed key-mgmt, SSID not found, whatever), the
    # error message is persisted on disk by `_join_then_signal`.
    # Consume it now and seed it as the initial captive-portal error
    # so the customer sees what went wrong immediately on next AP
    # load instead of staring at an empty form again.
    initial_error = _consume_join_error()
    if initial_error is not None:
        log.info("seeded captive portal with prior join error: %s", initial_error)
    handler = type("HandlerWithConfig", (_Handler,), {
        "cfg": cfg,
        "nearby_networks": networks,
        "last_error": initial_error,
    })
    server = _ThreadingHTTPServer((cfg.bind_addr, cfg.http_port), handler)
    log.info("captive portal listening on http://%s:%d", cfg.bind_addr, cfg.http_port)

    def _serve() -> None:
        server.serve_forever()

    server_thread = threading.Thread(target=_serve, daemon=True)
    server_thread.start()

    if _DONE.wait(timeout=_SETUP_TIMEOUT_S):
        log.info("provisioning succeeded for ssid=%r", _PROVISION_RESULT.get("ssid"))
        mark_provisioned(cfg.marker_path)
    else:
        log.warning("setup timed out after %d s with no successful provision", _SETUP_TIMEOUT_S)

    server.shutdown()
    stop_ap(cfg)


# How long to sleep between re-checks of the provisioned state when
# the Pi is paired and idle. Short enough that a customer who clears
# the marker (factory-reset path) gets the AP back within a minute
# without rebooting; long enough that the supervisor uses negligible
# CPU.
_IDLE_RECHECK_INTERVAL_S: int = 30


def main() -> None:
    """Supervisor loop: bring up the AP whenever the Pi is unpaired,
    and keep checking once paired so a customer-initiated factory
    reset (removing the marker file) auto-rearms the AP without a
    reboot.

    Behavior:
      - Unpaired (no marker, no saved WiFi): run `serve(cfg)` which
        brings up the AP, serves the captive portal, joins the home
        WiFi on submit, touches the marker, returns.
      - Paired (marker present or saved WiFi): sleep
        `_IDLE_RECHECK_INTERVAL_S` and re-check. Next iteration sees
        the same state and sleeps again. If the marker disappears,
        next iteration falls into the unpaired branch and AP comes
        back up.
      - Fatal exception inside serve(): log, sleep
        `_IDLE_RECHECK_INTERVAL_S` (so we do not hot-loop on a
        persistent failure), continue.
    """
    logging.basicConfig(
        level=os.environ.get("NEMARA_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = load_config()
    log.info(
        "supervisor.start marker=%s recheck=%ds",
        cfg.marker_path, _IDLE_RECHECK_INTERVAL_S,
    )
    while True:
        try:
            if already_provisioned(cfg.marker_path):
                # Paired and configured. Sleep and re-check; this is
                # the steady-state branch for the lifetime of the
                # device after first provisioning.
                time.sleep(_IDLE_RECHECK_INTERVAL_S)
                continue
            # Unpaired: run the one-shot AP + captive portal flow.
            # Returns on successful provision (marker written) or
            # timeout. Either way the next loop iteration re-checks.
            log.info(
                "unpaired (marker=%s, no saved WiFi); starting AP",
                cfg.marker_path.exists(),
            )
            serve(cfg)
        except KeyboardInterrupt:
            log.info("stopped by signal")
            try:
                stop_ap(cfg)
            except Exception:
                pass
            sys.exit(0)
        except Exception:
            # A persistent failure (nmcli stuck, radio offline) should
            # not hot-loop. Tear down whatever the AP got partway
            # through, sleep, then retry. systemd's Restart=on-failure
            # would also catch this, but the supervisor loop keeps
            # state in-process across retries so a long backoff doesn't
            # require systemd churn.
            log.exception("provisioning iteration failed; backing off")
            try:
                stop_ap(cfg)
            except Exception:
                pass
            time.sleep(_IDLE_RECHECK_INTERVAL_S)


if __name__ == "__main__":
    # Defensive: a Python that does not have threading enabled is
    # unusable for this daemon. Better to abort with a clear message
    # than crash mid-serve.
    assert hasattr(threading, "Event"), "Python threading required"
    _ = time  # ensure time module is loaded even if unused above
    main()
