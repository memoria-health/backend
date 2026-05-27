#!/usr/bin/env python3
"""Nemara camera WiFi-AP provisioning daemon. Runs on the Pi.

When the Pi has no saved WiFi connection, this daemon brings up an
open WiFi access point (named "Nemara-Setup-XXXXXX" using the last
6 of the WiFi MAC) and serves a captive portal at 192.168.4.1.

Customer flow, identical on iOS / Android / Windows / macOS:

  1. Customer plugs in the camera. Pi boots, this daemon notices
     there is no saved WiFi connection, and starts the AP.
  2. Customer connects their phone or laptop to the
     "Nemara-Setup-XXXXXX" WiFi network. On iOS the captive portal
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
  NEMARA_AP_LABEL_PREFIX   optional, default "Nemara-Setup"
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
    prefix = os.environ.get("NEMARA_AP_LABEL_PREFIX", "Nemara-Setup")
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
    """Skip provisioning when EITHER the marker file is present (a
    previous successful run touched it) OR NetworkManager already
    knows a saved WiFi connection (the device was manually flashed
    with credentials by Pi Imager).
    """
    if marker.exists():
        return True
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE", "connection", "show"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout
        return any(
            line.strip() == "802-11-wireless" for line in out.splitlines()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
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


def start_ap(cfg: Config) -> None:
    """Bring up the WiFi access point via NetworkManager.

    We delete any old `Nemara-Setup` connection first so a previous
    aborted run does not leave a stale profile blocking the create.
    """
    log.info("starting AP ssid=%r channel=%s", cfg.ap_ssid, cfg.ap_channel)
    subprocess.run(
        ["nmcli", "connection", "delete", cfg.ap_ssid],
        capture_output=True, check=False,
    )
    subprocess.run(
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
        check=True, capture_output=True,
    )
    # Set an open network (no PSK); customers should not be typing a
    # PSK to access a temporary setup network.
    subprocess.run(
        [
            "nmcli", "connection", "modify", cfg.ap_ssid,
            "wifi-sec.key-mgmt", "none",
        ],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["nmcli", "connection", "up", cfg.ap_ssid],
        check=True, capture_output=True,
    )


def stop_ap(cfg: Config) -> None:
    """Tear down the AP. Best effort; if NM is already shutting it
    down (e.g. because we joined another network) the down call is
    harmless.
    """
    log.info("tearing down AP")
    subprocess.run(
        ["nmcli", "connection", "down", cfg.ap_ssid],
        capture_output=True, check=False,
    )
    subprocess.run(
        ["nmcli", "connection", "delete", cfg.ap_ssid],
        capture_output=True, check=False,
    )


def join_wifi(cfg: Config, ssid: str, psk: str) -> tuple[bool, str]:
    """Attempt to connect to the customer's home WiFi. Returns
    (success, message). On success NetworkManager saves the
    connection so subsequent boots auto-connect.
    """
    log.info("joining home WiFi ssid=%r", ssid)
    try:
        proc = subprocess.run(
            [
                "nmcli", "device", "wifi", "connect", ssid,
                "password", psk, "ifname", cfg.wifi_iface,
            ],
            capture_output=True, text=True, timeout=60,
        )
        if proc.returncode == 0:
            return True, proc.stdout.strip()
        msg = (proc.stderr or proc.stdout).strip()
        log.warning("nmcli connect failed: %s", msg)
        return False, msg
    except subprocess.TimeoutExpired:
        return False, "Connection attempt timed out after 60 seconds."


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
<title>Nemara Camera Setup</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif;
          max-width: 480px; margin: 32px auto; padding: 0 20px;
          color: #1a1a1a; background: #fafafa; }}
  h1 {{ font-size: 22px; margin-bottom: 8px; }}
  .sub {{ color: #666; margin-bottom: 24px; }}
  label {{ display: block; margin-top: 16px; font-weight: 600; }}
  input, select {{ width: 100%; padding: 10px; margin-top: 4px;
                   font-size: 16px; border: 1px solid #ccc; border-radius: 6px; }}
  button {{ width: 100%; margin-top: 24px; padding: 12px;
            font-size: 16px; font-weight: 600;
            background: #1a1a1a; color: #fff; border: 0;
            border-radius: 6px; }}
  .err {{ background: #fee; color: #900; padding: 10px; border-radius: 6px;
          margin-top: 16px; font-size: 14px; }}
</style>
</head><body>
<h1>Set up your Nemara Camera</h1>
<p class="sub">Pick your home WiFi network and enter the password.
The camera will join your home WiFi and disappear from this setup
network within about a minute.</p>
{error_block}
<form method="POST" action="/provision">
  <label for="ssid">Home WiFi network</label>
  {ssid_input}
  <label for="psk">Password</label>
  <input type="password" id="psk" name="psk" autocomplete="off"
         minlength="8" maxlength="63" required>
  <button type="submit">Connect</button>
</form>
</body></html>"""


_SUCCESS_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Success</title>
<style>body{{font-family:-apple-system,system-ui,sans-serif;
max-width:480px;margin:60px auto;padding:0 20px;text-align:center;}}</style>
</head><body>
<h1>Connected!</h1>
<p>Your camera is joining the home WiFi. In about a minute it will
appear in the Nemara app under the &quot;Add a camera&quot; screen.</p>
<p>You can now reconnect this phone or laptop to your home WiFi.</p>
</body></html>"""


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
        return self.path != "/" and self.path != "/provision"

    def do_GET(self) -> None:
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

        ok, msg = join_wifi(self.cfg, ssid, psk)
        if not ok:
            return self._reshow_with_error(f"Connect failed: {msg}")

        _PROVISION_RESULT["ssid"] = ssid
        body = _SUCCESS_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        # Schedule the daemon to shut down so AP mode tears down
        # cleanly; defer briefly so the success page renders on the
        # client before the radio mode switches.
        threading.Timer(_TEARDOWN_DELAY_S, _DONE.set).start()

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
    handler = type("HandlerWithConfig", (_Handler,), {
        "cfg": cfg,
        "nearby_networks": networks,
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
