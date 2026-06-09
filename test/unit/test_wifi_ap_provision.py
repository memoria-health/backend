"""Unit tests for the Pi WiFi-AP provisioning daemon's join/verify and
captive-portal rendering — the false-success + missing-manual-entry
bugs from 2026-06-06.

Runs on any machine (no Pi, no nmcli): network calls are either pure
functions or monkeypatched. Locks the regression so a join can never
again be reported successful without a real DHCP lease, and the captive
portal always offers manual SSID entry.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# scripts/pi/ has no __init__.py (deployed scripts), so import by path.
_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "pi" / "wifi_ap_provision.py"
)
_spec = importlib.util.spec_from_file_location("wifi_ap_provision", _MODULE_PATH)
assert _spec is not None and _spec.loader is not None
wap = importlib.util.module_from_spec(_spec)
# Register before exec_module: @dataclass resolves field types via
# sys.modules[cls.__module__], which is None for an unregistered module.
sys.modules["wifi_ap_provision"] = wap
_spec.loader.exec_module(wap)


# ── _parse_device_connectivity (pure) ────────────────────────────────

def test_parse_connected_with_ip():
    out = "GENERAL.STATE:100 (connected)\nIP4.ADDRESS[1]:192.168.1.50/24\n"
    connected, ip = wap._parse_device_connectivity(out)
    assert connected is True
    assert ip == "192.168.1.50/24"


def test_parse_wrong_password_no_lease():
    # Wrong PSK: never reaches connected, no IP.
    connected, ip = wap._parse_device_connectivity("GENERAL.STATE:30 (disconnected)\n")
    assert connected is False
    assert ip is None


def test_parse_connecting_no_ip():
    assert wap._parse_device_connectivity("GENERAL.STATE:50 (connecting)\n") == (
        False,
        None,
    )


def test_parse_connected_but_no_ip_is_failure():
    # Associated but DHCP failed -> not a usable join.
    connected, ip = wap._parse_device_connectivity("GENERAL.STATE:100 (connected)\n")
    assert connected is False
    assert ip is None


# ── join_wifi branching (subprocess + verify mocked) ─────────────────

def _cfg():
    return SimpleNamespace(wifi_iface="wlan0")


def _fake_run_factory(up_rc=0, calls=None):
    def fake_run(args, **kwargs):
        if calls is not None:
            calls.append(list(args))
        rc = 0
        if list(args[:3]) == ["nmcli", "connection", "up"]:
            rc = up_rc
        return subprocess.CompletedProcess(args, rc, stdout="", stderr="")
    return fake_run


def test_join_wifi_success_requires_lease(monkeypatch):
    monkeypatch.setattr(wap.subprocess, "run", _fake_run_factory(up_rc=0))
    monkeypatch.setattr(wap, "_verify_connectivity", lambda iface, t: (True, "10.0.0.5/24"))
    ok, msg = wap.join_wifi(_cfg(), "MyNet", "password123")
    assert ok is True
    assert "10.0.0.5" in msg


def test_join_wifi_rejects_up_rc0_without_lease(monkeypatch):
    # THE bug: nmcli `up` returns 0 but there is no real connection.
    calls: list[list[str]] = []
    monkeypatch.setattr(wap.subprocess, "run", _fake_run_factory(up_rc=0, calls=calls))
    monkeypatch.setattr(wap, "_verify_connectivity", lambda iface, t: (False, None))
    ok, msg = wap.join_wifi(_cfg(), "WrongNet", "wrongpass")
    assert ok is False
    assert "password" in msg.lower()
    # The dead profile must be scrubbed so NM doesn't autoconnect to it.
    assert any(
        list(c[:3]) == ["nmcli", "connection", "delete"] and "WrongNet" in c
        for c in calls
    )


def test_join_wifi_rejects_up_nonzero(monkeypatch):
    # A non-zero `up` is already a failure regardless of any later check.
    monkeypatch.setattr(wap.subprocess, "run", _fake_run_factory(up_rc=4))
    monkeypatch.setattr(wap, "_verify_connectivity", lambda iface, t: (True, "1.2.3.4/24"))
    ok, _msg = wap.join_wifi(_cfg(), "WrongNet", "wrongpass")
    assert ok is False


# ── captive portal: manual SSID entry always available ───────────────

def test_render_dropdown_plus_optional_textbox():
    html = wap._render_setup_page(["Suburban Living", "Neighbor"]).decode()
    assert "<select" in html            # readable dropdown of scanned nets
    assert "Suburban Living" in html    # as a dropdown option
    assert 'name="ssid_manual"' in html  # separate optional text box
    assert 'type="text"' in html


def test_render_textbox_when_scan_empty():
    html = wap._render_setup_page([]).decode()
    assert 'name="ssid_manual"' in html
    assert 'type="text"' in html


def test_pick_ssid_manual_overrides_dropdown():
    assert wap._pick_ssid("Neighbor", "Suburban Living") == "Suburban Living"


def test_pick_ssid_falls_back_to_dropdown():
    assert wap._pick_ssid("Suburban Living", "") == "Suburban Living"
    assert wap._pick_ssid("Suburban Living", "   ") == "Suburban Living"


def test_pick_ssid_empty():
    assert wap._pick_ssid("", "") == ""
    assert wap._pick_ssid(None, None) == ""


# ── scan_nearby_networks: multi-pass union catches a missed beacon ───

def test_scan_unions_multiple_passes(monkeypatch):
    # Pass 1's scan misses Suburban Living (weak beacon); a later pass
    # catches it. The union must include it, de-duped, order preserved.
    outputs = iter([
        "Neighbor\n\n",                 # pass 1: missed Suburban
        "Neighbor\nSuburban Living\n",  # pass 2: caught it
        "Suburban Living\n",            # pass 3
    ])
    monkeypatch.setattr(
        wap.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=next(outputs), stderr=""),
    )
    nets = wap.scan_nearby_networks("wlan0", passes=3)
    assert nets == ["Neighbor", "Suburban Living"]


def test_scan_survives_a_failed_pass(monkeypatch):
    outputs = iter([
        subprocess.CalledProcessError(1, "nmcli"),  # pass 1 errors
        "Suburban Living\n",                          # pass 2 succeeds
        "Suburban Living\n",
    ])

    def fake_run(*a, **k):
        nxt = next(outputs)
        if isinstance(nxt, Exception):
            raise nxt
        return subprocess.CompletedProcess(a, 0, stdout=nxt, stderr="")

    monkeypatch.setattr(wap.subprocess, "run", fake_run)
    assert wap.scan_nearby_networks("wlan0", passes=3) == ["Suburban Living"]


# ── _verify_connectivity polling (subprocess mocked) ─────────────────

def test_verify_connectivity_true_on_lease(monkeypatch):
    monkeypatch.setattr(wap.time, "sleep", lambda *_: None)
    out = "GENERAL.STATE:100 (connected)\nIP4.ADDRESS[1]:10.0.0.9/24\n"
    monkeypatch.setattr(
        wap.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=out, stderr=""),
    )
    ok, ip = wap._verify_connectivity("wlan0", timeout_s=5.0)
    assert ok is True
    assert ip == "10.0.0.9/24"


def test_verify_connectivity_times_out_without_lease(monkeypatch):
    monkeypatch.setattr(wap.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        wap.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(
            a, 0, stdout="GENERAL.STATE:30 (disconnected)\n", stderr=""),
    )
    ok, ip = wap._verify_connectivity("wlan0", timeout_s=0.0)
    assert ok is False
    assert ip is None


# ── failed join must signal a prompt AP-restart (no 30-min blackout) ──

def test_join_then_signal_failure_signals_retry(monkeypatch):
    wap._RETRY.clear()
    wap._DONE.clear()
    monkeypatch.setattr(wap.time, "sleep", lambda *_: None)
    monkeypatch.setattr(wap, "stop_ap", lambda cfg: None)
    monkeypatch.setattr(wap, "join_wifi", lambda cfg, ssid, psk: (False, "wrong password"))
    monkeypatch.setattr(wap, "_persist_join_error", lambda msg: None)
    wap._join_then_signal(SimpleNamespace(wifi_iface="wlan0"), "WrongNet", "badpass")
    assert wap._RETRY.is_set()      # serve() will bring the AP back now
    assert not wap._DONE.is_set()   # and it must NOT mark provisioned


def test_join_then_signal_success_signals_done(monkeypatch):
    wap._RETRY.clear()
    wap._DONE.clear()
    monkeypatch.setattr(wap.time, "sleep", lambda *_: None)
    monkeypatch.setattr(wap, "stop_ap", lambda cfg: None)
    monkeypatch.setattr(wap, "join_wifi", lambda cfg, ssid, psk: (True, "connected ip=10.0.0.5/24"))
    wap._join_then_signal(SimpleNamespace(wifi_iface="wlan0"), "GoodNet", "goodpass")
    assert wap._DONE.is_set()
    assert not wap._RETRY.is_set()
