"""Sentinel — host-side watchdog + telemetry for the whole local stack.

One dumb loop that watches every service and, for exactly ONE failure
class, acts. Everything else it observes and journals. Born from the
2026-07-29 05:37 incident: the backend container's outbound
connections to LiveKit wedged for 108 minutes while every health
endpoint answered 200 — process-alive is not data-flowing, so the
sentinel's core probe is FRAME FRESHNESS (newest analyses row age),
the one signal that can't lie about whether the product is working.

Probes (every cycle, journaled to JSONL):
  backend   GET :8002/health
  identity  GET :8012/state (ready flag)
  tunnel    GET :8000/health (pod via ssh forward)
  dashboard GET :8010/
  frames    newest analyses row age via psql (THE product signal)
  gateway   ping memoria-gateway.local
  docker    gvisor in-progress TCP connection count (wedge trend)

Action (the only one): frames stale > STALE_S while backend answers
and the gateway is reachable — the exact 05:37 signature — capture
forensics (curl LiveKit from INSIDE the container and from the host,
docker network state, last gvisor lines) into a timestamped dir, then
`docker restart memoria-backend`. Cooldown + hourly cap so a
persistent failure can't restart-loop the stack. The identity service
is deliberately NOT auto-restarted: it runs on the host (its own
supervisor owns it) and a restart wipes the in-memory body pool.

Run on the HOST (it restarts the container, so it cannot live inside
it):  nohup uv run python tools/sentinel.py >> /tmp/sentinel.log 2>&1 &
Stdlib only — the watchdog must be simpler than what it watches.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CYCLE_S = 30
STALE_S = 300                 # frames older than this = product is blind
COOLDOWN_S = 600              # min gap between restarts
MAX_RESTARTS_PER_HOUR = 3

LIVEKIT_URL = "https://memoria-dev-5v5d1l62.livekit.cloud/"
CONTAINER = "memoria-backend"
GVISOR_LOG = (
    Path.home()
    / "Library/Containers/com.docker.docker/Data/log/host/com.docker.backend.log"
)
JOURNAL = Path("/tmp/sentinel.jsonl")
FORENSICS_ROOT = Path("/tmp/sentinel_forensics")

PROBES = {
    "backend": "http://localhost:8002/health",
    "identity": "http://localhost:8012/state",
    "tunnel": "http://localhost:8000/health",
    "dashboard": "http://localhost:8010/",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# The identity service's endpoints are token-gated. Without this the
# sentinel's identity probe would get 401 — and urllib RAISES on 401, so
# it would read as "identity down" and could restart a healthy service.
IDENTITY_TOKEN = os.environ.get("NEMARA_IDENTITY_STATE_TOKEN", "")


def _identity_req(url: str) -> urllib.request.Request:
    req = urllib.request.Request(url)
    if IDENTITY_TOKEN:
        req.add_header("Authorization", f"Bearer {IDENTITY_TOKEN}")
    return req


def _http(url: str, timeout: float = 4.0) -> int:
    try:
        req = _identity_req(url) if ":8012" in url else url
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except Exception:
        return 0


def _sh(cmd: list[str], timeout: float = 10.0) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout or p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}>"


def probe_frames_age() -> int | None:
    """Worst per-camera analysis age via the backend's own
    /health/freshness — NOT raw SQL. The endpoint excludes cameras the
    user intentionally turned off (disabled / ai_paused / offline), so
    a deliberate disconnect-all reads as "nothing to watch" (None),
    never as staleness — the sentinel must never fight the product's
    disconnect / privacy buttons (Aryaman 2026-07-29). Returns None
    when there are no active cameras or the endpoint is unreachable
    (the backend probe covers the latter)."""
    try:
        with urllib.request.urlopen(
            "http://localhost:8002/health/freshness", timeout=4
        ) as r:
            payload = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code != 503:
            return None
        try:
            payload = json.load(e).get("detail", {})
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None
    ages = payload.get("ages_s") or {}
    return max(ages.values()) if ages else None


def probe_identity_lag() -> float | None:
    """Max per-camera intake lag from the identity service's /state —
    ITS data-freshness signal (the /state HTTP answering proves
    nothing; today it served a frozen snapshot for 4 hours)."""
    try:
        with urllib.request.urlopen(
            "http://localhost:8012/state", timeout=4
        ) as r:
            d = json.load(r)
        lags = [
            v.get("lag_s")
            for v in (d.get("cameras") or {}).values()
            if isinstance(v, dict) and isinstance(v.get("lag_s"), (int, float))
        ]
        return max(lags) if lags else None
    except Exception:  # noqa: BLE001
        return None


IDENTITY_STALE_S = 300


def restart_identity() -> bool:
    """Kill the identity service so its supervisor relaunches it.
    Returns whether anything was actually killed.

    Signals the exact python PID rather than `pkill -f <path>`: that
    pattern also matches the `uv run ...` wrapper, an editor, a tail,
    or the sentinel's own shell — and it silently matches NOTHING if
    identity is running on another host, in which case the sentinel
    would report a restart it never performed and burn a slot.

    NOTE: this wipes the in-memory body pool, which is why the bar is
    high (stale identity intake WHILE backend frames still flow, so
    the fault is provably identity-side) and why the in-service
    resubscribe walk is the first line of defence and this the
    backstop.
    """
    out = _sh(["pgrep", "-f", "python.*gpu/identity/service.py"], timeout=10)
    pids = [p for p in out.split() if p.isdigit()]
    if not pids:
        print(f"{_now()} sentinel: identity restart requested but no local "
              "process matched (running off-host?) — no action taken")
        return False
    for pid in pids:
        _sh(["kill", pid], timeout=5)
    return True


def probe_gateway() -> bool:
    return "1 packets received" in _sh(
        ["ping", "-c", "1", "-W", "2000", "memoria-gateway.local"], timeout=6
    ) or "1 received" in _sh(
        ["ping", "-c", "1", "-W", "2000", "memoria-gateway.local"], timeout=6
    )


def probe_docker_connections() -> int | None:
    """Newest 'N in-progress TCP connections' from the gvisor forwarder
    log — the wedge-suspect trend line."""
    try:
        lines = GVISOR_LOG.read_text(errors="replace").splitlines()[-200:]
    except OSError:
        return None
    for line in reversed(lines):
        if "in-progress TCP connections" in line:
            try:
                return int(line.rsplit("(", 1)[1].split(" ")[0])
            except (IndexError, ValueError):
                return None
    return None


def gateway_zombie_signature() -> bool:
    """True when the fault is on the GATEWAY side of the wire.

    The test is IDENTITY's view: the identity service subscribes to
    the same camera tracks the backend does, from this host. If its
    intake is also stale, both subscribers are starved by the same
    upstream publisher — the gateway is publishing nothing real, and
    restarting the backend spends a restart on the wrong side.

    (The first version counted a "walking for missed track_subscribed"
    log line. That line is only emitted when a participant JOINS, so
    in the incident it was written for — participant stays registered,
    media stops, no rejoin — it never appeared and the sentinel would
    have restarted the backend every cooldown instead. It detects a
    flapping box, not a silent one.)
    """
    lag = probe_identity_lag()
    return lag is not None and lag > 60


# Key-only access to the gateway box. There is NO password path here on
# purpose: this reaches hardware that sits in someone's home, and a
# shared password that works from any machine on the LAN is not an
# acceptable control for that, however carefully it is passed around.
# The box holds `memoria_ops.pub` in authorized_keys, and one narrow
# sudoers rule lets this user restart exactly one service and nothing
# else — so the automation never needs a secret it could leak.
GATEWAY_HOST = os.environ.get(
    "NEMARA_GATEWAY_SSH_HOST", "potaton00b@memoria-gateway.local")
GATEWAY_KEY = os.environ.get(
    "NEMARA_GATEWAY_SSH_KEY", str(Path.home() / ".ssh" / "memoria_ops"))


def restart_gateway() -> str:
    """systemctl-restart the gateway box service over key-auth ssh.

    Requires, one time on the box:
      * `memoria_ops.pub` in ~/.ssh/authorized_keys
      * /etc/sudoers.d/memoria-gateway-watchdog granting NOPASSWD on
        `systemctl restart|is-active memoria-gateway` only

    BatchMode means a box missing either one fails fast and says so,
    rather than hanging the sentinel on a prompt nobody will answer.
    """
    if not Path(GATEWAY_KEY).exists():
        return f"GATEWAY_RESTART_SKIPPED: no ssh key at {GATEWAY_KEY}"
    out = _sh([
        "ssh", "-i", GATEWAY_KEY,
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        GATEWAY_HOST,
        "sudo -n systemctl restart memoria-gateway && "
        "sudo -n systemctl is-active memoria-gateway",
    ], timeout=60)
    if "active" in out and "inactive" not in out:
        return f"GATEWAY_RESTARTED {out.strip()}"
    if "authentication is required" in out or "a password is required" in out:
        return (
            "GATEWAY_RESTART_FAILED: sudo wants a password — install "
            "/etc/sudoers.d/memoria-gateway-watchdog on the box. " + out.strip()
        )
    return f"GATEWAY_RESTART_FAILED {out.strip()}"


def capture_forensics(tag: str) -> Path:
    d = FORENSICS_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "container_curl_livekit.txt").write_text(_sh([
        "docker", "exec", CONTAINER, "python", "-c",
        "import urllib.request,time;t=time.time()\n"
        "try:\n"
        f"  r=urllib.request.urlopen('{LIVEKIT_URL}',timeout=8)\n"
        "  print('status',r.status,'in',round(time.time()-t,2),'s')\n"
        "except Exception as e:\n"
        "  print('FAIL',type(e).__name__,e,'in',round(time.time()-t,2),'s')",
    ], timeout=20))
    host_status = _http(LIVEKIT_URL, timeout=8)
    (d / "host_curl_livekit.txt").write_text(f"status {host_status}")
    (d / "docker_network.txt").write_text(
        _sh(["docker", "network", "inspect", "bridge"], timeout=10))
    (d / "container_netstat.txt").write_text(_sh(
        ["docker", "exec", CONTAINER, "sh", "-c",
         "cat /proc/net/tcp | wc -l; cat /proc/net/tcp6 | wc -l"], timeout=10))
    try:
        (d / "gvisor_tail.txt").write_text(
            "\n".join(GVISOR_LOG.read_text(errors="replace").splitlines()[-300:]))
    except OSError:
        pass
    return d


def main() -> None:
    print(f"{_now()} sentinel: start (stale>{STALE_S}s -> forensics + restart)")
    budgets: dict[str, list[float]] = {
        "identity": [], "backend": [], "gateway": [],
    }
    while True:
        time.sleep(CYCLE_S)
        row: dict = {"ts": _now()}
        for name, url in PROBES.items():
            row[name] = _http(url)
        row["frames_age_s"] = probe_frames_age()
        row["docker_conns"] = probe_docker_connections()
        row["identity_lag_s"] = probe_identity_lag()

        # Identity-side stall: its intake is stale while the BACKEND's
        # frames are fresh — the same tracks reach one subscriber and
        # not the other, so the fault is provably in identity's
        # subscription, not upstream. (Both stale = upstream; handled
        # by the frames rule below.)
        idl = row["identity_lag_s"]
        now = time.monotonic()
        # PER-TARGET budgets. A shared one let a repeatedly-failing
        # identity restart consume the whole hour's allowance, after
        # which a real backend wedge — the product going blind — could
        # not be acted on at all.
        for tgt in budgets:
            budgets[tgt] = [t for t in budgets[tgt] if now - t < 3600]

        def _may(target: str) -> bool:
            b = budgets[target]
            return (
                len(b) < MAX_RESTARTS_PER_HOUR
                and (not b or now - b[-1] >= COOLDOWN_S)
            )

        def _spend(target: str) -> None:
            budgets[target].append(now)

        if (
            idl is not None and idl > IDENTITY_STALE_S
            and row["frames_age_s"] is not None
            and row["frames_age_s"] < STALE_S
            and _may("identity")
        ):
            print(f"{_now()} sentinel: identity intake stale {idl:.0f}s "
                  "while backend frames are fresh — restarting identity "
                  "(supervisor relaunches; body pool is lost)")
            if restart_identity():
                row["action"] = "identity_restart"
                _spend("identity")
            else:
                row["action"] = "identity_restart_no_local_process"

        stale = (
            row["frames_age_s"] is not None and row["frames_age_s"] > STALE_S
        )
        if stale:
            # Only act on the 05:37 signature: product blind while the
            # pieces claim health. A dead gateway or dead backend is a
            # DIFFERENT failure the sentinel reports but must not
            # "fix" by restarting the wrong thing.
            row["gateway_ping"] = probe_gateway()
            actionable = row["backend"] == 200 and row["gateway_ping"]
            # Route the restart to the FAULTY side of the wire: if the
            # identity service is starved of the same tracks, the
            # publisher upstream is the problem, not the backend.
            gateway_side = gateway_zombie_signature()
            target = "gateway" if gateway_side else "backend"
            if actionable and _may(target):
                print(f"{_now()} sentinel: frames stale "
                      f"{row['frames_age_s']}s with healthy backend+gateway "
                      f"— forensics, then restarting {target.upper()}")
                d = capture_forensics("stale_frames")
                row["forensics"] = str(d)
                if gateway_side:
                    result = restart_gateway()
                    row["action"] = "gateway_restart"
                    row["action_result"] = result[-120:]
                else:
                    _sh(["docker", "restart", CONTAINER], timeout=90)
                    row["action"] = "docker_restart"
                _spend(target)
            else:
                row["action"] = "observed_only"
        with JOURNAL.open("a") as f:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
