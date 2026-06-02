#!/usr/bin/env bash
# Provision a fresh Raspberry Pi into a Nemara sensor camera in one
# shot from the developer's Mac.
#
# What this does, in order:
#   1. Verifies we can SSH into the Pi (key-based, no password prompt).
#   2. Copies the scripts/pi/ directory to /tmp/nemara-pi/ on the Pi.
#   3. Runs install.sh on the Pi with the chosen backend URL. install.sh
#      handles: apt deps, dtoverlay=imx219, persistent journal, dnsmasq
#      captive-portal wildcard, systemd units for announce + stream
#      + WiFi-AP + recovery timer.
#   4. (Optional) Wipes the Pi's saved WiFi via wifi_wipe_safe.sh so
#      it reboots into AP mode on a fresh power cycle. Use this for
#      a Pi that is being handed to a customer.
#
# Usage:
#   ./tools/provision_pi.sh <pi-host> <backend-url> [--for-customer]
#
# Example (dev, leave on local WiFi):
#   ./tools/provision_pi.sh nemara-cam.local https://api.dev.nemara
#
# Example (production, ready to ship):
#   ./tools/provision_pi.sh 192.168.1.42 \
#       https://nemara-backend-production.up.railway.app \
#       --for-customer
#
# Env overrides:
#   PI_USER     SSH user on the Pi (default: pi)
#   SSH_OPTS    extra ssh / scp options (default: "")

set -euo pipefail

PI_USER="${PI_USER:-pi}"
SSH_OPTS="${SSH_OPTS:-}"

usage() {
  cat >&2 <<EOF
Usage: $0 <pi-host> <backend-url> [--for-customer]

  <pi-host>       hostname or IP of the Pi (e.g. nemara-cam.local, 10.0.0.44)
  <backend-url>   backend the Pi will announce to and push frames into
  --for-customer  wipe saved WiFi at the end so the Pi boots into AP mode
                  on its next power cycle (use when shipping to a customer)

Env overrides:
  PI_USER         SSH user on the Pi (default: pi)
  SSH_OPTS        extra ssh / scp options (default: "")
EOF
  exit 1
}

if [[ $# -lt 2 ]]; then
  usage
fi

PI_HOST="$1"
BACKEND_URL="$2"
shift 2

FOR_CUSTOMER=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --for-customer) FOR_CUSTOMER=1 ;;
    -h|--help)      usage ;;
    *)              echo "unknown arg: $1" >&2; usage ;;
  esac
  shift
done

# Resolve scripts/pi/ relative to this script (tools/provision_pi.sh).
# Works whether invoked from repo root, from tools/, or via an absolute path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI_SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../scripts/pi" && pwd)"

if [[ ! -f "$PI_SCRIPTS_DIR/install.sh" ]]; then
  echo "could not find install.sh at $PI_SCRIPTS_DIR" >&2
  exit 1
fi

# ServerAliveInterval+CountMax are critical for the --for-customer path:
# wifi_wipe_safe.sh ends in `reboot`, which kills the Pi's network before
# OpenSSH on this end sees a clean FIN. Without keepalive, the SSH socket
# sits in a wait-forever loop until TCP's own ~2hr timeout fires; the
# whole provision script appears to hang. Setting interval=5/count=2
# means we detect the disconnect within ~10s and the wifi_wipe call
# returns with non-zero (already wrapped in `|| true` below), letting
# the trailing "ready for power cycle" instructions actually print.
SSH="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 $SSH_OPTS"
SCP="scp -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 $SSH_OPTS"
TARGET="${PI_USER}@${PI_HOST}"

echo ">>> Verifying SSH access to ${TARGET}"
if ! $SSH "$TARGET" 'echo ok' >/dev/null 2>&1; then
  cat >&2 <<EOF
SSH to ${TARGET} failed.

If this Pi is fresh and uses password auth, run this once first:
    ssh-copy-id ${TARGET}

Then re-run this script.
EOF
  exit 1
fi

echo ">>> Confirming Pi OS + sudo"
PI_INFO=$($SSH "$TARGET" 'cat /etc/os-release 2>/dev/null | grep -E "^(PRETTY_NAME|VERSION_CODENAME)=" ; sudo -n true 2>&1 || echo "sudo-needs-password"')
echo "$PI_INFO" | sed 's/^/    /'
if echo "$PI_INFO" | grep -q "sudo-needs-password"; then
  cat >&2 <<EOF

The Pi requires a password for sudo. install.sh runs as root and
expects passwordless sudo. Add this on the Pi (one-time):

    sudo visudo -f /etc/sudoers.d/nemara
    # add this line:
    ${PI_USER} ALL=(ALL) NOPASSWD: ALL

Then re-run this script.
EOF
  exit 1
fi

echo ">>> Copying scripts/pi/ to ${TARGET}:/tmp/nemara-pi/"
$SSH "$TARGET" 'rm -rf /tmp/nemara-pi && mkdir -p /tmp/nemara-pi'
$SCP -r "$PI_SCRIPTS_DIR"/* "${TARGET}:/tmp/nemara-pi/"
$SSH "$TARGET" 'chmod +x /tmp/nemara-pi/*.sh /tmp/nemara-pi/*.py 2>/dev/null || true'

echo ">>> Running install.sh on Pi with BACKEND_URL=${BACKEND_URL}"
$SSH "$TARGET" "sudo /tmp/nemara-pi/install.sh '${BACKEND_URL}'"

if [[ $FOR_CUSTOMER -eq 1 ]]; then
  echo ">>> Wiping saved WiFi (Pi will reboot into AP mode)"
  # wifi_wipe_safe.sh ends with `reboot`, which kills the SSH session
  # before ssh sees a clean exit. We treat non-zero here as expected.
  # If the wipe step itself errored, its stderr already printed above.
  $SSH "$TARGET" 'sudo /opt/nemara/wifi_wipe_safe.sh' || true
  cat <<EOF

================================================================
Pi provisioned and prepared for a customer.

Next power cycle, the Pi will:
  1. Boot into AP mode with SSID like "Memoria-Setup-XXXXXX"
  2. Serve a captive portal at http://192.168.4.1
  3. Wait for the customer's phone to submit their home WiFi
  4. Reconnect, then announce itself to ${BACKEND_URL}

You can power-cycle the Pi now (unplug + replug).
================================================================
EOF
else
  cat <<EOF

================================================================
Pi provisioned and left on its current WiFi.

It will start announcing to ${BACKEND_URL} after the next service
restart or reboot. To verify:

    ssh ${TARGET} 'sudo journalctl -u nemara-announce.service -f'

When you are ready to hand it to a customer, re-run with
--for-customer to wipe the WiFi.
================================================================
EOF
fi
