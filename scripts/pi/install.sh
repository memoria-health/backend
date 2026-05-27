#!/usr/bin/env bash
# Install the Nemara camera daemons on a fresh Raspberry Pi.
#
# Requires Raspberry Pi OS Bookworm or Trixie (Lite is fine).
# Must be run with sudo.
#
# Usage on a fresh Pi (after burning the image and SSHing in):
#
#   git clone https://github.com/<your-fork>/nemara-backend.git
#   cd nemara-backend/scripts/pi
#   sudo ./install.sh https://api.your-nemara-backend.example
#
# The single positional argument is the backend URL the announce
# daemon will POST to. It is written into /etc/nemara/announce.env.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "install.sh must be run with sudo" >&2
  exit 1
fi

if [[ $# -ne 1 ]]; then
  echo "Usage: sudo $0 <backend-url>" >&2
  echo "  e.g. sudo $0 https://api.nemara.example" >&2
  exit 1
fi

BACKEND_URL="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Installing system packages"
apt-get update
# bluez:           BLE peripheral support via D-Bus
# python3-pip:     to install bless
# network-manager: nmcli for WiFi credential handoff
apt-get install -y --no-install-recommends \
  bluez network-manager python3-pip python3-httpx

echo ">>> Installing Python deps for BLE (system-wide pip with PEP 668 override)"
# PEP 668 marks Pi OS Python as externally managed; we need the
# --break-system-packages flag since we are installing into the
# system interpreter that systemd will run as.
pip install --break-system-packages bless

echo ">>> Deploying daemon scripts to /opt/nemara"
install -d /opt/nemara
install -m 0755 "$SCRIPT_DIR/announce_daemon.py" /opt/nemara/announce_daemon.py
install -m 0755 "$SCRIPT_DIR/ble_provision.py"   /opt/nemara/ble_provision.py

echo ">>> Writing /etc/nemara/announce.env"
install -d /etc/nemara
cat > /etc/nemara/announce.env <<EOF
NEMARA_BACKEND_URL=$BACKEND_URL
# `{ip}` is replaced with this device's current WiFi IP on every
# announcement (so a new DHCP lease pushes the new URL).
NEMARA_CAMERA_STREAM_URL=http://{ip}:8000/stream.mjpg
# Uncomment when a USB mic is attached and an audio daemon serves PCM:
# NEMARA_CAMERA_AUDIO_URL=http://{ip}:8001/audio.pcm
NEMARA_ANNOUNCE_INTERVAL_S=60
NEMARA_WIFI_IFACE=wlan0
NEMARA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/nemara/announce.env

echo ">>> Writing /etc/nemara/ble.env"
cat > /etc/nemara/ble.env <<EOF
NEMARA_WIFI_IFACE=wlan0
NEMARA_BLE_LABEL_PREFIX=Nemara Camera
NEMARA_PROVISIONED_FILE=/var/lib/nemara/provisioned
NEMARA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/nemara/ble.env

echo ">>> Installing systemd units"
install -m 0644 "$SCRIPT_DIR/nemara-announce.service"      /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/nemara-ble-provision.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable nemara-announce.service
systemctl enable nemara-ble-provision.service

echo ""
echo ">>> Installed. Both daemons will start on next boot."
echo "    nemara-ble-provision: advertises BLE until WiFi is configured"
echo "    nemara-announce:     POSTs to $BACKEND_URL once on WiFi"
echo ""
echo ">>> To start now without rebooting:"
echo "    sudo systemctl start nemara-ble-provision.service"
echo "    sudo systemctl start nemara-announce.service"
echo ""
echo ">>> To watch logs:"
echo "    journalctl -u nemara-ble-provision.service -f"
echo "    journalctl -u nemara-announce.service -f"
