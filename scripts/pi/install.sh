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
#
# What this installs:
#   wifi_ap_provision.py    First-boot WiFi setup via an open AP
#                           + captive portal. Runs once on a fresh
#                           Pi with no saved WiFi, then short-
#                           circuits forever after a successful
#                           provision (marker file).
#   announce_daemon.py      Heartbeat to the backend so the caregiver
#                           app's "Add a camera" picker can see this
#                           device.

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
# network-manager:    nmcli for both AP mode and home-WiFi joining
# python3-httpx:      used by both announce daemon and push streamer
# python3-picamera2:  used by the push streamer to capture frames
# python3-pil:        Pillow, used by the push streamer for JPEG encode
apt-get install -y --no-install-recommends \
  network-manager python3-httpx python3-picamera2 python3-pil

echo ">>> Ensuring camera dtoverlay is enabled"
# IMX219 (Camera Module 2 / Arducam IMX219) needs the dtoverlay
# explicitly enabled in /boot/firmware/config.txt. Pi OS Lite does NOT
# auto-detect MIPI camera modules at boot; without this line,
# libcamera reports zero cameras and picamera2 raises IndexError.
# Idempotent: only appends if not already present.
CONFIG_TXT="/boot/firmware/config.txt"
if [[ -f "$CONFIG_TXT" ]] && ! grep -q "^dtoverlay=imx219" "$CONFIG_TXT"; then
  echo "dtoverlay=imx219" >> "$CONFIG_TXT"
  echo "    added dtoverlay=imx219 (reboot required to take effect)"
  NEED_REBOOT=1
else
  echo "    dtoverlay already configured"
fi

echo ">>> Deploying daemon scripts to /opt/nemara"
install -d /opt/nemara
install -m 0755 "$SCRIPT_DIR/announce_daemon.py"     /opt/nemara/announce_daemon.py
install -m 0755 "$SCRIPT_DIR/wifi_ap_provision.py"   /opt/nemara/wifi_ap_provision.py
install -m 0755 "$SCRIPT_DIR/mjpeg_pusher.py"        /opt/nemara/mjpeg_pusher.py
install -m 0755 "$SCRIPT_DIR/wifi_wipe_safe.sh"      /opt/nemara/wifi_wipe_safe.sh
install -m 0755 "$SCRIPT_DIR/wifi_recovery.sh"       /opt/nemara/wifi_recovery.sh

echo ">>> Enabling persistent journal so previous-boot logs survive a reboot"
# Pi OS ships /usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf
# which forces Storage=volatile, overriding /etc/systemd/journald.conf.
# We need a higher-numbered drop-in in /etc/ to win the merge.
# Essential for AP-mode debug: when AP setup fails and the recovery
# timer reboots us back to home WiFi, we want
# `journalctl --boot=-1 -u nemara-wifi-ap.service` to show why.
install -d -m 2755 /var/log/journal
install -d /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/99-nemara-persistent.conf <<'EOF'
[Journal]
Storage=persistent
SystemMaxUse=64M
EOF
chmod 0644 /etc/systemd/journald.conf.d/99-nemara-persistent.conf
systemctl restart systemd-journald

echo ">>> Writing /etc/nemara/announce.env"
install -d /etc/nemara
cat > /etc/nemara/announce.env <<EOF
NEMARA_BACKEND_URL=$BACKEND_URL
NEMARA_ANNOUNCE_INTERVAL_S=60
NEMARA_WIFI_IFACE=wlan0
NEMARA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/nemara/announce.env

echo ">>> Writing /etc/nemara/camera-stream.env (per-camera tuning)"
# Pinned to 720p15 @ 4 Mbps: matches the iPhone publish ceiling
# closely, stays comfortably within typical home WiFi uplink, and is
# the rate the MJPEG hardware encoder was validated against on Pi
# Zero W (sustains the full 15 fps without dropping). The QUALITY
# env var is legacy from the libjpeg path and is ignored by the
# current MJPEGEncoder; left in place so existing installs don't
# error on a missing var, but BITRATE_BPS is the real knob now.
cat > /etc/nemara/camera-stream.env <<EOF
NEMARA_CAMERA_WIDTH=1280
NEMARA_CAMERA_HEIGHT=720
NEMARA_CAMERA_FPS=15
NEMARA_CAMERA_BITRATE_BPS=4000000
NEMARA_CAMERA_QUALITY=80
NEMARA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/nemara/camera-stream.env

echo ">>> Installing captive-portal DNS wildcard for NM shared mode"
# When NetworkManager runs a wifi connection with ipv4.method=shared
# (our AP), it launches dnsmasq for DHCP + DNS. By default that
# dnsmasq forwards DNS upstream, but the AP has no upstream, so
# captive-portal probes (captive.apple.com, connectivitycheck.gstatic.com,
# etc.) just time out and the phone shows a "no internet" toast
# instead of popping the portal sheet. This drop-in tells dnsmasq to
# answer 192.168.4.1 (the Pi) for every name, which makes the probe
# 302-redirect to our setup page and triggers the captive portal UI.
install -d /etc/NetworkManager/dnsmasq-shared.d
cat > /etc/NetworkManager/dnsmasq-shared.d/01-nemara-captive.conf <<EOF
address=/#/192.168.4.1
EOF
chmod 0644 /etc/NetworkManager/dnsmasq-shared.d/01-nemara-captive.conf

echo ">>> Writing /etc/nemara/wifi-ap.env"
cat > /etc/nemara/wifi-ap.env <<EOF
NEMARA_WIFI_IFACE=wlan0
NEMARA_AP_LABEL_PREFIX=Nemara-Setup
NEMARA_AP_CHANNEL=6
NEMARA_AP_BIND_ADDR=192.168.4.1
NEMARA_AP_HTTP_PORT=80
NEMARA_PROVISIONED_FILE=/var/lib/nemara/provisioned
NEMARA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/nemara/wifi-ap.env

# Pre-bake the Nemara-Setup AP nmconnection so the supervisors start_ap()
# only has to `nmcli connection up`, NOT `connection add`. On Pi OS Trixie,
# `nmcli connection add` triggers NM -> netplan-regenerate -> systemctl
# daemon-reload (~13s) and the chain runs 2-3 times during a single up.
# Pre-baking eliminates the cascade entirely; AP comes up in ~3s instead
# of ~50s. Confirmed via 2026-05-28 FAT32 debug dump.
#
# The keyfile is static config + dynamic SSID suffix (last 6 of MAC,
# matching wifi_ap_provision.load_config()). autoconnect=no means NM
# wont activate it automatically -- only when supervisor calls `up`.
echo ">>> Pre-baking Nemara-Setup AP nmconnection (eliminates start_ap netplan cascade)"
WLAN_MAC=$(cat /sys/class/net/wlan0/address 2>/dev/null || echo "00:00:00:00:00:00")
MAC_SUFFIX=$(echo "$WLAN_MAC" | tr -d ':' | tr '[:lower:]' '[:upper:]' | tail -c 7 | head -c 6)
AP_SSID="Nemara-Setup-${MAC_SUFFIX}"
AP_KEYFILE="/etc/NetworkManager/system-connections/${AP_SSID}.nmconnection"
AP_UUID=$(cat /proc/sys/kernel/random/uuid)
mkdir -p /etc/NetworkManager/system-connections
cat > "$AP_KEYFILE" <<EOF
[connection]
id=${AP_SSID}
uuid=${AP_UUID}
type=wifi
autoconnect=false
interface-name=wlan0

[wifi]
band=bg
channel=6
mode=ap
ssid=${AP_SSID}

[ipv4]
address1=192.168.4.1/24
method=shared

[ipv6]
method=ignore
EOF
chmod 0600 "$AP_KEYFILE"
chown root:root "$AP_KEYFILE"
echo "    wrote $AP_KEYFILE (ssid=$AP_SSID)"
# Reload NM so it picks up the new keyfile without a restart.
nmcli connection reload 2>/dev/null || true

echo ">>> Installing systemd units"
install -m 0644 "$SCRIPT_DIR/nemara-announce.service"      /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/nemara-wifi-ap.service"       /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/nemara-camera-stream.service" /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/nemara-wifi-recovery.service" /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/nemara-wifi-recovery.timer"   /etc/systemd/system/
# Remove the old BLE provisioning unit from any prior install of
# this script. Failure here is fine (the unit may never have been
# installed).
systemctl disable --now nemara-ble-provision.service 2>/dev/null || true
rm -f /etc/systemd/system/nemara-ble-provision.service /opt/nemara/ble_provision.py /etc/nemara/ble.env
systemctl daemon-reload
systemctl enable nemara-announce.service
systemctl enable nemara-wifi-ap.service
# The camera-stream unit is enabled but won't start until
# /etc/nemara/ingest.env exists (ConditionPathExists). The announce
# daemon writes that file the moment the backend hands us a fresh
# token, then triggers `systemctl restart` on this unit.
systemctl enable nemara-camera-stream.service

# Touch the provisioning marker. wifi_ap_provision.py treats the marker
# as the sole signal for "this Pi has wifi, skip AP mode." Dev-mode
# installs (no `--for-customer`) keep the marker so the Pi stays on
# the imager's wifi across reboots. `wifi_wipe_safe.sh` explicitly
# clears the marker to force the next boot into AP-mode provisioning
# for a customer handoff.
mkdir -p /var/lib/nemara
touch /var/lib/nemara/provisioned

echo ""
if [[ "${NEED_REBOOT:-0}" == "1" ]]; then
  echo ">>> REBOOT REQUIRED for the camera dtoverlay change to take effect:"
  echo "    sudo reboot"
  echo ""
fi
echo ">>> Installed. On next boot:"
echo "    nemara-wifi-ap   advertises an open WiFi AP named"
echo "                     'Nemara-Setup-XXXXXX' (with a captive"
echo "                     portal at http://192.168.4.1) UNTIL"
echo "                     the customer enters home WiFi creds."
echo "    nemara-announce  POSTs to $BACKEND_URL once on home WiFi."
echo ""
echo ">>> To start now without rebooting:"
echo "    sudo systemctl start nemara-wifi-ap.service"
echo "    sudo systemctl start nemara-announce.service"
echo ""
echo ">>> To watch logs:"
echo "    journalctl -u nemara-wifi-ap.service -f"
echo "    journalctl -u nemara-announce.service -f"
echo ""
echo ">>> To re-trigger setup mode (wipe saved WiFi + marker):"
echo "    sudo rm /var/lib/nemara/provisioned"
echo "    sudo nmcli connection delete <home-wifi-ssid>"
echo "    sudo systemctl restart nemara-wifi-ap.service"
