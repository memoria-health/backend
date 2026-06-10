#!/usr/bin/env bash
# Install the Memoria camera daemons on a fresh Raspberry Pi.
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
# daemon will POST to. It is written into /etc/memoria/announce.env.
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

echo ">>> Deploying daemon scripts to /opt/memoria"
install -d /opt/memoria
install -m 0755 "$SCRIPT_DIR/announce_daemon.py"     /opt/memoria/announce_daemon.py
install -m 0755 "$SCRIPT_DIR/wifi_ap_provision.py"   /opt/memoria/wifi_ap_provision.py
install -m 0755 "$SCRIPT_DIR/mjpeg_pusher.py"        /opt/memoria/mjpeg_pusher.py
install -m 0755 "$SCRIPT_DIR/wifi_wipe_safe.sh"      /opt/memoria/wifi_wipe_safe.sh
install -m 0755 "$SCRIPT_DIR/wifi_recovery.sh"       /opt/memoria/wifi_recovery.sh

echo ">>> Enabling persistent journal so previous-boot logs survive a reboot"
# Pi OS ships /usr/lib/systemd/journald.conf.d/40-rpi-volatile-storage.conf
# which forces Storage=volatile, overriding /etc/systemd/journald.conf.
# We need a higher-numbered drop-in in /etc/ to win the merge.
# Essential for AP-mode debug: when AP setup fails and the recovery
# timer reboots us back to home WiFi, we want
# `journalctl --boot=-1 -u memoria-wifi-ap.service` to show why.
install -d -m 2755 /var/log/journal
install -d /etc/systemd/journald.conf.d
cat > /etc/systemd/journald.conf.d/99-memoria-persistent.conf <<'EOF'
[Journal]
Storage=persistent
SystemMaxUse=64M
EOF
chmod 0644 /etc/systemd/journald.conf.d/99-memoria-persistent.conf
systemctl restart systemd-journald

echo ">>> Writing /etc/memoria/announce.env"
install -d /etc/memoria
cat > /etc/memoria/announce.env <<EOF
MEMORIA_BACKEND_URL=$BACKEND_URL
MEMORIA_ANNOUNCE_INTERVAL_S=60
MEMORIA_WIFI_IFACE=wlan0
MEMORIA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/memoria/announce.env

echo ">>> Deploying lens-shading/color tuning for the Arducam 105° M12 lens"
# The stock libcamera imx219.json is calibrated for the Pi Camera Module 2,
# NOT this Arducam 8MP IMX219 105° M12 lens — its lens-shading (ALSC) leaves
# a magenta vignette ring. imx219_arducam_105_m12.json carries the
# lens-specific ALSC calibration (luminance ~2.05 vs stock ~2.9) measured
# against a flat white field. Deploy it as the active VC4 tuning (the Zero
# 2 W / Pi 3-class ISP), backing up the stock file once. A re-flash without
# this step reverts to stock and the ring returns.
TUNING_SRC="$SCRIPT_DIR/imx219_arducam_105_m12.json"
TUNING_DST="/usr/share/libcamera/ipa/rpi/vc4/imx219.json"
if [[ -f "$TUNING_SRC" && -f "$TUNING_DST" ]]; then
  [[ -f "$TUNING_DST.stock-orig" ]] || cp "$TUNING_DST" "$TUNING_DST.stock-orig"
  install -m 0644 "$TUNING_SRC" "$TUNING_DST"
  echo "    installed Arducam-calibrated tuning -> $TUNING_DST"
else
  echo "    WARN: tuning src/dst missing — leaving stock imx219.json in place"
fi

echo ">>> Writing /etc/memoria/camera-stream.env (per-camera tuning)"
# Full-FOV 1080p15 for a whole-room shot. The IMX219's native 1920x1080
# readout is a center CROP (narrower FOV), so we pin the raw sensor mode
# to the full 3280x2464 and let the ISP downscale to 1080p — this keeps
# the whole-room field of view. Full resolution caps the sensor at
# ~15 fps. Drop to 1280x720 + remove the SENSOR_* lines to fall back to
# the binned full-FOV 720p mode if the Zero 2 W uplink can't sustain it.
#
# QUALITY_LEVEL: the bare hardware MJPEGEncoder is bitrate-targeted and
# caps ~90-100 KB/frame at 1080p regardless of level (MEDIUM..VERY_HIGH
# differ by <15%); VERY_HIGH just requests the encoder's max target.
#
# ── IMAGE QUALITY: CALIBRATED FOR THE ARDUCAM 8MP IMX219 105° M12 LENS ──
# The image tuning here is specific to THIS lens/sensor module (Arducam
# B09VSRH14M, standard IR-cut). It is NOT generic — a different lens/module
# needs its own calibration. Two lens-specific pieces:
#
#   1. Colour — lens shading (ALSC) AND white balance (AWB ct_curve) are
#      baked into the deployed imx219_arducam_105_m12.json tuning (see the
#      deploy step above). The stock tuning targets the original Pi Camera
#      Module 2; on this lens it leaves a magenta vignette ring (ALSC) and
#      a ~25% magenta cast (AWB grey-point model). The bundled tuning
#      carries a flat-field ALSC calibration and an AWB ct_curve corrected
#      (r*1.05, b*1.15) so AUTO white balance lands ~neutral on this lens
#      while still adapting to room lighting. A faint residual cool cast
#      remains; removing it fully needs a multi-colour-temperature grey
#      calibration.
#
#   2. DENOISE=off — the stock libcamera "Fast" denoise massively over-
#      smooths this low-gain (~2.0) module, throwing away ~73% of real
#      detail (lap-var 260 -> 71) to fight grain that barely exists in a
#      lit room. "off" restores detail. Caveat: in genuine low light the
#      gain climbs and "off" gets noisy — switch to a tuned rpi.sdn for
#      night use.
#
# To PIN a fixed white balance instead of auto (perfectly neutral but
# static — only better if the room's light colour never changes), set
# MEMORIA_CAMERA_AWB_GAINS="<red>,<blue>" (1.385,1.448 was the wall-
# calibrated value for this room) to disable auto-AWB.
cat > /etc/memoria/camera-stream.env <<EOF
MEMORIA_CAMERA_WIDTH=1920
MEMORIA_CAMERA_HEIGHT=1080
MEMORIA_CAMERA_FPS=15
MEMORIA_CAMERA_SENSOR_WIDTH=3280
MEMORIA_CAMERA_SENSOR_HEIGHT=2464
MEMORIA_CAMERA_QUALITY_LEVEL=VERY_HIGH
MEMORIA_CAMERA_DENOISE=off
MEMORIA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/memoria/camera-stream.env

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
cat > /etc/NetworkManager/dnsmasq-shared.d/01-memoria-captive.conf <<EOF
address=/#/192.168.4.1
EOF
chmod 0644 /etc/NetworkManager/dnsmasq-shared.d/01-memoria-captive.conf

echo ">>> Writing /etc/memoria/wifi-ap.env"
cat > /etc/memoria/wifi-ap.env <<EOF
MEMORIA_WIFI_IFACE=wlan0
MEMORIA_AP_LABEL_PREFIX=Memoria-Setup
MEMORIA_AP_CHANNEL=6
MEMORIA_AP_BIND_ADDR=192.168.4.1
MEMORIA_AP_HTTP_PORT=80
MEMORIA_PROVISIONED_FILE=/var/lib/memoria/provisioned
MEMORIA_LOG_LEVEL=INFO
EOF
chmod 0644 /etc/memoria/wifi-ap.env

# Pre-bake the Memoria-Setup AP nmconnection so the supervisors start_ap()
# only has to `nmcli connection up`, NOT `connection add`. On Pi OS Trixie,
# `nmcli connection add` triggers NM -> netplan-regenerate -> systemctl
# daemon-reload (~13s) and the chain runs 2-3 times during a single up.
# Pre-baking eliminates the cascade entirely; AP comes up in ~3s instead
# of ~50s. Confirmed via 2026-05-28 FAT32 debug dump.
#
# The keyfile is static config + dynamic SSID suffix (last 6 of MAC,
# matching wifi_ap_provision.load_config()). autoconnect=no means NM
# wont activate it automatically -- only when supervisor calls `up`.
echo ">>> Pre-baking Memoria-Setup AP nmconnection (eliminates start_ap netplan cascade)"
WLAN_MAC=$(cat /sys/class/net/wlan0/address 2>/dev/null || echo "00:00:00:00:00:00")
MAC_SUFFIX=$(echo "$WLAN_MAC" | tr -d ':' | tr '[:lower:]' '[:upper:]' | tail -c 7 | head -c 6)
AP_SSID="Memoria-Setup-${MAC_SUFFIX}"
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
install -m 0644 "$SCRIPT_DIR/memoria-announce.service"      /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/memoria-wifi-ap.service"       /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/memoria-camera-stream.service" /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/memoria-wifi-recovery.service" /etc/systemd/system/
install -m 0644 "$SCRIPT_DIR/memoria-wifi-recovery.timer"   /etc/systemd/system/
# Remove the old BLE provisioning unit from any prior install of
# this script. Failure here is fine (the unit may never have been
# installed).
systemctl disable --now nemara-ble-provision.service 2>/dev/null || true
rm -f /etc/systemd/system/nemara-ble-provision.service /opt/memoria/ble_provision.py /etc/memoria/ble.env
systemctl daemon-reload
systemctl enable memoria-announce.service
systemctl enable memoria-wifi-ap.service
# The camera-stream unit is enabled but won't start until
# /etc/memoria/ingest.env exists (ConditionPathExists). The announce
# daemon writes that file the moment the backend hands us a fresh
# token, then triggers `systemctl restart` on this unit.
systemctl enable memoria-camera-stream.service

# Touch the provisioning marker. wifi_ap_provision.py treats the marker
# as the sole signal for "this Pi has wifi, skip AP mode." Dev-mode
# installs (no `--for-customer`) keep the marker so the Pi stays on
# the imager's wifi across reboots. `wifi_wipe_safe.sh` explicitly
# clears the marker to force the next boot into AP-mode provisioning
# for a customer handoff.
mkdir -p /var/lib/memoria
touch /var/lib/memoria/provisioned

echo ""
if [[ "${NEED_REBOOT:-0}" == "1" ]]; then
  echo ">>> REBOOT REQUIRED for the camera dtoverlay change to take effect:"
  echo "    sudo reboot"
  echo ""
fi
echo ">>> Installed. On next boot:"
echo "    memoria-wifi-ap   advertises an open WiFi AP named"
echo "                     'Memoria-Setup-XXXXXX' (with a captive"
echo "                     portal at http://192.168.4.1) UNTIL"
echo "                     the customer enters home WiFi creds."
echo "    memoria-announce  POSTs to $BACKEND_URL once on home WiFi."
echo ""
echo ">>> To start now without rebooting:"
echo "    sudo systemctl start memoria-wifi-ap.service"
echo "    sudo systemctl start memoria-announce.service"
echo ""
echo ">>> To watch logs:"
echo "    journalctl -u memoria-wifi-ap.service -f"
echo "    journalctl -u memoria-announce.service -f"
echo ""
echo ">>> To re-trigger setup mode (wipe saved WiFi + marker):"
echo "    sudo rm /var/lib/memoria/provisioned"
echo "    sudo nmcli connection delete <home-wifi-ssid>"
echo "    sudo systemctl restart memoria-wifi-ap.service"
