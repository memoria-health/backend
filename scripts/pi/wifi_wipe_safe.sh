#!/usr/bin/env bash
# wifi_wipe_safe.sh — wipe home WiFi to force the Pi into AP-mode
# provisioning, with an auto-recovery safety net so the device doesn't
# go permanently dark when the AP path fails.
#
# Pi OS Trixie has TWO sources of WiFi config that NM reads:
#   1. /etc/NetworkManager/system-connections/*.nmconnection
#        Canonical NM keyfiles. Created by `nmcli connection add`
#        (what wifi_ap_provision.py's join_wifi does after a customer
#        completes the captive portal).
#   2. /etc/netplan/*.yaml
#        Netplan configs. NM auto-generates one of these for every
#        keyfile on boot. So if we only delete one source, the other
#        regenerates it and the Pi rejoins home WiFi instead of
#        entering AP mode.
# We back up + wipe both so the wipe is actually effective, and
# recovery restores both so we don't lose the home WiFi config.
#
# Flow:
#   1. Back up NM keyfiles to $BACKUP_DIR/nm-system-connections/
#   2. Back up netplan WiFi YAMLs to $BACKUP_DIR/netplan/
#   3. Delete originals from both locations
#   4. Remove the provisioning marker
#   5. Enable nemara-wifi-recovery.timer (5-min auto-restore)
#   6. Reboot
#
# After reboot:
#   - nemara-wifi-ap.service brings up the AP (or fails loudly)
#   - If a customer completes the captive portal within 5 min,
#     /var/lib/nemara/provisioned is touched; recovery short-circuits.
#   - If 5 min passes with no marker, recovery restores both backups
#     and reboots so we keep SSH access.
#
# MUST be run as root. Triggers an immediate reboot.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "wifi_wipe_safe.sh must be run with sudo" >&2
  exit 1
fi

BACKUP_DIR="/var/lib/nemara/wifi-backup"
NM_BACKUP="$BACKUP_DIR/nm-system-connections"
NETPLAN_BACKUP="$BACKUP_DIR/netplan"

NM_DIR="/etc/NetworkManager/system-connections"
NETPLAN_DIR="/etc/netplan"

# Use nullglob so empty globs expand to nothing (not a literal `*.nmconnection`),
# letting the for-loops below short-circuit cleanly under `set -e`.
shopt -s nullglob

echo ">>> Backing up NM keyfiles to $NM_BACKUP"
mkdir -p "$NM_BACKUP"
rm -f "$NM_BACKUP"/*.nmconnection 2>/dev/null || true
nm_backed_up=0
for kf in "$NM_DIR"/*.nmconnection; do
  cp -p "$kf" "$NM_BACKUP/"
  echo "    backed up: $kf"
  rm -v "$kf"
  nm_backed_up=$((nm_backed_up + 1))
done
[[ $nm_backed_up -eq 0 ]] && echo "    (none found)"

echo ">>> Backing up netplan WiFi YAMLs to $NETPLAN_BACKUP"
mkdir -p "$NETPLAN_BACKUP"
rm -f "$NETPLAN_BACKUP"/*.yaml 2>/dev/null || true
np_backed_up=0
for yaml in "$NETPLAN_DIR"/*.yaml; do
  # grep -q exits non-zero on no match; allow that without aborting.
  if grep -qE "^[[:space:]]*wifis:" "$yaml"; then
    cp -p "$yaml" "$NETPLAN_BACKUP/"
    echo "    backed up: $yaml"
    rm -v "$yaml"
    np_backed_up=$((np_backed_up + 1))
  fi
done
[[ $np_backed_up -eq 0 ]] && echo "    (no netplan YAMLs declared wifis:)"

if [[ $nm_backed_up -eq 0 && $np_backed_up -eq 0 ]]; then
  echo "ERROR: nothing to wipe; would reboot into AP mode with no home WiFi to restore." >&2
  echo "       Refusing to reboot. Connect to home WiFi first or use a fresh Pi." >&2
  exit 1
fi

echo ">>> Removing provisioning marker"
rm -fv /var/lib/nemara/provisioned

echo ">>> Enabling 5-minute auto-recovery timer"
systemctl enable --now nemara-wifi-recovery.timer 2>&1 | tail -3

echo ">>> Rebooting in 3s to apply (will lose SSH if no home WiFi)"
sleep 3
reboot
