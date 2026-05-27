#!/usr/bin/env bash
# wifi_wipe_safe.sh — wipe home WiFi to force the Pi into AP-mode
# provisioning, with an auto-recovery safety net so the device doesn't
# go permanently dark when the AP path fails.
#
# Flow:
#   1. Back up every netplan YAML that declares a `wifis:` block to
#      /var/lib/nemara/wifi-backup/
#   2. Delete those YAMLs from /etc/netplan/
#   3. Remove the provisioning marker
#   4. Enable the nemara-wifi-recovery.timer (5-min auto-restore)
#   5. Reboot so netplan generates configs without WiFi
#
# After reboot:
#   - nemara-wifi-ap.service brings up the AP (or fails loudly)
#   - If a customer completes the captive-portal flow within 5 min,
#     /var/lib/nemara/provisioned is touched, recovery short-circuits.
#   - If 5 min passes with no marker, recovery restores the YAMLs and
#     reboots back to home WiFi so we keep SSH access.
#
# MUST be run as root. Triggers an immediate reboot.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "wifi_wipe_safe.sh must be run with sudo" >&2
  exit 1
fi

BACKUP_DIR="/var/lib/nemara/wifi-backup"
NETPLAN_DIR="/etc/netplan"

echo ">>> Backing up netplan WiFi configs to $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
# Clean any prior backup so a re-run reflects the current state.
rm -f "$BACKUP_DIR"/*.yaml 2>/dev/null || true

found_wifi=0
for yaml in "$NETPLAN_DIR"/*.yaml; do
  [[ -f "$yaml" ]] || continue
  # Match `wifis:` at the start of a line (netplan's top-level key
  # for WiFi config). grep -q is silent; falsy exit just skips.
  if grep -qE "^[[:space:]]*wifis:" "$yaml"; then
    cp "$yaml" "$BACKUP_DIR/"
    echo "    backed up: $yaml"
    rm -v "$yaml"
    found_wifi=1
  fi
done

if [[ $found_wifi -eq 0 ]]; then
  echo "    no netplan WiFi YAMLs found; nothing to back up"
fi

echo ">>> Removing provisioning marker"
rm -fv /var/lib/nemara/provisioned

echo ">>> Enabling 5-minute auto-recovery timer"
systemctl enable --now nemara-wifi-recovery.timer 2>&1 | tail -3

echo ">>> Rebooting in 3s to apply (will lose SSH if no home WiFi)"
sleep 3
reboot
