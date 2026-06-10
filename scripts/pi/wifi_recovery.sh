#!/usr/bin/env bash
# wifi_recovery.sh — fired 5 minutes after boot by
# memoria-wifi-recovery.timer. If the customer hasn't completed
# provisioning by then (no marker file), restore the WiFi backups
# (both NM keyfiles and netplan YAMLs) and reboot back to home WiFi
# so we don't lose SSH access permanently while debugging the AP-mode
# flow.
#
# If the marker DOES exist (customer succeeded), this is a no-op.

set -euo pipefail

MARKER="/var/lib/memoria/provisioned"
BACKUP_DIR="/var/lib/memoria/wifi-backup"
NM_BACKUP="$BACKUP_DIR/nm-system-connections"
NETPLAN_BACKUP="$BACKUP_DIR/netplan"

NM_DIR="/etc/NetworkManager/system-connections"
NETPLAN_DIR="/etc/netplan"

if [[ -f "$MARKER" ]]; then
  echo "marker present (provisioning succeeded); recovery is a no-op" \
    | systemd-cat -t memoria-wifi-recovery
  exit 0
fi

# Belt-and-suspenders: also disable the timer so it doesn't fire on
# every subsequent boot until someone explicitly runs wifi_wipe_safe
# again. The timer is one-shot per provisioning attempt by design.
systemctl disable memoria-wifi-recovery.timer >/dev/null 2>&1 || true

shopt -s nullglob
restored_any=0

for kf in "$NM_BACKUP"/*.nmconnection; do
  cp -p "$kf" "$NM_DIR/"
  echo "restored $kf to $NM_DIR/" \
    | systemd-cat -t memoria-wifi-recovery
  restored_any=1
done

for yaml in "$NETPLAN_BACKUP"/*.yaml; do
  cp -p "$yaml" "$NETPLAN_DIR/"
  echo "restored $yaml to $NETPLAN_DIR/" \
    | systemd-cat -t memoria-wifi-recovery
  restored_any=1
done

if [[ $restored_any -eq 0 ]]; then
  echo "no WiFi backup to restore; staying in current state" \
    | systemd-cat -t memoria-wifi-recovery
  exit 0
fi

echo "rebooting to apply restored WiFi" | systemd-cat -t memoria-wifi-recovery
reboot
