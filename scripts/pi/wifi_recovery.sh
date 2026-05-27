#!/usr/bin/env bash
# wifi_recovery.sh — fired 5 minutes after boot by
# nemara-wifi-recovery.timer. If the customer hasn't completed
# provisioning by then (no marker file), restore the netplan WiFi
# backup and reboot back to home WiFi so we don't lose SSH access
# permanently while debugging the AP-mode flow.
#
# If the marker DOES exist (customer succeeded), this is a no-op.

set -euo pipefail

MARKER="/var/lib/nemara/provisioned"
BACKUP_DIR="/var/lib/nemara/wifi-backup"
NETPLAN_DIR="/etc/netplan"

if [[ -f "$MARKER" ]]; then
  echo "marker present (provisioning succeeded); recovery is a no-op" \
    | systemd-cat -t nemara-wifi-recovery
  exit 0
fi

# Belt-and-suspenders: also disable the timer so it doesn't fire on
# every subsequent boot until someone explicitly runs wifi_wipe_safe
# again. The timer is one-shot per provisioning attempt by design.
systemctl disable nemara-wifi-recovery.timer >/dev/null 2>&1 || true

shopt -s nullglob
restored_any=0
for yaml in "$BACKUP_DIR"/*.yaml; do
  cp "$yaml" "$NETPLAN_DIR/"
  echo "restored $yaml to $NETPLAN_DIR/" \
    | systemd-cat -t nemara-wifi-recovery
  restored_any=1
done

if [[ $restored_any -eq 0 ]]; then
  echo "no WiFi backup to restore; staying in current state" \
    | systemd-cat -t nemara-wifi-recovery
  exit 0
fi

echo "rebooting to apply restored WiFi" | systemd-cat -t nemara-wifi-recovery
reboot
