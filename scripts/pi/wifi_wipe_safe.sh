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
#   5. Enable memoria-wifi-recovery.timer (5-min auto-restore)
#   6. Reboot
#
# After reboot:
#   - memoria-wifi-ap.service brings up the AP (or fails loudly)
#   - If a customer completes the captive portal within 5 min,
#     /var/lib/memoria/provisioned is touched; recovery short-circuits.
#   - If 5 min passes with no marker, recovery restores both backups
#     and reboots so we keep SSH access.
#
# MUST be run as root. Triggers an immediate reboot.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "wifi_wipe_safe.sh must be run with sudo" >&2
  exit 1
fi

BACKUP_DIR="/var/lib/memoria/wifi-backup"
NM_BACKUP="$BACKUP_DIR/nm-system-connections"
NETPLAN_BACKUP="$BACKUP_DIR/netplan"

NM_DIR="/etc/NetworkManager/system-connections"
NETPLAN_DIR="/etc/netplan"

# Use nullglob so empty globs expand to nothing (not a literal `*.nmconnection`),
# letting the for-loops below short-circuit cleanly under `set -e`.
shopt -s nullglob

# Stop the AP-mode supervisor BEFORE doing anything destructive. Its
# already_provisioned() check ticks every 30s and self-heals the marker
# if it sees marker-absent + nmcli-reports-active-wifi. During our wipe,
# theres an 8-15s window where:
#   - the marker has been rmed (step 1 of inner block)
#   - nmcli reports the wifi as "deactivating" (step 4 deletes it but NM
#     reports it as briefly active during teardown)
# If the supervisor ticks in that window it re-touches the marker and
# the next boot sees state="provisioned" -> no AP. Confirmed 2026-05-28
# from /var/log/memoria-wipe.log + journal: supervisor.tick at 22:34:29
# self-healed between marker-rm at 22:34:11 and reboot at 22:34:21.
# Stopping the service here is sufficient because (a) systemctl stop is
# not "failure" so on-failure restart wont fire, and (b) on next boot
# the service auto-starts and sees marker-absent + no-active-wifi
# (because by then the wipe + reboot has cleared everything) -> AP mode.
echo ">>> Stopping memoria-wifi-ap.service so its self-heal tick cant race the wipe"
systemctl stop memoria-wifi-ap.service 2>/dev/null || true

echo ">>> Backing up NM keyfiles to $NM_BACKUP (skipping Memoria-Setup AP keyfile)"
mkdir -p "$NM_BACKUP"
rm -f "$NM_BACKUP"/*.nmconnection 2>/dev/null || true
nm_backed_up=0
for kf in "$NM_DIR"/*.nmconnection; do
  # The pre-baked Memoria-Setup AP keyfile (installed by install.sh) is the
  # fast path for the supervisors start_ap (~3s vs ~50s). We MUST keep it
  # across the wipe; otherwise the post-wipe AP startup falls back to
  # `nmcli connection add` which triggers the netplan/daemon-reload cascade
  # and the supervisor times out before NM finishes activation.
  if [[ "$(basename "$kf")" == Memoria-Setup-*.nmconnection ]]; then
    echo "    preserving AP keyfile: $kf"
    continue
  fi
  cp -p "$kf" "$NM_BACKUP/"
  echo "    backed up: $kf"
  rm -v "$kf"
  nm_backed_up=$((nm_backed_up + 1))
done
[[ $nm_backed_up -eq 0 ]] && echo "    (no home-wifi keyfiles found)"

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

# Removing files on disk is not enough by itself on Pi OS Trixie. NM keeps
# the connection in runtime state, and on next boot netplan/cloud-init
# re-derives a YAML from that state plus the imager wifi seed in the boot
# partition. So we ALSO have to: remove the AP-mode marker, remove the
# cloud-init wifi seeds from the boot partition, delete the in-memory NM
# connection, and reboot.
#
# Every one of those steps either kills the active wifi mid-execution
# (so SSH dies), or only takes effect on next boot. If we tried to run
# them inline from the SSH session, the destructive nmcli call would
# tear our own connection down before the script reached the reboot
# line — 2026-05-28 incident #1. So we hand the destructive block to
# a transient systemd unit and let it run after this SSH session has
# already exited cleanly.
#
# But that wasnt enough either. 2026-05-28 incident #2: install.sh had
# appended `dtoverlay=imx219` to /boot/firmware/config.txt, which queues
# a "reboot to apply config.txt changes" via systemd-logind. That queued
# reboot fired ~32s into our wipe-execute unit (right between the slow
# nmcli call and cloud-init clean), killing the script BEFORE step 4
# (marker removal). Result: Pi rebooted with wifi torn down but marker
# still present -> supervisor stayed in "provisioned" mode -> no AP.
#
# Fix: assume the unit can be killed at ANY moment. Order operations
# by criticality, fastest+most-important first:
#   1. rm marker          - milliseconds; without this AP wont start
#   2. rm boot seeds      - milliseconds; without this cloud-init regens wifi
#   3. enable recovery    - cheap; safety net for botched provisioning
#   4. nmcli delete       - 20+ seconds, triggers netplan/systemd cascade
#   5. systemctl reboot   - final
# If a queued reboot fires after step 2, the wipe is already effective
# on next boot. Steps 4-5 are nice-to-have polish.
#
# Also dropped `cloud-init clean --logs` entirely: it consumed 17s of CPU
# in the failed run, and removing the boot-partition seeds already
# achieves the goal (cloud-init re-reads them on next boot).
echo ">>> Scheduling destructive wipe + reboot via systemd (will run in 2s, detached)"
systemd-run --no-block --on-active=2s --unit=memoria-wipe-execute \
  --property=TimeoutStartSec=infinity \
  --property=KillMode=process \
  bash -c '
set -uxo pipefail
exec >>/var/log/memoria-wipe.log 2>&1
echo "=== memoria-wipe-execute starting at $(date) ==="

# 1. CRITICAL: remove the provisioning marker FIRST. wifi_ap_provision.py
#    reads this as is-this-Pi-paired; absence forces AP mode on next boot.
#    If everything else fails after this point, the Pi still boots into
#    AP mode and is recoverable via captive portal.
echo "[1/4] removing provisioning marker"
rm -fv /var/lib/memoria/provisioned || true

# 2. CRITICAL: remove the boot-partition seeds. Without this, cloud-init
#    plus netplan would regenerate the wifi connection on the very next
#    boot from the imager seed. Only network-config is purely wifi;
#    user-data has SSH / hostname config we want to preserve, so only
#    strip the network section there.
echo "[2/4] removing boot-partition wifi seeds"
if [ -f /boot/firmware/network-config ]; then
  echo "  removing /boot/firmware/network-config"
  rm -f /boot/firmware/network-config
fi
if [ -f /boot/firmware/user-data ] && grep -qE "^[[:space:]]*(wifi|network):" /boot/firmware/user-data; then
  echo "  stripping network/wifi block from /boot/firmware/user-data"
  cp -p /boot/firmware/user-data "/boot/firmware/user-data.bak.$(date +%Y%m%d%H%M%S)"
  awk "
    /^[[:space:]]*(wifi|network):/ { skip=1; next }
    /^[^[:space:]]/                { skip=0 }
    !skip                          { print }
  " /boot/firmware/user-data > /boot/firmware/user-data.tmp \
    && mv /boot/firmware/user-data.tmp /boot/firmware/user-data
fi

# 3. (Disabled for customer shipping.) The recovery timer used to fire
#    N minutes after boot and restore the home-wifi backup if AP-mode
#    provisioning hadnt completed. For a customer Pi the backed-up
#    wifi is YOUR test wifi from the factory flash -- restoring it at
#    the customers house just makes the Pi try to join a network that
#    isnt there. The supervisor itself loops indefinitely (30-min AP
#    window -> 30s idle -> 30-min AP window -> forever) so a customer
#    who steps away and comes back will still find the AP up. If you
#    need the safety net for in-house dev, enable memoria-wifi-recovery
#    .timer manually after install.
echo "[3/4] recovery timer disabled for customer-mode wipe"

# 4. Polish: delete in-memory NM wifi connection so the active wifi
#    drops NOW rather than at reboot. Not strictly required - steps 1-2
#    are already enough to enter AP mode on next boot. Wrapped in
#    `timeout` because on Pi OS Trixie this triggers netplan regenerate
#    -> systemctl daemon-reload (~13s) -> cascading session shutdowns.
#    If we cant complete in 30s, skip and let the reboot do the work.
#
#    SKIP the pre-baked Memoria-Setup-XXXXXX AP keyfile. Deleting it here
#    would force the supervisors start_ap fast-path check to fail next
#    boot, which then runs the slow `connection add` path and re-triggers
#    the netplan/daemon-reload cascade we just paid the price to avoid
#    by pre-baking. Skipping the AP connection here mirrors the outer
#    scripts NM keyfile loop, which preserves the same file on disk.
echo "[4/4] deleting in-memory NM wifi connections (best-effort, 30s cap; preserving Memoria-Setup-* AP)"
timeout 30 bash -c "
  nmcli -t -f TYPE,NAME connection show \
    | awk -F: \"\\\$1==\\\"802-11-wireless\\\" && \\\$2 !~ /^Memoria-Setup-/{print \\\$2}\" \
    | while IFS= read -r conn; do
        [ -z \"\$conn\" ] && continue
        echo \"  deleting NM connection: \$conn\"
        timeout 8 nmcli connection delete \"\$conn\" || echo \"  (timeout or error, continuing)\"
      done
" || echo "[4/4] timed out at 30s, continuing"

echo "=== rebooting at $(date) ==="
systemctl reboot
'
echo ">>> Wipe scheduled. SSH will exit cleanly now; Pi reboots in ~5s."
exit 0
