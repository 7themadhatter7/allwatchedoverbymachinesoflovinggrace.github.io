#!/bin/bash
# RM Script 01 — Persist OS tuning across reboots
# Requires: sudo
# Safe to re-run.

CONF=/etc/sysctl.d/99-rm-tuning.conf

cat > /tmp/rm-sysctl.conf << 'EOF'
# RM OS tuning — Ghost in the Machine Labs
# Applied: vm params optimised for 121GB RAM, no swap
# Network params optimised for bridge_server short-lived connections
vm.swappiness=1
vm.dirty_ratio=10
vm.dirty_background_ratio=3
vm.nr_hugepages=512
net.core.somaxconn=8192
net.core.netdev_max_backlog=4096
net.ipv4.tcp_tw_reuse=1
net.ipv4.tcp_fin_timeout=15
EOF

sudo cp /tmp/rm-sysctl.conf $CONF
sudo chmod 644 $CONF
sudo sysctl --system > /dev/null 2>&1

echo "[OK] $CONF written and applied"
echo "     Params will survive reboot."
