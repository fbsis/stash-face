#!/usr/bin/env bash
set -euo pipefail
umask 022
[[ $EUID -eq 0 ]] || { echo "Run with sudo on the target Ubuntu LXC."; exit 1; }
SOURCE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
apt-get update
GLIB=libglib2.0-0
if apt-cache show libglib2.0-0t64 &>/dev/null; then GLIB=libglib2.0-0t64; fi
apt-get install -y python3-venv ca-certificates "$GLIB"
getent group stash-face >/dev/null || groupadd --system stash-face
id stash-face &>/dev/null || useradd --system --gid stash-face --home-dir /var/lib/stash-face --shell /usr/sbin/nologin stash-face
install -d -m 0755 /opt/stash-face
install -d -m 0700 -o stash-face -g stash-face /var/lib/stash-face
install -d -m 0750 -o root -g stash-face /etc/stash-face
python3 -m venv /opt/stash-face/.venv
/opt/stash-face/.venv/bin/pip install --upgrade pip
TARGET="$SOURCE"
if [[ "${WITH_OPENVINO:-0}" == 1 ]]; then TARGET="$SOURCE[intel]"; fi
/opt/stash-face/.venv/bin/pip install "$TARGET"
if [[ ! -e /etc/stash-face/env ]]; then
  install -m 0640 -o root -g stash-face "$SOURCE/.env.example" /etc/stash-face/env
fi
install -m 0644 "$SOURCE/deploy/stash-face@.service" /etc/systemd/system/stash-face@.service
systemctl daemon-reload
echo "Installed, but NOT started. Configure an instance environment file such as /etc/stash-face/local.env, then:"
echo "  systemctl enable --now stash-face@local"
echo "For multiple Stash instances, create one environment file and data directory per instance."
echo "The InsightFace model pack downloads on first start if it is not already present."
