#!/usr/bin/env bash
# 安裝資料收集的 systemd user timer（尖峰每 5 分鐘、離峰每 30 分鐘錄一段 CCTV）
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
dest="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$dest"
for f in "$project"/deploy/systemd/*; do
    sed "s|__PROJECT_DIR__|$project|g" "$f" > "$dest/$(basename "$f")"
done
systemctl --user daemon-reload
systemctl --user enable --now tv-record.timer
systemctl --user list-timers 'tv-*'
