#!/usr/bin/env bash
# 安裝資料收集的 systemd user timer（每 5 分鐘錄影、每天 04:00 下載官方資料）
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
dest="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$dest"
for f in "$project"/deploy/systemd/*; do
    sed "s|__PROJECT_DIR__|$project|g" "$f" > "$dest/$(basename "$f")"
done
systemctl --user daemon-reload
systemctl --user enable --now tv-record.timer tv-official.timer
systemctl --user list-timers 'tv-*'
