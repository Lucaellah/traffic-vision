#!/usr/bin/env bash
# 把網站部署在這台機器：Streamlit（GPU 推論）+ ngrok 固定網址
#
# 事前準備（只要做一次）：
#   1. 到 https://dashboard.ngrok.com 免費註冊，在 Domains 領一個固定網域
#   2. ngrok config add-authtoken <你的 authtoken>
#   3. 在 ~/.config/traffic-vision/web.env 寫入：
#        TV_PASSWORD=<網站密碼>
#        NGROK_DOMAIN=<你的網域>.ngrok-free.dev
set -euo pipefail
project="$(cd "$(dirname "$0")/.." && pwd)"
dest="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
env_file="$HOME/.config/traffic-vision/web.env"
ngrok="$HOME/.local/bin/ngrok"

if [[ ! -x "$ngrok" ]]; then
    mkdir -p "$HOME/.local/bin"
    curl -fsSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz \
        | tar -xz -C "$HOME/.local/bin"
fi

if ! grep -q '^TV_PASSWORD=.' "$env_file" 2>/dev/null \
        || ! grep -q '^NGROK_DOMAIN=.' "$env_file"; then
    echo "請先在 $env_file 設定 TV_PASSWORD 和 NGROK_DOMAIN（見本檔開頭說明）" >&2
    exit 1
fi
chmod 600 "$env_file"
"$ngrok" config check >/dev/null || { echo "請先執行 ngrok config add-authtoken <token>" >&2; exit 1; }

mkdir -p "$dest"
for f in "$project"/deploy/web/*.service; do
    sed "s|__PROJECT_DIR__|$project|g" "$f" > "$dest/$(basename "$f")"
done
systemctl --user daemon-reload
systemctl --user enable tv-web.service tv-tunnel.service
systemctl --user restart tv-web.service tv-tunnel.service
# 沒登入時也繼續跑、開機自動啟動；需要權限時請改用 sudo loginctl enable-linger "$USER"
loginctl enable-linger "$USER" || echo "無法開啟 linger：登出後服務會停止"

echo "網址：https://$(grep '^NGROK_DOMAIN=' "$env_file" | cut -d= -f2)"
