#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get upgrade -y

sudo apt-get install -y --no-install-recommends \
	libvulkan1 \
	vulkan-tools \
	mesa-vulkan-drivers \
	libegl1 \
	libgl1 \
	chromium \
	pkg-config

bash scripts/devcontainer/setup_nvidia_vulkan.sh || true

echo
echo "== Vulkan adapters visible to wgpu =="
vulkaninfo --summary 2>/dev/null | sed -n '/Devices:/,/^$/p' || echo "vulkaninfo failed — WebGPU will not come up"

sudo apt-get install -y --no-install-recommends tmux
sudo rm -rf /var/lib/apt/lists/*

TMUX_CONF=~/.tmux.conf
if ! grep -q '^# >>> devcontainer tmux (claude) >>>' "$TMUX_CONF" 2>/dev/null; then
	cat >>"$TMUX_CONF" <<'EOF'
# >>> devcontainer tmux (claude) >>>
set -g mouse on
set -g history-limit 200000
set -g default-terminal "tmux-256color"
set -ag terminal-overrides ",xterm-256color:RGB,*256col*:RGB"
set -s escape-time 0
set -g focus-events on
set -g set-clipboard on
set -g allow-passthrough on
set -g base-index 1
setw -g pane-base-index 1
setw -g mode-keys vi
setw -g aggressive-resize on
set -g status-interval 5
# <<< devcontainer tmux (claude) <<<
EOF
fi

tmux has-session -t harnesses 2>/dev/null || tmux new-session -d -s harnesses -c "$PWD"
tmux source-file "$TMUX_CONF"
