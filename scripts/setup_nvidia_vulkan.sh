#!/usr/bin/env bash
# Make the NVIDIA GPU visible to Vulkan (and therefore wgpu) inside a
# container where the toolkit mounted the driver libs but not the ICD
# manifest or the DRM device nodes.
#
# Diagnosed requirements (driver 580.x, RTX A5000):
#   1. an ICD manifest pointing at libGLX_nvidia.so.0
#   2. GLVND (libEGL.so.1 / libGL.so.1) — the ICD dlopen's it during init
#   3. /dev/nvidia-modeset and the /dev/dri/{card*,renderD*} nodes for the
#      NVIDIA GPU, openable under the container's device cgroup
#      (needs --device-cgroup-rule='c 226:* rmw' and 'c 195:* rmw')
#
# Safe to re-run; skips anything already present.
set -euo pipefail

ICD_DIR=/usr/share/vulkan/icd.d
if [ ! -f "$ICD_DIR/nvidia_icd.json" ] && [ -e /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 ]; then
    sudo mkdir -p "$ICD_DIR"
    sudo tee "$ICD_DIR/nvidia_icd.json" >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libGLX_nvidia.so.0",
        "api_version" : "1.3.277"
    }
}
EOF
    echo "installed $ICD_DIR/nvidia_icd.json"
fi

if [ ! -e /dev/nvidia-modeset ]; then
    sudo mknod -m 666 /dev/nvidia-modeset c 195 254 && echo "created /dev/nvidia-modeset"
fi

# Create DRM nodes for every NVIDIA (vendor 0x10de) DRM device in sysfs.
sudo mkdir -p /dev/dri
for dev in /sys/class/drm/card* /sys/class/drm/renderD*; do
    [ -e "$dev/device/vendor" ] || continue
    [ "$(cat "$dev/device/vendor")" = "0x10de" ] || continue
    name=$(basename "$dev")
    if [ ! -e "/dev/dri/$name" ]; then
        IFS=: read -r major minor < "$dev/dev"
        sudo mknod -m 666 "/dev/dri/$name" c "$major" "$minor" && echo "created /dev/dri/$name"
    fi
done

# Verify the ICD initializes (returns a non-null vkCreateInstance).
python3 - <<'EOF' || echo "warning: NVIDIA ICD still not initializing (check device cgroup rules)"
import ctypes, sys
lib = ctypes.CDLL("libGLX_nvidia.so.0")
fn = lib.vk_icdGetInstanceProcAddr
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
sys.exit(0 if fn(None, b"vkCreateInstance") else 1)
EOF
