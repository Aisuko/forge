#!/usr/bin/env bash
# Make the NVIDIA GPU visible to Vulkan, and so to wgpu, inside a container
# where the toolkit mounted the driver libs but not the ICD manifest, the GLVND
# vendor config, or usable DRM node permissions.
#
# Three separate causes, each of which alone produces "Found no drivers!".
# CONTRIBUTING.md, "Vulkan in the devcontainer", records how each was diagnosed.
#
# Safe to re-run.
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

EGL_VENDOR_DIR=/usr/share/glvnd/egl_vendor.d
if [ ! -f "$EGL_VENDOR_DIR/10_nvidia.json" ] && [ -e /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 ]; then
    sudo mkdir -p "$EGL_VENDOR_DIR"
    sudo tee "$EGL_VENDOR_DIR/10_nvidia.json" >/dev/null <<'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
EOF
    echo "installed $EGL_VENDOR_DIR/10_nvidia.json"
fi

if [ ! -e /dev/nvidia-modeset ]; then
    sudo mknod -m 666 /dev/nvidia-modeset c 195 254 && echo "created /dev/nvidia-modeset"
fi

# Create, or fix permissions on, the DRM nodes of every NVIDIA (0x10de) device.
# --device=/dev/dri bind-mounts the host's nodes with the host's groups, so an
# existing node needs chmod, not just the ones we mknod.
sudo mkdir -p /dev/dri
for dev in /sys/class/drm/card* /sys/class/drm/renderD*; do
    [ -e "$dev/device/vendor" ] || continue
    [ "$(cat "$dev/device/vendor")" = "0x10de" ] || continue
    name=$(basename "$dev")
    if [ ! -e "/dev/dri/$name" ]; then
        IFS=: read -r major minor < "$dev/dev"
        sudo mknod -m 666 "/dev/dri/$name" c "$major" "$minor" && echo "created /dev/dri/$name"
    else
        sudo chmod 666 "/dev/dri/$name"
    fi
done

# Verify the ICD initializes, by replicating the loader's own negotiate-then-
# getInstanceProcAddr protocol. On driver 580.x, vk_icdGetInstanceProcAddr
# alone returns a valid-looking pointer while negotiate is still failing.
python3 - <<'EOF' || echo "warning: NVIDIA ICD still not initializing — see CONTRIBUTING.md"
import ctypes, sys
lib = ctypes.CDLL("libGLX_nvidia.so.0")
negotiate = lib.vk_icdNegotiateLoaderICDInterfaceVersion
negotiate.restype = ctypes.c_int32
negotiate.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
ver = ctypes.c_uint32(5)
result = negotiate(ctypes.byref(ver))
if result != 0:
    print(f"vk_icdNegotiateLoaderICDInterfaceVersion returned {result} (expected 0/VK_SUCCESS)")
    sys.exit(1)
fn = lib.vk_icdGetInstanceProcAddr
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
sys.exit(0 if fn(None, b"vkCreateInstance") else 1)
EOF
