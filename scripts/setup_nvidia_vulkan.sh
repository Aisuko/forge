#!/usr/bin/env bash
# Make the NVIDIA GPU visible to Vulkan (and therefore wgpu) inside a
# container where the toolkit mounted the driver libs but not the ICD
# manifest, the GLVND vendor config, or usable DRM device node perms.
#
# Diagnosed requirements (driver 580.x, RTX A5000), in the order that
# actually gated initialization when this was debugged:
#   1. /dev/dri/{card*,renderD*} nodes for the NVIDIA GPU, openable under
#      the container's device cgroup (needs --device-cgroup-rule='c 226:*
#      rmw' and 'c 195:* rmw', both already in devcontainer.json) AND with
#      DAC permissions the container user can actually use. --device=/dev/dri
#      bind-mounts the *host's* nodes verbatim (root:video / root:<host-render-gid>,
#      mode 660) — the container user is in neither group, so every open()
#      fails EACCES even though the cgroup rule allows it. Fixed by chmod'ing
#      the nodes to 666 (container-local device nodes; does not touch the host).
#   2. an ICD manifest (/usr/share/vulkan/icd.d/nvidia_icd.json) pointing at
#      libGLX_nvidia.so.0, so the Vulkan loader dlopen's the NVIDIA driver
#      instead of only finding Mesa's llvmpipe/lvp ICDs.
#   3. a GLVND EGL vendor manifest (/usr/share/glvnd/egl_vendor.d/10_nvidia.json)
#      pointing at libEGL_nvidia.so.0. This one is easy to miss: only
#      /usr/share/glvnd/egl_vendor.d/50_mesa.json existed. Without it, the
#      combined libGLX_nvidia.so.0's internal graphics-stack bring-up (shared
#      by its GLX/EGL/Vulkan entry points) never completes, so even though the
#      Vulkan loader successfully loads and calls into libGLX_nvidia.so.0, its
#      vk_icdNegotiateLoaderICDInterfaceVersion call returns
#      VK_ERROR_INITIALIZATION_FAILED (-3) with vkCreateInstance staying NULL —
#      *before* the driver ever attempts to open /dev/nvidiactl or the DRM
#      nodes, and identically whether run as the container user or as root
#      (i.e. it is not a permissions/cgroup symptom, even though the error
#      text "Found no drivers!" looks like one). Confirmed via
#      vk_icdNegotiateLoaderICDInterfaceVersion returning 0 and vkCreateInstance
#      resolving to a real pointer immediately after adding this manifest.
#
# Safe to re-run; skips/repeats anything idempotently.
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

# Create (or fix perms on) DRM nodes for every NVIDIA (vendor 0x10de) DRM
# device in sysfs. /dev/dri is normally bind-mounted by --device=/dev/dri
# with the host's restrictive perms, so existing nodes need chmod, not just
# nodes we mknod ourselves.
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

# Verify the ICD actually initializes: replicate the loader's own
# negotiate-then-getInstanceProcAddr protocol rather than skipping straight
# to vk_icdGetInstanceProcAddr, since on driver 580.x that call alone can
# return a valid-looking pointer table lookup while negotiate still fails.
python3 - <<'EOF' || echo "warning: NVIDIA ICD still not initializing (see comments in this script)"
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
