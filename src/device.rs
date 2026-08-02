use std::sync::Arc;

use crate::backend::wgpu::{DispatchScope, WgpuContext};
use crate::error::Result;

/// Where a tensor lives and where its ops execute.
#[derive(Clone, Debug)]
pub enum Device {
    /// Reference implementation used for testing and verification.
    Cpu,
    /// Production backend: WebGPU via wgpu (Vulkan/Metal/D3D12/Browser).
    Wgpu(Arc<WgpuContext>),
}

impl Device {
    /// Create the default WebGPU device (highest-performance adapter).
    /// Sync facade — native only; on wasm use [`Device::wgpu_async`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wgpu() -> Result<Device> {
        Ok(Device::Wgpu(WgpuContext::new()?))
    }

    /// Async device creation — the primary form; works on native and wasm32.
    ///
    /// Safe to call from several threads or tasks at once, and safe to mix
    /// with [`Device::wgpu`]: creation is serialized process-wide, because
    /// drivers wedge when two devices are built concurrently. The returned
    /// future is `Send`, so it can be spawned on a multi-threaded runtime.
    pub async fn wgpu_async() -> Result<Device> {
        Ok(Device::Wgpu(WgpuContext::new_async().await?))
    }

    /// Batch every dispatch made while the returned guard is alive into one
    /// command buffer and one submit. `None` on the CPU backend, which has no
    /// queue to batch.
    ///
    /// Hold it for the span of a logical step — a decode step issues ~100
    /// kernels, and a submit each leaves the GPU waiting on the CPU between
    /// every one of them. Readbacks flush automatically, so a scope can wrap
    /// code that reads results without special care.
    #[must_use = "the batching ends when the guard drops"]
    pub fn dispatch_scope(&self) -> Option<DispatchScope> {
        match self {
            Device::Cpu => None,
            Device::Wgpu(ctx) => Some(ctx.scope()),
        }
    }

    /// Human-readable adapter description.
    pub fn describe(&self) -> String {
        match self {
            Device::Cpu => "CPU (reference)".to_string(),
            Device::Wgpu(ctx) => format!(
                "WebGPU: {} ({:?})",
                ctx.adapter_info.name, ctx.adapter_info.backend
            ),
        }
    }

    /// Largest single buffer binding this device supports, in bytes. Large
    /// weights (GPT-2's wte) are row-chunked to stay under this.
    pub fn max_binding_bytes(&self) -> usize {
        match self {
            Device::Cpu => usize::MAX,
            Device::Wgpu(ctx) => ctx.device.limits().max_storage_buffer_binding_size as usize,
        }
    }

    /// Two devices are compatible when tensors on them can be combined.
    pub fn same_as(&self, other: &Device) -> bool {
        match (self, other) {
            (Device::Cpu, Device::Cpu) => true,
            (Device::Wgpu(a), Device::Wgpu(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}
