use std::sync::Arc;

use crate::backend::wgpu::WgpuContext;
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

    /// Async device creation — the primary form (roadmap v4, pitfall 14);
    /// works on native and wasm32/browser.
    pub async fn wgpu_async() -> Result<Device> {
        Ok(Device::Wgpu(WgpuContext::new_async().await?))
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
