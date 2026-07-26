//! WebGPU backend: device context, buffer management, pipeline cache, and
//! kernel dispatch. Production backend of Forge; the CPU backend is the
//! numerical reference.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

use crate::error::{ForgeError, Result};

pub mod ops;

/// Storage-buffer offsets must respect this alignment when creating views.
pub const OFFSET_ALIGN_BYTES: usize = 256;

const SHADERS: &[(&str, &str)] = &[
    ("add", include_str!("../../../shaders/add.wgsl")),
    ("gelu", include_str!("../../../shaders/gelu.wgsl")),
    ("matmul", include_str!("../../../shaders/matmul.wgsl")),
    ("softmax", include_str!("../../../shaders/softmax.wgsl")),
    ("layernorm", include_str!("../../../shaders/layernorm.wgsl")),
    ("embedding", include_str!("../../../shaders/embedding.wgsl")),
    (
        "split_heads",
        include_str!("../../../shaders/split_heads.wgsl"),
    ),
    (
        "merge_heads",
        include_str!("../../../shaders/merge_heads.wgsl"),
    ),
    ("kv_append", include_str!("../../../shaders/kv_append.wgsl")),
    ("gelu_bwd", include_str!("../../../shaders/gelu_bwd.wgsl")),
    (
        "softmax_bwd",
        include_str!("../../../shaders/softmax_bwd.wgsl"),
    ),
    (
        "layernorm_bwd_dx",
        include_str!("../../../shaders/layernorm_bwd_dx.wgsl"),
    ),
    (
        "layernorm_bwd_dp",
        include_str!("../../../shaders/layernorm_bwd_dp.wgsl"),
    ),
    ("sum_rows", include_str!("../../../shaders/sum_rows.wgsl")),
    (
        "scatter_add",
        include_str!("../../../shaders/scatter_add.wgsl"),
    ),
    (
        "gather_nll",
        include_str!("../../../shaders/gather_nll.wgsl"),
    ),
    ("ce_bwd", include_str!("../../../shaders/ce_bwd.wgsl")),
    ("dropout", include_str!("../../../shaders/dropout.wgsl")),
    (
        "unsplit_heads",
        include_str!("../../../shaders/unsplit_heads.wgsl"),
    ),
    (
        "unmerge_heads",
        include_str!("../../../shaders/unmerge_heads.wgsl"),
    ),
    ("sumsq", include_str!("../../../shaders/sumsq.wgsl")),
    ("scale", include_str!("../../../shaders/scale.wgsl")),
    ("adamw", include_str!("../../../shaders/adamw.wgsl")),
];

/// Owns the wgpu device/queue and a cache of compiled compute pipelines.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pipelines: Mutex<HashMap<&'static str, Arc<wgpu::ComputePipeline>>>,
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuContext({})", self.adapter_info.name)
    }
}

impl WgpuContext {
    /// Sync device creation — native only. On wasm use [`Self::new_async`].
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Arc<Self>> {
        pollster::block_on(Self::new_async())
    }

    /// Async device creation (works on native and wasm32; roadmap v4,
    /// pitfall 14: the async form is primary, the sync API is the facade).
    pub async fn new_async() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| ForgeError::Wgpu(format!("no adapter: {e}")))?;
        let adapter_info = adapter.get_info();
        // GPT-2's token embedding (~147 MiB) exceeds the 128 MiB default
        // max_storage_buffer_binding_size, so request the adapter's limits.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("forge"),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .map_err(|e| ForgeError::Wgpu(format!("request_device: {e}")))?;
        Ok(Arc::new(WgpuContext {
            device,
            queue,
            adapter_info,
            pipelines: Mutex::new(HashMap::new()),
        }))
    }

    fn pipeline(&self, name: &'static str) -> Arc<wgpu::ComputePipeline> {
        let mut cache = self.pipelines.lock().unwrap();
        cache
            .entry(name)
            .or_insert_with(|| {
                let src = SHADERS
                    .iter()
                    .find(|(n, _)| *n == name)
                    .unwrap_or_else(|| panic!("unknown shader {name}"))
                    .1;
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                Arc::new(
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(name),
                            layout: None,
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        }),
                )
            })
            .clone()
    }

    pub fn create_storage(&self, size_bytes: usize) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes.max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn upload(&self, bytes: &[u8]) -> wgpu::Buffer {
        let buf = self.create_storage(bytes.len());
        self.queue.write_buffer(&buf, 0, bytes);
        buf
    }

    fn stage_copy(
        &self,
        buf: &wgpu::Buffer,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> wgpu::Buffer {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buf, offset_bytes as u64, &staging, 0, size_bytes as u64);
        self.queue.submit([encoder.finish()]);
        staging
    }

    /// Read `size_bytes` starting at `offset_bytes` back to the host.
    /// Sync facade — native only (`device.poll(Wait)` cannot exist on wasm).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn readback(
        &self,
        buf: &wgpu::Buffer,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> Result<Vec<u8>> {
        let staging = self.stage_copy(buf, offset_bytes, size_bytes);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| ForgeError::Wgpu(format!("poll: {e:?}")))?;
        rx.recv()
            .map_err(|_| ForgeError::Wgpu("map_async callback dropped".into()))?
            .map_err(|e| ForgeError::Wgpu(format!("map_async: {e:?}")))?;
        let out = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Async readback — the primary form; on wasm the browser event loop
    /// drives the mapping.
    pub async fn readback_async(
        &self,
        buf: &wgpu::Buffer,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> Result<Vec<u8>> {
        let staging = self.stage_copy(buf, offset_bytes, size_bytes);
        let slice = staging.slice(..);
        let (tx, rx) = oneshot::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r));
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| ForgeError::Wgpu(format!("poll: {e:?}")))?;
        #[cfg(target_arch = "wasm32")]
        let _ = self.device.poll(wgpu::PollType::Poll);
        rx.await
            .map_err(|e| ForgeError::Wgpu(format!("map_async: {e:?}")))?;
        let out = slice.get_mapped_range().to_vec();
        staging.unmap();
        Ok(out)
    }

    /// Read several regions back in one submit and one fence wait.
    ///
    /// [`WgpuContext::readback_async`] costs a submit and a wait *each*, which
    /// dominates when a single logical step wants several small tensors: the
    /// attention probe reads `n_layer + 1` per generated token, and one at a
    /// time that cost more than the decode itself. Staged into one encoder
    /// they cost one round trip regardless of how many there are.
    ///
    /// Regions are returned in the order given.
    pub async fn readback_many_async(
        &self,
        regions: &[(&wgpu::Buffer, usize, usize)],
    ) -> Result<Vec<Vec<u8>>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let staging: Vec<wgpu::Buffer> = regions
            .iter()
            .map(|(buf, offset_bytes, size_bytes)| {
                let s = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: *size_bytes as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                encoder.copy_buffer_to_buffer(buf, *offset_bytes as u64, &s, 0, *size_bytes as u64);
                s
            })
            .collect();
        self.queue.submit([encoder.finish()]);

        // Every map request is issued before anything is awaited, so one poll
        // services all of them.
        let waits: Vec<_> = staging
            .iter()
            .map(|s| {
                let (tx, rx) = oneshot::channel();
                s.slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r));
                rx
            })
            .collect();
        #[cfg(not(target_arch = "wasm32"))]
        self.device
            .poll(wgpu::PollType::Wait)
            .map_err(|e| ForgeError::Wgpu(format!("poll: {e:?}")))?;
        #[cfg(target_arch = "wasm32")]
        let _ = self.device.poll(wgpu::PollType::Poll);

        let mut out = Vec::with_capacity(regions.len());
        for (rx, s) in waits.into_iter().zip(&staging) {
            rx.await
                .map_err(|e| ForgeError::Wgpu(format!("map_async: {e:?}")))?;
            out.push(s.slice(..).get_mapped_range().to_vec());
            s.unmap();
        }
        Ok(out)
    }

    /// Dispatch `name` with binding 0 = `params` (uniform, raw words) and
    /// bindings 1.. = `buffers` (storage). Each buffer entry is
    /// (buffer, offset_bytes, size_bytes).
    pub fn dispatch(
        &self,
        name: &'static str,
        params: &[u32],
        buffers: &[(&wgpu::Buffer, usize, usize)],
        workgroups: (u32, u32, u32),
    ) {
        let pipeline = self.pipeline(name);
        let params_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(name),
                contents: bytemuck::cast_slice(params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: params_buf.as_entire_binding(),
        }];
        for (i, (buf, off, size)) in buffers.iter().enumerate() {
            debug_assert!(off % OFFSET_ALIGN_BYTES == 0, "storage offset misaligned");
            entries.push(wgpu::BindGroupEntry {
                binding: (i + 1) as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: *off as u64,
                    size: Some(std::num::NonZeroU64::new((*size).max(4) as u64).unwrap()),
                }),
            });
        }
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(name),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.queue.submit([encoder.finish()]);
    }
}

/// Minimal single-value channel whose receiver is a `Future` — lets
/// `map_async` results be awaited without extra dependencies (wasm has no
/// blocking receive).
mod oneshot {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};
    use std::task::{Context, Poll, Waker};

    struct State<T> {
        value: Option<T>,
        waker: Option<Waker>,
    }

    pub struct Sender<T>(Arc<Mutex<State<T>>>);
    pub struct Receiver<T>(Arc<Mutex<State<T>>>);

    pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
        let shared = Arc::new(Mutex::new(State {
            value: None,
            waker: None,
        }));
        (Sender(shared.clone()), Receiver(shared))
    }

    impl<T> Sender<T> {
        pub fn send(self, value: T) {
            let mut s = self.0.lock().unwrap();
            s.value = Some(value);
            if let Some(w) = s.waker.take() {
                w.wake();
            }
        }
    }

    impl<T> Future for Receiver<T> {
        type Output = T;
        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
            let mut s = self.0.lock().unwrap();
            match s.value.take() {
                Some(v) => Poll::Ready(v),
                None => {
                    s.waker = Some(cx.waker().clone());
                    Poll::Pending
                }
            }
        }
    }
}

/// Split a linear element count into a (x, y, 1) workgroup grid of 256-thread
/// groups, respecting the 65535 per-dimension dispatch limit.
pub fn linear_grid(numel: usize) -> (u32, u32, u32) {
    let groups = numel.div_ceil(256).max(1) as u32;
    if groups <= 65535 {
        (groups, 1, 1)
    } else {
        let y = groups.div_ceil(65535);
        (65535, y, 1)
    }
}
