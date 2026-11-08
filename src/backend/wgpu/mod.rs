use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

use crate::error::{ForgeError, Result};

pub mod ops;
mod serial;

pub const OFFSET_ALIGN_BYTES: usize = 256;

const F32_B: &str = "var<storage, read> b: array<f32>;";

const Q4_B: &str = "var<storage, read> b: array<u32>;";

const F32_BVAL: &str = "    return b[i];";

const Q4_BVAL: &str = "    let base = (i >> 6u) * 10u;
    let q = (b[base + 2u + ((i & 63u) >> 3u)] >> ((i & 7u) * 4u)) & 15u;
    return f32(q) * bitcast<f32>(b[base]) + bitcast<f32>(b[base + 1u]);";

const SHADERS_CORE: &[(&str, &str)] = &[
    ("add", include_str!("../../../shaders/add.wgsl")),
    ("gelu", include_str!("../../../shaders/gelu.wgsl")),
    ("matmul", include_str!("../../../shaders/matmul.wgsl")),
    ("gemv", include_str!("../../../shaders/gemv.wgsl")),
    ("q4_gemv", include_str!("../../../shaders/q4_gemv.wgsl")),
    (
        "gemv_reduce",
        include_str!("../../../shaders/gemv_reduce.wgsl"),
    ),
    ("softmax", include_str!("../../../shaders/softmax.wgsl")),
    (
        "softmax_masked",
        include_str!("../../../shaders/softmax_masked.wgsl"),
    ),
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
    (
        "unmerge_heads",
        include_str!("../../../shaders/unmerge_heads.wgsl"),
    ),
];

#[cfg(feature = "train")]
const SHADERS_TRAIN: &[(&str, &str)] = &[
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
    ("sumsq", include_str!("../../../shaders/sumsq.wgsl")),
    ("scale", include_str!("../../../shaders/scale.wgsl")),
    ("adamw", include_str!("../../../shaders/adamw.wgsl")),
];

#[cfg(not(feature = "train"))]
const SHADERS_TRAIN: &[(&str, &str)] = &[];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub dispatches: usize,
    pub submits: usize,
    pub buffers_created: usize,
    pub bytes_allocated: usize,
}

impl Stats {
    pub fn since(self, earlier: Stats) -> Stats {
        Stats {
            dispatches: self.dispatches - earlier.dispatches,
            submits: self.submits - earlier.submits,
            buffers_created: self.buffers_created - earlier.buffers_created,
            bytes_allocated: self.bytes_allocated - earlier.bytes_allocated,
        }
    }
}

#[derive(Default)]
struct Counters {
    dispatches: AtomicUsize,
    submits: AtomicUsize,
    buffers_created: AtomicUsize,
    bytes_allocated: AtomicUsize,
}

#[derive(Default)]
struct Pool {
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    bytes: usize,
}

const MAX_POOL_BYTES: usize = 512 * 1024 * 1024;

const MAX_STAGING_BYTES: usize = 64 * 1024 * 1024;

const MAX_UNIFORM_ENTRIES: usize = 8192;

fn staging_class(size_bytes: usize) -> u64 {
    size_bytes.max(256).next_power_of_two() as u64
}

pub struct PooledBuffer {
    buf: Option<wgpu::Buffer>,
    ctx: Arc<WgpuContext>,
    size: u64,
    recycle: bool,
}

impl std::ops::Deref for PooledBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        self.buf.as_ref().expect("PooledBuffer used after drop")
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PooledBuffer({} B)", self.size)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        let Some(buf) = self.buf.take() else { return };
        if !self.recycle {
            return;
        }
        let mut pool = self.ctx.pool.lock().unwrap();
        if pool.bytes + self.size as usize > MAX_POOL_BYTES {
            return;
        }
        pool.bytes += self.size as usize;
        pool.free.entry(self.size).or_default().push(buf);
    }
}

pub struct DispatchScope {
    ctx: Arc<WgpuContext>,
}

impl Drop for DispatchScope {
    fn drop(&mut self) {
        let mut s = self.ctx.scope.lock().unwrap();
        s.depth -= 1;
        if s.depth == 0 {
            self.ctx.submit_locked(&mut s);
        }
    }
}

#[derive(Default)]
struct ScopeState {
    encoder: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    depth: usize,
}

type Kernel = (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>);

static SERIAL: serial::Gate = serial::Gate::new();

pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pipelines: Mutex<HashMap<&'static str, Kernel>>,
    counters: Counters,
    pool: Mutex<Pool>,
    staging: Mutex<Pool>,
    uniforms: Mutex<HashMap<Box<[u32]>, Arc<wgpu::Buffer>>>,
    scope: Mutex<ScopeState>,
    #[cfg(not(target_arch = "wasm32"))]
    _teardown: Option<serial::Guard<'static>>,
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuContext({})", self.adapter_info.name)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for WgpuContext {
    fn drop(&mut self) {
        self._teardown = Some(SERIAL.lock_blocking());
    }
}

impl WgpuContext {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Arc<Self>> {
        pollster::block_on(Self::new_async())
    }

    pub async fn new_async() -> Result<Arc<Self>> {
        let _serial = SERIAL.lock().await;
        Self::create().await
    }

    async fn create() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .map_err(|e| ForgeError::Wgpu(format!("no adapter: {e}")))?;
        let adapter_info = adapter.get_info();
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
            counters: Counters::default(),
            pool: Mutex::new(Pool::default()),
            staging: Mutex::new(Pool::default()),
            uniforms: Mutex::new(HashMap::new()),
            scope: Mutex::new(ScopeState::default()),
            #[cfg(not(target_arch = "wasm32"))]
            _teardown: None,
        }))
    }

    pub fn scope(self: &Arc<Self>) -> DispatchScope {
        let mut s = self.scope.lock().unwrap();
        s.depth += 1;
        DispatchScope { ctx: self.clone() }
    }

    pub fn flush(&self) {
        self.submit_locked(&mut self.scope.lock().unwrap());
    }

    fn submit_locked(&self, s: &mut ScopeState) {
        s.pass = None;
        if let Some(encoder) = s.encoder.take() {
            self.submit(encoder);
        }
    }

    fn submit_with_copies(&self, copies: &[(&wgpu::Buffer, u64, &wgpu::Buffer, u64)]) {
        let mut s = self.scope.lock().unwrap();
        s.pass = None;
        let mut encoder = s
            .encoder
            .take()
            .unwrap_or_else(|| self.device.create_command_encoder(&Default::default()));
        for (src, src_off, dst, size) in copies {
            encoder.copy_buffer_to_buffer(src, *src_off, dst, 0, *size);
        }
        self.submit(encoder);
    }

    pub fn create_pooled(self: &Arc<Self>, size_bytes: usize) -> PooledBuffer {
        let size = size_bytes.max(4).next_multiple_of(4) as u64;
        let recycled = {
            let mut pool = self.pool.lock().unwrap();
            let hit = pool.free.get_mut(&size).and_then(Vec::pop);
            if hit.is_some() {
                pool.bytes -= size as usize;
            }
            hit
        };
        let buf = recycled.unwrap_or_else(|| self.create_storage(size as usize));
        PooledBuffer {
            buf: Some(buf),
            ctx: self.clone(),
            size,
            recycle: true,
        }
    }

    pub fn stats(&self) -> Stats {
        Stats {
            dispatches: self.counters.dispatches.load(Ordering::Relaxed),
            submits: self.counters.submits.load(Ordering::Relaxed),
            buffers_created: self.counters.buffers_created.load(Ordering::Relaxed),
            bytes_allocated: self.counters.bytes_allocated.load(Ordering::Relaxed),
        }
    }

    fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.counters.submits.fetch_add(1, Ordering::Relaxed);
        self.queue.submit([encoder.finish()]);
    }

    fn pipeline(&self, name: &'static str) -> Kernel {
        let mut cache = self.pipelines.lock().unwrap();
        cache
            .entry(name)
            .or_insert_with(|| {
                let base = name.strip_suffix("_q4").unwrap_or(name);
                let src = SHADERS_CORE
                    .iter()
                    .chain(SHADERS_TRAIN)
                    .find(|(n, _)| *n == base)
                    .unwrap_or_else(|| panic!("unknown shader {name}"))
                    .1;
                let src = if base == name {
                    src.to_string()
                } else {
                    src.replace(F32_B, Q4_B).replace(F32_BVAL, Q4_BVAL)
                };
                let module = self
                    .device
                    .create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(name),
                        source: wgpu::ShaderSource::Wgsl(src.into()),
                    });
                let pipeline =
                    self.device
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(name),
                            layout: None,
                            module: &module,
                            entry_point: Some("main"),
                            compilation_options: Default::default(),
                            cache: None,
                        });
                let layout = Arc::new(pipeline.get_bind_group_layout(0));
                (Arc::new(pipeline), layout)
            })
            .clone()
    }

    pub fn create_storage(&self, size_bytes: usize) -> wgpu::Buffer {
        let size = size_bytes.max(4) as u64;
        self.counters
            .buffers_created
            .fetch_add(1, Ordering::Relaxed);
        self.counters
            .bytes_allocated
            .fetch_add(size as usize, Ordering::Relaxed);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub fn create_zeroed(self: &Arc<Self>, size_bytes: usize) -> PooledBuffer {
        PooledBuffer {
            buf: Some(self.create_storage(size_bytes.max(4))),
            ctx: self.clone(),
            size: size_bytes as u64,
            recycle: false,
        }
    }

    pub fn upload(self: &Arc<Self>, bytes: &[u8]) -> PooledBuffer {
        let buf = self.create_storage(bytes.len().max(4));
        self.queue.write_buffer(&buf, 0, bytes);
        PooledBuffer {
            buf: Some(buf),
            ctx: self.clone(),
            size: bytes.len() as u64,
            recycle: false,
        }
    }

    fn take_staging(&self, size_bytes: usize) -> wgpu::Buffer {
        let size = staging_class(size_bytes);
        let recycled = {
            let mut pool = self.staging.lock().unwrap();
            let hit = pool.free.get_mut(&size).and_then(Vec::pop);
            if hit.is_some() {
                pool.bytes -= size as usize;
            }
            hit
        };
        recycled.unwrap_or_else(|| {
            self.counters
                .buffers_created
                .fetch_add(1, Ordering::Relaxed);
            self.counters
                .bytes_allocated
                .fetch_add(size as usize, Ordering::Relaxed);
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
    }

    fn give_staging(&self, buf: wgpu::Buffer) {
        let size = buf.size();
        let mut pool = self.staging.lock().unwrap();
        if pool.bytes + size as usize > MAX_STAGING_BYTES {
            return;
        }
        pool.bytes += size as usize;
        pool.free.entry(size).or_default().push(buf);
    }

    fn stage_copy(
        &self,
        buf: &wgpu::Buffer,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> wgpu::Buffer {
        let staging = self.take_staging(size_bytes);
        self.submit_with_copies(&[(buf, offset_bytes as u64, &staging, size_bytes as u64)]);
        staging
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn readback(
        &self,
        buf: &wgpu::Buffer,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> Result<Vec<u8>> {
        let staging = self.stage_copy(buf, offset_bytes, size_bytes);
        let slice = staging.slice(..size_bytes as u64);
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
        self.give_staging(staging);
        Ok(out)
    }

    pub async fn readback_async(
        &self,
        buf: &wgpu::Buffer,
        offset_bytes: usize,
        size_bytes: usize,
    ) -> Result<Vec<u8>> {
        let staging = self.stage_copy(buf, offset_bytes, size_bytes);
        let slice = staging.slice(..size_bytes as u64);
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
        self.give_staging(staging);
        Ok(out)
    }

    pub async fn readback_many_async(
        &self,
        regions: &[(&wgpu::Buffer, usize, usize)],
    ) -> Result<Vec<Vec<u8>>> {
        if regions.is_empty() {
            return Ok(Vec::new());
        }
        let staging: Vec<wgpu::Buffer> = regions
            .iter()
            .map(|(_, _, size_bytes)| self.take_staging(*size_bytes))
            .collect();
        let copies: Vec<_> = regions
            .iter()
            .zip(&staging)
            .map(|((buf, off, size), s)| (*buf, *off as u64, s, *size as u64))
            .collect();
        self.submit_with_copies(&copies);

        let waits: Vec<_> = staging
            .iter()
            .zip(regions)
            .map(|(s, (_, _, size))| {
                let (tx, rx) = oneshot::channel();
                s.slice(..*size as u64)
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
        for ((rx, s), (_, _, size)) in waits.into_iter().zip(&staging).zip(regions) {
            rx.await
                .map_err(|e| ForgeError::Wgpu(format!("map_async: {e:?}")))?;
            out.push(s.slice(..*size as u64).get_mapped_range().to_vec());
            s.unmap();
        }
        for s in staging {
            self.give_staging(s);
        }
        Ok(out)
    }

    fn params_buffer(&self, name: &'static str, params: &[u32]) -> Arc<wgpu::Buffer> {
        let mut cache = self.uniforms.lock().unwrap();
        if let Some(buf) = cache.get(params) {
            return buf.clone();
        }
        if cache.len() >= MAX_UNIFORM_ENTRIES {
            cache.clear();
        }
        let buf = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: bytemuck::cast_slice(params),
                    usage: wgpu::BufferUsages::UNIFORM,
                }),
        );
        cache.insert(params.into(), buf.clone());
        buf
    }

    pub fn dispatch(
        &self,
        name: &'static str,
        params: &[u32],
        buffers: &[(&wgpu::Buffer, usize, usize)],
        workgroups: (u32, u32, u32),
    ) {
        self.counters.dispatches.fetch_add(1, Ordering::Relaxed);
        let (pipeline, layout) = self.pipeline(name);
        let params_buf = self.params_buffer(name, params);
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
            layout: &layout,
            entries: &entries,
        });
        let mut scope = self.scope.lock().unwrap();
        if scope.depth > 0 {
            let ScopeState { encoder, pass, .. } = &mut *scope;
            let encoder = encoder
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            let pass = match pass {
                Some(p) => p,
                None => pass.insert(
                    encoder
                        .begin_compute_pass(&Default::default())
                        .forget_lifetime(),
                ),
            };
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
            return;
        }

        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }
        self.submit(encoder);
    }
}

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

pub fn linear_grid(numel: usize) -> (u32, u32, u32) {
    let groups = numel.div_ceil(256).max(1) as u32;
    if groups <= 65535 {
        (groups, 1, 1)
    } else {
        let y = groups.div_ceil(65535);
        (65535, y, 1)
    }
}
