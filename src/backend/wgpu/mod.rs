//! WebGPU backend: device context, buffer management, pipeline cache, and
//! kernel dispatch. Production backend of Forge; the CPU backend is the
//! numerical reference.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use wgpu::util::DeviceExt;

use crate::error::{ForgeError, Result};

pub mod ops;
mod serial;

/// Storage-buffer offsets must respect this alignment when creating views.
pub const OFFSET_ALIGN_BYTES: usize = 256;

/// The forward kernels — every build has these.
const SHADERS_CORE: &[(&str, &str)] = &[
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
];

/// The backward and optimizer kernels. Split out of one table so `train` can
/// gate them: `include_str!` bakes each shader's source into the binary, and
/// a `const` slice referenced by `pipeline()` is live whether or not anything
/// dispatches it. Splitting the table is what actually keeps ~14 KB of WGSL
/// out of a default build — gating the Rust alone does not, because the
/// linker had already eliminated that.
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
    (
        "unmerge_heads",
        include_str!("../../../shaders/unmerge_heads.wgsl"),
    ),
    ("sumsq", include_str!("../../../shaders/sumsq.wgsl")),
    ("scale", include_str!("../../../shaders/scale.wgsl")),
    ("adamw", include_str!("../../../shaders/adamw.wgsl")),
];

#[cfg(not(feature = "train"))]
const SHADERS_TRAIN: &[(&str, &str)] = &[];

/// Cumulative device counters, read with [`WgpuContext::stats`].
///
/// Always compiled rather than sitting behind a `bench` feature: they are four
/// relaxed atomic increments on a path that is about to talk to a GPU, so they
/// cost nothing measurable, and a counter nobody can read in a normal build is
/// a counter that never catches the regression it exists for.
///
/// `dispatches` counts kernel invocations; `submits` counts
/// `queue.submit` calls. Their ratio is the whole point — one submit per
/// dispatch means the GPU is idling between kernels waiting on the CPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    pub dispatches: usize,
    pub submits: usize,
    pub buffers_created: usize,
    pub bytes_allocated: usize,
}

impl Stats {
    /// Counters accumulated between two snapshots.
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

/// Buffers waiting to be handed out again, keyed by exact size in bytes.
///
/// Keyed exactly rather than by power-of-two size class on purpose: a
/// transformer decode step allocates the *same* handful of shapes every token,
/// so exact keys hit essentially always, while rounding a 41 MiB weight up to a
/// 64 MiB class would waste more than the pool saves.
#[derive(Default)]
struct Pool {
    free: HashMap<u64, Vec<wgpu::Buffer>>,
    bytes: usize,
}

/// How much the free list may hold before returned buffers are dropped instead
/// of kept. Bounds a long session; large one-off buffers (weights) are the ones
/// this evicts, which is the right choice — they are never asked for twice.
const MAX_POOL_BYTES: usize = 512 * 1024 * 1024;

/// A storage buffer that returns to its context's free list when dropped
/// rather than being destroyed.
///
/// Recycling GPU memory sounds alarming and is safe here for two reasons. A
/// buffer only reaches the free list when its last host-side handle drops, and
/// the pool can then only hand it to a *later* dispatch — and later commands on
/// one queue execute after earlier ones, so a new write can never overtake a
/// pending read. And recycled buffers are not zeroed, which is fine because
/// every kernel writes the whole of its output before anything reads it;
/// `tests/op_parity.rs` would show stale tail elements immediately if that ever
/// stopped being true.
///
/// **`recycle` is the exception those reasons do not cover, and it is not
/// theoretical — it corrupted `wte` gradients before it was understood.**
/// `queue.write_buffer` does not run in command order: its data is applied at
/// the *start* of the next submit, ahead of every command buffer in it. So a
/// buffer that is dropped, recycled, and then filled by `upload` while earlier
/// dispatches reading its previous contents are still recorded and unsubmitted
/// would have those dispatches read the new bytes. Buffers written by
/// `write_buffer` therefore never join the pool: [`WgpuContext::upload`] sets
/// `recycle: false`. They are weights and per-step id tensors, so the pool
/// loses almost nothing.
pub struct PooledBuffer {
    buf: Option<wgpu::Buffer>,
    ctx: Arc<WgpuContext>,
    size: u64,
    recycle: bool,
}

impl std::ops::Deref for PooledBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        // Only `Drop` ever takes it, and `Drop` cannot be observed.
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
            return; // over the cap: let wgpu destroy it
        }
        pool.bytes += self.size as usize;
        pool.free.entry(self.size).or_default().push(buf);
    }
}

/// An open recording scope: every dispatch made while one is alive goes into a
/// single command encoder and a single `queue.submit`.
///
/// The reason this type exists: a GPT-2 block issues ~16 kernels, and one
/// submit each leaves the GPU idle between them waiting on the CPU. See
/// [`WgpuContext::scope`].
pub struct DispatchScope {
    ctx: Arc<WgpuContext>,
}

impl Drop for DispatchScope {
    fn drop(&mut self) {
        let outermost = {
            let mut s = self.ctx.scope.lock().unwrap();
            s.depth -= 1;
            s.depth == 0
        };
        if outermost {
            self.ctx.flush();
        }
    }
}

#[derive(Default)]
struct ScopeState {
    encoder: Option<wgpu::CommandEncoder>,
    /// One compute pass for the whole scope, rather than one per dispatch.
    ///
    /// A pass boundary is a pipeline barrier and, on some drivers, a cache
    /// flush; ~100 of them per decoded token is real time. `forget_lifetime`
    /// is what lets the pass and the encoder it borrows live in the same
    /// struct — it is a safe wgpu API that turns the borrow into a runtime
    /// check, and the check is upheld here by ending the pass (dropping it)
    /// before the encoder is touched for anything else.
    ///
    /// Ordering within a pass is still guaranteed: dispatches execute in the
    /// order recorded, and wgpu inserts the barriers a read-after-write
    /// between them needs.
    pass: Option<wgpu::ComputePass<'static>>,
    depth: usize,
}

/// A compiled kernel: the pipeline, and the bind-group layout `dispatch` needs
/// for every bind group it builds.
///
/// The layout is cached rather than fetched because
/// `ComputePipeline::get_bind_group_layout` constructs a new one on every call,
/// and `dispatch` would call it ~100 times per decoded token.
type Kernel = (Arc<wgpu::ComputePipeline>, Arc<wgpu::BindGroupLayout>);

/// Owns the wgpu device/queue and a cache of compiled compute pipelines.
pub struct WgpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pipelines: Mutex<HashMap<&'static str, Kernel>>,
    counters: Counters,
    pool: Mutex<Pool>,
    scope: Mutex<ScopeState>,
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuContext({})", self.adapter_info.name)
    }
}

impl WgpuContext {
    /// Sync device creation — native only. On wasm use [`Self::new_async`].
    ///
    /// A facade over [`Self::new_async`], holding no lock of its own: one gate
    /// covers both, so a thread here and a thread there cannot race.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Arc<Self>> {
        pollster::block_on(Self::new_async())
    }

    /// Async device creation, on native and wasm32. This is the primary form;
    /// the sync API is a facade over it.
    ///
    /// Serialized against every other creator in the process: building two
    /// adapters and devices at once wedges in the driver, with no dispatch or
    /// compute involved, and that is what hung `tests/pool.rs` for minutes.
    /// The gate is async (`serial::Gate`) because a `std::sync` guard held
    /// across the `.await` below would make this future `!Send` and break
    /// dependents that spawn it.
    pub async fn new_async() -> Result<Arc<Self>> {
        static CREATE: serial::Gate = serial::Gate::new();
        let _serial = CREATE.lock().await;
        Self::create().await
    }

    /// Creation itself, without the gate. Every caller comes through
    /// `new_async`, which is where the serialization lives.
    async fn create() -> Result<Arc<Self>> {
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
            counters: Counters::default(),
            pool: Mutex::new(Pool::default()),
            scope: Mutex::new(ScopeState::default()),
        }))
    }

    /// Open a recording scope: every [`WgpuContext::dispatch`] made while the
    /// returned guard is alive is recorded into one command encoder and
    /// submitted once, when the outermost guard drops.
    ///
    /// This is the difference between ~100 submits per decoded token and one.
    /// Scopes nest — only the outermost submits — so a caller can wrap a whole
    /// decode step without knowing whether its callees also wrap their bodies.
    ///
    /// Reads are safe across a scope: any readback flushes first (see
    /// [`WgpuContext::flush`]), and compute passes within one encoder execute
    /// in the order they were recorded.
    pub fn scope(self: &Arc<Self>) -> DispatchScope {
        let mut s = self.scope.lock().unwrap();
        s.depth += 1;
        DispatchScope { ctx: self.clone() }
    }

    /// Submit whatever a scope has recorded so far, without closing it.
    ///
    /// Every readback path calls this first. Making it the readback's job
    /// rather than the caller's is deliberate: "remember to flush before you
    /// read" is exactly the kind of rule that silently rots into a
    /// stale-bytes bug, and the cost is one uncontended mutex on a path that
    /// is about to wait on a fence anyway.
    pub fn flush(&self) {
        let encoder = {
            let mut s = self.scope.lock().unwrap();
            s.pass = None; // ends the pass; the encoder is locked while one lives
            s.encoder.take()
        };
        if let Some(encoder) = encoder {
            self.submit(encoder);
        }
    }

    /// Submit everything a scope has recorded *plus* these buffer-to-buffer
    /// copies, in one command buffer.
    ///
    /// Readback is why this exists. Flushing the compute and then submitting
    /// the staging copy separately costs two submits and two trips through the
    /// driver for what is logically one step; folding the copy into the same
    /// command buffer makes a decode step one submit and one fence wait.
    ///
    /// Each copy is `(src, src_offset_bytes, dst, size_bytes)`, written to
    /// offset 0 of `dst`.
    fn submit_with_copies(&self, copies: &[(&wgpu::Buffer, u64, &wgpu::Buffer, u64)]) {
        let mut encoder = {
            let mut s = self.scope.lock().unwrap();
            s.pass = None;
            s.encoder.take()
        }
        .unwrap_or_else(|| self.device.create_command_encoder(&Default::default()));
        for (src, src_off, dst, size) in copies {
            encoder.copy_buffer_to_buffer(src, *src_off, dst, 0, *size);
        }
        self.submit(encoder);
    }

    /// A storage buffer, from the free list when one of this exact size is
    /// waiting. See [`PooledBuffer`].
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

    /// Snapshot of this device's cumulative counters. Subtract two snapshots
    /// with [`Stats::since`] to measure one region.
    pub fn stats(&self) -> Stats {
        Stats {
            dispatches: self.counters.dispatches.load(Ordering::Relaxed),
            submits: self.counters.submits.load(Ordering::Relaxed),
            buffers_created: self.counters.buffers_created.load(Ordering::Relaxed),
            bytes_allocated: self.counters.bytes_allocated.load(Ordering::Relaxed),
        }
    }

    /// The one place a command buffer reaches the queue. Everything submits
    /// through here so `Stats::submits` cannot drift from reality.
    fn submit(&self, encoder: wgpu::CommandEncoder) {
        self.counters.submits.fetch_add(1, Ordering::Relaxed);
        self.queue.submit([encoder.finish()]);
    }

    /// The compiled pipeline for `name`, and its bind-group layout.
    ///
    /// The layout is cached alongside the pipeline because
    /// `ComputePipeline::get_bind_group_layout` is not an accessor — it
    /// constructs a new `BindGroupLayout` on every call, and `dispatch` calls
    /// it once per kernel. At ~100 kernels per decoded token that is pure
    /// CPU-side overhead on the critical path.
    fn pipeline(&self, name: &'static str) -> Kernel {
        let mut cache = self.pipelines.lock().unwrap();
        cache
            .entry(name)
            .or_insert_with(|| {
                let src = SHADERS_CORE
                    .iter()
                    .chain(SHADERS_TRAIN)
                    .find(|(n, _)| *n == name)
                    .unwrap_or_else(|| panic!("unknown shader {name}"))
                    .1;
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

    /// A storage buffer that never came from the pool, and never returns to
    /// it — so it is guaranteed zero-filled, which WebGPU promises for a newly
    /// created buffer and a recycled one obviously cannot.
    ///
    /// For the kernels that write only part of their output and rely on the
    /// rest being zero. There is exactly one today (`unsplit_heads`, which
    /// fills one third of a `[t, 3c]` gradient), and it is worth a dedicated
    /// allocator rather than zeroing every recycled buffer: zeroing through
    /// `queue.write_buffer` would reintroduce the ordering hazard described on
    /// [`PooledBuffer`], and clearing through the encoder would force a pass
    /// boundary on a path that has no other reason for one.
    pub fn create_zeroed(self: &Arc<Self>, size_bytes: usize) -> PooledBuffer {
        PooledBuffer {
            buf: Some(self.create_storage(size_bytes.max(4))),
            ctx: self.clone(),
            size: size_bytes as u64,
            recycle: false,
        }
    }

    /// A fresh storage buffer holding `bytes`.
    ///
    /// Deliberately outside the pool in both directions — it neither takes a
    /// recycled buffer nor returns this one. See [`PooledBuffer`]: a
    /// `queue.write_buffer` into a recycled buffer can land ahead of recorded
    /// dispatches that still expect its old contents.
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
        // Whatever a scope has recorded must reach the GPU ahead of this copy,
        // or the read returns stale bytes. Going through `submit_with_copies`
        // rather than `flush` then a second submit keeps a decode step at one
        // submit and one fence wait.
        self.submit_with_copies(&[(buf, offset_bytes as u64, &staging, size_bytes as u64)]);
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
        let staging: Vec<wgpu::Buffer> = regions
            .iter()
            .map(|(_, _, size_bytes)| {
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: *size_bytes as u64,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        // See `stage_copy`: pending scope work and every one of these copies go
        // out in one command buffer, so N regions still cost one round trip.
        let copies: Vec<_> = regions
            .iter()
            .zip(&staging)
            .map(|((buf, off, size), s)| (*buf, *off as u64, s, *size as u64))
            .collect();
        self.submit_with_copies(&copies);

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
        self.counters.dispatches.fetch_add(1, Ordering::Relaxed);
        let (pipeline, layout) = self.pipeline(name);
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
            layout: &layout,
            entries: &entries,
        });
        // Inside a scope this records into the shared encoder and pass and
        // returns; the submit happens once, when the outermost scope drops.
        // Outside one the behaviour is what it has always been, which is what
        // lets this exist without touching a single caller.
        let mut scope = self.scope.lock().unwrap();
        if scope.depth > 0 {
            let ScopeState { encoder, pass, .. } = &mut *scope;
            // `flush` may have taken both mid-scope; remake what is missing.
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
        drop(scope);

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
