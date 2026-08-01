use std::sync::Arc;

use crate::backend::wgpu::{OFFSET_ALIGN_BYTES, PooledBuffer, WgpuContext};
use crate::device::Device;
use crate::dtype::DType;
use crate::error::{ForgeError, Result};
use crate::shape::Shape;

#[derive(Clone, Debug)]
pub enum CpuStorage {
    F32(Arc<Vec<f32>>),
    U32(Arc<Vec<u32>>),
}

#[derive(Clone)]
pub struct WgpuStorage {
    pub ctx: Arc<WgpuContext>,
    /// Recycled through the context's free list when the last handle drops
    /// (see [`PooledBuffer`]). `pub(crate)` rather than `pub`: it was only ever
    /// public by accident, and `buffer()` is the supported way to reach the
    /// underlying `wgpu::Buffer`.
    pub(crate) buf: Arc<PooledBuffer>,
    /// View offset into `buf`, in elements.
    pub offset: usize,
}

impl WgpuStorage {
    /// The underlying GPU buffer. `offset` is *not* applied — a storage is a
    /// view, and the offset is the caller's to add.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }
}

impl std::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuStorage(offset={})", self.offset)
    }
}

#[derive(Clone, Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    Wgpu(WgpuStorage),
}

/// A contiguous, row-major tensor on a specific device.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub(crate) storage: Storage,
    pub(crate) shape: Shape,
    pub(crate) dtype: DType,
}

impl Tensor {
    pub fn from_f32(data: &[f32], shape: impl Into<Shape>, device: &Device) -> Result<Tensor> {
        let shape = shape.into();
        if shape.numel() != data.len() {
            return Err(ForgeError::Shape(format!(
                "data length {} does not match shape {shape}",
                data.len()
            )));
        }
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuStorage::F32(Arc::new(data.to_vec()))),
            Device::Wgpu(ctx) => Storage::Wgpu(WgpuStorage {
                ctx: ctx.clone(),
                buf: Arc::new(ctx.upload(bytemuck::cast_slice(data))),
                offset: 0,
            }),
        };
        Ok(Tensor {
            storage,
            shape,
            dtype: DType::F32,
        })
    }

    pub fn from_u32(data: &[u32], shape: impl Into<Shape>, device: &Device) -> Result<Tensor> {
        let shape = shape.into();
        if shape.numel() != data.len() {
            return Err(ForgeError::Shape(format!(
                "data length {} does not match shape {shape}",
                data.len()
            )));
        }
        let storage = match device {
            Device::Cpu => Storage::Cpu(CpuStorage::U32(Arc::new(data.to_vec()))),
            Device::Wgpu(ctx) => Storage::Wgpu(WgpuStorage {
                ctx: ctx.clone(),
                buf: Arc::new(ctx.upload(bytemuck::cast_slice(data))),
                offset: 0,
            }),
        };
        Ok(Tensor {
            storage,
            shape,
            dtype: DType::U32,
        })
    }

    pub fn zeros(shape: impl Into<Shape>, device: &Device) -> Result<Tensor> {
        let shape = shape.into();
        Tensor::from_f32(&vec![0.0f32; shape.numel()], shape, device)
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        match &self.storage {
            Storage::Cpu(_) => Device::Cpu,
            Storage::Wgpu(s) => Device::Wgpu(s.ctx.clone()),
        }
    }

    pub(crate) fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Copy the tensor contents to a host `Vec<f32>`. Sync facade: on wasm32
    /// only CPU tensors can be read synchronously — use
    /// [`Tensor::to_vec_f32_async`] for WebGPU tensors there.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(ForgeError::Shape("to_vec_f32 on non-f32 tensor".into()));
        }
        match &self.storage {
            Storage::Cpu(CpuStorage::F32(v)) => Ok(v.as_ref().clone()),
            Storage::Cpu(_) => unreachable!("dtype checked above"),
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Wgpu(s) => {
                let bytes = s
                    .ctx
                    .readback(&s.buf, s.offset * 4, self.shape.numel() * 4)?;
                Ok(bytemuck::pod_collect_to_vec(&bytes))
            }
            #[cfg(target_arch = "wasm32")]
            Storage::Wgpu(_) => Err(ForgeError::Wgpu(
                "sync readback unavailable on wasm32; use to_vec_f32_async".into(),
            )),
        }
    }

    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        if self.dtype != DType::U32 {
            return Err(ForgeError::Shape("to_vec_u32 on non-u32 tensor".into()));
        }
        match &self.storage {
            Storage::Cpu(CpuStorage::U32(v)) => Ok(v.as_ref().clone()),
            Storage::Cpu(_) => unreachable!("dtype checked above"),
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Wgpu(s) => {
                let bytes = s
                    .ctx
                    .readback(&s.buf, s.offset * 4, self.shape.numel() * 4)?;
                Ok(bytemuck::pod_collect_to_vec(&bytes))
            }
            #[cfg(target_arch = "wasm32")]
            Storage::Wgpu(_) => Err(ForgeError::Wgpu(
                "sync readback unavailable on wasm32; use to_vec_u32_async".into(),
            )),
        }
    }

    /// Async readback — the primary form (identical result to `to_vec_f32`);
    /// on wasm the browser event loop drives the buffer mapping.
    pub async fn to_vec_f32_async(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(ForgeError::Shape("to_vec_f32 on non-f32 tensor".into()));
        }
        match &self.storage {
            Storage::Cpu(CpuStorage::F32(v)) => Ok(v.as_ref().clone()),
            Storage::Cpu(_) => unreachable!("dtype checked above"),
            Storage::Wgpu(s) => {
                let bytes = s
                    .ctx
                    .readback_async(&s.buf, s.offset * 4, self.shape.numel() * 4)
                    .await?;
                Ok(bytemuck::pod_collect_to_vec(&bytes))
            }
        }
    }

    /// Read several f32 tensors back in one GPU round trip, in the order
    /// given.
    ///
    /// [`Tensor::to_vec_f32_async`] is a submit and a fence wait per call, so
    /// a step that wants several small tensors — the attention probe reads
    /// `n_layer + 1` per generated token — pays for the round trips, not the
    /// bytes. Every tensor must be on the same device.
    pub async fn to_vec_f32_batch(tensors: &[Tensor]) -> Result<Vec<Vec<f32>>> {
        let mixed = || ForgeError::Shape("to_vec_f32_batch needs one device".into());
        let mut ctx: Option<&Arc<WgpuContext>> = None;
        for t in tensors {
            if t.dtype != DType::F32 {
                return Err(ForgeError::Shape("to_vec_f32 on non-f32 tensor".into()));
            }
            if let Storage::Wgpu(s) = &t.storage {
                match ctx {
                    None => ctx = Some(&s.ctx),
                    Some(c) if Arc::ptr_eq(c, &s.ctx) => {}
                    Some(_) => return Err(mixed()),
                }
            }
        }
        let Some(ctx) = ctx else {
            // All host-side: nothing to stage, and no round trip to save.
            return tensors.iter().map(Tensor::to_vec_f32).collect();
        };
        let mut regions = Vec::with_capacity(tensors.len());
        for t in tensors {
            match &t.storage {
                Storage::Wgpu(s) => regions.push((s.buffer(), s.offset * 4, t.shape.numel() * 4)),
                // A host tensor has nothing to stage, so a mixed batch would
                // misalign the results with their inputs.
                Storage::Cpu(_) => return Err(mixed()),
            }
        }
        Ok(ctx
            .readback_many_async(&regions)
            .await?
            .iter()
            .map(|b| bytemuck::pod_collect_to_vec(b))
            .collect())
    }

    pub async fn to_vec_u32_async(&self) -> Result<Vec<u32>> {
        if self.dtype != DType::U32 {
            return Err(ForgeError::Shape("to_vec_u32 on non-u32 tensor".into()));
        }
        match &self.storage {
            Storage::Cpu(CpuStorage::U32(v)) => Ok(v.as_ref().clone()),
            Storage::Cpu(_) => unreachable!("dtype checked above"),
            Storage::Wgpu(s) => {
                let bytes = s
                    .ctx
                    .readback_async(&s.buf, s.offset * 4, self.shape.numel() * 4)
                    .await?;
                Ok(bytemuck::pod_collect_to_vec(&bytes))
            }
        }
    }

    /// Reinterpret the shape; element count must be unchanged. Zero-copy.
    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        let shape = shape.into();
        if shape.numel() != self.shape.numel() {
            return Err(ForgeError::Shape(format!(
                "reshape {} -> {shape} changes element count",
                self.shape
            )));
        }
        Ok(Tensor {
            storage: self.storage.clone(),
            shape,
            dtype: self.dtype,
        })
    }

    /// View of `count` rows of a rank-2 tensor starting at `start`.
    /// Zero-copy on WebGPU (subject to 256-byte offset alignment), copied on CPU.
    pub fn narrow_rows(&self, start: usize, count: usize) -> Result<Tensor> {
        if self.shape.rank() != 2 {
            return Err(ForgeError::Shape(
                "narrow_rows needs a rank-2 tensor".into(),
            ));
        }
        let (rows, cols) = (self.shape.dim(0), self.shape.dim(1));
        if start + count > rows {
            return Err(ForgeError::Shape(format!(
                "narrow_rows {start}+{count} out of bounds for {rows} rows"
            )));
        }
        let shape = Shape::new(&[count, cols]);
        match &self.storage {
            Storage::Cpu(CpuStorage::F32(v)) => {
                let slice = v[start * cols..(start + count) * cols].to_vec();
                Ok(Tensor {
                    storage: Storage::Cpu(CpuStorage::F32(Arc::new(slice))),
                    shape,
                    dtype: self.dtype,
                })
            }
            Storage::Cpu(CpuStorage::U32(v)) => {
                let slice = v[start * cols..(start + count) * cols].to_vec();
                Ok(Tensor {
                    storage: Storage::Cpu(CpuStorage::U32(Arc::new(slice))),
                    shape,
                    dtype: self.dtype,
                })
            }
            Storage::Wgpu(s) => {
                let offset = s.offset + start * cols;
                if !(offset * self.dtype.size_bytes()).is_multiple_of(OFFSET_ALIGN_BYTES) {
                    return Err(ForgeError::Wgpu(format!(
                        "narrow_rows offset {offset} elems violates {OFFSET_ALIGN_BYTES}-byte alignment"
                    )));
                }
                Ok(Tensor {
                    storage: Storage::Wgpu(WgpuStorage {
                        ctx: s.ctx.clone(),
                        buf: s.buf.clone(),
                        offset,
                    }),
                    shape,
                    dtype: self.dtype,
                })
            }
        }
    }
}
