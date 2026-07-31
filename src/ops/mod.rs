//! Backend-agnostic tensor operations. Each op validates shapes, dispatches
//! to the CPU reference or the WGPU kernels, and returns a tensor on the same
//! device as its inputs.

use crate::backend::{cpu, wgpu as gpu};
use crate::error::{ForgeError, Result};
use crate::shape::Shape;
use crate::tensor::{CpuStorage, Storage, Tensor, WgpuStorage};

pub(crate) fn cpu_f32(t: &Tensor) -> Result<&[f32]> {
    match t.storage() {
        Storage::Cpu(CpuStorage::F32(v)) => Ok(v),
        _ => Err(ForgeError::Device("expected CPU f32 tensor".into())),
    }
}

pub(crate) fn cpu_u32(t: &Tensor) -> Result<&[u32]> {
    match t.storage() {
        Storage::Cpu(CpuStorage::U32(v)) => Ok(v),
        _ => Err(ForgeError::Device("expected CPU u32 tensor".into())),
    }
}

pub(crate) fn gpu_storage(t: &Tensor) -> Result<&WgpuStorage> {
    match t.storage() {
        Storage::Wgpu(s) => Ok(s),
        _ => Err(ForgeError::Device("expected WGPU tensor".into())),
    }
}

pub(crate) fn same_device(tensors: &[&Tensor]) -> Result<()> {
    let first = tensors[0].device();
    for t in &tensors[1..] {
        if !first.same_as(&t.device()) {
            return Err(ForgeError::Device(format!(
                "tensors on different devices: {} vs {}",
                first.describe(),
                t.device().describe()
            )));
        }
    }
    Ok(())
}

pub(crate) fn f32_tensor(storage: Storage, shape: Shape) -> Tensor {
    Tensor {
        storage,
        shape,
        dtype: crate::dtype::DType::F32,
    }
}

/// Elementwise `a + b`; shapes must match exactly.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    same_device(&[a, b])?;
    if a.shape() != b.shape() {
        return Err(ForgeError::Shape(format!(
            "add shape mismatch: {} vs {}",
            a.shape(),
            b.shape()
        )));
    }
    let n = a.shape().numel();
    let storage = match a.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(cpu::add(cpu_f32(a)?, cpu_f32(b)?).into())),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::add(gpu_storage(a)?, gpu_storage(b)?, n)),
    };
    Ok(f32_tensor(storage, a.shape().clone()))
}

/// GELU (tanh approximation).
pub fn gelu(x: &Tensor) -> Result<Tensor> {
    let n = x.shape().numel();
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(cpu::gelu(cpu_f32(x)?).into())),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::gelu(gpu_storage(x)?, n)),
    };
    Ok(f32_tensor(storage, x.shape().clone()))
}

#[derive(Clone, Copy)]
pub struct MatmulSpec {
    /// Interpret A as [k, m] instead of [m, k] (used by backward passes:
    /// dB = Aᵀ·dY without materializing the transpose).
    pub trans_a: bool,
    /// Interpret B as [n, k] instead of [k, n].
    pub trans_b: bool,
    /// Scale applied to the product (before bias).
    pub alpha: f32,
    /// Logical row count of each batch element of B, when B is a view over
    /// a larger preallocated buffer (KV cache): only the first `b_rows` rows
    /// are read, while the batch stride comes from the physical shape.
    pub b_rows: Option<usize>,
}

impl Default for MatmulSpec {
    fn default() -> Self {
        MatmulSpec {
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            b_rows: None,
        }
    }
}

/// Batched matmul with optional A/B-transpose, scaling, and bias.
///
/// A: `[m, k]` or `[batch, m, k]` (`[k, m]` with `trans_a`); B: `[k, n]`
/// (`[n, k]` transposed), optionally batched. A rank-2 operand broadcasts
/// across the batch. Bias: `[n]`.
pub fn matmul(a: &Tensor, b: &Tensor, bias: Option<&Tensor>, spec: MatmulSpec) -> Result<Tensor> {
    same_device(&[a, b])?;
    if let Some(bias) = bias {
        same_device(&[a, bias])?;
    }
    let (batch_a, ad0, ad1) = rank23_dims(a, "A")?;
    let (m, k) = if spec.trans_a { (ad1, ad0) } else { (ad0, ad1) };
    let (batch_b, bd0_phys, bd1) = rank23_dims(b, "B")?;
    let bd0 = match spec.b_rows {
        Some(rows) if rows <= bd0_phys => rows,
        Some(rows) => {
            return Err(ForgeError::Shape(format!(
                "matmul b_rows {rows} exceeds physical rows {bd0_phys}"
            )));
        }
        None => bd0_phys,
    };
    let (bk, n) = if spec.trans_b { (bd1, bd0) } else { (bd0, bd1) };
    if bk != k {
        return Err(ForgeError::Shape(format!(
            "matmul inner dim mismatch: A {} vs B {}",
            a.shape(),
            b.shape()
        )));
    }
    let batch = match (batch_a, batch_b) {
        (None, None) => 1,
        (Some(x), None) | (None, Some(x)) => x,
        (Some(x), Some(y)) if x == y => x,
        (Some(x), Some(y)) => {
            return Err(ForgeError::Shape(format!(
                "matmul batch mismatch: {x} vs {y}"
            )));
        }
    };
    // Batch strides come from the *physical* shapes.
    let a_stride = if batch_a.is_some() { ad0 * ad1 } else { 0 };
    let b_stride = if batch_b.is_some() { bd0_phys * bd1 } else { 0 };
    if let Some(bias) = bias
        && bias.shape().numel() != n
    {
        return Err(ForgeError::Shape(format!(
            "bias length {} != n {n}",
            bias.shape().numel()
        )));
    }
    let out_shape = if batch_a.is_some() || batch_b.is_some() {
        Shape::new(&[batch, m, n])
    } else {
        Shape::new(&[m, n])
    };
    let storage = match a.storage() {
        Storage::Cpu(_) => {
            let bias = bias.map(cpu_f32).transpose()?;
            Storage::Cpu(CpuStorage::F32(
                cpu::matmul(
                    cpu_f32(a)?,
                    cpu_f32(b)?,
                    bias,
                    m,
                    k,
                    n,
                    batch,
                    a_stride,
                    b_stride,
                    spec.trans_a,
                    spec.trans_b,
                    spec.alpha,
                )
                .into(),
            ))
        }
        Storage::Wgpu(_) => {
            let bias = bias.map(gpu_storage).transpose()?;
            Storage::Wgpu(gpu::ops::matmul(
                gpu_storage(a)?,
                gpu_storage(b)?,
                bias,
                m,
                k,
                n,
                batch,
                a_stride,
                b_stride,
                spec.trans_a,
                spec.trans_b,
                spec.alpha,
            ))
        }
    };
    Ok(f32_tensor(storage, out_shape))
}

fn rank23_dims(t: &Tensor, which: &str) -> Result<(Option<usize>, usize, usize)> {
    match t.shape().dims() {
        [d0, d1] => Ok((None, *d0, *d1)),
        [bb, d0, d1] => Ok((Some(*bb), *d0, *d1)),
        d => Err(ForgeError::Shape(format!(
            "matmul {which} rank {} unsupported",
            d.len()
        ))),
    }
}

/// Softmax over the last dim. With `causal`, row r (query position
/// `r % q_len`, where q_len is the second-to-last dim) sees key j only when
/// `j <= query + off`; `off` is `key_len - query_len` when using a KV cache.
pub fn softmax(x: &Tensor, causal: bool, off: usize) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(ForgeError::Shape("softmax on scalar".into()));
    }
    let cols = dims[dims.len() - 1];
    let rows = x.shape().numel() / cols;
    let q_len = if dims.len() >= 2 {
        dims[dims.len() - 2]
    } else {
        1
    };
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::softmax(cpu_f32(x)?, rows, cols, q_len, causal, off).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::softmax(
            gpu_storage(x)?,
            rows,
            cols,
            q_len,
            causal,
            off,
        )),
    };
    Ok(f32_tensor(storage, x.shape().clone()))
}

/// LayerNorm over the last dim.
pub fn layernorm(x: &Tensor, gamma: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    same_device(&[x, gamma, beta])?;
    let dims = x.shape().dims();
    let cols = dims[dims.len() - 1];
    let rows = x.shape().numel() / cols;
    if gamma.shape().numel() != cols || beta.shape().numel() != cols {
        return Err(ForgeError::Shape(
            "layernorm gamma/beta length mismatch".into(),
        ));
    }
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::layernorm(
                cpu_f32(x)?,
                cpu_f32(gamma)?,
                cpu_f32(beta)?,
                rows,
                cols,
                eps,
            )
            .into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::layernorm(
            gpu_storage(x)?,
            gpu_storage(gamma)?,
            gpu_storage(beta)?,
            rows,
            cols,
            eps,
        )),
    };
    Ok(f32_tensor(storage, x.shape().clone()))
}

/// Fused token + positional embedding: `out[t] = wte[ids[t]] + wpe[t + pos]`.
pub fn embedding(ids: &Tensor, wte: &Tensor, wpe: Option<&Tensor>, pos: usize) -> Result<Tensor> {
    let chunk_rows = wte
        .shape()
        .dims()
        .first()
        .copied()
        .ok_or_else(|| ForgeError::Shape("wte must be rank 2".into()))?;
    embedding_chunked(ids, std::slice::from_ref(wte), chunk_rows, wpe, pos)
}

/// Embedding gather from a row-chunked table. `chunks[i]` holds rows
/// [i * chunk_rows, ...) of the full [vocab, c] table — chunking keeps every
/// GPU binding under max_storage_buffer_binding_size (GPT-2's wte alone is
/// ~147 MiB, above the WebGPU default and llvmpipe's hard 128 MiB limit).
pub fn embedding_chunked(
    ids: &Tensor,
    chunks: &[Tensor],
    chunk_rows: usize,
    wpe: Option<&Tensor>,
    pos: usize,
) -> Result<Tensor> {
    if chunks.is_empty() {
        return Err(ForgeError::Shape(
            "embedding needs at least one chunk".into(),
        ));
    }
    same_device(&[ids, &chunks[0]])?;
    let t = ids.shape().numel();
    let c = match chunks[0].shape().dims() {
        [_, c] => *c,
        _ => return Err(ForgeError::Shape("wte chunks must be rank 2".into())),
    };
    for (i, ch) in chunks.iter().enumerate() {
        match ch.shape().dims() {
            [r, cc] if *cc == c && (*r == chunk_rows || i + 1 == chunks.len()) => {}
            _ => return Err(ForgeError::Shape("inconsistent wte chunk shapes".into())),
        }
    }
    let n_ctx = match wpe {
        Some(wpe) => match wpe.shape().dims() {
            [n, wc] if *wc == c => *n,
            _ => return Err(ForgeError::Shape("wpe must be [n_ctx, c]".into())),
        },
        None => 0,
    };
    if let Some(wpe) = wpe {
        same_device(&[&chunks[0], wpe])?;
        if t + pos > n_ctx {
            return Err(ForgeError::Shape(format!(
                "sequence {t}+{pos} exceeds context length {n_ctx}"
            )));
        }
    }
    let storage = match ids.storage() {
        Storage::Cpu(_) => {
            let ids = cpu_u32(ids)?;
            let wpe = wpe.map(cpu_f32).transpose()?;
            let mut out = vec![0.0f32; t * c];
            for (tt, &id) in ids.iter().enumerate() {
                let (chunk_i, row) = ((id as usize) / chunk_rows, (id as usize) % chunk_rows);
                let table = cpu_f32(&chunks[chunk_i])?;
                let dst = &mut out[tt * c..(tt + 1) * c];
                dst.copy_from_slice(&table[row * c..(row + 1) * c]);
                if let Some(wpe) = wpe {
                    for (d, &pv) in dst.iter_mut().zip(&wpe[(tt + pos) * c..(tt + pos + 1) * c]) {
                        *d += pv;
                    }
                }
            }
            Storage::Cpu(CpuStorage::F32(out.into()))
        }
        Storage::Wgpu(_) => {
            let wpe = wpe.map(gpu_storage).transpose()?;
            let gpu_chunks: Vec<(&crate::tensor::WgpuStorage, usize)> = chunks
                .iter()
                .map(|ch| Ok((gpu_storage(ch)?, ch.shape().dim(0))))
                .collect::<Result<_>>()?;
            Storage::Wgpu(gpu::ops::embedding(
                gpu_storage(ids)?,
                &gpu_chunks,
                chunk_rows,
                wpe,
                t,
                c,
                n_ctx,
                pos,
            ))
        }
    };
    Ok(f32_tensor(storage, Shape::new(&[t, c])))
}

/// a [m, k] @ concat(chunks)^T -> [m, sum n_i], chunks each [n_i, k].
/// The column-chunked equivalent of `matmul` with `trans_b` — used for the
/// weight-tied LM head over a chunked wte.
pub fn matmul_chunked_transb(a: &Tensor, chunks: &[Tensor], alpha: f32) -> Result<Tensor> {
    if chunks.is_empty() {
        return Err(ForgeError::Shape("matmul needs at least one chunk".into()));
    }
    same_device(&[a, &chunks[0]])?;
    let (m, k) = match a.shape().dims() {
        [m, k] => (*m, *k),
        _ => return Err(ForgeError::Shape("chunked matmul A must be rank 2".into())),
    };
    let mut n_total = 0usize;
    for ch in chunks {
        match ch.shape().dims() {
            [n_i, kk] if *kk == k => n_total += n_i,
            _ => {
                return Err(ForgeError::Shape(format!(
                    "chunk shape {} incompatible with k={k}",
                    ch.shape()
                )));
            }
        }
    }
    let out_shape = Shape::new(&[m, n_total]);
    match a.storage() {
        Storage::Cpu(_) => {
            let a_data = cpu_f32(a)?;
            let mut out = vec![0.0f32; m * n_total];
            let mut n_off = 0usize;
            for ch in chunks {
                let n_i = ch.shape().dim(0);
                let part = cpu::matmul(
                    a_data,
                    cpu_f32(ch)?,
                    None,
                    m,
                    k,
                    n_i,
                    1,
                    0,
                    0,
                    false,
                    true,
                    alpha,
                );
                for row in 0..m {
                    out[row * n_total + n_off..row * n_total + n_off + n_i]
                        .copy_from_slice(&part[row * n_i..(row + 1) * n_i]);
                }
                n_off += n_i;
            }
            Ok(f32_tensor(
                Storage::Cpu(CpuStorage::F32(out.into())),
                out_shape,
            ))
        }
        Storage::Wgpu(s) => {
            let out = gpu::ops::alloc_storage(&s.ctx, m * n_total);
            let mut n_off = 0usize;
            for ch in chunks {
                let n_i = ch.shape().dim(0);
                gpu::ops::matmul_into(
                    &out,
                    gpu_storage(a)?,
                    gpu_storage(ch)?,
                    None,
                    m,
                    k,
                    n_i,
                    1,
                    0,
                    0,
                    false,
                    true,
                    alpha,
                    n_off,
                    n_total,
                );
                n_off += n_i;
            }
            Ok(f32_tensor(Storage::Wgpu(out), out_shape))
        }
    }
}

/// Append `src` [h, t, hd] into the preallocated cache [h, cap, hd] at row
/// offset `len` within each head, in place. Used by KV-cache decode.
pub fn kv_append(cache: &mut Tensor, src: &Tensor, len: usize) -> Result<()> {
    same_device(&[cache, src])?;
    let (h, cap, hd) = match cache.shape().dims() {
        [h, cap, hd] => (*h, *cap, *hd),
        _ => return Err(ForgeError::Shape("kv_append cache must be rank 3".into())),
    };
    let t = match src.shape().dims() {
        [sh, t, shd] if *sh == h && *shd == hd => *t,
        _ => {
            return Err(ForgeError::Shape(format!(
                "kv_append src {} incompatible with cache {}",
                src.shape(),
                cache.shape()
            )));
        }
    };
    if len + t > cap {
        return Err(ForgeError::Shape(format!(
            "kv_append {len}+{t} exceeds cache capacity {cap}"
        )));
    }
    match (&mut cache.storage, src.storage()) {
        (Storage::Cpu(CpuStorage::F32(dst)), Storage::Cpu(CpuStorage::F32(s))) => {
            let dst: &mut Vec<f32> = std::sync::Arc::make_mut(dst);
            cpu::kv_append(dst, s, h, t, hd, cap, len);
            Ok(())
        }
        (Storage::Wgpu(c), Storage::Wgpu(s)) => {
            gpu::ops::kv_append(c, s, h, t, hd, cap, len);
            Ok(())
        }
        _ => Err(ForgeError::Device(
            "kv_append expects f32 tensors on one device".into(),
        )),
    }
}

/// qkv [t, 3c] -> (q, k, v) each [h, t, c/h].
pub fn split_heads(qkv: &Tensor, n_head: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let (t, c3) = match qkv.shape().dims() {
        [t, c3] => (*t, *c3),
        _ => return Err(ForgeError::Shape("split_heads needs [t, 3c]".into())),
    };
    let c = c3 / 3;
    if c3 != 3 * c || c % n_head != 0 {
        return Err(ForgeError::Shape(format!(
            "split_heads: dim {c3} not divisible into 3 x {n_head} heads"
        )));
    }
    let shape = Shape::new(&[n_head, t, c / n_head]);
    match qkv.storage() {
        Storage::Cpu(_) => {
            let (q, k, v) = cpu::split_heads(cpu_f32(qkv)?, t, c, n_head);
            Ok((
                f32_tensor(Storage::Cpu(CpuStorage::F32(q.into())), shape.clone()),
                f32_tensor(Storage::Cpu(CpuStorage::F32(k.into())), shape.clone()),
                f32_tensor(Storage::Cpu(CpuStorage::F32(v.into())), shape),
            ))
        }
        Storage::Wgpu(_) => {
            let (q, k, v) = gpu::ops::split_heads(gpu_storage(qkv)?, t, c, n_head);
            Ok((
                f32_tensor(Storage::Wgpu(q), shape.clone()),
                f32_tensor(Storage::Wgpu(k), shape.clone()),
                f32_tensor(Storage::Wgpu(v), shape),
            ))
        }
    }
}

/// x [h, t, hd] -> [t, h*hd].
pub fn merge_heads(x: &Tensor) -> Result<Tensor> {
    let (h, t, hd) = match x.shape().dims() {
        [h, t, hd] => (*h, *t, *hd),
        _ => return Err(ForgeError::Shape("merge_heads needs [h, t, hd]".into())),
    };
    let c = h * hd;
    let shape = Shape::new(&[t, c]);
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::merge_heads(cpu_f32(x)?, t, c, h).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::merge_heads(gpu_storage(x)?, t, c, h)),
    };
    Ok(f32_tensor(storage, shape))
}

/// The backward and optimizer ops. Behind `train`: a caller doing inference
/// dispatches none of them, and this is the whole reason the feature exists.
#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
mod train;
#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub use train::*;
