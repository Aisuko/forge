//! Backward and optimizer ops — the training half of the op API.
//!
//! Split out of `ops/mod.rs` so the whole set can sit behind the `train`
//! feature without scattering `#[cfg]` through the forward ops it used to be
//! interleaved with. `ops/mod.rs` re-exports every one of these, so the public
//! paths are unchanged: `forge::ops::adamw` still resolves, when `train` is on.

use crate::backend::{cpu, wgpu as gpu};
use crate::error::{ForgeError, Result};
use crate::shape::Shape;
use crate::tensor::{CpuStorage, Storage, Tensor};

use super::{cpu_f32, cpu_u32, f32_tensor, gpu_storage, same_device};

/// Rows and last-dim width of a tensor, for the ops that reduce over rows.
fn last_dim_rows(x: &Tensor) -> Result<(usize, usize)> {
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(ForgeError::Shape("op needs rank >= 1".into()));
    }
    let cols = dims[dims.len() - 1];
    Ok((x.shape().numel() / cols, cols))
}

/// GELU backward: dx = gelu'(x) * dy.
pub fn gelu_bwd(x: &Tensor, dy: &Tensor) -> Result<Tensor> {
    same_device(&[x, dy])?;
    if x.shape() != dy.shape() {
        return Err(ForgeError::Shape("gelu_bwd shape mismatch".into()));
    }
    let n = x.shape().numel();
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::gelu_bwd(cpu_f32(x)?, cpu_f32(dy)?).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::gelu_bwd(gpu_storage(x)?, gpu_storage(dy)?, n)),
    };
    Ok(f32_tensor(storage, x.shape().clone()))
}

/// Softmax backward from the forward *output* y: dx = y * (dy - sum(y*dy))
/// per row. Causal masking needs no parameters here — masked forward outputs
/// are exact zeros.
pub fn softmax_bwd(y: &Tensor, dy: &Tensor) -> Result<Tensor> {
    same_device(&[y, dy])?;
    if y.shape() != dy.shape() {
        return Err(ForgeError::Shape("softmax_bwd shape mismatch".into()));
    }
    let (rows, cols) = last_dim_rows(y)?;
    let storage = match y.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::softmax_bwd(cpu_f32(y)?, cpu_f32(dy)?, rows, cols).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::softmax_bwd(
            gpu_storage(y)?,
            gpu_storage(dy)?,
            rows,
            cols,
        )),
    };
    Ok(f32_tensor(storage, y.shape().clone()))
}

/// LayerNorm backward: (dx, dgamma, dbeta).
pub fn layernorm_bwd(
    x: &Tensor,
    gamma: &Tensor,
    dy: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor, Tensor)> {
    same_device(&[x, gamma, dy])?;
    if x.shape() != dy.shape() {
        return Err(ForgeError::Shape("layernorm_bwd shape mismatch".into()));
    }
    let (rows, cols) = last_dim_rows(x)?;
    if gamma.shape().numel() != cols {
        return Err(ForgeError::Shape("layernorm_bwd gamma length".into()));
    }
    let pshape = gamma.shape().clone();
    match x.storage() {
        Storage::Cpu(_) => {
            let dx =
                cpu::layernorm_bwd_dx(cpu_f32(x)?, cpu_f32(gamma)?, cpu_f32(dy)?, rows, cols, eps);
            let (dg, db) = cpu::layernorm_bwd_dparams(cpu_f32(x)?, cpu_f32(dy)?, rows, cols, eps);
            Ok((
                f32_tensor(Storage::Cpu(CpuStorage::F32(dx.into())), x.shape().clone()),
                f32_tensor(Storage::Cpu(CpuStorage::F32(dg.into())), pshape.clone()),
                f32_tensor(Storage::Cpu(CpuStorage::F32(db.into())), pshape),
            ))
        }
        Storage::Wgpu(_) => {
            let (dx, dg, db) = gpu::ops::layernorm_bwd(
                gpu_storage(x)?,
                gpu_storage(gamma)?,
                gpu_storage(dy)?,
                rows,
                cols,
                eps,
            );
            Ok((
                f32_tensor(Storage::Wgpu(dx), x.shape().clone()),
                f32_tensor(Storage::Wgpu(dg), pshape.clone()),
                f32_tensor(Storage::Wgpu(db), pshape),
            ))
        }
    }
}

/// Column sums over all leading dims: `[.., cols]` -> `[cols]`. Bias gradients.
pub fn sum_rows(x: &Tensor) -> Result<Tensor> {
    let (rows, cols) = last_dim_rows(x)?;
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::sum_rows(cpu_f32(x)?, rows, cols).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::sum_rows(gpu_storage(x)?, rows, cols)),
    };
    Ok(f32_tensor(storage, Shape::new(&[cols])))
}

/// `dst[ids[r]] += src[r]`, in place. Embedding-table gradient scatter.
pub fn scatter_add_rows(dst: &mut Tensor, ids: &Tensor, src: &Tensor) -> Result<()> {
    same_device(&[dst, ids, src])?;
    let (vocab, c) = match dst.shape().dims() {
        [v, c] => (*v, *c),
        _ => return Err(ForgeError::Shape("scatter_add dst must be rank 2".into())),
    };
    let t = ids.shape().numel();
    if src.shape().dims() != [t, c] {
        return Err(ForgeError::Shape(format!(
            "scatter_add src {} != [{t}, {c}]",
            src.shape()
        )));
    }
    match (&mut dst.storage, src.storage()) {
        (Storage::Cpu(CpuStorage::F32(d)), Storage::Cpu(CpuStorage::F32(s))) => {
            let ids = cpu_u32(ids)?;
            if let Some(&bad) = ids.iter().find(|&&id| id as usize >= vocab) {
                return Err(ForgeError::Shape(format!(
                    "scatter_add id {bad} >= {vocab}"
                )));
            }
            let d: &mut Vec<f32> = std::sync::Arc::make_mut(d);
            cpu::scatter_add_rows(d, ids, s, c);
            Ok(())
        }
        (Storage::Wgpu(d), Storage::Wgpu(s)) => {
            gpu::ops::scatter_add_rows(d, gpu_storage(ids)?, s, t, c, vocab * c);
            Ok(())
        }
        _ => Err(ForgeError::Device(
            "scatter_add expects f32 tensors on one device".into(),
        )),
    }
}

/// Per-row NLL: `out[r] = -log(probs[r, ids[r]])`.
pub fn gather_nll(probs: &Tensor, ids: &Tensor) -> Result<Tensor> {
    same_device(&[probs, ids])?;
    let (rows, cols) = last_dim_rows(probs)?;
    if ids.shape().numel() != rows {
        return Err(ForgeError::Shape("gather_nll ids length".into()));
    }
    let storage = match probs.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::gather_nll(cpu_f32(probs)?, cpu_u32(ids)?, cols).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::gather_nll(
            gpu_storage(probs)?,
            gpu_storage(ids)?,
            rows,
            cols,
        )),
    };
    Ok(f32_tensor(storage, Shape::new(&[rows])))
}

/// Cross-entropy backward: dlogits = (probs - onehot(ids)) * scale.
pub fn ce_bwd(probs: &Tensor, ids: &Tensor, scale: f32) -> Result<Tensor> {
    same_device(&[probs, ids])?;
    let (rows, cols) = last_dim_rows(probs)?;
    if ids.shape().numel() != rows {
        return Err(ForgeError::Shape("ce_bwd ids length".into()));
    }
    let storage = match probs.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::ce_bwd(cpu_f32(probs)?, cpu_u32(ids)?, rows, cols, scale).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::ce_bwd(
            gpu_storage(probs)?,
            gpu_storage(ids)?,
            rows,
            cols,
            scale,
        )),
    };
    Ok(f32_tensor(storage, probs.shape().clone()))
}

/// Inverted dropout with a deterministic counter RNG (identical masks on
/// both backends). Apply the same (p, seed) to dy for the backward pass.
pub fn dropout(x: &Tensor, p: f32, seed: u32) -> Result<Tensor> {
    if !(0.0..1.0).contains(&p) {
        return Err(ForgeError::Shape(format!("dropout p {p} outside [0, 1)")));
    }
    if p == 0.0 {
        return Ok(x.clone());
    }
    // Compute the keep-scale once on the CPU and hand the same bits to both
    // backends: GPU division isn't guaranteed correctly-rounded (unlike
    // multiplication), so an independent 1.0/(1.0-p) on the GPU can be a
    // few ULP off from the CPU's, breaking bit-for-bit parity.
    let scale = 1.0f32 / (1.0 - p);
    let n = x.shape().numel();
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::dropout(cpu_f32(x)?, p, scale, seed).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::dropout(gpu_storage(x)?, n, p, scale, seed)),
    };
    Ok(f32_tensor(storage, x.shape().clone()))
}

/// split_heads backward for one of q/k/v (`which` in 0..3):
/// d [h, t, hd] -> [t, 3c] with the other thirds zero.
pub fn unsplit_head(d: &Tensor, which: usize) -> Result<Tensor> {
    unsplit_head_batched(d, which, 1)
}

/// [`unsplit_head`] over a stacked batch: d [b*h, t, hd] -> [b*t, 3c].
pub fn unsplit_head_batched(d: &Tensor, which: usize, batch: usize) -> Result<Tensor> {
    let (bh, t, hd) = match d.shape().dims() {
        [bh, t, hd] => (*bh, *t, *hd),
        _ => return Err(ForgeError::Shape("unsplit_head needs [b*h, t, hd]".into())),
    };
    if batch == 0 || bh % batch != 0 {
        return Err(ForgeError::Shape(format!(
            "unsplit_head: {bh} planes not divisible into {batch} sequences"
        )));
    }
    let h = bh / batch;
    let c = h * hd;
    let shape = Shape::new(&[batch * t, 3 * c]);
    let storage = match d.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::unsplit_head(cpu_f32(d)?, batch, t, c, h, which).into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::unsplit_head(
            gpu_storage(d)?,
            batch,
            t,
            c,
            h,
            which,
        )),
    };
    Ok(f32_tensor(storage, shape))
}

/// merge_heads backward: dy [t, c] -> [h, t, hd].
pub fn unmerge_heads(dy: &Tensor, h: usize) -> Result<Tensor> {
    unmerge_heads_batched(dy, h, 1)
}

/// [`unmerge_heads`] over a stacked batch: dy [b*t, c] -> [b*h, t, hd].
pub fn unmerge_heads_batched(dy: &Tensor, h: usize, batch: usize) -> Result<Tensor> {
    let (bt, c) = match dy.shape().dims() {
        [bt, c] => (*bt, *c),
        _ => return Err(ForgeError::Shape("unmerge_heads needs [b*t, c]".into())),
    };
    if c % h != 0 {
        return Err(ForgeError::Shape(format!("unmerge_heads c={c} % h={h}")));
    }
    if batch == 0 || bt % batch != 0 {
        return Err(ForgeError::Shape(format!(
            "unmerge_heads: {bt} rows not divisible into {batch} sequences"
        )));
    }
    let t = bt / batch;
    let shape = Shape::new(&[batch * h, t, c / h]);
    let storage = match dy.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu::unmerge_heads(cpu_f32(dy)?, batch, t, c, h).into(),
        )),
        Storage::Wgpu(_) => {
            Storage::Wgpu(gpu::ops::unmerge_heads(gpu_storage(dy)?, batch, t, c, h))
        }
    };
    Ok(f32_tensor(storage, shape))
}

/// Sum of squares of all elements (for the global gradient norm).
/// Training-path op with a sync readback — WebGPU tensors are native-only
/// here (browser training is out of scope for 1.0).
pub fn sumsq(x: &Tensor) -> Result<f32> {
    Ok(sumsq_all(std::slice::from_ref(x))?[0])
}

/// [`sumsq`] over many tensors, in **one** GPU round trip.
///
/// The global gradient norm needs this for every parameter at once, and a
/// round trip is ~90 µs against a reduction that takes single-digit µs — so
/// per-tensor, GPT-2's 148 parameters cost more in fence waits than the whole
/// optimizer step does in arithmetic. Every reduction is recorded first, then
/// all the partials come back together.
pub fn sumsq_all(xs: &[Tensor]) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; xs.len()];
    // Reductions for the GPU tensors, recorded now and read once below.
    let mut pending: Vec<(usize, Tensor)> = Vec::new();
    for (i, x) in xs.iter().enumerate() {
        match x.storage() {
            Storage::Cpu(_) => out[i] = cpu_f32(x)?.iter().map(|&v| v * v).sum(),
            #[cfg(not(target_arch = "wasm32"))]
            Storage::Wgpu(s) => {
                let (partials, groups) = gpu::ops::sumsq_partials(s, x.shape().numel());
                pending.push((
                    i,
                    crate::ops::f32_tensor(
                        Storage::Wgpu(partials),
                        crate::shape::Shape::new(&[groups]),
                    ),
                ));
            }
            #[cfg(target_arch = "wasm32")]
            Storage::Wgpu(_) => {
                return Err(crate::error::ForgeError::Wgpu(
                    "sumsq readback unavailable on wasm32 (training is native-only)".into(),
                ));
            }
        }
    }
    if pending.is_empty() {
        return Ok(out);
    }
    let tensors: Vec<Tensor> = pending.iter().map(|(_, t)| t.clone()).collect();
    // Sync facade over the batched async read; native-only, as above.
    let all = pollster::block_on(Tensor::to_vec_f32_batch(&tensors))?;
    for ((i, _), vals) in pending.iter().zip(all) {
        out[*i] = vals.iter().sum();
    }
    Ok(out)
}

/// y = x * alpha.
pub fn scale(x: &Tensor, alpha: f32) -> Result<Tensor> {
    let n = x.shape().numel();
    let storage = match x.storage() {
        Storage::Cpu(_) => Storage::Cpu(CpuStorage::F32(
            cpu_f32(x)?
                .iter()
                .map(|&v| v * alpha)
                .collect::<Vec<_>>()
                .into(),
        )),
        Storage::Wgpu(_) => Storage::Wgpu(gpu::ops::scale(gpu_storage(x)?, n, alpha)),
    };
    Ok(f32_tensor(storage, x.shape().clone()))
}

/// AdamW step with decoupled weight decay, updating param/m/v in place.
/// `step` is 1-based.
#[allow(clippy::too_many_arguments)]
pub fn adamw(
    param: &mut Tensor,
    grad: &Tensor,
    m: &mut Tensor,
    v: &mut Tensor,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) -> Result<()> {
    if param.shape() != grad.shape() || param.shape() != m.shape() || param.shape() != v.shape() {
        return Err(ForgeError::Shape("adamw shape mismatch".into()));
    }
    let n = param.shape().numel();
    match (&mut param.storage, grad.storage()) {
        (Storage::Cpu(CpuStorage::F32(p)), Storage::Cpu(CpuStorage::F32(g))) => {
            let (Storage::Cpu(CpuStorage::F32(ms)), Storage::Cpu(CpuStorage::F32(vs))) =
                (&mut m.storage, &mut v.storage)
            else {
                return Err(ForgeError::Device("adamw state device mismatch".into()));
            };
            let p: &mut Vec<f32> = std::sync::Arc::make_mut(p);
            let ms: &mut Vec<f32> = std::sync::Arc::make_mut(ms);
            let vs: &mut Vec<f32> = std::sync::Arc::make_mut(vs);
            cpu::adamw(p, g, ms, vs, lr, beta1, beta2, eps, weight_decay, step);
            Ok(())
        }
        (Storage::Wgpu(p), Storage::Wgpu(g)) => {
            let (Storage::Wgpu(ms), Storage::Wgpu(vs)) = (&m.storage, &v.storage) else {
                return Err(ForgeError::Device("adamw state device mismatch".into()));
            };
            gpu::ops::adamw(p, g, ms, vs, n, lr, beta1, beta2, eps, weight_decay, step);
            Ok(())
        }
        _ => Err(ForgeError::Device(
            "adamw expects f32 tensors on one device".into(),
        )),
    }
}
