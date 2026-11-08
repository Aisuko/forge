use crate::backend::{cpu, wgpu as gpu};
use crate::error::{ForgeError, Result};
use crate::shape::Shape;
use crate::tensor::{CpuStorage, Storage, Tensor};

use super::{cpu_f32, cpu_u32, f32_tensor, gpu_storage, same_device};

fn last_dim_rows(x: &Tensor) -> Result<(usize, usize)> {
    let dims = x.shape().dims();
    if dims.is_empty() {
        return Err(ForgeError::Shape("op needs rank >= 1".into()));
    }
    let cols = dims[dims.len() - 1];
    Ok((x.shape().numel() / cols, cols))
}

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

pub fn dropout(x: &Tensor, p: f32, seed: u32) -> Result<Tensor> {
    if !(0.0..1.0).contains(&p) {
        return Err(ForgeError::Shape(format!("dropout p {p} outside [0, 1)")));
    }
    if p == 0.0 {
        return Ok(x.clone());
    }
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

pub fn unsplit_head(d: &Tensor, which: usize) -> Result<Tensor> {
    unsplit_head_batched(d, which, 1)
}

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

pub fn unmerge_heads(dy: &Tensor, h: usize) -> Result<Tensor> {
    unmerge_heads_batched(dy, h, 1)
}

pub fn unmerge_heads_batched(dy: &Tensor, h: usize, batch: usize) -> Result<Tensor> {
    super::split_heads_single(dy, h, batch)
}

pub fn sumsq(x: &Tensor) -> Result<f32> {
    Ok(sumsq_all(std::slice::from_ref(x))?[0])
}

pub fn sumsq_all(xs: &[Tensor]) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; xs.len()];
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
    let all = pollster::block_on(Tensor::to_vec_f32_batch(&tensors))?;
    for ((i, _), vals) in pending.iter().zip(all) {
        out[*i] = vals.iter().sum();
    }
    Ok(out)
}

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
