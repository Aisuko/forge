//! CPU reference backend. Mathematically identical semantics to the WGSL
//! kernels; used for testing, verification, and gradient checking.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// GELU, tanh approximation ("gelu_new") — the variant GPT-2 was trained with.
pub fn gelu(x: &[f32]) -> Vec<f32> {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + (C * (v + 0.044715 * v * v * v)).tanh()))
        .collect()
}

/// Batched matmul matching `shaders/matmul.wgsl`:
/// C[b] = alpha * A[b] @ B[b] (+ bias broadcast over rows).
/// `a_stride`/`b_stride` are per-batch element strides (0 broadcasts).
/// With `trans_a`, A[b] is stored [k, m]; with `trans_b`, B[b] is stored
/// [n, k] instead of [k, n].
#[allow(clippy::too_many_arguments)]
pub fn matmul(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    a_stride: usize,
    b_stride: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    // rayon on native; serial on wasm32 (no threads there).
    #[cfg(not(target_arch = "wasm32"))]
    let rows = out.par_chunks_mut(n);
    #[cfg(target_arch = "wasm32")]
    let rows = out.chunks_mut(n);
    rows.enumerate().for_each(|(row_idx, orow)| {
        let bat = row_idx / m;
        let i = row_idx % m;
        let a_base = bat * a_stride;
        let b_base = bat * b_stride;
        let a_at = |kk: usize| {
            if trans_a {
                a[a_base + kk * m + i]
            } else {
                a[a_base + i * k + kk]
            }
        };
        if trans_b {
            for (j, o) in orow.iter_mut().enumerate() {
                let b_row = &b[b_base + j * k..b_base + (j + 1) * k];
                let dot: f32 = b_row
                    .iter()
                    .enumerate()
                    .map(|(kk, &bv)| a_at(kk) * bv)
                    .sum();
                *o = dot * alpha;
            }
        } else {
            for kk in 0..k {
                let av = a_at(kk);
                let b_row = &b[b_base + kk * n..b_base + (kk + 1) * n];
                for (o, &bv) in orow.iter_mut().zip(b_row) {
                    *o += av * bv;
                }
            }
            if alpha != 1.0 {
                for o in orow.iter_mut() {
                    *o *= alpha;
                }
            }
        }
        if let Some(bias) = bias {
            for (o, &bb) in orow.iter_mut().zip(bias) {
                *o += bb;
            }
        }
    });
    out
}

/// Append src [h, t, hd] into dst [h, cap, hd] at row offset `len` per head.
pub fn kv_append(
    dst: &mut [f32],
    src: &[f32],
    h: usize,
    t: usize,
    hd: usize,
    cap: usize,
    len: usize,
) {
    for hh in 0..h {
        for tt in 0..t {
            let d0 = hh * cap * hd + (len + tt) * hd;
            let s0 = hh * t * hd + tt * hd;
            dst[d0..d0 + hd].copy_from_slice(&src[s0..s0 + hd]);
        }
    }
}

/// Stable softmax over the last dim with optional causal masking, matching
/// `shaders/softmax.wgsl`. Row r's query position is `r % q_len`; key j is
/// visible when `j <= query + off`. Masked entries produce 0.
pub fn softmax(
    x: &[f32],
    rows: usize,
    cols: usize,
    q_len: usize,
    causal: bool,
    off: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let visible = |j: usize| !causal || j <= (r % q_len) + off;
        let mut max = f32::NEG_INFINITY;
        for j in 0..cols {
            if visible(j) {
                max = max.max(x[base + j]);
            }
        }
        let mut sum = 0.0f32;
        for j in 0..cols {
            if visible(j) {
                sum += (x[base + j] - max).exp();
            }
        }
        for j in 0..cols {
            if visible(j) {
                out[base + j] = (x[base + j] - max).exp() / sum;
            }
        }
    }
    out
}

/// LayerNorm over the last dim (biased variance, like PyTorch).
pub fn layernorm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    let nf = cols as f32;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean: f32 = row.iter().sum::<f32>() / nf;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / nf;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..cols {
            out[r * cols + j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
        }
    }
    out
}

/// Fused token + positional embedding gather.
pub fn embedding(ids: &[u32], wte: &[f32], wpe: Option<&[f32]>, c: usize, pos: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; ids.len() * c];
    for (t, &id) in ids.iter().enumerate() {
        let dst = &mut out[t * c..(t + 1) * c];
        dst.copy_from_slice(&wte[id as usize * c..(id as usize + 1) * c]);
        if let Some(wpe) = wpe {
            for (d, &pv) in dst.iter_mut().zip(&wpe[(t + pos) * c..(t + pos + 1) * c]) {
                *d += pv;
            }
        }
    }
    out
}

/// qkv: [t, 3c] -> (q, k, v) each [h, t, hd], hd = c / h.
pub fn split_heads(qkv: &[f32], t: usize, c: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hd = c / h;
    let mut q = vec![0.0f32; t * c];
    let mut k = vec![0.0f32; t * c];
    let mut v = vec![0.0f32; t * c];
    for hh in 0..h {
        for tt in 0..t {
            for d in 0..hd {
                let dst = hh * t * hd + tt * hd + d;
                let src_row = tt * 3 * c;
                let col = hh * hd + d;
                q[dst] = qkv[src_row + col];
                k[dst] = qkv[src_row + c + col];
                v[dst] = qkv[src_row + 2 * c + col];
            }
        }
    }
    (q, k, v)
}

/// x: [h, t, hd] -> [t, c], c = h * hd.
pub fn merge_heads(x: &[f32], t: usize, c: usize, h: usize) -> Vec<f32> {
    let hd = c / h;
    let mut out = vec![0.0f32; t * c];
    for tt in 0..t {
        for hh in 0..h {
            for d in 0..hd {
                out[tt * c + hh * hd + d] = x[hh * t * hd + tt * hd + d];
            }
        }
    }
    out
}

// ---- backward / training primitives (roadmap v4, Stages 8-9) ----

/// d/dx of GELU (tanh approximation), applied to upstream dy.
pub fn gelu_bwd(x: &[f32], dy: &[f32]) -> Vec<f32> {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    const A: f32 = 0.044715;
    x.iter()
        .zip(dy)
        .map(|(&v, &g)| {
            let u = C * (v + A * v * v * v);
            let th = u.tanh();
            let sech2 = 1.0 - th * th;
            let d = 0.5 * (1.0 + th) + 0.5 * v * sech2 * C * (1.0 + 3.0 * A * v * v);
            d * g
        })
        .collect()
}

/// Softmax backward: dx = y * (dy - sum(y * dy)) per row. The causal-masked
/// forward writes exact zeros at masked entries, so no mask is needed here.
pub fn softmax_bwd(y: &[f32], dy: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let base = r * cols;
        let s: f32 = (0..cols).map(|j| y[base + j] * dy[base + j]).sum();
        for j in 0..cols {
            out[base + j] = y[base + j] * (dy[base + j] - s);
        }
    }
    out
}

/// LayerNorm backward for x: returns dx; gamma/beta gradients are computed
/// by `layernorm_bwd_dparams`.
pub fn layernorm_bwd_dx(
    x: &[f32],
    gamma: &[f32],
    dy: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let nf = cols as f32;
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean: f32 = row.iter().sum::<f32>() / nf;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / nf;
        let inv_std = 1.0 / (var + eps).sqrt();
        let mut s1 = 0.0f32; // sum(gamma * dy)
        let mut s2 = 0.0f32; // sum(gamma * dy * xhat)
        for j in 0..cols {
            let gd = gamma[j] * dy[r * cols + j];
            s1 += gd;
            s2 += gd * (row[j] - mean) * inv_std;
        }
        for j in 0..cols {
            let xhat = (row[j] - mean) * inv_std;
            let gd = gamma[j] * dy[r * cols + j];
            out[r * cols + j] = inv_std * (gd - s1 / nf - xhat * s2 / nf);
        }
    }
    out
}

/// LayerNorm gamma/beta gradients: dgamma = sum_r dy * xhat, dbeta = sum_r dy.
pub fn layernorm_bwd_dparams(
    x: &[f32],
    dy: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let nf = cols as f32;
    let mut dgamma = vec![0.0f32; cols];
    let mut dbeta = vec![0.0f32; cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean: f32 = row.iter().sum::<f32>() / nf;
        let var: f32 = row.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / nf;
        let inv_std = 1.0 / (var + eps).sqrt();
        for j in 0..cols {
            let d = dy[r * cols + j];
            dgamma[j] += d * (row[j] - mean) * inv_std;
            dbeta[j] += d;
        }
    }
    (dgamma, dbeta)
}

/// Column sums: [rows, cols] -> [cols]. Bias gradients.
pub fn sum_rows(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    for r in 0..rows {
        for j in 0..cols {
            out[j] += x[r * cols + j];
        }
    }
    out
}

/// dwte scatter-add: dst[ids[r]] += src[r] for rows of width c.
pub fn scatter_add_rows(dst: &mut [f32], ids: &[u32], src: &[f32], c: usize) {
    for (r, &id) in ids.iter().enumerate() {
        let d0 = id as usize * c;
        for j in 0..c {
            dst[d0 + j] += src[r * c + j];
        }
    }
}

/// -log(probs[r, ids[r]]) per row.
pub fn gather_nll(probs: &[f32], ids: &[u32], cols: usize) -> Vec<f32> {
    ids.iter()
        .enumerate()
        .map(|(r, &id)| -probs[r * cols + id as usize].max(f32::MIN_POSITIVE).ln())
        .collect()
}

/// Cross-entropy backward: dlogits = (probs - onehot(ids)) * scale.
pub fn ce_bwd(probs: &[f32], ids: &[u32], rows: usize, cols: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for j in 0..cols {
            let onehot = (j as u32 == ids[r]) as u32 as f32;
            out[r * cols + j] = (probs[r * cols + j] - onehot) * scale;
        }
    }
    out
}

/// PCG output hash — identical to shaders/dropout.wgsl so masks match
/// bit-for-bit across backends.
pub fn pcg_hash(x: u32) -> u32 {
    let state = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

/// Inverted dropout with a counter-based RNG: element i is kept when
/// hash(seed, i) maps to u >= p, and scaled by `scale` (the caller-computed
/// 1/(1-p)). Applying the same (seed, p, scale) to dy gives the backward
/// pass. `scale` is passed in (rather than recomputed here) so the CPU and
/// WGSL kernel multiply by the exact same bits — see ops::dropout.
pub fn dropout(x: &[f32], p: f32, scale: f32, seed: u32) -> Vec<f32> {
    x.iter()
        .enumerate()
        .map(|(i, &v)| {
            let r = pcg_hash(seed ^ (i as u32).wrapping_mul(0x9E37_79B9));
            let u = (r >> 8) as f32 / 16_777_216.0; // [0, 1)
            if u >= p { v * scale } else { 0.0 }
        })
        .collect()
}

/// split_heads backward for one of q/k/v: place d [h, t, hd] into the
/// `which` third of a zeroed [t, 3c] buffer.
pub fn unsplit_head(d: &[f32], t: usize, c: usize, h: usize, which: usize) -> Vec<f32> {
    let hd = c / h;
    let mut out = vec![0.0f32; t * 3 * c];
    for hh in 0..h {
        for tt in 0..t {
            for dd in 0..hd {
                out[tt * 3 * c + which * c + hh * hd + dd] = d[hh * t * hd + tt * hd + dd];
            }
        }
    }
    out
}

/// merge_heads backward: dy [t, c] -> [h, t, hd].
pub fn unmerge_heads(dy: &[f32], t: usize, c: usize, h: usize) -> Vec<f32> {
    let hd = c / h;
    let mut out = vec![0.0f32; t * c];
    for hh in 0..h {
        for tt in 0..t {
            for dd in 0..hd {
                out[hh * t * hd + tt * hd + dd] = dy[tt * c + hh * hd + dd];
            }
        }
    }
    out
}

/// AdamW step (decoupled weight decay), in place. `step` is 1-based.
#[allow(clippy::too_many_arguments)]
pub fn adamw(
    param: &mut [f32],
    grad: &[f32],
    m: &mut [f32],
    v: &mut [f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    for i in 0..param.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        param[i] -= lr * (mhat / (vhat.sqrt() + eps) + weight_decay * param[i]);
    }
}
