//! GPU op wrappers: bind tensors, compute grid sizes, dispatch WGSL kernels.
//! Semantics mirror `backend::cpu` exactly; see the shader sources for the
//! kernel-side contracts.

use std::sync::Arc;

use super::{WgpuContext, linear_grid};
use crate::tensor::WgpuStorage;

pub fn alloc_storage(ctx: &Arc<WgpuContext>, numel: usize) -> WgpuStorage {
    alloc(ctx, numel)
}

fn alloc(ctx: &Arc<WgpuContext>, numel: usize) -> WgpuStorage {
    WgpuStorage {
        ctx: ctx.clone(),
        buf: Arc::new(ctx.create_storage(numel * 4)),
        offset: 0,
    }
}

fn bind(s: &WgpuStorage, numel: usize) -> (&wgpu::Buffer, usize, usize) {
    (&s.buf, s.offset * 4, numel * 4)
}

pub fn add(a: &WgpuStorage, b: &WgpuStorage, n: usize) -> WgpuStorage {
    let out = alloc(&a.ctx, n);
    a.ctx.dispatch(
        "add",
        &[n as u32, 0, 0, 0],
        &[bind(a, n), bind(b, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

pub fn gelu(x: &WgpuStorage, n: usize) -> WgpuStorage {
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "gelu",
        &[n as u32, 0, 0, 0],
        &[bind(x, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

#[allow(clippy::too_many_arguments)]
pub fn matmul(
    a: &WgpuStorage,
    b: &WgpuStorage,
    bias: Option<&WgpuStorage>,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    a_stride: usize,
    b_stride: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
) -> WgpuStorage {
    let out = alloc(&a.ctx, batch * m * n);
    matmul_into(
        &out, a, b, bias, m, k, n, batch, a_stride, b_stride, trans_a, trans_b, alpha, 0, n,
    );
    out
}

/// Compute `n` columns of a wider [m, n_out] output, writing at column
/// offset `n_off` — used to keep each weight-chunk binding under
/// max_storage_buffer_binding_size.
#[allow(clippy::too_many_arguments)]
pub fn matmul_into(
    out: &WgpuStorage,
    a: &WgpuStorage,
    b: &WgpuStorage,
    bias: Option<&WgpuStorage>,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    a_stride: usize,
    b_stride: usize,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    n_off: usize,
    n_out: usize,
) {
    let ctx = &a.ctx;
    // The bias binding must exist even when unused; bind a 1-element dummy.
    let dummy;
    let bias_binding = match bias {
        Some(bias) => bind(bias, n),
        None => {
            dummy = alloc(ctx, 1);
            bind(&dummy, 1)
        }
    };
    let a_numel = if a_stride == 0 {
        m * k
    } else {
        batch * a_stride
    };
    let b_numel = if b_stride == 0 {
        // Physical element count of one B; with b_rows the logical (k or n)
        // undercounts, but an unbatched B is always bound whole below.
        k * n
    } else {
        batch * b_stride
    };
    let params = [
        m as u32,
        k as u32,
        n as u32,
        batch as u32,
        a_stride as u32,
        b_stride as u32,
        trans_b as u32,
        bias.is_some() as u32,
        alpha.to_bits(),
        n_off as u32,
        n_out as u32,
        trans_a as u32,
    ];
    ctx.dispatch(
        "matmul",
        &params,
        &[
            bind(a, a_numel),
            bind(b, b_numel),
            bias_binding,
            bind(out, batch * m * n_out),
        ],
        (n.div_ceil(16) as u32, m.div_ceil(16) as u32, batch as u32),
    );
}

pub fn softmax(
    x: &WgpuStorage,
    rows: usize,
    cols: usize,
    q_len: usize,
    causal: bool,
    off: usize,
) -> WgpuStorage {
    let n = rows * cols;
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "softmax",
        &[
            rows as u32,
            cols as u32,
            q_len as u32,
            causal as u32,
            off as u32,
            0,
            0,
            0,
        ],
        &[bind(x, n), bind(&out, n)],
        (rows as u32, 1, 1),
    );
    out
}

pub fn layernorm(
    x: &WgpuStorage,
    gamma: &WgpuStorage,
    beta: &WgpuStorage,
    rows: usize,
    cols: usize,
    eps: f32,
) -> WgpuStorage {
    let n = rows * cols;
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "layernorm",
        &[rows as u32, cols as u32, eps.to_bits(), 0],
        &[
            bind(x, n),
            bind(gamma, cols),
            bind(beta, cols),
            bind(&out, n),
        ],
        (rows as u32, 1, 1),
    );
    out
}

/// Gather from row-chunked embedding tables. `chunks[i]` holds rows
/// [i * chunk_rows, ...) of the full table; each token is written by exactly
/// the dispatch whose chunk owns its id.
#[allow(clippy::too_many_arguments)]
pub fn embedding(
    ids: &WgpuStorage,
    chunks: &[(&WgpuStorage, usize)], // (buffer, rows in this chunk)
    chunk_rows: usize,
    wpe: Option<&WgpuStorage>,
    t: usize,
    c: usize,
    n_ctx: usize,
    pos: usize,
) -> WgpuStorage {
    let ctx = &ids.ctx;
    let n = t * c;
    let out = alloc(ctx, n);
    let dummy;
    let wpe_binding = match wpe {
        Some(wpe) => bind(wpe, n_ctx * c),
        None => {
            dummy = alloc(ctx, 1);
            bind(&dummy, 1)
        }
    };
    for (i, (chunk, rows)) in chunks.iter().enumerate() {
        let row_start = i * chunk_rows;
        ctx.dispatch(
            "embedding",
            &[
                t as u32,
                c as u32,
                pos as u32,
                wpe.is_some() as u32,
                row_start as u32,
                (row_start + rows) as u32,
                0,
                0,
            ],
            &[
                bind(ids, t),
                bind(chunk, rows * c),
                wpe_binding,
                bind(&out, n),
            ],
            linear_grid(n),
        );
    }
    out
}

/// Append src [h, t, hd] into cache [h, cap, hd] at row offset `len`.
pub fn kv_append(
    cache: &WgpuStorage,
    src: &WgpuStorage,
    h: usize,
    t: usize,
    hd: usize,
    cap: usize,
    len: usize,
) {
    let n = h * t * hd;
    src.ctx.dispatch(
        "kv_append",
        &[
            h as u32, t as u32, hd as u32, cap as u32, len as u32, 0, 0, 0,
        ],
        &[bind(src, n), bind(cache, h * cap * hd)],
        linear_grid(n),
    );
}

pub fn split_heads(
    qkv: &WgpuStorage,
    t: usize,
    c: usize,
    h: usize,
) -> (WgpuStorage, WgpuStorage, WgpuStorage) {
    let ctx = &qkv.ctx;
    let n = t * c;
    let (q, k, v) = (alloc(ctx, n), alloc(ctx, n), alloc(ctx, n));
    ctx.dispatch(
        "split_heads",
        &[t as u32, c as u32, h as u32, 0],
        &[bind(qkv, t * 3 * c), bind(&q, n), bind(&k, n), bind(&v, n)],
        linear_grid(n),
    );
    (q, k, v)
}

pub fn merge_heads(x: &WgpuStorage, t: usize, c: usize, h: usize) -> WgpuStorage {
    let n = t * c;
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "merge_heads",
        &[t as u32, c as u32, h as u32, 0],
        &[bind(x, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

// ---- backward / training kernels (roadmap v4, Stages 8-9) ----

pub fn gelu_bwd(x: &WgpuStorage, dy: &WgpuStorage, n: usize) -> WgpuStorage {
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "gelu_bwd",
        &[n as u32, 0, 0, 0],
        &[bind(x, n), bind(dy, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

pub fn softmax_bwd(y: &WgpuStorage, dy: &WgpuStorage, rows: usize, cols: usize) -> WgpuStorage {
    let n = rows * cols;
    let out = alloc(&y.ctx, n);
    y.ctx.dispatch(
        "softmax_bwd",
        &[rows as u32, cols as u32, 0, 0],
        &[bind(y, n), bind(dy, n), bind(&out, n)],
        (rows as u32, 1, 1),
    );
    out
}

pub fn layernorm_bwd(
    x: &WgpuStorage,
    gamma: &WgpuStorage,
    dy: &WgpuStorage,
    rows: usize,
    cols: usize,
    eps: f32,
) -> (WgpuStorage, WgpuStorage, WgpuStorage) {
    let n = rows * cols;
    let ctx = &x.ctx;
    let dx = alloc(ctx, n);
    let stats = alloc(ctx, rows * 2);
    ctx.dispatch(
        "layernorm_bwd_dx",
        &[rows as u32, cols as u32, eps.to_bits(), 0],
        &[
            bind(x, n),
            bind(gamma, cols),
            bind(dy, n),
            bind(&dx, n),
            bind(&stats, rows * 2),
        ],
        (rows as u32, 1, 1),
    );
    let dgamma = alloc(ctx, cols);
    let dbeta = alloc(ctx, cols);
    ctx.dispatch(
        "layernorm_bwd_dp",
        &[rows as u32, cols as u32, 0, 0],
        &[
            bind(x, n),
            bind(dy, n),
            bind(&stats, rows * 2),
            bind(&dgamma, cols),
            bind(&dbeta, cols),
        ],
        linear_grid(cols),
    );
    (dx, dgamma, dbeta)
}

pub fn sum_rows(x: &WgpuStorage, rows: usize, cols: usize) -> WgpuStorage {
    let out = alloc(&x.ctx, cols);
    x.ctx.dispatch(
        "sum_rows",
        &[rows as u32, cols as u32, 0, 0],
        &[bind(x, rows * cols), bind(&out, cols)],
        linear_grid(cols),
    );
    out
}

/// dst[ids[r]] += src[r], in place (CAS-loop f32 atomics).
pub fn scatter_add_rows(
    dst: &WgpuStorage,
    ids: &WgpuStorage,
    src: &WgpuStorage,
    t: usize,
    c: usize,
    dst_numel: usize,
) {
    src.ctx.dispatch(
        "scatter_add",
        &[t as u32, c as u32, 0, 0],
        &[bind(ids, t), bind(src, t * c), bind(dst, dst_numel)],
        linear_grid(t * c),
    );
}

pub fn gather_nll(
    probs: &WgpuStorage,
    ids: &WgpuStorage,
    rows: usize,
    cols: usize,
) -> WgpuStorage {
    let out = alloc(&probs.ctx, rows);
    probs.ctx.dispatch(
        "gather_nll",
        &[rows as u32, cols as u32, 0, 0],
        &[bind(probs, rows * cols), bind(ids, rows), bind(&out, rows)],
        linear_grid(rows),
    );
    out
}

pub fn ce_bwd(
    probs: &WgpuStorage,
    ids: &WgpuStorage,
    rows: usize,
    cols: usize,
    scale: f32,
) -> WgpuStorage {
    let n = rows * cols;
    let out = alloc(&probs.ctx, n);
    probs.ctx.dispatch(
        "ce_bwd",
        &[rows as u32, cols as u32, scale.to_bits(), 0],
        &[bind(probs, n), bind(ids, rows), bind(&out, n)],
        linear_grid(n),
    );
    out
}

pub fn dropout(x: &WgpuStorage, n: usize, p: f32, scale: f32, seed: u32) -> WgpuStorage {
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "dropout",
        &[n as u32, seed, p.to_bits(), scale.to_bits()],
        &[bind(x, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

pub fn unsplit_head(d: &WgpuStorage, t: usize, c: usize, h: usize, which: usize) -> WgpuStorage {
    let n = t * c;
    // Fresh wgpu buffers are zero-initialized; the kernel writes one third.
    let out = alloc(&d.ctx, t * 3 * c);
    d.ctx.dispatch(
        "unsplit_heads",
        &[t as u32, c as u32, h as u32, which as u32],
        &[bind(d, n), bind(&out, t * 3 * c)],
        linear_grid(n),
    );
    out
}

pub fn unmerge_heads(dy: &WgpuStorage, t: usize, c: usize, h: usize) -> WgpuStorage {
    let n = t * c;
    let out = alloc(&dy.ctx, n);
    dy.ctx.dispatch(
        "unmerge_heads",
        &[t as u32, c as u32, h as u32, 0],
        &[bind(dy, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

/// Per-workgroup partial sums of squares; the caller reads back and sums.
/// The partial count covers the whole (possibly 2-D) dispatch grid; groups
/// past the data contribute zeros.
pub fn sumsq_partials(x: &WgpuStorage, n: usize) -> (WgpuStorage, usize) {
    let grid = linear_grid(n);
    let groups = (grid.0 * grid.1) as usize;
    let out = alloc(&x.ctx, groups);
    x.ctx.dispatch(
        "sumsq",
        &[n as u32, 0, 0, 0],
        &[bind(x, n), bind(&out, groups)],
        grid,
    );
    (out, groups)
}

pub fn scale(x: &WgpuStorage, n: usize, alpha: f32) -> WgpuStorage {
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "scale",
        &[n as u32, alpha.to_bits(), 0, 0],
        &[bind(x, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

/// AdamW step, updating param/m/v in place.
#[allow(clippy::too_many_arguments)]
pub fn adamw(
    param: &WgpuStorage,
    grad: &WgpuStorage,
    m: &WgpuStorage,
    v: &WgpuStorage,
    n: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u32,
) {
    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);
    param.ctx.dispatch(
        "adamw",
        &[
            n as u32,
            lr.to_bits(),
            beta1.to_bits(),
            beta2.to_bits(),
            eps.to_bits(),
            weight_decay.to_bits(),
            bc1.to_bits(),
            bc2.to_bits(),
        ],
        &[bind(grad, n), bind(param, n), bind(m, n), bind(v, n)],
        linear_grid(n),
    );
}
