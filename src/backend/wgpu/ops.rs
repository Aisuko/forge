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
        buf: Arc::new(ctx.create_pooled(numel * 4)),
        offset: 0,
    }
}

/// An output buffer guaranteed to be all zeros.
///
/// `alloc` recycles, and a recycled buffer holds whatever the last op left in
/// it. Only for kernels that write part of their output and need the remainder
/// zero — see `unsplit_head`, the sole caller.
fn alloc_zeroed(ctx: &Arc<WgpuContext>, numel: usize) -> WgpuStorage {
    WgpuStorage {
        ctx: ctx.clone(),
        buf: Arc::new(ctx.create_zeroed(numel * 4)),
        offset: 0,
    }
}

fn bind(s: &WgpuStorage, numel: usize) -> (&wgpu::Buffer, usize, usize) {
    (&s.buf, s.offset * 4, numel * 4)
}

/// A grid of one workgroup per row, folded into two dimensions.
///
/// The row-per-workgroup kernels — softmax, layernorm, and their backwards —
/// used a flat `(rows, 1, 1)`. A batched training step overruns it: 64
/// sequences x 6 heads x 256 query positions is 98304 attention rows against a
/// 65535 per-dimension limit, and wgpu rejects the dispatch. The kernels
/// recover the row as `wid.y * nwg.x + wid.x` and bound-check it, so the
/// rounding up here is harmless.
fn row_grid(rows: usize) -> (u32, u32, u32) {
    if rows <= 65535 {
        (rows.max(1) as u32, 1, 1)
    } else {
        (65535, rows.div_ceil(65535) as u32, 1)
    }
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
    if use_gemv(m, n, batch) {
        gemv_into(
            out,
            a,
            b,
            bias_binding,
            a_numel,
            b_numel,
            &params,
            m,
            k,
            n,
            batch,
            alpha,
            n_off,
            n_out,
        );
        return;
    }
    ctx.dispatch(
        "matmul",
        &params,
        &[
            bind(a, a_numel),
            bind(b, b_numel),
            bias_binding,
            bind(out, batch * m * n_out),
        ],
        // One workgroup per 64x64 output block — see shaders/matmul.wgsl.
        (n.div_ceil(64) as u32, m.div_ceil(64) as u32, batch as u32),
    );
}

/// Rows one `gemv` workgroup carries — must match `MROWS` in gemv.wgsl.
const GEMV_ROWS: usize = 16;

/// Which of the two matmul kernels this shape goes to.
///
/// Measured, not assumed. Both branches are the same choice seen twice: the
/// tiled GEMM claims a 64x64 output block per workgroup, so a shape that does
/// not have ~64 such blocks leaves most of an RTX A5000 idle and the GEMM
/// flattens onto a floor — a 1536x384 projection costs the same ~290 us at
/// m = 24 as at m = 256. `gemv` splits k instead, manufacturing workgroups the
/// shape does not have, and pays one extra pass over B per 16 rows.
///
/// So: take `gemv` when the rows alone are too few (decode is m = 1), and take
/// it when the whole grid is too small. Stop at 16 row blocks, past which those
/// extra passes over B cost more than the idle SMs did.
fn use_gemv(m: usize, n: usize, batch: usize) -> bool {
    if m <= 4 * GEMV_ROWS {
        return true;
    }
    let gemm_groups = n.div_ceil(64) * m.div_ceil(64) * batch;
    gemm_groups < GEMV_TARGET_GROUPS / 4 && m <= 16 * GEMV_ROWS
}

/// Columns one `gemv` workgroup covers — must match `COLS` in gemv.wgsl.
const GEMV_COLS: usize = 64;

/// Shallowest k-slice worth giving a split its own workgroup.
const GEMV_MIN_SPLIT_K: usize = 32;

/// How many workgroups to aim a split-k matvec at. Around 4x an RTX A5000's 64
/// SMs, which is enough to cover the tail without making the reduction pass
/// wider than the work it saves.
const GEMV_TARGET_GROUPS: usize = 256;

/// How many ways to split k for a matvec of this shape.
///
/// The column blocks alone are the natural parallelism, and for a narrow output
/// there are far too few of them — a 384-column projection is six. This tops
/// them up to [`GEMV_TARGET_GROUPS`]. Row blocks count toward the total for the
/// same reason they cost a pass over B: they are already real workgroups.
fn gemv_split(m: usize, k: usize, n: usize, batch: usize) -> usize {
    let blocks = n.div_ceil(GEMV_COLS) * m.div_ceil(GEMV_ROWS) * batch;
    let want = GEMV_TARGET_GROUPS.div_ceil(blocks.max(1));
    want.clamp(1, (k / GEMV_MIN_SPLIT_K).max(1))
}

#[allow(clippy::too_many_arguments)]
fn gemv_into(
    out: &WgpuStorage,
    a: &WgpuStorage,
    b: &WgpuStorage,
    bias_binding: (&wgpu::Buffer, usize, usize),
    a_numel: usize,
    b_numel: usize,
    matmul_params: &[u32; 12],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    alpha: f32,
    n_off: usize,
    n_out: usize,
) {
    let ctx = &a.ctx;
    let nsplit = gemv_split(m, k, n, batch);
    let nrowblk = m.div_ceil(GEMV_ROWS);
    let mut params = [0u32; 16];
    params[..12].copy_from_slice(matmul_params);
    params[12] = nsplit as u32;
    params[13] = nrowblk as u32;

    let grid = (
        n.div_ceil(GEMV_COLS) as u32,
        (nsplit * nrowblk) as u32,
        batch as u32,
    );
    if nsplit == 1 {
        ctx.dispatch(
            "gemv",
            &params,
            &[
                bind(a, a_numel),
                bind(b, b_numel),
                bias_binding,
                bind(out, batch * m * n_out),
            ],
            grid,
        );
        return;
    }

    // Partials are [nsplit][batch][m][n] — packed at stride n, not n_out: a
    // column-chunked head writes its chunk contiguously here and only spreads
    // out across n_out in the reduction.
    let per_split = batch * m * n;
    let partials = alloc(ctx, nsplit * per_split);
    ctx.dispatch(
        "gemv",
        &params,
        &[
            bind(a, a_numel),
            bind(b, b_numel),
            bias_binding,
            bind(&partials, nsplit * per_split),
        ],
        grid,
    );
    let rparams = [
        m as u32,
        n as u32,
        batch as u32,
        nsplit as u32,
        matmul_params[7], // has_bias
        alpha.to_bits(),
        n_off as u32,
        n_out as u32,
    ];
    ctx.dispatch(
        "gemv_reduce",
        &rparams,
        &[
            bind(&partials, nsplit * per_split),
            bias_binding,
            bind(out, batch * m * n_out),
        ],
        linear_grid(per_split),
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
        row_grid(rows),
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
        row_grid(rows),
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
    seq: usize,
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
                seq as u32,
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
    b: usize,
    t: usize,
    c: usize,
    h: usize,
) -> (WgpuStorage, WgpuStorage, WgpuStorage) {
    let ctx = &qkv.ctx;
    let n = b * t * c;
    let (q, k, v) = (alloc(ctx, n), alloc(ctx, n), alloc(ctx, n));
    ctx.dispatch(
        "split_heads",
        &[t as u32, c as u32, h as u32, b as u32],
        &[
            bind(qkv, b * t * 3 * c),
            bind(&q, n),
            bind(&k, n),
            bind(&v, n),
        ],
        linear_grid(n),
    );
    (q, k, v)
}

pub fn merge_heads(x: &WgpuStorage, b: usize, t: usize, c: usize, h: usize) -> WgpuStorage {
    let n = b * t * c;
    let out = alloc(&x.ctx, n);
    x.ctx.dispatch(
        "merge_heads",
        &[t as u32, c as u32, h as u32, b as u32],
        &[bind(x, n), bind(&out, n)],
        linear_grid(n),
    );
    out
}

// ---- backward / training kernels ----

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
        row_grid(rows),
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
        row_grid(rows),
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

/// `dst[ids[r]] += src[r]`, in place (CAS-loop f32 atomics).
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

pub fn gather_nll(probs: &WgpuStorage, ids: &WgpuStorage, rows: usize, cols: usize) -> WgpuStorage {
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

pub fn unsplit_head(
    d: &WgpuStorage,
    b: usize,
    t: usize,
    c: usize,
    h: usize,
    which: usize,
) -> WgpuStorage {
    let n = b * t * c;
    // The kernel writes one third and needs the other two to be zero, so this
    // is the one op that must not take a recycled buffer. `alloc` here silently
    // corrupts wte gradients — it did, before `alloc_zeroed` existed.
    let out = alloc_zeroed(&d.ctx, b * t * 3 * c);
    d.ctx.dispatch(
        "unsplit_heads",
        &[
            t as u32,
            c as u32,
            h as u32,
            which as u32,
            b as u32,
            0,
            0,
            0,
        ],
        &[bind(d, n), bind(&out, b * t * 3 * c)],
        linear_grid(n),
    );
    out
}

pub fn unmerge_heads(dy: &WgpuStorage, b: usize, t: usize, c: usize, h: usize) -> WgpuStorage {
    let n = b * t * c;
    let out = alloc(&dy.ctx, n);
    dy.ctx.dispatch(
        "unmerge_heads",
        &[t as u32, c as u32, h as u32, b as u32],
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
