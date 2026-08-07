// Tall-skinny matmul: the m <= GEMV_MAX_M case that `matmul.wgsl` cannot serve.
//
// KV-cached decode runs every projection at m = 1 — one token, one row. The
// tiled GEMM claims a 64-row output block per workgroup, so at m = 1 it
// discards 63/64 of the arithmetic it schedules, and it measured 13 GFLOP/s
// there.
//
// Two things make a tall-skinny shape hard, and this kernel answers both.
//
// It is bandwidth bound, not compute bound: B must be read whole and each
// element used once. So B is read exactly once no matter how many rows there
// are — a thread owns one *column* for the life of the kernel and carries one
// accumulator per row, while A's slab sits in workgroup memory. The obvious
// alternative, a row loop around a matvec, re-reads B m times; it measured
// exactly that, 8 rows costing 8x one row once B outgrew the L2.
//
// And the shape supplies too little parallelism: a 384-column projection is six
// 64-column blocks, six workgroups on a 64-SM card, which is why the GEMM
// bottoms out around 70 us for every m from 12 to 128. So k is *split* —
// `nsplit` workgroups each reduce a slice of k, and when there is more than one
// they write partial sums that `gemv_reduce.wgsl` adds up. The host picks
// nsplit (`backend::wgpu::ops::gemv_split`) to land near 256 workgroups.
//
// Layout matches matmul.wgsl exactly — same transposes, same n_off/n_out
// column-chunked output, same bias and alpha — because op_parity.rs holds the
// two to the same CPU reference.

struct Params {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    a_stride: u32,
    b_stride: u32,
    trans_b: u32,
    has_bias: u32,
    alpha: f32,
    n_off: u32,
    n_out: u32,
    trans_a: u32,
    nsplit: u32,
    nrowblk: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const COLS: u32 = 64u;  // columns per workgroup = threads per workgroup
const MROWS: u32 = 16u; // rows one workgroup carries; must equal GEMV_ROWS
const KT: u32 = 64u;    // k staged per pass

var<workgroup> asub: array<f32, 1024>; // MROWS * KT

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wid.x * COLS + lid.x;
    // y carries (split, row block). More than one row block means B is read
    // once per block rather than once outright, which is why the host only
    // reaches for them while that is still cheaper than the GEMM's idle SMs.
    let split = wid.y / p.nrowblk;
    let row_off = (wid.y % p.nrowblk) * MROWS;
    let bat = wid.z;
    let a_base = bat * p.a_stride;
    let b_base = bat * p.b_stride;

    let per = (p.k + p.nsplit - 1u) / p.nsplit;
    let k0 = split * per;
    let k1 = min(k0 + per, p.k);

    // Row-major B: consecutive threads read consecutive columns of one k-row,
    // which is the coalesced direction. Transposed B walks each thread down a
    // contiguous column instead — the layout attention's K^T arrives in, and
    // still one stream per thread rather than a gather.
    let b_row = select(p.n, 1u, p.trans_b == 1u);
    let b_col = select(1u, p.k, p.trans_b == 1u);

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    // Decode is m = 1, and there the staging below is all cost and no benefit:
    // one row's A slice is a single value per k, already in L1, and staging it
    // buys nothing while the index arithmetic and two barriers per pass cost
    // 0.4 ms/token. So four rows or fewer read A straight from global.
    if (p.m <= 4u) {
        for (var gk = k0; gk < k1; gk = gk + 1u) {
            var bv = 0.0;
            if (col < p.n) {
                bv = b[b_base + gk * b_row + col * b_col];
            }
            var av = vec4<f32>(aval(a_base, 0u, gk), 0.0, 0.0, 0.0);
            if (p.m > 1u) {
                av.y = aval(a_base, 1u, gk);
            }
            if (p.m > 2u) {
                av.z = aval(a_base, 2u, gk);
            }
            if (p.m > 3u) {
                av.w = aval(a_base, 3u, gk);
            }
            acc0 = fma(av, vec4<f32>(bv), acc0);
        }
        if (col < p.n) {
            store4(0u, col, bat, split, acc0);
        }
        return;
    }

    // `kt`, `k1` and `p.m` are workgroup-uniform, so every barrier below is
    // reached by every invocation — including the ones whose column is past n,
    // which is why an out-of-range column clamps its loads instead of returning.
    // Stage only the rows this block actually has, rounded up to the 4 that an
    // accumulator covers.
    let rows_here = min(MROWS, p.m - row_off);
    let staged = (rows_here + 3u) / 4u * 4u;

    var kt = k0;
    while (kt < k1) {
        for (var i = 0u; i < staged * KT / COLS; i = i + 1u) {
            let idx = lid.x + i * COLS;
            let r = row_off + idx / KT;
            let gk = kt + idx % KT;
            var v = 0.0;
            if (r < p.m && gk < p.k) {
                if (p.trans_a == 0u) {
                    v = a[a_base + r * p.k + gk];
                } else {
                    v = a[a_base + gk * p.m + r];
                }
            }
            asub[idx] = v;
        }
        workgroupBarrier();

        let kend = min(KT, k1 - kt);
        for (var kk = 0u; kk < kend; kk = kk + 1u) {
            var bv = 0.0;
            if (col < p.n) {
                bv = b[b_base + (kt + kk) * b_row + col * b_col];
            }
            let bb = vec4<f32>(bv);
            // Every thread reads the same asub element, so these broadcast
            // rather than conflict. The guards are uniform: no divergence.
            acc0 = fma(row4(0u, kk), bb, acc0);
            if (p.m > row_off + 4u) {
                acc1 = fma(row4(4u, kk), bb, acc1);
            }
            if (p.m > row_off + 8u) {
                acc2 = fma(row4(8u, kk), bb, acc2);
            }
            if (p.m > row_off + 12u) {
                acc3 = fma(row4(12u, kk), bb, acc3);
            }
        }
        workgroupBarrier();
        kt = kt + KT;
    }

    if (col >= p.n) {
        return;
    }
    store4(row_off + 0u, col, bat, split, acc0);
    store4(row_off + 4u, col, bat, split, acc1);
    store4(row_off + 8u, col, bat, split, acc2);
    store4(row_off + 12u, col, bat, split, acc3);
}

/// A[row, gk], straight from global — the m <= 4 path's read.
fn aval(a_base: u32, row: u32, gk: u32) -> f32 {
    if (p.trans_a == 0u) {
        return a[a_base + row * p.k + gk];
    }
    return a[a_base + gk * p.m + row];
}

/// Four consecutive rows of the staged A slab at depth `kk`.
fn row4(r0: u32, kk: u32) -> vec4<f32> {
    return vec4<f32>(
        asub[(r0 + 0u) * KT + kk],
        asub[(r0 + 1u) * KT + kk],
        asub[(r0 + 2u) * KT + kk],
        asub[(r0 + 3u) * KT + kk],
    );
}

fn store4(r0: u32, col: u32, bat: u32, split: u32, acc: vec4<f32>) {
    for (var j = 0u; j < 4u; j = j + 1u) {
        let row = r0 + j;
        if (row >= p.m) {
            return;
        }
        if (p.nsplit == 1u) {
            var v = acc[j] * p.alpha;
            if (p.has_bias == 1u) {
                v = v + bias[col];
            }
            out[bat * p.m * p.n_out + row * p.n_out + p.n_off + col] = v;
        } else {
            // Partials, [nsplit][batch][m][n]. alpha and bias are the reduce
            // kernel's job — applying them per split would apply them nsplit
            // times.
            out[((split * p.batch + bat) * p.m + row) * p.n + col] = acc[j];
        }
    }
}
