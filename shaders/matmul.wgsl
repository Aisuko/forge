// Tiled batched matmul: C[b] = alpha * A[b] @ B[b] (+ bias, broadcast over rows).
//
// A: [batch, m, k], or [batch, k, m] if trans_a == 1
//    (a_stride per batch, or 0 to broadcast one A)
// B: [batch, k, n] if trans_b == 0, [batch, n, k] if trans_b == 1
//    (b_stride per batch, or 0 to broadcast one B)
// C: [batch, m, n]
// bias: [n], added when has_bias == 1 (bind a 1-element dummy otherwise).
//
// Register-tiled: a 64x64 output block per workgroup, 4x4 outputs per thread,
// accumulated over 16-deep k-tiles staged in workgroup memory.
//
// The tile shape is what the arithmetic costs. The predecessor gave each
// thread one output, so every fused multiply-add needed two workgroup-memory
// reads — a ratio that pins the kernel to shared-memory bandwidth, and it
// measured 1.1 TFLOP/s against an RTX A5000's 27.8 TFLOP/s fp32 peak. Holding
// a 4x4 block in registers instead amortizes 8 reads over 16 FMAs, a 4x better
// ratio, and the reads are `vec4`-shaped so they issue as 128-bit loads.
//
// Sizes: asub is 64x17 and bsub 16x64 f32 = 8.5 KiB, inside the 16 KiB
// workgroup-storage floor that WebGPU guarantees everywhere. The 17 is
// deliberate padding — at a stride of 16 the four rows a thread reads land in
// one bank, and every warp serializes on the conflict.

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
    // Column-chunked output: this dispatch computes n columns of a wider
    // [m, n_out] matrix, writing at column offset n_off.
    n_off: u32,
    n_out: u32,
    trans_a: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> c: array<f32>;

const BM: u32 = 64u; // output rows per workgroup
const BN: u32 = 64u; // output cols per workgroup
const BK: u32 = 16u; // k-tile depth
const TM: u32 = 4u;  // output rows per thread
const TN: u32 = 4u;  // output cols per thread
const AS: u32 = 17u; // asub row stride: BK + 1, to break the bank conflict
const THREADS: u32 = 256u;

var<workgroup> asub: array<f32, 1088>; // BM * AS
var<workgroup> bsub: array<f32, 1024>; // BK * BN

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bat = wid.z;
    let a_base = bat * p.a_stride;
    let b_base = bat * p.b_stride;
    let row0 = wid.y * BM;
    let col0 = wid.x * BN;
    let tid = lid.y * 16u + lid.x;

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let ntiles = (p.k + BK - 1u) / BK;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let kt = t * BK;

        // ── stage A's tile. Both index maps walk `tid` along the axis that is
        // contiguous in memory for that layout, so the global loads coalesce
        // either way round.
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            var r: u32;
            var kk: u32;
            if (p.trans_a == 0u) {
                r = idx / BK;
                kk = idx % BK;
            } else {
                kk = idx / BM;
                r = idx % BM;
            }
            let grow = row0 + r;
            let gk = kt + kk;
            var v = 0.0;
            if (grow < p.m && gk < p.k) {
                if (p.trans_a == 0u) {
                    v = a[a_base + grow * p.k + gk];
                } else {
                    v = a[a_base + gk * p.m + grow];
                }
            }
            asub[r * AS + kk] = v;
        }

        // ── stage B's tile, same reasoning.
        for (var i = 0u; i < 4u; i = i + 1u) {
            let idx = tid + i * THREADS;
            var kk: u32;
            var cc: u32;
            if (p.trans_b == 0u) {
                kk = idx / BN;
                cc = idx % BN;
            } else {
                cc = idx / BK;
                kk = idx % BK;
            }
            let gcol = col0 + cc;
            let gk = kt + kk;
            var v = 0.0;
            if (gk < p.k && gcol < p.n) {
                if (p.trans_b == 0u) {
                    v = b[b_base + gk * p.n + gcol];
                } else {
                    v = b[b_base + gcol * p.k + gk];
                }
            }
            bsub[kk * BN + cc] = v;
        }

        workgroupBarrier();

        let ar = lid.y * TM;
        let bc = lid.x * TN;
        for (var kk = 0u; kk < BK; kk = kk + 1u) {
            let av = vec4<f32>(
                asub[(ar + 0u) * AS + kk],
                asub[(ar + 1u) * AS + kk],
                asub[(ar + 2u) * AS + kk],
                asub[(ar + 3u) * AS + kk],
            );
            let bo = kk * BN + bc;
            let bv = vec4<f32>(bsub[bo], bsub[bo + 1u], bsub[bo + 2u], bsub[bo + 3u]);
            acc0 = fma(vec4<f32>(av.x), bv, acc0);
            acc1 = fma(vec4<f32>(av.y), bv, acc1);
            acc2 = fma(vec4<f32>(av.z), bv, acc2);
            acc3 = fma(vec4<f32>(av.w), bv, acc3);
        }

        workgroupBarrier();
    }

    write_row(row0 + lid.y * TM + 0u, col0 + lid.x * TN, bat, acc0);
    write_row(row0 + lid.y * TM + 1u, col0 + lid.x * TN, bat, acc1);
    write_row(row0 + lid.y * TM + 2u, col0 + lid.x * TN, bat, acc2);
    write_row(row0 + lid.y * TM + 3u, col0 + lid.x * TN, bat, acc3);
}

fn write_row(grow: u32, gcol0: u32, bat: u32, acc: vec4<f32>) {
    if (grow >= p.m) {
        return;
    }
    let base = bat * p.m * p.n_out + grow * p.n_out + p.n_off;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let gcol = gcol0 + j;
        if (gcol < p.n) {
            var v = acc[j] * p.alpha;
            if (p.has_bias == 1u) {
                v = v + bias[gcol];
            }
            c[base + gcol] = v;
        }
    }
}
