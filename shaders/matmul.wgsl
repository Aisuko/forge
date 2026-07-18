// Tiled batched matmul: C[b] = alpha * A[b] @ B[b] (+ bias, broadcast over rows).
//
// A: [batch, m, k], or [batch, k, m] if trans_a == 1
//    (a_stride per batch, or 0 to broadcast one A)
// B: [batch, k, n] if trans_b == 0, [batch, n, k] if trans_b == 1
//    (b_stride per batch, or 0 to broadcast one B)
// C: [batch, m, n]
// bias: [n], added when has_bias == 1 (bind a 1-element dummy otherwise).

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

const TILE: u32 = 16u;
var<workgroup> asub: array<f32, 256>;
var<workgroup> bsub: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bat = wid.z;
    let row = wid.y * TILE + lid.y; // m index
    let col = wid.x * TILE + lid.x; // n index
    let a_base = bat * p.a_stride;
    let b_base = bat * p.b_stride;

    var acc = 0.0;
    let ntiles = (p.k + TILE - 1u) / TILE;
    for (var t = 0u; t < ntiles; t = t + 1u) {
        let ak = t * TILE + lid.x;
        var av = 0.0;
        if (row < p.m && ak < p.k) {
            if (p.trans_a == 0u) {
                av = a[a_base + row * p.k + ak];
            } else {
                av = a[a_base + ak * p.m + row];
            }
        }
        asub[lid.y * TILE + lid.x] = av;

        let bk = t * TILE + lid.y;
        var bv = 0.0;
        if (bk < p.k && col < p.n) {
            if (p.trans_b == 0u) {
                bv = b[b_base + bk * p.n + col];
            } else {
                bv = b[b_base + col * p.k + bk];
            }
        }
        bsub[lid.y * TILE + lid.x] = bv;

        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {
            acc = acc + asub[lid.y * TILE + i] * bsub[i * TILE + lid.x];
        }
        workgroupBarrier();
    }

    if (row < p.m && col < p.n) {
        var v = acc * p.alpha;
        if (p.has_bias == 1u) {
            v = v + bias[col];
        }
        c[bat * p.m * p.n_out + row * p.n_out + p.n_off + col] = v;
    }
}
