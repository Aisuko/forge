// LayerNorm backward (input gradient). One workgroup per row.
// dx = inv_std * (gamma*dy - mean(gamma*dy) - xhat * mean(gamma*dy*xhat))
// Also writes per-row (mean, inv_std) to `stats` for the dgamma/dbeta pass.

struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> stats: array<f32>; // [rows, 2]

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

fn reduce_sum(li: u32, v: f32) -> f32 {
    red[li] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) { red[li] = red[li] + red[li + st]; }
        workgroupBarrier();
    }
    let s = red[0];
    workgroupBarrier();
    return s;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    // One workgroup per row, laid out 2-D: a batched training step has more
    // rows than the 65535 a single grid dimension may hold (64 sequences x 6
    // heads x 256 queries is 98304). See `row_grid`.
    let r = wid.y * nwg.x + wid.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;
    let nf = f32(p.cols);

    var s = 0.0;
    for (var j = li; j < p.cols; j = j + WG) { s = s + x[base + j]; }
    let mean = reduce_sum(li, s) / nf;

    var v = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        let d = x[base + j] - mean;
        v = v + d * d;
    }
    let variance = reduce_sum(li, v) / nf;
    let inv_std = 1.0 / sqrt(variance + p.eps);

    var t1 = 0.0; // sum(gamma * dy)
    var t2 = 0.0; // sum(gamma * dy * xhat)
    for (var j = li; j < p.cols; j = j + WG) {
        let gd = gamma[j] * dy[base + j];
        t1 = t1 + gd;
        t2 = t2 + gd * (x[base + j] - mean) * inv_std;
    }
    let s1 = reduce_sum(li, t1);
    let s2 = reduce_sum(li, t2);

    for (var j = li; j < p.cols; j = j + WG) {
        let xhat = (x[base + j] - mean) * inv_std;
        let gd = gamma[j] * dy[base + j];
        dx[base + j] = inv_std * (gd - s1 / nf - xhat * s2 / nf);
    }
    if (li == 0u) {
        stats[r * 2u] = mean;
        stats[r * 2u + 1u] = inv_std;
    }
}
