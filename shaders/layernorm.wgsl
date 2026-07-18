// LayerNorm over the last dimension. One workgroup per row.
// y = (x - mean) / sqrt(var + eps) * gamma + beta
// Variance is the biased (population) estimate, matching PyTorch LayerNorm.

struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let r = wid.x;
    let base = r * p.cols;
    let nf = f32(p.cols);

    // Pass 1: mean.
    var s = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        s = s + x[base + j];
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) { red[li] = red[li] + red[li + st]; }
        workgroupBarrier();
    }
    let mean = red[0] / nf;
    workgroupBarrier();

    // Pass 2: variance.
    var v = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        let d = x[base + j] - mean;
        v = v + d * d;
    }
    red[li] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) { red[li] = red[li] + red[li + st]; }
        workgroupBarrier();
    }
    let variance = red[0] / nf;
    let inv_std = 1.0 / sqrt(variance + p.eps);

    // Pass 3: normalize, scale, shift.
    for (var j = li; j < p.cols; j = j + WG) {
        y[base + j] = (x[base + j] - mean) * inv_std * gamma[j] + beta[j];
    }
}
