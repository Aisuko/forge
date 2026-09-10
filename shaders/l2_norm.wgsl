struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let r = wid.y * nwg.x + wid.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;

    var s = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        let v = x[base + j];
        s = s + v * v;
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) { red[li] = red[li] + red[li + st]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sqrt(red[0] + p.eps);

    for (var j = li; j < p.cols; j = j + WG) {
        y[base + j] = x[base + j] * inv;
    }
}
