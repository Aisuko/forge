// Cross-entropy backward: dlogits = (probs - onehot(ids)) * scale.
// One thread per element.

struct Params {
    rows: u32,
    cols: u32,
    scale: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> probs: array<f32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.rows * p.cols) { return; }
    let r = i / p.cols;
    let j = i % p.cols;
    var v = probs[i];
    if (j == ids[r]) { v = v - 1.0; }
    out[i] = v * p.scale;
}
