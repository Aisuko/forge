// Negative log-likelihood gather: out[r] = -log(probs[r, ids[r]]).
// One thread per row.

struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
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
    let r = (wid.y * nwg.x + wid.x) * 256u + li;
    if (r >= p.rows) { return; }
    let v = max(probs[r * p.cols + ids[r]], 1.17549435e-38);
    out[r] = -log(v);
}
