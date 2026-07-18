// Partial sum of squares: each workgroup reduces its 256-element slice to
// out[group]. The host sums the (small) partial array — used for the global
// gradient norm.

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let group = wid.y * nwg.x + wid.x;
    let i = group * 256u + li;
    var v = 0.0;
    if (i < p.n) { v = x[i] * x[i]; }
    red[li] = v;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) { red[li] = red[li] + red[li + st]; }
        workgroupBarrier();
    }
    if (li == 0u) { out[group] = red[0]; }
}
