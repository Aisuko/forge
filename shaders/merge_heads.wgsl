// Merge per-head attention output back to model layout:
// x: [h, t, hd]  ->  out: [t, c], where c = h * hd.

struct Params {
    t: u32,
    c: u32,
    h: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    let n = p.t * p.c;
    if (i >= n) { return; }
    let hd = p.c / p.h;
    // i indexes [t, c]
    let tt = i / p.c;
    let col = i % p.c;
    let hh = col / hd;
    let d = col % hd;
    out[i] = x[hh * p.t * hd + tt * hd + d];
}
