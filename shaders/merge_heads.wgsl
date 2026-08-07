// Merge per-head attention output back to model layout:
// x: [b*h, t, hd]  ->  out: [b*t, c], where c = h * hd.
// See split_heads.wgsl for what `b` is; at b = 1 this is the original.

struct Params {
    t: u32,
    c: u32,
    h: u32,
    b: u32,
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
    let n = p.b * p.t * p.c;
    if (i >= n) { return; }
    let hd = p.c / p.h;
    // i indexes [b*t, c]
    let bt = i / p.c;
    let col = i % p.c;
    let bb = bt / p.t;
    let tt = bt % p.t;
    let hh = col / hd;
    let d = col % hd;
    out[i] = x[(bb * p.h + hh) * p.t * hd + tt * hd + d];
}
