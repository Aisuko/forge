// merge_heads backward: dy [b*t, c] -> dx [b*h, t, hd], hd = c / h.
// One thread per element. See split_heads.wgsl for `b`.

struct Params {
    t: u32,
    c: u32,
    h: u32,
    b: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.b * p.t * p.c) { return; }
    let hd = p.c / p.h;
    let bh = i / (p.t * hd);
    let rem = i % (p.t * hd);
    let tt = rem / hd;
    let dd = rem % hd;
    let bb = bh / p.h;
    let hh = bh % p.h;
    dx[i] = dy[(bb * p.t + tt) * p.c + hh * hd + dd];
}
