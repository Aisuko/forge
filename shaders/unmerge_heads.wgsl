// merge_heads backward: dy [t, c] -> dx [h, t, hd], hd = c / h.
// One thread per element.

struct Params {
    t: u32,
    c: u32,
    h: u32,
    _pad0: u32,
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
    if (i >= p.t * p.c) { return; }
    let hd = p.c / p.h;
    let hh = i / (p.t * hd);
    let rem = i % (p.t * hd);
    let tt = rem / hd;
    let dd = rem % hd;
    dx[i] = dy[tt * p.c + hh * hd + dd];
}
