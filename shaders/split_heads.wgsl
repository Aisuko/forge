// Split fused QKV projection into per-head layouts:
// qkv: [t, 3c]  ->  q, k, v: [h, t, hd] each, where hd = c / h.

struct Params {
    t: u32,
    c: u32,
    h: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> qkv: array<f32>;
@group(0) @binding(2) var<storage, read_write> q: array<f32>;
@group(0) @binding(3) var<storage, read_write> k: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    let n = p.t * p.c; // elements per output tensor
    if (i >= n) { return; }
    let hd = p.c / p.h;
    // i indexes [h, t, hd]
    let hh = i / (p.t * hd);
    let rem = i % (p.t * hd);
    let tt = rem / hd;
    let d = rem % hd;
    let src_row = tt * 3u * p.c;
    let col = hh * hd + d;
    q[i] = qkv[src_row + col];
    k[i] = qkv[src_row + p.c + col];
    v[i] = qkv[src_row + 2u * p.c + col];
}
