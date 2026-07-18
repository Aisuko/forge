// Append src [h, t, hd] into a preallocated KV cache [h, cap, hd] at row
// offset `len` within each head. One thread per src element.

struct Params {
    h: u32,
    t: u32,
    hd: u32,
    cap: u32,
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> cache: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let idx = (wid.y * nwg.x + wid.x) * 256u + li;
    let n = p.h * p.t * p.hd;
    if (idx >= n) { return; }
    let hh = idx / (p.t * p.hd);
    let rem = idx % (p.t * p.hd);
    let tt = rem / p.hd;
    let d = rem % p.hd;
    cache[hh * p.cap * p.hd + (p.len + tt) * p.hd + d] = src[idx];
}
