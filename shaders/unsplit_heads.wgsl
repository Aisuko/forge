// split_heads backward for one of q/k/v: place d [h, t, hd] into the
// `which` third of a zero-initialized [t, 3c] buffer. One thread per element
// of d; untouched entries stay zero (output buffer is freshly zeroed).

struct Params {
    t: u32,
    c: u32,
    h: u32,
    which: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> d: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

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
    out[tt * 3u * p.c + p.which * p.c + hh * hd + dd] = d[i];
}
