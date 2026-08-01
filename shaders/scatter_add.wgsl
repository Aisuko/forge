// Embedding backward: dst[ids[r]] += src[r] (rows of width c).
// WGSL has no f32 atomicAdd, so this uses a compare-exchange loop over the
// f32 bits. One thread per src element.

struct Params {
    t: u32,
    c: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> ids: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.t * p.c) { return; }
    let r = i / p.c;
    let j = i % p.c;
    let val = src[i];
    if (val == 0.0) { return; }
    let di = ids[r] * p.c + j;
    var old = atomicLoad(&dst[di]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old) + val);
        let res = atomicCompareExchangeWeak(&dst[di], old, new_bits);
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}
