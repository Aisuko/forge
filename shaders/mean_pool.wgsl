struct Params {
    batch: u32,
    seq: u32,
    cols: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> key_len: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.batch * p.cols) { return; }
    let b = i / p.cols;
    let j = i % p.cols;
    let n = min(key_len[b], p.seq);
    if (n == 0u) {
        y[i] = 0.0;
        return;
    }
    var s = 0.0;
    for (var t = 0u; t < n; t = t + 1u) {
        s = s + x[(b * p.seq + t) * p.cols + j];
    }
    y[i] = s / f32(n);
}
