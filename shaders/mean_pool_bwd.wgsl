struct Params {
    batch: u32,
    seq: u32,
    cols: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> dy: array<f32>;
@group(0) @binding(2) var<storage, read> key_len: array<u32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.batch * p.seq * p.cols) { return; }
    let row = i / p.cols;
    let j = i % p.cols;
    let b = row / p.seq;
    let t = row % p.seq;
    let n = min(key_len[b], p.seq);
    if (t < n) {
        dx[i] = dy[b * p.cols + j] / f32(n);
    } else {
        dx[i] = 0.0;
    }
}
