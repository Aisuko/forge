struct Params {
    rows: u32,
    cols: u32,
    group: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> key_len: array<u32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

const NEG_INF: f32 = -3.402823e38;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let r = wid.y * nwg.x + wid.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;
    let klen = key_len[r / p.group];

    var m = NEG_INF;
    for (var j = li; j < p.cols; j = j + WG) {
        if (j < klen) { m = max(m, x[base + j]); }
    }
    red[li] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = max(red[li], red[li + s]); }
        workgroupBarrier();
    }
    let row_max = red[0];
    workgroupBarrier();

    var sum = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        if (j < klen) { sum = sum + exp(x[base + j] - row_max); }
    }
    red[li] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    let row_sum = red[0];

    for (var j = li; j < p.cols; j = j + WG) {
        if (j < klen && row_sum > 0.0) {
            y[base + j] = exp(x[base + j] - row_max) / row_sum;
        } else {
            y[base + j] = 0.0;
        }
    }
}
