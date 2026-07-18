// Numerically stable softmax over the last dimension, with optional causal
// masking. One workgroup per row.
//
// x: [rows, cols]. Row r's query position is (r % q_len); key j is visible
// when j <= query + off, where off = key_len - query_len (0 without KV cache).
// The mask is applied before max-subtraction; masked entries produce 0.

struct Params {
    rows: u32,
    cols: u32,
    q_len: u32,
    causal: u32,
    off: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

const NEG_INF: f32 = -3.402823e38;

fn visible(r: u32, j: u32) -> bool {
    if (p.causal == 0u) { return true; }
    let q = r % p.q_len;
    return j <= q + p.off;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let r = wid.x;
    let base = r * p.cols;

    // Pass 1: row max over visible entries.
    var m = NEG_INF;
    for (var j = li; j < p.cols; j = j + WG) {
        if (visible(r, j)) { m = max(m, x[base + j]); }
    }
    red[li] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = max(red[li], red[li + s]); }
        workgroupBarrier();
    }
    let row_max = red[0];
    workgroupBarrier();

    // Pass 2: sum of exp(x - max) over visible entries.
    var sum = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        if (visible(r, j)) { sum = sum + exp(x[base + j] - row_max); }
    }
    red[li] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (li < s) { red[li] = red[li] + red[li + s]; }
        workgroupBarrier();
    }
    let row_sum = red[0];

    // Pass 3: write normalized probabilities; masked entries get 0.
    for (var j = li; j < p.cols; j = j + WG) {
        if (visible(r, j)) {
            y[base + j] = exp(x[base + j] - row_max) / row_sum;
        } else {
            y[base + j] = 0.0;
        }
    }
}
