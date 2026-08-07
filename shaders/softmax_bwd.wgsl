// Softmax backward: dx = y * (dy - sum(y * dy)) per row. One workgroup per
// row. Masked entries of the causal forward hold exact zeros in y, so they
// contribute nothing and receive zero gradient — no mask parameters needed.

struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

const WG: u32 = 256u;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    // One workgroup per row, laid out 2-D: a batched training step has more
    // rows than the 65535 a single grid dimension may hold (64 sequences x 6
    // heads x 256 queries is 98304). See `row_grid`.
    let r = wid.y * nwg.x + wid.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;

    var s = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        s = s + y[base + j] * dy[base + j];
    }
    red[li] = s;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) { red[li] = red[li] + red[li + st]; }
        workgroupBarrier();
    }
    let total = red[0];

    for (var j = li; j < p.cols; j = j + WG) {
        dx[base + j] = y[base + j] * (dy[base + j] - total);
    }
}
