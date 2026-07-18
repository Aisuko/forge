// Column sums: [rows, cols] -> [cols]. One thread per column.

struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let j = (wid.y * nwg.x + wid.x) * 256u + li;
    if (j >= p.cols) { return; }
    var s = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        s = s + x[r * p.cols + j];
    }
    out[j] = s;
}
