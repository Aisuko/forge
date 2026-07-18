// LayerNorm backward (parameter gradients). One thread per column, using
// the per-row (mean, inv_std) written by layernorm_bwd_dx.
// dgamma[j] = sum_r dy[r,j] * xhat[r,j]; dbeta[j] = sum_r dy[r,j].

struct Params {
    rows: u32,
    cols: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read> stats: array<f32>; // [rows, 2]
@group(0) @binding(4) var<storage, read_write> dgamma: array<f32>;
@group(0) @binding(5) var<storage, read_write> dbeta: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let j = (wid.y * nwg.x + wid.x) * 256u + li;
    if (j >= p.cols) { return; }
    var dg = 0.0;
    var db = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        let mean = stats[r * 2u];
        let inv_std = stats[r * 2u + 1u];
        let d = dy[r * p.cols + j];
        dg = dg + d * (x[r * p.cols + j] - mean) * inv_std;
        db = db + d;
    }
    dgamma[j] = dg;
    dbeta[j] = db;
}
