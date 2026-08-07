// Second half of the split-k matvec: sum `gemv.wgsl`'s partial sums.
//
// partials is [nsplit][batch][m][n]; one thread owns one (batch, row, col) and
// walks the nsplit stride. alpha and the bias are applied here, once, rather
// than in each split.

struct Params {
    m: u32,
    n: u32,
    batch: u32,
    nsplit: u32,
    has_bias: u32,
    alpha: f32,
    n_off: u32,
    n_out: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    let per_split = p.batch * p.m * p.n;
    if (i >= per_split) {
        return;
    }
    var acc = 0.0;
    for (var s = 0u; s < p.nsplit; s = s + 1u) {
        acc = acc + partials[s * per_split + i];
    }
    let col = i % p.n;
    let row = (i / p.n) % p.m;
    let bat = i / (p.n * p.m);
    var v = acc * p.alpha;
    if (p.has_bias == 1u) {
        v = v + bias[col];
    }
    out[bat * p.m * p.n_out + row * p.n_out + p.n_off + col] = v;
}
