// Inverted dropout with a counter-based PCG hash (identical to
// backend::cpu::pcg_hash, so masks match across backends bit-for-bit).
// Applying the same (seed, p) to dy gives the backward pass.
// `scale` (1/(1-p)) is computed host-side and passed in rather than
// recomputed here: GPU division isn't guaranteed correctly-rounded, so an
// independent 1.0/(1.0-p) can differ from the CPU's by a few ULP.

struct Params {
    n: u32,
    seed: u32,
    p: f32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;

fn pcg_hash(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.n) { return; }
    let r = pcg_hash(p.seed ^ (i * 0x9E3779B9u));
    let u = f32(r >> 8u) / 16777216.0;
    if (u >= p.p) {
        y[i] = x[i] * p.scale;
    } else {
        y[i] = 0.0;
    }
}
