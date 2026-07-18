// GELU (tanh approximation) backward: dx = gelu'(x) * dy.

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

const C: f32 = 0.7978845608028654; // sqrt(2/pi)
const A: f32 = 0.044715;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.n) { return; }
    let v = x[i];
    let u = C * (v + A * v * v * v);
    let th = tanh(u);
    let sech2 = 1.0 - th * th;
    let d = 0.5 * (1.0 + th) + 0.5 * v * sech2 * C * (1.0 + 3.0 * A * v * v);
    dx[i] = d * dy[i];
}
