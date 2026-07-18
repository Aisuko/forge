// AdamW step with decoupled weight decay, in place. Bias corrections
// (1 - beta^step) are computed host-side.

struct Params {
    n: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bc1: f32, // 1 - beta1^step
    bc2: f32, // 1 - beta2^step
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> param: array<f32>;
@group(0) @binding(3) var<storage, read_write> m: array<f32>;
@group(0) @binding(4) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let i = (wid.y * nwg.x + wid.x) * 256u + li;
    if (i >= p.n) { return; }
    let g = grad[i];
    let mi = p.beta1 * m[i] + (1.0 - p.beta1) * g;
    let vi = p.beta2 * v[i] + (1.0 - p.beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    let mhat = mi / p.bc1;
    let vhat = vi / p.bc2;
    param[i] = param[i] - p.lr * (mhat / (sqrt(vhat) + p.eps) + p.weight_decay * param[i]);
}
