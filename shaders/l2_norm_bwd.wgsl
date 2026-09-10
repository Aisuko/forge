struct Params {
    rows: u32,
    cols: u32,
    eps: f32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> dx: array<f32>;

const WG: u32 = 256u;
var<workgroup> red_ss: array<f32, 256>;
var<workgroup> red_dot: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_index) li: u32,
) {
    let r = wid.y * nwg.x + wid.x;
    if (r >= p.rows) { return; }
    let base = r * p.cols;

    var ss = 0.0;
    var dot = 0.0;
    for (var j = li; j < p.cols; j = j + WG) {
        let v = x[base + j];
        ss = ss + v * v;
        dot = dot + v * dy[base + j];
    }
    red_ss[li] = ss;
    red_dot[li] = dot;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (li < st) {
            red_ss[li] = red_ss[li] + red_ss[li + st];
            red_dot[li] = red_dot[li] + red_dot[li + st];
        }
        workgroupBarrier();
    }
    let inv2 = 1.0 / (red_ss[0] + p.eps);
    let inv = sqrt(inv2);
    let scaled = red_dot[0] * inv2;

    for (var j = li; j < p.cols; j = j + WG) {
        dx[base + j] = inv * (dy[base + j] - x[base + j] * scaled);
    }
}
