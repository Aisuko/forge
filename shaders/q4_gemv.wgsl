struct Params {
    m: u32,
    k: u32,
    n: u32,
    trans_b: u32,
    has_bias: u32,
    alpha: f32,
    n_off: u32,
    n_out: u32,
    nsplit: u32,
    per: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn word(i: u32) -> vec3<u32> {
    let base = (i >> 6u) * 10u;
    return vec3<u32>(b[base + 2u + ((i & 63u) >> 3u)], b[base], b[base + 1u]);
}

fn half(w: vec3<u32>, shift: u32) -> vec4<f32> {
    let q = (vec4<u32>(w.x) >> vec4<u32>(shift, shift + 4u, shift + 8u, shift + 12u)) & vec4<u32>(15u);
    return fma(vec4<f32>(q), vec4<f32>(bitcast<f32>(w.y)), vec4<f32>(bitcast<f32>(w.z)));
}

fn put(row: u32, col: u32, split: u32, v: f32) {
    if (p.nsplit == 1u) {
        var y = v * p.alpha;
        if (p.has_bias == 1u) {
            y = y + bias[col];
        }
        out[row * p.n_out + p.n_off + col] = y;
    } else {
        out[(split * p.m + row) * p.n + col] = v;
    }
}

fn put8(row: u32, col: u32, split: u32, lo: vec4<f32>, hi: vec4<f32>) {
    for (var j = 0u; j < 4u; j = j + 1u) {
        put(row, col + j, split, lo[j]);
        put(row, col + 4u + j, split, hi[j]);
    }
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let split = wid.y;
    let k0 = split * p.per;
    let k1 = min(k0 + p.per, p.k);
    let t = wid.x * 64u + lid.x;

    if (p.trans_b == 1u) {
        if (t >= p.n) {
            return;
        }
        var acc = vec4<f32>(0.0);
        for (var kk = k0; kk < k1; kk = kk + 8u) {
            let w = word(t * p.k + kk);
            let lo = half(w, 0u);
            let hi = half(w, 16u);
            for (var r = 0u; r < p.m; r = r + 1u) {
                let o = r * p.k + kk;
                let alo = vec4<f32>(a[o], a[o + 1u], a[o + 2u], a[o + 3u]);
                let ahi = vec4<f32>(a[o + 4u], a[o + 5u], a[o + 6u], a[o + 7u]);
                acc[r] = acc[r] + dot(alo, lo) + dot(ahi, hi);
            }
        }
        for (var r = 0u; r < p.m; r = r + 1u) {
            put(r, t, split, acc[r]);
        }
        return;
    }

    let col = t * 8u;
    if (col >= p.n) {
        return;
    }
    var lo0 = vec4<f32>(0.0);
    var hi0 = vec4<f32>(0.0);
    var lo1 = vec4<f32>(0.0);
    var hi1 = vec4<f32>(0.0);
    var lo2 = vec4<f32>(0.0);
    var hi2 = vec4<f32>(0.0);
    var lo3 = vec4<f32>(0.0);
    var hi3 = vec4<f32>(0.0);
    for (var kk = k0; kk < k1; kk = kk + 1u) {
        let w = word(kk * p.n + col);
        let lo = half(w, 0u);
        let hi = half(w, 16u);
        let a0 = vec4<f32>(a[kk]);
        lo0 = fma(a0, lo, lo0);
        hi0 = fma(a0, hi, hi0);
        if (p.m > 1u) {
            let a1 = vec4<f32>(a[p.k + kk]);
            lo1 = fma(a1, lo, lo1);
            hi1 = fma(a1, hi, hi1);
        }
        if (p.m > 2u) {
            let a2 = vec4<f32>(a[2u * p.k + kk]);
            lo2 = fma(a2, lo, lo2);
            hi2 = fma(a2, hi, hi2);
        }
        if (p.m > 3u) {
            let a3 = vec4<f32>(a[3u * p.k + kk]);
            lo3 = fma(a3, lo, lo3);
            hi3 = fma(a3, hi, hi3);
        }
    }
    put8(0u, col, split, lo0, hi0);
    if (p.m > 1u) {
        put8(1u, col, split, lo1, hi1);
    }
    if (p.m > 2u) {
        put8(2u, col, split, lo2, hi2);
    }
    if (p.m > 3u) {
        put8(3u, col, split, lo3, hi3);
    }
}
