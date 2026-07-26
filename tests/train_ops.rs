//! Stage 8/9 op gates (roadmap v4): backward kernels and training modules
//! match the CPU reference (<= 1e-3 abs), dropout masks are identical across
//! backends, and AdamW matches a hand-computed step.

use forge::ops;
use forge::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn wgpu_device() -> Device {
    // One device per test binary: concurrent device creation from parallel
    // test threads segfaults some Vulkan drivers (observed on llvmpipe).
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Device::wgpu().expect("wgpu adapter required for parity tests"))
        .clone()
}

fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-2.0f32..2.0)).collect()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        max = max.max((x - y).abs());
    }
    assert!(max <= 1e-3, "{what}: max abs diff {max:.2e} > 1e-3");
}

#[test]
fn gelu_bwd_parity_and_numeric() {
    let mut rng = StdRng::seed_from_u64(20);
    let gpu = wgpu_device();
    let n = 1000;
    let x = rand_vec(&mut rng, n);
    let dy = rand_vec(&mut rng, n);
    let run = |dev: &Device| {
        let xt = Tensor::from_f32(&x, [n], dev).unwrap();
        let dyt = Tensor::from_f32(&dy, [n], dev).unwrap();
        ops::gelu_bwd(&xt, &dyt).unwrap().to_vec_f32().unwrap()
    };
    let cpu = run(&Device::Cpu);
    assert_close(&cpu, &run(&gpu), "gelu_bwd");
    // Numeric check: (gelu(x+h) - gelu(x-h)) / 2h ~ gelu'(x).
    let h = 1e-3f32;
    let dev = Device::Cpu;
    let xp = Tensor::from_f32(&x.iter().map(|v| v + h).collect::<Vec<_>>(), [n], &dev).unwrap();
    let xm = Tensor::from_f32(&x.iter().map(|v| v - h).collect::<Vec<_>>(), [n], &dev).unwrap();
    let gp = ops::gelu(&xp).unwrap().to_vec_f32().unwrap();
    let gm = ops::gelu(&xm).unwrap().to_vec_f32().unwrap();
    for i in 0..n {
        let numeric = (gp[i] - gm[i]) / (2.0 * h) * dy[i];
        assert!(
            (numeric - cpu[i]).abs() <= 2e-2 * (1.0 + numeric.abs()),
            "gelu' numeric mismatch at {i}: {numeric} vs {}",
            cpu[i]
        );
    }
}

#[test]
fn softmax_bwd_parity() {
    let mut rng = StdRng::seed_from_u64(21);
    let gpu = wgpu_device();
    let (h, t) = (3usize, 17usize);
    let x = rand_vec(&mut rng, h * t * t);
    let dy = rand_vec(&mut rng, h * t * t);
    let run = |dev: &Device| {
        let xt = Tensor::from_f32(&x, [h, t, t], dev).unwrap();
        let dyt = Tensor::from_f32(&dy, [h, t, t], dev).unwrap();
        let y = ops::softmax(&xt, true, 0).unwrap();
        ops::softmax_bwd(&y, &dyt).unwrap().to_vec_f32().unwrap()
    };
    let cpu = run(&Device::Cpu);
    assert_close(&cpu, &run(&gpu), "softmax_bwd");
    // Masked (upper-triangle) entries must get exactly zero gradient.
    for hh in 0..h {
        for q in 0..t {
            for j in (q + 1)..t {
                assert_eq!(cpu[hh * t * t + q * t + j], 0.0, "masked grad not zero");
            }
        }
    }
}

#[test]
fn layernorm_bwd_parity() {
    let mut rng = StdRng::seed_from_u64(22);
    let gpu = wgpu_device();
    let (rows, cols) = (37usize, 300usize);
    let x = rand_vec(&mut rng, rows * cols);
    let gamma = rand_vec(&mut rng, cols);
    let dy = rand_vec(&mut rng, rows * cols);
    let run = |dev: &Device| {
        let xt = Tensor::from_f32(&x, [rows, cols], dev).unwrap();
        let gt = Tensor::from_f32(&gamma, [cols], dev).unwrap();
        let dyt = Tensor::from_f32(&dy, [rows, cols], dev).unwrap();
        let (dx, dg, db) = ops::layernorm_bwd(&xt, &gt, &dyt, 1e-5).unwrap();
        (
            dx.to_vec_f32().unwrap(),
            dg.to_vec_f32().unwrap(),
            db.to_vec_f32().unwrap(),
        )
    };
    let (cdx, cdg, cdb) = run(&Device::Cpu);
    let (gdx, gdg, gdb) = run(&gpu);
    assert_close(&cdx, &gdx, "layernorm_bwd dx");
    assert_close(&cdg, &gdg, "layernorm_bwd dgamma");
    assert_close(&cdb, &gdb, "layernorm_bwd dbeta");
}

#[test]
fn sum_rows_scale_sumsq_parity() {
    let mut rng = StdRng::seed_from_u64(23);
    let gpu = wgpu_device();
    let (rows, cols) = (129usize, 517usize);
    let x = rand_vec(&mut rng, rows * cols);
    let run = |dev: &Device| {
        let xt = Tensor::from_f32(&x, [rows, cols], dev).unwrap();
        (
            ops::sum_rows(&xt).unwrap().to_vec_f32().unwrap(),
            ops::scale(&xt, 0.25).unwrap().to_vec_f32().unwrap(),
            ops::sumsq(&xt).unwrap(),
        )
    };
    let (cs, csc, cq) = run(&Device::Cpu);
    let (gs, gsc, gq) = run(&gpu);
    assert_close(&cs, &gs, "sum_rows");
    assert_close(&csc, &gsc, "scale");
    let rel = (cq - gq).abs() / cq.max(1.0);
    assert!(rel <= 1e-4, "sumsq rel diff {rel:.2e}");
}

#[test]
fn scatter_add_parity_with_repeats() {
    let mut rng = StdRng::seed_from_u64(24);
    let gpu = wgpu_device();
    let (vocab, c, t) = (50usize, 32usize, 200usize);
    // Heavy id repetition to exercise the CAS loop.
    let ids: Vec<u32> = (0..t).map(|_| rng.random_range(0..8u32)).collect();
    let src = rand_vec(&mut rng, t * c);
    let run = |dev: &Device| {
        let mut dst = Tensor::zeros([vocab, c], dev).unwrap();
        let idst = Tensor::from_u32(&ids, [t], dev).unwrap();
        let srct = Tensor::from_f32(&src, [t, c], dev).unwrap();
        ops::scatter_add_rows(&mut dst, &idst, &srct).unwrap();
        dst.to_vec_f32().unwrap()
    };
    // Atomic order differs, so compare within the 1e-3 bwd tolerance.
    assert_close(&run(&Device::Cpu), &run(&gpu), "scatter_add");
}

#[test]
fn cross_entropy_parity_and_reference() {
    let mut rng = StdRng::seed_from_u64(25);
    let gpu = wgpu_device();
    let (rows, cols) = (13usize, 97usize);
    let logits = rand_vec(&mut rng, rows * cols);
    let ids: Vec<u32> = (0..rows)
        .map(|_| rng.random_range(0..cols as u32))
        .collect();
    let run = |dev: &Device| {
        let lt = Tensor::from_f32(&logits, [rows, cols], dev).unwrap();
        let it = Tensor::from_u32(&ids, [rows], dev).unwrap();
        let probs = ops::softmax(&lt, false, 0).unwrap();
        let nll = ops::gather_nll(&probs, &it).unwrap().to_vec_f32().unwrap();
        let dlogits = ops::ce_bwd(&probs, &it, 1.0 / rows as f32)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        (nll, dlogits)
    };
    let (cn, cd) = run(&Device::Cpu);
    let (gn, gd) = run(&gpu);
    assert_close(&cn, &gn, "gather_nll");
    assert_close(&cd, &gd, "ce_bwd");
    // Hand-computed reference on row 0.
    let row = &logits[0..cols];
    let max = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let z: f32 = row.iter().map(|&v| (v - max).exp()).sum();
    let p0 = (row[ids[0] as usize] - max).exp() / z;
    assert!((cn[0] - (-p0.ln())).abs() <= 1e-5, "nll reference");
    // Gradient rows must sum to ~0 (softmax minus one-hot).
    let s: f32 = cd[0..cols].iter().sum();
    assert!(s.abs() <= 1e-6, "ce grad row sum {s}");
}

#[test]
fn dropout_deterministic_across_backends() {
    let mut rng = StdRng::seed_from_u64(26);
    let gpu = wgpu_device();
    let n = 4096;
    let x = rand_vec(&mut rng, n);
    let run = |dev: &Device| {
        let xt = Tensor::from_f32(&x, [n], dev).unwrap();
        ops::dropout(&xt, 0.3, 1234).unwrap().to_vec_f32().unwrap()
    };
    let cpu = run(&Device::Cpu);
    let wg = run(&gpu);
    assert_eq!(cpu, wg, "dropout masks must match bit-for-bit");
    let kept = cpu.iter().filter(|v| **v != 0.0).count() as f32 / n as f32;
    assert!((kept - 0.7).abs() < 0.05, "keep rate {kept} far from 0.7");
    // Same seed reproduces; different seed differs.
    assert_eq!(cpu, run(&Device::Cpu), "dropout not reproducible");
    let other = {
        let xt = Tensor::from_f32(&x, [n], &Device::Cpu).unwrap();
        ops::dropout(&xt, 0.3, 99).unwrap().to_vec_f32().unwrap()
    };
    assert_ne!(cpu, other, "different seeds gave identical masks");
}

#[test]
fn unsplit_unmerge_roundtrip() {
    let mut rng = StdRng::seed_from_u64(27);
    let gpu = wgpu_device();
    let (t, c, h) = (9usize, 24usize, 4usize);
    let qkv = rand_vec(&mut rng, t * 3 * c);
    let run = |dev: &Device| {
        let qkvt = Tensor::from_f32(&qkv, [t, 3 * c], dev).unwrap();
        let (q, k, v) = ops::split_heads(&qkvt, h).unwrap();
        // unsplit(split(x)) summed over q/k/v must reconstruct x.
        let uq = ops::unsplit_head(&q, 0).unwrap();
        let uk = ops::unsplit_head(&k, 1).unwrap();
        let uv = ops::unsplit_head(&v, 2).unwrap();
        let sum = ops::add(&ops::add(&uq, &uk).unwrap(), &uv).unwrap();
        // unmerge(merge(q)) must equal q.
        let um = ops::unmerge_heads(&ops::merge_heads(&q).unwrap(), h).unwrap();
        (
            sum.to_vec_f32().unwrap(),
            um.to_vec_f32().unwrap(),
            q.to_vec_f32().unwrap(),
        )
    };
    for dev in [Device::Cpu, gpu] {
        let (sum, um, q) = run(&dev);
        assert_eq!(sum, qkv, "unsplit roundtrip on {}", dev.describe());
        assert_eq!(um, q, "unmerge roundtrip on {}", dev.describe());
    }
}

#[test]
fn adamw_parity_and_reference() {
    let mut rng = StdRng::seed_from_u64(28);
    let gpu = wgpu_device();
    let n = 1000;
    let p0 = rand_vec(&mut rng, n);
    let g1 = rand_vec(&mut rng, n);
    let g2 = rand_vec(&mut rng, n);
    let (lr, b1, b2, eps, wd) = (1e-2f32, 0.9f32, 0.999f32, 1e-8f32, 0.1f32);
    let run = |dev: &Device| {
        let mut p = Tensor::from_f32(&p0, [n], dev).unwrap();
        let mut m = Tensor::zeros([n], dev).unwrap();
        let mut v = Tensor::zeros([n], dev).unwrap();
        for (step, g) in [(1u32, &g1), (2u32, &g2)] {
            let gt = Tensor::from_f32(g, [n], dev).unwrap();
            ops::adamw(&mut p, &gt, &mut m, &mut v, lr, b1, b2, eps, wd, step).unwrap();
        }
        p.to_vec_f32().unwrap()
    };
    let cpu = run(&Device::Cpu);
    assert_close(&cpu, &run(&gpu), "adamw");
    // Scalar reference for element 0, two steps.
    let (mut p, mut m, mut v) = (p0[0], 0.0f32, 0.0f32);
    for (step, g) in [(1i32, g1[0]), (2, g2[0])] {
        m = b1 * m + (1.0 - b1) * g;
        v = b2 * v + (1.0 - b2) * g * g;
        let mhat = m / (1.0 - b1.powi(step));
        let vhat = v / (1.0 - b2.powi(step));
        p -= lr * (mhat / (vhat.sqrt() + eps) + wd * p);
    }
    assert!((cpu[0] - p).abs() <= 1e-6, "adamw scalar reference");
}
