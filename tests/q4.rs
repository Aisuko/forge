use forge::dtype::Q4;
use forge::ops::{self, MatmulSpec};
use forge::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn wgpu_device() -> Device {
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Device::wgpu().expect("wgpu device"))
        .clone()
}

fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-1.0..1.0)).collect()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1.0 + y.abs()),
            "{what}: element {i} differs: {x} vs {y}"
        );
    }
}

#[test]
fn q4_tensor_reads_back_dequantized() {
    let mut rng = StdRng::seed_from_u64(1);
    let q = Q4::quantize(&rand_vec(&mut rng, 37 * 11));
    let t = Tensor::from_q4(&q, [37, 11], &wgpu_device()).unwrap();
    assert_eq!(t.dtype(), forge::DType::Q4);
    assert_eq!(t.to_vec_f32().unwrap(), q.dequant());
    let host = Tensor::from_q4(&q, [37, 11], &Device::Cpu).unwrap();
    assert_eq!(host.dtype(), forge::DType::F32);
    assert_eq!(host.to_vec_f32().unwrap(), q.dequant());
    assert!(t.narrow_rows(0, 1).is_err());
    assert!(ops::add(&t, &t).is_err());
}

#[test]
fn q4_matmul_matches_dequantized_reference() {
    let gpu = wgpu_device();
    let mut rng = StdRng::seed_from_u64(2);
    for (m, k, n) in [
        (1, 384, 1152),
        (2, 64, 72),
        (4, 256, 520),
        (3, 96, 100),
        (17, 128, 64),
        (70, 64, 192),
        (300, 384, 1536),
        (5, 1536, 384),
    ] {
        let a = rand_vec(&mut rng, m * k);
        let q = Q4::quantize(&rand_vec(&mut rng, k * n));
        let bias = rand_vec(&mut rng, n);
        let want = ops::matmul(
            &Tensor::from_f32(&a, [m, k], &Device::Cpu).unwrap(),
            &Tensor::from_f32(&q.dequant(), [k, n], &Device::Cpu).unwrap(),
            Some(&Tensor::from_f32(&bias, [n], &Device::Cpu).unwrap()),
            MatmulSpec::default(),
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();
        let got = ops::matmul(
            &Tensor::from_f32(&a, [m, k], &gpu).unwrap(),
            &Tensor::from_q4(&q, [k, n], &gpu).unwrap(),
            Some(&Tensor::from_f32(&bias, [n], &gpu).unwrap()),
            MatmulSpec::default(),
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();
        assert_close(&got, &want, 1e-4, &format!("q4 matmul {m}x{k}x{n}"));
    }
}

#[test]
fn q4_lm_head_and_embedding_match_reference() {
    let gpu = wgpu_device();
    let mut rng = StdRng::seed_from_u64(3);
    let (vocab, d) = (1000, 96);
    let wte = Q4::quantize(&rand_vec(&mut rng, vocab * d));
    let host = wte.dequant();
    let wpe = rand_vec(&mut rng, 8 * d);
    let cpu_emb = forge::nn::Embedding::from_host_wte(
        &host,
        vocab,
        d,
        Tensor::from_f32(&wpe, [8, d], &Device::Cpu).unwrap(),
        &Device::Cpu,
    )
    .unwrap();
    let gpu_emb = forge::nn::Embedding::from_q4_wte(
        &wte,
        vocab,
        d,
        Tensor::from_f32(&wpe, [8, d], &gpu).unwrap(),
        &gpu,
    )
    .unwrap();
    assert!(
        gpu_emb
            .wte_chunks
            .iter()
            .all(|c| c.dtype() == forge::DType::Q4)
    );

    let ids = [0u32, 999, 17, 513, 64];
    let x_cpu = cpu_emb
        .forward(&Tensor::from_u32(&ids, [5], &Device::Cpu).unwrap(), 2)
        .unwrap();
    let x_gpu = gpu_emb
        .forward(&Tensor::from_u32(&ids, [5], &gpu).unwrap(), 2)
        .unwrap();
    assert_close(
        &x_gpu.to_vec_f32().unwrap(),
        &x_cpu.to_vec_f32().unwrap(),
        1e-6,
        "q4 embedding",
    );

    for m in [1, 3, 5] {
        let a = rand_vec(&mut rng, m * d);
        let want = ops::matmul_chunked_transb(
            &Tensor::from_f32(&a, [m, d], &Device::Cpu).unwrap(),
            &cpu_emb.wte_chunks,
            1.0,
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();
        let got = ops::matmul_chunked_transb(
            &Tensor::from_f32(&a, [m, d], &gpu).unwrap(),
            &gpu_emb.wte_chunks,
            1.0,
        )
        .unwrap()
        .to_vec_f32()
        .unwrap();
        assert_close(&got, &want, 1e-4, &format!("q4 lm head m={m}"));
    }
}

#[test]
fn rows_gathers_unaligned_rows_on_the_gpu() {
    let gpu = wgpu_device();
    let data: Vec<f32> = (0..7 * 10).map(|i| i as f32).collect();
    let t = Tensor::from_f32(&data, [7, 10], &gpu).unwrap();
    let r = ops::rows(&t, 3, 2).unwrap();
    assert_eq!(r.to_vec_f32().unwrap(), data[30..50].to_vec());
    assert!(ops::rows(&t, 6, 2).is_err());
}
