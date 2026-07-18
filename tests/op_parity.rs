//! Per-op numerical parity: every WGSL kernel must match the CPU reference
//! within 1e-4 absolute tolerance (roadmap acceptance criteria), including
//! non-square and non-power-of-two shapes.

use forge::ops::{self, MatmulSpec};
use forge::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const TOL: f32 = 1e-4;

fn rand_vec(rng: &mut StdRng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-2.0..2.0)).collect()
}

fn assert_close(cpu: &[f32], gpu: &[f32], what: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{what}: length mismatch");
    let mut worst = 0.0f32;
    for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
        let d = (a - b).abs();
        if d > worst {
            worst = d;
        }
        assert!(
            d <= TOL,
            "{what}: element {i} differs: cpu={a} gpu={b} (|d|={d})"
        );
    }
    println!("{what}: max abs diff {worst:.2e}");
}

fn wgpu_device() -> Device {
    // One device per test binary: concurrent device creation from parallel
    // test threads segfaults some Vulkan drivers (observed on llvmpipe).
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Device::wgpu().expect("wgpu device")).clone()
}

#[test]
fn add_parity() {
    let mut rng = StdRng::seed_from_u64(1);
    let gpu = wgpu_device();
    for n in [1usize, 255, 256, 1000, 70_000] {
        let a = rand_vec(&mut rng, n);
        let b = rand_vec(&mut rng, n);
        let cpu_out = ops::add(
            &Tensor::from_f32(&a, [n], &Device::Cpu).unwrap(),
            &Tensor::from_f32(&b, [n], &Device::Cpu).unwrap(),
        )
        .unwrap();
        let gpu_out = ops::add(
            &Tensor::from_f32(&a, [n], &gpu).unwrap(),
            &Tensor::from_f32(&b, [n], &gpu).unwrap(),
        )
        .unwrap();
        assert_close(
            &cpu_out.to_vec_f32().unwrap(),
            &gpu_out.to_vec_f32().unwrap(),
            &format!("add n={n}"),
        );
    }
}

#[test]
fn gelu_parity() {
    let mut rng = StdRng::seed_from_u64(2);
    let gpu = wgpu_device();
    let n = 4097;
    let x = rand_vec(&mut rng, n);
    let cpu_out = ops::gelu(&Tensor::from_f32(&x, [n], &Device::Cpu).unwrap()).unwrap();
    let gpu_out = ops::gelu(&Tensor::from_f32(&x, [n], &gpu).unwrap()).unwrap();
    assert_close(
        &cpu_out.to_vec_f32().unwrap(),
        &gpu_out.to_vec_f32().unwrap(),
        "gelu",
    );
}

#[test]
fn matmul_parity() {
    let mut rng = StdRng::seed_from_u64(3);
    let gpu = wgpu_device();
    // (m, k, n) including non-multiples of the 16x16 tile.
    for (m, k, n) in [
        (1usize, 768usize, 50257usize / 64),
        (17, 33, 5),
        (16, 16, 16),
        (64, 64, 100),
    ] {
        for trans_a in [false, true] {
            for trans_b in [false, true] {
                for with_bias in [false, true] {
                    let a = rand_vec(&mut rng, m * k);
                    let b = rand_vec(&mut rng, k * n);
                    let bias = rand_vec(&mut rng, n);
                    let a_shape = if trans_a { [k, m] } else { [m, k] };
                    let b_shape = if trans_b { [n, k] } else { [k, n] };
                    let spec = MatmulSpec {
                        trans_a,
                        trans_b,
                        alpha: 0.5,
                        ..Default::default()
                    };
                    let run = |dev: &Device| {
                        let at = Tensor::from_f32(&a, a_shape, dev).unwrap();
                        let bt = Tensor::from_f32(&b, b_shape, dev).unwrap();
                        let biast = Tensor::from_f32(&bias, [n], dev).unwrap();
                        ops::matmul(&at, &bt, with_bias.then_some(&biast), spec)
                            .unwrap()
                            .to_vec_f32()
                            .unwrap()
                    };
                    assert_close(
                        &run(&Device::Cpu),
                        &run(&gpu),
                        &format!(
                            "matmul m={m} k={k} n={n} trans_a={trans_a} trans_b={trans_b} bias={with_bias}"
                        ),
                    );
                }
            }
        }
    }

    // trans_a must equal an explicit host-side transpose (CPU).
    let (m, k, n) = (7usize, 13usize, 9usize);
    let a = rand_vec(&mut rng, m * k);
    let b = rand_vec(&mut rng, k * n);
    let mut a_t = vec![0.0f32; k * m]; // [k, m]
    for i in 0..m {
        for kk in 0..k {
            a_t[kk * m + i] = a[i * k + kk];
        }
    }
    let run_cpu = |data: &[f32], shape: [usize; 2], trans_a: bool| {
        let at = Tensor::from_f32(data, shape, &Device::Cpu).unwrap();
        let bt = Tensor::from_f32(&b, [k, n], &Device::Cpu).unwrap();
        let spec = MatmulSpec {
            trans_a,
            ..Default::default()
        };
        ops::matmul(&at, &bt, None, spec)
            .unwrap()
            .to_vec_f32()
            .unwrap()
    };
    assert_eq!(
        run_cpu(&a, [m, k], false),
        run_cpu(&a_t, [k, m], true),
        "trans_a semantics"
    );
}

#[test]
fn matmul_b_rows_view_parity() {
    // A batched B stored [batch, cap, d] with b_rows = kv must equal the
    // compact [batch, kv, d] matmul — on both devices, both transposes.
    let mut rng = StdRng::seed_from_u64(11);
    let gpu = wgpu_device();
    let (batch, m, d, kv, cap) = (4usize, 3usize, 16usize, 5usize, 8usize);
    let padded = rand_vec(&mut rng, batch * cap * d);
    let mut compact = vec![0.0f32; batch * kv * d];
    for bb in 0..batch {
        compact[bb * kv * d..(bb + 1) * kv * d]
            .copy_from_slice(&padded[bb * cap * d..bb * cap * d + kv * d]);
    }
    for trans_b in [false, true] {
        // trans_b: A [batch, m, d] @ B[.., :kv, :]^T -> [batch, m, kv]
        // else:    A [batch, m, kv] @ B[.., :kv, :]  -> [batch, m, d]
        let ak = if trans_b { d } else { kv };
        let a = rand_vec(&mut rng, batch * m * ak);
        let run = |dev: &Device, b_data: &[f32], rows: usize, b_rows: Option<usize>| {
            let at = Tensor::from_f32(&a, [batch, m, ak], dev).unwrap();
            let bt = Tensor::from_f32(b_data, [batch, rows, d], dev).unwrap();
            let spec = MatmulSpec {
                trans_b,
                b_rows,
                ..Default::default()
            };
            ops::matmul(&at, &bt, None, spec)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        };
        let reference = run(&Device::Cpu, &compact, kv, None);
        assert_close(
            &reference,
            &run(&Device::Cpu, &padded, cap, Some(kv)),
            &format!("b_rows view CPU trans_b={trans_b}"),
        );
        assert_close(
            &reference,
            &run(&gpu, &padded, cap, Some(kv)),
            &format!("b_rows view WGPU trans_b={trans_b}"),
        );
    }
}

#[test]
fn kv_append_parity() {
    let mut rng = StdRng::seed_from_u64(12);
    let gpu = wgpu_device();
    let (h, hd, cap) = (3usize, 8usize, 16usize);
    let first = rand_vec(&mut rng, h * 5 * hd); // t = 5
    let second = rand_vec(&mut rng, h * 1 * hd); // t = 1
    let run = |dev: &Device| {
        let mut cache = Tensor::zeros([h, cap, hd], dev).unwrap();
        let f = Tensor::from_f32(&first, [h, 5, hd], dev).unwrap();
        let s = Tensor::from_f32(&second, [h, 1, hd], dev).unwrap();
        ops::kv_append(&mut cache, &f, 0).unwrap();
        ops::kv_append(&mut cache, &s, 5).unwrap();
        cache.to_vec_f32().unwrap()
    };
    let cpu = run(&Device::Cpu);
    let wgpu = run(&gpu);
    assert_eq!(cpu, wgpu, "kv_append CPU vs WGPU");
    // Spot-check placement: head 1, position 5 must hold `second`'s head 1.
    let base = 1 * cap * hd + 5 * hd;
    assert_eq!(&cpu[base..base + hd], &second[hd..2 * hd], "kv placement");
}

#[test]
fn batched_matmul_parity() {
    let mut rng = StdRng::seed_from_u64(4);
    let gpu = wgpu_device();
    let (batch, m, k, n) = (12usize, 21usize, 64usize, 21usize);
    for trans_b in [false, true] {
        let a = rand_vec(&mut rng, batch * m * k);
        let b = rand_vec(&mut rng, batch * k * n);
        let b_shape = if trans_b {
            [batch, n, k]
        } else {
            [batch, k, n]
        };
        let spec = MatmulSpec {
            trans_b,
            alpha: 0.125,
            ..Default::default()
        };
        let run = |dev: &Device| {
            let at = Tensor::from_f32(&a, [batch, m, k], dev).unwrap();
            let bt = Tensor::from_f32(&b, b_shape, dev).unwrap();
            ops::matmul(&at, &bt, None, spec)
                .unwrap()
                .to_vec_f32()
                .unwrap()
        };
        assert_close(
            &run(&Device::Cpu),
            &run(&gpu),
            &format!("batched matmul trans_b={trans_b}"),
        );
    }
}

#[test]
fn softmax_parity() {
    let mut rng = StdRng::seed_from_u64(5);
    let gpu = wgpu_device();
    for (h, t, causal) in [(1usize, 7usize, false), (12, 21, true), (12, 300, true)] {
        let x = rand_vec(&mut rng, h * t * t);
        let run = |dev: &Device| {
            let xt = Tensor::from_f32(&x, [h, t, t], dev).unwrap();
            ops::softmax(&xt, causal, 0).unwrap().to_vec_f32().unwrap()
        };
        let cpu_out = run(&Device::Cpu);
        let gpu_out = run(&gpu);
        assert_close(
            &cpu_out,
            &gpu_out,
            &format!("softmax h={h} t={t} causal={causal}"),
        );
        // Rows must sum to 1.
        for r in 0..h * t {
            let s: f32 = cpu_out[r * t..(r + 1) * t].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "softmax row {r} sums to {s}");
        }
    }
}

#[test]
fn layernorm_parity() {
    let mut rng = StdRng::seed_from_u64(6);
    let gpu = wgpu_device();
    let (rows, cols) = (37usize, 768usize);
    let x = rand_vec(&mut rng, rows * cols);
    let g = rand_vec(&mut rng, cols);
    let b = rand_vec(&mut rng, cols);
    let run = |dev: &Device| {
        let xt = Tensor::from_f32(&x, [rows, cols], dev).unwrap();
        let gt = Tensor::from_f32(&g, [cols], dev).unwrap();
        let bt = Tensor::from_f32(&b, [cols], dev).unwrap();
        ops::layernorm(&xt, &gt, &bt, 1e-5)
            .unwrap()
            .to_vec_f32()
            .unwrap()
    };
    assert_close(&run(&Device::Cpu), &run(&gpu), "layernorm");
}

#[test]
fn embedding_parity() {
    let mut rng = StdRng::seed_from_u64(7);
    let gpu = wgpu_device();
    let (vocab, n_ctx, c, t) = (1000usize, 64usize, 48usize, 9usize);
    let wte = rand_vec(&mut rng, vocab * c);
    let wpe = rand_vec(&mut rng, n_ctx * c);
    let ids: Vec<u32> = (0..t).map(|_| rng.random_range(0..vocab as u32)).collect();
    let run = |dev: &Device| {
        let wte_t = Tensor::from_f32(&wte, [vocab, c], dev).unwrap();
        let wpe_t = Tensor::from_f32(&wpe, [n_ctx, c], dev).unwrap();
        let ids_t = Tensor::from_u32(&ids, [t], dev).unwrap();
        ops::embedding(&ids_t, &wte_t, Some(&wpe_t), 3)
            .unwrap()
            .to_vec_f32()
            .unwrap()
    };
    assert_close(&run(&Device::Cpu), &run(&gpu), "embedding");
}

#[test]
fn chunked_embedding_and_lm_head_parity() {
    let mut rng = StdRng::seed_from_u64(9);
    let gpu = wgpu_device();
    let (vocab, n_ctx, c, t, chunk_rows) = (100usize, 32usize, 16usize, 11usize, 32usize);
    let wte = rand_vec(&mut rng, vocab * c);
    let wpe = rand_vec(&mut rng, n_ctx * c);
    let ids: Vec<u32> = (0..t).map(|_| rng.random_range(0..vocab as u32)).collect();
    let h = rand_vec(&mut rng, 3 * c);

    let make_chunks = |dev: &Device| -> Vec<Tensor> {
        let mut chunks = Vec::new();
        let mut start = 0;
        while start < vocab {
            let rows = chunk_rows.min(vocab - start);
            chunks.push(
                Tensor::from_f32(&wte[start * c..(start + rows) * c], [rows, c], dev).unwrap(),
            );
            start += rows;
        }
        chunks
    };
    let run = |dev: &Device| {
        let chunks = make_chunks(dev);
        let wpe_t = Tensor::from_f32(&wpe, [n_ctx, c], dev).unwrap();
        let ids_t = Tensor::from_u32(&ids, [t], dev).unwrap();
        let emb = ops::embedding_chunked(&ids_t, &chunks, chunk_rows, Some(&wpe_t), 2)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let ht = Tensor::from_f32(&h, [3, c], dev).unwrap();
        let logits = ops::matmul_chunked_transb(&ht, &chunks, 1.0)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        (emb, logits)
    };
    let (cpu_emb, cpu_logits) = run(&Device::Cpu);
    let (gpu_emb, gpu_logits) = run(&gpu);
    assert_close(&cpu_emb, &gpu_emb, "chunked embedding");
    assert_close(&cpu_logits, &gpu_logits, "chunked lm head");

    // Chunked must equal unchunked exactly (CPU).
    let wte_t = Tensor::from_f32(&wte, [vocab, c], &Device::Cpu).unwrap();
    let wpe_t = Tensor::from_f32(&wpe, [n_ctx, c], &Device::Cpu).unwrap();
    let ids_t = Tensor::from_u32(&ids, [t], &Device::Cpu).unwrap();
    let single = ops::embedding(&ids_t, &wte_t, Some(&wpe_t), 2)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    assert_eq!(single, cpu_emb, "chunked embedding != unchunked");
    let ht = Tensor::from_f32(&h, [3, c], &Device::Cpu).unwrap();
    let full = ops::matmul(
        &ht,
        &wte_t,
        None,
        MatmulSpec {
            trans_b: true,
            ..Default::default()
        },
    )
    .unwrap()
    .to_vec_f32()
    .unwrap();
    assert_close(&full, &cpu_logits, "chunked vs full lm head");
}

#[test]
fn split_merge_heads_parity() {
    let mut rng = StdRng::seed_from_u64(8);
    let gpu = wgpu_device();
    let (t, c, h) = (21usize, 48usize, 12usize);
    let qkv = rand_vec(&mut rng, t * 3 * c);
    let run = |dev: &Device| {
        let qkv_t = Tensor::from_f32(&qkv, [t, 3 * c], dev).unwrap();
        let (q, k, v) = ops::split_heads(&qkv_t, h).unwrap();
        let merged = ops::merge_heads(&q).unwrap();
        (
            q.to_vec_f32().unwrap(),
            k.to_vec_f32().unwrap(),
            v.to_vec_f32().unwrap(),
            merged.to_vec_f32().unwrap(),
        )
    };
    let (cq, ck, cv, cm) = run(&Device::Cpu);
    let (gq, gk, gv, gm) = run(&gpu);
    assert_close(&cq, &gq, "split_heads q");
    assert_close(&ck, &gk, "split_heads k");
    assert_close(&cv, &gv, "split_heads v");
    assert_close(&cm, &gm, "merge_heads");
    // merge(split(x).q) must equal the q columns of the input.
    for tt in 0..t {
        for cc in 0..c {
            assert_eq!(
                cm[tt * c + cc],
                qkv[tt * 3 * c + cc],
                "merge/split roundtrip"
            );
        }
    }
}
