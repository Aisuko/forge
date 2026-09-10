use forge::autograd::Tape;
use forge::models::encoder::{Encoder, EncoderConfig};
use forge::ops::{self, MatmulSpec};
use forge::optim::{AdamW, AdamWOpts};
use forge::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn tiny_config() -> EncoderConfig {
    EncoderConfig {
        vocab_size: 64,
        d_model: 32,
        embed_rank: 16,
        n_layer: 2,
        n_head: 4,
        d_ff: 64,
        max_seq: 16,
        embed_dim: 16,
        layer_norm_epsilon: 1e-5,
        embed_init_std: 0.05,
    }
}

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("forge_encoder_{name}"));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn wgpu_device() -> Device {
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Device::wgpu().expect("wgpu device"))
        .clone()
}

#[test]
fn safetensors_round_trip_is_exact() {
    let cfg = tiny_config();
    let model = Encoder::init_random(cfg, &Device::Cpu, 1).unwrap();
    let dir = scratch("safetensors");
    let path = dir.join("enc.safetensors");
    model.save_checkpoint(&path).unwrap();

    let back = Encoder::from_checkpoint(&path, cfg, &Device::Cpu).unwrap();
    for ((name, _), (a, b)) in model
        .param_specs()
        .iter()
        .zip(model.params().unwrap().iter().zip(back.params().unwrap()))
    {
        assert_eq!(
            a.to_vec_f32().unwrap(),
            b.to_vec_f32().unwrap(),
            "{name} changed across a safetensors round trip"
        );
    }
}

#[test]
fn fzm_round_trip_is_a_fixed_point() {
    let cfg = tiny_config();
    let model = Encoder::init_random(cfg, &Device::Cpu, 2).unwrap();
    let dir = scratch("fzm");
    let path = dir.join("enc.fzm");
    model.save_fzm(&path).unwrap();

    let once = Encoder::from_checkpoint(&path, cfg, &Device::Cpu).unwrap();
    let path2 = dir.join("enc2.fzm");
    once.save_fzm(&path2).unwrap();
    let twice = Encoder::from_fzm(&path2, cfg, &Device::Cpu).unwrap();

    for ((name, _), (a, b)) in model
        .param_specs()
        .iter()
        .zip(once.params().unwrap().iter().zip(twice.params().unwrap()))
    {
        assert_eq!(
            a.to_vec_f32().unwrap(),
            b.to_vec_f32().unwrap(),
            "{name} is not a fixed point of fzm q4"
        );
    }
    let bytes = std::fs::metadata(&path).unwrap().len();
    let params = cfg.n_params() as u64;
    assert!(
        bytes < params,
        "{bytes} bytes for {params} parameters is not a 4-bit checkpoint"
    );
}

#[test]
fn embedding_ignores_padding() {
    let cfg = tiny_config();
    let model = Encoder::init_random(cfg, &Device::Cpu, 3).unwrap();
    let real = [5u32, 9, 2, 7];

    let embed = |seq: usize, pad: u32| {
        let mut ids = real.to_vec();
        ids.resize(seq, pad);
        model
            .embed(&ids, &[real.len() as u32], seq)
            .unwrap()
            .to_vec_f32()
            .unwrap()
    };
    let short = embed(6, 0);
    let long = embed(12, 0);
    let other_pad = embed(12, 63);

    for (i, ((a, b), c)) in short.iter().zip(&long).zip(&other_pad).enumerate() {
        assert!(
            (a - b).abs() < 1e-5 && (a - c).abs() < 1e-5,
            "dim {i}: padding changed the embedding: {a} / {b} / {c}"
        );
    }
}

#[test]
fn embeddings_are_unit_vectors() {
    let cfg = tiny_config();
    let model = Encoder::init_random(cfg, &Device::Cpu, 4).unwrap();
    let ids: Vec<u32> = (0..24).map(|i| (i * 7 % 64) as u32).collect();
    let e = model
        .embed(&ids, &[8, 3, 8], 8)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    for (r, row) in e.chunks(cfg.embed_dim).enumerate() {
        let n: f32 = row.iter().map(|v| v * v).sum();
        assert!((n - 1.0).abs() < 1e-4, "row {r} has norm {n}");
    }
}

#[test]
fn matryoshka_prefixes_stay_normalized() {
    let cfg = tiny_config();
    let model = Encoder::init_random(cfg, &Device::Cpu, 5).unwrap();
    let ids: Vec<u32> = (0..16).map(|i| (i * 5 % 64) as u32).collect();
    let e = model.embed(&ids, &[8, 6], 8).unwrap().to_vec_f32().unwrap();
    for dim in [4usize, 8, 16] {
        let t = Encoder::truncate(&e, cfg.embed_dim, dim).unwrap();
        assert_eq!(t.len(), 2 * dim);
        for row in t.chunks(dim) {
            let n: f32 = row.iter().map(|v| v * v).sum();
            assert!((n - 1.0).abs() < 1e-4, "dim {dim} row norm {n}");
        }
    }
    assert!(Encoder::truncate(&e, cfg.embed_dim, 0).is_err());
    assert!(Encoder::truncate(&e, cfg.embed_dim, 17).is_err());
}

#[test]
fn cpu_and_wgpu_embed_identically() {
    let cfg = tiny_config();
    let gpu = wgpu_device();
    let cpu_model = Encoder::init_random(cfg, &Device::Cpu, 6).unwrap();
    let dir = scratch("portable");
    let path = dir.join("enc.fzm");
    cpu_model.save_fzm(&path).unwrap();

    let a = Encoder::from_fzm(&path, cfg, &Device::Cpu).unwrap();
    let b = Encoder::from_fzm(&path, cfg, &gpu).unwrap();
    let ids: Vec<u32> = (0..32).map(|i| (i * 13 % 64) as u32).collect();
    let key_len = [8u32, 5, 8, 2];
    let ea = a.embed(&ids, &key_len, 8).unwrap().to_vec_f32().unwrap();
    let eb = b.embed(&ids, &key_len, 8).unwrap().to_vec_f32().unwrap();
    for (i, (x, y)) in ea.iter().zip(&eb).enumerate() {
        assert!((x - y).abs() < 1e-4, "element {i}: cpu {x} vs wgpu {y}");
    }
}

fn infonce_step(
    model: &mut Encoder,
    opt: &mut AdamW,
    queries: &[u32],
    positives: &[u32],
    key_len: &[u32],
    seq: usize,
    scale: f32,
) -> f32 {
    let batch = key_len.len();
    let device = Device::Cpu;
    let mut tape = Tape::new();
    let pvars = model.leaves(&mut tape).unwrap();

    let fq = model
        .forward_tape(&mut tape, &pvars, queries, key_len, seq, 0.0, 0)
        .unwrap();
    let eq = model.pool_tape(&mut tape, &pvars, &fq).unwrap();
    let fp = model
        .forward_tape(&mut tape, &pvars, positives, key_len, seq, 0.0, 0)
        .unwrap();
    let ep = model.pool_tape(&mut tape, &pvars, &fp).unwrap();

    let sim = tape
        .matmul(
            &eq,
            &ep,
            None,
            MatmulSpec {
                trans_b: true,
                alpha: scale,
                ..Default::default()
            },
        )
        .unwrap();

    let tgt: Vec<u32> = (0..batch as u32).collect();
    let tgt_t = Tensor::from_u32(&tgt, [batch], &device).unwrap();
    let probs = ops::softmax(&sim.t, false, 0).unwrap();
    let nll = ops::gather_nll(&probs, &tgt_t)
        .unwrap()
        .to_vec_f32()
        .unwrap();
    let loss = nll.iter().sum::<f32>() / batch as f32;

    let dsim = ops::ce_bwd(&probs, &tgt_t, 1.0 / batch as f32).unwrap();
    let all = tape.backward(&sim, dsim).unwrap();
    let grads = model.collect_grads(all).unwrap();
    let mut params = model.params_mut().unwrap();
    let norm = opt.step(&mut params, &grads).unwrap();
    assert!(norm.is_finite(), "gradient norm went non-finite: {norm}");
    loss
}

#[test]
fn infonce_overfits_a_handful_of_pairs() {
    let cfg = tiny_config();
    let mut model = Encoder::init_random(cfg, &Device::Cpu, 7).unwrap();
    let (batch, seq) = (8usize, 6usize);

    let mut rng = StdRng::seed_from_u64(70);
    let mut draw = |n: usize| -> Vec<u32> { (0..n).map(|_| rng.random_range(0..64u32)).collect() };
    let queries = draw(batch * seq);
    let positives = draw(batch * seq);
    let key_len = vec![seq as u32; batch];

    let specs = model.param_specs();
    let params = model.params().unwrap();
    let reg: Vec<(&Tensor, bool)> = params
        .iter()
        .zip(&specs)
        .map(|(p, (_, decay))| (*p, *decay))
        .collect();
    let mut opt = AdamW::new(
        &reg,
        AdamWOpts {
            lr: 3e-3,
            weight_decay: 0.0,
            ..Default::default()
        },
    )
    .unwrap();
    drop(params);

    let scale = 20.0;
    let first = infonce_step(
        &mut model, &mut opt, &queries, &positives, &key_len, seq, scale,
    );
    let chance = (batch as f32).ln();
    assert!(
        first > 0.9 * chance,
        "step 0 loss {first} is already below chance {chance} — the pairs leak"
    );

    let mut last = first;
    for _ in 0..120 {
        last = infonce_step(
            &mut model, &mut opt, &queries, &positives, &key_len, seq, scale,
        );
        assert!(last.is_finite(), "loss went non-finite");
    }
    assert!(
        last < 0.1 * chance,
        "InfoNCE did not overfit: {first} -> {last} (chance {chance})"
    );
}
