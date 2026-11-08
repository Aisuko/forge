use forge::serialization::fzm::{self, FzmDtype};
use forge::{Device, EncoderDecoder, EncoderDecoderConfig};

fn tiny() -> EncoderDecoderConfig {
    EncoderDecoderConfig {
        vocab_size: 37,
        d_model: 16,
        n_encoder_layer: 2,
        n_decoder_layer: 2,
        n_head: 2,
        d_ff: 32,
        max_source: 16,
        max_target: 8,
        layer_norm_epsilon: 1e-5,
        pad_id: 0,
        bos_id: 1,
        eos_id: 2,
    }
}

fn wgpu_device() -> Device {
    static DEV: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
    DEV.get_or_init(|| Device::wgpu().expect("wgpu device"))
        .clone()
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length mismatch");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol,
            "{what}: element {i} differs: {x} vs {y}"
        );
    }
}

fn temp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("forge_encdec_{name}_{}", std::process::id()))
}

const SRC: [u32; 6] = [5, 9, 3, 30, 11, 7];

#[test]
fn config_validates_and_round_trips_json() {
    let c = tiny();
    let back = EncoderDecoderConfig::from_json_str(&serde_json::to_string(&c).unwrap()).unwrap();
    assert_eq!(back, c);
    let bad = EncoderDecoderConfig { n_head: 3, ..c };
    assert!(bad.validate().is_err());
    let bad = EncoderDecoderConfig { eos_id: 99, ..c };
    assert!(bad.validate().is_err());
}

#[test]
fn param_count_matches_saved_entries() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 1).unwrap();
    let n: usize = model
        .save_entries()
        .unwrap()
        .iter()
        .map(|e| e.2.len())
        .sum();
    assert_eq!(n, model.n_params());
}

#[test]
fn incremental_decoding_matches_full_recompute() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 2).unwrap();
    let memory = model.encode(&SRC, &[SRC.len() as u32], SRC.len()).unwrap();
    let target = [1u32, 4, 8, 15];

    let mut full = model.start(&memory, 0).unwrap();
    let all = model.step(&target, &mut full).unwrap();
    assert_eq!(full.len(), target.len());

    let mut inc = model.start(&memory, 0).unwrap();
    let mut last = Vec::new();
    for &t in &target {
        last = model.step(&[t], &mut inc).unwrap();
    }
    assert_close(&all, &last, 1e-4, "last-step logits");
}

#[test]
fn padding_does_not_change_unpadded_rows() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 3).unwrap();
    let short = [12u32, 6, 21];
    let seq = SRC.len();
    let mut ids = SRC.to_vec();
    ids.extend_from_slice(&short);
    ids.extend(std::iter::repeat_n(0u32, seq - short.len()));
    let batched = model
        .encode(&ids, &[seq as u32, short.len() as u32], seq)
        .unwrap();
    let alone = model
        .encode(&short, &[short.len() as u32], short.len())
        .unwrap();

    let d = tiny().d_model;
    let rows = batched.hidden().to_vec_f32().unwrap();
    let want = alone.hidden().to_vec_f32().unwrap();
    assert_close(
        &rows[seq * d..(seq + short.len()) * d],
        &want,
        1e-4,
        "encoder rows",
    );

    let mut a = model.start(&batched, 1).unwrap();
    let mut b = model.start(&alone, 0).unwrap();
    let la = model.step(&[1, 4], &mut a).unwrap();
    let lb = model.step(&[1, 4], &mut b).unwrap();
    assert_close(&la, &lb, 1e-4, "cross-attention over padded memory");
}

#[test]
fn rejects_bad_inputs() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 4).unwrap();
    assert!(model.encode(&SRC, &[7], SRC.len()).is_err());
    assert!(model.encode(&SRC, &[0], SRC.len()).is_err());
    assert!(model.encode(&[99], &[1], 1).is_err());
    assert!(model.encode(&[3; 17], &[17], 17).is_err());
    let memory = model.encode(&SRC, &[6], 6).unwrap();
    let mut state = model.start(&memory, 0).unwrap();
    assert!(model.step(&[1; 9], &mut state).is_err());
    assert!(model.start(&memory, 1).is_err());
}

#[test]
fn greedy_generation_is_deterministic_and_bounded() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 5).unwrap();
    let a = model.generate_greedy(&SRC, 5).unwrap();
    let b = model.generate_greedy(&SRC, 5).unwrap();
    assert_eq!(a, b);
    assert!(a.len() <= 5);
    assert!(!a.contains(&tiny().eos_id));
    assert!(model.generate_greedy(&SRC, 100).unwrap().len() <= tiny().max_target);
}

#[test]
fn fzm_v2_round_trip_reads_config_from_manifest() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 6).unwrap();
    let path = temp("f32.fzm");
    let manifest = model.manifest().unwrap();
    model
        .save_fzm_with(
            &path,
            &manifest,
            FzmDtype::F32,
            &[("tokenizer.json".into(), b"{}".to_vec())],
        )
        .unwrap();
    let file = fzm::load_fzm_file(&path).unwrap();
    assert_eq!(file.version, 2);
    assert_eq!(file.manifest.unwrap().architecture, "encoder_decoder");
    assert_eq!(file.assets.len(), 1);

    let back = EncoderDecoder::from_checkpoint(&path, None, &Device::Cpu).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(back.config(), model.config());
    assert_eq!(
        model.generate_greedy(&SRC, 6).unwrap(),
        back.generate_greedy(&SRC, 6).unwrap()
    );

    let q4 = temp("q4.fzm");
    model.save_fzm(&q4).unwrap();
    let back = EncoderDecoder::from_checkpoint(&q4, None, &Device::Cpu).unwrap();
    std::fs::remove_file(&q4).ok();
    let memory = back.encode(&SRC, &[6], 6).unwrap();
    assert!(
        memory
            .hidden()
            .to_vec_f32()
            .unwrap()
            .iter()
            .all(|x| x.is_finite())
    );
}

#[test]
fn safetensors_round_trip_needs_config() {
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 7).unwrap();
    let path = temp("model.safetensors");
    model.save_safetensors(&path).unwrap();
    assert!(EncoderDecoder::from_checkpoint(&path, None, &Device::Cpu).is_err());
    let back = EncoderDecoder::from_checkpoint(&path, Some(tiny()), &Device::Cpu).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(
        model.generate_greedy(&SRC, 6).unwrap(),
        back.generate_greedy(&SRC, 6).unwrap()
    );
}

#[test]
fn cpu_and_wgpu_agree() {
    let path = temp("parity.fzm");
    let cpu = EncoderDecoder::init_random(tiny(), &Device::Cpu, 8).unwrap();
    cpu.save_fzm_with(&path, &cpu.manifest().unwrap(), FzmDtype::F32, &[])
        .unwrap();
    let gpu = EncoderDecoder::from_checkpoint(&path, None, &wgpu_device()).unwrap();
    std::fs::remove_file(&path).ok();

    let short = [12u32, 6, 21, 0, 0, 0];
    let mut ids = SRC.to_vec();
    ids.extend_from_slice(&short);
    let a = cpu.encode(&ids, &[6, 3], 6).unwrap();
    let b = gpu.encode(&ids, &[6, 3], 6).unwrap();
    assert_close(
        &a.hidden().to_vec_f32().unwrap(),
        &b.hidden().to_vec_f32().unwrap(),
        1e-3,
        "encoder hidden",
    );

    for row in 0..2 {
        let mut sa = cpu.start(&a, row).unwrap();
        let mut sb = gpu.start(&b, row).unwrap();
        for ids in [&[1u32, 4][..], &[8], &[15]] {
            let la = cpu.step(ids, &mut sa).unwrap();
            let lb = gpu.step(ids, &mut sb).unwrap();
            assert_close(&la, &lb, 1e-3, "decoder logits");
        }
    }
    assert_eq!(
        cpu.generate_greedy(&SRC, 6).unwrap(),
        gpu.generate_greedy(&SRC, 6).unwrap()
    );
}

#[test]
fn q4_checkpoint_runs_packed_on_wgpu_and_matches_cpu() {
    let config = EncoderDecoderConfig {
        d_model: 64,
        d_ff: 128,
        n_head: 4,
        ..tiny()
    };
    let path = temp("q4parity.fzm");
    EncoderDecoder::init_random(config, &Device::Cpu, 9)
        .unwrap()
        .save_fzm(&path)
        .unwrap();
    let cpu = EncoderDecoder::from_checkpoint(&path, None, &Device::Cpu).unwrap();
    let gpu = EncoderDecoder::from_checkpoint(&path, None, &wgpu_device()).unwrap();
    std::fs::remove_file(&path).ok();

    let a = cpu.encode(&SRC, &[6], 6).unwrap();
    let b = gpu.encode(&SRC, &[6], 6).unwrap();
    let mut sa = cpu.start(&a, 0).unwrap();
    let mut sb = gpu.start(&b, 0).unwrap();
    for ids in [&[1u32, 4][..], &[8], &[15]] {
        let la = cpu.step(ids, &mut sa).unwrap();
        let lb = pollster::block_on(gpu.step_async(ids, &mut sb)).unwrap();
        assert_close(&la, &lb, 1e-3, "q4 decoder logits");
    }
    assert_eq!(
        cpu.generate_greedy(&SRC, 6).unwrap(),
        pollster::block_on(gpu.generate_greedy_async(&SRC, 6)).unwrap()
    );
}

#[test]
fn async_step_matches_sync_step() {
    let gpu = EncoderDecoder::init_random(tiny(), &wgpu_device(), 10).unwrap();
    let memory = gpu.encode(&SRC, &[6], 6).unwrap();
    let mut a = gpu.start(&memory, 0).unwrap();
    let mut b = gpu.start(&memory, 0).unwrap();
    for ids in [&[1u32, 4, 8][..], &[15]] {
        let la = gpu.step(ids, &mut a).unwrap();
        let lb = pollster::block_on(gpu.step_async(ids, &mut b)).unwrap();
        assert_eq!(la, lb);
    }
    assert_eq!(
        gpu.generate_greedy(&SRC, 6).unwrap(),
        pollster::block_on(gpu.generate_greedy_async(&SRC, 6)).unwrap()
    );
}

#[test]
fn loaders_reject_the_wrong_architecture() {
    let path = temp("arch.fzm");
    let model = EncoderDecoder::init_random(tiny(), &Device::Cpu, 11).unwrap();
    model.save_fzm(&path).unwrap();
    let err = forge::Gpt2::from_checkpoint(&path, forge::Gpt2Config::gpt2(), &Device::Cpu)
        .err()
        .unwrap()
        .to_string();
    assert!(err.contains("architecture"), "{err}");

    let gpt2 = forge::Gpt2::init_random(
        forge::Gpt2Config {
            n_layer: 1,
            n_head: 2,
            n_embd: 16,
            n_ctx: 8,
            vocab_size: 37,
            layer_norm_epsilon: 1e-5,
            eos_token_id: None,
        },
        &Device::Cpu,
        1,
    )
    .unwrap();
    gpt2.save_fzm(&path).unwrap();
    let err = EncoderDecoder::from_checkpoint(&path, Some(tiny()), &Device::Cpu)
        .err()
        .unwrap()
        .to_string();
    std::fs::remove_file(&path).ok();
    assert!(err.contains("architecture"), "{err}");
}
