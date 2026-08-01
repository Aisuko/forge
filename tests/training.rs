//! A scaled-down random-init model
//! overfits a fixed sequence (loss drops sharply) on both backends, and
//! checkpoints round-trip bit-identically.

use forge::optim::{AdamW, AdamWOpts};
use forge::{Device, Gpt2, Gpt2Config};

fn tiny_config() -> Gpt2Config {
    Gpt2Config {
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        n_ctx: 32,
        vocab_size: 61,
        layer_norm_epsilon: 1e-5,
        eos_token_id: None,
    }
}

fn fixed_batch() -> (Vec<u32>, Vec<u32>) {
    // A short repeating "language" the model can memorize quickly.
    let seq: Vec<u32> = (0..25).map(|i| [3u32, 14, 15, 9, 2, 6][i % 6]).collect();
    (seq[..24].to_vec(), seq[1..25].to_vec())
}

fn overfit(device: &Device) -> (f32, f32) {
    let mut model = Gpt2::init_random(tiny_config(), device, 5).unwrap();
    let specs = model.param_specs();
    let mut opt = {
        let params = model.params().unwrap();
        let with_decay: Vec<(&forge::Tensor, bool)> = params
            .iter()
            .zip(&specs)
            .map(|(p, (_, d))| (*p, *d))
            .collect();
        AdamW::new(
            &with_decay,
            AdamWOpts {
                lr: 1e-2,
                weight_decay: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let (input, target) = fixed_batch();
    let mut first = None;
    let mut last = 0.0;
    for _ in 0..60 {
        let (loss, grads) = model.loss_grads(&input, &target, 0.0, 0).unwrap();
        first.get_or_insert(loss);
        last = loss;
        let mut params = model.params_mut().unwrap();
        opt.step(&mut params, &grads).unwrap();
    }
    (first.unwrap(), last)
}

#[test]
fn overfit_fixed_sequence_cpu() {
    let (first, last) = overfit(&Device::Cpu);
    println!("CPU overfit: {first:.3} -> {last:.3}");
    assert!(
        last < first * 0.25 && last < 1.0,
        "loss did not collapse: {first:.3} -> {last:.3}"
    );
}

#[test]
fn overfit_fixed_sequence_wgpu() {
    let gpu = Device::wgpu().expect("wgpu adapter required");
    let (first, last) = overfit(&gpu);
    println!("WGPU overfit: {first:.3} -> {last:.3}");
    assert!(
        last < first * 0.25 && last < 1.0,
        "loss did not collapse: {first:.3} -> {last:.3}"
    );
}

#[test]
fn checkpoint_roundtrip() {
    let dir = std::env::temp_dir().join("forge_ckpt_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny.safetensors");

    let model = Gpt2::init_random(tiny_config(), &Device::Cpu, 11).unwrap();
    model.save_safetensors(&path).unwrap();
    let reloaded = Gpt2::from_safetensors(&path, tiny_config(), &Device::Cpu).unwrap();

    // Bit-identical parameters...
    let a = model.params().unwrap();
    let b = reloaded.params().unwrap();
    for ((name, _), (x, y)) in model.param_specs().iter().zip(a.iter().zip(&b)) {
        assert_eq!(
            x.to_vec_f32().unwrap(),
            y.to_vec_f32().unwrap(),
            "checkpoint mismatch in {name}"
        );
    }
    // ...and identical loss.
    let (input, target) = fixed_batch();
    let (l1, _) = model.loss_grads(&input, &target, 0.0, 0).unwrap();
    let (l2, _) = reloaded.loss_grads(&input, &target, 0.0, 0).unwrap();
    assert_eq!(l1, l2, "loss changed after checkpoint round trip");
    std::fs::remove_file(&path).ok();
}
