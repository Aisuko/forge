//! KV-cache decode produces token-identical
//! greedy continuations to the no-cache path for >= 64 generated tokens,
//! on both backends.
//!
//! Requires models/gpt2/ (scripts/download_gpt2.sh); skipped when absent.

use forge::{Device, Gpt2, Gpt2Config, Gpt2Tokenizer};

const PROMPT: &str = "The meaning of life is";
const N_TOKENS: usize = 64;

fn greedy_nocache(model: &Gpt2, ids0: &[u32], n: usize) -> Vec<u32> {
    let mut ids = ids0.to_vec();
    for _ in 0..n {
        let logits = model.logits_last(&ids).unwrap();
        ids.push(argmax(&logits));
    }
    ids
}

fn greedy_cached(model: &Gpt2, ids0: &[u32], n: usize) -> Vec<u32> {
    let mut ids = ids0.to_vec();
    let mut cache = model.new_cache().unwrap();
    let mut logits = model.logits_step(&ids, &mut cache).unwrap();
    for _ in 0..n {
        let next = argmax(&logits);
        ids.push(next);
        logits = model.logits_step(&[next], &mut cache).unwrap();
    }
    ids
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

#[test]
fn kv_cache_matches_nocache() {
    if !std::path::Path::new("models/gpt2/model.safetensors").exists() {
        eprintln!("skipping: models/gpt2 not downloaded");
        return;
    }
    let tok = Gpt2Tokenizer::from_dir("models/gpt2").unwrap();
    let config = Gpt2Config::from_json("models/gpt2/config.json").unwrap();
    let ids0 = tok.encode(PROMPT).unwrap();

    for device in [Device::Cpu, Device::wgpu().unwrap()] {
        println!("device: {}", device.describe());
        let model =
            Gpt2::from_safetensors("models/gpt2/model.safetensors", config.clone(), &device)
                .unwrap();
        let nocache = greedy_nocache(&model, &ids0, N_TOKENS);
        let cached = greedy_cached(&model, &ids0, N_TOKENS);
        assert_eq!(
            nocache,
            cached,
            "KV-cache decode diverged from no-cache on {}\nnocache: {:?}\ncached:  {:?}",
            device.describe(),
            tok.decode(&nocache),
            tok.decode(&cached),
        );
        println!(
            "greedy ({N_TOKENS} tokens) identical: {:?}",
            tok.decode(&cached)
        );
    }
}
