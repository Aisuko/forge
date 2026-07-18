//! End-to-end GPT-2 acceptance gates (roadmap v3, Stage 6):
//!   - CPU vs WGPU last-position logits within 5e-3 abs
//!   - greedy continuations identical on both backends
//!   - (when tests/data/hf_golden.json exists) logits within 1e-2 abs of
//!     HF transformers, and identical greedy tokens
//!
//! Requires models/gpt2/ (scripts/download_gpt2.sh); skipped when absent.

use forge::{Device, Gpt2, Gpt2Config, Gpt2Tokenizer, Sampling};

const PROMPT: &str = "Hello, my dog is cute";
const GREEDY_TOKENS: usize = 15;

#[derive(serde::Deserialize)]
struct Golden {
    prompt: String,
    prompt_ids: Vec<u32>,
    logits_last: Vec<f32>,
    greedy_ids: Vec<u32>,
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn greedy_ids(model: &Gpt2, tok: &Gpt2Tokenizer, n: usize) -> Vec<u32> {
    // Re-encode inside so both backends start identically.
    let mut ids = tok.encode(PROMPT).unwrap();
    for _ in 0..n {
        let logits = model.logits_last(&ids).unwrap();
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        ids.push(next);
    }
    ids
}

#[test]
fn gpt2_cpu_wgpu_parity_and_golden() {
    if !std::path::Path::new("models/gpt2/model.safetensors").exists() {
        eprintln!("skipping: models/gpt2 not downloaded");
        return;
    }
    let tok = Gpt2Tokenizer::from_dir("models/gpt2").unwrap();
    let config = Gpt2Config::from_json("models/gpt2/config.json").unwrap();
    let prompt_ids = tok.encode(PROMPT).unwrap();

    println!("loading CPU model...");
    let cpu_model = Gpt2::from_safetensors(
        "models/gpt2/model.safetensors",
        config.clone(),
        &Device::Cpu,
    )
    .unwrap();
    let cpu_logits = cpu_model.logits_last(&prompt_ids).unwrap();

    println!("loading WGPU model...");
    let gpu_dev = Device::wgpu().unwrap();
    println!("wgpu adapter: {}", gpu_dev.describe());
    let gpu_model =
        Gpt2::from_safetensors("models/gpt2/model.safetensors", config, &gpu_dev).unwrap();
    let gpu_logits = gpu_model.logits_last(&prompt_ids).unwrap();

    let d = max_abs_diff(&cpu_logits, &gpu_logits);
    println!("CPU vs WGPU last-logits max abs diff: {d:.2e}");
    assert!(d <= 5e-3, "CPU/WGPU logit divergence {d} exceeds 5e-3");

    let cpu_greedy = greedy_ids(&cpu_model, &tok, GREEDY_TOKENS);
    let gpu_greedy = greedy_ids(&gpu_model, &tok, GREEDY_TOKENS);
    println!("CPU greedy: {:?}", tok.decode(&cpu_greedy));
    println!("GPU greedy: {:?}", tok.decode(&gpu_greedy));
    assert_eq!(
        cpu_greedy, gpu_greedy,
        "greedy tokens differ across backends"
    );

    // Sampling API smoke test (same seed => deterministic).
    let sampled = gpu_model
        .generate(
            &tok,
            PROMPT,
            10,
            Sampling::TopK {
                k: 40,
                temperature: 0.8,
                seed: 7,
            },
        )
        .unwrap();
    println!("sampled: {sampled:?}");
    assert!(sampled.starts_with(PROMPT));

    // Golden comparison against HF transformers, when available.
    let golden_path = "tests/data/hf_golden.json";
    if let Ok(text) = std::fs::read_to_string(golden_path) {
        let golden: Golden = serde_json::from_str(&text).unwrap();
        assert_eq!(golden.prompt, PROMPT, "golden file for different prompt");
        assert_eq!(prompt_ids, golden.prompt_ids, "tokenizer disagrees with HF");
        let dg = max_abs_diff(&cpu_logits, &golden.logits_last);
        println!("CPU vs HF logits max abs diff: {dg:.2e}");
        assert!(dg <= 1e-2, "CPU vs HF logit divergence {dg} exceeds 1e-2");
        let n = golden.greedy_ids.len().min(cpu_greedy.len());
        assert_eq!(
            &cpu_greedy[..n],
            &golden.greedy_ids[..n],
            "greedy tokens disagree with HF"
        );
        println!("HF golden checks passed");
    } else {
        eprintln!("note: {golden_path} absent; HF golden checks skipped");
    }
}
