//! Stage 11 gate reference (roadmap v4): native WGPU greedy continuation as
//! raw token ids, written to `tests/data/gate_expected.json` for the browser
//! demo to compare against. Mirrors `WasmGpt2::greedy_ids` exactly (fixed
//! length, no EOS early-stop).
//!
//! Defaults to the model the site actually ships, so opening the deployed
//! demo with `?gate` checks the browser against native WGPU token-for-token.
//!
//! ```bash
//! cargo run --release --example gate_tokens -- [--model DIR] [--backend wgpu] \
//!     [--prompt "..."] [--tokens N]
//! ```

use forge::{AnyTokenizer, Device, Gpt2, Gpt2Config, Tokenizer as _};

const OUT: &str = "tests/data/gate_expected.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let backend = get("--backend").unwrap_or_else(|| "wgpu".into());
    let model_dir = get("--model").unwrap_or_else(|| "assets/shakespeare_char".into());
    let max_new_tokens: usize = get("--tokens").map_or(30, |s| s.parse().expect("--tokens"));

    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "wgpu" => Device::wgpu()?,
        other => return Err(format!("unknown backend {other:?}").into()),
    };
    println!("device: {}", device.describe());

    let dir = std::path::Path::new(&model_dir);
    let config =
        Gpt2Config::from_json(dir.join("config.json")).unwrap_or_else(|_| Gpt2Config::gpt2());
    let model = Gpt2::from_safetensors(dir.join("model.safetensors"), config, &device)?;
    let tokenizer = AnyTokenizer::from_dir(dir)?;
    // A 65-character vocabulary cannot encode a prompt about lighthouses.
    let prompt = get("--prompt").unwrap_or_else(|| match tokenizer.kind() {
        "char" => "ROMEO:".into(),
        _ => "The old lighthouse keeper".into(),
    });

    let mut ids = tokenizer.encode(&prompt)?;
    let mut cache = model.new_cache()?;
    let mut logits = model.logits_step(&ids, &mut cache)?;
    for _ in 0..max_new_tokens {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        ids.push(next);
        if ids.len() >= model.config.n_ctx {
            break;
        }
        logits = model.logits_step(&[next], &mut cache)?;
    }

    let json = serde_json::json!({
        "model": model_dir,
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "backend": device.describe(),
        "ids": ids,
    });
    std::fs::write(OUT, serde_json::to_string_pretty(&json)?)?;
    println!(
        "wrote {OUT} ({} ids)",
        json["ids"].as_array().unwrap().len()
    );
    println!("text: {}", tokenizer.decode(&ids));
    Ok(())
}
