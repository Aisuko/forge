//! Text generation on either backend, with either tokenizer.
//!
//! Usage:
//!   cargo run --release --example generate -- \
//!       [--model DIR] [--backend cpu|wgpu] [--prompt "..."] \
//!       [--tokens N] [--topk K] [--temp T]
//!
//! `--model` is a directory holding `model.safetensors`, `config.json`, and
//! `vocab.json` (plus `merges.txt` for a BPE model). It defaults to
//! `models/gpt2/` — see scripts/download_gpt2.sh. The 43 MB character-level
//! Shakespeare model ships in the repo:
//!
//!   cargo run --release --example generate -- \
//!       --model assets/shakespeare_char --prompt "ROMEO:"

use forge::{AnyTokenizer, Device, Gpt2, Gpt2Config, Sampling, Tokenizer as _};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let dir = get("--model").unwrap_or_else(|| "models/gpt2".into());
    let backend = get("--backend").unwrap_or_else(|| "wgpu".into());
    let tokens: usize = get("--tokens").map_or(30, |s| s.parse().expect("--tokens"));
    let topk: Option<usize> = get("--topk").map(|s| s.parse().expect("--topk"));
    let temp: f32 = get("--temp").map_or(0.8, |s| s.parse().expect("--temp"));

    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "wgpu" => Device::wgpu()?,
        other => return Err(format!("unknown backend {other:?} (use cpu|wgpu)").into()),
    };
    println!("device: {}", device.describe());

    let dir = std::path::Path::new(&dir);
    let config =
        Gpt2Config::from_json(dir.join("config.json")).unwrap_or_else(|_| Gpt2Config::gpt2());
    let start = std::time::Instant::now();
    let model = Gpt2::from_safetensors(dir.join("model.safetensors"), config, &device)?;
    // Picks BPE or char by which sidecar files are present.
    let tokenizer = AnyTokenizer::from_dir(dir)?;
    println!(
        "loaded {} in {:.1}s — {} vocab, {} tokens",
        dir.display(),
        start.elapsed().as_secs_f32(),
        tokenizer.kind(),
        tokenizer.vocab_size()
    );

    // A char model has no idea what a lighthouse keeper is.
    let prompt = get("--prompt").unwrap_or_else(|| match tokenizer.kind() {
        "char" => "ROMEO:".into(),
        _ => "The old lighthouse keeper".into(),
    });
    let sampling = match topk {
        Some(k) => Sampling::TopK {
            k,
            temperature: temp,
            seed: 42,
        },
        None => Sampling::Greedy,
    };
    let start = std::time::Instant::now();
    let text = model.generate(&tokenizer, &prompt, tokens, sampling)?;
    let dt = start.elapsed().as_secs_f32();
    println!(
        "--- output ({tokens} tokens, {dt:.1}s, {:.2} tok/s) ---",
        tokens as f32 / dt
    );
    println!("{text}");
    Ok(())
}
