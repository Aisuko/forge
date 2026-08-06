//! One-shot `.safetensors` -> `.fzm` q4 converter for shipped checkpoints.
//!
//! No `--features train` needed: `Gpt2::from_safetensors`/`save_fzm` are
//! plain inference-side code, unlike `fzm_bench`'s accuracy check.
//!
//! ```bash
//! cargo run --release --example to_fzm -- \
//!   --config assets/shakespeare_char/config.json \
//!   --in assets/shakespeare_char/model.safetensors \
//!   --out assets/shakespeare_char/model.fzm
//! ```

use forge::{Device, Gpt2, Gpt2Config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let config_path = get("--config").expect("--config <path to config.json>");
    let input = get("--in").expect("--in <path to .safetensors>");
    let output = get("--out").expect("--out <path to .fzm>");

    let config = Gpt2Config::from_json(&config_path)?;
    let model = Gpt2::from_safetensors(&input, config, &Device::Cpu)?;
    model.save_fzm(&output)?;

    let in_size = std::fs::metadata(&input)?.len();
    let out_size = std::fs::metadata(&output)?.len();
    println!(
        "{input} ({in_size} B) -> {output} ({out_size} B), {:.1}x smaller",
        in_size as f64 / out_size as f64
    );
    Ok(())
}
