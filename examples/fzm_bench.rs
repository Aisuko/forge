//! Baseline-then-compare accuracy check for the `.fzm` q4 checkpoint format.
//!
//! Loads a safetensors checkpoint (`assets/shakespeare_char` by default),
//! scores it with the same deterministic strided-window loss
//! `train_shakespeare --eval-only` uses, quantizes and saves it to `.fzm`,
//! reloads that `.fzm` file, and scores it the same way. This is the
//! accuracy check the `6_aug` checkpoint-format plan called for before
//! trusting `.fzm` q4 for anything beyond this one model.
//!
//! ```bash
//! cargo run --release --features train --example fzm_bench
//! ```

use forge::{CharTokenizer, Device, Gpt2, Gpt2Config, Tokenizer as _};

/// Fraction of the corpus held out for validation — matches
/// `train_shakespeare`'s `VAL_FRACTION` and nanoGPT's split.
const VAL_FRACTION: f64 = 0.1;

/// Deterministic mean loss over evenly spaced windows covering the whole
/// split. Copied from `train_shakespeare.rs::eval_loss_strided` rather than
/// shared — it's 15 lines and pulling it into the library for one example
/// would be the wrong direction of dependency.
fn eval_loss_strided(model: &Gpt2, ids: &[u32], seq_len: usize, max_windows: usize) -> f32 {
    if ids.len() <= seq_len + 1 {
        return f32::NAN;
    }
    let usable = ids.len() - seq_len - 1;
    let n = (usable / seq_len + 1).min(max_windows).max(1);
    let stride = if n > 1 { usable / (n - 1) } else { 0 };
    let mut total = 0.0f32;
    for i in 0..n {
        let s = i * stride;
        total += model
            .loss(&ids[s..s + seq_len], &ids[s + 1..s + seq_len + 1])
            .unwrap();
    }
    total / n as f32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let model_dir = get("--model").unwrap_or_else(|| "assets/shakespeare_char".into());
    let backend = get("--backend").unwrap_or_else(|| "cpu".into());
    let eval_windows: usize =
        get("--eval-windows").map_or(512, |s| s.parse().expect("--eval-windows"));
    let fzm_path = get("--out").unwrap_or_else(|| "checkpoints/shakespeare_char.fzm".into());
    let log_path = get("--log").unwrap_or_else(|| "checkpoints/fzm_bench_report.json".into());

    let device = match backend.as_str() {
        "cpu" => Device::Cpu,
        "wgpu" => Device::wgpu()?,
        other => return Err(format!("unknown backend {other:?} (use cpu|wgpu)").into()),
    };
    println!("device: {}", device.describe());

    let dir = std::path::Path::new(&model_dir);
    let config = Gpt2Config::from_json(dir.join("config.json"))?;
    let seq_len = config.n_ctx;

    let text = std::fs::read_to_string("data/tinyshakespeare.txt").map_err(|e| {
        format!("cannot read data/tinyshakespeare.txt ({e}) — run scripts/download_shakespeare.sh")
    })?;
    let tok = CharTokenizer::from_corpus(&text);
    if tok.vocab_size() != config.vocab_size {
        return Err(format!(
            "corpus vocab ({}) != checkpoint vocab ({}) — wrong --model?",
            tok.vocab_size(),
            config.vocab_size
        )
        .into());
    }
    let ids = tok.encode(&text)?;
    let split = ((ids.len() as f64) * (1.0 - VAL_FRACTION)) as usize;
    let (train_ids, val_ids) = ids.split_at(split);
    println!(
        "corpus: {} char tokens (vocab {}) — {} train / {} val",
        ids.len(),
        tok.vocab_size(),
        train_ids.len(),
        val_ids.len()
    );

    let safetensors_path = dir.join("model.safetensors");
    println!(
        "\n== safetensors baseline: {} ==",
        safetensors_path.display()
    );
    let st_start = std::time::Instant::now();
    let st_model = Gpt2::from_safetensors(&safetensors_path, config.clone(), &device)?;
    let st_load_s = st_start.elapsed().as_secs_f32();
    let st_train = eval_loss_strided(&st_model, train_ids, seq_len, eval_windows);
    let st_val = eval_loss_strided(&st_model, val_ids, seq_len, eval_windows);
    let st_size = std::fs::metadata(&safetensors_path)?.len();
    println!(
        "train {st_train:.4}  val {st_val:.4}  ({} windows, {:.1}s load, {:.2} MB)",
        eval_windows,
        st_load_s,
        st_size as f64 / 1e6
    );

    println!("\n== quantizing to .fzm q4: {fzm_path} ==");
    if let Some(parent) = std::path::Path::new(&fzm_path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    st_model.save_fzm(&fzm_path)?;
    let fzm_size = std::fs::metadata(&fzm_path)?.len();

    println!("== .fzm reload ==");
    let fzm_start = std::time::Instant::now();
    let fzm_model = Gpt2::from_fzm(&fzm_path, config.clone(), &device)?;
    let fzm_load_s = fzm_start.elapsed().as_secs_f32();
    let fzm_train = eval_loss_strided(&fzm_model, train_ids, seq_len, eval_windows);
    let fzm_val = eval_loss_strided(&fzm_model, val_ids, seq_len, eval_windows);
    println!(
        "train {fzm_train:.4}  val {fzm_val:.4}  ({} windows, {:.1}s load, {:.2} MB)",
        eval_windows,
        fzm_load_s,
        fzm_size as f64 / 1e6
    );

    let val_delta = fzm_val - st_val;
    println!("\n== summary ==");
    println!(
        "size: {:.2} MB -> {:.2} MB ({:.1}x smaller)",
        st_size as f64 / 1e6,
        fzm_size as f64 / 1e6,
        st_size as f64 / fzm_size as f64
    );
    println!("val loss: {st_val:.4} -> {fzm_val:.4}  (delta {val_delta:+.4})");

    let report = format!(
        "{{\n  \"model\": {model_dir:?},\n  \"eval_windows\": {eval_windows},\n  \"safetensors\": {{\"path\": {:?}, \"bytes\": {st_size}, \"train_loss\": {st_train:.4}, \"val_loss\": {st_val:.4}}},\n  \"fzm\": {{\"path\": {fzm_path:?}, \"bytes\": {fzm_size}, \"train_loss\": {fzm_train:.4}, \"val_loss\": {fzm_val:.4}}},\n  \"val_loss_delta\": {val_delta:.4},\n  \"size_ratio\": {:.3}\n}}\n",
        safetensors_path.display(),
        st_size as f64 / fzm_size as f64
    );
    std::fs::write(&log_path, &report)?;
    println!("\nreport written to {log_path}");

    Ok(())
}
