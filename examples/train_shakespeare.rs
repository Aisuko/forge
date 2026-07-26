//! Stage 10 (roadmap v4): train a from-scratch GPT-2-style model on Tiny
//! Shakespeare, with either the character-level vocabulary (nanoGPT's
//! `shakespeare_char`, 65 tokens — the default) or the GPT-2 BPE vocabulary.
//!
//! Gates: char mode targets nanoGPT's published **val loss ≈ 1.48**; BPE mode
//! keeps the original gate of smoothed loss falling from ~10.8 (= ln 50257,
//! random init) to < 4.0.
//!
//! ```bash
//! ./scripts/download_shakespeare.sh   # data/tinyshakespeare.txt
//! ./scripts/download_gpt2.sh          # BPE mode only (vocab/merges)
//!
//! # nanoGPT shakespeare_char defaults — 6L/6H/384d, 10.77M params, 43.1 MB
//! cargo run --release --example train_shakespeare -- --backend wgpu
//!
//! cargo run --release --example train_shakespeare -- --tokenizer bpe --steps 1500
//! ```

use std::io::Write as _;

use forge::optim::{AdamW, AdamWOpts};
use forge::{
    AnyTokenizer, CharTokenizer, Device, Gpt2, Gpt2Config, Gpt2Tokenizer, Sampling, Tokenizer as _,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Fraction of the corpus held out for validation, matching nanoGPT.
const VAL_FRACTION: f64 = 0.1;

struct Args {
    backend: String,
    tokenizer: String,
    steps: usize,
    seq_len: usize,
    accum: usize,
    lr: f32,
    beta2: f32,
    warmup: usize,
    cosine: bool,
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    dropout: f32,
    seed: u64,
    checkpoint: String,
    resume: bool,
    sample_every: usize,
    eval_every: usize,
    eval_batches: usize,
}

/// Per-tokenizer defaults. The char run reproduces nanoGPT's
/// `config/train_shakespeare_char.py`; the BPE run keeps the Stage 10 config
/// this example shipped with.
fn defaults(tokenizer: &str) -> Args {
    let char_mode = tokenizer == "char";
    Args {
        backend: "wgpu".into(),
        tokenizer: tokenizer.into(),
        steps: if char_mode { 5000 } else { 1500 },
        seq_len: 256,
        // Roadmap v4 keeps the op surface single-sequence, so nanoGPT's
        // batch_size=64 is reached through gradient accumulation. Training at
        // the BPE default of 2 with lr=1e-3 (tuned for 64) will not converge.
        accum: if char_mode { 64 } else { 2 },
        lr: if char_mode { 1e-3 } else { 6e-4 },
        // AdamWOpts defaults to 0.95; nanoGPT sets 0.99 for this run.
        beta2: if char_mode { 0.99 } else { 0.95 },
        warmup: 100,
        cosine: char_mode,
        n_layer: if char_mode { 6 } else { 4 },
        n_head: 6,
        n_embd: if char_mode { 384 } else { 192 },
        dropout: if char_mode { 0.2 } else { 0.0 },
        seed: 1337,
        checkpoint: if char_mode {
            "checkpoints/shakespeare_char.safetensors".into()
        } else {
            "checkpoints/shakespeare.safetensors".into()
        },
        resume: false,
        sample_every: if char_mode { 500 } else { 0 },
        eval_every: 250,
        eval_batches: 20,
    }
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // Defaults depend on the tokenizer, so resolve that before the main pass.
    let tokenizer = argv
        .iter()
        .position(|a| a == "--tokenizer")
        .and_then(|i| argv.get(i + 1).cloned())
        .unwrap_or_else(|| "char".into());
    assert!(
        tokenizer == "char" || tokenizer == "bpe",
        "--tokenizer must be char or bpe"
    );
    let mut a = defaults(&tokenizer);

    let mut it = argv.into_iter();
    while let Some(flag) = it.next() {
        let mut val = || it.next().expect("missing value for flag");
        match flag.as_str() {
            "--backend" => a.backend = val(),
            "--tokenizer" => a.tokenizer = val(),
            "--steps" => a.steps = val().parse().unwrap(),
            "--seq-len" => a.seq_len = val().parse().unwrap(),
            "--accum" => a.accum = val().parse().unwrap(),
            "--lr" => a.lr = val().parse().unwrap(),
            "--beta2" => a.beta2 = val().parse().unwrap(),
            "--warmup" => a.warmup = val().parse().unwrap(),
            "--cosine" => a.cosine = true,
            "--no-cosine" => a.cosine = false,
            "--layers" => a.n_layer = val().parse().unwrap(),
            "--heads" => a.n_head = val().parse().unwrap(),
            "--embd" => a.n_embd = val().parse().unwrap(),
            "--dropout" => a.dropout = val().parse().unwrap(),
            "--seed" => a.seed = val().parse().unwrap(),
            "--checkpoint" => a.checkpoint = val(),
            "--resume" => a.resume = true,
            "--sample-every" => a.sample_every = val().parse().unwrap(),
            "--eval-every" => a.eval_every = val().parse().unwrap(),
            "--eval-batches" => a.eval_batches = val().parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn corpus() -> String {
    std::fs::read_to_string("data/tinyshakespeare.txt")
        .expect("run scripts/download_shakespeare.sh first")
}

/// Encode the corpus, caching the ids on disk. The cache is **keyed by
/// tokenizer**: reusing BPE ids for a char model would train on ids that index
/// a 65-row embedding table with values up to 50256.
fn load_tokens(tok: &AnyTokenizer, text: &str) -> Vec<u32> {
    let cache = format!("data/tinyshakespeare.{}.ids", tok.kind());
    if let Ok(bytes) = std::fs::read(&cache) {
        return bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
    }
    eprintln!(
        "encoding {} bytes with the {} vocab...",
        text.len(),
        tok.kind()
    );
    let ids = tok.encode(text).unwrap();
    let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
    std::fs::write(&cache, &bytes).ok();
    ids
}

/// Mean loss over `n` random windows of the given id slice, forward only.
fn eval_loss(model: &Gpt2, ids: &[u32], seq_len: usize, n: usize, seed: u64) -> f32 {
    if ids.len() <= seq_len + 1 {
        return f32::NAN;
    }
    // A fixed seed makes the figure comparable across steps.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut total = 0.0f32;
    for _ in 0..n {
        let s = rng.random_range(0..ids.len() - seq_len - 1);
        total += model
            .loss(&ids[s..s + seq_len], &ids[s + 1..s + seq_len + 1])
            .unwrap();
    }
    total / n as f32
}

fn main() {
    let a = parse_args();
    let device = match a.backend.as_str() {
        "cpu" => Device::Cpu,
        "wgpu" => Device::wgpu().expect("no WebGPU adapter"),
        other => panic!("unknown backend {other}"),
    };
    println!("device: {}", device.describe());

    let text = corpus();
    let tok = match a.tokenizer.as_str() {
        "char" => AnyTokenizer::Char(CharTokenizer::from_corpus(&text)),
        _ => AnyTokenizer::bpe(
            Gpt2Tokenizer::from_dir("models/gpt2").expect("run scripts/download_gpt2.sh first"),
        ),
    };
    let vocab_size = tok.vocab_size();
    let ids = load_tokens(&tok, &text);
    // nanoGPT's split: the last 10% of the corpus is never trained on.
    let split = ((ids.len() as f64) * (1.0 - VAL_FRACTION)) as usize;
    let (train_ids, val_ids) = ids.split_at(split);
    println!(
        "corpus: {} {} tokens (vocab {vocab_size}) — {} train / {} val",
        ids.len(),
        tok.kind(),
        train_ids.len(),
        val_ids.len()
    );

    let config = Gpt2Config {
        n_layer: a.n_layer,
        n_head: a.n_head,
        n_embd: a.n_embd,
        n_ctx: a.seq_len.max(64),
        vocab_size,
        layer_norm_epsilon: 1e-5,
        eos_token_id: None,
    };
    let mut model = if a.resume {
        println!("resuming from {}", a.checkpoint);
        Gpt2::from_safetensors(&a.checkpoint, config.clone(), &device).unwrap()
    } else {
        Gpt2::init_random(config.clone(), &device, a.seed).unwrap()
    };
    let n_params: usize = model
        .params()
        .unwrap()
        .iter()
        .map(|p| p.shape().numel())
        .sum();
    println!(
        "model: {} layers, {} heads, {} embd — {:.2}M params ({:.1} MB f32)",
        a.n_layer,
        a.n_head,
        a.n_embd,
        n_params as f64 / 1e6,
        (n_params * 4) as f64 / 1e6
    );
    println!(
        "train: {} steps, accum {} (effective batch), lr {:.1e}, beta2 {}, dropout {}",
        a.steps, a.accum, a.lr, a.beta2, a.dropout
    );

    let opts = AdamWOpts {
        lr: a.lr,
        beta2: a.beta2,
        ..Default::default()
    };
    let specs = model.param_specs();
    let mut opt = {
        let params = model.params().unwrap();
        let with_decay: Vec<(&forge::Tensor, bool)> = params
            .iter()
            .zip(&specs)
            .map(|(p, (_, d))| (*p, *d))
            .collect();
        AdamW::new(&with_decay, opts).unwrap()
    };

    let mut rng = StdRng::seed_from_u64(a.seed);
    let mut ema: Option<f32> = None;
    let mut best_val = f32::INFINITY;
    let start = std::time::Instant::now();
    if let Some(dir) = std::path::Path::new(&a.checkpoint).parent() {
        std::fs::create_dir_all(dir).ok();
    }
    // The vocab must ship with the weights: silently re-deriving it from a
    // different corpus at inference time would shift every token id.
    let save_sidecars = |model: &Gpt2| {
        write_sidecars(&a.checkpoint, &model.config, &tok);
    };

    for step in 1..=a.steps {
        // Linear warmup, then constant lr (or cosine decay to 10% with --cosine).
        opt.opts.lr = if step <= a.warmup {
            a.lr * step as f32 / a.warmup as f32
        } else if a.cosine {
            let t = (step - a.warmup) as f32 / (a.steps - a.warmup).max(1) as f32;
            let min_lr = 0.1 * a.lr;
            min_lr + 0.5 * (a.lr - min_lr) * (1.0 + (std::f32::consts::PI * t).cos())
        } else {
            a.lr
        };
        let mut loss_sum = 0.0f32;
        let mut grads_acc: Option<Vec<forge::Tensor>> = None;
        for micro in 0..a.accum {
            let s = rng.random_range(0..train_ids.len() - a.seq_len - 1);
            let input = &train_ids[s..s + a.seq_len];
            let target = &train_ids[s + 1..s + a.seq_len + 1];
            let seed = (step * 1_000 + micro) as u32;
            let (loss, grads) = model.loss_grads(input, target, a.dropout, seed).unwrap();
            loss_sum += loss;
            grads_acc = Some(match grads_acc {
                None => grads,
                Some(acc) => acc
                    .iter()
                    .zip(&grads)
                    .map(|(x, y)| forge::ops::add(x, y).unwrap())
                    .collect(),
            });
        }
        let loss = loss_sum / a.accum as f32;
        // Mean over micro-batches.
        let grads: Vec<forge::Tensor> = grads_acc
            .unwrap()
            .iter()
            .map(|g| forge::ops::scale(g, 1.0 / a.accum as f32).unwrap())
            .collect();
        let mut params = model.params_mut().unwrap();
        let norm = opt.step(&mut params, &grads).unwrap();
        drop(params);

        assert!(
            loss.is_finite(),
            "step {step}: loss went non-finite ({loss}) — lower --lr"
        );
        ema = Some(match ema {
            None => loss,
            Some(e) => 0.95 * e + 0.05 * loss,
        });
        if step % 10 == 0 || step == 1 {
            println!(
                "step {step:5}  loss {loss:7.4}  ema {:7.4}  gnorm {norm:6.2}  lr {:.1e}  {:5.1}s",
                ema.unwrap(),
                opt.opts.lr,
                start.elapsed().as_secs_f32()
            );
            std::io::stdout().flush().ok();
        }
        if a.eval_every > 0 && (step % a.eval_every == 0 || step == a.steps) {
            let tr = eval_loss(&model, train_ids, a.seq_len, a.eval_batches, 99);
            let va = eval_loss(&model, val_ids, a.seq_len, a.eval_batches, 99);
            best_val = best_val.min(va);
            println!("eval  @ {step:5}  train {tr:7.4}  val {va:7.4}  (best val {best_val:7.4})");
            std::io::stdout().flush().ok();
        }
        if a.sample_every > 0 && step % a.sample_every == 0 {
            let text = model
                .generate(
                    &tok,
                    "ROMEO:",
                    200,
                    Sampling::TopK {
                        k: 40,
                        temperature: 0.8,
                        seed: 7,
                    },
                )
                .unwrap();
            println!("--- sample @ {step}:\n{text}\n---");
            std::io::stdout().flush().ok();
        }
        if step % 500 == 0 || step == a.steps {
            model.save_safetensors(&a.checkpoint).unwrap();
            save_sidecars(&model);
            println!("checkpoint saved to {}", a.checkpoint);
        }
    }
    let final_ema = ema.unwrap();
    let val = eval_loss(&model, val_ids, a.seq_len, a.eval_batches.max(50), 99);
    let gate = if a.tokenizer == "char" {
        "gate: nanoGPT reference val loss ~1.48"
    } else {
        "gate: start ~10.8, target < 4.0"
    };
    println!(
        "done: smoothed train loss {final_ema:.4}, val loss {val:.4} after {} steps ({:.1}s) — {gate}",
        a.steps,
        start.elapsed().as_secs_f32()
    );
}

/// Write `config.json` and `vocab.json` beside the checkpoint so the artifact
/// is self-describing and loadable by `generate`, the TUI, and the web demo.
fn write_sidecars(checkpoint: &str, config: &Gpt2Config, tok: &AnyTokenizer) {
    let dir = std::path::Path::new(checkpoint)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let stem = std::path::Path::new(checkpoint)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let config_json = format!(
        "{{\n  \"n_layer\": {},\n  \"n_head\": {},\n  \"n_embd\": {},\n  \"n_ctx\": {},\n  \
         \"vocab_size\": {},\n  \"layer_norm_epsilon\": {},\n  \"tokenizer\": \"{}\"\n}}\n",
        config.n_layer,
        config.n_head,
        config.n_embd,
        config.n_ctx,
        config.vocab_size,
        config.layer_norm_epsilon,
        tok.kind(),
    );
    std::fs::write(dir.join(format!("{stem}.config.json")), config_json).ok();
    if let AnyTokenizer::Char(c) = tok {
        std::fs::write(dir.join(format!("{stem}.vocab.json")), c.to_json()).ok();
    }
}
