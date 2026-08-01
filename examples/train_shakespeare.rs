//! Train a from-scratch GPT-2-style model on Tiny Shakespeare, with either
//! the character-level vocabulary (nanoGPT's `shakespeare_char`, 65 tokens —
//! the default) or the GPT-2 BPE vocabulary.
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
    /// Decoupled weight decay. Was fixed at the `AdamWOpts` default, which
    /// made the one knob most likely to help an overfitting run unsearchable.
    wd: f32,
    /// Windows in the deterministic evaluation (see [`eval_loss_strided`]).
    eval_windows: usize,
    /// Stop after this many evaluations without a new best val loss. 0 is off.
    early_stop: usize,
    /// Load the checkpoint, evaluate it, print JSON, exit. No training.
    eval_only: bool,
    /// Text to train and validate on. The **vocabulary always comes from the
    /// full corpus** regardless, so a model fine-tuned on a slice keeps token
    /// ids that mean the same thing everywhere.
    data: String,
    /// Hold `wte` and `wpe` fixed. This is what makes the council's experts
    /// commensurable: branched from one ancestor with the embeddings frozen,
    /// they share a basis and a wte-tied decoder, so their hidden states can
    /// be added. Unfreeze them and the merge is meaningless.
    freeze_embeddings: bool,
}

/// Corpus the vocabulary is always derived from, whatever `--data` says.
const FULL_CORPUS: &str = "data/tinyshakespeare.txt";

/// Per-tokenizer defaults. The char run reproduces nanoGPT's
/// `config/train_shakespeare_char.py`; the BPE run keeps the config this
/// example shipped with.
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
        // Evaluation is ~0.3% of the cost of a training step, and the best
        // checkpoint can only be caught by an eval that ran near it.
        eval_every: 100,
        eval_batches: 20,
        wd: 0.1,
        eval_windows: 512,
        early_stop: 0,
        eval_only: false,
        data: FULL_CORPUS.into(),
        freeze_embeddings: false,
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
            "--wd" => a.wd = val().parse().unwrap(),
            "--eval-windows" => a.eval_windows = val().parse().unwrap(),
            "--early-stop" => a.early_stop = val().parse().unwrap(),
            "--eval-only" => a.eval_only = true,
            "--data" => a.data = val(),
            "--freeze-embeddings" => a.freeze_embeddings = true,
            other => panic!("unknown flag {other}"),
        }
    }
    a
}

fn corpus(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("cannot read {path} ({e}) — run scripts/download_shakespeare.sh")
    })
}

/// Encode the corpus, caching the ids on disk. The cache is keyed by
/// **tokenizer and by data file**: reusing BPE ids for a char model would train
/// on ids that index a 65-row embedding table with values up to 50256, and
/// reusing the full corpus's ids for a `--data` slice would train every
/// council expert on the whole corpus while claiming otherwise.
fn load_tokens(tok: &AnyTokenizer, text: &str, data: &str) -> Vec<u32> {
    let stem = std::path::Path::new(data)
        .with_extension("")
        .to_string_lossy()
        .replace('/', "_");
    let cache = format!("data/.ids/{stem}.{}.ids", tok.kind());
    std::fs::create_dir_all("data/.ids").ok();
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

/// Deterministic mean loss over evenly spaced windows covering the whole
/// split, forward only.
///
/// Unlike the random-window version below, this is a property of the weights
/// alone: two checkpoints evaluated a month apart are directly comparable, and
/// a figure recorded today still means the same thing later. At ~17 ms per
/// window on an A5000 the full validation split (434 windows at seq_len 256)
/// costs about 7 s — 0.3% of the 2.7 s a training step takes — so it can run
/// every 100 steps without mattering.
fn eval_loss_strided(model: &Gpt2, ids: &[u32], seq_len: usize, max_windows: usize) -> f32 {
    if ids.len() <= seq_len + 1 {
        return f32::NAN;
    }
    let usable = ids.len() - seq_len - 1;
    // Non-overlapping by default; on a split too large for `max_windows`,
    // spread the windows evenly rather than truncating to a prefix.
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

/// Mean loss over `n` random windows of the given id slice, forward only.
#[allow(dead_code)] // kept as the reference the strided version is checked against
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

    // The vocabulary is a property of the whole corpus, never of `--data`:
    // deriving it from a slice would give each council expert a different
    // meaning for the same token id.
    let vocab_text = corpus(FULL_CORPUS);
    let text = if a.data == FULL_CORPUS {
        vocab_text.clone()
    } else {
        corpus(&a.data)
    };
    let tok = match a.tokenizer.as_str() {
        "char" => AnyTokenizer::Char(CharTokenizer::from_corpus(&vocab_text)),
        _ => AnyTokenizer::bpe(
            Gpt2Tokenizer::from_dir("models/gpt2").expect("run scripts/download_gpt2.sh first"),
        ),
    };
    let vocab_size = tok.vocab_size();
    let ids = load_tokens(&tok, &text, &a.data);
    // nanoGPT's split: the last 10% of the corpus is never trained on.
    let split = ((ids.len() as f64) * (1.0 - VAL_FRACTION)) as usize;
    let (train_ids, val_ids) = ids.split_at(split);
    println!(
        "corpus: {} — {} {} tokens (vocab {vocab_size}) — {} train / {} val",
        a.data,
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
    // --eval-only implies --resume: there is nothing to evaluate about a
    // randomly initialised model, and silently scoring one returns ln(vocab)
    // rather than an error.
    let mut model = if a.resume || a.eval_only {
        println!("loading {}", a.checkpoint);
        Gpt2::from_safetensors(&a.checkpoint, config.clone(), &device).unwrap()
    } else {
        Gpt2::init_random(config.clone(), &device, a.seed).unwrap()
    };
    // Evaluate and leave. This is the path the training script uses to score
    // an existing checkpoint — including the one the site currently ships —
    // without the `--lr 0 --steps 1` trick that was the only way before.
    if a.eval_only {
        let tr = eval_loss_strided(&model, train_ids, a.seq_len, a.eval_windows);
        let va = eval_loss_strided(&model, val_ids, a.seq_len, a.eval_windows);
        println!(
            "{{\"checkpoint\":{:?},\"train\":{tr:.4},\"val\":{va:.4},\"windows\":{}}}",
            a.checkpoint, a.eval_windows
        );
        return;
    }

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
        weight_decay: a.wd,
        ..Default::default()
    };
    let specs = model.param_specs();
    // `wte` and `wpe` are the first two entries of `param_specs`, by contract.
    let frozen: Vec<bool> = specs
        .iter()
        .map(|(name, _)| a.freeze_embeddings && (name == "wte.weight" || name == "wpe.weight"))
        .collect();
    if a.freeze_embeddings {
        let n = frozen.iter().filter(|f| **f).count();
        assert_eq!(n, 2, "expected to freeze wte and wpe, matched {n} params");
        println!("freezing wte + wpe: gradients zeroed and weight decay disabled");
    }
    let mut opt = {
        let params = model.params().unwrap();
        let with_decay: Vec<(&forge::Tensor, bool)> = params
            .iter()
            .zip(&specs)
            .zip(&frozen)
            // Zeroing the gradient is not enough on its own: AdamW's weight
            // decay is decoupled, so a "frozen" param with a zero gradient
            // would still shrink by lr*wd*p on every step.
            .map(|((p, (_, d)), f)| (*p, *d && !f))
            .collect();
        AdamW::new(&with_decay, opts).unwrap()
    };

    let mut rng = StdRng::seed_from_u64(a.seed);
    let mut ema: Option<f32> = None;
    let mut best_val = f32::INFINITY;
    let mut best_step = 0usize;
    // Evaluations since the last improvement, for --early-stop.
    let mut stale = 0usize;
    let start = std::time::Instant::now();

    // Where the best-so-far weights go. This is the file worth shipping: for
    // this recipe the validation loss bottoms out around step 2000 of 5000 and
    // then climbs, so the *last* checkpoint is the worst one the run produced.
    // Saving only on a fixed cadence threw that away on every run so far.
    let stem = std::path::Path::new(&a.checkpoint)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    let dir = std::path::Path::new(&a.checkpoint)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .to_path_buf();
    let best_path = dir.join(format!("{stem}.best.safetensors"));
    let log_path = dir.join(format!("{stem}.metrics.jsonl"));
    let mut log = std::fs::File::create(&log_path).ok();
    // An "improvement" smaller than this is evaluation noise, not progress.
    const MIN_DELTA: f32 = 1e-3;
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
            .zip(&frozen)
            .map(|(g, f)| {
                let s = if *f { 0.0 } else { 1.0 / a.accum as f32 };
                forge::ops::scale(g, s).unwrap()
            })
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
            let tr = eval_loss_strided(&model, train_ids, a.seq_len, a.eval_windows);
            let va = eval_loss_strided(&model, val_ids, a.seq_len, a.eval_windows);
            let improved = va < best_val - MIN_DELTA;
            if improved {
                best_val = va;
                best_step = step;
                stale = 0;
                model.save_safetensors(&best_path).unwrap();
                write_sidecars(best_path.to_str().unwrap(), &model.config, &tok);
            } else {
                stale += 1;
            }
            println!(
                "eval  @ {step:5}  train {tr:7.4}  val {va:7.4}  \
                 (best {best_val:7.4} @ {best_step}){}",
                if improved { "  ← saved" } else { "" }
            );
            if let Some(f) = log.as_mut() {
                writeln!(
                    f,
                    "{{\"step\":{step},\"train\":{tr:.4},\"val\":{va:.4},\
                      \"lr\":{:.3e},\"best_val\":{best_val:.4},\"best_step\":{best_step},\
                      \"elapsed_s\":{:.1}}}",
                    opt.opts.lr,
                    start.elapsed().as_secs_f32()
                )
                .ok();
            }
            std::io::stdout().flush().ok();
            if a.early_stop > 0 && stale >= a.early_stop {
                println!(
                    "early stop at {step}: {stale} evaluations without improvement \
                     (best val {best_val:.4} @ step {best_step})"
                );
                break;
            }
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
    let val = eval_loss_strided(&model, val_ids, a.seq_len, a.eval_windows);
    let gate = if a.tokenizer == "char" {
        "gate: nanoGPT reference val loss ~1.48"
    } else {
        "gate: start ~10.8, target < 4.0"
    };
    println!(
        "done: smoothed train loss {final_ema:.4}, final val loss {val:.4} ({:.1}s) — {gate}",
        start.elapsed().as_secs_f32()
    );
    // The last checkpoint is not the best one for this recipe, so say which
    // file to actually use rather than leaving it to be guessed.
    if best_val.is_finite() {
        println!(
            "best:  val {best_val:.4} at step {best_step} — {}",
            best_path.display()
        );
        println!("log:   {}", log_path.display());
    }
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
