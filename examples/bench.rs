//! The inference benchmark. Every performance claim about Forge should be
//! reproducible with one command:
//!
//!   cargo run --release --example bench -- --model assets/shakespeare_char
//!
//! Flags: [--model DIR] [--tokens N] [--warmup N] [--runs N] [--prompt "..."]
//!
//! It reports two phases separately because they are bounded by different
//! things. **Prompt encode** runs `t` positions through one forward pass — wide
//! kernels, GPU-bound. **Decode** runs one position at a time behind a KV
//! cache — narrow kernels, and historically bounded not by arithmetic but by
//! how many times the CPU hands work to the GPU.
//!
//! Which is why `submits/token` sits beside `ms/token`. A GPT-2 block issues
//! ~16 kernels; if each one is its own `queue.submit`, the GPU spends most of a
//! token idle waiting on the CPU, and no amount of faster arithmetic shows up
//! in the wall clock. Those counters are what make that visible instead of
//! merely suspected.
//!
//! Timings are the median of `--runs` (default 5) after `--warmup` (default 3)
//! discarded tokens; the first dispatch of every kernel pays a one-time shader
//! compile that would otherwise land entirely in run 1.

use std::time::Instant;

use forge::backend::wgpu::Stats;
use forge::{AnyTokenizer, Device, Gpt2, Gpt2Config, Tokenizer as _};

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

/// One measured phase: wall time, and what the device was asked to do.
struct Phase {
    label: &'static str,
    /// Unit the rates are per — "token" for both phases here.
    unit: &'static str,
    units: usize,
    ms: f64,
    stats: Stats,
}

impl Phase {
    fn report(&self) {
        let per = |n: usize| n as f64 / self.units as f64;
        println!("\n  {}", self.label);
        println!(
            "    {:>12.3} ms/{}      {:>10.1} {}s/sec",
            self.ms / self.units as f64,
            self.unit,
            1000.0 * self.units as f64 / self.ms,
            self.unit,
        );
        println!(
            "    {:>12.1} dispatches/{}  {:>8.1} submits/{}",
            per(self.stats.dispatches),
            self.unit,
            per(self.stats.submits),
            self.unit,
        );
        println!(
            "    {:>12.1} buffers/{}     {:>8.2} MiB allocated/{}",
            per(self.stats.buffers_created),
            self.unit,
            per(self.stats.bytes_allocated) / (1024.0 * 1024.0),
            self.unit,
        );
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let dir = get("--model").unwrap_or_else(|| "assets/shakespeare_char".into());
    let tokens: usize = get("--tokens").map_or(128, |s| s.parse().expect("--tokens"));
    let warmup: usize = get("--warmup").map_or(3, |s| s.parse().expect("--warmup"));
    let runs: usize = get("--runs").map_or(5, |s| s.parse().expect("--runs"));

    // wgpu only, and deliberately no --backend flag: this measures the
    // production backend. The CPU backend exists to be a correctness
    // reference, so benchmarking it would invite a comparison that means
    // nothing.
    let device = Device::wgpu()?;
    let ctx = match &device {
        Device::Wgpu(ctx) => ctx.clone(),
        _ => unreachable!("Device::wgpu returns a Wgpu device"),
    };

    let dir = std::path::Path::new(&dir);
    let config =
        Gpt2Config::from_json(dir.join("config.json")).unwrap_or_else(|_| Gpt2Config::gpt2());
    let weights = dir.join("model.safetensors");
    let disk_bytes = std::fs::metadata(&weights).map(|m| m.len()).unwrap_or(0);

    let t0 = Instant::now();
    let model = Gpt2::from_safetensors(&weights, config.clone(), &device)?;
    let tokenizer = AnyTokenizer::from_dir(dir)?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let after_load = ctx.stats();

    let prompt = get("--prompt").unwrap_or_else(|| match tokenizer.kind() {
        "char" => "ROMEO:\nWhat light through yonder window breaks?".into(),
        _ => "The old lighthouse keeper".into(),
    });
    let ids = tokenizer.encode(&prompt)?;

    println!("== forge bench");
    println!("  device      {}", device.describe());
    println!("  model       {}", dir.display());
    println!(
        "  config      n_layer={} n_head={} n_embd={} n_ctx={} vocab={}",
        config.n_layer, config.n_head, config.n_embd, config.n_ctx, config.vocab_size
    );
    println!(
        "  weights     {:.2} MiB on disk, loaded in {:.0} ms into {:.2} MiB of GPU buffers",
        disk_bytes as f64 / (1024.0 * 1024.0),
        load_ms,
        after_load.bytes_allocated as f64 / (1024.0 * 1024.0),
    );
    println!(
        "  prompt      {} tokens, generating {tokens}, {runs} runs after {warmup} warmup",
        ids.len()
    );

    // ── prompt encode: the whole prompt through one forward pass.
    let mut encode_ms = Vec::with_capacity(runs);
    let mut encode_stats = Stats::default();
    for r in 0..(warmup.min(1) + runs) {
        let mut cache = model.new_cache()?;
        let before = ctx.stats();
        let t = Instant::now();
        let _ = model.logits_step(&ids, &mut cache)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        // Run 0 is the warmup: it pays every shader's first-time compile.
        if r > 0 || warmup == 0 {
            encode_ms.push(ms);
            encode_stats = ctx.stats().since(before);
        }
    }

    // ── decode: one token at a time behind the KV cache. This is the phase
    // the site's demo actually spends its time in.
    let mut decode_ms = Vec::with_capacity(runs);
    let mut decode_stats = Stats::default();
    for r in 0..(warmup.min(1) + runs) {
        let mut cache = model.new_cache()?;
        let mut logits = model.logits_step(&ids, &mut cache)?;
        // Warmup tokens are generated but not timed.
        for _ in 0..warmup {
            let next = argmax(&logits);
            logits = model.logits_step(&[next], &mut cache)?;
        }
        let before = ctx.stats();
        let t = Instant::now();
        for _ in 0..tokens {
            let next = argmax(&logits);
            if cache.len() + 1 >= config.n_ctx {
                break;
            }
            logits = model.logits_step(&[next], &mut cache)?;
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        if r > 0 || warmup == 0 {
            decode_ms.push(ms);
            decode_stats = ctx.stats().since(before);
        }
    }

    Phase {
        label: "prompt encode (one forward pass over the prompt)",
        unit: "token",
        units: ids.len(),
        ms: median(encode_ms),
        stats: encode_stats,
    }
    .report();

    Phase {
        label: "decode (KV-cached, one token per step)",
        unit: "token",
        units: tokens,
        ms: median(decode_ms),
        stats: decode_stats,
    }
    .report();

    let total = ctx.stats();
    println!(
        "\n  session total  {} dispatches, {} submits, {} buffers, {:.1} MiB allocated",
        total.dispatches,
        total.submits,
        total.buffers_created,
        total.bytes_allocated as f64 / (1024.0 * 1024.0),
    );
    Ok(())
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
