//! Run the council natively and show its working — the gate the web page is
//! not allowed to run ahead of.
//!
//! Four experts branched from one ancestor generate one character at a time.
//! For each character this prints what every expert wanted on its own, the
//! router weight it earned, and what the merged hidden state actually decoded
//! to. If the experts never disagree, the page has nothing to show and the
//! fine-tuning needs more steps or a higher learning rate — so the disagreement
//! rate is printed at the end and is the number that matters.
//!
//! ```bash
//! ./scripts/train_council.sh
//! cargo run --release --example council_demo -- --prompt "ROMEO:" --chars 200
//! ```

use forge::{
    AnyTokenizer, CharTokenizer, Council, Device, Gpt2, Gpt2Config, Sampling, Tokenizer as _,
};

const MANIFEST: &str = "data/council/manifest.json";

fn label(tok: &AnyTokenizer, id: u32) -> String {
    match tok.decode(&[id]).as_str() {
        "\n" => "\\n".into(),
        " " => "␣".into(),
        s => s.to_string(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let dir = get("--dir").unwrap_or_else(|| "checkpoints/council".into());
    let prompt = get("--prompt").unwrap_or_else(|| "ROMEO:".into());
    let chars: usize = get("--chars").map_or(160, |s| s.parse().expect("--chars"));
    let show: usize = get("--show").map_or(24, |s| s.parse().expect("--show"));
    let temp: f32 = get("--temp").map_or(0.8, |s| s.parse().expect("--temp"));
    let seed: u64 = get("--seed").map_or(42, |s| s.parse().expect("--seed"));
    let beta: f32 = get("--beta").map_or(forge::models::council::DEFAULT_BETA, |s| {
        s.parse().expect("--beta")
    });
    let greedy = args.iter().any(|a| a == "--greedy");

    let device = Device::wgpu()?;
    println!("device: {}", device.describe());

    let dir = std::path::Path::new(&dir);
    let names: Vec<String> = {
        let text = std::fs::read_to_string(MANIFEST).unwrap_or_else(|e| {
            panic!("cannot read {MANIFEST} ({e}) — run scripts/split_corpus.py")
        });
        let m: serde_json::Value = serde_json::from_str(&text)?;
        m["experts"]
            .as_array()
            .expect("manifest.experts")
            .iter()
            .map(|e| e["label"].as_str().unwrap_or("expert").to_string())
            .collect()
    };
    let config = Gpt2Config::from_json(dir.join("expert0.config.json"))?;
    let tokenizer = AnyTokenizer::Char(CharTokenizer::from_json(&std::fs::read_to_string(
        dir.join("expert0.vocab.json"),
    )?)?);

    let mut models = Vec::new();
    for k in 0..names.len() {
        let path = dir.join(format!("expert{k}.best.safetensors"));
        models.push(Gpt2::from_safetensors(&path, config.clone(), &device)?);
    }
    let per_expert: usize = models[0].params()?.iter().map(|p| p.shape().numel()).sum();
    let mut council = Council::new(models, names.clone(), seed)?;
    council.beta = beta;

    println!(
        "council: {} experts x {:.2}M params ({:.1} MB f32 each), n_embd {}, vocab {}",
        council.n_experts(),
        per_expert as f64 / 1e6,
        (per_expert * 4) as f64 / 1e6,
        council.n_embd(),
        council.vocab_size(),
    );
    // The load-bearing invariant: a shared embedding table is what puts every
    // expert's hidden state in one basis, and it is the only reason adding them
    // together produces something the shared head can still read.
    let shared = council.embeddings_agree()?;
    println!(
        "shared wte: {}",
        if shared {
            "identical across all experts — hidden states are commensurable"
        } else {
            "DIFFERENT — the merge below is meaningless, retrain with --freeze-embeddings"
        }
    );
    println!(
        "router: beta {beta}  |  sampling: {}\n",
        if greedy {
            "greedy".to_string()
        } else {
            format!("top-k 40, temp {temp}")
        }
    );

    let sampling = if greedy {
        Sampling::Greedy
    } else {
        Sampling::TopK {
            k: 40,
            temperature: temp,
            seed,
        }
    };

    let mut ids = tokenizer.encode(&prompt)?;
    assert!(!ids.is_empty(), "prompt encodes to nothing");
    let mut out = String::new();
    let mut split = 0usize;
    let mut consensus_sum = 0.0f32;
    let mut wins = vec![0usize; council.n_experts()];

    let header: String = names
        .iter()
        .map(|n| format!("{:>22}", short(n)))
        .collect::<Vec<_>>()
        .join("");
    println!("  #  merged {header}");

    let start = std::time::Instant::now();
    let mut next: Vec<u32> = ids.clone();
    for i in 0..chars {
        let step = council.step(&next, sampling, 5)?;
        // Every expert disagreeing with at least one other is the whole point:
        // if this never happens the council is four copies of one model.
        if step.consensus < 1.0 {
            split += 1;
        }
        consensus_sum += step.consensus;
        let leader = step
            .experts
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.weight.total_cmp(&b.1.weight))
            .map(|(k, _)| k)
            .unwrap_or(0);
        wins[leader] += 1;

        if i < show {
            let cells: String = step
                .experts
                .iter()
                .map(|e| {
                    let (id, p) = e.top[0];
                    format!("{:>12} {:.2} w{:.2}", label(&tokenizer, id), p, e.weight)
                })
                .collect::<Vec<_>>()
                .join("");
            println!("{i:>3}  {:>6} {cells}", label(&tokenizer, step.chosen),);
        }

        out.push_str(&tokenizer.decode(&[step.chosen]));
        ids.push(step.chosen);
        next = vec![step.chosen];
        if ids.len() >= council.n_ctx() {
            break;
        }
    }
    let secs = start.elapsed().as_secs_f32();

    println!("\n--- {prompt}{out}\n---\n");
    println!(
        "{chars} chars in {secs:.1}s ({:.0} chars/s)",
        chars as f32 / secs
    );
    println!(
        "experts split on {split}/{chars} characters ({:.0}%) — mean consensus {:.2}",
        100.0 * split as f32 / chars as f32,
        consensus_sum / chars as f32
    );
    for (n, w) in names.iter().zip(&wins) {
        println!("  led {:>4} times  {n}", w);
    }
    if split * 5 < chars {
        println!(
            "\nWARNING: the experts almost never disagree, so the page would show four \
             identical models. Fine-tune longer or at a higher learning rate."
        );
    }
    Ok(())
}

/// Keep the table readable: "Duke Vincentio & Petruchio" is 26 columns.
fn short(name: &str) -> String {
    let first = name.split(" & ").next().unwrap_or(name);
    first.chars().take(11).collect()
}
