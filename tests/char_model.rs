//! The shipped character-level Shakespeare model (Plan 04): the artifact the
//! website and TUI both load.
//!
//! Unlike models/gpt2, `assets/shakespeare_char/` is tracked in git, so these
//! run on a fresh clone. They still self-skip, because the artifact is only
//! present once someone has trained and shipped it.

use std::ops::ControlFlow;

use forge::serialization::checkpoint_in_dir;
use forge::{
    AnyTokenizer, AttnStep, Device, Gpt2, Gpt2Config, Sampling, StepTrace, Tokenizer as _,
};

const DIR: &str = "assets/shakespeare_char";
const PROMPT: &str = "ROMEO:";

fn model_on(config: Gpt2Config, device: &Device) -> Gpt2 {
    Gpt2::from_checkpoint(checkpoint_in_dir(DIR).unwrap(), config, device).unwrap()
}

fn assets() -> Option<(Gpt2Config, AnyTokenizer)> {
    if checkpoint_in_dir(DIR).is_err() {
        eprintln!("skipping: {DIR} not present (train and run scripts/ship_char_model.sh)");
        return None;
    }
    let config = Gpt2Config::from_json(format!("{DIR}/config.json")).expect("config.json");
    let tok = AnyTokenizer::from_dir(DIR).expect("vocab.json");
    Some((config, tok))
}

#[test]
fn ships_a_char_vocab_matching_the_checkpoint() {
    let Some((config, tok)) = assets() else {
        return;
    };
    assert_eq!(tok.kind(), "char", "no merges.txt should be present");
    assert_eq!(tok.vocab_size(), 65, "Tiny Shakespeare's character set");
    assert_eq!(
        config.vocab_size,
        tok.vocab_size(),
        "config.json and vocab.json disagree — every token id would be wrong"
    );
}

#[test]
fn greedy_is_identical_on_cpu_and_wgpu() {
    let Some((config, tok)) = assets() else {
        return;
    };
    let mut outputs = Vec::new();
    for device in [Device::Cpu, Device::wgpu().unwrap()] {
        let model = model_on(config.clone(), &device);
        let text = model.generate(&tok, PROMPT, 48, Sampling::Greedy).unwrap();
        println!("{}: {text:?}", device.describe());
        outputs.push(text);
    }
    assert_eq!(
        outputs[0], outputs[1],
        "CPU and WGPU produced different greedy continuations"
    );
}

#[test]
fn generation_stays_inside_the_vocabulary() {
    let Some((config, tok)) = assets() else {
        return;
    };
    let model = model_on(config, &Device::Cpu);
    let text = model
        .generate(
            &tok,
            PROMPT,
            200,
            Sampling::TopK {
                k: 40,
                temperature: 0.8,
                seed: 7,
            },
        )
        .unwrap();
    // Round-tripping proves every generated id maps back to a real character.
    assert_eq!(tok.decode(&tok.encode(&text).unwrap()), text);
    // A char model trained on Shakespeare emits speaker names and line breaks;
    // a broken one emits a single run of one character.
    assert!(text.contains('\n'), "no line breaks in {text:?}");
    assert!(
        text.chars().filter(|c| c.is_ascii_alphabetic()).count() > 100,
        "output is not mostly letters: {text:?}"
    );
}

// ── Attention probe ───────────────────────────────────────────────────────
// The website renders these numbers as a live 3D view while it generates, so
// what matters is that they are the model's own softmax output and that
// capturing them changes nothing.

#[test]
fn attention_probe_captures_the_real_softmax() {
    let Some((config, tok)) = assets() else {
        return;
    };
    // Both backends: WGPU reads every block back in one batched round trip,
    // which is a different path from the CPU backend's plain clone.
    for device in [Device::Cpu, Device::wgpu().unwrap()] {
        let model = model_on(config.clone(), &device);
        let ids = tok.encode(PROMPT).unwrap();
        let mut cache = model.new_cache().unwrap();
        let (_, steps) =
            pollster::block_on(model.logits_step_attn_async(&ids, &mut cache)).unwrap();

        assert_eq!(steps.len(), config.n_layer, "one capture per block");
        for (layer, s) in steps.iter().enumerate() {
            assert_eq!(s.layer, layer);
            assert_eq!(s.n_head, config.n_head);
            // Prefill attends with every prompt position at once.
            assert_eq!((s.q_len, s.kv_len), (ids.len(), ids.len()));
            assert_eq!(s.probs.len(), s.n_head * s.q_len * s.kv_len);

            for head in 0..s.n_head {
                for q in 0..s.q_len {
                    let row = &s.probs[(head * s.q_len + q) * s.kv_len..][..s.kv_len];
                    let sum: f32 = row.iter().sum();
                    assert!(
                        (sum - 1.0).abs() < 1e-5,
                        "{}: layer {layer} head {head} row {q} sums to {sum}, not 1",
                        device.describe()
                    );
                    // Causal mask: a query never attends to a later position.
                    for (k, &w) in row.iter().enumerate().skip(q + 1) {
                        assert_eq!(
                            w,
                            0.0,
                            "{}: layer {layer} head {head} row {q} sees future {k}",
                            device.describe()
                        );
                    }
                }
            }
        }

        // One decode step: a single query against one more cached position.
        let (_, steps) =
            pollster::block_on(model.logits_step_attn_async(&[ids[0]], &mut cache)).unwrap();
        for s in &steps {
            assert_eq!((s.q_len, s.kv_len), (1, ids.len() + 1));
        }
    }
}

#[test]
fn attention_probe_does_not_change_the_output() {
    let Some((config, tok)) = assets() else {
        return;
    };
    let model = model_on(config.clone(), &Device::Cpu);
    const N: usize = 24;

    let plain = pollster::block_on(model.generate_async(&tok, PROMPT, N, Sampling::Greedy, |_| {}))
        .unwrap();

    let mut seen = 0usize;
    let probed = pollster::block_on(model.generate_async_probe(
        &tok,
        PROMPT,
        N,
        Sampling::Greedy,
        |_| ControlFlow::Continue(()),
        Some(|steps: &[AttnStep]| {
            assert_eq!(steps.len(), config.n_layer);
            seen += 1;
        }),
    ))
    .unwrap();

    assert_eq!(
        plain, probed,
        "the probe perturbed the computation it is meant to observe"
    );
    // Prefill plus one capture per generated token.
    assert_eq!(seen, N + 1, "the probe skipped or duplicated a decode step");
}

/// The full trace must be a strict superset of the attention probe, not a
/// second implementation of it: the same run, the same text, and byte-identical
/// probabilities for every block of every step.
#[test]
fn the_trace_is_a_superset_of_the_attention_probe() {
    let Some((config, tok)) = assets() else {
        return;
    };
    let model = model_on(config.clone(), &Device::wgpu().unwrap());
    const N: usize = 8;
    const DETAIL: usize = 3;
    const TOP: usize = 24;
    let sampling = Sampling::Greedy;

    let mut attn_runs: Vec<Vec<AttnStep>> = Vec::new();
    let plain = pollster::block_on(model.generate_async_probe(
        &tok,
        PROMPT,
        N,
        sampling,
        |_| ControlFlow::Continue(()),
        Some(|steps: &[AttnStep]| attn_runs.push(steps.to_vec())),
    ))
    .unwrap();

    let mut traces: Vec<StepTrace> = Vec::new();
    let traced = pollster::block_on(model.generate_async_trace(
        &tok,
        PROMPT,
        N,
        sampling,
        |_| ControlFlow::Continue(()),
        Some(|t: &StepTrace| traces.push(t.clone())),
        DETAIL,
        TOP,
    ))
    .unwrap();

    assert_eq!(
        plain, traced,
        "reading the detail tensors changed the generated text"
    );
    assert_eq!(traces.len(), attn_runs.len(), "different step counts");

    let prompt_len = tok.encode(PROMPT).unwrap().len();
    for (step, (trace, attn)) in traces.iter().zip(&attn_runs).enumerate() {
        assert_eq!(trace.attn.len(), config.n_layer, "one capture per block");
        for (a, b) in trace.attn.iter().zip(attn) {
            assert_eq!((a.layer, a.n_head), (b.layer, b.n_head));
            assert_eq!((a.q_len, a.kv_len), (b.q_len, b.kv_len));
            assert_eq!(a.probs, b.probs, "step {step}: attention differs");
        }

        // Prefill paints the whole prompt at once; every decode step is one
        // query row against one more cached position.
        let want_q = if step == 0 { prompt_len } else { 1 };
        assert_eq!(trace.q_len, want_q);
        assert_eq!(trace.kv_len, prompt_len + step);
        assert_eq!(trace.embedding.len(), trace.q_len * config.n_embd);

        let hd = config.n_embd / config.n_head;
        assert_eq!(trace.detail.len(), DETAIL);
        for (layer, d) in trace.detail.iter().enumerate() {
            assert_eq!(d.layer, layer);
            for (name, got) in [("q", &d.q), ("k", &d.k), ("v", &d.v)] {
                assert_eq!(
                    got.len(),
                    config.n_head * trace.q_len * hd,
                    "step {step} layer {layer} {name}"
                );
            }
            assert_eq!(d.mlp_hidden.len(), trace.q_len * 4 * config.n_embd);
            assert_eq!(d.block_out.len(), trace.q_len * config.n_embd);
        }

        // Real probabilities over the whole vocabulary, ranked.
        assert_eq!(trace.top.len(), TOP.min(config.vocab_size));
        let mut sum = 0.0;
        for w in trace.top.windows(2) {
            assert!(w[0].1 >= w[1].1, "step {step}: top-n is not ranked");
        }
        for (id, p) in &trace.top {
            assert!((*id as usize) < config.vocab_size);
            assert!((0.0..=1.0).contains(p), "step {step}: probability {p}");
            sum += p;
        }
        assert!(sum <= 1.0 + 1e-4, "step {step}: probabilities sum to {sum}");
    }
}
