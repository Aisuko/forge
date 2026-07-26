//! The shipped character-level Shakespeare model (Plan 04): the artifact the
//! website and TUI both load.
//!
//! Unlike models/gpt2, `assets/shakespeare_char/` is tracked in git, so these
//! run on a fresh clone. They still self-skip, because the artifact is only
//! present once someone has trained and shipped it.

use std::ops::ControlFlow;

use forge::{AnyTokenizer, AttnStep, Device, Gpt2, Gpt2Config, Sampling, Tokenizer as _};

const DIR: &str = "assets/shakespeare_char";
const PROMPT: &str = "ROMEO:";

fn assets() -> Option<(Gpt2Config, AnyTokenizer)> {
    if !std::path::Path::new(DIR).join("model.safetensors").exists() {
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
        let model =
            Gpt2::from_safetensors(format!("{DIR}/model.safetensors"), config.clone(), &device)
                .unwrap();
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
    let model =
        Gpt2::from_safetensors(format!("{DIR}/model.safetensors"), config, &Device::Cpu).unwrap();
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

fn model_on(config: Gpt2Config, device: &Device) -> Gpt2 {
    Gpt2::from_safetensors(format!("{DIR}/model.safetensors"), config, device).unwrap()
}

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
