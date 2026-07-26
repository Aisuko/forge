//! `Gpt2::generate_streaming` (Plan 02 Task 1): the sync per-token hook the
//! TUI measures throughput with.
//!
//! Uses a tiny randomly-initialized model on the CPU backend, so this needs no
//! downloaded weights and no GPU — the properties under test are about the
//! decode loop, not about the quality of what it emits.

use forge::{CharTokenizer, Device, Gpt2, Gpt2Config, Sampling, Tokenizer as _};

const CORPUS: &str = "To be, or not to be: that is the question.\n";
const PROMPT: &str = "To be";
const N_NEW: usize = 32;

fn tiny() -> (Gpt2, CharTokenizer) {
    let tok = CharTokenizer::from_corpus(CORPUS);
    let config = Gpt2Config {
        n_layer: 2,
        n_head: 2,
        n_embd: 32,
        n_ctx: 64,
        vocab_size: tok.vocab_size(),
        layer_norm_epsilon: 1e-5,
        eos_token_id: None,
    };
    let model = Gpt2::init_random(config, &Device::Cpu, 7).unwrap();
    (model, tok)
}

fn topk() -> Sampling {
    Sampling::TopK {
        k: 8,
        temperature: 0.8,
        seed: 1234,
    }
}

#[test]
fn streaming_matches_generate() {
    let (model, tok) = tiny();
    for sampling in [Sampling::Greedy, topk()] {
        let plain = model.generate(&tok, PROMPT, N_NEW, sampling).unwrap();
        let streamed = model
            .generate_streaming(&tok, PROMPT, N_NEW, sampling, |_, _| {})
            .unwrap();
        assert_eq!(
            plain.as_bytes(),
            streamed.as_bytes(),
            "generate and generate_streaming diverged for {sampling:?}"
        );
    }
}

#[test]
fn callback_fires_once_per_new_token() {
    let (model, tok) = tiny();
    let mut ids = Vec::new();
    let text = model
        .generate_streaming(&tok, PROMPT, N_NEW, topk(), |id, _| ids.push(id))
        .unwrap();

    assert_eq!(
        ids.len(),
        N_NEW,
        "expected one callback per generated token, got {}",
        ids.len()
    );
    // The callback must cover generated tokens only, never the prompt.
    let prompt_ids = tok.encode(PROMPT).unwrap();
    let all = tok.encode(&text).unwrap();
    assert_eq!(all.len(), prompt_ids.len() + N_NEW);
    assert_eq!(&all[prompt_ids.len()..], &ids[..]);
}

#[test]
fn deltas_concatenate_to_the_completion() {
    let (model, tok) = tiny();
    let mut streamed = String::new();
    let text = model
        .generate_streaming(&tok, PROMPT, N_NEW, topk(), |_, delta| {
            streamed.push_str(delta)
        })
        .unwrap();
    assert_eq!(text, format!("{PROMPT}{streamed}"));
}
