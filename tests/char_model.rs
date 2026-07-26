//! The shipped character-level Shakespeare model (Plan 04): the artifact the
//! website and TUI both load.
//!
//! Unlike models/gpt2, `assets/shakespeare_char/` is tracked in git, so these
//! run on a fresh clone. They still self-skip, because the artifact is only
//! present once someone has trained and shipped it.

use forge::{AnyTokenizer, Device, Gpt2, Gpt2Config, Sampling, Tokenizer as _};

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
