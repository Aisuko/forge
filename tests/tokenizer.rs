//! Tokenizer tests against known GPT-2 encodings and round-trips.
//! Requires models/gpt2/vocab.json + merges.txt (scripts/download_gpt2.sh);
//! skipped when absent.

use forge::Gpt2Tokenizer;

fn tokenizer() -> Option<Gpt2Tokenizer> {
    if !std::path::Path::new("models/gpt2/vocab.json").exists() {
        eprintln!("skipping: models/gpt2 not downloaded");
        return None;
    }
    Some(Gpt2Tokenizer::from_dir("models/gpt2").expect("load tokenizer"))
}

#[test]
fn known_encodings() {
    let Some(tok) = tokenizer() else { return };
    // Canonical GPT-2 encodings (verified against HF transformers).
    assert_eq!(tok.encode("Hello world").unwrap(), vec![15496, 995]);
    assert_eq!(
        tok.encode("Hello, my dog is cute").unwrap(),
        vec![15496, 11, 616, 3290, 318, 13779]
    );
}

#[test]
fn round_trips() {
    let Some(tok) = tokenizer() else { return };
    for text in [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "  leading spaces and   runs of spaces ",
        "unicode: naïve café 日本語 🚀 emoji",
        "numbers 123 456.789 and symbols !@#$%^&*()",
        "newlines\nand\ttabs",
        "don't can't won't it's",
    ] {
        let ids = tok.encode(text).unwrap();
        assert_eq!(tok.decode(&ids), text, "round trip failed for {text:?}");
    }
}

#[test]
fn vocab_size() {
    let Some(tok) = tokenizer() else { return };
    assert_eq!(tok.vocab_size(), 50257);
}
