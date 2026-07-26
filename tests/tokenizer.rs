//! Tokenizer tests against known GPT-2 encodings and round-trips.
//! The BPE tests require models/gpt2/vocab.json + merges.txt
//! (scripts/download_gpt2.sh) and are skipped when absent; the char-level
//! tests require data/tinyshakespeare.txt (scripts/download_shakespeare.sh).

use forge::{CharTokenizer, Gpt2Tokenizer, Tokenizer as _};

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

// ---- character-level tokenizer (nanoGPT shakespeare_char) ----

fn shakespeare() -> Option<String> {
    match std::fs::read_to_string("data/tinyshakespeare.txt") {
        Ok(t) => Some(t),
        Err(_) => {
            eprintln!("skipping: run scripts/download_shakespeare.sh first");
            None
        }
    }
}

#[test]
fn char_round_trips_whole_corpus() {
    let Some(text) = shakespeare() else { return };
    let tok = CharTokenizer::from_corpus(&text);
    assert_eq!(
        tok.vocab_size(),
        65,
        "Tiny Shakespeare has 65 distinct chars"
    );
    let ids = tok.encode(&text).unwrap();
    assert_eq!(ids.len(), text.chars().count());
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn char_vocab_is_sorted_unique_chars() {
    let tok = CharTokenizer::from_corpus("banana\n");
    // sorted(set("banana\n")) == ['\n', 'a', 'b', 'n']
    assert_eq!(tok.chars(), ['\n', 'a', 'b', 'n']);
    assert_eq!(tok.encode("nab").unwrap(), vec![3, 1, 2]);
}

#[test]
fn char_json_round_trips() {
    let Some(text) = shakespeare() else { return };
    let tok = CharTokenizer::from_corpus(&text);
    let back = CharTokenizer::from_json(&tok.to_json()).unwrap();
    assert_eq!(back.chars(), tok.chars(), "vocab order must survive JSON");
    let sample = "ROMEO:\nBut soft, what light through yonder window breaks?";
    assert_eq!(back.encode(sample).unwrap(), tok.encode(sample).unwrap());
}

#[test]
fn char_rejects_unknown_but_can_report_them() {
    let tok = CharTokenizer::from_corpus("abc");
    assert!(tok.encode("abé").is_err());
    assert_eq!(tok.unknown_chars("abéxé"), vec!['é', 'x']);
    assert_eq!(tok.encode_lossy("abéx"), vec![0, 1]);
}

#[test]
fn char_decode_bytes_is_append_only() {
    // 'é' is two UTF-8 bytes but one token, so splitting the id stream can
    // never split a character — the property the streaming path relies on.
    let tok = CharTokenizer::from_corpus("héllo");
    let all = tok.encode("héllo").unwrap();
    let (head, tail) = all.split_at(2);
    let mut joined = tok.decode_bytes(head);
    joined.extend(tok.decode_bytes(tail));
    assert_eq!(joined, tok.decode_bytes(&all));
}
