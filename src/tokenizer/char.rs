//! Character-level tokenizer, matching nanoGPT's `shakespeare_char` prepare
//! step: the vocabulary is the sorted set of distinct characters in the
//! corpus, and a token id is that character's index in the sorted order.
//!
//! For Tiny Shakespeare this yields 65 tokens, which shrinks the (weight-tied)
//! embedding table from 9.65M parameters at the GPT-2 BPE vocab to 24,960 —
//! the difference between a 45.9 MB checkpoint that is 84% untrained random
//! rows and a 43.1 MB one where every parameter does work.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use crate::error::{ForgeError, Result};
use crate::tokenizer::Tokenizer;

pub struct CharTokenizer {
    /// id -> char, indexed directly.
    itos: Vec<char>,
    stoi: HashMap<char, u32>,
}

impl CharTokenizer {
    /// Build the vocab from the sorted unique characters of `text`
    /// (nanoGPT's rule). Ids are indices into that sorted order.
    pub fn from_corpus(text: &str) -> Self {
        Self::from_chars(text.chars().collect::<BTreeSet<char>>())
    }

    fn from_chars(sorted: BTreeSet<char>) -> Self {
        let itos: Vec<char> = sorted.into_iter().collect();
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        CharTokenizer { itos, stoi }
    }

    /// The vocabulary in id order.
    pub fn chars(&self) -> &[char] {
        &self.itos
    }

    /// Characters of `text` that are outside the vocabulary, deduplicated and
    /// in first-appearance order. Callers that want to warn a user before
    /// [`CharTokenizer::encode`] rejects the input use this.
    pub fn unknown_chars(&self, text: &str) -> Vec<char> {
        let mut seen = Vec::new();
        for c in text.chars() {
            if !self.stoi.contains_key(&c) && !seen.contains(&c) {
                seen.push(c);
            }
        }
        seen
    }

    /// Encode, silently dropping characters outside the vocabulary. Pair with
    /// [`CharTokenizer::unknown_chars`] to tell the user what was dropped —
    /// the strict [`CharTokenizer::encode`] is the default for good reason.
    pub fn encode_lossy(&self, text: &str) -> Vec<u32> {
        text.chars()
            .filter_map(|c| self.stoi.get(&c).copied())
            .collect()
    }

    /// `{"<char>": id}` — the same shape as GPT-2's `vocab.json`, so the same
    /// loader path and the same filename work for both tokenizers.
    pub fn to_json(&self) -> String {
        let map: HashMap<String, u32> = self
            .stoi
            .iter()
            .map(|(&c, &i)| (c.to_string(), i))
            .collect();
        // Infallible: keys are strings and values are u32.
        serde_json::to_string(&map).expect("char vocab serializes")
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let map: HashMap<String, u32> = serde_json::from_str(s)?;
        let mut itos = vec!['\0'; map.len()];
        let mut filled = vec![false; map.len()];
        for (k, id) in &map {
            let mut cs = k.chars();
            let (Some(c), None) = (cs.next(), cs.next()) else {
                return Err(ForgeError::Tokenizer(format!(
                    "char vocab entry {k:?} is not a single character"
                )));
            };
            let idx = *id as usize;
            if idx >= itos.len() {
                return Err(ForgeError::Tokenizer(format!(
                    "char vocab id {id} out of range for a {}-entry vocab",
                    itos.len()
                )));
            }
            if filled[idx] {
                return Err(ForgeError::Tokenizer(format!(
                    "char vocab id {id} assigned twice"
                )));
            }
            itos[idx] = c;
            filled[idx] = true;
        }
        let stoi = itos
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Ok(CharTokenizer { itos, stoi })
    }

    /// Write `vocab.json` next to a checkpoint. Re-deriving the vocab at
    /// inference time from a different corpus would silently shift every id,
    /// so it ships with the weights.
    pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
        std::fs::write(path, self.to_json())?;
        Ok(())
    }

    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json(&std::fs::read_to_string(path)?)
    }
}

impl Tokenizer for CharTokenizer {
    /// Unknown characters are an error rather than a silent drop: a prompt
    /// containing them would otherwise generate a plausible continuation of
    /// something the user did not type. See [`CharTokenizer::encode_lossy`].
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        text.chars()
            .map(|c| {
                self.stoi.get(&c).copied().ok_or_else(|| {
                    ForgeError::Tokenizer(format!(
                        "character {c:?} is outside the {}-token vocabulary",
                        self.itos.len()
                    ))
                })
            })
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&i| self.itos.get(i as usize))
            .collect()
    }

    /// A char vocab has no partial-UTF-8 problem — every token is a whole
    /// character — but the streaming path calls this for both tokenizers.
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        self.decode(ids).into_bytes()
    }

    fn vocab_size(&self) -> usize {
        self.itos.len()
    }
}
