//! GPT-2 byte-level BPE tokenizer, implemented from scratch.
//!
//! Loads the original `vocab.json` + `merges.txt`. The split pattern uses a
//! negative lookahead, which the standard `regex` crate cannot express, so
//! `fancy-regex` is used (see roadmap "Known Pitfalls").

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::error::{ForgeError, Result};

const SPLIT_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

pub struct Gpt2Tokenizer {
    encoder: HashMap<String, u32>,
    decoder: HashMap<u32, String>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    byte_decoder: HashMap<char, u8>,
    pattern: fancy_regex::Regex,
    cache: Mutex<HashMap<String, Vec<String>>>,
}

/// GPT-2's reversible byte -> printable-unicode mapping.
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u16> = (b'!'..=b'~').map(u16::from).collect();
    bs.extend((0xA1u16..=0xACu16).chain(0xAEu16..=0xFFu16));
    let mut map = HashMap::new();
    for &b in &bs {
        map.insert(b as u8, char::from_u32(b as u32).unwrap());
    }
    let mut n = 0u32;
    for b in 0u16..256 {
        if !bs.contains(&b) {
            map.insert(b as u8, char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    map
}

impl Gpt2Tokenizer {
    pub fn from_files(vocab_path: impl AsRef<Path>, merges_path: impl AsRef<Path>) -> Result<Self> {
        let vocab_json = std::fs::read_to_string(vocab_path)?;
        let merges = std::fs::read_to_string(merges_path)?;
        Self::from_strs(&vocab_json, &merges)
    }

    /// Build from the contents of `vocab.json` and `merges.txt` — the primary
    /// form; on wasm the strings arrive from an HTTP fetch.
    pub fn from_strs(vocab_json: &str, merges: &str) -> Result<Self> {
        let encoder: HashMap<String, u32> = serde_json::from_str(vocab_json)?;
        let decoder = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        let mut bpe_ranks = HashMap::new();
        for (i, line) in merges
            .lines()
            .filter(|l| !l.starts_with("#version") && !l.trim().is_empty())
            .enumerate()
        {
            let mut parts = line.split(' ');
            let (a, b) = (
                parts.next().ok_or_else(|| bad_merge(line))?,
                parts.next().ok_or_else(|| bad_merge(line))?,
            );
            bpe_ranks.insert((a.to_string(), b.to_string()), i);
        }

        let byte_encoder = bytes_to_unicode();
        let byte_decoder = byte_encoder.iter().map(|(&b, &c)| (c, b)).collect();
        let pattern = fancy_regex::Regex::new(SPLIT_PATTERN)
            .map_err(|e| ForgeError::Tokenizer(format!("pattern: {e}")))?;
        Ok(Gpt2Tokenizer {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
            pattern,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Load `vocab.json` + `merges.txt` from a directory.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        Self::from_files(dir.join("vocab.json"), dir.join("merges.txt"))
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for m in self.pattern.find_iter(text) {
            let piece = m
                .map_err(|e| ForgeError::Tokenizer(format!("regex: {e}")))?
                .as_str();
            let mapped: String = piece.bytes().map(|b| self.byte_encoder[&b]).collect();
            for token in self.bpe(&mapped) {
                let id = self.encoder.get(&token).ok_or_else(|| {
                    ForgeError::Tokenizer(format!("token {token:?} not in vocab"))
                })?;
                ids.push(*id);
            }
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(ids)).into_owned()
    }

    /// Raw byte-level decode. Unlike [`Gpt2Tokenizer::decode`] this is exact
    /// and append-only per token (id sequence a ++ b decodes to
    /// bytes(a) ++ bytes(b)), which streaming builds on: a multi-byte UTF-8
    /// character split across BPE tokens completes once the next token's
    /// bytes arrive.
    pub fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|id| self.decoder.get(id))
            .flat_map(|s| s.chars())
            .filter_map(|c| self.byte_decoder.get(&c).copied())
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Standard BPE merge loop over a byte-mapped word.
    fn bpe(&self, word: &str) -> Vec<String> {
        if let Some(hit) = self.cache.lock().unwrap().get(word) {
            return hit.clone();
        }
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        while parts.len() > 1 {
            let best = parts
                .windows(2)
                .filter_map(|w| {
                    self.bpe_ranks
                        .get(&(w[0].clone(), w[1].clone()))
                        .map(|&r| (r, (w[0].clone(), w[1].clone())))
                })
                .min_by_key(|(r, _)| *r);
            let Some((_, (a, b))) = best else { break };
            let mut merged = Vec::with_capacity(parts.len());
            let mut i = 0;
            while i < parts.len() {
                if i + 1 < parts.len() && parts[i] == a && parts[i + 1] == b {
                    merged.push(format!("{a}{b}"));
                    i += 2;
                } else {
                    merged.push(parts[i].clone());
                    i += 1;
                }
            }
            parts = merged;
        }
        self.cache
            .lock()
            .unwrap()
            .insert(word.to_string(), parts.clone());
        parts
    }
}

fn bad_merge(line: &str) -> ForgeError {
    ForgeError::Tokenizer(format!("malformed merges line: {line:?}"))
}
