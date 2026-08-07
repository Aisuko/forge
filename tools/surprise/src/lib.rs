//! How surprised a model was by text that is already there.
//!
//! This is reading, not writing: for each position the model is asked what it
//! expected *before* seeing the character that actually followed, and the
//! answer is scored against what did. It is a teacher-forced scoring pass, so
//! the whole sequence costs **one forward pass** rather than `t` decode steps —
//! the model reads a paragraph in the time it would take to generate one
//! character.
//!
//! The runtime keeps the primitive this stands on: [`Gpt2::forward`] returns
//! the `[t, vocab]` logits, and everything below is host arithmetic over them —
//! a log-sum-exp, a gather, a partial sort. Which of those numbers a reader
//! should be shown, and in what units, is a page's opinion and lives here.

use forge::{ForgeError, Gpt2, Result, top_probs};

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// Per-position surprisal from [`surprisal`]. `bits` is the length of the
/// input and index 0 is a placeholder — nothing precedes the first token.
///
/// The candidates are flattened rather than `Vec<Vec<_>>`: it matches how
/// `bits` is already stored and costs no allocation per character. Column 0 is
/// the model's own first choice, so [`Surprisal::top`] and
/// [`Surprisal::top_p`] read it off directly rather than keeping a second copy
/// that could drift from this one.
#[derive(Clone, Debug)]
pub struct Surprisal {
    /// `-log2 p(ids[i] | ids[..i])`, in bits.
    pub bits: Vec<f32>,
    /// `k` candidates per position, descending by probability:
    /// position `i` occupies `alt_ids[i*k .. (i+1)*k]`.
    pub alt_ids: Vec<u32>,
    /// The matching probabilities, same layout, in `0..=1`.
    pub alt_p: Vec<f32>,
    /// Candidates per position. Always at least 1.
    pub k: usize,
}

impl Surprisal {
    /// The token the model would have chosen at position `i`.
    pub fn top(&self, i: usize) -> u32 {
        self.alt_ids[i * self.k]
    }

    /// How probable the model thought its own choice was, in `0..=1`.
    pub fn top_p(&self, i: usize) -> f32 {
        self.alt_p[i * self.k]
    }

    /// Number of scored positions.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }
}

/// Score `ids` for surprise: one forward pass, every position at once.
///
/// `bits[i]` is `-log2 p(ids[i] | ids[..i])`: 0 means "entirely expected", and
/// `log2(vocab_size)` is what a uniform guess would score. `bits[0]` is 0 —
/// nothing precedes the first token, so nothing about it can be a surprise.
///
/// `k` is how many alternatives to keep per position — the tokens the model
/// genuinely weighed there, which is what makes a surprise legible: "it
/// expected `e`, and was also considering `a` and `o`" says more than a
/// number, and is the only truthful thing a page can show a reader in place of
/// a character that has not resolved yet. `k = 1` keeps only the winner.
///
/// A free function rather than an extension trait on [`Gpt2`]: method syntax
/// would buy one character at the call site and cost a `use` at every one.
///
/// The softmax and the ranking are done on the host deliberately. The GPU ops
/// that would do them (`ops::softmax` over `[t, vocab]`, and `ops::gather_nll`)
/// sit behind the runtime's `train` feature, and for a character-level
/// vocabulary the host loop is free. [`forge::top_probs`] partitions rather
/// than sorts, so `k` costs a scan and not a `vocab log vocab`. Note the cost
/// model for a BPE model, though: the readback is `t × vocab × 4` bytes, which
/// for GPT-2's 50257-token vocabulary is ~196 KB per position.
pub async fn surprisal(model: &Gpt2, ids: &[u32], k: usize) -> Result<Surprisal> {
    let t = ids.len();
    if t == 0 {
        return Err(ForgeError::Shape(
            "surprisal needs a non-empty sequence".into(),
        ));
    }
    let vocab = model.config.vocab_size;
    let k = k.clamp(1, vocab);
    let logits = model.forward(ids)?.to_vec_f32_async().await?;

    let mut out = Surprisal {
        bits: vec![0.0; t],
        // Position 0 has no predecessor to have predicted it, so its row is
        // the character itself at probability 0 — a placeholder the page
        // never spins.
        alt_ids: vec![ids[0]; t * k],
        alt_p: vec![0.0; t * k],
        k,
    };
    // Position i is predicted by row i-1: the model saw ids[..i] and the
    // logits it produced there are its guess about ids[i].
    for i in 1..t {
        let row = &logits[(i - 1) * vocab..i * vocab];
        // Log-sum-exp in the stable form; a char model's logits are small,
        // but a shifted exp costs nothing and cannot overflow.
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|l| (l - max).exp()).sum();
        let log_z = max + sum.ln();

        let target = ids[i] as usize;
        if target >= vocab {
            return Err(ForgeError::Shape(format!(
                "token id {target} >= vocab_size {vocab}"
            )));
        }
        // nats -> bits: a bit is the unit a reader can reason about.
        out.bits[i] = -(row[target] - log_z) / std::f32::consts::LN_2;

        for (j, (id, p)) in top_probs(row, k).into_iter().enumerate() {
            out.alt_ids[i * k + j] = id;
            out.alt_p[i * k + j] = p;
        }
    }
    Ok(out)
}
