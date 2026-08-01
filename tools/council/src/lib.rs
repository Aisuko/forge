//! A council: several small GPT-2s that run on the same prompt in parallel,
//! exchange hidden states rather than text, and produce one character.
//!
//! Every expert branched from one ancestor checkpoint and was fine-tuned with
//! `wte`/`wpe` **frozen**, so all of them read the same token ids into the same
//! basis and all of them are decoded by the same wte-tied head. That is the
//! whole trick: it makes `Σ wᵢ·hᵢ` a vector the head can still read. Merge
//! hidden states from models that do *not* share an embedding table and the
//! result is noise that merely has the right shape.
//!
//! The router does no learning. Each expert's own hidden state decodes to its
//! own distribution; the expert that is most certain — lowest entropy — gets
//! the most weight, with `beta` setting how sharply. At `beta = 0` every expert
//! counts equally; as `beta` grows the most confident expert takes over.

use forge::{ForgeError, Gpt2, KvCache, Result, Sampler, Sampling, top_probs};

#[cfg(target_arch = "wasm32")]
pub mod wasm;

/// What one expert contributed to one character.
#[derive(Debug, Clone)]
pub struct ExpertStep {
    /// Router weight, in `[0, 1]`; the weights over all experts sum to 1.
    pub weight: f32,
    /// Entropy of this expert's own next-character distribution, in nats.
    /// Low means certain. This is what the router reads.
    pub entropy: f32,
    /// This expert's own top-n `(token id, probability)` — what it would have
    /// said on its own, which is the thing worth showing next to the merge.
    pub top: Vec<(u32, f32)>,
    /// `[n_embd]` — this expert's post-`ln_f` hidden state.
    pub hidden: Vec<f32>,
}

/// Everything one council step decided, in the order it decided it.
#[derive(Debug, Clone)]
pub struct CouncilStep {
    pub experts: Vec<ExpertStep>,
    /// `[n_embd]` — the merged hidden state, `Σ wᵢ·hᵢ`.
    pub hidden: Vec<f32>,
    /// Top-n of the merged distribution.
    pub top: Vec<(u32, f32)>,
    /// The character actually emitted.
    pub chosen: u32,
    /// Fraction of experts whose own first choice matches the council's.
    /// 1.0 means the council added nothing here; low means it genuinely
    /// arbitrated.
    pub consensus: f32,
}

pub struct Council {
    /// Display names, one per expert, in expert order.
    pub names: Vec<String>,
    /// Router sharpness. 0 weights every expert equally; larger lets the most
    /// confident expert dominate.
    pub beta: f32,
    models: Vec<Gpt2>,
    caches: Vec<KvCache>,
    sampler: Sampler,
}

impl Council {
    /// All experts must agree on their configuration — same depth, width and
    /// vocabulary — or their hidden states are not in the same space and the
    /// merge below is meaningless.
    pub fn new(models: Vec<Gpt2>, names: Vec<String>, seed: u64) -> Result<Council> {
        if models.is_empty() {
            return Err(ForgeError::Shape(
                "a council needs at least one expert".into(),
            ));
        }
        if models.len() != names.len() {
            return Err(ForgeError::Shape(format!(
                "{} experts but {} names",
                models.len(),
                names.len()
            )));
        }
        let c0 = &models[0].config;
        for (i, m) in models.iter().enumerate().skip(1) {
            let c = &m.config;
            if (c.n_embd, c.vocab_size, c.n_layer, c.n_head, c.n_ctx)
                != (c0.n_embd, c0.vocab_size, c0.n_layer, c0.n_head, c0.n_ctx)
            {
                return Err(ForgeError::Shape(format!(
                    "expert {i} ({}) has a different shape from expert 0 — \
                     hidden states from differently shaped models cannot be merged",
                    names[i]
                )));
            }
        }
        let caches = models
            .iter()
            .map(|m| m.new_cache())
            .collect::<Result<Vec<_>>>()?;
        Ok(Council {
            names,
            beta: DEFAULT_BETA,
            models,
            caches,
            sampler: Sampler::new(seed),
        })
    }

    pub fn n_experts(&self) -> usize {
        self.models.len()
    }

    pub fn n_embd(&self) -> usize {
        self.models[0].config.n_embd
    }

    pub fn vocab_size(&self) -> usize {
        self.models[0].config.vocab_size
    }

    pub fn n_ctx(&self) -> usize {
        self.models[0].config.n_ctx
    }

    /// Forget the conversation: fresh KV caches, fresh sampling stream.
    pub fn reset(&mut self, seed: u64) -> Result<()> {
        self.caches = self
            .models
            .iter()
            .map(|m| m.new_cache())
            .collect::<Result<Vec<_>>>()?;
        self.sampler.reseed(seed);
        Ok(())
    }

    /// One character: every expert runs, the router weighs them, the merge is
    /// decoded by the shared head.
    ///
    /// `ids` is the whole prompt on the first call and one token per call
    /// after that, exactly like [`Gpt2::logits_step`].
    pub fn step(&mut self, ids: &[u32], sampling: Sampling, top_n: usize) -> Result<CouncilStep> {
        let mut hidden = Vec::with_capacity(self.models.len());
        for (m, cache) in self.models.iter().zip(&mut self.caches) {
            hidden.push(m.hidden_step(ids, cache)?);
        }
        let mut logits = Vec::with_capacity(hidden.len());
        for (m, h) in self.models.iter().zip(&hidden) {
            logits.push(m.logits_from_hidden(h)?);
        }
        let merged = self.merge(&hidden, &logits);
        let merged_logits = self.models[0].logits_from_hidden(&merged)?;
        Ok(self.assemble(hidden, logits, merged, merged_logits, sampling, top_n))
    }

    /// Async form of [`Council::step`] — identical math, awaited readbacks so
    /// it works on wasm32.
    pub async fn step_async(
        &mut self,
        ids: &[u32],
        sampling: Sampling,
        top_n: usize,
    ) -> Result<CouncilStep> {
        let mut hidden = Vec::with_capacity(self.models.len());
        for (m, cache) in self.models.iter().zip(&mut self.caches) {
            hidden.push(m.hidden_step_async(ids, cache).await?);
        }
        let mut logits = Vec::with_capacity(hidden.len());
        for (m, h) in self.models.iter().zip(&hidden) {
            logits.push(m.logits_from_hidden_async(h).await?);
        }
        let merged = self.merge(&hidden, &logits);
        let merged_logits = self.models[0].logits_from_hidden_async(&merged).await?;
        Ok(self.assemble(hidden, logits, merged, merged_logits, sampling, top_n))
    }

    /// True when every expert's token embedding table is bit-identical to
    /// expert 0's — the invariant the whole merge rests on. Reads `wte` off
    /// the device, so this is a check to run once, not per step.
    pub fn embeddings_agree(&self) -> Result<bool> {
        let first = self.models[0].wte_host()?;
        for m in self.models.iter().skip(1) {
            if m.wte_host()? != first {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `wᵢ = softmax(−β·Hᵢ)` over the experts' entropies.
    fn weights(&self, logits: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
        let entropy: Vec<f32> = logits.iter().map(|l| entropy_nats(l)).collect();
        let scores: Vec<f32> = entropy.iter().map(|h| -self.beta * h).collect();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exp.iter().sum();
        (exp.iter().map(|e| e / total).collect(), entropy)
    }

    fn merge(&self, hidden: &[Vec<f32>], logits: &[Vec<f32>]) -> Vec<f32> {
        let (w, _) = self.weights(logits);
        let mut merged = vec![0.0f32; self.n_embd()];
        for (h, wi) in hidden.iter().zip(&w) {
            for (m, x) in merged.iter_mut().zip(h) {
                *m += wi * x;
            }
        }
        merged
    }

    fn assemble(
        &mut self,
        hidden: Vec<Vec<f32>>,
        logits: Vec<Vec<f32>>,
        merged: Vec<f32>,
        merged_logits: Vec<f32>,
        sampling: Sampling,
        top_n: usize,
    ) -> CouncilStep {
        let (w, entropy) = self.weights(&logits);
        let top_n = top_n.max(1);
        let experts: Vec<ExpertStep> = hidden
            .into_iter()
            .zip(logits)
            .zip(w)
            .zip(entropy)
            .map(|(((hidden, l), weight), entropy)| ExpertStep {
                weight,
                entropy,
                top: top_probs(&l, top_n),
                hidden,
            })
            .collect();
        let chosen = self.sampler.pick(&merged_logits, sampling);
        let agreed = experts
            .iter()
            .filter(|e| e.top.first().map(|(id, _)| *id) == Some(chosen))
            .count();
        CouncilStep {
            consensus: agreed as f32 / experts.len() as f32,
            experts,
            hidden: merged,
            top: top_probs(&merged_logits, top_n),
            chosen,
        }
    }
}

/// Router sharpness that separates the experts without letting one silence the
/// rest: over 65 characters the entropy spread between a confident and an
/// unsure expert is a few tenths of a nat, so β around 2 turns that into a
/// visible 2-3x weight ratio.
pub const DEFAULT_BETA: f32 = 2.0;

/// Entropy of `softmax(logits)`, in nats.
fn entropy_nats(logits: &[f32]) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    -exp.iter()
        .map(|e| {
            let p = e / total;
            if p > 0.0 { p * p.ln() } else { 0.0 }
        })
        .sum::<f32>()
}
