//! GPT-2 model assembly and generation.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

#[cfg(feature = "train")]
use crate::autograd::{TVar, Tape};
use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::nn::{Embedding, LayerNorm, Linear};
use crate::ops::{self, MatmulSpec};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

fn default_eps() -> f32 {
    1e-5
}

#[derive(Debug, Clone)]
pub struct Gpt2Config {
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub layer_norm_epsilon: f32,
    pub eos_token_id: Option<u32>,
}

/// HF config.json may carry `n_ctx`, `n_positions`, or both.
#[derive(Deserialize)]
struct RawConfig {
    n_layer: usize,
    n_head: usize,
    n_embd: usize,
    #[serde(default)]
    n_ctx: Option<usize>,
    #[serde(default)]
    n_positions: Option<usize>,
    vocab_size: usize,
    #[serde(default = "default_eps")]
    layer_norm_epsilon: f32,
    #[serde(default)]
    eos_token_id: Option<u32>,
}

impl Gpt2Config {
    /// GPT-2 small (124M).
    pub fn gpt2() -> Self {
        Gpt2Config {
            n_layer: 12,
            n_head: 12,
            n_embd: 768,
            n_ctx: 1024,
            vocab_size: 50257,
            layer_norm_epsilon: 1e-5,
            eos_token_id: Some(50256),
        }
    }

    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json_str(&text)
    }

    /// Parse an HF `config.json` string — the primary form; on wasm the
    /// string arrives from an HTTP fetch.
    pub fn from_json_str(text: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(text)?;
        let n_ctx = raw.n_ctx.or(raw.n_positions).ok_or_else(|| {
            ForgeError::Json(serde::de::Error::custom("missing n_ctx/n_positions"))
        })?;
        Ok(Gpt2Config {
            n_layer: raw.n_layer,
            n_head: raw.n_head,
            n_embd: raw.n_embd,
            n_ctx,
            vocab_size: raw.vocab_size,
            layer_norm_epsilon: raw.layer_norm_epsilon,
            eos_token_id: raw.eos_token_id,
        })
    }
}

struct Block {
    ln_1: LayerNorm,
    attn_qkv: Linear,
    attn_proj: Linear,
    ln_2: LayerNorm,
    mlp_fc: Linear,
    mlp_proj: Linear,
}

pub struct Gpt2 {
    emb: Embedding,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    pub config: Gpt2Config,
}

#[derive(Debug, Clone, Copy)]
pub enum Sampling {
    Greedy,
    TopK {
        k: usize,
        temperature: f32,
        seed: u64,
    },
}

/// One block's attention probabilities for one decode step, read back from
/// the device.
///
/// `probs` is the row-major `[n_head, q_len, kv_len]` tensor the model
/// actually attended with — captured after the softmax, not recomputed — so a
/// visualization built from it shows the same arithmetic that produced the
/// text. `ops::softmax` is out-of-place, so capturing it perturbs nothing.
#[derive(Debug, Clone)]
pub struct AttnStep {
    pub layer: usize,
    pub n_head: usize,
    /// Query positions in this step: the whole prompt on prefill, 1 per decode.
    pub q_len: usize,
    /// Past positions attended to, including the queries themselves.
    pub kv_len: usize,
    pub probs: Vec<f32>,
}

/// One block's Q/K/V, MLP and output tensors for one step, read back from the
/// device. Captured only for the first `detail_layers` blocks — the expensive
/// kinds — while [`AttnStep`] is captured for every block.
///
/// `q`/`k`/`v` are the tensors for the *new* positions only (`[n_head, q_len,
/// head_dim]`), which is what the step computed: the older keys and values
/// live in the cache and were never recomputed.
#[derive(Debug, Clone)]
pub struct LayerDetail {
    pub layer: usize,
    /// `[q_len, n_embd]`, the block's first LayerNorm.
    pub ln1_out: Vec<f32>,
    /// `[n_head, q_len, head_dim]`
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    /// `[n_head, q_len, kv_len]` — `q · kᵢ / √head_dim` *before* the causal
    /// mask and the softmax. The masked triangle only means something next to
    /// the full rectangle it was cut from.
    pub scores: Vec<f32>,
    /// `[q_len, n_embd]` — the heads concatenated, before the output
    /// projection. Kept separate from `attn_proj_out` because this is the one
    /// place `n_head × head_dim` becomes `n_embd`.
    pub attn_head_out: Vec<f32>,
    /// `[q_len, n_embd]`, after the attention output projection.
    pub attn_proj_out: Vec<f32>,
    /// `[q_len, n_embd]`, after the first residual add.
    pub resid_attn: Vec<f32>,
    /// `[q_len, n_embd]`, the block's second LayerNorm.
    pub ln2_out: Vec<f32>,
    /// `[q_len, 4 * n_embd]`, post-GELU.
    pub mlp_hidden: Vec<f32>,
    /// `[q_len, n_embd]`, after the second residual add.
    pub block_out: Vec<f32>,
}

/// Everything one decode step computed, read back in a single batched round
/// trip. A strict superset of [`AttnStep`]: `attn` is exactly what
/// [`Gpt2::logits_step_attn_async`] reports for the same step.
#[derive(Debug, Clone)]
pub struct StepTrace {
    /// Query positions in this step: the whole prompt on prefill, 1 per decode.
    pub q_len: usize,
    /// Past positions attended to, including the queries themselves.
    pub kv_len: usize,
    pub n_head: usize,
    pub n_embd: usize,
    /// `[q_len, n_embd]` — token + position embedding, before block 0.
    /// Empty when `detail_layers` is 0.
    pub embedding: Vec<f32>,
    /// Every block, in layer order.
    pub attn: Vec<AttnStep>,
    /// The first `detail_layers` blocks, in layer order.
    pub detail: Vec<LayerDetail>,
    /// `[q_len, n_embd]` — after the final LayerNorm, i.e. what the LM head
    /// multiplies. Empty when `detail_layers` is 0.
    pub ln_f_out: Vec<f32>,
    /// Top-n `(token id, probability)`, most likely first. Empty when `top_n`
    /// is 0.
    pub top: Vec<(u32, f32)>,
}

/// What a captured tensor is, so the batched readback can be interpreted
/// without guessing from rank.
#[derive(Debug, Clone, Copy)]
enum ProbeKind {
    Embedding,
    Ln1Out { layer: usize },
    Query { layer: usize },
    Key { layer: usize },
    Value { layer: usize },
    Scores { layer: usize },
    Attention { layer: usize },
    AttnHeadOut { layer: usize },
    AttnProjOut { layer: usize },
    ResidAttn { layer: usize },
    Ln2Out { layer: usize },
    MlpHidden { layer: usize },
    BlockOut { layer: usize },
    LnFOut,
}

/// Capture request + accumulator, threaded through the forward pass.
///
/// Every `push` is an `Arc` clone of a tensor the step computes anyway, so the
/// probing and non-probing paths run identical arithmetic. `detail_layers`
/// bounds the expensive kinds; `Attention` is captured for every layer
/// regardless, because the folded block stack still needs it.
struct Probe {
    detail_layers: usize,
    kinds: Vec<ProbeKind>,
    tensors: Vec<Tensor>,
}

impl Probe {
    fn new(detail_layers: usize) -> Self {
        Probe {
            detail_layers,
            kinds: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn detail(&self, layer: usize) -> bool {
        layer < self.detail_layers
    }

    fn push(&mut self, kind: ProbeKind, t: &Tensor) {
        self.kinds.push(kind);
        self.tensors.push(t.clone());
    }
}

/// Preallocated per-layer K/V tensors (`[n_head, n_ctx, head_dim]`) for
/// incremental decode. `len` positions are filled; new tokens append.
pub struct KvCache {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    len: usize,
}

impl KvCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Gpt2 {
    /// Load a checkpoint of either format from disk, by content and not by
    /// extension — see [`Gpt2::from_checkpoint_bytes`]. Pair it with
    /// [`crate::serialization::checkpoint_in_dir`] to load a model directory
    /// without caring which format it ships.
    pub fn from_checkpoint(
        path: impl AsRef<Path>,
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_checkpoint_bytes(&bytes, config, device)
    }

    /// Load a checkpoint of either format, chosen by sniffing the leading
    /// magic bytes: `.fzm`'s `"FZM1"` header, otherwise `.safetensors`. This
    /// is what the wasm bindings use, so a page can fetch either extension
    /// without the JS side knowing which codec is behind it.
    pub fn from_checkpoint_bytes(
        bytes: &[u8],
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        if bytes.starts_with(b"FZM1") {
            Self::from_fzm_bytes(bytes, config, device)
        } else {
            Self::from_safetensors_bytes(bytes, config, device)
        }
    }

    /// Load HF GPT-2 weights. `path` is the .safetensors file. Weight names
    /// may or may not carry the "transformer." prefix; both are accepted.
    /// The LM head is tied to `wte` (the checkpoint has no separate tensor).
    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_safetensors_bytes(&bytes, config, device)
    }

    /// Load HF GPT-2 weights from in-memory .safetensors bytes — the primary
    /// form; on wasm the bytes arrive from an HTTP fetch. `wte` never touches
    /// the device un-chunked: it is uploaded row-chunked straight from the
    /// file buffer.
    pub fn from_safetensors_bytes(
        bytes: &[u8],
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| ForgeError::SafeTensors(format!("deserialize: {e}")))?;
        let view = |name: &str| -> Result<safetensors::tensor::TensorView<'_>> {
            st.tensor(name)
                .or_else(|_| st.tensor(&format!("transformer.{name}")))
                .map_err(|_| ForgeError::SafeTensors(format!("missing tensor {name}")))
        };
        let host = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let v = view(name)?;
            if v.dtype() != safetensors::tensor::Dtype::F32 {
                return Err(ForgeError::SafeTensors(format!(
                    "{name}: expected f32, got {:?}",
                    v.dtype()
                )));
            }
            Ok((v.shape().to_vec(), bytemuck::pod_collect_to_vec(v.data())))
        };
        let take = |name: &str| -> Result<Tensor> {
            let (shape, data) = host(name)?;
            Tensor::from_f32(&data, shape, device)
        };
        let eps = config.layer_norm_epsilon;
        // wte (~147 MiB for GPT-2) can exceed a single GPU binding, so it
        // goes host buffer -> row-chunked device tensors directly.
        let (_, wte_host) = host("wte.weight")?;
        let emb = Embedding::from_host_wte(
            &wte_host,
            config.vocab_size,
            config.n_embd,
            take("wpe.weight")?,
            device,
        )?;
        drop(wte_host);
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: take(&format!("h.{i}.ln_1.weight"))?,
                    beta: take(&format!("h.{i}.ln_1.bias"))?,
                    eps,
                },
                attn_qkv: Linear {
                    w: take(&format!("h.{i}.attn.c_attn.weight"))?,
                    b: Some(take(&format!("h.{i}.attn.c_attn.bias"))?),
                },
                attn_proj: Linear {
                    w: take(&format!("h.{i}.attn.c_proj.weight"))?,
                    b: Some(take(&format!("h.{i}.attn.c_proj.bias"))?),
                },
                ln_2: LayerNorm {
                    gamma: take(&format!("h.{i}.ln_2.weight"))?,
                    beta: take(&format!("h.{i}.ln_2.bias"))?,
                    eps,
                },
                mlp_fc: Linear {
                    w: take(&format!("h.{i}.mlp.c_fc.weight"))?,
                    b: Some(take(&format!("h.{i}.mlp.c_fc.bias"))?),
                },
                mlp_proj: Linear {
                    w: take(&format!("h.{i}.mlp.c_proj.weight"))?,
                    b: Some(take(&format!("h.{i}.mlp.c_proj.bias"))?),
                },
            });
        }
        let ln_f = LayerNorm {
            gamma: take("ln_f.weight")?,
            beta: take("ln_f.bias")?,
            eps,
        };
        Ok(Gpt2 {
            emb,
            blocks,
            ln_f,
            config,
        })
    }

    /// Load a `.fzm` q4 checkpoint. `path` is the `.fzm` file.
    pub fn from_fzm(path: impl AsRef<Path>, config: Gpt2Config, device: &Device) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_fzm_bytes(&bytes, config, device)
    }

    /// Load a `.fzm` q4 checkpoint from in-memory bytes. Every tensor is
    /// dequantized to f32 host-side on parse (see
    /// [`crate::serialization::fzm`]), then follows the exact same
    /// host-\>device upload path as [`Gpt2::from_safetensors_bytes`] — there
    /// is no packed GPU-side q4 kernel yet.
    pub fn from_fzm_bytes(bytes: &[u8], config: Gpt2Config, device: &Device) -> Result<Self> {
        let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> =
            crate::serialization::fzm::load_fzm_q4(bytes)?
                .into_iter()
                .map(|(name, shape, data)| (name, (shape, data)))
                .collect();
        fn host(
            map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
            name: &str,
        ) -> Result<(Vec<usize>, Vec<f32>)> {
            map.remove(name)
                .or_else(|| map.remove(&format!("transformer.{name}")))
                .ok_or_else(|| ForgeError::Fzm(format!("missing tensor {name}")))
        }
        fn take(
            map: &mut HashMap<String, (Vec<usize>, Vec<f32>)>,
            name: &str,
            device: &Device,
        ) -> Result<Tensor> {
            let (shape, data) = host(map, name)?;
            Tensor::from_f32(&data, shape, device)
        }
        let eps = config.layer_norm_epsilon;
        let (_, wte_host) = host(&mut map, "wte.weight")?;
        let emb = Embedding::from_host_wte(
            &wte_host,
            config.vocab_size,
            config.n_embd,
            take(&mut map, "wpe.weight", device)?,
            device,
        )?;
        drop(wte_host);
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: take(&mut map, &format!("h.{i}.ln_1.weight"), device)?,
                    beta: take(&mut map, &format!("h.{i}.ln_1.bias"), device)?,
                    eps,
                },
                attn_qkv: Linear {
                    w: take(&mut map, &format!("h.{i}.attn.c_attn.weight"), device)?,
                    b: Some(take(&mut map, &format!("h.{i}.attn.c_attn.bias"), device)?),
                },
                attn_proj: Linear {
                    w: take(&mut map, &format!("h.{i}.attn.c_proj.weight"), device)?,
                    b: Some(take(&mut map, &format!("h.{i}.attn.c_proj.bias"), device)?),
                },
                ln_2: LayerNorm {
                    gamma: take(&mut map, &format!("h.{i}.ln_2.weight"), device)?,
                    beta: take(&mut map, &format!("h.{i}.ln_2.bias"), device)?,
                    eps,
                },
                mlp_fc: Linear {
                    w: take(&mut map, &format!("h.{i}.mlp.c_fc.weight"), device)?,
                    b: Some(take(&mut map, &format!("h.{i}.mlp.c_fc.bias"), device)?),
                },
                mlp_proj: Linear {
                    w: take(&mut map, &format!("h.{i}.mlp.c_proj.weight"), device)?,
                    b: Some(take(&mut map, &format!("h.{i}.mlp.c_proj.bias"), device)?),
                },
            });
        }
        let ln_f = LayerNorm {
            gamma: take(&mut map, "ln_f.weight", device)?,
            beta: take(&mut map, "ln_f.bias", device)?,
            eps,
        };
        Ok(Gpt2 {
            emb,
            blocks,
            ln_f,
            config,
        })
    }

    fn attention(&self, block: &Block, h: &Tensor) -> Result<Tensor> {
        let n_head = self.config.n_head;
        let hd = self.config.n_embd / n_head;
        let qkv = block.attn_qkv.forward(h)?; // [t, 3c]
        let (q, k, v) = ops::split_heads(&qkv, n_head)?; // [h, t, hd] each
        let att = ops::matmul(
            &q,
            &k,
            None,
            MatmulSpec {
                trans_b: true,
                alpha: 1.0 / (hd as f32).sqrt(),
                ..Default::default()
            },
        )?; // [h, t, t]
        let att = ops::softmax(&att, true, 0)?;
        let y = ops::matmul(&att, &v, None, MatmulSpec::default())?; // [h, t, hd]
        let y = ops::merge_heads(&y)?; // [t, c]
        block.attn_proj.forward(&y)
    }

    /// Attention over new tokens only, appending their K/V to the cache and
    /// attending to all `len + t` cached positions.
    ///
    /// `probe`, when present, collects the post-softmax probabilities — and,
    /// for a detail layer, Q/K/V as well. Every capture is an `Arc` handle to
    /// a tensor that is computed regardless, so the probing and non-probing
    /// paths run identical arithmetic.
    #[allow(clippy::too_many_arguments)] // the cache slots and the probe are
    // all per-call state; bundling them would only move the argument list.
    fn attention_cached(
        &self,
        layer: usize,
        block: &Block,
        h: &Tensor,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        len: usize,
        mut probe: Option<&mut Probe>,
    ) -> Result<Tensor> {
        let n_head = self.config.n_head;
        let hd = self.config.n_embd / n_head;
        let qkv = block.attn_qkv.forward(h)?; // [t, 3c]
        let (q, k, v) = ops::split_heads(&qkv, n_head)?; // [h, t, hd]
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
            p.push(ProbeKind::Query { layer }, &q);
            p.push(ProbeKind::Key { layer }, &k);
            p.push(ProbeKind::Value { layer }, &v);
        }
        let t = q.shape().dim(1);
        ops::kv_append(k_cache, &k, len)?;
        ops::kv_append(v_cache, &v, len)?;
        let kv_len = len + t;
        let scores = ops::matmul(
            &q,
            k_cache,
            None,
            MatmulSpec {
                trans_b: true,
                alpha: 1.0 / (hd as f32).sqrt(),
                b_rows: Some(kv_len),
                ..Default::default()
            },
        )?; // [h, t, kv_len]
        // The raw dot products, before the mask and the softmax: the causal
        // triangle the next capture holds is only legible beside the full
        // rectangle it was cut from.
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
            p.push(ProbeKind::Scores { layer }, &scores);
        }
        let att = ops::softmax(&scores, true, len)?; // off = kv_len - t
        if let Some(p) = probe.as_deref_mut() {
            p.push(ProbeKind::Attention { layer }, &att);
        }
        let y = ops::matmul(
            &att,
            v_cache,
            None,
            MatmulSpec {
                b_rows: Some(kv_len),
                ..Default::default()
            },
        )?; // [h, t, hd]
        let y = ops::merge_heads(&y)?;
        // Concatenation and projection are two stages of the picture, and
        // returning only the projected tensor hides the one place
        // `n_head × head_dim` becomes `n_embd`.
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
            p.push(ProbeKind::AttnHeadOut { layer }, &y);
        }
        let proj = block.attn_proj.forward(&y)?;
        if let Some(p) = probe.filter(|p| p.detail(layer)) {
            p.push(ProbeKind::AttnProjOut { layer }, &proj);
        }
        Ok(proj)
    }

    /// Hidden states after the final LayerNorm: [t, n_embd].
    fn hidden(&self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(ForgeError::Shape("empty token sequence".into()));
        }
        if ids.len() > self.config.n_ctx {
            return Err(ForgeError::Shape(format!(
                "sequence length {} exceeds n_ctx {}",
                ids.len(),
                self.config.n_ctx
            )));
        }
        let device = self.emb.wpe.device();
        // One command buffer for the whole pass instead of one per kernel.
        let _scope = device.dispatch_scope();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let mut x = self.emb.forward(&ids_t, 0)?;
        for block in &self.blocks {
            let attn_out = self.attention(block, &block.ln_1.forward(&x)?)?;
            x = ops::add(&x, &attn_out)?;
            let mlp_in = block.ln_2.forward(&x)?;
            let mlp_out = block
                .mlp_proj
                .forward(&ops::gelu(&block.mlp_fc.forward(&mlp_in)?)?)?;
            x = ops::add(&x, &mlp_out)?;
        }
        self.ln_f.forward(&x)
    }

    /// Hidden states for `ids` (new tokens) continuing from `cache`;
    /// appends their K/V and advances `cache.len`. `probe` is threaded through
    /// to [`Gpt2::attention_cached`] and collects the embedding, every block's
    /// attention, and the detail layers' Q/K/V, MLP and block outputs.
    fn hidden_cached(
        &self,
        ids: &[u32],
        cache: &mut KvCache,
        mut probe: Option<&mut Probe>,
    ) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(ForgeError::Shape("empty token sequence".into()));
        }
        let pos = cache.len;
        if pos + ids.len() > self.config.n_ctx {
            return Err(ForgeError::Shape(format!(
                "sequence length {}+{} exceeds n_ctx {}",
                pos,
                ids.len(),
                self.config.n_ctx
            )));
        }
        let device = self.emb.wpe.device();
        // As in `hidden`: one submit for the step, not one per kernel. The
        // probe paths read tensors back mid-pass, which flushes and reopens —
        // correct, and the reason drawing the attention costs what it does.
        let _scope = device.dispatch_scope();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let mut x = self.emb.forward(&ids_t, pos)?;
        // The embedding is only worth a round trip when the detail stages are
        // being drawn; `detail_layers == 0` is the cheap attention-only probe.
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail_layers > 0) {
            p.push(ProbeKind::Embedding, &x);
        }
        for (layer, (block, (kc, vc))) in self
            .blocks
            .iter()
            .zip(cache.k.iter_mut().zip(cache.v.iter_mut()))
            .enumerate()
        {
            // Bound rather than passed inline so it can be captured: the
            // walkthrough draws a LayerNorm as its own stage.
            let ln1 = block.ln_1.forward(&x)?;
            if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
                p.push(ProbeKind::Ln1Out { layer }, &ln1);
            }
            let attn_out =
                self.attention_cached(layer, block, &ln1, kc, vc, pos, probe.as_deref_mut())?;
            x = ops::add(&x, &attn_out)?;
            if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
                p.push(ProbeKind::ResidAttn { layer }, &x);
            }
            let mlp_in = block.ln_2.forward(&x)?;
            if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
                p.push(ProbeKind::Ln2Out { layer }, &mlp_in);
            }
            let mlp_hidden = ops::gelu(&block.mlp_fc.forward(&mlp_in)?)?;
            if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
                p.push(ProbeKind::MlpHidden { layer }, &mlp_hidden);
            }
            let mlp_out = block.mlp_proj.forward(&mlp_hidden)?;
            x = ops::add(&x, &mlp_out)?;
            if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
                p.push(ProbeKind::BlockOut { layer }, &x);
            }
        }
        cache.len += ids.len();
        let out = self.ln_f.forward(&x)?;
        if let Some(p) = probe.filter(|p| p.detail_layers > 0) {
            p.push(ProbeKind::LnFOut, &out);
        }
        Ok(out)
    }

    /// Empty KV cache sized for this model's full context.
    pub fn new_cache(&self) -> Result<KvCache> {
        let device = self.emb.wpe.device();
        let hd = self.config.n_embd / self.config.n_head;
        let shape = [self.config.n_head, self.config.n_ctx, hd];
        let mut k = Vec::with_capacity(self.config.n_layer);
        let mut v = Vec::with_capacity(self.config.n_layer);
        for _ in 0..self.config.n_layer {
            k.push(Tensor::zeros(shape, &device)?);
            v.push(Tensor::zeros(shape, &device)?);
        }
        Ok(KvCache { k, v, len: 0 })
    }

    /// Full logits [t, vocab] — used by verification tests.
    /// The LM head is wte-tied: logits = h @ wte^T.
    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        // Nests inside `hidden`'s scope so the wte-tied head joins the same
        // command buffer: one submit for the whole step.
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden(ids)?;
        ops::matmul_chunked_transb(&h, &self.emb.wte_chunks, 1.0)
    }

    /// Logits for the last position only, recomputing the full context
    /// (no cache) — the reference decode path used by verification tests.
    pub fn logits_last(&self, ids: &[u32]) -> Result<Vec<f32>> {
        // As in `forward`: the head joins the body's command buffer.
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden(ids)?;
        let last = h.narrow_rows(ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)?.to_vec_f32()
    }

    /// Last-position logits for `ids` continuing from `cache` (incremental
    /// decode: pass the full prompt once, then one token at a time).
    pub fn logits_step(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        // As in `forward`: the head joins the body's command buffer.
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden_cached(ids, cache, None)?;
        let last = h.narrow_rows(ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)?.to_vec_f32()
    }

    /// Async form of [`Gpt2::logits_step`] — identical math; the readback is
    /// awaited so it works on wasm32.
    pub async fn logits_step_async(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        // As in `forward`: the head joins the body's command buffer.
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden_cached(ids, cache, None)?;
        let last = h.narrow_rows(ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)?
            .to_vec_f32_async()
            .await
    }

    /// The last position's hidden state **after the final LayerNorm** —
    /// `[n_embd]`, exactly the vector [`Gpt2::logits_step`] hands to the head.
    ///
    /// This is the only stage at which two models' activations can be merged
    /// and still decode: it is what `wte^T` multiplies, so models that share a
    /// (frozen) `wte` share a basis here and nowhere else. Anything deeper —
    /// per-head Q/K/V, the post-GELU MLP hidden — is private to one model.
    pub fn hidden_step(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let h = self.hidden_cached(ids, cache, None)?;
        h.narrow_rows(ids.len() - 1, 1)?.to_vec_f32()
    }

    /// Async form of [`Gpt2::hidden_step`] — identical math, awaited readback
    /// so it works on wasm32.
    pub async fn hidden_step_async(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let h = self.hidden_cached(ids, cache, None)?;
        h.narrow_rows(ids.len() - 1, 1)?.to_vec_f32_async().await
    }

    /// Logits for a hidden state supplied from outside: `h @ wte^T`, the same
    /// wte-tied head every other decode path applies. Lets a vector this model
    /// did not produce — a merge of several models' [`Gpt2::hidden_step`]
    /// outputs, say — be decoded to a distribution over the vocabulary.
    pub fn logits_from_hidden(&self, h: &[f32]) -> Result<Vec<f32>> {
        ops::matmul_chunked_transb(&self.hidden_tensor(h)?, &self.emb.wte_chunks, 1.0)?.to_vec_f32()
    }

    /// Async form of [`Gpt2::logits_from_hidden`].
    pub async fn logits_from_hidden_async(&self, h: &[f32]) -> Result<Vec<f32>> {
        ops::matmul_chunked_transb(&self.hidden_tensor(h)?, &self.emb.wte_chunks, 1.0)?
            .to_vec_f32_async()
            .await
    }

    /// The token embedding table as host floats, chunks reassembled. Exists so
    /// a council can verify its experts really do share one `wte` — the
    /// invariant that makes merging their hidden states mean anything.
    pub fn wte_host(&self) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity(self.config.vocab_size * self.config.n_embd);
        for chunk in &self.emb.wte_chunks {
            out.extend(chunk.to_vec_f32()?);
        }
        Ok(out)
    }

    fn hidden_tensor(&self, h: &[f32]) -> Result<Tensor> {
        if h.len() != self.config.n_embd {
            return Err(ForgeError::Shape(format!(
                "hidden state has {} elements, expected n_embd {}",
                h.len(),
                self.config.n_embd
            )));
        }
        Tensor::from_f32(h, [1, self.config.n_embd], &self.emb.wpe.device())
    }

    /// [`Gpt2::logits_step_async`] plus every block's attention probabilities
    /// for this step, in layer order.
    ///
    /// The cheap probe: exactly [`Gpt2::logits_step_trace_async`] with no
    /// detail layers and no top-n, so the two can never disagree about what
    /// the model attended with.
    pub async fn logits_step_attn_async(
        &self,
        ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<(Vec<f32>, Vec<AttnStep>)> {
        let (logits, trace) = self.logits_step_trace_async(ids, cache, 0, 0).await?;
        Ok((logits, trace.attn))
    }

    /// [`Gpt2::logits_step_async`] plus a full calculation trace for this step:
    /// the embedding, every block's attention, the whole forward pass through
    /// the first `detail_layers` blocks (both LayerNorms, Q/K/V, pre-softmax
    /// scores, the concatenated heads and their projection, both residual
    /// adds, post-GELU MLP), the final LayerNorm, and the `top_n` most likely
    /// next tokens with their probabilities.
    ///
    /// The logits are identical to `logits_step_async` — the probe only reads
    /// tensors the step computes anyway. The whole step is enqueued first, and
    /// every captured tensor comes back in a *single* batched readback: one at
    /// a time they cost ~2.7x the decode itself, since the price is the round
    /// trip rather than the bytes.
    ///
    /// `detail_layers` is the cost knob. At 0 this is the attention-only probe
    /// and nothing but the post-softmax tensors is read. Each detail layer adds
    /// `q_len * (13 * n_embd + n_head * kv_len)` floats, so a long prefill is
    /// the only expensive call — a decode step is `q_len == 1`.
    pub async fn logits_step_trace_async(
        &self,
        ids: &[u32],
        cache: &mut KvCache,
        detail_layers: usize,
        top_n: usize,
    ) -> Result<(Vec<f32>, StepTrace)> {
        let q_len = ids.len();
        let detail_layers = detail_layers.min(self.config.n_layer);
        let mut probe = Probe::new(detail_layers);
        let h = self.hidden_cached(ids, cache, Some(&mut probe))?;
        let last = h.narrow_rows(q_len - 1, 1)?;
        let Probe {
            kinds,
            mut tensors,
            detail_layers,
        } = probe;
        // Logits last, so the kinds stay aligned with the tensors that
        // produced them and the shapes below line up.
        let shapes: Vec<Vec<usize>> = tensors.iter().map(|t| t.shape().dims().to_vec()).collect();
        tensors.push(ops::matmul_chunked_transb(
            &last,
            &self.emb.wte_chunks,
            1.0,
        )?);

        let mut read = Tensor::to_vec_f32_batch(&tensors).await?;
        let logits = read.pop().expect("logits were pushed last");

        let mut trace = StepTrace {
            q_len,
            kv_len: cache.len,
            n_head: self.config.n_head,
            n_embd: self.config.n_embd,
            embedding: Vec::new(),
            attn: Vec::with_capacity(self.config.n_layer),
            detail: (0..detail_layers)
                .map(|layer| LayerDetail {
                    layer,
                    ln1_out: Vec::new(),
                    q: Vec::new(),
                    k: Vec::new(),
                    v: Vec::new(),
                    scores: Vec::new(),
                    attn_head_out: Vec::new(),
                    attn_proj_out: Vec::new(),
                    resid_attn: Vec::new(),
                    ln2_out: Vec::new(),
                    mlp_hidden: Vec::new(),
                    block_out: Vec::new(),
                })
                .collect(),
            ln_f_out: Vec::new(),
            top: top_probs(&logits, top_n),
        };
        for ((kind, dims), data) in kinds.iter().zip(&shapes).zip(read) {
            match *kind {
                ProbeKind::Embedding => trace.embedding = data,
                ProbeKind::Ln1Out { layer } => trace.detail[layer].ln1_out = data,
                ProbeKind::Query { layer } => trace.detail[layer].q = data,
                ProbeKind::Key { layer } => trace.detail[layer].k = data,
                ProbeKind::Value { layer } => trace.detail[layer].v = data,
                ProbeKind::Scores { layer } => trace.detail[layer].scores = data,
                ProbeKind::AttnHeadOut { layer } => trace.detail[layer].attn_head_out = data,
                ProbeKind::AttnProjOut { layer } => trace.detail[layer].attn_proj_out = data,
                ProbeKind::ResidAttn { layer } => trace.detail[layer].resid_attn = data,
                ProbeKind::Ln2Out { layer } => trace.detail[layer].ln2_out = data,
                ProbeKind::MlpHidden { layer } => trace.detail[layer].mlp_hidden = data,
                ProbeKind::BlockOut { layer } => trace.detail[layer].block_out = data,
                ProbeKind::LnFOut => trace.ln_f_out = data,
                ProbeKind::Attention { layer } => {
                    let [n_head, q_len, kv_len] = dims[..] else {
                        return Err(ForgeError::Shape(format!(
                            "attention probe expected rank 3, got {dims:?}"
                        )));
                    };
                    trace.attn.push(AttnStep {
                        layer,
                        n_head,
                        q_len,
                        kv_len,
                        probs: data,
                    });
                }
            }
        }
        Ok((logits, trace))
    }

    // ---- construction, parameters, serialization ----
    //
    // Motivated by training but not gated behind it: `init_random` builds a
    // model, `params`/`param_specs` name its tensors and `save_safetensors`
    // writes them. None records a tape, and `tests/streaming.rs` and
    // `tests/council.rs` — both pure inference — call `init_random`. Only
    // `loss` and `loss_grads` below are `#[cfg(feature = "train")]`.

    /// Random initialization for training from scratch: N(0, 0.02) weights,
    /// residual projections scaled by 1/sqrt(2*n_layer), zero biases,
    /// identity LayerNorms.
    pub fn init_random(config: Gpt2Config, device: &Device, seed: u64) -> Result<Gpt2> {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = config.n_embd;
        let std = 0.02f32;
        let resid_std = std / (2.0 * config.n_layer as f32).sqrt();
        let mut normal = |n: usize, std: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let u1: f32 = rng.random::<f32>().max(1e-7);
                    let u2: f32 = rng.random::<f32>();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
                })
                .collect()
        };
        let wte_host = normal(config.vocab_size * c, std);
        let wpe = Tensor::from_f32(&normal(config.n_ctx * c, std), [config.n_ctx, c], device)?;
        let emb = Embedding::from_host_wte(&wte_host, config.vocab_size, c, wpe, device)?;
        drop(wte_host);
        let eps = config.layer_norm_epsilon;
        let ones = |dev: &Device| Tensor::from_f32(&vec![1.0f32; c], [c], dev);
        let zeros1 = |dev: &Device, n: usize| Tensor::zeros([n], dev);
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: ones(device)?,
                    beta: zeros1(device, c)?,
                    eps,
                },
                attn_qkv: Linear {
                    w: Tensor::from_f32(&normal(c * 3 * c, std), [c, 3 * c], device)?,
                    b: Some(zeros1(device, 3 * c)?),
                },
                attn_proj: Linear {
                    w: Tensor::from_f32(&normal(c * c, resid_std), [c, c], device)?,
                    b: Some(zeros1(device, c)?),
                },
                ln_2: LayerNorm {
                    gamma: ones(device)?,
                    beta: zeros1(device, c)?,
                    eps,
                },
                mlp_fc: Linear {
                    w: Tensor::from_f32(&normal(c * 4 * c, std), [c, 4 * c], device)?,
                    b: Some(zeros1(device, 4 * c)?),
                },
                mlp_proj: Linear {
                    w: Tensor::from_f32(&normal(4 * c * c, resid_std), [4 * c, c], device)?,
                    b: Some(zeros1(device, c)?),
                },
            });
        }
        let ln_f = LayerNorm {
            gamma: ones(device)?,
            beta: zeros1(device, c)?,
            eps,
        };
        Ok(Gpt2 {
            emb,
            blocks,
            ln_f,
            config,
        })
    }

    /// Parameter names (safetensors keys) and weight-decay flags, in the
    /// canonical order used by `params`, `params_mut`, and `loss_grads`.
    pub fn param_specs(&self) -> Vec<(String, bool)> {
        let mut out = vec![
            ("wte.weight".to_string(), true),
            ("wpe.weight".to_string(), true),
        ];
        for i in 0..self.config.n_layer {
            out.push((format!("h.{i}.ln_1.weight"), false));
            out.push((format!("h.{i}.ln_1.bias"), false));
            out.push((format!("h.{i}.attn.c_attn.weight"), true));
            out.push((format!("h.{i}.attn.c_attn.bias"), false));
            out.push((format!("h.{i}.attn.c_proj.weight"), true));
            out.push((format!("h.{i}.attn.c_proj.bias"), false));
            out.push((format!("h.{i}.ln_2.weight"), false));
            out.push((format!("h.{i}.ln_2.bias"), false));
            out.push((format!("h.{i}.mlp.c_fc.weight"), true));
            out.push((format!("h.{i}.mlp.c_fc.bias"), false));
            out.push((format!("h.{i}.mlp.c_proj.weight"), true));
            out.push((format!("h.{i}.mlp.c_proj.bias"), false));
        }
        out.push(("ln_f.weight".to_string(), false));
        out.push(("ln_f.bias".to_string(), false));
        out
    }

    /// The single-chunk token embedding table, required for training.
    fn wte_single(&self) -> Result<&Tensor> {
        if self.emb.wte_chunks.len() != 1 {
            return Err(ForgeError::Shape(format!(
                "training requires a single-chunk wte; got {} chunks (device \
                 binding limit too small for vocab x n_embd)",
                self.emb.wte_chunks.len()
            )));
        }
        Ok(&self.emb.wte_chunks[0])
    }

    /// Parameters in `param_specs` order.
    pub fn params(&self) -> Result<Vec<&Tensor>> {
        let mut out = vec![self.wte_single()?, &self.emb.wpe];
        for b in &self.blocks {
            out.extend([&b.ln_1.gamma, &b.ln_1.beta, &b.attn_qkv.w]);
            out.push(bias(&b.attn_qkv)?);
            out.push(&b.attn_proj.w);
            out.push(bias(&b.attn_proj)?);
            out.extend([&b.ln_2.gamma, &b.ln_2.beta, &b.mlp_fc.w]);
            out.push(bias(&b.mlp_fc)?);
            out.push(&b.mlp_proj.w);
            out.push(bias(&b.mlp_proj)?);
        }
        out.extend([&self.ln_f.gamma, &self.ln_f.beta]);
        Ok(out)
    }

    /// Mutable parameters in `param_specs` order (for the optimizer).
    pub fn params_mut(&mut self) -> Result<Vec<&mut Tensor>> {
        if self.emb.wte_chunks.len() != 1 {
            return Err(ForgeError::Shape(
                "training requires a single-chunk wte".into(),
            ));
        }
        let mut out: Vec<&mut Tensor> = Vec::new();
        out.push(&mut self.emb.wte_chunks[0]);
        out.push(&mut self.emb.wpe);
        for b in &mut self.blocks {
            out.push(&mut b.ln_1.gamma);
            out.push(&mut b.ln_1.beta);
            out.push(&mut b.attn_qkv.w);
            out.push(
                b.attn_qkv
                    .b
                    .as_mut()
                    .ok_or_else(|| ForgeError::Shape("training requires biases".into()))?,
            );
            out.push(&mut b.attn_proj.w);
            out.push(
                b.attn_proj
                    .b
                    .as_mut()
                    .ok_or_else(|| ForgeError::Shape("training requires biases".into()))?,
            );
            out.push(&mut b.ln_2.gamma);
            out.push(&mut b.ln_2.beta);
            out.push(&mut b.mlp_fc.w);
            out.push(
                b.mlp_fc
                    .b
                    .as_mut()
                    .ok_or_else(|| ForgeError::Shape("training requires biases".into()))?,
            );
            out.push(&mut b.mlp_proj.w);
            out.push(
                b.mlp_proj
                    .b
                    .as_mut()
                    .ok_or_else(|| ForgeError::Shape("training requires biases".into()))?,
            );
        }
        out.push(&mut self.ln_f.gamma);
        out.push(&mut self.ln_f.beta);
        Ok(out)
    }

    /// Save a checkpoint loadable by `from_safetensors`. A chunked wte is
    /// reassembled host-side first.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialization::save_safetensors(path, &self.save_entries()?)
    }

    /// Save a checkpoint loadable by `from_fzm`, quantized to `.fzm` q4. A
    /// chunked wte is reassembled host-side first, same as `save_safetensors`.
    pub fn save_fzm(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialization::fzm::save_fzm_q4(path, &self.save_entries()?)
    }

    /// Save in whichever format `path`'s extension names — `.fzm` q4, else
    /// safetensors. The write-side counterpart of [`Gpt2::from_checkpoint`],
    /// so a caller carrying a checkpoint path around (the trainer's
    /// `--checkpoint`) cannot write one format under the other's name.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if path.extension().and_then(|s| s.to_str()) == Some("fzm") {
            self.save_fzm(path)
        } else {
            self.save_safetensors(path)
        }
    }

    /// `(name, shape, host f32)` for every parameter, in `param_specs` order.
    /// Both savers want exactly this; the only difference is the codec.
    fn save_entries(&self) -> Result<crate::serialization::fzm::TensorEntries> {
        let c = self.config.n_embd;
        let mut wte_host = Vec::with_capacity(self.config.vocab_size * c);
        for ch in &self.emb.wte_chunks {
            wte_host.extend(ch.to_vec_f32()?);
        }
        let mut entries = vec![(
            "wte.weight".to_string(),
            vec![self.config.vocab_size, c],
            wte_host,
        )];
        let specs = self.param_specs();
        let params = self.params_for_save()?;
        for ((name, _), t) in specs.iter().zip(params).skip(1) {
            entries.push((name.clone(), t.shape().dims().to_vec(), t.to_vec_f32()?));
        }
        Ok(entries)
    }

    /// Like `params` but tolerates a chunked wte (slot 0 is unused by the
    /// caller, which reassembles wte itself).
    fn params_for_save(&self) -> Result<Vec<&Tensor>> {
        let mut out = vec![&self.emb.wte_chunks[0], &self.emb.wpe];
        for b in &self.blocks {
            out.extend([&b.ln_1.gamma, &b.ln_1.beta, &b.attn_qkv.w]);
            out.push(bias(&b.attn_qkv)?);
            out.push(&b.attn_proj.w);
            out.push(bias(&b.attn_proj)?);
            out.extend([&b.ln_2.gamma, &b.ln_2.beta, &b.mlp_fc.w]);
            out.push(bias(&b.mlp_fc)?);
            out.push(&b.mlp_proj.w);
            out.push(bias(&b.mlp_proj)?);
        }
        out.extend([&self.ln_f.gamma, &self.ln_f.beta]);
        Ok(out)
    }

    #[cfg(feature = "train")]
    #[cfg_attr(docsrs, doc(cfg(feature = "train")))]
    /// Mean cross-entropy of `targets` given `input`, forward only — no tape,
    /// no gradients, no dropout. This is the evaluation path: a validation
    /// split scored with [`Gpt2::loss_grads`] would build and discard a full
    /// backward graph per window.
    pub fn loss(&self, input: &[u32], targets: &[u32]) -> Result<f32> {
        if input.is_empty() || input.len() != targets.len() {
            return Err(ForgeError::Shape(
                "loss needs equal, non-empty input/target lengths".into(),
            ));
        }
        let t = input.len();
        let device = self.emb.wpe.device();
        let tgt_t = Tensor::from_u32(targets, [t], &device)?;
        let logits = self.forward(input)?;
        let probs = ops::softmax(&logits, false, 0)?;
        let nll = ops::gather_nll(&probs, &tgt_t)?.to_vec_f32()?;
        Ok(nll.iter().sum::<f32>() / t as f32)
    }

    #[cfg(feature = "train")]
    #[cfg_attr(docsrs, doc(cfg(feature = "train")))]
    /// One training forward + backward on a single sequence: mean
    /// cross-entropy of `targets` given `input`, and gradients for every
    /// parameter in `param_specs` order. Batch by accumulating grads over calls.
    pub fn loss_grads(
        &self,
        input: &[u32],
        targets: &[u32],
        dropout_p: f32,
        seed: u32,
    ) -> Result<(f32, Vec<Tensor>)> {
        if input.is_empty() || input.len() != targets.len() {
            return Err(ForgeError::Shape(
                "loss_grads needs equal, non-empty input/target lengths".into(),
            ));
        }
        if input.len() > self.config.n_ctx {
            return Err(ForgeError::Shape(format!(
                "sequence length {} exceeds n_ctx {}",
                input.len(),
                self.config.n_ctx
            )));
        }
        let device = self.emb.wpe.device();
        let t = input.len();
        let ids_t = Tensor::from_u32(input, [t], &device)?;
        let tgt_t = Tensor::from_u32(targets, [t], &device)?;
        let eps = self.config.layer_norm_epsilon;
        let n_head = self.config.n_head;
        let hd = self.config.n_embd / n_head;

        let mut tape = Tape::new();
        // Leaves in param_specs order (ids 0..n_params).
        let pvars: Vec<TVar> = self
            .params()?
            .into_iter()
            .map(|p| tape.leaf(p.clone()))
            .collect();
        let n_params = pvars.len();

        let mut site = 0u32;
        let mut dseed = move || {
            site = site.wrapping_add(1);
            seed.wrapping_mul(0x9E37_79B1)
                .wrapping_add(site.wrapping_mul(0x85EB_CA77))
        };

        let mut x = tape.embedding(&ids_t, &pvars[0], &pvars[1], 0)?;
        x = tape.dropout(&x, dropout_p, dseed())?;
        for i in 0..self.config.n_layer {
            let base = 2 + i * 12;
            let [g1, b1, wqkv, bqkv, wproj, bproj, g2, b2, wfc, bfc, wmp, bmp] =
                std::array::from_fn(|j| &pvars[base + j]);
            let a = tape.layernorm(&x, g1, b1, eps)?;
            let qkv = tape.matmul(&a, wqkv, Some(bqkv), MatmulSpec::default())?;
            let (q, k, v) = tape.split_heads(&qkv, n_head)?;
            let att = tape.matmul(
                &q,
                &k,
                None,
                MatmulSpec {
                    trans_b: true,
                    alpha: 1.0 / (hd as f32).sqrt(),
                    ..Default::default()
                },
            )?;
            let probs = tape.softmax(&att, true, 0)?;
            let probs = tape.dropout(&probs, dropout_p, dseed())?;
            let y = tape.matmul(&probs, &v, None, MatmulSpec::default())?;
            let y = tape.merge_heads(&y)?;
            let y = tape.matmul(&y, wproj, Some(bproj), MatmulSpec::default())?;
            let y = tape.dropout(&y, dropout_p, dseed())?;
            x = tape.add(&x, &y)?;
            let a2 = tape.layernorm(&x, g2, b2, eps)?;
            let f = tape.matmul(&a2, wfc, Some(bfc), MatmulSpec::default())?;
            let f = tape.gelu(&f)?;
            let f = tape.matmul(&f, wmp, Some(bmp), MatmulSpec::default())?;
            let f = tape.dropout(&f, dropout_p, dseed())?;
            x = tape.add(&x, &f)?;
        }
        let xf = tape.layernorm(&x, &pvars[n_params - 2], &pvars[n_params - 1], eps)?;
        // Weight-tied LM head: wte gets gradient from here *and* from the
        // embedding scatter.
        let logits = tape.matmul(
            &xf,
            &pvars[0],
            None,
            MatmulSpec {
                trans_b: true,
                ..Default::default()
            },
        )?;

        let probs = ops::softmax(&logits.t, false, 0)?;
        let nll = ops::gather_nll(&probs, &tgt_t)?.to_vec_f32()?;
        let loss = nll.iter().sum::<f32>() / t as f32;
        let dlogits = ops::ce_bwd(&probs, &tgt_t, 1.0 / t as f32)?;
        let all = tape.backward(&logits, dlogits)?;
        let mut grads = Vec::with_capacity(n_params);
        for (i, g) in all.into_iter().take(n_params).enumerate() {
            grads.push(
                g.ok_or_else(|| ForgeError::Shape(format!("parameter {i} received no gradient")))?,
            );
        }
        Ok((loss, grads))
    }

    /// Autoregressive generation with KV-cache decode.
    /// Returns the full text (prompt + completion).
    pub fn generate(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
    ) -> Result<String> {
        self.generate_streaming(tokenizer, prompt, max_new_tokens, sampling, |_, _| {})
    }

    /// Sync streaming generation with KV-cache decode. `on_token` fires
    /// exactly once per *generated* token (never for the prompt) with the
    /// token id and the newly decodable text delta — which may be empty when
    /// a multi-byte character spans several BPE tokens, since a partial UTF-8
    /// sequence is held back until its trailing bytes arrive.
    ///
    /// This is the accurate throughput hook: callback invocations map 1:1 to
    /// tokens, unlike [`Gpt2::generate_async`]'s text-delta callback.
    pub fn generate_streaming(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        mut on_token: impl FnMut(u32, &str),
    ) -> Result<String> {
        self.generate_streaming_ctl(tokenizer, prompt, max_new_tokens, sampling, |id, text| {
            on_token(id, text);
            ControlFlow::Continue(())
        })
    }

    /// [`Gpt2::generate_streaming`] with an interruptible callback: returning
    /// [`ControlFlow::Break`] stops after the current token and returns the
    /// text generated so far. Long interactive runs need this — a callback
    /// that cannot abort can only be "cancelled" once the whole run finishes.
    pub fn generate_streaming_ctl(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        mut on_token: impl FnMut(u32, &str) -> ControlFlow<()>,
    ) -> Result<String> {
        let mut ids = tokenizer.encode(prompt)?;
        if ids.is_empty() {
            return Err(ForgeError::Tokenizer("prompt produced no tokens".into()));
        }
        let mut rng = match sampling {
            Sampling::TopK { seed, .. } => Some(StdRng::seed_from_u64(seed)),
            Sampling::Greedy => None,
        };
        let mut cache = self.new_cache()?;
        let mut logits = self.logits_step(&ids, &mut cache)?; // prompt prefill
        // Same append-only byte stream as `generate_async`, but the delta is
        // captured per token instead of being forwarded as it is produced.
        let mut bytes = tokenizer.decode_bytes(&ids);
        let mut sent = emit_valid_prefix(&bytes, 0, &mut |_: &str| {});
        for _ in 0..max_new_tokens {
            let next = sample(&logits, sampling, rng.as_mut());
            if Some(next) == self.config.eos_token_id {
                break;
            }
            ids.push(next);
            bytes.extend(tokenizer.decode_bytes(&[next]));
            let mut delta = String::new();
            sent = emit_valid_prefix(&bytes, sent, &mut |s: &str| delta.push_str(s));
            if on_token(next, &delta).is_break() {
                break;
            }
            if ids.len() >= self.config.n_ctx {
                break;
            }
            logits = self.logits_step(&[next], &mut cache)?; // single-token decode
        }
        Ok(tokenizer.decode(&ids))
    }

    /// Async autoregressive generation with KV-cache decode — the browser
    /// form of [`Gpt2::generate`]. `on_text` receives each newly decoded text
    /// fragment (the delta of the full decode, so multi-byte characters that
    /// span BPE tokens are never split) so a page can stream output.
    pub async fn generate_async(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        mut on_text: impl FnMut(&str),
    ) -> Result<String> {
        self.generate_async_ctl(tokenizer, prompt, max_new_tokens, sampling, |s| {
            on_text(s);
            ControlFlow::Continue(())
        })
        .await
    }

    /// [`Gpt2::generate_async`] with an interruptible callback — the async
    /// counterpart of [`Gpt2::generate_streaming_ctl`]. Returning
    /// [`ControlFlow::Break`] stops after the current token, which is how a
    /// page offers a working "stop" button.
    pub async fn generate_async_ctl(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<String> {
        // `None` turns off the probe entirely: no capture, no readback, and
        // the same `logits_step_async` this method has always called.
        self.generate_async_probe(
            tokenizer,
            prompt,
            max_new_tokens,
            sampling,
            on_text,
            None::<fn(&[AttnStep])>,
        )
        .await
    }

    /// [`Gpt2::generate_async_ctl`] with an optional attention probe:
    /// `on_attn`, when present, fires once per decode step with every block's
    /// attention probabilities — including the prompt prefill, whose `q_len`
    /// is the prompt length rather than 1.
    ///
    /// Opt-in because it costs one readback per block per token. Passing
    /// `None` is exactly the non-probing path; the generated text is identical
    /// either way, since the probe reads tensors the step already computed.
    pub async fn generate_async_probe(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
        on_attn: Option<impl FnMut(&[AttnStep])>,
    ) -> Result<String> {
        // No detail layers and no top-n: the attention-only probe, which is
        // what this signature has always promised.
        self.generate_async_trace(
            tokenizer,
            prompt,
            max_new_tokens,
            sampling,
            on_text,
            on_attn.map(|mut f| move |t: &StepTrace| f(&t.attn)),
            0,
            0,
        )
        .await
    }

    /// [`Gpt2::generate_async_probe`] with the full calculation trace:
    /// `on_trace`, when present, fires once per step — including the prompt
    /// prefill, whose `q_len` is the prompt length rather than 1 — with
    /// everything [`Gpt2::logits_step_trace_async`] captured.
    ///
    /// `None` is byte-for-byte the non-probing path. With `Some`, the
    /// generated text is still identical: the probe reads tensors the step
    /// already computed and changes no arithmetic.
    #[allow(clippy::too_many_arguments)] // sampling, two sinks and the two
    // cost knobs; a builder here would be ceremony over a call made twice.
    pub async fn generate_async_trace(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        mut on_text: impl FnMut(&str) -> ControlFlow<()>,
        mut on_trace: Option<impl FnMut(&StepTrace)>,
        detail_layers: usize,
        top_n: usize,
    ) -> Result<String> {
        // The streaming helper takes a plain sink, so the break request is
        // captured here and checked once the delta has been forwarded. A Cell
        // keeps the flag readable while the closure holds it.
        let stop = std::cell::Cell::new(false);
        let mut on_text = |s: &str| {
            if on_text(s).is_break() {
                stop.set(true);
            }
        };
        let mut ids = tokenizer.encode(prompt)?;
        if ids.is_empty() {
            return Err(ForgeError::Tokenizer("prompt produced no tokens".into()));
        }
        let mut rng = match sampling {
            Sampling::TopK { seed, .. } => Some(StdRng::seed_from_u64(seed)),
            Sampling::Greedy => None,
        };
        let mut cache = self.new_cache()?;
        let mut logits = self
            .step_async(&ids, &mut cache, &mut on_trace, detail_layers, top_n)
            .await?; // prompt prefill
        // Stream over the raw byte-level decode (append-only per token),
        // emitting only its valid-UTF-8 prefix: a multi-byte character split
        // across BPE tokens is held back until its trailing bytes arrive.
        let mut bytes = tokenizer.decode_bytes(&ids);
        let mut sent = emit_valid_prefix(&bytes, 0, &mut on_text);
        for _ in 0..max_new_tokens {
            let next = sample(&logits, sampling, rng.as_mut());
            if Some(next) == self.config.eos_token_id {
                break;
            }
            ids.push(next);
            bytes.extend(tokenizer.decode_bytes(&[next]));
            sent = emit_valid_prefix(&bytes, sent, &mut on_text);
            if stop.get() {
                break;
            }
            if ids.len() >= self.config.n_ctx {
                break;
            }
            logits = self
                .step_async(&[next], &mut cache, &mut on_trace, detail_layers, top_n)
                .await?; // single-token decode
        }
        Ok(tokenizer.decode(&ids))
    }

    /// One decode step, forwarding the trace to `on_trace` when probing.
    async fn step_async(
        &self,
        ids: &[u32],
        cache: &mut KvCache,
        on_trace: &mut Option<impl FnMut(&StepTrace)>,
        detail_layers: usize,
        top_n: usize,
    ) -> Result<Vec<f32>> {
        match on_trace {
            Some(f) => {
                let (logits, trace) = self
                    .logits_step_trace_async(ids, cache, detail_layers, top_n)
                    .await?;
                f(&trace);
                Ok(logits)
            }
            None => self.logits_step_async(ids, cache).await,
        }
    }
}

/// Send `bytes[sent..]` to `on_text` up to the longest decodable prefix;
/// returns the new high-water mark. Invalid sequences become U+FFFD (matching
/// `from_utf8_lossy`); an *incomplete* trailing character is held back until
/// its remaining bytes arrive.
fn emit_valid_prefix(bytes: &[u8], mut sent: usize, on_text: &mut impl FnMut(&str)) -> usize {
    loop {
        match std::str::from_utf8(&bytes[sent..]) {
            Ok(s) => {
                if !s.is_empty() {
                    on_text(s);
                }
                return bytes.len();
            }
            Err(e) => {
                let valid = e.valid_up_to();
                if valid > 0 {
                    on_text(std::str::from_utf8(&bytes[sent..sent + valid]).unwrap());
                    sent += valid;
                }
                match e.error_len() {
                    // Invalid sequence: emit a replacement char and skip it.
                    Some(bad) => {
                        on_text("\u{FFFD}");
                        sent += bad;
                    }
                    // Incomplete tail: wait for the next token's bytes.
                    None => return sent,
                }
            }
        }
    }
}

/// The `n` most likely next tokens and their probabilities, most likely first.
///
/// A full softmax over the row, so a bar labelled 0.21 means 21% of the model's
/// probability mass — not 21% of whatever survived a top-k truncation. This is
/// the model's own distribution, before any sampling temperature: the sampler
/// picks from it, but the picture is of the model.
///
/// No GPU work: the logits were already read back for sampling.
pub fn top_probs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let n = n.min(logits.len());
    if n == 0 {
        return Vec::new();
    }
    let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    // Partition first: `n` is ~24 and the vocabulary is up to 50257, so a full
    // sort per token would be the most expensive thing in the trace.
    idx.select_nth_unstable_by(n - 1, by_logit);
    idx.truncate(n);
    idx.sort_unstable_by(by_logit);

    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let total: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    idx.into_iter()
        .map(|i| (i, (logits[i as usize] - max).exp() / total))
        .collect()
}

fn bias(l: &Linear) -> Result<&Tensor> {
    l.b.as_ref()
        .ok_or_else(|| ForgeError::Shape("training requires biases".into()))
}

pub(crate) fn sample(logits: &[f32], sampling: Sampling, rng: Option<&mut StdRng>) -> u32 {
    match sampling {
        Sampling::Greedy => argmax(logits),
        Sampling::TopK { k, temperature, .. } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
            indexed.truncate(k.max(1));
            let t = temperature.max(1e-4);
            let max = indexed[0].1;
            let weights: Vec<f32> = indexed.iter().map(|(_, l)| ((l - max) / t).exp()).collect();
            let total: f32 = weights.iter().sum();
            let mut r = rng.expect("rng required for TopK").random::<f32>() * total;
            for ((idx, _), w) in indexed.iter().zip(&weights) {
                if r <= *w {
                    return *idx as u32;
                }
                r -= w;
            }
            indexed[0].0 as u32
        }
    }
}

/// Draws tokens from a row of logits, carrying the sampling stream with it.
///
/// [`Gpt2::generate`] and friends sample internally; this is the same drawing
/// logic for callers that own the loop themselves, and it owns its RNG so no
/// caller has to name `rand`'s types.
///
/// ```no_run
/// # use forge::{Sampler, Sampling};
/// let mut s = Sampler::new(1337);
/// let logits = vec![0.1f32, 2.0, 0.3];
/// let id = s.pick(&logits, Sampling::TopK { k: 2, temperature: 0.8, seed: 0 });
/// ```
///
/// `Sampling::TopK`'s own `seed` field is ignored here — the stream is this
/// sampler's, seeded once at [`Sampler::new`], so successive picks continue it
/// rather than restarting it.
pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// One token id from `logits`. Deterministic under [`Sampling::Greedy`];
    /// otherwise it advances this sampler's stream.
    pub fn pick(&mut self, logits: &[f32], sampling: Sampling) -> u32 {
        sample(logits, sampling, Some(&mut self.rng))
    }

    /// Restart the stream. Same contract as constructing a fresh sampler.
    pub fn reseed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}
