use std::ops::ControlFlow;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use super::{Weights, argmax};
#[cfg(feature = "train")]
use crate::autograd::{TVar, Tape};
use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::nn::{Embedding, LayerNorm, Linear};
use crate::ops::{self, MatmulSpec};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

pub const ARCHITECTURE: &str = "gpt2";

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

#[derive(Debug, Clone)]
pub struct AttnStep {
    pub layer: usize,
    pub n_head: usize,
    pub q_len: usize,
    pub kv_len: usize,
    pub probs: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerDetail {
    pub layer: usize,
    pub ln1_out: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub scores: Vec<f32>,
    pub attn_head_out: Vec<f32>,
    pub attn_proj_out: Vec<f32>,
    pub resid_attn: Vec<f32>,
    pub ln2_out: Vec<f32>,
    pub mlp_hidden: Vec<f32>,
    pub block_out: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct StepTrace {
    pub q_len: usize,
    pub kv_len: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub embedding: Vec<f32>,
    pub attn: Vec<AttnStep>,
    pub detail: Vec<LayerDetail>,
    pub ln_f_out: Vec<f32>,
    pub top: Vec<(u32, f32)>,
}

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
    pub fn from_checkpoint(
        path: impl AsRef<Path>,
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_checkpoint_bytes(&bytes, config, device)
    }

    pub fn from_checkpoint_bytes(
        bytes: &[u8],
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        let (mut w, _) = Weights::parse(bytes, ARCHITECTURE)?;
        let eps = config.layer_norm_epsilon;
        let wpe = w.param("wpe.weight", None, device)?;
        let emb = w.embedding("wte.weight", config.vocab_size, config.n_embd, wpe, device)?;
        let ln = |w: &mut Weights<'_>, name: &str| -> Result<LayerNorm> {
            Ok(LayerNorm {
                gamma: w.param(&format!("{name}.weight"), None, device)?,
                beta: w.param(&format!("{name}.bias"), None, device)?,
                eps,
            })
        };
        let linear = |w: &mut Weights<'_>, name: &str| -> Result<Linear> {
            Ok(Linear {
                w: w.weight(&format!("{name}.weight"), None, device)?,
                b: Some(w.param(&format!("{name}.bias"), None, device)?),
            })
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block {
                ln_1: ln(&mut w, &format!("h.{i}.ln_1"))?,
                attn_qkv: linear(&mut w, &format!("h.{i}.attn.c_attn"))?,
                attn_proj: linear(&mut w, &format!("h.{i}.attn.c_proj"))?,
                ln_2: ln(&mut w, &format!("h.{i}.ln_2"))?,
                mlp_fc: linear(&mut w, &format!("h.{i}.mlp.c_fc"))?,
                mlp_proj: linear(&mut w, &format!("h.{i}.mlp.c_proj"))?,
            });
        }
        let ln_f = ln(&mut w, "ln_f")?;
        Ok(Gpt2 {
            emb,
            blocks,
            ln_f,
            config,
        })
    }

    pub fn from_safetensors(
        path: impl AsRef<Path>,
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        Self::from_checkpoint(path, config, device)
    }

    pub fn from_safetensors_bytes(
        bytes: &[u8],
        config: Gpt2Config,
        device: &Device,
    ) -> Result<Self> {
        Self::from_checkpoint_bytes(bytes, config, device)
    }

    pub fn from_fzm(path: impl AsRef<Path>, config: Gpt2Config, device: &Device) -> Result<Self> {
        Self::from_checkpoint(path, config, device)
    }

    pub fn from_fzm_bytes(bytes: &[u8], config: Gpt2Config, device: &Device) -> Result<Self> {
        Self::from_checkpoint_bytes(bytes, config, device)
    }

    fn attention(&self, block: &Block, h: &Tensor) -> Result<Tensor> {
        let n_head = self.config.n_head;
        let hd = self.config.n_embd / n_head;
        let qkv = block.attn_qkv.forward(h)?;
        let (q, k, v) = ops::split_heads(&qkv, n_head)?;
        let att = ops::matmul(
            &q,
            &k,
            None,
            MatmulSpec {
                trans_b: true,
                alpha: 1.0 / (hd as f32).sqrt(),
                ..Default::default()
            },
        )?;
        let att = ops::softmax(&att, true, 0)?;
        let y = ops::matmul(&att, &v, None, MatmulSpec::default())?;
        let y = ops::merge_heads(&y)?;
        block.attn_proj.forward(&y)
    }

    #[allow(clippy::too_many_arguments)]
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
        let qkv = block.attn_qkv.forward(h)?;
        let (q, k, v) = ops::split_heads(&qkv, n_head)?;
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
        )?;
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
            p.push(ProbeKind::Scores { layer }, &scores);
        }
        let att = ops::softmax(&scores, true, len)?;
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
        )?;
        let y = ops::merge_heads(&y)?;
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail(layer)) {
            p.push(ProbeKind::AttnHeadOut { layer }, &y);
        }
        let proj = block.attn_proj.forward(&y)?;
        if let Some(p) = probe.filter(|p| p.detail(layer)) {
            p.push(ProbeKind::AttnProjOut { layer }, &proj);
        }
        Ok(proj)
    }

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
        let _scope = device.dispatch_scope();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let mut x = self.emb.forward(&ids_t, pos)?;
        if let Some(p) = probe.as_deref_mut().filter(|p| p.detail_layers > 0) {
            p.push(ProbeKind::Embedding, &x);
        }
        for (layer, (block, (kc, vc))) in self
            .blocks
            .iter()
            .zip(cache.k.iter_mut().zip(cache.v.iter_mut()))
            .enumerate()
        {
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

    pub fn forward(&self, ids: &[u32]) -> Result<Tensor> {
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden(ids)?;
        ops::matmul_chunked_transb(&h, &self.emb.wte_chunks, 1.0)
    }

    pub fn logits_last(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden(ids)?;
        let last = h.narrow_rows(ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)?.to_vec_f32()
    }

    pub fn logits_step(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden_cached(ids, cache, None)?;
        let last = h.narrow_rows(ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)?.to_vec_f32()
    }

    pub async fn logits_step_async(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let _scope = self.emb.wpe.device().dispatch_scope();
        let h = self.hidden_cached(ids, cache, None)?;
        let last = h.narrow_rows(ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)?
            .to_vec_f32_async()
            .await
    }

    pub fn hidden_step(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let h = self.hidden_cached(ids, cache, None)?;
        h.narrow_rows(ids.len() - 1, 1)?.to_vec_f32()
    }

    pub async fn hidden_step_async(&self, ids: &[u32], cache: &mut KvCache) -> Result<Vec<f32>> {
        let h = self.hidden_cached(ids, cache, None)?;
        h.narrow_rows(ids.len() - 1, 1)?.to_vec_f32_async().await
    }

    pub fn logits_from_hidden(&self, h: &[f32]) -> Result<Vec<f32>> {
        ops::matmul_chunked_transb(&self.hidden_tensor(h)?, &self.emb.wte_chunks, 1.0)?.to_vec_f32()
    }

    pub async fn logits_from_hidden_async(&self, h: &[f32]) -> Result<Vec<f32>> {
        ops::matmul_chunked_transb(&self.hidden_tensor(h)?, &self.emb.wte_chunks, 1.0)?
            .to_vec_f32_async()
            .await
    }

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

    pub async fn logits_step_attn_async(
        &self,
        ids: &[u32],
        cache: &mut KvCache,
    ) -> Result<(Vec<f32>, Vec<AttnStep>)> {
        let (logits, trace) = self.logits_step_trace_async(ids, cache, 0, 0).await?;
        Ok((logits, trace.attn))
    }

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

    pub fn init_random(config: Gpt2Config, device: &Device, seed: u64) -> Result<Gpt2> {
        let mut rng = StdRng::seed_from_u64(seed);
        let c = config.n_embd;
        let std = 0.02f32;
        let resid_std = std / (2.0 * config.n_layer as f32).sqrt();
        let mut normal = |n: usize, std: f32| super::normal(&mut rng, n, std);
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

    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialization::save_safetensors(path, &self.save_entries()?)
    }

    pub fn save_fzm(&self, path: impl AsRef<Path>) -> Result<()> {
        use crate::serialization::fzm::{FzmDtype, FzmManifest, save_fzm_v2};
        let c = &self.config;
        let config = serde_json::json!({
            "n_layer": c.n_layer,
            "n_head": c.n_head,
            "n_embd": c.n_embd,
            "n_ctx": c.n_ctx,
            "vocab_size": c.vocab_size,
            "layer_norm_epsilon": c.layer_norm_epsilon,
            "eos_token_id": c.eos_token_id,
        });
        let manifest = FzmManifest::new(ARCHITECTURE, config);
        save_fzm_v2(path, &manifest, &self.save_entries()?, FzmDtype::Q4, &[])
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if path.extension().and_then(|s| s.to_str()) == Some("fzm") {
            self.save_fzm(path)
        } else {
            self.save_safetensors(path)
        }
    }

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
    pub fn loss_grads(
        &self,
        input: &[u32],
        targets: &[u32],
        dropout_p: f32,
        seed: u32,
    ) -> Result<(f32, Vec<Tensor>)> {
        self.loss_grads_batched(input, targets, input.len(), dropout_p, seed)
    }

    #[cfg(feature = "train")]
    #[cfg_attr(docsrs, doc(cfg(feature = "train")))]
    pub fn loss_grads_batched(
        &self,
        input: &[u32],
        targets: &[u32],
        seq_len: usize,
        dropout_p: f32,
        seed: u32,
    ) -> Result<(f32, Vec<Tensor>)> {
        if input.is_empty() || input.len() != targets.len() {
            return Err(ForgeError::Shape(
                "loss_grads needs equal, non-empty input/target lengths".into(),
            ));
        }
        if seq_len == 0 || !input.len().is_multiple_of(seq_len) {
            return Err(ForgeError::Shape(format!(
                "loss_grads: {} tokens is not a whole number of {seq_len}-token sequences",
                input.len()
            )));
        }
        if seq_len > self.config.n_ctx {
            return Err(ForgeError::Shape(format!(
                "sequence length {seq_len} exceeds n_ctx {}",
                self.config.n_ctx
            )));
        }
        let batch = input.len() / seq_len;
        let device = self.emb.wpe.device();
        let _scope = device.dispatch_scope();
        let t = input.len();
        let ids_t = Tensor::from_u32(input, [t], &device)?;
        let tgt_t = Tensor::from_u32(targets, [t], &device)?;
        let eps = self.config.layer_norm_epsilon;
        let n_head = self.config.n_head;
        let hd = self.config.n_embd / n_head;

        let mut tape = Tape::new();
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

        let mut x = tape.embedding_batched(&ids_t, &pvars[0], &pvars[1], 0, batch)?;
        x = tape.dropout(&x, dropout_p, dseed())?;
        for i in 0..self.config.n_layer {
            let base = 2 + i * 12;
            let [g1, b1, wqkv, bqkv, wproj, bproj, g2, b2, wfc, bfc, wmp, bmp] =
                std::array::from_fn(|j| &pvars[base + j]);
            let a = tape.layernorm(&x, g1, b1, eps)?;
            let qkv = tape.matmul(&a, wqkv, Some(bqkv), MatmulSpec::default())?;
            let (q, k, v) = tape.split_heads_batched(&qkv, n_head, batch)?;
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
            let y = tape.merge_heads_batched(&y, batch)?;
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

    pub fn generate(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
    ) -> Result<String> {
        self.generate_streaming(tokenizer, prompt, max_new_tokens, sampling, |_, _| {})
    }

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
        let mut logits = self.logits_step(&ids, &mut cache)?;
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
            logits = self.logits_step(&[next], &mut cache)?;
        }
        Ok(tokenizer.decode(&ids))
    }

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

    pub async fn generate_async_ctl(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
    ) -> Result<String> {
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

    pub async fn generate_async_probe(
        &self,
        tokenizer: &impl Tokenizer,
        prompt: &str,
        max_new_tokens: usize,
        sampling: Sampling,
        on_text: impl FnMut(&str) -> ControlFlow<()>,
        on_attn: Option<impl FnMut(&[AttnStep])>,
    ) -> Result<String> {
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

    #[allow(clippy::too_many_arguments)]
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
            .await?;
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
                .await?;
        }
        Ok(tokenizer.decode(&ids))
    }

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
                    Some(bad) => {
                        on_text("\u{FFFD}");
                        sent += bad;
                    }
                    None => return sent,
                }
            }
        }
    }
}

pub fn top_probs(logits: &[f32], n: usize) -> Vec<(u32, f32)> {
    let n = n.min(logits.len());
    if n == 0 {
        return Vec::new();
    }
    let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
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

pub struct Sampler {
    rng: StdRng,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Sampler {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn pick(&mut self, logits: &[f32], sampling: Sampling) -> u32 {
        sample(logits, sampling, Some(&mut self.rng))
    }

    pub fn reseed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}
