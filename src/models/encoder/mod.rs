use std::collections::HashMap;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::nn::{LayerNorm, Linear};
use crate::ops::{self, MatmulSpec};
use crate::tensor::Tensor;

#[cfg(feature = "train")]
use crate::autograd::{TVar, Tape};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EncoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub embed_rank: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub max_seq: usize,
    pub embed_dim: usize,
    pub layer_norm_epsilon: f32,
    pub embed_init_std: f32,
}

impl EncoderConfig {
    pub fn ramanujan_10m() -> Self {
        EncoderConfig {
            vocab_size: 16384,
            d_model: 320,
            embed_rank: 160,
            n_layer: 6,
            n_head: 5,
            d_ff: 1280,
            max_seq: 256,
            embed_dim: 256,
            layer_norm_epsilon: 1e-5,
            embed_init_std: 0.05,
        }
    }

    pub fn n_params(&self) -> usize {
        let (v, d, r, f) = (self.vocab_size, self.d_model, self.embed_rank, self.d_ff);
        let per_layer = 2 * d + d * 3 * d + 3 * d + d * d + d + 2 * d + d * f + f + f * d + d;
        v * r
            + r * d
            + d
            + 2 * d
            + self.n_layer * per_layer
            + 2 * d
            + d * self.embed_dim
            + self.embed_dim
            + v
    }

    fn validate(&self) -> Result<()> {
        if !self.d_model.is_multiple_of(self.n_head) {
            return Err(ForgeError::Shape(format!(
                "d_model {} not divisible by n_head {}",
                self.d_model, self.n_head
            )));
        }
        if self.embed_dim > self.d_model {
            return Err(ForgeError::Shape(format!(
                "embed_dim {} exceeds d_model {}",
                self.embed_dim, self.d_model
            )));
        }
        Ok(())
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

pub struct Encoder {
    wte: Tensor,
    wte_proj: Linear,
    emb_ln: LayerNorm,
    blocks: Vec<Block>,
    ln_f: LayerNorm,
    proj: Linear,
    mlm_bias: Tensor,
    pos: Tensor,
    config: EncoderConfig,
}

fn sinusoidal(max_seq: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; max_seq * d];
    for (p, row) in out.chunks_mut(d).enumerate() {
        for i in 0..d / 2 {
            let freq = (p as f32) / 10000f32.powf(2.0 * i as f32 / d as f32);
            row[2 * i] = freq.sin();
            row[2 * i + 1] = freq.cos();
        }
        if d % 2 == 1 {
            row[d - 1] = 0.0;
        }
    }
    out
}

fn position_ids(batch: usize, seq: usize) -> Vec<u32> {
    (0..batch * seq).map(|i| (i % seq) as u32).collect()
}

impl Encoder {
    pub fn config(&self) -> &EncoderConfig {
        &self.config
    }

    pub fn init_random(config: EncoderConfig, device: &Device, seed: u64) -> Result<Encoder> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(seed);
        let (d, r, f) = (config.d_model, config.embed_rank, config.d_ff);
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
        let eps = config.layer_norm_epsilon;
        let ones = |n: usize| Tensor::from_f32(&vec![1.0f32; n], [n], device);
        let zeros = |n: usize| Tensor::zeros([n], device);

        let emb_std = config.embed_init_std;
        let wte = Tensor::from_f32(
            &normal(config.vocab_size * r, emb_std),
            [config.vocab_size, r],
            device,
        )?;
        let wte_proj = Linear {
            w: Tensor::from_f32(&normal(r * d, emb_std), [r, d], device)?,
            b: Some(zeros(d)?),
        };
        let emb_ln = LayerNorm {
            gamma: ones(d)?,
            beta: zeros(d)?,
            eps,
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: ones(d)?,
                    beta: zeros(d)?,
                    eps,
                },
                attn_qkv: Linear {
                    w: Tensor::from_f32(&normal(d * 3 * d, std), [d, 3 * d], device)?,
                    b: Some(zeros(3 * d)?),
                },
                attn_proj: Linear {
                    w: Tensor::from_f32(&normal(d * d, resid_std), [d, d], device)?,
                    b: Some(zeros(d)?),
                },
                ln_2: LayerNorm {
                    gamma: ones(d)?,
                    beta: zeros(d)?,
                    eps,
                },
                mlp_fc: Linear {
                    w: Tensor::from_f32(&normal(d * f, std), [d, f], device)?,
                    b: Some(zeros(f)?),
                },
                mlp_proj: Linear {
                    w: Tensor::from_f32(&normal(f * d, resid_std), [f, d], device)?,
                    b: Some(zeros(d)?),
                },
            });
        }
        let ln_f = LayerNorm {
            gamma: ones(d)?,
            beta: zeros(d)?,
            eps,
        };
        let proj = Linear {
            w: Tensor::from_f32(
                &normal(d * config.embed_dim, std),
                [d, config.embed_dim],
                device,
            )?,
            b: Some(zeros(config.embed_dim)?),
        };
        let mlm_bias = zeros(config.vocab_size)?;
        let pos = Tensor::from_f32(&sinusoidal(config.max_seq, d), [config.max_seq, d], device)?;
        Ok(Encoder {
            wte,
            wte_proj,
            emb_ln,
            blocks,
            ln_f,
            proj,
            mlm_bias,
            pos,
            config,
        })
    }

    pub fn param_specs(&self) -> Vec<(String, bool)> {
        let mut out = vec![
            ("wte.weight".to_string(), true),
            ("wte_proj.weight".to_string(), true),
            ("wte_proj.bias".to_string(), false),
            ("emb_ln.weight".to_string(), false),
            ("emb_ln.bias".to_string(), false),
        ];
        for i in 0..self.config.n_layer {
            out.push((format!("h.{i}.ln_1.weight"), false));
            out.push((format!("h.{i}.ln_1.bias"), false));
            out.push((format!("h.{i}.attn.qkv.weight"), true));
            out.push((format!("h.{i}.attn.qkv.bias"), false));
            out.push((format!("h.{i}.attn.proj.weight"), true));
            out.push((format!("h.{i}.attn.proj.bias"), false));
            out.push((format!("h.{i}.ln_2.weight"), false));
            out.push((format!("h.{i}.ln_2.bias"), false));
            out.push((format!("h.{i}.mlp.fc.weight"), true));
            out.push((format!("h.{i}.mlp.fc.bias"), false));
            out.push((format!("h.{i}.mlp.proj.weight"), true));
            out.push((format!("h.{i}.mlp.proj.bias"), false));
        }
        out.push(("ln_f.weight".to_string(), false));
        out.push(("ln_f.bias".to_string(), false));
        out.push(("proj.weight".to_string(), true));
        out.push(("proj.bias".to_string(), false));
        out.push(("mlm_head.bias".to_string(), false));
        out
    }

    pub const BLOCK_BASE: usize = 5;
    pub const BLOCK_PARAMS: usize = 12;

    fn bias(l: &Linear) -> Result<&Tensor> {
        l.b.as_ref()
            .ok_or_else(|| ForgeError::Shape("encoder requires biases".into()))
    }

    pub fn params(&self) -> Result<Vec<&Tensor>> {
        let mut out = vec![
            &self.wte,
            &self.wte_proj.w,
            Self::bias(&self.wte_proj)?,
            &self.emb_ln.gamma,
            &self.emb_ln.beta,
        ];
        for b in &self.blocks {
            out.extend([&b.ln_1.gamma, &b.ln_1.beta, &b.attn_qkv.w]);
            out.push(Self::bias(&b.attn_qkv)?);
            out.push(&b.attn_proj.w);
            out.push(Self::bias(&b.attn_proj)?);
            out.extend([&b.ln_2.gamma, &b.ln_2.beta, &b.mlp_fc.w]);
            out.push(Self::bias(&b.mlp_fc)?);
            out.push(&b.mlp_proj.w);
            out.push(Self::bias(&b.mlp_proj)?);
        }
        out.extend([&self.ln_f.gamma, &self.ln_f.beta, &self.proj.w]);
        out.push(Self::bias(&self.proj)?);
        out.push(&self.mlm_bias);
        Ok(out)
    }

    pub fn params_mut(&mut self) -> Result<Vec<&mut Tensor>> {
        let need = || ForgeError::Shape("encoder requires biases".into());
        let mut out: Vec<&mut Tensor> = vec![
            &mut self.wte,
            &mut self.wte_proj.w,
            self.wte_proj.b.as_mut().ok_or_else(need)?,
            &mut self.emb_ln.gamma,
            &mut self.emb_ln.beta,
        ];
        for b in &mut self.blocks {
            out.push(&mut b.ln_1.gamma);
            out.push(&mut b.ln_1.beta);
            out.push(&mut b.attn_qkv.w);
            out.push(b.attn_qkv.b.as_mut().ok_or_else(need)?);
            out.push(&mut b.attn_proj.w);
            out.push(b.attn_proj.b.as_mut().ok_or_else(need)?);
            out.push(&mut b.ln_2.gamma);
            out.push(&mut b.ln_2.beta);
            out.push(&mut b.mlp_fc.w);
            out.push(b.mlp_fc.b.as_mut().ok_or_else(need)?);
            out.push(&mut b.mlp_proj.w);
            out.push(b.mlp_proj.b.as_mut().ok_or_else(need)?);
        }
        out.push(&mut self.ln_f.gamma);
        out.push(&mut self.ln_f.beta);
        out.push(&mut self.proj.w);
        out.push(self.proj.b.as_mut().ok_or_else(need)?);
        out.push(&mut self.mlm_bias);
        Ok(out)
    }

    pub fn save_entries(&self) -> Result<crate::serialization::fzm::TensorEntries> {
        let specs = self.param_specs();
        let params = self.params()?;
        specs
            .iter()
            .zip(params)
            .map(|((name, _), t)| Ok((name.clone(), t.shape().dims().to_vec(), t.to_vec_f32()?)))
            .collect()
    }

    pub fn save_fzm(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialization::fzm::save_fzm_q4(path, &self.save_entries()?)
    }

    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialization::save_safetensors(path, &self.save_entries()?)
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if path.extension().and_then(|s| s.to_str()) == Some("fzm") {
            self.save_fzm(path)
        } else {
            self.save_safetensors(path)
        }
    }

    pub fn from_entries(
        entries: crate::serialization::fzm::TensorEntries,
        config: EncoderConfig,
        device: &Device,
    ) -> Result<Encoder> {
        config.validate()?;
        let mut map: HashMap<String, (Vec<usize>, Vec<f32>)> = entries
            .into_iter()
            .map(|(name, shape, data)| (name, (shape, data)))
            .collect();
        let mut take = |name: &str| -> Result<Tensor> {
            let (shape, data) = map
                .remove(name)
                .ok_or_else(|| ForgeError::Fzm(format!("missing tensor {name}")))?;
            Tensor::from_f32(&data, shape, device)
        };
        let eps = config.layer_norm_epsilon;
        let wte = take("wte.weight")?;
        let wte_proj = Linear {
            w: take("wte_proj.weight")?,
            b: Some(take("wte_proj.bias")?),
        };
        let emb_ln = LayerNorm {
            gamma: take("emb_ln.weight")?,
            beta: take("emb_ln.bias")?,
            eps,
        };
        let mut blocks = Vec::with_capacity(config.n_layer);
        for i in 0..config.n_layer {
            blocks.push(Block {
                ln_1: LayerNorm {
                    gamma: take(&format!("h.{i}.ln_1.weight"))?,
                    beta: take(&format!("h.{i}.ln_1.bias"))?,
                    eps,
                },
                attn_qkv: Linear {
                    w: take(&format!("h.{i}.attn.qkv.weight"))?,
                    b: Some(take(&format!("h.{i}.attn.qkv.bias"))?),
                },
                attn_proj: Linear {
                    w: take(&format!("h.{i}.attn.proj.weight"))?,
                    b: Some(take(&format!("h.{i}.attn.proj.bias"))?),
                },
                ln_2: LayerNorm {
                    gamma: take(&format!("h.{i}.ln_2.weight"))?,
                    beta: take(&format!("h.{i}.ln_2.bias"))?,
                    eps,
                },
                mlp_fc: Linear {
                    w: take(&format!("h.{i}.mlp.fc.weight"))?,
                    b: Some(take(&format!("h.{i}.mlp.fc.bias"))?),
                },
                mlp_proj: Linear {
                    w: take(&format!("h.{i}.mlp.proj.weight"))?,
                    b: Some(take(&format!("h.{i}.mlp.proj.bias"))?),
                },
            });
        }
        let ln_f = LayerNorm {
            gamma: take("ln_f.weight")?,
            beta: take("ln_f.bias")?,
            eps,
        };
        let proj = Linear {
            w: take("proj.weight")?,
            b: Some(take("proj.bias")?),
        };
        let mlm_bias = take("mlm_head.bias")?;
        let pos = Tensor::from_f32(
            &sinusoidal(config.max_seq, config.d_model),
            [config.max_seq, config.d_model],
            device,
        )?;
        Ok(Encoder {
            wte,
            wte_proj,
            emb_ln,
            blocks,
            ln_f,
            proj,
            mlm_bias,
            pos,
            config,
        })
    }

    pub fn from_fzm(
        path: impl AsRef<Path>,
        config: EncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_fzm_bytes(&bytes, config, device)
    }

    pub fn from_fzm_bytes(bytes: &[u8], config: EncoderConfig, device: &Device) -> Result<Self> {
        Self::from_entries(
            crate::serialization::fzm::load_fzm_q4(bytes)?,
            config,
            device,
        )
    }

    pub fn from_checkpoint(
        path: impl AsRef<Path>,
        config: EncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref())?;
        Self::from_checkpoint_bytes(&bytes, config, device)
    }

    pub fn from_checkpoint_bytes(
        bytes: &[u8],
        config: EncoderConfig,
        device: &Device,
    ) -> Result<Self> {
        if bytes.starts_with(b"FZM1") {
            Self::from_fzm_bytes(bytes, config, device)
        } else {
            let st = safetensors::SafeTensors::deserialize(bytes)
                .map_err(|e| ForgeError::SafeTensors(format!("deserialize: {e}")))?;
            let mut entries = crate::serialization::fzm::TensorEntries::new();
            for (name, _) in Self::specs_for(&config) {
                let v = st
                    .tensor(&name)
                    .map_err(|_| ForgeError::SafeTensors(format!("missing tensor {name}")))?;
                if v.dtype() != safetensors::tensor::Dtype::F32 {
                    return Err(ForgeError::SafeTensors(format!("{name}: expected f32")));
                }
                entries.push((
                    name,
                    v.shape().to_vec(),
                    bytemuck::pod_collect_to_vec(v.data()),
                ));
            }
            Self::from_entries(entries, config, device)
        }
    }

    fn specs_for(config: &EncoderConfig) -> Vec<(String, bool)> {
        let mut out = vec![
            ("wte.weight".to_string(), true),
            ("wte_proj.weight".to_string(), true),
            ("wte_proj.bias".to_string(), false),
            ("emb_ln.weight".to_string(), false),
            ("emb_ln.bias".to_string(), false),
        ];
        for i in 0..config.n_layer {
            for (suffix, decay) in [
                ("ln_1.weight", false),
                ("ln_1.bias", false),
                ("attn.qkv.weight", true),
                ("attn.qkv.bias", false),
                ("attn.proj.weight", true),
                ("attn.proj.bias", false),
                ("ln_2.weight", false),
                ("ln_2.bias", false),
                ("mlp.fc.weight", true),
                ("mlp.fc.bias", false),
                ("mlp.proj.weight", true),
                ("mlp.proj.bias", false),
            ] {
                out.push((format!("h.{i}.{suffix}"), decay));
            }
        }
        out.push(("ln_f.weight".to_string(), false));
        out.push(("ln_f.bias".to_string(), false));
        out.push(("proj.weight".to_string(), true));
        out.push(("proj.bias".to_string(), false));
        out.push(("mlm_head.bias".to_string(), false));
        out
    }

    fn check_batch(&self, ids: &[u32], key_len: &[u32], seq: usize) -> Result<usize> {
        if seq == 0 || !ids.len().is_multiple_of(seq) {
            return Err(ForgeError::Shape(format!(
                "{} ids is not a whole number of {seq}-token sequences",
                ids.len()
            )));
        }
        if seq > self.config.max_seq {
            return Err(ForgeError::Shape(format!(
                "sequence length {seq} exceeds max_seq {}",
                self.config.max_seq
            )));
        }
        let batch = ids.len() / seq;
        if key_len.len() != batch {
            return Err(ForgeError::Shape(format!(
                "key_len has {} entries, expected batch {batch}",
                key_len.len()
            )));
        }
        if let Some(bad) = key_len.iter().find(|&&k| k as usize > seq) {
            return Err(ForgeError::Shape(format!(
                "key_len {bad} exceeds sequence length {seq}"
            )));
        }
        Ok(batch)
    }

    fn device(&self) -> Device {
        self.wte.device()
    }

    pub fn hidden(&self, ids: &[u32], key_len: &[u32], seq: usize) -> Result<Tensor> {
        let batch = self.check_batch(ids, key_len, seq)?;
        let device = self.device();
        let _scope = device.dispatch_scope();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let klen_t = Tensor::from_u32(key_len, [batch], &device)?;
        let pos_ids = Tensor::from_u32(&position_ids(batch, seq), [ids.len()], &device)?;

        let e = ops::embedding(&ids_t, &self.wte, None, 0)?;
        let mut x = self.wte_proj.forward(&e)?;
        x = self.emb_ln.forward(&x)?;
        x = ops::add(&x, &ops::embedding(&pos_ids, &self.pos, None, 0)?)?;

        let n_head = self.config.n_head;
        let hd = self.config.d_model / n_head;
        for b in &self.blocks {
            let a = b.ln_1.forward(&x)?;
            let qkv = b.attn_qkv.forward(&a)?;
            let (q, k, v) = ops::split_heads_batched(&qkv, n_head, batch)?;
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
            let probs = ops::softmax_masked(&att, &klen_t, batch)?;
            let y = ops::matmul(&probs, &v, None, MatmulSpec::default())?;
            let y = ops::merge_heads_batched(&y, batch)?;
            let y = b.attn_proj.forward(&y)?;
            x = ops::add(&x, &y)?;
            let a2 = b.ln_2.forward(&x)?;
            let f = ops::gelu(&b.mlp_fc.forward(&a2)?)?;
            let f = b.mlp_proj.forward(&f)?;
            x = ops::add(&x, &f)?;
        }
        self.ln_f.forward(&x)
    }

    pub fn embed(&self, ids: &[u32], key_len: &[u32], seq: usize) -> Result<Tensor> {
        let batch = self.check_batch(ids, key_len, seq)?;
        let device = self.device();
        let klen_t = Tensor::from_u32(key_len, [batch], &device)?;
        let h = self.hidden(ids, key_len, seq)?;
        let pooled = ops::mean_pool(&h, &klen_t, batch)?;
        let projected = self.proj.forward(&pooled)?;
        ops::l2_norm(&projected, 1e-12)
    }

    pub fn truncate(rows: &[f32], embed_dim: usize, dim: usize) -> Result<Vec<f32>> {
        if dim == 0 || dim > embed_dim {
            return Err(ForgeError::Shape(format!(
                "truncate dim {dim} outside 1..={embed_dim}"
            )));
        }
        if !rows.len().is_multiple_of(embed_dim) {
            return Err(ForgeError::Shape("truncate: ragged embedding rows".into()));
        }
        let mut out = Vec::with_capacity(rows.len() / embed_dim * dim);
        for row in rows.chunks(embed_dim) {
            let head = &row[..dim];
            let inv = 1.0 / (head.iter().map(|v| v * v).sum::<f32>() + 1e-12).sqrt();
            out.extend(head.iter().map(|v| v * inv));
        }
        Ok(out)
    }
}

#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct EncoderForward {
    pub hidden: TVar,
    pub batch: usize,
    pub seq: usize,
    pub key_len: Tensor,
}

#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
impl Encoder {
    pub fn leaves(&self, tape: &mut Tape) -> Result<Vec<TVar>> {
        Ok(self
            .params()?
            .into_iter()
            .map(|p| tape.leaf(p.clone()))
            .collect())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_tape(
        &self,
        tape: &mut Tape,
        pvars: &[TVar],
        ids: &[u32],
        key_len: &[u32],
        seq: usize,
        dropout_p: f32,
        seed: u32,
    ) -> Result<EncoderForward> {
        let batch = self.check_batch(ids, key_len, seq)?;
        let device = self.device();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let klen_t = Tensor::from_u32(key_len, [batch], &device)?;
        let pos_ids = Tensor::from_u32(&position_ids(batch, seq), [ids.len()], &device)?;
        let pos_batch = tape.leaf(ops::embedding(&pos_ids, &self.pos, None, 0)?);

        let eps = self.config.layer_norm_epsilon;
        let n_head = self.config.n_head;
        let hd = self.config.d_model / n_head;

        let mut site = 0u32;
        let mut dseed = move || {
            site = site.wrapping_add(1);
            seed.wrapping_mul(0x9E37_79B1)
                .wrapping_add(site.wrapping_mul(0x85EB_CA77))
        };

        let e = tape.embedding_tokens(&ids_t, &pvars[0])?;
        let mut x = tape.matmul(&e, &pvars[1], Some(&pvars[2]), MatmulSpec::default())?;
        x = tape.layernorm(&x, &pvars[3], &pvars[4], eps)?;
        x = tape.add(&x, &pos_batch)?;
        x = tape.dropout(&x, dropout_p, dseed())?;

        for i in 0..self.config.n_layer {
            let base = Self::BLOCK_BASE + i * Self::BLOCK_PARAMS;
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
            let probs = tape.softmax_masked(&att, &klen_t, batch)?;
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
        let n = pvars.len();
        let hidden = tape.layernorm(&x, &pvars[n - 5], &pvars[n - 4], eps)?;
        Ok(EncoderForward {
            hidden,
            batch,
            seq,
            key_len: klen_t,
        })
    }

    pub fn pool_tape(&self, tape: &mut Tape, pvars: &[TVar], fwd: &EncoderForward) -> Result<TVar> {
        let n = pvars.len();
        let pooled = tape.mean_pool(&fwd.hidden, &fwd.key_len, fwd.batch)?;
        let projected = tape.matmul(
            &pooled,
            &pvars[n - 3],
            Some(&pvars[n - 2]),
            MatmulSpec::default(),
        )?;
        tape.l2_norm(&projected, 1e-12)
    }

    pub fn select_rows(&self, tape: &mut Tape, x: &TVar, sel: &Tensor) -> Result<TVar> {
        tape.embedding_tokens(sel, x)
    }

    pub fn mlm_logits_tape(&self, tape: &mut Tape, pvars: &[TVar], hidden: &TVar) -> Result<TVar> {
        let n = pvars.len();
        let down = tape.matmul(
            hidden,
            &pvars[1],
            None,
            MatmulSpec {
                trans_b: true,
                ..Default::default()
            },
        )?;
        tape.matmul(
            &down,
            &pvars[0],
            Some(&pvars[n - 1]),
            MatmulSpec {
                trans_b: true,
                ..Default::default()
            },
        )
    }

    pub fn collect_grads(&self, all: Vec<Option<Tensor>>) -> Result<Vec<Tensor>> {
        let params = self.params()?;
        params
            .iter()
            .zip(all)
            .map(|(p, g)| match g {
                Some(g) => Ok(g),
                None => Tensor::zeros(p.shape().clone(), &p.device()),
            })
            .collect()
    }

    pub fn truncation_matrix(embed_dim: usize, dim: usize, device: &Device) -> Result<Tensor> {
        if dim == 0 || dim > embed_dim {
            return Err(ForgeError::Shape(format!(
                "truncation dim {dim} outside 1..={embed_dim}"
            )));
        }
        let mut m = vec![0.0f32; embed_dim * dim];
        for i in 0..dim {
            m[i * dim + i] = 1.0;
        }
        Tensor::from_f32(&m, [embed_dim, dim], device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinusoidal_first_row_is_sin0_cos0() {
        let p = sinusoidal(4, 8);
        assert_eq!(&p[..8], &[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn param_count_matches_the_budget() {
        let c = EncoderConfig::ramanujan_10m();
        let n = c.n_params();
        assert!(
            (9_500_000..10_500_000).contains(&n),
            "ramanujan_10m has {n} parameters, outside the ~10M budget"
        );
    }

    #[test]
    fn specs_for_matches_param_specs() {
        let c = EncoderConfig::ramanujan_10m();
        let m = Encoder::init_random(c, &Device::Cpu, 0).unwrap();
        assert_eq!(Encoder::specs_for(&c), m.param_specs());
        assert_eq!(m.param_specs().len(), m.params().unwrap().len());
    }
}
