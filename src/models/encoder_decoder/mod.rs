mod attention;
mod config;
mod decoder;
mod encoder;

use std::path::Path;

use rand::SeedableRng;
use rand::rngs::StdRng;

use super::{Weights, argmax, normal};
use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::nn::{Embedding, LayerNorm, Linear};
use crate::ops;
use crate::serialization::fzm::{self, Body, FzmDtype, FzmManifest, TensorEntries};
use crate::tensor::Tensor;

pub use config::EncoderDecoderConfig;
pub use decoder::DecoderState;

use decoder::Decoder;
use encoder::Encoder;

pub const ARCHITECTURE: &str = "encoder_decoder";

pub(crate) enum Init {
    Normal(f32),
    Ones,
    Zeros,
}

enum Origin<'a> {
    Random(StdRng),
    Weights(Weights<'a>),
}

pub(crate) struct Source<'a> {
    device: Device,
    origin: Origin<'a>,
}

impl Source<'_> {
    fn host(&mut self, name: &str, shape: &[usize], init: Init) -> Result<Vec<f32>> {
        let n = shape.iter().product();
        match &mut self.origin {
            Origin::Weights(w) => Ok(w.host(name, Some(shape))?.1),
            Origin::Random(rng) => Ok(match init {
                Init::Ones => vec![1.0; n],
                Init::Zeros => vec![0.0; n],
                Init::Normal(std) => normal(rng, n, std),
            }),
        }
    }

    fn param(&mut self, name: &str, shape: &[usize], init: Init) -> Result<Tensor> {
        let data = self.host(name, shape, init)?;
        Tensor::from_f32(&data, shape.to_vec(), &self.device)
    }

    fn weight(&mut self, name: &str, shape: &[usize], std: f32) -> Result<Tensor> {
        match &mut self.origin {
            Origin::Weights(w) => w.weight(name, Some(shape), &self.device),
            Origin::Random(_) => self.param(name, shape, Init::Normal(std)),
        }
    }

    fn embedding(&mut self, v: usize, d: usize, wpe: Tensor) -> Result<Embedding> {
        match &mut self.origin {
            Origin::Weights(w) => w.embedding(SHARED, v, d, wpe, &self.device),
            Origin::Random(rng) => {
                Embedding::from_host_wte(&normal(rng, v * d, 0.02), v, d, wpe, &self.device)
            }
        }
    }
}

pub(crate) type Named<'a> = Vec<(String, &'a Tensor)>;

pub(crate) fn linear(
    src: &mut Source<'_>,
    name: &str,
    fan_in: usize,
    fan_out: usize,
    std: f32,
) -> Result<Linear> {
    Ok(Linear {
        w: src.weight(&format!("{name}.weight"), &[fan_in, fan_out], std)?,
        b: Some(src.param(&format!("{name}.bias"), &[fan_out], Init::Zeros)?),
    })
}

pub(crate) fn layer_norm(
    src: &mut Source<'_>,
    name: &str,
    d: usize,
    eps: f32,
) -> Result<LayerNorm> {
    Ok(LayerNorm {
        gamma: src.param(&format!("{name}.weight"), &[d], Init::Ones)?,
        beta: src.param(&format!("{name}.bias"), &[d], Init::Zeros)?,
        eps,
    })
}

pub(crate) fn push_linear<'a>(out: &mut Named<'a>, name: &str, l: &'a Linear) {
    out.push((format!("{name}.weight"), &l.w));
    if let Some(b) = &l.b {
        out.push((format!("{name}.bias"), b));
    }
}

pub(crate) fn push_layer_norm<'a>(out: &mut Named<'a>, name: &str, ln: &'a LayerNorm) {
    out.push((format!("{name}.weight"), &ln.gamma));
    out.push((format!("{name}.bias"), &ln.beta));
}

fn sinusoidal(rows: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * d];
    for (p, row) in out.chunks_mut(d).enumerate() {
        for i in 0..d / 2 {
            let angle = p as f32 / 10000f32.powf(2.0 * i as f32 / d as f32);
            row[2 * i] = angle.sin();
            row[2 * i + 1] = angle.cos();
        }
    }
    out
}

const SHARED: &str = "shared.weight";

pub struct EncoderOutput {
    hidden: Tensor,
    key_len: Vec<u32>,
    batch: usize,
    seq: usize,
}

impl EncoderOutput {
    pub fn hidden(&self) -> &Tensor {
        &self.hidden
    }

    pub fn key_len(&self) -> &[u32] {
        &self.key_len
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn seq(&self) -> usize {
        self.seq
    }

    pub fn row(&self, b: usize) -> Result<Tensor> {
        if b >= self.batch {
            return Err(ForgeError::Shape(format!(
                "encoder row {b} outside batch {}",
                self.batch
            )));
        }
        if self.batch == 1 {
            return Ok(self.hidden.clone());
        }
        ops::rows(&self.hidden, b * self.seq, self.seq)
    }
}

pub struct EncoderDecoder {
    emb: Embedding,
    encoder: Encoder,
    decoder: Decoder,
    config: EncoderDecoderConfig,
}

impl EncoderDecoder {
    pub fn config(&self) -> &EncoderDecoderConfig {
        &self.config
    }

    fn device(&self) -> Device {
        self.emb.wpe.device()
    }

    fn build(config: EncoderDecoderConfig, mut src: Source<'_>) -> Result<Self> {
        config.validate()?;
        let (v, d) = (config.vocab_size, config.d_model);
        let p = config.max_position();
        let wpe = Tensor::from_f32(&sinusoidal(p, d), [p, d], &src.device)?;
        let emb = src.embedding(v, d, wpe)?;
        let encoder = Encoder::build(&config, &mut src)?;
        let decoder = Decoder::build(&config, &mut src)?;
        Ok(EncoderDecoder {
            emb,
            encoder,
            decoder,
            config,
        })
    }

    pub fn init_random(config: EncoderDecoderConfig, device: &Device, seed: u64) -> Result<Self> {
        Self::build(
            config,
            Source {
                device: device.clone(),
                origin: Origin::Random(StdRng::seed_from_u64(seed)),
            },
        )
    }

    fn from_weights(
        weights: Weights<'_>,
        config: EncoderDecoderConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::build(
            config,
            Source {
                device: device.clone(),
                origin: Origin::Weights(weights),
            },
        )
    }

    pub fn from_entries(
        entries: TensorEntries,
        config: EncoderDecoderConfig,
        device: &Device,
    ) -> Result<Self> {
        let raw = entries
            .into_iter()
            .map(|(n, s, d)| (n, s, Body::F32(d)))
            .collect();
        Self::from_weights(Weights::from_entries(raw), config, device)
    }

    pub fn from_checkpoint_bytes(
        bytes: &[u8],
        config: Option<EncoderDecoderConfig>,
        device: &Device,
    ) -> Result<Self> {
        let (weights, manifest) = Weights::parse(bytes, ARCHITECTURE)?;
        let config = match (config, manifest) {
            (Some(c), _) => c,
            (None, Some(m)) => serde_json::from_value(m.config)?,
            (None, None) => {
                return Err(ForgeError::Fzm(
                    "checkpoint carries no config; pass one explicitly".into(),
                ));
            }
        };
        Self::from_weights(weights, config, device)
    }

    pub fn from_checkpoint(
        path: impl AsRef<Path>,
        config: Option<EncoderDecoderConfig>,
        device: &Device,
    ) -> Result<Self> {
        Self::from_checkpoint_bytes(&std::fs::read(path.as_ref())?, config, device)
    }

    fn named(&self) -> Named<'_> {
        let mut out = Vec::new();
        self.encoder.named(&mut out);
        self.decoder.named(&mut out);
        out
    }

    pub fn n_params(&self) -> usize {
        self.config.n_params()
    }

    pub fn save_entries(&self) -> Result<TensorEntries> {
        let (v, d) = (self.config.vocab_size, self.config.d_model);
        let mut wte = Vec::with_capacity(v * d);
        for chunk in &self.emb.wte_chunks {
            wte.extend(chunk.to_vec_f32()?);
        }
        let mut out = vec![(SHARED.to_string(), vec![v, d], wte)];
        for (name, t) in self.named() {
            out.push((name, t.shape().dims().to_vec(), t.to_vec_f32()?));
        }
        Ok(out)
    }

    pub fn manifest(&self) -> Result<FzmManifest> {
        let c = &self.config;
        let mut m = FzmManifest::new(ARCHITECTURE, serde_json::to_value(c)?);
        m.model = serde_json::json!({
            "hidden_size": c.d_model,
            "encoder_layers": c.n_encoder_layer,
            "decoder_layers": c.n_decoder_layer,
            "attention_heads": c.n_head,
            "vocab_size": c.vocab_size,
        });
        Ok(m)
    }

    pub fn save_fzm_with(
        &self,
        path: impl AsRef<Path>,
        manifest: &FzmManifest,
        dtype: FzmDtype,
        assets: &[(String, Vec<u8>)],
    ) -> Result<()> {
        fzm::save_fzm_v2(path, manifest, &self.save_entries()?, dtype, assets)
    }

    pub fn save_fzm(&self, path: impl AsRef<Path>) -> Result<()> {
        self.save_fzm_with(path, &self.manifest()?, FzmDtype::Q4, &[])
    }

    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        crate::serialization::save_safetensors(path, &self.save_entries()?)
    }

    fn check_ids(&self, ids: &[u32]) -> Result<()> {
        if let Some(bad) = ids.iter().find(|&&i| i as usize >= self.config.vocab_size) {
            return Err(ForgeError::Shape(format!(
                "token {bad} outside vocab {}",
                self.config.vocab_size
            )));
        }
        Ok(())
    }

    pub fn encode(&self, ids: &[u32], key_len: &[u32], seq: usize) -> Result<EncoderOutput> {
        if seq == 0 || ids.is_empty() || !ids.len().is_multiple_of(seq) {
            return Err(ForgeError::Shape(format!(
                "{} ids is not a whole number of {seq}-token sequences",
                ids.len()
            )));
        }
        if seq > self.config.max_source {
            return Err(ForgeError::Shape(format!(
                "source length {seq} exceeds max_source {}",
                self.config.max_source
            )));
        }
        let batch = ids.len() / seq;
        if key_len.len() != batch {
            return Err(ForgeError::Shape(format!(
                "key_len has {} entries, expected batch {batch}",
                key_len.len()
            )));
        }
        if let Some(bad) = key_len.iter().find(|&&k| k == 0 || k as usize > seq) {
            return Err(ForgeError::Shape(format!(
                "key_len {bad} outside 1..={seq}"
            )));
        }
        self.check_ids(ids)?;
        let device = self.device();
        let _scope = device.dispatch_scope();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let klen_t = Tensor::from_u32(key_len, [batch], &device)?;
        let x = self.emb.forward_batched(&ids_t, 0, seq)?;
        let hidden = self
            .encoder
            .forward(x, self.config.n_head, batch, &klen_t)?;
        Ok(EncoderOutput {
            hidden,
            key_len: key_len.to_vec(),
            batch,
            seq,
        })
    }

    pub fn start(&self, memory: &EncoderOutput, row: usize) -> Result<DecoderState> {
        let hidden = memory.row(row)?;
        let device = self.device();
        let _scope = device.dispatch_scope();
        self.decoder
            .start(&self.config, &hidden, memory.key_len[row], &device)
    }

    fn logits(&self, ids: &[u32], state: &mut DecoderState) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(ForgeError::Shape("empty decoder input".into()));
        }
        self.check_ids(ids)?;
        let device = self.device();
        let _scope = device.dispatch_scope();
        let ids_t = Tensor::from_u32(ids, [ids.len()], &device)?;
        let x = self.emb.forward(&ids_t, state.len())?;
        let h = self.decoder.forward(&self.config, x, state)?;
        let last = ops::rows(&h, ids.len() - 1, 1)?;
        ops::matmul_chunked_transb(&last, &self.emb.wte_chunks, 1.0)
    }

    pub fn step(&self, ids: &[u32], state: &mut DecoderState) -> Result<Vec<f32>> {
        self.logits(ids, state)?.to_vec_f32()
    }

    pub async fn step_async(&self, ids: &[u32], state: &mut DecoderState) -> Result<Vec<f32>> {
        self.logits(ids, state)?.to_vec_f32_async().await
    }

    fn prefill(&self, src: &[u32]) -> Result<DecoderState> {
        let memory = self.encode(src, &[src.len() as u32], src.len())?;
        self.start(&memory, 0)
    }

    pub fn generate_greedy(&self, src: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let mut state = self.prefill(src)?;
        let mut token = self.config.bos_id;
        let mut out = Vec::new();
        for _ in 0..max_new.min(self.config.max_target) {
            token = argmax(&self.step(&[token], &mut state)?);
            if token == self.config.eos_id {
                break;
            }
            out.push(token);
        }
        Ok(out)
    }

    pub async fn generate_greedy_async(&self, src: &[u32], max_new: usize) -> Result<Vec<u32>> {
        let mut state = self.prefill(src)?;
        let mut token = self.config.bos_id;
        let mut out = Vec::new();
        for _ in 0..max_new.min(self.config.max_target) {
            token = argmax(&self.step_async(&[token], &mut state).await?);
            if token == self.config.eos_id {
                break;
            }
            out.push(token);
        }
        Ok(out)
    }
}
