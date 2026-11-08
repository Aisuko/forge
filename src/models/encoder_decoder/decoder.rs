use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::tensor::Tensor;

use super::attention::{CrossAttention, SelfAttention};
use super::config::EncoderDecoderConfig;
use super::{Named, Source, layer_norm, linear, push_layer_norm, push_linear};

pub(crate) struct DecoderBlock {
    ln_1: LayerNorm,
    self_attn: SelfAttention,
    ln_2: LayerNorm,
    cross_attn: CrossAttention,
    ln_3: LayerNorm,
    mlp_fc: Linear,
    mlp_proj: Linear,
}

pub(crate) struct Decoder {
    blocks: Vec<DecoderBlock>,
    ln_f: LayerNorm,
}

pub struct DecoderState {
    self_k: Vec<Tensor>,
    self_v: Vec<Tensor>,
    cross_k: Vec<Tensor>,
    cross_v: Vec<Tensor>,
    key_len: Tensor,
    len: usize,
}

impl DecoderState {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Decoder {
    pub fn build(c: &EncoderDecoderConfig, src: &mut Source<'_>) -> Result<Self> {
        let (d, f, eps) = (c.d_model, c.d_ff, c.layer_norm_epsilon);
        let resid = 0.02 / (3.0 * c.n_decoder_layer.max(1) as f32).sqrt();
        let mut blocks = Vec::with_capacity(c.n_decoder_layer);
        for i in 0..c.n_decoder_layer {
            let p = format!("decoder.h.{i}");
            blocks.push(DecoderBlock {
                ln_1: layer_norm(src, &format!("{p}.ln_1"), d, eps)?,
                self_attn: SelfAttention {
                    qkv: linear(src, &format!("{p}.self_attn.qkv"), d, 3 * d, 0.02)?,
                    proj: linear(src, &format!("{p}.self_attn.proj"), d, d, resid)?,
                },
                ln_2: layer_norm(src, &format!("{p}.ln_2"), d, eps)?,
                cross_attn: CrossAttention {
                    q: linear(src, &format!("{p}.cross_attn.q"), d, d, 0.02)?,
                    k: linear(src, &format!("{p}.cross_attn.k"), d, d, 0.02)?,
                    v: linear(src, &format!("{p}.cross_attn.v"), d, d, 0.02)?,
                    proj: linear(src, &format!("{p}.cross_attn.proj"), d, d, resid)?,
                },
                ln_3: layer_norm(src, &format!("{p}.ln_3"), d, eps)?,
                mlp_fc: linear(src, &format!("{p}.mlp.fc"), d, f, 0.02)?,
                mlp_proj: linear(src, &format!("{p}.mlp.proj"), f, d, resid)?,
            });
        }
        Ok(Decoder {
            blocks,
            ln_f: layer_norm(src, "decoder.ln_f", d, eps)?,
        })
    }

    pub fn named<'a>(&'a self, out: &mut Named<'a>) {
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("decoder.h.{i}");
            push_layer_norm(out, &format!("{p}.ln_1"), &b.ln_1);
            push_linear(out, &format!("{p}.self_attn.qkv"), &b.self_attn.qkv);
            push_linear(out, &format!("{p}.self_attn.proj"), &b.self_attn.proj);
            push_layer_norm(out, &format!("{p}.ln_2"), &b.ln_2);
            push_linear(out, &format!("{p}.cross_attn.q"), &b.cross_attn.q);
            push_linear(out, &format!("{p}.cross_attn.k"), &b.cross_attn.k);
            push_linear(out, &format!("{p}.cross_attn.v"), &b.cross_attn.v);
            push_linear(out, &format!("{p}.cross_attn.proj"), &b.cross_attn.proj);
            push_layer_norm(out, &format!("{p}.ln_3"), &b.ln_3);
            push_linear(out, &format!("{p}.mlp.fc"), &b.mlp_fc);
            push_linear(out, &format!("{p}.mlp.proj"), &b.mlp_proj);
        }
        push_layer_norm(out, "decoder.ln_f", &self.ln_f);
    }

    pub fn start(
        &self,
        c: &EncoderDecoderConfig,
        memory: &Tensor,
        key_len: u32,
        device: &Device,
    ) -> Result<DecoderState> {
        let shape = [c.n_head, c.max_target, c.head_dim()];
        let n = self.blocks.len();
        let mut state = DecoderState {
            self_k: Vec::with_capacity(n),
            self_v: Vec::with_capacity(n),
            cross_k: Vec::with_capacity(n),
            cross_v: Vec::with_capacity(n),
            key_len: Tensor::from_u32(&[key_len], [1], device)?,
            len: 0,
        };
        for b in &self.blocks {
            let (k, v) = b.cross_attn.memory(memory, c.n_head)?;
            state.cross_k.push(k);
            state.cross_v.push(v);
            state.self_k.push(Tensor::zeros(shape, device)?);
            state.self_v.push(Tensor::zeros(shape, device)?);
        }
        Ok(state)
    }

    pub fn forward(
        &self,
        c: &EncoderDecoderConfig,
        mut x: Tensor,
        state: &mut DecoderState,
    ) -> Result<Tensor> {
        let t = x.shape().dim(0);
        if state.len + t > c.max_target {
            return Err(ForgeError::Shape(format!(
                "decoder length {}+{t} exceeds max_target {}",
                state.len, c.max_target
            )));
        }
        let len = state.len;
        for (i, b) in self.blocks.iter().enumerate() {
            let a = b.self_attn.forward_cached(
                &b.ln_1.forward(&x)?,
                c.n_head,
                &mut state.self_k[i],
                &mut state.self_v[i],
                len,
            )?;
            x = ops::add(&x, &a)?;
            let a = b.cross_attn.forward(
                &b.ln_2.forward(&x)?,
                c.n_head,
                &state.cross_k[i],
                &state.cross_v[i],
                &state.key_len,
            )?;
            x = ops::add(&x, &a)?;
            let h = ops::gelu(&b.mlp_fc.forward(&b.ln_3.forward(&x)?)?)?;
            x = ops::add(&x, &b.mlp_proj.forward(&h)?)?;
        }
        state.len += t;
        self.ln_f.forward(&x)
    }
}
