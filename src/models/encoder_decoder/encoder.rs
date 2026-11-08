use crate::error::Result;
use crate::nn::{LayerNorm, Linear};
use crate::ops;
use crate::tensor::Tensor;

use super::attention::SelfAttention;
use super::config::EncoderDecoderConfig;
use super::{Named, Source, layer_norm, linear, push_layer_norm, push_linear};

pub(crate) struct EncoderBlock {
    ln_1: LayerNorm,
    attn: SelfAttention,
    ln_2: LayerNorm,
    mlp_fc: Linear,
    mlp_proj: Linear,
}

pub(crate) struct Encoder {
    blocks: Vec<EncoderBlock>,
    ln_f: LayerNorm,
}

impl Encoder {
    pub fn build(c: &EncoderDecoderConfig, src: &mut Source<'_>) -> Result<Self> {
        let (d, f, eps) = (c.d_model, c.d_ff, c.layer_norm_epsilon);
        let resid = 0.02 / (2.0 * c.n_encoder_layer.max(1) as f32).sqrt();
        let mut blocks = Vec::with_capacity(c.n_encoder_layer);
        for i in 0..c.n_encoder_layer {
            let p = format!("encoder.h.{i}");
            blocks.push(EncoderBlock {
                ln_1: layer_norm(src, &format!("{p}.ln_1"), d, eps)?,
                attn: SelfAttention {
                    qkv: linear(src, &format!("{p}.attn.qkv"), d, 3 * d, 0.02)?,
                    proj: linear(src, &format!("{p}.attn.proj"), d, d, resid)?,
                },
                ln_2: layer_norm(src, &format!("{p}.ln_2"), d, eps)?,
                mlp_fc: linear(src, &format!("{p}.mlp.fc"), d, f, 0.02)?,
                mlp_proj: linear(src, &format!("{p}.mlp.proj"), f, d, resid)?,
            });
        }
        Ok(Encoder {
            blocks,
            ln_f: layer_norm(src, "encoder.ln_f", d, eps)?,
        })
    }

    pub fn named<'a>(&'a self, out: &mut Named<'a>) {
        for (i, b) in self.blocks.iter().enumerate() {
            let p = format!("encoder.h.{i}");
            push_layer_norm(out, &format!("{p}.ln_1"), &b.ln_1);
            push_linear(out, &format!("{p}.attn.qkv"), &b.attn.qkv);
            push_linear(out, &format!("{p}.attn.proj"), &b.attn.proj);
            push_layer_norm(out, &format!("{p}.ln_2"), &b.ln_2);
            push_linear(out, &format!("{p}.mlp.fc"), &b.mlp_fc);
            push_linear(out, &format!("{p}.mlp.proj"), &b.mlp_proj);
        }
        push_layer_norm(out, "encoder.ln_f", &self.ln_f);
    }

    pub fn forward(
        &self,
        mut x: Tensor,
        n_head: usize,
        batch: usize,
        key_len: &Tensor,
    ) -> Result<Tensor> {
        for b in &self.blocks {
            let a = b
                .attn
                .forward_masked(&b.ln_1.forward(&x)?, n_head, batch, key_len)?;
            x = ops::add(&x, &a)?;
            let h = ops::gelu(&b.mlp_fc.forward(&b.ln_2.forward(&x)?)?)?;
            x = ops::add(&x, &b.mlp_proj.forward(&h)?)?;
        }
        self.ln_f.forward(&x)
    }
}
