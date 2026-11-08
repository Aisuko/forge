use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{ForgeError, Result};

fn default_eps() -> f32 {
    1e-5
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncoderDecoderConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_encoder_layer: usize,
    pub n_decoder_layer: usize,
    pub n_head: usize,
    pub d_ff: usize,
    pub max_source: usize,
    pub max_target: usize,
    #[serde(default = "default_eps")]
    pub layer_norm_epsilon: f32,
    pub pad_id: u32,
    pub bos_id: u32,
    pub eos_id: u32,
}

impl EncoderDecoderConfig {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_json_str(&std::fs::read_to_string(path)?)
    }

    pub fn from_json_str(text: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_head
    }

    pub fn max_position(&self) -> usize {
        self.max_source.max(self.max_target)
    }

    pub fn validate(&self) -> Result<()> {
        let fail = |m: String| Err(ForgeError::Shape(m));
        if self.vocab_size == 0 || self.d_model == 0 || self.d_ff == 0 || self.n_head == 0 {
            return fail("encoder-decoder dims must be non-zero".into());
        }
        if !self.d_model.is_multiple_of(self.n_head) {
            return fail(format!(
                "d_model {} not divisible by n_head {}",
                self.d_model, self.n_head
            ));
        }
        if self.max_source == 0 || self.max_target == 0 {
            return fail("max_source and max_target must be non-zero".into());
        }
        for (name, id) in [
            ("pad", self.pad_id),
            ("bos", self.bos_id),
            ("eos", self.eos_id),
        ] {
            if id as usize >= self.vocab_size {
                return fail(format!("{name}_id {id} outside vocab {}", self.vocab_size));
            }
        }
        Ok(())
    }

    pub fn n_params(&self) -> usize {
        let (v, d, f) = (self.vocab_size, self.d_model, self.d_ff);
        let ln = 2 * d;
        let lin = |i: usize, o: usize| i * o + o;
        let enc = 2 * ln + lin(d, 3 * d) + lin(d, d) + lin(d, f) + lin(f, d);
        let dec = enc + ln + 4 * lin(d, d);
        v * d + self.n_encoder_layer * enc + ln + self.n_decoder_layer * dec + ln
    }
}
