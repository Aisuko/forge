//! Optimizers: AdamW with decoupled weight decay and optional global
//! gradient-norm clipping. Backend-agnostic — state lives on the same device
//! as the parameters, and updates run through `ops`.

use crate::error::{ForgeError, Result};
use crate::ops;
use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug)]
pub struct AdamWOpts {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// Clip the global gradient norm to this value when set.
    pub clip: Option<f32>,
}

impl Default for AdamWOpts {
    fn default() -> Self {
        AdamWOpts {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            clip: Some(1.0),
        }
    }
}

pub struct AdamW {
    pub opts: AdamWOpts,
    step: u32,
    /// (m, v) per parameter, in registration order.
    state: Vec<(Tensor, Tensor)>,
    /// Whether weight decay applies (false for biases and LayerNorm params).
    decay: Vec<bool>,
}

impl AdamW {
    /// `params` supplies each parameter with its decay flag; order must
    /// match the `step` calls.
    pub fn new(params: &[(&Tensor, bool)], opts: AdamWOpts) -> Result<AdamW> {
        let mut state = Vec::with_capacity(params.len());
        let mut decay = Vec::with_capacity(params.len());
        for (p, d) in params {
            let device = p.device();
            state.push((
                Tensor::zeros(p.shape().clone(), &device)?,
                Tensor::zeros(p.shape().clone(), &device)?,
            ));
            decay.push(*d);
        }
        Ok(AdamW {
            opts,
            step: 0,
            state,
            decay,
        })
    }

    /// Apply one update. Returns the pre-clip global gradient norm.
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &[Tensor]) -> Result<f32> {
        if params.len() != self.state.len() || grads.len() != self.state.len() {
            return Err(ForgeError::Shape(format!(
                "AdamW::step arity mismatch: {} params, {} grads, {} slots",
                params.len(),
                grads.len(),
                self.state.len()
            )));
        }
        self.step += 1;
        let mut total_sq = 0.0f32;
        for g in grads {
            total_sq += ops::sumsq(g)?;
        }
        let norm = total_sq.sqrt();
        let rescale = match self.opts.clip {
            Some(c) if norm > c && norm > 0.0 => Some(c / norm),
            _ => None,
        };
        let mut clipped;
        for ((p, g), ((m, v), decay)) in params
            .iter_mut()
            .zip(grads)
            .zip(self.state.iter_mut().zip(&self.decay))
        {
            let g = match rescale {
                Some(s) => {
                    clipped = ops::scale(g, s)?;
                    &clipped
                }
                None => g,
            };
            ops::adamw(
                p,
                g,
                m,
                v,
                self.opts.lr,
                self.opts.beta1,
                self.opts.beta2,
                self.opts.eps,
                if *decay { self.opts.weight_decay } else { 0.0 },
                self.step,
            )?;
        }
        Ok(norm)
    }
}
