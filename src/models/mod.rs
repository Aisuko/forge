pub mod encoder_decoder;
pub mod gpt2;

use std::collections::HashMap;

use rand::Rng;
use rand::rngs::StdRng;
use safetensors::SafeTensors;

use crate::device::Device;
use crate::error::{ForgeError, Result};
use crate::nn::Embedding;
use crate::serialization::fzm::{self, Body, FzmManifest, RawEntries};
use crate::tensor::Tensor;

pub(crate) fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

pub(crate) fn normal(rng: &mut StdRng, n: usize, std: f32) -> Vec<f32> {
    (0..n)
        .map(|_| {
            let u1: f32 = rng.random::<f32>().max(1e-7);
            let u2: f32 = rng.random::<f32>();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos() * std
        })
        .collect()
}

pub(crate) enum Weights<'a> {
    Map(HashMap<String, (Vec<usize>, Body)>),
    Safetensors(SafeTensors<'a>),
}

impl<'a> Weights<'a> {
    pub fn from_entries(entries: RawEntries) -> Self {
        Weights::Map(
            entries
                .into_iter()
                .map(|(name, shape, body)| (name, (shape, body)))
                .collect(),
        )
    }

    pub fn parse(bytes: &'a [u8], architecture: &str) -> Result<(Self, Option<FzmManifest>)> {
        if !fzm::is_fzm(bytes) {
            let st = SafeTensors::deserialize(bytes)
                .map_err(|e| ForgeError::SafeTensors(format!("deserialize: {e}")))?;
            return Ok((Weights::Safetensors(st), None));
        }
        let file = fzm::load_fzm(bytes)?;
        fzm::check_architecture(file.manifest.as_ref(), architecture)?;
        Ok((Self::from_entries(file.entries), file.manifest))
    }

    fn body(&mut self, name: &str) -> Result<(Vec<usize>, Body)> {
        let alias = format!("transformer.{name}");
        match self {
            Weights::Map(map) => map
                .remove(name)
                .or_else(|| map.remove(&alias))
                .ok_or_else(|| ForgeError::Fzm(format!("missing tensor {name}"))),
            Weights::Safetensors(st) => {
                let view = st
                    .tensor(name)
                    .or_else(|_| st.tensor(&alias))
                    .map_err(|_| ForgeError::SafeTensors(format!("missing tensor {name}")))?;
                Ok((
                    view.shape().to_vec(),
                    Body::F32(crate::serialization::view_f32(name, &view)?),
                ))
            }
        }
    }

    fn checked(&mut self, name: &str, shape: Option<&[usize]>) -> Result<(Vec<usize>, Body)> {
        let (got, body) = self.body(name)?;
        match shape {
            Some(want) if want != got.as_slice() => Err(ForgeError::Shape(format!(
                "{name}: checkpoint shape {got:?}, expected {want:?}"
            ))),
            _ => Ok((got, body)),
        }
    }

    pub fn host(&mut self, name: &str, shape: Option<&[usize]>) -> Result<(Vec<usize>, Vec<f32>)> {
        let (shape, body) = self.checked(name, shape)?;
        Ok((shape, body.into_f32()))
    }

    pub fn param(
        &mut self,
        name: &str,
        shape: Option<&[usize]>,
        device: &Device,
    ) -> Result<Tensor> {
        let (shape, data) = self.host(name, shape)?;
        Tensor::from_f32(&data, shape, device)
    }

    pub fn weight(
        &mut self,
        name: &str,
        shape: Option<&[usize]>,
        device: &Device,
    ) -> Result<Tensor> {
        match self.checked(name, shape)? {
            (shape, Body::Q4(q)) => Tensor::from_q4(&q, shape, device),
            (shape, Body::F32(data)) => Tensor::from_f32(&data, shape, device),
        }
    }

    pub fn embedding(
        &mut self,
        name: &str,
        vocab: usize,
        d: usize,
        wpe: Tensor,
        device: &Device,
    ) -> Result<Embedding> {
        match self.checked(name, Some(&[vocab, d]))?.1 {
            Body::Q4(q) => Embedding::from_q4_wte(&q, vocab, d, wpe, device),
            Body::F32(data) => Embedding::from_host_wte(&data, vocab, d, wpe, device),
        }
    }
}
