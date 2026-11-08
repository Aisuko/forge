//! Forge — a WebGPU-native machine learning runtime in Rust for GPT-2 and
//! encoder–decoder transformers, with native packed Q4 weights on the GPU.
//!
//! The `train` feature (off by default) adds [`autograd`], [`optim`] and the
//! backward kernels.

#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub mod autograd;
pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod models;
pub mod nn;
pub mod ops;
#[cfg(feature = "train")]
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub mod optim;
pub mod serialization;
pub mod shape;
pub mod tensor;
pub mod tokenizer;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub use device::Device;
pub use dtype::DType;
pub use error::{ForgeError, Result};
pub use models::encoder_decoder::{
    DecoderState, EncoderDecoder, EncoderDecoderConfig, EncoderOutput,
};
pub use models::gpt2::{
    AttnStep, Gpt2, Gpt2Config, KvCache, LayerDetail, Sampler, Sampling, StepTrace, top_probs,
};
pub use shape::Shape;
pub use tensor::Tensor;
pub use tokenizer::{AnyTokenizer, CharTokenizer, Gpt2Tokenizer, Tokenizer};
