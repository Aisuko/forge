//! Forge — a WebGPU-native machine learning framework in Rust, scoped to GPT-2.
//!
//! Production execution targets WebGPU via `wgpu`; the CPU backend is a
//! mathematically identical reference used for testing and verification.

pub mod autograd;
pub mod backend;
pub mod device;
pub mod dtype;
pub mod error;
pub mod models;
pub mod nn;
pub mod ops;
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
pub use models::gpt2::{Gpt2, Gpt2Config, KvCache, Sampling};
pub use shape::Shape;
pub use tensor::Tensor;
pub use tokenizer::{AnyTokenizer, CharTokenizer, Gpt2Tokenizer, Tokenizer};
