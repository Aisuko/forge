//! Forge — a WebGPU-native machine learning framework in Rust, scoped to GPT-2.
//!
//! Production execution targets WebGPU via `wgpu`; the CPU backend is a
//! mathematically identical reference used for testing and verification.
//!
//! # Optional features
//!
//! Neither is on by default, so `cargo add forge-ml` gets the runtime alone.
//!
//! - **`council`** — [`Council`], several small GPT-2s run on one prompt in
//!   parallel and merged in their own hidden space. Composition over the
//!   runtime: no extra dependency, no extra kernel.
//! - **`tui`** — the `forge-top` terminal model browser. Its dependencies are
//!   optional so they never reach this library's dependents or the wasm build.

// Nightly-only, and only ever set by docs.rs (see `[package.metadata.docs.rs]`),
// so this is inert on stable. It is what puts the "Available on crate feature"
// badges on the items below.
#![cfg_attr(docsrs, feature(doc_cfg))]

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
#[cfg(feature = "council")]
#[cfg_attr(docsrs, doc(cfg(feature = "council")))]
pub use models::council::{Council, CouncilStep, ExpertStep};
pub use models::gpt2::{AttnStep, Gpt2, Gpt2Config, KvCache, LayerDetail, Sampling, StepTrace};
pub use shape::Shape;
pub use tensor::Tensor;
pub use tokenizer::{AnyTokenizer, CharTokenizer, Gpt2Tokenizer, Tokenizer};
