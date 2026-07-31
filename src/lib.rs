//! Forge — a WebGPU-native machine learning framework in Rust, scoped to GPT-2.
//!
//! Production execution targets WebGPU via `wgpu`; the CPU backend is a
//! mathematically identical reference used for testing and verification.
//!
//! # Optional features
//!
//! None is on by default, so `cargo add forge-ml` gets the runtime alone.
//!
//! - **`council`** — [`Council`], several small GPT-2s run on one prompt in
//!   parallel and merged in their own hidden space. Composition over the
//!   runtime: no extra dependency, no extra kernel.
//! - **`train`** — [`autograd`] and [`optim`], the backward kernels, and
//!   `Gpt2::loss`/`loss_grads`. Off by default because Forge is an inference
//!   runtime that also happens to train; `cargo add forge-ml` should not
//!   compile a tape you never record. Construction and serialization
//!   (`Gpt2::init_random`, `params`, `save_safetensors`) stay core — they are
//!   not training, and the inference tests use them.
//! - **`tui`** — the `forge-top` terminal model browser. Its dependencies are
//!   optional so they never reach this library's dependents or the wasm build.

// Nightly-only, and only ever set by docs.rs (see `[package.metadata.docs.rs]`),
// so this is inert on stable. It is what puts the "Available on crate feature"
// badges on the items below.
#![cfg_attr(docsrs, feature(doc_cfg))]

// Behind `train`. The module's own `//!` docs carry the description; an outer
// `///` here would move intra-doc link resolution into this file's scope and
// break `[`Tape`]`.
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
// Behind `train`; see the note on `autograd` above.
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
#[cfg(feature = "council")]
#[cfg_attr(docsrs, doc(cfg(feature = "council")))]
pub use models::council::{Council, CouncilStep, ExpertStep};
pub use models::gpt2::{
    AttnStep, Gpt2, Gpt2Config, KvCache, LayerDetail, Sampling, StepTrace, Surprisal,
};
pub use shape::Shape;
pub use tensor::Tensor;
pub use tokenizer::{AnyTokenizer, CharTokenizer, Gpt2Tokenizer, Tokenizer};
