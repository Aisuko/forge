//! The generation worker.
//!
//! Inference never runs on the main thread: a single `logits_step` on the CPU
//! backend takes long enough to visibly freeze input handling. Everything here
//! happens on a thread spawned per run, reporting back over the shared
//! `Event` channel.

use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::time::Instant;

use forge::{AnyTokenizer, CharTokenizer, Device, Gpt2, Gpt2Tokenizer, Sampling, Tokenizer as _};

use crate::Event;
use crate::scan::{ModelInfo, TokenizerKind};

#[derive(Clone, Copy, PartialEq)]
pub enum Backend {
    Wgpu,
    Cpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Backend::Wgpu => "wgpu",
            Backend::Cpu => "cpu",
        }
    }

    pub fn toggled(self) -> Self {
        match self {
            Backend::Wgpu => Backend::Cpu,
            Backend::Cpu => Backend::Wgpu,
        }
    }
}

pub struct RunSpec {
    pub model: ModelInfo,
    pub backend: Backend,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub sampling: Sampling,
}

/// Spawn a run. The returned flag cancels it: the generation callback checks
/// it between tokens, so cancellation lands after the token in flight.
pub fn spawn(spec: RunSpec, send: Sender<Event>) -> Arc<AtomicBool> {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    std::thread::spawn(move || {
        let outcome = execute(spec, &send, &flag);
        let _ = send.send(Event::RunFinished(outcome.err()));
    });
    cancel
}

fn execute(spec: RunSpec, send: &Sender<Event>, cancel: &AtomicBool) -> Result<(), String> {
    let load_start = Instant::now();
    let device = match spec.backend {
        Backend::Cpu => Device::Cpu,
        Backend::Wgpu => Device::wgpu().map_err(|e| format!("no WebGPU device: {e}"))?,
    };
    let config = spec
        .model
        .config
        .clone()
        .ok_or("model has no usable config.json")?;
    let tokenizer = load_tokenizer(&spec.model)?;
    let model = Gpt2::from_safetensors(&spec.model.path, config, &device)
        .map_err(|e| format!("load failed: {e}"))?;
    let _ = send.send(Event::RunStarted {
        device: device.describe(),
        load: load_start.elapsed(),
        at: Instant::now(),
    });

    // Reject a prompt the char vocab cannot represent *before* generating,
    // naming the offending characters rather than failing opaquely.
    if let AnyTokenizer::Char(c) = &tokenizer {
        let unknown = c.unknown_chars(&spec.prompt);
        if !unknown.is_empty() {
            return Err(format!(
                "prompt has {} character(s) outside the {}-token vocab: {}",
                unknown.len(),
                c.vocab_size(),
                unknown
                    .iter()
                    .map(|c| format!("{c:?}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            ));
        }
    }

    model
        .generate_streaming_ctl(
            &tokenizer,
            &spec.prompt,
            spec.max_new_tokens,
            spec.sampling,
            |id, text| {
                let sent = send.send(Event::Token {
                    id,
                    text: text.to_string(),
                    at: Instant::now(),
                });
                if sent.is_err() || cancel.load(Ordering::Relaxed) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            },
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}

fn load_tokenizer(model: &ModelInfo) -> Result<AnyTokenizer, String> {
    let vocab = model.vocab_path.as_ref().ok_or("no vocab.json found")?;
    match model.tokenizer {
        TokenizerKind::Bpe => {
            let merges = model.merges_path.as_ref().ok_or("no merges.txt found")?;
            Gpt2Tokenizer::from_files(vocab, merges)
                .map(AnyTokenizer::bpe)
                .map_err(|e| e.to_string())
        }
        TokenizerKind::Char => CharTokenizer::from_json_file(vocab)
            .map(AnyTokenizer::Char)
            .map_err(|e| e.to_string()),
        TokenizerKind::None => Err("no tokenizer beside the weights".into()),
    }
}
