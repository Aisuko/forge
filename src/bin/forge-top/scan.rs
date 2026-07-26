//! Model discovery: walk the given roots for `*.safetensors` and read each
//! file's *header only*.
//!
//! The whole point is that listing a 548 MB checkpoint must not cost 548 MB of
//! RSS. `SafeTensors::read_metadata` looks like it accepts just the header but
//! asserts the buffer runs to the end of the file
//! (`safetensors-0.5.3/src/tensor.rs:320`), so the file is memory-mapped and
//! the full mapped slice handed over — the OS pages in only the few KB the
//! parser actually touches. `SafeTensors::deserialize` is never called here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use forge::Gpt2Config;

use crate::Event;

/// Deepest directory level walked below each root.
const MAX_DEPTH: usize = 4;

#[derive(Clone)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
}

/// Which tokenizer files sit beside a checkpoint, and hence whether it can
/// actually be run.
#[derive(Clone, Copy, PartialEq)]
pub enum TokenizerKind {
    Bpe,
    Char,
    None,
}

impl TokenizerKind {
    pub fn label(self) -> &'static str {
        match self {
            TokenizerKind::Bpe => "bpe",
            TokenizerKind::Char => "char",
            TokenizerKind::None => "—",
        }
    }
}

#[derive(Clone)]
pub struct ModelInfo {
    /// The `.safetensors` file itself.
    pub path: PathBuf,
    /// Display name — the parent directory for `models/gpt2/model.safetensors`,
    /// otherwise the file stem.
    pub name: String,
    pub file_size: u64,
    pub params: usize,
    pub tensors: Vec<TensorEntry>,
    pub config: Option<Gpt2Config>,
    /// Path of the config sidecar, when one parsed.
    pub config_path: Option<PathBuf>,
    pub vocab_path: Option<PathBuf>,
    pub merges_path: Option<PathBuf>,
    pub tokenizer: TokenizerKind,
    /// Why the model cannot be run, if it cannot.
    pub blocked: Option<String>,
}

impl ModelInfo {
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn runnable(&self) -> bool {
        self.blocked.is_none()
    }
}

pub fn human_bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = n as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{n}{}", UNITS[u])
    } else {
        format!("{v:.1}{}", UNITS[u])
    }
}

pub fn human_params(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

/// Walk `roots`, streaming each model to `send` as it is parsed so a slow
/// mount cannot stall the UI. Runs on its own thread.
pub fn scan(roots: Vec<PathBuf>, send: std::sync::mpsc::Sender<Event>) {
    let mut files = Vec::new();
    for root in &roots {
        collect(root, 0, &mut files);
    }
    files.sort();
    files.dedup();
    for f in files {
        match read_model(&f) {
            Ok(info) => {
                if send.send(Event::ModelFound(Box::new(info))).is_err() {
                    return; // UI gone
                }
            }
            Err(e) => {
                let _ = send.send(Event::ScanWarning(format!("{}: {e}", f.display())));
            }
        }
    }
    let _ = send.send(Event::ScanDone);
}

fn collect(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(ft) = entry.file_type() else { continue };
        if ft.is_dir() {
            // Build output and VCS internals never hold a model the user
            // meant; `dist/` in particular would list the site's copy of the
            // demo checkpoint a second time.
            let skip = matches!(
                path.file_name().and_then(|s| s.to_str()),
                Some("target" | ".git" | "node_modules" | "dist")
            );
            if !skip {
                collect(&path, depth + 1, out);
            }
        } else if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
            out.push(path);
        }
    }
}

fn read_model(path: &Path) -> Result<ModelInfo, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let file_size = file.metadata().map_err(|e| e.to_string())?.len();
    // SAFETY: the usual mmap caveat — another process truncating the file
    // under us would fault. These are local checkpoints, and the alternative
    // (reading 548 MB to draw a list) is what this whole module exists to
    // avoid.
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
    let (_, meta) =
        safetensors::SafeTensors::read_metadata(&mmap).map_err(|e| format!("header: {e}"))?;

    let map: HashMap<String, &safetensors::tensor::TensorInfo> = meta.tensors();
    let mut tensors: Vec<TensorEntry> = map
        .iter()
        .map(|(name, info)| TensorEntry {
            name: name.clone(),
            shape: info.shape.clone(),
            dtype: format!("{:?}", info.dtype),
        })
        .collect();
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    let params: usize = tensors
        .iter()
        .map(|t| t.shape.iter().product::<usize>())
        .sum();

    let dir = path.parent().unwrap_or(Path::new("."));
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    // Sidecars are either bare (`config.json`, the layout of models/gpt2 and
    // the shipped assets/) or stem-prefixed (`<stem>.config.json`, what
    // train_shakespeare.rs writes beside a checkpoint directory holding
    // several models).
    let sidecar = |suffix: &str| -> Option<PathBuf> {
        [dir.join(format!("{stem}.{suffix}")), dir.join(suffix)]
            .into_iter()
            .find(|cand| cand.exists())
    };
    let config_path = sidecar("config.json");
    let vocab_path = sidecar("vocab.json");
    let merges_path = sidecar("merges.txt");
    let config = config_path
        .as_ref()
        .and_then(|p| Gpt2Config::from_json(p).ok());

    let tokenizer = match (&vocab_path, &merges_path) {
        (Some(_), Some(_)) => TokenizerKind::Bpe,
        (Some(_), None) => TokenizerKind::Char,
        _ => TokenizerKind::None,
    };
    // Report the *first* blocking reason, so the status line can name it.
    let blocked = if config.is_none() {
        Some(match &config_path {
            Some(p) => format!("{} did not parse as a GPT-2 config", p.display()),
            None => "no config.json beside the weights".to_string(),
        })
    } else if tokenizer == TokenizerKind::None {
        Some("no vocab.json beside the weights".to_string())
    } else {
        None
    };

    let name = if stem == "model" {
        dir.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(stem)
            .to_string()
    } else {
        stem.to_string()
    };

    Ok(ModelInfo {
        path: path.to_path_buf(),
        name,
        file_size,
        params,
        tensors,
        config,
        config_path,
        vocab_path,
        merges_path,
        tokenizer,
        blocked,
    })
}
