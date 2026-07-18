use std::fmt;

/// Unified error type for all Forge operations.
#[derive(Debug)]
pub enum ForgeError {
    Io(std::io::Error),
    Json(serde_json::Error),
    SafeTensors(String),
    /// Shape or dtype mismatch in a tensor operation.
    Shape(String),
    /// WebGPU runtime failure (no adapter, device request, mapping, ...).
    Wgpu(String),
    Tokenizer(String),
    /// A tensor expected on one device was found on another.
    Device(String),
}

impl fmt::Display for ForgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ForgeError::Io(e) => write!(f, "io error: {e}"),
            ForgeError::Json(e) => write!(f, "json error: {e}"),
            ForgeError::SafeTensors(m) => write!(f, "safetensors error: {m}"),
            ForgeError::Shape(m) => write!(f, "shape error: {m}"),
            ForgeError::Wgpu(m) => write!(f, "wgpu error: {m}"),
            ForgeError::Tokenizer(m) => write!(f, "tokenizer error: {m}"),
            ForgeError::Device(m) => write!(f, "device error: {m}"),
        }
    }
}

impl std::error::Error for ForgeError {}

impl From<std::io::Error> for ForgeError {
    fn from(e: std::io::Error) -> Self {
        ForgeError::Io(e)
    }
}

impl From<serde_json::Error> for ForgeError {
    fn from(e: serde_json::Error) -> Self {
        ForgeError::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, ForgeError>;
