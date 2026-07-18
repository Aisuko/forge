//! Minimal neural-network modules — exactly what GPT-2 needs.

use crate::error::Result;
use crate::ops::{self, MatmulSpec};
use crate::tensor::Tensor;

/// y = x @ W + b, with W stored [in_features, out_features].
///
/// This matches the HF GPT-2 "Conv1D" convention, so checkpoint weights load
/// without transposition (see roadmap "Known Pitfalls").
pub struct Linear {
    pub w: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::matmul(x, &self.w, self.b.as_ref(), MatmulSpec::default())
    }
}

pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        ops::layernorm(x, &self.gamma, &self.beta, self.eps)
    }
}

/// Token + positional embedding table. The token table is stored row-chunked
/// so no single GPU binding exceeds the device limit (see roadmap pitfalls);
/// on CPU it is a single chunk.
pub struct Embedding {
    /// Row chunks of the [vocab, n_embd] token table.
    pub wte_chunks: Vec<Tensor>,
    /// Rows per chunk (the last chunk may be smaller).
    pub chunk_rows: usize,
    /// [n_ctx, n_embd]
    pub wpe: Tensor,
}

impl Embedding {
    /// Split a host-side [vocab, n_embd] table into device chunks that each
    /// fit in a single binding on `device`.
    pub fn from_host_wte(
        wte: &[f32],
        vocab: usize,
        n_embd: usize,
        wpe: Tensor,
        device: &crate::device::Device,
    ) -> Result<Self> {
        let row_bytes = n_embd * 4;
        let max_rows = (device.max_binding_bytes() / row_bytes).min(vocab);
        let chunk_rows = max_rows.max(1);
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < vocab {
            let rows = chunk_rows.min(vocab - start);
            chunks.push(Tensor::from_f32(
                &wte[start * n_embd..(start + rows) * n_embd],
                [rows, n_embd],
                device,
            )?);
            start += rows;
        }
        Ok(Embedding {
            wte_chunks: chunks,
            chunk_rows,
            wpe,
        })
    }

    /// ids: u32 tensor [t]; `pos` is the absolute position of ids[0].
    pub fn forward(&self, ids: &Tensor, pos: usize) -> Result<Tensor> {
        ops::embedding_chunked(ids, &self.wte_chunks, self.chunk_rows, Some(&self.wpe), pos)
    }
}
