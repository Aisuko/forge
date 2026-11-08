use crate::error::Result;
use crate::nn::Linear;
use crate::ops::{self, MatmulSpec};
use crate::tensor::Tensor;

fn scores(q: &Tensor, k: &Tensor, head_dim: usize, b_rows: Option<usize>) -> Result<Tensor> {
    ops::matmul(
        q,
        k,
        None,
        MatmulSpec {
            trans_b: true,
            alpha: 1.0 / (head_dim as f32).sqrt(),
            b_rows,
            ..Default::default()
        },
    )
}

fn mix(probs: &Tensor, v: &Tensor, b_rows: Option<usize>) -> Result<Tensor> {
    ops::matmul(
        probs,
        v,
        None,
        MatmulSpec {
            b_rows,
            ..Default::default()
        },
    )
}

pub(crate) struct SelfAttention {
    pub qkv: Linear,
    pub proj: Linear,
}

impl SelfAttention {
    pub fn forward_masked(
        &self,
        x: &Tensor,
        n_head: usize,
        batch: usize,
        key_len: &Tensor,
    ) -> Result<Tensor> {
        let qkv = self.qkv.forward(x)?;
        let (q, k, v) = ops::split_heads_batched(&qkv, n_head, batch)?;
        let hd = q.shape().dim(2);
        let probs = ops::softmax_masked(&scores(&q, &k, hd, None)?, key_len, batch)?;
        let y = ops::merge_heads_batched(&mix(&probs, &v, None)?, batch)?;
        self.proj.forward(&y)
    }

    pub fn forward_cached(
        &self,
        x: &Tensor,
        n_head: usize,
        k_cache: &mut Tensor,
        v_cache: &mut Tensor,
        len: usize,
    ) -> Result<Tensor> {
        let qkv = self.qkv.forward(x)?;
        let (q, k, v) = ops::split_heads(&qkv, n_head)?;
        let (t, hd) = (q.shape().dim(1), q.shape().dim(2));
        ops::kv_append(k_cache, &k, len)?;
        ops::kv_append(v_cache, &v, len)?;
        let kv_len = Some(len + t);
        let probs = ops::softmax(&scores(&q, k_cache, hd, kv_len)?, true, len)?;
        let y = ops::merge_heads(&mix(&probs, v_cache, kv_len)?)?;
        self.proj.forward(&y)
    }
}

pub(crate) struct CrossAttention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub proj: Linear,
}

impl CrossAttention {
    pub fn memory(&self, memory: &Tensor, n_head: usize) -> Result<(Tensor, Tensor)> {
        let k = ops::split_heads_single(&self.k.forward(memory)?, n_head, 1)?;
        let v = ops::split_heads_single(&self.v.forward(memory)?, n_head, 1)?;
        Ok((k, v))
    }

    pub fn forward(
        &self,
        x: &Tensor,
        n_head: usize,
        k: &Tensor,
        v: &Tensor,
        key_len: &Tensor,
    ) -> Result<Tensor> {
        let q = ops::split_heads_single(&self.q.forward(x)?, n_head, 1)?;
        let hd = q.shape().dim(2);
        let probs = ops::softmax_masked(&scores(&q, k, hd, None)?, key_len, 1)?;
        let y = ops::merge_heads(&mix(&probs, v, None)?)?;
        self.proj.forward(&y)
    }
}
