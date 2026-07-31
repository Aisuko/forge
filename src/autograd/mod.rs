//! Tape-based reverse-mode autograd (roadmap v4, Stage 8).
//!
//! Define-by-run: a training forward records one `Node` per op on the
//! [`Tape`]; [`Tape::backward`] sweeps the nodes in reverse, composing each
//! backward pass out of the already-verified forward ops (matmul with
//! trans_a/trans_b, add, sum_rows) plus the dedicated backward kernels
//! (gelu_bwd, layernorm_bwd, softmax_bwd, scatter_add). One implementation
//! serves both backends because every rule goes through `ops`.

use crate::error::{ForgeError, Result};
use crate::ops::{self, MatmulSpec};
use crate::tensor::Tensor;

/// A tensor tracked on the tape.
#[derive(Clone)]
pub struct TVar {
    pub id: usize,
    pub t: Tensor,
}

type BackwardFn = Box<dyn Fn(&Tensor) -> Result<Vec<(usize, Tensor)>>>;

struct Node {
    /// None for leaves (parameters, inputs).
    backward: Option<BackwardFn>,
}

#[derive(Default)]
pub struct Tape {
    nodes: Vec<Node>,
}

impl Tape {
    pub fn new() -> Tape {
        Tape::default()
    }

    /// Register a leaf (parameter or input); its gradient is what
    /// `backward` reports.
    pub fn leaf(&mut self, t: Tensor) -> TVar {
        let id = self.nodes.len();
        self.nodes.push(Node { backward: None });
        TVar { id, t }
    }

    fn record(&mut self, t: Tensor, f: BackwardFn) -> TVar {
        let id = self.nodes.len();
        self.nodes.push(Node { backward: Some(f) });
        TVar { id, t }
    }

    pub fn add(&mut self, a: &TVar, b: &TVar) -> Result<TVar> {
        let out = ops::add(&a.t, &b.t)?;
        let (ia, ib) = (a.id, b.id);
        Ok(self.record(
            out,
            Box::new(move |dy| Ok(vec![(ia, dy.clone()), (ib, dy.clone())])),
        ))
    }

    /// Matmul with the specs the training forward uses: `trans_a == false`,
    /// `b_rows == None`, operands of equal rank (2x2 or batched 3x3).
    pub fn matmul(
        &mut self,
        a: &TVar,
        b: &TVar,
        bias: Option<&TVar>,
        spec: MatmulSpec,
    ) -> Result<TVar> {
        if spec.trans_a || spec.b_rows.is_some() {
            return Err(ForgeError::Shape(
                "tape matmul: trans_a/b_rows are backward-only specs".into(),
            ));
        }
        if a.t.shape().rank() != b.t.shape().rank() {
            return Err(ForgeError::Shape(
                "tape matmul requires equal-rank operands (no broadcast grads)".into(),
            ));
        }
        let out = ops::matmul(&a.t, &b.t, bias.map(|b| &b.t), spec)?;
        let (ia, ib) = (a.id, b.id);
        let ibias = bias.map(|b| b.id);
        let (at, bt) = (a.t.clone(), b.t.clone());
        let alpha = spec.alpha;
        let trans_b = spec.trans_b;
        Ok(self.record(
            out,
            Box::new(move |dy| {
                let da = if !trans_b {
                    // Y = aAB: dA = a dY Bt
                    ops::matmul(
                        dy,
                        &bt,
                        None,
                        MatmulSpec {
                            trans_b: true,
                            alpha,
                            ..Default::default()
                        },
                    )?
                } else {
                    // Y = aABt: dA = a dY B
                    ops::matmul(
                        dy,
                        &bt,
                        None,
                        MatmulSpec {
                            alpha,
                            ..Default::default()
                        },
                    )?
                };
                let db = if !trans_b {
                    // dB = a At dY
                    ops::matmul(
                        &at,
                        dy,
                        None,
                        MatmulSpec {
                            trans_a: true,
                            alpha,
                            ..Default::default()
                        },
                    )?
                } else {
                    // dB = a dYt A
                    ops::matmul(
                        dy,
                        &at,
                        None,
                        MatmulSpec {
                            trans_a: true,
                            alpha,
                            ..Default::default()
                        },
                    )?
                };
                let mut grads = vec![(ia, da), (ib, db)];
                if let Some(ibias) = ibias {
                    grads.push((ibias, ops::sum_rows(dy)?));
                }
                Ok(grads)
            }),
        ))
    }

    pub fn gelu(&mut self, x: &TVar) -> Result<TVar> {
        let out = ops::gelu(&x.t)?;
        let ix = x.id;
        let xt = x.t.clone();
        Ok(self.record(
            out,
            Box::new(move |dy| Ok(vec![(ix, ops::gelu_bwd(&xt, dy)?)])),
        ))
    }

    pub fn layernorm(&mut self, x: &TVar, gamma: &TVar, beta: &TVar, eps: f32) -> Result<TVar> {
        let out = ops::layernorm(&x.t, &gamma.t, &beta.t, eps)?;
        let (ix, ig, ib) = (x.id, gamma.id, beta.id);
        let (xt, gt) = (x.t.clone(), gamma.t.clone());
        Ok(self.record(
            out,
            Box::new(move |dy| {
                let (dx, dgamma, dbeta) = ops::layernorm_bwd(&xt, &gt, dy, eps)?;
                Ok(vec![(ix, dx), (ig, dgamma), (ib, dbeta)])
            }),
        ))
    }

    /// Softmax over the last dim (optionally causal). Saves the output for
    /// backward; the causal mask needs no bookkeeping there because masked
    /// outputs are exact zeros.
    pub fn softmax(&mut self, x: &TVar, causal: bool, off: usize) -> Result<TVar> {
        let out = ops::softmax(&x.t, causal, off)?;
        let ix = x.id;
        let yt = out.clone();
        Ok(self.record(
            out,
            Box::new(move |dy| Ok(vec![(ix, ops::softmax_bwd(&yt, dy)?)])),
        ))
    }

    pub fn split_heads(&mut self, qkv: &TVar, n_head: usize) -> Result<(TVar, TVar, TVar)> {
        let (q, k, v) = ops::split_heads(&qkv.t, n_head)?;
        let iq = qkv.id;
        let mk = move |which: usize| -> BackwardFn {
            Box::new(move |dy| Ok(vec![(iq, ops::unsplit_head(dy, which)?)]))
        };
        Ok((
            self.record(q, mk(0)),
            self.record(k, mk(1)),
            self.record(v, mk(2)),
        ))
    }

    pub fn merge_heads(&mut self, x: &TVar) -> Result<TVar> {
        let h = x.t.shape().dim(0);
        let out = ops::merge_heads(&x.t)?;
        let ix = x.id;
        Ok(self.record(
            out,
            Box::new(move |dy| Ok(vec![(ix, ops::unmerge_heads(dy, h)?)])),
        ))
    }

    /// Fused token + positional embedding over a single-chunk wte.
    /// dwte comes from scatter-add over the token ids; dwpe is the upstream
    /// gradient placed at rows `pos..pos+t`.
    pub fn embedding(&mut self, ids: &Tensor, wte: &TVar, wpe: &TVar, pos: usize) -> Result<TVar> {
        let out = ops::embedding(ids, &wte.t, Some(&wpe.t), pos)?;
        let (iw, ip) = (wte.id, wpe.id);
        let ids = ids.clone();
        let wte_shape = wte.t.shape().clone();
        let (n_ctx, c) = (wpe.t.shape().dim(0), wpe.t.shape().dim(1));
        Ok(self.record(
            out,
            Box::new(move |dy| {
                let device = dy.device();
                let mut dwte = Tensor::zeros(wte_shape.clone(), &device)?;
                ops::scatter_add_rows(&mut dwte, &ids, dy)?;
                // dwpe: place dy rows at pos.. via kv_append on a rank-3
                // view created *before* the write (CPU copy-on-write).
                let t = ids.shape().numel();
                let mut dwpe3 = Tensor::zeros([1, n_ctx, c], &device)?;
                ops::kv_append(&mut dwpe3, &dy.reshape([1, t, c])?, pos)?;
                Ok(vec![(iw, dwte), (ip, dwpe3.reshape([n_ctx, c])?)])
            }),
        ))
    }

    /// Inverted dropout; the deterministic counter RNG regenerates the same
    /// mask in backward, so nothing is saved.
    pub fn dropout(&mut self, x: &TVar, p: f32, seed: u32) -> Result<TVar> {
        if p == 0.0 {
            return Ok(x.clone());
        }
        let out = ops::dropout(&x.t, p, seed)?;
        let ix = x.id;
        Ok(self.record(
            out,
            Box::new(move |dy| Ok(vec![(ix, ops::dropout(dy, p, seed)?)])),
        ))
    }

    /// Reverse sweep from `root` seeded with `seed_grad` (dL/droot).
    /// Returns per-node gradients; leaves keep theirs, interior gradients
    /// are freed as soon as they have been consumed.
    pub fn backward(&mut self, root: &TVar, seed_grad: Tensor) -> Result<Vec<Option<Tensor>>> {
        let mut grads: Vec<Option<Tensor>> = (0..self.nodes.len()).map(|_| None).collect();
        grads[root.id] = Some(seed_grad);
        for id in (0..=root.id).rev() {
            if grads[id].is_none() {
                continue;
            }
            let Some(f) = self.nodes[id].backward.take() else {
                continue; // leaf: keep the gradient
            };
            let dy = grads[id].take().expect("checked above");
            for (pid, g) in f(&dy)? {
                debug_assert!(pid < id, "backward edge must point to an earlier node");
                grads[pid] = Some(match grads[pid].take() {
                    Some(acc) => ops::add(&acc, &g)?,
                    None => g,
                });
            }
        }
        Ok(grads)
    }
}
