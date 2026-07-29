//! Browser bindings (Stage 11, roadmap v4): a wasm-bindgen facade over the
//! async inference API. Inference only — training is out of browser scope
//! for 1.0.

use wasm_bindgen::prelude::*;

use crate::Device;
use crate::models::gpt2::{AttnStep, Gpt2, Gpt2Config, Sampling, StepTrace};
use crate::tokenizer::{AnyTokenizer, CharTokenizer, Gpt2Tokenizer, Tokenizer as _};

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Set a property on a freshly built object. `Reflect::set` can only fail on an
/// exotic target (a frozen object, a proxy that throws); these are plain
/// `Object::new()` results, so the Result carries no information.
fn set(o: &js_sys::Object, key: &str, v: &JsValue) {
    let _ = js_sys::Reflect::set(o, &JsValue::from_str(key), v);
}

fn num(v: usize) -> JsValue {
    JsValue::from_f64(v as f64)
}

fn f32a(v: &[f32]) -> JsValue {
    js_sys::Float32Array::from(v).into()
}

/// One step's trace as a plain JS object — see `generate_with_trace`.
///
/// Hand-built with `Reflect::set` rather than through a serde bridge: the big
/// fields are `Float32Array`s that must not be copied through JSON, and the
/// handful of scalars do not justify a new crate dependency.
fn trace_to_js(t: &StepTrace, tokenizer: &AnyTokenizer) -> JsValue {
    let o = js_sys::Object::new();
    set(&o, "qLen", &num(t.q_len));
    set(&o, "kvLen", &num(t.kv_len));
    set(&o, "nHead", &num(t.n_head));
    set(&o, "nEmbd", &num(t.n_embd));
    set(&o, "headDim", &num(t.n_embd / t.n_head.max(1)));
    // With a KV cache a full q_len only happens on the prompt prefill; that is
    // the one step with a whole triangular matrix to draw.
    set(&o, "isPrefill", &JsValue::from_bool(t.q_len > 1));
    set(&o, "embedding", &f32a(&t.embedding));
    set(&o, "lnFOut", &f32a(&t.ln_f_out));

    let attn = js_sys::Array::new_with_length(t.attn.len() as u32);
    for (i, a) in t.attn.iter().enumerate() {
        let e = js_sys::Object::new();
        set(&e, "layer", &num(a.layer));
        set(&e, "nHead", &num(a.n_head));
        set(&e, "qLen", &num(a.q_len));
        set(&e, "kvLen", &num(a.kv_len));
        set(&e, "probs", &f32a(&a.probs));
        attn.set(i as u32, e.into());
    }
    set(&o, "attn", &attn);

    let detail = js_sys::Array::new_with_length(t.detail.len() as u32);
    for (i, d) in t.detail.iter().enumerate() {
        let e = js_sys::Object::new();
        set(&e, "layer", &num(d.layer));
        set(&e, "ln1Out", &f32a(&d.ln1_out));
        set(&e, "q", &f32a(&d.q));
        set(&e, "k", &f32a(&d.k));
        set(&e, "v", &f32a(&d.v));
        set(&e, "scores", &f32a(&d.scores));
        set(&e, "attnHeadOut", &f32a(&d.attn_head_out));
        set(&e, "attnProjOut", &f32a(&d.attn_proj_out));
        set(&e, "residAttn", &f32a(&d.resid_attn));
        set(&e, "ln2Out", &f32a(&d.ln2_out));
        set(&e, "mlpHidden", &f32a(&d.mlp_hidden));
        set(&e, "blockOut", &f32a(&d.block_out));
        detail.set(i as u32, e.into());
    }
    set(&o, "detail", &detail);

    // Decoded here rather than in JS: the tokenizer is Rust's, and the page
    // has no way to turn an id back into text on its own.
    let top = js_sys::Array::new_with_length(t.top.len() as u32);
    for (i, (id, p)) in t.top.iter().enumerate() {
        let e = js_sys::Object::new();
        set(&e, "id", &num(*id as usize));
        set(&e, "token", &JsValue::from_str(&tokenizer.decode(&[*id])));
        set(&e, "p", &JsValue::from_f64(*p as f64));
        top.set(i as u32, e.into());
    }
    set(&o, "top", &top);
    o.into()
}

/// A GPT-2 model + tokenizer on a WebGPU device, driven from JavaScript.
#[wasm_bindgen]
pub struct WasmGpt2 {
    model: Gpt2,
    tokenizer: AnyTokenizer,
    device: Device,
}

fn js_err(e: crate::ForgeError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
impl WasmGpt2 {
    /// Build a BPE model from fetched assets: `model.safetensors` bytes,
    /// `config.json`, `vocab.json`, and `merges.txt` contents.
    pub async fn load(
        model_bytes: Vec<u8>,
        config_json: &str,
        vocab_json: &str,
        merges: &str,
    ) -> Result<WasmGpt2, JsValue> {
        let tokenizer =
            AnyTokenizer::bpe(Gpt2Tokenizer::from_strs(vocab_json, merges).map_err(js_err)?);
        Self::build(model_bytes, config_json, tokenizer).await
    }

    /// Build a character-level model. Same assets minus `merges.txt`, which
    /// is BPE-specific and does not exist for a char vocabulary.
    pub async fn load_char(
        model_bytes: Vec<u8>,
        config_json: &str,
        vocab_json: &str,
    ) -> Result<WasmGpt2, JsValue> {
        let tokenizer = AnyTokenizer::Char(CharTokenizer::from_json(vocab_json).map_err(js_err)?);
        Self::build(model_bytes, config_json, tokenizer).await
    }

    async fn build(
        model_bytes: Vec<u8>,
        config_json: &str,
        tokenizer: AnyTokenizer,
    ) -> Result<WasmGpt2, JsValue> {
        let device = Device::wgpu_async().await.map_err(js_err)?;
        let config = Gpt2Config::from_json_str(config_json).map_err(js_err)?;
        let model = Gpt2::from_safetensors_bytes(&model_bytes, config, &device).map_err(js_err)?;
        Ok(WasmGpt2 {
            model,
            tokenizer,
            device,
        })
    }

    /// Human-readable adapter description (shown by the demo page).
    pub fn device_info(&self) -> String {
        self.device.describe()
    }

    /// `"bpe"` or `"char"`.
    pub fn tokenizer_kind(&self) -> String {
        self.tokenizer.kind().to_string()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    // The page's architecture view is sized from the model it is actually
    // running, not from GPT-2 124M constants baked into JavaScript.
    pub fn n_layer(&self) -> usize {
        self.model.config.n_layer
    }

    pub fn n_head(&self) -> usize {
        self.model.config.n_head
    }

    pub fn n_embd(&self) -> usize {
        self.model.config.n_embd
    }

    pub fn n_ctx(&self) -> usize {
        self.model.config.n_ctx
    }

    /// Characters of `prompt` this model's vocabulary cannot represent,
    /// deduplicated. Empty when the prompt is fine. The char model knows only
    /// 65 characters, so the page checks before generating rather than
    /// surfacing an encode error mid-run.
    pub fn unsupported_chars(&self, prompt: &str) -> String {
        match &self.tokenizer {
            AnyTokenizer::Char(c) => c.unknown_chars(prompt).into_iter().collect(),
            // Byte-level BPE encodes any UTF-8 input by construction.
            AnyTokenizer::Bpe(_) => String::new(),
        }
    }

    /// Generate with KV-cache decode, streaming each newly decoded text
    /// fragment to `on_text(fragment)`. Greedy when `top_k` is 0, otherwise
    /// top-k sampling at `temperature` with `seed`.
    ///
    /// Returning `false` from `on_text` stops generation after the current
    /// token — that is how a page implements a working "stop" button.
    pub async fn generate(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        top_k: usize,
        temperature: f32,
        seed: u64,
        on_text: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let sampling = if top_k == 0 {
            Sampling::Greedy
        } else {
            Sampling::TopK {
                k: top_k,
                temperature,
                seed,
            }
        };
        let this = JsValue::NULL;
        self.model
            .generate_async_ctl(&self.tokenizer, prompt, max_new_tokens, sampling, |s| {
                match on_text.call1(&this, &JsValue::from_str(s)) {
                    // Only an explicit `false` stops; `undefined` (the common
                    // case for a callback with no return) continues.
                    Ok(v) if v.is_falsy() && !v.is_undefined() && !v.is_null() => {
                        std::ops::ControlFlow::Break(())
                    }
                    _ => std::ops::ControlFlow::Continue(()),
                }
            })
            .await
            .map_err(js_err)
    }

    /// [`WasmGpt2::generate`] plus a live attention feed: after every token,
    /// `on_attn(layer, n_head, weights)` fires once per block with that
    /// block's attention probabilities as a `Float32Array` of
    /// `n_head * kv_len` — head-major, so `kv_len` is `weights.length /
    /// n_head`, and `weights[h * kv_len + p]` is how much head `h` weighted
    /// position `p` when producing this token.
    ///
    /// Only the newest query row is sent: on the prompt prefill the model
    /// attends with every prompt position at once, but the row that produced
    /// the next token is the last one.
    ///
    /// Opt-in — `generate` does no attention readback at all.
    pub async fn generate_with_attention(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        top_k: usize,
        temperature: f32,
        seed: u64,
        on_text: &js_sys::Function,
        on_attn: &js_sys::Function,
    ) -> Result<String, JsValue> {
        let sampling = if top_k == 0 {
            Sampling::Greedy
        } else {
            Sampling::TopK {
                k: top_k,
                temperature,
                seed,
            }
        };
        let this = JsValue::NULL;
        self.model
            .generate_async_probe(
                &self.tokenizer,
                prompt,
                max_new_tokens,
                sampling,
                |s| match on_text.call1(&this, &JsValue::from_str(s)) {
                    Ok(v) if v.is_falsy() && !v.is_undefined() && !v.is_null() => {
                        std::ops::ControlFlow::Break(())
                    }
                    _ => std::ops::ControlFlow::Continue(()),
                },
                Some(|steps: &[AttnStep]| {
                    for s in steps {
                        let last = (s.q_len - 1) * s.kv_len;
                        let mut row = Vec::with_capacity(s.n_head * s.kv_len);
                        for head in 0..s.n_head {
                            let base = head * s.q_len * s.kv_len + last;
                            row.extend_from_slice(&s.probs[base..base + s.kv_len]);
                        }
                        // A failing visualization callback must not abort
                        // generation; the text is the point.
                        let _ = on_attn.call3(
                            &this,
                            &JsValue::from_f64(s.layer as f64),
                            &JsValue::from_f64(s.n_head as f64),
                            &js_sys::Float32Array::from(&row[..]),
                        );
                    }
                }),
            )
            .await
            .map_err(js_err)
    }

    /// [`WasmGpt2::generate`] plus a full calculation trace: after every step
    /// — including the prompt prefill — `on_trace(trace)` fires once with a
    /// plain JS object carrying what the model actually computed.
    ///
    /// | field | type |
    /// | --- | --- |
    /// | `qLen`, `kvLen`, `nHead`, `nEmbd`, `headDim` | number |
    /// | `isPrefill` | boolean (`qLen > 1`) |
    /// | `embedding` | `Float32Array` `[qLen × nEmbd]` |
    /// | `lnFOut` | `Float32Array` `[qLen × nEmbd]`, after the final LayerNorm |
    /// | `attn` | `[{ layer, nHead, qLen, kvLen, probs }]`, every block |
    /// | `detail` | `[{ layer, ln1Out, q, k, v, scores, attnHeadOut, attnProjOut, residAttn, ln2Out, mlpHidden, blockOut }]`, `detail_layers` of them |
    /// | `top` | `[{ id, token, p }]`, ranked |
    ///
    /// `scores` is `q · kᵢ / √headDim` **before** the causal mask and the
    /// softmax; `probs` in `attn` is the same tensor after both.
    /// `attnHeadOut` is the concatenated heads before the output projection,
    /// `attnProjOut` after it.
    ///
    /// `detail_layers` is the cost knob: 0 is exactly
    /// [`WasmGpt2::generate_with_attention`]'s readback. The expensive call is
    /// the prefill, where `qLen` is the whole prompt; every decode step after
    /// it has `qLen == 1`.
    #[allow(clippy::too_many_arguments)] // a wasm-bindgen export: JS has no
    // named arguments, and an options object would only move the list into JS.
    pub async fn generate_with_trace(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        top_k: usize,
        temperature: f32,
        seed: u64,
        on_text: &js_sys::Function,
        on_trace: &js_sys::Function,
        detail_layers: usize,
        top_n: usize,
    ) -> Result<String, JsValue> {
        let sampling = if top_k == 0 {
            Sampling::Greedy
        } else {
            Sampling::TopK {
                k: top_k,
                temperature,
                seed,
            }
        };
        let this = JsValue::NULL;
        self.model
            .generate_async_trace(
                &self.tokenizer,
                prompt,
                max_new_tokens,
                sampling,
                |s| match on_text.call1(&this, &JsValue::from_str(s)) {
                    Ok(v) if v.is_falsy() && !v.is_undefined() && !v.is_null() => {
                        std::ops::ControlFlow::Break(())
                    }
                    _ => std::ops::ControlFlow::Continue(()),
                },
                Some(|t: &StepTrace| {
                    // A failing visualization callback must not abort
                    // generation; the text is the point.
                    let _ = on_trace.call1(&this, &trace_to_js(t, &self.tokenizer));
                }),
                detail_layers,
                top_n,
            )
            .await
            .map_err(js_err)
    }

    /// One string per token of `prompt`, in order — the explainer's column
    /// labels, and the only way a page can know where this model's token
    /// boundaries fall.
    pub fn tokenize_display(&self, prompt: &str) -> Result<js_sys::Array, JsValue> {
        let ids = self.tokenizer.encode(prompt).map_err(js_err)?;
        Ok(self.decode_ids(&ids))
    }

    /// Display strings for arbitrary ids — one per id, decoded individually so
    /// each stays addressable as a column or a bar label.
    pub fn decode_ids(&self, ids: &[u32]) -> js_sys::Array {
        ids.iter()
            .map(|id| JsValue::from_str(&self.tokenizer.decode(&[*id])))
            .collect()
    }

    /// Greedy continuation as raw token ids — used by the Stage 11 gate to
    /// compare browser output against native WGPU token-for-token.
    pub async fn greedy_ids(
        &self,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, JsValue> {
        let mut ids = self.tokenizer.encode(prompt).map_err(js_err)?;
        let mut cache = self.model.new_cache().map_err(js_err)?;
        let mut logits = self
            .model
            .logits_step_async(&ids, &mut cache)
            .await
            .map_err(js_err)?;
        for _ in 0..max_new_tokens {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);
            ids.push(next);
            if ids.len() >= self.model.config.n_ctx {
                break;
            }
            logits = self
                .model
                .logits_step_async(&[next], &mut cache)
                .await
                .map_err(js_err)?;
        }
        Ok(ids)
    }
}
