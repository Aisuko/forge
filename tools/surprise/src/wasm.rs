//! Browser bindings for the surprise page — a wasm-bindgen facade over
//! [`surprisal`](crate::surprisal), driven by `web/react.js`.
//!
//! Deliberately not the whole of `WasmGpt2`: this exposes the six things the
//! page calls and nothing else. `#[wasm_bindgen]` exports are GC roots the
//! linker cannot eliminate, so every method here is weight in the bundle
//! whether or not it is reachable from the page.

use forge::{AnyTokenizer, CharTokenizer, Device, Gpt2, Gpt2Config, Tokenizer as _};
use wasm_bindgen::prelude::*;

/// Set a property on a freshly built object. `Reflect::set` can only fail on an
/// exotic target (a frozen object, a proxy that throws); these are plain
/// `Object::new()` results, so the Result carries no information.
fn set(o: &js_sys::Object, key: &str, v: &JsValue) {
    let _ = js_sys::Reflect::set(o, &JsValue::from_str(key), v);
}

fn f32a(v: &[f32]) -> JsValue {
    js_sys::Float32Array::from(v).into()
}

fn js_err(e: forge::ForgeError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// A character-level GPT-2 on a WebGPU device, scored rather than sampled —
/// the model behind the surprise page.
#[wasm_bindgen]
pub struct WasmSurprise {
    model: Gpt2,
    tokenizer: AnyTokenizer,
    device: Device,
}

#[wasm_bindgen]
impl WasmSurprise {
    /// Build a character-level model from fetched assets: checkpoint bytes
    /// (`.safetensors` or `.fzm`, detected automatically), `config.json` and
    /// `vocab.json`. There is no BPE constructor: this page ships one model.
    pub async fn load_char(
        model_bytes: Vec<u8>,
        config_json: &str,
        vocab_json: &str,
    ) -> Result<WasmSurprise, JsValue> {
        let device = Device::wgpu_async().await.map_err(js_err)?;
        let config = Gpt2Config::from_json_str(config_json).map_err(js_err)?;
        let tokenizer = AnyTokenizer::Char(CharTokenizer::from_json(vocab_json).map_err(js_err)?);
        let model = Gpt2::from_checkpoint_bytes(&model_bytes, config, &device).map_err(js_err)?;
        Ok(WasmSurprise {
            model,
            tokenizer,
            device,
        })
    }

    /// Human-readable adapter description, shown in the page header.
    pub fn device_info(&self) -> String {
        self.device.describe()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.vocab_size()
    }

    pub fn n_layer(&self) -> usize {
        self.model.config.n_layer
    }

    /// Characters of `text` this model's vocabulary cannot represent,
    /// deduplicated. Empty when the text is fine. The char model knows only 65
    /// characters, so the page checks before scoring rather than surfacing an
    /// encode error mid-hover.
    pub fn unsupported_chars(&self, text: &str) -> String {
        match &self.tokenizer {
            AnyTokenizer::Char(c) => c.unknown_chars(text).into_iter().collect(),
            // Byte-level BPE encodes any UTF-8 input by construction.
            AnyTokenizer::Bpe(_) => String::new(),
        }
    }

    /// Score `text` for surprise: one forward pass, every position at once,
    /// keeping `k` alternatives per position.
    ///
    /// Returns `{ tokens, bits, top, topP, alt, altP, k }`. The first four are
    /// one entry per position:
    ///
    /// - `tokens[i]` — the decoded text of position `i`, so the page can lay
    ///   out exactly the characters the model saw rather than re-splitting the
    ///   string itself and hoping the two agree.
    /// - `bits[i]` — `-log2 p` of that token given everything before it.
    ///   `bits[0]` is 0.
    /// - `top[i]` / `topP[i]` — what the model expected instead, and how sure
    ///   it was. Read off column 0 of the arrays below, never stored twice.
    ///
    /// `alt` and `altP` are flat and `k` wide: position `i` occupies
    /// `[i*k, (i+1)*k)`, descending by probability. These are the characters
    /// the model actually weighed at that position, which is what the page
    /// flickers through while a character is still resolving — the alternative
    /// being random draws from the alphabet, which would be a lie about the
    /// model.
    ///
    /// A *scoring* call, not a generation call: nothing is sampled, and the
    /// cost is one pass over the text regardless of its length.
    ///
    /// Errors if the text contains characters the tokenizer has never seen —
    /// call [`WasmSurprise::unsupported_chars`] first and tell the reader,
    /// which is friendlier than a thrown exception mid-hover.
    pub async fn surprisal(&self, text: &str, k: usize) -> Result<JsValue, JsValue> {
        let ids = self.tokenizer.encode(text).map_err(js_err)?;
        if ids.is_empty() {
            return Err(JsValue::from_str("surprisal needs non-empty text"));
        }
        let s = crate::surprisal(&self.model, &ids, k)
            .await
            .map_err(js_err)?;

        let tokens = js_sys::Array::new_with_length(ids.len() as u32);
        let top = js_sys::Array::new_with_length(ids.len() as u32);
        for (i, id) in ids.iter().enumerate() {
            tokens.set(i as u32, JsValue::from_str(&self.tokenizer.decode(&[*id])));
            top.set(
                i as u32,
                JsValue::from_str(&self.tokenizer.decode(&[s.top(i)])),
            );
        }
        // Decoded here, not in JS: the tokenizer is Rust's, and the page has no
        // way to turn an id back into text on its own.
        let alt = js_sys::Array::new_with_length(s.alt_ids.len() as u32);
        for (i, id) in s.alt_ids.iter().enumerate() {
            alt.set(i as u32, JsValue::from_str(&self.tokenizer.decode(&[*id])));
        }

        let o = js_sys::Object::new();
        set(&o, "tokens", &tokens.into());
        set(&o, "top", &top.into());
        set(&o, "bits", &f32a(&s.bits));
        set(
            &o,
            "topP",
            &f32a(&(0..s.len()).map(|i| s.top_p(i)).collect::<Vec<_>>()),
        );
        set(&o, "alt", &alt.into());
        set(&o, "altP", &f32a(&s.alt_p));
        set(&o, "k", &JsValue::from_f64(s.k as f64));
        Ok(o.into())
    }
}
