//! Browser bindings (Stage 11, roadmap v4): a wasm-bindgen facade over the
//! async inference API. Inference only — training is out of browser scope
//! for 1.0.

use wasm_bindgen::prelude::*;

use crate::models::gpt2::{Gpt2, Gpt2Config, Sampling};
use crate::tokenizer::Gpt2Tokenizer;
use crate::Device;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// A GPT-2 model + tokenizer on a WebGPU device, driven from JavaScript.
#[wasm_bindgen]
pub struct WasmGpt2 {
    model: Gpt2,
    tokenizer: Gpt2Tokenizer,
    device: Device,
}

fn js_err(e: crate::ForgeError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

#[wasm_bindgen]
impl WasmGpt2 {
    /// Build from fetched assets: `model.safetensors` bytes, `config.json`,
    /// `vocab.json`, and `merges.txt` contents.
    pub async fn load(
        model_bytes: Vec<u8>,
        config_json: &str,
        vocab_json: &str,
        merges: &str,
    ) -> Result<WasmGpt2, JsValue> {
        let device = Device::wgpu_async().await.map_err(js_err)?;
        let config = Gpt2Config::from_json_str(config_json).map_err(js_err)?;
        let model = Gpt2::from_safetensors_bytes(&model_bytes, config, &device).map_err(js_err)?;
        let tokenizer = Gpt2Tokenizer::from_strs(vocab_json, merges).map_err(js_err)?;
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

    /// Generate with KV-cache decode, streaming each newly decoded text
    /// fragment to `on_text(fragment)`. Greedy when `top_k` is 0, otherwise
    /// top-k sampling at `temperature` with `seed`.
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
            .generate_async(&self.tokenizer, prompt, max_new_tokens, sampling, |s| {
                let _ = on_text.call1(&this, &JsValue::from_str(s));
            })
            .await
            .map_err(js_err)
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
