//! Browser bindings for the council — a wasm-bindgen facade over
//! [`Council`](crate::Council), driven by `web/council.js`.

use forge::{AnyTokenizer, CharTokenizer, Device, Gpt2, Gpt2Config, Sampling, Tokenizer as _};
use wasm_bindgen::prelude::*;

use crate::{Council, CouncilStep};

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

fn js_err(e: forge::ForgeError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// One council step as a plain JS object.
///
/// The per-expert `hidden` arrays are the whole point of the page — the
/// vectors the experts actually exchanged — so they cross as `Float32Array`s,
/// never through JSON.
fn council_step_to_js(s: &CouncilStep, names: &[String], tokenizer: &AnyTokenizer) -> JsValue {
    let top_to_js = |top: &[(u32, f32)]| -> JsValue {
        let a = js_sys::Array::new_with_length(top.len() as u32);
        for (i, (id, p)) in top.iter().enumerate() {
            let e = js_sys::Object::new();
            set(&e, "id", &num(*id as usize));
            set(&e, "token", &JsValue::from_str(&tokenizer.decode(&[*id])));
            set(&e, "p", &JsValue::from_f64(*p as f64));
            a.set(i as u32, e.into());
        }
        a.into()
    };

    let experts = js_sys::Array::new_with_length(s.experts.len() as u32);
    for (i, e) in s.experts.iter().enumerate() {
        let o = js_sys::Object::new();
        set(
            &o,
            "name",
            &JsValue::from_str(names.get(i).map_or("", |n| n)),
        );
        set(&o, "weight", &JsValue::from_f64(e.weight as f64));
        set(&o, "entropy", &JsValue::from_f64(e.entropy as f64));
        set(&o, "top", &top_to_js(&e.top));
        set(&o, "hidden", &f32a(&e.hidden));
        experts.set(i as u32, o.into());
    }

    let o = js_sys::Object::new();
    set(&o, "experts", &experts);
    set(&o, "hidden", &f32a(&s.hidden));
    set(&o, "top", &top_to_js(&s.top));
    set(&o, "id", &num(s.chosen as usize));
    set(
        &o,
        "token",
        &JsValue::from_str(&tokenizer.decode(&[s.chosen])),
    );
    set(&o, "consensus", &JsValue::from_f64(s.consensus as f64));
    o.into()
}

/// Several small GPT-2s on one WebGPU device, run in parallel and merged in
/// their own hidden space — the model behind the council page.
///
/// Every expert shares one `Device`: four adapters would be four copies of
/// every shader and, on some browsers, four times the device limit.
#[wasm_bindgen]
pub struct WasmCouncil {
    council: Council,
    names: Vec<String>,
    tokenizer: AnyTokenizer,
    device: Device,
}

#[wasm_bindgen]
impl WasmCouncil {
    /// `experts` is an array of `Uint8Array` safetensors blobs and `names` the
    /// matching array of display strings. All experts share one `config.json`
    /// and one `vocab.json` — if they needed their own, they would not be
    /// mergeable in the first place.
    pub async fn load(
        experts: js_sys::Array,
        names: js_sys::Array,
        config_json: &str,
        vocab_json: &str,
        seed: u64,
    ) -> Result<WasmCouncil, JsValue> {
        let device = Device::wgpu_async().await.map_err(js_err)?;
        let config = Gpt2Config::from_json_str(config_json).map_err(js_err)?;
        let tokenizer = AnyTokenizer::Char(CharTokenizer::from_json(vocab_json).map_err(js_err)?);

        let mut models = Vec::with_capacity(experts.length() as usize);
        for v in experts.iter() {
            let bytes = js_sys::Uint8Array::new(&v).to_vec();
            models.push(
                Gpt2::from_safetensors_bytes(&bytes, config.clone(), &device).map_err(js_err)?,
            );
        }
        let names: Vec<String> = names.iter().filter_map(|v| v.as_string()).collect();
        let council = Council::new(models, names.clone(), seed).map_err(js_err)?;
        Ok(WasmCouncil {
            council,
            names,
            tokenizer,
            device,
        })
    }

    pub fn device_info(&self) -> String {
        self.device.describe()
    }

    pub fn n_experts(&self) -> usize {
        self.council.n_experts()
    }

    pub fn n_embd(&self) -> usize {
        self.council.n_embd()
    }

    pub fn n_ctx(&self) -> usize {
        self.council.n_ctx()
    }

    pub fn vocab_size(&self) -> usize {
        self.council.vocab_size()
    }

    pub fn expert_names(&self) -> js_sys::Array {
        self.names.iter().map(|n| JsValue::from_str(n)).collect()
    }

    /// Router sharpness. 0 weights every expert equally; larger lets the most
    /// confident expert dominate. The page drives this from a slider.
    pub fn set_beta(&mut self, beta: f32) {
        self.council.beta = beta;
    }

    pub fn beta(&self) -> f32 {
        self.council.beta
    }

    /// Characters of `prompt` this vocabulary cannot represent, as one string.
    /// Empty means the prompt is safe to run.
    pub fn unsupported_chars(&self, prompt: &str) -> String {
        match &self.tokenizer {
            AnyTokenizer::Char(c) => c.unknown_chars(prompt).into_iter().collect(),
            AnyTokenizer::Bpe(_) => String::new(),
        }
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, JsValue> {
        self.tokenizer.encode(prompt).map_err(js_err)
    }

    pub fn decode_ids(&self, ids: &[u32]) -> js_sys::Array {
        ids.iter()
            .map(|id| JsValue::from_str(&self.tokenizer.decode(&[*id])))
            .collect()
    }

    /// Forget everything: fresh KV caches for every expert, fresh sampling.
    pub fn reset(&mut self, seed: u64) -> Result<(), JsValue> {
        self.council.reset(seed).map_err(js_err)
    }

    /// One character. Pass the whole prompt on the first call after `reset`,
    /// then one id per call — the same contract as `generate_with_trace`.
    pub async fn step(
        &mut self,
        ids: &[u32],
        top_n: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<JsValue, JsValue> {
        let sampling = if temperature <= 0.0 {
            Sampling::Greedy
        } else {
            Sampling::TopK {
                k: 40,
                temperature,
                seed,
            }
        };
        let step = self
            .council
            .step_async(ids, sampling, top_n)
            .await
            .map_err(js_err)?;
        Ok(council_step_to_js(&step, &self.names, &self.tokenizer))
    }
}
