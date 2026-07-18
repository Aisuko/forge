/* tslint:disable */
/* eslint-disable */

/**
 * A GPT-2 model + tokenizer on a WebGPU device, driven from JavaScript.
 */
export class WasmGpt2 {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Human-readable adapter description (shown by the demo page).
     */
    device_info(): string;
    /**
     * Generate with KV-cache decode, streaming each newly decoded text
     * fragment to `on_text(fragment)`. Greedy when `top_k` is 0, otherwise
     * top-k sampling at `temperature` with `seed`.
     */
    generate(prompt: string, max_new_tokens: number, top_k: number, temperature: number, seed: bigint, on_text: Function): Promise<string>;
    /**
     * Greedy continuation as raw token ids — used by the Stage 11 gate to
     * compare browser output against native WGPU token-for-token.
     */
    greedy_ids(prompt: string, max_new_tokens: number): Promise<Uint32Array>;
    /**
     * Build from fetched assets: `model.safetensors` bytes, `config.json`,
     * `vocab.json`, and `merges.txt` contents.
     */
    static load(model_bytes: Uint8Array, config_json: string, vocab_json: string, merges: string): Promise<WasmGpt2>;
}

export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmgpt2_free: (a: number, b: number) => void;
    readonly wasmgpt2_device_info: (a: number) => [number, number];
    readonly wasmgpt2_generate: (a: number, b: number, c: number, d: number, e: number, f: number, g: bigint, h: any) => any;
    readonly wasmgpt2_greedy_ids: (a: number, b: number, c: number, d: number) => any;
    readonly wasmgpt2_load: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => any;
    readonly start: () => void;
    readonly wasm_bindgen_37dbbe0f84118608___convert__closures_____invoke___wasm_bindgen_37dbbe0f84118608___JsValue__core_9b3796e30d99ddb7___result__Result_____wasm_bindgen_37dbbe0f84118608___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_37dbbe0f84118608___convert__closures_____invoke___js_sys_6044d517200822db___Function_fn_wasm_bindgen_37dbbe0f84118608___JsValue_____wasm_bindgen_37dbbe0f84118608___sys__Undefined___js_sys_6044d517200822db___Function_fn_wasm_bindgen_37dbbe0f84118608___JsValue_____wasm_bindgen_37dbbe0f84118608___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_37dbbe0f84118608___convert__closures_____invoke___wasm_bindgen_37dbbe0f84118608___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
