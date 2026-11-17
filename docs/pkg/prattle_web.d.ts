/* tslint:disable */
/* eslint-disable */
/**
 * The paired statistic over `(prompt, native ms, bridge ms)` triples, as
 * `[speedup, ci low, ci high, median ratio, prompts, pairs]`, or empty with no pairs.
 */
export function summarize(prompts: Uint32Array, native_ms: Float64Array, bridge_ms: Float64Array): Float64Array;
export function first_arm(prompt: number, rep: number, arms: number): number;
export function prompt_hash(text: string): string;
export class Session {
  private constructor();
  free(): void;
  /**
   * A session from a model's config.json, tokenizer.json and model.safetensors.
   */
  static from_bytes(config: Uint8Array, tokenizer: Uint8Array, weights: Uint8Array): Session;
  has_target(): boolean;
  /**
   * A session from ONE FAST_LLM file (`prattle_core::fast_llm`): the reading model, the answering
   * model and their bridge, attached as `attach_target_bytes` would. The file's bytes are let go as
   * soon as they are split, so the page's 4 GiB holds one copy of the weights while the models build.
   */
  static from_fast_llm(bytes: Uint8Array): Session;
  prompt_tokens(question: string): number;
  /**
   * A synthetic prompt of at least `tokens` prompt tokens; `variant` makes it distinct.
   */
  synthetic_prompt(tokens: number, variant: number): string;
  /**
   * Attach the TARGET that answers from this model's reading when the bridge is on, from its
   * config.json and model.safetensors; it shares this session's tokens. `bridge` is a fitted bridge
   * file, or empty for direct copy between equal shapes.
   */
  attach_target_bytes(config: Uint8Array, weights: Uint8Array, bridge: Uint8Array): void;
  /**
   * The text the next token adds, or `undefined` when the reply is finished.
   */
  step(): string | undefined;
  start(question: string, bridge: boolean): void;
  /**
   * `[prompt tokens, reader tokens, generator history, bridge tokens]` for the reply in progress.
   */
  counters(): Uint32Array;
  /**
   * Begin one control arm: `native`, `bridge`, `restore`, or `donor` (from `donor`'s memory).
   */
  start_arm(question: string, arm: string, donor: string): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_session_free: (a: number, b: number) => void;
  readonly first_arm: (a: number, b: number, c: number) => number;
  readonly prompt_hash: (a: number, b: number) => [number, number];
  readonly session_attach_target_bytes: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly session_counters: (a: number) => [number, number];
  readonly session_from_bytes: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
  readonly session_from_fast_llm: (a: number, b: number) => [number, number, number];
  readonly session_has_target: (a: number) => number;
  readonly session_prompt_tokens: (a: number, b: number, c: number) => [number, number, number];
  readonly session_start: (a: number, b: number, c: number, d: number) => [number, number];
  readonly session_start_arm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly session_step: (a: number) => [number, number, number, number];
  readonly session_synthetic_prompt: (a: number, b: number, c: number) => [number, number, number, number];
  readonly summarize: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
