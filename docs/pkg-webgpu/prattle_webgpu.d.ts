/* tslint:disable */
/* eslint-disable */
export function first_arm(prompt: number, rep: number, arms: number): number;
/**
 * prattle-core's paired statistic, as `prattle-web` exports it.
 */
export function summarize(prompts: Uint32Array, native_ms: Float64Array, bridge_ms: Float64Array): Float64Array;
export function prompt_hash(text: string): string;
export class Session {
  private constructor();
  free(): void;
  /**
   * A session from a model's config.json, tokenizer.json and model.safetensors: a promise.
   */
  static from_bytes(config: Uint8Array, tokenizer: Uint8Array, weights: Uint8Array): Promise<any>;
  has_target(): boolean;
  /**
   * Begin the answering model; its weights arrive through `feed`, then `finish_target`.
   */
  begin_target(config: Uint8Array): void;
  finish_source(): void;
  finish_target(bridge: Uint8Array): void;
  prompt_tokens(question: string): number;
  synthetic_prompt(tokens: number, variant: number): string;
  attach_target_bytes(config: Uint8Array, weights: Uint8Array, bridge: Uint8Array): void;
  /**
   * The next piece of the file arriving, in order.
   */
  feed(bytes: Uint8Array): void;
  /**
   * A session whose reading model's weights then arrive in pieces — `feed`, `next_file` between
   * the files of a sharded checkpoint, `finish_source` at the end — so a file larger than this
   * module's 4 GiB of memory still loads: a promise.
   */
  static open(config: Uint8Array, tokenizer: Uint8Array): Promise<any>;
  /**
   * A promise of the text the next token adds, or `undefined` when the reply is finished. The
   * worker awaits each step before the next, so the model is never borrowed twice.
   */
  step(): Promise<any>;
  start(question: string, bridge: boolean): void;
  counters(): Uint32Array;
  /**
   * The model arriving goes on to its next file.
   */
  next_file(): void;
  start_arm(question: string, arm: string, donor: string): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly __wbg_session_free: (a: number, b: number) => void;
  readonly first_arm: (a: number, b: number, c: number) => number;
  readonly prompt_hash: (a: number, b: number) => [number, number];
  readonly session_attach_target_bytes: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly session_begin_target: (a: number, b: number, c: number) => [number, number];
  readonly session_counters: (a: number) => [number, number];
  readonly session_feed: (a: number, b: number, c: number) => [number, number];
  readonly session_finish_source: (a: number) => [number, number];
  readonly session_finish_target: (a: number, b: number, c: number) => [number, number];
  readonly session_from_bytes: (a: number, b: number, c: number, d: number, e: number, f: number) => any;
  readonly session_has_target: (a: number) => number;
  readonly session_next_file: (a: number) => [number, number];
  readonly session_open: (a: number, b: number, c: number, d: number) => any;
  readonly session_prompt_tokens: (a: number, b: number, c: number) => [number, number, number];
  readonly session_start: (a: number, b: number, c: number, d: number) => [number, number];
  readonly session_start_arm: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number];
  readonly session_step: (a: number) => any;
  readonly session_synthetic_prompt: (a: number, b: number, c: number) => [number, number, number, number];
  readonly summarize: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number];
  readonly __wbindgen_exn_store: (a: number) => void;
  readonly __externref_table_alloc: () => number;
  readonly __wbindgen_export_2: WebAssembly.Table;
  readonly __wbindgen_malloc: (a: number, b: number) => number;
  readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
  readonly __wbindgen_export_5: WebAssembly.Table;
  readonly __wbindgen_free: (a: number, b: number, c: number) => void;
  readonly __externref_table_dealloc: (a: number) => void;
  readonly closure1472_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure1499_externref_shim: (a: number, b: number, c: any) => void;
  readonly closure1515_externref_shim: (a: number, b: number, c: any, d: any) => void;
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
