// The page's models, off the main thread: this worker owns the wasm Session, fetches the weights
// (keeping files pinned to a hub commit in Cache Storage), and runs every forward pass, so a long
// prompt never blocks the page. Every clock that is measured runs here, in the worker.
//
// Protocol: the page posts {id, op, args}; the worker answers {id, ok} or {id, err}, and while an
// op is in flight may post {id, progress}, {id, counters}, {id, ttft} and {id, piece}.
//
// Two engines behind one Session API: ?backend=webgpu loads prattle-webgpu (our WGSL kernels), and
// otherwise prattle-web (candle on the CPU). The WebGPU engine's loading and step return promises,
// so every call is awaited; awaiting a plain value costs the CPU engine nothing.
//
// The message handler is installed before the engine is imported: a message that arrived while the
// module was still awaiting its import would otherwise be dispatched to no handler, and lost.
const BACKEND = new URL(self.location.href).searchParams.get('backend') === 'webgpu' ? 'webgpu' : 'cpu';
let Session, summarize, first_arm, prompt_hash, wasm;
const ready = (async () => {
  const pkg = await import(BACKEND === 'webgpu' ? './pkg-webgpu/prattle_webgpu.js' : './pkg/prattle_web.js');
  ({ Session, summarize, first_arm, prompt_hash } = pkg);
  wasm = await pkg.default();
})();

const CACHE = 'prattle-weights-v1';
// Only files pinned to a commit on the hub are kept: a local file (models/ beside the page, or a
// bridge fitted on this machine) can change under the same URL, so it is always fetched fresh.
const keepable = url => url.startsWith('https://huggingface.co/') && /\/resolve\/[0-9a-f]{40}\//.test(url);

let session = null;

async function openCache() {
  try { return await caches.open(CACHE); } catch { return null; }
}

// The bytes of `url`, reporting progress as {label, loaded, total, from}.
async function fetchBytes(url, label, progress) {
  const keep = keepable(url);
  const cache = keep ? await openCache() : null;
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const bytes = new Uint8Array(await hit.arrayBuffer());
    progress({ label, loaded: bytes.length, total: bytes.length, from: 'cache' });
    return { bytes, from: 'cache' };
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} answered ${response.status}`);
  let from = keep ? 'network' : 'local';
  const total = +response.headers.get('content-length') || 0;
  // The cache is fed from one branch of the stream while the other is read here, so the browser
  // stores the file without this worker holding a second copy of it beside the wasm heap.
  let body = response.body, stored = null;
  if (cache) {
    const [mine, theirs] = body.tee();
    body = mine;
    stored = cache.put(url, new Response(theirs, { headers: { 'content-type': 'application/octet-stream' } }));
  }
  const reader = body.getReader();
  const chunks = [];
  let loaded = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    if (loaded - last > 4 << 20) { last = loaded; progress({ label, loaded, total, from }); }
  }
  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) { bytes.set(c, at); at += c.length; }
  chunks.length = 0;
  if (stored) {
    try { await stored; } catch (e) { from = `network, not kept: ${e?.name ?? e}`; }
  }
  progress({ label, loaded, total: total || loaded, from });
  return { bytes, from };
}

// A control arm runs to completion, capped.
const CONTROL_TOKENS = 32;
// A timed run stops at the first visible text.
const TEXT_STEPS = 16;

const ops = {
  async load({ base, label }, post) {
    const progress = p => post({ progress: p });
    const config = await fetchBytes(base + 'config.json', `${label} config`, progress);
    const tokenizer = await fetchBytes(base + 'tokenizer.json', `${label} tokenizer`, progress);
    const weights = await fetchBytes(base + 'model.safetensors', label, progress);
    session = await Session.from_bytes(config.bytes, tokenizer.bytes, weights.bytes);
    return { [label]: weights.from };
  },

  async attach({ base, label, bridge }, post) {
    const progress = p => post({ progress: p });
    const config = await fetchBytes(base + 'config.json', `${label} config`, progress);
    const weights = await fetchBytes(base + 'model.safetensors', label, progress);
    const map = await fetchBytes(bridge, 'the bridge', progress);
    session.attach_target_bytes(config.bytes, weights.bytes, map.bytes);
    return { [label]: weights.from, bridge: map.from };
  },

  // ONE FAST_LLM file: the reading model, the answering model and the bridge, in one download.
  async loadFast({ url, label }, post) {
    if (!Session.from_fast_llm) throw new Error('this engine does not read a FAST_LLM yet — open the page without ?backend=webgpu');
    const progress = p => post({ progress: p });
    const file = await fetchBytes(url, label, progress);
    session = await Session.from_fast_llm(file.bytes);
    return { [label]: file.from };
  },

  hasTarget() { return session?.has_target() ?? false; },
  backend() { return BACKEND; },
  // How much memory the engine's wasm holds: it has 4 GiB in all, and an allocation past it aborts.
  memory() { return wasm?.memory?.buffer?.byteLength ?? 0; },

  // One reply, streamed: the clock starts before the prompt is read and stops at the first token.
  async ask({ question, bridge }, post) {
    const t0 = performance.now();
    let ttft = null;
    session.start(question, bridge);
    post({ counters: Array.from(session.counters()) });
    for (;;) {
      const piece = await session.step();
      if (piece === undefined) break;
      if (piece && ttft === null) { ttft = performance.now() - t0; post({ ttft }); }
      post({ piece });
    }
    return { ttft_ms: ttft, counters: Array.from(session.counters()) };
  },

  async arm({ question, arm, donor }) {
    session.start_arm(question, arm, donor || '');
    let text = '';
    for (let i = 0; i < CONTROL_TOKENS; i++) {
      const piece = await session.step();
      if (piece === undefined) break;
      text += piece;
    }
    return text.trim();
  },

  // One timed run. The token clock stops when the first token is sampled; the text clock when the
  // first visible text arrives. Nothing is excluded: the bridge arm's source prefill is inside.
  async time({ question, arm }) {
    const t0 = performance.now();
    session.start_arm(question, arm, '');
    let token = null, text = null, acc = '';
    for (let i = 0; i < TEXT_STEPS; i++) {
      const piece = await session.step();
      const now = performance.now();
      if (token === null) token = now - t0;
      if (piece === undefined) break;
      acc += piece;
      if (acc.trim()) { text = now - t0; break; }
    }
    const [prompt, reader, history, bridge] = session.counters();
    return { arm, ttft_token_ms: token, ttft_text_ms: text, counters: { prompt, reader, history, bridge } };
  },

  synthetic({ tokens, variant }) { return session.synthetic_prompt(tokens, variant); },
  promptTokens({ text }) { return session.prompt_tokens(text); },
  hash({ text }) { return prompt_hash(text); },
  firstArm({ prompt, rep, arms }) { return first_arm(prompt, rep, arms); },
  summarize({ prompts, native, bridge }) {
    return Array.from(summarize(new Uint32Array(prompts), new Float64Array(native), new Float64Array(bridge)));
  },
};

// A Rust panic arrives as an uncaught "unreachable"; the engine's panic hook left the real message.
self.addEventListener('error', e => {
  const where = (e.error?.stack || '').split('\n').slice(0, 12).join(' ← ');
  self.postMessage({ panic: self.prattlePanic || `${e.message}${where ? ` at ${where}` : ''}` });
});

// One op at a time, in arrival order: the WebGPU engine's steps are promises, and an op that began
// while another awaited its GPU would reach the model mid-reply.
let queue = Promise.resolve();
self.onmessage = ({ data: { id, op, args } }) => {
  const post = m => self.postMessage({ id, ...m });
  queue = queue.then(async () => {
    try {
      await ready;
      if (!ops[op]) throw new Error(`no op called ${op}`);
      post({ ok: await ops[op](args || {}, post) });
    } catch (e) {
      post({ err: String(e?.message ?? e) });
    }
  });
};
