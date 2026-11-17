//! The ORACLE: upstream LatentPort's native runtime (patched llama.cpp), built by `just oracle`
//! and driven over its C ABI (`native/latentport.h` at upstream `6be1956f`). It is the numeric
//! reference every Rust port of a piece is checked against — docs/BRIDGE.md §7, slice 11.

pub mod prompt;
pub mod reproduce;
pub mod template;

use std::path::{Path, PathBuf};

pub struct Oracle {
    lib: libloading::Library,
    path: PathBuf,
    /// The CUDA libraries it links against, loaded first and kept alive with it.
    _cuda: Vec<libloading::Library>,
}

/// The CUDA runtime libraries the oracle links by soname. The toolkit lives in user space
/// (`just cuda`), off the system search path, so they are loaded by absolute path first — a
/// soname already loaded satisfies the oracle's dependency.
const CUDA_LIBS: [&str; 3] = ["libcudart.so.13", "libcublasLt.so.13", "libcublas.so.13"];

fn preload_cuda() -> Result<Vec<libloading::Library>, String> {
    let Ok(home) = std::env::var("PRATTLE_CUDA_HOME") else { return Ok(Vec::new()) };
    CUDA_LIBS
        .iter()
        .map(|name| {
            let path = Path::new(&home).join("lib64").join(name);
            // SAFETY: NVIDIA's own runtime libraries; RTLD_GLOBAL so the oracle resolves against them.
            unsafe {
                libloading::os::unix::Library::open(Some(&path), libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL)
            }
            .map(libloading::Library::from)
            .map_err(|e| format!("{}: {e}", path.display()))
        })
        .collect()
}

impl Oracle {
    /// Where `just oracle` puts the runtime; `PRATTLE_ORACLE_LIB` overrides. `PRATTLE_CUDA_HOME`
    /// names the toolkit its CUDA libraries come from — `just` recipes set both.
    pub fn default_path() -> PathBuf {
        std::env::var("PRATTLE_ORACLE_LIB").map(PathBuf::from).unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default()).join(".cache/prattle/oracle/build/bin/libllama.so")
        })
    }

    pub fn load_default() -> Result<Self, String> {
        Self::load(&Self::default_path())
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Err(format!("no oracle runtime at {}", path.display()));
        }
        let cuda = preload_cuda()?;
        // SAFETY: loading runs the library's initialisers; it is our own build of pinned source.
        let lib = unsafe { libloading::Library::new(path) }.map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self { lib, path: path.to_path_buf(), _cuda: cuda })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// `lp_abi()`: the native ABI version the runtime implements.
    pub fn abi(&self) -> i32 {
        // SAFETY: `int lp_abi(void)` per latentport.h; no arguments, no state.
        unsafe {
            match self.lib.get::<unsafe extern "C" fn() -> i32>(b"lp_abi\0") {
                Ok(f) => f(),
                Err(_) => -1,
            }
        }
    }
}

/// Which side of the pair a model is opened as: the target allocates multi-token-prediction drafts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Source,
    Target,
}

/// One timed native run: the model read the whole prompt itself.
#[derive(Debug, Clone, Copy)]
pub struct NativeRun {
    pub prompt_tokens: usize,
    pub first_token: i32,
    pub seconds: f64,
    pub prefill_delta: u64,
}

#[repr(C)]
#[derive(Default)]
struct Info {
    abi: i32,
    layers: i32,
    hidden: i32,
    kv_heads: i32,
    head_dim: i32,
    conv_channels: i32,
    recurrent_heads: i32,
    mtp_heads: i32,
    max_drafts: i32,
    context_size: i32,
    vocabulary: i32,
    auxiliary_bytes: u64,
}

#[repr(C)]
#[derive(Default)]
struct Trace {
    prefill_tokens: u64,
    committed_tokens: u64,
    verified_tokens: u64,
    head_calls: u64,
    installs: u64,
    exports: u64,
    model_bytes: u64,
    context_bytes: u64,
    compute_bytes: u64,
    auxiliary_bytes: u64,
    last_install_s: f64,
    last_head_s: f64,
}

type Handle = *mut std::ffi::c_void;

/// One model resident in the oracle, closed on drop.
pub struct Model<'a> {
    oracle: &'a Oracle,
    handle: Handle,
    info: Info,
    tokenizer: Option<tokenizers::Tokenizer>,
}

/// Upstream's engine defaults (`_native/engine.py`): batch 512, 8 threads.
const BATCH: i32 = 512;
const THREADS: i32 = 8;

impl Oracle {
    fn sym<T: Copy>(&self, name: &[u8]) -> Result<T, String> {
        // SAFETY: each caller names a function with the signature latentport.h declares for it.
        unsafe { self.lib.get::<T>(name).map(|s| *s).map_err(|e| e.to_string()) }
    }

    fn error(&self) -> String {
        let f: Result<unsafe extern "C" fn() -> *const std::ffi::c_char, _> = self.sym(b"lp_error\0");
        match f {
            // SAFETY: lp_error returns a NUL-terminated string owned by the runtime, or null.
            Ok(f) => unsafe {
                let p = f();
                if p.is_null() { "unknown oracle error".into() } else { std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned() }
            },
            Err(e) => e,
        }
    }

    fn check(&self, what: &str, rc: i32) -> Result<i32, String> {
        if rc < 0 { Err(format!("{what}: {}", self.error())) } else { Ok(rc) }
    }

    /// Open a GGUF as one side of the pair with `context` tokens of working cache. The target gets
    /// no MTP drafts: the timed runs decode with multi-token prediction off, as upstream's do.
    pub fn open(&self, gguf: &str, side: Side, context: usize) -> Result<Model<'_>, String> {
        let open: unsafe extern "C" fn(*const std::ffi::c_char, i32, i32, i32, i32, i32) -> Handle = self.sym(b"lp_open\0")?;
        let path = std::ffi::CString::new(gguf).map_err(|e| e.to_string())?;
        // SAFETY: a valid C string; the remaining arguments are plain integers.
        let handle = unsafe { open(path.as_ptr(), i32::from(side == Side::Target), context as i32, BATCH, THREADS, 0) };
        if handle.is_null() {
            return Err(format!("lp_open {gguf}: {}", self.error()));
        }
        let mut info = Info::default();
        let get_info: unsafe extern "C" fn(Handle, *mut Info) -> i32 = self.sym(b"lp_get_info\0")?;
        // SAFETY: handle is live; info is a correctly laid-out lp_info.
        self.check("lp_get_info", unsafe { get_info(handle, &mut info) })?;
        Ok(Model { oracle: self, handle, info, tokenizer: None })
    }
}

impl Model<'_> {
    pub fn vocabulary(&self) -> usize {
        self.info.vocabulary as usize
    }

    pub fn layers(&self) -> usize {
        self.info.layers as usize
    }

    /// The tokenizer upstream's codec uses for this side (its pinned `tokenizer.json`).
    pub fn with_tokenizer(mut self, tokenizer_json: &str) -> Result<Self, String> {
        self.tokenizer = Some(tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|e| format!("{tokenizer_json}: {e}"))?);
        Ok(self)
    }

    fn prefill_tokens(&self) -> Result<u64, String> {
        let f: unsafe extern "C" fn(Handle, *mut Trace) -> i32 = self.oracle.sym(b"lp_get_trace\0")?;
        let mut t = Trace::default();
        // SAFETY: handle is live; t is a correctly laid-out lp_trace.
        self.oracle.check("lp_get_trace", unsafe { f(self.handle, &mut t) })?;
        Ok(t.prefill_tokens)
    }

    /// The native arm: render one user turn with the model's template, tokenize, reset, prefill the
    /// whole prompt, and take the greedy first token. Timed from before rendering to the argmax.
    pub fn native_first_token(&mut self, template: &str, kwargs: &serde_json::Value, user: &str) -> Result<NativeRun, String> {
        let reset: unsafe extern "C" fn(Handle) -> i32 = self.oracle.sym(b"lp_reset\0")?;
        let prefill: unsafe extern "C" fn(Handle, *const i32, i32) -> i32 = self.oracle.sym(b"lp_prefill\0")?;
        let logits: unsafe extern "C" fn(Handle, *mut f32, i32) -> i32 = self.oracle.sym(b"lp_logits\0")?;
        let tok = self.tokenizer.as_ref().ok_or("no tokenizer attached — with_tokenizer first")?;
        let before = self.prefill_tokens()?;
        let mut out = vec![0f32; self.vocabulary()];

        let t0 = std::time::Instant::now();
        let rendered = template::render_user(template, user, kwargs)?;
        let ids: Vec<i32> = tok.encode(rendered, false).map_err(|e| e.to_string())?.get_ids().iter().map(|&i| i as i32).collect();
        // SAFETY: handle is live; ids and out outlive the calls and their lengths are passed.
        unsafe {
            self.oracle.check("lp_reset", reset(self.handle))?;
            self.oracle.check("lp_prefill", prefill(self.handle, ids.as_ptr(), ids.len() as i32))?;
            self.oracle.check("lp_logits", logits(self.handle, out.as_mut_ptr(), out.len() as i32))?;
        }
        let first = out.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i as i32).ok_or("no logits")?;
        let seconds = t0.elapsed().as_secs_f64();

        Ok(NativeRun { prompt_tokens: ids.len(), first_token: first, seconds, prefill_delta: self.prefill_tokens()? - before })
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        if let Ok(close) = self.oracle.sym::<unsafe extern "C" fn(Handle)>(b"lp_close\0") {
            // SAFETY: handle came from lp_open and is closed exactly once.
            unsafe { close(self.handle) };
        }
    }
}
