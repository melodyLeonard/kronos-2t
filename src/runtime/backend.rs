//! The target engine behind [`crate::runtime::chat::ChatModel`]: the Llama/Qwen2/Qwen3 family
//! (`RealLlamaEngine`) or a Qwen3.5 hybrid (`MetalQwen35`).
//!
//! The chat layer treats both the same, with one difference: Qwen3.5's Gated DeltaNet state is
//! cumulative, so it cannot be cut back to an arbitrary earlier position. It rolls back only
//! inside the last speculative verify, to a full [`StateSnapshot`] taken at an exact position, or
//! to an empty context. [`Backend::can_truncate_kv_to`] says which cuts are possible.

use crate::backend::metal_qwen35::{MetalQwen35, StateSnapshot};
use crate::model::kv_blob::KvBlobLayout;
use crate::model::qwen35::{QuantSpec, Qwen35Config, Qwen35Model};
use crate::model::real_llm::{KvPrefixSnapshot, RealLlamaEngine};
use crate::model::sparse_moe::MoeRuntimeStats;
use std::path::Path;

/// Tokens one Qwen3.5 verify pass can check.
const QWEN35_MAX_VERIFY: usize = crate::backend::metal_gpu::GEMV_MAX_TOKENS;

/// A saved KV prefix of either engine.
pub(crate) enum PrefixSnap {
    Llama(KvPrefixSnapshot),
    /// Restorable only in full, at exactly the position it was taken.
    Qwen35(Box<StateSnapshot>),
}

impl PrefixSnap {
    pub(crate) fn byte_len(&self) -> usize {
        match self {
            PrefixSnap::Llama(s) => s.byte_len(),
            PrefixSnap::Qwen35(s) => s.bytes(),
        }
    }
}

pub(crate) struct Qwen35Backend {
    model: Qwen35Model,
    gpu: MetalQwen35,
    /// Logits of the last forward call, `[rows, vocab]`.
    logits: Vec<f32>,
    /// `(pos0, n)` of the last verify until a cut or the next forward settles it.
    pending: Option<(usize, usize)>,
}

impl Qwen35Backend {
    /// Settles an open verify as fully accepted.
    fn settle(&mut self) {
        if let Some((_, n)) = self.pending.take() {
            self.gpu.accept(n);
        }
    }

    /// Feeds `tokens` and returns the logits after the last one.
    pub(crate) fn forward_last(&mut self, tokens: &[usize]) -> &[f32] {
        self.settle();
        let v = self.gpu.vocab();
        let mut logits = std::mem::take(&mut self.logits);
        logits.resize(v, 0.0);
        let want = [tokens.len() - 1];
        if let Err(e) = self.gpu.forward(&self.model, tokens, &want, &mut |_, l| logits.copy_from_slice(l)) {
            panic!("Qwen3.5 forward: {e}");
        }
        self.logits = logits;
        &self.logits
    }

    /// Feeds `tokens` without reading logits.
    pub(crate) fn feed(&mut self, tokens: &[usize]) {
        self.settle();
        if let Err(e) = self.gpu.forward(&self.model, tokens, &[], &mut |_, _| {}) {
            panic!("Qwen3.5 forward: {e}");
        }
    }

    pub(crate) fn reset(&mut self) {
        self.pending = None;
        self.gpu.reset();
    }

    pub(crate) fn snapshot(&mut self) -> StateSnapshot {
        self.settle();
        self.gpu.snapshot()
    }

    pub(crate) fn restore(&mut self, s: &StateSnapshot) {
        self.pending = None;
        self.gpu.restore(s);
    }
}

pub(crate) enum Backend {
    Llama(RealLlamaEngine),
    Qwen35(Box<Qwen35Backend>),
}

/// Loads a compiled Qwen3.5 model (`kronos-qwen35-<mode>.k2tb`), compiling it from `calib` first
/// when missing. Returns `None` when `model_dir` is not a Qwen3.5 checkpoint.
pub(crate) fn load_qwen35(
    model_dir: &Path,
    cache_dir: Option<&Path>,
    mode_key: &str,
    ctx: usize,
    calib: Option<(&Path, usize)>,
) -> Option<Result<Backend, String>> {
    let cfg = Qwen35Config::from_model_dir(model_dir).ok()?;
    Some((|| {
        let bits = match mode_key {
            "nf4g128" => 4,
            "nf3g128" => 3,
            other => return Err(format!("Qwen3.5 supports --mode nf4g128 or nf3g128 (got {other})")),
        };
        let cache = match cache_dir {
            Some(dir) => {
                let tag = model_dir.file_name().map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
                dir.join(format!("{tag}-kronos-qwen35-{mode_key}.k2tb"))
            }
            None => model_dir.join(format!("kronos-qwen35-{mode_key}.k2tb")),
        };
        let model = match Qwen35Model::load(model_dir, &cache) {
            Ok(m) => m,
            Err(_) => {
                let (path, n) = calib.ok_or_else(|| {
                    format!("no compiled Qwen3.5 model at {}; provide calibration text to compile it", cache.display())
                })?;
                let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
                let tok = crate::model::bpe_tokenizer::BpeTokenizer::from_model_dir(model_dir)
                    .map_err(|e| format!("tokenizer: {e}"))?;
                let ids = tok.encode(&text);
                let seqs: Vec<Vec<usize>> = ids.chunks_exact(512).take(n.div_ceil(512).max(1)).map(|c| c.to_vec()).collect();
                if seqs.is_empty() {
                    return Err("calibration text is shorter than 512 tokens".to_string());
                }
                eprintln!("compiling Qwen3.5 {} ({mode_key}), one time only; this takes about 30 minutes...", model_dir.display());
                let spec = QuantSpec { bits, head_bits: 4, scale_group_chunks: 2, gptq: true, mlp_bits: bits, mlp_err_max: f32::INFINITY, in_bits: bits };
                let threads = std::thread::available_parallelism().map_or(8, |p| p.get());
                let m = crate::model::qwen35::compile(model_dir, spec, &seqs, threads, &mut |s| eprintln!("{s}"))
                    .map_err(|e| format!("compile: {e}"))?;
                if let Some(parent) = cache.parent() {
                    std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                m.save(&cache).map_err(|e| format!("save {}: {e}", cache.display()))?;
                m
            }
        };
        let gpu = MetalQwen35::new(&model, ctx)?;
        let _ = cfg;
        Ok(Backend::Qwen35(Box::new(Qwen35Backend { model, gpu, logits: Vec::new(), pending: None })))
    })())
}

impl Backend {
    pub(crate) fn llama(&self) -> Option<&RealLlamaEngine> {
        match self {
            Backend::Llama(e) => Some(e),
            Backend::Qwen35(_) => None,
        }
    }

    pub(crate) fn llama_mut(&mut self) -> Option<&mut RealLlamaEngine> {
        match self {
            Backend::Llama(e) => Some(e),
            Backend::Qwen35(_) => None,
        }
    }

    pub(crate) fn is_qwen35(&self) -> bool {
        matches!(self, Backend::Qwen35(_))
    }

    pub(crate) fn vocab_size(&self) -> usize {
        match self {
            Backend::Llama(e) => e.cfg.vocab_size,
            Backend::Qwen35(q) => q.model.cfg.vocab_size,
        }
    }

    pub(crate) fn hidden_size(&self) -> usize {
        match self {
            Backend::Llama(e) => e.cfg.hidden_size,
            Backend::Qwen35(q) => q.model.cfg.hidden_size,
        }
    }

    pub(crate) fn head_dim(&self) -> usize {
        match self {
            Backend::Llama(e) => e.cfg.head_dim(),
            Backend::Qwen35(q) => q.model.cfg.head_dim,
        }
    }

    pub(crate) fn embed(&self, token: usize, out: &mut [f32]) {
        match self {
            Backend::Llama(e) => e.embed(token, out),
            Backend::Qwen35(q) => q.model.embed_into(token, out),
        }
    }

    pub(crate) fn position(&self) -> usize {
        match self {
            Backend::Llama(e) => e.position(),
            Backend::Qwen35(q) => q.gpu.pos(),
        }
    }

    pub(crate) fn reset_context(&mut self) {
        match self {
            Backend::Llama(e) => e.reset_context(),
            Backend::Qwen35(q) => {
                q.pending = None;
                q.gpu.reset();
            }
        }
    }

    /// Whether the context can be cut back to `n` positions: always for Llama outside sliding
    /// windows; for Qwen3.5 only to the current position, to 0, or inside the open verify.
    pub(crate) fn can_truncate_kv_to(&self, n: usize) -> bool {
        match self {
            Backend::Llama(e) => e.can_truncate_kv_to(n),
            Backend::Qwen35(q) => {
                n == 0 || n >= q.gpu.pos() || q.pending.is_some_and(|(pos0, _)| n > pos0)
            }
        }
    }

    pub(crate) fn truncate_context(&mut self, n: usize) {
        match self {
            Backend::Llama(e) => e.truncate_context(n),
            Backend::Qwen35(q) => {
                if n >= q.gpu.pos() {
                    q.settle();
                } else if let Some((pos0, _)) = q.pending.filter(|&(pos0, _)| n > pos0) {
                    q.pending = None;
                    q.gpu.accept(n - pos0);
                } else if n == 0 {
                    q.pending = None;
                    q.gpu.reset();
                } else {
                    panic!("Qwen3.5 cannot roll back from {} to {n}", q.gpu.pos());
                }
            }
        }
    }

    pub(crate) fn forward_prompt(&mut self, tokens: &[usize]) -> &[f32] {
        match self {
            Backend::Llama(e) => e.forward_prompt(tokens),
            Backend::Qwen35(q) => q.forward_last(tokens),
        }
    }

    pub(crate) fn forward_token(&mut self, token: usize) -> &[f32] {
        match self {
            Backend::Llama(e) => e.forward_token(token),
            Backend::Qwen35(q) => q.forward_last(&[token]),
        }
    }

    /// Feeds `tokens` and writes `[tokens.len(), vocab]` logits. On Qwen3.5 this is a verify pass
    /// of at most [`Self::max_verify_batch`] tokens that a following cut can partly undo.
    pub(crate) fn forward_batch_logits(&mut self, tokens: &[usize], out: &mut [f32]) {
        match self {
            Backend::Llama(e) => e.forward_batch_logits(tokens, out),
            Backend::Qwen35(q) => {
                q.settle();
                let pos0 = q.gpu.pos();
                let v = q.gpu.vocab();
                let model = &q.model;
                if let Err(e) = q.gpu.verify(model, tokens, &mut |i, l| out[i * v..(i + 1) * v].copy_from_slice(l)) {
                    panic!("Qwen3.5 verify: {e}");
                }
                q.pending = Some((pos0, tokens.len()));
            }
        }
    }

    /// Most tokens one speculative verify may carry.
    pub(crate) fn max_verify_batch(&self) -> usize {
        match self {
            Backend::Llama(_) => usize::MAX,
            Backend::Qwen35(_) => QWEN35_MAX_VERIFY,
        }
    }

    pub(crate) fn is_all_sdot(&self) -> bool {
        match self {
            Backend::Llama(e) => e.is_all_sdot(),
            Backend::Qwen35(_) => true,
        }
    }

    /// Saves the first `n` positions; Qwen3.5 only at exactly the current position.
    pub(crate) fn save_kv_prefix(&mut self, n: usize) -> Option<PrefixSnap> {
        match self {
            Backend::Llama(e) => e.save_kv_prefix(n).map(PrefixSnap::Llama),
            Backend::Qwen35(q) => (n == q.gpu.pos() && n > 0).then(|| PrefixSnap::Qwen35(Box::new(q.snapshot()))),
        }
    }

    /// Restores the first `copy` positions of `snap` as the whole context; Qwen3.5 needs `copy`
    /// to be the snapshot's own length. Returns whether it restored.
    pub(crate) fn restore_prefix(&mut self, snap: &PrefixSnap, copy: usize) -> bool {
        match (self, snap) {
            (Backend::Llama(e), PrefixSnap::Llama(s)) => {
                e.restore_kv_prefix_slice(s, copy, copy);
                true
            }
            (Backend::Qwen35(q), PrefixSnap::Qwen35(s)) if copy == s.pos => {
                q.restore(s);
                true
            }
            _ => false,
        }
    }

    /// Positions of `snap` [`Self::restore_prefix`] can use given `common` shared tokens.
    pub(crate) fn usable_prefix(snap: &PrefixSnap, common: usize) -> usize {
        match snap {
            PrefixSnap::Llama(_) => common,
            PrefixSnap::Qwen35(s) => if common >= s.pos { s.pos } else { 0 },
        }
    }

    pub(crate) fn kv_blob_layout(&self) -> Option<KvBlobLayout> {
        self.llama().and_then(RealLlamaEngine::kv_blob_layout)
    }

    pub(crate) fn export_kv_positions(&self, lo: usize, hi: usize, out: &mut Vec<u8>) -> bool {
        self.llama().is_some_and(|e| e.export_kv_positions(lo, hi, out))
    }

    pub(crate) fn import_kv_positions(&mut self, layout: &KvBlobLayout, data: &[u8], n: usize) -> bool {
        self.llama_mut().is_some_and(|e| e.import_kv_positions(layout, data, n))
    }

    pub(crate) fn enable_metal(&mut self) -> Result<(), String> {
        match self {
            Backend::Llama(e) => e.enable_metal(),
            Backend::Qwen35(_) => Ok(()),
        }
    }

    pub(crate) fn disable_metal(&mut self) {
        if let Backend::Llama(e) = self {
            e.disable_metal();
        }
    }

    pub(crate) fn is_metal_enabled(&self) -> bool {
        self.llama().is_none_or(RealLlamaEngine::is_metal_enabled)
    }

    pub(crate) fn set_kv_bits(&mut self, bits: u8) {
        if let Backend::Llama(e) = self {
            e.set_kv_bits(bits);
        }
    }

    /// Qwen3.5 keeps attention KV as Q8.
    pub(crate) fn kv_bits(&self) -> u8 {
        self.llama().map_or(8, RealLlamaEngine::kv_bits)
    }

    pub(crate) fn enable_q8_kv_cache(&mut self, enable: bool) {
        if let Backend::Llama(e) = self {
            e.enable_q8_kv_cache(enable);
        }
    }

    pub(crate) fn is_q8_kv_enabled(&self) -> bool {
        self.llama().is_none_or(RealLlamaEngine::is_q8_kv_enabled)
    }

    pub(crate) fn set_kv_sliding_window(&mut self, window: Option<usize>, sink_tokens: usize) {
        if let Backend::Llama(e) = self {
            e.set_kv_sliding_window(window, sink_tokens);
        }
    }

    pub(crate) fn kv_sliding_window(&self) -> Option<(usize, usize)> {
        self.llama().and_then(RealLlamaEngine::kv_sliding_window)
    }

    pub(crate) fn is_moe(&self) -> bool {
        self.llama().is_some_and(RealLlamaEngine::is_moe)
    }

    pub(crate) fn moe_stats(&self) -> Option<MoeRuntimeStats> {
        self.llama().and_then(RealLlamaEngine::moe_stats)
    }

    pub(crate) fn num_threads(&self) -> usize {
        self.llama().map_or_else(crate::kernels::thread_pool::default_inference_threads, RealLlamaEngine::num_threads)
    }

    pub(crate) fn resident_weight_bytes(&self) -> usize {
        match self {
            Backend::Llama(e) => e.resident_weight_bytes(),
            Backend::Qwen35(q) => q.model.weight_bytes_per_token() + q.model.embed.pair_stream.len(),
        }
    }

    pub(crate) fn kv_cache_bytes(&self) -> usize {
        match self {
            Backend::Llama(e) => e.kv_cache_bytes(),
            Backend::Qwen35(q) => q.gpu.state_bytes(),
        }
    }
}
