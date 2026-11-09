//! Chat on a compiled Kronos model: ChatML template, zero-allocation sampling, `O(1)` incremental
//! UTF-8 token streaming, KV-cache prefix reuse, and Dual-Tier Lossless Speculative Decoding
//! (Zero-RAM Prompt/History Suffix Lookup + Draft-Model Leviathan-Chen Speculative Verification).

use crate::model::bpe_tokenizer::{BpeTokenizer, PreTokenizerKind};
use crate::util::json::{quote, Json};
use crate::model::real_llm::{argmax, RealLlamaEngine, RealWeightMode};
use std::path::Path;
use std::time::Instant;
pub use crate::config::prompts::{CREATOR_PUBLIC_PROFILE, DEFAULT_SYSTEM, DEFAULT_SYSTEM_QWEN};

#[cfg(all(feature = "ane", target_os = "macos"))]
use crate::config::limits::ANE_MIN_NEW_TOKENS;
use crate::runtime::backend::{Backend, PrefixSnap};
use crate::runtime::kv_sessions::{KvBlob, KvSource, SessionKvStore};



use crate::config::limits::{SPEC_MTP_K, 
    SPEC_DRAFT_K_GREEDY, SPEC_DRAFT_K_INIT, SPEC_DRAFT_K_SAMPLE, SPEC_LOOKUP_K_INIT, SPEC_LOOKUP_K_MAX,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Raw JSON object string for arguments, e.g. `{"path":"index.html","content":"..."}`.
    pub arguments: String,
}

/// How tool schemas and calls are written in the prompt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolFormat {
    /// Qwen2.5: `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
    Json,
    /// Qwen3-Coder: `<tool_call><function=name><parameter=k>v</parameter></function></tool_call>`.
    Xml,
}

impl ToolFormat {
    /// Reads the model's chat template to pick the format; JSON when there is none.
    pub fn detect(model_dir: &Path) -> Self {
        let has_fn_tag = |p: &Path| std::fs::read_to_string(p).is_ok_and(|s| s.contains("<function="));
        if has_fn_tag(&model_dir.join("chat_template.jinja")) || has_fn_tag(&model_dir.join("tokenizer_config.json")) {
            Self::Xml
        } else {
            Self::Json
        }
    }

    /// Assistant prefix that forces the next output to be a tool call.
    pub fn call_prefix(self) -> &'static str {
        match self {
            Self::Json => "<tool_call>\n{\"name\": \"",
            Self::Xml => "<tool_call>\n<function=",
        }
    }
}

/// Which chat template the model was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PromptStyle {
    /// `<|im_start|>role\n...<|im_end|>\n` (Qwen, SmolLM2).
    ChatMl,
    /// HunYuan / Hy-MT2: `<bos>[system<sys_end>]<user>...<assistant>...<eos>`, no newlines, no
    /// default system prompt, no tool-calling markup.
    HunYuan,
}

impl PromptStyle {
    /// Picks the template from the tokenizer's special tokens.
    pub fn detect(tok: &BpeTokenizer) -> Self {
        if tok.token_id(crate::config::markup::HY_USER).is_some() {
            Self::HunYuan
        } else {
            Self::ChatMl
        }
    }

    /// Text that opens an assistant turn (up to where the reply tokens start).
    fn assistant_open(self) -> &'static str {
        match self {
            Self::ChatMl => "<|im_start|>assistant\n",
            Self::HunYuan => crate::config::markup::HY_ASSISTANT,
        }
    }

    /// Text that closes an assistant turn.
    fn assistant_close(self) -> &'static str {
        match self {
            Self::ChatMl => "<|im_end|>\n",
            Self::HunYuan => crate::config::markup::HY_EOS,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }

    pub fn assistant_tool_calls(content: &str, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
            tool_calls,
            tool_call_id: None,
        }
    }

    pub fn tool_result(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: content.to_string(),
            tool_calls: Vec::new(),
            tool_call_id: Some(tool_call_id.to_string()),
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenParams {
    /// 0 = greedy.
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub max_tokens: usize,
    pub seed: u64,
}

impl Default for GenParams {
    fn default() -> Self {
        Self { temperature: 0.7, top_k: 40, top_p: 0.9, repeat_penalty: 1.1, max_tokens: 512, seed: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub text: String,
    pub token_ids: Vec<usize>,
    pub tool_calls: Vec<ToolCall>,
    pub prompt_tokens: usize,
    /// Prompt tokens actually run (the rest came from the KV cache).
    pub prefilled_tokens: usize,
    pub completion_tokens: usize,
    pub drafted_tokens: usize,
    pub accepted_tokens: usize,
    pub finish: FinishReason,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    /// `"ane"` when the Neural Engine ran the prefill, else `"cpu"`.
    pub prefill_unit: &'static str,
}

impl Completion {
    pub fn acceptance_rate(&self) -> Option<f64> {
        (self.drafted_tokens > 0).then(|| self.accepted_tokens as f64 / self.drafted_tokens as f64)
    }
}

/// Parses a mode key: `nf5`, `nf4`, `nf3`, `e8q2`, `nf3nf4`, `e8q2nf4` (optional `g128`),
/// `q2c` (exact `q2_0c` GGUF weights),
/// `nfpQOGDH` (`G=64`), `nfgQOGDH` (`G=128`), `vq2`, `vq2g128`, `v2b4`, `v2b5`, `b4`, `b5`, `b6`, or `pQOGDH`.
pub fn parse_mode(m: &str) -> Result<RealWeightMode, String> {
    Ok(match m {
        "nf5" => RealWeightMode::KronosNfGptq {
            bits: 5,
            scale_group_chunks: 1,
        },
        "nf5g128" => RealWeightMode::KronosNfGptq {
            bits: 5,
            scale_group_chunks: 2,
        },
        "nf4" => RealWeightMode::KronosNfGptq {
            bits: 4,
            scale_group_chunks: 1,
        },
        "nf4g128" => RealWeightMode::KronosNfGptq {
            bits: 4,
            scale_group_chunks: 2,
        },
        "nf3nf4" => RealWeightMode::KronosNfMixedGptq {
            attn_head_bits: 4,
            ffn_bits: 3,
            scale_group_chunks: 1,
        },
        "nf3nf4g128" => RealWeightMode::KronosNfMixedGptq {
            attn_head_bits: 4,
            ffn_bits: 3,
            scale_group_chunks: 2,
        },
        "nf3" => RealWeightMode::KronosNfGptq {
            bits: 3,
            scale_group_chunks: 1,
        },
        "nf3g128" => RealWeightMode::KronosNfGptq {
            bits: 3,
            scale_group_chunks: 2,
        },
        "e8q2" => RealWeightMode::KronosNfGptq {
            bits: 2,
            scale_group_chunks: 1,
        },
        "e8q2g128" => RealWeightMode::KronosNfGptq {
            bits: 2,
            scale_group_chunks: 2,
        },
        "e8q2nf4" => RealWeightMode::KronosNfMixedGptq {
            attn_head_bits: 4,
            ffn_bits: 2,
            scale_group_chunks: 1,
        },
        "e8q2nf4g128" => RealWeightMode::KronosNfMixedGptq {
            attn_head_bits: 4,
            ffn_bits: 2,
            scale_group_chunks: 2,
        },
        "q2c" => RealWeightMode::KronosQ2c,
        "vq2" => RealWeightMode::KronosVq2Gptq { scale_group_chunks: 1 },
        "vq2g128" => RealWeightMode::KronosVq2Gptq { scale_group_chunks: 2 },
        "v2b4" => RealWeightMode::KronosVq2MixedGptq {
            attn_head_planes: 4,
            scale_group_chunks: 1,
        },
        "v2b5" => RealWeightMode::KronosVq2MixedGptq {
            attn_head_planes: 5,
            scale_group_chunks: 1,
        },
        "b4" => RealWeightMode::KronosBipolarGptq { num_planes: 4 },
        "b5" => RealWeightMode::KronosBipolarGptq { num_planes: 5 },
        "b6" => RealWeightMode::KronosBipolarGptq { num_planes: 6 },
        p if p.len() == 8
            && (p.starts_with("nfp") || p.starts_with("nfg"))
            && p[3..].bytes().all(|b| (b'2'..=b'5').contains(&b)) =>
        {
            let d: Vec<usize> = p[3..].bytes().map(|b| (b - b'0') as usize).collect();
            let sgc = if p.starts_with("nfg") { 2 } else { 1 };
            RealWeightMode::KronosNfPlanesGptq {
                qkv: d[0],
                o: d[1],
                gate_up: d[2],
                down: d[3],
                head: d[4],
                scale_group_chunks: sgc,
            }
        }
        p if p.len() == 6 && p.starts_with('p') && p[1..].bytes().all(|b| (b'4'..=b'6').contains(&b)) => {
            let d: Vec<usize> = p[1..].bytes().map(|b| (b - b'0') as usize).collect();
            RealWeightMode::KronosBipolarGptqPlanes { qkv: d[0], o: d[1], gate_up: d[2], down: d[3], head: d[4] }
        }
        other => {
            return Err(format!(
                "unknown mode {other}; use nf5, nf5g128, nf4, nf4g128, nf3nf4, nf3nf4g128, nf3, nf3g128, e8q2, e8q2g128, e8q2nf4, e8q2nf4g128, q2c, nfpQOGDH, nfgQOGDH (digits 2-5), vq2, vq2g128, v2b4, v2b5, b4, b5, b6 or pQOGDH (digits 4-6)"
            ));
        }
    })
}

/// Compiles `model_dir` with streaming layer-by-layer Hadamard GPTQ on `calib_text` (one-time, `< 10 GiB` peak RAM even on `14B`).
pub fn compile(model_dir: &Path, mode: RealWeightMode, calib_text: &str, calib_tokens: usize) -> Result<RealLlamaEngine, String> {
    const SEQ: usize = 512;
    let tok = BpeTokenizer::from_model_dir(model_dir).map_err(|e| format!("tokenizer: {e}"))?;
    let ids = tok.encode(calib_text);
    if ids.len() < SEQ {
        return Err("calibration text is too short (need at least 512 tokens)".to_string());
    }
    let seqs: Vec<Vec<usize>> = ids[..calib_tokens.min(ids.len())].chunks(SEQ).map(|c| c.to_vec()).collect();
    if mode == RealWeightMode::KronosQ2c {
        return RealLlamaEngine::compile_q2c_from_dir(model_dir, &seqs, crate::kernels::thread_pool::physical_p_cores())
            .map_err(|e| format!("q2c compile: {e}"));
    }
    RealLlamaEngine::compile_streaming_gptq_from_dir(
        model_dir,
        mode,
        &seqs,
        crate::kernels::thread_pool::physical_p_cores(),
    )
    .map_err(|e| format!("streaming GPTQ compile: {e}"))
}

/// Compiles a Sparse-MoE `model_dir` into a `K2TKRONOS5\n` pinned core (`.k2tb`) and `F_NOCACHE` expert shard (`.k2tm`).
pub fn compile_moe(
    model_dir: &Path,
    core_k2tb_path: &Path,
    experts_k2tm_path: &Path,
    mode: RealWeightMode,
    calib_text: &str,
    calib_tokens: usize,
) -> Result<(), String> {
    const SEQ: usize = 512;
    let tok = BpeTokenizer::from_model_dir(model_dir).map_err(|e| format!("tokenizer: {e}"))?;
    let ids = tok.encode(calib_text);
    if ids.len() < SEQ {
        return Err("calibration text is too short (need at least 512 tokens)".to_string());
    }
    let seqs: Vec<Vec<usize>> = ids[..calib_tokens.min(ids.len())]
        .chunks(SEQ)
        .map(|c| c.to_vec())
        .collect();
    RealLlamaEngine::compile_streaming_moe_to_files(
        model_dir,
        core_k2tb_path,
        experts_k2tm_path,
        mode,
        &seqs,
        crate::kernels::thread_pool::physical_p_cores(),
    )
    .map_err(|e| format!("streaming Sparse-MoE GPTQ compile: {e}"))
}

/// Loads the target engine: a Qwen3.5 hybrid when `model_dir` is one, else the Llama-family engine.
fn load_target(
    model_dir: &Path,
    cache_dir: Option<&Path>,
    mode_key: &str,
    ctx: usize,
    threads: usize,
    calib: Option<(&Path, usize)>,
) -> Result<Backend, String> {
    if let Some(q) = crate::runtime::backend::load_qwen35(model_dir, cache_dir, mode_key, ctx, calib) {
        return q;
    }
    load_or_compile_engine(model_dir, cache_dir, mode_key, ctx, threads, calib).map(Backend::Llama)
}

fn load_or_compile_engine(
    model_dir: &Path,
    cache_dir: Option<&Path>,
    mode_key: &str,
    ctx: usize,
    threads: usize,
    calib: Option<(&Path, usize)>,
) -> Result<RealLlamaEngine, String> {
    let mode = parse_mode(mode_key)?;
    let probe_cfg = crate::model::real_llm::LlamaConfig::from_config_json(
        &model_dir.join("config.json"),
        ctx.max(16),
    )
    .map_err(|e| format!("config.json: {e}"))?;

    if probe_cfg.is_moe() {
        let (core_cache, experts_cache) = match cache_dir {
            Some(dir) => {
                let tag = model_dir
                    .file_name()
                    .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
                (
                    dir.join(format!("{tag}-kronos-moe-core-{mode_key}.k2tb")),
                    dir.join(format!("{tag}-kronos-moe-experts-{mode_key}.k2tm")),
                )
            }
            None => (
                model_dir.join(format!("kronos-moe-core-{mode_key}.k2tb")),
                model_dir.join(format!("kronos-moe-experts-{mode_key}.k2tm")),
            ),
        };
        // Map the whole shard when it fits in RAM so Metal can read experts in place.
        let ram = crate::model::memory_budget::MemoryBudgetProfile::active().physical_ram_bytes;
        let shard_len = std::fs::metadata(&experts_cache).map_or(0, |m| m.len() as usize);
        let policy = if shard_len > 0 && shard_len <= ram / 10 * 7 {
            crate::model::sparse_moe::MoeExpertOpenPolicy::ForceZeroCopyMmap
        } else {
            crate::model::sparse_moe::MoeExpertOpenPolicy::AutoBudget
        };
        let mut engine = match RealLlamaEngine::load_moe_from_files(
            model_dir,
            &core_cache,
            &experts_cache,
            ctx,
            policy,
        ) {
            Ok(e) => e,
            Err(_) => {
                let (path, n) = calib.ok_or_else(|| {
                    format!(
                        "no compiled Sparse-MoE cache at {} / {}; provide calibration text to compile it",
                        core_cache.display(),
                        experts_cache.display()
                    )
                })?;
                let text = std::fs::read_to_string(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                eprintln!(
                    "compiling Sparse-MoE {} ({mode_key}) to .k2tb + .k2tm...",
                    model_dir.display()
                );
                if let Some(parent) = core_cache.parent() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("{}: {e}", parent.display()))?;
                }
                compile_moe(model_dir, &core_cache, &experts_cache, mode, &text, n)?;
                RealLlamaEngine::load_moe_from_files(
                    model_dir,
                    &core_cache,
                    &experts_cache,
                    ctx,
                    policy,
                )
                .map_err(|e| e.to_string())?
            }
        };
        engine.set_num_threads(threads);
        return Ok(engine);
    }

    let cache = match cache_dir {
        Some(dir) => {
            let tag = model_dir
                .file_name()
                .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
            dir.join(format!("{tag}-kronos-{mode_key}.k2tb"))
        }
        None => model_dir.join(format!("kronos-{mode_key}.k2tb")),
    };
    let mut engine = match RealLlamaEngine::load_bipolar(model_dir, &cache, ctx) {
        Ok(e) => e,
        Err(_) => {
            let (path, n) = calib.ok_or_else(|| {
                format!("no compiled model at {}; provide calibration text to compile it", cache.display())
            })?;
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!("compiling {} ({mode_key}), one time only; this takes minutes...", model_dir.display());
            if let Some(parent) = cache.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
            }
            compile(model_dir, mode, &text, n)?
                .save_bipolar(&cache)
                .map_err(|e| format!("save {}: {e}", cache.display()))?;
            RealLlamaEngine::load_bipolar(model_dir, &cache, ctx).map_err(|e| e.to_string())?
        }
    };
    engine.repack_sdot();
    engine.set_num_threads(threads);
    Ok(engine)
}

struct DraftEngine {
    engine: RealLlamaEngine,
    cached: Vec<usize>,
    shared_vocab: usize,
    draft_vocab: usize,
}

/// Where the speculative tokens of one verify round came from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SpecSource {
    None,
    Lookup,
    Draft,
    Mtp,
}

/// Drops target and draft KV entries past `n` after a rejected speculative token.
fn rollback_spec(engine: &mut Backend, draft: &mut Option<DraftEngine>, n: usize) {
    engine.truncate_context(n);
    if let Some(d) = draft.as_mut() {
        d.engine.truncate_context(n);
        d.cached.truncate(n);
    }
}

/// Reusable zero-allocation sampling & speculative verification scratch buffers.
struct ChatScratch {
    cur_logits: Vec<f32>,
    pen_logits: Vec<f32>,
    verify_logits: Vec<f32>,
    top_heap: Vec<(usize, f32)>,
    target_dist: Vec<(usize, f32)>,
    draft_dists: Vec<Vec<(usize, f32)>>,
}

impl ChatScratch {
    fn new(vocab_size: usize) -> Self {
        Self {
            cur_logits: vec![0.0f32; vocab_size],
            pen_logits: vec![0.0f32; vocab_size],
            verify_logits: vec![0.0f32; (SPEC_LOOKUP_K_MAX + 1) * vocab_size],
            top_heap: Vec::with_capacity(128),
            target_dist: Vec::with_capacity(128),
            draft_dists: (0..SPEC_DRAFT_K_GREEDY).map(|_| Vec::with_capacity(128)).collect(),
        }
    }
}

/// Prefill checkpoint spacing for Qwen3.5 (see `ChatModel::prefill_tail`).
const Q35_CKPT_TOKENS: usize = 1024;

/// Byte-budgeted LRU of KV snapshots keyed by their token prefix.
struct ConvSnapshots {
    entries: Vec<(Vec<usize>, PrefixSnap, u64)>,
    budget_bytes: usize,
    clock: u64,
}

impl ConvSnapshots {
    /// Switching conversations is only worth a snapshot when the lost prefix is at least this long.
    const MIN_TOKENS: usize = 256;

    fn new() -> Self {
        let mib = std::env::var("KRONOS_KV_SNAP_MIB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024);
        Self { entries: Vec::new(), budget_bytes: mib << 20, clock: 0 }
    }

    fn common_prefix(a: &[usize], b: &[usize]) -> usize {
        a.iter().zip(b).take_while(|(x, y)| x == y).count()
    }

    /// Index and usable length of the snapshot that covers the longest prefix of `prompt`.
    fn best(&self, prompt: &[usize]) -> Option<(usize, usize)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(i, (toks, snap, _))| (i, Backend::usable_prefix(snap, Self::common_prefix(toks, prompt))))
            .max_by_key(|&(_, n)| n)
            .filter(|&(_, n)| n > 0)
    }

    fn touch(&mut self, i: usize) {
        self.clock += 1;
        self.entries[i].2 = self.clock;
    }

    /// Marks every snapshot that is a prefix of `prompt` as used, so the shared system-prompt point
    /// outlives the conversation snapshots built on it.
    fn touch_prefixes(&mut self, prompt: &[usize]) {
        self.clock += 1;
        for e in &mut self.entries {
            if prompt.starts_with(&e.0) {
                e.2 = self.clock;
            }
        }
    }

    fn insert(&mut self, toks: Vec<usize>, snap: PrefixSnap) {
        if snap.byte_len() == 0 || snap.byte_len() > self.budget_bytes {
            return;
        }
        // A snapshot that is a prefix of a stored one adds nothing; one that extends a stored one replaces it.
        // Qwen3.5 snapshots resume only at their own length, so each prefix point is kept on its own.
        let exact_only = matches!(snap, PrefixSnap::Qwen35(_));
        if self.entries.iter().any(|(t, _, _)| if exact_only { *t == toks } else { t.len() >= toks.len() && t[..toks.len()] == toks[..] }) {
            return;
        }
        if !exact_only {
            self.entries.retain(|(t, _, _)| !(t.len() <= toks.len() && toks[..t.len()] == t[..]));
        }
        while self.entries.iter().map(|e| e.1.byte_len()).sum::<usize>() + snap.byte_len() > self.budget_bytes {
            let Some(oldest) = self.entries.iter().enumerate().min_by_key(|(_, e)| (e.2, std::cmp::Reverse(e.0.len()))).map(|(i, _)| i) else { break };
            self.entries.remove(oldest);
        }
        self.clock += 1;
        self.entries.push((toks, snap, self.clock));
    }
}

pub struct ChatModel {
    engine: Backend,
    /// Prompt-lookup index over the live context (doc `look_ctx`) and recent conversations; drafts
    /// Qwen3.5 speculation.
    look: crate::model::lookup::SessionLookup,
    look_ctx: usize,
    /// Qwen3.5-style template: the open assistant turn starts with an empty `<think>` block.
    think_block: bool,
    /// `KRONOS_THINK=1`: open a real `<think>` block instead of the empty one; reasoning is
    /// stripped before tool-call parsing.
    thinking: bool,
    draft: Option<DraftEngine>,
    tok: BpeTokenizer,
    im_start: usize,
    im_end: usize,
    /// Token ids currently in the target KV cache.
    cached: Vec<usize>,
    /// Small 2-slot LRU KV prefix cache (`<= 80` tokens/slot) accelerating repeated `Kronos-Jev` calls.
    jev_caches: Vec<(Vec<usize>, crate::model::real_llm::KvPrefixSnapshot)>,
    /// Qwen3.5 Jev states after the scorer system prompt, most recent first.
    jev_q35: Vec<(Vec<usize>, crate::backend::metal_qwen35::StateSnapshot)>,
    /// KV snapshots of recent conversations, so switching back restores instead of re-prefilling.
    conv_snaps: ConvSnapshots,
    /// Generated token ids per assistant message, keyed by the hash of the canonical ChatML text up to
    /// and including it, so the re-rendered history reuses the KV written during decoding.
    gen_memo: Vec<(u64, Vec<usize>)>,
    last_gen: Option<Vec<usize>>,
    prev_reply: Vec<usize>,
    ctx: usize,
    scratch: ChatScratch,
    /// Lookup and draft speculation (off = plain token-by-token decoding).
    spec_enabled: bool,
    /// Per-session KV prefixes (`--kv-sessions`), keyed by session id.
    sessions: Option<SessionKvStore>,
    active_session: Option<String>,
    pub name: String,
    tool_format: ToolFormat,
    style: PromptStyle,
    #[cfg(all(feature = "ane", target_os = "macos"))]
    ane: Vec<(usize, crate::backend::coreml::CoreMlModel)>,
}

impl ChatModel {
    /// Loads `<model_dir>/kronos-<mode>.k2tb`, compiling it first from `calib` (text file, tokens)
    /// when the cache is missing.
    pub fn load(
        model_dir: &Path,
        mode_key: &str,
        ctx: usize,
        threads: usize,
        calib: Option<(&Path, usize)>,
    ) -> Result<Self, String> {
        Self::load_with_cache_dir(model_dir, None, mode_key, ctx, threads, calib)
    }

    /// Like [`Self::load`], with `.k2tb` caches in `cache_dir` if given, else `model_dir`.
    pub fn load_with_cache_dir(
        model_dir: &Path,
        cache_dir: Option<&Path>,
        mode_key: &str,
        ctx: usize,
        threads: usize,
        calib: Option<(&Path, usize)>,
    ) -> Result<Self, String> {
        let mut engine = load_target(model_dir, cache_dir, mode_key, ctx, threads, calib)?;
        crate::model::memory_budget::mem_trace("load_target done");
        // Hadamard KV cache (`Q4` for `> 1B` models under `<= 8 GB` budget profile, `Q8` otherwise).
        if !engine.is_qwen35() && matches!(engine.head_dim(), 64 | 128) {
            let kv_bits = if crate::model::memory_budget::MemoryBudgetProfile::active().prefer_q4_kv
                && engine.hidden_size() > 1024
            {
                4
            } else {
                8
            };
            engine.set_kv_bits(kv_bits);
        }
        let vocab_size = engine.vocab_size();
        crate::model::memory_budget::mem_trace("kv bits set");
        let mut tok = BpeTokenizer::from_model_dir(model_dir).map_err(|e| format!("tokenizer: {e}"))?;
        crate::model::memory_budget::mem_trace("tokenizer loaded");
        let mut emb_probe = vec![0.0f32; engine.hidden_size()];
        tok.retain_added_tokens(|s, id| {
            if s.starts_with("<|") && s.ends_with("|>") {
                return true;
            }
            if id >= vocab_size {
                return false;
            }
            engine.embed(id, &mut emb_probe);
            emb_probe.iter().any(|&v| v != 0.0)
        });
        let special =
            |s: &str| tok.token_id(s).ok_or(format!("tokenizer lacks {s}; use an Instruct model"));
        let style = PromptStyle::detect(&tok);
        // HunYuan: the user-turn marker is masked like `<|im_start|>` and `<eos>` ends the reply.
        let (im_start, im_end) = match style {
            PromptStyle::ChatMl => (special(crate::config::markup::IM_START)?, special(crate::config::markup::IM_END)?),
            PromptStyle::HunYuan => (special(crate::config::markup::HY_USER)?, special(crate::config::markup::HY_EOS)?),
        };
        let name = format!(
            "{}-kronos-{mode_key}",
            model_dir.file_name().map_or("model".into(), |n| n.to_string_lossy().to_lowercase())
        );
        let think_block = std::fs::read_to_string(model_dir.join("chat_template.jinja"))
            .or_else(|_| std::fs::read_to_string(model_dir.join("tokenizer_config.json")))
            .is_ok_and(|t| t.contains("enable_thinking"));
        Ok(Self {
            engine,
            think_block,
            thinking: think_block && std::env::var("KRONOS_THINK").is_ok_and(|v| v == "1"),
            look: crate::model::lookup::SessionLookup::new(),
            look_ctx: 0,
            draft: None,
            tok,
            im_start,
            im_end,
            cached: Vec::new(),
            jev_caches: Vec::with_capacity(2),
            jev_q35: Vec::new(),
            conv_snaps: ConvSnapshots::new(),
            gen_memo: Vec::new(),
            last_gen: None,
            prev_reply: Vec::new(),
            ctx,
            scratch: { crate::model::memory_budget::mem_trace("before chat scratch"); ChatScratch::new(vocab_size) },
            spec_enabled: true,
            sessions: None,
            active_session: None,
            name,
            tool_format: ToolFormat::detect(model_dir),
            style,
            #[cfg(all(feature = "ane", target_os = "macos"))]
            ane: Vec::new(),
        })
    }

    /// Toggles the Hadamard INT8 KV cache.
    pub fn enable_q8_kv_cache(&mut self, enable: bool) {
        self.engine.enable_q8_kv_cache(enable);
        self.cached.clear();
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
        if let Some(d) = self.draft.as_mut() {
            d.engine.enable_q8_kv_cache(enable);
            d.cached.clear();
        }
    }

    pub fn is_q8_kv_enabled(&self) -> bool {
        self.engine.is_q8_kv_enabled()
    }

    /// Sets KV cache quantization bit-width (`4`, `8`, or `32`) on target and draft engines.
    pub fn set_kv_bits(&mut self, bits: u8) {
        self.engine.set_kv_bits(bits);
        self.cached.clear();
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
        if let Some(d) = self.draft.as_mut() {
            d.engine.set_kv_bits(bits);
            d.cached.clear();
        }
    }

    pub fn kv_bits(&self) -> u8 {
        self.engine.kv_bits()
    }

    /// Configures a sliding-window + attention-sink KV ring buffer on target and draft engines.
    pub fn set_kv_sliding_window(&mut self, window: Option<usize>, sink_tokens: usize) {
        self.engine.set_kv_sliding_window(window, sink_tokens);
        self.cached.clear();
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
        if let Some(d) = self.draft.as_mut() {
            d.engine.set_kv_sliding_window(window, sink_tokens);
            d.cached.clear();
        }
    }

    pub fn kv_sliding_window(&self) -> Option<usize> {
        self.engine.kv_sliding_window().map(|(w, _)| w)
    }

    pub fn is_moe(&self) -> bool {
        self.engine.is_moe()
    }

    pub fn moe_stats(&self) -> Option<crate::model::sparse_moe::MoeRuntimeStats> {
        self.engine.moe_stats()
    }

    /// Enables zero-copy Metal GPU inference on the target model; small draft models stay on CPU.
    pub fn enable_metal(&mut self) -> Result<(), String> {
        self.engine.enable_metal()?;
        self.cached.clear();
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
        let target_bits = if self.engine.kv_bits() == 4 { 4 } else { 8 };
        let target_win = self.engine.kv_sliding_window();
        if let Some(d) = self.draft.as_mut() {
            if d.engine.cfg.hidden_size > 1536 {
                let _ = d.engine.enable_metal();
            } else {
                d.engine.disable_metal();
                if matches!(d.engine.cfg.head_dim(), 64 | 128) {
                    let draft_bits = if d.engine.cfg.hidden_size > 1024 {
                        target_bits
                    } else {
                        8
                    };
                    d.engine.set_kv_bits(draft_bits);
                    if let Some((w, s)) = target_win {
                        d.engine.set_kv_sliding_window(Some(w), s);
                    }
                }
                d.engine
                    .set_num_threads(crate::kernels::thread_pool::default_inference_threads());
                crate::kernels::thread_pool::SpinThreadPool::park_idle_workers();
            }
            d.cached.clear();
        }
        Ok(())
    }

    /// Disables Metal GPU inference and restores CPU execution.
    pub fn disable_metal(&mut self) {
        self.engine.disable_metal();
        self.cached.clear();
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
        if let Some(d) = self.draft.as_mut() {
            d.engine.disable_metal();
            d.cached.clear();
        }
    }

    pub fn is_metal_enabled(&self) -> bool {
        self.engine.is_metal_enabled()
    }

    pub fn resident_weight_bytes(&self) -> usize {
        self.engine.resident_weight_bytes()
    }

    pub fn kv_cache_bytes(&self) -> usize {
        self.engine.kv_cache_bytes()
    }

    pub fn mlock_weights(&self) -> usize {
        self.engine.mlock_weights()
    }

    pub fn tool_format(&self) -> ToolFormat {
        self.tool_format
    }

    pub fn default_system(&self) -> &'static str {
        match self.tok.pre_tokenizer {
            PreTokenizerKind::Qwen2 => DEFAULT_SYSTEM_QWEN,
            PreTokenizerKind::SmolLm2DigitsGpt2 => DEFAULT_SYSTEM,
            // Hy-MT2 is trained without a system prompt.
            PreTokenizerKind::HunYuan => "",
        }
    }

    /// The chat template this model uses.
    pub fn prompt_style(&self) -> PromptStyle {
        self.style
    }

    /// Clears the target and draft KV caches.
    pub fn reset_cache(&mut self) {
        self.engine.reset_context();
        self.cached.clear();
        if let Some(d) = self.draft.as_mut() {
            d.engine.reset_context();
            d.cached.clear();
        }
    }

    pub fn has_draft(&self) -> bool {
        self.draft.is_some()
    }

    /// Enables or disables lookup and draft speculation; output is identical either way.
    pub fn set_speculation(&mut self, enabled: bool) {
        self.spec_enabled = enabled;
    }

    /// Keeps the KV prefix of the last `slots` sessions in RAM and spills older ones under `dir/kv`.
    /// Qwen3.5 has no position-sliced KV export, so it relies on the snapshot LRU instead.
    pub fn enable_session_kv(&mut self, slots: usize, dir: Option<&Path>) {
        self.active_session = None;
        if slots == 0 || self.engine.is_qwen35() {
            self.sessions = None;
            return;
        }
        let tag = format!("{}:ctx{}", self.name, self.ctx);
        self.sessions = Some(SessionKvStore::new(slots, dir.map(|d| d.join("kv")), tag));
    }

    pub fn session_kv_enabled(&self) -> bool {
        self.sessions.is_some()
    }

    /// Bytes held by in-RAM session prefixes (excluding the live engine cache).
    pub fn session_kv_bytes(&self) -> usize {
        self.sessions.as_ref().map_or(0, SessionKvStore::resident_bytes)
    }

    /// Makes `id` the active session, restoring its KV prefix from RAM or disk when it is a
    /// different session. Returns the restored token count and where it came from.
    pub fn switch_session(&mut self, id: &str) -> (usize, KvSource) {
        if self.sessions.is_none() {
            return (0, KvSource::None);
        }
        if self.active_session.as_deref() == Some(id) {
            return (self.cached.len(), KvSource::Active);
        }
        self.save_active_session();
        let target_layout = self.engine.kv_blob_layout();
        let draft_layout = self.draft.as_ref().and_then(|d| d.engine.kv_blob_layout());
        let store = self.sessions.as_mut().unwrap();
        let mut source = KvSource::None;
        let mut entry = store.take(id).inspect(|_| source = KvSource::Memory);
        if entry.is_none() {
            if let Some(tl) = target_layout.as_ref() {
                entry = store.load(id, tl, draft_layout.as_ref()).inspect(|_| source = KvSource::Disk);
            }
        }

        self.engine.reset_context();
        self.cached.clear();
        self.jev_caches.clear();
        if let Some(d) = self.draft.as_mut() {
            d.engine.reset_context();
            d.cached.clear();
        }
        let mut restored = 0usize;
        if let Some(e) = entry.as_mut() {
            let t = &e.target;
            if self.engine.import_kv_positions(&t.layout, &t.data, t.len()) {
                self.cached = t.tokens.clone();
                restored = t.len();
                if let (Some(d), Some(db)) = (self.draft.as_mut(), e.draft.as_ref()) {
                    if d.engine.import_kv_positions(&db.layout, &db.data, db.len()) {
                        d.cached = db.tokens.clone();
                    } else {
                        d.engine.reset_context();
                    }
                }
                if e.draft.is_some() && self.draft.is_none() {
                    e.draft = None;
                }
            } else {
                entry = None;
                source = KvSource::None;
            }
        }
        let store = self.sessions.as_mut().unwrap();
        if let Some(e) = entry {
            store.insert(e);
        }
        self.active_session = Some(id.to_string());
        store.evict_to_capacity(id);
        (restored, source)
    }

    /// Drops the stored KV prefix of `id` from RAM and disk.
    pub fn forget_session(&mut self, id: &str) {
        if let Some(store) = self.sessions.as_mut() {
            store.remove(id);
        }
        if self.active_session.as_deref() == Some(id) {
            self.active_session = None;
        }
    }

    /// Copies the live KV tail of the active session into its RAM entry and appends it on disk.
    fn save_active_session(&mut self) {
        let (Some(id), Some(store)) = (self.active_session.as_deref(), self.sessions.as_mut()) else {
            return;
        };
        let Some(tl) = self.engine.kv_blob_layout() else { return };
        let live = self.cached.len().min(self.engine.position());
        let e = store.upsert(id, tl);
        if e.target.layout != tl {
            e.target = KvBlob::new(tl);
        }
        let common = e.target.common_prefix(&self.cached[..live]);
        e.target.truncate(common);
        if live > common {
            let eng = &self.engine;
            e.target.extend(&self.cached[common..live], |out| eng.export_kv_positions(common, live, out));
        }
        if let Some(d) = self.draft.as_ref() {
            if let Some(dl) = d.engine.kv_blob_layout() {
                let d_live = d.cached.len().min(d.engine.position());
                let db = e.draft.get_or_insert_with(|| KvBlob::new(dl));
                if db.layout != dl {
                    *db = KvBlob::new(dl);
                }
                let dc = db.common_prefix(&d.cached[..d_live]);
                db.truncate(dc);
                if d_live > dc {
                    let deng = &d.engine;
                    db.extend(&d.cached[dc..d_live], |out| deng.export_kv_positions(dc, d_live, out));
                }
            }
        }
        if let Err(err) = store.persist_id(id) {
            eprintln!("kv persist {id}: {err}");
        }
    }

    pub fn detach_draft(&mut self) {
        self.draft = None;
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
    }

    /// Attaches a draft model for lossless speculative decoding.
    pub fn attach_draft(
        &mut self,
        draft_dir: &Path,
        draft_mode_key: &str,
        calib: Option<(&Path, usize)>,
    ) -> Result<(), String> {
        self.attach_draft_with_cache_dir(draft_dir, None, draft_mode_key, calib)
    }

    /// Like [`Self::attach_draft`], with caches in `cache_dir` if given.
    pub fn attach_draft_with_cache_dir(
        &mut self,
        draft_dir: &Path,
        cache_dir: Option<&Path>,
        draft_mode_key: &str,
        calib: Option<(&Path, usize)>,
    ) -> Result<(), String> {
        if self.engine.is_qwen35() {
            return Err("Qwen3.5 drafts with prompt lookup; separate draft models are not supported".to_string());
        }
        let draft_tok =
            BpeTokenizer::from_model_dir(draft_dir).map_err(|e| format!("draft tokenizer: {e}"))?;
        let threads = if self.engine.is_metal_enabled() {
            crate::kernels::thread_pool::default_inference_threads()
        } else {
            self.engine.num_threads()
        };
        let mut draft_eng =
            load_or_compile_engine(draft_dir, cache_dir, draft_mode_key, self.ctx, threads, calib)?;
        let draft_vocab = draft_eng.cfg.vocab_size;
        let target_vocab = self.engine.vocab_size();
        let shared_vocab = draft_vocab.min(target_vocab);
        let check_len = draft_tok.vocab_len().min(self.tok.vocab_len()).min(shared_vocab);
        if check_len == 0
            || draft_tok.pre_tokenizer != self.tok.pre_tokenizer
            || !draft_tok.same_vocab_prefix(&self.tok, check_len)
        {
            return Err(format!(
                "draft tokenizer mismatch over shared {check_len} ids (draft vocab {draft_vocab}, target vocab {target_vocab})"
            ));
        }
        if self.scratch.pen_logits.len() < shared_vocab {
            self.scratch.pen_logits.resize(shared_vocab, 0.0);
        }
        let target_bits = if self.engine.kv_bits() == 4 { 4 } else { 8 };
        let target_win = self.engine.kv_sliding_window();
        if self.engine.is_metal_enabled() && draft_eng.cfg.hidden_size > 1536 {
            let _ = draft_eng.enable_metal();
            draft_eng.set_kv_bits(target_bits);
            if let Some((w, s)) = target_win {
                draft_eng.set_kv_sliding_window(Some(w), s);
            }
        } else if (self.engine.is_q8_kv_enabled() || self.engine.is_metal_enabled())
            && matches!(draft_eng.cfg.head_dim(), 64 | 128)
        {
            let draft_bits = if draft_eng.cfg.hidden_size > 1024 {
                target_bits
            } else {
                8
            };
            draft_eng.set_kv_bits(draft_bits);
            if let Some((w, s)) = target_win {
                draft_eng.set_kv_sliding_window(Some(w), s);
            }
        }
        if self.engine.is_metal_enabled() {
            crate::kernels::thread_pool::SpinThreadPool::park_idle_workers();
        }
        self.jev_caches.clear();
        self.conv_snaps.entries.clear();
        self.draft = Some(DraftEngine {
            engine: draft_eng,
            cached: Vec::new(),
            shared_vocab,
            draft_vocab,
        });
        Ok(())
    }

    /// CPU threads the engine uses (the Metal path keeps the default pool size).
    pub fn num_threads(&self) -> usize {
        if self.engine.is_metal_enabled() {
            crate::kernels::thread_pool::default_inference_threads()
        } else {
            self.engine.num_threads()
        }
    }

    pub fn context_size(&self) -> usize {
        self.ctx
    }

    pub fn cached_prefix_len(&self) -> usize {
        self.cached.len()
    }

    pub fn count_prompt_tokens(&self, messages: &[Message], tools_json: &[String]) -> usize {
        self.render_with_tools(messages, tools_json).len()
    }

    /// Returns true if the target KV cache already holds the exact `system + tools` prefix.
    pub fn has_hot_prefix(&self, system_prompt: &str, tools_json: &[String]) -> bool {
        let full = render_prompt_with(&[Message::new("system", system_prompt)], tools_json, "", self.tool_format, self.style);
        let prefix_str = match self.style {
            PromptStyle::ChatMl => full.strip_suffix("assistant\n").unwrap_or(&full),
            PromptStyle::HunYuan => full.strip_suffix(self.style.assistant_open()).unwrap_or(&full),
        };
        let tokens = self.tok.encode(prefix_str);
        !tokens.is_empty()
            && self.cached.len() >= tokens.len()
            && self.cached[..tokens.len()] == tokens[..]
    }

    /// Evaluates first-token log-probabilities for `anchors` in a single forward pass (`0` decode tokens)
    /// while preserving both target and draft KV caches. Returns `(anchor_logprobs, captured_mass)`.
    pub fn jev_score_anchors(
        &mut self,
        system_hint: &str,
        user_query: &str,
        anchors: &[&str],
    ) -> Option<(Vec<f32>, f32)> {
        if anchors.len() < 2 {
            return None;
        }
        let prompt_str = render_chatml_prompt(
            &[
                Message::new("system", system_hint),
                Message::new("user", user_query),
            ],
            &[],
            "",
        );
        let raw_tokens = self.tok.encode(&prompt_str);
        if raw_tokens.is_empty() || raw_tokens.len() > 768 || raw_tokens.len() >= self.ctx {
            return None;
        }

        let vocab_cap = if let Some(d) = self.draft.as_ref() {
            d.draft_vocab
        } else {
            self.engine.vocab_size()
        };

        let mut candidate_sets: Vec<Vec<usize>> = Vec::with_capacity(anchors.len());
        for &a in anchors {
            let trimmed = a.trim();
            if trimmed.is_empty() {
                return None;
            }
            let lower = trimmed.to_ascii_lowercase();
            let variants = [
                trimmed.to_string(),
                format!(" {trimmed}"),
                lower.clone(),
                format!(" {lower}"),
            ];
            let mut ids = Vec::with_capacity(4);
            for v in &variants {
                let enc = self.tok.encode(v);
                if enc.len() == 1 && enc[0] < vocab_cap && !ids.contains(&enc[0]) {
                    ids.push(enc[0]);
                }
            }
            if ids.is_empty() {
                for v in &variants {
                    let enc = self.tok.encode(v);
                    if let Some(&first) = enc.first() {
                        if first < vocab_cap && !ids.contains(&first) {
                            ids.push(first);
                        }
                    }
                }
            }
            candidate_sets.push(ids);
        }

        // Remove any token ID shared between two distinct anchors so anchor sets stay disjoint.
        let mut unique_sets: Vec<Vec<usize>> = Vec::with_capacity(anchors.len());
        for (idx, set) in candidate_sets.iter().enumerate() {
            let filtered: Vec<usize> = set
                .iter()
                .copied()
                .filter(|id| {
                    !candidate_sets
                        .iter()
                        .enumerate()
                        .any(|(other_idx, other)| other_idx != idx && other.contains(id))
                })
                .collect();
            if filtered.is_empty() {
                return None;
            }
            unique_sets.push(filtered);
        }

        let score_logits = |logits: &[f32]| -> (Vec<f32>, f32) {
            let mut max_z = f32::NEG_INFINITY;
            for &z in logits {
                if z > max_z {
                    max_z = z;
                }
            }
            let mut sum_exp = 0.0f64;
            for &z in logits {
                sum_exp += ((z - max_z) as f64).exp();
            }
            let log_z = (max_z as f64) + sum_exp.ln();
            let mut out = Vec::with_capacity(unique_sets.len());
            let mut mass = 0.0f64;
            for set in &unique_sets {
                let mut anchor_exp = 0.0f64;
                for &id in set {
                    if let Some(&z) = logits.get(id) {
                        anchor_exp += ((z as f64) - log_z).exp();
                    }
                }
                let p = anchor_exp.clamp(1e-12, 1.0);
                mass += p;
                out.push(p.ln() as f32);
            }
            (out, mass.clamp(0.0, 1.0) as f32)
        };

        use crate::config::limits::{JEV_CACHE_SLOTS, JEV_PREFIX_CAP};

        fn eval_with_jev_prefix<F, R>(
            eng: &mut RealLlamaEngine,
            saved_total: usize,
            tokens: &[usize],
            caches: &mut Vec<(Vec<usize>, crate::model::real_llm::KvPrefixSnapshot)>,
            score_fn: F,
        ) -> Option<R>
        where
            F: FnOnce(&[f32]) -> R,
        {
            let overlap = saved_total.min(tokens.len());
            let snap = eng.save_kv_prefix(overlap)?;

            let max_reuse = tokens.len().saturating_sub(1);
            let mut best_slot: Option<(usize, usize)> = None;
            for (idx, (cached_toks, _)) in caches.iter().enumerate() {
                let common = cached_toks
                    .iter()
                    .zip(tokens)
                    .take_while(|(a, b)| a == b)
                    .count()
                    .min(max_reuse);
                if common >= 8 && best_slot.is_none_or(|(_, best_c)| common > best_c) {
                    best_slot = Some((idx, common));
                }
            }

            let start_pos = if let Some((idx, common)) = best_slot {
                eng.restore_kv_prefix_slice(&caches[idx].1, common, common);
                if idx > 0 {
                    let entry = caches.remove(idx);
                    caches.insert(0, entry);
                }
                common
            } else {
                eng.reset_context();
                0
            };

            let res = {
                let logits = eng.forward_prompt(&tokens[start_pos..]);
                score_fn(logits)
            };

            let save_len = max_reuse.min(JEV_PREFIX_CAP);
            if start_pos < 16 && save_len >= 12 {
                if let Some(jev_snap) = eng.save_linear_kv_prefix(save_len) {
                    if caches.len() >= JEV_CACHE_SLOTS {
                        caches.pop();
                    }
                    caches.insert(0, (tokens[..save_len].to_vec(), jev_snap));
                }
            }

            eng.restore_kv_prefix(&snap, saved_total);
            Some(res)
        }

        let metal_active = self.engine.is_metal_enabled();
        if let Some(d) = self.draft.as_mut() {
            let dv = d.draft_vocab;
            let mapped: Vec<usize> = raw_tokens
                .iter()
                .map(|&id| if id < dv { id } else { 0 })
                .collect();
            let saved_total = d.cached.len();
            let res = eval_with_jev_prefix(
                &mut d.engine,
                saved_total,
                &mapped,
                &mut self.jev_caches,
                score_logits,
            )?;
            if metal_active {
                crate::kernels::thread_pool::SpinThreadPool::park_idle_workers();
            }
            return Some(res);
        }

        // Metal included: the scorer overwrites only the first `tokens.len()` KV positions, which are
        // snapshotted and restored, so the conversation KV (and `self.cached`) stay intact.
        let sys_ids = self.tok.encode(&format!("<|im_start|>system\n{system_hint}<|im_end|>\n"));
        let saved_total = self.cached.len();
        match &mut self.engine {
            Backend::Llama(e) => eval_with_jev_prefix(e, saved_total, &raw_tokens, &mut self.jev_caches, score_logits),
            Backend::Qwen35(q) => {
                // The recurrent state cannot be partly overwritten, so score from a clean context and put the
                // whole conversation state back afterwards.
                let saved = q.snapshot();
                let hit = self.jev_q35.iter().position(|(t, _)| t.len() < raw_tokens.len() && raw_tokens.starts_with(t));
                let start = if let Some(i) = hit {
                    let e = self.jev_q35.remove(i);
                    q.restore(&e.1);
                    let n = e.0.len();
                    self.jev_q35.insert(0, e);
                    n
                } else {
                    q.reset();
                    if sys_ids.len() >= 8 && sys_ids.len() < raw_tokens.len() && raw_tokens.starts_with(&sys_ids) {
                        q.feed(&sys_ids);
                        self.jev_q35.truncate(JEV_CACHE_SLOTS - 1);
                        self.jev_q35.insert(0, (sys_ids.clone(), q.snapshot()));
                        sys_ids.len()
                    } else {
                        0
                    }
                };
                let res = score_logits(q.forward_last(&raw_tokens[start..]));
                q.restore(&saved);
                Some(res)
            }
        }
    }

    /// Pre-warms the fixed system + tool-schema prefix into target and draft KV caches.
    pub fn prewarm_prefix(&mut self, system_prompt: &str, tools_json: &[String]) -> usize {
        let full = render_prompt_with(&[Message::new("system", system_prompt)], tools_json, "", self.tool_format, self.style);
        let prefix_str = match self.style {
            PromptStyle::ChatMl => full.strip_suffix("assistant\n").unwrap_or(&full),
            PromptStyle::HunYuan => full.strip_suffix(self.style.assistant_open()).unwrap_or(&full),
        };
        let tokens = self.tok.encode(prefix_str);
        if tokens.is_empty() || tokens.len() >= self.ctx {
            return 0;
        }
        let common = self.cached.iter().zip(&tokens).take_while(|(a, b)| a == b).count();
        if common == tokens.len() {
            return tokens.len();
        }
        // Snapshots a different warmed prefix instead of discarding it.
        let common = self.align_kv(&tokens);
        let tail = &tokens[common..];
        if self.engine.is_metal_enabled() && self.draft.is_some() {
            let t_eng = &mut self.engine;
            let cur_logits = &mut self.scratch.cur_logits;
            let d = self.draft.as_mut().unwrap();
            let dv = d.draft_vocab;
            let mut d_common = d
                .cached
                .iter()
                .zip(&tokens)
                .take_while(|(&a, &b)| a == if b < dv { b } else { 0 })
                .count();
            if !d.engine.can_truncate_kv_to(d_common) {
                d_common = 0;
            }
            d.engine.truncate_context(d_common);
            d.cached.truncate(d_common);
            let mapped: Vec<usize> = tokens[d_common..]
                .iter()
                .map(|&id| if id < dv { id } else { 0 })
                .collect();
            std::thread::scope(|s| {
                let dj = s.spawn(|| {
                    if !mapped.is_empty() {
                        d.engine.forward_prompt(&mapped);
                        d.cached.extend_from_slice(&mapped);
                    }
                });
                let l = t_eng.forward_prompt(tail);
                cur_logits.copy_from_slice(l);
                let _ = dj.join();
            });
        } else {
            self.prefill_tail(tail);
            if let Some(d) = self.draft.as_mut() {
                let dv = d.draft_vocab;
                let mut d_common = d
                    .cached
                    .iter()
                    .zip(&tokens)
                    .take_while(|(&a, &b)| a == if b < dv { b } else { 0 })
                    .count();
                if !d.engine.can_truncate_kv_to(d_common) {
                    d_common = 0;
                }
                d.engine.truncate_context(d_common);
                d.cached.truncate(d_common);
                let mapped: Vec<usize> = tokens[d_common..]
                    .iter()
                    .map(|&id| if id < dv { id } else { 0 })
                    .collect();
                if !mapped.is_empty() {
                    d.engine.forward_prompt(&mapped);
                    d.cached.extend_from_slice(&mapped);
                }
            }
        }
        self.cached.extend_from_slice(tail);
        self.cached.len()
    }

    /// Loads Neural Engine prefill models `prefill_s<N>.mlmodelc` from `dir` (see
    /// `ane/export_prefill.py`); returns the prompt-length buckets found.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn enable_ane(&mut self, dir: &Path) -> Result<Vec<usize>, String> {
        use crate::backend::coreml::{ComputeUnits, CoreMlModel};
        let entries = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            let Some(n) = name.strip_prefix("prefill_s").and_then(|r| r.strip_suffix(".mlmodelc")) else {
                continue;
            };
            let seq: usize = n.parse().map_err(|_| format!("bad bucket name {name}"))?;
            if seq <= self.ctx {
                let mut m = CoreMlModel::load(&e.path(), ComputeUnits::CpuAndNeuralEngine)?;
                // The first prediction pays a one-time Neural Engine setup cost.
                let h = self.engine.hidden_size();
                m.predict("x", &[1, seq, h], &vec![0.0; seq * h], &["hidden"])?;
                self.ane.push((seq, m));
            }
        }
        self.ane.sort_by_key(|(s, _)| *s);
        Ok(self.ane.iter().map(|(s, _)| *s).collect())
    }

    /// Prefills `prompt` from an empty cache on the Neural Engine, if a bucket fits.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    fn ane_prefill(&mut self, prompt: &[usize]) -> Option<Result<Vec<f32>, String>> {
        if self.engine.is_qwen35() {
            return None;
        }
        let n = prompt.len();
        let (seq, model) = self.ane.iter_mut().find(|(s, _)| *s >= n)?;
        let seq = *seq;
        let h = self.engine.hidden_size();
        let mut x = vec![0.0f32; seq * h];
        for (t, &tok) in prompt.iter().enumerate() {
            self.engine.embed(tok, &mut x[t * h..(t + 1) * h]);
        }
        Some(model.predict("x", &[1, seq, h], &x, &["hidden", "keys", "values"]).map(|out| {
            let Backend::Llama(e) = &mut self.engine else { unreachable!() };
            e.load_prefix_kv(&out[1], &out[2], seq, n);
            e.logits_from_hidden(&out[0][(n - 1) * h..n * h]).to_vec()
        }))
    }

    /// ChatML prompt ending with an open assistant turn; adds the default system prompt if none.
    pub fn render(&self, messages: &[Message]) -> Vec<usize> {
        self.render_with_tools(messages, &[])
    }

    /// Renders a ChatML / Qwen2.5 tool-calling prompt (including `<tools>` system block,
    /// `<tool_call>` assistant turns, and grouped `<tool_response>` user turns) into token IDs.
    pub fn render_with_tools(&self, messages: &[Message], tools_json: &[String]) -> Vec<usize> {
        self.render_tokens(messages, tools_json, "")
    }

    /// Token-level render: assistant messages whose generated ids are memoised are spliced in as
    /// decoded, everything else is encoded from the canonical ChatML text.
    fn render_tokens(&self, messages: &[Message], tools_json: &[String], assistant_prefix: &str) -> Vec<usize> {
        use std::hash::{Hash, Hasher};
        let mut ids = Vec::new();
        let mut buf = String::new();
        let mut hits = 0usize;
        let mut ctx_hash = std::collections::hash_map::DefaultHasher::new();
        render_chatml_pieces(messages, tools_json, self.default_system(), self.tool_format, self.style, &mut |piece| match piece {
            ChatPiece::Text(t) => {
                t.hash(&mut ctx_hash);
                buf.push_str(t);
            }
            ChatPiece::Assistant(m) => {
                let mut seg = String::new();
                push_assistant_segment(m, self.tool_format, self.style, &mut seg);
                seg.hash(&mut ctx_hash);
                let key = ctx_hash.clone().finish();
                match self.gen_memo.iter().rev().find(|(k, _)| *k == key) {
                    Some((_, toks)) => {
                        hits += 1;
                        buf.push_str(self.style.assistant_open());
                        if self.think_block {
                            buf.push_str(THINK_EMPTY);
                        }
                        ids.extend(self.tok.encode(&buf));
                        buf.clear();
                        ids.extend_from_slice(toks);
                        buf.push_str(self.style.assistant_close());
                    }
                    None if self.think_block => buf.push_str(&with_think_block(&seg)),
                    None => buf.push_str(&seg),
                }
            }
        });
        if self.thinking {
            buf.push_str("<think>\n");
        } else if self.think_block {
            buf.push_str(THINK_EMPTY);
        }
        buf.push_str(assistant_prefix);
        ids.extend(self.tok.encode(&buf));
        if std::env::var_os("KRONOS_KV_DEBUG").is_some() {
            if let Some(p) = std::env::var_os("KRONOS_PROMPT_DUMP") {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
                    let _ = writeln!(f, "===== {} tokens\n{}", ids.len(), self.tok.decode(&ids));
                }
            }
            let n_asst = messages.iter().filter(|m| m.role == "assistant").count();
            eprintln!("[kv] render {} tokens, {hits}/{n_asst} assistant messages from generated ids", ids.len());
        }
        ids
    }

    /// Records the ids generated by the last `generate_with_tools*` call as the rendering of the
    /// last message of `messages` (the history entry built from that completion), while the KV
    /// cache still holds them. `tools_json` must be what the next render will use.
    pub fn remember_generated(&mut self, messages: &[Message], tools_json: &[String]) {
        use std::hash::{Hash, Hasher};
        let Some(gen) = self.last_gen.take() else { return };
        if self.thinking {
            return;
        }
        if messages.last().is_none_or(|m| m.role != "assistant") || gen.is_empty() || !self.cached.ends_with(&gen) {
            return;
        }
        let mut ctx_hash = std::collections::hash_map::DefaultHasher::new();
        let mut key = 0u64;
        render_chatml_pieces(messages, tools_json, self.default_system(), self.tool_format, self.style, &mut |piece| match piece {
            ChatPiece::Text(t) => t.hash(&mut ctx_hash),
            ChatPiece::Assistant(m) => {
                let mut seg = String::new();
                push_assistant_segment(m, self.tool_format, self.style, &mut seg);
                seg.hash(&mut ctx_hash);
                key = ctx_hash.clone().finish();
            }
        });
        if self.gen_memo.len() >= 512 {
            self.gen_memo.remove(0);
        }
        self.gen_memo.push((key, gen));
    }

    /// Rebuilds the lookup index from the live context plus stored conversations.
    fn rebuild_lookup(&mut self) {
        let mut look = crate::model::lookup::SessionLookup::new();
        let mut budget = 200_000usize;
        for (toks, _, _) in &self.conv_snaps.entries {
            if toks.len() > budget {
                break;
            }
            budget -= toks.len();
            look.add_doc(toks);
        }
        self.look_ctx = look.add_doc(&self.cached);
        self.look = look;
    }

    /// Appends newly cached tokens to the lookup index (rebuilding if the context was cut).
    fn sync_lookup(&mut self) {
        if self.look_ctx >= self.look.num_docs() {
            self.rebuild_lookup();
            return;
        }
        let have = self.look.doc(self.look_ctx).len();
        if have > self.cached.len() || (have > 0 && self.look.doc(self.look_ctx)[have - 1] != self.cached[have - 1]) {
            self.rebuild_lookup();
            return;
        }
        for i in have..self.cached.len() {
            self.look.push(self.look_ctx, self.cached[i]);
        }
    }

    /// Feeds `tail` after `self.cached` (which must match the engine) and copies the last logits into
    /// `scratch.cur_logits`. On Qwen3.5, whose recurrent state only resumes from exact snapshots, it
    /// also snapshots every `Q35_CKPT_TOKENS` positions so a later prompt that shares only part of
    /// this one resumes from the nearest checkpoint instead of from scratch.
    fn prefill_tail(&mut self, tail: &[usize]) {
        let start = self.engine.position();
        let mut done = 0;
        if self.engine.is_qwen35() && !self.session_kv_enabled() {
            // Also checkpoint right after the system turn (the second `<|im_start|>`): compaction keeps it.
            let sys_end = self.cached.iter().chain(tail).enumerate().filter(|(_, &t)| t == self.im_start).nth(1).map(|(i, _)| i);
            loop {
                let mut mark = ((start + done) / Q35_CKPT_TOKENS + 1) * Q35_CKPT_TOKENS - start;
                if let Some(s) = sys_end.filter(|&s| s > start + done && s - start < mark && s >= 16) {
                    mark = s - start;
                }
                if mark >= tail.len() {
                    break;
                }
                self.engine.forward_prompt(&tail[done..mark]);
                done = mark;
                if let Some(snap) = self.engine.save_kv_prefix(start + done) {
                    let mut toks = self.cached.clone();
                    toks.extend_from_slice(&tail[..done]);
                    self.conv_snaps.insert(toks, snap);
                }
            }
        }
        let vocab = self.engine.vocab_size();
        let logits = self.engine.forward_prompt(&tail[done..]);
        self.scratch.cur_logits[..vocab].copy_from_slice(&logits[..vocab]);
    }

    /// Aligns the target KV cache with `prompt`: keeps the common prefix, snapshots a long context
    /// being left behind, and restores a stored snapshot when it covers more of `prompt`.
    /// Returns the number of cached tokens to reuse (at most `prompt.len() - 1`).
    fn align_kv(&mut self, prompt: &[usize]) -> usize {
        let mut common = ConvSnapshots::common_prefix(&self.cached, prompt);
        common = common.min(prompt.len().saturating_sub(1));
        if !self.engine.can_truncate_kv_to(common) {
            common = 0;
        }
        // Id-keyed session KV (when enabled) already restores before the turn; skip the content-keyed LRU.
        let leaving = if self.session_kv_enabled() { 0 } else { self.cached.len().saturating_sub(common) };
        if leaving >= ConvSnapshots::MIN_TOKENS {
            if let Some(snap) = self.engine.save_kv_prefix(self.cached.len()) {
                self.conv_snaps.insert(self.cached.clone(), snap);
            }
        }
        let min_restore = if self.engine.is_qwen35() { 16 } else { ConvSnapshots::MIN_TOKENS };
        if let Some((i, n)) = if self.session_kv_enabled() { None } else { self.conv_snaps.best(prompt) } {
            let n = n.min(prompt.len().saturating_sub(1));
            let n = Backend::usable_prefix(&self.conv_snaps.entries[i].1, n);
            if n >= common + min_restore && self.engine.restore_prefix(&self.conv_snaps.entries[i].1, n) {
                self.cached.clear();
                self.cached.extend_from_slice(&prompt[..n]);
                self.conv_snaps.touch(i);
                common = n;
            }
        }
        self.conv_snaps.touch_prefixes(prompt);
        self.engine.truncate_context(common);
        self.cached.truncate(common);
        common
    }

    /// Prefills `messages` (reusing any hot KV prefix) and returns `(first_greedy_token, prefilled_tokens)`.
    pub fn prefill_messages(&mut self, messages: &[Message]) -> Result<(usize, usize), String> {
        let prompt = self.render(messages);
        if prompt.is_empty() {
            return Err("rendered prompt is empty".to_string());
        }
        if prompt.len() >= self.ctx {
            return Err(format!(
                "prompt of {} tokens exceeds {}-token context",
                prompt.len(),
                self.ctx
            ));
        }
        let common = self.align_kv(&prompt);
        let tail = &prompt[common..];
        let vocab = self.engine.vocab_size();
        self.prefill_tail(tail);
        let actual = self.engine.position().saturating_sub(self.cached.len()).min(tail.len());
        self.cached.extend_from_slice(&tail[..actual]);
        if actual < tail.len() || crate::model::real_llm::is_abort_active() {
            return Err("aborted".to_string());
        }
        if self.im_start < self.scratch.cur_logits.len() {
            self.scratch.cur_logits[self.im_start] = f32::NEG_INFINITY;
        }
        let next = argmax(&self.scratch.cur_logits[..vocab]);
        Ok((next, tail.len()))
    }

    /// Returns whether `tok_id` is an end-of-turn or end-of-sequence stop token.
    pub fn is_stop_token(&self, tok_id: usize) -> bool {
        tok_id == self.im_end
            || tok_id == self.im_start
            || tok_id == self.tok.eos_token_id
            || self.tok.eos_token_ids.contains(&tok_id)
    }

    /// Appends raw BPE bytes for `tok_id` onto `raw_bytes`.
    pub fn append_token_bytes(&self, tok_id: usize, raw_bytes: &mut Vec<u8>) {
        self.tok.append_token_bytes(tok_id, raw_bytes);
    }

    /// Runs one greedy autoregressive forward step for `tok_id` and returns the next predicted token ID.
    pub fn advance_greedy_token(&mut self, tok_id: usize) -> Result<usize, String> {
        if self.cached.len() >= self.ctx {
            return Err(format!("context limit ({}) reached", self.ctx));
        }
        let vocab = self.engine.vocab_size();
        let logits = self.engine.forward_token(tok_id);
        self.scratch.cur_logits[..vocab].copy_from_slice(&logits[..vocab]);
        self.cached.push(tok_id);
        if self.im_start < self.scratch.cur_logits.len() {
            self.scratch.cur_logits[self.im_start] = f32::NEG_INFINITY;
        }
        Ok(argmax(&self.scratch.cur_logits[..vocab]))
    }

    /// Generates a reply, calling `on_text` with each new piece of text; return `false` from it
    /// to stop early. Oldest non-system messages are dropped if the conversation outgrows the
    /// context window.
    pub fn generate(
        &mut self,
        messages: &[Message],
        params: &GenParams,
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<Completion, String> {
        self.generate_with_tools(messages, &[], params, on_text)
    }

    /// Like [`Self::generate`], with optional OpenAI/Qwen2.5 JSON tool signatures (`tools_json`).
    /// Automatically parses `<tool_call>...</tool_call>` blocks into [`Completion::tool_calls`]
    /// and sets [`FinishReason::ToolCalls`] when tools are invoked.
    pub fn generate_with_tools(
        &mut self,
        messages: &[Message],
        tools_json: &[String],
        params: &GenParams,
        on_text: impl FnMut(&str) -> bool,
    ) -> Result<Completion, String> {
        self.generate_with_tools_prefixed(messages, tools_json, "", params, on_text)
    }

    /// Like [`Self::generate_with_tools`], optionally pre-seeding the open assistant turn with
    /// `assistant_prefix` (for example `"<tool_call>\n{\"name\": \""` during deterministic repair steps)
    /// so generation enters structured tool invocation immediately from token 0.
    pub fn generate_with_tools_prefixed(
        &mut self,
        messages: &[Message],
        tools_json: &[String],
        assistant_prefix: &str,
        params: &GenParams,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<Completion, String> {
        let mut msgs = messages.to_vec();
        let reserve = params.max_tokens.min(512).min(self.ctx / 4).max(1);
        self.last_gen = None;
        let render_prefixed = |m: &[Message], model: &Self| -> Vec<usize> { model.render_tokens(m, tools_json, assistant_prefix) };
        let mut prompt = render_prefixed(&msgs, self);
        while prompt.len() + reserve > self.ctx {
            let first = usize::from(msgs.first().is_some_and(|m| m.role == "system"));
            if msgs.len() <= first + 1 {
                return Err(format!("prompt of {} tokens exceeds the {}-token context", prompt.len(), self.ctx));
            }
            // Drop the oldest complete conversational turn (user + following reply/tool
            // replies, plus any leading non-user messages) so ChatML role alternation after
            // `system` always starts with a `user` turn.
            msgs.remove(first);
            while msgs.len() > first + 1
                && msgs
                    .get(first)
                    .is_some_and(|m| m.role == "assistant" || m.role == "tool")
            {
                msgs.remove(first);
            }
            prompt = render_prefixed(&msgs, self);
        }
        let max_new = params.max_tokens.min(self.ctx - prompt.len());
        let vocab = self.engine.vocab_size();

        // Reuse the cached prefix; at least one prompt token must run to produce logits.
        let mut common = self.align_kv(&prompt);
        if std::env::var_os("KRONOS_KV_DEBUG").is_some() {
            eprintln!("[kv] prompt {} tokens, {} cached, {} to prefill", prompt.len(), common, prompt.len() - common);
        }
        let t0 = Instant::now();
        let mut prefill_unit = if self.engine.is_metal_enabled() {
            "metal"
        } else {
            "cpu"
        };
        #[cfg(all(feature = "ane", target_os = "macos"))]
        // The Neural Engine recomputes the whole prompt from position 0; worth it once there is
        // a meaningful amount of new text, even when a shared prefix (system prompt) is cached.
        let ane_logits = if prompt.len() - common >= ANE_MIN_NEW_TOKENS {
            self.ane_prefill(&prompt).transpose()?
        } else {
            None
        };
        #[cfg(not(all(feature = "ane", target_os = "macos")))]
        let ane_logits: Option<Vec<f32>> = None;
        let mut draft_synced_in_parallel = false;
        match ane_logits {
            Some(l) => {
                prefill_unit = "ane";
                common = 0;
                self.cached.clear();
                self.scratch.cur_logits.copy_from_slice(&l);
            }
            None => {
                if self.engine.is_metal_enabled() && self.draft.is_some() {
                    let target_slice = &prompt[common..];
                    let t_eng = &mut self.engine;
                    let cur_logits = &mut self.scratch.cur_logits;
                    let d = self.draft.as_mut().unwrap();
                    let dv = d.draft_vocab;
                    let mut d_common = d
                        .cached
                        .iter()
                        .zip(&prompt)
                        .take_while(|(&a, &b)| a == if b < dv { b } else { 0 })
                        .count();
                    if !d.engine.can_truncate_kv_to(d_common) {
                        d_common = 0;
                    }
                    d.engine.truncate_context(d_common);
                    d.cached.truncate(d_common);
                    let mapped: Vec<usize> = prompt[d_common..]
                        .iter()
                        .map(|&id| if id < dv { id } else { 0 })
                        .collect();
                    std::thread::scope(|s| {
                        let draft_job = s.spawn(|| {
                            if !mapped.is_empty() {
                                d.engine.forward_prompt(&mapped);
                                d.cached.extend_from_slice(&mapped);
                            }
                        });
                        let l = t_eng.forward_prompt(target_slice);
                        cur_logits.copy_from_slice(l);
                        let _ = draft_job.join();
                    });
                    draft_synced_in_parallel = true;
                } else {
                    self.prefill_tail(&prompt[common..]);
                    let actual = self
                        .engine
                        .position()
                        .saturating_sub(self.cached.len())
                        .min(prompt.len() - common);
                    if actual < prompt.len() - common || crate::model::real_llm::is_abort_active() {
                        self.cached.extend_from_slice(&prompt[common..common + actual]);
                        return Err("aborted".to_string());
                    }
                }
            }
        }
        self.cached.extend_from_slice(&prompt[common..]);

        // Sync draft KV cache on the prompt when not already overlapped with Metal GPU prefill.
        if !draft_synced_in_parallel {
            if let Some(d) = self.draft.as_mut() {
                let dv = d.draft_vocab;
                let mut d_common = d
                    .cached
                    .iter()
                    .zip(&self.cached)
                    .take_while(|(&a, &b)| a == if b < dv { b } else { 0 })
                    .count();
                if !d.engine.can_truncate_kv_to(d_common) {
                    d_common = 0;
                }
                d.engine.truncate_context(d_common);
                d.cached.truncate(d_common);
                if d_common < self.cached.len() {
                    let mapped: Vec<usize> = self.cached[d_common..]
                        .iter()
                        .map(|&id| if id < dv { id } else { 0 })
                        .collect();
                    d.engine.forward_prompt(&mapped);
                    d.cached.extend_from_slice(&mapped);
                }
            }
        }
        let prefill_secs = t0.elapsed().as_secs_f64();

        let mut rng = Rng(params.seed ^ 0x9E37_79B9_7F4A_7C15);
        let t1 = Instant::now();
        let mut reply: Vec<usize> = Vec::with_capacity(max_new);
        let mut raw_bytes: Vec<u8> = Vec::with_capacity(max_new * 4);
        let mut emitted_bytes = 0usize;
        let mut finish = FinishReason::Length;
        let mut drafted_total = 0usize;
        let mut accepted_total = 0usize;

        if max_new == 0 {
            return Ok(Completion {
                text: String::new(),
                token_ids: Vec::new(),
                tool_calls: Vec::new(),
                prompt_tokens: prompt.len(),
                prefilled_tokens: prompt.len() - common,
                completion_tokens: 0,
                drafted_tokens: 0,
                accepted_tokens: 0,
                finish,
                prefill_secs,
                prefill_unit,
                decode_secs: 0.0,
            });
        }

        if !assistant_prefix.is_empty() {
            raw_bytes.extend_from_slice(assistant_prefix.as_bytes());
            emitted_bytes = raw_bytes.len();
            if !on_text(assistant_prefix) {
                return Ok(Completion {
                    text: assistant_prefix.to_string(),
                    token_ids: Vec::new(),
                    tool_calls: Vec::new(),
                    prompt_tokens: prompt.len(),
                    prefilled_tokens: prompt.len() - common,
                    completion_tokens: 0,
                    drafted_tokens: 0,
                    accepted_tokens: 0,
                    finish: FinishReason::Stop,
                    prefill_secs,
                    prefill_unit,
                    decode_secs: 0.0,
                });
            }
        }

        for &stop_id in &[self.im_start, self.im_end, self.tok.eos_token_id] {
            if stop_id < self.scratch.cur_logits.len() {
                self.scratch.cur_logits[stop_id] = f32::NEG_INFINITY;
            }
        }
        for &stop_id in &self.tok.eos_token_ids {
            if stop_id < self.scratch.cur_logits.len() {
                self.scratch.cur_logits[stop_id] = f32::NEG_INFINITY;
            }
        }
        let mut next = sample_token(
            &self.scratch.cur_logits,
            &[],
            params,
            &mut rng,
            &mut self.scratch.pen_logits,
            &mut self.scratch.top_heap,
        );

        let im_start_tok = self.im_start;
        let is_stop = move |tok_id: usize, im_end: usize, tok: &BpeTokenizer| {
            tok_id == im_end
                || tok_id == im_start_tok
                || tok_id == tok.eos_token_id
                || tok.eos_token_ids.contains(&tok_id)
        };
        let emit_tok = |tok_id: usize,
                        tok: &BpeTokenizer,
                        reply: &mut Vec<usize>,
                        raw_bytes: &mut Vec<u8>,
                        emitted_bytes: &mut usize,
                        on_text: &mut dyn FnMut(&str) -> bool|
         -> bool {
            reply.push(tok_id);
            tok.append_token_bytes(tok_id, raw_bytes);
            let pending = &raw_bytes[*emitted_bytes..];
            let valid_len = match std::str::from_utf8(pending) {
                Ok(s) => s.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => return true,
            };
            if valid_len > 0 {
                // SAFETY: `pending[..valid_len]` was just validated by `std::str::from_utf8`.
                let piece = unsafe { std::str::from_utf8_unchecked(&pending[..valid_len]) };
                *emitted_bytes += valid_len;
                return on_text(piece);
            }
            true
        };

        let mut cur_draft_k = SPEC_DRAFT_K_INIT;
        let mut cur_lookup_k = SPEC_LOOKUP_K_INIT;
        let mut ewma_accept = 0.70f32;
        let mut zero_accept_streak = 0usize;
        let mut draft_cooldown = 0usize;
        let mut lookup_cooldown = 0usize;
        let spec_ok = self.spec_enabled && self.engine.is_all_sdot();
        // Lookup pays off only where a verify batch costs little more than one token (Metal, large
        // models); track single-step and batch times and back off when a round loses time.
        let mut t_single: Option<f64> = None;
        let mut lookup_gain = 1.5f64;
        'outer: loop {
            if crate::model::real_llm::is_abort_active() {
                finish = FinishReason::Stop;
                break;
            }
            if is_stop(next, self.im_end, &self.tok) {
                finish = FinishReason::Stop;
                break;
            }
            if !emit_tok(next, &self.tok, &mut reply, &mut raw_bytes, &mut emitted_bytes, &mut on_text) {
                finish = FinishReason::Stop;
                break;
            }
            if reply.len() >= max_new {
                finish = FinishReason::Length;
                break;
            }

            let rem = max_new - reply.len();
            let avail_ctx = self.ctx.saturating_sub(self.cached.len() + 1);
            let greedy = params.temperature <= 0.0;
            let max_k = if greedy { SPEC_DRAFT_K_GREEDY } else { SPEC_DRAFT_K_SAMPLE };
            let verify_cap = self.engine.max_verify_batch().saturating_sub(1);
            let max_draft = rem.min(avail_ctx).min(cur_draft_k.min(max_k)).min(verify_cap);
            let max_lookup = rem.min(avail_ctx).min(cur_lookup_k).min(verify_cap);

            let mut draft_tokens = [0usize; SPEC_LOOKUP_K_MAX];
            let mut drafted = 0usize;
            let mut source = SpecSource::None;

            // Tier 1: previous-reply continuation or prompt/history suffix lookup (greedy only).
            if lookup_cooldown > 0 {
                lookup_cooldown -= 1;
            } else if spec_ok && greedy && max_lookup >= 2 {
                drafted = draft_from_prev_reply(&self.prev_reply, &reply, max_lookup, &mut draft_tokens);
                if drafted < 2 && max_lookup >= 3 {
                    drafted = if self.engine.is_qwen35() {
                        self.sync_lookup();
                        let d = self.look.draft_with(self.look_ctx, &[next], max_lookup);
                        draft_tokens[..d.len()].copy_from_slice(&d);
                        d.len()
                    } else {
                        find_suffix_continuation(&self.cached, next, max_lookup, &mut draft_tokens)
                    };
                }
                if drafted >= 2 {
                    source = SpecSource::Lookup;
                } else {
                    drafted = 0;
                }
            }

            // Tier 2: draft model (`--draft`) when lookup found nothing.
            if source == SpecSource::None && draft_cooldown > 0 {
                draft_cooldown -= 1;
            } else if source == SpecSource::None && spec_ok && max_draft >= 1 && self.draft.is_some() {
                source = SpecSource::Draft;
                {
                    let d = self.draft.as_mut().unwrap();
                    let dv = d.draft_vocab;
                    let sv = d.shared_vocab;
                    let clamp_d = |id: usize| if id < dv { id } else { 0 };
                    let mut d_common = d
                        .cached
                        .iter()
                        .zip(&self.cached)
                        .take_while(|(&a, &b)| a == clamp_d(b))
                        .count();
                    if !d.engine.can_truncate_kv_to(d_common) {
                        d_common = 0;
                    }
                    d.engine.truncate_context(d_common);
                    d.cached.truncate(d_common);
                    let next_d = clamp_d(next);
                    let gap = self.cached.len() - d_common;
                    let mut dl: &[f32] = if gap == 0 {
                        d.cached.push(next_d);
                        d.engine.forward_token(next_d)
                    } else if gap <= 2 {
                        for &id in &self.cached[d_common..] {
                            let cid = clamp_d(id);
                            d.cached.push(cid);
                            d.engine.forward_token(cid);
                        }
                        d.cached.push(next_d);
                        d.engine.forward_token(next_d)
                    } else {
                        let mut sync_buf = Vec::with_capacity(gap + 1);
                        sync_buf.extend(self.cached[d_common..].iter().map(|&id| clamp_d(id)));
                        sync_buf.push(next_d);
                        d.cached.extend_from_slice(&sync_buf);
                        d.engine.forward_prompt(&sync_buf)
                    };

                    let mut sim_recent: Vec<usize> =
                        reply[reply.len().saturating_sub(64 + SPEC_DRAFT_K_GREEDY)..].to_vec();
                    for (j, dt_slot) in draft_tokens.iter_mut().enumerate().take(max_draft) {
                        let rec = &sim_recent[sim_recent.len().saturating_sub(64)..];
                        let dl_shared = &dl[..sv];
                        let (dj, low_conf) = if params.temperature <= 0.0 {
                            let (tok_id, margin) = sample_greedy_with_margin(
                                dl_shared,
                                rec,
                                params.repeat_penalty,
                                &mut self.scratch.pen_logits,
                            );
                            (tok_id, margin < 0.55)
                        } else {
                            build_sampling_dist(
                                dl_shared,
                                rec,
                                params,
                                &mut self.scratch.pen_logits,
                                &mut self.scratch.top_heap,
                                &mut self.scratch.draft_dists[j],
                            );
                            let top_p = self.scratch.draft_dists[j].first().map_or(0.0, |e| e.1);
                            (sample_from_dist(&self.scratch.draft_dists[j], &mut rng), top_p < 0.35)
                        };
                        *dt_slot = dj;
                        drafted += 1;
                        if is_stop(dj, self.im_end, &self.tok) {
                            break;
                        }
                        // Stop extending the burst past an uncertain branch once >= 2 tokens are drafted.
                        if low_conf && drafted >= 2 {
                            break;
                        }
                        sim_recent.push(dj);
                        if j + 1 < max_draft {
                            d.cached.push(dj);
                            dl = d.engine.forward_token(dj);
                        }
                    }
                }
            }

            // Tier 3: the model's own MTP head (greedy only; a 2-4 token verify costs ~1.2x one token).
            if source == SpecSource::None && spec_ok && greedy && self.engine.has_mtp() {
                let k = SPEC_MTP_K.min(verify_cap).min(rem).min(avail_ctx);
                if k >= 1 {
                    let d = self.engine.mtp_draft(next, k);
                    if !d.is_empty() {
                        draft_tokens[..d.len()].copy_from_slice(&d);
                        drafted = d.len();
                        source = SpecSource::Mtp;
                    }
                }
            }

            if source != SpecSource::None {
                // Verify `[next, d_0, ..., d_{drafted-1}]` in one batched target pass.
                let mut batch = [0usize; SPEC_LOOKUP_K_MAX + 1];
                batch[0] = next;
                batch[1..1 + drafted].copy_from_slice(&draft_tokens[..drafted]);
                let batch_len = 1 + drafted;
                let pos0 = self.engine.position();
                if self.scratch.verify_logits.len() < batch_len * vocab {
                    self.scratch.verify_logits.resize(batch_len * vocab, 0.0);
                }
                let t_batch0 = Instant::now();
                self.engine
                    .forward_batch_logits(&batch[..batch_len], &mut self.scratch.verify_logits);
                let t_batch = t_batch0.elapsed().as_secs_f64();

                let mut accepted_all = true;
                let mut accepted_in_round = 0usize;
                for j in 0..drafted {
                    self.cached.push(batch[j]);
                    let t_logits = &self.scratch.verify_logits[j * vocab..(j + 1) * vocab];
                    let rec = &reply[reply.len().saturating_sub(64)..];
                    let dj = draft_tokens[j];

                    let rejected = if greedy {
                        let tj = sample_greedy(t_logits, rec, params.repeat_penalty, &mut self.scratch.pen_logits);
                        (tj != dj).then_some(tj)
                    } else {
                        build_sampling_dist(
                            t_logits,
                            rec,
                            params,
                            &mut self.scratch.pen_logits,
                            &mut self.scratch.top_heap,
                            &mut self.scratch.target_dist,
                        );
                        let p_d = prob_in_dist(&self.scratch.target_dist, dj);
                        let q_d = prob_in_dist(&self.scratch.draft_dists[j], dj).max(1e-12);
                        (p_d < q_d && rng.next_f32() >= (p_d / q_d)).then(|| {
                            sample_residual_dist(
                                &self.scratch.target_dist,
                                &self.scratch.draft_dists[j],
                                &mut rng,
                            )
                        })
                    };
                    if let Some(tj) = rejected {
                        rollback_spec(&mut self.engine, &mut self.draft, pos0 + j + 1);
                        next = tj;
                        accepted_all = false;
                        break;
                    }

                    accepted_in_round += 1;
                    let stop_here = if is_stop(dj, self.im_end, &self.tok)
                        || !emit_tok(dj, &self.tok, &mut reply, &mut raw_bytes, &mut emitted_bytes, &mut on_text)
                    {
                        Some(FinishReason::Stop)
                    } else if reply.len() >= max_new {
                        Some(FinishReason::Length)
                    } else {
                        None
                    };
                    if let Some(f) = stop_here {
                        drafted_total += j + 1;
                        accepted_total += accepted_in_round;
                        rollback_spec(&mut self.engine, &mut self.draft, pos0 + j + 1);
                        finish = f;
                        break 'outer;
                    }
                }

                drafted_total += drafted;
                accepted_total += accepted_in_round;
                if accepted_all {
                    self.cached.push(batch[drafted]);
                    let bonus_logits = &self.scratch.verify_logits[drafted * vocab..(drafted + 1) * vocab];
                    let rec = &reply[reply.len().saturating_sub(64)..];
                    next = sample_token(
                        bonus_logits,
                        rec,
                        params,
                        &mut rng,
                        &mut self.scratch.pen_logits,
                        &mut self.scratch.top_heap,
                    );
                }

                match source {
                    SpecSource::Mtp => {}
                    SpecSource::Draft => {
                        let round_rate = (accepted_in_round as f32) / (drafted.max(1) as f32);
                        ewma_accept = 0.75 * ewma_accept + 0.25 * round_rate;
                        if accepted_all {
                            zero_accept_streak = 0;
                            cur_draft_k = if ewma_accept >= 0.75 {
                                (cur_draft_k + 2).min(SPEC_DRAFT_K_GREEDY)
                            } else {
                                (cur_draft_k + 1).min(SPEC_DRAFT_K_GREEDY)
                            };
                        } else {
                            cur_draft_k = if ewma_accept >= 0.80 {
                                7
                            } else if ewma_accept >= 0.68 {
                                6
                            } else if ewma_accept >= 0.55 {
                                5
                            } else if ewma_accept >= 0.40 {
                                4
                            } else if ewma_accept >= 0.25 {
                                3
                            } else {
                                2
                            };
                            if accepted_in_round == 0 {
                                zero_accept_streak += 1;
                                if zero_accept_streak >= 4 {
                                    draft_cooldown = 4;
                                    zero_accept_streak = 0;
                                }
                            } else {
                                zero_accept_streak = 0;
                            }
                        }
                    }
                    _ => {
                        // Copied text tends to continue; grow the window on full accepts, shrink on misses.
                        cur_lookup_k = if accepted_all {
                            (cur_lookup_k * 2).min(SPEC_LOOKUP_K_MAX)
                        } else if accepted_in_round == 0 {
                            lookup_cooldown = 8;
                            SPEC_LOOKUP_K_INIT.min(cur_lookup_k)
                        } else {
                            (accepted_in_round + 2).clamp(3, SPEC_LOOKUP_K_MAX)
                        };
                        if let Some(t1) = t_single {
                            let gain = (accepted_in_round + 1) as f64 * t1 / t_batch.max(1e-9);
                            lookup_gain = 0.7 * lookup_gain + 0.3 * gain;
                            if lookup_gain < 0.9 {
                                lookup_cooldown = 32;
                                lookup_gain = 1.2;
                            }
                        }
                    }
                }
                continue;
            }
            // Direct single-token Megakernel step (`0` allocations!)
            let recent = &reply[reply.len().saturating_sub(64)..];
            let t_step0 = Instant::now();
            let l = self.engine.forward_token(next);
            let dt = t_step0.elapsed().as_secs_f64();
            t_single = Some(t_single.map_or(dt, |t| 0.7 * t + 0.3 * dt));
            self.cached.push(next);
            next = sample_token(
                l,
                recent,
                params,
                &mut rng,
                &mut self.scratch.pen_logits,
                &mut self.scratch.top_heap,
            );
        }

        // Flush any remaining lossy bytes at end of generation
        if emitted_bytes < raw_bytes.len() {
            let tail = String::from_utf8_lossy(&raw_bytes[emitted_bytes..]);
            if !tail.is_empty() {
                let _ = on_text(&tail);
            }
        }

        let decode_secs = t1.elapsed().as_secs_f64();
        if self.engine.is_metal_enabled() && self.draft.is_some() {
            crate::kernels::thread_pool::SpinThreadPool::park_idle_workers();
        }
        let raw_text = String::from_utf8_lossy(&raw_bytes).into_owned();
        let raw_text = match raw_text.find("</think>") {
            Some(i) if self.thinking => raw_text[i + "</think>".len()..].trim_start().to_string(),
            _ => raw_text,
        };
        let (text, tool_calls) = if !tools_json.is_empty()
            || raw_text.contains(crate::config::markup::TOOL_CALL_OPEN)
            || raw_text.contains(crate::config::markup::TOOLS_OPEN)
            || raw_text.contains("<tool ")
        {
            let (prose, calls) = match self.tool_format {
                ToolFormat::Json => parse_qwen_tool_calls(&raw_text),
                ToolFormat::Xml => parse_xml_tool_calls(&raw_text),
            };
            if calls.is_empty() {
                (raw_text, Vec::new())
            } else {
                if finish == FinishReason::Stop {
                    finish = FinishReason::ToolCalls;
                }
                (prose, calls)
            }
        } else {
            (raw_text, Vec::new())
        };

        self.save_active_session();
        self.prev_reply.clone_from(&reply);
        let completion_tokens = reply.len();
        // Everything after `<|im_start|>assistant\n` in the KV cache: forced prefix + reply.
        let base_len = if assistant_prefix.is_empty() { prompt.len() } else { self.render_tokens(&msgs, tools_json, "").len() };
        if base_len <= prompt.len() {
            let mut gen = prompt[base_len..].to_vec();
            gen.extend_from_slice(&reply);
            self.last_gen = Some(gen);
        }
        Ok(Completion {
            text,
            token_ids: reply,
            tool_calls,
            prompt_tokens: prompt.len(),
            prefilled_tokens: prompt.len() - common,
            completion_tokens,
            drafted_tokens: drafted_total,
            accepted_tokens: accepted_total,
            finish,
            prefill_secs,
            decode_secs,
            prefill_unit,
        })
    }
}

/// Renders a ChatML / Qwen2.5 tool-calling conversation into its exact prompt string
/// (matching `Qwen2.5-Coder-Instruct`'s Jinja `chat_template` bit-for-bit).
pub fn render_chatml_prompt(messages: &[Message], tools_json: &[String], default_system: &str) -> String {
    render_chatml_prompt_with(messages, tools_json, default_system, ToolFormat::Json)
}

/// Renders one tool schema (OpenAI function JSON) in the Qwen3-Coder `<function>` layout.
fn render_xml_tool_schema(tool_json: &str, out: &mut String) {
    let Ok(j) = Json::parse(tool_json) else {
        out.push('\n');
        out.push_str(tool_json.trim());
        return;
    };
    let f = j.get("function").unwrap_or(&j);
    out.push_str("\n<function>\n<name>");
    out.push_str(f.get("name").and_then(Json::as_str).unwrap_or(""));
    out.push_str("</name>");
    if let Some(d) = f.get("description").and_then(Json::as_str) {
        out.push_str("\n<description>");
        out.push_str(d.trim());
        out.push_str("</description>");
    }
    out.push_str("\n<parameters>");
    if let Some(props) = f.get("parameters").and_then(|p| p.get("properties")).and_then(Json::as_obj) {
        for (pname, pf) in props {
            out.push_str("\n<parameter>\n<name>");
            out.push_str(pname);
            out.push_str("</name>");
            if let Some(t) = pf.get("type").and_then(Json::as_str) {
                out.push_str("\n<type>");
                out.push_str(t);
                out.push_str("</type>");
            }
            if let Some(d) = pf.get("description").and_then(Json::as_str) {
                out.push_str("\n<description>");
                out.push_str(d.trim());
                out.push_str("</description>");
            }
            out.push_str("\n</parameter>");
        }
    }
    if let Some(req) = f.get("parameters").and_then(|p| p.get("required")) {
        out.push_str("\n<required>");
        out.push_str(&req.stringify());
        out.push_str("</required>");
    }
    out.push_str("\n</parameters>\n</function>");
}

/// Renders one assistant tool call in the Qwen3-Coder layout; arguments must be a JSON object.
fn render_xml_tool_call(tc: &ToolCall, out: &mut String) {
    out.push_str("\n<tool_call>\n<function=");
    out.push_str(&tc.name);
    out.push('>');
    if let Some(args) = Json::parse(&tc.arguments).ok().as_ref().and_then(Json::as_obj) {
        for (k, v) in args {
            out.push_str("\n<parameter=");
            out.push_str(k);
            out.push_str(">\n");
            match v {
                Json::Str(s) => out.push_str(s),
                other => out.push_str(&other.stringify()),
            }
            out.push_str("\n</parameter>");
        }
    }
    out.push_str("\n</function>\n</tool_call>");
}

const XML_TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// Parses Qwen3-Coder `<tool_call><function=name><parameter=k>v</parameter>...</function></tool_call>`
/// blocks; parameter values that are valid JSON literals/objects are kept as such, others become strings.
pub fn parse_xml_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find(OPEN) {
        let start = cursor + rel;
        prose.push_str(&text[cursor..start]);
        let body_start = start + OPEN.len();
        let body_end = text[body_start..].find(CLOSE).map_or(text.len(), |e| body_start + e);
        let body = &text[body_start..body_end];
        if let Some(call) = parse_xml_function_block(body, calls.len()) {
            calls.push(call);
        } else {
            prose.push_str(&text[start..body_end]);
        }
        cursor = (body_end + CLOSE.len()).min(text.len());
    }
    prose.push_str(&text[cursor..]);
    (prose.trim().to_string(), calls)
}

fn parse_xml_function_block(body: &str, idx: usize) -> Option<ToolCall> {
    let fn_start = body.find("<function=")? + "<function=".len();
    let name_end = fn_start + body[fn_start..].find('>')?;
    let name = body[fn_start..name_end].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let mut rest = &body[name_end + 1..];
    let mut args = String::from("{");
    let mut first = true;
    while let Some(p) = rest.find("<parameter=") {
        let k_start = p + "<parameter=".len();
        let Some(k_len) = rest[k_start..].find('>') else { break };
        let key = rest[k_start..k_start + k_len].trim();
        let v_start = k_start + k_len + 1;
        let v_end = rest[v_start..].find("</parameter>").map_or(rest.len(), |e| v_start + e);
        let raw = rest[v_start..v_end].strip_prefix('\n').unwrap_or(&rest[v_start..v_end]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        if !first {
            args.push(',');
        }
        first = false;
        args.push_str(&quote(key));
        args.push(':');
        let t = raw.trim();
        let literal = (t.starts_with('{') || t.starts_with('[') || t == "true" || t == "false" || t == "null"
            || t.parse::<f64>().is_ok())
            && Json::parse(t).is_ok();
        if literal {
            args.push_str(t);
        } else {
            args.push_str(&quote(raw));
        }
        rest = &rest[(v_end + "</parameter>".len()).min(rest.len())..];
    }
    args.push('}');
    Some(ToolCall { id: format!("call_{idx}"), name, arguments: args })
}

/// Empty reasoning block that opens a non-thinking Qwen3.5 assistant turn.
const THINK_EMPTY: &str = "<think>\n\n</think>\n\n";

/// An assistant segment with [`THINK_EMPTY`] after its role line, as Qwen3.5 renders history.
fn with_think_block(seg: &str) -> String {
    let head = "<|im_start|>assistant";
    match seg.strip_prefix(head) {
        Some(rest) => format!("{head}\n{THINK_EMPTY}{}", rest.strip_prefix('\n').unwrap_or(rest)),
        None => seg.to_string(),
    }
}

enum ChatPiece<'a> {
    Text(&'a str),
    Assistant(&'a Message),
}

/// Canonical text of an assistant message: ChatML `<|im_start|>assistant` through `<|im_end|>\n`,
/// or HunYuan `<assistant>content<eos>` (tool calls are not part of that template).
fn push_assistant_segment(m: &Message, fmt: ToolFormat, style: PromptStyle, out: &mut String) {
    if style == PromptStyle::HunYuan {
        out.push_str(style.assistant_open());
        out.push_str(&m.content);
        out.push_str(style.assistant_close());
        return;
    }
    if m.tool_calls.is_empty() {
        out.push_str(crate::config::markup::IM_START);
        out.push_str("assistant\n");
        out.push_str(&m.content);
        out.push_str("<|im_end|>\n");
        return;
    }
    out.push_str("<|im_start|>assistant");
    if !m.content.is_empty() {
        out.push('\n');
        out.push_str(&m.content);
    }
    for tc in &m.tool_calls {
        if fmt == ToolFormat::Xml {
            render_xml_tool_call(tc, out);
            continue;
        }
        let args_obj = if tc.arguments.trim().is_empty() {
            "{}"
        } else {
            tc.arguments.trim()
        };
        out.push_str("\n<tool_call>\n{\"name\": ");
        out.push_str(&quote(&tc.name));
        out.push_str(", \"arguments\": ");
        out.push_str(args_obj);
        out.push_str("}\n</tool_call>");
    }
    out.push_str("<|im_end|>\n");
}

pub fn render_chatml_prompt_with(
    messages: &[Message],
    tools_json: &[String],
    default_system: &str,
    fmt: ToolFormat,
) -> String {
    render_prompt_with(messages, tools_json, default_system, fmt, PromptStyle::ChatMl)
}

/// Renders a conversation in `style`, ending with an open assistant turn.
pub fn render_prompt_with(
    messages: &[Message],
    tools_json: &[String],
    default_system: &str,
    fmt: ToolFormat,
    style: PromptStyle,
) -> String {
    let mut out = String::new();
    render_chatml_pieces(messages, tools_json, default_system, fmt, style, &mut |piece| match piece {
        ChatPiece::Text(t) => out.push_str(t),
        ChatPiece::Assistant(m) => push_assistant_segment(m, fmt, style, &mut out),
    });
    out
}

/// HunYuan template (`chat_template.jinja` of Hy-MT2): tools are not rendered and tool results
/// are passed as user turns.
fn render_hunyuan_pieces(messages: &[Message], default_system: &str, sink: &mut dyn FnMut(ChatPiece<'_>)) {
    use crate::config::markup::{HY_ASSISTANT, HY_BOS, HY_SYSTEM_END, HY_USER};
    let mut out = String::from(HY_BOS);
    let (sys_content, start_idx) = if messages.first().is_some_and(|m| m.role == "system") {
        (messages[0].content.as_str(), 1usize)
    } else {
        (default_system, 0usize)
    };
    if !sys_content.is_empty() {
        out.push_str(sys_content);
        out.push_str(HY_SYSTEM_END);
    }
    for m in &messages[start_idx..] {
        if m.role == "assistant" {
            sink(ChatPiece::Text(&out));
            out.clear();
            sink(ChatPiece::Assistant(m));
        } else {
            out.push_str(HY_USER);
            out.push_str(&m.content);
        }
    }
    out.push_str(HY_ASSISTANT);
    sink(ChatPiece::Text(&out));
}

fn render_chatml_pieces(
    messages: &[Message],
    tools_json: &[String],
    default_system: &str,
    fmt: ToolFormat,
    style: PromptStyle,
    sink: &mut dyn FnMut(ChatPiece<'_>),
) {
    if style == PromptStyle::HunYuan {
        return render_hunyuan_pieces(messages, default_system, sink);
    }
    let mut out = String::new();
    let (sys_content, start_idx) = if messages.first().is_some_and(|m| m.role == "system") {
        (messages[0].content.as_str(), 1usize)
    } else {
        (default_system, 0usize)
    };

    out.push_str("<|im_start|>system\n");
    out.push_str(sys_content);
    if !tools_json.is_empty() && fmt == ToolFormat::Xml {
        out.push_str("\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in tools_json {
            render_xml_tool_schema(tool, &mut out);
        }
        out.push_str("\n</tools>");
        out.push_str(XML_TOOL_INSTRUCTIONS);
    } else if !tools_json.is_empty() {
        out.push_str(
            "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        );
        for tool in tools_json {
            out.push('\n');
            out.push_str(tool.trim());
        }
        out.push_str(
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        );
    }
    out.push_str("<|im_end|>\n");

    for (idx, m) in messages.iter().enumerate().skip(start_idx) {
        if m.role == "assistant" {
            sink(ChatPiece::Text(&out));
            out.clear();
            sink(ChatPiece::Assistant(m));
        } else if m.role == "tool" {
            let prev_is_tool = idx > start_idx && messages[idx - 1].role == "tool";
            if !prev_is_tool {
                out.push_str("<|im_start|>user");
            }
            out.push_str("\n<tool_response>\n");
            out.push_str(&m.content);
            out.push_str("\n</tool_response>");
            let next_is_tool = idx + 1 < messages.len() && messages[idx + 1].role == "tool";
            if !next_is_tool {
                out.push_str("<|im_end|>\n");
            }
        } else {
            out.push_str(crate::config::markup::IM_START);
            out.push_str(&m.role);
            out.push('\n');
            out.push_str(&m.content);
            out.push_str("<|im_end|>\n");
        }
    }
    out.push_str("<|im_start|>assistant\n");
    sink(ChatPiece::Text(&out));
}

/// Escapes raw control bytes (`\n`, `\r`, `\t`), single-line code string `\n` escapes (`split('\n')`,
/// `f"...\n"`), LaTeX backslash commands (`\frac`, `\times`, `\rangle`, `\beta`), and non-RFC8259
/// backslash sequences (`\'`, `\ `, `\d`, `\s`, `\u{...}`) inside JSON string literals.
fn sanitize_json_control_chars(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 32);
    let mut in_str = false;
    let mut esc = false;
    // Tracks inner single-quoted spans inside a JSON string:
    // `None` = outside `'...'`, `Some(false)` = opened by plain `'`, `Some(true)` = opened by over-escaped `\'`.
    let mut sq_mode: Option<bool> = None;
    let mut inner_sq_open_on_line = false;
    let mut inner_dq_open_on_line = false;
    let mut after_json_escaped_backslash = false;
    let bytes = s.as_bytes();
    let mut iter = s.char_indices().peekable();
    while let Some((byte_idx, ch)) = iter.next() {
        if in_str {
            if esc {
                esc = false;
                after_json_escaped_backslash = false;
                let rest = &s[byte_idx + ch.len_utf8()..];
                match ch {
                    '"' => {
                        out.push('"');
                        if out.ends_with("\\\"\\\"\\\"") {
                            inner_dq_open_on_line = false;
                        } else {
                            inner_dq_open_on_line = !inner_dq_open_on_line;
                        }
                    }
                    '/' => out.push('/'),
                    'b' | 'f' => {
                        if rest.as_bytes().first().is_some_and(u8::is_ascii_alphabetic) {
                            // LaTeX / regex identifier escape (`\frac`, `\forall`, `\beta`, `\mathbf`, `\bmod`).
                            out.push('\\');
                            out.push(ch);
                        } else {
                            out.push(ch);
                        }
                    }
                    't' => {
                        let inline_sq_tab = inner_sq_open_on_line
                            && (rest.starts_with('\'') || rest.starts_with("\\'"));
                        let latex_t = [
                            "imes", "ext{", "extrm{", "extbf{", "exttt{", "heta", "au", "op",
                            "o ", "riangle", "ilde", "frac", "anh",
                        ]
                        .iter()
                        .any(|p| rest.starts_with(p));
                        if inline_sq_tab || latex_t {
                            out.push('\\');
                            out.push('t');
                        } else {
                            out.push('t');
                        }
                    }
                    'r' => {
                        let inline_sq_cr = inner_sq_open_on_line
                            && (rest.starts_with('\'')
                                || rest.starts_with("\\'")
                                || rest.starts_with("\\n'")
                                || rest.starts_with("\\n\\'"));
                        let latex_r = ["angle", "ight", "ho", "mathbb", "mathrm"]
                            .iter()
                            .any(|p| rest.starts_with(p));
                        if inline_sq_cr || latex_r {
                            out.push('\\');
                            out.push('r');
                        } else {
                            out.push('r');
                        }
                    }
                    'n' => {
                        let inline_sq_nl = inner_sq_open_on_line
                            && (rest.starts_with('\'') || rest.starts_with("\\'"));
                        let inline_dq_nl = inner_dq_open_on_line
                            && rest.starts_with("\\\"")
                            && !rest.starts_with("\\\"\\\"\\\"")
                            && matches!(
                                rest.as_bytes().get(2).copied(),
                                Some(b')' | b',' | b';' | b'+' | b']' | b'}' | b':')
                            );
                        if inline_sq_nl || inline_dq_nl {
                            out.push('\\');
                            out.push('n');
                        } else {
                            out.push('n');
                            sq_mode = None;
                            inner_sq_open_on_line = false;
                            inner_dq_open_on_line = false;
                        }
                    }
                    '\\' => {
                        out.push('\\');
                        after_json_escaped_backslash = true;
                    }
                    '\'' => match sq_mode {
                        None => {
                            // Over-escaped opening single-quote delimiter (`owner:\'Core\'`).
                            out.pop();
                            out.push('\'');
                            sq_mode = Some(true);
                            inner_sq_open_on_line = true;
                        }
                        Some(true) => {
                            let next_is_word = iter
                                .peek()
                                .is_some_and(|&(_, nc)| nc.is_ascii_alphanumeric());
                            if next_is_word {
                                // Apostrophe inside `\'it\'s live\'` stays escaped in JS (`\'`).
                                out.push('\\');
                                out.push('\'');
                            } else {
                                out.pop();
                                out.push('\'');
                                sq_mode = None;
                                inner_sq_open_on_line = false;
                            }
                        }
                        Some(false) => {
                            let closes_before_delim = rest.starts_with(',')
                                || rest.starts_with(']')
                                || rest.starts_with(')')
                                || rest.starts_with('}')
                                || rest.starts_with(':')
                                || rest.starts_with(';')
                                || rest.starts_with("\\n")
                                || rest.starts_with("\\r")
                                || rest.starts_with('\n')
                                || rest.starts_with('\r')
                                || {
                                    let trimmed = rest.trim_start_matches([' ', '\t']);
                                    trimmed.len() < rest.len()
                                        && (trimmed.starts_with(',')
                                            || trimmed.starts_with(']')
                                            || trimmed.starts_with(')')
                                            || trimmed.starts_with('#')
                                            || trimmed.starts_with("\\n")
                                            || trimmed.starts_with('\n'))
                                };
                            if closes_before_delim {
                                // Over-escaped closing delimiter on a string opened with plain `'` (`'0\', \'-i\'`).
                                out.pop();
                                out.push('\'');
                                sq_mode = None;
                                inner_sq_open_on_line = false;
                            } else {
                                // Escaped apostrophe inside plain `'it\'s ok'` stays escaped (`\\'`).
                                out.push('\\');
                                out.push('\'');
                            }
                        }
                    },
                    'u' => {
                        let valid_hex4 = bytes
                            .get(byte_idx + 1..byte_idx + 5)
                            .is_some_and(|h| h.iter().all(u8::is_ascii_hexdigit));
                        if valid_hex4 {
                            out.push('u');
                        } else {
                            out.push('\\');
                            out.push('u');
                        }
                    }
                    '\n' => {
                        out.push('n');
                        sq_mode = None;
                        inner_sq_open_on_line = false;
                        inner_dq_open_on_line = false;
                    }
                    '\r' => {
                        if iter.peek().is_some_and(|&(_, next_c)| next_c == '\n') {
                            let _ = iter.next();
                        }
                        out.push('n');
                        sq_mode = None;
                        inner_sq_open_on_line = false;
                        inner_dq_open_on_line = false;
                    }
                    '`' => {
                        // Drop stray line-ending `\`\n` / `\`` before newline emitted around JS template literals.
                        let before_newline = match iter.peek().copied() {
                            Some((_, '\n' | '\r')) => true,
                            Some((next_idx, '\\')) => matches!(
                                bytes.get(next_idx + 1).copied(),
                                Some(b'n' | b'r')
                            ),
                            _ => false,
                        };
                        if before_newline {
                            out.pop();
                        } else {
                            out.push('\\');
                            out.push('`');
                        }
                    }
                    other => {
                        out.push('\\');
                        out.push(other);
                    }
                }
                continue;
            }
            let prev_esc_bs = after_json_escaped_backslash;
            after_json_escaped_backslash = false;
            match ch {
                '\\' => {
                    esc = true;
                    out.push('\\');
                }
                '"' => {
                    in_str = false;
                    sq_mode = None;
                    inner_sq_open_on_line = false;
                    inner_dq_open_on_line = false;
                    out.push('"');
                }
                '\'' => {
                    if !prev_esc_bs {
                        let prev_is_word = byte_idx > 0
                            && bytes[byte_idx - 1].is_ascii_alphanumeric();
                        let next_is_word = iter
                            .peek()
                            .is_some_and(|&(_, nc)| nc.is_ascii_alphanumeric());
                        let is_contraction_apostrophe =
                            sq_mode.is_none() && prev_is_word && next_is_word;
                        if !is_contraction_apostrophe {
                            sq_mode = match sq_mode {
                                None => {
                                    inner_sq_open_on_line = !out.ends_with("''");
                                    Some(false)
                                }
                                Some(_) => {
                                    inner_sq_open_on_line = false;
                                    None
                                }
                            };
                        }
                    }
                    out.push('\'');
                }
                '\n' => {
                    sq_mode = None;
                    inner_sq_open_on_line = false;
                    inner_dq_open_on_line = false;
                    out.push_str("\\n");
                }
                '\r' => {
                    sq_mode = None;
                    inner_sq_open_on_line = false;
                    inner_dq_open_on_line = false;
                    out.push_str("\\r");
                }
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => {
                    let code = c as u32;
                    out.push_str(&format!("\\u{code:04x}"));
                }
                c => out.push(c),
            }
        } else {
            if ch == '"' {
                in_str = true;
                sq_mode = None;
                inner_sq_open_on_line = false;
                inner_dq_open_on_line = false;
                after_json_escaped_backslash = false;
            }
            out.push(ch);
        }
    }
    out
}

fn parse_json_lenient(s: &str) -> Option<Json> {
    let sanitized = sanitize_json_control_chars(s);
    if let Ok(v) = Json::parse(&sanitized) {
        return Some(v);
    }
    if let Ok(v) = Json::parse(s) {
        return Some(v);
    }
    let trimmed = sanitized.trim();
    if trimmed.starts_with('{') && trimmed.contains("\"name\"") && trimmed.contains("\"arguments\"") {
        let with_brace = format!("{trimmed}}}");
        if let Ok(v) = Json::parse(&with_brace) {
            return Some(v);
        }
    }
    None
}

/// Repairs unescaped `"` inside the final string argument (`content`, `new_text`, `cmd`, or `old_text`)
/// and missing outer `}` of a `<tool_call>{...}</tool_call>` or fenced JSON tool block.
fn repair_tool_call_inner_quotes(raw_obj: &str) -> Option<Json> {
    let trimmed = raw_obj.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    if let Some(v) = parse_json_lenient(trimmed) {
        return Some(v);
    }
    let mut best_val_start: Option<(usize, u8, u8)> = None;
    for (prio, key) in [
        (2u8, "\"content\""),
        (2u8, "\"new_text\""),
        (2u8, "\"cmd\""),
        (1u8, "\"old_text\""),
    ] {
        let mut search_from = 0usize;
        while search_from < trimmed.len() {
            let Some(rel) = trimmed[search_from..].find(key) else {
                break;
            };
            let key_pos = search_from + rel;
            // Ignore escaped `\"content\"` occurrences inside string payloads.
            if key_pos > 0 && trimmed.as_bytes()[key_pos - 1] == b'\\' {
                search_from = key_pos + key.len();
                continue;
            }
            let after_key = key_pos + key.len();
            let rest = trimmed[after_key..].trim_start();
            if let Some(after_colon) = rest.strip_prefix(':') {
                let after_ws = after_colon.trim_start();
                if let Some(&delim) = after_ws.as_bytes().first() {
                    if delim == b'"' || delim == b'`' {
                        let quote_offset = trimmed.len() - after_ws.len();
                        let val_start = quote_offset + 1;
                        let better = match best_val_start {
                            None => true,
                            Some((prev_pos, _, prev_prio)) => {
                                // Prefer higher-priority argument keys (`new_text` over `old_text`),
                                // and the earliest occurrence of that key so inner user code containing
                                // `"content": "..."` never shadows the outer tool argument key.
                                prio > prev_prio || (prio == prev_prio && val_start < prev_pos)
                            }
                        };
                        if better {
                            best_val_start = Some((val_start, delim, prio));
                        }
                    }
                }
            }
            search_from = after_key;
        }
    }
    let (val_start, open_delim, _) = best_val_start?;
    // Locate the closing delimiter of the string value right before `}}` (or single/omitted outer brace).
    let mut tail_idx = trimmed.len();
    let bytes = trimmed.as_bytes();
    let mut braces_seen = 0usize;
    while tail_idx > val_start && braces_seen < 2 {
        let b = bytes[tail_idx - 1];
        if b.is_ascii_whitespace() {
            tail_idx -= 1;
            continue;
        }
        if b == b'}' {
            braces_seen += 1;
            tail_idx -= 1;
        } else {
            break;
        }
    }
    while tail_idx > val_start && (bytes[tail_idx - 1].is_ascii_whitespace() || bytes[tail_idx - 1] == b',') {
        tail_idx -= 1;
    }
    // Handle reversed argument order `{"content": "...", "path": "foo.py"}}`.
    if tail_idx > val_start && bytes[tail_idx - 1] == b'"' {
        let prefix = &trimmed[val_start..tail_idx - 1];
        if let Some(path_key_rel) = prefix.rfind("\"path\"") {
            let before_path = prefix[..path_key_rel].trim_end();
            let after_path = prefix[path_key_rel + "\"path\"".len()..].trim_start();
            if let (Some(before_comma), Some(after_colon)) =
                (before_path.strip_suffix(','), after_path.strip_prefix(':'))
            {
                let path_val = after_colon.trim_start();
                if path_val.starts_with('"') && !path_val[1..].contains('"') {
                    let trimmed_before = before_comma.trim_end();
                    if trimmed_before.as_bytes().last().copied() == Some(open_delim) {
                        tail_idx = val_start + trimmed_before.len();
                    }
                }
            }
        }
    }
    if tail_idx <= val_start || bytes[tail_idx - 1] != open_delim {
        return None;
    }
    let val_end = tail_idx - 1;
    let inner = &trimmed[val_start..val_end];
    let mut repaired = String::with_capacity(trimmed.len() + 40);
    repaired.push_str(&trimmed[..val_start - 1]);
    repaired.push('"');
    let mut backslashes = 0usize;
    for ch in inner.chars() {
        if ch == '\\' {
            backslashes += 1;
            repaired.push('\\');
        } else {
            if ch == '"' && backslashes.is_multiple_of(2) {
                repaired.push('\\');
            }
            backslashes = 0;
            repaired.push(ch);
        }
    }
    if !backslashes.is_multiple_of(2) {
        repaired.push('\\');
    }
    repaired.push('"');
    let suffix = trimmed[val_end + 1..].trim_start();
    if let Some(after_comma) = suffix.strip_prefix(',') {
        let rest_t = after_comma.trim_start();
        if rest_t.starts_with('}') || rest_t.is_empty() {
            repaired.push_str(rest_t);
        } else {
            repaired.push_str(suffix);
        }
    } else {
        repaired.push_str(suffix);
    }
    if braces_seen == 0 {
        repaired.push_str("}}");
    }
    if let Some(v) = parse_json_lenient(&repaired) {
        return Some(v);
    }
    if braces_seen == 1 {
        repaired.push('}');
        return parse_json_lenient(&repaired);
    }
    None
}

fn extract_name_and_args(v: &Json) -> Option<(String, String)> {
    let fn_obj = v.get("function").unwrap_or(v);
    let name = fn_obj.get("name").and_then(Json::as_str)?.to_string();
    let arguments = match fn_obj.get("arguments") {
        Some(Json::Str(s)) => s.clone(),
        Some(other) => other.stringify(),
        None => {
            if let Some(kvs) = fn_obj.as_obj() {
                let extra: Vec<(String, Json)> = kvs
                    .iter()
                    .filter(|(k, _)| !matches!(k.as_str(), "name" | "type" | "id" | "function"))
                    .cloned()
                    .collect();
                if !extra.is_empty() {
                    Json::Obj(extra).stringify()
                } else {
                    "{}".to_string()
                }
            } else {
                "{}".to_string()
            }
        }
    };
    Some((name, arguments))
}

/// Returns the byte offset of the next genuine `<tool_call>` block header (followed by `{` or `` ``` ``)
/// starting at or after `from`, ignoring inline prose mentions such as `` `<tool_call>` ``.
fn find_next_real_tool_open(text: &str, from: usize) -> Option<usize> {
    const OPEN_TAG: &str = crate::config::markup::TOOL_CALL_OPEN;
    let mut scan = from;
    while let Some(rel) = text[scan..].find(OPEN_TAG) {
        let pos = scan + rel;
        let after = text[pos + OPEN_TAG.len()..].trim_start();
        if after.starts_with('{') || after.starts_with("```") {
            return Some(pos);
        }
        scan = pos + OPEN_TAG.len();
    }
    None
}

/// Locates the `</tool_call>` closing tag belonging to the `<tool_call>` object starting at `obj_start`,
/// stopping before any subsequent `<tool_call>` tag and preferring a `</tool_call>` preceded by `}`.
fn find_enclosing_tool_close(text: &str, obj_start: usize) -> Option<usize> {
    const CLOSE_TAG: &str = crate::config::markup::TOOL_CALL_CLOSE;
    let next_open = find_next_real_tool_open(text, obj_start).unwrap_or(text.len());
    let mut candidate: Option<usize> = None;
    let mut scan = obj_start;
    while scan < next_open {
        let Some(rel) = text[scan..next_open].find(CLOSE_TAG) else {
            break;
        };
        let close_pos = scan + rel;
        let before = text[obj_start..close_pos]
            .trim_end()
            .trim_end_matches("```")
            .trim_end();
        if before.ends_with('}') || candidate.is_none() {
            candidate = Some(close_pos);
        }
        scan = close_pos + CLOSE_TAG.len();
    }
    candidate
}

/// Extracts tool `(name, arguments)` only when explicit `"name"` and `"arguments"` keys are present.
fn extract_explicit_tool_name_and_args(v: &Json) -> Option<(String, String)> {
    let fn_obj = v.get("function").unwrap_or(v);
    let name = fn_obj.get("name").and_then(Json::as_str)?.to_string();
    let args_val = fn_obj.get("arguments")?;
    let arguments = match args_val {
        Json::Str(s) => s.clone(),
        other => other.stringify(),
    };
    Some((name, arguments))
}

/// Extracts non-`<tool_call>` invocations (`<tool .../>`, fenced JSON tool blocks, and line-start
/// bare `{"name": ..., "arguments": ...}` objects) from `span` in exact left-to-right document order.
fn extract_non_xml_tool_calls_in_span(
    span: &str,
    prose_out: &mut String,
    calls_out: &mut Vec<ToolCall>,
) {
    if span.is_empty() {
        return;
    }
    let flush_gap = |gap: &str, p_out: &mut String, c_out: &mut Vec<ToolCall>| {
        if gap.is_empty() {
            return;
        }
        let mut cur = gap.to_string();
        if cur.contains("<tool ") || cur.contains("<tool\t") {
            let (xml_p, xml_c) = parse_xml_attr_tool_calls(&cur);
            if !xml_c.is_empty() {
                for mut tc in xml_c {
                    tc.id = format!("call_{}", c_out.len());
                    c_out.push(tc);
                }
                cur = xml_p;
            }
        }
        if cur.contains("\"name\"") && cur.contains("\"arguments\"") {
            let (bare_p, bare_c) = parse_bare_json_tool_calls(&cur);
            if !bare_c.is_empty() {
                for mut tc in bare_c {
                    tc.id = format!("call_{}", c_out.len());
                    c_out.push(tc);
                }
                cur = bare_p;
            }
        }
        p_out.push_str(&cur);
    };

    if !span.contains("```") {
        flush_gap(span, prose_out, calls_out);
        return;
    }

    const FENCE: &str = crate::config::markup::CODE_FENCE;
    let mut cursor = 0usize;
    while let Some(rel) = span[cursor..].find(FENCE) {
        let fence_start = cursor + rel;
        let after_ticks = fence_start + FENCE.len();
        let lang_end = span[after_ticks..]
            .find(['\n', '{', '['])
            .map(|i| after_ticks + i)
            .unwrap_or(after_ticks);
        let lang_tag = span[after_ticks..lang_end].trim();
        let is_tool_fence_lang = lang_tag.is_empty()
            || lang_tag.eq_ignore_ascii_case("json")
            || lang_tag.eq_ignore_ascii_case("tool_call")
            || lang_tag.eq_ignore_ascii_case("tool");
        if !is_tool_fence_lang {
            let close_end = span[lang_end..]
                .find("\n```")
                .map(|i| lang_end + i + 4)
                .or_else(|| span[lang_end..].find(FENCE).map(|i| lang_end + i + FENCE.len()))
                .unwrap_or(span.len());
            flush_gap(&span[cursor..close_end], prose_out, calls_out);
            cursor = close_end;
            continue;
        }

        let mut p = lang_end;
        let ws1 = span[p..].trim_start();
        p += span[p..].len() - ws1.len();
        let opened_array = span[p..].starts_with('[');
        if opened_array {
            p += 1;
            let ws2 = span[p..].trim_start();
            p += span[p..].len() - ws2.len();
        }
        if !span[p..].starts_with('{') {
            let close_end = span[lang_end..]
                .find("\n```")
                .map(|i| lang_end + i + 4)
                .or_else(|| span[lang_end..].find(FENCE).map(|i| lang_end + i + FENCE.len()))
                .unwrap_or(span.len());
            flush_gap(&span[cursor..close_end], prose_out, calls_out);
            cursor = close_end;
            continue;
        }

        let mut block_calls: Vec<(String, String)> = Vec::new();
        let mut scan = p;
        while span[scan..].starts_with('{') {
            let primary = find_balanced_json_object_end(&span[scan..]).and_then(|len| {
                let end = scan + len;
                let parsed = parse_json_lenient(&span[scan..end])
                    .and_then(|v| extract_explicit_tool_name_and_args(&v))?;
                Some((end, parsed))
            });
            let Some((obj_end, pair)) = primary else {
                break;
            };
            block_calls.push(pair);
            scan = obj_end;
            loop {
                let t = span[scan..].trim_start();
                scan += span[scan..].len() - t.len();
                if span[scan..].starts_with(',') {
                    scan += 1;
                } else {
                    break;
                }
            }
        }

        // If an object inside this fence broke brace balance due to unescaped inner quotes,
        // locate the closing ``` fence preceded by `}` and rescue via `repair_tool_call_inner_quotes`.
        if span[scan..].starts_with('{') {
            let mut close_fence: Option<usize> = None;
            let mut fscan = scan;
            let next_json_fence = span[scan + 1..]
                .find("```json")
                .map(|i| scan + 1 + i)
                .unwrap_or(span.len());
            while fscan < next_json_fence {
                let Some(frel) = span[fscan..next_json_fence].find(FENCE) else {
                    break;
                };
                let fpos = fscan + frel;
                let candidate = span[scan..fpos]
                    .trim_end()
                    .trim_end_matches(']')
                    .trim_end();
                if candidate.ends_with('}') {
                    close_fence = Some(fpos);
                }
                fscan = fpos + FENCE.len();
            }
            let end_bound = close_fence.unwrap_or(next_json_fence);
            let candidate = span[scan..end_bound]
                .trim_end()
                .trim_end_matches(']')
                .trim_end();
            if let Some(pair) = repair_tool_call_inner_quotes(candidate)
                .and_then(|v| extract_explicit_tool_name_and_args(&v))
            {
                block_calls.push(pair);
                scan = if close_fence.is_some() {
                    end_bound + FENCE.len()
                } else {
                    end_bound
                };
            }
        }

        if block_calls.is_empty() {
            let close_end = span[lang_end..]
                .find("\n```")
                .map(|i| lang_end + i + 4)
                .or_else(|| span[lang_end..].find(FENCE).map(|i| lang_end + i + FENCE.len()))
                .unwrap_or(span.len());
            flush_gap(&span[cursor..close_end], prose_out, calls_out);
            cursor = close_end;
            continue;
        }

        flush_gap(&span[cursor..fence_start], prose_out, calls_out);
        for (name, arguments) in block_calls {
            calls_out.push(ToolCall {
                id: format!("call_{}", calls_out.len()),
                name,
                arguments,
            });
        }
        let mut after = scan;
        if opened_array {
            let t0 = span[after..].trim_start();
            after += span[after..].len() - t0.len();
            if span[after..].starts_with(']') {
                after += 1;
            }
        }
        let t1 = span[after..].trim_start();
        after += span[after..].len() - t1.len();
        if span[after..].starts_with(FENCE) {
            after += FENCE.len();
        }
        cursor = after;
    }

    if cursor < span.len() {
        flush_gap(&span[cursor..], prose_out, calls_out);
    }
}

/// Extracts Qwen2.5 `<tool_call>...</tool_call>` JSON invocations (including multi-object
/// `<tool_call>` blocks and interleaved fenced/bare JSON tool blocks in exact document order),
/// returning `(remaining_prose, parsed_tool_calls)` using JSON-string-aware brace matching.
pub fn parse_qwen_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    const OPEN_TAG: &str = crate::config::markup::TOOL_CALL_OPEN;
    const CLOSE_TAG: &str = crate::config::markup::TOOL_CALL_CLOSE;

    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;

    while let Some(rel) = text[cursor..].find(OPEN_TAG) {
        let tag_start = cursor + rel;
        extract_non_xml_tool_calls_in_span(&text[cursor..tag_start], &mut prose, &mut calls);
        let mut p = tag_start + OPEN_TAG.len();
        let rest = &text[p..];

        // Skip whitespace, optional ```json / ``` fence, and optional `[` array wrapper.
        let trimmed_ws = rest.trim_start();
        p += rest.len() - trimmed_ws.len();
        if text[p..].starts_with("```") {
            p += 3;
            if let Some(nl) = text[p..].find('\n') {
                p += nl + 1;
            }
        }
        let ws_arr = text[p..].trim_start();
        p += text[p..].len() - ws_arr.len();
        let opened_array = text[p..].starts_with('[');
        if opened_array {
            p += 1;
        }

        let next_open_guard = find_next_real_tool_open(text, tag_start + OPEN_TAG.len())
            .unwrap_or(text.len());
        let Some(brace_rel) = text[p..next_open_guard].find('{') else {
            prose.push_str(&text[tag_start..next_open_guard]);
            cursor = next_open_guard;
            continue;
        };
        let mut obj_start = p + brace_rel;
        let mut extracted_in_tag = 0usize;

        while text[obj_start..].starts_with('{') {
            let primary = find_balanced_json_object_end(&text[obj_start..]).and_then(|len| {
                let obj_end = obj_start + len;
                let parsed = parse_json_lenient(&text[obj_start..obj_end])
                    .and_then(|v| extract_name_and_args(&v))?;
                Some((obj_end, parsed))
            });
            let Some((obj_end, (name, arguments))) = primary else {
                break;
            };
            calls.push(ToolCall {
                id: format!("call_{}", calls.len()),
                name,
                arguments,
            });
            extracted_in_tag += 1;
            let mut after = obj_end;
            loop {
                let t = text[after..].trim_start();
                after += text[after..].len() - t.len();
                if text[after..].starts_with(',') {
                    after += 1;
                } else {
                    break;
                }
            }
            obj_start = after;
        }

        if extracted_in_tag > 0 && !text[obj_start..].starts_with('{') {
            let mut after = obj_start;
            if opened_array && text[after..].starts_with(']') {
                after += 1;
                let t0 = text[after..].trim_start();
                after += text[after..].len() - t0.len();
            }
            if text[after..].starts_with("```") {
                after += 3;
                let t2 = text[after..].trim_start();
                after += text[after..].len() - t2.len();
            }
            if text[after..].starts_with(CLOSE_TAG) {
                after += CLOSE_TAG.len();
            }
            cursor = after;
            continue;
        }

        // Fallback: if unescaped inner quotes or non-standard escapes broke brace balance,
        // rescue using the enclosing `</tool_call>` boundary without aborting later `<tool_call>` blocks.
        if let Some(close_pos) = find_enclosing_tool_close(text, obj_start) {
            let candidate = text[obj_start..close_pos]
                .trim_end()
                .trim_end_matches("```")
                .trim_end()
                .trim_end_matches(']')
                .trim_end();
            if let Some((name, arguments)) =
                repair_tool_call_inner_quotes(candidate).and_then(|v| extract_name_and_args(&v))
            {
                calls.push(ToolCall {
                    id: format!("call_{}", calls.len()),
                    name,
                    arguments,
                });
                cursor = close_pos + CLOSE_TAG.len();
                continue;
            }
            if extracted_in_tag == 0 {
                prose.push_str(&text[tag_start..close_pos + CLOSE_TAG.len()]);
            } else {
                prose.push_str(&text[obj_start..close_pos + CLOSE_TAG.len()]);
            }
            cursor = close_pos + CLOSE_TAG.len();
            continue;
        }

        // Unclosed `<tool_call>` followed by another `<tool_call>`: rescue up to `next_open`
        // and continue scanning subsequent `<tool_call>` blocks.
        if let Some(next_open) = find_next_real_tool_open(text, obj_start + 1) {
            let between = text[obj_start..next_open]
                .trim_end()
                .trim_end_matches("```")
                .trim_end()
                .trim_end_matches(']')
                .trim_end();
            if let Some((name, arguments)) =
                repair_tool_call_inner_quotes(between).and_then(|v| extract_name_and_args(&v))
            {
                calls.push(ToolCall {
                    id: format!("call_{}", calls.len()),
                    name,
                    arguments,
                });
            } else if extracted_in_tag == 0 {
                prose.push_str(&text[tag_start..next_open]);
            } else {
                prose.push_str(&text[obj_start..next_open]);
            }
            cursor = next_open;
            continue;
        }

        // Unclosed `<tool_call>` at end of turn: rescue if the JSON object closed before `<|im_end|>` without `</tool_call>`.
        let tail_candidate = text[obj_start..]
            .trim_end()
            .trim_end_matches("```")
            .trim_end()
            .trim_end_matches(']')
            .trim_end();
        if let Some((name, arguments)) =
            repair_tool_call_inner_quotes(tail_candidate).and_then(|v| extract_name_and_args(&v))
        {
            calls.push(ToolCall {
                id: format!("call_{}", calls.len()),
                name,
                arguments,
            });
            cursor = text.len();
            break;
        }
        if extracted_in_tag == 0 {
            prose.push_str(&text[tag_start..]);
        } else {
            prose.push_str(&text[obj_start..]);
        }
        cursor = text.len();
        break;
    }

    if cursor < text.len() {
        extract_non_xml_tool_calls_in_span(&text[cursor..], &mut prose, &mut calls);
    }
    (strip_empty_tool_wrappers(&prose), calls)
}

fn parse_xml_attr_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    const TAG: &str = crate::config::markup::XML_TOOL_OPEN;
    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;

    while let Some(rel) = text[cursor..].find(TAG) {
        let tag_start = cursor + rel;
        let mut p = tag_start + TAG.len();
        let Some(name_pos) = text[p..].find("name=") else {
            prose.push_str(&text[cursor..p]);
            cursor = p;
            continue;
        };
        let nq_pos = p + name_pos + "name=".len();
        let Some(&nq) = text.as_bytes().get(nq_pos) else {
            break;
        };
        if nq != b'"' && nq != b'\'' {
            prose.push_str(&text[cursor..nq_pos]);
            cursor = nq_pos;
            continue;
        }
        let name_start = nq_pos + 1;
        let Some(name_end_rel) = text[name_start..].find(nq as char) else {
            break;
        };
        let name = text[name_start..name_start + name_end_rel].trim().to_string();
        p = name_start + name_end_rel + 1;

        let Some(arg_pos) = text[p..].find("arguments=") else {
            prose.push_str(&text[cursor..p]);
            cursor = p;
            continue;
        };
        let aq_pos = p + arg_pos + "arguments=".len();
        let Some(&aq) = text.as_bytes().get(aq_pos) else {
            break;
        };
        let brace_search_start = if aq == b'"' || aq == b'\'' {
            aq_pos + 1
        } else {
            aq_pos
        };
        let trimmed = text[brace_search_start..].trim_start();
        let obj_start = brace_search_start + (text[brace_search_start..].len() - trimmed.len());
        let Some(obj_len) = find_balanced_json_object_end(&text[obj_start..]) else {
            prose.push_str(&text[cursor..obj_start]);
            cursor = obj_start;
            continue;
        };
        let obj_end = obj_start + obj_len;
        let raw_args = &text[obj_start..obj_end];
        let Some(parsed_args) = parse_json_lenient(raw_args) else {
            prose.push_str(&text[cursor..obj_end]);
            cursor = obj_end;
            continue;
        };
        prose.push_str(&text[cursor..tag_start]);
        calls.push(ToolCall {
            id: format!("call_{}", calls.len()),
            name,
            arguments: parsed_args.stringify(),
        });
        let mut after = obj_end;
        if text.as_bytes().get(after).copied() == Some(aq) {
            after += 1;
        }
        let tail_ws = text[after..].trim_start();
        after += text[after..].len() - tail_ws.len();
        if text[after..].starts_with("/>") {
            after += 2;
        } else if text[after..].starts_with('>') {
            after += 1;
            let t2 = text[after..].trim_start();
            after += text[after..].len() - t2.len();
            if text[after..].starts_with("</tool>") {
                after += "</tool>".len();
            }
        }
        cursor = after;
    }
    if cursor < text.len() {
        prose.push_str(&text[cursor..]);
    }
    (prose, calls)
}

fn strip_empty_tool_wrappers(s: &str) -> String {
    let mut cur = s.to_string();
    loop {
        let mut changed = false;
        for (open, close) in [
            (crate::config::markup::TOOLS_OPEN, crate::config::markup::TOOLS_CLOSE),
            (crate::config::markup::TOOL_CALL_OPEN, crate::config::markup::TOOL_CALL_CLOSE),
            ("```xml", "```"),
            ("```json", "```"),
        ] {
            if let Some(i) = cur.find(open) {
                let body_start = i + open.len();
                if let Some(rel_end) = cur[body_start..].find(close) {
                    let body = &cur[body_start..body_start + rel_end];
                    if body.trim().is_empty() {
                        let end = body_start + rel_end + close.len();
                        cur.replace_range(i..end, "");
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    cur.trim().to_string()
}

fn parse_bare_json_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut cursor = 0usize;

    while let Some(rel) = text[cursor..].find('{') {
        let obj_start = cursor + rel;
        // Require the bare JSON tool object to start at the beginning of a line (after optional spaces).
        let line_prefix = text[cursor..obj_start]
            .rsplit('\n')
            .next()
            .unwrap_or("");
        if !line_prefix.trim().is_empty() {
            prose.push_str(&text[cursor..=obj_start]);
            cursor = obj_start + 1;
            continue;
        }
        let Some(obj_len) = find_balanced_json_object_end(&text[obj_start..]) else {
            prose.push_str(&text[cursor..=obj_start]);
            cursor = obj_start + 1;
            continue;
        };
        let obj_end = obj_start + obj_len;
        let parsed = parse_json_lenient(&text[obj_start..obj_end]).and_then(|v| {
            let fn_obj = v.get("function").unwrap_or(&v);
            let name = fn_obj.get("name").and_then(Json::as_str)?.to_string();
            let args_val = fn_obj.get("arguments")?;
            let arguments = match args_val {
                Json::Str(s) => s.clone(),
                other => other.stringify(),
            };
            Some((name, arguments))
        });
        let Some((name, arguments)) = parsed else {
            prose.push_str(&text[cursor..obj_end]);
            cursor = obj_end;
            continue;
        };
        prose.push_str(&text[cursor..obj_start]);
        calls.push(ToolCall {
            id: format!("call_{}", calls.len()),
            name,
            arguments,
        });
        let mut after = obj_end;
        let t = text[after..].trim_start();
        let ws_len = text[after..].len() - t.len();
        if t.starts_with(crate::config::markup::TOOL_CALL_CLOSE) {
            after += ws_len + crate::config::markup::TOOL_CALL_CLOSE.len();
        } else if t.starts_with("```") {
            after += ws_len + 3;
        }
        cursor = after;
    }
    if cursor < text.len() {
        prose.push_str(&text[cursor..]);
    }
    (prose.trim().to_string(), calls)
}

fn find_balanced_json_object_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first().copied() != Some(b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    for (idx, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// Drafts up to `max_k` continuation tokens from `prev` (the previous turn's reply) when `reply`
/// matches a prefix or a 2..=3-token n-gram of `prev` (e.g. incremental live translation edits).
fn draft_from_prev_reply(prev: &[usize], reply: &[usize], max_k: usize, out: &mut [usize]) -> usize {
    let r_len = reply.len();
    let limit = max_k.min(out.len());
    if r_len == 0 || prev.len() <= r_len || limit == 0 {
        return 0;
    }
    if prev.starts_with(reply) {
        let count = (prev.len() - r_len).min(limit);
        out[..count].copy_from_slice(&prev[r_len..r_len + count]);
        return count;
    }
    for ngram in [3usize, 2] {
        if r_len < ngram || prev.len() <= ngram {
            continue;
        }
        let pat = &reply[r_len - ngram..];
        for i in (ngram..prev.len()).rev() {
            if &prev[i - ngram..i] == pat {
                let count = (prev.len() - i).min(limit);
                if count >= 2 {
                    out[..count].copy_from_slice(&prev[i..i + count]);
                    return count;
                }
            }
        }
    }
    0
}

/// Scans `cached` + `next` backwards for a 6-, 5-, or 4-token suffix match and copies up to `max_k` continuation tokens.
fn find_suffix_continuation(cached: &[usize], next: usize, max_k: usize, out: &mut [usize]) -> usize {
    let n = cached.len();
    if n < 5 {
        return 0;
    }
    let limit = max_k.min(out.len());
    for ngram in [6usize, 5, 4] {
        if n + 1 < ngram + 2 {
            continue;
        }
        // Suffix ending at `next` has previous `ngram - 1` tokens `cached[n - (ngram - 1)..n]` followed by `next`.
        let prev = &cached[n - (ngram - 1)..n];
        // Search backwards from `n - 2` down to `ngram - 1` for `cached[i] == next` and matching `prev`.
        let start_search = n.saturating_sub(2);
        for i in (ngram - 1..=start_search).rev() {
            if cached[i] == next && &cached[i - (ngram - 1)..i] == prev {
                let mut count = 0usize;
                let mut src = i + 1;
                while count < limit && src <= n {
                    out[count] = if src < n { cached[src] } else { next };
                    count += 1;
                    src += 1;
                }
                if count >= 2 {
                    return count;
                }
            }
        }
    }
    0
}

/// xorshift64* sampler state.
struct Rng(u64);

impl Rng {
    #[inline]
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

#[inline]
fn top1_and_margin(logits: &[f32]) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let mut second_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            second_val = best_val;
            best_val = v;
            best_idx = i;
        } else if v > second_val {
            second_val = v;
        }
    }
    let margin = if second_val.is_finite() { best_val - second_val } else { f32::INFINITY };
    (best_idx, margin)
}

#[inline]
fn apply_repeat_penalty_unique(l: &mut [f32], recent: &[usize], repeat_penalty: f32) {
    let mut seen = [usize::MAX; 128];
    let mut seen_len = 0usize;
    for (idx, &t) in recent.iter().enumerate() {
        if t >= l.len() {
            continue;
        }
        let already_seen = if idx <= seen.len() {
            seen[..seen_len].contains(&t)
        } else {
            recent[..idx].contains(&t)
        };
        if already_seen {
            continue;
        }
        if seen_len < seen.len() {
            seen[seen_len] = t;
            seen_len += 1;
        }
        l[t] = if l[t] > 0.0 {
            l[t] / repeat_penalty
        } else {
            l[t] * repeat_penalty
        };
    }
}

#[inline]
fn sample_greedy_with_margin(
    logits: &[f32],
    recent: &[usize],
    repeat_penalty: f32,
    scratch: &mut [f32],
) -> (usize, f32) {
    if recent.is_empty() || (repeat_penalty - 1.0).abs() <= f32::EPSILON {
        return top1_and_margin(logits);
    }
    if repeat_penalty >= 1.0 {
        let (best, margin) = top1_and_margin(logits);
        if !recent.contains(&best) {
            return (best, margin);
        }
    }
    let l = &mut scratch[..logits.len()];
    l.copy_from_slice(logits);
    apply_repeat_penalty_unique(l, recent, repeat_penalty);
    top1_and_margin(l)
}

/// Zero-copy greedy argmax with repetition penalty.
#[inline]
fn sample_greedy(logits: &[f32], recent: &[usize], repeat_penalty: f32, scratch: &mut [f32]) -> usize {
    if recent.is_empty() || (repeat_penalty - 1.0).abs() <= f32::EPSILON {
        return argmax(logits);
    }
    if repeat_penalty >= 1.0 {
        let best = argmax(logits);
        if !recent.contains(&best) {
            return best;
        }
    }
    let l = &mut scratch[..logits.len()];
    l.copy_from_slice(logits);
    apply_repeat_penalty_unique(l, recent, repeat_penalty);
    argmax(l)
}

#[inline]
fn sample_token(
    logits: &[f32],
    recent: &[usize],
    p: &GenParams,
    rng: &mut Rng,
    pen_scratch: &mut [f32],
    heap_scratch: &mut Vec<(usize, f32)>,
) -> usize {
    if p.temperature <= 0.0 {
        return sample_greedy(logits, recent, p.repeat_penalty, pen_scratch);
    }
    let mut dist = Vec::new();
    std::mem::swap(&mut dist, heap_scratch);
    let mut tmp_heap = Vec::with_capacity(p.top_k.clamp(1, 128));
    build_sampling_dist(logits, recent, p, pen_scratch, &mut tmp_heap, &mut dist);
    let tok = sample_from_dist(&dist, rng);
    std::mem::swap(&mut dist, heap_scratch);
    tok
}

/// Builds the normalized `(token_id, prob)` `top_k` + `top_p` distribution in a single `O(V)` scan
/// using a `k`-element binary min-heap (`0` `49,152`-index allocations).
fn build_sampling_dist(
    logits: &[f32],
    recent: &[usize],
    p: &GenParams,
    pen_scratch: &mut [f32],
    heap: &mut Vec<(usize, f32)>,
    out_dist: &mut Vec<(usize, f32)>,
) {
    let slice: &[f32] = if recent.is_empty() || (p.repeat_penalty - 1.0).abs() <= f32::EPSILON {
        logits
    } else {
        let l = &mut pen_scratch[..logits.len()];
        l.copy_from_slice(logits);
        apply_repeat_penalty_unique(l, recent, p.repeat_penalty);
        l
    };

    let k = p.top_k.clamp(1, slice.len());
    heap.clear();
    // Min-heap ordered so the SMALLEST logit among the current top-k sits at `heap[0]`
    // (tie-breaking favors smaller index on ties).
    let less_in_topk = |a: (usize, f32), b: (usize, f32)| -> bool {
        a.1.total_cmp(&b.1).then_with(|| b.0.cmp(&a.0)).is_lt()
    };

    for (idx, &val) in slice.iter().enumerate() {
        if heap.len() < k {
            heap.push((idx, val));
            if heap.len() == k {
                // Floyd bottom-up heapify
                for start in (0..k / 2).rev() {
                    sift_down_min(heap, start, less_in_topk);
                }
            }
        } else if less_in_topk(heap[0], (idx, val)) {
            heap[0] = (idx, val);
            sift_down_min(heap, 0, less_in_topk);
        }
    }

    // Sort top-k descending by logit
    heap.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let max_l = heap[0].1;
    let inv_temp = 1.0 / p.temperature;

    out_dist.clear();
    let mut total = 0.0f32;
    for &(idx, val) in heap.iter() {
        let e = ((val - max_l) * inv_temp).exp();
        out_dist.push((idx, e));
        total += e;
    }

    let mut cum = 0.0f32;
    let mut keep = out_dist.len();
    for (n, entry) in out_dist.iter_mut().enumerate() {
        entry.1 /= total;
        cum += entry.1;
        if cum >= p.top_p {
            keep = n + 1;
            break;
        }
    }
    out_dist.truncate(keep);
    let kept_sum: f32 = out_dist.iter().map(|e| e.1).sum();
    if kept_sum > 0.0 {
        let inv_kept = 1.0 / kept_sum;
        for entry in out_dist.iter_mut() {
            entry.1 *= inv_kept;
        }
    }
}

#[inline]
fn sift_down_min(heap: &mut [(usize, f32)], mut pos: usize, less: impl Fn((usize, f32), (usize, f32)) -> bool) {
    let len = heap.len();
    loop {
        let left = 2 * pos + 1;
        if left >= len {
            break;
        }
        let right = left + 1;
        let smallest = if right < len && less(heap[right], heap[left]) { right } else { left };
        if !less(heap[smallest], heap[pos]) {
            break;
        }
        heap.swap(pos, smallest);
        pos = smallest;
    }
}

#[inline]
fn prob_in_dist(dist: &[(usize, f32)], tok: usize) -> f32 {
    for &(id, prob) in dist {
        if id == tok {
            return prob;
        }
    }
    0.0
}

#[inline]
fn sample_from_dist(dist: &[(usize, f32)], rng: &mut Rng) -> usize {
    let mut r = rng.next_f32();
    for &(id, prob) in dist {
        r -= prob;
        if r <= 0.0 {
            return id;
        }
    }
    dist.last().map_or(0, |e| e.0)
}

/// Samples from $\text{norm}(\max(0, p(x) - q(x)))$ over `p`'s sparse `top_k` support (Leviathan et al. 2023).
fn sample_residual_dist(p_dist: &[(usize, f32)], q_dist: &[(usize, f32)], rng: &mut Rng) -> usize {
    let mut resid_sum = 0.0f32;
    for &(id, pv) in p_dist {
        let diff = (pv - prob_in_dist(q_dist, id)).max(0.0);
        resid_sum += diff;
    }
    if resid_sum <= 1e-12 {
        return sample_from_dist(p_dist, rng);
    }
    let mut r = rng.next_f32() * resid_sum;
    let mut last_pos = p_dist[0].0;
    for &(id, pv) in p_dist {
        let diff = (pv - prob_in_dist(q_dist, id)).max(0.0);
        if diff > 0.0 {
            last_pos = id;
            r -= diff;
            if r <= 0.0 {
                return id;
            }
        }
    }
    last_pos
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xml_tool_format_round_trip() {
        let raw = "Let me write it.\n<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>\n<parameter=content>\nprint(1)\n\nprint(2)\n</parameter>\n<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>";
        let (prose, calls) = parse_xml_tool_calls(raw);
        assert_eq!(prose, "Let me write it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "write_file");
        assert_eq!(calls[0].arguments, "{\"path\":\"a.py\",\"content\":\"print(1)\\n\\nprint(2)\",\"count\":3}");

        let tools = vec![r#"{"type":"function","function":{"name":"write_file","description":"Write a file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative path"}},"required":["path"]}}}"#.to_string()];
        let msgs = vec![
            Message::new("user", "go"),
            Message { role: "assistant".into(), content: String::new(), tool_calls: calls.clone(), tool_call_id: None },
            Message::tool_result("call_0", "ok"),
        ];
        let p = render_chatml_prompt_with(&msgs, &tools, "sys", ToolFormat::Xml);
        // HunYuan: `chat_template.jinja` of Hy-MT2 (no newlines, tools dropped, no default system).
        let hy = render_prompt_with(
            &[Message::new("system", "sys"), Message::new("user", "hi")],
            &tools,
            "",
            ToolFormat::Json,
            PromptStyle::HunYuan,
        );
        assert_eq!(hy, "<｜hy_begin▁of▁sentence｜>sys<｜hy_place▁holder▁no▁3｜><｜hy_User｜>hi<｜hy_Assistant｜>");
        let hy_turns = render_prompt_with(
            &[Message::new("user", "a"), Message::new("assistant", "b"), Message::new("user", "c")],
            &[],
            "",
            ToolFormat::Json,
            PromptStyle::HunYuan,
        );
        assert_eq!(
            hy_turns,
            "<｜hy_begin▁of▁sentence｜><｜hy_User｜>a<｜hy_Assistant｜>b<｜hy_place▁holder▁no▁2｜><｜hy_User｜>c<｜hy_Assistant｜>"
        );
        assert!(p.contains("<function>\n<name>write_file</name>\n<description>Write a file.</description>\n<parameters>\n<parameter>\n<name>path</name>\n<type>string</type>"));
        assert!(p.contains("<required>[\"path\"]</required>"));
        assert!(p.contains("<tool_call>\n<function=write_file>\n<parameter=path>\na.py\n</parameter>"));
        assert!(p.contains("<parameter=count>\n3\n</parameter>\n</function>\n</tool_call><|im_end|>"));
        assert!(p.contains("<tool_response>\nok\n</tool_response>"));
        assert_eq!(ToolFormat::Xml.call_prefix(), "<tool_call>\n<function=");
    }

    #[test]
    fn test_zero_alloc_topk_dist_matches_reference() {
        let logits: Vec<f32> = (0..1024).map(|i| ((i as f32) * 0.37).sin() * 4.0).collect();
        let recent = [17usize, 99, 17, 512];
        let p = GenParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.15,
            max_tokens: 32,
            seed: 42,
        };
        let mut pen = vec![0.0f32; logits.len()];
        let mut heap = Vec::new();
        let mut dist = Vec::new();
        build_sampling_dist(&logits, &recent, &p, &mut pen, &mut heap, &mut dist);

        let sum: f32 = dist.iter().map(|e| e.1).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(dist.len() <= 40);

        let g = sample_greedy(&logits, &recent, p.repeat_penalty, &mut pen);
        let mut ref_l = logits.clone();
        for &t in &[17usize, 99, 512] {
            ref_l[t] = if ref_l[t] > 0.0 { ref_l[t] / p.repeat_penalty } else { ref_l[t] * p.repeat_penalty };
        }
        assert_eq!(g, argmax(&ref_l));
        let mut dup_probe = vec![2.0f32, 1.90f32];
        apply_repeat_penalty_unique(&mut dup_probe, &[0, 0, 0, 0], 1.05);
        assert!((dup_probe[0] - (2.0f32 / 1.05f32)).abs() < 1e-6);
    }

    #[test]
    fn test_suffix_continuation_lookup() {
        let history = vec![10, 20, 30, 40, 50, 60, 70, 10, 20, 30];
        let mut out = [0usize; 8];
        let n = find_suffix_continuation(&history, 40, 4, &mut out);
        assert_eq!(n, 4);
        assert_eq!(&out[..4], &[50, 60, 70, 10]);
    }

    #[test]
    fn test_qwen25_tool_chat_template_exact_jinja_match() {
        let tools = vec![
            r#"{"type":"function","function":{"name":"write_file","parameters":{"type":"object"}}}"#.to_string(),
        ];
        let msgs = vec![
            Message::new("user", "Build index.html"),
            Message::assistant_tool_calls(
                "",
                vec![
                    ToolCall {
                        id: "call_0".to_string(),
                        name: "write_file".to_string(),
                        arguments: r#"{"path":"index.html","content":"<h1>Hi</h1>"}"#.to_string(),
                    },
                    ToolCall {
                        id: "call_1".to_string(),
                        name: "list_dir".to_string(),
                        arguments: r#"{"path":"."}"#.to_string(),
                    },
                ],
            ),
            Message::tool_result("call_0", "wrote 11 bytes"),
            Message::tool_result("call_1", "index.html"),
        ];

        let rendered = render_chatml_prompt(&msgs, &tools, DEFAULT_SYSTEM_QWEN);
        let expected = format!(
            concat!(
                "<|im_start|>system\n",
                "{}",
                "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n",
                "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n",
                r#"{{"type":"function","function":{{"name":"write_file","parameters":{{"type":"object"}}}}}}"#,
                "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n",
                "<tool_call>\n{{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n</tool_call><|im_end|>\n",
                "<|im_start|>user\nBuild index.html<|im_end|>\n",
                "<|im_start|>assistant\n<tool_call>\n",
                r#"{{"name": "write_file", "arguments": {{"path":"index.html","content":"<h1>Hi</h1>"}}}}"#,
                "\n</tool_call>\n<tool_call>\n",
                r#"{{"name": "list_dir", "arguments": {{"path":"."}}}}"#,
                "\n</tool_call><|im_end|>\n",
                "<|im_start|>user\n<tool_response>\nwrote 11 bytes\n</tool_response>\n<tool_response>\nindex.html\n</tool_response><|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
            DEFAULT_SYSTEM_QWEN
        );
        assert_eq!(rendered, expected);
    }

    #[test]
    fn test_parse_qwen_keeps_write_before_run_with_unescaped_python_quotes() {
        // A write_file whose Python payload has raw unescaped `"` (breaking brace balance)
        // followed by a run_command in the same turn must yield both calls in order.
        let raw = concat!(
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "build_video.py", "content": "import os\nmsg = "hello {world}"\ndoc = """triple"""\nprint(msg)\n"}}"#,
            "\n</tool_call>\n",
            "<tool_call>\n",
            r#"{"name": "run_command", "arguments": {"cmd": "python3 build_video.py"}}"#,
            "\n</tool_call>",
        );
        let (_prose, calls) = parse_qwen_tool_calls(raw);
        assert_eq!(calls.len(), 2, "calls: {calls:?}");
        assert_eq!(calls[0].name, "write_file");
        let a0 = Json::parse(&calls[0].arguments).unwrap();
        assert_eq!(a0.get("path").and_then(Json::as_str), Some("build_video.py"));
        let content = a0.get("content").and_then(Json::as_str).unwrap();
        assert!(content.contains("msg = \"hello {world}\""), "content: {content}");
        assert!(content.contains("\"\"\"triple\"\"\""), "content: {content}");
        assert_eq!(calls[1].name, "run_command");
    }

    #[test]
    fn test_parse_qwen_extracts_fenced_and_bare_calls_after_first_tool_call() {
        // Calls 2..N emitted as fenced JSON or bare JSON (without repeating `<tool_call>`)
        // must not be dropped once the first `<tool_call>` parsed.
        let raw = concat!(
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "styles/tokens.css", "content": ":root { --bg: #0b1120; }"}}"#,
            "\n</tool_call>\nNext the layout file:\n```json\n",
            r#"{"name": "write_file", "arguments": {"path": "styles/layout.css", "content": "body { margin: 0; }"}}"#,
            "\n```\nAnd the script:\n",
            r#"{"name": "write_file", "arguments": {"path": "js/app.js", "content": "console.log('ok');"}}"#,
            "\n",
        );
        let (prose, calls) = parse_qwen_tool_calls(raw);
        let paths: Vec<String> = calls
            .iter()
            .map(|c| {
                Json::parse(&c.arguments)
                    .ok()
                    .and_then(|v| v.get("path").and_then(Json::as_str).map(str::to_string))
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(paths, vec!["styles/tokens.css", "styles/layout.css", "js/app.js"]);
        assert!(!prose.contains("\"arguments\""), "prose leaked tool JSON: {prose}");
    }

    #[test]
    fn test_parse_qwen_tool_calls_with_embedded_tags_and_fences() {
        let raw = concat!(
            "Creating the file and running a quick check.\n",
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "doc.html", "content": "<pre>literal </tool_call> inside HTML</pre>"}}"#,
            "\n</tool_call>\n",
            "<tool_call>\n```json\n",
            r#"{"name": "run_command", "arguments": {"cmd": "ls -la"}}"#,
            "\n```\n</tool_call>",
        );
        let (prose, calls) = parse_qwen_tool_calls(raw);
        assert_eq!(prose, "Creating the file and running a quick check.");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "write_file");
        let args0 = Json::parse(&calls[0].arguments).unwrap();
        assert_eq!(
            args0.get("content").and_then(Json::as_str),
            Some("<pre>literal </tool_call> inside HTML</pre>")
        );
        assert_eq!(calls[1].name, "run_command");
        let args1 = Json::parse(&calls[1].arguments).unwrap();
        assert_eq!(args1.get("cmd").and_then(Json::as_str), Some("ls -la"));

        let bare = concat!(
            "Checking CSS and JS endpoints:\n",
            r#"{"name": "run_command", "arguments": {"cmd": "curl -fsSI http://127.0.0.1:18123/style.css"}}"#,
            "\n",
            r#"{"name": "run_command", "arguments": {"cmd": "curl -fsSI http://127.0.0.1:18123/app.js"}}"#,
            "\n```",
        );
        let (b_prose, b_calls) = parse_qwen_tool_calls(bare);
        assert_eq!(b_prose, "Checking CSS and JS endpoints:");
        assert_eq!(b_calls.len(), 2);
        assert_eq!(b_calls[0].name, "run_command");
        assert_eq!(b_calls[1].name, "run_command");

        let multiline_css = concat!(
            "<tool_call>\n",
            "{\"name\": \"write_file\", \"arguments\": {\"path\": \"style.css\", \"content\": \"body {\n\tbackground: #0f172a;\n\tcolor: #f8fafc;\n}\"}}\n",
            "</tool_call>",
        );
        let (_, m_calls) = parse_qwen_tool_calls(multiline_css);
        assert_eq!(m_calls.len(), 1);
        assert_eq!(m_calls[0].name, "write_file");
        let m_args = Json::parse(&m_calls[0].arguments).unwrap();
        assert_eq!(
            m_args.get("content").and_then(Json::as_str),
            Some("body {\n\tbackground: #0f172a;\n\tcolor: #f8fafc;\n}")
        );

        let generic_fence = "Applying styles:\n```\n{\"name\":\"edit_file\",\"arguments\":{\"path\":\"style.css\",\"old_text\":\"\",\"new_text\":\"h1 { color: teal; }\"}}\n```";
        let (gf_prose, gf_calls) = parse_qwen_tool_calls(generic_fence);
        assert_eq!(gf_prose, "Applying styles:");
        assert_eq!(gf_calls.len(), 1);
        assert_eq!(gf_calls[0].name, "edit_file");

        let xml_attr = concat!(
            "Updating script:\n```xml\n<tools>\n",
            "  <tool name=\"edit_file\" arguments='{\"path\":\"script.md\",\"old_text\":\"a\",\"new_text\":\"Here's line 1\nLine 2\"}' />\n",
            "</tools>\n```",
        );
        let (xa_prose, xa_calls) = parse_qwen_tool_calls(xml_attr);
        assert_eq!(xa_prose, "Updating script:");
        assert_eq!(xa_calls.len(), 1);
        assert_eq!(xa_calls[0].name, "edit_file");
        let xa_args = Json::parse(&xa_calls[0].arguments).unwrap();
        assert_eq!(
            xa_args.get("new_text").and_then(Json::as_str),
            Some("Here's line 1\nLine 2")
        );

        // Non-RFC8259 JS/shell backslash escapes (`\'`, `\ `, `\d`, `\u{...}`) and unescaped inner HTML quotes.
        let sloppy_coder = concat!(
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "tests/verify_app.mjs", "content": "const chrome = '/Applications/Google\ Chrome.app';\nconst re = /\d+\s+/;\nconst u = '\u{1F680}';\nconst msg = 'it\'s ok';\n"}}"#,
            "\n</tool_call>\n",
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "index.html", "content": "<div class="kpi-card" data-id="uptime">99.98%</div>"}}"#,
            "\n</tool_call>\n",
            "<tool_call>\n",
            r#"{"name": "run_command", "arguments": {"cmd": "node tests/verify_app.mjs"}}"#,
            "\n</tool_call>",
        );
        let (sc_prose, sc_calls) = parse_qwen_tool_calls(sloppy_coder);
        assert!(sc_prose.is_empty(), "unexpected leftover prose: {sc_prose}");
        assert_eq!(sc_calls.len(), 3);
        assert_eq!(sc_calls[0].name, "write_file");
        let sc0 = Json::parse(&sc_calls[0].arguments).unwrap();
        let sc0_content = sc0.get("content").and_then(Json::as_str).unwrap();
        assert!(sc0_content.contains(r"Google\ Chrome.app"));
        assert!(sc0_content.contains(r"/\d+\s+/"));
        assert!(sc0_content.contains(r"\u{1F680}"));
        assert!(sc0_content.contains(r"const msg = 'it\'s ok';"));
        assert_eq!(sc_calls[1].name, "write_file");
        let sc1 = Json::parse(&sc_calls[1].arguments).unwrap();
        assert_eq!(
            sc1.get("content").and_then(Json::as_str),
            Some(r#"<div class="kpi-card" data-id="uptime">99.98%</div>"#)
        );
        assert_eq!(sc_calls[2].name, "run_command");

        // Over-escaped single-quote JS delimiters (`owner:\'Core\'`) + backtick-wrapped content without `</tool_call>`.
        let over_escaped_js = concat!(
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "js/data.js", "content": "export const S = [{ owner:\'Core Networking\', region:\'us-east-1', note:\'it\'s live\' }];"}}"#,
            "\n</tool_call>",
        );
        let (_, oe_calls) = parse_qwen_tool_calls(over_escaped_js);
        assert_eq!(oe_calls.len(), 1);
        let oe0 = Json::parse(&oe_calls[0].arguments).unwrap();
        assert_eq!(
            oe0.get("content").and_then(Json::as_str),
            Some(r#"export const S = [{ owner:'Core Networking', region:'us-east-1', note:'it\'s live' }];"#)
        );

        let backtick_unclosed = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"js/state.js\", \"content\": `export const state = { filter: \"all\" };`}}";
        let (_, bt_calls) = parse_qwen_tool_calls(backtick_unclosed);
        assert_eq!(bt_calls.len(), 1);
        let bt0 = Json::parse(&bt_calls[0].arguments).unwrap();
        assert_eq!(
            bt0.get("content").and_then(Json::as_str),
            Some("export const state = { filter: \"all\" };")
        );

        let stray_line_end_bt = "<tool_call>\n{\"name\": \"write_file\", \"arguments\": {\"path\": \"js/render.js\", \"content\": \"function f() {\\n  render();\\`\\n}\\n\"}}\n</tool_call>";
        let (_, sl_calls) = parse_qwen_tool_calls(stray_line_end_bt);
        assert_eq!(sl_calls.len(), 1);
        let sl0 = Json::parse(&sl_calls[0].arguments).unwrap();
        assert_eq!(
            sl0.get("content").and_then(Json::as_str),
            Some("function f() {\n  render();\n}\n")
        );

        // Single-line Python `split('\n')` and `f"file '{s}'\n"` inside JSON string + LaTeX `\frac`/`\times`/`\rangle` + missing outer `}`.
        let py_and_latex = concat!(
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "build_video.py", "content": "title = section.split('\n')[0]\nf.write(f\"file '{slide}'\n\")\n"}"#,
            "\n</tool_call>\n",
            "<tool_call>\n",
            r#"{"name": "write_file", "arguments": {"path": "script.md", "content": "Transform $Q_b = \frac{1}{8} H_{64} D_b$ over $C_1 \times C_2$ with $\langle x, y \rangle$ and `<tool_call>` tags ("Kronos")."}"#,
            "\n</tool_call>",
        );
        let (_, pl_calls) = parse_qwen_tool_calls(py_and_latex);
        assert_eq!(pl_calls.len(), 2);
        let pl0 = Json::parse(&pl_calls[0].arguments).unwrap();
        assert_eq!(
            pl0.get("content").and_then(Json::as_str),
            Some("title = section.split('\\n')[0]\nf.write(f\"file '{slide}'\\n\")\n")
        );
        let pl1 = Json::parse(&pl_calls[1].arguments).unwrap();
        let pl1_c = pl1.get("content").and_then(Json::as_str).unwrap();
        assert!(pl1_c.contains(r"\frac{1}{8}"), "got: {pl1_c}");
        assert!(pl1_c.contains(r"\times"), "got: {pl1_c}");
        assert!(pl1_c.contains(r"\langle"), "got: {pl1_c}");
        assert!(pl1_c.contains(r"\rangle"), "got: {pl1_c}");
        assert!(pl1_c.contains(r#"("Kronos")"#), "got: {pl1_c}");
    }

    #[test]
    fn test_tiered_speculation_matches_plain_greedy_05b() {
        let dir = Path::new("models/Qwen2.5-Coder-0.5B-Instruct");
        if !dir.join("kronos-nf4g128.k2tb").exists() {
            return;
        }
        let threads = crate::kernels::thread_pool::default_inference_threads();
        let mut model = ChatModel::load(dir, "nf4g128", 4096, threads, None).unwrap();
        let file: Vec<String> = (0..60)
            .map(|i| format!("pub fn item_{i}(x: i32) -> i32 {{\n    x * {i} + {}\n}}\n", i * 7))
            .collect();
        let prompts = [
            format!(
                "Here is a Rust file:\n```rust\n{}```\nChange `item_30` so it returns `x * 31`. Reply with the complete file.",
                file.concat()
            ),
            "Explain the difference between a Vec and a slice in Rust in three sentences.".to_string(),
        ];
        let params = GenParams {
            temperature: 0.0,
            top_k: 40,
            top_p: 1.0,
            repeat_penalty: 1.05,
            max_tokens: 160,
            seed: 0,
        };
        let run = |model: &mut ChatModel, p: &str| {
            model.reset_cache();
            let msgs = [Message::new("user", p)];
            model.generate(&msgs, &params, |_| true).unwrap()
        };

        model.set_speculation(false);
        let plain: Vec<Completion> = prompts.iter().map(|p| run(&mut model, p)).collect();
        assert!(plain.iter().all(|c| c.drafted_tokens == 0 && c.completion_tokens > 20));

        model.set_speculation(true);
        for (i, p) in prompts.iter().enumerate() {
            let c = run(&mut model, p);
            assert_eq!(c.token_ids, plain[i].token_ids, "lookup tier changed output on prompt {i}");
            if i == 0 {
                assert!(c.drafted_tokens > 0, "code-edit prompt never used lookup");
            }
            eprintln!(
                "lookup prompt {i}: {} tok, accept {}/{}, {:.1} -> {:.1} tok/s",
                c.completion_tokens,
                c.accepted_tokens,
                c.drafted_tokens,
                plain[i].completion_tokens as f64 / plain[i].decode_secs.max(1e-9),
                c.completion_tokens as f64 / c.decode_secs.max(1e-9)
            );
        }

        model.attach_draft(dir, "nf4g128", None).unwrap();
        for (i, p) in prompts.iter().enumerate() {
            let c = run(&mut model, p);
            assert_eq!(c.token_ids, plain[i].token_ids, "lookup+draft tiers changed output on prompt {i}");
            assert!(c.drafted_tokens > 0);
            eprintln!(
                "lookup+draft prompt {i}: {} tok, accept {}/{}, {:.1} tok/s",
                c.completion_tokens,
                c.accepted_tokens,
                c.drafted_tokens,
                c.completion_tokens as f64 / c.decode_secs.max(1e-9)
            );
        }
    }

    static HEAVY_14B_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_qwen25_14b_draft_05b_greedy_exact_5_coding_prompts() {
        let _guard = HEAVY_14B_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let target_dir = Path::new("models/Qwen2.5-Coder-14B-Instruct");
        let draft_dir = Path::new("models/Qwen2.5-Coder-0.5B-Instruct");
        if !target_dir.join("kronos-nf4g128.k2tb").exists()
            || !draft_dir.join("kronos-nf4g128.k2tb").exists()
        {
            return;
        }

        let threads = crate::kernels::thread_pool::default_inference_threads();
        let mut model = ChatModel::load(target_dir, "nf4g128", 2048, threads, None).unwrap();

        let bad_draft = Path::new("models/SmolLM2-135M-Instruct");
        if bad_draft.join("kronos-nf4g128.k2tb").exists() {
            assert!(model.attach_draft(bad_draft, "nf4g128", None).is_err());
        }

        let prompts = [
            "Write a Rust function `pub fn binary_search(xs: &[i32], target: i32) -> Option<usize>` without comments.",
            "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]`.",
            "Write a JavaScript function `debounce(fn, delayMs)` using `setTimeout` and `clearTimeout`.",
            "Write an SQL query returning the top 3 departments by average salary from `employees(id, dept, salary)`.",
            "Write a C function `size_t str_lcpy(char *dst, const char *src, size_t cap)` that always NUL-terminates when `cap > 0`.",
        ];
        let params = GenParams {
            temperature: 0.0,
            top_k: 40,
            top_p: 1.0,
            repeat_penalty: 1.0,
            max_tokens: 64,
            seed: 0,
        };

        let mut baseline = Vec::with_capacity(prompts.len());
        for p in &prompts {
            model.reset_cache();
            let msgs = [Message::new("user", p)];
            let out = model.generate(&msgs, &params, |_| true).unwrap();
            assert!(!out.token_ids.is_empty());
            baseline.push((out.token_ids, out.text, out.decode_secs));
        }

        model.attach_draft(draft_dir, "nf4g128", None).unwrap();
        assert!(model.has_draft());

        for (idx, p) in prompts.iter().enumerate() {
            model.reset_cache();
            let msgs = [Message::new("user", p)];
            let out = model.generate(&msgs, &params, |_| true).unwrap();
            assert!(out.drafted_tokens > 0, "prompt {idx} did not use draft");
            assert_eq!(
                out.token_ids, baseline[idx].0,
                "token-id mismatch on coding prompt {idx}"
            );
            assert_eq!(out.text, baseline[idx].1, "text mismatch on coding prompt {idx}");
            eprintln!(
                "prompt {}: {} tok | base {:.1} tok/s -> spec {:.1} tok/s | accept {:.1}% ({}/{})",
                idx,
                out.completion_tokens,
                out.completion_tokens as f64 / baseline[idx].2.max(1e-9),
                out.completion_tokens as f64 / out.decode_secs.max(1e-9),
                out.acceptance_rate().unwrap_or(0.0) * 100.0,
                out.accepted_tokens,
                out.drafted_tokens,
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_qwen25_14b_metal_vs_cpu_kl_and_greedy_10_prompts() {
        let _guard = HEAVY_14B_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let target_dir = Path::new("models/Qwen2.5-Coder-14B-Instruct");
        let draft_dir = Path::new("models/Qwen2.5-Coder-0.5B-Instruct");
        if !target_dir.join("kronos-nf4g128.k2tb").exists() {
            return;
        }

        let kl_div = |ref_l: &[f32], test_l: &[f32]| -> f64 {
            let max_r = ref_l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let max_t = test_l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum_r: f64 = ref_l.iter().map(|&v| ((v as f64) - max_r).exp()).sum();
            let sum_t: f64 = test_l.iter().map(|&v| ((v as f64) - max_t).exp()).sum();
            let lse_r = max_r + sum_r.ln();
            let lse_t = max_t + sum_t.ln();
            let mut kl = 0.0f64;
            for (&rv, &tv) in ref_l.iter().zip(test_l.iter()) {
                let log_p = (rv as f64) - lse_r;
                let log_q = (tv as f64) - lse_t;
                let p = log_p.exp();
                if p > 1e-15 {
                    kl += p * (log_p - log_q);
                }
            }
            kl
        };

        let threads = crate::kernels::thread_pool::default_inference_threads();
        let mut model = ChatModel::load(target_dir, "nf4g128", 2048, threads, None).unwrap();

        let prompts = [
            "Write a Rust function `pub fn gcd(mut a: u64, mut b: u64) -> u64` using Euclid's algorithm.",
            "Write a Python function `toposort(adj: dict[str, list[str]]) -> list[str]` using Kahn's algorithm.",
            "Write a JavaScript function `lruCache(capacity)` returning `{ get, put }` backed by a `Map`.",
            "Write an SQL query with a CTE that finds consecutive 3-day login streaks in `logins(user_id, day)`.",
            "Write a C function `uint32_t fnv1a32(const uint8_t *buf, size_t len)` with standard FNV-1a constants.",
            "Explain in two concise sentences why causal self-attention scales quadratically with sequence length.",
            "Write a Go function `ReverseRunes(s string) string` that handles multi-byte UTF-8 runes properly.",
            "Write a bash one-liner that counts unique IPv4 addresses in column 1 of `access.log` sorted descending.",
            "Write a TypeScript type `ReadonlyDeep<T>` that recursively marks object and array fields readonly.",
            "Write a CSS rule `.grid-cards` using `grid-template-columns: repeat(auto-fill, minmax(240px, 1fr))` and `gap: 16px`.",
        ];
        let params = GenParams {
            temperature: 0.0,
            top_k: 40,
            top_p: 1.0,
            repeat_penalty: 1.0,
            max_tokens: 32,
            seed: 0,
        };

        let mut cpu_logits = Vec::with_capacity(prompts.len());
        let mut cpu_replies = Vec::with_capacity(prompts.len());
        model.disable_metal();
        for p in &prompts {
            let msgs = [Message::new("user", p)];
            let toks = model.render(&msgs);
            model.reset_cache();
            let l = model.engine.forward_prompt(&toks).to_vec();
            cpu_logits.push(l);
            model.reset_cache();
            let out = model.generate(&msgs, &params, |_| true).unwrap();
            cpu_replies.push((out.token_ids, out.text));
        }

        model.enable_metal().unwrap();
        assert!(model.is_metal_enabled());
        let mut max_kl = 0.0f64;
        let mut exact_greedy_matches = 0usize;
        for (idx, p) in prompts.iter().enumerate() {
            let msgs = [Message::new("user", p)];
            let toks = model.render(&msgs);
            model.reset_cache();
            let ml = model.engine.forward_prompt(&toks).to_vec();
            let kl = kl_div(&cpu_logits[idx], &ml);
            max_kl = max_kl.max(kl);
            assert!(
                kl < 0.02,
                "prompt {idx} Metal vs CPU last-token KL = {kl:.6} exceeds 0.02"
            );

            model.reset_cache();
            let out = model.generate(&msgs, &params, |_| true).unwrap();
            assert_eq!(out.prefill_unit, "metal");
            let prefix_agree = out
                .token_ids
                .iter()
                .zip(&cpu_replies[idx].0)
                .take_while(|(a, b)| a == b)
                .count();
            if out.token_ids == cpu_replies[idx].0 {
                exact_greedy_matches += 1;
            }
            eprintln!(
                "metal parity prompt {idx}: prefill {} tok @ {:.0} tok/s | KL={kl:.6} | greedy prefix match {}/{}",
                out.prefilled_tokens,
                out.prefilled_tokens as f64 / out.prefill_secs.max(1e-9),
                prefix_agree,
                out.token_ids.len(),
            );
            assert!(
                prefix_agree >= 4,
                "prompt {idx} greedy prefix diverged too early ({prefix_agree} tokens): metal={:?} vs cpu={:?}",
                out.text,
                cpu_replies[idx].1
            );

            if draft_dir.join("kronos-nf4g128.k2tb").exists() && idx == 9 {
                model.attach_draft(draft_dir, "nf4g128", None).unwrap();
                model.reset_cache();
                let spec_out = model.generate(&msgs, &params, |_| true).unwrap();
                assert_eq!(
                    spec_out.token_ids, out.token_ids,
                    "Metal speculative greedy must match Metal target greedy token-for-token"
                );
                model.detach_draft();
            }
        }
        eprintln!(
            "10-prompt 14B Metal vs CPU summary: max_kl={max_kl:.6}, exact_32tok_greedy={exact_greedy_matches}/10"
        );
    }

    #[test]
    fn test_parse_qwen_tool_calls_mixed_plain_and_escaped_single_quotes() {
        let raw = "<tool_call>\n{\"name\":\"write_file\",\"arguments\":{\"path\":\"build_video.py\",\"content\":\"note = 'Viazovska\\'s theorem'\\nsubprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0\\', \\'-i\\', \\'concat.txt\\', \\'-c\\', \\'copy\\', \\'out.mp4\\'], check=True)\\n\"}}\n</tool_call>";
        let (_, calls) = parse_qwen_tool_calls(raw);
        assert_eq!(calls.len(), 1);
        let parsed = Json::parse(&calls[0].arguments).expect("valid tool arguments json");
        let content = parsed.get("content").and_then(Json::as_str).unwrap_or("");
        assert!(
            content.contains("note = 'Viazovska\\'s theorem'"),
            "apostrophe inside plain single-quoted string should stay escaped: {content}"
        );
        assert!(
            content.contains("'-safe', '0', '-i', 'concat.txt', '-c', 'copy', 'out.mp4']"),
            "mixed plain/over-escaped single-quote list delimiters should decode cleanly: {content}"
        );
    }
}
