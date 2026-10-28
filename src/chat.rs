//! Chat on a compiled Kronos model: ChatML template, zero-allocation sampling, `O(1)` incremental
//! UTF-8 token streaming, KV-cache prefix reuse, and Dual-Tier Lossless Speculative Decoding
//! (Zero-RAM Prompt/History Suffix Lookup + Draft-Model Leviathan-Chen Speculative Verification).

use crate::bpe_tokenizer::BpeTokenizer;
use crate::real_llm::{argmax, RealLlamaEngine, RealWeightMode};
use std::path::Path;
use std::time::Instant;

/// Minimum new prompt tokens before Neural Engine prefill beats extending the CPU cache.
#[cfg(all(feature = "ane", target_os = "macos"))]
const ANE_MIN_NEW_TOKENS: usize = 32;

pub const DEFAULT_SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

/// Maximum draft lookahead tokens verified in a single batched target DRAM pass (`1 + 7 = 8` tokens = exact `NT = 8` NEON tile!).
const SPEC_DRAFT_K_GREEDY: usize = 7;
const SPEC_DRAFT_K_SAMPLE: usize = 7;

#[derive(Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self { role: role.to_string(), content: content.to_string() }
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
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            FinishReason::Stop => "stop",
            FinishReason::Length => "length",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Completion {
    pub text: String,
    pub prompt_tokens: usize,
    /// Prompt tokens actually run (the rest came from the KV cache).
    pub prefilled_tokens: usize,
    pub completion_tokens: usize,
    pub finish: FinishReason,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    /// `"ane"` when the Neural Engine ran the prefill, else `"cpu"`.
    pub prefill_unit: &'static str,
}

/// `b4`, `b5`, `b6` or `pQOGDH` (plane counts 4–6 for qkv, o, gate_up, down, lm_head).
pub fn parse_mode(m: &str) -> Result<RealWeightMode, String> {
    Ok(match m {
        "b4" => RealWeightMode::KronosBipolarGptq { num_planes: 4 },
        "b5" => RealWeightMode::KronosBipolarGptq { num_planes: 5 },
        "b6" => RealWeightMode::KronosBipolarGptq { num_planes: 6 },
        p if p.len() == 6 && p.starts_with('p') && p[1..].bytes().all(|b| (b'4'..=b'6').contains(&b)) => {
            let d: Vec<usize> = p[1..].bytes().map(|b| (b - b'0') as usize).collect();
            RealWeightMode::KronosBipolarGptqPlanes { qkv: d[0], o: d[1], gate_up: d[2], down: d[3], head: d[4] }
        }
        other => return Err(format!("unknown mode {other}; use b4, b5, b6 or pQOGDH (digits 4-6)")),
    })
}

/// Compiles `model_dir` with GPTQ on `calib_text` (one-time, minutes).
pub fn compile(model_dir: &Path, mode: RealWeightMode, calib_text: &str, calib_tokens: usize) -> Result<RealLlamaEngine, String> {
    const SEQ: usize = 512;
    let tok = BpeTokenizer::from_model_dir(model_dir).map_err(|e| format!("tokenizer: {e}"))?;
    let ids = tok.encode(calib_text);
    if ids.len() < SEQ {
        return Err("calibration text is too short (need at least 512 tokens)".to_string());
    }
    let seqs: Vec<Vec<usize>> = ids[..calib_tokens.min(ids.len())].chunks(SEQ).map(|c| c.to_vec()).collect();
    let fp32 = RealLlamaEngine::load_fp32_reference(model_dir, SEQ).map_err(|e| format!("load model: {e}"))?;
    Ok(fp32.compile_sequential_bipolar_gptq(mode, &seqs, crate::thread_pool::physical_p_cores()))
}

fn load_or_compile_engine(
    model_dir: &Path,
    mode_key: &str,
    ctx: usize,
    threads: usize,
    calib: Option<(&Path, usize)>,
) -> Result<RealLlamaEngine, String> {
    let mode = parse_mode(mode_key)?;
    let cache = model_dir.join(format!("kronos-{mode_key}.k2tb"));
    let mut engine = match RealLlamaEngine::load_bipolar(model_dir, &cache, ctx) {
        Ok(e) => e,
        Err(_) => {
            let (path, n) = calib.ok_or_else(|| {
                format!("no compiled model at {}; provide calibration text to compile it", cache.display())
            })?;
            let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            eprintln!("compiling {} ({mode_key}), one time only; this takes minutes...", model_dir.display());
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
            verify_logits: vec![0.0f32; (SPEC_DRAFT_K_GREEDY + 1) * vocab_size],
            top_heap: Vec::with_capacity(128),
            target_dist: Vec::with_capacity(128),
            draft_dists: (0..SPEC_DRAFT_K_GREEDY).map(|_| Vec::with_capacity(128)).collect(),
        }
    }
}

pub struct ChatModel {
    engine: RealLlamaEngine,
    draft: Option<DraftEngine>,
    tok: BpeTokenizer,
    im_start: usize,
    im_end: usize,
    newline: Vec<usize>,
    /// Token ids currently in the target KV cache.
    cached: Vec<usize>,
    ctx: usize,
    scratch: ChatScratch,
    pub name: String,
    #[cfg(all(feature = "ane", target_os = "macos"))]
    ane: Vec<(usize, crate::coreml::CoreMlModel)>,
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
        let engine = load_or_compile_engine(model_dir, mode_key, ctx, threads, calib)?;
        let vocab_size = engine.cfg.vocab_size;
        let tok = BpeTokenizer::from_model_dir(model_dir).map_err(|e| format!("tokenizer: {e}"))?;
        let special =
            |s: &str| tok.token_to_id.get(s).copied().ok_or(format!("tokenizer lacks {s}; use an Instruct model"));
        let (im_start, im_end) = (special("<|im_start|>")?, special("<|im_end|>")?);
        let newline = tok.encode("\n");
        let name = format!(
            "{}-kronos-{mode_key}",
            model_dir.file_name().map_or("model".into(), |n| n.to_string_lossy().to_lowercase())
        );
        Ok(Self {
            engine,
            draft: None,
            tok,
            im_start,
            im_end,
            newline,
            cached: Vec::new(),
            ctx,
            scratch: ChatScratch::new(vocab_size),
            name,
            #[cfg(all(feature = "ane", target_os = "macos"))]
            ane: Vec::new(),
        })
    }

    /// Attaches a fast Kronos draft model (e.g. `SmolLM2-135M-Instruct`) for **100% lossless**
    /// speculative decoding (exact greedy match at `temp == 0` and exact Leviathan-Chen rejection
    /// sampling at `temp > 0`), sharing the global P-core thread pool with zero thread contention!
    pub fn attach_draft(
        &mut self,
        draft_dir: &Path,
        draft_mode_key: &str,
        calib: Option<(&Path, usize)>,
    ) -> Result<(), String> {
        let threads = self.engine.num_threads();
        let draft_eng = load_or_compile_engine(draft_dir, draft_mode_key, self.ctx, threads, calib)?;
        if draft_eng.cfg.vocab_size != self.engine.cfg.vocab_size {
            return Err(format!(
                "draft vocab_size ({}) != target vocab_size ({})",
                draft_eng.cfg.vocab_size, self.engine.cfg.vocab_size
            ));
        }
        self.draft = Some(DraftEngine { engine: draft_eng, cached: Vec::new() });
        Ok(())
    }

    pub fn context_size(&self) -> usize {
        self.ctx
    }

    /// Loads Neural Engine prefill models `prefill_s<N>.mlmodelc` from `dir` (see
    /// `ane/export_prefill.py`); returns the prompt-length buckets found.
    #[cfg(all(feature = "ane", target_os = "macos"))]
    pub fn enable_ane(&mut self, dir: &Path) -> Result<Vec<usize>, String> {
        use crate::coreml::{ComputeUnits, CoreMlModel};
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
                let h = self.engine.cfg.hidden_size;
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
        let n = prompt.len();
        let (seq, model) = self.ane.iter_mut().find(|(s, _)| *s >= n)?;
        let seq = *seq;
        let h = self.engine.cfg.hidden_size;
        let mut x = vec![0.0f32; seq * h];
        for (t, &tok) in prompt.iter().enumerate() {
            self.engine.embed(tok, &mut x[t * h..(t + 1) * h]);
        }
        Some(model.predict("x", &[1, seq, h], &x, &["hidden", "keys", "values"]).map(|out| {
            self.engine.load_prefix_kv(&out[1], &out[2], seq, n);
            self.engine.logits_from_hidden(&out[0][(n - 1) * h..n * h]).to_vec()
        }))
    }

    /// ChatML prompt ending with an open assistant turn; adds the default system prompt if none.
    pub fn render(&self, messages: &[Message]) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut turn = |role: &str, content: &str| {
            ids.push(self.im_start);
            ids.extend(self.tok.encode(&format!("{role}\n{content}")));
            ids.push(self.im_end);
            ids.extend(&self.newline);
        };
        if messages.first().is_none_or(|m| m.role != "system") {
            turn("system", DEFAULT_SYSTEM);
        }
        for m in messages {
            turn(&m.role, &m.content);
        }
        ids.push(self.im_start);
        ids.extend(self.tok.encode("assistant\n"));
        ids
    }

    /// Generates a reply, calling `on_text` with each new piece of text; return `false` from it
    /// to stop early. Oldest non-system messages are dropped if the conversation outgrows the
    /// context window.
    pub fn generate(
        &mut self,
        messages: &[Message],
        params: &GenParams,
        mut on_text: impl FnMut(&str) -> bool,
    ) -> Result<Completion, String> {
        let mut msgs = messages.to_vec();
        let reserve = params.max_tokens.min(self.ctx / 2).max(1);
        let mut prompt = self.render(&msgs);
        while prompt.len() + reserve > self.ctx {
            let first = usize::from(msgs.first().is_some_and(|m| m.role == "system"));
            if msgs.len() <= first + 1 {
                return Err(format!("prompt of {} tokens exceeds the {}-token context", prompt.len(), self.ctx));
            }
            msgs.remove(first);
            prompt = self.render(&msgs);
        }
        let max_new = params.max_tokens.min(self.ctx - prompt.len());
        let vocab = self.engine.cfg.vocab_size;

        // Reuse the cached prefix; at least one prompt token must run to produce logits.
        let mut common = self.cached.iter().zip(&prompt).take_while(|(a, b)| a == b).count();
        common = common.min(prompt.len() - 1);
        self.engine.truncate_context(common);
        self.cached.truncate(common);
        let t0 = Instant::now();
        let mut prefill_unit = "cpu";
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
        match ane_logits {
            Some(l) => {
                prefill_unit = "ane";
                common = 0;
                self.cached.clear();
                self.scratch.cur_logits.copy_from_slice(&l);
            }
            None => {
                let l = self.engine.forward_prompt(&prompt[common..]);
                self.scratch.cur_logits.copy_from_slice(l);
            }
        }
        self.cached.extend_from_slice(&prompt[common..]);

        // Pre-sync draft KV cache on the prompt inside prefill timing so decode starts hot!
        if let Some(d) = self.draft.as_mut() {
            let d_common = d.cached.iter().zip(&self.cached).take_while(|(a, b)| a == b).count();
            d.engine.truncate_context(d_common);
            d.cached.truncate(d_common);
            if d_common < self.cached.len() {
                d.engine.forward_prompt(&self.cached[d_common..]);
                d.cached.extend_from_slice(&self.cached[d_common..]);
            }
        }
        let prefill_secs = t0.elapsed().as_secs_f64();

        let mut rng = Rng(params.seed ^ 0x9E37_79B9_7F4A_7C15);
        let t1 = Instant::now();
        let mut reply: Vec<usize> = Vec::with_capacity(max_new);
        let mut raw_bytes: Vec<u8> = Vec::with_capacity(max_new * 4);
        let mut emitted_bytes = 0usize;
        let mut finish = FinishReason::Length;

        if max_new == 0 {
            return Ok(Completion {
                text: String::new(),
                prompt_tokens: prompt.len(),
                prefilled_tokens: prompt.len() - common,
                completion_tokens: 0,
                finish,
                prefill_secs,
                prefill_unit,
                decode_secs: 0.0,
            });
        }

        let mut next = sample_token(
            &self.scratch.cur_logits,
            &[],
            params,
            &mut rng,
            &mut self.scratch.pen_logits,
            &mut self.scratch.top_heap,
        );

        let is_stop = |tok: usize, im_end: usize, eos: usize| tok == im_end || tok == eos;
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

        let mut cur_draft_k = 3usize; // `1 + 3 = 4` (`NT = 4` single NEON tile); scales to `7` (`NT = 8` tile) on streaks!
        let mut zero_accept_streak = 0usize;
        let mut draft_cooldown = 0usize;
        'outer: loop {
            if is_stop(next, self.im_end, self.tok.eos_token_id) {
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
            let max_k = if params.temperature <= 0.0 { SPEC_DRAFT_K_GREEDY } else { SPEC_DRAFT_K_SAMPLE };
            let max_draft = rem.min(avail_ctx).min(cur_draft_k.min(max_k));

            // TIER 1: Draft-Model Lossless Speculative Decoding (`--draft`)
            if draft_cooldown > 0 {
                draft_cooldown -= 1;
            } else if max_draft >= 2 && self.engine.is_all_sdot() && self.draft.is_some() {
                let mut draft_tokens = [0usize; SPEC_DRAFT_K_GREEDY];
                let mut drafted = 0usize;
                {
                    let d = self.draft.as_mut().unwrap();
                    let d_common = d.cached.iter().zip(&self.cached).take_while(|(a, b)| a == b).count();
                    d.engine.truncate_context(d_common);
                    d.cached.truncate(d_common);
                    if d_common < self.cached.len() {
                        let start = d_common;
                        let mut sync_buf = Vec::with_capacity(self.cached.len() - start + 1);
                        sync_buf.extend_from_slice(&self.cached[start..]);
                        sync_buf.push(next);
                        let dl = d.engine.forward_prompt(&sync_buf);
                        self.scratch.cur_logits.copy_from_slice(dl);
                        d.cached.extend_from_slice(&sync_buf);
                    } else {
                        let dl = d.engine.forward_token(next);
                        self.scratch.cur_logits.copy_from_slice(dl);
                        d.cached.push(next);
                    }

                    let mut sim_recent: Vec<usize> =
                        reply[reply.len().saturating_sub(64 + SPEC_DRAFT_K_GREEDY)..].to_vec();
                    for (j, dt_slot) in draft_tokens.iter_mut().enumerate().take(max_draft) {
                        let rec = &sim_recent[sim_recent.len().saturating_sub(64)..];
                        let dj = if params.temperature <= 0.0 {
                            sample_greedy(
                                &self.scratch.cur_logits,
                                rec,
                                params.repeat_penalty,
                                &mut self.scratch.pen_logits,
                            )
                        } else {
                            build_sampling_dist(
                                &self.scratch.cur_logits,
                                rec,
                                params,
                                &mut self.scratch.pen_logits,
                                &mut self.scratch.top_heap,
                                &mut self.scratch.draft_dists[j],
                            );
                            sample_from_dist(&self.scratch.draft_dists[j], &mut rng)
                        };
                        *dt_slot = dj;
                        drafted += 1;
                        if is_stop(dj, self.im_end, self.tok.eos_token_id) {
                            break;
                        }
                        sim_recent.push(dj);
                        if j + 1 < max_draft {
                            let dl = d.engine.forward_token(dj);
                            self.scratch.cur_logits.copy_from_slice(dl);
                            d.cached.push(dj);
                        }
                    }
                }

                // Single batched DRAM pass on target engine verifying `[next, d_0, ..., d_{drafted-1}]`!
                let mut batch = [0usize; SPEC_DRAFT_K_GREEDY + 1];
                batch[0] = next;
                batch[1..1 + drafted].copy_from_slice(&draft_tokens[..drafted]);
                let batch_len = 1 + drafted;
                let pos0 = self.engine.position();
                if self.scratch.verify_logits.len() < batch_len * vocab {
                    self.scratch.verify_logits.resize(batch_len * vocab, 0.0);
                }
                self.engine
                    .forward_batch_logits(&batch[..batch_len], &mut self.scratch.verify_logits);

                let mut accepted_all = true;
                let mut accepted_in_round = 0usize;
                for j in 0..drafted {
                    self.cached.push(batch[j]);
                    let t_logits = &self.scratch.verify_logits[j * vocab..(j + 1) * vocab];
                    let rec = &reply[reply.len().saturating_sub(64)..];
                    let dj = draft_tokens[j];

                    if params.temperature <= 0.0 {
                        let tj = sample_greedy(t_logits, rec, params.repeat_penalty, &mut self.scratch.pen_logits);
                        if tj != dj {
                            self.engine.truncate_context(pos0 + j + 1);
                            if let Some(d) = self.draft.as_mut() {
                                d.engine.truncate_context(pos0 + j + 1);
                                d.cached.truncate(pos0 + j + 1);
                            }
                            next = tj;
                            accepted_all = false;
                            break;
                        }
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
                        if p_d < q_d && rng.next_f32() >= (p_d / q_d) {
                            let tj = sample_residual_dist(
                                &self.scratch.target_dist,
                                &self.scratch.draft_dists[j],
                                &mut rng,
                            );
                            self.engine.truncate_context(pos0 + j + 1);
                            if let Some(d) = self.draft.as_mut() {
                                d.engine.truncate_context(pos0 + j + 1);
                                d.cached.truncate(pos0 + j + 1);
                            }
                            next = tj;
                            accepted_all = false;
                            break;
                        }
                    }

                    // `dj` accepted!
                    accepted_in_round += 1;
                    if is_stop(dj, self.im_end, self.tok.eos_token_id) {
                        self.engine.truncate_context(pos0 + j + 1);
                        if let Some(d) = self.draft.as_mut() {
                            d.engine.truncate_context(pos0 + j + 1);
                            d.cached.truncate(pos0 + j + 1);
                        }
                        finish = FinishReason::Stop;
                        break 'outer;
                    }
                    if !emit_tok(dj, &self.tok, &mut reply, &mut raw_bytes, &mut emitted_bytes, &mut on_text) {
                        self.engine.truncate_context(pos0 + j + 1);
                        if let Some(d) = self.draft.as_mut() {
                            d.engine.truncate_context(pos0 + j + 1);
                            d.cached.truncate(pos0 + j + 1);
                        }
                        finish = FinishReason::Stop;
                        break 'outer;
                    }
                    if reply.len() >= max_new {
                        self.engine.truncate_context(pos0 + j + 1);
                        if let Some(d) = self.draft.as_mut() {
                            d.engine.truncate_context(pos0 + j + 1);
                            d.cached.truncate(pos0 + j + 1);
                        }
                        finish = FinishReason::Length;
                        break 'outer;
                    }
                }

                if accepted_all {
                    // All `drafted` tokens accepted -> bonus token `drafted + 1` is FREE; step up to `NT = 8` (`1 + 7`)!
                    cur_draft_k = SPEC_DRAFT_K_GREEDY;
                    zero_accept_streak = 0;
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
                } else {
                    cur_draft_k = 3;
                    if accepted_in_round == 0 {
                        zero_accept_streak += 1;
                        if zero_accept_streak >= 2 {
                            draft_cooldown = 8;
                            zero_accept_streak = 0;
                        }
                    } else {
                        zero_accept_streak = 0;
                    }
                }
                continue;
            }

            // TIER 2: Zero-RAM Prompt/History Suffix-Lookup Speculation (greedy, `0` extra RAM)
            if params.temperature <= 0.0 && max_draft >= 3 && self.engine.is_all_sdot() {
                let mut draft_tokens = [0usize; SPEC_DRAFT_K_GREEDY];
                let drafted = find_suffix_continuation(&self.cached, next, max_draft, &mut draft_tokens);
                if drafted >= 2 {
                    let mut batch = [0usize; SPEC_DRAFT_K_GREEDY + 1];
                    batch[0] = next;
                    batch[1..1 + drafted].copy_from_slice(&draft_tokens[..drafted]);
                    let batch_len = 1 + drafted;
                    let pos0 = self.engine.position();
                    if self.scratch.verify_logits.len() < batch_len * vocab {
                        self.scratch.verify_logits.resize(batch_len * vocab, 0.0);
                    }
                    self.engine
                        .forward_batch_logits(&batch[..batch_len], &mut self.scratch.verify_logits);

                    let mut accepted_all = true;
                    for j in 0..drafted {
                        self.cached.push(batch[j]);
                        let t_logits = &self.scratch.verify_logits[j * vocab..(j + 1) * vocab];
                        let rec = &reply[reply.len().saturating_sub(64)..];
                        let tj = sample_greedy(t_logits, rec, params.repeat_penalty, &mut self.scratch.pen_logits);
                        let dj = draft_tokens[j];
                        if tj != dj {
                            self.engine.truncate_context(pos0 + j + 1);
                            next = tj;
                            accepted_all = false;
                            break;
                        }
                        if is_stop(dj, self.im_end, self.tok.eos_token_id) {
                            self.engine.truncate_context(pos0 + j + 1);
                            finish = FinishReason::Stop;
                            break 'outer;
                        }
                        if !emit_tok(dj, &self.tok, &mut reply, &mut raw_bytes, &mut emitted_bytes, &mut on_text) {
                            self.engine.truncate_context(pos0 + j + 1);
                            finish = FinishReason::Stop;
                            break 'outer;
                        }
                        if reply.len() >= max_new {
                            self.engine.truncate_context(pos0 + j + 1);
                            finish = FinishReason::Length;
                            break 'outer;
                        }
                    }
                    if accepted_all {
                        self.cached.push(batch[drafted]);
                        let bonus_logits = &self.scratch.verify_logits[drafted * vocab..(drafted + 1) * vocab];
                        let rec = &reply[reply.len().saturating_sub(64)..];
                        next = sample_greedy(
                            bonus_logits,
                            rec,
                            params.repeat_penalty,
                            &mut self.scratch.pen_logits,
                        );
                    }
                    continue;
                }
            }

            // Direct single-token Megakernel step (`0` allocations!)
            let recent = &reply[reply.len().saturating_sub(64)..];
            let l = self.engine.forward_token(next);
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

        Ok(Completion {
            text: String::from_utf8_lossy(&raw_bytes).into_owned(),
            prompt_tokens: prompt.len(),
            prefilled_tokens: prompt.len() - common,
            completion_tokens: reply.len(),
            finish,
            prefill_secs,
            decode_secs: t1.elapsed().as_secs_f64(),
            prefill_unit,
        })
    }
}

/// Scans `cached` + `next` backwards for a suffix match of length `4`, `3`, or `2` and copies up
/// to `max_k` continuation tokens into `out` (`< 0.5 µs`, `0` heap allocations).
fn find_suffix_continuation(cached: &[usize], next: usize, max_k: usize, out: &mut [usize]) -> usize {
    let n = cached.len();
    if n < 4 {
        return 0;
    }
    let limit = max_k.min(out.len());
    for ngram in [4usize, 3, 2] {
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

/// Fast zero-allocation greedy argmax with repetition penalty (`2.5 µs` zero-copy fast path!).
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
    for &t in recent {
        l[t] = if l[t] > 0.0 { l[t] / repeat_penalty } else { l[t] * repeat_penalty };
    }
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
/// using a `k`-element binary min-heap (`0` `49,152`-index allocations!).
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
        for &t in recent {
            l[t] = if l[t] > 0.0 { l[t] / p.repeat_penalty } else { l[t] * p.repeat_penalty };
        }
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
        for &t in &recent {
            ref_l[t] = if ref_l[t] > 0.0 { ref_l[t] / p.repeat_penalty } else { ref_l[t] * p.repeat_penalty };
        }
        assert_eq!(g, argmax(&ref_l));
    }

    #[test]
    fn test_suffix_continuation_lookup() {
        let history = vec![10, 20, 30, 40, 50, 60, 70, 10, 20, 30];
        let mut out = [0usize; 8];
        let n = find_suffix_continuation(&history, 40, 4, &mut out);
        assert_eq!(n, 4);
        assert_eq!(&out[..4], &[50, 60, 70, 10]);
    }
}
