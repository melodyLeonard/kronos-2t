//! Chat on a compiled Kronos model: ChatML template, sampling, token streaming and KV-cache
//! prefix reuse across requests.

use crate::bpe_tokenizer::BpeTokenizer;
use crate::real_llm::{argmax, RealLlamaEngine, RealWeightMode};
use std::path::Path;
use std::time::Instant;

pub const DEFAULT_SYSTEM: &str = "You are a helpful AI assistant named SmolLM, trained by Hugging Face";

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
    Ok(fp32.compile_sequential_bipolar_gptq(mode, &seqs, 8))
}

pub struct ChatModel {
    engine: RealLlamaEngine,
    tok: BpeTokenizer,
    im_start: usize,
    im_end: usize,
    newline: Vec<usize>,
    /// Token ids currently in the KV cache.
    cached: Vec<usize>,
    ctx: usize,
    pub name: String,
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
        let tok = BpeTokenizer::from_model_dir(model_dir).map_err(|e| format!("tokenizer: {e}"))?;
        let special = |s: &str| tok.token_to_id.get(s).copied().ok_or(format!("tokenizer lacks {s}; use an Instruct model"));
        let (im_start, im_end) = (special("<|im_start|>")?, special("<|im_end|>")?);
        let newline = tok.encode("\n");
        let name = format!(
            "{}-kronos-{mode_key}",
            model_dir.file_name().map_or("model".into(), |n| n.to_string_lossy().to_lowercase())
        );
        Ok(Self { engine, tok, im_start, im_end, newline, cached: Vec::new(), ctx, name })
    }

    pub fn context_size(&self) -> usize {
        self.ctx
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

        // Reuse the cached prefix; at least one prompt token must run to produce logits.
        let mut common = self.cached.iter().zip(&prompt).take_while(|(a, b)| a == b).count();
        common = common.min(prompt.len() - 1);
        self.engine.truncate_context(common);
        self.cached.truncate(common);
        let t0 = Instant::now();
        let mut logits = self.engine.forward_prompt(&prompt[common..]).to_vec();
        self.cached.extend_from_slice(&prompt[common..]);
        let prefill_secs = t0.elapsed().as_secs_f64();

        let mut rng = Rng(params.seed ^ 0x9E37_79B9_7F4A_7C15);
        let t1 = Instant::now();
        let mut reply: Vec<usize> = Vec::new();
        let mut shown = 0usize;
        let mut finish = FinishReason::Length;
        for _ in 0..max_new {
            let recent = &reply[reply.len().saturating_sub(64)..];
            let next = sample(&logits, recent, params, &mut rng);
            if next == self.im_end || next == self.tok.eos_token_id {
                finish = FinishReason::Stop;
                break;
            }
            reply.push(next);
            let text = self.tok.decode(&reply);
            // Hold back a trailing partial UTF-8 character until it completes.
            let ready = text.trim_end_matches('\u{FFFD}');
            if ready.len() > shown {
                let keep_going = on_text(&ready[shown..]);
                shown = ready.len();
                if !keep_going {
                    finish = FinishReason::Stop;
                    break;
                }
            }
            logits = self.engine.forward_token(next).to_vec();
            self.cached.push(next);
        }
        Ok(Completion {
            text: self.tok.decode(&reply),
            prompt_tokens: prompt.len(),
            prefilled_tokens: prompt.len() - common,
            completion_tokens: reply.len(),
            finish,
            prefill_secs,
            decode_secs: t1.elapsed().as_secs_f64(),
        })
    }
}

/// xorshift64* sampler state.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        ((self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn sample(logits: &[f32], recent: &[usize], p: &GenParams, rng: &mut Rng) -> usize {
    let mut l = logits.to_vec();
    for &t in recent {
        l[t] = if l[t] > 0.0 { l[t] / p.repeat_penalty } else { l[t] * p.repeat_penalty };
    }
    if p.temperature <= 0.0 {
        return argmax(&l);
    }
    let k = p.top_k.clamp(1, l.len());
    let mut idx: Vec<usize> = (0..l.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| l[b].total_cmp(&l[a]));
    idx.truncate(k);
    idx.sort_unstable_by(|&a, &b| l[b].total_cmp(&l[a]));
    let max = l[idx[0]];
    let mut probs: Vec<f32> = idx.iter().map(|&i| ((l[i] - max) / p.temperature).exp()).collect();
    let total: f32 = probs.iter().sum();
    let (mut cum, mut keep) = (0.0, probs.len());
    for (n, q) in probs.iter_mut().enumerate() {
        *q /= total;
        cum += *q;
        if cum >= p.top_p {
            keep = n + 1;
            break;
        }
    }
    let mut r = rng.next_f32() * probs[..keep].iter().sum::<f32>();
    for (n, &q) in probs[..keep].iter().enumerate() {
        r -= q;
        if r <= 0.0 {
            return idx[n];
        }
    }
    idx[keep - 1]
}
