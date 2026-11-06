//! Qwen3.5 (Gated DeltaNet hybrid) compile, quality and speed tool.
//!
//! Usage: `qwen35_bench <command> <model_dir> [flags]`
//!   `ref-ppl`  BF16 reference perplexity on CPU (`--text F --seq-len N --seqs K`)
//!   `compile`  Hadamard-GPTQ compile to `--out FILE` (`--bits 4 --head-bits 4 --calib F
//!              --calib-seqs N --seq-len L --rtn`)
//!   `ppl`      perplexity of `--cache FILE` on the GPU (`--cpu` also scores it on the CPU)
//!   `bench`    prefill and decode tok/s (`--cache FILE --prompt-len N --gen N --ctx N --text F`)
//!   `gen`      greedy chat completion (`--cache FILE --prompt TEXT --max-new N`)

use kronos_2t::backend::metal_qwen35::{MetalQwen35, MAX_CHUNK};
use kronos_2t::model::bpe_tokenizer::BpeTokenizer;
use kronos_2t::model::qwen35::{compile, reference_ppl, QuantSpec, Qwen35Model};
use kronos_2t::model::real_llm::argmax;
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Args {
    cmd: String,
    model_dir: PathBuf,
    flags: Vec<(String, String)>,
    switches: Vec<String>,
}

impl Args {
    fn parse() -> Self {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_else(|| usage());
        let model_dir = PathBuf::from(it.next().unwrap_or_else(|| usage()));
        let rest: Vec<String> = it.collect();
        let (mut flags, mut switches) = (Vec::new(), Vec::new());
        let mut i = 0;
        while i < rest.len() {
            let k = rest[i].trim_start_matches("--").to_string();
            if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
                flags.push((k, rest[i + 1].clone()));
                i += 2;
            } else {
                switches.push(k);
                i += 1;
            }
        }
        Self { cmd, model_dir, flags, switches }
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.flags.iter().find(|f| f.0 == k).map(|f| f.1.as_str())
    }

    fn num(&self, k: &str, default: usize) -> usize {
        self.get(k).map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("--{k} expects a number")))
    }

    fn has(&self, k: &str) -> bool {
        self.switches.iter().any(|s| s == k)
    }

    fn path(&self, k: &str, default: &str) -> PathBuf {
        self.get(k).map_or_else(|| self.model_dir.join(default), PathBuf::from)
    }
}

fn usage() -> ! {
    eprintln!("usage: qwen35_bench <ref-ppl|compile|ppl|bench|gen> <model_dir> [flags]");
    std::process::exit(2)
}

/// `count` sequences of `len` tokens from `text`, taken from the start.
fn token_seqs(tok: &BpeTokenizer, text_path: &Path, len: usize, count: usize) -> Vec<Vec<usize>> {
    let text = std::fs::read_to_string(text_path).unwrap_or_else(|e| panic!("{}: {e}", text_path.display()));
    let ids = tok.encode(&text);
    let seqs: Vec<Vec<usize>> = ids.chunks_exact(len).take(count).map(|c| c.to_vec()).collect();
    assert!(!seqs.is_empty(), "{} has fewer than {len} tokens", text_path.display());
    seqs
}

fn log_line(s: &str) {
    eprintln!("[{}] {s}", chrono_like_now());
}

fn chrono_like_now() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
    format!("{:02}:{:02}:{:02}", (secs / 3600) % 24, (secs / 60) % 60, secs % 60)
}

fn load(a: &Args) -> Qwen35Model {
    let cache = a.path("cache", "kronos-qwen35-nf4g128.k2tb");
    let t0 = Instant::now();
    let m = Qwen35Model::load(&a.model_dir, &cache).unwrap_or_else(|e| panic!("{}: {e}", cache.display()));
    log_line(&format!(
        "loaded {} in {:.1}s ({:.2} GiB of weights read per token)",
        cache.display(),
        t0.elapsed().as_secs_f64(),
        m.weight_bytes_per_token() as f64 / (1u64 << 30) as f64
    ));
    m
}

fn gpu_ppl(model: &Qwen35Model, eng: &mut MetalQwen35, seqs: &[Vec<usize>]) -> f64 {
    let (mut nll, mut n) = (0.0f64, 0usize);
    for s in seqs {
        eng.reset();
        let want: Vec<usize> = (0..s.len() - 1).collect();
        eng.forward(model, s, &want, &mut |i, logits| {
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + logits.iter().map(|&v| (v as f64 - mx).exp()).sum::<f64>().ln();
            nll += lse - logits[s[i + 1]] as f64;
            n += 1;
        })
        .expect("GPU forward");
    }
    (nll / n as f64).exp()
}

/// Prefills `prompt`; with `mtp`, also fills the MTP cache for every prompt pair. Returns the first
/// greedy token, which is not fed yet.
fn prefill(model: &Qwen35Model, eng: &mut MetalQwen35, prompt: &[usize], mtp: bool) -> usize {
    let p = prompt.len();
    let mut next = 0usize;
    let mut s = 0;
    while s < p {
        let e = (s + MAX_CHUNK).min(p);
        let want: Vec<usize> = if e == p { vec![e - s - 1] } else { vec![] };
        eng.forward(model, &prompt[s..e], &want, &mut |_, l| next = argmax(l)).expect("prefill");
        if mtp {
            let pairs = &prompt[s + 1..e.min(p)];
            if !pairs.is_empty() {
                eng.mtp_fill(model, pairs, 0, s, false).expect("MTP fill");
            }
        }
        s = e;
    }
    next
}

fn decode_plain(model: &Qwen35Model, eng: &mut MetalQwen35, mut next: usize, gen: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(gen);
    for _ in 0..gen {
        out.push(next);
        eng.forward(model, &[next], &[0], &mut |_, l| next = argmax(l)).expect("decode");
    }
    out
}

/// Greedy decoding with MTP drafts of up to `k` tokens verified in one batched pass. Returns the
/// tokens and the number of verify steps.
fn decode_mtp(model: &Qwen35Model, eng: &mut MetalQwen35, first: usize, prompt_len: usize, gen: usize, k: usize) -> (Vec<usize>, usize) {
    let prof = std::env::var_os("QWEN35_SPEC_PROF").is_some();
    let (mut t_draft, mut t_verify, mut t_fill) = (0.0f64, 0.0f64, 0.0f64);
    let last_row = (prompt_len - 1) % MAX_CHUNK;
    let mut x = first;
    let mut pos = prompt_len;
    let mut d = eng.mtp_fill(model, &[x], last_row, pos - 1, true).expect("MTP").expect("draft");
    let mut out = Vec::with_capacity(gen + k + 1);
    let mut steps = 0;
    while out.len() < gen {
        let tq = Instant::now();
        let mut drafts = vec![d];
        for j in 1..k {
            drafts.push(eng.mtp_chain(model, drafts[j - 1], pos - 1 + j).expect("MTP chain"));
        }
        t_draft += tq.elapsed().as_secs_f64();
        let tq = Instant::now();
        let mut feed = vec![x];
        feed.extend(&drafts);
        let mut am = vec![0usize; feed.len()];
        eng.verify(model, &feed, &mut |i, l| am[i] = argmax(l)).expect("verify");
        t_verify += tq.elapsed().as_secs_f64();
        let mut m = 0;
        while m < k && am[m] == drafts[m] {
            m += 1;
        }
        eng.accept(m + 1);
        out.push(x);
        out.extend(&drafts[..m]);
        let y = am[m];
        let mut next_toks = drafts[..m].to_vec();
        next_toks.push(y);
        let tq = Instant::now();
        d = eng.mtp_fill(model, &next_toks, 0, pos, true).expect("MTP").expect("draft");
        t_fill += tq.elapsed().as_secs_f64();
        pos += m + 1;
        x = y;
        steps += 1;
    }
    if prof {
        let s = steps.max(1) as f64 * 1e-3;
        eprintln!("per step: draft {:.1}ms verify {:.1}ms fill {:.1}ms", t_draft / s, t_verify / s, t_fill / s);
    }
    out.truncate(gen);
    (out, steps)
}

fn main() {
    let a = Args::parse();
    let tok = BpeTokenizer::from_model_dir(&a.model_dir).expect("tokenizer");
    let text = a.path("text", "../../README.md");
    match a.cmd.as_str() {
        "ref-ppl" => {
            let seqs = token_seqs(&tok, &text, a.num("seq-len", 512), a.num("seqs", 2));
            let t0 = Instant::now();
            let ppl = reference_ppl(&a.model_dir, &seqs, &mut |s| log_line(s)).expect("reference");
            println!("reference BF16 ppl {ppl:.4} over {} x {} tokens ({:.0}s)", seqs.len(), seqs[0].len(), t0.elapsed().as_secs_f64());
        }
        "compile" => {
            let bits = a.num("bits", 4);
            let spec = QuantSpec { bits, head_bits: a.num("head-bits", bits), scale_group_chunks: 2, gptq: !a.has("rtn") };
            let calib = a.path("calib", "../calib.txt");
            let seqs = token_seqs(&tok, &calib, a.num("seq-len", 512), a.num("calib-seqs", 8));
            let out = a.path("out", "kronos-qwen35-nf4g128.k2tb");
            let threads = std::thread::available_parallelism().map_or(8, |p| p.get());
            let t0 = Instant::now();
            log_line(&format!("compiling {spec:?} on {} x {} calibration tokens", seqs.len(), seqs[0].len()));
            let model = compile(&a.model_dir, spec, &seqs, threads, &mut |s| log_line(s)).expect("compile");
            model.save(&out).expect("save");
            log_line(&format!("wrote {} in {:.1} min", out.display(), t0.elapsed().as_secs_f64() / 60.0));
        }
        "ppl" => {
            let model = load(&a);
            let seqs = token_seqs(&tok, &text, a.num("seq-len", 512), a.num("seqs", 2));
            let mut eng = MetalQwen35::new(&model, a.num("ctx", 4096)).expect("Metal");
            let t0 = Instant::now();
            let ppl = gpu_ppl(&model, &mut eng, &seqs);
            println!("GPU ppl {ppl:.4} over {} x {} tokens ({:.1}s)", seqs.len(), seqs[0].len(), t0.elapsed().as_secs_f64());
            if a.has("cpu") {
                let t0 = Instant::now();
                println!("CPU ppl {:.4} ({:.1}s)", model.ppl_cpu(&seqs), t0.elapsed().as_secs_f64());
            }
        }
        "bench" => {
            let model = if a.has("synthetic") {
                let mut cfg = kronos_2t::model::qwen35::Qwen35Config::from_model_dir(&a.model_dir).expect("config");
                if let Some(n) = a.get("layers").and_then(|v| v.parse::<usize>().ok()) {
                    cfg.num_layers = n;
                    cfg.full_attn.truncate(n);
                }
                let t0 = Instant::now();
                let m = Qwen35Model::synthetic(cfg, 1, true);
                log_line(&format!("built random-weight model with the real shapes in {:.0}s", t0.elapsed().as_secs_f64()));
                m
            } else {
                load(&a)
            };
            let ctx = a.num("ctx", 8192);
            let mut eng = MetalQwen35::new(&model, ctx + 256).expect("Metal");
            let ids = tok.encode(&std::fs::read_to_string(&text).expect("text"));
            let gen = a.num("gen", 128);
            let draft = a.num("draft", 3);
            for &plen in &[a.num("prompt-len", 512), ctx] {
                let prompt: Vec<usize> = ids.iter().cycle().take(plen).copied().collect();
                // Warm-up so pipeline creation and first-touch faults are not timed.
                eng.reset();
                eng.forward(&model, &prompt[..8], &[7], &mut |_, _| {}).expect("warm-up");

                eng.reset();
                let t0 = Instant::now();
                let first = prefill(&model, &mut eng, &prompt, false);
                let pf = t0.elapsed().as_secs_f64();
                let t1 = Instant::now();
                let plain = decode_plain(&model, &mut eng, first, gen);
                let dc = t1.elapsed().as_secs_f64();
                print!(
                    "prompt {plen:>5}: prefill {:7.1} tok/s ({pf:.2}s) | decode {:6.2} tok/s",
                    plen as f64 / pf,
                    gen as f64 / dc
                );
                if eng.has_mtp() && draft > 0 {
                    eng.set_draft_vocab(a.num("draft-vocab", 32768));
                    eng.reset();
                    let first = prefill(&model, &mut eng, &prompt, true);
                    let t2 = Instant::now();
                    let (spec, steps) = decode_mtp(&model, &mut eng, first, plen, gen, draft);
                    let sc = t2.elapsed().as_secs_f64();
                    let same = spec.iter().zip(&plain).take_while(|(x, y)| x == y).count();
                    print!(
                        " | MTP x{draft} {:6.2} tok/s, {:.2} tokens/step, matches greedy for {same}/{gen}",
                        gen as f64 / sc,
                        gen as f64 / steps as f64
                    );
                }
                println!();
                if a.has("show") {
                    println!("  continuation: {:?}", tok.decode(&plain));
                }
            }
        }
        "gen" => {
            let model = load(&a);
            let user = a.get("prompt").unwrap_or("Write a Rust function that reverses a string.");
            let prompt = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
            let ids = tok.encode(&prompt);
            let mut eng = MetalQwen35::new(&model, 8192).expect("Metal");
            let mut next = 0usize;
            let t0 = Instant::now();
            eng.forward(&model, &ids, &[ids.len() - 1], &mut |_, l| next = argmax(l)).expect("prefill");
            let pf = t0.elapsed().as_secs_f64();
            let im_end = tok.encode("<|im_end|>").first().copied().unwrap_or(usize::MAX);
            let t1 = Instant::now();
            let mut out = Vec::new();
            for _ in 0..a.num("max-new", 256) {
                if next == im_end || next == model.cfg.eos_token_id {
                    break;
                }
                out.push(next);
                eng.forward(&model, &[next], &[0], &mut |_, l| next = argmax(l)).expect("decode");
            }
            println!("{}", tok.decode(&out));
            eprintln!(
                "prefill {} tokens in {pf:.2}s, decode {:.2} tok/s",
                ids.len(),
                out.len() as f64 / t1.elapsed().as_secs_f64()
            );
        }
        _ => usage(),
    }
}
