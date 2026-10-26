//! WikiText-2 perplexity / KL / top-1 evaluation, matched 1:1 to `llama-perplexity`, plus an
//! optional `llama-bench`-style decode-speed test.
//!
//! Usage: `wikitext_ppl <model_dir> <token_ids_file> [flags]`
//!   `--chunks N`        evaluation chunks (default 64)
//!   `--ctx N`           chunk length (default 256)
//!   `--modes LIST`      comma list of `q40,b4,mk,b5` (default all)
//!   `--calib-ids FILE`  calibrate GPTQ on this token-id file (e.g. WikiText-2 *train*) instead of
//!                       the 8 built-in passages
//!   `--calib-tokens N`  tokens taken from `--calib-ids` (default 8192), split into `--ctx` sequences
//!   `--cache-dir DIR`   reuse/save compiled Bipolar engines (keyed by mode + calibration)
//!   `--speed`           also measure decode tok/s (`tg32`, 1 and 8 threads, mean of 5 runs)
//!
//! `token_ids_file` holds one token id per line, produced by
//! `llama-tokenize -m <smollm2.gguf> -f wiki.test.raw --ids --no-bos` so both engines score the
//! exact same token stream. Each chunk of `n_ctx` tokens is evaluated from an empty KV cache and,
//! like `llama-perplexity`, only the second half (`n_ctx/2 .. n_ctx-1`) is scored. KL divergence
//! and top-1 agreement are measured against the Exact FP32 reference at the same positions.

use kronos_2t::{
    bpe_tokenizer::BpeTokenizer,
    real_llm::{argmax, RealLlamaEngine, RealWeightMode},
};
use std::path::Path;
use std::time::Instant;

fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max_v + logits.iter().map(|&v| ((v as f64) - max_v).exp()).sum::<f64>().ln();
    logits.iter().map(|&v| (v as f64) - lse).collect()
}

fn argmax_f64(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}

fn read_ids(path: &str) -> Vec<usize> {
    std::fs::read_to_string(path)
        .expect("failed to read token ids")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().expect("bad token id"))
        .collect()
}

#[derive(Default)]
struct Acc {
    nll: f64,
    nll_sq: f64,
    kl: f64,
    top1: usize,
    n: usize,
}

impl Acc {
    fn add(&mut self, nll: f64, kl: f64, top1: bool) {
        self.nll += nll;
        self.nll_sq += nll * nll;
        self.kl += kl;
        self.top1 += top1 as usize;
        self.n += 1;
    }

    /// `(ppl, ppl_stderr, mean_kl, top1_pct)`; stderr via delta method like `llama-perplexity`.
    fn report(&self) -> (f64, f64, f64, f64) {
        let n = self.n as f64;
        let mean = self.nll / n;
        let var = (self.nll_sq / n - mean * mean).max(0.0);
        let ppl = mean.exp();
        (ppl, ppl * (var / (n - 1.0)).sqrt(), self.kl / n, 100.0 * self.top1 as f64 / n)
    }
}

/// `llama-bench tg32`-style decode speed: `n_gen` single-token forwards from an empty KV cache,
/// `runs` timed repetitions after one warm-up, returning `(mean, stddev)` tok/s.
fn decode_speed(eng: &mut RealLlamaEngine, ids: &[usize], n_gen: usize, runs: usize) -> (f64, f64) {
    let one_run = |eng: &mut RealLlamaEngine| {
        eng.reset_context();
        let t = Instant::now();
        for &tok in &ids[..n_gen] {
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }
        n_gen as f64 / t.elapsed().as_secs_f64()
    };
    one_run(eng);
    let tps: Vec<f64> = (0..runs).map(|_| one_run(eng)).collect();
    let mean = tps.iter().sum::<f64>() / runs as f64;
    let var = tps.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / (runs as f64 - 1.0);
    (mean, var.sqrt())
}

/// Compiles a Bipolar engine, or loads it from `cache_dir` when an identical compile was saved.
#[allow(clippy::too_many_arguments)]
fn compile_or_load(
    fp32: &RealLlamaEngine,
    model_dir: &Path,
    mode: RealWeightMode,
    mode_key: &str,
    calib_tag: &str,
    calib_sequences: &[Vec<usize>],
    cache_dir: Option<&str>,
    n_ctx: usize,
) -> RealLlamaEngine {
    let Some(dir) = cache_dir else {
        return fp32.compile_sequential_bipolar_gptq(mode, calib_sequences, 8);
    };
    let path = Path::new(dir).join(format!("{mode_key}-{calib_tag}.k2tb"));
    if let Ok(eng) = RealLlamaEngine::load_bipolar(model_dir, &path, n_ctx) {
        eprintln!("  loaded cached {}", path.display());
        return eng;
    }
    let eng = fp32.compile_sequential_bipolar_gptq(mode, calib_sequences, 8);
    std::fs::create_dir_all(dir).expect("create cache dir");
    eng.save_bipolar(&path).expect("save bipolar cache");
    eng
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <model_dir> <token_ids_file> [--chunks N] [--ctx N] [--modes q40,b4,mk,b5] [--calib-ids FILE] [--calib-tokens N] [--speed]", args[0]);
        std::process::exit(1);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).map(|i| args[i + 1].clone());
    let model_dir = Path::new(&args[1]);
    let ids = read_ids(&args[2]);
    let n_ctx: usize = flag("--ctx").map_or(256, |s| s.parse().unwrap());
    let n_chunks = flag("--chunks").map_or(64, |s| s.parse().unwrap()).min(ids.len() / n_ctx);
    let modes = flag("--modes").unwrap_or_else(|| "q40,b4,mk,b5".to_string());
    let speed = args.iter().any(|a| a == "--speed");
    let cache_dir = flag("--cache-dir");
    let calib_tag = match flag("--calib-ids") {
        Some(p) => format!(
            "{}-{}",
            Path::new(&p).file_stem().unwrap().to_string_lossy(),
            flag("--calib-tokens").unwrap_or_else(|| "8192".to_string())
        ),
        None => "builtin".to_string(),
    };
    let first = n_ctx / 2;
    let p_cores = 8usize;

    println!(
        "WikiText-2: {} tokens in file | {} chunks x n_ctx={} | {} scored tokens",
        ids.len(),
        n_chunks,
        n_ctx,
        n_chunks * (n_ctx - 1 - first)
    );

    let tokenizer = BpeTokenizer::from_model_dir(model_dir).expect("failed to load BPE tokenizer");
    let calib_sequences: Vec<Vec<usize>> = if let Some(path) = flag("--calib-ids") {
        let calib_tokens: usize = flag("--calib-tokens").map_or(8192, |s| s.parse().unwrap());
        let calib = read_ids(&path);
        calib[..calib_tokens.min(calib.len())].chunks(n_ctx).map(|c| c.to_vec()).collect()
    } else {
        // Same 8 calibration passages as `verify_real_weights` (disjoint from WikiText-2 test).
        [
            "Photosynthesis converts sunlight, water, and carbon dioxide into chemical energy inside plant chloroplasts while releasing oxygen into the atmosphere.",
            "Ocean tides rise and fall twice each lunar day because gravitational attraction varies across opposite sides of a rotating coastal basin.",
            "String instruments produce harmonic overtones when a vibrating bow excites standing waves along a wooden soundboard in a concert hall.",
            "Suspension bridges distribute heavy roadway loads through vertical suspender cables anchored into bedrock towers on either river bank.",
            "Artisan sourdough fermentation relies on wild yeast and lactic acid bacteria metabolizing flour starches into aromatic organic acids.",
            "Spoken human languages organize vowel and consonant phonemes into syllables, morphemes, and recursive grammatical clauses.",
            "Migratory songbirds navigate across continents at night using geomagnetic inclination cues combined with starlight polarization patterns.",
            "Metamorphic marble recrystallizes deep within mountain belts when limestone strata experience sustained tectonic pressure and geothermal heat.",
        ]
        .iter()
        .map(|txt| {
            let mut v = vec![tokenizer.eos_token_id];
            v.extend(tokenizer.encode(txt));
            v
        })
        .collect()
    };
    println!(
        "Calibration: {} sequences, {} tokens",
        calib_sequences.len(),
        calib_sequences.iter().map(|s| s.len()).sum::<usize>()
    );

    let mut fp32 = RealLlamaEngine::load_fp32_reference(model_dir, n_ctx).expect("load fp32");
    let mut candidates: Vec<(&str, RealLlamaEngine)> = Vec::new();
    for m in modes.split(',') {
        let t = Instant::now();
        let (label, eng) = match m {
            "q40" => (
                "In-house GGUF Q4_0 reimpl",
                fp32.compile_from_reference(RealWeightMode::GgufQ4_0, None),
            ),
            "b4" | "mk" | "b5" => {
                let (label, mode) = match m {
                    "b4" => ("Kronos Bipolar-4 + GPTQ", RealWeightMode::KronosBipolarGptq { num_planes: 4 }),
                    "mk" => ("Kronos Bipolar Mixed-K + GPTQ", RealWeightMode::KronosBipolarGptqMixedK),
                    _ => ("Kronos Bipolar-5 + GPTQ", RealWeightMode::KronosBipolarGptq { num_planes: 5 }),
                };
                (label, compile_or_load(&fp32, model_dir, mode, m, &calib_tag, &calib_sequences, cache_dir.as_deref(), n_ctx))
            }
            // `pQOGDH`: planes for qkv, o, gate_up, down, lm_head (e.g. `p44455`).
            p if p.len() == 6 && p.starts_with('p') => {
                let d: Vec<usize> = p[1..].bytes().map(|b| (b - b'0') as usize).collect();
                let mode = RealWeightMode::KronosBipolarGptqPlanes {
                    qkv: d[0],
                    o: d[1],
                    gate_up: d[2],
                    down: d[3],
                    head: d[4],
                };
                let label: &'static str = Box::leak(format!("Kronos Bipolar planes {}", &p[1..]).into_boxed_str());
                (label, compile_or_load(&fp32, model_dir, mode, m, &calib_tag, &calib_sequences, cache_dir.as_deref(), n_ctx))
            }
            other => panic!("unknown mode {other}"),
        };
        eprintln!("  compiled {label} in {:.1}s", t.elapsed().as_secs_f64());
        candidates.push((label, eng));
    }
    fp32.set_num_threads(p_cores);
    for (_, eng) in candidates.iter_mut() {
        eng.set_num_threads(p_cores);
    }

    let mut fp32_acc = Acc::default();
    let mut cand_acc: Vec<Acc> = candidates.iter().map(|_| Acc::default()).collect();
    let t0 = Instant::now();

    for c in 0..n_chunks {
        let chunk = &ids[c * n_ctx..(c + 1) * n_ctx];

        // FP32 reference log-probs at the scored positions.
        fp32.reset_context();
        let mut ref_lp: Vec<Vec<f64>> = Vec::with_capacity(n_ctx - 1 - first);
        for (i, &tok) in chunk[..n_ctx - 1].iter().enumerate() {
            let logits = fp32.forward_token(tok);
            if i >= first {
                let lp = log_softmax(logits);
                fp32_acc.add(-lp[chunk[i + 1]], 0.0, true);
                ref_lp.push(lp);
            }
        }

        for ((_, eng), acc) in candidates.iter_mut().zip(cand_acc.iter_mut()) {
            eng.reset_context();
            for (i, &tok) in chunk[..n_ctx - 1].iter().enumerate() {
                let logits = eng.forward_token(tok);
                if i >= first {
                    let lq = log_softmax(logits);
                    let lp = &ref_lp[i - first];
                    let kl: f64 = lp
                        .iter()
                        .zip(lq.iter())
                        .filter(|(&p, _)| p > -40.0)
                        .map(|(&p, &q)| p.exp() * (p - q))
                        .sum();
                    acc.add(-lq[chunk[i + 1]], kl.max(0.0), argmax(logits) == argmax_f64(lp));
                }
            }
        }
        if (c + 1) % 16 == 0 {
            eprintln!("  chunk {}/{} ({:.0}s)", c + 1, n_chunks, t0.elapsed().as_secs_f64());
        }
    }

    println!(
        "\n  {:<32} | {:>6} | {:>7} | {:>18} | {:>8} | {:>7}",
        "Engine / Mode", "Bits", "RAM", "PPL (± stderr)", "KL(nat)", "Top-1%"
    );
    println!("  {}", "-".repeat(94));
    let (ppl, se, _, _) = fp32_acc.report();
    println!(
        "  {:<32} | {:>5.2}b | {:>6.1}M | {:>9.4} ± {:<6.4} | {:>8} | {:>7}",
        "Exact FP32 Reference",
        fp32.physical_bits_per_weight(),
        fp32.resident_weight_bytes() as f64 / (1024.0 * 1024.0),
        ppl,
        se,
        "-",
        "-"
    );
    for ((label, eng), acc) in candidates.iter().zip(cand_acc.iter()) {
        let (ppl, se, kl, top1) = acc.report();
        println!(
            "  {:<32} | {:>5.2}b | {:>6.1}M | {:>9.4} ± {:<6.4} | {:>8.5} | {:>6.2}%",
            label,
            eng.physical_bits_per_weight(),
            eng.resident_weight_bytes() as f64 / (1024.0 * 1024.0),
            ppl,
            se,
            kl,
            top1
        );
    }

    if speed {
        println!("\n  Decode speed (tg32, mean ± sd of 5 runs, tok/s)");
        for (label, eng) in std::iter::once(("Exact FP32 Reference", &mut fp32))
            .chain(candidates.iter_mut().map(|(l, e)| (*l, e)))
        {
            eng.set_num_threads(1);
            let (t1, s1) = decode_speed(eng, &ids, 32, 5);
            eng.set_num_threads(p_cores);
            let (t8, s8) = decode_speed(eng, &ids, 32, 5);
            println!("  {:<32} | 1T {:>7.1} ± {:<5.1} | 8T {:>7.1} ± {:<5.1}", label, t1, s1, t8, s8);
        }
    }
}
