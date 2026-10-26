//! WikiText-2 perplexity / KL / top-1 evaluation, matched 1:1 to `llama-perplexity`.
//!
//! Usage: `wikitext_ppl <model_dir> <token_ids_file> [n_chunks=64] [n_ctx=256]`
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: {} <model_dir> <token_ids_file> [n_chunks=64] [n_ctx=256]", args[0]);
        std::process::exit(1);
    }
    let model_dir = Path::new(&args[1]);
    let ids: Vec<usize> = std::fs::read_to_string(&args[2])
        .expect("failed to read token ids")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().expect("bad token id"))
        .collect();
    let n_chunks: usize = args.get(3).map_or(64, |s| s.parse().unwrap());
    let n_ctx: usize = args.get(4).map_or(256, |s| s.parse().unwrap());
    let n_chunks = n_chunks.min(ids.len() / n_ctx);
    let first = n_ctx / 2;
    let p_cores = 8usize;

    println!(
        "WikiText-2: {} tokens in file | {} chunks x n_ctx={} | {} scored tokens",
        ids.len(),
        n_chunks,
        n_ctx,
        n_chunks * (n_ctx - 1 - first)
    );

    // Same 8 calibration passages as `verify_real_weights` (disjoint from WikiText-2 test).
    let tokenizer = BpeTokenizer::from_model_dir(model_dir).expect("failed to load BPE tokenizer");
    let raw_calib_texts = [
        "Photosynthesis converts sunlight, water, and carbon dioxide into chemical energy inside plant chloroplasts while releasing oxygen into the atmosphere.",
        "Ocean tides rise and fall twice each lunar day because gravitational attraction varies across opposite sides of a rotating coastal basin.",
        "String instruments produce harmonic overtones when a vibrating bow excites standing waves along a wooden soundboard in a concert hall.",
        "Suspension bridges distribute heavy roadway loads through vertical suspender cables anchored into bedrock towers on either river bank.",
        "Artisan sourdough fermentation relies on wild yeast and lactic acid bacteria metabolizing flour starches into aromatic organic acids.",
        "Spoken human languages organize vowel and consonant phonemes into syllables, morphemes, and recursive grammatical clauses.",
        "Migratory songbirds navigate across continents at night using geomagnetic inclination cues combined with starlight polarization patterns.",
        "Metamorphic marble recrystallizes deep within mountain belts when limestone strata experience sustained tectonic pressure and geothermal heat.",
    ];
    let calib_sequences: Vec<Vec<usize>> = raw_calib_texts
        .iter()
        .map(|txt| {
            let mut v = vec![tokenizer.eos_token_id];
            v.extend(tokenizer.encode(txt));
            v
        })
        .collect();

    let mut fp32 = RealLlamaEngine::load_fp32_reference(model_dir, n_ctx).expect("load fp32");
    let mut candidates: Vec<(&str, RealLlamaEngine)> = vec![(
        "In-house GGUF Q4_0 reimpl",
        fp32.compile_from_reference(RealWeightMode::GgufQ4_0, None),
    )];
    for (label, mode) in [
        ("Kronos Bipolar-4 + GPTQ", RealWeightMode::KronosBipolarGptq { num_planes: 4 }),
        ("Kronos Bipolar Mixed-K + GPTQ", RealWeightMode::KronosBipolarGptqMixedK),
        ("Kronos Bipolar-5 + GPTQ", RealWeightMode::KronosBipolarGptq { num_planes: 5 }),
    ] {
        candidates.push((label, fp32.compile_sequential_bipolar_gptq(mode, &calib_sequences, p_cores)));
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
                    let ref_top1 = argmax_f64(lp);
                    acc.add(-lq[chunk[i + 1]], kl.max(0.0), argmax(logits) == ref_top1);
                }
            }
        }
        if (c + 1) % 8 == 0 {
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
        fp32.linear_weight_bytes() as f64 / (1024.0 * 1024.0),
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
            eng.linear_weight_bytes() as f64 / (1024.0 * 1024.0),
            ppl,
            se,
            kl,
            top1
        );
    }
}

fn argmax_f64(v: &[f64]) -> usize {
    v.iter()
        .enumerate()
        .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) })
        .0
}
