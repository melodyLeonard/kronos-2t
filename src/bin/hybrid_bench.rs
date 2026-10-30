//! Neural Engine prefill + Kronos CPU decode vs Kronos CPU only.
//!
//! Usage: `hybrid_bench <model_dir> <token_ids_file> [--mode p55454] [--chunks 16]`
//!
//! Needs `<model_dir>/kronos-<mode>.k2tb` (from `kronos_chat`) and Neural Engine buckets
//! `<model_dir>/ane/prefill_s<N>.mlmodelc` (from `ane/export_prefill.py`).

use kronos_2t::backend::coreml::{ComputeUnits, CoreMlModel};
use kronos_2t::model::real_llm::{argmax, RealLlamaEngine};
use std::path::Path;
use std::time::Instant;

fn log_softmax(l: &[f32]) -> Vec<f64> {
    let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z = m + l.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln();
    l.iter().map(|&v| v as f64 - z).collect()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

struct Ane(Vec<(usize, CoreMlModel)>);

impl Ane {
    fn prefill(&mut self, eng: &mut RealLlamaEngine, prompt: &[usize]) -> Vec<f32> {
        let n = prompt.len();
        let h = eng.cfg.hidden_size;
        let (seq, model) = self.0.iter_mut().find(|(s, _)| *s >= n).expect("no bucket fits");
        let seq = *seq;
        let mut x = vec![0.0f32; seq * h];
        for (t, &tok) in prompt.iter().enumerate() {
            eng.embed(tok, &mut x[t * h..(t + 1) * h]);
        }
        let out = model.predict("x", &[1, seq, h], &x, &["hidden", "keys", "values"]).expect("predict");
        eng.reset_context();
        eng.load_prefix_kv(&out[1], &out[2], seq, n);
        eng.logits_from_hidden(&out[0][(n - 1) * h..n * h]).to_vec()
    }
}

fn decode_speed(eng: &mut RealLlamaEngine, first: usize, n: usize) -> f64 {
    let t = Instant::now();
    let mut tok = first;
    for _ in 0..n {
        tok = argmax(eng.forward_token(tok));
    }
    n as f64 / t.elapsed().as_secs_f64()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let model_dir = Path::new(&args[1]);
    let ids: Vec<usize> = std::fs::read_to_string(&args[2])
        .expect("token ids")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect();
    let mode = flag("--mode").unwrap_or_else(|| "p55454".into());
    let chunks: usize = flag("--chunks").map_or(16, |v| v.parse().unwrap());

    let mut eng = RealLlamaEngine::load_bipolar(model_dir, &model_dir.join(format!("kronos-{mode}.k2tb")), 1024)
        .expect("compiled model (run kronos_chat once)");
    eng.repack_sdot();
    eng.set_num_threads(8);
    let mut buckets = Vec::new();
    for e in std::fs::read_dir(model_dir.join("ane")).expect("ane dir").flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(n) = name.strip_prefix("prefill_s").and_then(|r| r.strip_suffix(".mlmodelc")) {
            let seq: usize = n.parse().unwrap();
            let mut m = CoreMlModel::load(&e.path(), ComputeUnits::CpuAndNeuralEngine).expect("load");
            let h = eng.cfg.hidden_size;
            m.predict("x", &[1, seq, h], &vec![0.0; seq * h], &["hidden"]).expect("warm-up");
            buckets.push((seq, m));
        }
    }
    buckets.sort_by_key(|(s, _)| *s);
    let max_bucket = buckets.last().map_or(0, |(s, _)| *s);
    println!("buckets: {:?}", buckets.iter().map(|(s, _)| *s).collect::<Vec<_>>());
    let mut ane = Ane(buckets);

    println!("\n prompt | CPU prefill        | ANE prefill (bucket)     | decode after CPU / ANE | last-token KL, top-1");
    for n in [32usize, 64, 128, 256, 512].into_iter().filter(|&n| n <= max_bucket) {
        let prompt = &ids[..n];
        let mut cpu_t = Vec::new();
        let mut cpu_logits = Vec::new();
        for _ in 0..3 {
            eng.reset_context();
            let t = Instant::now();
            cpu_logits = eng.forward_prompt(prompt).to_vec();
            cpu_t.push(t.elapsed().as_secs_f64());
        }
        let cpu_dec = decode_speed(&mut eng, argmax(&cpu_logits), 64);
        let mut ane_t = Vec::new();
        let mut ane_logits = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            ane_logits = ane.prefill(&mut eng, prompt);
            ane_t.push(t.elapsed().as_secs_f64());
        }
        let ane_dec = decode_speed(&mut eng, argmax(&ane_logits), 64);
        let (p, q) = (log_softmax(&cpu_logits), log_softmax(&ane_logits));
        let kl: f64 = p.iter().zip(&q).filter(|(a, _)| **a > -40.0).map(|(a, b)| a.exp() * (a - b)).sum();
        let bucket = ane.0.iter().find(|(s, _)| *s >= n).unwrap().0;
        let (c, a) = (median(cpu_t), median(ane_t));
        println!(
            " {n:>6} | {:>6.1} ms {:>6.0} t/s | {:>6.1} ms {:>6.0} t/s ({bucket:>3}) | {:>6.1} / {:>6.1} tok/s   | {kl:.4}, {}",
            c * 1e3,
            n as f64 / c,
            a * 1e3,
            n as f64 / a,
            cpu_dec,
            ane_dec,
            argmax(&cpu_logits) == argmax(&ane_logits)
        );
    }

    // Teacher-forced: prefill the first half of each 256-token chunk, decode the second half.
    let (half, ctx) = (128usize, 256usize);
    let (mut nll_cpu, mut nll_hyb, mut n_tok) = (0.0f64, 0.0f64, 0usize);
    for c in 0..chunks.min(ids.len() / ctx) {
        let chunk = &ids[c * ctx..(c + 1) * ctx];
        for (hybrid, nll) in [(false, &mut nll_cpu), (true, &mut nll_hyb)] {
            let mut logits = if hybrid {
                ane.prefill(&mut eng, &chunk[..half])
            } else {
                eng.reset_context();
                eng.forward_prompt(&chunk[..half]).to_vec()
            };
            for (i, &tok) in chunk.iter().enumerate().skip(half) {
                *nll -= log_softmax(&logits)[tok];
                if i + 1 < ctx {
                    logits = eng.forward_token(tok).to_vec();
                }
            }
        }
        n_tok += ctx - half;
    }
    println!(
        "\nWikiText, {n_tok} tokens decoded after a {half}-token prefill: perplexity CPU-only {:.3} | hybrid {:.3}",
        (nll_cpu / n_tok as f64).exp(),
        (nll_hyb / n_tok as f64).exp()
    );
}
