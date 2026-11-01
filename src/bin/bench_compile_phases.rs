//! Times each per-expert GPTQ compile phase at Qwen3-30B-A3B shapes on synthetic data.
//!
//! `bench_compile_phases [--threads N] [--bits 4] [--tokens 1024] [--experts 4] [--gpu]`
//! Prints seconds per phase per expert and the projected full-model (48 x 128) time.
//! `--gpu` runs the same phases through the Metal GPTQ compiler.
//! `--model-dir DIR` also times FP8 expert reads (dequantized to f32) from a real checkpoint.

use kronos_2t::kernels::hadamard::fwht64_signed_inplace;
use kronos_2t::kernels::simd_lut::{nf_sdot_stripe_range_int8, Int8ChunkActivation};
use kronos_2t::model::compiler::{compile_hadamard_gptq_nf, FullRowGptqHessian};
use std::time::Instant;

const H: usize = 2048;
const INTER: usize = 768;
const LAYERS: usize = 48;
const EXPERTS: usize = 128;

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
}

fn main() {
    let threads = arg("--threads", kronos_2t::kernels::thread_pool::physical_p_cores());
    let bits = arg("--bits", 4);
    let tokens = arg("--tokens", 1024);
    let n_exp = arg("--experts", 4);
    let use_gpu = std::env::args().any(|a| a == "--gpu");
    if let Some(dir) = std::env::args().skip_while(|a| a != "--model-dir").nth(1) {
        use kronos_2t::model::sparse_moe::read_moe_expert_f32;
        use kronos_2t::storage::safetensors::SafeTensorsStore;
        let store = SafeTensorsStore::open_dir(std::path::Path::new(&dir)).expect("safetensors dir");
        let t = Instant::now();
        for e in 0..n_exp {
            let (gu, dn) = read_moe_expert_f32(&store, 0, e, EXPERTS, H, INTER, true).expect("expert read");
            assert_eq!(gu.len(), 2 * INTER * H);
            assert_eq!(dn.len(), H * INTER);
        }
        let per = t.elapsed().as_secs_f64() / n_exp as f64;
        println!("FP8 expert read+dequant: {per:.3}s/expert -> full model {:.1} min", per * (LAYERS * EXPERTS) as f64 / 60.0);
    }
    let sgc = 2;
    let mut seed = 7u64;

    let gu: Vec<f32> = (0..2 * INTER * H).map(|_| 0.02 * lcg(&mut seed)).collect();
    let dn: Vec<f32> = (0..H * INTER).map(|_| 0.02 * lcg(&mut seed)).collect();
    let acts: Vec<Vec<f32>> = (0..tokens)
        .map(|_| (0..H).map(|_| lcg(&mut seed)).collect())
        .collect();

    // Per-layer: input Hessian over all calibration tokens (shared by the layer's experts).
    let t = Instant::now();
    let mut hess_in = FullRowGptqHessian::new(H);
    for a in &acts {
        hess_in.observe_activation(a);
    }
    hess_in.finalize(0.01);
    let t_hess_in = t.elapsed().as_secs_f64();

    let act_i8: Vec<Int8ChunkActivation> = acts
        .iter()
        .map(|a| {
            let mut r = a.clone();
            fwht64_signed_inplace(&mut r);
            let mut i8a = Int8ChunkActivation::new_zeroed(H);
            i8a.populate(&r);
            i8a
        })
        .collect();

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    if use_gpu {
        use kronos_2t::backend::metal_compile::{HessianSet, MetalGptqCompiler, Slot};
        let batch = arg("--batch", 8).max(1);
        let mut gpu = MetalGptqCompiler::new().expect("Metal device");
        gpu.upload_layer_acts(&act_i8).unwrap();
        let ids: Vec<u32> = (0..tokens as u32).collect();
        let gus: Vec<&[f32]> = (0..batch).map(|_| gu.as_slice()).collect();
        let dns: Vec<&[f32]> = (0..batch).map(|_| dn.as_slice()).collect();
        let (mut t_gu, mut t_hdn, mut t_dn, mut t_fwd) = (0.0, 0.0, 0.0, 0.0);
        let n_batches = n_exp.div_ceil(batch);
        for _ in 0..n_batches {
            let t = Instant::now();
            let _gu_mats = gpu.compile_nf_batch(Slot::GateUp, &gus, 2 * INTER, H, bits, sgc, HessianSet::Shared(&hess_in)).unwrap();
            t_gu += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let mut hds: Vec<FullRowGptqHessian> = Vec::with_capacity(batch);
            for m in 0..batch {
                let sw_all = gpu.forward_gate_up_swiglu(m, &ids).unwrap();
                let mut hd = FullRowGptqHessian::new(INTER);
                for row in sw_all.chunks_exact(INTER) {
                    hd.observe_prerotated(row);
                }
                hds.push(hd);
            }
            std::thread::scope(|s| {
                for hd in hds.iter_mut() {
                    s.spawn(move || hd.finalize(0.01));
                }
            });
            t_hdn += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let hrefs: Vec<&FullRowGptqHessian> = hds.iter().collect();
            let _dn_mats = gpu.compile_nf_batch(Slot::Down, &dns, H, INTER, bits, sgc, HessianSet::PerMatrix(&hrefs)).unwrap();
            t_dn += t.elapsed().as_secs_f64();
            let t = Instant::now();
            for m in 0..batch {
                let _out = gpu.forward_down(m).unwrap();
            }
            t_fwd += t.elapsed().as_secs_f64();
        }
        report(&format!("gpu batch={batch}"), threads, bits, tokens, t_hess_in, n_batches * batch, [t_gu, t_hdn, t_dn, t_fwd]);
        return;
    }
    let (mut t_gu, mut t_hdn, mut t_dn, mut t_fwd) = (0.0, 0.0, 0.0, 0.0);
    let mut gu_tmp = vec![0.0f32; 2 * INTER];
    let mut sw = vec![0.0f32; INTER];
    let mut dn_tmp = vec![0.0f32; H];
    let mut mid = Int8ChunkActivation::new_zeroed(INTER);
    for _ in 0..n_exp {
        let t = Instant::now();
        let gu_mat = compile_hadamard_gptq_nf(&gu, 2 * INTER, H, bits, sgc, Some(&hess_in), threads);
        t_gu += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut hd = FullRowGptqHessian::new(INTER);
        for a in &act_i8 {
            unsafe { nf_sdot_stripe_range_int8(&gu_mat, a, 0, gu_mat.num_stripes, gu_tmp.as_mut_ptr()) };
            let (g, u) = gu_tmp.split_at(INTER);
            for ((s, &gv), &uv) in sw.iter_mut().zip(g).zip(u) {
                *s = gv / (1.0 + (-gv).exp()) * uv;
            }
            fwht64_signed_inplace(&mut sw);
            hd.observe_prerotated(&sw);
        }
        hd.finalize(0.01);
        t_hdn += t.elapsed().as_secs_f64();

        let t = Instant::now();
        let dn_mat = compile_hadamard_gptq_nf(&dn, H, INTER, bits, sgc, Some(&hd), threads);
        t_dn += t.elapsed().as_secs_f64();

        let t = Instant::now();
        for a in &act_i8 {
            unsafe { nf_sdot_stripe_range_int8(&gu_mat, a, 0, gu_mat.num_stripes, gu_tmp.as_mut_ptr()) };
            let (g, u) = gu_tmp.split_at(INTER);
            for ((s, &gv), &uv) in sw.iter_mut().zip(g).zip(u) {
                *s = gv / (1.0 + (-gv).exp()) * uv;
            }
            fwht64_signed_inplace(&mut sw);
            mid.populate(&sw);
            unsafe { nf_sdot_stripe_range_int8(&dn_mat, &mid, 0, dn_mat.num_stripes, dn_tmp.as_mut_ptr()) };
        }
        t_fwd += t.elapsed().as_secs_f64();
    }
    report("cpu", threads, bits, tokens, t_hess_in, n_exp, [t_gu, t_hdn, t_dn, t_fwd]);
}

fn report(label: &str, threads: usize, bits: usize, tokens: usize, t_hess_in: f64, n_exp: usize, t: [f64; 4]) {
    let n = n_exp as f64;
    let names = [
        "gate_up GPTQ",
        "down Hessian (fwd gu + SYRK + Cholesky)",
        "down GPTQ",
        "expert forward for next layer",
    ];
    println!("[{label}] threads={threads} bits={bits} tokens/expert={tokens}");
    println!("layer input Hessian ({tokens} tokens, {H}x{H}): {t_hess_in:.2}s  -> x{LAYERS} = {:.1}s", t_hess_in * LAYERS as f64);
    let mut tot = 0.0;
    for (name, s) in names.iter().zip(t) {
        let s = s / n;
        tot += s;
        println!("{name:<42} {s:7.3}s/expert  -> full model {:6.1} min", s * (LAYERS * EXPERTS) as f64 / 60.0);
    }
    println!("per-expert total {tot:.3}s -> full model {:.1} min (+{:.1} min layer Hessians)", tot * (LAYERS * EXPERTS) as f64 / 60.0, t_hess_in * LAYERS as f64 / 60.0);
}
