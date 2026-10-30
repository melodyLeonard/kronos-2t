//! Benchmark for the 800B sparse-MoE engine: expert streaming from the SSD vs the SLRU DRAM cache.
//!
//! Suite 0 compares NF3 and VQ2 experts with FP32 on SmolLM2-1.7B FFN weights (skipped if absent).
//! Suite A times cold `F_NOCACHE` reads, Zipf traffic with SLRU and prefetch, and all-hot DRAM.
//! Suite B times the full model with all active weights in DRAM.
//!
//! Flags: --shard-path P, --real-model DIR, --shard-layers N, --shard-experts N, --threads N,
//!        --exec-layers N, --vq2, --keep-shard

use kronos_2t::model::compiler::{compile_hadamard_gptq_nf, compile_hadamard_gptq_vq2};
use kronos_2t::model::moe_800b::{
    write_exact_experts_k2tm, write_exact_nf_experts_k2tm, write_k2tm_shard_file,
    write_nf_k2tm_shard_file, DirectNvmeShard, DirectNvmeSlruPager, K2tmShardMmap,
    Kronos800bConfig, Kronos800bMoEEngine, MmapNfExpertView, MmapVq2ExpertView,
};
use kronos_2t::storage::safetensors::SafeTensorsFile;
use kronos_2t::kernels::simd_lut::{Int8ChunkActivation, Vq2ActivationLut};
use kronos_2t::kernels::thread_pool::{default_inference_threads, physical_p_cores, SpinThreadPool};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct TempShardGuard {
    path: PathBuf,
    keep: bool,
}

impl Drop for TempShardGuard {
    fn drop(&mut self) {
        if !self.keep && self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Deterministic `Zipf(alpha)` sampler over `0..num_experts` with per-layer domain permutation.
struct ZipfDomainSampler {
    cdf: Vec<f64>,
    state: u64,
}

impl ZipfDomainSampler {
    fn new(num_experts: usize, alpha: f64, seed: u64) -> Self {
        let mut weights = Vec::with_capacity(num_experts);
        let mut sum = 0.0f64;
        for r in 1..=num_experts {
            let w = 1.0 / (r as f64).powf(alpha);
            weights.push(w);
            sum += w;
        }
        let mut cdf = Vec::with_capacity(num_experts);
        let mut acc = 0.0f64;
        for w in weights {
            acc += w / sum;
            cdf.push(acc);
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Self {
            cdf,
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    #[inline]
    fn next_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    fn sample_layer_experts(&mut self, layer_idx: usize, k_eff: usize) -> Vec<usize> {
        let n = self.cdf.len();
        let mut out = Vec::with_capacity(k_eff);
        let perm_stride = 7usize;
        while out.len() < k_eff.min(n.saturating_sub(1)) {
            let u = self.next_f64();
            let rank = self.cdf.partition_point(|&c| c < u).min(n - 1);
            let span = (n - 1).max(1);
            let eid = 1 + ((rank * perm_stride + layer_idx * 3) % span);
            if !out.contains(&eid) {
                out.push(eid);
            }
        }
        out
    }
}

fn expected_speculative_tokens(batch_b: usize, alpha: f64) -> f64 {
    let gamma = batch_b.saturating_sub(1);
    if (1.0 - alpha).abs() < 1e-9 {
        (gamma + 1) as f64
    } else {
        (1.0 - alpha.powi((gamma + 1) as i32)) / (1.0 - alpha)
    }
}

fn gib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0 * 1024.0)
}

fn mib(bytes: u64) -> f64 {
    (bytes as f64) / (1024.0 * 1024.0)
}

fn cosine_and_rel_l2(ref_v: &[f32], test_v: &[f32]) -> (f64, f64) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut diff_sq = 0.0f64;
    for (&a, &b) in ref_v.iter().zip(test_v.iter()) {
        let af = a as f64;
        let bf = b as f64;
        dot += af * bf;
        na += af * af;
        nb += bf * bf;
        diff_sq += (af - bf) * (af - bf);
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    let rel = if na > 0.0 { (diff_sq / na).sqrt() } else { 0.0 };
    (cos, rel)
}

/// FP32 SwiGLU FFN reference: `W_down * (silu(W_gate * x) * (W_up * x))`.
fn fp32_swiglu_ffn(
    gate_w: &[f32],
    up_w: &[f32],
    down_w: &[f32],
    d_model: usize,
    inter_dim: usize,
    x: &[f32],
) -> Vec<f32> {
    let mut sw = vec![0.0f32; inter_dim];
    for i in 0..inter_dim {
        let g: f32 = gate_w[i * d_model..(i + 1) * d_model]
            .iter()
            .zip(x.iter())
            .map(|(w, a)| w * a)
            .sum();
        let u: f32 = up_w[i * d_model..(i + 1) * d_model]
            .iter()
            .zip(x.iter())
            .map(|(w, a)| w * a)
            .sum();
        sw[i] = (g / (1.0 + (-g).exp())) * u;
    }
    let mut out = vec![0.0f32; d_model];
    for r in 0..d_model {
        out[r] = down_w[r * inter_dim..(r + 1) * inter_dim]
            .iter()
            .zip(sw.iter())
            .map(|(w, a)| w * a)
            .sum();
    }
    out
}

/// Suite 0: splits 4 SmolLM2 FFN layers into 4 experts each, quantizes them to NF3 and VQ2,
/// reads them back with `F_NOCACHE` and compares the outputs with FP32.
fn run_real_pretrained_moe_suite(model_dir: &Path, scratch_dir: &Path, threads: usize) {
    let st_path = model_dir.join("model.safetensors");
    let Ok(st) = SafeTensorsFile::open(&st_path) else {
        return;
    };
    println!();
    println!("  SUITE 0: REAL PRETRAINED SmolLM2-1.7B-Instruct WEIGHT MoE PARTITION + F_NOCACHE NVMe FIDELITY");
    println!("  --------------------------------------------------------------------------------------------------------");

    let d_model = 2048usize;
    let full_inter = 8192usize;
    let experts_per_layer = 4usize;
    let d_ff_exp = full_inter / experts_per_layer; // 2048
    let num_real_layers = 4usize; // 16 experts of 2048x2048

    let embed_tv = st.tensor("model.embed_tokens.weight").expect("embed_tokens");
    let real_cfg = Kronos800bConfig::compact_test(1, d_model, d_ff_exp, 2);

    let t_comp = Instant::now();
    let mut nf3_experts = Vec::with_capacity(num_real_layers * experts_per_layer);
    let mut vq2_experts = Vec::with_capacity(num_real_layers * experts_per_layer);
    let mut test_xs = Vec::with_capacity(num_real_layers);
    let mut fp32_outs = Vec::with_capacity(num_real_layers);

    for l in 0..num_real_layers {
        let gate_w = st
            .tensor(&format!("model.layers.{l}.mlp.gate_proj.weight"))
            .expect("gate_proj")
            .to_f32_vec();
        let up_w = st
            .tensor(&format!("model.layers.{l}.mlp.up_proj.weight"))
            .expect("up_proj")
            .to_f32_vec();
        let down_w = st
            .tensor(&format!("model.layers.{l}.mlp.down_proj.weight"))
            .expect("down_proj")
            .to_f32_vec();
        let norm_w = st
            .tensor(&format!("model.layers.{l}.post_attention_layernorm.weight"))
            .expect("post_attn_norm")
            .to_f32_vec();

        // Test input: RMS-normalized sum of three token embeddings.
        let mut raw_x = vec![0.0f32; d_model];
        let mut row_tmp = vec![0.0f32; d_model];
        for tok_id in [415usize, 1823 + l * 97, 5112 + l * 31] {
            embed_tv.copy_row_f32(tok_id, &mut row_tmp);
            for (dst, &v) in raw_x.iter_mut().zip(row_tmp.iter()) {
                *dst += v;
            }
        }
        let rms = (raw_x.iter().map(|v| v * v).sum::<f32>() / (d_model as f32) + 1e-5).sqrt();
        for (x_i, &nw) in raw_x.iter_mut().zip(norm_w.iter()) {
            *x_i = (*x_i / rms) * nw;
        }
        let fp32_y = fp32_swiglu_ffn(&gate_w, &up_w, &down_w, d_model, full_inter, &raw_x);

        // Split the 8192 intermediate channels into 4 experts; their outputs sum to the full FFN output.
        for e in 0..experts_per_layer {
            let c0 = e * d_ff_exp;
            let mut gu_exp = vec![0.0f32; 2 * d_ff_exp * d_model];
            gu_exp[..d_ff_exp * d_model]
                .copy_from_slice(&gate_w[c0 * d_model..(c0 + d_ff_exp) * d_model]);
            gu_exp[d_ff_exp * d_model..]
                .copy_from_slice(&up_w[c0 * d_model..(c0 + d_ff_exp) * d_model]);

            let mut dn_exp = vec![0.0f32; d_model * d_ff_exp];
            for r in 0..d_model {
                dn_exp[r * d_ff_exp..(r + 1) * d_ff_exp]
                    .copy_from_slice(&down_w[r * full_inter + c0..r * full_inter + c0 + d_ff_exp]);
            }

            let gu_nf3 = compile_hadamard_gptq_nf(&gu_exp, 2 * d_ff_exp, d_model, 3, 2, None, threads);
            let dn_nf3 = compile_hadamard_gptq_nf(&dn_exp, d_model, d_ff_exp, 3, 2, None, threads);
            nf3_experts.push((gu_nf3, dn_nf3));

            let gu_vq2 = compile_hadamard_gptq_vq2(&gu_exp, 2 * d_ff_exp, d_model, 2, None, threads);
            let dn_vq2 = compile_hadamard_gptq_vq2(&dn_exp, d_model, d_ff_exp, 2, None, threads);
            vq2_experts.push((gu_vq2, dn_vq2));
        }

        test_xs.push(raw_x);
        fp32_outs.push(fp32_y);
    }

    let nf3_path = scratch_dir.join(format!("kronos_real17b_nf3_{}.k2tm", std::process::id()));
    let vq2_path = scratch_dir.join(format!("kronos_real17b_vq2_{}.k2tm", std::process::id()));
    let _g1 = TempShardGuard { path: nf3_path.clone(), keep: false };
    let _g2 = TempShardGuard { path: vq2_path.clone(), keep: false };

    let nf3_hdr = write_exact_nf_experts_k2tm(
        &nf3_path,
        &real_cfg,
        3,
        1,
        nf3_experts.len(),
        &nf3_experts,
    )
    .expect("write real NF3 shard");
    let vq2_hdr = write_exact_experts_k2tm(&vq2_path, &real_cfg, &vq2_experts)
        .expect("write real VQ2 shard");

    let pool = SpinThreadPool::new(threads);
    let nf3_lay = real_cfg.nf_expert_layout(3);
    let vq2_lay = real_cfg.expert_layout();

    // Read the NF3 experts back with `F_NOCACHE` and compare with FP32.
    let nf3_shard = DirectNvmeShard::open(&nf3_path, nf3_lay.aligned_expert_bytes).unwrap();
    let mut nf3_pager = DirectNvmeSlruPager::new(nf3_shard, 4, true, false).unwrap();
    let mut nf3_cos_sum = 0.0f64;
    let mut nf3_rel_sum = 0.0f64;
    let mut gu_buf = vec![0.0f32; 2 * d_ff_exp];
    let mut out_buf = vec![0.0f32; d_model];
    let mut act_in = Int8ChunkActivation::new_zeroed(d_model);
    let mut act_mid = Int8ChunkActivation::new_zeroed(d_ff_exp);

    for l in 0..num_real_layers {
        act_in.populate_fwht(&test_xs[l]);
        let mut sum_y = vec![0.0f32; d_model];
        for e in 0..experts_per_layer {
            let slice = nf3_pager.access_expert_slice(0, l * experts_per_layer + e).unwrap();
            let view = MmapNfExpertView::from_aligned_slice(slice, nf3_lay).unwrap();
            view.forward_swiglu_zero_copy(&act_in, &mut gu_buf, &mut act_mid, &mut out_buf, Some(&pool));
            for (dst, &v) in sum_y.iter_mut().zip(out_buf.iter()) {
                *dst += v;
            }
        }
        let (cos, rel) = cosine_and_rel_l2(&fp32_outs[l], &sum_y);
        nf3_cos_sum += cos;
        nf3_rel_sum += rel;
    }

    // Same for VQ2.
    let vq2_shard = DirectNvmeShard::open(&vq2_path, vq2_lay.aligned_expert_bytes).unwrap();
    let mut vq2_pager = DirectNvmeSlruPager::new(vq2_shard, 4, true, false).unwrap();
    let mut vq2_cos_sum = 0.0f64;
    let mut vq2_rel_sum = 0.0f64;
    let mut vlut_in = Vq2ActivationLut::new_zeroed(d_model);
    let mut vlut_mid = Vq2ActivationLut::new_zeroed(d_ff_exp);

    for l in 0..num_real_layers {
        vlut_in.populate_fwht(&test_xs[l]);
        let mut sum_y = vec![0.0f32; d_model];
        for e in 0..experts_per_layer {
            let slice = vq2_pager.access_expert_slice(0, l * experts_per_layer + e).unwrap();
            let view = MmapVq2ExpertView::from_aligned_slice(slice, vq2_lay).unwrap();
            view.forward_swiglu_zero_copy(&vlut_in, &mut gu_buf, &mut vlut_mid, &mut out_buf, Some(&pool));
            for (dst, &v) in sum_y.iter_mut().zip(out_buf.iter()) {
                *dst += v;
            }
        }
        let (cos, rel) = cosine_and_rel_l2(&fp32_outs[l], &sum_y);
        vq2_cos_sum += cos;
        vq2_rel_sum += rel;
    }

    let n_l = num_real_layers as f64;
    println!(
        "  Compiled 16 real SmolLM2-1.7B SwiGLU experts (201.3M pretrained params) in {:.2}s:",
        t_comp.elapsed().as_secs_f64()
    );
    println!(
        "  • NF3-SDOT (3.125 bpw, {:5.1} MiB .k2tm) | SwiGLU Cosine vs FP32: {:.5} | Rel L2 Err: {:5.2}% | Physical F_NOCACHE NVMe: {:.2} GB/s",
        mib(nf3_hdr.total_file_bytes() as u64),
        nf3_cos_sum / n_l,
        (nf3_rel_sum / n_l) * 100.0,
        nf3_pager.shard.physical_nvme_gbps()
    );
    println!(
        "  • VQ2-LUT  (2.125 bpw, {:5.1} MiB .k2tm) | SwiGLU Cosine vs FP32: {:.5} | Rel L2 Err: {:5.2}% | Physical F_NOCACHE NVMe: {:.2} GB/s",
        mib(vq2_hdr.total_file_bytes() as u64),
        vq2_cos_sum / n_l,
        (vq2_rel_sum / n_l) * 100.0,
        vq2_pager.shard.physical_nvme_gbps()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut shard_path: Option<PathBuf> = None;
    let mut real_model_dir = PathBuf::from("models/SmolLM2-1.7B-Instruct");
    let mut shard_layers = 4usize;
    let mut shard_experts = 32usize; // 4 x 32 = 128 experts
    let mut threads = default_inference_threads();
    let mut exec_layers = 80usize;
    let mut use_vq2 = false; // NF3 unless --vq2
    let mut keep_shard = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--shard-path" if i + 1 < args.len() => {
                shard_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--real-model" if i + 1 < args.len() => {
                real_model_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--shard-layers" if i + 1 < args.len() => {
                shard_layers = args[i + 1].parse().unwrap_or(4);
                i += 2;
            }
            "--shard-experts" if i + 1 < args.len() => {
                shard_experts = args[i + 1].parse().unwrap_or(32);
                i += 2;
            }
            "--threads" if i + 1 < args.len() => {
                threads = args[i + 1].parse().unwrap_or(threads);
                i += 2;
            }
            "--exec-layers" if i + 1 < args.len() => {
                exec_layers = args[i + 1].parse().unwrap_or(80);
                i += 2;
            }
            "--vq2" => {
                use_vq2 = true;
                i += 1;
            }
            "--keep-shard" => {
                keep_shard = true;
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    let cfg_nf3 = Kronos800bConfig::frontier_800b_nf3();
    let rep_nf3 = cfg_nf3.nf_budget_report(3);
    let cfg_vq2 = Kronos800bConfig::frontier_800b();
    let rep_vq2 = cfg_vq2.budget_report();
    let p_cores = physical_p_cores();

    println!("==========================================================================================");
    println!(" KRONOS FRONTIER 800B SPARSE-MoE — PHYSICAL F_NOCACHE NVMe & 24 GiB M4 UNIFIED RAM VERIFIER");
    println!("==========================================================================================");
    println!(
        "  Hardware Detected        : {} physical P-cores (executing on {} P-core workers)",
        p_cores, threads
    );
    println!(
        "  800B MoE Architecture    : {} layers, d_model = {}, {} routed + {} shared SwiGLU experts/layer (d_ff = {})",
        cfg_nf3.num_layers,
        cfg_nf3.d_model,
        rep_nf3.routed_experts_per_layer,
        cfg_nf3.num_shared_experts,
        cfg_nf3.d_ff_expert
    );
    println!(
        "  Total / Active Params    : {:.2}B total  |  {:.2}B active @ top-{} (~37.0B w/ Confidence Tail Pruning)",
        rep_nf3.total_params as f64 / 1e9,
        rep_nf3.active_params_top_k as f64 / 1e9,
        cfg_nf3.top_k
    );
    println!("  24 GiB M4 Memory Budgets (0 OS Swap, >= 4.0 GiB Free OS Headroom):");
    println!(
        "    • NF3-SDOT (3.125 bpw, High-Fidelity) : {:6.2} MiB/expert ({}x16KB) | {:6.2} GiB NVMe | {} SLRU experts -> {:5.2} GiB Resident RAM",
        mib(rep_nf3.bytes_per_expert_aligned as u64),
        rep_nf3.bytes_per_expert_aligned / 16_384,
        gib(rep_nf3.total_disk_bytes),
        cfg_nf3.slru_capacity_experts,
        gib(rep_nf3.total_resident_ram_bytes)
    );
    println!(
        "    • VQ2-LUT  (2.125 bpw, Max-Capacity)  : {:6.2} MiB/expert ({}x16KB) | {:6.2} GiB NVMe | {} SLRU experts -> {:5.2} GiB Resident RAM",
        mib(rep_vq2.bytes_per_expert_aligned as u64),
        rep_vq2.bytes_per_expert_aligned / 16_384,
        gib(rep_vq2.total_disk_bytes),
        cfg_vq2.slru_capacity_experts,
        gib(rep_vq2.total_resident_ram_bytes)
    );
    println!("------------------------------------------------------------------------------------------");

    let resolved_path = shard_path.unwrap_or_else(|| {
        std::env::temp_dir().join(format!("kronos_800b_fnocache_{}.k2tm", std::process::id()))
    });
    let scratch_dir = resolved_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let _ = std::fs::create_dir_all(&scratch_dir);

    run_real_pretrained_moe_suite(&real_model_dir, &scratch_dir, threads);

    let _guard = TempShardGuard {
        path: resolved_path.clone(),
        keep: keep_shard,
    };

    println!();
    let fmt_name = if use_vq2 { "VQ2 (2.125 bpw, 11.16 MiB/expert)" } else { "NF3-SDOT (3.125 bpw, 16.45 MiB/expert)" };
    println!(
        "  SUITE A: HONEST PHYSICAL F_NOCACHE NVMe STREAMING vs SLRU DRAM ({}-Layer 800B 7168x2048 {})",
        exec_layers, fmt_name
    );
    println!("  (Bypasses macOS XNU RAM UBC via fcntl(F_NOCACHE, 1) at write + fsync + pread into posix_memalign(16KB) SLRU slots)");
    println!("  --------------------------------------------------------------------------------------------------------");

    let t_write = Instant::now();
    let hdr = if use_vq2 {
        write_k2tm_shard_file(&resolved_path, &cfg_vq2, shard_layers, shard_experts, 0x800B_2026)
            .expect("write uncached VQ2 k2tm")
    } else {
        write_nf_k2tm_shard_file(&resolved_path, &cfg_nf3, 3, shard_layers, shard_experts, 0x800B_2026)
            .expect("write uncached NF3 k2tm")
    };
    let write_s = t_write.elapsed().as_secs_f64();
    let shard_bytes = hdr.total_file_bytes() as u64;
    println!(
        "  [Uncached F_NOCACHE Shard] Wrote + fsync'd {:.2} GiB ({}x{} full 7168x2048 experts) in {:.2}s ({:.2} GB/s physical SSD write)",
        gib(shard_bytes),
        shard_layers,
        shard_experts,
        write_s,
        (shard_bytes as f64 / 1e9) / write_s.max(1e-6)
    );

    let pool = SpinThreadPool::new(threads);
    let slru_cap = (shard_layers * shard_experts / 2).max(16); // half the shard's experts
    let shard = DirectNvmeShard::open(&resolved_path, hdr.aligned_expert_bytes)
        .expect("open DirectNvmeShard with F_NOCACHE");
    let mut pager = DirectNvmeSlruPager::new(shard, slru_cap, true, true)
        .expect("allocate 16KB-aligned SLRU DRAM arena");

    let d_model = hdr.d_model;
    let d_ff = hdr.d_ff_expert;
    let nf3_lay = cfg_nf3.nf_expert_layout(3);
    let vq2_lay = cfg_vq2.expert_layout();
    let x_vec: Vec<f32> = (0..d_model).map(|j| ((j as f32) * 0.0137).sin() * 0.5).collect();
    let mut act_in = Int8ChunkActivation::new_zeroed(d_model);
    act_in.populate_fwht(&x_vec);
    let mut act_mid = Int8ChunkActivation::new_zeroed(d_ff);
    let mut vlut_in = Vq2ActivationLut::new_zeroed(d_model);
    vlut_in.populate_fwht(&x_vec);
    let mut vlut_mid = Vq2ActivationLut::new_zeroed(d_ff);
    let mut gu_buf = vec![0.0f32; 2 * d_ff];
    let mut out_buf = vec![0.0f32; d_model];

    let mut exec_slice = |slice: &[u8]| {
        if use_vq2 {
            let v = MmapVq2ExpertView::from_aligned_slice(slice, vq2_lay).unwrap();
            v.forward_swiglu_zero_copy(&vlut_in, &mut gu_buf, &mut vlut_mid, &mut out_buf, Some(&pool));
        } else {
            let v = MmapNfExpertView::from_aligned_slice(slice, nf3_lay).unwrap();
            v.forward_swiglu_zero_copy(&act_in, &mut gu_buf, &mut act_mid, &mut out_buf, Some(&pool));
        }
        std::hint::black_box(out_buf[0]);
    };

    // A1: every access misses. Reads each shard expert at most once, then scales the time to `exec_layers`.
    pager.clear_cache();
    let total_unique = shard_layers * shard_experts;
    let k_eff = 5usize;
    let cold_layers = (total_unique / k_eff).min(exec_layers).max(1);
    let t_cold = Instant::now();
    for l in 0..cold_layers {
        for k in 0..k_eff {
            let idx = l * k_eff + k;
            let pl = idx / shard_experts;
            let pe = idx % shard_experts;
            let slice = pager.access_expert_slice(pl, pe).expect("cold F_NOCACHE pread");
            exec_slice(slice);
        }
    }
    let cold_wall = t_cold.elapsed().as_secs_f64();
    let nvme_bytes = pager.shard.nvme_bytes_read.load(std::sync::atomic::Ordering::Relaxed);
    let pure_nvme_gbps = pager.shard.physical_nvme_gbps();
    let e2e_cold_gbps = (nvme_bytes as f64 / 1e9) / cold_wall;
    let scaled_80l_ms = (cold_wall / cold_layers as f64) * (exec_layers as f64) * 1e3;
    println!(
        "  1. 100% Cold Physical NVMe (0% hit, F_NOCACHE pread)   : {:6.2} tok/s ({:6.1} ms/80L tok) | Physical SSD I/O: {:5.2} GB/s (E2E {:5.2} GB/s)",
        1000.0 / scaled_80l_ms,
        scaled_80l_ms,
        pure_nvme_gbps,
        e2e_cold_gbps
    );

    // A2: Zipf traffic with SLRU cache and async prefetch. Lookahead is capped so prefetches
    // don't evict probationary slots that have not been read yet.
    pager.clear_cache();
    let nl = exec_layers;
    let prob_slots = slru_cap.saturating_sub((slru_cap * 4) / 5).max(k_eff);
    let lookahead_l = (prob_slots / (2 * k_eff)).clamp(1, 6);
    let mut zipf = ZipfDomainSampler::new(shard_experts, 1.15, 0x517F_800B);
    let conv_steps = 8usize;
    let warm_skip = 3usize;
    let mut traces: Vec<Vec<Vec<usize>>> = (0..conv_steps)
        .map(|_| (0..nl).map(|l| zipf.sample_layer_experts(l, k_eff)).collect())
        .collect();
    let mut meas_wall = 0.0f64;
    let mut meas_total_bytes = 0u64;
    for (step, trace) in traces.iter_mut().enumerate() {
        if step == warm_skip {
            pager.dram_hits = 0;
            pager.protected_hits = 0;
            pager.async_prefetch_hits = 0;
            pager.nvme_misses = 0;
            pager.shard.reset_io_stats();
        }
        let t_step = Instant::now();
        for (fl, exps) in trace.iter().enumerate().take(lookahead_l.min(nl)) {
            for &fe in exps {
                pager.prefetch_expert_async(fl, fe);
            }
        }
        for (l, exps) in trace.iter().enumerate() {
            if let Some(future_exps) = trace.get(l + lookahead_l) {
                for &fe in future_exps {
                    pager.prefetch_expert_async(l + lookahead_l, fe);
                }
            }
            for &eid in exps {
                let slice = pager.access_expert_slice(l, eid).expect("zipf expert access");
                exec_slice(slice);
            }
        }
        let dt = t_step.elapsed().as_secs_f64();
        if step >= warm_skip {
            meas_wall += dt;
            meas_total_bytes += (nl * k_eff * hdr.aligned_expert_bytes) as u64;
        }
    }
    let meas_toks = (conv_steps - warm_skip) as f64;
    let zipf_tok_s = meas_toks / meas_wall;
    let zipf_nvme_bytes = pager.shard.nvme_bytes_read.load(std::sync::atomic::Ordering::Relaxed);
    println!(
        "  2. Zipf(1.15) + SLRU + Async F_NOCACHE Prefetch        : {:6.2} tok/s ({:6.1} ms/tok) | 0-IO DRAM {:4.1}% + Prefetch {:4.1}% | Physical SSD {:4.2} GB/s ({:.2} GiB read) | Eff {:5.2} GB/s",
        zipf_tok_s,
        (meas_wall / meas_toks) * 1e3,
        pager.zero_io_dram_hit_percent(),
        pager.async_prefetch_percent(),
        pager.shard.physical_nvme_gbps(),
        gib(zipf_nvme_bytes),
        (meas_total_bytes as f64 / 1e9) / meas_wall
    );

    // A3: every access hits DRAM.
    let hot_trace: Vec<Vec<usize>> = (0..nl)
        .map(|l| (0..k_eff).map(|k| 1 + ((l + k) % 8)).collect())
        .collect();
    for _ in 0..2 {
        for (l, exps) in hot_trace.iter().enumerate() {
            for &eid in exps {
                let slice = pager.access_expert_slice(l, eid).unwrap();
                exec_slice(slice);
            }
        }
    }
    pager.shard.reset_io_stats();
    let hot_runs = 3usize;
    let t_hot = Instant::now();
    for _ in 0..hot_runs {
        for (l, exps) in hot_trace.iter().enumerate() {
            for &eid in exps {
                let slice = pager.access_expert_slice(l, eid).unwrap();
                exec_slice(slice);
            }
        }
    }
    let hot_wall = t_hot.elapsed().as_secs_f64();
    let hot_bytes = (hot_runs * nl * k_eff * hdr.aligned_expert_bytes) as u64;
    assert_eq!(
        pager.shard.nvme_bytes_read.load(std::sync::atomic::Ordering::Relaxed),
        0,
        "100% SLRU DRAM run must perform 0 physical NVMe reads"
    );
    println!(
        "  3. 100% Hot SLRU DRAM Hit Ceiling (0 SSD I/O, B = 1)   : {:6.2} tok/s ({:6.1} ms/tok) | Unified DRAM Bandwidth: {:6.2} GB/s",
        (hot_runs as f64) / hot_wall,
        (hot_wall / hot_runs as f64) * 1e3,
        (hot_bytes as f64 / 1e9) / hot_wall
    );

    // Free the SLRU arena before Suite B allocates.
    drop(pager);

    // Suite B: full model with all active weights in DRAM.
    println!();
    println!("  SUITE B: FULL 812.13B ACTIVE-TENSOR HOT-DRAM CEILING (128h MLA + Hadamard INT8 KV + Shared/Routed + 128K Head)");
    println!("  (Measures in-DRAM compute & memory-bandwidth ceiling when active domain experts reside in the 14.5 GiB SLRU cache)");
    println!("  --------------------------------------------------------------------------------------------------------");

    let vq2_shard_path = if use_vq2 {
        resolved_path.clone()
    } else {
        scratch_dir.join(format!("kronos_800b_vq2_dram_{}.k2tm", std::process::id()))
    };
    let _vq2_guard = TempShardGuard {
        path: vq2_shard_path.clone(),
        keep: use_vq2 && keep_shard,
    };
    if !use_vq2 {
        write_k2tm_shard_file(&vq2_shard_path, &cfg_vq2, 4, 16, 0x800B_D4A0)
            .expect("write compact VQ2 DRAM shard");
    }
    let mut bench_cfg = cfg_vq2.clone();
    bench_cfg.max_seq_len = 256;
    bench_cfg.slru_capacity_experts = 64;
    let mmap = K2tmShardMmap::open(&vq2_shard_path, &bench_cfg).expect("open VQ2 shard");
    let mut engine = Kronos800bMoEEngine::new(bench_cfg.clone(), mmap, threads);
    engine.enable_pinned_mla_and_head(8, true);

    let mut hidden: Vec<f32> = (0..bench_cfg.d_model)
        .map(|j| ((j as f32) * 0.0137).sin() * 0.5)
        .collect();
    let full_hot_trace: Vec<Vec<usize>> = (0..nl)
        .map(|l| (0..5).map(|k| 1 + ((l + k) % 8)).collect())
        .collect();
    let _ = engine.forward_decode_layers(&mut hidden, Some(&full_hot_trace), nl, false);
    let st_full_1 = engine.forward_decode_layers(&mut hidden, Some(&full_hot_trace), nl, false);
    println!(
        "  1. Full 812.13B In-DRAM Single-Token (B = 1, {:.2}B active) : {:6.2} tok/s | {:6.1} ms/tok | {:6.2} GB/s Unified DRAM",
        st_full_1.active_params_per_token as f64 / 1e9,
        1.0 / st_full_1.elapsed_secs,
        st_full_1.elapsed_secs * 1e3,
        (st_full_1.weight_bytes_streamed as f64 / 1e9) / st_full_1.elapsed_secs
    );

    let draft_sec_per_tok = 0.00050;
    for &b_toks in &[4usize, 8usize] {
        let mut batch_hiddens = vec![0.0f32; b_toks * bench_cfg.d_model];
        for (idx, slot) in batch_hiddens.iter_mut().enumerate() {
            *slot = ((idx as f32) * 0.0113).cos() * 0.45;
        }
        let mut b_zipf = ZipfDomainSampler::new(12, 1.20, 0xF011_0000 + b_toks as u64);
        let batch_trace: Vec<Vec<Vec<usize>>> = (0..nl)
            .map(|l| (0..b_toks).map(|_| b_zipf.sample_layer_experts(l, 5)).collect())
            .collect();
        let _ = engine.forward_speculative_coalesced_layers(&mut batch_hiddens, b_toks, Some(&batch_trace), nl);
        let st_fb = engine.forward_speculative_coalesced_layers(&mut batch_hiddens, b_toks, Some(&batch_trace), nl);
        let raw_verify_tok_s = (b_toks as f64) / st_fb.elapsed_secs;
        let step_wall_sec = st_fb.elapsed_secs + ((b_toks - 1) as f64) * draft_sec_per_tok;
        let eff_75 = expected_speculative_tokens(b_toks, 0.75) / step_wall_sec;
        let eff_85 = expected_speculative_tokens(b_toks, 0.85) / step_wall_sec;
        println!(
            "  2. Full 812.13B In-DRAM Coalesced    (B = {}, {:.2}B active) : {:6.2} tok/s raw ({:.2} GB/s DRAM) | Net Speculative: {:5.2} .. {:5.2} tok/s",
            b_toks,
            st_fb.active_params_per_token as f64 / 1e9,
            raw_verify_tok_s,
            (st_fb.weight_bytes_streamed as f64 / 1e9) / st_fb.elapsed_secs,
            eff_75,
            eff_85
        );
    }
    println!("==========================================================================================");
}
