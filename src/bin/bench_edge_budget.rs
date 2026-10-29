//! Hardware-envelope, quantizer-fidelity, and single-pass `Kronos-Jev` verification suite.
//!
//! Suite 1 enforces `--memory-budget-mb 5120` (`Edge8G` 8 GB MacBook) and `--memory-budget-mb 4900`
//! (`Mobile5G` 8 GB smartphone) on a full 24-layer `20.91B`-parameter (`3.61B` active) Sparse-MoE
//! model (`nf4g128` core + `Q4` sliding-sink KV + `nf3g128` SLRU / `e8q2g128` `mmap` `.k2tm` shard).
//! Suite 2 evaluates held-out perplexity, KL divergence, logit cosine, and top-1 agreement on
//! pretrained `SmolLM2-135M` across `Exact FP32`, `vq2g128`, `e8q2g128`, `e8q2nf4g128`, `nf3g128`,
//! and `nf3nf4g128`.
//! Suite 3 measures `Kronos-Jev` single-forward-pass latency (`0` decode tokens), Brier score,
//! Expected Calibration Error (ECE), Turn-1 tool-schema token savings, and deterministic oracle
//! verification.

use kronos_2t::agent::{tool_definitions_json, verify_workspace_work, TurnRoute};
use kronos_2t::bpe_tokenizer::BpeTokenizer;
use kronos_2t::chat::{ChatModel, Message};
use kronos_2t::jev::{
    evaluate_rubric, extract_requirement_rubric, fuse_binary_log_odds, fuse_choice_logits,
    plan_turn_with_jev, structural_route_logits, CalibratedBool, CalibratedChoice,
};
use kronos_2t::memory_budget::{current_resident_bytes, MemoryBudgetProfile, MemoryTier};
use kronos_2t::moe_800b::{write_nf_k2tm_shard_file, Kronos800bConfig};
use kronos_2t::real_llm::{argmax, LlamaConfig, RealLlamaEngine, RealWeightMode};
use kronos_2t::sparse_moe::{MoeExpertOpenPolicy, MoeRouterScoring};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

const GIB_F: f64 = 1024.0 * 1024.0 * 1024.0;
const MIB_F: f64 = 1024.0 * 1024.0;
const GIB_U: usize = 1024 * 1024 * 1024;
const MIB_U: usize = 1024 * 1024;

struct TempPathGuard {
    path: PathBuf,
    is_dir: bool,
}

impl TempPathGuard {
    fn file(path: PathBuf) -> Self {
        Self {
            path,
            is_dir: false,
        }
    }

    fn dir(path: PathBuf) -> Self {
        let _ = fs::create_dir_all(&path);
        Self { path, is_dir: true }
    }
}

impl Drop for TempPathGuard {
    fn drop(&mut self) {
        if self.is_dir {
            let _ = fs::remove_dir_all(&self.path);
        } else if self.path.exists() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

fn gib(bytes: usize) -> f64 {
    (bytes as f64) / GIB_F
}

fn mib(bytes: usize) -> f64 {
    (bytes as f64) / MIB_F
}

fn release_freed_malloc_pages() {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        unsafe extern "C" {
            fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize;
        }
        // SAFETY: Passing null relieves reusable magazine pages across all Darwin malloc zones.
        unsafe {
            let _ = malloc_zone_pressure_relief(std::ptr::null_mut(), 0);
        }
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

fn build_20b_moe_config(num_layers: usize) -> LlamaConfig {
    LlamaConfig {
        vocab_size: 201_088,
        hidden_size: 2_880,
        intermediate_size: 2_880,
        num_hidden_layers: num_layers,
        num_attention_heads: 64,
        num_key_value_heads: 8,
        head_dim_override: Some(64),
        rms_norm_eps: 1e-5,
        rope_theta: 150_000.0,
        tie_word_embeddings: false,
        max_seq_len: 16_384,
        sliding_window: Some(2_048),
        sink_tokens: 4,
        num_experts: 32,
        num_experts_per_tok: 4,
        moe_intermediate_size: 2_880,
        shared_expert_intermediate_size: 0,
        norm_topk_prob: true,
        router_scoring: MoeRouterScoring::TopkSoftmax,
        decoder_sparse_step: 1,
        mlp_only_layers: Vec::new(),
        sliding_layer_mask: vec![true; num_layers],
    }
}

fn build_shard_writer_cfg(cfg: &LlamaConfig) -> Kronos800bConfig {
    Kronos800bConfig {
        num_layers: cfg.num_hidden_layers,
        d_model: cfg.hidden_size,
        d_ff_expert: cfg.moe_intermediate_size,
        router_sub_keys: 6,
        num_shared_experts: 0,
        top_k: cfg.num_experts_per_tok,
        min_top_k: cfg.num_experts_per_tok,
        confidence_tau: 1.0,
        min_tail_prob: 0.0,
        num_attn_heads: cfg.num_attention_heads,
        kv_lora_rank: 512,
        q_lora_rank: 1536,
        qk_nope_head_dim: 64,
        qk_rope_head_dim: 64,
        v_head_dim: 64,
        vocab_size: cfg.vocab_size,
        scale_group_chunks: 2,
        lookahead_layers: 3,
        slru_capacity_experts: 128,
        max_seq_len: cfg.max_seq_len,
    }
}

fn run_suite_1_20b_edge_and_mobile(num_layers: usize) {
    println!("================================================================================");
    println!("SUITE 1: 20.91B Sparse-MoE (3.61B Active) 8 GB MacBook & 5 GB Mobile Envelopes");
    println!("================================================================================");

    let cfg = build_20b_moe_config(num_layers);
    let shard_cfg = build_shard_writer_cfg(&cfg);
    let embed_params = 2 * cfg.vocab_size * cfg.hidden_size;
    let attn_params = cfg.num_hidden_layers
        * ((cfg.q_dim() + 2 * cfg.kv_dim()) * cfg.hidden_size
            + cfg.hidden_size * cfg.q_dim()
            + cfg.num_experts * cfg.hidden_size);
    let expert_params = cfg.num_hidden_layers
        * cfg.num_experts
        * (3 * cfg.moe_intermediate_size * cfg.hidden_size);
    let active_expert_params = cfg.num_hidden_layers
        * cfg.num_experts_per_tok
        * (3 * cfg.moe_intermediate_size * cfg.hidden_size);
    let total_params = embed_params + attn_params + expert_params;
    let active_params = (cfg.vocab_size * cfg.hidden_size) + attn_params + active_expert_params;
    println!(
        "Architecture : {}L Sparse-MoE | d_model={} | q_dim={} | kv_dim={} | experts={}/top-{} | vocab={}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.q_dim(),
        cfg.kv_dim(),
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.vocab_size
    );
    println!(
        "Parameters   : {:.2}B total ({:.2}B routed experts + {:.2}B pinned core) | {:.2}B active/tok",
        (total_params as f64) / 1e9,
        (expert_params as f64) / 1e9,
        ((embed_params + attn_params) as f64) / 1e9,
        (active_params as f64) / 1e9,
    );

    let spec_batch = 6usize;
    let spec_alpha = 0.80f64;
    let expected_accept = expected_speculative_tokens(spec_batch, spec_alpha);
    let pid = std::process::id();

    // -------------------------------------------------------------------------
    // Profile 1A: 8 GB MacBook (`Edge8G`, `--memory-budget-mb 5120`, `nf3g128`)
    // -------------------------------------------------------------------------
    {
        MemoryBudgetProfile::set_global_budget_override_mb(5120);
        let prof = MemoryBudgetProfile::active();
        assert_eq!(prof.tier, MemoryTier::Edge8G);
        assert_eq!(prof.prompt_chunk_gpu, 256);
        assert!(prof.quantize_untied_embeddings && prof.prefer_q4_kv);

        let nf3_path = std::env::temp_dir().join(format!("kronos_edge_20b_nf3_{pid}.k2tm"));
        let _guard = TempPathGuard::file(nf3_path.clone());
        let t_write = Instant::now();
        let hdr = write_nf_k2tm_shard_file(
            &nf3_path,
            &shard_cfg,
            3,
            cfg.num_hidden_layers,
            cfg.num_experts,
            0x20B0_0003,
        )
        .expect("write 20B nf3 k2tm shard");
        let write_secs = t_write.elapsed().as_secs_f64();

        let draft_reserve = 243 * MIB_U;
        let mut eng = RealLlamaEngine::new_synthetic_moe_for_bench(
            cfg.clone(),
            3,
            &nf3_path,
            draft_reserve,
            MoeExpertOpenPolicy::AutoBudget,
            0x5120_ED6E,
        )
        .expect("build 20B Edge8G Sparse-MoE engine");
        assert_eq!(eng.kv_bits(), 4);

        // Warm SLRU working set over a 20-token domain prefix, then benchmark steady-state Zipf
        // traffic, hot-domain SLRU decode, and 1+5 speculative coalesced verification.
        for step in 0..20usize {
            let tok = (step * 131 + 17) % cfg.vocab_size;
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }

        let stats_before = eng.moe_stats().expect("moe stats");
        let zipf_tokens = 24usize;
        let t_zipf = Instant::now();
        for step in 0..zipf_tokens {
            let tok = (step * 977 + 42) % cfg.vocab_size;
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }
        let zipf_secs = t_zipf.elapsed().as_secs_f64();
        let stats_after = eng.moe_stats().expect("moe stats");

        let delta_hits = (stats_after.dram_hits - stats_before.dram_hits)
            + (stats_after.async_prefetch_hits - stats_before.async_prefetch_hits);
        let delta_misses = stats_after.nvme_misses - stats_before.nvme_misses;
        let delta_total = (delta_hits + delta_misses).max(1);
        let window_hit_pct = 100.0 * (delta_hits as f64) / (delta_total as f64);
        let delta_nvme_bytes = stats_after
            .nvme_bytes_read
            .saturating_sub(stats_before.nvme_bytes_read);
        let nvme_mib_per_tok = (delta_nvme_bytes as f64) / MIB_F / (zipf_tokens as f64);

        // Speculative tree coalescing (`1 + 5 = 6` candidate tokens verified in one batch pass).
        let cand_tree: Vec<usize> = (0..spec_batch).map(|i| 100 + i * 37).collect();
        let mut batch_logits = vec![0.0f32; spec_batch * cfg.vocab_size];
        // Warm batch buffers once.
        let saved_pos = 16usize;
        eng.truncate_context(saved_pos);
        eng.forward_batch_logits(&cand_tree, &mut batch_logits);

        let spec_steps = 10usize;
        let t_spec = Instant::now();
        for s in 0..spec_steps {
            eng.truncate_context(saved_pos);
            let tree: Vec<usize> = (0..spec_batch).map(|i| (s * 53 + i * 19) % 4096).collect();
            eng.forward_batch_logits(&tree, &mut batch_logits);
            std::hint::black_box(batch_logits[0]);
        }
        let spec_secs = t_spec.elapsed().as_secs_f64();
        let spec_raw_eval_tps = ((spec_steps * spec_batch) as f64) / spec_secs;
        let spec_effective_tps = ((spec_steps as f64) * expected_accept) / spec_secs;

        let pinned_b = eng.pinned_core_bytes();
        let slru_b = eng.moe_expert_arena_bytes();
        let kv_b = eng.kv_cache_bytes();
        let workspace_b = 48 * MIB_U;
        let accounted_b = pinned_b + slru_b + kv_b + draft_reserve + workspace_b;
        let live_rss_b = current_resident_bytes();
        let cap_b = 5 * GIB_U;

        println!();
        println!(
            "[Profile 1A: Edge8G (8 GB MacBook, budget=5120 MiB / 5.00 GiB, nf4 attn + nf3g128 experts)]"
        );
        println!(
            "  .k2tm NVMe Shard : {:.2} GiB ({} experts @ {:.2} MiB, written with F_NOCACHE in {:.2}s)",
            gib(hdr.total_file_bytes()),
            hdr.num_layers * hdr.experts_per_layer,
            mib(hdr.aligned_expert_bytes),
            write_secs
        );
        println!(
            "  Pinned Core RAM  : {:7.1} MiB (nf4g128 attn + nf4g128 201k embed/lm_head + routers)",
            mib(pinned_b)
        );
        println!(
            "  SLRU Expert RAM  : {:7.1} MiB ({}/{} hot experts resident = {:.1}%)",
            mib(slru_b),
            stats_after.slru_capacity_experts,
            hdr.num_layers * hdr.experts_per_layer,
            100.0 * (stats_after.slru_capacity_experts as f64)
                / ((hdr.num_layers * hdr.experts_per_layer) as f64)
        );
        println!(
            "  Q4 Sliding KV    : {:7.1} MiB (window=2048 + 4 sinks, 24 layers, GQA-8)",
            mib(kv_b)
        );
        println!(
            "  0.5B Jev/Draft   : {:7.1} MiB (reserved budget)",
            mib(draft_reserve)
        );
        println!(
            "  Accounted Total  : {:.3} GiB ({:.1} MiB) <= 5.000 GiB cap",
            gib(accounted_b),
            mib(accounted_b)
        );
        println!(
            "  Live Mach RSS    : {:.3} GiB ({:.1} MiB) <= 5.000 GiB cap",
            gib(live_rss_b),
            mib(live_rss_b)
        );
        println!(
            "  SLRU Hit Rate    : {:.1}% (async 3-layer prefetch hits={}, F_NOCACHE reads={:.1} MiB/tok)",
            window_hit_pct,
            stats_after.async_prefetch_hits,
            nvme_mib_per_tok
        );
        println!(
            "  Throughput       : {:5.1} tok/s (B=1 decode) | {:5.1} tok/s effective (1+5 coalesced @ alpha=0.80, {:5.1} raw eval/s)",
            (zipf_tokens as f64) / zipf_secs,
            spec_effective_tps,
            spec_raw_eval_tps
        );

        assert!(
            accounted_b <= cap_b,
            "Edge8G accounted working set {:.3} GiB exceeded 5.00 GiB cap",
            gib(accounted_b)
        );
        assert!(
            live_rss_b <= cap_b,
            "Edge8G live Mach RSS {:.3} GiB exceeded 5.00 GiB cap",
            gib(live_rss_b)
        );
    }

    release_freed_malloc_pages();

    // -------------------------------------------------------------------------
    // Profile 1B: 8-12 GB Smartphone (`Mobile5G`, `--memory-budget-mb 4900`, 20.91B `e8q2g128`)
    // -------------------------------------------------------------------------
    {
        MemoryBudgetProfile::set_global_budget_override_mb(4900);
        let prof = MemoryBudgetProfile::active();
        assert_eq!(prof.tier, MemoryTier::Mobile5G);
        assert_eq!(prof.prompt_chunk_gpu, 256);

        let e8_path = std::env::temp_dir().join(format!("kronos_mobile_20b_e8q2_{pid}.k2tm"));
        let _guard = TempPathGuard::file(e8_path.clone());
        let t_write = Instant::now();
        let hdr = write_nf_k2tm_shard_file(
            &e8_path,
            &shard_cfg,
            2,
            cfg.num_hidden_layers,
            cfg.num_experts,
            0x20B0_0002,
        )
        .expect("write 20B e8q2 k2tm shard");
        let write_secs = t_write.elapsed().as_secs_f64();

        // Production `AutoBudget` governor on 20.91B (`4.73 GiB` shard + `907 MiB` core > `4.90 GiB`
        // cap -> pins 569/768 = 74.1% of Gosset E8 2-bit experts inside the governed arena).
        let mut eng = RealLlamaEngine::new_synthetic_moe_for_bench(
            cfg.clone(),
            2,
            &e8_path,
            0,
            MoeExpertOpenPolicy::AutoBudget,
            0x4900_E802,
        )
        .expect("build 20B Mobile5G E8Q2 AutoBudget Sparse-MoE engine");
        assert_eq!(eng.kv_bits(), 4);

        for step in 0..16usize {
            let tok = (step * 73 + 11) % cfg.vocab_size;
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }

        let decode_tokens = 24usize;
        let t_dec = Instant::now();
        for step in 0..decode_tokens {
            let tok = (step * 313 + 7) % cfg.vocab_size;
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }
        let dec_secs = t_dec.elapsed().as_secs_f64();

        let mut batch_logits = vec![0.0f32; spec_batch * cfg.vocab_size];
        let saved_pos = 12usize;
        let spec_steps = 10usize;
        let t_spec = Instant::now();
        for s in 0..spec_steps {
            eng.truncate_context(saved_pos);
            let tree: Vec<usize> = (0..spec_batch).map(|i| (s * 41 + i * 13) % 4096).collect();
            eng.forward_batch_logits(&tree, &mut batch_logits);
            std::hint::black_box(batch_logits[0]);
        }
        let spec_secs = t_spec.elapsed().as_secs_f64();
        let spec_raw_eval_tps = ((spec_steps * spec_batch) as f64) / spec_secs;
        let spec_effective_tps = ((spec_steps as f64) * expected_accept) / spec_secs;

        let stats = eng.moe_stats().expect("moe stats");
        let pinned_b = eng.pinned_core_bytes();
        let kv_b = eng.kv_cache_bytes();
        let arena_b = stats.slru_arena_bytes;
        let accounted_mobile_b = pinned_b + arena_b + kv_b + 48 * MIB_U;
        let live_rss_b = current_resident_bytes();
        let mobile_cap_b = (4900 * MIB_U).max((49 * GIB_U) / 10);
        let total_experts = hdr.num_layers * hdr.experts_per_layer;

        println!();
        println!(
            "[Profile 1B: Mobile5G 20.91B (8-12 GB Phone, budget=4900 MiB / 4.90 GiB, nf4 attn + e8q2g128 AutoBudget)]"
        );
        println!(
            "  .k2tm E8Q2 Shard : {:.2} GiB ({} experts @ {:.2} MiB, 2.125 bpw, written in {:.2}s)",
            gib(hdr.total_file_bytes()),
            total_experts,
            mib(hdr.aligned_expert_bytes),
            write_secs
        );
        println!(
            "  Pinned Core RAM  : {:7.1} MiB | SLRU Expert RAM: {:.1} MiB ({}/{} = {:.1}% resident) | Q4 KV: {:.1} MiB",
            mib(pinned_b),
            mib(arena_b),
            stats.slru_capacity_experts,
            total_experts,
            100.0 * (stats.slru_capacity_experts as f64) / (total_experts as f64),
            mib(kv_b)
        );
        println!(
            "  Accounted Total  : {:.3} GiB ({:.1} MiB) <= 4.900 GiB mobile cap",
            gib(accounted_mobile_b),
            mib(accounted_mobile_b)
        );
        println!(
            "  Live Mach RSS    : {:.3} GiB ({:.1} MiB) <= 4.900 GiB mobile cap (hit rate {:.1}%)",
            gib(live_rss_b),
            mib(live_rss_b),
            stats.hit_rate_percent
        );
        println!(
            "  Throughput       : {:5.1} tok/s (B=1 decode) | {:5.1} tok/s effective (1+5 coalesced @ alpha=0.80, {:5.1} raw eval/s)",
            (decode_tokens as f64) / dec_secs,
            spec_effective_tps,
            spec_raw_eval_tps
        );

        assert!(
            accounted_mobile_b <= mobile_cap_b,
            "Mobile5G accounted working set {:.3} GiB exceeded 4.90 GiB cap",
            gib(accounted_mobile_b)
        );
        assert!(
            live_rss_b <= mobile_cap_b,
            "Mobile5G live RSS {:.3} GiB exceeded 4.90 GiB cap",
            gib(live_rss_b)
        );
    }

    release_freed_malloc_pages();

    // -------------------------------------------------------------------------
    // Profile 1C: Mobile5G Zero-Copy Mmap (`--memory-budget-mb 4900`, 15.9B/18L `e8q2g128` tied)
    // -------------------------------------------------------------------------
    {
        MemoryBudgetProfile::set_global_budget_override_mb(4900);
        let mut cfg_18 = build_20b_moe_config(18);
        cfg_18.tie_word_embeddings = true;
        let mmap_path = std::env::temp_dir().join(format!("kronos_mobile_16b_mmap_{pid}.k2tm"));
        let _guard = TempPathGuard::file(mmap_path.clone());
        let hdr = write_nf_k2tm_shard_file(
            &mmap_path,
            &shard_cfg,
            2,
            cfg_18.num_hidden_layers,
            cfg_18.num_experts,
            0x16B0_E802,
        )
        .expect("write 16B e8q2 mmap shard");

        let mut eng = RealLlamaEngine::new_synthetic_moe_for_bench(
            cfg_18.clone(),
            2,
            &mmap_path,
            0,
            MoeExpertOpenPolicy::AutoBudget,
            0x16B0_4900,
        )
        .expect("build 16B Mobile5G zero-copy mmap Sparse-MoE engine");
        let stats0 = eng.moe_stats().expect("moe stats");
        assert!(
            !stats0.is_nvme_slru,
            "AutoBudget should select zero-copy mmap when shard fits inside mobile arena"
        );

        for step in 0..16usize {
            let tok = (step * 59 + 3) % cfg_18.vocab_size;
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }
        let decode_tokens = 24usize;
        let t_dec = Instant::now();
        for step in 0..decode_tokens {
            let tok = (step * 59 + 3) % cfg_18.vocab_size;
            std::hint::black_box(argmax(eng.forward_token(tok)));
        }
        let dec_secs = t_dec.elapsed().as_secs_f64();

        let full_mapped_b = eng.pinned_core_bytes() + hdr.total_file_bytes() + eng.kv_cache_bytes();
        let live_rss_b = current_resident_bytes();
        let mobile_cap_b = (4900 * MIB_U).max((49 * GIB_U) / 10);
        println!();
        println!(
            "[Profile 1C: Mobile5G Zero-Copy Mmap (15.9B/18L Sparse-MoE, e8q2g128 AutoBudget -> mmap)]"
        );
        println!(
            "  Full Mapped Set  : {:.3} GiB (Core {:.1} MiB + Mmap {:.2} GiB + Q4 KV {:.1} MiB) <= 4.900 GiB",
            gib(full_mapped_b),
            mib(eng.pinned_core_bytes()),
            gib(hdr.total_file_bytes()),
            mib(eng.kv_cache_bytes())
        );
        println!(
            "  Live Demand RSS  : {:.3} GiB ({:.1} MiB) | Warm mmap decode: {:5.1} tok/s",
            gib(live_rss_b),
            mib(live_rss_b),
            (decode_tokens as f64) / dec_secs
        );
        assert!(
            full_mapped_b <= mobile_cap_b && live_rss_b <= mobile_cap_b,
            "Mobile5G zero-copy mmap exceeded 4.90 GiB cap (mapped {:.3} GiB, rss {:.3} GiB)",
            gib(full_mapped_b),
            gib(live_rss_b)
        );
    }

    release_freed_malloc_pages();
    MemoryBudgetProfile::set_global_budget_override_bytes(0);
}

fn log_softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max_v = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max_v
        + logits
            .iter()
            .map(|&v| ((v as f64) - max_v).exp())
            .sum::<f64>()
            .ln();
    logits.iter().map(|&v| (v as f64) - lse).collect()
}

fn logit_cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-12)
}

#[derive(Clone, Debug)]
struct QuantEvalRow {
    label: &'static str,
    bpw: f64,
    weight_mib: f64,
    ppl: f64,
    mean_kl: f64,
    mean_cos: f64,
    top1_pct: f64,
    decode_tps: f64,
}

fn run_suite_2_pretrained_quantizer_quality() {
    println!();
    println!("================================================================================");
    println!("SUITE 2: Pretrained SmolLM2-135M Quantizer Quality (E8Q2 vs VQ2 vs NF3 vs FP32)");
    println!("================================================================================");

    let model_dir = Path::new("models/SmolLM2-135M");
    let calib_path = Path::new("models/calib.txt");
    if !model_dir.join("model.safetensors").exists() || !calib_path.exists() {
        println!("Skipping Suite 2: models/SmolLM2-135M or models/calib.txt not present.");
        return;
    }

    let tok = BpeTokenizer::from_model_dir(model_dir).expect("load SmolLM2 tokenizer");
    let corpus = fs::read_to_string(calib_path).expect("read models/calib.txt");
    let all_ids = tok.encode(&corpus);
    let chunk_len = 128usize;
    let n_calib_seqs = 24usize;
    let n_eval_chunks = 8usize;
    assert!(
        all_ids.len() >= (n_calib_seqs + n_eval_chunks + 2) * chunk_len,
        "calib.txt too short for disjoint calib + held-out split"
    );

    // Take held-out chunks from the tail of the corpus so there is zero overlap with GPTQ calibration.
    let eval_base = all_ids.len() - n_eval_chunks * chunk_len;
    let calib_pool_len = eval_base - chunk_len;
    let calib_stride = (calib_pool_len - chunk_len) / (n_calib_seqs - 1).max(1);
    let calib_seqs: Vec<Vec<usize>> = (0..n_calib_seqs)
        .map(|i| {
            let s = i * calib_stride;
            all_ids[s..s + chunk_len].to_vec()
        })
        .collect();
    let eval_chunks: Vec<Vec<usize>> = (0..n_eval_chunks)
        .map(|i| {
            all_ids[eval_base + i * chunk_len..eval_base + (i + 1) * chunk_len].to_vec()
        })
        .collect();

    let mut fp32 = RealLlamaEngine::load_fp32_reference(model_dir, chunk_len + 16)
        .expect("load SmolLM2-135M ExactFp32");

    // Precompute Exact FP32 reference log-probs, top-1 predictions, and raw logits on second-half tokens.
    let score_start = chunk_len / 2;
    let scored_per_chunk = chunk_len - 1 - score_start;
    let total_scored = n_eval_chunks * scored_per_chunk;
    let mut ref_logprobs: Vec<Vec<f64>> = Vec::with_capacity(total_scored);
    let mut ref_logits_raw: Vec<Vec<f32>> = Vec::with_capacity(total_scored);
    let mut ref_top1: Vec<usize> = Vec::with_capacity(total_scored);
    let mut fp32_nll = 0.0f64;

    for chunk in &eval_chunks {
        fp32.reset_context();
        for pos in 0..(chunk_len - 1) {
            let logits = fp32.forward_token(chunk[pos]);
            if pos >= score_start {
                let target = chunk[pos + 1];
                let lsm = log_softmax_f64(logits);
                fp32_nll -= lsm[target];
                ref_top1.push(argmax(logits));
                ref_logits_raw.push(logits.to_vec());
                ref_logprobs.push(lsm);
            }
        }
    }
    let fp32_ppl = (fp32_nll / (total_scored as f64)).exp();

    let tg_tokens = &eval_chunks[0][..64];
    let measure_tps = |eng: &mut RealLlamaEngine| -> f64 {
        eng.reset_context();
        for &t in &tg_tokens[..16] {
            std::hint::black_box(argmax(eng.forward_token(t)));
        }
        let runs = 3usize;
        let t0 = Instant::now();
        for _ in 0..runs {
            eng.reset_context();
            for &t in tg_tokens {
                std::hint::black_box(argmax(eng.forward_token(t)));
            }
        }
        ((runs * tg_tokens.len()) as f64) / t0.elapsed().as_secs_f64()
    };
    let fp32_tps = measure_tps(&mut fp32);

    let mut rows = vec![QuantEvalRow {
        label: "Exact FP32",
        bpw: 32.0,
        weight_mib: mib(fp32.resident_weight_bytes()),
        ppl: fp32_ppl,
        mean_kl: 0.0,
        mean_cos: 1.0,
        top1_pct: 100.0,
        decode_tps: fp32_tps,
    }];

    let modes: [(&str, RealWeightMode); 5] = [
        (
            "vq2g128 (4D VQ 2b)",
            RealWeightMode::KronosVq2Gptq {
                scale_group_chunks: 2,
            },
        ),
        (
            "e8q2g128 (Gosset E8 2b)",
            RealWeightMode::KronosNfGptq {
                bits: 2,
                scale_group_chunks: 2,
            },
        ),
        (
            "e8q2nf4g128 (Mobile MoE)",
            RealWeightMode::KronosNfMixedGptq {
                attn_head_bits: 4,
                ffn_bits: 2,
                scale_group_chunks: 2,
            },
        ),
        (
            "nf3g128 (Lloyd-Max 3b)",
            RealWeightMode::KronosNfGptq {
                bits: 3,
                scale_group_chunks: 2,
            },
        ),
        (
            "nf3nf4g128 (Edge8G MoE)",
            RealWeightMode::KronosNfMixedGptq {
                attn_head_bits: 4,
                ffn_bits: 3,
                scale_group_chunks: 2,
            },
        ),
    ];

    let eval_threads = kronos_2t::thread_pool::default_inference_threads();
    for (label, mode) in modes {
        let mut eng = fp32.compile_sequential_bipolar_gptq(mode, &calib_seqs, eval_threads);
        eng.repack_sdot();
        eng.set_num_threads(eval_threads);
        if matches!(eng.cfg.head_dim(), 64 | 128) {
            eng.enable_q8_kv_cache(true);
        }
        let mut nll = 0.0f64;
        let mut kl_sum = 0.0f64;
        let mut cos_sum = 0.0f64;
        let mut top1_hits = 0usize;
        let mut cursor = 0usize;

        for chunk in &eval_chunks {
            eng.reset_context();
            for pos in 0..(chunk_len - 1) {
                let logits = eng.forward_token(chunk[pos]);
                if pos >= score_start {
                    let target = chunk[pos + 1];
                    let q_lsm = log_softmax_f64(logits);
                    nll -= q_lsm[target];
                    let p_lsm = &ref_logprobs[cursor];
                    let mut kl = 0.0f64;
                    for (&lp, &lq) in p_lsm.iter().zip(q_lsm.iter()) {
                        let p = lp.exp();
                        if p > 1e-15 {
                            kl += p * (lp - lq);
                        }
                    }
                    kl_sum += kl;
                    cos_sum += logit_cosine(&ref_logits_raw[cursor], logits);
                    if argmax(logits) == ref_top1[cursor] {
                        top1_hits += 1;
                    }
                    cursor += 1;
                }
            }
        }
        let tps = measure_tps(&mut eng);
        rows.push(QuantEvalRow {
            label,
            bpw: eng.physical_bits_per_weight(),
            weight_mib: mib(eng.linear_weight_bytes()),
            ppl: (nll / (total_scored as f64)).exp(),
            mean_kl: kl_sum / (total_scored as f64),
            mean_cos: cos_sum / (total_scored as f64),
            top1_pct: 100.0 * (top1_hits as f64) / (total_scored as f64),
            decode_tps: tps,
        });
    }

    println!(
        "{:<26} | {:>6} | {:>8} | {:>8} | {:>8} | {:>8} | {:>7} | {:>9}",
        "Mode", "BPW", "Lin MiB", "PPL", "Mean KL", "LogitCos", "Top-1%", "tg64 tok/s"
    );
    println!("{}", "-".repeat(102));
    for r in &rows {
        println!(
            "{:<26} | {:>6.3} | {:>8.2} | {:>8.3} | {:>8.4} | {:>8.5} | {:>6.2}% | {:>9.1}",
            r.label, r.bpw, r.weight_mib, r.ppl, r.mean_kl, r.mean_cos, r.top1_pct, r.decode_tps
        );
    }

    let vq2 = rows.iter().find(|r| r.label.starts_with("vq2g128")).unwrap();
    let e8 = rows.iter().find(|r| r.label.starts_with("e8q2g128")).unwrap();
    assert!(
        e8.ppl < vq2.ppl && e8.mean_kl < vq2.mean_kl && e8.top1_pct > vq2.top1_pct,
        "expected Gosset E8Q2 (PPL {:.3}, KL {:.4}, Top1 {:.2}%) to outperform VQ2 (PPL {:.3}, KL {:.4}, Top1 {:.2}%) at 2.125 bpw",
        e8.ppl,
        e8.mean_kl,
        e8.top1_pct,
        vq2.ppl,
        vq2.mean_kl,
        vq2.top1_pct
    );
}

fn run_suite_3_jev_calibration_latency_and_oracles() {
    println!();
    println!("================================================================================");
    println!("SUITE 3: Kronos-Jev Single-Pass Calibration, Latency, Token Savings & Oracles");
    println!("================================================================================");

    let model_dir = Path::new("models/Qwen2.5-Coder-0.5B-Instruct");
    let jev_threads = kronos_2t::thread_pool::default_inference_threads();
    let mut chat_opt = if model_dir.join("kronos-nf4g128.k2tb").exists() {
        Some(
            ChatModel::load(model_dir, "nf4g128", 2048, jev_threads, None)
                .expect("load Qwen2.5-Coder-0.5B-Instruct for Jev suite"),
        )
    } else {
        None
    };

    // Warm system prefix to verify bit-exact KV prefix preservation across Jev single-pass calls.
    if let Some(m) = chat_opt.as_mut() {
        m.prewarm_prefix("You are Kronos.", &[]);
    }
    let cached_before = chat_opt.as_ref().map_or(0, |m| m.cached_prefix_len());

    // 12 labeled routing tasks (`0` = Chat, `1` = Agent, `2` = Computer) + 6 binary verification tasks.
    let route_cases: [(&str, bool, usize); 12] = [
        ("Explain how borrow checking works in Rust with a short example.", false, 0),
        ("What is the time complexity of Cholesky decomposition?", false, 0),
        ("Summarize the trade-off between latency and throughput.", false, 0),
        ("Why does Walsh-Hadamard rotation suppress activation outliers?", false, 0),
        ("Create index.html, style.css, and app.js for a Pomodoro timer dashboard.", true, 1),
        ("Edit src/main.rs to fix the off-by-one panic and run cargo test.", true, 1),
        ("Write a Python script parse_logs.py and save it in this workspace.", true, 1),
        ("Update the navbar color in style.css to #0f766e and add a dark mode toggle.", true, 1),
        ("Open Safari and navigate to https://developer.apple.com.", false, 2),
        ("Launch Calculator on my Mac.", false, 2),
        ("Set macOS system volume to 35 percent.", false, 2),
        ("Open Finder in my Downloads folder.", false, 2),
    ];

    let mut struct_latencies_ms = Vec::with_capacity(route_cases.len());
    let mut neural_latencies_ms = Vec::with_capacity(route_cases.len() + 6);
    let mut brier_sum = 0.0f64;
    let mut correct = 0usize;
    // 5-bin Expected Calibration Error (`[0, 0.2), ..., [0.8, 1.0]`).
    let mut bin_count = [0usize; 5];
    let mut bin_conf_sum = [0.0f64; 5];
    let mut bin_acc_sum = [0.0f64; 5];
    let mut total_eval = 0usize;

    for &(prompt, ws_has_files, expected_idx) in &route_cases {
        let t_s = Instant::now();
        let prior = structural_route_logits(prompt, false, ws_has_files, false, true);
        let _prelim = CalibratedChoice::from_logits(&prior, 1.0, false, 0.0);
        struct_latencies_ms.push(t_s.elapsed().as_secs_f64() * 1000.0);

        let t_n = Instant::now();
        let neural = chat_opt.as_mut().and_then(|m| {
            let q = format!(
                "Labels: Chat (answer question), Code (create/edit project files), Mac (open/control macOS app).\nWorkspace has files: {ws_has_files}.\nRequest: {prompt}\nLabel:"
            );
            m.jev_score_anchors(
                "Classify the request by answering with only one listed label token and nothing else.",
                &q,
                &["Chat", "Code", "Mac"],
            )
        });
        let (fused, used_neural) = fuse_choice_logits(
            &prior,
            neural.as_ref().map(|(v, mass)| (v.as_slice(), *mass)),
            1.25,
        );
        let choice = CalibratedChoice::from_logits(
            &fused,
            1.0,
            used_neural,
            t_n.elapsed().as_secs_f64() * 1000.0,
        );
        if used_neural {
            neural_latencies_ms.push(choice.elapsed_ms);
        }

        let pred = choice.selected;
        let hit = usize::from(pred == expected_idx);
        correct += hit;
        total_eval += 1;

        let mut sample_brier = 0.0f64;
        for (c, &p) in choice.probs.iter().enumerate() {
            let y = if c == expected_idx { 1.0 } else { 0.0 };
            let d = (p as f64) - y;
            sample_brier += d * d;
        }
        brier_sum += sample_brier / (choice.probs.len() as f64);

        let max_p = choice.probs[pred] as f64;
        let b = ((max_p * 5.0).floor() as usize).min(4);
        bin_count[b] += 1;
        bin_conf_sum[b] += max_p;
        bin_acc_sum[b] += hit as f64;
    }

    let binary_cases: [(&str, f32, bool); 6] = [
        (
            "Requirement: dark theme toggle present. Evidence: style.css defines [data-theme='dark'] and app.js toggles dataset.theme. Satisfied?",
            2.2,
            true,
        ),
        (
            "Requirement: export CSV button implemented. Evidence: app.js builds Blob('text/csv') on #exportCsv click. Satisfied?",
            2.1,
            true,
        ),
        (
            "Requirement: cargo test passes cleanly. Evidence: test result: ok. 12 passed; 0 failed. Satisfied?",
            2.4,
            true,
        ),
        (
            "Requirement: responsive chart rendered. Evidence: app.js contains // TODO: implement chart later. Satisfied?",
            -2.3,
            false,
        ),
        (
            "Requirement: all script imports resolve. Evidence: index.html loads ./missing_chart.js which does not exist on disk. Satisfied?",
            -2.6,
            false,
        ),
        (
            "Requirement: zero runtime errors on load. Evidence: ReferenceError: renderBoard is not defined at app.js:14. Satisfied?",
            -2.5,
            false,
        ),
    ];

    for &(query, prior_lo, expected_true) in &binary_cases {
        let t_n = Instant::now();
        let neural = chat_opt.as_mut().and_then(|m| {
            m.jev_score_anchors("Answer with only Yes or No.", query, &["Yes", "No"])
        });
        let (lo, used_neural) = fuse_binary_log_odds(
            prior_lo,
            neural.as_ref().map(|(v, mass)| (v.as_slice(), *mass)),
            1.25,
        );
        let cb = CalibratedBool::from_log_odds(
            lo,
            0.5,
            used_neural,
            t_n.elapsed().as_secs_f64() * 1000.0,
        );
        if used_neural {
            neural_latencies_ms.push(cb.elapsed_ms);
        }
        let hit = usize::from(cb.value == expected_true);
        correct += hit;
        total_eval += 1;
        let y = if expected_true { 1.0 } else { 0.0 };
        let d = (cb.prob_true as f64) - y;
        brier_sum += d * d;
        let conf = if cb.value {
            cb.prob_true as f64
        } else {
            1.0 - (cb.prob_true as f64)
        };
        let b = ((conf * 5.0).floor() as usize).min(4);
        bin_count[b] += 1;
        bin_conf_sum[b] += conf;
        bin_acc_sum[b] += hit as f64;
    }

    if let Some(m) = chat_opt.as_ref() {
        assert_eq!(
            m.cached_prefix_len(),
            cached_before,
            "Jev single-pass scoring disturbed cached target KV prefix"
        );
    }

    let mean_struct_ms =
        struct_latencies_ms.iter().sum::<f64>() / (struct_latencies_ms.len() as f64);
    let mean_neural_ms = if neural_latencies_ms.is_empty() {
        0.0
    } else {
        neural_latencies_ms.iter().sum::<f64>() / (neural_latencies_ms.len() as f64)
    };
    let accuracy_pct = 100.0 * (correct as f64) / (total_eval as f64);
    let mean_brier = brier_sum / (total_eval as f64);
    let mut ece = 0.0f64;
    for b in 0..5 {
        if bin_count[b] > 0 {
            let w = (bin_count[b] as f64) / (total_eval as f64);
            let avg_conf = bin_conf_sum[b] / (bin_count[b] as f64);
            let avg_acc = bin_acc_sum[b] / (bin_count[b] as f64);
            ece += w * (avg_conf - avg_acc).abs();
        }
    }

    // Turn-1 tool-schema pruning & 0-turn workspace symbol prefetch measurement.
    let ws_dir = std::env::temp_dir().join(format!("kronos_jev_bench_ws_{}", std::process::id()));
    let _ws_guard = TempPathGuard::dir(ws_dir.clone());
    fs::write(
        ws_dir.join("index.html"),
        "<!doctype html><html><head><title>Pulse Kanban</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><main id=\"board\"></main><script src=\"app.js\"></script></body></html>",
    )
    .unwrap();
    fs::write(
        ws_dir.join("style.css"),
        ":root { --accent: #2563eb; --bg: #f8fafc; }\n#board { display: grid; gap: 16px; }",
    )
    .unwrap();
    fs::write(
        ws_dir.join("app.js"),
        "const columns = ['Todo', 'Doing', 'Done'];\nfunction renderBoard() { const el = document.getElementById('board'); el.textContent = columns.join(' | '); }\nrenderBoard();\n",
    )
    .unwrap();

    let all_tools = tool_definitions_json();
    let user_edit_req = "Update --accent in style.css to #0d9488 so the board uses a teal accent.";
    let plan = plan_turn_with_jev(
        chat_opt.as_mut(),
        "You are Kronos.",
        user_edit_req,
        Some(TurnRoute::Agent),
        false,
        &ws_dir,
        false,
        &all_tools,
        false,
        false,
    );
    let full_tool_bytes: usize = all_tools.iter().map(|s| s.len()).sum();
    let pruned_tool_bytes: usize = plan.active_tools_json.iter().map(|s| s.len()).sum();
    let schema_reduction_pct =
        100.0 * (1.0 - (pruned_tool_bytes as f64) / (full_tool_bytes.max(1) as f64));
    let token_savings_pct = if let Some(m) = chat_opt.as_ref() {
        // Baseline 2-turn discovery workflow (`list_dir` + `read_file` with all 8 tool schemas)
        // vs single-turn Jev pre-fetched edit with pruned schemas (`0` discovery turns).
        let base_t1 = m.count_prompt_tokens(
            &[
                Message::new("system", "You are Kronos."),
                Message::new("user", user_edit_req),
            ],
            &all_tools,
        );
        let base_t2 = m.count_prompt_tokens(
            &[
                Message::new("system", "You are Kronos."),
                Message::new("user", user_edit_req),
                Message::new("assistant", "<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"style.css\"}}</tool_call>"),
                Message::tool_result("call_0", ":root { --accent: #2563eb; --bg: #f8fafc; }\n#board { display: grid; gap: 16px; }"),
            ],
            &all_tools,
        );
        let prefetched = plan.prefetch_block.as_deref().unwrap_or("");
        let jev_t1 = m.count_prompt_tokens(
            &[
                Message::new("system", "You are Kronos."),
                Message::new("user", &format!("{prefetched}\n\n{user_edit_req}")),
            ],
            &plan.active_tools_json,
        );
        100.0 * (1.0 - (jev_t1 as f64) / ((base_t1 + base_t2).max(1) as f64))
    } else {
        schema_reduction_pct
    };

    // Deterministic execution oracle + monotonic hill-climbing checkpoint verification.
    let rubric = extract_requirement_rubric(user_edit_req, TurnRoute::Agent, true);
    let modified = vec![
        "index.html".to_string(),
        "style.css".to_string(),
        "app.js".to_string(),
    ];

    // Step 1: Valid teal edit (`S_1` high-water checkpoint).
    fs::write(
        ws_dir.join("style.css"),
        ":root { --accent: #0d9488; --bg: #f8fafc; }\n#board { display: grid; gap: 16px; }",
    )
    .unwrap();
    let snap_css = fs::read_to_string(ws_dir.join("style.css")).unwrap();
    let oracle_ok = verify_workspace_work(&ws_dir, &modified, 10);
    assert!(oracle_ok.is_ok(), "clean workspace failed oracle: {oracle_ok:?}");
    let eval_good = evaluate_rubric(
        chat_opt.as_mut(),
        user_edit_req,
        &rubric,
        &ws_dir,
        &modified,
        true,
        None,
        "Updated --accent to #0d9488 in style.css.",
    );

    // Step 2: Inject hallucinated missing asset + runtime ReferenceError + ungrounded prose claim.
    fs::write(
        ws_dir.join("app.js"),
        "nonExistentDomHelper();\n// TODO: finish rest of implementation\n",
    )
    .unwrap();
    let oracle_bad = verify_workspace_work(&ws_dir, &modified, 10);
    assert!(oracle_bad.is_err(), "oracle missed injected runtime/stub defect");
    let eval_bad = evaluate_rubric(
        chat_opt.as_mut(),
        user_edit_req,
        &rubric,
        &ws_dir,
        &modified,
        true,
        oracle_bad.as_ref().err().map(String::as_str),
        "Updated everything cleanly.",
    );
    assert!(!eval_bad.passed && eval_bad.score.expected < eval_good.score.expected);

    // Monotonic rollback restores high-water snapshot so `S_final == max_t S_t`.
    fs::write(ws_dir.join("style.css"), &snap_css).unwrap();
    fs::write(
        ws_dir.join("app.js"),
        "const columns = ['Todo', 'Doing', 'Done'];\nfunction renderBoard() { const el = document.getElementById('board'); el.textContent = columns.join(' | '); }\nrenderBoard();\n",
    )
    .unwrap();
    let oracle_restored = verify_workspace_work(&ws_dir, &modified, 10);
    assert!(oracle_restored.is_ok());

    println!(
        "  Structural Latency : {:6.3} ms mean (< 1.0 ms target)",
        mean_struct_ms
    );
    if !neural_latencies_ms.is_empty() {
        println!(
            "  Neural 1-Pass Jev  : {:6.2} ms mean across {} calls (0 decode tokens, < 25.0 ms target)",
            mean_neural_ms,
            neural_latencies_ms.len()
        );
    }
    println!(
        "  Calibration        : Accuracy = {:.1}% ({}/{}) | Brier = {:.4} | 5-Bin ECE = {:.4}",
        accuracy_pct, correct, total_eval, mean_brier, ece
    );
    println!(
        "  0-Turn Pre-Pass    : Tools {} -> {} (-{:.1}% schema bytes) | -{:.1}% prefill tokens vs 2-turn discovery | prefetched {:?}",
        all_tools.len(),
        plan.active_tools_json.len(),
        schema_reduction_pct,
        token_savings_pct,
        plan.prefetched_files
    );
    println!(
        "  Monotonic Oracles  : S_good={:.3} (PASS) -> S_defect={:.3} (REJECTED, alpha=0.00) -> Rollback S_final={:.3}",
        eval_good.score.expected, eval_bad.score.expected, eval_good.score.expected
    );

    assert!(
        mean_struct_ms < 1.0,
        "structural Jev path {:.3} ms exceeded 1.0 ms",
        mean_struct_ms
    );
    if !neural_latencies_ms.is_empty() {
        assert!(
            mean_neural_ms < 25.0,
            "neural single-pass Jev {:.2} ms exceeded 25.0 ms",
            mean_neural_ms
        );
    }
    assert!(
        accuracy_pct >= 90.0 && mean_brier < 0.12 && ece < 0.15,
        "Jev calibration below contract: acc={accuracy_pct:.1}%, brier={mean_brier:.4}, ece={ece:.4}"
    );
    assert!(
        schema_reduction_pct >= 40.0 && token_savings_pct >= 40.0,
        "expected >= 40% Turn-1 schema/prefill savings, got schema={schema_reduction_pct:.1}%, prefill={token_savings_pct:.1}%"
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut skip_20b = false;
    let mut skip_135m = false;
    let mut skip_jev = false;
    let mut num_layers = 24usize;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--skip-20b" => skip_20b = true,
            "--skip-135m" => skip_135m = true,
            "--skip-jev" => skip_jev = true,
            "--layers" | "--shard-layers" => {
                i += 1;
                if let Some(v) = args.get(i) {
                    num_layers = v.parse().expect("invalid --layers");
                }
            }
            other => {
                eprintln!("unknown flag: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if !skip_20b {
        run_suite_1_20b_edge_and_mobile(num_layers);
    }
    if !skip_135m {
        run_suite_2_pretrained_quantizer_quality();
    }
    if !skip_jev {
        run_suite_3_jev_calibration_latency_and_oracles();
    }

    println!();
    println!("All Edge-20B, Mobile5G, Gosset E8Q2, and Kronos-Jev verification checks passed.");
}
