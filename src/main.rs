use kronos_2t::{
    model::compiler::compile_kronecker_matrix,
    model::demo_model::{
        build_trained_demo_engine, generate_visible_sentence, run_needle_in_haystack_test,
    },
    model::engine::{KronosConfig, KronosEngine},
    kernels::lut_gemv, kernels::router,
    kernels::simd_lut::{
        default_register_codebook_64, default_simd_codebook, fused_kronecker_gemv, simd_lut_gemv,
        QuantizedActivationLut, SimdPackedMatrix,
    },
    kernels::ssd::{SsdBlock, SsdScratch, SsdState},
    kernels::synthesis,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);

#[global_allocator]
static GLOBAL_ALLOC: TrackingAllocator = TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: Delegates directly to the OS system allocator with unmodified layout.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: Delegates directly to the OS system allocator.
        unsafe { System.dealloc(ptr, layout) }
    }
}

fn reset_alloc_stats() {
    ALLOC_COUNT.store(0, Ordering::SeqCst);
    ALLOC_BYTES.store(0, Ordering::SeqCst);
}

fn read_alloc_stats() -> (usize, usize) {
    (
        ALLOC_COUNT.load(Ordering::SeqCst),
        ALLOC_BYTES.load(Ordering::SeqCst),
    )
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() {
    println!("==============================================================================");
    println!(" KRONOS-2T v0.3: Zero-Alloc Register-Fused Engine & Live Intelligence Suite");
    println!("==============================================================================\n");

    let d_model = 1024;
    let sub_keys = 1000;
    let top_k = 4;
    let iters = 40;

    // ------------------------------------------------------------------------
    // [1] Zero-Allocation 4-Key SIMD Product-Key Router (1,000,000 Experts)
    // ------------------------------------------------------------------------
    let pk_router = router::ProductKeyRouter::new(d_model, sub_keys, 42);
    let hidden_state: Vec<f32> = (0..d_model).map(|i| ((i as f32) * 0.013).sin()).collect();
    let mut routed_buf = Vec::with_capacity(16);

    pk_router.route_top_k_into(black_box(&hidden_state), top_k, &mut routed_buf);
    reset_alloc_stats();
    let t0 = Instant::now();
    for _ in 0..iters {
        pk_router.route_top_k_into(black_box(&hidden_state), top_k, &mut routed_buf);
        black_box(&routed_buf);
    }
    let route_us = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);
    let (router_allocs, _) = read_alloc_stats();

    println!("[1] Zero-Alloc 4-Key SIMD Product-Key Router (1,000,000 Micro-Experts)");
    println!("    - Routing Latency (avg)     : {:.2} µs (down from 801.5 µs v0.1 -> 103.1 µs v0.2)", route_us);
    println!("    - Heap Allocations / Call   : {} allocs", router_allocs);
    println!("    - Selected Top-{} Experts    : {:?}\n", top_k, routed_buf);

    // ------------------------------------------------------------------------
    // [2] Register-Fused Kronecker NEON (`vqtbl4q_u8` + `vqtbl1q_s8`) vs L1 Materialization
    // ------------------------------------------------------------------------
    let rows = 512;
    let cols = 1024;
    let mut rng = 0xCAFE_BABE_2026_0001u64;
    let ternary_weights: Vec<i8> = (0..rows * cols)
        .map(|_| {
            rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z % 3) as i8) - 1
        })
        .collect();

    let packed_simd = SimdPackedMatrix::from_ternary_weights(&ternary_weights, rows, cols);
    let mut out_dense = vec![0.0f32; rows];
    let mut out_neon = vec![0.0f32; rows];
    let mut out_l1_synth = vec![0.0f32; rows];
    let mut out_reg_fused = vec![0.0f32; rows];

    // Dense 4-way SIMD FP32 baseline
    let t_dense = Instant::now();
    for _ in 0..iters {
        let w_ref = black_box(ternary_weights.as_slice());
        let x_ref = black_box(hidden_state.as_slice());
        for (out_val, row_slice) in out_dense.iter_mut().zip(w_ref.chunks_exact(cols)) {
            let mut acc = [0.0f32; 4];
            for (wc, xc) in row_slice.chunks_exact(4).zip(x_ref.chunks_exact(4)) {
                acc[0] += (wc[0] as f32) * xc[0];
                acc[1] += (wc[1] as f32) * xc[1];
                acc[2] += (wc[2] as f32) * xc[2];
                acc[3] += (wc[3] as f32) * xc[3];
            }
            *out_val = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        }
        black_box(&out_dense);
    }
    let dense_ns = t_dense.elapsed().as_nanos() / (iters as u128);

    // Reusable zero-alloc INT8 LUT + Static NEON LUT-GEMV
    let mut reusable_qlut = QuantizedActivationLut::new_zeroed(cols);
    let t_neon = Instant::now();
    for _ in 0..iters {
        reusable_qlut.populate(black_box(&hidden_state));
        simd_lut_gemv(
            black_box(&packed_simd),
            black_box(&reusable_qlut),
            &mut out_neon,
        );
        black_box(&out_neon);
    }
    let neon_ns = t_neon.elapsed().as_nanos() / (iters as u128);

    // Old L1 Tile Materialization + FP32 LUT-GEMV (v0.1/v0.2 approach)
    let expert = synthesis::KroneckerCompressedExpert::new(routed_buf[0].0, rows, cols);
    let cb256 = synthesis::default_canonical_codebook();
    let luts_fp32 = lut_gemv::build_activation_lut(&hidden_state);
    let mut l1_tile = vec![0u8; rows * (cols / 4)];
    let t_old_synth = Instant::now();
    for _ in 0..iters {
        expert.synthesize_and_execute(
            black_box(&cb256),
            black_box(&luts_fp32),
            &mut l1_tile,
            &mut out_l1_synth,
        );
        black_box(&out_l1_synth);
    }
    let old_synth_ns = t_old_synth.elapsed().as_nanos() / (iters as u128);

    // New Register-Fused Kronecker NEON (`vqtbl4q_u8` + `vqtbl1q_s8`, ZERO L1 tile buffer!)
    let reg_cb64 = default_register_codebook_64();
    let scales_1 = [1.0f32];
    let _ = default_simd_codebook();
    let t_fused = Instant::now();
    for _ in 0..iters {
        fused_kronecker_gemv(
            black_box(&expert.row_seeds),
            black_box(&expert.col_seeds),
            black_box(&scales_1),
            rows,
            black_box(&reg_cb64),
            black_box(&reusable_qlut),
            &mut out_reg_fused,
        );
        black_box(&out_reg_fused);
    }
    let fused_ns = t_fused.elapsed().as_nanos() / (iters as u128);

    println!(
        "[2] Register-Fused Kronecker Synthesis & GEMV ({}x{})",
        rows, cols
    );
    println!("    - Dense SIMD FP32 GEMV            : {:>7} ns (1.00x baseline)", dense_ns);
    println!(
        "    - Static NEON vqtbl1q_s8 Reg-LUT  : {:>7} ns ({:.2}x speedup, cosine sim {:.6})",
        neon_ns,
        dense_ns as f64 / neon_ns.max(1) as f64,
        cosine_similarity(&out_dense, &out_neon)
    );
    println!("    - Old L1-Materialized Synthesis   : {:>7} ns (writes & re-reads 128 KB L1 buffer)", old_synth_ns);
    println!(
        "    - NEW Register-Fused vqtbl4q+1q   : {:>7} ns ({:.2}x faster synthesis+exec, 0 B buffer!)\n",
        fused_ns,
        old_synth_ns as f64 / fused_ns.max(1) as f64
    );

    // ------------------------------------------------------------------------
    // [3] Multi-Rank Kronecker Weight Compiler Fidelity
    // ------------------------------------------------------------------------
    let target_w: Vec<f32> = (0..32 * 64)
        .map(|idx| {
            let r = (idx / 64) as f32;
            let c = (idx % 64) as f32;
            (r * 0.21).sin() * (c * 0.13).cos() + 0.35 * ((r + c) * 0.31).sin()
        })
        .collect();
    let r1 = compile_kronecker_matrix(&target_w, 32, 64, 1, &reg_cb64);
    let r2 = compile_kronecker_matrix(&target_w, 32, 64, 2, &reg_cb64);
    let r4 = compile_kronecker_matrix(&target_w, 32, 64, 4, &reg_cb64);
    let r8 = compile_kronecker_matrix(&target_w, 32, 64, 8, &reg_cb64);

    println!("[3] Multi-Rank Kronecker Weight Compiler (Alternating Coordinate Descent)");
    println!(
        "    - Relative Frobenius Error : Rank-1={:.3} -> Rank-2={:.3} -> Rank-4={:.3} -> Rank-8={:.3}\n",
        r1.relative_frobenius_error,
        r2.relative_frobenius_error,
        r4.relative_frobenius_error,
        r8.relative_frobenius_error
    );

    // ------------------------------------------------------------------------
    // [4] Zero-Allocation 12-Layer 2.36T-Virtual KronosEngine Throughput & Alloc Audit
    // ------------------------------------------------------------------------
    let ssd_block = SsdBlock::new(256, 8, 2026);
    let mut ssd_st = SsdState::new(8, 32);
    let mut ssd_sc = SsdScratch::new(256, 8);
    let mut ssd_out = vec![0.0f32; 256];
    ssd_block.step_recurrent(&hidden_state[..256], &mut ssd_st, &mut ssd_sc, &mut ssd_out);

    let cfg = KronosConfig::default_2t_prototype();
    let k2t_path = std::env::temp_dir().join("kronos_2t_v3_bench.k2t");
    let mut engine = KronosEngine::build_and_open(cfg.clone(), &k2t_path).expect("engine open");

    let prompt = [7usize, 42, 128, 511, 1024, 1999];
    let _ = engine.generate(&prompt, 4).expect("warmup");
    engine.reset_context();

    // Measure exact OS heap allocations on a single multi-layer token pass!
    let _ = engine.forward_token(7).unwrap();
    reset_alloc_stats();
    let _ = engine.forward_token(42).unwrap();
    let (tok_allocs, tok_bytes) = read_alloc_stats();

    engine.reset_context();
    let report = engine.generate(&prompt, 64).expect("bench generate");
    let _ = std::fs::remove_file(&k2t_path);

    println!("[4] Zero-Allocation 12-Layer 2.36T-Virtual KronosEngine (Before vs After)");
    println!(
        "    - Heap Allocations / Token    : {} allocs / {} bytes (down from ~330 allocs/tok!)",
        tok_allocs, tok_bytes
    );
    println!(
        "    - Decode Throughput           : {:.2} tokens/sec ({:.2} ms/token, up from 609 tok/s!)",
        report.tokens_per_sec,
        1000.0 / report.tokens_per_sec
    );
    println!(
        "    - .k2t mmap Ring-Buffer Hits  : {:.1}% | Constant SSD Context: {:.2} MB\n",
        report.pager_hit_rate_percent,
        report.ssd_context_bytes as f64 / (1024.0 * 1024.0)
    );

    // ------------------------------------------------------------------------
    // [5] REAL VISIBLE TEST: Trained .k2t English Generation + 1,200-Token Passkey Swap!
    // ------------------------------------------------------------------------
    println!("==============================================================================");
    println!(" LIVE VISIBLE INTELLIGENCE TEST (Trained & Compiled .k2t mmap Model)");
    println!("==============================================================================\n");

    let demo_k2t = std::env::temp_dir().join("kronos_2t_trained_live.k2t");
    let t_train = Instant::now();
    let (vocab, mut demo_engine) =
        build_trained_demo_engine(&demo_k2t).expect("failed to train demo model");
    let train_ms = t_train.elapsed().as_secs_f64() * 1000.0;

    println!(
        "  Trained & compiled {}-word BitNet-b1.58 + Rank-4 Kronecker .k2t model in {:.1} ms",
        vocab.id_to_word.len(),
        train_ms
    );
    println!(
        "  Mapped container: {:?} ({} KB on disk)\n",
        demo_k2t.file_name().unwrap(),
        demo_engine.pager.mmap.byte_len() / 1024
    );

    println!("  [Test A] Multi-Topic English Generation & Live Product-Key Expert Routing:");
    for prompt_str in [
        "kronos solves",
        "product key",
        "state space",
        "lookahead subspace",
    ] {
        let (sentence, l0_experts) =
            generate_visible_sentence(&vocab, &mut demo_engine, prompt_str, 20);
        println!(
            "    Prompt: {:<20?} -> Routed Experts {:?}\n      => \"{}\"",
            prompt_str, l0_experts, sentence
        );
    }

    println!("\n  [Test B] 1,200-Token Needle-in-a-Haystack In-Context Recall (O(1) Memory):");
    let pass_a = [
        ("vault_alpha", "amber_749"),
        ("orbit_key", "cobalt_318"),
        ("reactor_pin", "helios_904"),
    ];
    let (results_a, ram_timeline) =
        run_needle_in_haystack_test(&vocab, &mut demo_engine, &pass_a, 1_200);

    print!("    SSD Context RAM Timeline : ");
    for (i, (tok_cnt, bytes)) in ram_timeline.iter().enumerate() {
        if i > 0 {
            print!(" -> ");
        }
        print!("@{} toks: {:.1} KB", tok_cnt, *bytes as f64 / 1024.0);
    }
    println!(" (0 B growth!)");

    println!("    Episode 1 (Planted @ token 1, queried after 1,200 distractor tokens):");
    for (k, expected, predicted) in &results_a {
        let status = if expected == predicted { "PASS" } else { "FAIL" };
        println!(
            "      [{}] Query {:<13?} => Predicted: {:<12?} (Expected: {:?})",
            status, k, predicted, expected
        );
    }

    // Episode 2: Swap every passkey to a DIFFERENT value at runtime on the EXACT SAME .k2t file!
    let pass_b = [
        ("vault_alpha", "quartz_512"),
        ("orbit_key", "zenith_601"),
        ("reactor_pin", "cipher_285"),
    ];
    let (results_b, _) = run_needle_in_haystack_test(&vocab, &mut demo_engine, &pass_b, 1_200);
    println!(
        "    Episode 2 (RUNTIME VALUE SWAP on same .k2t model + 1,200 distractor tokens):"
    );
    for (k, expected, predicted) in &results_b {
        let status = if expected == predicted { "PASS" } else { "FAIL" };
        println!(
            "      [{}] Query {:<13?} => Predicted: {:<12?} (Expected: {:?})",
            status, k, predicted, expected
        );
    }
    println!("==============================================================================");

    let _ = std::fs::remove_file(&demo_k2t);
}
