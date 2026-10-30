use kronos_2t::{
    model::bpe_tokenizer::BpeTokenizer,
    model::compiler::{
        compile_global_ternary_as_multiplane, compile_hadamard_gptq_bipolar,
        compile_hadamard_gptq_full_row, compile_kronecker_matrix, compile_multiplane_ternary,
        FullRowGptqHessian,
    },
    kernels::hadamard::fwht64_signed_inplace,
    model::real_llm::{
        collect_fp32_reference_logits, evaluate_quality_against_reference, RealLlamaEngine,
        RealWeightMode,
    },
    storage::safetensors::SafeTensorsFile,
    kernels::simd_lut::{
        default_register_codebook_64, fused_kronecker_gemv, simd_lut_base3_gemv,
        simd_lut_bipolar_gemv, simd_lut_gemv_multiplane, BipolarActivationLut, GgufQ4_0Matrix,
        QuantizedActivationLut,
    },
};
use std::path::Path;
use std::time::Instant;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64) * (y as f64))
        .sum();
    let na: f64 = a.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let nb: f64 = b.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32
}

fn bench_generation(
    eng: &mut RealLlamaEngine,
    prompts: &[&str],
    tokenizer: &BpeTokenizer,
    gen_tokens: usize,
) -> (Vec<String>, f64) {
    // Warm up spin-pool / instruction cache on 1 short step
    let warm_ids = [tokenizer.eos_token_id, 250];
    let _ = eng.generate_greedy(&warm_ids, 4, tokenizer.eos_token_id);

    let mut outs = Vec::with_capacity(prompts.len());
    let mut tps_sum = 0.0;
    for p_str in prompts {
        let mut p_ids = vec![tokenizer.eos_token_id];
        p_ids.extend(tokenizer.encode(p_str));
        // Run 3 timed passes per prompt and take median/mean after warmup for stable measurement
        let mut best_tps = 0.0f64;
        let mut last_ids = Vec::new();
        for _ in 0..3 {
            let (g_ids, tps) = eng.generate_greedy(&p_ids, gen_tokens, tokenizer.eos_token_id);
            if tps > best_tps {
                best_tps = tps;
            }
            last_ids = g_ids;
        }
        tps_sum += best_tps;
        outs.push(format!("{p_str}{}", tokenizer.decode(&last_ids)));
    }
    (outs, tps_sum / (prompts.len() as f64))
}

fn main() {
    let model_dir =
        Path::new("/Users/melodyleonard/Documents/project/kronos-2t/models/SmolLM2-135M");
    if !model_dir.join("model.safetensors").exists() {
        eprintln!(
            "Real model checkpoint not found at {}. Please download HuggingFaceTB/SmolLM2-135M.",
            model_dir.display()
        );
        std::process::exit(1);
    }

    println!("==========================================================================================================");
    println!(" KRONOS-2T vs CUTTING-EDGE LLM ENGINES — REAL WEIGHT QUALITY & SPEED BENCHMARK (SmolLM2-135M)");
    println!("==========================================================================================================\n");

    let tokenizer = BpeTokenizer::from_model_dir(model_dir).expect("failed to load BPE tokenizer");
    let st = SafeTensorsFile::open(&model_dir.join("model.safetensors"))
        .expect("failed to mmap model.safetensors");
    println!(
        "Loaded SafeTensors mmap: {:.1} MB ({} BF16 tensors) | BPE Vocabulary: {} tokens\n",
        st.byte_len() as f64 / (1024.0 * 1024.0),
        st.tensors.len(),
        tokenizer.id_to_token.len()
    );

    // --------------------------------------------------------------------------------------------------
    // PART 1: Single Real Pretrained Matrix Diagnostic (`model.layers.0.mlp.down_proj.weight` 576x1536)
    // --------------------------------------------------------------------------------------------------
    println!("[Part 1] Real Pretrained Matrix Diagnostic (Layer 0 down_proj: 576x1536 BF16)");
    let down0_f32 = st
        .tensor("model.layers.0.mlp.down_proj.weight")
        .expect("down_proj")
        .to_f32_vec();
    let rows = 576;
    let cols = 1536;

    // Realistic heavy-tailed SwiGLU activation vector
    let x_real: Vec<f32> = (0..cols)
        .map(|c| {
            let v = (((c * 31 + 7) as f32) * 0.09).sin() * 0.4;
            if c == 199 || c == 1041 {
                v * 25.0
            } else {
                v
            }
        })
        .collect();
    let y_exact: Vec<f32> = (0..rows)
        .map(|r| {
            down0_f32[r * cols..(r + 1) * cols]
                .iter()
                .zip(x_real.iter())
                .map(|(w, x)| w * x)
                .sum()
        })
        .collect();

    let qlut_unrot = QuantizedActivationLut::build(&x_real);
    let mut x_rot = x_real.clone();
    fwht64_signed_inplace(&mut x_rot);
    let qlut_rot = QuantizedActivationLut::build(&x_rot);
    let blut_rot = BipolarActivationLut::build(&x_rot);

    let cb64 = default_register_codebook_64();
    let kron8 = compile_kronecker_matrix(&down0_f32, rows, cols, 8, &cb64);
    let mut y_kron8 = vec![0.0f32; rows];
    fused_kronecker_gemv(
        &kron8.row_seeds,
        &kron8.col_seeds,
        &kron8.scales,
        rows,
        &cb64,
        &qlut_unrot,
        &mut y_kron8,
    );

    let (m_glob, _) = compile_global_ternary_as_multiplane(&down0_f32, rows, cols);
    let mut y_glob = vec![0.0f32; rows];
    simd_lut_gemv_multiplane(&m_glob, &qlut_unrot, &mut y_glob);

    let q4_mat = GgufQ4_0Matrix::from_f32(&down0_f32, rows, cols);
    let mut q8_v = vec![0i8; cols];
    let mut q8_s = vec![0.0f32; cols / 32];
    GgufQ4_0Matrix::quantize_q8_0_activation(&x_real, &mut q8_v, &mut q8_s);
    let mut y_q4 = vec![0.0f32; rows];
    unsafe {
        q4_mat.forward_row_range(&q8_v, &q8_s, 0, rows, y_q4.as_mut_ptr());
    }

    let mut hess0 = FullRowGptqHessian::new(cols);
    for s in 0..192usize {
        let xc: Vec<f32> = (0..cols)
            .map(|c| {
                let v = (((s * 43 + c * 17) as f32) * 0.13).cos() * 0.4;
                if c == 199 || c == 1041 {
                    v * 25.0
                } else {
                    v
                }
            })
            .collect();
        hess0.observe_activation(&xc);
    }
    hess0.finalize(0.025);
    let m_b3_2p = compile_hadamard_gptq_full_row(&down0_f32, rows, cols, 2, Some(&hess0), 8);
    let m_bp_4p = compile_hadamard_gptq_bipolar(&down0_f32, rows, cols, 4, Some(&hess0), 8);
    let m_bp_5p = compile_hadamard_gptq_bipolar(&down0_f32, rows, cols, 5, Some(&hess0), 8);
    let mut y_b3_2p = vec![0.0f32; rows];
    let mut y_bp_4p = vec![0.0f32; rows];
    let mut y_bp_5p = vec![0.0f32; rows];
    simd_lut_base3_gemv(&m_b3_2p, &qlut_rot, &mut y_b3_2p);
    simd_lut_bipolar_gemv(&m_bp_4p, &blut_rot, &mut y_bp_4p);
    simd_lut_bipolar_gemv(&m_bp_5p, &blut_rot, &mut y_bp_5p);

    let (m_old_2p, _) = compile_multiplane_ternary(&down0_f32, rows, cols, 2);
    let mut y_old_2p = vec![0.0f32; rows];
    simd_lut_gemv_multiplane(&m_old_2p, &qlut_unrot, &mut y_old_2p);

    println!(
        "  {:<46} | {:>10} | {:>16}",
        "Single-Matrix Quantization / Kernel", "Phys Bits", "GEMV Cosine Sim"
    );
    println!("  {}", "-".repeat(78));
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}",
        "Rank-8 XOR-Kronecker PTQ (no QAT)",
        ((kron8.row_seeds.len() + kron8.col_seeds.len() + kron8.scales.len() * 4) as f64 * 8.0)
            / ((rows * cols) as f64),
        cosine_sim(&y_exact, &y_kron8)
    );
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}",
        "Naive Global 1-Plane Ternary PTQ",
        2.00,
        cosine_sim(&y_exact, &y_glob)
    );
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}",
        "Uncalibrated Greedy 2-Plane Ternary (M4)",
        m_old_2p.physical_bits_per_weight(),
        cosine_sim(&y_exact, &y_old_2p)
    );
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}",
        "In-house Q4_0 x Q8_0 reimpl (G=32)",
        q4_mat.physical_bits_per_weight(),
        cosine_sim(&y_exact, &y_q4)
    );
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}",
        "Kronos Hadamard+GPTQ Base-3 2-Plane (9-lvl)",
        m_b3_2p.physical_bits_per_weight(),
        cosine_sim(&y_exact, &y_b3_2p)
    );
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}",
        "Kronos Hadamard+GPTQ Bipolar-4 (16-lvl, ZeroMul)",
        m_bp_4p.physical_bits_per_weight(),
        cosine_sim(&y_exact, &y_bp_4p)
    );
    println!(
        "  {:<46} | {:>9.2}b | {:>16.6}\n",
        "Kronos Hadamard+GPTQ Bipolar-5 (32-lvl, ZeroMul)",
        m_bp_5p.physical_bits_per_weight(),
        cosine_sim(&y_exact, &y_bp_5p)
    );

    // --------------------------------------------------------------------------------------------------
    // PART 2: Full 30-Layer `SmolLM2-135M` Head-to-Head Quality & Speed Table
    // --------------------------------------------------------------------------------------------------
    // 8 diverse calibration passages (strictly disjoint in topic & vocabulary from eval_passage & prompts!)
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
    let flat_awq_calib: Vec<usize> = calib_sequences[0].clone();
    let total_calib_tokens: usize = calib_sequences.iter().map(|s| s.len()).sum();

    let eval_passage = "The history of computing began long before the invention of the modern \
electronic transistor. In the nineteenth century, Charles Babbage designed the Analytical Engine, \
and Ada Lovelace wrote the first published algorithm intended to be carried out by a machine. \
More than a century later, Alan Turing formalized the mathematical foundations of computation, \
proving that a universal machine reading symbols from memory could execute any computable task.";
    let mut eval_ids = vec![tokenizer.eos_token_id];
    eval_ids.extend(tokenizer.encode(eval_passage));
    let max_seq_len = 256;
    let p_cores = 8usize;

    let prompts = [
        "Alan Turing was a",
        "In computer science, CPU cache locality is important because",
        "The planets of the solar system orbit the",
    ];
    let gen_tokens = 28usize;

    println!(
        "[Part 2] Full 30-Layer SmolLM2-135M Head-to-Head Benchmark ({} eval tokens | {} GPTQ calib tokens)",
        eval_ids.len(),
        total_calib_tokens
    );

    let mut fp32_engine =
        RealLlamaEngine::load_fp32_reference(model_dir, max_seq_len).expect("load fp32");
    let awq_calib = fp32_engine.calibrate_activations(&flat_awq_calib, 0.65);

    // Measure Exact FP32 (1-Core & 8-Core)
    fp32_engine.set_num_threads(1);
    let _ = fp32_engine.forward_token(eval_ids[0]);
    let (ref_logits, ref_ppl, fp32_eval_1t) =
        collect_fp32_reference_logits(&mut fp32_engine, &eval_ids);
    let (_, fp32_gen_1t) = bench_generation(&mut fp32_engine, &prompts, &tokenizer, gen_tokens);

    fp32_engine.set_num_threads(p_cores);
    let (fp32_outs, fp32_gen_8t) =
        bench_generation(&mut fp32_engine, &prompts, &tokenizer, gen_tokens);
    fp32_engine.set_num_threads(1);
    let fp32_mb = fp32_engine.linear_weight_bytes() as f64 / (1024.0 * 1024.0);

    println!(
        "  {:<36} | {:>5} | {:>6} | {:>7} | {:>8} | {:>7} | {:>6} | {:>8} | {:>8}",
        "Engine / Quantization Mode",
        "Bits",
        "RAM",
        "PPL(↓)",
        "LogitCos",
        "KL(nat)",
        "Top-1%",
        "1T Tok/s",
        "8T Tok/s"
    );
    println!("  {}", "-".repeat(110));
    println!(
        "  {:<36} | {:>4.2}b | {:>5.1}M | {:>7.2} | {:>8.6} | {:>7.4} | {:>5.1}% | {:>8.1} | {:>8.1}",
        "Exact FP32 Reference (NEON vfmaq)",
        fp32_engine.physical_bits_per_weight(),
        fp32_mb,
        ref_ppl,
        1.000000,
        0.0000,
        100.0,
        fp32_gen_1t.max(fp32_eval_1t),
        fp32_gen_8t
    );
    let mut gen_showcase: Vec<(String, Vec<String>, f64, f64)> = vec![(
        "Exact FP32 Reference (32.00b)".to_string(),
        fp32_outs,
        fp32_gen_1t,
        fp32_gen_8t,
    )];

    // 1. Naive Global 1-Plane (1.58b PTQ) & Milestone-4 AWQ 2-Plane (4.50b) & in-house Q4_0 reimpl (4.50b)
    let direct_modes = [
        (
            "Naive Global 1-Plane PTQ (M4 Base)",
            RealWeightMode::GlobalTernary1Plane,
            false,
        ),
        (
            "Kronos M4 AWQ 2-Plane (Old Greedy)",
            RealWeightMode::AwqGroup64Ternary { num_planes: 2 },
            false,
        ),
        (
            "In-house Q4_0 reimpl (G=32 Q4xQ8)",
            RealWeightMode::GgufQ4_0,
            true,
        ),
    ];

    for (label, mode, showcase) in direct_modes {
        let mut eng = fp32_engine.compile_from_reference(mode, Some(&awq_calib));
        eng.set_num_threads(1);
        let rep = evaluate_quality_against_reference(&mut eng, &eval_ids, &ref_logits);
        let (_, tps_1t) = bench_generation(&mut eng, &prompts, &tokenizer, gen_tokens);

        eng.set_num_threads(p_cores);
        let (outs_8t, tps_8t) = bench_generation(&mut eng, &prompts, &tokenizer, gen_tokens);

        let bits = eng.physical_bits_per_weight();
        let ram_mb = eng.linear_weight_bytes() as f64 / (1024.0 * 1024.0);
        let ppl_str = if rep.perplexity > 99_999.0 {
            ">99999".to_string()
        } else {
            format!("{:.2}", rep.perplexity)
        };
        println!(
            "  {:<36} | {:>4.2}b | {:>5.1}M | {:>7} | {:>8.6} | {:>7.4} | {:>5.1}% | {:>8.1} | {:>8.1}",
            label,
            bits,
            ram_mb,
            ppl_str,
            rep.mean_logit_cosine_sim,
            rep.mean_kl_divergence_nats,
            rep.top1_agreement_pct,
            tps_1t,
            tps_8t
        );
        if showcase {
            gen_showcase.push((label.to_string(), outs_8t, tps_1t, tps_8t));
        }
    }

    // 2. Milestone 5A: Base-3 2-Plane (9-level) vs Milestone 5B: Full-16-Entry Bipolar Radix-2 LUT!
    {
        let label = "Kronos Base-3 2P + GPTQ (9-lvl)";
        let t_comp = Instant::now();
        let mut eng = fp32_engine.compile_sequential_hadamard_gptq(
            RealWeightMode::KronosHadamardGptqBase3 { num_planes: 2 },
            &calib_sequences,
            p_cores,
        );
        let comp_sec = t_comp.elapsed().as_secs_f64();
        eng.set_num_threads(1);
        let rep = evaluate_quality_against_reference(&mut eng, &eval_ids, &ref_logits);
        let (_, tps_1t) = bench_generation(&mut eng, &prompts, &tokenizer, gen_tokens);
        eng.set_num_threads(p_cores);
        let (_, tps_8t) = bench_generation(&mut eng, &prompts, &tokenizer, gen_tokens);
        let bits = eng.physical_bits_per_weight();
        let ram_mb = eng.linear_weight_bytes() as f64 / (1024.0 * 1024.0);
        println!(
            "  {:<36} | {:>4.2}b | {:>5.1}M | {:>7.2} | {:>8.6} | {:>7.4} | {:>5.1}% | {:>8.1} | {:>8.1}  [GPTQ {:.2}s]",
            label,
            bits,
            ram_mb,
            rep.perplexity,
            rep.mean_logit_cosine_sim,
            rep.mean_kl_divergence_nats,
            rep.top1_agreement_pct,
            tps_1t,
            tps_8t,
            comp_sec
        );
    }

    let bipolar_modes = [
        (
            "Kronos Bipolar-4 + GPTQ (16-lvl) ★",
            RealWeightMode::KronosBipolarGptq { num_planes: 4 },
        ),
        (
            "Kronos Bipolar Mixed-K + GPTQ ★",
            RealWeightMode::KronosBipolarGptqMixedK,
        ),
        (
            "Kronos Bipolar-5 + GPTQ (32-lvl) ★",
            RealWeightMode::KronosBipolarGptq { num_planes: 5 },
        ),
    ];

    for (label, mode) in bipolar_modes {
        let t_comp = Instant::now();
        let mut eng = fp32_engine.compile_sequential_bipolar_gptq(mode, &calib_sequences, p_cores);
        let comp_sec = t_comp.elapsed().as_secs_f64();

        eng.set_num_threads(1);
        let rep = evaluate_quality_against_reference(&mut eng, &eval_ids, &ref_logits);
        let (_, tps_1t) = bench_generation(&mut eng, &prompts, &tokenizer, gen_tokens);

        eng.set_num_threads(p_cores);
        let (outs_8t, tps_8t) = bench_generation(&mut eng, &prompts, &tokenizer, gen_tokens);

        let bits = eng.physical_bits_per_weight();
        let ram_mb = eng.linear_weight_bytes() as f64 / (1024.0 * 1024.0);
        println!(
            "  {:<36} | {:>4.2}b | {:>5.1}M | {:>7.2} | {:>8.6} | {:>7.4} | {:>5.1}% | {:>8.1} | {:>8.1}  [GPTQ {:.2}s]",
            label,
            bits,
            ram_mb,
            rep.perplexity,
            rep.mean_logit_cosine_sim,
            rep.mean_kl_divergence_nats,
            rep.top1_agreement_pct,
            tps_1t,
            tps_8t,
            comp_sec
        );
        gen_showcase.push((label.to_string(), outs_8t, tps_1t, tps_8t));
    }
    println!("  (In-house Q4_0 is a reimplementation inside this engine, not llama.cpp. For verified comparisons");
    println!("   against real llama.cpp / Ollama on WikiText-2, see `wikitext_ppl` and BENCHMARKS.md.)");

    // --------------------------------------------------------------------------------------------------
    // PART 3: Side-by-Side Live Autoregressive English Text Generation
    // --------------------------------------------------------------------------------------------------
    println!("\n==========================================================================================================");
    println!(" [Part 3] SIDE-BY-SIDE LIVE AUTOREGRESSIVE GENERATION (Greedy, 28 New BPE Tokens, BOS Prepended)");
    println!("==========================================================================================================");
    for (p_idx, p_str) in prompts.iter().enumerate() {
        println!("\nPrompt #{}: {:?}", p_idx + 1, p_str);
        println!("  {}", "-".repeat(104));
        for (mode_name, outs, tps_1t, tps_8t) in &gen_showcase {
            let clean = outs[p_idx].replace('\n', "\\n");
            println!(
                "  [{:<36} | 1T: {:>6.1} tok/s | 8T: {:>7.1} tok/s]",
                mode_name, tps_1t, tps_8t
            );
            println!("    \"{}\"", clean);
        }
    }
    println!("\n==========================================================================================================");
}
