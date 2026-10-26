#![forbid(unsafe_code)]
//! Real Trainable & `.k2t`-Compiled Micro-Language & 1,200-Token Recall Model.
//!
//! Builds a word-level vocabulary, specializes `.k2t` Kronecker SwiGLU micro-experts across
//! technical topics, configures $O(1)$ `SsdBlock` induction + n-gram heads, trains the
//! vocabulary readout end-to-end on real `KronosEngine` hidden states, compiles it to
//! BitNet $b1.58$ ternary (`{-1, 0, +1}`), and verifies:
//! 1. Coherent multi-topic English generation with visible topic-to-expert routing.
//! 2. **1,200-token Needle-in-a-Haystack associative recall** with runtime value swapping
//!    (proving true in-context $O(1)$ fast-weight storage rather than static memorization).

use crate::compiler::{compile_bitnet_ternary, compile_swiglu_expert};
use crate::engine::{argmax, KronosConfig, KronosEngine};
use crate::k2t_format::write_compiled_k2t_container;
use crate::mmap_pager::{AsyncFlashPager, K2tMmap};
use crate::prefetch::SubspaceProjector;
use crate::router::ProductKeyRouter;
use crate::ssd::SsdBlock;
use std::collections::HashMap;
use std::io;
use std::path::Path;

/// Technical corpus sentences spanning 4 distinct architectural domains.
pub const DOMAIN_CORPUS: [&str; 4] = [
    "kronos solves the mobile memory wall using kronecker synthesis and ternary lookup tables .",
    "product key routing searches one million micro experts in sublinear square root time .",
    "state space dual blocks fold long context into constant fast weight memory .",
    "lookahead subspace prefetching hides flash latency six layers ahead via mmap pages .",
];

/// Passkey query keys for in-context Needle-in-a-Haystack evaluation.
pub const PASSKEY_KEYS: [&str; 3] = ["vault_alpha", "orbit_key", "reactor_pin"];

/// Candidate passkey values (dynamically bound to any key at runtime in the prompt!).
pub const PASSKEY_VALUES: [&str; 6] = [
    "amber_749",
    "cobalt_318",
    "helios_904",
    "quartz_512",
    "zenith_601",
    "cipher_285",
];

/// Simple deterministic word-level vocabulary padded to a multiple of 16.
#[derive(Clone, Debug)]
pub struct WordVocab {
    pub id_to_word: Vec<String>,
    pub word_to_id: HashMap<String, usize>,
}

impl WordVocab {
    pub fn build() -> Self {
        let mut words = vec!["<pad>".to_string(), "<bos>".to_string()];
        let mut seen = HashMap::new();
        for (idx, w) in words.iter().enumerate() {
            seen.insert(w.clone(), idx);
        }

        for sentence in DOMAIN_CORPUS {
            for tok in sentence.split_whitespace() {
                if !seen.contains_key(tok) {
                    let id = words.len();
                    words.push(tok.to_string());
                    seen.insert(tok.to_string(), id);
                }
            }
        }
        for &k in &PASSKEY_KEYS {
            if !seen.contains_key(k) {
                let id = words.len();
                words.push(k.to_string());
                seen.insert(k.to_string(), id);
            }
        }
        for &v in &PASSKEY_VALUES {
            if !seen.contains_key(v) {
                let id = words.len();
                words.push(v.to_string());
                seen.insert(v.to_string(), id);
            }
        }

        while !words.len().is_multiple_of(16) {
            let filler = format!("<unused_{}>", words.len());
            let id = words.len();
            seen.insert(filler.clone(), id);
            words.push(filler);
        }

        Self {
            id_to_word: words,
            word_to_id: seen,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.split_whitespace()
            .map(|w| *self.word_to_id.get(w).expect("unknown word in demo text"))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .map(|&id| self.id_to_word[id].as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Trains and compiles a real word-level [`KronosEngine`] backed by a `.k2t` file at `k2t_path`.
pub fn build_trained_demo_engine(k2t_path: &Path) -> io::Result<(WordVocab, KronosEngine)> {
    let vocab = WordVocab::build();
    let v_size = vocab.id_to_word.len();

    let cfg = KronosConfig {
        vocab_size: v_size,
        num_layers: 4,
        d_model: 128,
        d_ff: 64,
        num_heads: 4, // head_dim = 32: heads 0,2 = n-gram context; heads 1,3 = long-range induction!
        kronecker_rank: 4,
        router_sub_keys: 64, // 4,096 virtual Cartesian micro-experts per layer
        physical_experts_per_layer: 16,
        top_k_experts: 2,
        lookahead_layers: 2,
        lookahead_rank: 8,
        ram_ring_buffer_pages: 64,
        residual_depth_scale: Some(1.0),
        seed: 0x2026_ABCD,
    };

    let dm = cfg.d_model;
    let dff = cfg.d_ff;
    let dh = dm / cfg.num_heads; // 32

    // ------------------------------------------------------------------------
    // 1. Construct Structured Orthogonal Embeddings
    //    - dims [0..32]   : Word semantic + topic identity (Head 0 SSM context)
    //    - dims [32..64]  : Induction KEY / QUERY subspace (Head 1 shift-induction)
    //    - dims [64..96]  : Secondary semantic representation (Head 2 SSM context)
    //    - dims [96..128] : Induction VALUE subspace (written by Head 1)
    // ------------------------------------------------------------------------
    let mut embeddings = vec![0.0f32; v_size * dm];
    let mut rng = 0x1234_5678_9ABC_DEF0u64;

    let key_ids: Vec<usize> = PASSKEY_KEYS.iter().map(|k| vocab.word_to_id[*k]).collect();
    let val_ids: Vec<usize> = PASSKEY_VALUES
        .iter()
        .map(|v| vocab.word_to_id[*v])
        .collect();

    for tok_id in 0..v_size {
        let emb = &mut embeddings[tok_id * dm..(tok_id + 1) * dm];
        if let Some(k_idx) = key_ids.iter().position(|&id| id == tok_id) {
            // Orthogonal Hadamard-like block in dims [32..64] (8 dims per key -> zero overlap!)
            for d in 0..8 {
                emb[32 + k_idx * 8 + d] = if d % 2 == 0 { 4.0 } else { -4.0 };
            }
            emb[k_idx * 2] = 0.5;
        } else if let Some(v_idx) = val_ids.iter().position(|&id| id == tok_id) {
            // Orthogonal block in dims [96..128] (5 dims per value -> zero overlap across 6 values!)
            for d in 0..5 {
                emb[96 + v_idx * 5 + d] = if d % 2 == 0 { 4.0 } else { -4.0 };
            }
            emb[16 + v_idx] = 0.5;
        } else {
            // Corpus words populate dims [0..32] and [64..96] with zero-sum balanced vectors,
            // plus a positive carrier on dim 31 & dim 95 for SSM shift-register clocking!
            for &base in &[0usize, 64usize] {
                for d in 0..30 {
                    rng = rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let mut z = rng;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                    let u = (z ^ (z >> 31)) as f64 / (u64::MAX as f64);
                    emb[base + d] = ((u as f32) * 2.0 - 1.0) * 1.4;
                }
                emb[base + 31] = 2.5; // Constant positive clock channel for SSM convolution!
            }
            let word = &vocab.id_to_word[tok_id];
            for (topic_idx, sentence) in DOMAIN_CORPUS.iter().enumerate() {
                if sentence.split_whitespace().any(|w| w == word) && word != "." {
                    emb[topic_idx * 2] += 2.2;
                    emb[64 + topic_idx * 2] += 2.2;
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // 2. Configure Multi-Layer SSD Blocks (Ternary Compiled via BitNet b1.58!)
    //    - Heads 0 & 2: Multi-timescale SSM Exponential Context Registers (gamma = 0.60, 0.85)
    //    - Head 1     : Content-Addressable Causal Shift Induction Head (gamma = 0.99998)
    // ------------------------------------------------------------------------
    let mut ssd_blocks = Vec::with_capacity(cfg.num_layers);
    for l in 0..cfg.num_layers {
        let mut block = SsdBlock::new(dm, cfg.num_heads, cfg.seed.wrapping_add(l as u64 * 77));
        let mut wq_dense = vec![0.0f32; dm * dm];
        let mut wk_dense = vec![0.0f32; dm * dm];
        let mut wv_dense = vec![0.0f32; dm * dm];
        let mut wo_dense = vec![0.0f32; dm * dm];

        if l == 0 {
            // Head 0 (dims [0..32], gamma ≈ 0.60) & Head 2 (dims [64..96], gamma ≈ 0.85):
            // Clock Q & K off carrier dim (base + 31) so S_t[*, 0] accumulates sum_{s<=t} gamma^{t-s} V x_s!
            let mut s_rng = 0x55AA_0101u64;
            for &(h_even, in_base, out_base) in &[(0usize, 0usize, 0usize), (2usize, 64usize, 64usize)] {
                let h_off = h_even * dh;
                wq_dense[h_off * dm + (in_base + 31)] = 1.0;
                wk_dense[h_off * dm + (in_base + 31)] = 1.0;

                for r in 0..dh {
                    for c in 0..31 {
                        s_rng = s_rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
                        let v = (((s_rng >> 33) % 3) as i8 - 1) as f32;
                        wv_dense[(h_off + r) * dm + (in_base + c)] = v;
                    }
                    wo_dense[(out_base + r) * dm + (h_off + r)] = 1.0;
                }
            }

            // Odd Head 1 (dims [32..64]): Exact Causal-Shift Induction Associative Head!
            // Binds K_1(x_{t-1}[32..64]) -> V_1(x_t[96..128]) and reads out into dims [96..128]!
            for i in 0..dh {
                wq_dense[(32 + i) * dm + (32 + i)] = 1.0;
                wk_dense[(32 + i) * dm + (32 + i)] = 1.0;
                wv_dense[(32 + i) * dm + (96 + i)] = 1.0;
                wo_dense[(96 + i) * dm + (32 + i)] = 1.0;
            }

            block.w_q = compile_bitnet_ternary(&wq_dense, dm, dm).0;
            block.w_k = compile_bitnet_ternary(&wk_dense, dm, dm).0;
            block.w_v = compile_bitnet_ternary(&wv_dense, dm, dm).0;
            block.w_o = compile_bitnet_ternary(&wo_dense, dm, dm).0;
            block.proj_scale = 3.5;
            block.w_gamma.fill(0.0);
            block.b_gamma = vec![0.4, 11.2, 1.7, 0.0];
        } else {
            // Upper layers mix dims [0..96] lightly while keeping induction channel [96..128] clean!
            let mut u_rng = 0x7788_0000u64 + l as u64;
            for r in 0..dh {
                for c in 0..30 {
                    u_rng = u_rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
                    let v = (((u_rng >> 33) % 3) as i8 - 1) as f32;
                    wv_dense[r * dm + c] = v;
                }
                wo_dense[r * dm + r] = 1.0;
            }
            wq_dense[31] = 1.0;
            wk_dense[31] = 1.0;
            block.enable_induction_shift = false;
            block.w_q = compile_bitnet_ternary(&wq_dense, dm, dm).0;
            block.w_k = compile_bitnet_ternary(&wk_dense, dm, dm).0;
            block.w_v = compile_bitnet_ternary(&wv_dense, dm, dm).0;
            block.w_o = compile_bitnet_ternary(&wo_dense, dm, dm).0;
            block.proj_scale = 0.25;
            block.w_gamma.fill(0.0);
            block.b_gamma = vec![1.1, 0.0, 0.0, 0.0];
        }
        ssd_blocks.push(block);
    }

    // ------------------------------------------------------------------------
    // 3. Specialize Product-Key Routers & Compile SwiGLU Experts into `.k2t`!
    // ------------------------------------------------------------------------
    let mut routers = Vec::with_capacity(cfg.num_layers);
    for l in 0..cfg.num_layers {
        let mut router =
            ProductKeyRouter::new(dm, cfg.router_sub_keys, cfg.seed.wrapping_add(l as u64 * 31));
        // Align sub-keys 0..4 with the 4 domain topic directions so each topic routes to distinct `.k2t` experts!
        let half_d = dm / 2;
        for topic_idx in 0..4 {
            let k1_slot = topic_idx * 3;
            let k2_slot = topic_idx * 5;
            router.codebook_1[k1_slot * half_d..(k1_slot + 1) * half_d].fill(0.0);
            router.codebook_2[k2_slot * half_d..(k2_slot + 1) * half_d].fill(0.0);
            router.codebook_1[k1_slot * half_d + topic_idx * 2] = 2.5;
            router.codebook_2[k2_slot * half_d + topic_idx * 2] = 2.5;
        }
        routers.push(router);
    }

    let mut compiled_experts =
        Vec::with_capacity(cfg.num_layers * cfg.physical_experts_per_layer);
    for l in 0..cfg.num_layers {
        for slot in 0..cfg.physical_experts_per_layer {
            // Structured harmonic dense matrices compiled via Multi-Rank Kronecker Factorization!
            let phase = (l * 17 + slot * 7 + 1) as f32;
            let w_gate: Vec<f32> = (0..dff * dm)
                .map(|idx| {
                    let r = (idx / dm) as f32;
                    let c = (idx % dm) as f32;
                    // Zero out dims [96..128] so FFN experts don't overwrite the passkey value channel!
                    if (idx % dm) >= 96 {
                        0.0
                    } else {
                        ((r * 0.19 + phase).sin() * (c * 0.13).cos()) * 0.5
                    }
                })
                .collect();
            let w_up: Vec<f32> = (0..dff * dm)
                .map(|idx| {
                    if (idx % dm) >= 96 {
                        0.0
                    } else {
                        let r = (idx / dm) as f32;
                        let c = (idx % dm) as f32;
                        ((r * 0.11).cos() * (c * 0.23 + phase).sin()) * 0.5
                    }
                })
                .collect();
            let w_down: Vec<f32> = (0..dm * dff)
                .map(|idx| {
                    let r = idx / dff;
                    if r >= 96 {
                        0.0
                    } else {
                        let c = (idx % dff) as f32;
                        (((r as f32) * 0.17 + phase).cos() * (c * 0.29).sin()) * 0.4
                    }
                })
                .collect();

            compiled_experts.push(compile_swiglu_expert(
                l as u32,
                slot as u64,
                dm,
                dff,
                cfg.kronecker_rank,
                0.08,
                &w_gate,
                &w_up,
                &w_down,
            ));
        }
    }

    write_compiled_k2t_container(
        k2t_path,
        cfg.num_layers as u32,
        cfg.physical_experts_per_layer as u32,
        dm as u32,
        dff as u32,
        cfg.kronecker_rank as u32,
        &compiled_experts,
    )?;

    let mmap = K2tMmap::open(k2t_path)?;
    let pager = AsyncFlashPager::new(mmap, cfg.ram_ring_buffer_pages);

    let lookahead_projectors: Vec<SubspaceProjector> = (0..cfg.num_layers)
        .map(|l| SubspaceProjector::new(dm, cfg.lookahead_rank, cfg.seed + 500 + l as u64))
        .collect();
    let entry_projectors: Vec<SubspaceProjector> = (0..cfg.lookahead_layers)
        .map(|l| SubspaceProjector::new(dm, cfg.lookahead_rank, cfg.seed + 900 + l as u64))
        .collect();

    let dummy_head = compile_bitnet_ternary(&vec![0.0f32; v_size * dm], v_size, dm).0;
    let mut engine = KronosEngine::from_compiled_parts(
        cfg.clone(),
        pager,
        routers,
        lookahead_projectors,
        entry_projectors,
        ssd_blocks,
        embeddings,
        dummy_head,
        1.0,
        vec![0.0f32; v_size],
    );

    // ------------------------------------------------------------------------
    // 4. End-to-End Readout Training + BitNet b1.58 Ternary Compilation!
    //    Collects real `final_normed` states from `KronosEngine` on corpus sequences &
    //    synthetic key-value episodes, trains dense readout via cross-entropy GD, then
    //    quantizes `W_head` to 100% ternary `{-1, 0, +1}` with STE fine-tuning!
    // ------------------------------------------------------------------------
    let mut samples: Vec<(Vec<f32>, usize)> = Vec::new();
    let bos_id = vocab.word_to_id["<bos>"];

    // 4a. Collect real hidden states across all 4 domain corpus sentences
    for sentence in DOMAIN_CORPUS {
        let mut seq = vec![bos_id];
        seq.extend(vocab.encode(sentence));
        engine.reset_context();
        for w_idx in 0..seq.len() - 1 {
            let _ = engine.forward_token(seq[w_idx]).unwrap();
            let h_norm = extract_final_normed(&engine);
            for _ in 0..2 {
                samples.push((h_norm.clone(), seq[w_idx + 1]));
            }
        }
    }

    // 4b. Collect real hidden states on passkey recall episodes across ALL (key, value) pairs
    let filler_tokens = vocab.encode(DOMAIN_CORPUS[0]);
    for &k_str in &PASSKEY_KEYS {
        let k_id = vocab.word_to_id[k_str];
        for &v_str in &PASSKEY_VALUES {
            let v_id = vocab.word_to_id[v_str];
            engine.reset_context();
            let _ = engine.forward_token(bos_id).unwrap();
            let _ = engine.forward_token(k_id).unwrap();
            let _ = engine.forward_token(v_id).unwrap();
            for &ft in &filler_tokens {
                let _ = engine.forward_token(ft).unwrap();
            }
            let _ = engine.forward_token(k_id).unwrap();
            let h_norm = extract_final_normed(&engine);
            for _ in 0..2 {
                samples.push((h_norm.clone(), v_id));
            }
        }
    }

    // 4c. Quantization-Aware Training (Straight-Through Estimator for BitNet b1.58 Ternary!)
    let mut w_dense = vec![0.0f32; v_size * dm];
    let mut bias = vec![-2.0f32; v_size];
    let mut probs = vec![0.0f32; v_size];

    for epoch in 0..200 {
        let lr = if epoch < 120 { 0.45 } else { 0.15 };

        let w_eval: Vec<f32> = if epoch < 40 {
            w_dense.clone()
        } else {
            let (_, beta) = compile_bitnet_ternary(&w_dense, v_size, dm);
            let b_safe = beta.max(1e-4);
            let mean_abs =
                (w_dense.iter().map(|v| v.abs()).sum::<f32>() / (w_dense.len() as f32)).max(1e-4);
            w_dense
                .iter()
                .map(|&w| (w / mean_abs).round().clamp(-1.0, 1.0) * b_safe)
                .collect()
        };

        for (feat, target) in &samples {
            let mut max_l = f32::NEG_INFINITY;
            for (v, (&b_v, w_row)) in bias
                .iter()
                .zip(w_eval.chunks_exact(dm))
                .enumerate()
            {
                let dot: f32 = w_row.iter().zip(feat.iter()).map(|(a, b)| a * b).sum();
                let l = dot + b_v;
                probs[v] = l;
                if l > max_l {
                    max_l = l;
                }
            }
            let mut sum_e = 0.0f32;
            for p in &mut probs {
                *p = (*p - max_l).exp();
                sum_e += *p;
            }
            let inv_sum = 1.0 / sum_e;
            let step = lr / (samples.len() as f32).sqrt();

            for v in 0..v_size {
                let grad = probs[v] * inv_sum - if v == *target { 1.0 } else { 0.0 };
                if grad.abs() < 1e-6 {
                    continue;
                }
                bias[v] -= step * grad * 0.2;
                let w_row = &mut w_dense[v * dm..(v + 1) * dm];
                for (w_ij, &xj) in w_row.iter_mut().zip(feat.iter()) {
                    *w_ij -= step * grad * xj;
                }
            }
        }
    }

    let (ternary_head, beta_scale) = compile_bitnet_ternary(&w_dense, v_size, dm);
    engine.lm_head = ternary_head;
    engine.lm_head_scale = beta_scale;
    engine.lm_head_bias = bias;
    engine.reset_context();

    Ok((vocab, engine))
}

/// Generates text from `prompt_text` until `"."` or `max_words`, returning `(decoded_text, layer0_experts)`.
pub fn generate_visible_sentence(
    vocab: &WordVocab,
    engine: &mut KronosEngine,
    prompt_text: &str,
    max_words: usize,
) -> (String, Vec<usize>) {
    engine.reset_context();
    let bos = vocab.word_to_id["<bos>"];
    let stop_dot = vocab.word_to_id["."];
    let prompt_ids = vocab.encode(prompt_text);

    let _ = engine.forward_token(bos).unwrap();
    let mut traced_experts = Vec::new();
    let mut next_id = 0usize;

    for (idx, &tid) in prompt_ids.iter().enumerate() {
        let trace_opt = if idx == 0 {
            Some(&mut traced_experts)
        } else {
            None
        };
        let logits = engine.forward_token_with_trace(tid, trace_opt).unwrap();
        next_id = argmax(logits);
    }

    let mut out_ids = prompt_ids;
    for _ in 0..max_words {
        out_ids.push(next_id);
        if next_id == stop_dot {
            break;
        }
        let logits = engine.forward_token(next_id).unwrap();
        next_id = argmax(logits);
    }

    (vocab.decode(&out_ids), traced_experts)
}

/// `(query_key, expected_value, predicted_value)` tuple from Needle-in-a-Haystack recall.
pub type RecallOutcome = (String, String, String);

/// `(tokens_seen, ssd_state_bytes)` sample from Needle-in-a-Haystack streaming.
pub type RamSample = (usize, usize);

/// Runs a **Long-Context Needle-in-a-Haystack Recall Test** over `distractor_tokens_target`+ tokens:
/// 1. Injects `bindings` (`[(key, value)]`) at token 0.
/// 2. Streams `distractor_tokens_target`+ tokens of technical prose while recording constant SSD RAM.
/// 3. Queries each `key` and returns `(outcomes, ssd_bytes_timeline)`.
pub fn run_needle_in_haystack_test(
    vocab: &WordVocab,
    engine: &mut KronosEngine,
    bindings: &[(&str, &str)],
    distractor_tokens_target: usize,
) -> (Vec<RecallOutcome>, Vec<RamSample>) {
    engine.reset_context();
    let bos = vocab.word_to_id["<bos>"];
    let _ = engine.forward_token(bos).unwrap();

    // 1. Plant secret passkey pairs at the very beginning of the context!
    for &(k_str, v_str) in bindings {
        let _ = engine.forward_token(vocab.word_to_id[k_str]).unwrap();
        let _ = engine.forward_token(vocab.word_to_id[v_str]).unwrap();
    }

    let mut ram_timeline = Vec::new();
    ram_timeline.push((
        engine.ssd_states[0].tokens_seen,
        engine.total_ssd_state_bytes(),
    ));

    // 2. Flood the engine with > `distractor_tokens_target` tokens of English technical prose!
    let mut distractor_seq = Vec::new();
    for s in DOMAIN_CORPUS {
        distractor_seq.extend(vocab.encode(s));
    }

    while engine.ssd_states[0].tokens_seen < distractor_tokens_target {
        for &tok in &distractor_seq {
            let _ = engine.forward_token(tok).unwrap();
            let seen = engine.ssd_states[0].tokens_seen;
            if seen.is_multiple_of(400) {
                ram_timeline.push((seen, engine.total_ssd_state_bytes()));
            }
            if seen >= distractor_tokens_target {
                break;
            }
        }
    }
    ram_timeline.push((
        engine.ssd_states[0].tokens_seen,
        engine.total_ssd_state_bytes(),
    ));

    // 3. Query each passkey key after 1,200+ tokens!
    let mut results = Vec::with_capacity(bindings.len());
    for &(k_str, expected_v) in bindings {
        let logits = engine.forward_token(vocab.word_to_id[k_str]).unwrap();
        let pred_id = argmax(logits);
        let pred_word = vocab.id_to_word[pred_id].clone();
        results.push((k_str.to_string(), expected_v.to_string(), pred_word));
    }

    (results, ram_timeline)
}

fn extract_final_normed(engine: &KronosEngine) -> Vec<f32> {
    // Reconstruct the exact unit-RMS feature vector that feeds `lm_head`?
    // Even cleaner: expose `last_final_normed(&self) -> &[f32]` on `KronosEngine`!
    engine.last_final_normed().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_trained_k2t_language_generation_and_1200_token_passkey_swap() {
        let tmp_path = std::env::temp_dir().join("kronos_2t_trained_demo_test.k2t");
        let (vocab, mut engine) =
            build_trained_demo_engine(&tmp_path).expect("failed to train & compile demo engine");

        // 1. Verify exact coherent English completion across all 4 domain prompts!
        let prompts = [
            ("kronos solves", DOMAIN_CORPUS[0]),
            ("product key", DOMAIN_CORPUS[1]),
            ("state space", DOMAIN_CORPUS[2]),
            ("lookahead subspace", DOMAIN_CORPUS[3]),
        ];
        for (prompt, expected_full) in prompts {
            let (generated, experts) = generate_visible_sentence(&vocab, &mut engine, prompt, 20);
            assert_eq!(generated, expected_full);
            assert!(!experts.is_empty());
        }

        // 2. Run A: 1,200-token Needle-in-a-Haystack with Assignment A
        let bindings_a = [
            ("vault_alpha", "amber_749"),
            ("orbit_key", "cobalt_318"),
            ("reactor_pin", "helios_904"),
        ];
        let (res_a, ram_a) = run_needle_in_haystack_test(&vocab, &mut engine, &bindings_a, 1_200);
        for (k, exp, got) in &res_a {
            assert_eq!(
                exp, got,
                "Run A failed 1,200-token recall for {k}: expected {exp}, got {got}"
            );
        }
        // Verify strictly 0 bytes of context RAM growth from token 7 to token 1,200!
        let first_ram = ram_a.first().unwrap().1;
        assert!(ram_a.iter().all(|&(_, b)| b == first_ram));

        // 3. Run B: SWAP ALL VALUES AT RUNTIME on the EXACT SAME `.k2t` model!
        let bindings_b = [
            ("vault_alpha", "quartz_512"),
            ("orbit_key", "zenith_601"),
            ("reactor_pin", "cipher_285"),
        ];
        let (res_b, _) = run_needle_in_haystack_test(&vocab, &mut engine, &bindings_b, 1_200);
        for (k, exp, got) in &res_b {
            assert_eq!(
                exp, got,
                "Run B (swapped runtime values!) failed 1,200-token recall for {k}: expected {exp}, got {got}"
            );
        }

        let _ = std::fs::remove_file(&tmp_path);
    }
}
