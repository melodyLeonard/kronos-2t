//! Real Pretrained LLM (`LlamaForCausalLM` / `SmolLM2-135M`) Quality & Speed Engine.
//!
//! Loads real open-weight HuggingFace SafeTensors checkpoints (`BF16` / `F16` / `F32`) and executes:
//! - **Exact Hardware-SIMD `FP32` Reference** (`vfmaq_f32`, single-core & multi-core)
//! - **`llama.cpp` `GGUF Q4_0` Reference** (`4.50` bits/weight `G=32` `Q4_0 x Q8_0` NEON `vmull_s8`)
//! - **`kronos-2t` Hadamard + Sequential `GPTQ` Fused Base-3 Zero-Multiply NEON `LUT-GEMV`**
//!   (`4.25` bits/weight 2-Plane, `4.72` bits/weight Mixed-K, and `6.25` bits/weight 3-Plane)
//!
//! Includes fused `[q_proj; k_proj; v_proj]` and `[gate_proj; up_proj]` matrices, precomputed
//! `RoPE` tables, and persistent P-core `SpinThreadPool` work-stealing execution.

use crate::compiler::{
    compile_global_ternary_as_multiplane, compile_hadamard_gptq_bipolar,
    compile_hadamard_gptq_full_row, compile_hadamard_gptq_nf, compile_hadamard_gptq_vq2,
    compile_multiplane_ternary, FullRowGptqHessian,
};
use crate::hadamard::{fwht64_signed_inplace, ifwht64_signed_inplace};
use crate::mmap_pager::sys::RawMmap;
use crate::prefetch::SubspaceProjector;
use crate::safetensors::{JsonCursor, SafeTensorsStore};
use crate::sparse_moe::{
    fit_subspace_projector, read_moe_expert_f32, read_moe_router_f32, read_moe_shared_expert_f32,
    DenseMoeRouter, MoeExpertOpenPolicy, MoeLayerCore, MoeRouterScoring, MoeRuntimeStats,
    SharedExpertBackend, SharedMoeExpert, SparseMoeRuntime, StreamingK2tmWriter,
};
use crate::simd_lut::{
    bf16_to_f32, f32_to_bf16, nf_sdot_batch_stripe_range, nf_sdot_stripe_range,
    nf_sdot_stripe_range_int8, nf_sdot_stripe_range_raw, nibble4_sdot_batch_stripe_range,
    nibble4_sdot_stripe_range, nibble4_sdot_stripe_range_int8, nibble4_sdot_stripe_range_raw,
    simd_lut_base3_gemv, simd_lut_base3_stripe_range, simd_lut_bipolar_gemv,
    simd_lut_bipolar_stripe_range, simd_lut_gemv_multiplane, vq2_lut_gemv, vq2_lut_stripe_range,
    vq2_lut_stripe_range_raw, vq2_sdot_batch_stripe_range, BipolarActivationLut,
    FusedBase3LutMatrix, FusedBipolarLutMatrix, GgufQ4_0Matrix, Int8ChunkActivation,
    MultiPlaneTernaryMatrix, NfSdotMatrix, Nibble4SdotMatrix, QuantizedActivationLut,
    Vq2ActivationLut, Vq2LutMatrix, NF4_DEFAULT_CODEBOOK_I8, SIMD_ROW_STRIPE,
};
use crate::thread_pool::{physical_p_cores, SpinBarrier, SpinThreadPool};
use std::fs;
use std::io;
use std::path::Path;

/// Hessian off-diagonal shrinkage for the Bipolar / VQ2 GPTQ compile.
const BIPOLAR_GPTQ_DAMP: f64 = 0.01;

/// Cache magic for Bipolar-only engines.
const BIPOLAR_CACHE_MAGIC: &[u8] = b"K2TBIPOLAR1\n";

/// Cache magic for VQ2, NF, and mixed VQ2/NF/Bipolar engines.
const KRONOS2_CACHE_MAGIC: &[u8] = b"K2TKRONOS2\n";

/// Cache magic adding optional fused `qkv_bias` and `BF16` untied embeddings (`Qwen2` / `Qwen2.5-Coder`).
const KRONOS3_CACHE_MAGIC: &[u8] = b"K2TKRONOS3\n";

/// Cache magic adding `NF4` quantized untied embeddings (`-1.08 GiB` wired RAM on `152k`-vocab models).
const KRONOS4_CACHE_MAGIC: &[u8] = b"K2TKRONOS4\n";

/// Cache magic for pinned Sparse-MoE cores paired with `.k2tm` routed expert shards.
const KRONOS5_CACHE_MAGIC: &[u8] = b"K2TKRONOS5\n";

/// Little-endian cursor over a [`RealLlamaEngine::save_bipolar`] cache.
struct CacheReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> CacheReader<'a> {
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let out = self.bytes.get(self.pos..self.pos + n).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "truncated bipolar cache")
        })?;
        self.pos += n;
        Ok(out)
    }

    fn u64(&mut self) -> io::Result<usize> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()) as usize)
    }

    fn f32s(&mut self) -> io::Result<Vec<f32>> {
        let n = self.u64()?;
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Model hyperparameters parsed from HuggingFace `config.json`.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim_override: Option<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub max_seq_len: usize,
    pub sliding_window: Option<usize>,
    pub sink_tokens: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub router_scoring: MoeRouterScoring,
    pub decoder_sparse_step: usize,
    pub mlp_only_layers: Vec<usize>,
    pub sliding_layer_mask: Vec<bool>,
}

impl LlamaConfig {
    pub fn from_config_json(path: &Path, max_seq_len: usize) -> io::Result<Self> {
        let text = fs::read_to_string(path)?;
        let mut cur = JsonCursor::new(&text);
        cur.expect_byte(b'{')
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;

        let mut vocab_size = 49152usize;
        let mut hidden_size = 576usize;
        let mut intermediate_size = 1536usize;
        let mut num_hidden_layers = 30usize;
        let mut num_attention_heads = 9usize;
        let mut num_key_value_heads = 3usize;
        let mut head_dim_override: Option<usize> = None;
        let mut rms_norm_eps = 1e-5f32;
        let mut rope_theta = 100_000.0f32;
        let mut tie_word_embeddings = true;
        let mut raw_sliding_window: Option<usize> = None;
        let mut use_sliding_window: Option<bool> = None;
        let mut explicit_sinks: Option<usize> = None;
        let mut num_experts = 0usize;
        let mut num_experts_per_tok = 0usize;
        let mut moe_intermediate_size = 0usize;
        let mut shared_expert_intermediate_size = 0usize;
        let mut norm_topk_prob = true;
        let mut router_scoring: Option<MoeRouterScoring> = None;
        let mut is_gpt_oss_arch = false;
        let mut decoder_sparse_step = 1usize;
        let mut mlp_only_layers: Vec<usize> = Vec::new();
        let mut sliding_layer_mask: Vec<bool> = Vec::new();

        let parse_opt_usize = |cur: &mut JsonCursor| -> io::Result<Option<usize>> {
            if cur.peek() == Some(b'n') {
                cur.consume_literal(b"null")
                    .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                Ok(None)
            } else {
                let v = cur
                    .parse_u64()
                    .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?
                    as usize;
                Ok((v > 0).then_some(v))
            }
        };
        let parse_bool = |cur: &mut JsonCursor| -> io::Result<bool> {
            if cur.peek() == Some(b't') {
                cur.consume_literal(b"true")
                    .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                Ok(true)
            } else {
                cur.consume_literal(b"false")
                    .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                Ok(false)
            }
        };

        while cur.peek() != Some(b'}') {
            let key = cur
                .parse_string()
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
            cur.expect_byte(b':')
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
            match key.as_str() {
                "architectures" => {
                    if cur.peek() == Some(b'[') {
                        cur.expect_byte(b'[').unwrap();
                        while cur.peek() != Some(b']') {
                            let arch = cur
                                .parse_string()
                                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                            if arch.contains("GptOss") {
                                is_gpt_oss_arch = true;
                            }
                            if cur.peek() == Some(b',') {
                                cur.expect_byte(b',').unwrap();
                            } else {
                                break;
                            }
                        }
                        cur.expect_byte(b']')
                            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    } else {
                        cur.skip_value()
                            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    }
                }
                "vocab_size" => vocab_size = cur.parse_u64().unwrap() as usize,
                "hidden_size" => hidden_size = cur.parse_u64().unwrap() as usize,
                "intermediate_size" => intermediate_size = cur.parse_u64().unwrap() as usize,
                "num_hidden_layers" => num_hidden_layers = cur.parse_u64().unwrap() as usize,
                "num_attention_heads" => num_attention_heads = cur.parse_u64().unwrap() as usize,
                "num_key_value_heads" => num_key_value_heads = cur.parse_u64().unwrap() as usize,
                "head_dim" => head_dim_override = parse_opt_usize(&mut cur)?,
                "rms_norm_eps" => rms_norm_eps = cur.parse_f64().unwrap() as f32,
                "rope_theta" => rope_theta = cur.parse_f64().unwrap() as f32,
                "tie_word_embeddings" => tie_word_embeddings = parse_bool(&mut cur)?,
                "sliding_window" => raw_sliding_window = parse_opt_usize(&mut cur)?,
                "use_sliding_window" => use_sliding_window = Some(parse_bool(&mut cur)?),
                "num_sink_tokens" | "attention_sinks" => {
                    if cur.peek() == Some(b't') || cur.peek() == Some(b'f') {
                        let b = parse_bool(&mut cur)?;
                        explicit_sinks = Some(if b { 4 } else { 0 });
                    } else {
                        explicit_sinks = parse_opt_usize(&mut cur)?.or(Some(0));
                    }
                }
                "num_experts" | "num_local_experts" => {
                    num_experts = parse_opt_usize(&mut cur)?.unwrap_or(0);
                }
                "num_experts_per_tok" | "experts_per_token" | "moe_topk" => {
                    num_experts_per_tok = parse_opt_usize(&mut cur)?.unwrap_or(0);
                }
                "moe_intermediate_size" => {
                    moe_intermediate_size = parse_opt_usize(&mut cur)?.unwrap_or(0);
                }
                "shared_expert_intermediate_size" => {
                    shared_expert_intermediate_size = parse_opt_usize(&mut cur)?.unwrap_or(0);
                }
                "norm_topk_prob" => norm_topk_prob = parse_bool(&mut cur)?,
                "scoring_func" => {
                    let s = cur
                        .parse_string()
                        .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    router_scoring = Some(match s.as_str() {
                        "sigmoid" => MoeRouterScoring::Sigmoid,
                        "topk_softmax" => MoeRouterScoring::TopkSoftmax,
                        _ => MoeRouterScoring::Softmax,
                    });
                }
                "decoder_sparse_step" => {
                    decoder_sparse_step = parse_opt_usize(&mut cur)?.unwrap_or(1).max(1);
                }
                "mlp_only_layers" => {
                    if cur.peek() == Some(b'[') {
                        cur.expect_byte(b'[').unwrap();
                        while cur.peek() != Some(b']') {
                            let idx = cur
                                .parse_u64()
                                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?
                                as usize;
                            mlp_only_layers.push(idx);
                            if cur.peek() == Some(b',') {
                                cur.expect_byte(b',').unwrap();
                            } else {
                                break;
                            }
                        }
                        cur.expect_byte(b']')
                            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    } else {
                        cur.skip_value()
                            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    }
                }
                "layer_types" => {
                    if cur.peek() == Some(b'[') {
                        cur.expect_byte(b'[').unwrap();
                        while cur.peek() != Some(b']') {
                            let lt = cur
                                .parse_string()
                                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                            sliding_layer_mask.push(lt.contains("sliding"));
                            if cur.peek() == Some(b',') {
                                cur.expect_byte(b',').unwrap();
                            } else {
                                break;
                            }
                        }
                        cur.expect_byte(b']')
                            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    } else {
                        cur.skip_value()
                            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
                    }
                }
                _ => cur
                    .skip_value()
                    .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?,
            }
            if cur.peek() == Some(b',') {
                cur.expect_byte(b',').unwrap();
            } else {
                break;
            }
        }

        let all_sliding = sliding_layer_mask.is_empty() || sliding_layer_mask.iter().all(|&b| b);
        let sliding_window = if use_sliding_window == Some(false) || !all_sliding {
            None
        } else {
            raw_sliding_window.filter(|&w| w >= 8)
        };
        let sink_tokens = explicit_sinks.unwrap_or(if sliding_window.is_some() { 4 } else { 0 });
        if num_experts > 1 && moe_intermediate_size == 0 {
            moe_intermediate_size = intermediate_size;
        }
        if intermediate_size == 0 && moe_intermediate_size > 0 {
            intermediate_size = moe_intermediate_size;
        }
        let resolved_scoring = router_scoring.unwrap_or(if is_gpt_oss_arch {
            MoeRouterScoring::TopkSoftmax
        } else {
            MoeRouterScoring::Softmax
        });

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim_override,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            max_seq_len,
            sliding_window,
            sink_tokens,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            norm_topk_prob,
            router_scoring: resolved_scoring,
            decoder_sparse_step,
            mlp_only_layers,
            sliding_layer_mask,
        })
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim()
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
    }

    #[inline]
    pub fn is_moe(&self) -> bool {
        self.num_experts > 1 && self.num_experts_per_tok > 0
    }

    #[inline]
    pub fn eff_moe_inter(&self) -> usize {
        if self.moe_intermediate_size > 0 {
            self.moe_intermediate_size
        } else {
            self.intermediate_size
        }
    }

    #[inline]
    pub fn layer_is_moe(&self, layer_idx: usize) -> bool {
        self.is_moe()
            && !self.mlp_only_layers.contains(&layer_idx)
            && (layer_idx + 1).is_multiple_of(self.decoder_sparse_step.max(1))
    }

    #[inline]
    pub fn num_moe_layers(&self) -> usize {
        (0..self.num_hidden_layers)
            .filter(|&l| self.layer_is_moe(l))
            .count()
    }
}

/// Weight compilation & execution mode for real pretrained checkpoints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealWeightMode {
    /// Exact `FP32` reference executed via hardware NEON `vfmaq_f32` SIMD dot products.
    ExactFp32,
    /// Naive single-scale BitNet `1.58`-bit PTQ (`1` scalar $\beta$ per matrix, zero retraining).
    GlobalTernary1Plane,
    /// Uncalibrated Group-64 Exact $L_2$-Optimal Multi-Plane Ternary (`num_planes` $\in \{1..=4\}$).
    Group64Ternary { num_planes: usize },
    /// **Activation-Aware Equilibrated (`AWQ-Smooth`)** Group-64 Multi-Plane Ternary.
    AwqGroup64Ternary { num_planes: usize },
    /// Standard **`llama.cpp` `GGUF Q4_0`** (`4.50` bits/weight, `G=32`, `Q4_0 x Q8_0` NEON SIMD).
    GgufQ4_0,
    /// **`kronos-2t` Hadamard + Sequential `GPTQ` Fused Base-3 LUT** (`4.25b` at `2` planes, `6.25b` at `3` planes).
    KronosHadamardGptqBase3 { num_planes: usize },
    /// **`kronos-2t` Hadamard + Sequential `GPTQ` Mixed-K (`5.07` bits/weight)**:
    KronosHadamardGptqMixedK,
    /// **`kronos-2t` Full-16-Entry Bipolar Radix-2 LUT + Hadamard + Sequential `GPTQ`**
    /// (`3` planes = `3.25b`, **`4` planes = `4.25b` `16`-level**, **`5` planes = `5.25b` `32`-level**,
    /// **`6` planes = `6.25b` `64`-level**).
    KronosBipolarGptq { num_planes: usize },
    /// **`kronos-2t` Bipolar Mixed-K (`4.66` bits/weight, `74.6 MB`)**:
    /// `4-Plane` (`4.25b`, `16` levels) on `qkv_proj`/`o_proj`/`gate_up_proj` + `5-Plane` (`5.25b`,
    /// `32` levels) on `down_proj` & `lm_head` (beats `GGUF Q4_K_M` at lower bit-width!).
    KronosBipolarGptqMixedK,
    /// Bipolar with per-matrix plane counts.
    KronosBipolarGptqPlanes {
        qkv: usize,
        o: usize,
        gate_up: usize,
        down: usize,
        head: usize,
    },
    /// 2D vector-quantized GPTQ ([`Vq2LutMatrix`]). `scale_group_chunks` 1, 2, 4 give 2.25, 2.125,
    /// 2.0625 bits/weight.
    KronosVq2Gptq { scale_group_chunks: usize },
    /// VQ2 FFN with `attn_head_planes`-plane Bipolar attention and LM head.
    KronosVq2MixedGptq {
        attn_head_planes: usize,
        scale_group_chunks: usize,
    },
    /// Lloyd-Max NF5/NF4/NF3 with GPTQ. `scale_group_chunks` 1 or 2 (`g128`).
    KronosNfGptq {
        bits: usize,
        scale_group_chunks: usize,
    },
    /// `ffn_bits` NF FFN with `attn_head_bits` NF attention and LM head.
    KronosNfMixedGptq {
        attn_head_bits: usize,
        ffn_bits: usize,
        scale_group_chunks: usize,
    },
    /// Per-matrix Lloyd-Max NF3/NF4/NF5 bit allocation with GPTQ (`scale_group_chunks` 1 = G64, 2 = G128).
    KronosNfPlanesGptq {
        qkv: usize,
        o: usize,
        gate_up: usize,
        down: usize,
        head: usize,
        scale_group_chunks: usize,
    },
}

enum LinearBackend {
    Empty,
    Fp32 {
        rows: usize,
        cols: usize,
        weights: Vec<f32>,
    },
    LutMultiPlane(MultiPlaneTernaryMatrix),
    GgufQ4_0(GgufQ4_0Matrix),
    FusedBase3(FusedBase3LutMatrix),
    FusedBipolar(FusedBipolarLutMatrix),
    /// Bipolar weights repacked for `SDOT`.
    Nibble4(Nibble4SdotMatrix),
    /// Lloyd-Max NF4/NF3 weights for `SDOT`.
    NfSdot(NfSdotMatrix),
    /// 2-bit vector-quantized weights.
    Vq2(Vq2LutMatrix),
}

impl LinearBackend {
    fn from_f32_slice(
        weights: &[f32],
        rows: usize,
        cols: usize,
        mode: RealWeightMode,
        col_scales: Option<&[f32]>,
    ) -> (Self, f32) {
        let scaled_buf: Option<Vec<f32>> = col_scales.map(|scales| {
            assert_eq!(scales.len(), cols);
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let src = &weights[r * cols..(r + 1) * cols];
                let dst = &mut out[r * cols..(r + 1) * cols];
                for ((d, &w), &s) in dst.iter_mut().zip(src.iter()).zip(scales.iter()) {
                    *d = w * s;
                }
            }
            out
        });
        let w_ref = scaled_buf.as_deref().unwrap_or(weights);

        match mode {
            RealWeightMode::ExactFp32 => (
                Self::Fp32 {
                    rows,
                    cols,
                    weights: w_ref.to_vec(),
                },
                0.0,
            ),
            RealWeightMode::GlobalTernary1Plane => {
                let (mat, err) = compile_global_ternary_as_multiplane(w_ref, rows, cols);
                (Self::LutMultiPlane(mat), err)
            }
            RealWeightMode::Group64Ternary { num_planes }
            | RealWeightMode::AwqGroup64Ternary { num_planes } => {
                let (mat, err) = compile_multiplane_ternary(w_ref, rows, cols, num_planes);
                (Self::LutMultiPlane(mat), err)
            }
            RealWeightMode::GgufQ4_0 => (Self::GgufQ4_0(GgufQ4_0Matrix::from_f32(w_ref, rows, cols)), 0.0),
            RealWeightMode::KronosHadamardGptqBase3 { num_planes } => (
                Self::FusedBase3(compile_hadamard_gptq_full_row(
                    w_ref, rows, cols, num_planes, None, 4,
                )),
                0.0,
            ),
            RealWeightMode::KronosHadamardGptqMixedK => (
                Self::FusedBase3(compile_hadamard_gptq_full_row(w_ref, rows, cols, 2, None, 4)),
                0.0,
            ),
            RealWeightMode::KronosBipolarGptq { num_planes } => (
                Self::FusedBipolar(compile_hadamard_gptq_bipolar(
                    w_ref, rows, cols, num_planes, None, 4,
                )),
                0.0,
            ),
            RealWeightMode::KronosBipolarGptqMixedK
            | RealWeightMode::KronosBipolarGptqPlanes { .. } => (
                Self::FusedBipolar(compile_hadamard_gptq_bipolar(w_ref, rows, cols, 4, None, 4)),
                0.0,
            ),
            RealWeightMode::KronosVq2Gptq { scale_group_chunks }
            | RealWeightMode::KronosVq2MixedGptq { scale_group_chunks, .. } => (
                Self::Vq2(compile_hadamard_gptq_vq2(
                    w_ref,
                    rows,
                    cols,
                    scale_group_chunks,
                    None,
                    4,
                )),
                0.0,
            ),
            RealWeightMode::KronosNfGptq {
                bits,
                scale_group_chunks,
            } => (
                Self::NfSdot(compile_hadamard_gptq_nf(
                    w_ref,
                    rows,
                    cols,
                    bits,
                    scale_group_chunks,
                    None,
                    4,
                )),
                0.0,
            ),
            RealWeightMode::KronosNfMixedGptq {
                attn_head_bits,
                scale_group_chunks,
                ..
            } => (
                Self::NfSdot(compile_hadamard_gptq_nf(
                    w_ref,
                    rows,
                    cols,
                    attn_head_bits,
                    scale_group_chunks,
                    None,
                    4,
                )),
                0.0,
            ),
            RealWeightMode::KronosNfPlanesGptq {
                qkv,
                scale_group_chunks,
                ..
            } => (
                Self::NfSdot(compile_hadamard_gptq_nf(
                    w_ref,
                    rows,
                    cols,
                    qkv,
                    scale_group_chunks,
                    None,
                    4,
                )),
                0.0,
            ),
        }
    }

    fn as_fp32_slice(&self) -> &[f32] {
        match self {
            Self::Fp32 { weights, .. } => weights,
            _ => panic!("expected ExactFp32 weights"),
        }
    }

    fn physical_bytes(&self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Fp32 { weights, .. } => weights.len() * std::mem::size_of::<f32>(),
            Self::LutMultiPlane(m) => m.physical_bytes(),
            Self::GgufQ4_0(m) => m.physical_bytes(),
            Self::FusedBase3(m) => m.physical_bytes(),
            Self::FusedBipolar(m) => m.physical_bytes(),
            Self::Nibble4(m) => m.physical_bytes(),
            Self::NfSdot(m) => m.physical_bytes(),
            Self::Vq2(m) => m.physical_bytes(),
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn forward_exec(
        &self,
        x: &[f32],
        qlut: Option<&QuantizedActivationLut>,
        blut: Option<&BipolarActivationLut>,
        vlut: Option<&Vq2ActivationLut>,
        q8_vals: &[i8],
        q8_scales: &[f32],
        pool: Option<&SpinThreadPool>,
        out: &mut [f32],
    ) {
        match self {
            Self::Empty => panic!("LinearBackend::Empty executed on dense FFN path"),
            Self::Fp32 {
                rows,
                cols,
                weights,
            } => {
                let r_tot = *rows;
                let c_tot = *cols;
                debug_assert_eq!(x.len(), c_tot);
                debug_assert_eq!(out.len(), r_tot);
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && r_tot >= 128) {
                    let grain = (r_tot / (p.num_threads() * 2)).max(32);
                    let out_addr = out.as_mut_ptr() as usize;
                    p.parallel_for(r_tot, grain, |r_start, r_end| {
                        // SAFETY: Disjoint row sub-ranges `[r_start, r_end)` within `0..r_tot`.
                        unsafe {
                            let dst_ptr = out_addr as *mut f32;
                            for r in r_start..r_end {
                                let row = &weights[r * c_tot..(r + 1) * c_tot];
                                *dst_ptr.add(r) = dot_f32_fast(row, x);
                            }
                        }
                    });
                } else {
                    fp32_simd_gemv(weights, x, r_tot, c_tot, out);
                }
            }
            Self::LutMultiPlane(mat) => {
                let lut = qlut.expect("QuantizedActivationLut required for LutMultiPlane");
                simd_lut_gemv_multiplane(mat, lut, out);
            }
            Self::GgufQ4_0(mat) => {
                let r_tot = mat.rows;
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && r_tot >= 128) {
                    let grain = (r_tot / (p.num_threads() * 2)).max(32);
                    let out_addr = out.as_mut_ptr() as usize;
                    p.parallel_for(r_tot, grain, |r_start, r_end| {
                        // SAFETY: Disjoint row sub-ranges `[r_start, r_end)` within `0..r_tot`.
                        unsafe {
                            mat.forward_row_range(
                                q8_vals,
                                q8_scales,
                                r_start,
                                r_end,
                                out_addr as *mut f32,
                            );
                        }
                    });
                } else {
                    // SAFETY: Single-threaded full range `0..r_tot`.
                    unsafe {
                        mat.forward_row_range(q8_vals, q8_scales, 0, r_tot, out.as_mut_ptr());
                    }
                }
            }
            Self::FusedBase3(mat) => {
                let lut = qlut.expect("QuantizedActivationLut required for FusedBase3");
                let n_stripes = mat.num_stripes;
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && n_stripes >= 16) {
                    let grain = (n_stripes / (p.num_threads() * 2)).clamp(4, 128);
                    let out_addr = out.as_mut_ptr() as usize;
                    p.parallel_for(n_stripes, grain, |s_start, s_end| {
                        // SAFETY: Disjoint 16-row SIMD stripes `[s_start, s_end)` write exclusively
                        // to disjoint `out` slices `[s_start*16 .. min(s_end*16, rows)]`.
                        unsafe {
                            simd_lut_base3_stripe_range(
                                mat,
                                lut,
                                s_start,
                                s_end,
                                out_addr as *mut f32,
                                false,
                            );
                        }
                    });
                } else {
                    simd_lut_base3_gemv(mat, lut, out);
                }
            }
            Self::FusedBipolar(mat) => {
                let lut = blut.expect("BipolarActivationLut required for FusedBipolar");
                let n_stripes = mat.num_stripes;
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && n_stripes >= 16) {
                    let grain = (n_stripes / (p.num_threads() * 2)).clamp(4, 128);
                    let out_addr = out.as_mut_ptr() as usize;
                    p.parallel_for(n_stripes, grain, |s_start, s_end| {
                        // SAFETY: Disjoint 16-row SIMD stripes `[s_start, s_end)` write exclusively
                        // to disjoint `out` slices `[s_start*16 .. min(s_end*16, rows)]`.
                        unsafe {
                            simd_lut_bipolar_stripe_range(
                                mat,
                                lut,
                                s_start,
                                s_end,
                                out_addr as *mut f32,
                            );
                        }
                    });
                } else {
                    simd_lut_bipolar_gemv(mat, lut, out);
                }
            }
            Self::Nibble4(mat) => {
                let act = blut.expect("BipolarActivationLut required for Nibble4");
                let n_stripes = mat.num_stripes;
                let out_addr = out.as_mut_ptr() as usize;
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && n_stripes >= 16) {
                    let grain = ((n_stripes / (p.num_threads() * 2)) & !1).clamp(2, 128);
                    p.parallel_for(n_stripes, grain, |s_start, s_end| {
                        // SAFETY: Disjoint 16-row stripes write disjoint `out` slices.
                        unsafe {
                            nibble4_sdot_stripe_range(mat, act, s_start, s_end, out_addr as *mut f32);
                        }
                    });
                } else {
                    // SAFETY: Single-threaded full stripe range.
                    unsafe { nibble4_sdot_stripe_range(mat, act, 0, n_stripes, out_addr as *mut f32) };
                }
            }
            Self::NfSdot(mat) => {
                let act = blut.expect("BipolarActivationLut required for NfSdot");
                let n_stripes = mat.num_stripes;
                let out_addr = out.as_mut_ptr() as usize;
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && n_stripes >= 16) {
                    let grain = ((n_stripes / (p.num_threads() * 2)) & !1).clamp(2, 128);
                    p.parallel_for(n_stripes, grain, |s_start, s_end| {
                        // SAFETY: Disjoint 16-row stripes write disjoint `out` slices.
                        unsafe {
                            nf_sdot_stripe_range(mat, act, s_start, s_end, out_addr as *mut f32);
                        }
                    });
                } else {
                    // SAFETY: Single-threaded full stripe range.
                    unsafe { nf_sdot_stripe_range(mat, act, 0, n_stripes, out_addr as *mut f32) };
                }
            }
            Self::Vq2(mat) => {
                let lut = vlut.expect("Vq2ActivationLut required for Vq2");
                let n_stripes = mat.num_stripes;
                let out_addr = out.as_mut_ptr() as usize;
                if let Some(p) = pool.filter(|p| p.num_threads() > 1 && n_stripes >= 16) {
                    let grain = ((n_stripes / (p.num_threads() * 2)) & !3).clamp(4, 128);
                    p.parallel_for(n_stripes, grain, |s_start, s_end| {
                        // SAFETY: Disjoint 16-row stripes write disjoint `out` slices.
                        unsafe {
                            vq2_lut_stripe_range(mat, lut, s_start, s_end, out_addr as *mut f32);
                        }
                    });
                } else {
                    vq2_lut_gemv(mat, lut, out);
                }
            }
        }
    }
}

#[inline]
fn fp32_simd_gemv(weights: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(weights.len(), rows * cols);
    for (dst, row) in out.iter_mut().zip(weights.chunks_exact(cols)) {
        *dst = dot_f32_fast(row, x);
    }
}

#[inline]
fn dot_f32_fast(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and `a.len() == b.len()`.
        unsafe { dot_f32_neon_16(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = [0.0f32; 4];
        let mut ac = a.chunks_exact(4);
        let mut bc = b.chunks_exact(4);
        for (va, vb) in (&mut ac).zip(&mut bc) {
            acc[0] += va[0] * vb[0];
            acc[1] += va[1] * vb[1];
            acc[2] += va[2] * vb[2];
            acc[3] += va[3] * vb[3];
        }
        let mut sum = (acc[0] + acc[1]) + (acc[2] + acc[3]);
        for (&x, &y) in ac.remainder().iter().zip(bc.remainder().iter()) {
            sum += x * y;
        }
        sum
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_neon_16(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;

    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);

    while i + 16 <= n {
        // SAFETY: `i + 16 <= n` holds for both `a` and `b`.
        unsafe {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        }
        i += 16;
    }

    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        // SAFETY: `i < n`.
        unsafe {
            sum += (*pa.add(i)) * (*pb.add(i));
        }
        i += 1;
    }
    sum
}

struct LlamaLayer {
    input_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    o_inv_scale: Option<Vec<f32>>,
    down_inv_scale: Option<Vec<f32>>,
    /// Optional fused `[q_bias; k_bias; v_bias]` (`Qwen2` / `Qwen2.5-Coder`).
    qkv_bias: Option<Vec<f32>>,
    /// Optional `o_proj.bias` (`GptOss`).
    o_bias: Option<Vec<f32>>,
    /// Optional per-head RMSNorm `[head_dim]` weights (`Qwen3Moe`).
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    /// Fused `[q_proj; k_proj; v_proj]` (`[q_dim + 2 * kv_dim, hidden_size]`).
    qkv_proj: LinearBackend,
    o_proj: LinearBackend,
    /// Fused `[gate_proj; up_proj]` (`[2 * intermediate_size, hidden_size]`), or `Empty` on MoE layers.
    gate_up_proj: LinearBackend,
    /// `gate_up_proj` rows interleaved in 64-row blocks `[gate_64(ic), up_64(ic)]`.
    gate_up_interleaved: bool,
    down_proj: LinearBackend,
    moe: Option<MoeLayerCore>,
}

/// Per-layer and final-head channel activation statistics gathered on a calibration passage.
pub struct ActivationCalibration {
    attn_in_scales: Vec<Vec<f32>>,
    o_in_scales: Vec<Vec<f32>>,
    mlp_in_scales: Vec<Vec<f32>>,
    down_in_scales: Vec<Vec<f32>>,
    head_in_scales: Vec<f32>,
}

/// Reusable zero-allocation batch workspace for [`RealLlamaEngine::forward_prompt`] and speculative verification.
#[derive(Default)]
struct PromptScratch {
    hidden: Vec<f32>,
    acts_h: Vec<BipolarActivationLut>,
    acts_q: Vec<BipolarActivationLut>,
    acts_i: Vec<BipolarActivationLut>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate_up: Vec<f32>,
    sw: Vec<f32>,
    moe_norm: Vec<f32>,
    scores: Vec<f32>,
}

impl PromptScratch {
    #[allow(clippy::too_many_arguments)]
    fn ensure_capacity(
        &mut self,
        n: usize,
        cfg: &LlamaConfig,
        has_dense_ffn: bool,
        qkv_rows: usize,
        lut_h: bool,
        lut_q: bool,
        lut_i: bool,
        need_fp32_scores: bool,
    ) {
        let h = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let inter = cfg.intermediate_size.max(64);
        let kv_group = (cfg.num_attention_heads / cfg.num_key_value_heads).max(1);
        let score_stride = if cfg.head_dim() == 64 {
            cfg.max_seq_len
        } else {
            kv_group * cfg.max_seq_len
        };
        if self.hidden.len() < n * h {
            self.hidden.resize(n * h, 0.0);
        }
        if self.qkv.len() < n * qkv_rows {
            self.qkv.resize(n * qkv_rows, 0.0);
        }
        if self.attn.len() < n * q_dim {
            self.attn.resize(n * q_dim, 0.0);
        }
        if self.proj.len() < n * h {
            self.proj.resize(n * h, 0.0);
        }
        if cfg.is_moe() && self.moe_norm.len() < n * h {
            self.moe_norm.resize(n * h, 0.0);
        }
        if has_dense_ffn {
            if self.gate_up.len() < n * 2 * inter {
                self.gate_up.resize(n * 2 * inter, 0.0);
            }
            if self.sw.len() < n * inter {
                self.sw.resize(n * inter, 0.0);
            }
        }
        if need_fp32_scores && self.scores.len() < n * score_stride {
            self.scores.resize(n * score_stride, 0.0);
        }
        while self.acts_h.len() < n {
            let mut b = BipolarActivationLut::new_zeroed(h);
            b.lut_enabled = lut_h;
            b.int8 = Some(Int8ChunkActivation::new_zeroed(h));
            self.acts_h.push(b);
        }
        for b in &mut self.acts_h[..n] {
            b.lut_enabled = lut_h;
        }
        while self.acts_q.len() < n {
            let mut b = BipolarActivationLut::new_zeroed(q_dim);
            b.lut_enabled = lut_q;
            b.int8 = Some(Int8ChunkActivation::new_zeroed(q_dim));
            self.acts_q.push(b);
        }
        for b in &mut self.acts_q[..n] {
            b.lut_enabled = lut_q;
        }
        if has_dense_ffn {
            while self.acts_i.len() < n {
                let mut b = BipolarActivationLut::new_zeroed(inter);
                b.lut_enabled = lut_i;
                b.int8 = Some(Int8ChunkActivation::new_zeroed(inter));
                self.acts_i.push(b);
            }
            for b in &mut self.acts_i[..n] {
                b.lut_enabled = lut_i;
            }
        }
    }
}

const KV_RING_BATCH_SLACK: usize = 512;

#[derive(Clone, Copy)]
struct QuantKvRawPtrs {
    num_kv_heads: usize,
    head_dim: usize,
    cap_seq_len: usize,
    bits: u8,
    window_size: usize,
    sink_tokens: usize,
    ring_len: usize,
    k_q8: usize,
    v_q8: usize,
    k_scales: usize,
    v_scales: usize,
    k_q4: usize,
    v_q4: usize,
    k_s16: usize,
    v_s16: usize,
}

impl QuantKvRawPtrs {
    #[inline(always)]
    fn slot_of_pos(self, pos: usize) -> usize {
        if self.ring_len == 0 || pos < self.sink_tokens {
            pos
        } else {
            self.sink_tokens + ((pos - self.sink_tokens) % self.ring_len)
        }
    }

    #[inline]
    unsafe fn store_k(self, layer_idx: usize, kvh: usize, pos: usize, head_s: &[f32]) {
        let slot = (layer_idx * self.num_kv_heads + kvh) * self.cap_seq_len + self.slot_of_pos(pos);
        let hd = self.head_dim;
        // SAFETY: `slot` is inside `cap_seq_len` for `(layer_idx, kvh)` and buffers match `self.bits`.
        unsafe {
            if self.bits == 8 {
                store_hadamard_q8_head_raw(
                    head_s,
                    hd,
                    (self.k_q8 as *mut i8).add(slot * hd),
                    (self.k_scales as *mut f32).add(slot),
                );
            } else {
                store_hadamard_q4_head_raw(
                    head_s,
                    hd,
                    (self.k_q4 as *mut u8).add(slot * (hd / 2)),
                    (self.k_s16 as *mut u16).add(slot * (hd / 64)),
                );
            }
        }
    }

    #[inline]
    unsafe fn store_v(self, layer_idx: usize, kvh: usize, pos: usize, head_s: &[f32]) {
        let slot = (layer_idx * self.num_kv_heads + kvh) * self.cap_seq_len + self.slot_of_pos(pos);
        let hd = self.head_dim;
        // SAFETY: `slot` is inside `cap_seq_len` for `(layer_idx, kvh)` and buffers match `self.bits`.
        unsafe {
            if self.bits == 8 {
                store_hadamard_q8_head_raw(
                    head_s,
                    hd,
                    (self.v_q8 as *mut i8).add(slot * hd),
                    (self.v_scales as *mut f32).add(slot),
                );
            } else {
                store_hadamard_q4_head_raw(
                    head_s,
                    hd,
                    (self.v_q4 as *mut u8).add(slot * (hd / 2)),
                    (self.v_s16 as *mut u16).add(slot * (hd / 64)),
                );
            }
        }
    }

    #[inline]
    unsafe fn eval_gqa_group(
        self,
        layer_idx: usize,
        kvh: usize,
        attn_scale: f32,
        queries: &mut [GqaQueryRef<'_>],
    ) {
        if queries.is_empty() {
            return;
        }
        let base = (layer_idx * self.num_kv_heads + kvh) * self.cap_seq_len;
        let hd = self.head_dim;
        if self.bits == 8 && self.window_size == 0 {
            let max_seq = queries.iter().map(|q| q.seq_len).max().unwrap_or(0);
            // SAFETY: `base + max_seq <= num_layers * num_kv_heads * cap_seq_len`.
            unsafe {
                let k_q8_h =
                    std::slice::from_raw_parts((self.k_q8 as *const i8).add(base * hd), max_seq * hd);
                let ks_h =
                    std::slice::from_raw_parts((self.k_scales as *const f32).add(base), max_seq);
                let v_q8_h =
                    std::slice::from_raw_parts((self.v_q8 as *const i8).add(base * hd), max_seq * hd);
                let vs_h =
                    std::slice::from_raw_parts((self.v_scales as *const f32).add(base), max_seq);
                causal_gqa_group_q8(hd, k_q8_h, ks_h, v_q8_h, vs_h, attn_scale, queries);
            }
        } else {
            // SAFETY: caller upholds `self` pointers and `cap_seq_len` capacity.
            unsafe {
                causal_gqa_group_q4_or_window(self, base, attn_scale, queries);
            }
        }
    }
}

/// KV-head-major Hadamard `INT8` / `INT4` (`NF4`) KV cache with optional `O(1)` sliding-window + sink ring buffer.
struct QuantizedKvCache {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    bits: u8,
    window_size: usize,
    sink_tokens: usize,
    ring_len: usize,
    max_seq_len: usize,
    cap_seq_len: usize,
    k_q8: Vec<i8>,
    v_q8: Vec<i8>,
    k_scales: Vec<f32>,
    v_scales: Vec<f32>,
    k_q4: Vec<u8>,
    v_q4: Vec<u8>,
    k_s16: Vec<u16>,
    v_s16: Vec<u16>,
}

impl QuantizedKvCache {
    #[allow(dead_code)]
    fn new(cfg: &LlamaConfig) -> Self {
        Self::new_with_opts(cfg, 8, cfg.sliding_window, cfg.sink_tokens)
    }

    fn new_with_opts(
        cfg: &LlamaConfig,
        bits: u8,
        sliding_window: Option<usize>,
        sink_tokens: usize,
    ) -> Self {
        assert!(bits == 8 || bits == 4, "KV cache bits must be 4 or 8 (got {bits})");
        let head_dim = cfg.head_dim();
        assert!(
            head_dim == 64 || head_dim == 128,
            "Hadamard KV-cache requires head_dim == 64 or 128 (got {head_dim})"
        );
        let (window_size, eff_sinks, ring_len, cap_seq_len) =
            if let Some(w) = sliding_window.filter(|&w| w >= 8 && w + KV_RING_BATCH_SLACK < cfg.max_seq_len) {
                let s = sink_tokens.min(w / 2);
                let rl = (w - s) + KV_RING_BATCH_SLACK;
                (w, s, rl, s + rl)
            } else {
                let init_cap = crate::memory_budget::MemoryBudgetProfile::active()
                    .initial_kv_cap_tokens
                    .clamp(64, 2048);
                (0, 0, 0, cfg.max_seq_len.min(init_cap))
            };
        let slots = cfg.num_hidden_layers * cfg.num_key_value_heads * cap_seq_len;
        let (k_q8, v_q8, k_scales, v_scales, k_q4, v_q4, k_s16, v_s16) = if bits == 8 {
            (
                vec![0i8; slots * head_dim],
                vec![0i8; slots * head_dim],
                vec![0.0f32; slots],
                vec![0.0f32; slots],
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        } else {
            let half_hd = head_dim / 2;
            let gph64 = head_dim / 64;
            (
                Vec::new(),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                vec![0x88u8; slots * half_hd],
                vec![0x88u8; slots * half_hd],
                vec![0u16; slots * gph64],
                vec![0u16; slots * gph64],
            )
        };
        Self {
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            bits,
            window_size,
            sink_tokens: eff_sinks,
            ring_len,
            max_seq_len: cfg.max_seq_len,
            cap_seq_len,
            k_q8,
            v_q8,
            k_scales,
            v_scales,
            k_q4,
            v_q4,
            k_s16,
            v_s16,
        }
    }

    #[inline]
    fn raw_ptrs(&self) -> QuantKvRawPtrs {
        QuantKvRawPtrs {
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            cap_seq_len: self.cap_seq_len,
            bits: self.bits,
            window_size: self.window_size,
            sink_tokens: self.sink_tokens,
            ring_len: self.ring_len,
            k_q8: self.k_q8.as_ptr() as usize,
            v_q8: self.v_q8.as_ptr() as usize,
            k_scales: self.k_scales.as_ptr() as usize,
            v_scales: self.v_scales.as_ptr() as usize,
            k_q4: self.k_q4.as_ptr() as usize,
            v_q4: self.v_q4.as_ptr() as usize,
            k_s16: self.k_s16.as_ptr() as usize,
            v_s16: self.v_s16.as_ptr() as usize,
        }
    }

    fn ensure_seq_capacity(&mut self, needed_len: usize, live_len: usize) {
        assert!(needed_len <= self.max_seq_len);
        if self.ring_len > 0 || needed_len <= self.cap_seq_len {
            return;
        }
        let mut new_cap = self.cap_seq_len.max(1);
        while new_cap < needed_len {
            new_cap = (new_cap * 2).min(self.max_seq_len);
        }
        let heads = self.num_layers * self.num_kv_heads;
        let hd = self.head_dim;
        let old_cap = self.cap_seq_len;
        let copy_len = live_len.min(old_cap);
        if self.bits == 8 {
            let mut new_k = vec![0i8; heads * new_cap * hd];
            let mut new_v = vec![0i8; heads * new_cap * hd];
            let mut new_ks = vec![0.0f32; heads * new_cap];
            let mut new_vs = vec![0.0f32; heads * new_cap];
            if copy_len > 0 {
                for h in 0..heads {
                    let so = h * old_cap;
                    let dn = h * new_cap;
                    new_k[dn * hd..(dn + copy_len) * hd]
                        .copy_from_slice(&self.k_q8[so * hd..(so + copy_len) * hd]);
                    new_v[dn * hd..(dn + copy_len) * hd]
                        .copy_from_slice(&self.v_q8[so * hd..(so + copy_len) * hd]);
                    new_ks[dn..dn + copy_len].copy_from_slice(&self.k_scales[so..so + copy_len]);
                    new_vs[dn..dn + copy_len].copy_from_slice(&self.v_scales[so..so + copy_len]);
                }
            }
            self.k_q8 = new_k;
            self.v_q8 = new_v;
            self.k_scales = new_ks;
            self.v_scales = new_vs;
        } else {
            let half_hd = hd / 2;
            let gph64 = hd / 64;
            let mut new_k = vec![0x88u8; heads * new_cap * half_hd];
            let mut new_v = vec![0x88u8; heads * new_cap * half_hd];
            let mut new_ks = vec![0u16; heads * new_cap * gph64];
            let mut new_vs = vec![0u16; heads * new_cap * gph64];
            if copy_len > 0 {
                for h in 0..heads {
                    let so = h * old_cap;
                    let dn = h * new_cap;
                    new_k[dn * half_hd..(dn + copy_len) * half_hd]
                        .copy_from_slice(&self.k_q4[so * half_hd..(so + copy_len) * half_hd]);
                    new_v[dn * half_hd..(dn + copy_len) * half_hd]
                        .copy_from_slice(&self.v_q4[so * half_hd..(so + copy_len) * half_hd]);
                    new_ks[dn * gph64..(dn + copy_len) * gph64]
                        .copy_from_slice(&self.k_s16[so * gph64..(so + copy_len) * gph64]);
                    new_vs[dn * gph64..(dn + copy_len) * gph64]
                        .copy_from_slice(&self.v_s16[so * gph64..(so + copy_len) * gph64]);
                }
            }
            self.k_q4 = new_k;
            self.v_q4 = new_v;
            self.k_s16 = new_ks;
            self.v_s16 = new_vs;
        }
        self.cap_seq_len = new_cap;
    }

    #[inline]
    fn physical_bytes(&self) -> usize {
        (self.k_q8.len() + self.v_q8.len()) * std::mem::size_of::<i8>()
            + (self.k_scales.len() + self.v_scales.len()) * std::mem::size_of::<f32>()
            + (self.k_q4.len() + self.v_q4.len()) * std::mem::size_of::<u8>()
            + (self.k_s16.len() + self.v_s16.len()) * std::mem::size_of::<u16>()
    }

    #[inline]
    fn store_k_head(&mut self, layer_idx: usize, kvh: usize, pos: usize, k_head: &[f32]) {
        debug_assert!(layer_idx < self.num_layers && kvh < self.num_kv_heads);
        let raw = self.raw_ptrs();
        // SAFETY: `&mut self` guarantees exclusive access and valid slot bounds.
        unsafe { raw.store_k(layer_idx, kvh, pos, k_head) };
    }

    #[inline]
    fn store_v_head(&mut self, layer_idx: usize, kvh: usize, pos: usize, v_head: &[f32]) {
        debug_assert!(layer_idx < self.num_layers && kvh < self.num_kv_heads);
        let raw = self.raw_ptrs();
        // SAFETY: `&mut self` guarantees exclusive access and valid slot bounds.
        unsafe { raw.store_v(layer_idx, kvh, pos, v_head) };
    }
}

/// Pre-allocated zero-allocation execution workspace and KV cache for [`RealLlamaEngine`].
struct LlamaWorkspace {
    pos: usize,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    q8_kv: Option<QuantizedKvCache>,
    hidden: Vec<f32>,
    norm_buf: Vec<f32>,
    qkv_buf: Vec<f32>,
    attn_out: Vec<f32>,
    proj_out: Vec<f32>,
    gate_up_buf: Vec<f32>,
    swiglu_buf: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    qlut_hidden: QuantizedActivationLut,
    qlut_q: QuantizedActivationLut,
    qlut_inter: QuantizedActivationLut,
    blut_hidden: BipolarActivationLut,
    blut_q: BipolarActivationLut,
    blut_inter: BipolarActivationLut,
    vqlut_hidden: Vq2ActivationLut,
    vqlut_q: Vq2ActivationLut,
    vqlut_inter: Vq2ActivationLut,
    q8_hidden_vals: Vec<i8>,
    q8_hidden_scales: Vec<f32>,
    q8_q_vals: Vec<i8>,
    q8_q_scales: Vec<f32>,
    q8_inter_vals: Vec<i8>,
    q8_inter_scales: Vec<f32>,
    prompt_scratch: PromptScratch,
}

impl LlamaWorkspace {
    fn new(cfg: &LlamaConfig) -> Self {
        let q_dim = cfg.q_dim();
        let qkv_rows = q_dim + 2 * cfg.kv_dim();
        let inter = cfg.intermediate_size.max(64);
        let inter_aligned = inter.next_multiple_of(64);
        Self {
            pos: 0,
            // Allocated lazily on first FP32-KV forward so Q8-KV models never allocate a multi-GiB throwaway FP32 cache.
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            q8_kv: None,
            hidden: vec![0.0f32; cfg.hidden_size],
            norm_buf: vec![0.0f32; cfg.hidden_size],
            qkv_buf: vec![0.0f32; qkv_rows],
            attn_out: vec![0.0f32; q_dim],
            proj_out: vec![0.0f32; cfg.hidden_size],
            gate_up_buf: vec![0.0f32; 2 * inter],
            swiglu_buf: vec![0.0f32; inter],
            scores: vec![0.0f32; cfg.max_seq_len * cfg.num_attention_heads],
            logits: vec![0.0f32; cfg.vocab_size],
            qlut_hidden: QuantizedActivationLut::new_zeroed(cfg.hidden_size),
            qlut_q: QuantizedActivationLut::new_zeroed(q_dim),
            qlut_inter: QuantizedActivationLut::new_zeroed(inter_aligned),
            blut_hidden: BipolarActivationLut::new_zeroed(cfg.hidden_size),
            blut_q: BipolarActivationLut::new_zeroed(q_dim),
            blut_inter: BipolarActivationLut::new_zeroed(inter_aligned),
            vqlut_hidden: Vq2ActivationLut::new_zeroed(cfg.hidden_size),
            vqlut_q: Vq2ActivationLut::new_zeroed(q_dim),
            vqlut_inter: Vq2ActivationLut::new_zeroed(inter_aligned),
            q8_hidden_vals: vec![0i8; cfg.hidden_size],
            q8_hidden_scales: vec![0.0f32; cfg.hidden_size.div_ceil(32)],
            q8_q_vals: vec![0i8; q_dim],
            q8_q_scales: vec![0.0f32; q_dim.div_ceil(32)],
            q8_inter_vals: vec![0i8; inter],
            q8_inter_scales: vec![0.0f32; inter.div_ceil(32)],
            prompt_scratch: PromptScratch::default(),
        }
    }

    #[inline]
    fn ensure_fp32_kv(&mut self, cfg: &LlamaConfig) {
        if self.q8_kv.is_none() {
            let kv_total = cfg.num_hidden_layers * cfg.max_seq_len * cfg.kv_dim();
            if self.k_cache.len() != kv_total {
                self.k_cache = vec![0.0f32; kv_total];
                self.v_cache = vec![0.0f32; kv_total];
            }
        }
    }

    fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Real pretrained `LlamaForCausalLM` / `Qwen2ForCausalLM` / Sparse-MoE engine.
pub struct RealLlamaEngine {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub(crate) metal: Option<crate::metal_gpu::MetalLlamaEngine>,
    pub cfg: LlamaConfig,
    pub mode: RealWeightMode,
    pub mean_weight_rel_error: f32,
    embed_f32_rows: Vec<f32>,
    embed_bf16_rows: Vec<u16>,
    embed_nf4: Option<NfSdotMatrix>,
    kv_bits_pref: u8,
    kv_window_pref: Option<usize>,
    kv_sinks_pref: usize,
    layers: Vec<LlamaLayer>,
    final_norm: Vec<f32>,
    lm_head: LinearBackend,
    /// Precomputed `[max_seq_len, half_dim * 2]` (`[cos_0..cos_{half-1}, sin_0..sin_{half-1}]`).
    rope_cos_sin: Vec<f32>,
    pool: Option<SpinThreadPool>,
    moe_runtime: Option<SparseMoeRuntime>,
    ws: LlamaWorkspace,
}

impl RealLlamaEngine {
    /// Configures persistent P-core thread pool (`num_threads <= 1` disables threading and parks workers).
    pub fn set_num_threads(&mut self, num_threads: usize) {
        if num_threads <= 1 {
            SpinThreadPool::park_idle_workers();
            self.pool = None;
        } else {
            self.pool = Some(SpinThreadPool::new(num_threads));
        }
    }

    /// Configures the engine to use all physical Performance cores (`10` on M4 Pro).
    pub fn set_default_p_cores(&mut self) {
        self.set_num_threads(physical_p_cores());
    }

    /// Current worker thread count (`1` when single-core).
    pub fn num_threads(&self) -> usize {
        self.pool.as_ref().map(|p| p.num_threads()).unwrap_or(1)
    }

    /// Toggles the Hadamard quantized KV cache (`head_dim == 64` or `128`). Frees the FP32 cache and resets `pos`.
    pub fn enable_q8_kv_cache(&mut self, enable: bool) {
        self.ws.pos = 0;
        if enable {
            let hd = self.cfg.head_dim();
            assert!(
                hd == 64 || hd == 128,
                "Hadamard KV-cache requires head_dim == 64 or 128 (got {hd})"
            );
            let bits = if self.kv_bits_pref == 4 { 4 } else { 8 };
            self.ws.k_cache = Vec::new();
            self.ws.v_cache = Vec::new();
            let needs_rebuild = match self.ws.q8_kv.as_ref() {
                None => true,
                Some(q) => q.bits != bits,
            };
            if needs_rebuild {
                self.ws.q8_kv = Some(QuantizedKvCache::new_with_opts(
                    &self.cfg,
                    bits,
                    self.kv_window_pref,
                    self.kv_sinks_pref,
                ));
            }
        } else {
            self.ws.q8_kv = None;
            self.ws.ensure_fp32_kv(&self.cfg);
        }
    }

    /// Selects 4-bit (`NF4` Hadamard) or 8-bit (`INT8` Hadamard) quantized KV cache (`bits == 4` or `8`).
    pub fn set_kv_bits(&mut self, bits: u8) {
        assert!(bits == 4 || bits == 8, "KV bits must be 4 or 8 (got {bits})");
        self.kv_bits_pref = bits;
        self.ws.pos = 0;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = self.metal.as_mut() {
            gpu.reconfigure_kv(&self.cfg, bits, self.kv_window_pref, self.kv_sinks_pref)
                .expect("Metal KV reconfigure failed");
            return;
        }
        if matches!(self.cfg.head_dim(), 64 | 128) {
            self.ws.k_cache = Vec::new();
            self.ws.v_cache = Vec::new();
            self.ws.q8_kv = Some(QuantizedKvCache::new_with_opts(
                &self.cfg,
                bits,
                self.kv_window_pref,
                self.kv_sinks_pref,
            ));
        }
    }

    /// Active KV-cache bit-width (`4`, `8`, or `32`).
    pub fn kv_bits(&self) -> u8 {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = &self.metal {
            return gpu.kv_bits();
        }
        self.ws.q8_kv.as_ref().map_or(32, |q| q.bits)
    }

    /// Configures an `O(1)` sliding-window + attention-sink KV ring buffer (`None` restores full causal window).
    pub fn set_kv_sliding_window(&mut self, window: Option<usize>, sink_tokens: usize) {
        self.kv_window_pref = window.filter(|&w| w >= 8);
        self.kv_sinks_pref = if self.kv_window_pref.is_some() {
            sink_tokens
        } else {
            0
        };
        self.ws.pos = 0;
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = self.metal.as_mut() {
            let bits = if self.kv_bits_pref == 4 { 4 } else { 8 };
            gpu.reconfigure_kv(&self.cfg, bits, self.kv_window_pref, self.kv_sinks_pref)
                .expect("Metal KV window reconfigure failed");
            return;
        }
        if self.ws.q8_kv.is_some() || (self.kv_window_pref.is_some() && matches!(self.cfg.head_dim(), 64 | 128)) {
            let bits = if self.kv_bits_pref == 4 { 4 } else { 8 };
            self.ws.k_cache = Vec::new();
            self.ws.v_cache = Vec::new();
            self.ws.q8_kv = Some(QuantizedKvCache::new_with_opts(
                &self.cfg,
                bits,
                self.kv_window_pref,
                self.kv_sinks_pref,
            ));
        }
    }

    /// Returns `(window_size, sink_tokens)` when a sliding-window ring buffer is active.
    pub fn kv_sliding_window(&self) -> Option<(usize, usize)> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = &self.metal {
            return gpu.kv_sliding_window();
        }
        self.ws
            .q8_kv
            .as_ref()
            .and_then(|q| (q.window_size > 0).then_some((q.window_size, q.sink_tokens)))
    }

    /// Whether `truncate_context(target_len)` can roll back from `self.ws.pos` without ring-buffer aliasing.
    #[inline]
    pub fn can_truncate_kv_to(&self, target_len: usize) -> bool {
        if target_len >= self.ws.pos {
            return true;
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = &self.metal {
            return gpu.can_truncate_kv_to(self.ws.pos, target_len);
        }
        if let Some(q) = &self.ws.q8_kv {
            if q.ring_len > 0 {
                let slack = q.ring_len.saturating_sub(q.window_size.saturating_sub(q.sink_tokens));
                return target_len <= q.sink_tokens || (self.ws.pos - target_len) <= slack;
            }
        }
        true
    }

    /// Compresses untied `BF16` embeddings into 4-bit Hadamard `NF4` (`-1.08 GiB` wired RAM on `152k` vocab)
    /// and releases the `BF16` heap allocation.
    pub fn compress_untied_embeddings_nf4(&mut self) {
        if self.embed_nf4.is_some()
            || self.embed_bf16_rows.is_empty()
            || !self.cfg.hidden_size.is_multiple_of(64)
        {
            return;
        }
        let workers = std::thread::available_parallelism()
            .map_or(4, |p| p.get())
            .clamp(1, 8);
        let mat = NfSdotMatrix::from_bf16_u16_embeddings_nf4(
            &self.embed_bf16_rows,
            self.cfg.vocab_size,
            self.cfg.hidden_size,
            2,
            workers,
        );
        self.embed_bf16_rows = Vec::new();
        self.embed_nf4 = Some(mat);
    }

    /// Whether untied input embeddings are stored in 4-bit Hadamard `NF4`.
    #[inline]
    pub fn has_nf4_untied_embeddings(&self) -> bool {
        self.embed_nf4.is_some()
    }

    /// Whether the quantized KV cache is active.
    #[inline]
    pub fn is_q8_kv_enabled(&self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.metal.is_some() {
            return true;
        }
        self.ws.q8_kv.is_some()
    }

    /// Whether this engine executes Sparse-MoE layers via [`SparseMoeRuntime`].
    #[inline]
    pub fn is_moe(&self) -> bool {
        self.moe_runtime.is_some()
    }

    /// Live SLRU / NVMe / zero-copy `mmap` telemetry when running a Sparse-MoE model.
    #[inline]
    pub fn moe_stats(&self) -> Option<MoeRuntimeStats> {
        self.moe_runtime.as_ref().map(|r| r.stats())
    }

    /// Evicts all DRAM SLRU slots on Sparse-MoE NVMe engines (used for cold-start benchmarks).
    pub fn clear_moe_slru_cache(&mut self) {
        if let Some(r) = self.moe_runtime.as_mut() {
            r.clear_slru_cache();
        }
    }

    /// Enables zero-copy Metal GPU execution for NfSdot engines (`0` duplicate weight RAM).
    pub fn enable_metal(&mut self) -> Result<(), String> {
        if self.is_moe() {
            return Err("Sparse-MoE engines execute hybrid CPU P-core + NVMe SLRU paging".to_string());
        }
        if self.cfg.q_dim() != self.cfg.hidden_size
            || self
                .layers
                .iter()
                .any(|l| l.q_norm.is_some() || l.k_norm.is_some() || l.o_bias.is_some())
        {
            return Err("Metal GPU requires q_dim == hidden_size without q_norm/k_norm/o_bias".to_string());
        }
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            fn as_nf4<'a>(b: &'a LinearBackend, name: &str) -> Result<&'a NfSdotMatrix, String> {
                match b {
                    LinearBackend::NfSdot(m) if (2..=5).contains(&m.bits) => Ok(m),
                    LinearBackend::NfSdot(m) => {
                        Err(format!("{name}: Metal GPU requires 2..=5-bit NfSdot (got {}-bit)", m.bits))
                    }
                    _ => Err(format!("{name}: Metal GPU requires NfSdot weights")),
                }
            }
            let mut borrows = Vec::with_capacity(self.layers.len());
            for (idx, l) in self.layers.iter().enumerate() {
                if l.o_inv_scale.is_some() || l.down_inv_scale.is_some() {
                    return Err(format!("layer.{idx}: Metal GPU does not support AWQ channel inv_scale"));
                }
                borrows.push(crate::metal_gpu::LayerBorrow {
                    input_norm: &l.input_norm,
                    post_attn_norm: &l.post_attn_norm,
                    qkv_bias: l.qkv_bias.as_deref(),
                    qkv_proj: as_nf4(&l.qkv_proj, &format!("layer.{idx}.qkv_proj"))?,
                    o_proj: as_nf4(&l.o_proj, &format!("layer.{idx}.o_proj"))?,
                    gate_up_proj: as_nf4(&l.gate_up_proj, &format!("layer.{idx}.gate_up_proj"))?,
                    gate_up_interleaved: l.gate_up_interleaved,
                    down_proj: as_nf4(&l.down_proj, &format!("layer.{idx}.down_proj"))?,
                });
            }
            let head_mat = as_nf4(&self.lm_head, "lm_head")?;
            let kv_bits = if self.kv_bits_pref == 4 { 4 } else { 8 };
            let gpu = crate::metal_gpu::MetalLlamaEngine::try_new(
                &self.cfg,
                &self.rope_cos_sin,
                &borrows,
                &self.final_norm,
                head_mat,
                kv_bits,
                self.kv_window_pref,
                self.kv_sinks_pref,
            )?;
            self.ws.pos = 0;
            self.ws.q8_kv = None;
            self.ws.k_cache = Vec::new();
            self.ws.v_cache = Vec::new();
            SpinThreadPool::park_idle_workers();
            self.pool = None;
            self.metal = Some(gpu);
            // Pre-wire zero-copy weight and scratch pages in IOGPU once during load.
            let warm_n = 16.min(self.cfg.max_seq_len);
            if warm_n > 0 {
                let warm = vec![0usize; warm_n];
                let _ = self.forward_prompt(&warm);
                self.reset_context();
            }
            if self.cfg.max_seq_len >= 2 {
                let mut tmp = vec![0.0f32; 2 * self.cfg.vocab_size];
                self.forward_batch_logits(&[0, 0], &mut tmp);
                self.reset_context();
            }
            if self.cfg.max_seq_len >= 1 {
                let _ = self.forward_token(0);
                self.reset_context();
            }
            Ok(())
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            Err("Metal GPU backend is only available on Apple platforms".to_string())
        }
    }

    /// Disables Metal GPU execution and restores CPU quantized KV cache and P-core worker pool.
    pub fn disable_metal(&mut self) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.metal.take().is_some() {
            self.ws.pos = 0;
            if matches!(self.cfg.head_dim(), 64 | 128) {
                self.enable_q8_kv_cache(true);
            }
            self.set_default_p_cores();
        }
    }

    /// Whether the Metal GPU backend is active.
    #[inline]
    pub fn is_metal_enabled(&self) -> bool {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            self.metal.is_some()
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            false
        }
    }

    /// Bytes allocated for the active KV cache.
    pub fn kv_cache_bytes(&self) -> usize {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = &self.metal {
            return gpu.kv_cache_bytes();
        }
        if let Some(q8) = &self.ws.q8_kv {
            q8.physical_bytes()
        } else {
            let kv_total = if self.ws.k_cache.is_empty() {
                2 * self.cfg.num_hidden_layers * self.cfg.max_seq_len * self.cfg.kv_dim()
            } else {
                self.ws.k_cache.len() + self.ws.v_cache.len()
            };
            kv_total * std::mem::size_of::<f32>()
        }
    }

    /// Loads the exact `FP32` reference model from `model_dir` (single or sharded SafeTensors, dense or Sparse-MoE).
    pub fn load_fp32_reference(model_dir: &Path, max_seq_len: usize) -> io::Result<Self> {
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_seq_len)?;
        let store = SafeTensorsStore::open_dir(model_dir)?;

        let get_f32 = |name: &str| -> io::Result<Vec<f32>> {
            store
                .tensor(name)
                .map(|t| t.to_f32_vec())
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))
        };

        let embed_f32_rows = get_f32("model.embed_tokens.weight")?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let q_dim = cfg.q_dim();
        let qkv_rows = q_dim + 2 * cfg.kv_dim();
        let mut fp32_moe_layers: Vec<Vec<(Vec<f32>, Vec<f32>)>> = Vec::new();
        let mut max_shared_inter = 0usize;

        for l in 0..cfg.num_hidden_layers {
            let pfx = format!("model.layers.{l}");
            let mut qkv_w = Vec::with_capacity(qkv_rows * cfg.hidden_size);
            qkv_w.extend_from_slice(&get_f32(&format!("{pfx}.self_attn.q_proj.weight"))?);
            qkv_w.extend_from_slice(&get_f32(&format!("{pfx}.self_attn.k_proj.weight"))?);
            qkv_w.extend_from_slice(&get_f32(&format!("{pfx}.self_attn.v_proj.weight"))?);

            let q_bias_name = format!("{pfx}.self_attn.q_proj.bias");
            let qkv_bias = if store.has_tensor(&q_bias_name) {
                let mut b = Vec::with_capacity(qkv_rows);
                b.extend_from_slice(&get_f32(&q_bias_name)?);
                b.extend_from_slice(&get_f32(&format!("{pfx}.self_attn.k_proj.bias"))?);
                b.extend_from_slice(&get_f32(&format!("{pfx}.self_attn.v_proj.bias"))?);
                Some(b)
            } else {
                None
            };
            let o_bias_name = format!("{pfx}.self_attn.o_proj.bias");
            let o_bias = if store.has_tensor(&o_bias_name) {
                Some(get_f32(&o_bias_name)?)
            } else {
                None
            };
            let q_norm_name = format!("{pfx}.self_attn.q_norm.weight");
            let q_norm = if store.has_tensor(&q_norm_name) {
                Some(get_f32(&q_norm_name)?)
            } else {
                None
            };
            let k_norm_name = format!("{pfx}.self_attn.k_norm.weight");
            let k_norm = if store.has_tensor(&k_norm_name) {
                Some(get_f32(&k_norm_name)?)
            } else {
                None
            };

            let (gate_up_proj, down_proj, moe) = if cfg.layer_is_moe(l) {
                let slot = fp32_moe_layers.len();
                let router = read_moe_router_f32(&store, l, cfg.num_experts, cfg.hidden_size, false)?;
                let shared = read_moe_shared_expert_f32(
                    &store,
                    l,
                    cfg.hidden_size,
                    cfg.shared_expert_intermediate_size,
                    false,
                )?;
                let shared_expert = shared.map(|(s_inter, gu, dn, gw)| {
                    max_shared_inter = max_shared_inter.max(s_inter);
                    SharedMoeExpert {
                        inter_dim: s_inter,
                        backend: SharedExpertBackend::Fp32 {
                            hidden_size: cfg.hidden_size,
                            inter_dim: s_inter,
                            gate_up: gu,
                            down: dn,
                        },
                        gate_weight: gw,
                    }
                });
                let moe_inter = cfg.eff_moe_inter();
                let mut exps = Vec::with_capacity(cfg.num_experts);
                for e in 0..cfg.num_experts {
                    exps.push(read_moe_expert_f32(
                        &store,
                        l,
                        e,
                        cfg.num_experts,
                        cfg.hidden_size,
                        moe_inter,
                        false,
                    )?);
                }
                fp32_moe_layers.push(exps);
                (
                    LinearBackend::Empty,
                    LinearBackend::Empty,
                    Some(MoeLayerCore {
                        moe_slot_idx: slot,
                        router,
                        shared_expert,
                        lookahead: None,
                    }),
                )
            } else {
                let mut gate_up_w = Vec::with_capacity(2 * cfg.intermediate_size * cfg.hidden_size);
                gate_up_w.extend_from_slice(&get_f32(&format!("{pfx}.mlp.gate_proj.weight"))?);
                gate_up_w.extend_from_slice(&get_f32(&format!("{pfx}.mlp.up_proj.weight"))?);
                (
                    LinearBackend::Fp32 {
                        rows: 2 * cfg.intermediate_size,
                        cols: cfg.hidden_size,
                        weights: gate_up_w,
                    },
                    LinearBackend::Fp32 {
                        rows: cfg.hidden_size,
                        cols: cfg.intermediate_size,
                        weights: get_f32(&format!("{pfx}.mlp.down_proj.weight"))?,
                    },
                    None,
                )
            };

            layers.push(LlamaLayer {
                input_norm: get_f32(&format!("{pfx}.input_layernorm.weight"))?,
                post_attn_norm: get_f32(&format!("{pfx}.post_attention_layernorm.weight"))?,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias,
                o_bias,
                q_norm,
                k_norm,
                qkv_proj: LinearBackend::Fp32 {
                    rows: qkv_rows,
                    cols: cfg.hidden_size,
                    weights: qkv_w,
                },
                o_proj: LinearBackend::Fp32 {
                    rows: cfg.hidden_size,
                    cols: q_dim,
                    weights: get_f32(&format!("{pfx}.self_attn.o_proj.weight"))?,
                },
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
                moe,
            });
        }

        let moe_runtime = if !fp32_moe_layers.is_empty() {
            Some(SparseMoeRuntime::new_fp32_reference(
                cfg.hidden_size,
                cfg.eff_moe_inter(),
                cfg.num_experts,
                cfg.num_experts_per_tok,
                cfg.norm_topk_prob,
                cfg.router_scoring,
                max_shared_inter,
                fp32_moe_layers,
            ))
        } else {
            None
        };

        let final_norm = get_f32("model.norm.weight")?;
        let lm_head_weights = if cfg.tie_word_embeddings {
            embed_f32_rows.clone()
        } else {
            get_f32("lm_head.weight")?
        };

        let rope_cos_sin = build_rope_table(&cfg);
        let ws = LlamaWorkspace::new(&cfg);
        Ok(Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: 8,
            kv_window_pref: cfg.sliding_window,
            kv_sinks_pref: cfg.sink_tokens,
            cfg: cfg.clone(),
            mode: RealWeightMode::ExactFp32,
            mean_weight_rel_error: 0.0,
            embed_f32_rows,
            embed_bf16_rows: Vec::new(),
            embed_nf4: None,
            layers,
            final_norm,
            lm_head: LinearBackend::Fp32 {
                rows: cfg.vocab_size,
                cols: cfg.hidden_size,
                weights: lm_head_weights,
            },
            rope_cos_sin,
            pool: None,
            moe_runtime,
            ws,
        })
    }

    /// Collects per-channel activation RMS scales across `calib_ids` on an `ExactFp32` reference model.
    pub fn calibrate_activations(&mut self, calib_ids: &[usize], alpha: f32) -> ActivationCalibration {
        assert_eq!(self.mode, RealWeightMode::ExactFp32);
        let nl = self.cfg.num_hidden_layers;
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;

        let mut attn_sq = vec![vec![0.0f64; h]; nl];
        let mut o_sq = vec![vec![0.0f64; h]; nl];
        let mut mlp_sq = vec![vec![0.0f64; h]; nl];
        let mut down_sq = vec![vec![0.0f64; inter]; nl];
        let mut head_sq = vec![0.0f64; h];

        self.reset_context();
        self.ws.ensure_fp32_kv(&self.cfg);
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_dim = self.cfg.kv_dim();
        let kv_group = self.cfg.num_attention_heads / self.cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let layer_kv_stride = self.cfg.max_seq_len * kv_dim;

        for &token_id in calib_ids {
            let pos = self.ws.pos;
            let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
            self.ws
                .hidden
                .copy_from_slice(&self.embed_f32_rows[token_id * h..(token_id + 1) * h]);

            for (l_idx, layer) in self.layers.iter().enumerate() {
                rms_norm(
                    &self.ws.hidden,
                    &layer.input_norm,
                    self.cfg.rms_norm_eps,
                    &mut self.ws.norm_buf,
                );
                accum_sq(&self.ws.norm_buf, &mut attn_sq[l_idx]);

                layer.qkv_proj.forward_exec(
                    &self.ws.norm_buf,
                    None,
                    None,
                    None,
                    &[],
                    &[],
                    None,
                    &mut self.ws.qkv_buf,
                );
                if let Some(bias) = &layer.qkv_bias {
                    add_inplace(&mut self.ws.qkv_buf, bias);
                }
                let (q_slice, kv_rest) = self.ws.qkv_buf.split_at_mut(h);
                let (k_slice, v_slice) = kv_rest.split_at_mut(kv_dim);

                apply_llama_rope_precomp(
                    q_slice,
                    self.cfg.num_attention_heads,
                    head_dim,
                    half_dim,
                    rope_row,
                );
                apply_llama_rope_precomp(
                    k_slice,
                    self.cfg.num_key_value_heads,
                    head_dim,
                    half_dim,
                    rope_row,
                );

                let kv_base = l_idx * layer_kv_stride;
                let step_off = kv_base + pos * kv_dim;
                self.ws.k_cache[step_off..step_off + kv_dim].copy_from_slice(k_slice);
                self.ws.v_cache[step_off..step_off + kv_dim].copy_from_slice(v_slice);

                causal_gqa_attention(
                    q_slice,
                    &self.ws.k_cache[kv_base..],
                    &self.ws.v_cache[kv_base..],
                    &mut self.ws.scores,
                    &mut self.ws.attn_out,
                    self.cfg.num_attention_heads,
                    kv_group,
                    head_dim,
                    kv_dim,
                    pos,
                    attn_scale,
                );

                accum_sq(&self.ws.attn_out, &mut o_sq[l_idx]);
                layer.o_proj.forward_exec(
                    &self.ws.attn_out,
                    None,
                    None,
                    None,
                    &[],
                    &[],
                    None,
                    &mut self.ws.proj_out,
                );
                add_inplace(&mut self.ws.hidden, &self.ws.proj_out);

                rms_norm(
                    &self.ws.hidden,
                    &layer.post_attn_norm,
                    self.cfg.rms_norm_eps,
                    &mut self.ws.norm_buf,
                );
                accum_sq(&self.ws.norm_buf, &mut mlp_sq[l_idx]);

                layer.gate_up_proj.forward_exec(
                    &self.ws.norm_buf,
                    None,
                    None,
                    None,
                    &[],
                    &[],
                    None,
                    &mut self.ws.gate_up_buf,
                );
                let (gate_s, up_s) = self.ws.gate_up_buf.split_at(inter);
                swiglu_activate(gate_s, up_s, &mut self.ws.swiglu_buf);

                accum_sq(&self.ws.swiglu_buf, &mut down_sq[l_idx]);
                layer.down_proj.forward_exec(
                    &self.ws.swiglu_buf,
                    None,
                    None,
                    None,
                    &[],
                    &[],
                    None,
                    &mut self.ws.proj_out,
                );
                add_inplace(&mut self.ws.hidden, &self.ws.proj_out);
            }

            rms_norm(
                &self.ws.hidden,
                &self.final_norm,
                self.cfg.rms_norm_eps,
                &mut self.ws.norm_buf,
            );
            accum_sq(&self.ws.norm_buf, &mut head_sq);
            self.ws.pos += 1;
        }

        let steps = calib_ids.len() as f64;
        let to_scales = |sq: &[f64]| -> Vec<f32> {
            let rms: Vec<f32> = sq
                .iter()
                .map(|&s| ((s / steps).sqrt() as f32).max(1e-5))
                .collect();
            let mean_rms = rms.iter().sum::<f32>() / (rms.len() as f32);
            rms.iter()
                .map(|&r| (r / mean_rms.max(1e-6)).powf(alpha).clamp(0.125, 32.0))
                .collect()
        };

        self.reset_context();
        ActivationCalibration {
            attn_in_scales: attn_sq.iter().map(|v| to_scales(v)).collect(),
            o_in_scales: o_sq.iter().map(|v| to_scales(v)).collect(),
            mlp_in_scales: mlp_sq.iter().map(|v| to_scales(v)).collect(),
            down_in_scales: down_sq.iter().map(|v| to_scales(v)).collect(),
            head_in_scales: to_scales(&head_sq),
        }
    }

    /// Compiles direct PTQ / `AWQ` / `GGUF Q4_0` modes from this `ExactFp32` reference model.
    pub fn compile_from_reference(
        &self,
        mode: RealWeightMode,
        calib: Option<&ActivationCalibration>,
    ) -> Self {
        assert_eq!(self.mode, RealWeightMode::ExactFp32);
        let use_awq = matches!(mode, RealWeightMode::AwqGroup64Ternary { .. });
        if use_awq {
            assert!(
                calib.is_some(),
                "ActivationCalibration required for AwqGroup64Ternary"
            );
        }

        let mut compiled_layers = Vec::with_capacity(self.cfg.num_hidden_layers);
        let mut err_sum = 0.0f64;
        let mut mat_count = 0usize;
        let qkv_rows = self.cfg.hidden_size + 2 * self.cfg.kv_dim();

        for (l, ref_layer) in self.layers.iter().enumerate() {
            let mut input_norm = ref_layer.input_norm.clone();
            let mut post_attn_norm = ref_layer.post_attn_norm.clone();

            let (s_attn, s_o, s_mlp, s_down) = if use_awq {
                let c = calib.unwrap();
                for (nw, &s) in input_norm.iter_mut().zip(c.attn_in_scales[l].iter()) {
                    *nw /= s;
                }
                for (nw, &s) in post_attn_norm.iter_mut().zip(c.mlp_in_scales[l].iter()) {
                    *nw /= s;
                }
                (
                    Some(c.attn_in_scales[l].as_slice()),
                    Some(c.o_in_scales[l].as_slice()),
                    Some(c.mlp_in_scales[l].as_slice()),
                    Some(c.down_in_scales[l].as_slice()),
                )
            } else {
                (None, None, None, None)
            };

            let o_inv_scale = s_o.map(|s| s.iter().map(|&v| 1.0 / v).collect());
            let down_inv_scale = s_down.map(|s| s.iter().map(|&v| 1.0 / v).collect());

            let (qkv_proj, eqkv) = LinearBackend::from_f32_slice(
                ref_layer.qkv_proj.as_fp32_slice(),
                qkv_rows,
                self.cfg.hidden_size,
                mode,
                s_attn,
            );
            let (o_proj, eo) = LinearBackend::from_f32_slice(
                ref_layer.o_proj.as_fp32_slice(),
                self.cfg.hidden_size,
                self.cfg.hidden_size,
                mode,
                s_o,
            );
            let (gate_up_proj, egu) = LinearBackend::from_f32_slice(
                ref_layer.gate_up_proj.as_fp32_slice(),
                2 * self.cfg.intermediate_size,
                self.cfg.hidden_size,
                mode,
                s_mlp,
            );
            let (down_proj, ed) = LinearBackend::from_f32_slice(
                ref_layer.down_proj.as_fp32_slice(),
                self.cfg.hidden_size,
                self.cfg.intermediate_size,
                mode,
                s_down,
            );

            err_sum += (eqkv + eo + egu + ed) as f64;
            mat_count += 4;

            compiled_layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale,
                down_inv_scale,
                qkv_bias: ref_layer.qkv_bias.clone(),
                o_bias: ref_layer.o_bias.clone(),
                q_norm: ref_layer.q_norm.clone(),
                k_norm: ref_layer.k_norm.clone(),
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
                moe: None,
            });
        }

        let mut final_norm = self.final_norm.clone();
        let s_head = if use_awq {
            let c = calib.unwrap();
            for (nw, &s) in final_norm.iter_mut().zip(c.head_in_scales.iter()) {
                *nw /= s;
            }
            Some(c.head_in_scales.as_slice())
        } else {
            None
        };

        let (lm_head, el) = LinearBackend::from_f32_slice(
            self.lm_head.as_fp32_slice(),
            self.cfg.vocab_size,
            self.cfg.hidden_size,
            mode,
            s_head,
        );
        err_sum += el as f64;
        mat_count += 1;

        let mut eng = Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: 8,
            kv_window_pref: self.cfg.sliding_window,
            kv_sinks_pref: self.cfg.sink_tokens,
            cfg: self.cfg.clone(),
            mode,
            mean_weight_rel_error: (err_sum / (mat_count as f64)) as f32,
            embed_f32_rows: self.embed_f32_rows.clone(),
            embed_bf16_rows: self.embed_bf16_rows.clone(),
            embed_nf4: self.embed_nf4.clone(),
            layers: compiled_layers,
            final_norm,
            lm_head,
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
            moe_runtime: None,
            ws: LlamaWorkspace::new(&self.cfg),
        };
        eng.init_activation_scratch();
        eng
    }

    /// Compiles **`kronos-2t` Sequential Layer-by-Layer Hadamard + Full-Row Cholesky `GPTQ`**
    /// across `calib_sequences` (propagating true quantized activations & `INT8` LUT states
    /// layer-by-layer so later projections actively cancel earlier quantization drift!).
    pub fn compile_sequential_hadamard_gptq(
        &self,
        mode: RealWeightMode,
        calib_sequences: &[Vec<usize>],
        compile_threads: usize,
    ) -> Self {
        assert_eq!(self.mode, RealWeightMode::ExactFp32);
        let (base_planes, down_planes, head_planes) = match mode {
            RealWeightMode::KronosHadamardGptqBase3 { num_planes } => {
                (num_planes, num_planes, num_planes)
            }
            RealWeightMode::KronosHadamardGptqMixedK => (2, 3, 3),
            _ => panic!("compile_sequential_hadamard_gptq expects KronosHadamardGptq* mode"),
        };

        let h_dim = self.cfg.hidden_size;
        let inter_dim = self.cfg.intermediate_size;
        let kv_dim = self.cfg.kv_dim();
        let qkv_rows = h_dim + 2 * kv_dim;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_group = self.cfg.num_attention_heads / self.cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        // Initialize per-sequence hidden trajectories `[num_seqs][seq_len * h_dim]`
        let mut seq_hiddens: Vec<Vec<f32>> = calib_sequences
            .iter()
            .map(|seq| {
                assert!(!seq.is_empty() && seq.len() <= self.cfg.max_seq_len);
                let mut buf = vec![0.0f32; seq.len() * h_dim];
                for (t, &tok_id) in seq.iter().enumerate() {
                    buf[t * h_dim..(t + 1) * h_dim]
                        .copy_from_slice(&self.embed_f32_rows[tok_id * h_dim..(tok_id + 1) * h_dim]);
                }
                buf
            })
            .collect();

        let total_tokens: usize = calib_sequences.iter().map(|s| s.len()).sum();
        let mut all_attn_out = vec![0.0f32; total_tokens * h_dim];
        let mut all_swiglu = vec![0.0f32; total_tokens * inter_dim];

        let mut norm_buf = vec![0.0f32; h_dim];
        let mut rot_h = vec![0.0f32; h_dim];
        let mut rot_inter = vec![0.0f32; inter_dim];
        let mut qkv_buf = vec![0.0f32; qkv_rows];
        let mut gate_up_buf = vec![0.0f32; 2 * inter_dim];
        let mut proj_out = vec![0.0f32; h_dim];
        let mut scores = vec![0.0f32; self.cfg.max_seq_len * self.cfg.num_attention_heads];
        let mut k_seq_cache = vec![0.0f32; self.cfg.max_seq_len * kv_dim];
        let mut v_seq_cache = vec![0.0f32; self.cfg.max_seq_len * kv_dim];
        let mut qlut_h = QuantizedActivationLut::new_zeroed(h_dim);
        let mut qlut_i = QuantizedActivationLut::new_zeroed(inter_dim);

        let mut compiled_layers = Vec::with_capacity(self.cfg.num_hidden_layers);

        for ref_layer in &self.layers {
            // 1. Gather pre-attention normalized activations -> compile `qkv_proj`
            let mut hess_attn = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(hid, &ref_layer.input_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    hess_attn.observe_activation(&norm_buf);
                }
            }
            hess_attn.finalize(0.03);
            let qkv_mat = compile_hadamard_gptq_full_row(
                ref_layer.qkv_proj.as_fp32_slice(),
                qkv_rows,
                h_dim,
                base_planes,
                Some(&hess_attn),
                compile_threads,
            );

            // 2. Execute quantized `qkv_proj` + causal GQA attention -> gather `o_proj` Hessian
            let mut hess_o = FullRowGptqHessian::new(h_dim);
            let mut flat_cursor = 0usize;
            for seq_h in &seq_hiddens {
                let seq_len = seq_h.len() / h_dim;
                for pos in 0..seq_len {
                    let hid = &seq_h[pos * h_dim..(pos + 1) * h_dim];
                    rms_norm(hid, &ref_layer.input_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    rot_h.copy_from_slice(&norm_buf);
                    fwht64_signed_inplace(&mut rot_h);
                    qlut_h.populate(&rot_h);
                    simd_lut_base3_gemv(&qkv_mat, &qlut_h, &mut qkv_buf);
                    if let Some(bias) = &ref_layer.qkv_bias {
                        add_inplace(&mut qkv_buf, bias);
                    }

                    let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
                    let (q_slice, kv_rest) = qkv_buf.split_at_mut(h_dim);
                    let (k_slice, v_slice) = kv_rest.split_at_mut(kv_dim);
                    apply_llama_rope_precomp(
                        q_slice,
                        self.cfg.num_attention_heads,
                        head_dim,
                        half_dim,
                        rope_row,
                    );
                    apply_llama_rope_precomp(
                        k_slice,
                        self.cfg.num_key_value_heads,
                        head_dim,
                        half_dim,
                        rope_row,
                    );
                    k_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k_slice);
                    v_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v_slice);

                    let attn_dst = &mut all_attn_out[flat_cursor * h_dim..(flat_cursor + 1) * h_dim];
                    causal_gqa_attention(
                        q_slice,
                        &k_seq_cache,
                        &v_seq_cache,
                        &mut scores,
                        attn_dst,
                        self.cfg.num_attention_heads,
                        kv_group,
                        head_dim,
                        kv_dim,
                        pos,
                        attn_scale,
                    );
                    hess_o.observe_activation(attn_dst);
                    flat_cursor += 1;
                }
            }
            hess_o.finalize(0.03);
            let o_mat = compile_hadamard_gptq_full_row(
                ref_layer.o_proj.as_fp32_slice(),
                h_dim,
                h_dim,
                base_planes,
                Some(&hess_o),
                compile_threads,
            );

            // Apply quantized `o_proj` into residual `seq_hiddens`
            let mut flat_cursor = 0usize;
            for seq_h in &mut seq_hiddens {
                for hid in seq_h.chunks_exact_mut(h_dim) {
                    rot_h.copy_from_slice(
                        &all_attn_out[flat_cursor * h_dim..(flat_cursor + 1) * h_dim],
                    );
                    fwht64_signed_inplace(&mut rot_h);
                    qlut_h.populate(&rot_h);
                    simd_lut_base3_gemv(&o_mat, &qlut_h, &mut proj_out);
                    add_inplace(hid, &proj_out);
                    flat_cursor += 1;
                }
            }

            // 3. Gather post-attention normalized activations -> compile `gate_up_proj`
            let mut hess_mlp = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(
                        hid,
                        &ref_layer.post_attn_norm,
                        self.cfg.rms_norm_eps,
                        &mut norm_buf,
                    );
                    hess_mlp.observe_activation(&norm_buf);
                }
            }
            hess_mlp.finalize(0.03);
            let gate_up_mat = compile_hadamard_gptq_full_row(
                ref_layer.gate_up_proj.as_fp32_slice(),
                2 * inter_dim,
                h_dim,
                base_planes,
                Some(&hess_mlp),
                compile_threads,
            );

            // 4. Execute quantized `gate_up_proj` + `SwiGLU` -> gather `down_proj` Hessian
            let mut hess_down = FullRowGptqHessian::new(inter_dim);
            let mut flat_cursor = 0usize;
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(
                        hid,
                        &ref_layer.post_attn_norm,
                        self.cfg.rms_norm_eps,
                        &mut norm_buf,
                    );
                    rot_h.copy_from_slice(&norm_buf);
                    fwht64_signed_inplace(&mut rot_h);
                    qlut_h.populate(&rot_h);
                    simd_lut_base3_gemv(&gate_up_mat, &qlut_h, &mut gate_up_buf);

                    let (gate_s, up_s) = gate_up_buf.split_at(inter_dim);
                    let sw_dst =
                        &mut all_swiglu[flat_cursor * inter_dim..(flat_cursor + 1) * inter_dim];
                    swiglu_activate(gate_s, up_s, sw_dst);
                    hess_down.observe_activation(sw_dst);
                    flat_cursor += 1;
                }
            }
            hess_down.finalize(0.035);
            let down_mat = compile_hadamard_gptq_full_row(
                ref_layer.down_proj.as_fp32_slice(),
                h_dim,
                inter_dim,
                down_planes,
                Some(&hess_down),
                compile_threads,
            );

            // Apply quantized `down_proj` into residual `seq_hiddens`
            let mut flat_cursor = 0usize;
            for seq_h in &mut seq_hiddens {
                for hid in seq_h.chunks_exact_mut(h_dim) {
                    rot_inter.copy_from_slice(
                        &all_swiglu[flat_cursor * inter_dim..(flat_cursor + 1) * inter_dim],
                    );
                    fwht64_signed_inplace(&mut rot_inter);
                    qlut_i.populate(&rot_inter);
                    simd_lut_base3_gemv(&down_mat, &qlut_i, &mut proj_out);
                    add_inplace(hid, &proj_out);
                    flat_cursor += 1;
                }
            }

            compiled_layers.push(LlamaLayer {
                input_norm: ref_layer.input_norm.clone(),
                post_attn_norm: ref_layer.post_attn_norm.clone(),
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias: ref_layer.qkv_bias.clone(),
                o_bias: ref_layer.o_bias.clone(),
                q_norm: ref_layer.q_norm.clone(),
                k_norm: ref_layer.k_norm.clone(),
                qkv_proj: LinearBackend::FusedBase3(qkv_mat),
                o_proj: LinearBackend::FusedBase3(o_mat),
                gate_up_proj: LinearBackend::FusedBase3(gate_up_mat),
                gate_up_interleaved: false,
                down_proj: LinearBackend::FusedBase3(down_mat),
                moe: None,
            });
        }

        // 5. Final `RMSNorm` + `lm_head` (`49,152 x 576`) GPTQ compilation on true 30-layer outputs!
        let mut hess_head = FullRowGptqHessian::new(h_dim);
        for seq_h in &seq_hiddens {
            for hid in seq_h.chunks_exact(h_dim) {
                rms_norm(hid, &self.final_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                hess_head.observe_activation(&norm_buf);
            }
        }
        hess_head.finalize(0.025);
        let head_mat = compile_hadamard_gptq_full_row(
            self.lm_head.as_fp32_slice(),
            self.cfg.vocab_size,
            h_dim,
            head_planes,
            Some(&hess_head),
            compile_threads,
        );

        Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: 8,
            kv_window_pref: self.cfg.sliding_window,
            kv_sinks_pref: self.cfg.sink_tokens,
            cfg: self.cfg.clone(),
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows: self.embed_f32_rows.clone(),
            embed_bf16_rows: self.embed_bf16_rows.clone(),
            embed_nf4: self.embed_nf4.clone(),
            layers: compiled_layers,
            final_norm: self.final_norm.clone(),
            lm_head: LinearBackend::FusedBase3(head_mat),
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
            moe_runtime: None,
            ws: LlamaWorkspace::new(&self.cfg),
        }
    }

    /// Sequential layer-by-layer Hadamard GPTQ compile for the Bipolar, NF and VQ2 modes.
    pub fn compile_sequential_bipolar_gptq(
        &self,
        mode: RealWeightMode,
        calib_sequences: &[Vec<usize>],
        compile_threads: usize,
    ) -> Self {
        assert_eq!(self.mode, RealWeightMode::ExactFp32);
        let (qkv_spec, o_spec, gate_up_spec, down_spec, head_spec) = proj_specs_for_mode(mode);

        let h_dim = self.cfg.hidden_size;
        let inter_dim = self.cfg.intermediate_size;
        let kv_dim = self.cfg.kv_dim();
        let qkv_rows = h_dim + 2 * kv_dim;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_group = self.cfg.num_attention_heads / self.cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();

        let mut seq_hiddens: Vec<Vec<f32>> = calib_sequences
            .iter()
            .map(|seq| {
                assert!(!seq.is_empty() && seq.len() <= self.cfg.max_seq_len);
                let mut buf = vec![0.0f32; seq.len() * h_dim];
                for (t, &tok_id) in seq.iter().enumerate() {
                    buf[t * h_dim..(t + 1) * h_dim]
                        .copy_from_slice(&self.embed_f32_rows[tok_id * h_dim..(tok_id + 1) * h_dim]);
                }
                buf
            })
            .collect();

        let total_tokens: usize = calib_sequences.iter().map(|s| s.len()).sum();
        let mut all_attn_out = vec![0.0f32; total_tokens * h_dim];
        let mut all_swiglu = vec![0.0f32; total_tokens * inter_dim];

        let mut norm_buf = vec![0.0f32; h_dim];

        let mut compiled_layers = Vec::with_capacity(self.cfg.num_hidden_layers);

        for ref_layer in &self.layers {
            // 1. Pre-attention normalized activations -> compile `qkv_proj`
            let mut hess_attn = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(hid, &ref_layer.input_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    hess_attn.observe_activation(&norm_buf);
                }
            }
            hess_attn.finalize(BIPOLAR_GPTQ_DAMP);
            let qkv_proj = compile_proj_spec(
                ref_layer.qkv_proj.as_fp32_slice(),
                qkv_rows,
                h_dim,
                qkv_spec,
                Some(&hess_attn),
                compile_threads,
            );

            // 2. Execute quantized `qkv_proj` + causal GQA attention -> gather `o_proj` Hessian
            par_calib(&mut seq_hiddens, &mut all_attn_out, h_dim, h_dim, |seq_h, attn_seq| {
                let mut norm_buf = vec![0.0f32; h_dim];
                let mut qkv_buf = vec![0.0f32; qkv_rows];
                let mut scores = vec![0.0f32; self.cfg.max_seq_len * self.cfg.num_attention_heads];
                let mut k_seq_cache = vec![0.0f32; self.cfg.max_seq_len * kv_dim];
                let mut v_seq_cache = vec![0.0f32; self.cfg.max_seq_len * kv_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                for (pos, (hid, attn_dst)) in seq_h
                    .chunks_exact(h_dim)
                    .zip(attn_seq.chunks_exact_mut(h_dim))
                    .enumerate()
                {
                    rms_norm(hid, &ref_layer.input_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    eval_proj_rot(&qkv_proj, &norm_buf, &mut blut_h, &mut vlut_h, &mut qkv_buf);
                    if let Some(bias) = &ref_layer.qkv_bias {
                        add_inplace(&mut qkv_buf, bias);
                    }

                    let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
                    let (q_slice, kv_rest) = qkv_buf.split_at_mut(h_dim);
                    let (k_slice, v_slice) = kv_rest.split_at_mut(kv_dim);
                    apply_llama_rope_precomp(q_slice, self.cfg.num_attention_heads, head_dim, half_dim, rope_row);
                    apply_llama_rope_precomp(k_slice, self.cfg.num_key_value_heads, head_dim, half_dim, rope_row);
                    k_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k_slice);
                    v_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v_slice);
                    causal_gqa_attention(
                        q_slice,
                        &k_seq_cache,
                        &v_seq_cache,
                        &mut scores,
                        attn_dst,
                        self.cfg.num_attention_heads,
                        kv_group,
                        head_dim,
                        kv_dim,
                        pos,
                        attn_scale,
                    );
                }
            });
            let mut hess_o = FullRowGptqHessian::new(h_dim);
            for row in all_attn_out.chunks_exact(h_dim) {
                hess_o.observe_activation(row);
            }
            hess_o.finalize(BIPOLAR_GPTQ_DAMP);
            let o_proj = compile_proj_spec(
                ref_layer.o_proj.as_fp32_slice(),
                h_dim,
                h_dim,
                o_spec,
                Some(&hess_o),
                compile_threads,
            );

            // Propagate quantized `o_proj` into residual `seq_hiddens`
            par_calib(&mut seq_hiddens, &mut all_attn_out, h_dim, h_dim, |seq_h, attn_seq| {
                let mut rot_h = vec![0.0f32; h_dim];
                let mut proj_out = vec![0.0f32; h_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                for (hid, attn) in seq_h.chunks_exact_mut(h_dim).zip(attn_seq.chunks_exact(h_dim)) {
                    rot_h.copy_from_slice(attn);
                    fwht64_signed_inplace(&mut rot_h);
                    eval_proj_rot(&o_proj, &rot_h, &mut blut_h, &mut vlut_h, &mut proj_out);
                    add_inplace(hid, &proj_out);
                }
            });

            // 3. Post-attention normalized activations -> compile `gate_up_proj`
            let mut hess_mlp = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(
                        hid,
                        &ref_layer.post_attn_norm,
                        self.cfg.rms_norm_eps,
                        &mut norm_buf,
                    );
                    hess_mlp.observe_activation(&norm_buf);
                }
            }
            hess_mlp.finalize(BIPOLAR_GPTQ_DAMP);
            let gate_up_proj = compile_proj_spec(
                ref_layer.gate_up_proj.as_fp32_slice(),
                2 * inter_dim,
                h_dim,
                gate_up_spec,
                Some(&hess_mlp),
                compile_threads,
            );

            // 4. Execute quantized `gate_up_proj` + `SwiGLU` -> gather `down_proj` Hessian
            par_calib(&mut seq_hiddens, &mut all_swiglu, h_dim, inter_dim, |seq_h, sw_seq| {
                let mut norm_buf = vec![0.0f32; h_dim];
                let mut gate_up_buf = vec![0.0f32; 2 * inter_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                for (hid, sw_dst) in seq_h.chunks_exact(h_dim).zip(sw_seq.chunks_exact_mut(inter_dim)) {
                    rms_norm(hid, &ref_layer.post_attn_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    eval_proj_rot(&gate_up_proj, &norm_buf, &mut blut_h, &mut vlut_h, &mut gate_up_buf);
                    let (gate_s, up_s) = gate_up_buf.split_at(inter_dim);
                    swiglu_activate(gate_s, up_s, sw_dst);
                }
            });
            let mut hess_down = FullRowGptqHessian::new(inter_dim);
            for row in all_swiglu.chunks_exact(inter_dim) {
                hess_down.observe_activation(row);
            }
            hess_down.finalize(BIPOLAR_GPTQ_DAMP);
            let down_proj = compile_proj_spec(
                ref_layer.down_proj.as_fp32_slice(),
                h_dim,
                inter_dim,
                down_spec,
                Some(&hess_down),
                compile_threads,
            );

            // Propagate quantized `down_proj` into residual `seq_hiddens`
            par_calib(&mut seq_hiddens, &mut all_swiglu, h_dim, inter_dim, |seq_h, sw_seq| {
                let mut rot_inter = vec![0.0f32; inter_dim];
                let mut proj_out = vec![0.0f32; h_dim];
                let mut blut_i = BipolarActivationLut::new_zeroed(inter_dim);
                let mut vlut_i = Vq2ActivationLut::new_zeroed(inter_dim);
                for (hid, sw) in seq_h.chunks_exact_mut(h_dim).zip(sw_seq.chunks_exact(inter_dim)) {
                    rot_inter.copy_from_slice(sw);
                    fwht64_signed_inplace(&mut rot_inter);
                    eval_proj_rot(&down_proj, &rot_inter, &mut blut_i, &mut vlut_i, &mut proj_out);
                    add_inplace(hid, &proj_out);
                }
            });

            compiled_layers.push(LlamaLayer {
                input_norm: ref_layer.input_norm.clone(),
                post_attn_norm: ref_layer.post_attn_norm.clone(),
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias: ref_layer.qkv_bias.clone(),
                o_bias: ref_layer.o_bias.clone(),
                q_norm: ref_layer.q_norm.clone(),
                k_norm: ref_layer.k_norm.clone(),
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
                moe: None,
            });
        }

        // Final RMSNorm + lm_head GPTQ on layer outputs.
        let mut hess_head = FullRowGptqHessian::new(h_dim);
        for seq_h in &seq_hiddens {
            for hid in seq_h.chunks_exact(h_dim) {
                rms_norm(hid, &self.final_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                hess_head.observe_activation(&norm_buf);
            }
        }
        hess_head.finalize(BIPOLAR_GPTQ_DAMP);
        let lm_head = compile_proj_spec(
            self.lm_head.as_fp32_slice(),
            self.cfg.vocab_size,
            h_dim,
            head_spec,
            Some(&hess_head),
            compile_threads,
        );

        let mut eng = Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: 8,
            kv_window_pref: self.cfg.sliding_window,
            kv_sinks_pref: self.cfg.sink_tokens,
            cfg: self.cfg.clone(),
            mode,
            mean_weight_rel_error: 0.0,
            // Tied embeddings are decoded from the quantized `lm_head`.
            embed_f32_rows: if self.cfg.tie_word_embeddings {
                Vec::new()
            } else {
                self.embed_f32_rows.clone()
            },
            embed_bf16_rows: if self.cfg.tie_word_embeddings {
                Vec::new()
            } else {
                self.embed_bf16_rows.clone()
            },
            embed_nf4: if self.cfg.tie_word_embeddings {
                None
            } else {
                self.embed_nf4.clone()
            },
            layers: compiled_layers,
            final_norm: self.final_norm.clone(),
            lm_head,
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
            moe_runtime: None,
            ws: LlamaWorkspace::new(&self.cfg),
        };
        eng.init_activation_scratch();
        eng
    }

    /// Layer-by-layer Hadamard GPTQ compiler streaming one weight tensor at a time from SafeTensors.
    pub fn compile_streaming_gptq_from_dir(
        model_dir: &Path,
        mode: RealWeightMode,
        calib_sequences: &[Vec<usize>],
        compile_threads: usize,
    ) -> io::Result<Self> {
        assert!(!calib_sequences.is_empty());
        let max_calib_len = calib_sequences.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_calib_len)?;
        let (qkv_spec, o_spec, gate_up_spec, down_spec, head_spec) = proj_specs_for_mode(mode);
        let store = SafeTensorsStore::open_dir(model_dir)?;

        let get_f32_and_release = |name: &str| -> io::Result<Vec<f32>> {
            let v = store
                .tensor(name)
                .map(|t| t.to_f32_vec())
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
            store.release_tensor_pages(name);
            Ok(v)
        };

        let mut embed_bf16_rows = store
            .tensor("model.embed_tokens.weight")
            .map(|t| t.to_bf16_vec())
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        store.release_tensor_pages("model.embed_tokens.weight");

        let h_dim = cfg.hidden_size;
        let inter_dim = cfg.intermediate_size;
        let kv_dim = cfg.kv_dim();
        let qkv_rows = h_dim + 2 * kv_dim;
        let head_dim = cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_group = cfg.num_attention_heads / cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let rope_cos_sin = build_rope_table(&cfg);

        let mut seq_hiddens: Vec<Vec<f32>> = calib_sequences
            .iter()
            .map(|seq| {
                assert!(!seq.is_empty() && seq.len() <= cfg.max_seq_len);
                let mut buf = vec![0.0f32; seq.len() * h_dim];
                for (t, &tok_id) in seq.iter().enumerate() {
                    let src = &embed_bf16_rows[tok_id * h_dim..(tok_id + 1) * h_dim];
                    for (dst, &b16) in buf[t * h_dim..(t + 1) * h_dim].iter_mut().zip(src.iter()) {
                        *dst = bf16_to_f32(b16);
                    }
                }
                buf
            })
            .collect();
        let mut embed_nf4 = None;
        if cfg.tie_word_embeddings {
            embed_bf16_rows = Vec::new();
        } else if crate::memory_budget::MemoryBudgetProfile::active().quantize_untied_embeddings
            && h_dim.is_multiple_of(64)
        {
            embed_nf4 = Some(NfSdotMatrix::from_bf16_u16_embeddings_nf4(
                &embed_bf16_rows,
                cfg.vocab_size,
                h_dim,
                2,
                compile_threads.max(1),
            ));
            embed_bf16_rows = Vec::new();
        }

        let total_tokens: usize = calib_sequences.iter().map(|s| s.len()).sum();
        let mut all_attn_out = vec![0.0f32; total_tokens * h_dim];
        let mut all_swiglu = vec![0.0f32; total_tokens * inter_dim];
        let mut norm_buf = vec![0.0f32; h_dim];
        let mut compiled_layers = Vec::with_capacity(cfg.num_hidden_layers);

        for l in 0..cfg.num_hidden_layers {
            let t_layer = std::time::Instant::now();
            let pfx = format!("model.layers.{l}");
            let input_norm = get_f32_and_release(&format!("{pfx}.input_layernorm.weight"))?;
            let post_attn_norm =
                get_f32_and_release(&format!("{pfx}.post_attention_layernorm.weight"))?;

            // 1. Pre-attention normalized activations -> compile `qkv_proj`
            let mut hess_attn = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(hid, &input_norm, cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    hess_attn.observe_prerotated(&norm_buf);
                }
            }
            hess_attn.finalize(BIPOLAR_GPTQ_DAMP);

            let mut qkv_w = Vec::with_capacity(qkv_rows * h_dim);
            qkv_w.extend_from_slice(&get_f32_and_release(&format!("{pfx}.self_attn.q_proj.weight"))?);
            qkv_w.extend_from_slice(&get_f32_and_release(&format!("{pfx}.self_attn.k_proj.weight"))?);
            qkv_w.extend_from_slice(&get_f32_and_release(&format!("{pfx}.self_attn.v_proj.weight"))?);
            let q_bias_name = format!("{pfx}.self_attn.q_proj.bias");
            let qkv_bias = if store.has_tensor(&q_bias_name) {
                let mut b = Vec::with_capacity(qkv_rows);
                b.extend_from_slice(&get_f32_and_release(&q_bias_name)?);
                b.extend_from_slice(&get_f32_and_release(&format!("{pfx}.self_attn.k_proj.bias"))?);
                b.extend_from_slice(&get_f32_and_release(&format!("{pfx}.self_attn.v_proj.bias"))?);
                Some(b)
            } else {
                None
            };
            let qkv_proj = compile_proj_spec(
                &qkv_w,
                qkv_rows,
                h_dim,
                qkv_spec,
                Some(&hess_attn),
                compile_threads,
            );
            drop(qkv_w);
            drop(hess_attn);

            // 2. Execute quantized `qkv_proj` (+ `qkv_bias`) + causal GQA attention -> rotate for `o_proj`
            par_calib(&mut seq_hiddens, &mut all_attn_out, h_dim, h_dim, |seq_h, attn_seq| {
                let mut norm_buf = vec![0.0f32; h_dim];
                let mut qkv_buf = vec![0.0f32; qkv_rows];
                let mut scores = vec![0.0f32; cfg.max_seq_len * cfg.num_attention_heads];
                let mut k_seq_cache = vec![0.0f32; cfg.max_seq_len * kv_dim];
                let mut v_seq_cache = vec![0.0f32; cfg.max_seq_len * kv_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                for (pos, (hid, attn_dst)) in seq_h
                    .chunks_exact(h_dim)
                    .zip(attn_seq.chunks_exact_mut(h_dim))
                    .enumerate()
                {
                    rms_norm(hid, &input_norm, cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    eval_proj_rot(&qkv_proj, &norm_buf, &mut blut_h, &mut vlut_h, &mut qkv_buf);
                    if let Some(bias) = &qkv_bias {
                        add_inplace(&mut qkv_buf, bias);
                    }

                    let rope_row = &rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
                    let (q_slice, kv_rest) = qkv_buf.split_at_mut(h_dim);
                    let (k_slice, v_slice) = kv_rest.split_at_mut(kv_dim);
                    apply_llama_rope_precomp(q_slice, cfg.num_attention_heads, head_dim, half_dim, rope_row);
                    apply_llama_rope_precomp(k_slice, cfg.num_key_value_heads, head_dim, half_dim, rope_row);
                    k_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k_slice);
                    v_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v_slice);
                    causal_gqa_attention(
                        q_slice,
                        &k_seq_cache,
                        &v_seq_cache,
                        &mut scores,
                        attn_dst,
                        cfg.num_attention_heads,
                        kv_group,
                        head_dim,
                        kv_dim,
                        pos,
                        attn_scale,
                    );
                    fwht64_signed_inplace(attn_dst);
                }
            });
            let mut hess_o = FullRowGptqHessian::new(h_dim);
            for row in all_attn_out.chunks_exact(h_dim) {
                hess_o.observe_prerotated(row);
            }
            hess_o.finalize(BIPOLAR_GPTQ_DAMP);
            let o_w = get_f32_and_release(&format!("{pfx}.self_attn.o_proj.weight"))?;
            let o_proj = compile_proj_spec(&o_w, h_dim, h_dim, o_spec, Some(&hess_o), compile_threads);
            drop(o_w);
            drop(hess_o);

            // Propagate quantized o_proj (all_attn_out is pre-rotated) into seq_hiddens.
            par_calib(&mut seq_hiddens, &mut all_attn_out, h_dim, h_dim, |seq_h, attn_seq| {
                let mut proj_out = vec![0.0f32; h_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                for (hid, attn_rot) in seq_h.chunks_exact_mut(h_dim).zip(attn_seq.chunks_exact(h_dim)) {
                    eval_proj_rot(&o_proj, attn_rot, &mut blut_h, &mut vlut_h, &mut proj_out);
                    add_inplace(hid, &proj_out);
                }
            });

            // 3. Post-attention normalized activations -> compile `gate_up_proj`
            let mut hess_mlp = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(hid, &post_attn_norm, cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    hess_mlp.observe_prerotated(&norm_buf);
                }
            }
            hess_mlp.finalize(BIPOLAR_GPTQ_DAMP);
            let mut gate_up_w = Vec::with_capacity(2 * inter_dim * h_dim);
            gate_up_w.extend_from_slice(&get_f32_and_release(&format!("{pfx}.mlp.gate_proj.weight"))?);
            gate_up_w.extend_from_slice(&get_f32_and_release(&format!("{pfx}.mlp.up_proj.weight"))?);
            let gate_up_proj = compile_proj_spec(
                &gate_up_w,
                2 * inter_dim,
                h_dim,
                gate_up_spec,
                Some(&hess_mlp),
                compile_threads,
            );
            drop(gate_up_w);
            drop(hess_mlp);

            // 4. Execute quantized `gate_up_proj` + `SwiGLU` -> rotate for `down_proj`
            par_calib(&mut seq_hiddens, &mut all_swiglu, h_dim, inter_dim, |seq_h, sw_seq| {
                let mut norm_buf = vec![0.0f32; h_dim];
                let mut gate_up_buf = vec![0.0f32; 2 * inter_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                for (hid, sw_dst) in seq_h.chunks_exact(h_dim).zip(sw_seq.chunks_exact_mut(inter_dim)) {
                    rms_norm(hid, &post_attn_norm, cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    eval_proj_rot(&gate_up_proj, &norm_buf, &mut blut_h, &mut vlut_h, &mut gate_up_buf);
                    let (gate_s, up_s) = gate_up_buf.split_at(inter_dim);
                    swiglu_activate(gate_s, up_s, sw_dst);
                    fwht64_signed_inplace(sw_dst);
                }
            });
            let mut hess_down = FullRowGptqHessian::new(inter_dim);
            for row in all_swiglu.chunks_exact(inter_dim) {
                hess_down.observe_prerotated(row);
            }
            hess_down.finalize(BIPOLAR_GPTQ_DAMP);
            let down_w = get_f32_and_release(&format!("{pfx}.mlp.down_proj.weight"))?;
            let down_proj = compile_proj_spec(
                &down_w,
                h_dim,
                inter_dim,
                down_spec,
                Some(&hess_down),
                compile_threads,
            );
            drop(down_w);
            drop(hess_down);

            // Propagate quantized down_proj (all_swiglu is pre-rotated) into seq_hiddens.
            par_calib(&mut seq_hiddens, &mut all_swiglu, h_dim, inter_dim, |seq_h, sw_seq| {
                let mut proj_out = vec![0.0f32; h_dim];
                let mut blut_i = BipolarActivationLut::new_zeroed(inter_dim);
                let mut vlut_i = Vq2ActivationLut::new_zeroed(inter_dim);
                for (hid, sw_rot) in seq_h.chunks_exact_mut(h_dim).zip(sw_seq.chunks_exact(inter_dim)) {
                    eval_proj_rot(&down_proj, sw_rot, &mut blut_i, &mut vlut_i, &mut proj_out);
                    add_inplace(hid, &proj_out);
                }
            });

            compiled_layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias,
                o_bias: None,
                q_norm: None,
                k_norm: None,
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
                moe: None,
            });
            if cfg.num_hidden_layers >= 16 && ((l + 1) % 4 == 0 || l + 1 == cfg.num_hidden_layers) {
                eprintln!(
                    "  [GPTQ layer {:>2}/{}] {:.1}s",
                    l + 1,
                    cfg.num_hidden_layers,
                    t_layer.elapsed().as_secs_f64()
                );
            }
        }

        drop(all_attn_out);
        drop(all_swiglu);

        // Final RMSNorm + lm_head GPTQ on layer outputs.
        let final_norm = get_f32_and_release("model.norm.weight")?;
        let mut hess_head = FullRowGptqHessian::new(h_dim);
        for seq_h in &seq_hiddens {
            for hid in seq_h.chunks_exact(h_dim) {
                rms_norm(hid, &final_norm, cfg.rms_norm_eps, &mut norm_buf);
                fwht64_signed_inplace(&mut norm_buf);
                hess_head.observe_prerotated(&norm_buf);
            }
        }
        drop(seq_hiddens);
        hess_head.finalize(BIPOLAR_GPTQ_DAMP);
        let lm_head_w = if cfg.tie_word_embeddings {
            get_f32_and_release("model.embed_tokens.weight")?
        } else {
            get_f32_and_release("lm_head.weight")?
        };
        let lm_head = compile_proj_spec(
            &lm_head_w,
            cfg.vocab_size,
            h_dim,
            head_spec,
            Some(&hess_head),
            compile_threads,
        );
        drop(lm_head_w);
        drop(hess_head);

        let mut eng = Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: 8,
            kv_window_pref: cfg.sliding_window,
            kv_sinks_pref: cfg.sink_tokens,
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows: Vec::new(),
            embed_bf16_rows,
            embed_nf4,
            layers: compiled_layers,
            final_norm,
            lm_head,
            rope_cos_sin,
            pool: None,
            moe_runtime: None,
        };
        eng.init_activation_scratch();
        Ok(eng)
    }

    fn init_activation_scratch(&mut self) {
        let is_lut = |b: &LinearBackend| matches!(b, LinearBackend::FusedBipolar(_));
        let is_int8 = |b: &LinearBackend| {
            matches!(
                b,
                LinearBackend::Nibble4(_) | LinearBackend::NfSdot(_) | LinearBackend::Vq2(_)
            )
        };
        let same_q = self.cfg.q_dim() == self.cfg.hidden_size;
        let hidden_lut = is_lut(&self.lm_head)
            || self.layers.iter().any(|l| {
                is_lut(&l.qkv_proj)
                    || (same_q && is_lut(&l.o_proj))
                    || (l.moe.is_none() && is_lut(&l.gate_up_proj))
            });
        let q_lut = self.layers.iter().any(|l| is_lut(&l.o_proj));
        let inter_lut = self
            .layers
            .iter()
            .any(|l| l.moe.is_none() && is_lut(&l.down_proj));
        let any_int8 = is_int8(&self.lm_head)
            || self.layers.iter().any(|l| {
                is_int8(&l.qkv_proj)
                    || is_int8(&l.o_proj)
                    || (l.moe.is_none() && (is_int8(&l.gate_up_proj) || is_int8(&l.down_proj)))
                    || l.moe.is_some()
            });
        self.ws.blut_hidden.lut_enabled = hidden_lut;
        self.ws.blut_q.lut_enabled = q_lut;
        self.ws.blut_inter.lut_enabled = inter_lut;
        if any_int8 {
            if self.ws.blut_hidden.int8.is_none() {
                self.ws.blut_hidden.int8 =
                    Some(Int8ChunkActivation::new_zeroed(self.cfg.hidden_size));
            }
            if self.ws.blut_q.int8.is_none() {
                self.ws.blut_q.int8 = Some(Int8ChunkActivation::new_zeroed(self.cfg.q_dim()));
            }
            if self.ws.blut_inter.int8.is_none() {
                let inter_aligned = self.cfg.intermediate_size.max(64).next_multiple_of(64);
                self.ws.blut_inter.int8 = Some(Int8ChunkActivation::new_zeroed(inter_aligned));
            }
        }
    }

    /// Repacks projections for `SDOT`, interleaves dense `gate_up_proj` in 64-row blocks, and enables INT8
    /// chunk activations.
    pub fn repack_sdot(&mut self) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            self.metal = None;
        }
        let repack = |b: &mut LinearBackend| {
            if let LinearBackend::FusedBipolar(m) = b {
                if (4..=6).contains(&m.num_planes) {
                    *b = LinearBackend::Nibble4(Nibble4SdotMatrix::from_bipolar(m));
                }
            }
        };
        let can_interleave_gu = self.cfg.intermediate_size >= 64
            && self.cfg.intermediate_size.is_multiple_of(64);
        for l in &mut self.layers {
            repack(&mut l.qkv_proj);
            repack(&mut l.o_proj);
            if l.moe.is_some() {
                continue;
            }
            if !l.gate_up_interleaved && can_interleave_gu {
                match &mut l.gate_up_proj {
                    LinearBackend::FusedBipolar(m) if (4..=6).contains(&m.num_planes) => {
                        l.gate_up_proj =
                            LinearBackend::Nibble4(Nibble4SdotMatrix::from_bipolar_with_perm(m, true));
                        l.gate_up_interleaved = true;
                    }
                    LinearBackend::NfSdot(m) => {
                        l.gate_up_proj = LinearBackend::NfSdot(m.to_gate_up_interleaved());
                        l.gate_up_interleaved = true;
                    }
                    LinearBackend::Vq2(m) => {
                        l.gate_up_proj = LinearBackend::Vq2(m.to_gate_up_interleaved());
                        l.gate_up_interleaved = true;
                    }
                    _ => repack(&mut l.gate_up_proj),
                }
            } else {
                repack(&mut l.gate_up_proj);
            }
            repack(&mut l.down_proj);
        }
        repack(&mut self.lm_head);
        self.init_activation_scratch();
    }

    /// Saves a compiled Bipolar, NF, VQ2 or mixed engine; see [`Self::load_bipolar`].
    pub fn save_bipolar(&self, path: &Path) -> io::Result<()> {
        use std::io::Write;
        if self.layers.iter().any(|l| l.gate_up_interleaved) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "save_bipolar must be called before repack_sdot()",
            ));
        }
        fn w_u64(w: &mut impl Write, v: usize) -> io::Result<()> {
            w.write_all(&(v as u64).to_le_bytes())
        }
        fn w_f32s(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
            w_u64(w, v.len())?;
            v.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        fn w_u16s(w: &mut impl Write, v: &[u16]) -> io::Result<()> {
            w_u64(w, v.len())?;
            v.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        fn w_bipolar_body(w: &mut impl Write, m: &FusedBipolarLutMatrix) -> io::Result<()> {
            w_u64(w, m.rows)?;
            w_u64(w, m.cols)?;
            w_u64(w, m.num_planes)?;
            w_u64(w, m.interleaved_data.len())?;
            w.write_all(&m.interleaved_data)?;
            w_u64(w, m.bf16_chunk_scales.len())?;
            m.bf16_chunk_scales.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        fn w_mat_v1(w: &mut impl Write, b: &LinearBackend) -> io::Result<()> {
            let LinearBackend::FusedBipolar(m) = b else {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "non-bipolar projection"));
            };
            w_bipolar_body(w, m)
        }
        fn w_mat_v2(w: &mut impl Write, b: &LinearBackend) -> io::Result<()> {
            match b {
                LinearBackend::FusedBipolar(m) => {
                    w_u64(w, 0)?;
                    w_bipolar_body(w, m)
                }
                LinearBackend::Vq2(m) => {
                    w_u64(w, 1)?;
                    w_u64(w, m.rows)?;
                    w_u64(w, m.cols)?;
                    w_u64(w, m.scale_group_chunks)?;
                    w_u64(w, m.packed_data.len())?;
                    w.write_all(&m.packed_data)?;
                    w_u64(w, m.bf16_chunk_scales.len())?;
                    m.bf16_chunk_scales.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
                }
                LinearBackend::NfSdot(m) => w_nf_mat(w, m),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "save_bipolar expects un-repacked FusedBipolar, NfSdot, or Vq2 projections",
                )),
            }
        }
        fn w_nf_mat(w: &mut impl Write, m: &NfSdotMatrix) -> io::Result<()> {
            w_u64(w, 2)?;
            w_u64(w, m.rows)?;
            w_u64(w, m.cols)?;
            w_u64(w, m.bits)?;
            w_u64(w, m.scale_group_chunks)?;
            let cb_bytes: [u8; 16] = std::array::from_fn(|i| m.codebook[i] as u8);
            w.write_all(&cb_bytes)?;
            w_u64(w, m.pair_stream.len())?;
            w.write_all(&m.pair_stream)?;
            w_u64(w, m.tail_stream.len())?;
            w.write_all(&m.tail_stream)
        }

        let (needs_v2_mats, mode_words) = match self.mode {
            RealWeightMode::KronosBipolarGptq { num_planes } => (false, [0, num_planes, 0, 0, 0, 0]),
            RealWeightMode::KronosBipolarGptqMixedK => (false, [1, 0, 0, 0, 0, 0]),
            RealWeightMode::KronosBipolarGptqPlanes { qkv, o, gate_up, down, head } => {
                (false, [2, qkv, o, gate_up, down, head])
            }
            RealWeightMode::KronosVq2Gptq { scale_group_chunks } => {
                (true, [3, scale_group_chunks, 0, 0, 0, 0])
            }
            RealWeightMode::KronosVq2MixedGptq {
                attn_head_planes,
                scale_group_chunks,
            } => (true, [4, attn_head_planes, scale_group_chunks, 0, 0, 0]),
            RealWeightMode::KronosNfGptq {
                bits,
                scale_group_chunks,
            } => (true, [5, bits, scale_group_chunks, 0, 0, 0]),
            RealWeightMode::KronosNfMixedGptq {
                attn_head_bits,
                ffn_bits,
                scale_group_chunks,
            } => (true, [6, attn_head_bits, ffn_bits, scale_group_chunks, 0, 0]),
            RealWeightMode::KronosNfPlanesGptq {
                qkv,
                o,
                gate_up,
                down,
                head,
                scale_group_chunks,
            } => (
                true,
                [
                    7,
                    (qkv & 0xFF) | (scale_group_chunks << 8),
                    o,
                    gate_up,
                    down,
                    head,
                ],
            ),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "not a bipolar/nf/vq2 engine",
                ));
            }
        };

        let needs_v4 = self.embed_nf4.is_some();
        let needs_v3 = self.layers.iter().any(|l| l.qkv_bias.is_some())
            || !self.embed_bf16_rows.is_empty();
        let version = if needs_v4 {
            4u8
        } else if needs_v3 {
            3u8
        } else if needs_v2_mats {
            2u8
        } else {
            1u8
        };

        let mut w = io::BufWriter::new(fs::File::create(path)?);
        w.write_all(match version {
            4 => KRONOS4_CACHE_MAGIC,
            3 => KRONOS3_CACHE_MAGIC,
            2 => KRONOS2_CACHE_MAGIC,
            _ => BIPOLAR_CACHE_MAGIC,
        })?;
        mode_words.iter().try_for_each(|&v| w_u64(&mut w, v))?;
        w_u64(&mut w, self.layers.len())?;
        for l in &self.layers {
            w_f32s(&mut w, &l.input_norm)?;
            w_f32s(&mut w, &l.post_attn_norm)?;
            if version >= 3 {
                w_f32s(&mut w, l.qkv_bias.as_deref().unwrap_or(&[]))?;
            }
            for b in [&l.qkv_proj, &l.o_proj, &l.gate_up_proj, &l.down_proj] {
                if version >= 2 {
                    w_mat_v2(&mut w, b)?;
                } else {
                    w_mat_v1(&mut w, b)?;
                }
            }
        }
        w_f32s(&mut w, &self.final_norm)?;
        if version >= 2 {
            w_mat_v2(&mut w, &self.lm_head)?;
        } else {
            w_mat_v1(&mut w, &self.lm_head)?;
        }
        w_f32s(&mut w, &self.embed_f32_rows)?;
        if version >= 3 {
            w_u16s(&mut w, &self.embed_bf16_rows)?;
        }
        if version >= 4 {
            if let Some(ref em) = self.embed_nf4 {
                w_u64(&mut w, 1)?;
                w_nf_mat(&mut w, em)?;
            } else {
                w_u64(&mut w, 0)?;
            }
        }
        w.flush()
    }

    /// Loads an engine written by [`Self::save_bipolar`]; `model_dir` supplies `config.json`.
    pub fn load_bipolar(model_dir: &Path, path: &Path, max_seq_len: usize) -> io::Result<Self> {
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_seq_len)?;
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len < BIPOLAR_CACHE_MAGIC.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty bipolar cache"));
        }
        let mmap = RawMmap::map_read_only(&file, len)?;
        let bytes = mmap.as_slice();
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        let (version, magic_len) = if bytes.starts_with(KRONOS4_CACHE_MAGIC) {
            (4u8, KRONOS4_CACHE_MAGIC.len())
        } else if bytes.starts_with(KRONOS3_CACHE_MAGIC) {
            (3u8, KRONOS3_CACHE_MAGIC.len())
        } else if bytes.starts_with(KRONOS2_CACHE_MAGIC) {
            (2u8, KRONOS2_CACHE_MAGIC.len())
        } else if bytes.starts_with(BIPOLAR_CACHE_MAGIC) {
            (1u8, BIPOLAR_CACHE_MAGIC.len())
        } else {
            return Err(bad("bad bipolar/kronos2/kronos3/kronos4 cache magic"));
        };
        let mut rd = CacheReader {
            bytes,
            pos: magic_len,
        };
        let r_bipolar_body = |rd: &mut CacheReader| -> io::Result<LinearBackend> {
            let rows = rd.u64()?;
            let cols = rd.u64()?;
            let num_planes = rd.u64()?;
            let n = rd.u64()?;
            let interleaved_data = rd.take(n)?.to_vec();
            let n = rd.u64()?;
            let bf16_chunk_scales =
                rd.take(n * 2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
            Ok(LinearBackend::FusedBipolar(FusedBipolarLutMatrix {
                rows,
                cols,
                num_planes,
                octets_per_row: cols / 8,
                num_stripes: rows.div_ceil(SIMD_ROW_STRIPE),
                num_chunks: cols / 64,
                interleaved_data,
                bf16_chunk_scales,
            }))
        };
        let r_mat = |rd: &mut CacheReader| -> io::Result<LinearBackend> {
            if version == 1 {
                return r_bipolar_body(rd);
            }
            match rd.u64()? {
                0 => r_bipolar_body(rd),
                1 => {
                    let rows = rd.u64()?;
                    let cols = rd.u64()?;
                    let scale_group_chunks = rd.u64()?.max(1);
                    let n = rd.u64()?;
                    let packed_data = rd.take(n)?.to_vec();
                    let n = rd.u64()?;
                    let bf16_chunk_scales = rd
                        .take(n * 2)?
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let num_chunks = cols / 64;
                    Ok(LinearBackend::Vq2(Vq2LutMatrix {
                        rows,
                        cols,
                        scale_group_chunks,
                        num_stripes: rows.div_ceil(SIMD_ROW_STRIPE),
                        num_chunks,
                        num_scale_groups: num_chunks.div_ceil(scale_group_chunks),
                        packed_data,
                        bf16_chunk_scales,
                    }))
                }
                2 => {
                    let rows = rd.u64()?;
                    let cols = rd.u64()?;
                    let bits = rd.u64()?;
                    let scale_group_chunks = rd.u64()?.max(1);
                    let cb_raw = rd.take(16)?;
                    let codebook: [i8; 16] = std::array::from_fn(|i| cb_raw[i] as i8);
                    let pair_len = rd.u64()?;
                    let pair_stream = crate::simd_lut::PageAlignedBytes::from_slice(rd.take(pair_len)?);
                    let tail_len = rd.u64()?;
                    let tail_stream = crate::simd_lut::PageAlignedBytes::from_slice(rd.take(tail_len)?);
                    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
                    let num_chunks = cols / 64;
                    Ok(LinearBackend::NfSdot(NfSdotMatrix {
                        rows,
                        cols,
                        bits,
                        scale_group_chunks,
                        num_stripes,
                        num_pairs: num_stripes / 2,
                        num_chunks,
                        num_scale_groups: num_chunks.div_ceil(scale_group_chunks),
                        codebook,
                        pair_stream,
                        tail_stream,
                    }))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown matrix tag in K2TKRONOS2/3 cache",
                )),
            }
        };

        let mut mw = [0usize; 6];
        for v in mw.iter_mut() {
            *v = rd.u64()?;
        }
        let mode = match mw[0] {
            0 => RealWeightMode::KronosBipolarGptq { num_planes: mw[1] },
            1 => RealWeightMode::KronosBipolarGptqMixedK,
            2 => RealWeightMode::KronosBipolarGptqPlanes {
                qkv: mw[1],
                o: mw[2],
                gate_up: mw[3],
                down: mw[4],
                head: mw[5],
            },
            3 => RealWeightMode::KronosVq2Gptq {
                scale_group_chunks: mw[1].max(1),
            },
            4 => RealWeightMode::KronosVq2MixedGptq {
                attn_head_planes: mw[1],
                scale_group_chunks: mw[2].max(1),
            },
            5 => RealWeightMode::KronosNfGptq {
                bits: mw[1],
                scale_group_chunks: mw[2].max(1),
            },
            6 => RealWeightMode::KronosNfMixedGptq {
                attn_head_bits: mw[1],
                ffn_bits: mw[2],
                scale_group_chunks: mw[3].max(1),
            },
            7 => RealWeightMode::KronosNfPlanesGptq {
                qkv: mw[1] & 0xFF,
                o: mw[2],
                gate_up: mw[3],
                down: mw[4],
                head: mw[5],
                scale_group_chunks: (mw[1] >> 8).max(1),
            },
            _ => return Err(bad("unknown mode tag in cache")),
        };
        let num_layers = rd.u64()?;
        if num_layers != cfg.num_hidden_layers {
            return Err(bad("layer count does not match config.json"));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let input_norm = rd.f32s()?;
            let post_attn_norm = rd.f32s()?;
            let qkv_bias = if version >= 3 {
                let b = rd.f32s()?;
                if b.is_empty() { None } else { Some(b) }
            } else {
                None
            };
            layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias,
                o_bias: None,
                q_norm: None,
                k_norm: None,
                qkv_proj: r_mat(&mut rd)?,
                o_proj: r_mat(&mut rd)?,
                gate_up_proj: r_mat(&mut rd)?,
                gate_up_interleaved: false,
                down_proj: r_mat(&mut rd)?,
                moe: None,
            });
        }
        let final_norm = rd.f32s()?;
        let lm_head = r_mat(&mut rd)?;
        let embed_f32_rows = rd.f32s()?;
        let auto_q_embed = crate::memory_budget::MemoryBudgetProfile::active()
            .quantize_untied_embeddings
            && cfg.hidden_size.is_multiple_of(64);
        let (mut embed_bf16_rows, mut embed_nf4) = if version >= 3 {
            let n_u16 = rd.u64()?;
            let raw_le = rd.take(n_u16 * 2)?;
            if n_u16 > 0 && auto_q_embed && n_u16 == cfg.vocab_size * cfg.hidden_size {
                let workers = std::thread::available_parallelism()
                    .map_or(4, |p| p.get())
                    .clamp(1, 8);
                let mat = NfSdotMatrix::from_bf16_le_embeddings_nf4(
                    raw_le,
                    cfg.vocab_size,
                    cfg.hidden_size,
                    2,
                    workers,
                );
                (Vec::new(), Some(mat))
            } else {
                let v: Vec<u16> = raw_le
                    .chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect();
                (v, None)
            }
        } else {
            (Vec::new(), None)
        };
        if version >= 4 && rd.u64()? != 0 {
            match r_mat(&mut rd)? {
                LinearBackend::NfSdot(m) => {
                    embed_bf16_rows = Vec::new();
                    embed_nf4 = Some(m);
                }
                _ => return Err(bad("expected NfSdot untied embedding matrix in K2TKRONOS4")),
            }
        }
        let page_aligned = len & !16_383usize;
        if page_aligned > 0 {
            mmap.advise_dontneed(0, page_aligned);
        }
        drop(mmap);

        let budget = crate::memory_budget::MemoryBudgetProfile::active();
        let mut eng = Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: if budget.prefer_q4_kv { 4 } else { 8 },
            kv_window_pref: cfg.sliding_window,
            kv_sinks_pref: cfg.sink_tokens,
            rope_cos_sin: build_rope_table(&cfg),
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows,
            embed_bf16_rows,
            embed_nf4,
            layers,
            final_norm,
            lm_head,
            pool: None,
            moe_runtime: None,
        };
        eng.init_activation_scratch();
        Ok(eng)
    }

    /// Streams a Sparse-MoE SafeTensors directory layer-by-layer and expert-by-expert into a
    /// pinned attention/router/shared-expert/lookahead core (`K2TKRONOS5\n` `.k2tb`) and a
    /// 16 KiB-aligned `F_NOCACHE` expert shard (`.k2tm`), keeping compile-time RAM bounded.
    pub fn compile_streaming_moe_to_files(
        model_dir: &Path,
        core_k2tb_path: &Path,
        experts_k2tm_path: &Path,
        mode: RealWeightMode,
        calib_sequences: &[Vec<usize>],
        compile_threads: usize,
    ) -> io::Result<()> {
        use std::collections::VecDeque;
        use std::io::Write;

        assert!(!calib_sequences.is_empty());
        let max_calib_len = calib_sequences
            .iter()
            .map(|s| s.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_calib_len)?;
        if !cfg.is_moe() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "compile_streaming_moe_to_files expects a Sparse-MoE config",
            ));
        }

        let (qkv_spec, o_spec, gate_up_spec, down_spec, head_spec) = proj_specs_for_mode(mode);
        let (expert_bits, expert_sgc) = match (gate_up_spec, down_spec) {
            (
                ProjSpec::Nf { bits: b1, sgc: s1 },
                ProjSpec::Nf { bits: b2, sgc: s2 },
            ) if b1 == b2 && s1 == s2 && (2..=5).contains(&b1) => (b1, s1),
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Sparse-MoE .k2tm requires matching 2..=5-bit NF/E8Q2 FFN specs",
                ));
            }
        };

        let mode_words = match mode {
            RealWeightMode::KronosBipolarGptq { num_planes } => [0, num_planes, 0, 0, 0, 0],
            RealWeightMode::KronosBipolarGptqMixedK => [1, 0, 0, 0, 0, 0],
            RealWeightMode::KronosBipolarGptqPlanes {
                qkv,
                o,
                gate_up,
                down,
                head,
            } => [2, qkv, o, gate_up, down, head],
            RealWeightMode::KronosVq2Gptq { scale_group_chunks } => {
                [3, scale_group_chunks, 0, 0, 0, 0]
            }
            RealWeightMode::KronosVq2MixedGptq {
                attn_head_planes,
                scale_group_chunks,
            } => [4, attn_head_planes, scale_group_chunks, 0, 0, 0],
            RealWeightMode::KronosNfGptq {
                bits,
                scale_group_chunks,
            } => [5, bits, scale_group_chunks, 0, 0, 0],
            RealWeightMode::KronosNfMixedGptq {
                attn_head_bits,
                ffn_bits,
                scale_group_chunks,
            } => [6, attn_head_bits, ffn_bits, scale_group_chunks, 0, 0],
            RealWeightMode::KronosNfPlanesGptq {
                qkv,
                o,
                gate_up,
                down,
                head,
                scale_group_chunks,
            } => [
                7,
                (qkv & 0xFF) | (scale_group_chunks << 8),
                o,
                gate_up,
                down,
                head,
            ],
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "unsupported RealWeightMode for K2TKRONOS5",
                ));
            }
        };

        let store = SafeTensorsStore::open_dir(model_dir)?;
        let get_f32_and_release = |name: &str| -> io::Result<Vec<f32>> {
            let v = store
                .tensor(name)
                .map(|t| t.to_f32_vec())
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
            store.release_tensor_pages(name);
            Ok(v)
        };

        let mut embed_bf16_rows = store
            .tensor("model.embed_tokens.weight")
            .map(|t| t.to_bf16_vec())
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        store.release_tensor_pages("model.embed_tokens.weight");

        let h_dim = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_rows = q_dim + 2 * kv_dim;
        let head_dim = cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_group = cfg.num_attention_heads / cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let moe_inter = cfg.eff_moe_inter();
        let rope_cos_sin = build_rope_table(&cfg);

        let mut seq_hiddens: Vec<Vec<f32>> = calib_sequences
            .iter()
            .map(|seq| {
                assert!(!seq.is_empty() && seq.len() <= cfg.max_seq_len);
                let mut buf = vec![0.0f32; seq.len() * h_dim];
                for (t, &tok_id) in seq.iter().enumerate() {
                    let src = &embed_bf16_rows[tok_id * h_dim..(tok_id + 1) * h_dim];
                    for (dst, &b16) in buf[t * h_dim..(t + 1) * h_dim].iter_mut().zip(src.iter()) {
                        *dst = bf16_to_f32(b16);
                    }
                }
                buf
            })
            .collect();

        let mut embed_nf4 = None;
        if cfg.tie_word_embeddings {
            embed_bf16_rows = Vec::new();
        } else if h_dim.is_multiple_of(64) {
            embed_nf4 = Some(NfSdotMatrix::from_bf16_u16_embeddings_nf4(
                &embed_bf16_rows,
                cfg.vocab_size,
                h_dim,
                2,
                compile_threads.max(1),
            ));
            embed_bf16_rows = Vec::new();
        }

        let num_moe_layers = (0..cfg.num_hidden_layers)
            .filter(|&l| cfg.layer_is_moe(l))
            .count();
        let mut k2tm_writer = StreamingK2tmWriter::create(
            experts_k2tm_path,
            num_moe_layers,
            cfg.num_experts,
            h_dim,
            moe_inter,
            expert_bits,
            expert_sgc,
        )?;

        let total_tokens: usize = calib_sequences.iter().map(|s| s.len()).sum();
        let mut all_attn_out = vec![0.0f32; total_tokens * q_dim];
        let mut norm_buf = vec![0.0f32; h_dim];
        let mut compiled_layers: Vec<LlamaLayer> = Vec::with_capacity(cfg.num_hidden_layers);
        let mut lookahead_history: VecDeque<(usize, Vec<f32>)> = VecDeque::with_capacity(4);
        let mut moe_slot_counter = 0usize;

        let subsample_post_attn = |seqs: &[Vec<f32>], max_m: usize| -> Vec<f32> {
            let tot: usize = seqs.iter().map(|s| s.len() / h_dim).sum();
            let m = tot.min(max_m).max(1);
            let mut out = vec![0.0f32; m * h_dim];
            let mut flat_refs: Vec<&[f32]> = Vec::with_capacity(tot);
            for s in seqs {
                for row in s.chunks_exact(h_dim) {
                    flat_refs.push(row);
                }
            }
            for i in 0..m {
                let idx = (i * tot) / m;
                out[i * h_dim..(i + 1) * h_dim].copy_from_slice(flat_refs[idx]);
            }
            out
        };

        for l in 0..cfg.num_hidden_layers {
            let t_layer = std::time::Instant::now();
            let pfx = format!("model.layers.{l}");
            let input_norm = get_f32_and_release(&format!("{pfx}.input_layernorm.weight"))?;
            let post_attn_norm =
                get_f32_and_release(&format!("{pfx}.post_attention_layernorm.weight"))?;

            let q_norm_name = format!("{pfx}.self_attn.q_norm.weight");
            let q_norm = if store.has_tensor(&q_norm_name) {
                Some(get_f32_and_release(&q_norm_name)?)
            } else {
                None
            };
            let k_norm_name = format!("{pfx}.self_attn.k_norm.weight");
            let k_norm = if store.has_tensor(&k_norm_name) {
                Some(get_f32_and_release(&k_norm_name)?)
            } else {
                None
            };

            // 1. Pre-attention normalized activations -> compile `qkv_proj`.
            let mut hess_attn = FullRowGptqHessian::new(h_dim);
            for seq_h in &seq_hiddens {
                for hid in seq_h.chunks_exact(h_dim) {
                    rms_norm(hid, &input_norm, cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    hess_attn.observe_prerotated(&norm_buf);
                }
            }
            hess_attn.finalize(BIPOLAR_GPTQ_DAMP);

            let mut qkv_w = Vec::with_capacity(qkv_rows * h_dim);
            qkv_w.extend_from_slice(&get_f32_and_release(&format!(
                "{pfx}.self_attn.q_proj.weight"
            ))?);
            qkv_w.extend_from_slice(&get_f32_and_release(&format!(
                "{pfx}.self_attn.k_proj.weight"
            ))?);
            qkv_w.extend_from_slice(&get_f32_and_release(&format!(
                "{pfx}.self_attn.v_proj.weight"
            ))?);
            let q_bias_name = format!("{pfx}.self_attn.q_proj.bias");
            let qkv_bias = if store.has_tensor(&q_bias_name) {
                let mut b = Vec::with_capacity(qkv_rows);
                b.extend_from_slice(&get_f32_and_release(&q_bias_name)?);
                b.extend_from_slice(&get_f32_and_release(&format!(
                    "{pfx}.self_attn.k_proj.bias"
                ))?);
                b.extend_from_slice(&get_f32_and_release(&format!(
                    "{pfx}.self_attn.v_proj.bias"
                ))?);
                Some(b)
            } else {
                None
            };
            let qkv_proj = compile_proj_spec(
                &qkv_w,
                qkv_rows,
                h_dim,
                qkv_spec,
                Some(&hess_attn),
                compile_threads,
            );
            drop(qkv_w);
            drop(hess_attn);

            // 2. Execute quantized `qkv_proj` (+ optional `q_norm`/`k_norm`) + causal GQA -> rotate `[q_dim]` for `o_proj`.
            par_calib(
                &mut seq_hiddens,
                &mut all_attn_out,
                h_dim,
                q_dim,
                |seq_h, attn_seq| {
                    let mut norm_local = vec![0.0f32; h_dim];
                    let mut qkv_buf = vec![0.0f32; qkv_rows];
                    let mut scores = vec![0.0f32; cfg.max_seq_len * cfg.num_attention_heads];
                    let mut k_seq_cache = vec![0.0f32; cfg.max_seq_len * kv_dim];
                    let mut v_seq_cache = vec![0.0f32; cfg.max_seq_len * kv_dim];
                    let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                    let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                    for (pos, (hid, attn_dst)) in seq_h
                        .chunks_exact(h_dim)
                        .zip(attn_seq.chunks_exact_mut(q_dim))
                        .enumerate()
                    {
                        rms_norm(hid, &input_norm, cfg.rms_norm_eps, &mut norm_local);
                        fwht64_signed_inplace(&mut norm_local);
                        eval_proj_rot(&qkv_proj, &norm_local, &mut blut_h, &mut vlut_h, &mut qkv_buf);
                        if let Some(bias) = &qkv_bias {
                            add_inplace(&mut qkv_buf, bias);
                        }
                        let rope_row = &rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
                        let (q_slice, kv_rest) = qkv_buf.split_at_mut(q_dim);
                        let (k_slice, v_slice) = kv_rest.split_at_mut(kv_dim);
                        if let Some(ref qn) = q_norm {
                            apply_per_head_rms_norm(q_slice, head_dim, qn, cfg.rms_norm_eps);
                        }
                        if let Some(ref kn) = k_norm {
                            apply_per_head_rms_norm(k_slice, head_dim, kn, cfg.rms_norm_eps);
                        }
                        apply_llama_rope_precomp(
                            q_slice,
                            cfg.num_attention_heads,
                            head_dim,
                            half_dim,
                            rope_row,
                        );
                        apply_llama_rope_precomp(
                            k_slice,
                            cfg.num_key_value_heads,
                            head_dim,
                            half_dim,
                            rope_row,
                        );
                        k_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(k_slice);
                        v_seq_cache[pos * kv_dim..(pos + 1) * kv_dim].copy_from_slice(v_slice);
                        causal_gqa_attention(
                            q_slice,
                            &k_seq_cache,
                            &v_seq_cache,
                            &mut scores,
                            attn_dst,
                            cfg.num_attention_heads,
                            kv_group,
                            head_dim,
                            kv_dim,
                            pos,
                            attn_scale,
                        );
                        fwht64_signed_inplace(attn_dst);
                    }
                },
            );

            let mut hess_o = FullRowGptqHessian::new(q_dim);
            for row in all_attn_out.chunks_exact(q_dim) {
                hess_o.observe_prerotated(row);
            }
            hess_o.finalize(BIPOLAR_GPTQ_DAMP);
            let o_w = get_f32_and_release(&format!("{pfx}.self_attn.o_proj.weight"))?;
            let o_bias_name = format!("{pfx}.self_attn.o_proj.bias");
            let o_bias = if store.has_tensor(&o_bias_name) {
                Some(get_f32_and_release(&o_bias_name)?)
            } else {
                None
            };
            let o_proj = compile_proj_spec(
                &o_w,
                h_dim,
                q_dim,
                o_spec,
                Some(&hess_o),
                compile_threads,
            );
            drop(o_w);
            drop(hess_o);

            par_calib(
                &mut seq_hiddens,
                &mut all_attn_out,
                h_dim,
                q_dim,
                |seq_h, attn_seq| {
                    let mut proj_out = vec![0.0f32; h_dim];
                    let mut blut_q = BipolarActivationLut::new_zeroed(q_dim);
                    let mut vlut_q = Vq2ActivationLut::new_zeroed(q_dim);
                    for (hid, attn_rot) in
                        seq_h.chunks_exact_mut(h_dim).zip(attn_seq.chunks_exact(q_dim))
                    {
                        eval_proj_rot(&o_proj, attn_rot, &mut blut_q, &mut vlut_q, &mut proj_out);
                        if let Some(ref ob) = o_bias {
                            add_inplace(&mut proj_out, ob);
                        }
                        add_inplace(hid, &proj_out);
                    }
                },
            );

            // 3. MoE or hybrid-dense FFN compilation.
            let (gate_up_proj, down_proj, moe) = if cfg.layer_is_moe(l) {
                let slot = moe_slot_counter;
                moe_slot_counter += 1;

                // Fit 3-layer-ahead residual drift projector `l_prev -> l` on subsampled post-attention states.
                let cur_sub = subsample_post_attn(&seq_hiddens, 256);
                while lookahead_history
                    .front()
                    .is_some_and(|(pl, _)| *pl + 3 < l)
                {
                    lookahead_history.pop_front();
                }
                if lookahead_history
                    .front()
                    .is_some_and(|(pl, _)| *pl + 3 == l)
                {
                    let (prev_l, prev_sub) = lookahead_history.pop_front().unwrap();
                    let rank = 32.min(h_dim / 2).max(4);
                    let proj =
                        fit_subspace_projector(&prev_sub, &cur_sub, h_dim, rank, 1e-2);
                    if let Some(ref mut prev_moe) = compiled_layers[prev_l].moe {
                        prev_moe.lookahead = Some((l, proj));
                    }
                }
                lookahead_history.push_back((l, cur_sub));

                let router =
                    read_moe_router_f32(&store, l, cfg.num_experts, h_dim, true)?;

                let mut hess_mlp = FullRowGptqHessian::new(h_dim);
                let mut x_norm_all = vec![0.0f32; total_tokens * h_dim];
                let mut act_in_all: Vec<Int8ChunkActivation> = Vec::with_capacity(total_tokens);
                let mut expert_buckets: Vec<Vec<(usize, f32)>> =
                    (0..cfg.num_experts).map(|_| Vec::new()).collect();
                let mut routes_buf = vec![(0usize, 0.0f32); cfg.num_experts_per_tok.max(1)];

                let mut tok_cursor = 0usize;
                for seq_h in &seq_hiddens {
                    for hid in seq_h.chunks_exact(h_dim) {
                        let xn = &mut x_norm_all[tok_cursor * h_dim..(tok_cursor + 1) * h_dim];
                        rms_norm(hid, &post_attn_norm, cfg.rms_norm_eps, xn);
                        let k = router.route_sorted(
                            xn,
                            cfg.num_experts_per_tok,
                            cfg.norm_topk_prob,
                            cfg.router_scoring,
                            &mut routes_buf,
                        );
                        for &(exp_id, weight) in &routes_buf[..k] {
                            expert_buckets[exp_id].push((tok_cursor, weight));
                        }
                        norm_buf.copy_from_slice(xn);
                        fwht64_signed_inplace(&mut norm_buf);
                        hess_mlp.observe_prerotated(&norm_buf);
                        let mut i8a = Int8ChunkActivation::new_zeroed(h_dim);
                        i8a.populate(&norm_buf);
                        act_in_all.push(i8a);
                        tok_cursor += 1;
                    }
                }
                hess_mlp.finalize(BIPOLAR_GPTQ_DAMP);

                let mut moe_out_all = vec![0.0f32; total_tokens * h_dim];

                // Optional shared SwiGLU expert (`Qwen2Moe`).
                let shared_expert = if let Some((s_inter, s_gu_w, s_dn_w, s_gate_w)) =
                    read_moe_shared_expert_f32(
                        &store,
                        l,
                        h_dim,
                        cfg.shared_expert_intermediate_size,
                        true,
                    )?
                {
                    let s_gu = compile_hadamard_gptq_nf(
                        &s_gu_w,
                        2 * s_inter,
                        h_dim,
                        expert_bits,
                        expert_sgc,
                        Some(&hess_mlp),
                        compile_threads,
                    );
                    drop(s_gu_w);
                    let mut hess_s_dn = FullRowGptqHessian::new(s_inter);
                    let mut gu_tmp = vec![0.0f32; 2 * s_inter];
                    let mut sw_tmp = vec![0.0f32; s_inter];
                    for act_in in &act_in_all {
                        // SAFETY: `gu_tmp` has `2 * s_inter == s_gu.rows` floats.
                        unsafe {
                            nf_sdot_stripe_range_int8(
                                &s_gu,
                                act_in,
                                0,
                                s_gu.num_stripes,
                                gu_tmp.as_mut_ptr(),
                            );
                        }
                        let (g, u) = gu_tmp.split_at(s_inter);
                        swiglu_activate(g, u, &mut sw_tmp);
                        fwht64_signed_inplace(&mut sw_tmp);
                        hess_s_dn.observe_prerotated(&sw_tmp);
                    }
                    hess_s_dn.finalize(BIPOLAR_GPTQ_DAMP);
                    let s_dn = compile_hadamard_gptq_nf(
                        &s_dn_w,
                        h_dim,
                        s_inter,
                        expert_bits,
                        expert_sgc,
                        Some(&hess_s_dn),
                        compile_threads,
                    );
                    drop(s_dn_w);
                    drop(hess_s_dn);
                    let se = SharedMoeExpert {
                        inter_dim: s_inter,
                        backend: SharedExpertBackend::Nf {
                            inter_dim: s_inter,
                            gate_up: s_gu,
                            down: s_dn,
                        },
                        gate_weight: s_gate_w,
                    };
                    let mut act_mid = Int8ChunkActivation::new_zeroed(s_inter);
                    for (t, act_in) in act_in_all.iter().enumerate() {
                        se.forward_single(
                            &x_norm_all[t * h_dim..(t + 1) * h_dim],
                            act_in,
                            &mut gu_tmp,
                            &mut sw_tmp,
                            &mut act_mid,
                            &mut moe_out_all[t * h_dim..(t + 1) * h_dim],
                        );
                    }
                    Some(se)
                } else {
                    None
                };

                // Stream routed experts one-by-one into `.k2tm` (non-interleaved `gu`).
                let mut gu_tmp = vec![0.0f32; 2 * moe_inter];
                let mut sw_tmp = vec![0.0f32; moe_inter];
                let mut dn_tmp = vec![0.0f32; h_dim];
                let mut act_mid = Int8ChunkActivation::new_zeroed(moe_inter);
                let fallback_step = (total_tokens / 32).max(1);

                for (e, bucket) in expert_buckets.iter().enumerate() {
                    let (gu_f32, dn_f32) = read_moe_expert_f32(
                        &store,
                        l,
                        e,
                        cfg.num_experts,
                        h_dim,
                        moe_inter,
                        true,
                    )?;
                    let gu_mat = compile_hadamard_gptq_nf(
                        &gu_f32,
                        2 * moe_inter,
                        h_dim,
                        expert_bits,
                        expert_sgc,
                        Some(&hess_mlp),
                        compile_threads,
                    );
                    drop(gu_f32);

                    let mut hess_e_dn = FullRowGptqHessian::new(moe_inter);
                    if bucket.len() >= 16 {
                        for &(tok_idx, _) in bucket {
                            // SAFETY: `gu_tmp` has `2 * moe_inter == gu_mat.rows` floats.
                            unsafe {
                                nf_sdot_stripe_range_int8(
                                    &gu_mat,
                                    &act_in_all[tok_idx],
                                    0,
                                    gu_mat.num_stripes,
                                    gu_tmp.as_mut_ptr(),
                                );
                            }
                            let (g, u) = gu_tmp.split_at(moe_inter);
                            swiglu_activate(g, u, &mut sw_tmp);
                            fwht64_signed_inplace(&mut sw_tmp);
                            hess_e_dn.observe_prerotated(&sw_tmp);
                        }
                    } else {
                        for tok_idx in (0..total_tokens).step_by(fallback_step) {
                            // SAFETY: `gu_tmp` has `2 * moe_inter == gu_mat.rows` floats.
                            unsafe {
                                nf_sdot_stripe_range_int8(
                                    &gu_mat,
                                    &act_in_all[tok_idx],
                                    0,
                                    gu_mat.num_stripes,
                                    gu_tmp.as_mut_ptr(),
                                );
                            }
                            let (g, u) = gu_tmp.split_at(moe_inter);
                            swiglu_activate(g, u, &mut sw_tmp);
                            fwht64_signed_inplace(&mut sw_tmp);
                            hess_e_dn.observe_prerotated(&sw_tmp);
                        }
                    }
                    hess_e_dn.finalize(BIPOLAR_GPTQ_DAMP);
                    let dn_mat = compile_hadamard_gptq_nf(
                        &dn_f32,
                        h_dim,
                        moe_inter,
                        expert_bits,
                        expert_sgc,
                        Some(&hess_e_dn),
                        compile_threads,
                    );
                    drop(dn_f32);
                    drop(hess_e_dn);

                    for &(tok_idx, weight) in bucket {
                        // SAFETY: `gu_tmp` and `dn_tmp` match `gu_mat.rows` and `dn_mat.rows`.
                        unsafe {
                            nf_sdot_stripe_range_int8(
                                &gu_mat,
                                &act_in_all[tok_idx],
                                0,
                                gu_mat.num_stripes,
                                gu_tmp.as_mut_ptr(),
                            );
                        }
                        let (g, u) = gu_tmp.split_at(moe_inter);
                        swiglu_activate(g, u, &mut sw_tmp);
                        fwht64_signed_inplace(&mut sw_tmp);
                        act_mid.populate(&sw_tmp);
                        // SAFETY: `dn_tmp` has `h_dim == dn_mat.rows` floats.
                        unsafe {
                            nf_sdot_stripe_range_int8(
                                &dn_mat,
                                &act_mid,
                                0,
                                dn_mat.num_stripes,
                                dn_tmp.as_mut_ptr(),
                            );
                        }
                        let dst = &mut moe_out_all[tok_idx * h_dim..(tok_idx + 1) * h_dim];
                        for (d, &s) in dst.iter_mut().zip(dn_tmp.iter()) {
                            *d += weight * s;
                        }
                    }

                    k2tm_writer.write_expert(&gu_mat, &dn_mat)?;
                }

                let mut tok_cursor = 0usize;
                for seq_h in &mut seq_hiddens {
                    for hid in seq_h.chunks_exact_mut(h_dim) {
                        add_inplace(
                            hid,
                            &moe_out_all[tok_cursor * h_dim..(tok_cursor + 1) * h_dim],
                        );
                        tok_cursor += 1;
                    }
                }

                (
                    LinearBackend::Empty,
                    LinearBackend::Empty,
                    Some(MoeLayerCore {
                        moe_slot_idx: slot,
                        router,
                        shared_expert,
                        lookahead: None,
                    }),
                )
            } else {
                let inter_dim = cfg.intermediate_size;
                let mut all_swiglu = vec![0.0f32; total_tokens * inter_dim];
                let mut hess_mlp = FullRowGptqHessian::new(h_dim);
                for seq_h in &seq_hiddens {
                    for hid in seq_h.chunks_exact(h_dim) {
                        rms_norm(hid, &post_attn_norm, cfg.rms_norm_eps, &mut norm_buf);
                        fwht64_signed_inplace(&mut norm_buf);
                        hess_mlp.observe_prerotated(&norm_buf);
                    }
                }
                hess_mlp.finalize(BIPOLAR_GPTQ_DAMP);
                let mut gate_up_w = Vec::with_capacity(2 * inter_dim * h_dim);
                gate_up_w.extend_from_slice(&get_f32_and_release(&format!(
                    "{pfx}.mlp.gate_proj.weight"
                ))?);
                gate_up_w.extend_from_slice(&get_f32_and_release(&format!(
                    "{pfx}.mlp.up_proj.weight"
                ))?);
                let gu_proj = compile_proj_spec(
                    &gate_up_w,
                    2 * inter_dim,
                    h_dim,
                    gate_up_spec,
                    Some(&hess_mlp),
                    compile_threads,
                );
                drop(gate_up_w);
                drop(hess_mlp);

                par_calib(
                    &mut seq_hiddens,
                    &mut all_swiglu,
                    h_dim,
                    inter_dim,
                    |seq_h, sw_seq| {
                        let mut norm_local = vec![0.0f32; h_dim];
                        let mut gu_buf = vec![0.0f32; 2 * inter_dim];
                        let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                        let mut vlut_h = Vq2ActivationLut::new_zeroed(h_dim);
                        for (hid, sw_dst) in seq_h
                            .chunks_exact(h_dim)
                            .zip(sw_seq.chunks_exact_mut(inter_dim))
                        {
                            rms_norm(hid, &post_attn_norm, cfg.rms_norm_eps, &mut norm_local);
                            fwht64_signed_inplace(&mut norm_local);
                            eval_proj_rot(&gu_proj, &norm_local, &mut blut_h, &mut vlut_h, &mut gu_buf);
                            let (g, u) = gu_buf.split_at(inter_dim);
                            swiglu_activate(g, u, sw_dst);
                            fwht64_signed_inplace(sw_dst);
                        }
                    },
                );
                let mut hess_down = FullRowGptqHessian::new(inter_dim);
                for row in all_swiglu.chunks_exact(inter_dim) {
                    hess_down.observe_prerotated(row);
                }
                hess_down.finalize(BIPOLAR_GPTQ_DAMP);
                let down_w = get_f32_and_release(&format!("{pfx}.mlp.down_proj.weight"))?;
                let dn_proj = compile_proj_spec(
                    &down_w,
                    h_dim,
                    inter_dim,
                    down_spec,
                    Some(&hess_down),
                    compile_threads,
                );
                drop(down_w);
                drop(hess_down);

                par_calib(
                    &mut seq_hiddens,
                    &mut all_swiglu,
                    h_dim,
                    inter_dim,
                    |seq_h, sw_seq| {
                        let mut proj_out = vec![0.0f32; h_dim];
                        let mut blut_i = BipolarActivationLut::new_zeroed(inter_dim);
                        let mut vlut_i = Vq2ActivationLut::new_zeroed(inter_dim);
                        for (hid, sw_rot) in seq_h
                            .chunks_exact_mut(h_dim)
                            .zip(sw_seq.chunks_exact(inter_dim))
                        {
                            eval_proj_rot(&dn_proj, sw_rot, &mut blut_i, &mut vlut_i, &mut proj_out);
                            add_inplace(hid, &proj_out);
                        }
                    },
                );
                (gu_proj, dn_proj, None)
            };

            compiled_layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias,
                o_bias,
                q_norm,
                k_norm,
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
                moe,
            });
            if cfg.num_hidden_layers >= 12 && ((l + 1) % 4 == 0 || l + 1 == cfg.num_hidden_layers) {
                eprintln!(
                    "  [MoE-GPTQ layer {:>2}/{}] {:.1}s",
                    l + 1,
                    cfg.num_hidden_layers,
                    t_layer.elapsed().as_secs_f64()
                );
            }
        }

        drop(all_attn_out);
        drop(lookahead_history);
        k2tm_writer.finish()?;

        let final_norm = get_f32_and_release("model.norm.weight")?;
        let mut hess_head = FullRowGptqHessian::new(h_dim);
        for seq_h in &seq_hiddens {
            for hid in seq_h.chunks_exact(h_dim) {
                rms_norm(hid, &final_norm, cfg.rms_norm_eps, &mut norm_buf);
                fwht64_signed_inplace(&mut norm_buf);
                hess_head.observe_prerotated(&norm_buf);
            }
        }
        drop(seq_hiddens);
        hess_head.finalize(BIPOLAR_GPTQ_DAMP);
        let lm_head_w = if cfg.tie_word_embeddings {
            get_f32_and_release("model.embed_tokens.weight")?
        } else {
            get_f32_and_release("lm_head.weight")?
        };
        let lm_head = compile_proj_spec(
            &lm_head_w,
            cfg.vocab_size,
            h_dim,
            head_spec,
            Some(&hess_head),
            compile_threads,
        );
        drop(lm_head_w);
        drop(hess_head);

        // Write `K2TKRONOS5\n` pinned core `.k2tb`.
        fn w_u64(w: &mut impl Write, v: usize) -> io::Result<()> {
            w.write_all(&(v as u64).to_le_bytes())
        }
        fn w_f32s(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
            w_u64(w, v.len())?;
            v.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        fn w_u16s(w: &mut impl Write, v: &[u16]) -> io::Result<()> {
            w_u64(w, v.len())?;
            v.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        fn w_nf_mat(w: &mut impl Write, m: &NfSdotMatrix) -> io::Result<()> {
            w_u64(w, 2)?;
            w_u64(w, m.rows)?;
            w_u64(w, m.cols)?;
            w_u64(w, m.bits)?;
            w_u64(w, m.scale_group_chunks)?;
            let cb_bytes: [u8; 16] = std::array::from_fn(|i| m.codebook[i] as u8);
            w.write_all(&cb_bytes)?;
            w_u64(w, m.pair_stream.len())?;
            w.write_all(&m.pair_stream)?;
            w_u64(w, m.tail_stream.len())?;
            w.write_all(&m.tail_stream)
        }
        fn w_mat_v2(w: &mut impl Write, b: &LinearBackend) -> io::Result<()> {
            match b {
                LinearBackend::FusedBipolar(m) => {
                    w_u64(w, 0)?;
                    w_u64(w, m.rows)?;
                    w_u64(w, m.cols)?;
                    w_u64(w, m.num_planes)?;
                    w_u64(w, m.interleaved_data.len())?;
                    w.write_all(&m.interleaved_data)?;
                    w_u64(w, m.bf16_chunk_scales.len())?;
                    m.bf16_chunk_scales
                        .iter()
                        .try_for_each(|x| w.write_all(&x.to_le_bytes()))
                }
                LinearBackend::Vq2(m) => {
                    w_u64(w, 1)?;
                    w_u64(w, m.rows)?;
                    w_u64(w, m.cols)?;
                    w_u64(w, m.scale_group_chunks)?;
                    w_u64(w, m.packed_data.len())?;
                    w.write_all(&m.packed_data)?;
                    w_u64(w, m.bf16_chunk_scales.len())?;
                    m.bf16_chunk_scales
                        .iter()
                        .try_for_each(|x| w.write_all(&x.to_le_bytes()))
                }
                LinearBackend::NfSdot(m) => w_nf_mat(w, m),
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "unexpected matrix backend in K2TKRONOS5",
                )),
            }
        }

        let mut w = io::BufWriter::new(fs::File::create(core_k2tb_path)?);
        w.write_all(KRONOS5_CACHE_MAGIC)?;
        mode_words.iter().try_for_each(|&v| w_u64(&mut w, v))?;
        w_u64(&mut w, expert_bits)?;
        w_u64(&mut w, expert_sgc)?;
        w_u64(&mut w, compiled_layers.len())?;
        for l in &compiled_layers {
            w_f32s(&mut w, &l.input_norm)?;
            w_f32s(&mut w, &l.post_attn_norm)?;
            w_f32s(&mut w, l.qkv_bias.as_deref().unwrap_or(&[]))?;
            w_f32s(&mut w, l.o_bias.as_deref().unwrap_or(&[]))?;
            w_f32s(&mut w, l.q_norm.as_deref().unwrap_or(&[]))?;
            w_f32s(&mut w, l.k_norm.as_deref().unwrap_or(&[]))?;
            w_mat_v2(&mut w, &l.qkv_proj)?;
            w_mat_v2(&mut w, &l.o_proj)?;
            if let Some(ref moe) = l.moe {
                w_u64(&mut w, 1)?;
                w_u64(&mut w, moe.moe_slot_idx)?;
                w_u64(&mut w, moe.router.num_experts)?;
                w_u64(&mut w, moe.router.hidden_size)?;
                w_f32s(&mut w, &moe.router.weights)?;
                w_f32s(&mut w, moe.router.bias.as_deref().unwrap_or(&[]))?;
                if let Some(ref sh) = moe.shared_expert {
                    w_u64(&mut w, 1)?;
                    w_u64(&mut w, sh.inter_dim)?;
                    w_f32s(&mut w, sh.gate_weight.as_deref().unwrap_or(&[]))?;
                    match &sh.backend {
                        SharedExpertBackend::Nf { gate_up, down, .. } => {
                            w_nf_mat(&mut w, gate_up)?;
                            w_nf_mat(&mut w, down)?;
                        }
                        _ => {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidInput,
                                "shared expert in K2TKRONOS5 must be NfSdot",
                            ));
                        }
                    }
                } else {
                    w_u64(&mut w, 0)?;
                }
                if let Some((future_l, ref proj)) = moe.lookahead {
                    w_u64(&mut w, 1)?;
                    w_u64(&mut w, future_l)?;
                    w_u64(&mut w, proj.d_model)?;
                    w_u64(&mut w, proj.rank)?;
                    w_f32s(&mut w, &proj.u_proj)?;
                    w_f32s(&mut w, &proj.v_proj)?;
                } else {
                    w_u64(&mut w, 0)?;
                }
            } else {
                w_u64(&mut w, 0)?;
                w_mat_v2(&mut w, &l.gate_up_proj)?;
                w_mat_v2(&mut w, &l.down_proj)?;
            }
        }
        w_f32s(&mut w, &final_norm)?;
        w_mat_v2(&mut w, &lm_head)?;
        w_f32s(&mut w, &[])?;
        w_u16s(&mut w, &embed_bf16_rows)?;
        if let Some(ref em) = embed_nf4 {
            w_u64(&mut w, 1)?;
            w_nf_mat(&mut w, em)?;
        } else {
            w_u64(&mut w, 0)?;
        }
        w.flush()
    }

    /// Loads a hybrid Sparse-MoE engine from its `K2TKRONOS5\n` pinned core (`.k2tb`) and
    /// 16 KiB-aligned expert shard (`.k2tm`) under `policy` and the active [`MemoryBudgetProfile`].
    pub fn load_moe_from_files(
        model_dir: &Path,
        core_k2tb_path: &Path,
        experts_k2tm_path: &Path,
        max_seq_len: usize,
        policy: MoeExpertOpenPolicy,
    ) -> io::Result<Self> {
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_seq_len)?;
        let file = fs::File::open(core_k2tb_path)?;
        let len = file.metadata()?.len() as usize;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if len < KRONOS5_CACHE_MAGIC.len() {
            return Err(bad("empty K2TKRONOS5 cache"));
        }
        let mmap = RawMmap::map_read_only(&file, len)?;
        let bytes = mmap.as_slice();
        if !bytes.starts_with(KRONOS5_CACHE_MAGIC) {
            return Err(bad("bad K2TKRONOS5 cache magic"));
        }
        let mut rd = CacheReader {
            bytes,
            pos: KRONOS5_CACHE_MAGIC.len(),
        };
        let r_mat = |rd: &mut CacheReader| -> io::Result<LinearBackend> {
            match rd.u64()? {
                0 => {
                    let rows = rd.u64()?;
                    let cols = rd.u64()?;
                    let num_planes = rd.u64()?;
                    let n = rd.u64()?;
                    let interleaved_data = rd.take(n)?.to_vec();
                    let n = rd.u64()?;
                    let bf16_chunk_scales = rd
                        .take(n * 2)?
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    Ok(LinearBackend::FusedBipolar(FusedBipolarLutMatrix {
                        rows,
                        cols,
                        num_planes,
                        octets_per_row: cols / 8,
                        num_stripes: rows.div_ceil(SIMD_ROW_STRIPE),
                        num_chunks: cols / 64,
                        interleaved_data,
                        bf16_chunk_scales,
                    }))
                }
                1 => {
                    let rows = rd.u64()?;
                    let cols = rd.u64()?;
                    let scale_group_chunks = rd.u64()?.max(1);
                    let n = rd.u64()?;
                    let packed_data = rd.take(n)?.to_vec();
                    let n = rd.u64()?;
                    let bf16_chunk_scales = rd
                        .take(n * 2)?
                        .chunks_exact(2)
                        .map(|c| u16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    let num_chunks = cols / 64;
                    Ok(LinearBackend::Vq2(Vq2LutMatrix {
                        rows,
                        cols,
                        scale_group_chunks,
                        num_stripes: rows.div_ceil(SIMD_ROW_STRIPE),
                        num_chunks,
                        num_scale_groups: num_chunks.div_ceil(scale_group_chunks),
                        packed_data,
                        bf16_chunk_scales,
                    }))
                }
                2 => {
                    let rows = rd.u64()?;
                    let cols = rd.u64()?;
                    let bits = rd.u64()?;
                    let scale_group_chunks = rd.u64()?.max(1);
                    let cb_raw = rd.take(16)?;
                    let codebook: [i8; 16] = std::array::from_fn(|i| cb_raw[i] as i8);
                    let pair_len = rd.u64()?;
                    let pair_stream =
                        crate::simd_lut::PageAlignedBytes::from_slice(rd.take(pair_len)?);
                    let tail_len = rd.u64()?;
                    let tail_stream =
                        crate::simd_lut::PageAlignedBytes::from_slice(rd.take(tail_len)?);
                    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
                    let num_chunks = cols / 64;
                    Ok(LinearBackend::NfSdot(NfSdotMatrix {
                        rows,
                        cols,
                        bits,
                        scale_group_chunks,
                        num_stripes,
                        num_pairs: num_stripes / 2,
                        num_chunks,
                        num_scale_groups: num_chunks.div_ceil(scale_group_chunks),
                        codebook,
                        pair_stream,
                        tail_stream,
                    }))
                }
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown matrix tag in K2TKRONOS5",
                )),
            }
        };

        let mut mw = [0usize; 6];
        for v in mw.iter_mut() {
            *v = rd.u64()?;
        }
        let mode = match mw[0] {
            0 => RealWeightMode::KronosBipolarGptq { num_planes: mw[1] },
            1 => RealWeightMode::KronosBipolarGptqMixedK,
            2 => RealWeightMode::KronosBipolarGptqPlanes {
                qkv: mw[1],
                o: mw[2],
                gate_up: mw[3],
                down: mw[4],
                head: mw[5],
            },
            3 => RealWeightMode::KronosVq2Gptq {
                scale_group_chunks: mw[1].max(1),
            },
            4 => RealWeightMode::KronosVq2MixedGptq {
                attn_head_planes: mw[1],
                scale_group_chunks: mw[2].max(1),
            },
            5 => RealWeightMode::KronosNfGptq {
                bits: mw[1],
                scale_group_chunks: mw[2].max(1),
            },
            6 => RealWeightMode::KronosNfMixedGptq {
                attn_head_bits: mw[1],
                ffn_bits: mw[2],
                scale_group_chunks: mw[3].max(1),
            },
            7 => RealWeightMode::KronosNfPlanesGptq {
                qkv: mw[1] & 0xFF,
                o: mw[2],
                gate_up: mw[3],
                down: mw[4],
                head: mw[5],
                scale_group_chunks: (mw[1] >> 8).max(1),
            },
            _ => return Err(bad("unknown mode tag in K2TKRONOS5")),
        };
        let expert_bits = rd.u64()?;
        let expert_sgc = rd.u64()?.max(1);
        let num_layers = rd.u64()?;
        if num_layers != cfg.num_hidden_layers {
            return Err(bad("K2TKRONOS5 layer count mismatch against config.json"));
        }

        let opt_f32s = |rd: &mut CacheReader| -> io::Result<Option<Vec<f32>>> {
            let v = rd.f32s()?;
            if v.is_empty() { Ok(None) } else { Ok(Some(v)) }
        };

        let mut layers = Vec::with_capacity(num_layers);
        let mut max_shared_inter = 0usize;
        for _ in 0..num_layers {
            let input_norm = rd.f32s()?;
            let post_attn_norm = rd.f32s()?;
            let qkv_bias = opt_f32s(&mut rd)?;
            let o_bias = opt_f32s(&mut rd)?;
            let q_norm = opt_f32s(&mut rd)?;
            let k_norm = opt_f32s(&mut rd)?;
            let qkv_proj = r_mat(&mut rd)?;
            let o_proj = r_mat(&mut rd)?;
            let is_moe_layer = rd.u64()? != 0;
            let (gate_up_proj, down_proj, moe) = if is_moe_layer {
                let moe_slot_idx = rd.u64()?;
                let r_exp = rd.u64()?;
                let r_hid = rd.u64()?;
                let r_weights = rd.f32s()?;
                let r_bias = opt_f32s(&mut rd)?;
                let router = DenseMoeRouter::new(r_exp, r_hid, r_weights, r_bias);

                let shared_expert = if rd.u64()? != 0 {
                    let s_inter = rd.u64()?;
                    max_shared_inter = max_shared_inter.max(s_inter);
                    let gate_weight = opt_f32s(&mut rd)?;
                    let LinearBackend::NfSdot(gate_up) = r_mat(&mut rd)? else {
                        return Err(bad("expected NfSdot shared gate_up"));
                    };
                    let LinearBackend::NfSdot(down) = r_mat(&mut rd)? else {
                        return Err(bad("expected NfSdot shared down"));
                    };
                    Some(SharedMoeExpert {
                        inter_dim: s_inter,
                        backend: SharedExpertBackend::Nf {
                            inter_dim: s_inter,
                            gate_up,
                            down,
                        },
                        gate_weight,
                    })
                } else {
                    None
                };

                let lookahead = if rd.u64()? != 0 {
                    let future_l = rd.u64()?;
                    let d_model = rd.u64()?;
                    let rank = rd.u64()?;
                    let u_proj = rd.f32s()?;
                    let v_proj = rd.f32s()?;
                    Some((
                        future_l,
                        SubspaceProjector {
                            d_model,
                            rank,
                            u_proj,
                            v_proj,
                        },
                    ))
                } else {
                    None
                };

                (
                    LinearBackend::Empty,
                    LinearBackend::Empty,
                    Some(MoeLayerCore {
                        moe_slot_idx,
                        router,
                        shared_expert,
                        lookahead,
                    }),
                )
            } else {
                (r_mat(&mut rd)?, r_mat(&mut rd)?, None)
            };

            layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias,
                o_bias,
                q_norm,
                k_norm,
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
                moe,
            });
        }

        let final_norm = rd.f32s()?;
        let lm_head = r_mat(&mut rd)?;
        let embed_f32_rows = rd.f32s()?;
        let n_u16 = rd.u64()?;
        let raw_le = rd.take(n_u16 * 2)?;
        let mut embed_bf16_rows: Vec<u16> = raw_le
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        let mut embed_nf4 = None;
        if rd.u64()? != 0 {
            let LinearBackend::NfSdot(m) = r_mat(&mut rd)? else {
                return Err(bad("expected NfSdot untied embedding matrix in K2TKRONOS5"));
            };
            embed_bf16_rows = Vec::new();
            embed_nf4 = Some(m);
        } else if !embed_bf16_rows.is_empty()
            && crate::memory_budget::MemoryBudgetProfile::active().quantize_untied_embeddings
            && cfg.hidden_size.is_multiple_of(64)
        {
            let workers = std::thread::available_parallelism()
                .map_or(4, |p| p.get())
                .clamp(1, 8);
            embed_nf4 = Some(NfSdotMatrix::from_bf16_u16_embeddings_nf4(
                &embed_bf16_rows,
                cfg.vocab_size,
                cfg.hidden_size,
                2,
                workers,
            ));
            embed_bf16_rows = Vec::new();
        }
        let page_aligned = len & !16_383usize;
        if page_aligned > 0 {
            mmap.advise_dontneed(0, page_aligned);
        }
        drop(mmap);

        let budget = crate::memory_budget::MemoryBudgetProfile::active();
        let constrained = matches!(
            budget.tier,
            crate::memory_budget::MemoryTier::Mobile5G | crate::memory_budget::MemoryTier::Edge8G
        );
        let (kv_window_pref, kv_sinks_pref) =
            if cfg.sliding_window.is_none() && constrained && cfg.max_seq_len > 2048 {
                (Some(2048), 4)
            } else {
                (cfg.sliding_window, cfg.sink_tokens)
            };

        let mut eng = Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: if budget.prefer_q4_kv { 4 } else { 8 },
            kv_window_pref,
            kv_sinks_pref,
            rope_cos_sin: build_rope_table(&cfg),
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows,
            embed_bf16_rows,
            embed_nf4,
            layers,
            final_norm,
            lm_head,
            pool: None,
            moe_runtime: None,
        };
        eng.repack_sdot();
        if matches!(eng.cfg.head_dim(), 64 | 128) {
            eng.enable_q8_kv_cache(true);
        }

        let pinned_bytes = eng.pinned_core_bytes();
        let kv_and_scratch_bytes = eng.kv_cache_bytes() + (24 << 20);
        let moe_rt = SparseMoeRuntime::open_k2tm(
            experts_k2tm_path,
            eng.cfg.hidden_size,
            eng.cfg.eff_moe_inter(),
            eng.cfg.num_experts,
            eng.cfg.num_experts_per_tok,
            eng.cfg.norm_topk_prob,
            eng.cfg.router_scoring,
            max_shared_inter,
            expert_bits,
            expert_sgc,
            pinned_bytes,
            kv_and_scratch_bytes,
            0,
            policy,
        )?;
        eng.moe_runtime = Some(moe_rt);
        Ok(eng)
    }

    /// Constructs a synthetic Sparse-MoE `RealLlamaEngine` with full `nf4g128` pinned attention,
    /// `nf4g128` untied embeddings and `lm_head`, `Q4`/`Q8` sliding-window KV cache, Zipf-skewed
    /// routers, 3-layer `SubspaceProjector` lookahead, and a physical `.k2tm` expert shard on disk.
    pub fn new_synthetic_moe_for_bench(
        cfg: LlamaConfig,
        expert_bits: usize,
        experts_k2tm_path: &Path,
        draft_reserve_bytes: usize,
        policy: MoeExpertOpenPolicy,
        seed: u64,
    ) -> io::Result<Self> {
        let h_dim = cfg.hidden_size;
        let q_dim = cfg.q_dim();
        let kv_dim = cfg.kv_dim();
        let qkv_rows = q_dim + 2 * kv_dim;
        let num_layers = cfg.num_hidden_layers;
        let num_exp = cfg.num_experts;
        let rank = 32usize.min(h_dim);

        let mut rng = seed ^ 0xA5A5_5A5A_F0F0_0F0F;
        let mut next_f32 = || -> f32 {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            let u = (rng.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32
                * (1.0 / ((1u32 << 24) as f32));
            (u - 0.5) * 2.0
        };

        let mut layers = Vec::with_capacity(num_layers);
        let router_scale = 0.85f32 / (h_dim as f32).sqrt();
        for l in 0..num_layers {
            let input_norm = vec![1.0f32; h_dim];
            let post_attn_norm = vec![1.0f32; h_dim];
            let qkv_proj = LinearBackend::NfSdot(NfSdotMatrix::new_synthetic(
                qkv_rows,
                h_dim,
                4,
                2,
                seed.wrapping_add((l * 17 + 1) as u64),
            ));
            let o_proj = LinearBackend::NfSdot(NfSdotMatrix::new_synthetic(
                h_dim,
                q_dim,
                4,
                2,
                seed.wrapping_add((l * 17 + 2) as u64),
            ));
            let mut r_weights = Vec::with_capacity(num_exp * h_dim);
            for _ in 0..(num_exp * h_dim) {
                r_weights.push(next_f32() * router_scale);
            }
            let mut r_bias = vec![0.0f32; num_exp];
            for rank_idx in 0..num_exp {
                let eid = (rank_idx * 7 + l * 3) % num_exp;
                r_bias[eid] = -1.65f32 * ((rank_idx + 1) as f32).ln();
            }
            let router = DenseMoeRouter::new(num_exp, h_dim, r_weights, Some(r_bias));
            let lookahead = (l + 3 < num_layers).then(|| {
                let mut u_proj = vec![0.0f32; rank * h_dim];
                let mut v_proj = vec![0.0f32; h_dim * rank];
                for r in 0..rank {
                    u_proj[r * h_dim + r] = 1.0;
                    v_proj[r * rank + r] = 1.0;
                }
                (
                    l + 3,
                    SubspaceProjector {
                        d_model: h_dim,
                        rank,
                        u_proj,
                        v_proj,
                    },
                )
            });

            layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias: None,
                o_bias: None,
                q_norm: None,
                k_norm: None,
                qkv_proj,
                o_proj,
                gate_up_proj: LinearBackend::Empty,
                gate_up_interleaved: false,
                down_proj: LinearBackend::Empty,
                moe: Some(MoeLayerCore {
                    moe_slot_idx: l,
                    router,
                    shared_expert: None,
                    lookahead,
                }),
            });
        }

        let final_norm = vec![1.0f32; h_dim];
        let lm_head = LinearBackend::NfSdot(NfSdotMatrix::new_synthetic(
            cfg.vocab_size,
            h_dim,
            4,
            2,
            seed.wrapping_add(0xBEEF),
        ));
        let embed_nf4 = (!cfg.tie_word_embeddings).then(|| {
            NfSdotMatrix::new_synthetic(
                cfg.vocab_size,
                h_dim,
                4,
                2,
                seed.wrapping_add(0xCAFE),
            )
        });

        let budget = crate::memory_budget::MemoryBudgetProfile::active();
        let kv_bits = if budget.prefer_q4_kv { 4 } else { 8 };
        let mut eng = Self {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            metal: None,
            kv_bits_pref: kv_bits,
            kv_window_pref: cfg.sliding_window,
            kv_sinks_pref: cfg.sink_tokens,
            rope_cos_sin: build_rope_table(&cfg),
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode: RealWeightMode::KronosNfMixedGptq {
                attn_head_bits: 4,
                ffn_bits: expert_bits,
                scale_group_chunks: 2,
            },
            mean_weight_rel_error: 0.0,
            embed_f32_rows: Vec::new(),
            embed_bf16_rows: Vec::new(),
            embed_nf4,
            layers,
            final_norm,
            lm_head,
            pool: None,
            moe_runtime: None,
        };
        eng.repack_sdot();
        eng.set_default_p_cores();
        eng.enable_q8_kv_cache(true);

        let pinned_bytes = eng.pinned_core_bytes();
        let kv_and_scratch_bytes = eng.kv_cache_bytes() + (48 << 20);
        let moe_rt = SparseMoeRuntime::open_k2tm(
            experts_k2tm_path,
            eng.cfg.hidden_size,
            eng.cfg.eff_moe_inter(),
            eng.cfg.num_experts,
            eng.cfg.num_experts_per_tok,
            eng.cfg.norm_topk_prob,
            eng.cfg.router_scoring,
            0,
            expert_bits,
            2,
            pinned_bytes,
            kv_and_scratch_bytes,
            draft_reserve_bytes,
            policy,
        )?;
        eng.moe_runtime = Some(moe_rt);
        Ok(eng)
    }

    /// Resident bytes occupied by the pinned transformer core (attention, norms, routers, shared experts,
    /// lookahead projectors, embeddings, and `lm_head`), excluding `.k2tm` routed experts.
    pub fn pinned_core_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.qkv_proj.physical_bytes()
                    + l.o_proj.physical_bytes()
                    + l.gate_up_proj.physical_bytes()
                    + l.down_proj.physical_bytes()
                    + (l.input_norm.len() + l.post_attn_norm.len()) * 4
                    + l.qkv_bias.as_ref().map_or(0, |b| b.len() * 4)
                    + l.o_bias.as_ref().map_or(0, |b| b.len() * 4)
                    + l.q_norm.as_ref().map_or(0, |b| b.len() * 4)
                    + l.k_norm.as_ref().map_or(0, |b| b.len() * 4)
                    + l.moe.as_ref().map_or(0, |m| m.physical_bytes())
            })
            .sum();
        layer_bytes
            + self.final_norm.len() * 4
            + self.lm_head.physical_bytes()
            + self.embed_f32_rows.len() * 4
            + self.embed_bf16_rows.len() * 2
            + self.embed_nf4.as_ref().map_or(0, |m| m.physical_bytes())
    }

    /// Resident DRAM bytes occupied by the Sparse-MoE expert arena (`0` on dense models).
    pub fn moe_expert_arena_bytes(&self) -> usize {
        self.moe_runtime
            .as_ref()
            .map_or(0, |r| r.resident_arena_bytes())
    }

    /// Total physical RAM occupied by all compiled linear projections (`layers` + `lm_head` + resident MoE arena).
    pub fn linear_weight_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.qkv_proj.physical_bytes()
                    + l.o_proj.physical_bytes()
                    + l.gate_up_proj.physical_bytes()
                    + l.down_proj.physical_bytes()
                    + l.moe.as_ref().map_or(0, |m| m.physical_bytes())
            })
            .sum();
        layer_bytes + self.lm_head.physical_bytes() + self.moe_expert_arena_bytes()
    }

    /// Linear weights plus biases, untied (`FP32`, `BF16`, or `NF4`) embeddings, and resident MoE SLRU/mmap arena.
    pub fn resident_weight_bytes(&self) -> usize {
        let bias_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.qkv_bias.as_ref().map_or(0, |b| b.len() * std::mem::size_of::<f32>())
                    + l.o_bias.as_ref().map_or(0, |b| b.len() * std::mem::size_of::<f32>())
            })
            .sum();
        self.linear_weight_bytes()
            + bias_bytes
            + self.embed_f32_rows.len() * std::mem::size_of::<f32>()
            + self.embed_bf16_rows.len() * std::mem::size_of::<u16>()
            + self.embed_nf4.as_ref().map_or(0, |m| m.physical_bytes())
    }

    /// Physical bits per weight across all transformer layers + `lm_head` (including full `.k2tm` shard on MoE).
    pub fn physical_bits_per_weight(&self) -> f64 {
        let q_dim = self.cfg.q_dim();
        let qkv_rows = q_dim + 2 * self.cfg.kv_dim();
        let h = self.cfg.hidden_size;
        if !self.is_moe() {
            let layer_params = qkv_rows * h
                + h * q_dim
                + 2 * self.cfg.intermediate_size * h
                + h * self.cfg.intermediate_size;
            let total_params =
                self.cfg.num_hidden_layers * layer_params + self.cfg.vocab_size * h;
            return (self.linear_weight_bytes() as f64 * 8.0) / (total_params as f64);
        }

        let mut total_params = self.cfg.vocab_size * h;
        let mut core_linear_bytes = self.lm_head.physical_bytes();
        if !self.cfg.tie_word_embeddings {
            total_params += self.cfg.vocab_size * h;
            core_linear_bytes += self.embed_f32_rows.len() * 4
                + self.embed_bf16_rows.len() * 2
                + self.embed_nf4.as_ref().map_or(0, |m| m.physical_bytes());
        }
        let moe_inter = self.cfg.eff_moe_inter();
        for l in &self.layers {
            total_params += qkv_rows * h + h * q_dim;
            core_linear_bytes += l.qkv_proj.physical_bytes() + l.o_proj.physical_bytes();
            if let Some(ref moe) = l.moe {
                total_params += moe.router.num_experts * h;
                total_params += self.cfg.num_experts * (3 * moe_inter * h);
                if let Some(ref sh) = moe.shared_expert {
                    total_params += 3 * sh.inter_dim * h;
                }
                core_linear_bytes += moe.physical_bytes();
            } else {
                total_params += 3 * self.cfg.intermediate_size * h;
                core_linear_bytes +=
                    l.gate_up_proj.physical_bytes() + l.down_proj.physical_bytes();
            }
        }
        let expert_total_bytes = self
            .moe_stats()
            .filter(|s| s.shard_disk_bytes > 0)
            .map_or_else(|| self.moe_expert_arena_bytes(), |s| s.shard_disk_bytes);
        ((core_linear_bytes + expert_total_bytes) as f64 * 8.0) / (total_params as f64)
    }
}

/// Compact snapshot of CPU KV-cache positions `0..len` for non-destructive single-pass scoring.
pub enum KvPrefixSnapshot {
    Q8 {
        slots_per_head: usize,
        k_q8: Vec<i8>,
        v_q8: Vec<i8>,
        k_scales: Vec<f32>,
        v_scales: Vec<f32>,
    },
    Q4 {
        slots_per_head: usize,
        k_q4: Vec<u8>,
        v_q4: Vec<u8>,
        k_s16: Vec<u16>,
        v_s16: Vec<u16>,
    },
    Fp32 {
        len: usize,
        k: Vec<f32>,
        v: Vec<f32>,
    },
}

impl RealLlamaEngine {
    /// Saves the first `n` CPU KV-cache positions (`None` on Metal-backed engines).
    pub fn save_kv_prefix(&self, n: usize) -> Option<KvPrefixSnapshot> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.metal.is_some() {
            return None;
        }
        let len = n.min(self.ws.pos);
        if let Some(q) = self.ws.q8_kv.as_ref() {
            let heads = q.num_layers * q.num_kv_heads;
            let hd = q.head_dim;
            let slots_per_head = if q.ring_len > 0 { q.cap_seq_len } else { len };
            if q.bits == 8 {
                let mut k_q8 = vec![0i8; heads * slots_per_head * hd];
                let mut v_q8 = vec![0i8; heads * slots_per_head * hd];
                let mut k_scales = vec![0.0f32; heads * slots_per_head];
                let mut v_scales = vec![0.0f32; heads * slots_per_head];
                if slots_per_head > 0 {
                    for h in 0..heads {
                        let src = h * q.cap_seq_len;
                        let dst = h * slots_per_head;
                        k_q8[dst * hd..(dst + slots_per_head) * hd]
                            .copy_from_slice(&q.k_q8[src * hd..(src + slots_per_head) * hd]);
                        v_q8[dst * hd..(dst + slots_per_head) * hd]
                            .copy_from_slice(&q.v_q8[src * hd..(src + slots_per_head) * hd]);
                        k_scales[dst..dst + slots_per_head]
                            .copy_from_slice(&q.k_scales[src..src + slots_per_head]);
                        v_scales[dst..dst + slots_per_head]
                            .copy_from_slice(&q.v_scales[src..src + slots_per_head]);
                    }
                }
                Some(KvPrefixSnapshot::Q8 {
                    slots_per_head,
                    k_q8,
                    v_q8,
                    k_scales,
                    v_scales,
                })
            } else {
                let half_hd = hd / 2;
                let gph64 = hd / 64;
                let mut k_q4 = vec![0x88u8; heads * slots_per_head * half_hd];
                let mut v_q4 = vec![0x88u8; heads * slots_per_head * half_hd];
                let mut k_s16 = vec![0u16; heads * slots_per_head * gph64];
                let mut v_s16 = vec![0u16; heads * slots_per_head * gph64];
                if slots_per_head > 0 {
                    for h in 0..heads {
                        let src = h * q.cap_seq_len;
                        let dst = h * slots_per_head;
                        k_q4[dst * half_hd..(dst + slots_per_head) * half_hd].copy_from_slice(
                            &q.k_q4[src * half_hd..(src + slots_per_head) * half_hd],
                        );
                        v_q4[dst * half_hd..(dst + slots_per_head) * half_hd].copy_from_slice(
                            &q.v_q4[src * half_hd..(src + slots_per_head) * half_hd],
                        );
                        k_s16[dst * gph64..(dst + slots_per_head) * gph64]
                            .copy_from_slice(&q.k_s16[src * gph64..(src + slots_per_head) * gph64]);
                        v_s16[dst * gph64..(dst + slots_per_head) * gph64]
                            .copy_from_slice(&q.v_s16[src * gph64..(src + slots_per_head) * gph64]);
                    }
                }
                Some(KvPrefixSnapshot::Q4 {
                    slots_per_head,
                    k_q4,
                    v_q4,
                    k_s16,
                    v_s16,
                })
            }
        } else {
            let kvd = self.cfg.kv_dim();
            let layers = self.layers.len();
            let mut k = vec![0.0f32; layers * len * kvd];
            let mut v = vec![0.0f32; layers * len * kvd];
            if len > 0 && !self.ws.k_cache.is_empty() {
                let stride = self.cfg.max_seq_len * kvd;
                for l in 0..layers {
                    let src = l * stride;
                    let dst = l * len * kvd;
                    k[dst..dst + len * kvd].copy_from_slice(&self.ws.k_cache[src..src + len * kvd]);
                    v[dst..dst + len * kvd].copy_from_slice(&self.ws.v_cache[src..src + len * kvd]);
                }
            }
            Some(KvPrefixSnapshot::Fp32 { len, k, v })
        }
    }

    /// Saves a linear (pre-wrap) CPU KV-cache prefix of `n` tokens without allocating full ring slack.
    pub fn save_linear_kv_prefix(&self, n: usize) -> Option<KvPrefixSnapshot> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if self.metal.is_some() {
            return None;
        }
        let len = n.min(self.ws.pos);
        if let Some(q) = self.ws.q8_kv.as_ref() {
            if q.ring_len > 0 && len > q.sink_tokens + q.ring_len {
                return None;
            }
            let heads = q.num_layers * q.num_kv_heads;
            let hd = q.head_dim;
            let slots_per_head = len;
            if q.bits == 8 {
                let mut k_q8 = vec![0i8; heads * slots_per_head * hd];
                let mut v_q8 = vec![0i8; heads * slots_per_head * hd];
                let mut k_scales = vec![0.0f32; heads * slots_per_head];
                let mut v_scales = vec![0.0f32; heads * slots_per_head];
                if slots_per_head > 0 {
                    for h in 0..heads {
                        let src = h * q.cap_seq_len;
                        let dst = h * slots_per_head;
                        k_q8[dst * hd..(dst + slots_per_head) * hd]
                            .copy_from_slice(&q.k_q8[src * hd..(src + slots_per_head) * hd]);
                        v_q8[dst * hd..(dst + slots_per_head) * hd]
                            .copy_from_slice(&q.v_q8[src * hd..(src + slots_per_head) * hd]);
                        k_scales[dst..dst + slots_per_head]
                            .copy_from_slice(&q.k_scales[src..src + slots_per_head]);
                        v_scales[dst..dst + slots_per_head]
                            .copy_from_slice(&q.v_scales[src..src + slots_per_head]);
                    }
                }
                Some(KvPrefixSnapshot::Q8 {
                    slots_per_head,
                    k_q8,
                    v_q8,
                    k_scales,
                    v_scales,
                })
            } else {
                let half_hd = hd / 2;
                let gph64 = hd / 64;
                let mut k_q4 = vec![0x88u8; heads * slots_per_head * half_hd];
                let mut v_q4 = vec![0x88u8; heads * slots_per_head * half_hd];
                let mut k_s16 = vec![0u16; heads * slots_per_head * gph64];
                let mut v_s16 = vec![0u16; heads * slots_per_head * gph64];
                if slots_per_head > 0 {
                    for h in 0..heads {
                        let src = h * q.cap_seq_len;
                        let dst = h * slots_per_head;
                        k_q4[dst * half_hd..(dst + slots_per_head) * half_hd].copy_from_slice(
                            &q.k_q4[src * half_hd..(src + slots_per_head) * half_hd],
                        );
                        v_q4[dst * half_hd..(dst + slots_per_head) * half_hd].copy_from_slice(
                            &q.v_q4[src * half_hd..(src + slots_per_head) * half_hd],
                        );
                        k_s16[dst * gph64..(dst + slots_per_head) * gph64]
                            .copy_from_slice(&q.k_s16[src * gph64..(src + slots_per_head) * gph64]);
                        v_s16[dst * gph64..(dst + slots_per_head) * gph64]
                            .copy_from_slice(&q.v_s16[src * gph64..(src + slots_per_head) * gph64]);
                    }
                }
                Some(KvPrefixSnapshot::Q4 {
                    slots_per_head,
                    k_q4,
                    v_q4,
                    k_s16,
                    v_s16,
                })
            }
        } else {
            self.save_kv_prefix(len)
        }
    }

    /// Restores a previously saved CPU KV-cache prefix and sets `pos = total_pos`.
    pub fn restore_kv_prefix(&mut self, snap: &KvPrefixSnapshot, total_pos: usize) {
        self.restore_kv_prefix_slice(snap, usize::MAX, total_pos);
    }

    /// Restores up to `copy_slots` leading tokens from `snap` and sets `pos = total_pos`.
    pub fn restore_kv_prefix_slice(
        &mut self,
        snap: &KvPrefixSnapshot,
        copy_slots: usize,
        total_pos: usize,
    ) {
        match snap {
            KvPrefixSnapshot::Q8 {
                slots_per_head,
                k_q8,
                v_q8,
                k_scales,
                v_scales,
            } => {
                if let Some(q) = self.ws.q8_kv.as_mut() {
                    let sp = *slots_per_head;
                    let cp = sp.min(copy_slots);
                    if cp > 0 {
                        q.ensure_seq_capacity(total_pos.max(cp.min(q.max_seq_len)), total_pos);
                        let heads = q.num_layers * q.num_kv_heads;
                        let hd = q.head_dim;
                        for h in 0..heads {
                            let src = h * sp;
                            let dst = h * q.cap_seq_len;
                            q.k_q8[dst * hd..(dst + cp) * hd]
                                .copy_from_slice(&k_q8[src * hd..(src + cp) * hd]);
                            q.v_q8[dst * hd..(dst + cp) * hd]
                                .copy_from_slice(&v_q8[src * hd..(src + cp) * hd]);
                            q.k_scales[dst..dst + cp].copy_from_slice(&k_scales[src..src + cp]);
                            v_scales[src..src + cp]
                                .iter()
                                .enumerate()
                                .for_each(|(i, &val)| q.v_scales[dst + i] = val);
                        }
                    }
                }
            }
            KvPrefixSnapshot::Q4 {
                slots_per_head,
                k_q4,
                v_q4,
                k_s16,
                v_s16,
            } => {
                if let Some(q) = self.ws.q8_kv.as_mut() {
                    let sp = *slots_per_head;
                    let cp = sp.min(copy_slots);
                    if cp > 0 {
                        q.ensure_seq_capacity(total_pos.max(cp.min(q.max_seq_len)), total_pos);
                        let heads = q.num_layers * q.num_kv_heads;
                        let half_hd = q.head_dim / 2;
                        let gph64 = q.head_dim / 64;
                        for h in 0..heads {
                            let src = h * sp;
                            let dst = h * q.cap_seq_len;
                            q.k_q4[dst * half_hd..(dst + cp) * half_hd]
                                .copy_from_slice(&k_q4[src * half_hd..(src + cp) * half_hd]);
                            q.v_q4[dst * half_hd..(dst + cp) * half_hd]
                                .copy_from_slice(&v_q4[src * half_hd..(src + cp) * half_hd]);
                            q.k_s16[dst * gph64..(dst + cp) * gph64]
                                .copy_from_slice(&k_s16[src * gph64..(src + cp) * gph64]);
                            q.v_s16[dst * gph64..(dst + cp) * gph64]
                                .copy_from_slice(&v_s16[src * gph64..(src + cp) * gph64]);
                        }
                    }
                }
            }
            KvPrefixSnapshot::Fp32 { len, k, v } => {
                let len = *len;
                let cp = len.min(copy_slots);
                if cp > 0 {
                    self.ws.ensure_fp32_kv(&self.cfg);
                    let kvd = self.cfg.kv_dim();
                    let stride = self.cfg.max_seq_len * kvd;
                    for l in 0..self.layers.len() {
                        let src = l * len * kvd;
                        let dst = l * stride;
                        self.ws.k_cache[dst..dst + cp * kvd]
                            .copy_from_slice(&k[src..src + cp * kvd]);
                        self.ws.v_cache[dst..dst + cp * kvd]
                            .copy_from_slice(&v[src..src + cp * kvd]);
                    }
                }
            }
        }
        self.ws.pos = total_pos.min(self.cfg.max_seq_len);
    }
}

impl RealLlamaEngine {
    /// Drops KV-cache entries after the first `n` tokens.
    pub fn truncate_context(&mut self, n: usize) {
        self.ws.pos = self.ws.pos.min(n);
    }

    /// Embedding row of `token` (decoded from `BF16`/`FP32`/`NF4` untied table, or quantized `lm_head` when tied).
    pub fn embed(&self, token: usize, out: &mut [f32]) {
        embed_token_into(
            &self.embed_f32_rows,
            &self.embed_bf16_rows,
            self.embed_nf4.as_ref(),
            &self.lm_head,
            self.cfg.hidden_size,
            token,
            out,
        );
    }

    /// Replaces the KV cache with positions `0..n` from `keys`/`values` laid out
    /// `[layers, rows, kv_dim]` (`rows >= n`), e.g. from an external prefill.
    pub fn load_prefix_kv(&mut self, keys: &[f32], values: &[f32], rows: usize, n: usize) {
        let kvd = self.cfg.kv_dim();
        let hd = self.cfg.head_dim();
        assert!(n <= rows && n <= self.cfg.max_seq_len);
        assert_eq!(keys.len(), self.layers.len() * rows * kvd);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = self.metal.as_mut() {
            gpu.load_prefix_kv(keys, values, rows, n)
                .expect("Metal load_prefix_kv failed");
            self.ws.pos = n;
            return;
        }
        if let Some(q8) = self.ws.q8_kv.as_mut() {
            q8.ensure_seq_capacity(n, 0);
            let nkvh = self.cfg.num_key_value_heads;
            for l in 0..self.layers.len() {
                let src = l * rows * kvd;
                for pos in 0..n {
                    for kvh in 0..nkvh {
                        let off = src + pos * kvd + kvh * hd;
                        q8.store_k_head(l, kvh, pos, &keys[off..off + hd]);
                        q8.store_v_head(l, kvh, pos, &values[off..off + hd]);
                    }
                }
            }
        } else {
            self.ws.ensure_fp32_kv(&self.cfg);
            let stride = self.cfg.max_seq_len * kvd;
            for l in 0..self.layers.len() {
                let src = l * rows * kvd;
                self.ws.k_cache[l * stride..l * stride + n * kvd].copy_from_slice(&keys[src..src + n * kvd]);
                self.ws.v_cache[l * stride..l * stride + n * kvd].copy_from_slice(&values[src..src + n * kvd]);
            }
        }
        self.ws.pos = n;
    }

    /// Logits from a last-layer hidden state (final RMSNorm + LM head).
    pub fn logits_from_hidden(&mut self, hidden: &[f32]) -> &[f32] {
        rms_norm(hidden, &self.final_norm, self.cfg.rms_norm_eps, &mut self.ws.norm_buf);
        fwht64_signed_inplace(&mut self.ws.norm_buf);
        if matches!(self.lm_head, LinearBackend::Vq2(_)) {
            self.ws.vqlut_hidden.populate(&self.ws.norm_buf);
        } else {
            self.ws.blut_hidden.populate(&self.ws.norm_buf);
        }
        self.lm_head.forward_exec(
            &self.ws.norm_buf,
            None,
            Some(&self.ws.blut_hidden),
            Some(&self.ws.vqlut_hidden),
            &self.ws.q8_hidden_vals,
            &self.ws.q8_hidden_scales,
            self.pool.as_ref(),
            &mut self.ws.logits,
        );
        &self.ws.logits
    }

    /// Tokens currently in the KV cache.
    pub fn position(&self) -> usize {
        self.ws.pos
    }

    /// Resets KV cache position to 0.
    pub fn reset_context(&mut self) {
        self.ws.reset();
    }

    /// Whether every projection supports batched `SDOT` and the INT8 activation scratch is set up.
    #[inline]
    pub fn is_all_sdot(&self) -> bool {
        let is_sdot = |b: &LinearBackend| {
            matches!(
                b,
                LinearBackend::Nibble4(_) | LinearBackend::NfSdot(_) | LinearBackend::Vq2(_)
            )
        };
        let has_dense_ffn = self.layers.iter().any(|l| l.moe.is_none());
        let moe_is_sdot = self
            .moe_runtime
            .as_ref()
            .is_none_or(|r| !r.is_fp32_reference());
        self.ws.blut_hidden.int8.is_some()
            && self.ws.blut_q.int8.is_some()
            && (!has_dense_ffn || self.ws.blut_inter.int8.is_some())
            && moe_is_sdot
            && is_sdot(&self.lm_head)
            && self.layers.iter().all(|l| {
                is_sdot(&l.qkv_proj)
                    && is_sdot(&l.o_proj)
                    && l.o_inv_scale.is_none()
                    && l.down_inv_scale.is_none()
                    && (l.moe.is_some()
                        || (is_sdot(&l.gate_up_proj) && is_sdot(&l.down_proj)))
            })
    }

    #[inline]
    fn can_run_sdot_megakernel(&self) -> bool {
        if self.is_moe()
            || self.cfg.q_dim() != self.cfg.hidden_size
            || self
                .layers
                .iter()
                .any(|l| l.q_norm.is_some() || l.k_norm.is_some() || l.o_bias.is_some())
        {
            return false;
        }
        let is_fast = |b: &LinearBackend| {
            matches!(
                b,
                LinearBackend::Nibble4(_) | LinearBackend::NfSdot(_) | LinearBackend::Vq2(_)
            )
        };
        let needs_int8 =
            |b: &LinearBackend| matches!(b, LinearBackend::Nibble4(_) | LinearBackend::NfSdot(_));
        let any_int8 = needs_int8(&self.lm_head)
            || self.layers.iter().any(|l| {
                needs_int8(&l.qkv_proj)
                    || needs_int8(&l.o_proj)
                    || needs_int8(&l.gate_up_proj)
                    || needs_int8(&l.down_proj)
            });
        let hd = self.cfg.head_dim();
        (hd == 64 || hd == 128)
            && self.cfg.hidden_size <= 8192
            && self.cfg.hidden_size.is_multiple_of(64)
            && self.cfg.intermediate_size.is_multiple_of(64)
            && (!any_int8 || (self.ws.blut_hidden.int8.is_some() && self.ws.blut_inter.int8.is_some()))
            && is_fast(&self.lm_head)
            && self.layers.iter().all(|l| {
                is_fast(&l.qkv_proj)
                    && is_fast(&l.o_proj)
                    && is_fast(&l.gate_up_proj)
                    && is_fast(&l.down_proj)
                    && l.o_inv_scale.is_none()
                    && l.down_inv_scale.is_none()
            })
    }

    /// Executes one full causal step for `token_id` and returns a slice of `[vocab_size]` logits.
    pub fn forward_token(&mut self, token_id: usize) -> &[f32] {
        assert!(
            self.ws.pos < self.cfg.max_seq_len,
            "sequence length {} reached max_seq_len {}",
            self.ws.pos,
            self.cfg.max_seq_len
        );
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = self.metal.as_mut() {
            let h_dim = self.cfg.hidden_size;
            let ef32 = &self.embed_f32_rows;
            let ebf16 = &self.embed_bf16_rows;
            let enf4 = self.embed_nf4.as_ref();
            let lm_head = &self.lm_head;
            let tok = [token_id];
            gpu.forward_tokens(
                &self.cfg,
                &tok,
                self.ws.pos,
                |t, dst| embed_token_into(ef32, ebf16, enf4, lm_head, h_dim, t, dst),
                None,
                &mut self.ws.logits,
            )
            .expect("Metal forward_token failed");
            self.ws.pos += 1;
            return &self.ws.logits;
        }
        if self.can_run_sdot_megakernel() {
            return self.forward_token_sdot_megakernel(token_id);
        }
        if let Some(q8) = self.ws.q8_kv.as_mut() {
            q8.ensure_seq_capacity(self.ws.pos + 1, self.ws.pos);
        } else {
            self.ws.ensure_fp32_kv(&self.cfg);
        }
        let pos = self.ws.pos;
        let h_dim = self.cfg.hidden_size;
        let q_dim = self.cfg.q_dim();
        let inter_dim = self.cfg.intermediate_size;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_dim = self.cfg.kv_dim();
        let num_q_heads = self.cfg.num_attention_heads;
        let num_kv_heads = self.cfg.num_key_value_heads;
        let kv_group = num_q_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let layer_kv_stride = self.cfg.max_seq_len * kv_dim;
        let eps = self.cfg.rms_norm_eps;

        embed_token_into(
            &self.embed_f32_rows,
            &self.embed_bf16_rows,
            self.embed_nf4.as_ref(),
            &self.lm_head,
            h_dim,
            token_id,
            &mut self.ws.hidden,
        );
        let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
        let layers = &self.layers;
        let final_norm = &self.final_norm;
        let lm_head = &self.lm_head;
        let pool_ref = self.pool.as_ref();
        let moe_rt = &mut self.moe_runtime;
        let ws = &mut self.ws;

        macro_rules! exec_hidden_proj {
            ($proj:expr, $buf:expr, $out:expr) => {{
                match &$proj {
                    LinearBackend::Vq2(_) => {
                        fwht64_signed_inplace($buf);
                        ws.vqlut_hidden.populate($buf);
                    }
                    LinearBackend::FusedBipolar(_)
                    | LinearBackend::Nibble4(_)
                    | LinearBackend::NfSdot(_) => {
                        fwht64_signed_inplace($buf);
                        ws.blut_hidden.populate($buf);
                    }
                    LinearBackend::FusedBase3(_) => {
                        fwht64_signed_inplace($buf);
                        ws.qlut_hidden.populate($buf);
                    }
                    LinearBackend::LutMultiPlane(_) => {
                        ws.qlut_hidden.populate($buf);
                    }
                    LinearBackend::GgufQ4_0(_) => {
                        GgufQ4_0Matrix::quantize_q8_0_activation(
                            $buf,
                            &mut ws.q8_hidden_vals,
                            &mut ws.q8_hidden_scales,
                        );
                    }
                    LinearBackend::Fp32 { .. } | LinearBackend::Empty => {}
                }
                $proj.forward_exec(
                    $buf,
                    Some(&ws.qlut_hidden),
                    Some(&ws.blut_hidden),
                    Some(&ws.vqlut_hidden),
                    &ws.q8_hidden_vals,
                    &ws.q8_hidden_scales,
                    pool_ref,
                    $out,
                );
            }};
        }

        macro_rules! exec_q_proj {
            ($proj:expr, $buf:expr, $out:expr) => {{
                match &$proj {
                    LinearBackend::Vq2(_) => {
                        fwht64_signed_inplace($buf);
                        ws.vqlut_q.populate($buf);
                    }
                    LinearBackend::FusedBipolar(_)
                    | LinearBackend::Nibble4(_)
                    | LinearBackend::NfSdot(_) => {
                        fwht64_signed_inplace($buf);
                        ws.blut_q.populate($buf);
                    }
                    LinearBackend::FusedBase3(_) => {
                        fwht64_signed_inplace($buf);
                        ws.qlut_q.populate($buf);
                    }
                    LinearBackend::LutMultiPlane(_) => {
                        ws.qlut_q.populate($buf);
                    }
                    LinearBackend::GgufQ4_0(_) => {
                        GgufQ4_0Matrix::quantize_q8_0_activation(
                            $buf,
                            &mut ws.q8_q_vals,
                            &mut ws.q8_q_scales,
                        );
                    }
                    LinearBackend::Fp32 { .. } | LinearBackend::Empty => {}
                }
                $proj.forward_exec(
                    $buf,
                    Some(&ws.qlut_q),
                    Some(&ws.blut_q),
                    Some(&ws.vqlut_q),
                    &ws.q8_q_vals,
                    &ws.q8_q_scales,
                    pool_ref,
                    $out,
                );
            }};
        }

        for (l_idx, layer) in layers.iter().enumerate() {
            // 1. Pre-Attention RMSNorm + Fused QKV Projection (+ optional QKV Bias)
            rms_norm(&ws.hidden, &layer.input_norm, eps, &mut ws.norm_buf);
            exec_hidden_proj!(layer.qkv_proj, &mut ws.norm_buf, &mut ws.qkv_buf);
            if let Some(bias) = &layer.qkv_bias {
                add_inplace(&mut ws.qkv_buf, bias);
            }

            // 2. Optional QK-Norm + Precomputed RoPE + KV Cache Store + Causal GQA Attention
            let (q_slice, kv_rest) = ws.qkv_buf.split_at_mut(q_dim);
            let (k_slice, v_slice) = kv_rest.split_at_mut(kv_dim);
            if let Some(ref qn) = layer.q_norm {
                apply_per_head_rms_norm(q_slice, head_dim, qn, eps);
            }
            if let Some(ref kn) = layer.k_norm {
                apply_per_head_rms_norm(k_slice, head_dim, kn, eps);
            }
            apply_llama_rope_precomp(q_slice, num_q_heads, head_dim, half_dim, rope_row);
            apply_llama_rope_precomp(k_slice, num_kv_heads, head_dim, half_dim, rope_row);

            if let Some(q8) = ws.q8_kv.as_mut() {
                for kvh in 0..num_kv_heads {
                    q8.store_k_head(
                        l_idx,
                        kvh,
                        pos,
                        &k_slice[kvh * head_dim..(kvh + 1) * head_dim],
                    );
                    q8.store_v_head(
                        l_idx,
                        kvh,
                        pos,
                        &v_slice[kvh * head_dim..(kvh + 1) * head_dim],
                    );
                }
                causal_gqa_attention_q8(
                    q_slice,
                    q8,
                    l_idx,
                    &mut ws.scores,
                    &mut ws.attn_out,
                    num_q_heads,
                    kv_group,
                    pos,
                    attn_scale,
                );
            } else {
                let kv_base = l_idx * layer_kv_stride;
                let step_off = kv_base + pos * kv_dim;
                ws.k_cache[step_off..step_off + kv_dim].copy_from_slice(k_slice);
                ws.v_cache[step_off..step_off + kv_dim].copy_from_slice(v_slice);

                causal_gqa_attention(
                    q_slice,
                    &ws.k_cache[kv_base..],
                    &ws.v_cache[kv_base..],
                    &mut ws.scores,
                    &mut ws.attn_out,
                    num_q_heads,
                    kv_group,
                    head_dim,
                    kv_dim,
                    pos,
                    attn_scale,
                );
            }

            // 3. Output Projection (+ optional `o_bias`) + Residual
            if let Some(inv_s) = &layer.o_inv_scale {
                for (a, &is) in ws.attn_out.iter_mut().zip(inv_s.iter()) {
                    *a *= is;
                }
            }
            exec_q_proj!(layer.o_proj, &mut ws.attn_out, &mut ws.proj_out);
            if let Some(ref ob) = layer.o_bias {
                add_inplace(&mut ws.proj_out, ob);
            }
            add_inplace(&mut ws.hidden, &ws.proj_out);

            // 4. Sparse-MoE (with 3-layer-ahead async NVMe prefetch) or Dense SwiGLU FFN
            if let Some(ref moe_core) = layer.moe {
                if let (Some((future_l, ref projector)), Some(rt)) =
                    (&moe_core.lookahead, moe_rt.as_mut())
                {
                    if let Some(future_layer) = layers.get(*future_l) {
                        if let Some(ref future_moe) = future_layer.moe {
                            rt.dispatch_lookahead_prefetch(
                                &ws.hidden,
                                projector,
                                &future_layer.post_attn_norm,
                                future_moe.moe_slot_idx,
                                &future_moe.router,
                                eps,
                            );
                        }
                    }
                }
                rms_norm(&ws.hidden, &layer.post_attn_norm, eps, &mut ws.norm_buf);
                moe_rt
                    .as_mut()
                    .expect("SparseMoeRuntime missing on MoE layer")
                    .forward_layer_single(moe_core, &ws.norm_buf, pool_ref, &mut ws.proj_out);
                add_inplace(&mut ws.hidden, &ws.proj_out);
            } else {
                rms_norm(&ws.hidden, &layer.post_attn_norm, eps, &mut ws.norm_buf);
                exec_hidden_proj!(
                    layer.gate_up_proj,
                    &mut ws.norm_buf,
                    &mut ws.gate_up_buf
                );

                if layer.gate_up_interleaved {
                    for ic in 0..(inter_dim / 64) {
                        swiglu_activate(
                            &ws.gate_up_buf[ic * 128..ic * 128 + 64],
                            &ws.gate_up_buf[ic * 128 + 64..ic * 128 + 128],
                            &mut ws.swiglu_buf[ic * 64..(ic + 1) * 64],
                        );
                    }
                } else {
                    let (gate_s, up_s) = ws.gate_up_buf.split_at(inter_dim);
                    swiglu_activate(gate_s, up_s, &mut ws.swiglu_buf);
                }

                if let Some(inv_s) = &layer.down_inv_scale {
                    for (g, &is) in ws.swiglu_buf.iter_mut().zip(inv_s.iter()) {
                        *g *= is;
                    }
                }
                match &layer.down_proj {
                    LinearBackend::Vq2(_) => {
                        fwht64_signed_inplace(&mut ws.swiglu_buf);
                        ws.vqlut_inter.populate(&ws.swiglu_buf);
                    }
                    LinearBackend::FusedBipolar(_)
                    | LinearBackend::Nibble4(_)
                    | LinearBackend::NfSdot(_) => {
                        fwht64_signed_inplace(&mut ws.swiglu_buf);
                        ws.blut_inter.populate(&ws.swiglu_buf);
                    }
                    LinearBackend::FusedBase3(_) => {
                        fwht64_signed_inplace(&mut ws.swiglu_buf);
                        ws.qlut_inter.populate(&ws.swiglu_buf);
                    }
                    LinearBackend::LutMultiPlane(_) => {
                        ws.qlut_inter.populate(&ws.swiglu_buf);
                    }
                    LinearBackend::GgufQ4_0(_) => {
                        GgufQ4_0Matrix::quantize_q8_0_activation(
                            &ws.swiglu_buf,
                            &mut ws.q8_inter_vals,
                            &mut ws.q8_inter_scales,
                        );
                    }
                    LinearBackend::Fp32 { .. } | LinearBackend::Empty => {}
                }
                layer.down_proj.forward_exec(
                    &ws.swiglu_buf,
                    Some(&ws.qlut_inter),
                    Some(&ws.blut_inter),
                    Some(&ws.vqlut_inter),
                    &ws.q8_inter_vals,
                    &ws.q8_inter_scales,
                    pool_ref,
                    &mut ws.proj_out,
                );
                add_inplace(&mut ws.hidden, &ws.proj_out);
            }
        }

        // Final RMSNorm + LM Head
        rms_norm(&ws.hidden, final_norm, eps, &mut ws.norm_buf);
        exec_hidden_proj!(*lm_head, &mut ws.norm_buf, &mut ws.logits);

        let _ = SIMD_ROW_STRIPE;
        ws.pos += 1;
        &ws.logits
    }

    /// Decodes one token for `SDOT`/VQ2 engines (`head_dim == 64` or `128`, `hidden_size <= 8192`)
    /// with all Transformer layers and the LM head fused inside a single thread-pool dispatch.
    fn forward_token_sdot_megakernel(&mut self, token_id: usize) -> &[f32] {
        use std::mem::MaybeUninit;

        if let Some(q8) = self.ws.q8_kv.as_mut() {
            q8.ensure_seq_capacity(self.ws.pos + 1, self.ws.pos);
        } else {
            self.ws.ensure_fp32_kv(&self.cfg);
        }
        let pos = self.ws.pos;
        let seq_len = pos + 1;
        let h_dim = self.cfg.hidden_size;
        let inter_dim = self.cfg.intermediate_size;
        let head_dim = self.cfg.head_dim();
        let stripes_per_head = head_dim / SIMD_ROW_STRIPE;
        let chunks_per_head = head_dim / 64;
        let nc_h = h_dim / 64;
        let nc_i = inter_dim / 64;
        let num_q_heads = self.cfg.num_attention_heads;
        let num_kv_heads = self.cfg.num_key_value_heads;
        let kv_dim = self.cfg.kv_dim();
        let kv_group = num_q_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let max_seq_len = self.cfg.max_seq_len;
        let layer_kv_stride = max_seq_len * kv_dim;
        let eps = self.cfg.rms_norm_eps;

        embed_token_into(
            &self.embed_f32_rows,
            &self.embed_bf16_rows,
            self.embed_nf4.as_ref(),
            &self.lm_head,
            h_dim,
            token_id,
            &mut self.ws.hidden,
        );

        let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
        let layers = &self.layers;
        let final_norm = &self.final_norm;
        let lm_head = &self.lm_head;

        let hidden_addr = self.ws.hidden.as_mut_ptr() as usize;
        let qkv_addr = self.ws.qkv_buf.as_mut_ptr() as usize;
        let proj_addr = self.ws.proj_out.as_mut_ptr() as usize;
        let gu_addr = self.ws.gate_up_buf.as_mut_ptr() as usize;
        let scores_addr = self.ws.scores.as_mut_ptr() as usize;
        let k_cache_addr = self.ws.k_cache.as_mut_ptr() as usize;
        let v_cache_addr = self.ws.v_cache.as_mut_ptr() as usize;
        let logits_addr = self.ws.logits.as_mut_ptr() as usize;

        let qkv_raw = self.ws.q8_kv.as_ref().map(|q| q.raw_ptrs());

        let (act_h_addr, act_hq_addr, act_hs_addr, act_hsum_addr) =
            if let Some(a) = self.ws.blut_hidden.int8.as_mut() {
                (
                    (a as *const Int8ChunkActivation) as usize,
                    a.q.as_mut_ptr() as usize,
                    a.scales.as_mut_ptr() as usize,
                    a.sums.as_mut_ptr() as usize,
                )
            } else {
                (0, 0, 0, 0)
            };
        let (act_i_addr, act_iq_addr, act_is_addr, act_isum_addr) =
            if let Some(a) = self.ws.blut_inter.int8.as_mut() {
                (
                    (a as *const Int8ChunkActivation) as usize,
                    a.q.as_mut_ptr() as usize,
                    a.scales.as_mut_ptr() as usize,
                    a.sums.as_mut_ptr() as usize,
                )
            } else {
                (0, 0, 0, 0)
            };

        let vlut_h_addr = self.ws.vqlut_hidden.lut.as_mut_ptr() as usize;
        let vscales_h_addr = self.ws.vqlut_hidden.chunk_scales.as_mut_ptr() as usize;
        let vlut_i_addr = self.ws.vqlut_inter.lut.as_mut_ptr() as usize;
        let vscales_i_addr = self.ws.vqlut_inter.chunk_scales.as_mut_ptr() as usize;

        let kernel = |_tid: usize, nt: usize, barrier: &SpinBarrier| {
            let hidden_ptr = hidden_addr as *mut f32;
            let qkv_ptr = qkv_addr as *mut f32;
            let proj_ptr = proj_addr as *mut f32;
            let gu_ptr = gu_addr as *mut f32;
            let scores_ptr = scores_addr as *mut f32;
            let k_cache_ptr = k_cache_addr as *mut f32;
            let v_cache_ptr = v_cache_addr as *mut f32;
            let logits_ptr = logits_addr as *mut f32;
            let act_hq = act_hq_addr as *mut i8;
            let act_hs = act_hs_addr as *mut f32;
            let act_hsum = act_hsum_addr as *mut i32;
            let act_iq = act_iq_addr as *mut i8;
            let act_is = act_is_addr as *mut f32;
            let act_isum = act_isum_addr as *mut i32;
            let vlut_h_ptr = vlut_h_addr as *mut [i8; 16];
            let vscales_h_ptr = vscales_h_addr as *mut f32;
            let vlut_i_ptr = vlut_i_addr as *mut [i8; 16];
            let vscales_i_ptr = vscales_i_addr as *mut f32;

            // Worker-private stack scratch (`hidden_size <= 8192`, left uninitialized).
            let mut lq = MaybeUninit::<[i8; 8192]>::uninit();
            let mut ls = MaybeUninit::<[f32; 128]>::uninit();
            let mut lsum = MaybeUninit::<[i32; 128]>::uninit();
            let mut vlq = MaybeUninit::<[[i8; 16]; 4096]>::uninit();
            let mut vls = MaybeUninit::<[f32; 128]>::uninit();
            let mut lis = MaybeUninit::<[f32; 256]>::uninit();
            let mut lisum = MaybeUninit::<[i32; 256]>::uninit();
            let lq_ptr = lq.as_mut_ptr().cast::<i8>();
            let ls_ptr = ls.as_mut_ptr().cast::<f32>();
            let lsum_ptr = lsum.as_mut_ptr().cast::<i32>();
            let vlq_ptr = vlq.as_mut_ptr().cast::<[i8; 16]>();
            let vls_ptr = vls.as_mut_ptr().cast::<f32>();

            let qh_grain = num_q_heads.div_ceil(nt * 2).clamp(1, kv_group.clamp(1, 8));
            let ic_grain = (nc_i / (nt * 4)).clamp(1, 2);

            for (l_idx, layer) in layers.iter().enumerate() {
                let kv_base = l_idx * layer_kv_stride;
                let qkv_rows = match &layer.qkv_proj {
                    LinearBackend::Nibble4(m) => m.rows,
                    LinearBackend::NfSdot(m) => m.rows,
                    LinearBackend::Vq2(m) => m.rows,
                    _ => unreachable!(),
                };

                // 1. Input norm + FWHT, then `qkv_proj` (+ optional bias) + RoPE + KV store per head.
                let qkv_heads = qkv_rows / head_dim;
                let qkv_grain = (qkv_heads / (nt * 4)).clamp(1, 2);
                let mut populated = false;
                while let Some((b0, b1)) = barrier.claim_work(qkv_grain, qkv_heads) {
                    // SAFETY: `hidden` is read-only here; `[b0*stripes_per_head, b1*stripes_per_head)` are disjoint per worker.
                    unsafe {
                        if !populated {
                            let hidden_ro = std::slice::from_raw_parts(hidden_ptr, h_dim);
                            let inv_rms = 1.0 / (dot_f32_fast(hidden_ro, hidden_ro) / (h_dim as f32) + eps).sqrt();
                            if matches!(layer.qkv_proj, LinearBackend::Vq2(_)) {
                                for c in 0..nc_h {
                                    Vq2ActivationLut::populate_chunk_fwht_raw(
                                        vlq_ptr,
                                        vls_ptr,
                                        c,
                                        &hidden_ro[c * 64..(c + 1) * 64],
                                        Some((&layer.input_norm[c * 64..(c + 1) * 64], inv_rms)),
                                    );
                                }
                            } else {
                                for c in 0..nc_h {
                                    Int8ChunkActivation::populate_chunk_fwht_raw(
                                        lq_ptr,
                                        ls_ptr,
                                        lsum_ptr,
                                        c,
                                        &hidden_ro[c * 64..(c + 1) * 64],
                                        Some((&layer.input_norm[c * 64..(c + 1) * 64], inv_rms)),
                                    );
                                }
                            }
                            populated = true;
                        }
                        let s0 = b0 * stripes_per_head;
                        let s1 = b1 * stripes_per_head;
                        match &layer.qkv_proj {
                            LinearBackend::Nibble4(qkv_mat) => {
                                nibble4_sdot_stripe_range_raw(
                                    qkv_mat,
                                    lq_ptr,
                                    ls_ptr,
                                    lsum_ptr,
                                    s0,
                                    s1,
                                    qkv_ptr,
                                );
                            }
                            LinearBackend::NfSdot(qkv_mat) => {
                                nf_sdot_stripe_range_raw(
                                    qkv_mat,
                                    lq_ptr,
                                    ls_ptr,
                                    s0,
                                    s1,
                                    qkv_ptr,
                                );
                            }
                            LinearBackend::Vq2(qkv_mat) => {
                                vq2_lut_stripe_range_raw(
                                    qkv_mat,
                                    vlq_ptr,
                                    vls_ptr,
                                    s0,
                                    s1,
                                    qkv_ptr,
                                );
                            }
                            _ => unreachable!(),
                        }
                        for b in b0..b1 {
                            let head_s = std::slice::from_raw_parts_mut(qkv_ptr.add(b * head_dim), head_dim);
                            if let Some(bias) = &layer.qkv_bias {
                                add_inplace(head_s, &bias[b * head_dim..(b + 1) * head_dim]);
                            }
                            if b < num_q_heads {
                                apply_llama_rope_head(head_s, rope_row);
                            } else if b < num_q_heads + num_kv_heads {
                                let kvh = b - num_q_heads;
                                apply_llama_rope_head(head_s, rope_row);
                                if let Some(raw) = qkv_raw {
                                    raw.store_k(l_idx, kvh, pos, head_s);
                                } else {
                                    std::ptr::copy_nonoverlapping(
                                        head_s.as_ptr(),
                                        k_cache_ptr.add(kv_base + pos * kv_dim + kvh * head_dim),
                                        head_dim,
                                    );
                                }
                            } else {
                                let kvh = b - num_q_heads - num_kv_heads;
                                if let Some(raw) = qkv_raw {
                                    raw.store_v(l_idx, kvh, pos, head_s);
                                } else {
                                    std::ptr::copy_nonoverlapping(
                                        head_s.as_ptr(),
                                        v_cache_ptr.add(kv_base + pos * kv_dim + kvh * head_dim),
                                        head_dim,
                                    );
                                }
                            }
                        }
                    }
                }
                barrier.wait();

                // 2. Per-head attention, quantized in-register for `o_proj`.
                let o_is_vq2 = matches!(layer.o_proj, LinearBackend::Vq2(_));
                while let Some((qh0, qh1)) = barrier.claim_work(qh_grain, num_q_heads) {
                    // SAFETY: inputs are read-only; head `qh` owns its `scores` slice and Hadamard chunks `[qh*cph, (qh+1)*cph)`.
                    unsafe {
                        if let Some(raw) = qkv_raw {
                            let mut out_group = [[0.0f32; 128]; 8];
                            let sc_len = active_kv_len(seq_len, raw.window_size);
                            let mut s0 = qh0;
                            while s0 < qh1 {
                                let kvh = s0 / kv_group;
                                let s1 = ((kvh + 1) * kv_group).min(qh1).min(s0 + 8);
                                let m = s1 - s0;
                                let mut out_iter = out_group[..m].iter_mut();
                                let mut queries: [GqaQueryRef<'_>; 8] = std::array::from_fn(|j| {
                                    if j < m {
                                        let qh = s0 + j;
                                        let ob = out_iter.next().unwrap();
                                        GqaQueryRef {
                                            q_head: std::slice::from_raw_parts(qkv_ptr.add(qh * head_dim), head_dim),
                                            seq_len,
                                            sc: std::slice::from_raw_parts_mut(scores_ptr.add(qh * sc_len), sc_len),
                                            out: &mut ob[..head_dim],
                                        }
                                    } else {
                                        GqaQueryRef {
                                            q_head: &[],
                                            seq_len: 0,
                                            sc: &mut [],
                                            out: &mut [],
                                        }
                                    }
                                });
                                raw.eval_gqa_group(l_idx, kvh, attn_scale, &mut queries[..m]);
                                for (j, og) in out_group[..m].iter().enumerate() {
                                    let qh = s0 + j;
                                    for sub in 0..chunks_per_head {
                                        let c_idx = qh * chunks_per_head + sub;
                                        let sub_slice = &og[sub * 64..(sub + 1) * 64];
                                        if o_is_vq2 {
                                            Vq2ActivationLut::populate_chunk_fwht_raw(
                                                vlut_h_ptr,
                                                vls_ptr,
                                                c_idx,
                                                sub_slice,
                                                None,
                                            );
                                        } else {
                                            Int8ChunkActivation::populate_chunk_fwht_raw(
                                                act_hq,
                                                ls_ptr,
                                                lsum_ptr,
                                                c_idx,
                                                sub_slice,
                                                None,
                                            );
                                        }
                                    }
                                }
                                s0 = s1;
                            }
                        } else {
                            let mut out_head = [0.0f32; 128];
                            let k_layer = std::slice::from_raw_parts(k_cache_ptr.add(kv_base), seq_len * kv_dim);
                            let v_layer = std::slice::from_raw_parts(v_cache_ptr.add(kv_base), seq_len * kv_dim);
                            for qh in qh0..qh1 {
                                let q_head = std::slice::from_raw_parts(qkv_ptr.add(qh * head_dim), head_dim);
                                let sc = std::slice::from_raw_parts_mut(scores_ptr.add(qh * max_seq_len), seq_len);
                                let kvh = qh / kv_group;
                                if head_dim == 64 {
                                    causal_gqa_head64(
                                        q_head,
                                        k_layer,
                                        v_layer,
                                        sc,
                                        (&mut out_head[..64]).try_into().unwrap(),
                                        kv_dim,
                                        kvh * 64,
                                        attn_scale,
                                    );
                                } else {
                                    causal_gqa_head_fp32(
                                        q_head,
                                        k_layer,
                                        v_layer,
                                        sc,
                                        &mut out_head[..head_dim],
                                        kv_dim,
                                        kvh * head_dim,
                                        attn_scale,
                                    );
                                }
                                for sub in 0..chunks_per_head {
                                    let c_idx = qh * chunks_per_head + sub;
                                    let sub_slice = &out_head[sub * 64..(sub + 1) * 64];
                                    if o_is_vq2 {
                                        Vq2ActivationLut::populate_chunk_fwht_raw(
                                            vlut_h_ptr,
                                            vls_ptr,
                                            c_idx,
                                            sub_slice,
                                            None,
                                        );
                                    } else {
                                        Int8ChunkActivation::populate_chunk_fwht_raw(
                                            act_hq,
                                            ls_ptr,
                                            lsum_ptr,
                                            c_idx,
                                            sub_slice,
                                            None,
                                        );
                                    }
                                }
                            }
                        }
                        let c0 = qh0 * chunks_per_head;
                        let n_c = (qh1 - qh0) * chunks_per_head;
                        if o_is_vq2 {
                            std::ptr::copy_nonoverlapping(vls_ptr.add(c0), vscales_h_ptr.add(c0), n_c);
                        } else {
                            std::ptr::copy_nonoverlapping(ls_ptr.add(c0), act_hs.add(c0), n_c);
                            std::ptr::copy_nonoverlapping(lsum_ptr.add(c0), act_hsum.add(c0), n_c);
                        }
                    }
                }
                barrier.wait();

                // 3. `o_proj` plus residual.
                let o_stripes = match &layer.o_proj {
                    LinearBackend::Nibble4(m) => m.num_stripes,
                    LinearBackend::NfSdot(m) => m.num_stripes,
                    LinearBackend::Vq2(m) => m.num_stripes,
                    _ => unreachable!(),
                };
                let o_pairs = o_stripes.div_ceil(2);
                let o_grain = (o_pairs / (nt * 4)).clamp(1, 2);
                while let Some((op0, op1)) = barrier.claim_work(o_grain, o_pairs) {
                    let os0 = (op0 * 2).min(o_stripes);
                    let os1 = (op1 * 2).min(o_stripes);
                    // SAFETY: `[os0*16, os1*16)` rows are disjoint per worker.
                    unsafe {
                        match &layer.o_proj {
                            LinearBackend::Nibble4(o_mat) => {
                                let act_h_ref = &*(act_h_addr as *const Int8ChunkActivation);
                                nibble4_sdot_stripe_range_int8(o_mat, act_h_ref, os0, os1, proj_ptr);
                            }
                            LinearBackend::NfSdot(o_mat) => {
                                let act_h_ref = &*(act_h_addr as *const Int8ChunkActivation);
                                nf_sdot_stripe_range_int8(o_mat, act_h_ref, os0, os1, proj_ptr);
                            }
                            LinearBackend::Vq2(o_mat) => {
                                vq2_lut_stripe_range_raw(o_mat, vlut_h_ptr, vscales_h_ptr, os0, os1, proj_ptr);
                            }
                            _ => unreachable!(),
                        }
                        let r0 = os0 * SIMD_ROW_STRIPE;
                        let r1 = (os1 * SIMD_ROW_STRIPE).min(h_dim);
                        add_inplace(
                            std::slice::from_raw_parts_mut(hidden_ptr.add(r0), r1 - r0),
                            std::slice::from_raw_parts(proj_ptr.add(r0), r1 - r0),
                        );
                    }
                }
                barrier.wait();

                // 4. Post-attention norm + FWHT, then fused `[gate; up]` -> SwiGLU -> quantize.
                let down_is_vq2 = matches!(layer.down_proj, LinearBackend::Vq2(_));
                let lis_ptr = if nc_i <= 256 {
                    lis.as_mut_ptr().cast::<f32>()
                } else if down_is_vq2 {
                    vscales_i_ptr
                } else {
                    act_is
                };
                let lisum_ptr = if nc_i <= 256 {
                    lisum.as_mut_ptr().cast::<i32>()
                } else {
                    act_isum
                };
                let mut gu_populated = false;
                while let Some((ic0, ic1)) = barrier.claim_work(ic_grain, nc_i) {
                    // SAFETY: `hidden` is read-only after barrier 3; `[ic0, ic1)` chunks are disjoint per worker.
                    unsafe {
                        if !gu_populated {
                            let hidden_ro = std::slice::from_raw_parts(hidden_ptr, h_dim);
                            let inv_rms = 1.0 / (dot_f32_fast(hidden_ro, hidden_ro) / (h_dim as f32) + eps).sqrt();
                            if matches!(layer.gate_up_proj, LinearBackend::Vq2(_)) {
                                for c in 0..nc_h {
                                    Vq2ActivationLut::populate_chunk_fwht_raw(
                                        vlq_ptr,
                                        vls_ptr,
                                        c,
                                        &hidden_ro[c * 64..(c + 1) * 64],
                                        Some((&layer.post_attn_norm[c * 64..(c + 1) * 64], inv_rms)),
                                    );
                                }
                            } else {
                                for c in 0..nc_h {
                                    Int8ChunkActivation::populate_chunk_fwht_raw(
                                        lq_ptr,
                                        ls_ptr,
                                        lsum_ptr,
                                        c,
                                        &hidden_ro[c * 64..(c + 1) * 64],
                                        Some((&layer.post_attn_norm[c * 64..(c + 1) * 64], inv_rms)),
                                    );
                                }
                            }
                            gu_populated = true;
                        }
                        if layer.gate_up_interleaved {
                            match &layer.gate_up_proj {
                                LinearBackend::Nibble4(gu_mat) => {
                                    nibble4_sdot_stripe_range_raw(gu_mat, lq_ptr, ls_ptr, lsum_ptr, ic0 * 8, ic1 * 8, gu_ptr);
                                }
                                LinearBackend::NfSdot(gu_mat) => {
                                    nf_sdot_stripe_range_raw(gu_mat, lq_ptr, ls_ptr, ic0 * 8, ic1 * 8, gu_ptr);
                                }
                                LinearBackend::Vq2(gu_mat) => {
                                    vq2_lut_stripe_range_raw(gu_mat, vlq_ptr, vls_ptr, ic0 * 8, ic1 * 8, gu_ptr);
                                }
                                _ => unreachable!(),
                            }
                        }
                        let mut sw64 = [0.0f32; 64];
                        for ic in ic0..ic1 {
                            let (g64, u64) = if layer.gate_up_interleaved {
                                (
                                    std::slice::from_raw_parts(gu_ptr.add(ic * 128), 64),
                                    std::slice::from_raw_parts(gu_ptr.add(ic * 128 + 64), 64),
                                )
                            } else {
                                match &layer.gate_up_proj {
                                    LinearBackend::Nibble4(gu_mat) => {
                                        nibble4_sdot_stripe_range_raw(gu_mat, lq_ptr, ls_ptr, lsum_ptr, ic * 4, ic * 4 + 4, gu_ptr);
                                        nibble4_sdot_stripe_range_raw(gu_mat, lq_ptr, ls_ptr, lsum_ptr, (nc_i + ic) * 4, (nc_i + ic) * 4 + 4, gu_ptr);
                                    }
                                    LinearBackend::NfSdot(gu_mat) => {
                                        nf_sdot_stripe_range_raw(gu_mat, lq_ptr, ls_ptr, ic * 4, ic * 4 + 4, gu_ptr);
                                        nf_sdot_stripe_range_raw(gu_mat, lq_ptr, ls_ptr, (nc_i + ic) * 4, (nc_i + ic) * 4 + 4, gu_ptr);
                                    }
                                    LinearBackend::Vq2(gu_mat) => {
                                        vq2_lut_stripe_range_raw(gu_mat, vlq_ptr, vls_ptr, ic * 4, ic * 4 + 4, gu_ptr);
                                        vq2_lut_stripe_range_raw(gu_mat, vlq_ptr, vls_ptr, (nc_i + ic) * 4, (nc_i + ic) * 4 + 4, gu_ptr);
                                    }
                                    _ => unreachable!(),
                                }
                                (
                                    std::slice::from_raw_parts(gu_ptr.add(ic * 64), 64),
                                    std::slice::from_raw_parts(gu_ptr.add(inter_dim + ic * 64), 64),
                                )
                            };
                            swiglu_activate(g64, u64, &mut sw64);
                            if down_is_vq2 {
                                Vq2ActivationLut::populate_chunk_fwht_raw(vlut_i_ptr, lis_ptr, ic, &sw64, None);
                            } else {
                                Int8ChunkActivation::populate_chunk_fwht_raw(act_iq, lis_ptr, lisum_ptr, ic, &sw64, None);
                            }
                        }
                        if nc_i <= 256 {
                            let n_ic = ic1 - ic0;
                            if down_is_vq2 {
                                std::ptr::copy_nonoverlapping(lis_ptr.add(ic0), vscales_i_ptr.add(ic0), n_ic);
                            } else {
                                std::ptr::copy_nonoverlapping(lis_ptr.add(ic0), act_is.add(ic0), n_ic);
                                std::ptr::copy_nonoverlapping(lisum_ptr.add(ic0), act_isum.add(ic0), n_ic);
                            }
                        }
                    }
                }
                barrier.wait();

                // 5. `down_proj` plus residual.
                let down_stripes = match &layer.down_proj {
                    LinearBackend::Nibble4(m) => m.num_stripes,
                    LinearBackend::NfSdot(m) => m.num_stripes,
                    LinearBackend::Vq2(m) => m.num_stripes,
                    _ => unreachable!(),
                };
                let down_pairs = down_stripes.div_ceil(2);
                let down_grain = (down_pairs / (nt * 4)).clamp(1, 2);
                while let Some((dp0, dp1)) = barrier.claim_work(down_grain, down_pairs) {
                    let ds0 = (dp0 * 2).min(down_stripes);
                    let ds1 = (dp1 * 2).min(down_stripes);
                    unsafe {
                        match &layer.down_proj {
                            LinearBackend::Nibble4(down_mat) => {
                                let act_i_ref = &*(act_i_addr as *const Int8ChunkActivation);
                                nibble4_sdot_stripe_range_int8(down_mat, act_i_ref, ds0, ds1, proj_ptr);
                            }
                            LinearBackend::NfSdot(down_mat) => {
                                let act_i_ref = &*(act_i_addr as *const Int8ChunkActivation);
                                nf_sdot_stripe_range_int8(down_mat, act_i_ref, ds0, ds1, proj_ptr);
                            }
                            LinearBackend::Vq2(down_mat) => {
                                vq2_lut_stripe_range_raw(down_mat, vlut_i_ptr, vscales_i_ptr, ds0, ds1, proj_ptr);
                            }
                            _ => unreachable!(),
                        }
                        let r0 = ds0 * SIMD_ROW_STRIPE;
                        let r1 = (ds1 * SIMD_ROW_STRIPE).min(h_dim);
                        add_inplace(
                            std::slice::from_raw_parts_mut(hidden_ptr.add(r0), r1 - r0),
                            std::slice::from_raw_parts(proj_ptr.add(r0), r1 - r0),
                        );
                    }
                }
                barrier.wait();
            }

            // 6. Final norm + FWHT, then `lm_head`.
            let head_stripes = match lm_head {
                LinearBackend::Nibble4(m) => m.num_stripes,
                LinearBackend::NfSdot(m) => m.num_stripes,
                LinearBackend::Vq2(m) => m.num_stripes,
                _ => unreachable!(),
            };
            let head_pairs = head_stripes.div_ceil(2);
            let head_grain = (head_pairs / (nt * 8)).clamp(2, 16);
            let mut head_populated = false;
            while let Some((hp0, hp1)) = barrier.claim_work(head_grain, head_pairs) {
                let hs0 = (hp0 * 2).min(head_stripes);
                let hs1 = (hp1 * 2).min(head_stripes);
                unsafe {
                    if !head_populated {
                        let hidden_ro = std::slice::from_raw_parts(hidden_ptr, h_dim);
                        let inv_rms = 1.0 / (dot_f32_fast(hidden_ro, hidden_ro) / (h_dim as f32) + eps).sqrt();
                        if matches!(lm_head, LinearBackend::Vq2(_)) {
                            for c in 0..nc_h {
                                Vq2ActivationLut::populate_chunk_fwht_raw(
                                    vlq_ptr,
                                    vls_ptr,
                                    c,
                                    &hidden_ro[c * 64..(c + 1) * 64],
                                    Some((&final_norm[c * 64..(c + 1) * 64], inv_rms)),
                                );
                            }
                        } else {
                            for c in 0..nc_h {
                                Int8ChunkActivation::populate_chunk_fwht_raw(
                                    lq_ptr,
                                    ls_ptr,
                                    lsum_ptr,
                                    c,
                                    &hidden_ro[c * 64..(c + 1) * 64],
                                    Some((&final_norm[c * 64..(c + 1) * 64], inv_rms)),
                                );
                            }
                        }
                        head_populated = true;
                    }
                    match lm_head {
                        LinearBackend::Nibble4(head_mat) => {
                            nibble4_sdot_stripe_range_raw(head_mat, lq_ptr, ls_ptr, lsum_ptr, hs0, hs1, logits_ptr);
                        }
                        LinearBackend::NfSdot(head_mat) => {
                            nf_sdot_stripe_range_raw(head_mat, lq_ptr, ls_ptr, hs0, hs1, logits_ptr);
                        }
                        LinearBackend::Vq2(head_mat) => {
                            vq2_lut_stripe_range_raw(head_mat, vlq_ptr, vls_ptr, hs0, hs1, logits_ptr);
                        }
                        _ => unreachable!(),
                    }
                }
            }
        };

        if let Some(pool) = self.pool.as_ref() {
            pool.execute_all(kernel);
        } else {
            let b1 = SpinBarrier::new(1);
            kernel(0, 1, &b1);
        }

        self.ws.pos += 1;
        &self.ws.logits
    }

    /// Runs a prompt as one batch (SDOT engines, see [`Self::repack_sdot`]) and returns the logits
    /// after its last token. Other engines fall back to token-by-token decoding.
    pub fn forward_prompt(&mut self, tokens: &[usize]) -> &[f32] {
        assert!(!tokens.is_empty());
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = self.metal.as_mut() {
            let h_dim = self.cfg.hidden_size;
            let ef32 = &self.embed_f32_rows;
            let ebf16 = &self.embed_bf16_rows;
            let enf4 = self.embed_nf4.as_ref();
            let lm_head = &self.lm_head;
            gpu.forward_tokens(
                &self.cfg,
                tokens,
                self.ws.pos,
                |t, dst| embed_token_into(ef32, ebf16, enf4, lm_head, h_dim, t, dst),
                None,
                &mut self.ws.logits,
            )
            .expect("Metal forward_prompt failed");
            self.ws.pos += tokens.len();
            return &self.ws.logits;
        }
        if !self.is_all_sdot() || tokens.len() == 1 {
            for &t in &tokens[..tokens.len() - 1] {
                self.forward_token(t);
            }
            return self.forward_token(tokens[tokens.len() - 1]);
        }
        const PROMPT_CHUNK: usize = 512;
        let chunks: Vec<&[usize]> = tokens.chunks(PROMPT_CHUNK).collect();
        for (idx, chunk) in chunks.iter().enumerate() {
            let last = idx + 1 == chunks.len();
            if chunk.len() == 1 && last {
                return self.forward_token(chunk[0]);
            }
            self.forward_batch_sdot_impl(chunk, None, last);
        }
        &self.ws.logits
    }

    /// Runs `tokens` (`len = m`) through one batched pass and writes `[m, vocab_size]` logits into
    /// `out_logits`.
    pub fn forward_batch_logits(&mut self, tokens: &[usize], out_logits: &mut [f32]) {
        assert!(!tokens.is_empty());
        let vocab = self.cfg.vocab_size;
        assert!(out_logits.len() >= tokens.len() * vocab);
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if let Some(gpu) = self.metal.as_mut() {
            let h_dim = self.cfg.hidden_size;
            let ef32 = &self.embed_f32_rows;
            let ebf16 = &self.embed_bf16_rows;
            let enf4 = self.embed_nf4.as_ref();
            let lm_head = &self.lm_head;
            gpu.forward_tokens(
                &self.cfg,
                tokens,
                self.ws.pos,
                |t, dst| embed_token_into(ef32, ebf16, enf4, lm_head, h_dim, t, dst),
                Some(&mut out_logits[..tokens.len() * vocab]),
                &mut self.ws.logits,
            )
            .expect("Metal forward_batch_logits failed");
            self.ws.pos += tokens.len();
            return;
        }
        if !self.is_all_sdot() || tokens.len() == 1 {
            for (t, &tok) in tokens.iter().enumerate() {
                let l = self.forward_token(tok);
                out_logits[t * vocab..(t + 1) * vocab].copy_from_slice(l);
            }
            return;
        }
        self.forward_batch_sdot_impl(tokens, Some(&mut out_logits[..tokens.len() * vocab]), true);
    }

    fn forward_batch_sdot_impl(
        &mut self,
        tokens: &[usize],
        all_logits_out: Option<&mut [f32]>,
        compute_last_logits: bool,
    ) {
        let n = tokens.len();
        let pos0 = self.ws.pos;
        assert!(pos0 + n <= self.cfg.max_seq_len, "prompt exceeds max_seq_len");
        let need_fp32_scores = self.ws.q8_kv.is_none();
        if let Some(q8) = self.ws.q8_kv.as_mut() {
            q8.ensure_seq_capacity(pos0 + n, pos0);
        } else {
            self.ws.ensure_fp32_kv(&self.cfg);
        }
        let h = self.cfg.hidden_size;
        let q_dim = self.cfg.q_dim();
        let inter = self.cfg.intermediate_size.max(64);
        let kv_dim = self.cfg.kv_dim();
        let qkv_rows = q_dim + 2 * kv_dim;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let num_kv_heads = self.cfg.num_key_value_heads;
        let kv_group = self.cfg.num_attention_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let max_seq_len = self.cfg.max_seq_len;
        let layer_kv_stride = max_seq_len * kv_dim;
        let eps = self.cfg.rms_norm_eps;
        let lut_h = self.ws.blut_hidden.lut_enabled;
        let lut_q = self.ws.blut_q.lut_enabled;
        let lut_i = self.ws.blut_inter.lut_enabled;
        let has_dense_ffn = self.layers.iter().any(|l| l.moe.is_none());
        let has_moe = self.cfg.is_moe();

        self.ws.prompt_scratch.ensure_capacity(
            n,
            &self.cfg,
            has_dense_ffn,
            qkv_rows,
            lut_h,
            lut_q,
            lut_i,
            need_fp32_scores,
        );

        let pool = self.pool.as_ref();
        let layers = &self.layers;
        let moe_rt = &mut self.moe_runtime;
        let ps = &mut self.ws.prompt_scratch;
        let hidden = &mut ps.hidden[..n * h];
        for (t, &tok) in tokens.iter().enumerate() {
            embed_token_into(
                &self.embed_f32_rows,
                &self.embed_bf16_rows,
                self.embed_nf4.as_ref(),
                &self.lm_head,
                h,
                tok,
                &mut hidden[t * h..(t + 1) * h],
            );
        }
        let acts_h = &mut ps.acts_h[..n];
        let acts_q = &mut ps.acts_q[..n];
        let acts_i = if has_dense_ffn {
            &mut ps.acts_i[..n]
        } else {
            &mut []
        };
        let qkv = &mut ps.qkv[..n * qkv_rows];
        let attn = &mut ps.attn[..n * q_dim];
        let proj = &mut ps.proj[..n * h];
        let gate_up = if has_dense_ffn {
            &mut ps.gate_up[..n * 2 * inter]
        } else {
            &mut []
        };
        let sw = if has_dense_ffn {
            &mut ps.sw[..n * inter]
        } else {
            &mut []
        };
        let moe_norm = if has_moe {
            &mut ps.moe_norm[..n * h]
        } else {
            &mut []
        };
        let score_stride = if head_dim == 64 {
            max_seq_len
        } else {
            kv_group.max(1) * max_seq_len
        };
        let scores_all = if need_fp32_scores {
            &mut ps.scores[..n * score_stride]
        } else {
            &mut []
        };

        for (l_idx, layer) in layers.iter().enumerate() {
            let kv_base = l_idx * layer_kv_stride;
            prompt_populate(pool, hidden, h, Some((&layer.input_norm, eps)), acts_h);
            prompt_gemm(pool, &layer.qkv_proj, acts_h, qkv);

            let qkv_ptr = qkv.as_mut_ptr() as usize;
            let qkv_bias = layer.qkv_bias.as_deref();
            let q_norm = layer.q_norm.as_deref();
            let k_norm = layer.k_norm.as_deref();
            let rope = &self.rope_cos_sin;
            let cfg = &self.cfg;
            if let Some(q8) = self.ws.q8_kv.as_mut() {
                let raw = q8.raw_ptrs();
                par_range(pool, n, |a, b| {
                    for t in a..b {
                        let pos = pos0 + t;
                        // SAFETY: token `t` owns row `t` of `qkv` and distinct ring/linear cache slot `slot_of_pos(pos)`.
                        unsafe {
                            let row = std::slice::from_raw_parts_mut(
                                (qkv_ptr as *mut f32).add(t * qkv_rows),
                                qkv_rows,
                            );
                            if let Some(bias) = qkv_bias {
                                add_inplace(row, bias);
                            }
                            let rope_row = &rope[pos * head_dim..(pos + 1) * head_dim];
                            let (q, kv) = row.split_at_mut(q_dim);
                            let (k, v) = kv.split_at_mut(kv_dim);
                            if let Some(qn) = q_norm {
                                apply_per_head_rms_norm(q, head_dim, qn, eps);
                            }
                            if let Some(kn) = k_norm {
                                apply_per_head_rms_norm(k, head_dim, kn, eps);
                            }
                            apply_llama_rope_precomp(q, cfg.num_attention_heads, head_dim, half_dim, rope_row);
                            apply_llama_rope_precomp(k, num_kv_heads, head_dim, half_dim, rope_row);
                            for kvh in 0..num_kv_heads {
                                raw.store_k(l_idx, kvh, pos, &k[kvh * head_dim..(kvh + 1) * head_dim]);
                                raw.store_v(l_idx, kvh, pos, &v[kvh * head_dim..(kvh + 1) * head_dim]);
                            }
                        }
                    }
                });

                let attn_ptr = attn.as_mut_ptr() as usize;
                let qkv_ref = &*qkv;
                let t_step = if n >= 16 { 4 } else if n >= 4 { 2 } else { 1 };
                let g_sub = kv_group.min(MAX_GQA_GROUP_BATCH / t_step).max(1);
                let g_blocks = kv_group.div_ceil(g_sub);
                let t_blocks = n.div_ceil(t_step);
                let total_items = num_kv_heads * g_blocks * t_blocks;
                let sc_cap = active_kv_len(pos0 + n, raw.window_size);
                let m_cap = t_step * g_sub;
                par_range(pool, total_items, |a, b| {
                    let needed = m_cap * sc_cap;
                    let mut sc_stack = [0.0f32; 16_384];
                    let mut sc_heap = Vec::new();
                    let sc_buf: &mut [f32] = if needed <= sc_stack.len() {
                        &mut sc_stack[..needed]
                    } else {
                        sc_heap.resize(needed, 0.0);
                        &mut sc_heap
                    };
                    for idx in a..b {
                        let kvh = idx / (g_blocks * t_blocks);
                        let rem = idx % (g_blocks * t_blocks);
                        let gb = rem / t_blocks;
                        let tb = rem % t_blocks;
                        let g0 = gb * g_sub;
                        let g1 = (g0 + g_sub).min(kv_group);
                        let g_len = g1 - g0;
                        let t0 = tb * t_step;
                        let t1 = (t0 + t_step).min(n);
                        let m = (t1 - t0) * g_len;
                        let mut sc_rest = &mut sc_buf[..m * sc_cap];
                        let mut queries: [GqaQueryRef<'_>; MAX_GQA_GROUP_BATCH] = std::array::from_fn(|j| {
                            if j < m {
                                let t = t0 + j / g_len;
                                let qh = kvh * kv_group + g0 + (j % g_len);
                                let seq_len = pos0 + t + 1;
                                let sc_len = active_kv_len(seq_len, raw.window_size);
                                let (sc_head, next) = std::mem::take(&mut sc_rest).split_at_mut(sc_cap);
                                sc_rest = next;
                                // SAFETY: each `(t, qh)` owns disjoint `attn[t * q_dim + qh * head_dim .. + head_dim]`.
                                let dst = unsafe {
                                    std::slice::from_raw_parts_mut(
                                        (attn_ptr as *mut f32).add(t * q_dim + qh * head_dim),
                                        head_dim,
                                    )
                                };
                                GqaQueryRef {
                                    q_head: &qkv_ref[t * qkv_rows + qh * head_dim..t * qkv_rows + (qh + 1) * head_dim],
                                    seq_len,
                                    sc: &mut sc_head[..sc_len],
                                    out: dst,
                                }
                            } else {
                                GqaQueryRef {
                                    q_head: &[],
                                    seq_len: 0,
                                    sc: &mut [],
                                    out: &mut [],
                                }
                            }
                        });
                        // SAFETY: KV stores completed across `par_range` above; `raw` buffers are read-only here.
                        unsafe {
                            raw.eval_gqa_group(l_idx, kvh, attn_scale, &mut queries[..m]);
                        }
                    }
                });
            } else {
                let k_ptr = self.ws.k_cache.as_mut_ptr() as usize;
                let v_ptr = self.ws.v_cache.as_mut_ptr() as usize;
                par_range(pool, n, |a, b| {
                    for t in a..b {
                        let pos = pos0 + t;
                        // SAFETY: token `t` owns row `t` of `qkv` and cache slot `pos`.
                        let row = unsafe {
                            std::slice::from_raw_parts_mut((qkv_ptr as *mut f32).add(t * qkv_rows), qkv_rows)
                        };
                        if let Some(bias) = qkv_bias {
                            add_inplace(row, bias);
                        }
                        let rope_row = &rope[pos * head_dim..(pos + 1) * head_dim];
                        let (q, kv) = row.split_at_mut(q_dim);
                        let (k, v) = kv.split_at_mut(kv_dim);
                        if let Some(qn) = q_norm {
                            apply_per_head_rms_norm(q, head_dim, qn, eps);
                        }
                        if let Some(kn) = k_norm {
                            apply_per_head_rms_norm(k, head_dim, kn, eps);
                        }
                        apply_llama_rope_precomp(q, cfg.num_attention_heads, head_dim, half_dim, rope_row);
                        apply_llama_rope_precomp(k, num_kv_heads, head_dim, half_dim, rope_row);
                        let off = kv_base + pos * kv_dim;
                        unsafe {
                            std::ptr::copy_nonoverlapping(k.as_ptr(), (k_ptr as *mut f32).add(off), kv_dim);
                            std::ptr::copy_nonoverlapping(v.as_ptr(), (v_ptr as *mut f32).add(off), kv_dim);
                        }
                    }
                });

                let k_cache = &self.ws.k_cache[kv_base..];
                let v_cache = &self.ws.v_cache[kv_base..];
                let attn_ptr = attn.as_mut_ptr() as usize;
                let scores_ptr = scores_all.as_mut_ptr() as usize;
                let qkv_ref = &*qkv;
                par_range(pool, n, |a, b| {
                    for t in a..b {
                        // SAFETY: token `t` owns row `t` of `attn` and `scores_all`.
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut((attn_ptr as *mut f32).add(t * q_dim), q_dim)
                        };
                        let sc = unsafe {
                            std::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(t * score_stride), score_stride)
                        };
                        causal_gqa_attention(
                            &qkv_ref[t * qkv_rows..t * qkv_rows + q_dim],
                            k_cache,
                            v_cache,
                            sc,
                            dst,
                            cfg.num_attention_heads,
                            kv_group,
                            head_dim,
                            kv_dim,
                            pos0 + t,
                            attn_scale,
                        );
                    }
                });
            }

            prompt_populate(pool, attn, q_dim, None, acts_q);
            prompt_gemm(pool, &layer.o_proj, acts_q, proj);
            if let Some(ref ob) = layer.o_bias {
                for row in proj.chunks_exact_mut(h) {
                    add_inplace(row, ob);
                }
            }
            add_inplace(hidden, proj);

            if let Some(ref moe_core) = layer.moe {
                if let (Some((future_l, ref projector)), Some(rt)) =
                    (&moe_core.lookahead, moe_rt.as_mut())
                {
                    if let Some(future_layer) = layers.get(*future_l) {
                        if let Some(ref future_moe) = future_layer.moe {
                            let pf_start = n.saturating_sub(4);
                            for t in pf_start..n {
                                rt.dispatch_lookahead_prefetch(
                                    &hidden[t * h..(t + 1) * h],
                                    projector,
                                    &future_layer.post_attn_norm,
                                    future_moe.moe_slot_idx,
                                    &future_moe.router,
                                    eps,
                                );
                            }
                        }
                    }
                }
                moe_rt
                    .as_mut()
                    .expect("SparseMoeRuntime missing on MoE layer")
                    .forward_layer_coalesced_batch(
                        moe_core,
                        hidden,
                        &layer.post_attn_norm,
                        eps,
                        n,
                        pool,
                        moe_norm,
                        proj,
                    );
                add_inplace(hidden, proj);
            } else {
                prompt_populate(pool, hidden, h, Some((&layer.post_attn_norm, eps)), acts_h);
                prompt_gemm(pool, &layer.gate_up_proj, acts_h, gate_up);
                let gu_ref = &*gate_up;
                let sw_ptr = sw.as_mut_ptr() as usize;
                let gu_interleaved = layer.gate_up_interleaved;
                par_range(pool, n, |a, b| {
                    for t in a..b {
                        let gu = &gu_ref[t * 2 * inter..(t + 1) * 2 * inter];
                        // SAFETY: token `t` owns row `t` of `sw`.
                        let out = unsafe {
                            std::slice::from_raw_parts_mut((sw_ptr as *mut f32).add(t * inter), inter)
                        };
                        if gu_interleaved {
                            for ic in 0..(inter / 64) {
                                swiglu_activate(
                                    &gu[ic * 128..ic * 128 + 64],
                                    &gu[ic * 128 + 64..ic * 128 + 128],
                                    &mut out[ic * 64..(ic + 1) * 64],
                                );
                            }
                        } else {
                            let (g, u) = gu.split_at(inter);
                            swiglu_activate(g, u, out);
                        }
                    }
                });
                prompt_populate(pool, sw, inter, None, acts_i);
                prompt_gemm(pool, &layer.down_proj, acts_i, proj);
                add_inplace(hidden, proj);
            }
        }

        if let Some(out_all) = all_logits_out {
            prompt_populate(pool, hidden, h, Some((&self.final_norm, eps)), acts_h);
            prompt_gemm(pool, &self.lm_head, acts_h, out_all);
            let vocab = self.cfg.vocab_size;
            self.ws.logits.copy_from_slice(&out_all[(n - 1) * vocab..n * vocab]);
        } else if compute_last_logits {
            let last_h = &hidden[(n - 1) * h..];
            if matches!(self.lm_head, LinearBackend::Vq2(_)) {
                let inv_rms = 1.0 / (dot_f32_fast(last_h, last_h) / (h as f32) + eps).sqrt();
                self.ws
                    .vqlut_hidden
                    .populate_rmsnorm_fwht(last_h, &self.final_norm, inv_rms);
            } else if !self.ws.blut_hidden.lut_enabled {
                let inv_rms = 1.0 / (dot_f32_fast(last_h, last_h) / (h as f32) + eps).sqrt();
                self.ws
                    .blut_hidden
                    .int8
                    .as_mut()
                    .unwrap()
                    .populate_rmsnorm_fwht(last_h, &self.final_norm, inv_rms);
            } else {
                rms_norm(last_h, &self.final_norm, eps, &mut self.ws.norm_buf);
                fwht64_signed_inplace(&mut self.ws.norm_buf);
                self.ws.blut_hidden.populate(&self.ws.norm_buf);
            }
            self.lm_head.forward_exec(
                &self.ws.norm_buf,
                None,
                Some(&self.ws.blut_hidden),
                Some(&self.ws.vqlut_hidden),
                &self.ws.q8_hidden_vals,
                &self.ws.q8_hidden_scales,
                pool,
                &mut self.ws.logits,
            );
        }
        self.ws.pos += n;
    }

    /// Greedy autoregressive generation (`temperature = 0`) returning `(generated_token_ids, tokens_per_sec)`.
    pub fn generate_greedy(
        &mut self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        eos_id: usize,
    ) -> (Vec<usize>, f64) {
        assert!(!prompt_ids.is_empty());
        self.reset_context();

        let t0 = std::time::Instant::now();
        let mut last_logits_argmax = 0usize;
        for &tok in prompt_ids {
            let logits = self.forward_token(tok);
            last_logits_argmax = argmax(logits);
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut next_tok = last_logits_argmax;
        for step in 0..max_new_tokens {
            if next_tok == eos_id {
                break;
            }
            generated.push(next_tok);
            if step + 1 < max_new_tokens {
                let logits = self.forward_token(next_tok);
                next_tok = argmax(logits);
            }
        }

        let total_forward_calls = prompt_ids.len() + generated.len().saturating_sub(1);
        let elapsed = t0.elapsed().as_secs_f64();
        let tok_per_sec = (total_forward_calls as f64) / elapsed.max(1e-9);
        (generated, tok_per_sec)
    }
}

#[derive(Clone, Copy)]
enum ProjSpec {
    Bipolar(usize),
    Nf { bits: usize, sgc: usize },
    Vq2(usize),
}

fn proj_specs_for_mode(mode: RealWeightMode) -> (ProjSpec, ProjSpec, ProjSpec, ProjSpec, ProjSpec) {
    match mode {
        RealWeightMode::KronosBipolarGptq { num_planes } => {
            let s = ProjSpec::Bipolar(num_planes);
            (s, s, s, s, s)
        }
        RealWeightMode::KronosBipolarGptqMixedK => (
            ProjSpec::Bipolar(4),
            ProjSpec::Bipolar(5),
            ProjSpec::Bipolar(4),
            ProjSpec::Bipolar(5),
            ProjSpec::Bipolar(5),
        ),
        RealWeightMode::KronosBipolarGptqPlanes { qkv, o, gate_up, down, head } => (
            ProjSpec::Bipolar(qkv),
            ProjSpec::Bipolar(o),
            ProjSpec::Bipolar(gate_up),
            ProjSpec::Bipolar(down),
            ProjSpec::Bipolar(head),
        ),
        RealWeightMode::KronosVq2Gptq { scale_group_chunks } => {
            let s = ProjSpec::Vq2(scale_group_chunks);
            (s, s, s, s, s)
        }
        RealWeightMode::KronosVq2MixedGptq {
            attn_head_planes,
            scale_group_chunks,
        } => (
            ProjSpec::Bipolar(attn_head_planes),
            ProjSpec::Bipolar(attn_head_planes),
            ProjSpec::Vq2(scale_group_chunks),
            ProjSpec::Vq2(scale_group_chunks),
            ProjSpec::Bipolar(attn_head_planes),
        ),
        RealWeightMode::KronosNfGptq {
            bits,
            scale_group_chunks,
        } => {
            let s = ProjSpec::Nf {
                bits,
                sgc: scale_group_chunks,
            };
            (s, s, s, s, s)
        }
        RealWeightMode::KronosNfMixedGptq {
            attn_head_bits,
            ffn_bits,
            scale_group_chunks,
        } => {
            let attn = ProjSpec::Nf {
                bits: attn_head_bits,
                sgc: scale_group_chunks,
            };
            let ffn = ProjSpec::Nf {
                bits: ffn_bits,
                sgc: scale_group_chunks,
            };
            (attn, attn, ffn, ffn, attn)
        }
        RealWeightMode::KronosNfPlanesGptq {
            qkv,
            o,
            gate_up,
            down,
            head,
            scale_group_chunks,
        } => (
            ProjSpec::Nf { bits: qkv, sgc: scale_group_chunks },
            ProjSpec::Nf { bits: o, sgc: scale_group_chunks },
            ProjSpec::Nf { bits: gate_up, sgc: scale_group_chunks },
            ProjSpec::Nf { bits: down, sgc: scale_group_chunks },
            ProjSpec::Nf { bits: head, sgc: scale_group_chunks },
        ),
        _ => panic!("sequential/streaming GPTQ expects KronosBipolarGptq*, KronosNf*, or KronosVq2* mode"),
    }
}

fn compile_proj_spec(
    weights: &[f32],
    rows: usize,
    cols: usize,
    spec: ProjSpec,
    hess: Option<&FullRowGptqHessian>,
    compile_threads: usize,
) -> LinearBackend {
    match spec {
        ProjSpec::Bipolar(num_planes) => LinearBackend::FusedBipolar(
            compile_hadamard_gptq_bipolar(weights, rows, cols, num_planes, hess, compile_threads),
        ),
        ProjSpec::Nf { bits, sgc } => LinearBackend::NfSdot(compile_hadamard_gptq_nf(
            weights,
            rows,
            cols,
            bits,
            sgc,
            hess,
            compile_threads,
        )),
        ProjSpec::Vq2(scale_group_chunks) => LinearBackend::Vq2(compile_hadamard_gptq_vq2(
            weights,
            rows,
            cols,
            scale_group_chunks,
            hess,
            compile_threads,
        )),
    }
}

fn eval_proj_rot(
    proj: &LinearBackend,
    rot_x: &[f32],
    blut: &mut BipolarActivationLut,
    vlut: &mut Vq2ActivationLut,
    out: &mut [f32],
) {
    match proj {
        LinearBackend::FusedBipolar(m) => {
            blut.populate(rot_x);
            simd_lut_bipolar_gemv(m, blut, out);
        }
        LinearBackend::NfSdot(m) => {
            let i8a = blut
                .int8
                .get_or_insert_with(|| Int8ChunkActivation::new_zeroed(m.cols));
            i8a.populate(rot_x);
            // SAFETY: `0..m.num_stripes` is in bounds and `out.len() == m.rows`.
            unsafe {
                nf_sdot_stripe_range_int8(m, i8a, 0, m.num_stripes, out.as_mut_ptr());
            }
        }
        LinearBackend::Vq2(m) => {
            vlut.populate(rot_x);
            vq2_lut_gemv(m, vlut, out);
        }
        _ => unreachable!("expected FusedBipolar, NfSdot, or Vq2 during sequential GPTQ"),
    }
}

#[inline]
fn embed_token_into(
    embed_f32_rows: &[f32],
    embed_bf16_rows: &[u16],
    embed_nf4: Option<&NfSdotMatrix>,
    lm_head: &LinearBackend,
    h_dim: usize,
    token: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), h_dim);
    if !embed_f32_rows.is_empty() {
        out.copy_from_slice(&embed_f32_rows[token * h_dim..(token + 1) * h_dim]);
        return;
    }
    if !embed_bf16_rows.is_empty() {
        let src = &embed_bf16_rows[token * h_dim..(token + 1) * h_dim];
        for (dst, &b16) in out.iter_mut().zip(src.iter()) {
            *dst = bf16_to_f32(b16);
        }
        return;
    }
    if let Some(m) = embed_nf4 {
        m.dequantize_row_rotated(token, out);
        ifwht64_signed_inplace(out);
        return;
    }
    match lm_head {
        LinearBackend::FusedBipolar(m) => m.dequantize_row_rotated(token, out),
        LinearBackend::Nibble4(m) => m.dequantize_row_rotated(token, out),
        LinearBackend::NfSdot(m) => m.dequantize_row_rotated(token, out),
        LinearBackend::Vq2(m) => m.dequantize_row_rotated(token, out),
        _ => unreachable!("embedding table dropped only for quantized tied lm_head"),
    }
    ifwht64_signed_inplace(out);
}

/// Runs `f(hidden_seq, buf_seq)` per calibration sequence in parallel; `buf_seq` is the sequence's
/// slice of a per-token buffer with `buf_dim` floats per token.
fn par_calib<F>(seq_hiddens: &mut [Vec<f32>], buf: &mut [f32], h_dim: usize, buf_dim: usize, f: F)
where
    F: Fn(&mut [f32], &mut [f32]) + Sync,
{
    let workers = std::thread::available_parallelism().map_or(1, |p| p.get()).clamp(1, 8);
    let mut buckets: Vec<Vec<(&mut Vec<f32>, &mut [f32])>> = (0..workers).map(|_| Vec::new()).collect();
    let mut rest = buf;
    for (i, seq_h) in seq_hiddens.iter_mut().enumerate() {
        let (head, tail) = rest.split_at_mut(seq_h.len() / h_dim * buf_dim);
        rest = tail;
        buckets[i % workers].push((seq_h, head));
    }
    let f = &f;
    std::thread::scope(|s| {
        for bucket in buckets {
            s.spawn(move || {
                for (seq_h, seq_buf) in bucket {
                    f(seq_h, seq_buf);
                }
            });
        }
    });
}

#[inline]
fn apply_per_head_rms_norm(heads_buf: &mut [f32], head_dim: usize, weight: &[f32], eps: f32) {
    debug_assert_eq!(weight.len(), head_dim);
    let inv_hd = 1.0 / (head_dim as f32);
    for head in heads_buf.chunks_exact_mut(head_dim) {
        let inv_rms = 1.0 / (dot_f32_fast(head, head) * inv_hd + eps).sqrt();
        for (v, &w) in head.iter_mut().zip(weight.iter()) {
            *v = (*v * inv_rms) * w;
        }
    }
}

/// `f(a, b)` over `0..n`, split across `pool` when present.
fn par_range(pool: Option<&SpinThreadPool>, n: usize, f: impl Fn(usize, usize) + Sync) {
    match pool {
        Some(p) if n > 1 => p.parallel_for(n, 1, f),
        _ => f(0, n),
    }
}

/// Per token: fused in-register optional RMSNorm + Hadamard rotation + INT8 activation quantization (`0` heap allocations!).
fn prompt_populate(
    pool: Option<&SpinThreadPool>,
    x: &[f32],
    dim: usize,
    norm: Option<(&[f32], f32)>,
    acts: &mut [BipolarActivationLut],
) {
    let acts_ptr = acts.as_mut_ptr() as usize;
    par_range(pool, acts.len(), |a, b| {
        let mut fallback_buf: Option<Vec<f32>> = None;
        for t in a..b {
            let row = &x[t * dim..(t + 1) * dim];
            // SAFETY: token `t` is owned by exactly one range.
            let act = unsafe { &mut *(acts_ptr as *mut BipolarActivationLut).add(t) };
            if !act.lut_enabled {
                let i8a = act.int8.as_mut().expect("int8 activations");
                match norm {
                    Some((w, eps)) => {
                        let inv_rms = 1.0 / (dot_f32_fast(row, row) / (dim as f32) + eps).sqrt();
                        i8a.populate_rmsnorm_fwht(row, w, inv_rms);
                    }
                    None => i8a.populate_fwht(row),
                }
            } else {
                let buf = fallback_buf.get_or_insert_with(|| vec![0.0f32; dim]);
                match norm {
                    Some((w, eps)) => rms_norm(row, w, eps, buf),
                    None => buf.copy_from_slice(row),
                }
                fwht64_signed_inplace(buf);
                act.populate(buf);
            }
        }
    });
}

/// Batched projection of all prompt tokens: `out[t * rows + r]`.
fn prompt_gemm(pool: Option<&SpinThreadPool>, mat: &LinearBackend, acts: &[BipolarActivationLut], out: &mut [f32]) {
    let out_ptr = out.as_mut_ptr() as usize;
    let par_stripes = |n_stripes: usize, grain: usize, f: &(dyn Fn(usize, usize) + Sync)| match pool {
        Some(p) if n_stripes > grain => p.parallel_for(n_stripes, grain, f),
        _ => f(0, n_stripes),
    };
    match mat {
        LinearBackend::Nibble4(m) => {
            debug_assert_eq!(out.len(), acts.len() * m.rows);
            par_stripes(m.num_stripes, 2, &|s0, s1| {
                // SAFETY: disjoint stripe ranges write disjoint rows of every token.
                unsafe { nibble4_sdot_batch_stripe_range(m, acts, s0, s1, out_ptr as *mut f32, m.rows) };
            });
        }
        LinearBackend::NfSdot(m) => {
            debug_assert_eq!(out.len(), acts.len() * m.rows);
            par_stripes(m.num_stripes, 8, &|s0, s1| {
                // SAFETY: disjoint stripe ranges write disjoint rows of every token.
                unsafe { nf_sdot_batch_stripe_range(m, acts, s0, s1, out_ptr as *mut f32, m.rows) };
            });
        }
        LinearBackend::Vq2(m) => {
            debug_assert_eq!(out.len(), acts.len() * m.rows);
            par_stripes(m.num_stripes, 4, &|s0, s1| {
                // SAFETY: disjoint stripe ranges write disjoint rows of every token.
                unsafe { vq2_sdot_batch_stripe_range(m, acts, s0, s1, out_ptr as *mut f32, m.rows) };
            });
        }
        _ => unreachable!("prompt_gemm requires Nibble4, NfSdot, or Vq2 SDOT projections"),
    }
}

/// Rotates `src` (`head_dim == 64` or `128`) by block-diagonal signed FWHT-64 and quantizes to
/// `head_dim` INT8 values plus one FP32 scale.
///
/// # Safety
/// `src.len() == head_dim` (`64` or `128`), `dst_q8` valid for `head_dim` `i8` writes, `dst_scale`
/// valid for `1` `f32` write.
#[inline]
unsafe fn store_hadamard_q8_head_raw(
    src: &[f32],
    head_dim: usize,
    dst_q8: *mut i8,
    dst_scale: *mut f32,
) {
    if head_dim == 64 {
        // SAFETY: caller upholds `64`-element bounds.
        unsafe { store_hadamard_q8_head64_raw(src, dst_q8, dst_scale) };
    } else {
        debug_assert_eq!(head_dim, 128);
        // SAFETY: caller upholds `128`-element bounds.
        unsafe { store_hadamard_q8_head128_raw(src, dst_q8, dst_scale) };
    }
}

/// Rotates `src` (64 floats) by the signed FWHT-64 and quantizes to 64 INT8 values plus one FP32 scale.
///
/// # Safety
/// `src.len() == 64`, `dst_q8` valid for `64` `i8` writes, `dst_scale` valid for `1` `f32` write.
#[inline]
unsafe fn store_hadamard_q8_head64_raw(src: &[f32], dst_q8: *mut i8, dst_scale: *mut f32) {
    debug_assert_eq!(src.len(), 64);
    let mut rot = [0.0f32; 64];
    rot.copy_from_slice(src);
    fwht64_signed_inplace(&mut rot);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`; caller upholds pointer bounds.
        let s = unsafe { quantize_rot64_to_i8_neon(&rot, dst_q8) };
        unsafe { *dst_scale = s };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let max_abs = rot.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if max_abs <= f32::EPSILON {
            unsafe {
                std::ptr::write_bytes(dst_q8, 0, 64);
                *dst_scale = 0.0;
            }
            return;
        }
        let inv = 127.0 / max_abs;
        for (i, &v) in rot.iter().enumerate() {
            unsafe { *dst_q8.add(i) = (v * inv).round_ties_even() as i32 as i8 };
        }
        unsafe { *dst_scale = max_abs / 127.0 };
    }
}

/// Rotates `src` (128 floats) by block-diagonal signed FWHT-64 and quantizes to 128 INT8 values
/// plus one FP32 scale (`3.88x` smaller than FP32 KV on `Qwen2.5-Coder-14B`).
///
/// # Safety
/// `src.len() == 128`, `dst_q8` valid for `128` `i8` writes, `dst_scale` valid for `1` `f32` write.
#[inline]
unsafe fn store_hadamard_q8_head128_raw(src: &[f32], dst_q8: *mut i8, dst_scale: *mut f32) {
    debug_assert_eq!(src.len(), 128);
    let mut rot = [0.0f32; 128];
    rot.copy_from_slice(src);
    fwht64_signed_inplace(&mut rot);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`; caller upholds pointer bounds.
        let s = unsafe { quantize_rot128_to_i8_neon(&rot, dst_q8) };
        unsafe { *dst_scale = s };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let max_abs = rot.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if max_abs <= f32::EPSILON {
            unsafe {
                std::ptr::write_bytes(dst_q8, 0, 128);
                *dst_scale = 0.0;
            }
            return;
        }
        let inv = 127.0 / max_abs;
        for (i, &v) in rot.iter().enumerate() {
            unsafe { *dst_q8.add(i) = (v * inv).round_ties_even() as i32 as i8 };
        }
        unsafe { *dst_scale = max_abs / 127.0 };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_rot64_to_i8_neon(rot: &[f32; 64], dst_q8: *mut i8) -> f32 {
    use std::arch::aarch64::*;
    let p = rot.as_ptr();
    let mut vmax = vdupq_n_f32(0.0);
    for i in 0..16 {
        // SAFETY: `i * 4 + 4 <= 64`.
        let v = unsafe { vld1q_f32(p.add(i * 4)) };
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    let max_abs = vmaxvq_f32(vmax);
    if max_abs <= f32::EPSILON {
        // SAFETY: caller guarantees `dst_q8` valid for `64` bytes.
        unsafe { std::ptr::write_bytes(dst_q8, 0, 64) };
        return 0.0;
    }
    let scale = max_abs / 127.0;
    let vinv = vdupq_n_f32(127.0 / max_abs);
    for g in 0..4 {
        let b = g * 16;
        // SAFETY: `b + 16 <= 64` for both `rot` and `dst_q8`.
        unsafe {
            let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b)), vinv));
            let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b + 4)), vinv));
            let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b + 8)), vinv));
            let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b + 12)), vinv));
            let s16_lo = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let s16_hi = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let s8 = vcombine_s8(vqmovn_s16(s16_lo), vqmovn_s16(s16_hi));
            vst1q_s8(dst_q8.add(b), s8);
        }
    }
    scale
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_rot128_to_i8_neon(rot: &[f32; 128], dst_q8: *mut i8) -> f32 {
    use std::arch::aarch64::*;
    let p = rot.as_ptr();
    let mut vmax = vdupq_n_f32(0.0);
    for i in 0..32 {
        // SAFETY: `i * 4 + 4 <= 128`.
        let v = unsafe { vld1q_f32(p.add(i * 4)) };
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    let max_abs = vmaxvq_f32(vmax);
    if max_abs <= f32::EPSILON {
        // SAFETY: caller guarantees `dst_q8` valid for `128` bytes.
        unsafe { std::ptr::write_bytes(dst_q8, 0, 128) };
        return 0.0;
    }
    let scale = max_abs / 127.0;
    let vinv = vdupq_n_f32(127.0 / max_abs);
    for g in 0..8 {
        let b = g * 16;
        // SAFETY: `b + 16 <= 128` for both `rot` and `dst_q8`.
        unsafe {
            let q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b)), vinv));
            let q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b + 4)), vinv));
            let q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b + 8)), vinv));
            let q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(p.add(b + 12)), vinv));
            let s16_lo = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            let s16_hi = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            let s8 = vcombine_s8(vqmovn_s16(s16_lo), vqmovn_s16(s16_hi));
            vst1q_s8(dst_q8.add(b), s8);
        }
    }
    scale
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_v16_i8(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SAFETY: `FEAT_DotProd` (`sdot`) is baseline on every Apple Silicon / ARMv8.4+ CPU.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod\nsdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

const Q8_ATTN_TILE: usize = 256;
const MAX_GQA_GROUP_BATCH: usize = 32;

struct GqaQueryRef<'a> {
    q_head: &'a [f32],
    seq_len: usize,
    sc: &'a mut [f32],
    out: &'a mut [f32],
}

/// L1D-tiled multi-query GQA attention over a shared INT8 Hadamard KV head; preserves exact `0..seq_len` accumulation order.
#[inline]
fn causal_gqa_group_q8(
    head_dim: usize,
    k_q8_head: &[i8],
    k_scales_head: &[f32],
    v_q8_head: &[i8],
    v_scales_head: &[f32],
    attn_scale: f32,
    queries: &mut [GqaQueryRef<'_>],
) {
    let m = queries.len();
    if m == 0 {
        return;
    }
    debug_assert!(m <= MAX_GQA_GROUP_BATCH && (head_dim == 64 || head_dim == 128));
    let mut q_i8 = [[0i8; 128]; MAX_GQA_GROUP_BATCH];
    let mut q_attn_scales = [0.0f32; MAX_GQA_GROUP_BATCH];
    let mut inv_sums = [0.0f32; MAX_GQA_GROUP_BATCH];
    let mut outs = [[0.0f32; 128]; MAX_GQA_GROUP_BATCH];
    let mut max_seq = 0usize;

    for (j, q) in queries.iter().enumerate() {
        debug_assert_eq!(q.q_head.len(), head_dim);
        debug_assert!(q.sc.len() >= q.seq_len && q.out.len() == head_dim);
        let mut q_scale = 0.0f32;
        // SAFETY: `q.q_head.len() == head_dim` and `q_i8[j]` holds 128 `i8`s.
        unsafe {
            if head_dim == 64 {
                store_hadamard_q8_head64_raw(q.q_head, q_i8[j].as_mut_ptr(), &mut q_scale);
            } else {
                store_hadamard_q8_head128_raw(q.q_head, q_i8[j].as_mut_ptr(), &mut q_scale);
            }
        }
        q_attn_scales[j] = q_scale * attn_scale;
        max_seq = max_seq.max(q.seq_len);
    }
    debug_assert!(
        k_q8_head.len() >= max_seq * head_dim
            && k_scales_head.len() >= max_seq
            && v_q8_head.len() >= max_seq * head_dim
            && v_scales_head.len() >= max_seq
    );

    for t0 in (0..max_seq).step_by(Q8_ATTN_TILE) {
        for (j, q) in queries.iter_mut().enumerate() {
            if t0 >= q.seq_len {
                continue;
            }
            let t1 = (t0 + Q8_ATTN_TILE).min(q.seq_len);
            let kt = &k_q8_head[t0 * head_dim..t1 * head_dim];
            let kst = &k_scales_head[t0..t1];
            let sct = &mut q.sc[t0..t1];
            let qas = q_attn_scales[j];
            #[cfg(target_arch = "aarch64")]
            // SAFETY: slice bounds verified above; NEON + `sdot` available on Apple Silicon.
            unsafe {
                if head_dim == 64 {
                    let qi: &[i8; 64] = (&q_i8[j][..64]).try_into().unwrap();
                    attn_scores_head64_q8_sdot_neon(qi, kt, kst, qas, sct);
                } else {
                    attn_scores_head128_q8_sdot_neon(&q_i8[j], kt, kst, qas, sct);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let qi = &q_i8[j][..head_dim];
                for (idx, dst_s) in sct.iter_mut().enumerate() {
                    let kt_row = &kt[idx * head_dim..(idx + 1) * head_dim];
                    let dot: i32 = qi
                        .iter()
                        .zip(kt_row.iter())
                        .map(|(&a, &b)| (a as i32) * (b as i32))
                        .sum();
                    *dst_s = (dot as f32) * kst[idx] * qas;
                }
            }
        }
    }

    for (j, q) in queries.iter_mut().enumerate() {
        let sc = &mut q.sc[..q.seq_len];
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        inv_sums[j] = 1.0 / softmax_unnorm_inplace(sc, max_score);
    }

    for t0 in (0..max_seq).step_by(Q8_ATTN_TILE) {
        for (j, q) in queries.iter().enumerate() {
            if t0 >= q.seq_len {
                continue;
            }
            let t1 = (t0 + Q8_ATTN_TILE).min(q.seq_len);
            let vt = &v_q8_head[t0 * head_dim..t1 * head_dim];
            let vst = &v_scales_head[t0..t1];
            let pt = &q.sc[t0..t1];
            let inv_s = inv_sums[j];
            #[cfg(target_arch = "aarch64")]
            // SAFETY: slice bounds verified above.
            unsafe {
                if head_dim == 64 {
                    let out64: &mut [f32; 64] = (&mut outs[j][..64]).try_into().unwrap();
                    attn_values_head64_q8_accum_neon(pt, vt, vst, inv_s, out64);
                } else {
                    attn_values_head128_q8_accum_neon(pt, vt, vst, inv_s, &mut outs[j]);
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let out_s = &mut outs[j][..head_dim];
                for idx in 0..(t1 - t0) {
                    let w = (pt[idx] * inv_s) * vst[idx];
                    let vt_row = &vt[idx * head_dim..(idx + 1) * head_dim];
                    for (dst, &vq) in out_s.iter_mut().zip(vt_row.iter()) {
                        *dst += w * (vq as f32);
                    }
                }
            }
        }
    }

    for (j, q) in queries.iter_mut().enumerate() {
        let out_s = &mut outs[j][..head_dim];
        ifwht64_signed_inplace(out_s);
        q.out.copy_from_slice(out_s);
    }
}

#[inline(always)]
fn active_kv_len(seq_len: usize, window_size: usize) -> usize {
    if window_size > 0 && seq_len > window_size {
        window_size
    } else {
        seq_len
    }
}

#[inline(always)]
fn active_kv_pos(idx: usize, seq_len: usize, window_size: usize, sink_tokens: usize) -> usize {
    if window_size > 0 && seq_len > window_size && idx >= sink_tokens {
        (seq_len - (window_size - sink_tokens)) + (idx - sink_tokens)
    } else {
        idx
    }
}

#[inline(always)]
fn nf4_nearest_idx(y: f32) -> u8 {
    let a = y.abs();
    let p = if a < 37.5 {
        if a < 12.0 {
            0
        } else if a < 24.5 {
            1
        } else {
            2
        }
    } else if a < 85.5 {
        if a < 51.0 {
            3
        } else if a < 66.5 {
            4
        } else {
            5
        }
    } else if a < 111.5 {
        6
    } else {
        7
    };
    if y >= 0.0 {
        8 + p
    } else {
        7 - p
    }
}

#[inline]
unsafe fn store_hadamard_q4_head_raw(
    head_s: &[f32],
    head_dim: usize,
    dst_q4: *mut u8,
    dst_s16: *mut u16,
) {
    debug_assert!(head_dim == 64 || head_dim == 128);
    let mut rot = [0.0f32; 128];
    rot[..head_dim].copy_from_slice(&head_s[..head_dim]);
    fwht64_signed_inplace(&mut rot[..head_dim]);
    let gph = head_dim / 64;
    for g in 0..gph {
        let blk = &rot[g * 64..(g + 1) * 64];
        let mut max_abs = 0.0f32;
        for &v in blk {
            let a = v.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        // SAFETY: caller guarantees `dst_q4` holds `head_dim / 2` bytes and `dst_s16` holds `gph` `u16`s.
        unsafe {
            if max_abs < 1e-30 {
                *dst_s16.add(g) = 0;
                std::ptr::write_bytes(dst_q4.add(g * 32), 0x88, 32);
                continue;
            }
            let inv0 = 127.0 / max_abs;
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for &v in blk {
                let c = NF4_DEFAULT_CODEBOOK_I8[nf4_nearest_idx(v * inv0) as usize] as f32;
                num += v * c;
                den += c * c;
            }
            let s_refined = if den > 1e-12 && num > 0.0 {
                num / den
            } else {
                max_abs / 127.0
            };
            let s_bf16 = f32_to_bf16(s_refined);
            let s_final = bf16_to_f32(s_bf16);
            let inv_final = if s_final > 1e-30 { 1.0 / s_final } else { 0.0 };
            *dst_s16.add(g) = s_bf16;
            for h in 0..2 {
                let base = h * 32;
                let out_p = dst_q4.add(g * 32 + h * 16);
                for i in 0..16 {
                    let lo = nf4_nearest_idx(blk[base + i] * inv_final);
                    let hi = nf4_nearest_idx(blk[base + 16 + i] * inv_final);
                    *out_p.add(i) = lo | (hi << 4);
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot64_q8_nf4_neon(q_i8: &[i8; 64], k_packed32: *const u8) -> i32 {
    use std::arch::aarch64::*;
    // SAFETY: `q_i8` has 64 bytes, `k_packed32` has 32 bytes, `NF4_DEFAULT_CODEBOOK_I8` has 16 bytes.
    unsafe {
        let cb = vld1q_s8(NF4_DEFAULT_CODEBOOK_I8.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let qp = q_i8.as_ptr();
        let p0 = vld1q_u8(k_packed32);
        let p1 = vld1q_u8(k_packed32.add(16));
        let k0 = vqtbl1q_s8(cb, vandq_u8(p0, mask));
        let k1 = vqtbl1q_s8(cb, vshrq_n_u8::<4>(p0));
        let k2 = vqtbl1q_s8(cb, vandq_u8(p1, mask));
        let k3 = vqtbl1q_s8(cb, vshrq_n_u8::<4>(p1));
        let mut acc = vdupq_n_s32(0);
        acc = sdot_v16_i8(acc, k0, vld1q_s8(qp));
        acc = sdot_v16_i8(acc, k1, vld1q_s8(qp.add(16)));
        acc = sdot_v16_i8(acc, k2, vld1q_s8(qp.add(32)));
        acc = sdot_v16_i8(acc, k3, vld1q_s8(qp.add(48)));
        vaddvq_s32(acc)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot64_q8_q8_neon(q_i8: &[i8; 64], k_q8: *const i8) -> i32 {
    use std::arch::aarch64::*;
    // SAFETY: `q_i8` and `k_q8` each hold 64 `i8` bytes.
    unsafe {
        let qp = q_i8.as_ptr();
        let mut acc = vdupq_n_s32(0);
        acc = sdot_v16_i8(acc, vld1q_s8(k_q8), vld1q_s8(qp));
        acc = sdot_v16_i8(acc, vld1q_s8(k_q8.add(16)), vld1q_s8(qp.add(16)));
        acc = sdot_v16_i8(acc, vld1q_s8(k_q8.add(32)), vld1q_s8(qp.add(32)));
        acc = sdot_v16_i8(acc, vld1q_s8(k_q8.add(48)), vld1q_s8(qp.add(48)));
        vaddvq_s32(acc)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accum64_nf4_neon(v_packed32: *const u8, w: f32, out: &mut [f32; 64]) {
    use std::arch::aarch64::*;
    // SAFETY: `v_packed32` holds 32 bytes and `out` holds 64 `f32` values.
    unsafe {
        let cb = vld1q_s8(NF4_DEFAULT_CODEBOOK_I8.as_ptr());
        let mask = vdupq_n_u8(0x0F);
        let vw = vdupq_n_f32(w);
        let op = out.as_mut_ptr();
        let p0 = vld1q_u8(v_packed32);
        let p1 = vld1q_u8(v_packed32.add(16));
        let vecs = [
            vqtbl1q_s8(cb, vandq_u8(p0, mask)),
            vqtbl1q_s8(cb, vshrq_n_u8::<4>(p0)),
            vqtbl1q_s8(cb, vandq_u8(p1, mask)),
            vqtbl1q_s8(cb, vshrq_n_u8::<4>(p1)),
        ];
        for (g, &vb) in vecs.iter().enumerate() {
            let lo16 = vmovl_s8(vget_low_s8(vb));
            let hi16 = vmovl_high_s8(vb);
            let b = g * 16;
            vst1q_f32(op.add(b), vfmaq_f32(vld1q_f32(op.add(b)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vw));
            vst1q_f32(op.add(b + 4), vfmaq_f32(vld1q_f32(op.add(b + 4)), vcvtq_f32_s32(vmovl_high_s16(lo16)), vw));
            vst1q_f32(op.add(b + 8), vfmaq_f32(vld1q_f32(op.add(b + 8)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vw));
            vst1q_f32(op.add(b + 12), vfmaq_f32(vld1q_f32(op.add(b + 12)), vcvtq_f32_s32(vmovl_high_s16(hi16)), vw));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn accum64_q8_neon(v_q8: *const i8, w: f32, out: &mut [f32; 64]) {
    use std::arch::aarch64::*;
    // SAFETY: `v_q8` holds 64 `i8` bytes and `out` holds 64 `f32` values.
    unsafe {
        let vw = vdupq_n_f32(w);
        let op = out.as_mut_ptr();
        for g in 0..4 {
            let vb = vld1q_s8(v_q8.add(g * 16));
            let lo16 = vmovl_s8(vget_low_s8(vb));
            let hi16 = vmovl_high_s8(vb);
            let b = g * 16;
            vst1q_f32(op.add(b), vfmaq_f32(vld1q_f32(op.add(b)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vw));
            vst1q_f32(op.add(b + 4), vfmaq_f32(vld1q_f32(op.add(b + 4)), vcvtq_f32_s32(vmovl_high_s16(lo16)), vw));
            vst1q_f32(op.add(b + 8), vfmaq_f32(vld1q_f32(op.add(b + 8)), vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vw));
            vst1q_f32(op.add(b + 12), vfmaq_f32(vld1q_f32(op.add(b + 12)), vcvtq_f32_s32(vmovl_high_s16(hi16)), vw));
        }
    }
}

unsafe fn causal_gqa_group_q4_or_window(
    raw: QuantKvRawPtrs,
    base_slot: usize,
    attn_scale: f32,
    queries: &mut [GqaQueryRef<'_>],
) {
    let hd = raw.head_dim;
    let gph = hd / 64;
    let half_hd = hd / 2;
    debug_assert!(hd == 64 || hd == 128);
    for q in queries.iter_mut() {
        let act_len = active_kv_len(q.seq_len, raw.window_size);
        debug_assert!(q.q_head.len() == hd && q.out.len() == hd && q.sc.len() >= act_len);
        let mut q_i8_flat = [0i8; 128];
        let mut q_scale = 0.0f32;
        // SAFETY: `q.q_head.len() == hd` (`64` or `128`) and `q_i8_flat` holds 128 `i8`s.
        unsafe {
            if hd == 64 {
                store_hadamard_q8_head64_raw(q.q_head, q_i8_flat.as_mut_ptr(), &mut q_scale);
            } else {
                store_hadamard_q8_head128_raw(q.q_head, q_i8_flat.as_mut_ptr(), &mut q_scale);
            }
        }
        let q_i8: [[i8; 64]; 2] = [
            q_i8_flat[..64].try_into().unwrap(),
            q_i8_flat[64..128].try_into().unwrap(),
        ];
        let q_s = [q_scale * attn_scale; 2];
        let sc = &mut q.sc[..act_len];
        for (idx, dst_s) in sc.iter_mut().enumerate() {
            let pos = active_kv_pos(idx, q.seq_len, raw.window_size, raw.sink_tokens);
            let slot = base_slot + raw.slot_of_pos(pos);
            let mut total = 0.0f32;
            if raw.bits == 4 {
                for g in 0..gph {
                    // SAFETY: `slot < (num_layers * num_kv_heads) * cap_seq_len`.
                    let ks = unsafe { bf16_to_f32(*(raw.k_s16 as *const u16).add(slot * gph + g)) };
                    let kp = unsafe { (raw.k_q4 as *const u8).add(slot * half_hd + g * 32) };
                    #[cfg(target_arch = "aarch64")]
                    let dot = unsafe { dot64_q8_nf4_neon(&q_i8[g], kp) };
                    #[cfg(not(target_arch = "aarch64"))]
                    let dot = unsafe {
                        let mut acc = 0i32;
                        for h in 0..2 {
                            for i in 0..16 {
                                let b = *kp.add(h * 16 + i);
                                let c0 = NF4_DEFAULT_CODEBOOK_I8[(b & 0x0F) as usize] as i32;
                                let c1 = NF4_DEFAULT_CODEBOOK_I8[(b >> 4) as usize] as i32;
                                acc += c0 * (q_i8[g][h * 32 + i] as i32)
                                    + c1 * (q_i8[g][h * 32 + 16 + i] as i32);
                            }
                        }
                        acc
                    };
                    total += (dot as f32) * ks * q_s[g];
                }
            } else {
                // SAFETY: `slot` in bounds for `k_q8` and `k_scales`.
                let ks = unsafe { *(raw.k_scales as *const f32).add(slot) };
                for g in 0..gph {
                    let kp = unsafe { (raw.k_q8 as *const i8).add(slot * hd + g * 64) };
                    #[cfg(target_arch = "aarch64")]
                    let dot = unsafe { dot64_q8_q8_neon(&q_i8[g], kp) };
                    #[cfg(not(target_arch = "aarch64"))]
                    let dot = unsafe {
                        let mut acc = 0i32;
                        for i in 0..64 {
                            acc += (*kp.add(i) as i32) * (q_i8[g][i] as i32);
                        }
                        acc
                    };
                    total += (dot as f32) * ks * q_s[g];
                }
            }
            *dst_s = total;
        }

        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        let mut out_blks = [[0.0f32; 64]; 2];
        for (idx, &p_raw) in sc.iter().enumerate() {
            let p_norm = p_raw * inv_sum;
            if p_norm == 0.0 {
                continue;
            }
            let pos = active_kv_pos(idx, q.seq_len, raw.window_size, raw.sink_tokens);
            let slot = base_slot + raw.slot_of_pos(pos);
            if raw.bits == 4 {
                for (g, out_blk) in out_blks.iter_mut().enumerate().take(gph) {
                    // SAFETY: `slot` in bounds for `v_q4` and `v_s16`.
                    let vs = unsafe { bf16_to_f32(*(raw.v_s16 as *const u16).add(slot * gph + g)) };
                    let w = p_norm * vs;
                    if w == 0.0 {
                        continue;
                    }
                    let vp = unsafe { (raw.v_q4 as *const u8).add(slot * half_hd + g * 32) };
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        accum64_nf4_neon(vp, w, out_blk);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    unsafe {
                        for h in 0..2 {
                            for i in 0..16 {
                                let b = *vp.add(h * 16 + i);
                                out_blk[h * 32 + i] +=
                                    w * (NF4_DEFAULT_CODEBOOK_I8[(b & 0x0F) as usize] as f32);
                                out_blk[h * 32 + 16 + i] +=
                                    w * (NF4_DEFAULT_CODEBOOK_I8[(b >> 4) as usize] as f32);
                            }
                        }
                    }
                }
            } else {
                // SAFETY: `slot` in bounds for `v_q8` and `v_scales`.
                let vs = unsafe { *(raw.v_scales as *const f32).add(slot) };
                let w = p_norm * vs;
                if w == 0.0 {
                    continue;
                }
                for (g, out_blk) in out_blks.iter_mut().enumerate().take(gph) {
                    let vp = unsafe { (raw.v_q8 as *const i8).add(slot * hd + g * 64) };
                    #[cfg(target_arch = "aarch64")]
                    unsafe {
                        accum64_q8_neon(vp, w, out_blk);
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    unsafe {
                        for (dst, i) in out_blk.iter_mut().zip(0..64) {
                            *dst += w * (*vp.add(i) as f32);
                        }
                    }
                }
            }
        }
        for (g, blk) in out_blks.iter().enumerate().take(gph) {
            q.out[g * 64..(g + 1) * 64].copy_from_slice(blk);
        }
        ifwht64_signed_inplace(&mut q.out[..hd]);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn causal_gqa_attention_q8(
    q_buf: &[f32],
    q8_kv: &QuantizedKvCache,
    layer_idx: usize,
    scores: &mut [f32],
    attn_out: &mut [f32],
    num_q_heads: usize,
    kv_group: usize,
    pos: usize,
    attn_scale: f32,
) {
    let seq_len = pos + 1;
    let hd = q8_kv.head_dim;
    let raw = q8_kv.raw_ptrs();
    let sc_len = active_kv_len(seq_len, raw.window_size);
    let mut qh0 = 0usize;
    while qh0 < num_q_heads {
        let kvh = qh0 / kv_group;
        let qh1 = ((kvh + 1) * kv_group).min(num_q_heads).min(qh0 + MAX_GQA_GROUP_BATCH);
        let m = qh1 - qh0;
        let mut sc_rest = &mut scores[..m * sc_len];
        let mut out_rest = &mut attn_out[qh0 * hd..qh1 * hd];
        let mut queries: [GqaQueryRef<'_>; MAX_GQA_GROUP_BATCH] = std::array::from_fn(|j| {
            if j < m {
                let qh = qh0 + j;
                let (sc_head, sc_next) = std::mem::take(&mut sc_rest).split_at_mut(sc_len);
                let (out_head, out_next) = std::mem::take(&mut out_rest).split_at_mut(hd);
                sc_rest = sc_next;
                out_rest = out_next;
                GqaQueryRef {
                    q_head: &q_buf[qh * hd..(qh + 1) * hd],
                    seq_len,
                    sc: sc_head,
                    out: out_head,
                }
            } else {
                GqaQueryRef {
                    q_head: &[],
                    seq_len: 0,
                    sc: &mut [],
                    out: &mut [],
                }
            }
        });
        // SAFETY: `q8_kv` owns valid buffers for `(layer_idx, kvh)` up to `pos`.
        unsafe {
            raw.eval_gqa_group(layer_idx, kvh, attn_scale, &mut queries[..m]);
        }
        qh0 = qh1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_scores_head64_q8_sdot_neon(
    q_i8: &[i8; 64],
    k_q8: &[i8],
    k_scales: &[f32],
    q_attn_scale: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let n = out.len();
    debug_assert!(k_q8.len() >= n * 64 && k_scales.len() >= n);
    let qp = q_i8.as_ptr();
    // SAFETY: `q_i8` holds 64 `i8` bytes.
    let (qv0, qv1, qv2, qv3) = unsafe {
        (
            vld1q_s8(qp),
            vld1q_s8(qp.add(16)),
            vld1q_s8(qp.add(32)),
            vld1q_s8(qp.add(48)),
        )
    };
    let kp = k_q8.as_ptr();
    let ksp = k_scales.as_ptr();
    let op = out.as_mut_ptr();
    let vqs = vdupq_n_f32(q_attn_scale);
    let mut t = 0usize;

    while t + 8 <= n {
        let mut acc = [vdupq_n_s32(0); 8];
        for (j, a) in acc.iter_mut().enumerate() {
            // SAFETY: `(t + j) * 64 + 64 <= n * 64`.
            unsafe {
                let kj = kp.add((t + j) * 64);
                let mut s = *a;
                s = sdot_v16_i8(s, vld1q_s8(kj), qv0);
                s = sdot_v16_i8(s, vld1q_s8(kj.add(16)), qv1);
                s = sdot_v16_i8(s, vld1q_s8(kj.add(32)), qv2);
                s = sdot_v16_i8(s, vld1q_s8(kj.add(48)), qv3);
                *a = s;
            }
        }
        let lo_i32 = vpaddq_s32(vpaddq_s32(acc[0], acc[1]), vpaddq_s32(acc[2], acc[3]));
        let hi_i32 = vpaddq_s32(vpaddq_s32(acc[4], acc[5]), vpaddq_s32(acc[6], acc[7]));
        // SAFETY: `t + 8 <= n`.
        unsafe {
            let s_lo = vmulq_f32(vld1q_f32(ksp.add(t)), vqs);
            let s_hi = vmulq_f32(vld1q_f32(ksp.add(t + 4)), vqs);
            vst1q_f32(op.add(t), vmulq_f32(vcvtq_f32_s32(lo_i32), s_lo));
            vst1q_f32(op.add(t + 4), vmulq_f32(vcvtq_f32_s32(hi_i32), s_hi));
        }
        t += 8;
    }

    while t < n {
        // SAFETY: `t < n`.
        unsafe {
            let kj = kp.add(t * 64);
            let mut a = vdupq_n_s32(0);
            a = sdot_v16_i8(a, vld1q_s8(kj), qv0);
            a = sdot_v16_i8(a, vld1q_s8(kj.add(16)), qv1);
            a = sdot_v16_i8(a, vld1q_s8(kj.add(32)), qv2);
            a = sdot_v16_i8(a, vld1q_s8(kj.add(48)), qv3);
            *op.add(t) = (vaddvq_s32(a) as f32) * (*ksp.add(t)) * q_attn_scale;
        }
        t += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_scores_head128_q8_sdot_neon(
    q_i8: &[i8; 128],
    k_q8: &[i8],
    k_scales: &[f32],
    q_attn_scale: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;
    let n = out.len();
    debug_assert!(k_q8.len() >= n * 128 && k_scales.len() >= n);
    let qp = q_i8.as_ptr();
    // SAFETY: `q_i8` holds 128 `i8` bytes.
    let qv: [int8x16_t; 8] = std::array::from_fn(|i| unsafe { vld1q_s8(qp.add(i * 16)) });
    let kp = k_q8.as_ptr();
    let ksp = k_scales.as_ptr();
    let op = out.as_mut_ptr();
    let vqs = vdupq_n_f32(q_attn_scale);
    let mut t = 0usize;

    while t + 8 <= n {
        let mut acc = [vdupq_n_s32(0); 8];
        for (j, a) in acc.iter_mut().enumerate() {
            // SAFETY: `(t + j) * 128 + 128 <= n * 128`.
            unsafe {
                let kj = kp.add((t + j) * 128);
                let mut s = *a;
                for (i, &qvi) in qv.iter().enumerate() {
                    s = sdot_v16_i8(s, vld1q_s8(kj.add(i * 16)), qvi);
                }
                *a = s;
            }
        }
        let lo_i32 = vpaddq_s32(vpaddq_s32(acc[0], acc[1]), vpaddq_s32(acc[2], acc[3]));
        let hi_i32 = vpaddq_s32(vpaddq_s32(acc[4], acc[5]), vpaddq_s32(acc[6], acc[7]));
        // SAFETY: `t + 8 <= n`.
        unsafe {
            let s_lo = vmulq_f32(vld1q_f32(ksp.add(t)), vqs);
            let s_hi = vmulq_f32(vld1q_f32(ksp.add(t + 4)), vqs);
            vst1q_f32(op.add(t), vmulq_f32(vcvtq_f32_s32(lo_i32), s_lo));
            vst1q_f32(op.add(t + 4), vmulq_f32(vcvtq_f32_s32(hi_i32), s_hi));
        }
        t += 8;
    }

    while t < n {
        // SAFETY: `t < n`.
        unsafe {
            let kj = kp.add(t * 128);
            let mut a = vdupq_n_s32(0);
            for (i, &qvi) in qv.iter().enumerate() {
                a = sdot_v16_i8(a, vld1q_s8(kj.add(i * 16)), qvi);
            }
            *op.add(t) = (vaddvq_s32(a) as f32) * (*ksp.add(t)) * q_attn_scale;
        }
        t += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_values_head64_q8_accum_neon(
    p: &[f32],
    v_q8: &[i8],
    v_scales: &[f32],
    inv_sum: f32,
    out: &mut [f32; 64],
) {
    use std::arch::aarch64::*;
    let n = p.len();
    debug_assert!(v_q8.len() >= n * 64 && v_scales.len() >= n);
    let vp = v_q8.as_ptr();
    let op = out.as_mut_ptr();
    // SAFETY: `out` holds 64 `f32` values.
    let mut acc: [float32x4_t; 16] = std::array::from_fn(|i| unsafe { vld1q_f32(op.add(i * 4)) });

    for t in 0..n {
        let w = (p[t] * inv_sum) * v_scales[t];
        if w == 0.0 {
            continue;
        }
        let vw = vdupq_n_f32(w);
        // SAFETY: `t * 64 + 64 <= n * 64`.
        unsafe {
            let vt = vp.add(t * 64);
            for g in 0..4 {
                let vb = vld1q_s8(vt.add(g * 16));
                let lo16 = vmovl_s8(vget_low_s8(vb));
                let hi16 = vmovl_high_s8(vb);
                let b = g * 4;
                acc[b] = vfmaq_f32(acc[b], vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vw);
                acc[b + 1] = vfmaq_f32(acc[b + 1], vcvtq_f32_s32(vmovl_high_s16(lo16)), vw);
                acc[b + 2] = vfmaq_f32(acc[b + 2], vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vw);
                acc[b + 3] = vfmaq_f32(acc[b + 3], vcvtq_f32_s32(vmovl_high_s16(hi16)), vw);
            }
        }
    }

    for (i, a) in acc.iter().enumerate() {
        // SAFETY: `i * 4 + 4 <= 64`.
        unsafe { vst1q_f32(op.add(i * 4), *a) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_values_head128_q8_accum_neon(
    p: &[f32],
    v_q8: &[i8],
    v_scales: &[f32],
    inv_sum: f32,
    out: &mut [f32; 128],
) {
    use std::arch::aarch64::*;
    let n = p.len();
    debug_assert!(v_q8.len() >= n * 128 && v_scales.len() >= n);
    let vp = v_q8.as_ptr();
    let op = out.as_mut_ptr();

    // Two 16-register passes (`0..64` then `64..128`) eliminate all register spills on 32-reg NEON.
    for half_off in [0usize, 64] {
        // SAFETY: `half_off + 64 <= 128`.
        let mut acc: [float32x4_t; 16] =
            std::array::from_fn(|i| unsafe { vld1q_f32(op.add(half_off + i * 4)) });
        for t in 0..n {
            let w = (p[t] * inv_sum) * v_scales[t];
            if w == 0.0 {
                continue;
            }
            let vw = vdupq_n_f32(w);
            // SAFETY: `t * 128 + half_off + 64 <= n * 128`.
            unsafe {
                let vt = vp.add(t * 128 + half_off);
                for g in 0..4 {
                    let vb = vld1q_s8(vt.add(g * 16));
                    let lo16 = vmovl_s8(vget_low_s8(vb));
                    let hi16 = vmovl_high_s8(vb);
                    let b = g * 4;
                    acc[b] = vfmaq_f32(acc[b], vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), vw);
                    acc[b + 1] = vfmaq_f32(acc[b + 1], vcvtq_f32_s32(vmovl_high_s16(lo16)), vw);
                    acc[b + 2] = vfmaq_f32(acc[b + 2], vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), vw);
                    acc[b + 3] = vfmaq_f32(acc[b + 3], vcvtq_f32_s32(vmovl_high_s16(hi16)), vw);
                }
            }
        }
        for (i, a) in acc.iter().enumerate() {
            // SAFETY: `half_off + i * 4 + 4 <= 128`.
            unsafe { vst1q_f32(op.add(half_off + i * 4), *a) };
        }
    }
}

fn build_rope_table(cfg: &LlamaConfig) -> Vec<f32> {
    let head_dim = cfg.head_dim();
    let half_dim = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| {
            let exp = (2 * i) as f64 / (head_dim as f64);
            (1.0 / (cfg.rope_theta as f64).powf(exp)) as f32
        })
        .collect();

    let mut table = vec![0.0f32; cfg.max_seq_len * head_dim];
    for pos in 0..cfg.max_seq_len {
        let pos_f = pos as f32;
        let row = &mut table[pos * head_dim..(pos + 1) * head_dim];
        for (i, &freq) in inv_freq.iter().enumerate() {
            let (s, c) = (pos_f * freq).sin_cos();
            row[i] = c;
            row[half_dim + i] = s;
        }
    }
    table
}

#[inline]
fn apply_llama_rope_head(head: &mut [f32], rope_row: &[f32]) {
    if head.len() == 64 {
        apply_llama_rope_head64(head, rope_row);
    } else if head.len() == 128 {
        apply_llama_rope_head128(head, rope_row);
    } else {
        let hd = head.len();
        apply_llama_rope_precomp(head, 1, hd, hd / 2, rope_row);
    }
}

#[inline]
fn apply_llama_rope_head64(head: &mut [f32], rope_row: &[f32]) {
    debug_assert_eq!(head.len(), 64);
    debug_assert_eq!(rope_row.len(), 64);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and `head`/`rope_row` hold 64 floats.
        unsafe { apply_llama_rope_head64_neon(head, rope_row) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        apply_llama_rope_precomp(head, 1, 64, 32, rope_row);
    }
}

#[inline]
fn apply_llama_rope_head128(head: &mut [f32], rope_row: &[f32]) {
    debug_assert_eq!(head.len(), 128);
    debug_assert_eq!(rope_row.len(), 128);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and `head`/`rope_row` hold 128 floats.
        unsafe { apply_llama_rope_head128_neon(head, rope_row) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (x0_slice, x1_slice) = head.split_at_mut(64);
        let (cos_s, sin_s) = rope_row.split_at(64);
        for i in 0..64 {
            let (x0, x1) = (x0_slice[i], x1_slice[i]);
            let (c, s) = (cos_s[i], sin_s[i]);
            x0_slice[i] = x0 * c - x1 * s;
            x1_slice[i] = x1 * c + x0 * s;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_llama_rope_head64_neon(head: &mut [f32], rope_row: &[f32]) {
    use std::arch::aarch64::*;
    let hp = head.as_mut_ptr();
    let cp = rope_row.as_ptr();
    let sp = unsafe { cp.add(32) };
    for k in 0..8 {
        let i = k * 4;
        // SAFETY: `i + 4 <= 32` and `32 + i + 4 <= 64`.
        unsafe {
            let x0 = vld1q_f32(hp.add(i));
            let x1 = vld1q_f32(hp.add(32 + i));
            let c = vld1q_f32(cp.add(i));
            let s = vld1q_f32(sp.add(i));
            vst1q_f32(hp.add(i), vsubq_f32(vmulq_f32(x0, c), vmulq_f32(x1, s)));
            vst1q_f32(hp.add(32 + i), vaddq_f32(vmulq_f32(x1, c), vmulq_f32(x0, s)));
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_llama_rope_head128_neon(head: &mut [f32], rope_row: &[f32]) {
    use std::arch::aarch64::*;
    let hp = head.as_mut_ptr();
    let cp = rope_row.as_ptr();
    let sp = unsafe { cp.add(64) };
    for k in 0..16 {
        let i = k * 4;
        // SAFETY: `i + 4 <= 64` and `64 + i + 4 <= 128`.
        unsafe {
            let x0 = vld1q_f32(hp.add(i));
            let x1 = vld1q_f32(hp.add(64 + i));
            let c = vld1q_f32(cp.add(i));
            let s = vld1q_f32(sp.add(i));
            vst1q_f32(hp.add(i), vsubq_f32(vmulq_f32(x0, c), vmulq_f32(x1, s)));
            vst1q_f32(hp.add(64 + i), vaddq_f32(vmulq_f32(x1, c), vmulq_f32(x0, s)));
        }
    }
}

#[inline]
fn apply_llama_rope_precomp(
    buf: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    half_dim: usize,
    rope_row: &[f32],
) {
    if head_dim == 64 && half_dim == 32 {
        for h in 0..num_heads {
            apply_llama_rope_head64(&mut buf[h * 64..(h + 1) * 64], rope_row);
        }
        return;
    }
    if head_dim == 128 && half_dim == 64 {
        for h in 0..num_heads {
            apply_llama_rope_head128(&mut buf[h * 128..(h + 1) * 128], rope_row);
        }
        return;
    }
    let cos_s = &rope_row[..half_dim];
    let sin_s = &rope_row[half_dim..head_dim];
    for h in 0..num_heads {
        let head = &mut buf[h * head_dim..(h + 1) * head_dim];
        let (x0_slice, x1_slice) = head.split_at_mut(half_dim);
        for i in 0..half_dim {
            let x0 = x0_slice[i];
            let x1 = x1_slice[i];
            let c = cos_s[i];
            let s = sin_s[i];
            x0_slice[i] = x0 * c - x1 * s;
            x1_slice[i] = x1 * c + x0 * s;
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn causal_gqa_head_fp32(
    q_head: &[f32],
    k_layer_cache: &[f32],
    v_layer_cache: &[f32],
    sc: &mut [f32],
    out_head: &mut [f32],
    kv_dim: usize,
    off: usize,
    attn_scale: f32,
) {
    let hd = q_head.len();
    debug_assert_eq!(out_head.len(), hd);
    for (t, dst) in sc.iter_mut().enumerate() {
        let k_head = &k_layer_cache[t * kv_dim + off..t * kv_dim + off + hd];
        *dst = dot_f32_fast(q_head, k_head) * attn_scale;
    }
    let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
    out_head.fill(0.0);
    for (t, &s) in sc.iter().enumerate() {
        let v_head = &v_layer_cache[t * kv_dim + off..t * kv_dim + off + hd];
        axpy_f32_fast(out_head, v_head, s * inv_sum);
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn causal_gqa_head64(
    q_head: &[f32],
    k_layer_cache: &[f32],
    v_layer_cache: &[f32],
    sc: &mut [f32],
    out_head: &mut [f32; 64],
    kv_dim: usize,
    off: usize,
    attn_scale: f32,
) {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `q_head.len() == 64`, `off + 64 <= kv_dim`, caches hold `sc.len()` rows.
        unsafe { attn_scores_head64_neon(q_head, k_layer_cache, kv_dim, off, attn_scale, sc) };
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        // SAFETY: as above, with `out_head.len() == 64`.
        unsafe { attn_values_head64_neon(sc, v_layer_cache, kv_dim, off, inv_sum, out_head) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (t, dst) in sc.iter_mut().enumerate() {
            let k_head = &k_layer_cache[t * kv_dim + off..t * kv_dim + off + 64];
            *dst = dot_f32_fast(q_head, k_head) * attn_scale;
        }
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        out_head.fill(0.0);
        for (t, &s) in sc.iter().enumerate() {
            let v_head = &v_layer_cache[t * kv_dim + off..t * kv_dim + off + 64];
            axpy_f32_fast(out_head, v_head, s * inv_sum);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn exp_f32_neon(x_raw: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    let x = vminq_f32(vmaxq_f32(x_raw, vdupq_n_f32(-87.336_54)), vdupq_n_f32(88.0));
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let kf = vrndnq_f32(vmulq_f32(x, log2e));
    // Cody-Waite two-part ln(2) reduction (`0.693145751953125 + 1.428606765330187e-6 == ln(2)`)
    let r = vfmsq_f32(
        vfmsq_f32(x, kf, vdupq_n_f32(0.693_145_75)),
        kf,
        vdupq_n_f32(1.428_606_8e-6),
    );

    let one = vdupq_n_f32(1.0);
    let mut y = vfmaq_f32(vdupq_n_f32(1.398_2e-3), vdupq_n_f32(1.987_569_1e-4), r);
    y = vfmaq_f32(vdupq_n_f32(8.333_452e-3), y, r);
    y = vfmaq_f32(vdupq_n_f32(4.166_579_6e-2), y, r);
    y = vfmaq_f32(vdupq_n_f32(1.666_666_6e-1), y, r);
    y = vfmaq_f32(vdupq_n_f32(0.5), y, r);
    y = vfmaq_f32(one, y, r);
    y = vfmaq_f32(one, y, r);

    let ki = vcvtnq_s32_f32(kf);
    let pow2k = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(ki, vdupq_n_s32(127))));
    vmulq_f32(y, pow2k)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn causal_gqa_attention(
    q_buf: &[f32],
    k_layer_cache: &[f32],
    v_layer_cache: &[f32],
    scores: &mut [f32],
    attn_out: &mut [f32],
    num_q_heads: usize,
    kv_group: usize,
    head_dim: usize,
    kv_dim: usize,
    pos: usize,
    attn_scale: f32,
) {
    let seq_len = pos + 1;
    #[cfg(target_arch = "aarch64")]
    if head_dim == 64 {
        for qh in 0..num_q_heads {
            let off = (qh / kv_group) * head_dim;
            let q_head = &q_buf[qh * head_dim..(qh + 1) * head_dim];
            let sc = &mut scores[..seq_len];
            // SAFETY: rows `0..seq_len` of the caches hold `kv_dim >= off + 64` floats each.
            unsafe { attn_scores_head64_neon(q_head, k_layer_cache, kv_dim, off, attn_scale, sc) };
            let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
            let out_head = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
            // SAFETY: as above.
            unsafe { attn_values_head64_neon(sc, v_layer_cache, kv_dim, off, inv_sum, out_head) };
        }
        return;
    }
    // Query heads sharing a KV head are processed together so each K/V row is read once.
    for kvh in 0..num_q_heads / kv_group {
        let q0 = kvh * kv_group;
        let sc = &mut scores[..kv_group * seq_len];
        for t in 0..seq_len {
            let k_off = t * kv_dim + kvh * head_dim;
            let k_head = &k_layer_cache[k_off..k_off + head_dim];
            for g in 0..kv_group {
                let q_head = &q_buf[(q0 + g) * head_dim..(q0 + g + 1) * head_dim];
                sc[g * seq_len + t] = dot_f32_fast(q_head, k_head) * attn_scale;
            }
        }
        for g in 0..kv_group {
            let row = &mut sc[g * seq_len..(g + 1) * seq_len];
            let max_score = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let inv_sum = 1.0 / softmax_unnorm_inplace(row, max_score);
            for p in row.iter_mut() {
                *p *= inv_sum;
            }
        }
        attn_out[q0 * head_dim..(q0 + kv_group) * head_dim].fill(0.0);
        for t in 0..seq_len {
            let v_off = t * kv_dim + kvh * head_dim;
            let v_head = &v_layer_cache[v_off..v_off + head_dim];
            for g in 0..kv_group {
                let out_head = &mut attn_out[(q0 + g) * head_dim..(q0 + g + 1) * head_dim];
                axpy_f32_fast(out_head, v_head, sc[g * seq_len + t]);
            }
        }
    }
}

/// `out[t] = scale * q . k_t` for a 64-wide head, 8 positions per pass with one reduction tree.
///
/// # Safety
/// `k_cache` holds `out.len()` rows of `kv_dim` floats with `off + 64 <= kv_dim`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_scores_head64_neon(q: &[f32], k_cache: &[f32], kv_dim: usize, off: usize, scale: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    debug_assert!(q.len() == 64 && off + 64 <= kv_dim && k_cache.len() >= out.len() * kv_dim);
    // SAFETY: `q` holds 64 floats.
    let qv: [float32x4_t; 16] = std::array::from_fn(|i| unsafe { vld1q_f32(q.as_ptr().add(4 * i)) });
    let kp = k_cache.as_ptr();
    let n = out.len();
    let mut t = 0;
    while t + 8 <= n {
        let mut acc = [vdupq_n_f32(0.0); 8];
        for (i, qi) in qv.iter().enumerate() {
            for (j, a) in acc.iter_mut().enumerate() {
                // SAFETY: row `t + j < n`, column `off + 4i + 4 <= kv_dim`.
                *a = vfmaq_f32(*a, *qi, unsafe { vld1q_f32(kp.add((t + j) * kv_dim + off + 4 * i)) });
            }
        }
        let lo = vpaddq_f32(vpaddq_f32(acc[0], acc[1]), vpaddq_f32(acc[2], acc[3]));
        let hi = vpaddq_f32(vpaddq_f32(acc[4], acc[5]), vpaddq_f32(acc[6], acc[7]));
        // SAFETY: `t + 8 <= n`.
        unsafe {
            vst1q_f32(out.as_mut_ptr().add(t), vmulq_n_f32(lo, scale));
            vst1q_f32(out.as_mut_ptr().add(t + 4), vmulq_n_f32(hi, scale));
        }
        t += 8;
    }
    for (j, o) in out.iter_mut().enumerate().skip(t) {
        let mut a = vdupq_n_f32(0.0);
        for (i, qi) in qv.iter().enumerate() {
            // SAFETY: row `j < n`.
            a = vfmaq_f32(a, *qi, unsafe { vld1q_f32(kp.add(j * kv_dim + off + 4 * i)) });
        }
        *o = vaddvq_f32(a) * scale;
    }
}

/// `out = inv_sum * sum_t p[t] * v_t` for a 64-wide head, accumulating in registers.
///
/// # Safety
/// As [`attn_scores_head64_neon`], with `out.len() == 64`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_values_head64_neon(p: &[f32], v_cache: &[f32], kv_dim: usize, off: usize, inv_sum: f32, out: &mut [f32]) {
    use std::arch::aarch64::*;
    debug_assert!(out.len() == 64 && off + 64 <= kv_dim && v_cache.len() >= p.len() * kv_dim);
    let vp = v_cache.as_ptr();
    let mut acc = [vdupq_n_f32(0.0); 16];
    for (t, &pt) in p.iter().enumerate() {
        for (i, a) in acc.iter_mut().enumerate() {
            // SAFETY: row `t < p.len()`, column `off + 4i + 4 <= kv_dim`.
            *a = vfmaq_n_f32(*a, unsafe { vld1q_f32(vp.add(t * kv_dim + off + 4 * i)) }, pt);
        }
    }
    for (i, a) in acc.iter().enumerate() {
        // SAFETY: `out` holds 64 floats.
        unsafe { vst1q_f32(out.as_mut_ptr().add(4 * i), vmulq_n_f32(*a, inv_sum)) };
    }
}

#[inline]
fn softmax_unnorm_inplace(scores: &mut [f32], max_score: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`.
        unsafe { softmax_unnorm_neon(scores, max_score) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum_exp = 0.0f32;
        for s in scores {
            *s = (*s - max_score).exp();
            sum_exp += *s;
        }
        sum_exp
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn softmax_unnorm_neon(scores: &mut [f32], max_score: f32) -> f32 {
    use std::arch::aarch64::*;
    let n = scores.len();
    let ptr = scores.as_mut_ptr();
    let vmax = vdupq_n_f32(max_score);
    let mut vacc = vdupq_n_f32(0.0);
    let mut i = 0usize;
    while i + 4 <= n {
        // SAFETY: `i + 4 <= n`.
        unsafe {
            let v = vsubq_f32(vld1q_f32(ptr.add(i)), vmax);
            let e = exp_f32_neon(v);
            vst1q_f32(ptr.add(i), e);
            vacc = vaddq_f32(vacc, e);
        }
        i += 4;
    }
    let mut sum_exp = vaddvq_f32(vacc);
    while i < n {
        // SAFETY: `i < n`.
        unsafe {
            let e = (*ptr.add(i) - max_score).exp();
            *ptr.add(i) = e;
            sum_exp += e;
        }
        i += 1;
    }
    sum_exp
}

#[inline]
fn axpy_f32_fast(dst: &mut [f32], src: &[f32], alpha: f32) {
    debug_assert_eq!(dst.len(), src.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and `dst.len() == src.len()`.
        unsafe { axpy_f32_neon(dst, src, alpha) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (d, &s) in dst.iter_mut().zip(src.iter()) {
            *d += alpha * s;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_f32_neon(dst: &mut [f32], src: &[f32], alpha: f32) {
    use std::arch::aarch64::*;
    let n = dst.len();
    let pd = dst.as_mut_ptr();
    let ps = src.as_ptr();
    let va = vdupq_n_f32(alpha);
    let mut i = 0usize;
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            vst1q_f32(pd.add(i), vfmaq_f32(vld1q_f32(pd.add(i)), vld1q_f32(ps.add(i)), va));
            vst1q_f32(
                pd.add(i + 4),
                vfmaq_f32(vld1q_f32(pd.add(i + 4)), vld1q_f32(ps.add(i + 4)), va),
            );
            vst1q_f32(
                pd.add(i + 8),
                vfmaq_f32(vld1q_f32(pd.add(i + 8)), vld1q_f32(ps.add(i + 8)), va),
            );
            vst1q_f32(
                pd.add(i + 12),
                vfmaq_f32(vld1q_f32(pd.add(i + 12)), vld1q_f32(ps.add(i + 12)), va),
            );
        }
        i += 16;
    }
    while i < n {
        // SAFETY: `i < n`.
        unsafe {
            *pd.add(i) += alpha * (*ps.add(i));
        }
        i += 1;
    }
}

#[inline]
fn swiglu_activate(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and slices have equal length.
        unsafe { swiglu_activate_neon(gate, up, out) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        swiglu_activate_scalar(gate, up, out);
    }
}

#[inline]
#[cfg(any(not(target_arch = "aarch64"), test))]
fn swiglu_activate_scalar(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((dst, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *dst = (g / (1.0 + (-g).exp())) * u;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn swiglu_activate_neon(gate: &[f32], up: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let n = out.len();
    let pg = gate.as_ptr();
    let pu = up.as_ptr();
    let po = out.as_mut_ptr();
    let one = vdupq_n_f32(1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        // SAFETY: `i + 8 <= n`.
        unsafe {
            let g0 = vld1q_f32(pg.add(i));
            let u0 = vld1q_f32(pu.add(i));
            let g1 = vld1q_f32(pg.add(i + 4));
            let u1 = vld1q_f32(pu.add(i + 4));

            let denom0 = vaddq_f32(one, exp_f32_neon(vnegq_f32(g0)));
            let denom1 = vaddq_f32(one, exp_f32_neon(vnegq_f32(g1)));

            vst1q_f32(po.add(i), vmulq_f32(vdivq_f32(g0, denom0), u0));
            vst1q_f32(po.add(i + 4), vmulq_f32(vdivq_f32(g1, denom1), u1));
        }
        i += 8;
    }
    while i < n {
        // SAFETY: `i < n`.
        unsafe {
            let g = *pg.add(i);
            let u = *pu.add(i);
            *po.add(i) = (g / (1.0 + (-g).exp())) * u;
        }
        i += 1;
    }
}

#[inline]
fn add_inplace(dst: &mut [f32], src: &[f32]) {
    axpy_f32_fast(dst, src, 1.0);
}

#[inline]
fn accum_sq(x: &[f32], sq: &mut [f64]) {
    for (s, &v) in sq.iter_mut().zip(x.iter()) {
        let vf = v as f64;
        *s += vf * vf;
    }
}

#[inline]
fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_sq = dot_f32_fast(x, x) / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`.
        unsafe {
            use std::arch::aarch64::*;
            let n = out.len();
            let px = x.as_ptr();
            let pw = weight.as_ptr();
            let po = out.as_mut_ptr();
            let vs = vdupq_n_f32(inv_rms);
            let mut i = 0usize;
            while i + 8 <= n {
                vst1q_f32(
                    po.add(i),
                    vmulq_f32(vmulq_f32(vld1q_f32(px.add(i)), vs), vld1q_f32(pw.add(i))),
                );
                vst1q_f32(
                    po.add(i + 4),
                    vmulq_f32(
                        vmulq_f32(vld1q_f32(px.add(i + 4)), vs),
                        vld1q_f32(pw.add(i + 4)),
                    ),
                );
                i += 8;
            }
            while i < n {
                *po.add(i) = (*px.add(i) * inv_rms) * (*pw.add(i));
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for ((o, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
            *o = (xi * inv_rms) * wi;
        }
    }
}

#[inline]
pub fn argmax(slice: &[f32]) -> usize {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`.
        unsafe { argmax_neon(slice) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        argmax_scalar(slice)
    }
}

#[inline]
fn argmax_scalar(slice: &[f32]) -> usize {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in slice.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn argmax_neon(slice: &[f32]) -> usize {
    use std::arch::aarch64::*;
    let n = slice.len();
    if n < 64 {
        return argmax_scalar(slice);
    }
    let ptr = slice.as_ptr();
    let mut best_val = f32::NEG_INFINITY;
    let mut best_block = 0usize;
    let mut b = 0usize;
    while b + 64 <= n {
        // SAFETY: `b + 64 <= n`.
        let block_max = unsafe {
            let m0 = vmaxq_f32(
                vmaxq_f32(vld1q_f32(ptr.add(b)), vld1q_f32(ptr.add(b + 4))),
                vmaxq_f32(vld1q_f32(ptr.add(b + 8)), vld1q_f32(ptr.add(b + 12))),
            );
            let m1 = vmaxq_f32(
                vmaxq_f32(vld1q_f32(ptr.add(b + 16)), vld1q_f32(ptr.add(b + 20))),
                vmaxq_f32(vld1q_f32(ptr.add(b + 24)), vld1q_f32(ptr.add(b + 28))),
            );
            let m2 = vmaxq_f32(
                vmaxq_f32(vld1q_f32(ptr.add(b + 32)), vld1q_f32(ptr.add(b + 36))),
                vmaxq_f32(vld1q_f32(ptr.add(b + 40)), vld1q_f32(ptr.add(b + 44))),
            );
            let m3 = vmaxq_f32(
                vmaxq_f32(vld1q_f32(ptr.add(b + 48)), vld1q_f32(ptr.add(b + 52))),
                vmaxq_f32(vld1q_f32(ptr.add(b + 56)), vld1q_f32(ptr.add(b + 60))),
            );
            vmaxvq_f32(vmaxq_f32(vmaxq_f32(m0, m1), vmaxq_f32(m2, m3)))
        };
        if block_max > best_val {
            best_val = block_max;
            best_block = b;
        }
        b += 64;
    }
    let mut best_idx = best_block + argmax_scalar(&slice[best_block..best_block + 64]);
    let mut exact_best = slice[best_idx];
    for (off, &val) in slice[b..].iter().enumerate() {
        if val > exact_best {
            exact_best = val;
            best_idx = b + off;
        }
    }
    best_idx
}

/// Quantitative output-quality metrics comparing a candidate engine against exact `FP32` reference.
#[derive(Clone, Debug)]
pub struct QualityEvalReport {
    pub perplexity: f64,
    pub mean_logit_cosine_sim: f64,
    pub mean_kl_divergence_nats: f64,
    pub top1_agreement_pct: f64,
    pub top5_agreement_pct: f64,
    pub eval_tokens_per_sec: f64,
}

/// Evaluates teacher-forced Perplexity on `eval_ids` (`len >= 2`) and compares step-by-step
/// full-vocabulary logits against `ref_logits_per_step` (from `ExactFp32`).
pub fn evaluate_quality_against_reference(
    engine: &mut RealLlamaEngine,
    eval_ids: &[usize],
    ref_logits_per_step: &[Vec<f32>],
) -> QualityEvalReport {
    assert!(eval_ids.len() >= 2);
    let steps = eval_ids.len() - 1;
    assert_eq!(ref_logits_per_step.len(), steps);

    engine.reset_context();
    let mut nll_sum = 0.0f64;
    let mut cos_sum = 0.0f64;
    let mut kl_sum = 0.0f64;
    let mut top1_matches = 0usize;
    let mut top5_matches = 0usize;
    let mut forward_secs = 0.0f64;

    for step in 0..steps {
        let tok_in = eval_ids[step];
        let target_next = eval_ids[step + 1];
        let t_fwd = std::time::Instant::now();
        let cand_logits = engine.forward_token(tok_in);
        forward_secs += t_fwd.elapsed().as_secs_f64();
        let ref_logits = &ref_logits_per_step[step];

        let cand_lse = log_sum_exp(cand_logits);
        let log_prob_target = (cand_logits[target_next] as f64) - cand_lse;
        nll_sum -= log_prob_target;

        cos_sum += logit_cosine_f64(ref_logits, cand_logits);

        let ref_lse = log_sum_exp(ref_logits);
        let mut step_kl = 0.0f64;
        for (&rl, &cl) in ref_logits.iter().zip(cand_logits.iter()) {
            let log_p = (rl as f64) - ref_lse;
            if log_p > -40.0 {
                let p = log_p.exp();
                let log_q = (cl as f64) - cand_lse;
                step_kl += p * (log_p - log_q);
            }
        }
        kl_sum += step_kl.max(0.0);

        let ref_top1 = argmax(ref_logits);
        let cand_top5 = top_5_indices(cand_logits);
        if cand_top5[0] == ref_top1 {
            top1_matches += 1;
        }
        if cand_top5.contains(&ref_top1) {
            top5_matches += 1;
        }
    }

    QualityEvalReport {
        perplexity: (nll_sum / (steps as f64)).exp(),
        mean_logit_cosine_sim: cos_sum / (steps as f64),
        mean_kl_divergence_nats: kl_sum / (steps as f64),
        top1_agreement_pct: (top1_matches as f64 / steps as f64) * 100.0,
        top5_agreement_pct: (top5_matches as f64 / steps as f64) * 100.0,
        eval_tokens_per_sec: (steps as f64) / forward_secs.max(1e-9),
    }
}

/// Collects exact `FP32` reference logits and reference Perplexity across `eval_ids[0..len-1]`.
pub fn collect_fp32_reference_logits(
    fp32_engine: &mut RealLlamaEngine,
    eval_ids: &[usize],
) -> (Vec<Vec<f32>>, f64, f64) {
    assert!(eval_ids.len() >= 2);
    let steps = eval_ids.len() - 1;
    fp32_engine.reset_context();

    let mut all_logits = Vec::with_capacity(steps);
    let mut nll_sum = 0.0f64;
    let mut forward_secs = 0.0f64;

    for step in 0..steps {
        let t_fwd = std::time::Instant::now();
        let logits = fp32_engine.forward_token(eval_ids[step]);
        forward_secs += t_fwd.elapsed().as_secs_f64();
        let lse = log_sum_exp(logits);
        nll_sum -= (logits[eval_ids[step + 1]] as f64) - lse;
        all_logits.push(logits.to_vec());
    }

    let ppl = (nll_sum / (steps as f64)).exp();
    let tps = (steps as f64) / forward_secs.max(1e-9);
    (all_logits, ppl, tps)
}

fn log_sum_exp(logits: &[f32]) -> f64 {
    let max_v = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&v| ((v as f64) - max_v).exp()).sum();
    max_v + sum.ln()
}

fn logit_cosine_f64(a: &[f32], b: &[f32]) -> f64 {
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

fn top_5_indices(logits: &[f32]) -> [usize; 5] {
    let mut top = [(f32::NEG_INFINITY, 0usize); 5];
    for (idx, &val) in logits.iter().enumerate() {
        if val > top[4].0 {
            let mut pos = 4;
            while pos > 0 && val > top[pos - 1].0 {
                top[pos] = top[pos - 1];
                pos -= 1;
            }
            top[pos] = (val, idx);
        }
    }
    [top[0].1, top[1].1, top[2].1, top[3].1, top[4].1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_swiglu_and_softmax_accuracy() {
        let gate: Vec<f32> = (0..1536)
            .map(|i| -40.0 + (i as f32) * (80.0 / 1535.0))
            .collect();
        let up: Vec<f32> = (0..1536).map(|i| ((i as f32) * 0.13).cos() * 3.0).collect();

        let mut out_neon = vec![0.0f32; 1536];
        let mut out_ref = vec![0.0f32; 1536];
        swiglu_activate(&gate, &up, &mut out_neon);
        swiglu_activate_scalar(&gate, &up, &mut out_ref);

        for (a, b) in out_neon.iter().zip(out_ref.iter()) {
            let tol = 2e-6 * b.abs().max(1.0);
            assert!(
                (a - b).abs() <= tol,
                "SwiGLU NEON vs scalar mismatch: {a} vs {b}"
            );
        }

        let mut scores_neon: Vec<f32> = (0..129).map(|i| ((i as f32) * 0.27).sin() * 8.0).collect();
        let mut scores_ref = scores_neon.clone();
        let max_s = scores_neon
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_neon = softmax_unnorm_inplace(&mut scores_neon, max_s);
        let mut sum_ref = 0.0f32;
        for s in &mut scores_ref {
            *s = (*s - max_s).exp();
            sum_ref += *s;
        }
        assert!((sum_neon - sum_ref).abs() < 1e-4);
        for (a, b) in scores_neon.iter().zip(scores_ref.iter()) {
            assert!((a - b).abs() < 1e-6, "Softmax mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn test_attention_matches_naive() {
        let (heads, kv_heads, hd) = (6, 2, 64);
        let kv_dim = kv_heads * hd;
        for pos in [0usize, 5, 7, 8, 30] {
            let n = pos + 1;
            let f = |i: usize, m: usize| ((i * m % 97) as f32 / 97.0 - 0.5) * 2.0;
            let q: Vec<f32> = (0..heads * hd).map(|i| f(i, 31)).collect();
            let k: Vec<f32> = (0..n * kv_dim).map(|i| f(i, 17)).collect();
            let v: Vec<f32> = (0..n * kv_dim).map(|i| f(i, 13)).collect();
            let mut scores = vec![0.0f32; n * heads];
            let mut out = vec![0.0f32; heads * hd];
            causal_gqa_attention(&q, &k, &v, &mut scores, &mut out, heads, heads / kv_heads, hd, kv_dim, pos, 0.125);
            for h in 0..heads {
                let off = (h / (heads / kv_heads)) * hd;
                let sc: Vec<f64> = (0..n)
                    .map(|t| (0..hd).map(|d| (q[h * hd + d] * k[t * kv_dim + off + d]) as f64).sum::<f64>() * 0.125)
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
                let z: f64 = e.iter().sum();
                for d in 0..hd {
                    let want: f64 = (0..n).map(|t| e[t] / z * v[t * kv_dim + off + d] as f64).sum();
                    let got = out[h * hd + d] as f64;
                    assert!((got - want).abs() < 1e-4, "pos {pos} head {h} dim {d}: {got} vs {want}");
                }
            }
        }
    }

    #[test]
    fn test_hadamard_q8_kv_cache_attention_accuracy() {
        for hd in [64usize, 128] {
            let heads = 6usize;
            let kv_heads = 2usize;
            let cfg = LlamaConfig {
                vocab_size: 256,
                hidden_size: heads * hd,
                intermediate_size: 1024,
                num_hidden_layers: 2,
                num_attention_heads: heads,
                num_key_value_heads: kv_heads,
                head_dim_override: None,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                tie_word_embeddings: true,
                max_seq_len: 64,
                sliding_window: None,
                sink_tokens: 0,
                num_experts: 0,
                num_experts_per_tok: 0,
                moe_intermediate_size: 0,
                shared_expert_intermediate_size: 0,
                norm_topk_prob: true,
                router_scoring: MoeRouterScoring::Softmax,
                decoder_sparse_step: 1,
                mlp_only_layers: Vec::new(),
                sliding_layer_mask: Vec::new(),
            };
            let kv_dim = kv_heads * hd;
            let attn_scale = 1.0 / (hd as f32).sqrt();
            let mut q8 = QuantizedKvCache::new(&cfg);
            assert_eq!(q8.physical_bytes(), 2 * 2 * 64 * (2 * hd + 8));

            // 12x outliers on two channels per head, as RoPE produces.
            let f = |i: usize, m: usize| {
                let base = ((i * m % 101) as f32 / 101.0 - 0.5) * 1.6;
                if (i % hd) == 7 || (i % hd) == (hd / 2 + 7) {
                    base * 12.0
                } else {
                    base
                }
            };
            let n_max = 37usize;
            let k_all: Vec<f32> = (0..n_max * kv_dim).map(|i| f(i, 19)).collect();
            let v_all: Vec<f32> = (0..n_max * kv_dim).map(|i| f(i, 29)).collect();
            for pos in 0..n_max {
                for kvh in 0..kv_heads {
                    let off = pos * kv_dim + kvh * hd;
                    q8.store_k_head(1, kvh, pos, &k_all[off..off + hd]);
                    q8.store_v_head(1, kvh, pos, &v_all[off..off + hd]);
                }
            }

            for pos in [0usize, 7, 15, 36] {
                let q: Vec<f32> = (0..heads * hd).map(|i| f(i + pos * 13, 37)).collect();
                let mut sc_f32 = vec![0.0f32; (pos + 1) * heads];
                let mut sc_q8 = vec![0.0f32; (pos + 1) * heads];
                let mut out_f32 = vec![0.0f32; heads * hd];
                let mut out_q8 = vec![0.0f32; heads * hd];

                causal_gqa_attention(
                    &q,
                    &k_all,
                    &v_all,
                    &mut sc_f32,
                    &mut out_f32,
                    heads,
                    heads / kv_heads,
                    hd,
                    kv_dim,
                    pos,
                    attn_scale,
                );
                causal_gqa_attention_q8(
                    &q,
                    &q8,
                    1,
                    &mut sc_q8,
                    &mut out_q8,
                    heads,
                    heads / kv_heads,
                    pos,
                    attn_scale,
                );

                let cos = logit_cosine_f64(&out_f32, &out_q8);
                assert!(
                    cos > 0.99980,
                    "Hadamard Q8 KV (hd={hd}) attention cosine at pos={pos} too low: {cos}"
                );
            }
        }
    }

    #[test]
    fn test_qwen2_head128_qkv_bias_streaming_gptq_and_megakernel() {
        use crate::simd_lut::f32_to_bf16;

        let dir = std::env::temp_dir().join(format!("kronos_qwen2_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let (vocab, h, inter, nl, nh, nkv) = (128usize, 512usize, 1024usize, 2usize, 4usize, 2usize);
        let hd = h / nh;
        assert_eq!(hd, 128);
        let kvd = nkv * hd;
        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"vocab_size":{vocab},"hidden_size":{h},"intermediate_size":{inter},"num_hidden_layers":{nl},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":false}}"#
            ),
        )
        .unwrap();

        let mk_bf16 = |n: usize, seed: usize, scale: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let mut z = (i as u64)
                    .wrapping_add((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
                let v = (2.0 * u - 1.0) * scale;
                out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
            out
        };
        let write_shard = |path: &Path, entries: Vec<(String, Vec<usize>, Vec<u8>)>| {
            let mut data = Vec::new();
            let mut hdr_items = Vec::new();
            for (name, shape, raw) in entries {
                let s = data.len();
                data.extend_from_slice(&raw);
                let e = data.len();
                let sh_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                hdr_items.push(format!(
                    r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{s},{e}]}}"#,
                    sh_str.join(",")
                ));
            }
            let hdr = format!("{{{}}}", hdr_items.join(","));
            let mut file_bytes = (hdr.len() as u64).to_le_bytes().to_vec();
            file_bytes.extend_from_slice(hdr.as_bytes());
            file_bytes.extend_from_slice(&data);
            fs::write(path, file_bytes).unwrap();
        };

        let mut s1 = vec![(
            "model.embed_tokens.weight".to_string(),
            vec![vocab, h],
            mk_bf16(vocab * h, 31, 0.4),
        )];
        let mut s2 = Vec::new();
        let mut wm = vec![r#""model.embed_tokens.weight":"s1.safetensors""#.to_string()];
        for l in 0..nl {
            let dst = if l == 0 { &mut s1 } else { &mut s2 };
            let sname = if l == 0 { "s1.safetensors" } else { "s2.safetensors" };
            let p = format!("model.layers.{l}");
            let specs = [
                (format!("{p}.input_layernorm.weight"), vec![h], mk_bf16(h, 7 + l, 1.0)),
                (format!("{p}.post_attention_layernorm.weight"), vec![h], mk_bf16(h, 11 + l, 1.0)),
                (format!("{p}.self_attn.q_proj.weight"), vec![h, h], mk_bf16(h * h, 13 + l, 0.08)),
                (format!("{p}.self_attn.k_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 17 + l, 0.08)),
                (format!("{p}.self_attn.v_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 19 + l, 0.08)),
                (format!("{p}.self_attn.q_proj.bias"), vec![h], mk_bf16(h, 23 + l, 0.25)),
                (format!("{p}.self_attn.k_proj.bias"), vec![kvd], mk_bf16(kvd, 29 + l, 0.25)),
                (format!("{p}.self_attn.v_proj.bias"), vec![kvd], mk_bf16(kvd, 37 + l, 0.25)),
                (format!("{p}.self_attn.o_proj.weight"), vec![h, h], mk_bf16(h * h, 41 + l, 0.08)),
                (format!("{p}.mlp.gate_proj.weight"), vec![inter, h], mk_bf16(inter * h, 43 + l, 0.08)),
                (format!("{p}.mlp.up_proj.weight"), vec![inter, h], mk_bf16(inter * h, 47 + l, 0.08)),
                (format!("{p}.mlp.down_proj.weight"), vec![h, inter], mk_bf16(h * inter, 53 + l, 0.055)),
            ];
            for (name, shape, raw) in specs {
                wm.push(format!(r#""{name}":"{sname}""#));
                dst.push((name, shape, raw));
            }
        }
        wm.push(r#""model.norm.weight":"s2.safetensors""#.to_string());
        s2.push(("model.norm.weight".to_string(), vec![h], mk_bf16(h, 59, 1.0)));
        wm.push(r#""lm_head.weight":"s2.safetensors""#.to_string());
        s2.push(("lm_head.weight".to_string(), vec![vocab, h], mk_bf16(vocab * h, 61, 0.3)));

        write_shard(&dir.join("s1.safetensors"), s1);
        write_shard(&dir.join("s2.safetensors"), s2);
        fs::write(
            dir.join("model.safetensors.index.json"),
            format!(r#"{{"weight_map":{{{}}}}}"#, wm.join(",")),
        )
        .unwrap();

        let mode = RealWeightMode::KronosNfPlanesGptq {
            qkv: 4,
            o: 5,
            gate_up: 4,
            down: 5,
            head: 4,
            scale_group_chunks: 2,
        };
        let calib = vec![vec![1usize, 5, 9, 13, 17, 21, 25, 29], vec![2, 6, 10, 14, 18, 22, 26, 30]];

        // Streaming GPTQ must match full FP32-loaded sequential GPTQ.
        let fp32_ref = RealLlamaEngine::load_fp32_reference(&dir, 32).unwrap();
        let seq_eng = fp32_ref.compile_sequential_bipolar_gptq(mode, &calib, 2);
        let stream_eng = RealLlamaEngine::compile_streaming_gptq_from_dir(&dir, mode, &calib, 2).unwrap();
        assert_eq!(stream_eng.embed_bf16_rows.len(), vocab * h);
        assert!(stream_eng.embed_f32_rows.is_empty());

        let cache_path = dir.join("qwen2_test.k2tb");
        stream_eng.save_bipolar(&cache_path).unwrap();
        let mut loaded = RealLlamaEngine::load_bipolar(&dir, &cache_path, 32).unwrap();

        // Compare non-megakernel FP32-KV against repacked Q8-KV megakernel and forward_prompt.
        let prompt = [3usize, 11, 19, 27, 35];
        let mut ref_eng = seq_eng;
        let mut logits_seq = Vec::new();
        for &t in &prompt {
            logits_seq = ref_eng.forward_token(t).to_vec();
        }

        // Un-repacked loaded engine (FP32-KV non-megakernel)
        let mut logits_unrepacked = Vec::new();
        for &t in &prompt {
            logits_unrepacked = loaded.forward_token(t).to_vec();
        }
        assert!(
            logit_cosine_f64(&logits_seq, &logits_unrepacked) > 0.999999,
            "streaming GPTQ + K2TKRONOS3 cache mismatch vs sequential GPTQ"
        );

        // Verify 1-thread vs 4-thread megakernel determinism and Q8-KV head_dim=128 accuracy.
        loaded.repack_sdot();
        assert!(loaded.can_run_sdot_megakernel());

        loaded.reset_context();
        loaded.set_num_threads(1);
        let mut logits_mega_fp32_1t = Vec::new();
        for &t in &prompt {
            logits_mega_fp32_1t = loaded.forward_token(t).to_vec();
        }
        loaded.reset_context();
        loaded.set_num_threads(4);
        let mut logits_mega_fp32_4t = Vec::new();
        for &t in &prompt {
            logits_mega_fp32_4t = loaded.forward_token(t).to_vec();
        }
        let cos_1t_4t = logit_cosine_f64(&logits_mega_fp32_1t, &logits_mega_fp32_4t);
        assert!(
            cos_1t_4t > 0.999999,
            "head_dim=128 FP32-KV 1t vs 4t megakernel mismatch: {cos_1t_4t}"
        );
        let cos_fp32_vs_ref = logit_cosine_f64(&logits_seq, &logits_mega_fp32_4t);
        assert!(
            cos_fp32_vs_ref > 0.9998,
            "head_dim=128 FP32-KV megakernel vs FP32-act reference cosine too low: {cos_fp32_vs_ref}"
        );

        loaded.enable_q8_kv_cache(true);
        loaded.reset_context();
        let mut logits_mega_q8 = Vec::new();
        for &t in &prompt {
            logits_mega_q8 = loaded.forward_token(t).to_vec();
        }
        let cos_q8_vs_fp32kv = logit_cosine_f64(&logits_mega_fp32_4t, &logits_mega_q8);
        assert!(
            cos_q8_vs_fp32kv > 0.99975,
            "head_dim=128 Q8-KV vs FP32-KV megakernel cosine too low: {cos_q8_vs_fp32kv}"
        );
        let cos_mega = logit_cosine_f64(&logits_seq, &logits_mega_q8);
        assert!(
            cos_mega > 0.99975,
            "head_dim=128 Q8-KV 4-thread megakernel cosine too low: {cos_mega}"
        );

        // Batched forward_prompt with Q8-KV + qkv_bias + head_dim=128 must match token-by-token megakernel.
        loaded.reset_context();
        let logits_batch_q8 = loaded.forward_prompt(&prompt).to_vec();
        let cos_batch = logit_cosine_f64(&logits_mega_q8, &logits_batch_q8);
        assert!(
            cos_batch > 0.99995,
            "head_dim=128 Q8-KV forward_prompt vs megakernel cosine too low: {cos_batch}"
        );

        loaded.set_num_threads(1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_phase2_nf4_untied_embeddings_k2tkronos4_and_q4_sliding_kv_parity() {
        let dir = std::env::temp_dir().join(format!("kronos_phase2_cpu_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let vocab = 192usize;
        let h = 256usize;
        let inter = 512usize;
        let nl = 2usize;
        let nh = 2usize;
        let nkv = 1usize;
        let kvd = 128usize;

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"vocab_size":{vocab},"hidden_size":{h},"intermediate_size":{inter},"num_hidden_layers":{nl},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":false}}"#
            ),
        )
        .unwrap();

        let mk_bf16 = |n: usize, seed: usize, scale: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let mut z = (i as u64).wrapping_add((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
                let v = (2.0 * u - 1.0) * scale;
                out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
            out
        };
        let mut data = Vec::new();
        let mut hdr_items = Vec::new();
        let mut push_t = |name: &str, shape: &[usize], raw: Vec<u8>| {
            let s = data.len();
            data.extend_from_slice(&raw);
            let e = data.len();
            let sh_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            hdr_items.push(format!(
                r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{s},{e}]}}"#,
                sh_str.join(",")
            ));
        };
        push_t("model.embed_tokens.weight", &[vocab, h], mk_bf16(vocab * h, 31, 0.4));
        for l in 0..nl {
            let p = format!("model.layers.{l}");
            push_t(&format!("{p}.input_layernorm.weight"), &[h], mk_bf16(h, 7 + l, 1.0));
            push_t(&format!("{p}.post_attention_layernorm.weight"), &[h], mk_bf16(h, 11 + l, 1.0));
            push_t(&format!("{p}.self_attn.q_proj.weight"), &[h, h], mk_bf16(h * h, 13 + l, 0.08));
            push_t(&format!("{p}.self_attn.k_proj.weight"), &[kvd, h], mk_bf16(kvd * h, 17 + l, 0.08));
            push_t(&format!("{p}.self_attn.v_proj.weight"), &[kvd, h], mk_bf16(kvd * h, 19 + l, 0.08));
            push_t(&format!("{p}.self_attn.o_proj.weight"), &[h, h], mk_bf16(h * h, 41 + l, 0.08));
            push_t(&format!("{p}.mlp.gate_proj.weight"), &[inter, h], mk_bf16(inter * h, 43 + l, 0.08));
            push_t(&format!("{p}.mlp.up_proj.weight"), &[inter, h], mk_bf16(inter * h, 47 + l, 0.08));
            push_t(&format!("{p}.mlp.down_proj.weight"), &[h, inter], mk_bf16(h * inter, 53 + l, 0.055));
        }
        push_t("model.norm.weight", &[h], mk_bf16(h, 59, 1.0));
        push_t("lm_head.weight", &[vocab, h], mk_bf16(vocab * h, 61, 0.3));
        let hdr = format!("{{{}}}", hdr_items.join(","));
        let mut st_bytes = (hdr.len() as u64).to_le_bytes().to_vec();
        st_bytes.extend_from_slice(hdr.as_bytes());
        st_bytes.extend_from_slice(&data);
        fs::write(dir.join("model.safetensors"), st_bytes).unwrap();

        let mode = RealWeightMode::KronosNfGptq {
            bits: 4,
            scale_group_chunks: 2,
        };
        let calib = vec![(0..16).map(|i| (i * 5 + 1) % vocab).collect::<Vec<_>>()];
        let mut eng = RealLlamaEngine::compile_streaming_gptq_from_dir(&dir, mode, &calib, 2).unwrap();
        let mut bf16_row = vec![0.0f32; h];
        eng.embed(17, &mut bf16_row);
        let before_bytes = eng.resident_weight_bytes();
        eng.compress_untied_embeddings_nf4();
        let after_bytes = eng.resident_weight_bytes();
        let saved_emb_bytes = before_bytes.saturating_sub(after_bytes);
        assert!(eng.has_nf4_untied_embeddings());
        assert!(
            saved_emb_bytes > (vocab * h * 2 * 7) / 10 && after_bytes < before_bytes,
            "expected >70% untied embedding RAM reduction, saved={saved_emb_bytes}"
        );
        let mut nf4_row = vec![0.0f32; h];
        eng.embed(17, &mut nf4_row);
        assert!(
            logit_cosine_f64(&bf16_row, &nf4_row) > 0.992,
            "NF4 untied embedding row cosine too low"
        );

        let v4_path = dir.join("kronos-v4.k2tb");
        eng.save_bipolar(&v4_path).unwrap();
        let mut q8_eng = RealLlamaEngine::load_bipolar(&dir, &v4_path, 1024).unwrap();
        assert!(q8_eng.has_nf4_untied_embeddings());
        q8_eng.repack_sdot();
        q8_eng.set_kv_bits(8);
        q8_eng.set_num_threads(2);

        let mut q4_eng = RealLlamaEngine::load_bipolar(&dir, &v4_path, 1024).unwrap();
        q4_eng.repack_sdot();
        q4_eng.set_kv_bits(4);
        q4_eng.set_num_threads(2);

        let prompt: Vec<usize> = (0..24).map(|i| (i * 11 + 3) % vocab).collect();
        let l_q8 = q8_eng.forward_prompt(&prompt).to_vec();
        let l_q4 = q4_eng.forward_prompt(&prompt).to_vec();
        assert!(
            q4_eng.kv_cache_bytes() * 100 < q8_eng.kv_cache_bytes() * 55,
            "Q4-KV bytes ({}) should be <55% of Q8-KV bytes ({})",
            q4_eng.kv_cache_bytes(),
            q8_eng.kv_cache_bytes()
        );
        assert!(
            logit_cosine_f64(&l_q8, &l_q4) > 0.998,
            "CPU Q4-KV vs Q8-KV prefill cosine too low"
        );

        let snap = q4_eng.save_kv_prefix(12).unwrap();
        let _ = q4_eng.forward_token(7);
        q4_eng.restore_kv_prefix(&snap, 12);
        let l_restored = q4_eng.forward_prompt(&prompt[12..]).to_vec();
        assert!(
            logit_cosine_f64(&l_q4, &l_restored) > 0.99999,
            "Q4-KV snapshot restore mismatch"
        );

        // Sliding-window + sink ring wraparound (`640 > 528` ring capacity): batch prefill must match decode.
        let mut win_batch = RealLlamaEngine::load_bipolar(&dir, &v4_path, 1024).unwrap();
        win_batch.repack_sdot();
        win_batch.set_kv_bits(4);
        win_batch.set_kv_sliding_window(Some(16), 4);
        win_batch.set_num_threads(2);

        let mut win_step = RealLlamaEngine::load_bipolar(&dir, &v4_path, 1024).unwrap();
        win_step.repack_sdot();
        win_step.set_kv_bits(4);
        win_step.set_kv_sliding_window(Some(16), 4);
        win_step.set_num_threads(2);

        let long_seq: Vec<usize> = (0..640).map(|i| (i * 13 + 5) % vocab).collect();
        let lb = win_batch.forward_prompt(&long_seq).to_vec();
        let mut ls = Vec::new();
        for &t in &long_seq {
            ls = win_step.forward_token(t).to_vec();
        }
        let cos_win = logit_cosine_f64(&lb, &ls);
        assert!(
            cos_win > 0.9999,
            "CPU Q4 sliding-window ring wraparound batch vs step mismatch: {cos_win}"
        );
        assert!(win_batch.can_truncate_kv_to(0));
        assert!(!win_batch.can_truncate_kv_to(8));
        assert!(win_batch.can_truncate_kv_to(638));

        q8_eng.set_num_threads(1);
        let _ = fs::remove_dir_all(&dir);
    }
}
