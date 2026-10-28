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
use crate::safetensors::{JsonCursor, SafeTensorsStore};
use crate::simd_lut::{
    bf16_to_f32, nf_sdot_batch_stripe_range, nf_sdot_stripe_range, nf_sdot_stripe_range_int8,
    nf_sdot_stripe_range_raw, nibble4_sdot_batch_stripe_range, nibble4_sdot_stripe_range,
    nibble4_sdot_stripe_range_int8, nibble4_sdot_stripe_range_raw, simd_lut_base3_gemv,
    simd_lut_base3_stripe_range, simd_lut_bipolar_gemv, simd_lut_bipolar_stripe_range,
    simd_lut_gemv_multiplane, vq2_lut_gemv, vq2_lut_stripe_range, vq2_lut_stripe_range_raw,
    vq2_sdot_batch_stripe_range, BipolarActivationLut, FusedBase3LutMatrix, FusedBipolarLutMatrix,
    GgufQ4_0Matrix, Int8ChunkActivation, MultiPlaneTernaryMatrix, NfSdotMatrix, Nibble4SdotMatrix,
    QuantizedActivationLut, Vq2ActivationLut, Vq2LutMatrix, SIMD_ROW_STRIPE,
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

    fn u16s(&mut self) -> io::Result<Vec<u16>> {
        let n = self.u64()?;
        Ok(self.take(n * 2)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
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
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
    pub max_seq_len: usize,
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
        let mut rms_norm_eps = 1e-5f32;
        let mut rope_theta = 100_000.0f32;
        let mut tie_word_embeddings = true;

        while cur.peek() != Some(b'}') {
            let key = cur
                .parse_string()
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
            cur.expect_byte(b':')
                .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
            match key.as_str() {
                "vocab_size" => vocab_size = cur.parse_u64().unwrap() as usize,
                "hidden_size" => hidden_size = cur.parse_u64().unwrap() as usize,
                "intermediate_size" => intermediate_size = cur.parse_u64().unwrap() as usize,
                "num_hidden_layers" => num_hidden_layers = cur.parse_u64().unwrap() as usize,
                "num_attention_heads" => num_attention_heads = cur.parse_u64().unwrap() as usize,
                "num_key_value_heads" => num_key_value_heads = cur.parse_u64().unwrap() as usize,
                "rms_norm_eps" => rms_norm_eps = cur.parse_f64().unwrap() as f32,
                "rope_theta" => rope_theta = cur.parse_f64().unwrap() as f32,
                "tie_word_embeddings" => {
                    if cur.peek() == Some(b't') {
                        cur.consume_literal(b"true").unwrap();
                        tie_word_embeddings = true;
                    } else {
                        cur.consume_literal(b"false").unwrap();
                        tie_word_embeddings = false;
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

        Ok(Self {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            max_seq_len,
        })
    }

    #[inline]
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim()
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
    /// Fused `[q_proj; k_proj; v_proj]` (`[hidden_size + 2 * kv_dim, hidden_size]`).
    qkv_proj: LinearBackend,
    o_proj: LinearBackend,
    /// Fused `[gate_proj; up_proj]` (`[2 * intermediate_size, hidden_size]`).
    gate_up_proj: LinearBackend,
    /// `gate_up_proj` rows interleaved in 64-row blocks `[gate_64(ic), up_64(ic)]`.
    gate_up_interleaved: bool,
    down_proj: LinearBackend,
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
    acts_i: Vec<BipolarActivationLut>,
    qkv: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate_up: Vec<f32>,
    sw: Vec<f32>,
    scores: Vec<f32>,
}

impl PromptScratch {
    fn ensure_capacity(
        &mut self,
        n: usize,
        cfg: &LlamaConfig,
        qkv_rows: usize,
        lut_h: bool,
        lut_i: bool,
    ) {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
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
        if self.attn.len() < n * h {
            self.attn.resize(n * h, 0.0);
        }
        if self.proj.len() < n * h {
            self.proj.resize(n * h, 0.0);
        }
        if self.gate_up.len() < n * 2 * inter {
            self.gate_up.resize(n * 2 * inter, 0.0);
        }
        if self.sw.len() < n * inter {
            self.sw.resize(n * inter, 0.0);
        }
        if self.scores.len() < n * score_stride {
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

/// KV-head-major INT8 KV cache (`head_dim == 64` or `128`) using signed FWHT-64 + per-head FP32 scale.
struct QuantizedKvCache {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    k_q8: Vec<i8>,
    v_q8: Vec<i8>,
    k_scales: Vec<f32>,
    v_scales: Vec<f32>,
}

impl QuantizedKvCache {
    fn new(cfg: &LlamaConfig) -> Self {
        let head_dim = cfg.head_dim();
        assert!(
            head_dim == 64 || head_dim == 128,
            "Hadamard Q8 KV-cache requires head_dim == 64 or 128 (got {head_dim})"
        );
        let slots = cfg.num_hidden_layers * cfg.num_key_value_heads * cfg.max_seq_len;
        Self {
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            max_seq_len: cfg.max_seq_len,
            k_q8: vec![0i8; slots * head_dim],
            v_q8: vec![0i8; slots * head_dim],
            k_scales: vec![0.0f32; slots],
            v_scales: vec![0.0f32; slots],
        }
    }

    #[inline]
    fn physical_bytes(&self) -> usize {
        (self.k_q8.len() + self.v_q8.len()) * std::mem::size_of::<i8>()
            + (self.k_scales.len() + self.v_scales.len()) * std::mem::size_of::<f32>()
    }

    #[inline]
    fn head_slot_base(&self, layer_idx: usize, kvh: usize) -> usize {
        debug_assert!(layer_idx < self.num_layers && kvh < self.num_kv_heads);
        (layer_idx * self.num_kv_heads + kvh) * self.max_seq_len
    }

    #[inline]
    fn store_k_head(&mut self, layer_idx: usize, kvh: usize, pos: usize, k_head: &[f32]) {
        let slot = self.head_slot_base(layer_idx, kvh) + pos;
        let hd = self.head_dim;
        // SAFETY: `slot < k_scales.len()` and `slot * hd + hd <= k_q8.len()`.
        unsafe {
            store_hadamard_q8_head_raw(
                k_head,
                hd,
                self.k_q8.as_mut_ptr().add(slot * hd),
                self.k_scales.as_mut_ptr().add(slot),
            );
        }
    }

    #[inline]
    fn store_v_head(&mut self, layer_idx: usize, kvh: usize, pos: usize, v_head: &[f32]) {
        let slot = self.head_slot_base(layer_idx, kvh) + pos;
        let hd = self.head_dim;
        // SAFETY: `slot < v_scales.len()` and `slot * hd + hd <= v_q8.len()`.
        unsafe {
            store_hadamard_q8_head_raw(
                v_head,
                hd,
                self.v_q8.as_mut_ptr().add(slot * hd),
                self.v_scales.as_mut_ptr().add(slot),
            );
        }
    }

    #[inline]
    fn head_views(&self, layer_idx: usize, kvh: usize, seq_len: usize) -> (&[i8], &[f32], &[i8], &[f32]) {
        let base = self.head_slot_base(layer_idx, kvh);
        let hd = self.head_dim;
        (
            &self.k_q8[base * hd..(base + seq_len) * hd],
            &self.k_scales[base..base + seq_len],
            &self.v_q8[base * hd..(base + seq_len) * hd],
            &self.v_scales[base..base + seq_len],
        )
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
    qlut_inter: QuantizedActivationLut,
    blut_hidden: BipolarActivationLut,
    blut_inter: BipolarActivationLut,
    vqlut_hidden: Vq2ActivationLut,
    vqlut_inter: Vq2ActivationLut,
    q8_hidden_vals: Vec<i8>,
    q8_hidden_scales: Vec<f32>,
    q8_inter_vals: Vec<i8>,
    q8_inter_scales: Vec<f32>,
    prompt_scratch: PromptScratch,
}

impl LlamaWorkspace {
    fn new(cfg: &LlamaConfig) -> Self {
        let qkv_rows = cfg.hidden_size + 2 * cfg.kv_dim();
        Self {
            pos: 0,
            // Allocated lazily on first FP32-KV forward so Q8-KV models never allocate a multi-GiB throwaway FP32 cache.
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            q8_kv: None,
            hidden: vec![0.0f32; cfg.hidden_size],
            norm_buf: vec![0.0f32; cfg.hidden_size],
            qkv_buf: vec![0.0f32; qkv_rows],
            attn_out: vec![0.0f32; cfg.hidden_size],
            proj_out: vec![0.0f32; cfg.hidden_size],
            gate_up_buf: vec![0.0f32; 2 * cfg.intermediate_size],
            swiglu_buf: vec![0.0f32; cfg.intermediate_size],
            scores: vec![0.0f32; cfg.max_seq_len * cfg.num_attention_heads],
            logits: vec![0.0f32; cfg.vocab_size],
            qlut_hidden: QuantizedActivationLut::new_zeroed(cfg.hidden_size),
            qlut_inter: QuantizedActivationLut::new_zeroed(cfg.intermediate_size),
            blut_hidden: BipolarActivationLut::new_zeroed(cfg.hidden_size),
            blut_inter: BipolarActivationLut::new_zeroed(cfg.intermediate_size),
            vqlut_hidden: Vq2ActivationLut::new_zeroed(cfg.hidden_size),
            vqlut_inter: Vq2ActivationLut::new_zeroed(cfg.intermediate_size),
            q8_hidden_vals: vec![0i8; cfg.hidden_size],
            q8_hidden_scales: vec![0.0f32; cfg.hidden_size / 32],
            q8_inter_vals: vec![0i8; cfg.intermediate_size],
            q8_inter_scales: vec![0.0f32; cfg.intermediate_size / 32],
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

/// Real pretrained `LlamaForCausalLM` / `Qwen2ForCausalLM` engine.
pub struct RealLlamaEngine {
    pub cfg: LlamaConfig,
    pub mode: RealWeightMode,
    pub mean_weight_rel_error: f32,
    embed_f32_rows: Vec<f32>,
    embed_bf16_rows: Vec<u16>,
    layers: Vec<LlamaLayer>,
    final_norm: Vec<f32>,
    lm_head: LinearBackend,
    /// Precomputed `[max_seq_len, half_dim * 2]` (`[cos_0..cos_{half-1}, sin_0..sin_{half-1}]`).
    rope_cos_sin: Vec<f32>,
    pool: Option<SpinThreadPool>,
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

    /// Toggles the Hadamard INT8 KV cache (`head_dim == 64` or `128`). Frees the FP32 cache and resets `pos`.
    pub fn enable_q8_kv_cache(&mut self, enable: bool) {
        self.ws.pos = 0;
        if enable {
            let hd = self.cfg.head_dim();
            assert!(
                hd == 64 || hd == 128,
                "Hadamard Q8 KV-cache requires head_dim == 64 or 128 (got {hd})"
            );
            self.ws.k_cache = Vec::new();
            self.ws.v_cache = Vec::new();
            if self.ws.q8_kv.is_none() {
                self.ws.q8_kv = Some(QuantizedKvCache::new(&self.cfg));
            }
        } else {
            self.ws.q8_kv = None;
            self.ws.ensure_fp32_kv(&self.cfg);
        }
    }

    /// Whether the INT8 KV cache is active.
    #[inline]
    pub fn is_q8_kv_enabled(&self) -> bool {
        self.ws.q8_kv.is_some()
    }

    /// Bytes allocated for the active KV cache.
    pub fn kv_cache_bytes(&self) -> usize {
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

    /// Loads the exact `FP32` reference model from `model_dir` (single or sharded SafeTensors).
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
        let qkv_rows = cfg.hidden_size + 2 * cfg.kv_dim();

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

            let mut gate_up_w = Vec::with_capacity(2 * cfg.intermediate_size * cfg.hidden_size);
            gate_up_w.extend_from_slice(&get_f32(&format!("{pfx}.mlp.gate_proj.weight"))?);
            gate_up_w.extend_from_slice(&get_f32(&format!("{pfx}.mlp.up_proj.weight"))?);

            layers.push(LlamaLayer {
                input_norm: get_f32(&format!("{pfx}.input_layernorm.weight"))?,
                post_attn_norm: get_f32(&format!("{pfx}.post_attention_layernorm.weight"))?,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_bias,
                qkv_proj: LinearBackend::Fp32 {
                    rows: qkv_rows,
                    cols: cfg.hidden_size,
                    weights: qkv_w,
                },
                o_proj: LinearBackend::Fp32 {
                    rows: cfg.hidden_size,
                    cols: cfg.hidden_size,
                    weights: get_f32(&format!("{pfx}.self_attn.o_proj.weight"))?,
                },
                gate_up_proj: LinearBackend::Fp32 {
                    rows: 2 * cfg.intermediate_size,
                    cols: cfg.hidden_size,
                    weights: gate_up_w,
                },
                gate_up_interleaved: false,
                down_proj: LinearBackend::Fp32 {
                    rows: cfg.hidden_size,
                    cols: cfg.intermediate_size,
                    weights: get_f32(&format!("{pfx}.mlp.down_proj.weight"))?,
                },
            });
        }

        let final_norm = get_f32("model.norm.weight")?;
        let lm_head_weights = if cfg.tie_word_embeddings {
            embed_f32_rows.clone()
        } else {
            get_f32("lm_head.weight")?
        };

        let rope_cos_sin = build_rope_table(&cfg);
        let ws = LlamaWorkspace::new(&cfg);
        Ok(Self {
            cfg: cfg.clone(),
            mode: RealWeightMode::ExactFp32,
            mean_weight_rel_error: 0.0,
            embed_f32_rows,
            embed_bf16_rows: Vec::new(),
            layers,
            final_norm,
            lm_head: LinearBackend::Fp32 {
                rows: cfg.vocab_size,
                cols: cfg.hidden_size,
                weights: lm_head_weights,
            },
            rope_cos_sin,
            pool: None,
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
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
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
            cfg: self.cfg.clone(),
            mode,
            mean_weight_rel_error: (err_sum / (mat_count as f64)) as f32,
            embed_f32_rows: self.embed_f32_rows.clone(),
            embed_bf16_rows: self.embed_bf16_rows.clone(),
            layers: compiled_layers,
            final_norm,
            lm_head,
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
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
                qkv_proj: LinearBackend::FusedBase3(qkv_mat),
                o_proj: LinearBackend::FusedBase3(o_mat),
                gate_up_proj: LinearBackend::FusedBase3(gate_up_mat),
                gate_up_interleaved: false,
                down_proj: LinearBackend::FusedBase3(down_mat),
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
            cfg: self.cfg.clone(),
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows: self.embed_f32_rows.clone(),
            embed_bf16_rows: self.embed_bf16_rows.clone(),
            layers: compiled_layers,
            final_norm: self.final_norm.clone(),
            lm_head: LinearBackend::FusedBase3(head_mat),
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
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
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
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
            layers: compiled_layers,
            final_norm: self.final_norm.clone(),
            lm_head,
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
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
        if cfg.tie_word_embeddings {
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
                qkv_proj,
                o_proj,
                gate_up_proj,
                gate_up_interleaved: false,
                down_proj,
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
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows: Vec::new(),
            embed_bf16_rows,
            layers: compiled_layers,
            final_norm,
            lm_head,
            rope_cos_sin,
            pool: None,
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
        let hidden_lut = is_lut(&self.lm_head)
            || self
                .layers
                .iter()
                .any(|l| is_lut(&l.qkv_proj) || is_lut(&l.o_proj) || is_lut(&l.gate_up_proj));
        let inter_lut = self.layers.iter().any(|l| is_lut(&l.down_proj));
        let any_int8 = is_int8(&self.lm_head)
            || self.layers.iter().any(|l| {
                is_int8(&l.qkv_proj)
                    || is_int8(&l.o_proj)
                    || is_int8(&l.gate_up_proj)
                    || is_int8(&l.down_proj)
            });
        self.ws.blut_hidden.lut_enabled = hidden_lut;
        self.ws.blut_inter.lut_enabled = inter_lut;
        if any_int8 {
            if self.ws.blut_hidden.int8.is_none() {
                self.ws.blut_hidden.int8 = Some(Int8ChunkActivation::new_zeroed(self.cfg.hidden_size));
            }
            if self.ws.blut_inter.int8.is_none() {
                self.ws.blut_inter.int8 =
                    Some(Int8ChunkActivation::new_zeroed(self.cfg.intermediate_size));
            }
        }
    }

    /// Repacks projections for `SDOT`, interleaves `gate_up_proj` in 64-row blocks and enables INT8
    /// chunk activations.
    pub fn repack_sdot(&mut self) {
        let repack = |b: &mut LinearBackend| {
            if let LinearBackend::FusedBipolar(m) = b {
                if (4..=6).contains(&m.num_planes) {
                    *b = LinearBackend::Nibble4(Nibble4SdotMatrix::from_bipolar(m));
                }
            }
        };
        let can_interleave_gu = self.cfg.intermediate_size.is_multiple_of(64);
        for l in &mut self.layers {
            repack(&mut l.qkv_proj);
            repack(&mut l.o_proj);
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
                LinearBackend::NfSdot(m) => {
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
                _ => Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "save_bipolar expects un-repacked FusedBipolar, NfSdot, or Vq2 projections",
                )),
            }
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

        let needs_v3 = self.layers.iter().any(|l| l.qkv_bias.is_some())
            || !self.embed_bf16_rows.is_empty();
        let version = if needs_v3 {
            3u8
        } else if needs_v2_mats {
            2u8
        } else {
            1u8
        };

        let mut w = io::BufWriter::new(fs::File::create(path)?);
        w.write_all(match version {
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
        let (version, magic_len) = if bytes.starts_with(KRONOS3_CACHE_MAGIC) {
            (3u8, KRONOS3_CACHE_MAGIC.len())
        } else if bytes.starts_with(KRONOS2_CACHE_MAGIC) {
            (2u8, KRONOS2_CACHE_MAGIC.len())
        } else if bytes.starts_with(BIPOLAR_CACHE_MAGIC) {
            (1u8, BIPOLAR_CACHE_MAGIC.len())
        } else {
            return Err(bad("bad bipolar/kronos2/kronos3 cache magic"));
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
                    let pair_stream = rd.take(pair_len)?.to_vec();
                    let tail_len = rd.u64()?;
                    let tail_stream = rd.take(tail_len)?.to_vec();
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
                qkv_proj: r_mat(&mut rd)?,
                o_proj: r_mat(&mut rd)?,
                gate_up_proj: r_mat(&mut rd)?,
                gate_up_interleaved: false,
                down_proj: r_mat(&mut rd)?,
            });
        }
        let final_norm = rd.f32s()?;
        let lm_head = r_mat(&mut rd)?;
        let embed_f32_rows = rd.f32s()?;
        let embed_bf16_rows = if version >= 3 {
            rd.u16s()?
        } else {
            Vec::new()
        };
        let page_aligned = len & !16_383usize;
        if page_aligned > 0 {
            mmap.advise_dontneed(0, page_aligned);
        }
        drop(mmap);

        let mut eng = Self {
            rope_cos_sin: build_rope_table(&cfg),
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows,
            embed_bf16_rows,
            layers,
            final_norm,
            lm_head,
            pool: None,
        };
        eng.init_activation_scratch();
        Ok(eng)
    }

    /// Total physical RAM occupied by all compiled linear projections (`layers` + `lm_head`).
    pub fn linear_weight_bytes(&self) -> usize {
        let layer_bytes: usize = self
            .layers
            .iter()
            .map(|l| {
                l.qkv_proj.physical_bytes()
                    + l.o_proj.physical_bytes()
                    + l.gate_up_proj.physical_bytes()
                    + l.down_proj.physical_bytes()
            })
            .sum();
        layer_bytes + self.lm_head.physical_bytes()
    }

    /// Linear weights plus biases and any untied (`FP32` or `BF16`) embedding table.
    pub fn resident_weight_bytes(&self) -> usize {
        let bias_bytes: usize = self
            .layers
            .iter()
            .map(|l| l.qkv_bias.as_ref().map_or(0, |b| b.len() * std::mem::size_of::<f32>()))
            .sum();
        self.linear_weight_bytes()
            + bias_bytes
            + self.embed_f32_rows.len() * std::mem::size_of::<f32>()
            + self.embed_bf16_rows.len() * std::mem::size_of::<u16>()
    }

    /// Physical bits per weight across all transformer layers + `lm_head`.
    pub fn physical_bits_per_weight(&self) -> f64 {
        let qkv_rows = self.cfg.hidden_size + 2 * self.cfg.kv_dim();
        let layer_params = qkv_rows * self.cfg.hidden_size
            + self.cfg.hidden_size * self.cfg.hidden_size
            + 2 * self.cfg.intermediate_size * self.cfg.hidden_size
            + self.cfg.hidden_size * self.cfg.intermediate_size;
        let total_params =
            self.cfg.num_hidden_layers * layer_params + self.cfg.vocab_size * self.cfg.hidden_size;
        (self.linear_weight_bytes() as f64 * 8.0) / (total_params as f64)
    }

    /// Drops KV-cache entries after the first `n` tokens.
    pub fn truncate_context(&mut self, n: usize) {
        self.ws.pos = self.ws.pos.min(n);
    }

    /// Embedding row of `token` (decoded from `BF16`/`FP32` untied table, or quantized `lm_head` when tied).
    pub fn embed(&self, token: usize, out: &mut [f32]) {
        embed_token_into(
            &self.embed_f32_rows,
            &self.embed_bf16_rows,
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
        if let Some(q8) = self.ws.q8_kv.as_mut() {
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
        self.ws.blut_hidden.int8.is_some()
            && self.ws.blut_inter.int8.is_some()
            && is_sdot(&self.lm_head)
            && self.layers.iter().all(|l| {
                is_sdot(&l.qkv_proj)
                    && is_sdot(&l.o_proj)
                    && is_sdot(&l.gate_up_proj)
                    && is_sdot(&l.down_proj)
                    && l.o_inv_scale.is_none()
                    && l.down_inv_scale.is_none()
            })
    }

    #[inline]
    fn can_run_sdot_megakernel(&self) -> bool {
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
        if self.can_run_sdot_megakernel() {
            return self.forward_token_sdot_megakernel(token_id);
        }
        if self.ws.q8_kv.is_none() {
            self.ws.ensure_fp32_kv(&self.cfg);
        }
        let pos = self.ws.pos;
        let h_dim = self.cfg.hidden_size;
        let inter_dim = self.cfg.intermediate_size;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_dim = self.cfg.kv_dim();
        let num_kv_heads = self.cfg.num_key_value_heads;
        let kv_group = self.cfg.num_attention_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let layer_kv_stride = self.cfg.max_seq_len * kv_dim;

        embed_token_into(
            &self.embed_f32_rows,
            &self.embed_bf16_rows,
            &self.lm_head,
            h_dim,
            token_id,
            &mut self.ws.hidden,
        );
        let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
        let pool_ref = self.pool.as_ref();

        macro_rules! exec_hidden_proj {
            ($proj:expr, $buf:expr, $out:expr) => {{
                match &$proj {
                    LinearBackend::Vq2(_) => {
                        fwht64_signed_inplace($buf);
                        self.ws.vqlut_hidden.populate($buf);
                    }
                    LinearBackend::FusedBipolar(_)
                    | LinearBackend::Nibble4(_)
                    | LinearBackend::NfSdot(_) => {
                        fwht64_signed_inplace($buf);
                        self.ws.blut_hidden.populate($buf);
                    }
                    LinearBackend::FusedBase3(_) => {
                        fwht64_signed_inplace($buf);
                        self.ws.qlut_hidden.populate($buf);
                    }
                    LinearBackend::LutMultiPlane(_) => {
                        self.ws.qlut_hidden.populate($buf);
                    }
                    LinearBackend::GgufQ4_0(_) => {
                        GgufQ4_0Matrix::quantize_q8_0_activation(
                            $buf,
                            &mut self.ws.q8_hidden_vals,
                            &mut self.ws.q8_hidden_scales,
                        );
                    }
                    LinearBackend::Fp32 { .. } => {}
                }
                $proj.forward_exec(
                    $buf,
                    Some(&self.ws.qlut_hidden),
                    Some(&self.ws.blut_hidden),
                    Some(&self.ws.vqlut_hidden),
                    &self.ws.q8_hidden_vals,
                    &self.ws.q8_hidden_scales,
                    pool_ref,
                    $out,
                );
            }};
        }

        for (l_idx, layer) in self.layers.iter().enumerate() {
            // 1. Pre-Attention RMSNorm + Fused QKV Projection (+ optional Qwen2 QKV Bias)
            rms_norm(
                &self.ws.hidden,
                &layer.input_norm,
                self.cfg.rms_norm_eps,
                &mut self.ws.norm_buf,
            );
            exec_hidden_proj!(layer.qkv_proj, &mut self.ws.norm_buf, &mut self.ws.qkv_buf);
            if let Some(bias) = &layer.qkv_bias {
                add_inplace(&mut self.ws.qkv_buf, bias);
            }

            // 2. Precomputed RoPE + KV Cache Store + Causal GQA Attention
            let (q_slice, kv_rest) = self.ws.qkv_buf.split_at_mut(h_dim);
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
                num_kv_heads,
                head_dim,
                half_dim,
                rope_row,
            );

            if let Some(q8) = self.ws.q8_kv.as_mut() {
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
                    &mut self.ws.scores,
                    &mut self.ws.attn_out,
                    self.cfg.num_attention_heads,
                    kv_group,
                    pos,
                    attn_scale,
                );
            } else {
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
            }

            // 3. Output Projection + Residual
            if let Some(inv_s) = &layer.o_inv_scale {
                for (a, &is) in self.ws.attn_out.iter_mut().zip(inv_s.iter()) {
                    *a *= is;
                }
            }
            exec_hidden_proj!(layer.o_proj, &mut self.ws.attn_out, &mut self.ws.proj_out);
            add_inplace(&mut self.ws.hidden, &self.ws.proj_out);

            // 4. Post-Attention RMSNorm + Fused Gate+Up SwiGLU Projection
            rms_norm(
                &self.ws.hidden,
                &layer.post_attn_norm,
                self.cfg.rms_norm_eps,
                &mut self.ws.norm_buf,
            );
            exec_hidden_proj!(
                layer.gate_up_proj,
                &mut self.ws.norm_buf,
                &mut self.ws.gate_up_buf
            );

            if layer.gate_up_interleaved {
                for ic in 0..(inter_dim / 64) {
                    swiglu_activate(
                        &self.ws.gate_up_buf[ic * 128..ic * 128 + 64],
                        &self.ws.gate_up_buf[ic * 128 + 64..ic * 128 + 128],
                        &mut self.ws.swiglu_buf[ic * 64..(ic + 1) * 64],
                    );
                }
            } else {
                let (gate_s, up_s) = self.ws.gate_up_buf.split_at(inter_dim);
                swiglu_activate(gate_s, up_s, &mut self.ws.swiglu_buf);
            }

            if let Some(inv_s) = &layer.down_inv_scale {
                for (g, &is) in self.ws.swiglu_buf.iter_mut().zip(inv_s.iter()) {
                    *g *= is;
                }
            }
            match &layer.down_proj {
                LinearBackend::Vq2(_) => {
                    fwht64_signed_inplace(&mut self.ws.swiglu_buf);
                    self.ws.vqlut_inter.populate(&self.ws.swiglu_buf);
                }
                LinearBackend::FusedBipolar(_)
                | LinearBackend::Nibble4(_)
                | LinearBackend::NfSdot(_) => {
                    fwht64_signed_inplace(&mut self.ws.swiglu_buf);
                    self.ws.blut_inter.populate(&self.ws.swiglu_buf);
                }
                LinearBackend::FusedBase3(_) => {
                    fwht64_signed_inplace(&mut self.ws.swiglu_buf);
                    self.ws.qlut_inter.populate(&self.ws.swiglu_buf);
                }
                LinearBackend::LutMultiPlane(_) => {
                    self.ws.qlut_inter.populate(&self.ws.swiglu_buf);
                }
                LinearBackend::GgufQ4_0(_) => {
                    GgufQ4_0Matrix::quantize_q8_0_activation(
                        &self.ws.swiglu_buf,
                        &mut self.ws.q8_inter_vals,
                        &mut self.ws.q8_inter_scales,
                    );
                }
                LinearBackend::Fp32 { .. } => {}
            }
            layer.down_proj.forward_exec(
                &self.ws.swiglu_buf,
                Some(&self.ws.qlut_inter),
                Some(&self.ws.blut_inter),
                Some(&self.ws.vqlut_inter),
                &self.ws.q8_inter_vals,
                &self.ws.q8_inter_scales,
                pool_ref,
                &mut self.ws.proj_out,
            );
            add_inplace(&mut self.ws.hidden, &self.ws.proj_out);
        }

        // Final RMSNorm + LM Head
        rms_norm(
            &self.ws.hidden,
            &self.final_norm,
            self.cfg.rms_norm_eps,
            &mut self.ws.norm_buf,
        );
        exec_hidden_proj!(self.lm_head, &mut self.ws.norm_buf, &mut self.ws.logits);

        let _ = SIMD_ROW_STRIPE;
        self.ws.pos += 1;
        &self.ws.logits
    }

    /// Decodes one token for `SDOT`/VQ2 engines (`head_dim == 64` or `128`, `hidden_size <= 8192`)
    /// with all Transformer layers and the LM head fused inside a single thread-pool dispatch.
    fn forward_token_sdot_megakernel(&mut self, token_id: usize) -> &[f32] {
        use std::mem::MaybeUninit;

        if self.ws.q8_kv.is_none() {
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

        let (use_q8_kv, q8_k_addr, q8_ks_addr, q8_v_addr, q8_vs_addr) =
            if let Some(q8) = self.ws.q8_kv.as_mut() {
                (
                    true,
                    q8.k_q8.as_mut_ptr() as usize,
                    q8.k_scales.as_mut_ptr() as usize,
                    q8.v_q8.as_mut_ptr() as usize,
                    q8.v_scales.as_mut_ptr() as usize,
                )
            } else {
                (false, 0, 0, 0, 0)
            };

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
            let q8_k_ptr = q8_k_addr as *mut i8;
            let q8_ks_ptr = q8_ks_addr as *mut f32;
            let q8_v_ptr = q8_v_addr as *mut i8;
            let q8_vs_ptr = q8_vs_addr as *mut f32;
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

            let qh_grain = (num_q_heads / (nt * 2)).clamp(1, 4);
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
                                if use_q8_kv {
                                    let slot = (l_idx * num_kv_heads + kvh) * max_seq_len + pos;
                                    store_hadamard_q8_head_raw(
                                        head_s,
                                        head_dim,
                                        q8_k_ptr.add(slot * head_dim),
                                        q8_ks_ptr.add(slot),
                                    );
                                } else {
                                    std::ptr::copy_nonoverlapping(
                                        head_s.as_ptr(),
                                        k_cache_ptr.add(kv_base + pos * kv_dim + kvh * head_dim),
                                        head_dim,
                                    );
                                }
                            } else {
                                let kvh = b - num_q_heads - num_kv_heads;
                                if use_q8_kv {
                                    let slot = (l_idx * num_kv_heads + kvh) * max_seq_len + pos;
                                    store_hadamard_q8_head_raw(
                                        head_s,
                                        head_dim,
                                        q8_v_ptr.add(slot * head_dim),
                                        q8_vs_ptr.add(slot),
                                    );
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
                    // SAFETY: inputs are read-only; head `qh` owns its `scores` row and Hadamard chunks `[qh*cph, (qh+1)*cph)`.
                    unsafe {
                        let mut out_head = [0.0f32; 128];
                        for qh in qh0..qh1 {
                            let q_head = std::slice::from_raw_parts(qkv_ptr.add(qh * head_dim), head_dim);
                            let sc = std::slice::from_raw_parts_mut(scores_ptr.add(qh * max_seq_len), seq_len);
                            let kvh = qh / kv_group;
                            if use_q8_kv {
                                let base = (l_idx * num_kv_heads + kvh) * max_seq_len;
                                let k_q8_h = std::slice::from_raw_parts(q8_k_ptr.add(base * head_dim), seq_len * head_dim);
                                let ks_h = std::slice::from_raw_parts(q8_ks_ptr.add(base), seq_len);
                                let v_q8_h = std::slice::from_raw_parts(q8_v_ptr.add(base * head_dim), seq_len * head_dim);
                                let vs_h = std::slice::from_raw_parts(q8_vs_ptr.add(base), seq_len);
                                if head_dim == 64 {
                                    causal_gqa_head64_q8(
                                        q_head,
                                        k_q8_h,
                                        ks_h,
                                        v_q8_h,
                                        vs_h,
                                        sc,
                                        (&mut out_head[..64]).try_into().unwrap(),
                                        attn_scale,
                                    );
                                } else {
                                    causal_gqa_head128_q8(
                                        q_head,
                                        k_q8_h,
                                        ks_h,
                                        v_q8_h,
                                        vs_h,
                                        sc,
                                        &mut out_head,
                                        attn_scale,
                                    );
                                }
                            } else {
                                let k_layer = std::slice::from_raw_parts(k_cache_ptr.add(kv_base), seq_len * kv_dim);
                                let v_layer = std::slice::from_raw_parts(v_cache_ptr.add(kv_base), seq_len * kv_dim);
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
        if !self.is_all_sdot() || tokens.len() == 1 {
            for &t in &tokens[..tokens.len() - 1] {
                self.forward_token(t);
            }
            return self.forward_token(tokens[tokens.len() - 1]);
        }
        self.forward_batch_sdot_impl(tokens, None);
        &self.ws.logits
    }

    /// Runs `tokens` (`len = m`) through one batched pass and writes `[m, vocab_size]` logits into
    /// `out_logits` (used for single-DRAM-sweep Lossless Speculative Decoding verification!).
    pub fn forward_batch_logits(&mut self, tokens: &[usize], out_logits: &mut [f32]) {
        assert!(!tokens.is_empty());
        let vocab = self.cfg.vocab_size;
        assert!(out_logits.len() >= tokens.len() * vocab);
        if !self.is_all_sdot() || tokens.len() == 1 {
            for (t, &tok) in tokens.iter().enumerate() {
                let l = self.forward_token(tok);
                out_logits[t * vocab..(t + 1) * vocab].copy_from_slice(l);
            }
            return;
        }
        self.forward_batch_sdot_impl(tokens, Some(&mut out_logits[..tokens.len() * vocab]));
    }

    fn forward_batch_sdot_impl(&mut self, tokens: &[usize], all_logits_out: Option<&mut [f32]>) {
        let n = tokens.len();
        let pos0 = self.ws.pos;
        assert!(pos0 + n <= self.cfg.max_seq_len, "prompt exceeds max_seq_len");
        if self.ws.q8_kv.is_none() {
            self.ws.ensure_fp32_kv(&self.cfg);
        }
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let kv_dim = self.cfg.kv_dim();
        let qkv_rows = h + 2 * kv_dim;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let num_kv_heads = self.cfg.num_key_value_heads;
        let kv_group = self.cfg.num_attention_heads / num_kv_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let max_seq_len = self.cfg.max_seq_len;
        let layer_kv_stride = max_seq_len * kv_dim;
        let eps = self.cfg.rms_norm_eps;
        let lut_h = self.ws.blut_hidden.lut_enabled;
        let lut_i = self.ws.blut_inter.lut_enabled;

        self.ws
            .prompt_scratch
            .ensure_capacity(n, &self.cfg, qkv_rows, lut_h, lut_i);

        let pool = self.pool.as_ref();
        let ps = &mut self.ws.prompt_scratch;
        let hidden = &mut ps.hidden[..n * h];
        for (t, &tok) in tokens.iter().enumerate() {
            embed_token_into(
                &self.embed_f32_rows,
                &self.embed_bf16_rows,
                &self.lm_head,
                h,
                tok,
                &mut hidden[t * h..(t + 1) * h],
            );
        }
        let acts_h = &mut ps.acts_h[..n];
        let acts_i = &mut ps.acts_i[..n];
        let qkv = &mut ps.qkv[..n * qkv_rows];
        let attn = &mut ps.attn[..n * h];
        let proj = &mut ps.proj[..n * h];
        let gate_up = &mut ps.gate_up[..n * 2 * inter];
        let sw = &mut ps.sw[..n * inter];
        let score_stride = if head_dim == 64 {
            max_seq_len
        } else {
            kv_group.max(1) * max_seq_len
        };
        let scores_all = &mut ps.scores[..n * score_stride];

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let kv_base = l_idx * layer_kv_stride;
            prompt_populate(pool, hidden, h, Some((&layer.input_norm, eps)), acts_h);
            prompt_gemm(pool, &layer.qkv_proj, acts_h, qkv);

            let qkv_ptr = qkv.as_mut_ptr() as usize;
            let qkv_bias = layer.qkv_bias.as_deref();
            let rope = &self.rope_cos_sin;
            let cfg = &self.cfg;
            if let Some(q8) = self.ws.q8_kv.as_mut() {
                let q8_k_addr = q8.k_q8.as_mut_ptr() as usize;
                let q8_ks_addr = q8.k_scales.as_mut_ptr() as usize;
                let q8_v_addr = q8.v_q8.as_mut_ptr() as usize;
                let q8_vs_addr = q8.v_scales.as_mut_ptr() as usize;
                par_range(pool, n, |a, b| {
                    for t in a..b {
                        let pos = pos0 + t;
                        // SAFETY: token `t` owns row `t` of `qkv` and cache slot `pos` across all KV heads.
                        unsafe {
                            let row = std::slice::from_raw_parts_mut(
                                (qkv_ptr as *mut f32).add(t * qkv_rows),
                                qkv_rows,
                            );
                            if let Some(bias) = qkv_bias {
                                add_inplace(row, bias);
                            }
                            let rope_row = &rope[pos * head_dim..(pos + 1) * head_dim];
                            let (q, kv) = row.split_at_mut(h);
                            let (k, v) = kv.split_at_mut(kv_dim);
                            apply_llama_rope_precomp(q, cfg.num_attention_heads, head_dim, half_dim, rope_row);
                            apply_llama_rope_precomp(k, num_kv_heads, head_dim, half_dim, rope_row);
                            for kvh in 0..num_kv_heads {
                                let slot = (l_idx * num_kv_heads + kvh) * max_seq_len + pos;
                                store_hadamard_q8_head_raw(
                                    &k[kvh * head_dim..(kvh + 1) * head_dim],
                                    head_dim,
                                    (q8_k_addr as *mut i8).add(slot * head_dim),
                                    (q8_ks_addr as *mut f32).add(slot),
                                );
                                store_hadamard_q8_head_raw(
                                    &v[kvh * head_dim..(kvh + 1) * head_dim],
                                    head_dim,
                                    (q8_v_addr as *mut i8).add(slot * head_dim),
                                    (q8_vs_addr as *mut f32).add(slot),
                                );
                            }
                        }
                    }
                });

                let q8_ro = &*q8;
                let attn_ptr = attn.as_mut_ptr() as usize;
                let scores_ptr = scores_all.as_mut_ptr() as usize;
                let qkv_ref = &*qkv;
                par_range(pool, n, |a, b| {
                    for t in a..b {
                        // SAFETY: token `t` owns row `t` of `attn` and `scores_all`.
                        let dst = unsafe { std::slice::from_raw_parts_mut((attn_ptr as *mut f32).add(t * h), h) };
                        let sc = unsafe {
                            std::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(t * score_stride), score_stride)
                        };
                        causal_gqa_attention_q8(
                            &qkv_ref[t * qkv_rows..t * qkv_rows + h],
                            q8_ro,
                            l_idx,
                            sc,
                            dst,
                            cfg.num_attention_heads,
                            kv_group,
                            pos0 + t,
                            attn_scale,
                        );
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
                        let (q, kv) = row.split_at_mut(h);
                        let (k, v) = kv.split_at_mut(kv_dim);
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
                        let dst = unsafe { std::slice::from_raw_parts_mut((attn_ptr as *mut f32).add(t * h), h) };
                        let sc = unsafe {
                            std::slice::from_raw_parts_mut((scores_ptr as *mut f32).add(t * score_stride), score_stride)
                        };
                        causal_gqa_attention(
                            &qkv_ref[t * qkv_rows..t * qkv_rows + h],
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

            prompt_populate(pool, attn, h, None, acts_h);
            prompt_gemm(pool, &layer.o_proj, acts_h, proj);
            add_inplace(hidden, proj);

            prompt_populate(pool, hidden, h, Some((&layer.post_attn_norm, eps)), acts_h);
            prompt_gemm(pool, &layer.gate_up_proj, acts_h, gate_up);
            let gu_ref = &*gate_up;
            let sw_ptr = sw.as_mut_ptr() as usize;
            let gu_interleaved = layer.gate_up_interleaved;
            par_range(pool, n, |a, b| {
                for t in a..b {
                    let gu = &gu_ref[t * 2 * inter..(t + 1) * 2 * inter];
                    // SAFETY: token `t` owns row `t` of `sw`.
                    let out = unsafe { std::slice::from_raw_parts_mut((sw_ptr as *mut f32).add(t * inter), inter) };
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

        if let Some(out_all) = all_logits_out {
            prompt_populate(pool, hidden, h, Some((&self.final_norm, eps)), acts_h);
            prompt_gemm(pool, &self.lm_head, acts_h, out_all);
            let vocab = self.cfg.vocab_size;
            self.ws.logits.copy_from_slice(&out_all[(n - 1) * vocab..n * vocab]);
        } else {
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
            par_stripes(m.num_stripes, 2, &|s0, s1| {
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
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

/// Single-head GQA attention (`head_dim == 64`) over the INT8 Hadamard KV cache using `SDOT`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn causal_gqa_head64_q8(
    q_head: &[f32],
    k_q8_head: &[i8],
    k_scales_head: &[f32],
    v_q8_head: &[i8],
    v_scales_head: &[f32],
    sc: &mut [f32],
    out_head: &mut [f32; 64],
    attn_scale: f32,
) {
    debug_assert_eq!(q_head.len(), 64);
    let seq_len = sc.len();
    debug_assert_eq!(k_q8_head.len(), seq_len * 64);
    debug_assert_eq!(k_scales_head.len(), seq_len);
    debug_assert_eq!(v_q8_head.len(), seq_len * 64);
    debug_assert_eq!(v_scales_head.len(), seq_len);

    let mut q_i8 = [0i8; 64];
    let mut q_scale = 0.0f32;
    // SAFETY: `q_head.len() == 64`, `q_i8` holds 64 `i8`s, `q_scale` is valid.
    unsafe {
        store_hadamard_q8_head64_raw(q_head, q_i8.as_mut_ptr(), &mut q_scale);
    }
    let q_attn_scale = q_scale * attn_scale;

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: slice lengths verified above; NEON + `sdot` available on Apple Silicon.
        unsafe {
            attn_scores_head64_q8_sdot_neon(&q_i8, k_q8_head, k_scales_head, q_attn_scale, sc);
        }
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        // SAFETY: slice lengths verified above.
        unsafe {
            attn_values_head64_q8_neon(sc, v_q8_head, v_scales_head, inv_sum, out_head);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for t in 0..seq_len {
            let kt = &k_q8_head[t * 64..(t + 1) * 64];
            let dot: i32 = q_i8.iter().zip(kt.iter()).map(|(&a, &b)| (a as i32) * (b as i32)).sum();
            sc[t] = (dot as f32) * k_scales_head[t] * q_attn_scale;
        }
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        out_head.fill(0.0);
        for t in 0..seq_len {
            let w = sc[t] * inv_sum * v_scales_head[t];
            let vt = &v_q8_head[t * 64..(t + 1) * 64];
            for (dst, &vq) in out_head.iter_mut().zip(vt.iter()) {
                *dst += w * (vq as f32);
            }
        }
    }
    ifwht64_signed_inplace(out_head);
}

/// Single-head GQA attention (`head_dim == 128`) over the INT8 Hadamard KV cache via NEON `SDOT`.
#[inline]
#[allow(clippy::too_many_arguments)]
fn causal_gqa_head128_q8(
    q_head: &[f32],
    k_q8_head: &[i8],
    k_scales_head: &[f32],
    v_q8_head: &[i8],
    v_scales_head: &[f32],
    sc: &mut [f32],
    out_head: &mut [f32; 128],
    attn_scale: f32,
) {
    debug_assert_eq!(q_head.len(), 128);
    let seq_len = sc.len();
    debug_assert_eq!(k_q8_head.len(), seq_len * 128);
    debug_assert_eq!(k_scales_head.len(), seq_len);
    debug_assert_eq!(v_q8_head.len(), seq_len * 128);
    debug_assert_eq!(v_scales_head.len(), seq_len);

    let mut q_i8 = [0i8; 128];
    let mut q_scale = 0.0f32;
    // SAFETY: `q_head.len() == 128`, `q_i8` holds 128 `i8`s, `q_scale` is valid.
    unsafe {
        store_hadamard_q8_head128_raw(q_head, q_i8.as_mut_ptr(), &mut q_scale);
    }
    let q_attn_scale = q_scale * attn_scale;

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: slice lengths verified above; NEON + `sdot` available on Apple Silicon.
        unsafe {
            attn_scores_head128_q8_sdot_neon(&q_i8, k_q8_head, k_scales_head, q_attn_scale, sc);
        }
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        // SAFETY: slice lengths verified above.
        unsafe {
            attn_values_head128_q8_neon(sc, v_q8_head, v_scales_head, inv_sum, out_head);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for t in 0..seq_len {
            let kt = &k_q8_head[t * 128..(t + 1) * 128];
            let dot: i32 = q_i8.iter().zip(kt.iter()).map(|(&a, &b)| (a as i32) * (b as i32)).sum();
            sc[t] = (dot as f32) * k_scales_head[t] * q_attn_scale;
        }
        let max_score = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let inv_sum = 1.0 / softmax_unnorm_inplace(sc, max_score);
        out_head.fill(0.0);
        for t in 0..seq_len {
            let w = sc[t] * inv_sum * v_scales_head[t];
            let vt = &v_q8_head[t * 128..(t + 1) * 128];
            for (dst, &vq) in out_head.iter_mut().zip(vt.iter()) {
                *dst += w * (vq as f32);
            }
        }
    }
    ifwht64_signed_inplace(out_head);
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
    if hd == 64 {
        let mut out_head = [0.0f32; 64];
        for qh in 0..num_q_heads {
            let kvh = qh / kv_group;
            let (k_q8_h, ks_h, v_q8_h, vs_h) = q8_kv.head_views(layer_idx, kvh, seq_len);
            let q_head = &q_buf[qh * 64..(qh + 1) * 64];
            let sc = &mut scores[..seq_len];
            causal_gqa_head64_q8(q_head, k_q8_h, ks_h, v_q8_h, vs_h, sc, &mut out_head, attn_scale);
            attn_out[qh * 64..(qh + 1) * 64].copy_from_slice(&out_head);
        }
    } else {
        debug_assert_eq!(hd, 128);
        let mut out_head = [0.0f32; 128];
        for qh in 0..num_q_heads {
            let kvh = qh / kv_group;
            let (k_q8_h, ks_h, v_q8_h, vs_h) = q8_kv.head_views(layer_idx, kvh, seq_len);
            let q_head = &q_buf[qh * 128..(qh + 1) * 128];
            let sc = &mut scores[..seq_len];
            causal_gqa_head128_q8(q_head, k_q8_h, ks_h, v_q8_h, vs_h, sc, &mut out_head, attn_scale);
            attn_out[qh * 128..(qh + 1) * 128].copy_from_slice(&out_head);
        }
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
unsafe fn attn_values_head64_q8_neon(
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
    let mut acc = [vdupq_n_f32(0.0); 16];

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

    let op = out.as_mut_ptr();
    for (i, a) in acc.iter().enumerate() {
        // SAFETY: `i * 4 + 4 <= 64`.
        unsafe { vst1q_f32(op.add(i * 4), *a) };
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn attn_values_head128_q8_neon(
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
        let mut acc = [vdupq_n_f32(0.0); 16];
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
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                tie_word_embeddings: true,
                max_seq_len: 64,
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
}
