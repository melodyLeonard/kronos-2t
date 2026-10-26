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
    compile_hadamard_gptq_full_row, compile_multiplane_ternary, FullRowGptqHessian,
};
use crate::hadamard::{fwht64_signed_inplace, ifwht64_signed_inplace};
use crate::safetensors::{JsonCursor, SafeTensorsFile};
use crate::simd_lut::{
    simd_lut_base3_gemv, simd_lut_base3_stripe_range, simd_lut_bipolar_gemv,
    simd_lut_bipolar_stripe_range, simd_lut_gemv_multiplane, BipolarActivationLut,
    nibble4_sdot_stripe_range, FusedBase3LutMatrix, FusedBipolarLutMatrix, GgufQ4_0Matrix,
    Int8ChunkActivation, MultiPlaneTernaryMatrix, Nibble4SdotMatrix, QuantizedActivationLut,
    SIMD_ROW_STRIPE,
};
use crate::thread_pool::SpinThreadPool;
use std::fs;
use std::io;
use std::path::Path;

/// File magic for [`RealLlamaEngine::save_bipolar`] caches.
const BIPOLAR_CACHE_MAGIC: &[u8] = b"K2TBIPOLAR1\n";

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
    /// **`kronos-2t` Bipolar per-matrix plane allocation** (`3..=6` planes each for `qkv_proj`,
    /// `o_proj`, `gate_up_proj`, `down_proj` and `lm_head`).
    KronosBipolarGptqPlanes {
        qkv: usize,
        o: usize,
        gate_up: usize,
        down: usize,
        head: usize,
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
    /// 4-plane Bipolar weights repacked for `SDOT` (see [`RealLlamaEngine::repack_sdot`]).
    Nibble4(Nibble4SdotMatrix),
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
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn forward_exec(
        &self,
        x: &[f32],
        qlut: Option<&QuantizedActivationLut>,
        blut: Option<&BipolarActivationLut>,
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
                    let grain = (n_stripes / (p.num_threads() * 2)).clamp(4, 128);
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
    /// Fused `[q_proj; k_proj; v_proj]` (`[hidden_size + 2 * kv_dim, hidden_size]`).
    qkv_proj: LinearBackend,
    o_proj: LinearBackend,
    /// Fused `[gate_proj; up_proj]` (`[2 * intermediate_size, hidden_size]`).
    gate_up_proj: LinearBackend,
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

/// Pre-allocated zero-allocation execution workspace and KV cache for [`RealLlamaEngine`].
struct LlamaWorkspace {
    pos: usize,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
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
    q8_hidden_vals: Vec<i8>,
    q8_hidden_scales: Vec<f32>,
    q8_inter_vals: Vec<i8>,
    q8_inter_scales: Vec<f32>,
}

impl LlamaWorkspace {
    fn new(cfg: &LlamaConfig) -> Self {
        let kv_total = cfg.num_hidden_layers * cfg.max_seq_len * cfg.kv_dim();
        let qkv_rows = cfg.hidden_size + 2 * cfg.kv_dim();
        Self {
            pos: 0,
            k_cache: vec![0.0f32; kv_total],
            v_cache: vec![0.0f32; kv_total],
            hidden: vec![0.0f32; cfg.hidden_size],
            norm_buf: vec![0.0f32; cfg.hidden_size],
            qkv_buf: vec![0.0f32; qkv_rows],
            attn_out: vec![0.0f32; cfg.hidden_size],
            proj_out: vec![0.0f32; cfg.hidden_size],
            gate_up_buf: vec![0.0f32; 2 * cfg.intermediate_size],
            swiglu_buf: vec![0.0f32; cfg.intermediate_size],
            scores: vec![0.0f32; cfg.max_seq_len],
            logits: vec![0.0f32; cfg.vocab_size],
            qlut_hidden: QuantizedActivationLut::new_zeroed(cfg.hidden_size),
            qlut_inter: QuantizedActivationLut::new_zeroed(cfg.intermediate_size),
            blut_hidden: BipolarActivationLut::new_zeroed(cfg.hidden_size),
            blut_inter: BipolarActivationLut::new_zeroed(cfg.intermediate_size),
            q8_hidden_vals: vec![0i8; cfg.hidden_size],
            q8_hidden_scales: vec![0.0f32; cfg.hidden_size / 32],
            q8_inter_vals: vec![0i8; cfg.intermediate_size],
            q8_inter_scales: vec![0.0f32; cfg.intermediate_size / 32],
        }
    }

    fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Real pretrained `LlamaForCausalLM` engine supporting `FP32`, `GGUF Q4_0`, and `kronos-2t` Base-3 LUTs.
pub struct RealLlamaEngine {
    pub cfg: LlamaConfig,
    pub mode: RealWeightMode,
    pub mean_weight_rel_error: f32,
    embed_f32_rows: Vec<f32>,
    layers: Vec<LlamaLayer>,
    final_norm: Vec<f32>,
    lm_head: LinearBackend,
    /// Precomputed `[max_seq_len, half_dim * 2]` (`[cos_0..cos_{half-1}, sin_0..sin_{half-1}]`).
    rope_cos_sin: Vec<f32>,
    pool: Option<SpinThreadPool>,
    ws: LlamaWorkspace,
}

impl RealLlamaEngine {
    /// Configures persistent P-core thread pool (`num_threads <= 1` disables threading).
    pub fn set_num_threads(&mut self, num_threads: usize) {
        if num_threads <= 1 {
            self.pool = None;
        } else {
            self.pool = Some(SpinThreadPool::new(num_threads));
        }
    }

    /// Current worker thread count (`1` when single-core).
    pub fn num_threads(&self) -> usize {
        self.pool.as_ref().map(|p| p.num_threads()).unwrap_or(1)
    }

    /// Loads the exact `FP32` reference model from `model_dir` (fusing `[q;k;v]` and `[gate;up]`).
    pub fn load_fp32_reference(model_dir: &Path, max_seq_len: usize) -> io::Result<Self> {
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_seq_len)?;
        let safetensors = SafeTensorsFile::open(&model_dir.join("model.safetensors"))?;

        let get_f32 = |name: &str| -> io::Result<Vec<f32>> {
            safetensors
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

            let mut gate_up_w = Vec::with_capacity(2 * cfg.intermediate_size * cfg.hidden_size);
            gate_up_w.extend_from_slice(&get_f32(&format!("{pfx}.mlp.gate_proj.weight"))?);
            gate_up_w.extend_from_slice(&get_f32(&format!("{pfx}.mlp.up_proj.weight"))?);

            layers.push(LlamaLayer {
                input_norm: get_f32(&format!("{pfx}.input_layernorm.weight"))?,
                post_attn_norm: get_f32(&format!("{pfx}.post_attention_layernorm.weight"))?,
                o_inv_scale: None,
                down_inv_scale: None,
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
                    &[],
                    &[],
                    None,
                    &mut self.ws.qkv_buf,
                );
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
                qkv_proj,
                o_proj,
                gate_up_proj,
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

        Self {
            cfg: self.cfg.clone(),
            mode,
            mean_weight_rel_error: (err_sum / (mat_count as f64)) as f32,
            embed_f32_rows: self.embed_f32_rows.clone(),
            layers: compiled_layers,
            final_norm,
            lm_head,
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
            ws: LlamaWorkspace::new(&self.cfg),
        }
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
        let mut scores = vec![0.0f32; self.cfg.max_seq_len];
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
                qkv_proj: LinearBackend::FusedBase3(qkv_mat),
                o_proj: LinearBackend::FusedBase3(o_mat),
                gate_up_proj: LinearBackend::FusedBase3(gate_up_mat),
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
            layers: compiled_layers,
            final_norm: self.final_norm.clone(),
            lm_head: LinearBackend::FusedBase3(head_mat),
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
            ws: LlamaWorkspace::new(&self.cfg),
        }
    }

    /// Compiles **`kronos-2t` Full-16-Entry Bipolar Radix-2 LUT + Sequential Hadamard-GPTQ**
    /// across `calib_sequences`.
    ///
    /// Packing `4` bipolar weights $s \in \{-1, +1\}^4$ per 4-bit nibble utilizes **all 16 slots**
    /// (`100%` register density) of ARM NEON `vqtbl1q_s8` (`1.00` bit/weight/plane).
    /// - `4` planes = **`4.25` bits/weight** (`16` uniform symmetric levels `{-15..+15}`)
    /// - `Mixed-K` = **`4.66` bits/weight** (`4` planes on `qkv/o/gate_up` + `5` planes on `down_proj/lm_head`)
    /// - `5` planes = **`5.25` bits/weight** (`32` uniform symmetric levels `{-31..+31}`)
    pub fn compile_sequential_bipolar_gptq(
        &self,
        mode: RealWeightMode,
        calib_sequences: &[Vec<usize>],
        compile_threads: usize,
    ) -> Self {
        assert_eq!(self.mode, RealWeightMode::ExactFp32);
        let (qkv_planes, o_planes, gate_up_planes, down_planes, head_planes) = match mode {
            RealWeightMode::KronosBipolarGptq { num_planes } => {
                (num_planes, num_planes, num_planes, num_planes, num_planes)
            }
            RealWeightMode::KronosBipolarGptqMixedK => (4, 5, 4, 5, 5),
            RealWeightMode::KronosBipolarGptqPlanes { qkv, o, gate_up, down, head } => {
                (qkv, o, gate_up, down, head)
            }
            _ => panic!("compile_sequential_bipolar_gptq expects KronosBipolarGptq* mode"),
        };

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
        let mut rot_h = vec![0.0f32; h_dim];
        let mut rot_inter = vec![0.0f32; inter_dim];
        let mut qkv_buf = vec![0.0f32; qkv_rows];
        let mut gate_up_buf = vec![0.0f32; 2 * inter_dim];
        let mut proj_out = vec![0.0f32; h_dim];
        let mut scores = vec![0.0f32; self.cfg.max_seq_len];
        let mut k_seq_cache = vec![0.0f32; self.cfg.max_seq_len * kv_dim];
        let mut v_seq_cache = vec![0.0f32; self.cfg.max_seq_len * kv_dim];
        let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
        let mut blut_i = BipolarActivationLut::new_zeroed(inter_dim);

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
            hess_attn.finalize(0.01);
            let qkv_mat = compile_hadamard_gptq_bipolar(
                ref_layer.qkv_proj.as_fp32_slice(),
                qkv_rows,
                h_dim,
                qkv_planes,
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
                    blut_h.populate(&rot_h);
                    simd_lut_bipolar_gemv(&qkv_mat, &blut_h, &mut qkv_buf);

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
            hess_o.finalize(0.01);
            let o_mat = compile_hadamard_gptq_bipolar(
                ref_layer.o_proj.as_fp32_slice(),
                h_dim,
                h_dim,
                o_planes,
                Some(&hess_o),
                compile_threads,
            );

            // Propagate quantized `o_proj` into residual `seq_hiddens`
            let mut flat_cursor = 0usize;
            for seq_h in &mut seq_hiddens {
                for hid in seq_h.chunks_exact_mut(h_dim) {
                    rot_h.copy_from_slice(
                        &all_attn_out[flat_cursor * h_dim..(flat_cursor + 1) * h_dim],
                    );
                    fwht64_signed_inplace(&mut rot_h);
                    blut_h.populate(&rot_h);
                    simd_lut_bipolar_gemv(&o_mat, &blut_h, &mut proj_out);
                    add_inplace(hid, &proj_out);
                    flat_cursor += 1;
                }
            }

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
            hess_mlp.finalize(0.01);
            let gate_up_mat = compile_hadamard_gptq_bipolar(
                ref_layer.gate_up_proj.as_fp32_slice(),
                2 * inter_dim,
                h_dim,
                gate_up_planes,
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
                    blut_h.populate(&rot_h);
                    simd_lut_bipolar_gemv(&gate_up_mat, &blut_h, &mut gate_up_buf);

                    let (gate_s, up_s) = gate_up_buf.split_at(inter_dim);
                    let sw_dst =
                        &mut all_swiglu[flat_cursor * inter_dim..(flat_cursor + 1) * inter_dim];
                    swiglu_activate(gate_s, up_s, sw_dst);
                    hess_down.observe_activation(sw_dst);
                    flat_cursor += 1;
                }
            }
            hess_down.finalize(0.01);
            let down_mat = compile_hadamard_gptq_bipolar(
                ref_layer.down_proj.as_fp32_slice(),
                h_dim,
                inter_dim,
                down_planes,
                Some(&hess_down),
                compile_threads,
            );

            // Propagate quantized `down_proj` into residual `seq_hiddens`
            let mut flat_cursor = 0usize;
            for seq_h in &mut seq_hiddens {
                for hid in seq_h.chunks_exact_mut(h_dim) {
                    rot_inter.copy_from_slice(
                        &all_swiglu[flat_cursor * inter_dim..(flat_cursor + 1) * inter_dim],
                    );
                    fwht64_signed_inplace(&mut rot_inter);
                    blut_i.populate(&rot_inter);
                    simd_lut_bipolar_gemv(&down_mat, &blut_i, &mut proj_out);
                    add_inplace(hid, &proj_out);
                    flat_cursor += 1;
                }
            }

            compiled_layers.push(LlamaLayer {
                input_norm: ref_layer.input_norm.clone(),
                post_attn_norm: ref_layer.post_attn_norm.clone(),
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_proj: LinearBackend::FusedBipolar(qkv_mat),
                o_proj: LinearBackend::FusedBipolar(o_mat),
                gate_up_proj: LinearBackend::FusedBipolar(gate_up_mat),
                down_proj: LinearBackend::FusedBipolar(down_mat),
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
        hess_head.finalize(0.01);
        let head_mat = compile_hadamard_gptq_bipolar(
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
            // Tied embeddings are decoded on the fly from the quantized `lm_head` rows (no FP32 copy).
            embed_f32_rows: if self.cfg.tie_word_embeddings {
                Vec::new()
            } else {
                self.embed_f32_rows.clone()
            },
            layers: compiled_layers,
            final_norm: self.final_norm.clone(),
            lm_head: LinearBackend::FusedBipolar(head_mat),
            rope_cos_sin: self.rope_cos_sin.clone(),
            pool: None,
            ws: LlamaWorkspace::new(&self.cfg),
        }
    }

    /// Repacks every `4..=6`-plane Bipolar projection into [`Nibble4SdotMatrix`] (bit-exact weights,
    /// `SDOT` kernel) and enables the `int8` activation path; the LUT build is skipped when no
    /// LUT-kernel projection remains. Call after compiling or loading, before inference.
    pub fn repack_sdot(&mut self) {
        let repack = |b: &mut LinearBackend| {
            if let LinearBackend::FusedBipolar(m) = b {
                if (4..=6).contains(&m.num_planes) {
                    *b = LinearBackend::Nibble4(Nibble4SdotMatrix::from_bipolar(m));
                }
            }
        };
        for l in &mut self.layers {
            repack(&mut l.qkv_proj);
            repack(&mut l.o_proj);
            repack(&mut l.gate_up_proj);
            repack(&mut l.down_proj);
        }
        repack(&mut self.lm_head);

        let is_lut = |b: &LinearBackend| match b {
            LinearBackend::FusedBipolar(_) => true,
            LinearBackend::Nibble4(m) => m.hi_planes > 0,
            _ => false,
        };
        let hidden_lut = is_lut(&self.lm_head)
            || self.layers.iter().any(|l| is_lut(&l.qkv_proj) || is_lut(&l.o_proj) || is_lut(&l.gate_up_proj));
        let inter_lut = self.layers.iter().any(|l| is_lut(&l.down_proj));
        self.ws.blut_hidden.lut_enabled = hidden_lut;
        self.ws.blut_inter.lut_enabled = inter_lut;
        self.ws.blut_hidden.int8 = Some(Int8ChunkActivation::new_zeroed(self.cfg.hidden_size));
        self.ws.blut_inter.int8 = Some(Int8ChunkActivation::new_zeroed(self.cfg.intermediate_size));
    }

    /// Saves a compiled Bipolar engine (every projection `FusedBipolar`) so later runs can skip the
    /// sequential GPTQ compile via [`Self::load_bipolar`].
    pub fn save_bipolar(&self, path: &Path) -> io::Result<()> {
        use std::io::Write;
        fn w_u64(w: &mut impl Write, v: usize) -> io::Result<()> {
            w.write_all(&(v as u64).to_le_bytes())
        }
        fn w_f32s(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
            w_u64(w, v.len())?;
            v.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }
        fn w_mat(w: &mut impl Write, b: &LinearBackend) -> io::Result<()> {
            let LinearBackend::FusedBipolar(m) = b else {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "non-bipolar projection"));
            };
            w_u64(w, m.rows)?;
            w_u64(w, m.cols)?;
            w_u64(w, m.num_planes)?;
            w_u64(w, m.interleaved_data.len())?;
            w.write_all(&m.interleaved_data)?;
            w_u64(w, m.bf16_chunk_scales.len())?;
            m.bf16_chunk_scales.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
        }

        let mut w = io::BufWriter::new(fs::File::create(path)?);
        w.write_all(BIPOLAR_CACHE_MAGIC)?;
        let mode_words = match self.mode {
            RealWeightMode::KronosBipolarGptq { num_planes } => [0, num_planes, 0, 0, 0, 0],
            RealWeightMode::KronosBipolarGptqMixedK => [1, 0, 0, 0, 0, 0],
            RealWeightMode::KronosBipolarGptqPlanes { qkv, o, gate_up, down, head } => {
                [2, qkv, o, gate_up, down, head]
            }
            _ => return Err(io::Error::new(io::ErrorKind::InvalidInput, "not a bipolar engine")),
        };
        mode_words.iter().try_for_each(|&v| w_u64(&mut w, v))?;
        w_u64(&mut w, self.layers.len())?;
        for l in &self.layers {
            w_f32s(&mut w, &l.input_norm)?;
            w_f32s(&mut w, &l.post_attn_norm)?;
            for b in [&l.qkv_proj, &l.o_proj, &l.gate_up_proj, &l.down_proj] {
                w_mat(&mut w, b)?;
            }
        }
        w_f32s(&mut w, &self.final_norm)?;
        w_mat(&mut w, &self.lm_head)?;
        w_f32s(&mut w, &self.embed_f32_rows)?;
        w.flush()
    }

    /// Loads an engine written by [`Self::save_bipolar`]; `model_dir` supplies `config.json`.
    pub fn load_bipolar(model_dir: &Path, path: &Path, max_seq_len: usize) -> io::Result<Self> {
        let cfg = LlamaConfig::from_config_json(&model_dir.join("config.json"), max_seq_len)?;
        let bytes = fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if !bytes.starts_with(BIPOLAR_CACHE_MAGIC) {
            return Err(bad("bad bipolar cache magic"));
        }
        let mut rd = CacheReader {
            bytes: &bytes,
            pos: BIPOLAR_CACHE_MAGIC.len(),
        };
        let r_mat = |rd: &mut CacheReader| -> io::Result<LinearBackend> {
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

        let mut mw = [0usize; 6];
        for v in mw.iter_mut() {
            *v = rd.u64()?;
        }
        let mode = match mw[0] {
            0 => RealWeightMode::KronosBipolarGptq { num_planes: mw[1] },
            1 => RealWeightMode::KronosBipolarGptqMixedK,
            _ => RealWeightMode::KronosBipolarGptqPlanes {
                qkv: mw[1],
                o: mw[2],
                gate_up: mw[3],
                down: mw[4],
                head: mw[5],
            },
        };
        let num_layers = rd.u64()?;
        if num_layers != cfg.num_hidden_layers {
            return Err(bad("layer count does not match config.json"));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let input_norm = rd.f32s()?;
            let post_attn_norm = rd.f32s()?;
            layers.push(LlamaLayer {
                input_norm,
                post_attn_norm,
                o_inv_scale: None,
                down_inv_scale: None,
                qkv_proj: r_mat(&mut rd)?,
                o_proj: r_mat(&mut rd)?,
                gate_up_proj: r_mat(&mut rd)?,
                down_proj: r_mat(&mut rd)?,
            });
        }
        let final_norm = rd.f32s()?;
        let lm_head = r_mat(&mut rd)?;
        let embed_f32_rows = rd.f32s()?;

        Ok(Self {
            rope_cos_sin: build_rope_table(&cfg),
            ws: LlamaWorkspace::new(&cfg),
            cfg,
            mode,
            mean_weight_rel_error: 0.0,
            embed_f32_rows,
            layers,
            final_norm,
            lm_head,
            pool: None,
        })
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

    /// Total resident weight RAM: linear projections plus any separate FP32 embedding table.
    pub fn resident_weight_bytes(&self) -> usize {
        self.linear_weight_bytes() + self.embed_f32_rows.len() * std::mem::size_of::<f32>()
    }

    /// Physical bits per weight across all 30 transformer layers + `lm_head`.
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

    /// Resets KV cache position to 0.
    pub fn reset_context(&mut self) {
        self.ws.reset();
    }

    /// Executes one full 30-layer causal step for `token_id` and returns a slice of `[vocab_size]` logits.
    pub fn forward_token(&mut self, token_id: usize) -> &[f32] {
        assert!(
            self.ws.pos < self.cfg.max_seq_len,
            "sequence length {} reached max_seq_len {}",
            self.ws.pos,
            self.cfg.max_seq_len
        );
        let pos = self.ws.pos;
        let h_dim = self.cfg.hidden_size;
        let inter_dim = self.cfg.intermediate_size;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_dim = self.cfg.kv_dim();
        let kv_group = self.cfg.num_attention_heads / self.cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let layer_kv_stride = self.cfg.max_seq_len * kv_dim;

        let is_bipolar = matches!(
            self.mode,
            RealWeightMode::KronosBipolarGptq { .. }
                | RealWeightMode::KronosBipolarGptqMixedK
                | RealWeightMode::KronosBipolarGptqPlanes { .. }
        );
        let is_hadamard = matches!(
            self.mode,
            RealWeightMode::KronosHadamardGptqBase3 { .. }
                | RealWeightMode::KronosHadamardGptqMixedK
        );
        let is_legacy_lut = matches!(
            self.mode,
            RealWeightMode::GlobalTernary1Plane
                | RealWeightMode::Group64Ternary { .. }
                | RealWeightMode::AwqGroup64Ternary { .. }
        );
        let is_q4_0 = matches!(self.mode, RealWeightMode::GgufQ4_0);

        if self.embed_f32_rows.is_empty() {
            match &self.lm_head {
                LinearBackend::FusedBipolar(m) => {
                    m.dequantize_row_rotated(token_id, &mut self.ws.hidden);
                    ifwht64_signed_inplace(&mut self.ws.hidden);
                }
                LinearBackend::Nibble4(m) => {
                    m.dequantize_row_rotated(token_id, &mut self.ws.hidden);
                    ifwht64_signed_inplace(&mut self.ws.hidden);
                }
                _ => unreachable!("embedding table dropped only for bipolar tied lm_head"),
            }
        } else {
            self.ws
                .hidden
                .copy_from_slice(&self.embed_f32_rows[token_id * h_dim..(token_id + 1) * h_dim]);
        }
        let rope_row = &self.rope_cos_sin[pos * head_dim..(pos + 1) * head_dim];
        let pool_ref = self.pool.as_ref();

        for (l_idx, layer) in self.layers.iter().enumerate() {
            // 1. Pre-Attention RMSNorm + Fused QKV Projection
            rms_norm(
                &self.ws.hidden,
                &layer.input_norm,
                self.cfg.rms_norm_eps,
                &mut self.ws.norm_buf,
            );

            let (qlut_qkv, blut_qkv) = if is_bipolar {
                fwht64_signed_inplace(&mut self.ws.norm_buf);
                self.ws.blut_hidden.populate(&self.ws.norm_buf);
                (None, Some(&self.ws.blut_hidden))
            } else if is_hadamard {
                fwht64_signed_inplace(&mut self.ws.norm_buf);
                self.ws.qlut_hidden.populate(&self.ws.norm_buf);
                (Some(&self.ws.qlut_hidden), None)
            } else if is_legacy_lut {
                self.ws.qlut_hidden.populate(&self.ws.norm_buf);
                (Some(&self.ws.qlut_hidden), None)
            } else {
                if is_q4_0 {
                    GgufQ4_0Matrix::quantize_q8_0_activation(
                        &self.ws.norm_buf,
                        &mut self.ws.q8_hidden_vals,
                        &mut self.ws.q8_hidden_scales,
                    );
                }
                (None, None)
            };

            layer.qkv_proj.forward_exec(
                &self.ws.norm_buf,
                qlut_qkv,
                blut_qkv,
                &self.ws.q8_hidden_vals,
                &self.ws.q8_hidden_scales,
                pool_ref,
                &mut self.ws.qkv_buf,
            );

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

            // 3. Output Projection + Residual
            if let Some(inv_s) = &layer.o_inv_scale {
                for (a, &is) in self.ws.attn_out.iter_mut().zip(inv_s.iter()) {
                    *a *= is;
                }
            }
            let (qlut_o, blut_o) = if is_bipolar {
                fwht64_signed_inplace(&mut self.ws.attn_out);
                self.ws.blut_hidden.populate(&self.ws.attn_out);
                (None, Some(&self.ws.blut_hidden))
            } else if is_hadamard {
                fwht64_signed_inplace(&mut self.ws.attn_out);
                self.ws.qlut_hidden.populate(&self.ws.attn_out);
                (Some(&self.ws.qlut_hidden), None)
            } else if is_legacy_lut {
                self.ws.qlut_hidden.populate(&self.ws.attn_out);
                (Some(&self.ws.qlut_hidden), None)
            } else {
                if is_q4_0 {
                    GgufQ4_0Matrix::quantize_q8_0_activation(
                        &self.ws.attn_out,
                        &mut self.ws.q8_hidden_vals,
                        &mut self.ws.q8_hidden_scales,
                    );
                }
                (None, None)
            };
            layer.o_proj.forward_exec(
                &self.ws.attn_out,
                qlut_o,
                blut_o,
                &self.ws.q8_hidden_vals,
                &self.ws.q8_hidden_scales,
                pool_ref,
                &mut self.ws.proj_out,
            );
            add_inplace(&mut self.ws.hidden, &self.ws.proj_out);

            // 4. Post-Attention RMSNorm + Fused Gate+Up SwiGLU Projection
            rms_norm(
                &self.ws.hidden,
                &layer.post_attn_norm,
                self.cfg.rms_norm_eps,
                &mut self.ws.norm_buf,
            );
            let (qlut_gu, blut_gu) = if is_bipolar {
                fwht64_signed_inplace(&mut self.ws.norm_buf);
                self.ws.blut_hidden.populate(&self.ws.norm_buf);
                (None, Some(&self.ws.blut_hidden))
            } else if is_hadamard {
                fwht64_signed_inplace(&mut self.ws.norm_buf);
                self.ws.qlut_hidden.populate(&self.ws.norm_buf);
                (Some(&self.ws.qlut_hidden), None)
            } else if is_legacy_lut {
                self.ws.qlut_hidden.populate(&self.ws.norm_buf);
                (Some(&self.ws.qlut_hidden), None)
            } else {
                if is_q4_0 {
                    GgufQ4_0Matrix::quantize_q8_0_activation(
                        &self.ws.norm_buf,
                        &mut self.ws.q8_hidden_vals,
                        &mut self.ws.q8_hidden_scales,
                    );
                }
                (None, None)
            };
            layer.gate_up_proj.forward_exec(
                &self.ws.norm_buf,
                qlut_gu,
                blut_gu,
                &self.ws.q8_hidden_vals,
                &self.ws.q8_hidden_scales,
                pool_ref,
                &mut self.ws.gate_up_buf,
            );

            let (gate_s, up_s) = self.ws.gate_up_buf.split_at(inter_dim);
            swiglu_activate(gate_s, up_s, &mut self.ws.swiglu_buf);

            if let Some(inv_s) = &layer.down_inv_scale {
                for (g, &is) in self.ws.swiglu_buf.iter_mut().zip(inv_s.iter()) {
                    *g *= is;
                }
            }
            let (qlut_down, blut_down) = if is_bipolar {
                fwht64_signed_inplace(&mut self.ws.swiglu_buf);
                self.ws.blut_inter.populate(&self.ws.swiglu_buf);
                (None, Some(&self.ws.blut_inter))
            } else if is_hadamard {
                fwht64_signed_inplace(&mut self.ws.swiglu_buf);
                self.ws.qlut_inter.populate(&self.ws.swiglu_buf);
                (Some(&self.ws.qlut_inter), None)
            } else if is_legacy_lut {
                self.ws.qlut_inter.populate(&self.ws.swiglu_buf);
                (Some(&self.ws.qlut_inter), None)
            } else {
                if is_q4_0 {
                    GgufQ4_0Matrix::quantize_q8_0_activation(
                        &self.ws.swiglu_buf,
                        &mut self.ws.q8_inter_vals,
                        &mut self.ws.q8_inter_scales,
                    );
                }
                (None, None)
            };
            layer.down_proj.forward_exec(
                &self.ws.swiglu_buf,
                qlut_down,
                blut_down,
                &self.ws.q8_inter_vals,
                &self.ws.q8_inter_scales,
                pool_ref,
                &mut self.ws.proj_out,
            );
            add_inplace(&mut self.ws.hidden, &self.ws.proj_out);
        }

        // Final RMSNorm + LM Head (`49,152` vocabulary logits)
        rms_norm(
            &self.ws.hidden,
            &self.final_norm,
            self.cfg.rms_norm_eps,
            &mut self.ws.norm_buf,
        );
        let (qlut_final, blut_final) = if is_bipolar {
            fwht64_signed_inplace(&mut self.ws.norm_buf);
            self.ws.blut_hidden.populate(&self.ws.norm_buf);
            (None, Some(&self.ws.blut_hidden))
        } else if is_hadamard {
            fwht64_signed_inplace(&mut self.ws.norm_buf);
            self.ws.qlut_hidden.populate(&self.ws.norm_buf);
            (Some(&self.ws.qlut_hidden), None)
        } else if is_legacy_lut {
            self.ws.qlut_hidden.populate(&self.ws.norm_buf);
            (Some(&self.ws.qlut_hidden), None)
        } else {
            if is_q4_0 {
                GgufQ4_0Matrix::quantize_q8_0_activation(
                    &self.ws.norm_buf,
                    &mut self.ws.q8_hidden_vals,
                    &mut self.ws.q8_hidden_scales,
                );
            }
            (None, None)
        };
        self.lm_head.forward_exec(
            &self.ws.norm_buf,
            qlut_final,
            blut_final,
            &self.ws.q8_hidden_vals,
            &self.ws.q8_hidden_scales,
            pool_ref,
            &mut self.ws.logits,
        );

        let _ = SIMD_ROW_STRIPE;
        self.ws.pos += 1;
        &self.ws.logits
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
fn apply_llama_rope_precomp(
    buf: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    half_dim: usize,
    rope_row: &[f32],
) {
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
    for qh in 0..num_q_heads {
        let kvh = qh / kv_group;
        let q_head = &q_buf[qh * head_dim..(qh + 1) * head_dim];

        let mut max_score = f32::NEG_INFINITY;
        for (t, sc_slot) in scores[..seq_len].iter_mut().enumerate() {
            let k_off = t * kv_dim + kvh * head_dim;
            let k_head = &k_layer_cache[k_off..k_off + head_dim];
            let s = dot_f32_fast(q_head, k_head) * attn_scale;
            *sc_slot = s;
            if s > max_score {
                max_score = s;
            }
        }

        let sum_exp = softmax_unnorm_inplace(&mut scores[..seq_len], max_score);
        let inv_sum = 1.0 / sum_exp;

        let out_head = &mut attn_out[qh * head_dim..(qh + 1) * head_dim];
        out_head.fill(0.0);
        for (t, &unnorm_p) in scores[..seq_len].iter().enumerate() {
            let p = unnorm_p * inv_sum;
            let v_off = t * kv_dim + kvh * head_dim;
            let v_head = &v_layer_cache[v_off..v_off + head_dim];
            axpy_f32_fast(out_head, v_head, p);
        }
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
}
