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
    nibble4_sdot_batch_stripe_range, nibble4_sdot_stripe_range, nibble4_sdot_stripe_range_int8,
    nibble4_sdot_stripe_range_raw, simd_lut_base3_gemv, simd_lut_base3_stripe_range,
    simd_lut_bipolar_gemv, simd_lut_bipolar_stripe_range, simd_lut_gemv_multiplane,
    BipolarActivationLut, FusedBase3LutMatrix, FusedBipolarLutMatrix, GgufQ4_0Matrix,
    Int8ChunkActivation, MultiPlaneTernaryMatrix, Nibble4SdotMatrix, QuantizedActivationLut,
    SIMD_ROW_STRIPE,
};
use crate::thread_pool::{physical_p_cores, SpinBarrier, SpinThreadPool};
use std::fs;
use std::io;
use std::path::Path;

/// Hessian off-diagonal shrinkage for the Bipolar GPTQ compile.
const BIPOLAR_GPTQ_DAMP: f64 = 0.01;

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
    /// Bipolar with per-matrix plane counts.
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
    /// Bipolar weights repacked for `SDOT`.
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
    prompt_scratch: PromptScratch,
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
            scores: vec![0.0f32; cfg.max_seq_len * cfg.num_attention_heads],
            logits: vec![0.0f32; cfg.vocab_size],
            qlut_hidden: QuantizedActivationLut::new_zeroed(cfg.hidden_size),
            qlut_inter: QuantizedActivationLut::new_zeroed(cfg.intermediate_size),
            blut_hidden: BipolarActivationLut::new_zeroed(cfg.hidden_size),
            blut_inter: BipolarActivationLut::new_zeroed(cfg.intermediate_size),
            q8_hidden_vals: vec![0i8; cfg.hidden_size],
            q8_hidden_scales: vec![0.0f32; cfg.hidden_size / 32],
            q8_inter_vals: vec![0i8; cfg.intermediate_size],
            q8_inter_scales: vec![0.0f32; cfg.intermediate_size / 32],
            prompt_scratch: PromptScratch::default(),
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
            let qkv_mat = compile_hadamard_gptq_bipolar(
                ref_layer.qkv_proj.as_fp32_slice(),
                qkv_rows,
                h_dim,
                qkv_planes,
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
                for (pos, (hid, attn_dst)) in seq_h
                    .chunks_exact(h_dim)
                    .zip(attn_seq.chunks_exact_mut(h_dim))
                    .enumerate()
                {
                    rms_norm(hid, &ref_layer.input_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    blut_h.populate(&norm_buf);
                    simd_lut_bipolar_gemv(&qkv_mat, &blut_h, &mut qkv_buf);

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
            let o_mat = compile_hadamard_gptq_bipolar(
                ref_layer.o_proj.as_fp32_slice(),
                h_dim,
                h_dim,
                o_planes,
                Some(&hess_o),
                compile_threads,
            );

            // Propagate quantized `o_proj` into residual `seq_hiddens`
            par_calib(&mut seq_hiddens, &mut all_attn_out, h_dim, h_dim, |seq_h, attn_seq| {
                let mut rot_h = vec![0.0f32; h_dim];
                let mut proj_out = vec![0.0f32; h_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                for (hid, attn) in seq_h.chunks_exact_mut(h_dim).zip(attn_seq.chunks_exact(h_dim)) {
                    rot_h.copy_from_slice(attn);
                    fwht64_signed_inplace(&mut rot_h);
                    blut_h.populate(&rot_h);
                    simd_lut_bipolar_gemv(&o_mat, &blut_h, &mut proj_out);
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
            let gate_up_mat = compile_hadamard_gptq_bipolar(
                ref_layer.gate_up_proj.as_fp32_slice(),
                2 * inter_dim,
                h_dim,
                gate_up_planes,
                Some(&hess_mlp),
                compile_threads,
            );

            // 4. Execute quantized `gate_up_proj` + `SwiGLU` -> gather `down_proj` Hessian
            par_calib(&mut seq_hiddens, &mut all_swiglu, h_dim, inter_dim, |seq_h, sw_seq| {
                let mut norm_buf = vec![0.0f32; h_dim];
                let mut gate_up_buf = vec![0.0f32; 2 * inter_dim];
                let mut blut_h = BipolarActivationLut::new_zeroed(h_dim);
                for (hid, sw_dst) in seq_h.chunks_exact(h_dim).zip(sw_seq.chunks_exact_mut(inter_dim)) {
                    rms_norm(hid, &ref_layer.post_attn_norm, self.cfg.rms_norm_eps, &mut norm_buf);
                    fwht64_signed_inplace(&mut norm_buf);
                    blut_h.populate(&norm_buf);
                    simd_lut_bipolar_gemv(&gate_up_mat, &blut_h, &mut gate_up_buf);
                    let (gate_s, up_s) = gate_up_buf.split_at(inter_dim);
                    swiglu_activate(gate_s, up_s, sw_dst);
                }
            });
            let mut hess_down = FullRowGptqHessian::new(inter_dim);
            for row in all_swiglu.chunks_exact(inter_dim) {
                hess_down.observe_activation(row);
            }
            hess_down.finalize(BIPOLAR_GPTQ_DAMP);
            let down_mat = compile_hadamard_gptq_bipolar(
                ref_layer.down_proj.as_fp32_slice(),
                h_dim,
                inter_dim,
                down_planes,
                Some(&hess_down),
                compile_threads,
            );

            // Propagate quantized `down_proj` into residual `seq_hiddens`
            par_calib(&mut seq_hiddens, &mut all_swiglu, h_dim, inter_dim, |seq_h, sw_seq| {
                let mut rot_inter = vec![0.0f32; inter_dim];
                let mut proj_out = vec![0.0f32; h_dim];
                let mut blut_i = BipolarActivationLut::new_zeroed(inter_dim);
                for (hid, sw) in seq_h.chunks_exact_mut(h_dim).zip(sw_seq.chunks_exact(inter_dim)) {
                    rot_inter.copy_from_slice(sw);
                    fwht64_signed_inplace(&mut rot_inter);
                    blut_i.populate(&rot_inter);
                    simd_lut_bipolar_gemv(&down_mat, &blut_i, &mut proj_out);
                    add_inplace(hid, &proj_out);
                }
            });

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
        hess_head.finalize(BIPOLAR_GPTQ_DAMP);
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
            // Tied embeddings are decoded from the quantized `lm_head`.
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

    /// Repacks `4..=6`-plane Bipolar projections for the `SDOT` kernel (same weights).
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

        // Pure-`SDOT` executes all `4..=6` planes directly from `Int8ChunkActivation` with zero D4 LUTs!
        let is_lut = |b: &LinearBackend| matches!(b, LinearBackend::FusedBipolar(_));
        let hidden_lut = is_lut(&self.lm_head)
            || self.layers.iter().any(|l| is_lut(&l.qkv_proj) || is_lut(&l.o_proj) || is_lut(&l.gate_up_proj));
        let inter_lut = self.layers.iter().any(|l| is_lut(&l.down_proj));
        self.ws.blut_hidden.lut_enabled = hidden_lut;
        self.ws.blut_inter.lut_enabled = inter_lut;
        self.ws.blut_hidden.int8 = Some(Int8ChunkActivation::new_zeroed(self.cfg.hidden_size));
        self.ws.blut_inter.int8 = Some(Int8ChunkActivation::new_zeroed(self.cfg.intermediate_size));
    }

    /// Saves a compiled Bipolar engine; see [`Self::load_bipolar`].
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

    /// Linear weights plus any FP32 embedding table.
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

    /// Drops KV-cache entries after the first `n` tokens.
    pub fn truncate_context(&mut self, n: usize) {
        self.ws.pos = self.ws.pos.min(n);
    }

    /// Tokens currently in the KV cache.
    pub fn position(&self) -> usize {
        self.ws.pos
    }

    /// Resets KV cache position to 0.
    pub fn reset_context(&mut self) {
        self.ws.reset();
    }

    /// Returns `true` when every projection uses Pure-`SDOT` [`Nibble4SdotMatrix`].
    #[inline]
    pub fn is_all_sdot(&self) -> bool {
        let is_sdot = |b: &LinearBackend| matches!(b, LinearBackend::Nibble4(_));
        is_sdot(&self.lm_head)
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
        self.cfg.head_dim() == 64
            && self.cfg.hidden_size <= 4096
            && self.cfg.hidden_size.is_multiple_of(64)
            && self.cfg.intermediate_size.is_multiple_of(64)
            && self.ws.blut_hidden.int8.is_some()
            && self.ws.blut_inter.int8.is_some()
            && self.is_all_sdot()
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

    /// Single-dispatch persistent P-core megakernel for Pure-`SDOT` (`4/5/6`-bit) decode.
    ///
    /// Executes all Transformer layers + `lm_head` inside **one** thread-pool dispatch with:
    /// - In-register fused `RMSNorm + FWHT-64 + Int8ChunkActivation` quantization (`0` `norm_buf` writes),
    /// - 64-row head-aligned `qkv_proj` fused directly with NEON `RoPE` and `KV` cache writes,
    /// - Multi-head parallel `GQA Attention` fused directly into in-register `FWHT-64 + Int8Quant` (`0` `attn_out` writes),
    /// - Worker-local 64-row L1-fused `[gate; up] -> SwiGLU -> FWHT-64 -> Int8Quant` (`0` extra barrier),
    /// - Even-stripe (`32`-row) worker partitions so 100% of stripes hit the 2-stripe `16`-accumulator NEON fast path.
    fn forward_token_sdot_megakernel(&mut self, token_id: usize) -> &[f32] {
        let pos = self.ws.pos;
        let seq_len = pos + 1;
        let h_dim = self.cfg.hidden_size;
        let inter_dim = self.cfg.intermediate_size;
        let nc_h = h_dim / 64;
        let nc_i = inter_dim / 64;
        let num_q_heads = self.cfg.num_attention_heads;
        let num_kv_heads = self.cfg.num_key_value_heads;
        let kv_dim = self.cfg.kv_dim();
        let kv_group = num_q_heads / num_kv_heads;
        let attn_scale = 1.0 / 8.0f32; // `head_dim == 64` -> `1 / sqrt(64) == 0.125`
        let max_seq_len = self.cfg.max_seq_len;
        let layer_kv_stride = max_seq_len * kv_dim;
        let eps = self.cfg.rms_norm_eps;

        let LinearBackend::Nibble4(head_mat) = &self.lm_head else {
            unreachable!("megakernel requires Nibble4 lm_head");
        };
        if self.embed_f32_rows.is_empty() {
            head_mat.dequantize_row_rotated(token_id, &mut self.ws.hidden);
            ifwht64_signed_inplace(&mut self.ws.hidden);
        } else {
            self.ws
                .hidden
                .copy_from_slice(&self.embed_f32_rows[token_id * h_dim..(token_id + 1) * h_dim]);
        }

        let rope_row = &self.rope_cos_sin[pos * 64..(pos + 1) * 64];
        let layers = &self.layers;
        let final_norm = &self.final_norm;

        let hidden_addr = self.ws.hidden.as_mut_ptr() as usize;
        let qkv_addr = self.ws.qkv_buf.as_mut_ptr() as usize;
        let proj_addr = self.ws.proj_out.as_mut_ptr() as usize;
        let gu_addr = self.ws.gate_up_buf.as_mut_ptr() as usize;
        let scores_addr = self.ws.scores.as_mut_ptr() as usize;
        let k_cache_addr = self.ws.k_cache.as_mut_ptr() as usize;
        let v_cache_addr = self.ws.v_cache.as_mut_ptr() as usize;
        let logits_addr = self.ws.logits.as_mut_ptr() as usize;

        let (act_h_addr, act_hq_addr, act_hs_addr, act_hsum_addr) = {
            let a = self.ws.blut_hidden.int8.as_mut().unwrap();
            (
                (a as *const Int8ChunkActivation) as usize,
                a.q.as_mut_ptr() as usize,
                a.scales.as_mut_ptr() as usize,
                a.sums.as_mut_ptr() as usize,
            )
        };
        let (act_i_addr, act_iq_addr, act_is_addr, act_isum_addr) = {
            let a = self.ws.blut_inter.int8.as_mut().unwrap();
            (
                (a as *const Int8ChunkActivation) as usize,
                a.q.as_mut_ptr() as usize,
                a.scales.as_mut_ptr() as usize,
                a.sums.as_mut_ptr() as usize,
            )
        };

        let even_stripes = |num_stripes: usize, tid: usize, nt: usize| -> (usize, usize) {
            let pairs = num_stripes.div_ceil(2);
            let s0 = (((pairs * tid) / nt) * 2).min(num_stripes);
            let s1 = (((pairs * (tid + 1)) / nt) * 2).min(num_stripes);
            (s0, s1)
        };

        let kernel = |tid: usize, nt: usize, barrier: &SpinBarrier| {
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

            // Worker-private L1D stack buffers (`4.6 KB`): eliminates 3 cross-core barriers per layer!
            let mut lq = [0i8; 4096];
            let mut ls = [0.0f32; 64];
            let mut lsum = [0i32; 64];

            let qh0 = (num_q_heads * tid) / nt;
            let qh1 = (num_q_heads * (tid + 1)) / nt;
            let ic0 = (nc_i * tid) / nt;
            let ic1 = (nc_i * (tid + 1)) / nt;

            for (l_idx, layer) in layers.iter().enumerate() {
                let (
                    LinearBackend::Nibble4(qkv_mat),
                    LinearBackend::Nibble4(o_mat),
                    LinearBackend::Nibble4(gu_mat),
                    LinearBackend::Nibble4(down_mat),
                ) = (&layer.qkv_proj, &layer.o_proj, &layer.gate_up_proj, &layer.down_proj)
                else {
                    unreachable!();
                };
                let kv_base = l_idx * layer_kv_stride;

                // 1. Zero-Barrier Worker-Local `input_norm + FWHT-64 + INT8` -> 64-Row `qkv_proj + RoPE + KV Store`
                let qkv_heads = qkv_mat.rows / 64;
                let b0 = (qkv_heads * tid) / nt;
                let b1 = (qkv_heads * (tid + 1)) / nt;
                if b0 < b1 {
                    // SAFETY: `hidden` is globally synchronized and read-only here; `(lq, ls, lsum)` are worker-private
                    // stack arrays (`h_dim <= 4096`), and `[b0*4, b1*4)` writes disjoint 64-row head blocks.
                    unsafe {
                        let hidden_ro = std::slice::from_raw_parts(hidden_ptr, h_dim);
                        let inv_rms = 1.0 / (dot_f32_fast(hidden_ro, hidden_ro) / (h_dim as f32) + eps).sqrt();
                        for c in 0..nc_h {
                            Int8ChunkActivation::populate_chunk_fwht_raw(
                                lq.as_mut_ptr(),
                                ls.as_mut_ptr(),
                                lsum.as_mut_ptr(),
                                c,
                                &hidden_ro[c * 64..(c + 1) * 64],
                                Some((&layer.input_norm[c * 64..(c + 1) * 64], inv_rms)),
                            );
                        }
                        nibble4_sdot_stripe_range_raw(
                            qkv_mat,
                            lq.as_ptr(),
                            ls.as_ptr(),
                            lsum.as_ptr(),
                            b0 * 4,
                            b1 * 4,
                            qkv_ptr,
                        );
                        for b in b0..b1 {
                            let head_s = std::slice::from_raw_parts_mut(qkv_ptr.add(b * 64), 64);
                            if b < num_q_heads {
                                apply_llama_rope_head64(head_s, rope_row);
                            } else if b < num_q_heads + num_kv_heads {
                                let kvh = b - num_q_heads;
                                apply_llama_rope_head64(head_s, rope_row);
                                std::ptr::copy_nonoverlapping(
                                    head_s.as_ptr(),
                                    k_cache_ptr.add(kv_base + pos * kv_dim + kvh * 64),
                                    64,
                                );
                            } else {
                                let kvh = b - num_q_heads - num_kv_heads;
                                std::ptr::copy_nonoverlapping(
                                    head_s.as_ptr(),
                                    v_cache_ptr.add(kv_base + pos * kv_dim + kvh * 64),
                                    64,
                                );
                            }
                        }
                    }
                }
                barrier.wait();

                // 2. Parallel Per-Head `GQA Attention` + Fused In-Register `FWHT-64 + INT8 Quantization`
                if qh0 < qh1 {
                    // SAFETY: `qkv_buf[..h_dim]`, `k_cache`, `v_cache` are immutable; each `qh` owns `scores[qh*max_seq_len..]`
                    // and writes exclusively to Hadamard chunk `c = qh` of `act_h`.
                    unsafe {
                        let k_layer = std::slice::from_raw_parts(k_cache_ptr.add(kv_base), seq_len * kv_dim);
                        let v_layer = std::slice::from_raw_parts(v_cache_ptr.add(kv_base), seq_len * kv_dim);
                        let mut out_head = [0.0f32; 64];
                        for qh in qh0..qh1 {
                            let q_head = std::slice::from_raw_parts(qkv_ptr.add(qh * 64), 64);
                            let sc = std::slice::from_raw_parts_mut(scores_ptr.add(qh * max_seq_len), seq_len);
                            let off = (qh / kv_group) * 64;
                            causal_gqa_head64(q_head, k_layer, v_layer, sc, &mut out_head, kv_dim, off, attn_scale);
                            Int8ChunkActivation::populate_chunk_fwht_raw(
                                act_hq,
                                act_hs,
                                act_hsum,
                                qh,
                                &out_head,
                                None,
                            );
                        }
                    }
                }
                barrier.wait();

                // 3. Parallel `o_proj` + Fused Disjoint Residual `hidden += proj_out`
                let (os0, os1) = even_stripes(o_mat.num_stripes, tid, nt);
                if os0 < os1 {
                    // SAFETY: `act_h` immutable; `[os0*16, os1*16)` partitions `proj_out` and `hidden` disjointly.
                    unsafe {
                        let act_h_ref = &*(act_h_addr as *const Int8ChunkActivation);
                        nibble4_sdot_stripe_range_int8(o_mat, act_h_ref, os0, os1, proj_ptr);
                        let r0 = os0 * SIMD_ROW_STRIPE;
                        let r1 = (os1 * SIMD_ROW_STRIPE).min(h_dim);
                        add_inplace(
                            std::slice::from_raw_parts_mut(hidden_ptr.add(r0), r1 - r0),
                            std::slice::from_raw_parts(proj_ptr.add(r0), r1 - r0),
                        );
                    }
                }
                barrier.wait();

                // 4. Zero-Barrier Worker-Local `post_attn_norm + FWHT-64 + INT8` -> L1-Fused 64-Row `[gate; up] -> SwiGLU -> FWHT-64 -> INT8`
                if ic0 < ic1 {
                    // SAFETY: `hidden` is read-only after barrier 3; `(lq, ls, lsum)` are worker-private stack buffers;
                    // each `ic` writes disjoint 64-float slices of `gate_up_buf` and `act_i[ic]`.
                    unsafe {
                        let hidden_ro = std::slice::from_raw_parts(hidden_ptr, h_dim);
                        let inv_rms = 1.0 / (dot_f32_fast(hidden_ro, hidden_ro) / (h_dim as f32) + eps).sqrt();
                        for c in 0..nc_h {
                            Int8ChunkActivation::populate_chunk_fwht_raw(
                                lq.as_mut_ptr(),
                                ls.as_mut_ptr(),
                                lsum.as_mut_ptr(),
                                c,
                                &hidden_ro[c * 64..(c + 1) * 64],
                                Some((&layer.post_attn_norm[c * 64..(c + 1) * 64], inv_rms)),
                            );
                        }
                        let mut sw64 = [0.0f32; 64];
                        for ic in ic0..ic1 {
                            nibble4_sdot_stripe_range_raw(
                                gu_mat,
                                lq.as_ptr(),
                                ls.as_ptr(),
                                lsum.as_ptr(),
                                ic * 4,
                                ic * 4 + 4,
                                gu_ptr,
                            );
                            nibble4_sdot_stripe_range_raw(
                                gu_mat,
                                lq.as_ptr(),
                                ls.as_ptr(),
                                lsum.as_ptr(),
                                (nc_i + ic) * 4,
                                (nc_i + ic) * 4 + 4,
                                gu_ptr,
                            );
                            let g64 = std::slice::from_raw_parts(gu_ptr.add(ic * 64), 64);
                            let u64 = std::slice::from_raw_parts(gu_ptr.add(inter_dim + ic * 64), 64);
                            swiglu_activate(g64, u64, &mut sw64);
                            Int8ChunkActivation::populate_chunk_fwht_raw(
                                act_iq,
                                act_is,
                                act_isum,
                                ic,
                                &sw64,
                                None,
                            );
                        }
                    }
                }
                barrier.wait();

                // 5. Parallel `down_proj` + Fused Disjoint Residual `hidden += proj_out`
                let (ds0, ds1) = even_stripes(down_mat.num_stripes, tid, nt);
                if ds0 < ds1 {
                    unsafe {
                        let act_i_ref = &*(act_i_addr as *const Int8ChunkActivation);
                        nibble4_sdot_stripe_range_int8(down_mat, act_i_ref, ds0, ds1, proj_ptr);
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

            // 6. Zero-Barrier Worker-Local `final_norm + FWHT-64 + INT8` -> Parallel `lm_head`
            let (hs0, hs1) = even_stripes(head_mat.num_stripes, tid, nt);
            if hs0 < hs1 {
                unsafe {
                    let hidden_ro = std::slice::from_raw_parts(hidden_ptr, h_dim);
                    let inv_rms = 1.0 / (dot_f32_fast(hidden_ro, hidden_ro) / (h_dim as f32) + eps).sqrt();
                    for c in 0..nc_h {
                        Int8ChunkActivation::populate_chunk_fwht_raw(
                            lq.as_mut_ptr(),
                            ls.as_mut_ptr(),
                            lsum.as_mut_ptr(),
                            c,
                            &hidden_ro[c * 64..(c + 1) * 64],
                            Some((&final_norm[c * 64..(c + 1) * 64], inv_rms)),
                        );
                    }
                    nibble4_sdot_stripe_range_raw(
                        head_mat,
                        lq.as_ptr(),
                        ls.as_ptr(),
                        lsum.as_ptr(),
                        hs0,
                        hs1,
                        logits_ptr,
                    );
                }
            }
        };

        if let Some(pool) = self.pool.as_ref() {
            // For sub-1B L2/SLC-resident models (`h_dim <= 1024`), 8 P-cores divide `nc_i` (`24` / `40`)
            // and `lm_head` (`1536` pairs) evenly while leaving 2 P-cores free for OS/desktop interrupts.
            let max_decode_nt = if h_dim <= 1024 { 8 } else { usize::MAX };
            pool.execute_up_to(max_decode_nt, kernel);
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
        let h = self.cfg.hidden_size;
        let inter = self.cfg.intermediate_size;
        let kv_dim = self.cfg.kv_dim();
        let qkv_rows = h + 2 * kv_dim;
        let head_dim = self.cfg.head_dim();
        let half_dim = head_dim / 2;
        let kv_group = self.cfg.num_attention_heads / self.cfg.num_key_value_heads;
        let attn_scale = 1.0 / (head_dim as f32).sqrt();
        let layer_kv_stride = self.cfg.max_seq_len * kv_dim;
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
            let row = &mut hidden[t * h..(t + 1) * h];
            if self.embed_f32_rows.is_empty() {
                if let LinearBackend::Nibble4(m) = &self.lm_head {
                    m.dequantize_row_rotated(tok, row);
                    ifwht64_signed_inplace(row);
                }
            } else {
                row.copy_from_slice(&self.embed_f32_rows[tok * h..(tok + 1) * h]);
            }
        }
        let acts_h = &mut ps.acts_h[..n];
        let acts_i = &mut ps.acts_i[..n];
        let qkv = &mut ps.qkv[..n * qkv_rows];
        let attn = &mut ps.attn[..n * h];
        let proj = &mut ps.proj[..n * h];
        let gate_up = &mut ps.gate_up[..n * 2 * inter];
        let sw = &mut ps.sw[..n * inter];
        let score_stride = if head_dim == 64 {
            self.cfg.max_seq_len
        } else {
            kv_group.max(1) * self.cfg.max_seq_len
        };
        let scores_all = &mut ps.scores[..n * score_stride];

        for (l_idx, layer) in self.layers.iter().enumerate() {
            let kv_base = l_idx * layer_kv_stride;
            prompt_populate(pool, hidden, h, Some((&layer.input_norm, eps)), acts_h);
            prompt_gemm(pool, &layer.qkv_proj, acts_h, qkv);

            let k_ptr = self.ws.k_cache.as_mut_ptr() as usize;
            let v_ptr = self.ws.v_cache.as_mut_ptr() as usize;
            let qkv_ptr = qkv.as_mut_ptr() as usize;
            let rope = &self.rope_cos_sin;
            let cfg = &self.cfg;
            par_range(pool, n, |a, b| {
                for t in a..b {
                    let pos = pos0 + t;
                    // SAFETY: token `t` owns row `t` of `qkv` and cache slot `pos`.
                    let row = unsafe {
                        std::slice::from_raw_parts_mut((qkv_ptr as *mut f32).add(t * qkv_rows), qkv_rows)
                    };
                    let rope_row = &rope[pos * head_dim..(pos + 1) * head_dim];
                    let (q, kv) = row.split_at_mut(h);
                    let (k, v) = kv.split_at_mut(kv_dim);
                    apply_llama_rope_precomp(q, cfg.num_attention_heads, head_dim, half_dim, rope_row);
                    apply_llama_rope_precomp(k, cfg.num_key_value_heads, head_dim, half_dim, rope_row);
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

            prompt_populate(pool, attn, h, None, acts_h);
            prompt_gemm(pool, &layer.o_proj, acts_h, proj);
            add_inplace(hidden, proj);

            prompt_populate(pool, hidden, h, Some((&layer.post_attn_norm, eps)), acts_h);
            prompt_gemm(pool, &layer.gate_up_proj, acts_h, gate_up);
            let gu_ref = &*gate_up;
            let sw_ptr = sw.as_mut_ptr() as usize;
            par_range(pool, n, |a, b| {
                for t in a..b {
                    let gu = &gu_ref[t * 2 * inter..(t + 1) * 2 * inter];
                    let (g, u) = gu.split_at(inter);
                    // SAFETY: token `t` owns row `t` of `sw`.
                    let out = unsafe { std::slice::from_raw_parts_mut((sw_ptr as *mut f32).add(t * inter), inter) };
                    swiglu_activate(g, u, out);
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
            if !self.ws.blut_hidden.lut_enabled {
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
    let LinearBackend::Nibble4(m) = mat else {
        unreachable!("prompt_gemm requires SDOT projections");
    };
    debug_assert_eq!(out.len(), acts.len() * m.rows);
    let out_ptr = out.as_mut_ptr() as usize;
    par_range(pool, m.num_stripes, |s0, s1| {
        // SAFETY: disjoint stripe ranges write disjoint rows of every token.
        unsafe { nibble4_sdot_batch_stripe_range(m, acts, s0, s1, out_ptr as *mut f32, m.rows) };
    });
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
}
