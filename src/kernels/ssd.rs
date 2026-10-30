#![forbid(unsafe_code)]
//! Constant-Memory $O(1)$ State-Space Dual (SSD) Recurrent Context Block with Induction Shift.
//!
//! Replaces $O(L)$ growing KV-caches with a fixed-footprint multi-head fast-weight state:
//! $$S_t^{(h)} = \gamma_t^{(h)} S_{t-1}^{(h)} + v_t^{(h)} (k_t^{(h)})^T, \qquad y_t^{(h)} = S_t^{(h)} q_t^{(h)}$$
//!
//! - **Zero-Allocation Hot Path**: All projection buffers and `QuantizedActivationLut` tables live
//!   inside [`SsdScratch`].
//! - **Causal 1-Token Induction Shift**: Odd-numbered heads optionally form keys $k_t^{(h)}$ from
//!   the previous normalized token $x_{t-1}$ while values $v_t^{(h)}$ come from $x_t$, binding
//!   in-context bigrams `(KEY_{t-1} -> VALUE_t)` into the fixed $O(1)$ matrix $S_t$ for arbitrary-distance
//!   associative recall!

use crate::kernels::simd_lut::{simd_lut_gemv, QuantizedActivationLut, SimdPackedMatrix};

/// Fixed-size $O(1)$ recurrent fast-weight state across all heads for one SSD layer.
#[derive(Clone, Debug)]
pub struct SsdState {
    pub num_heads: usize,
    pub head_dim: usize,
    /// Contiguous `[num_heads * head_dim * head_dim]` fast-weight matrices $S_t^{(h)}$.
    pub fast_weights: Vec<f32>,
    /// Single-step causal shift register `[num_heads * head_dim]` for induction-head key binding.
    pub prev_normed: Vec<f32>,
    /// Total tokens folded into this constant-size state.
    pub tokens_seen: usize,
}

impl SsdState {
    pub fn new(num_heads: usize, head_dim: usize) -> Self {
        assert!(num_heads > 0 && head_dim > 0);
        let d_model = num_heads * head_dim;
        Self {
            num_heads,
            head_dim,
            fast_weights: vec![0.0f32; num_heads * head_dim * head_dim],
            prev_normed: vec![0.0f32; d_model],
            tokens_seen: 0,
        }
    }

    /// Resets the recurrent state to zero without reallocating.
    pub fn reset(&mut self) {
        self.fast_weights.fill(0.0);
        self.prev_normed.fill(0.0);
        self.tokens_seen = 0;
    }

    /// Exact dynamic state RAM footprint in bytes (strictly constant for all sequence lengths).
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        (self.fast_weights.len() + self.prev_normed.len()) * std::mem::size_of::<f32>()
    }

    /// Equivalent FP16 attention KV-cache bytes that a standard Transformer layer would
    /// require at `seq_len` (`2 * seq_len * d_model * 2` bytes).
    #[inline]
    pub fn equivalent_fp16_kv_cache_bytes(&self, seq_len: usize) -> usize {
        let d_model = self.num_heads * self.head_dim;
        2 * seq_len * d_model * 2
    }
}

/// Pre-allocated scratch workspace for **zero-heap-allocation** single-token SSD execution.
#[derive(Clone, Debug)]
pub struct SsdScratch {
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub k_shifted: Vec<f32>,
    pub v: Vec<f32>,
    pub y_heads: Vec<f32>,
    pub gammas: Vec<f32>,
    pub qlut: QuantizedActivationLut,
}

impl SsdScratch {
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        Self {
            normed: vec![0.0f32; d_model],
            q: vec![0.0f32; d_model],
            k: vec![0.0f32; d_model],
            k_shifted: vec![0.0f32; d_model],
            v: vec![0.0f32; d_model],
            y_heads: vec![0.0f32; d_model],
            gammas: vec![0.0f32; num_heads],
            qlut: QuantizedActivationLut::new_zeroed(d_model),
        }
    }
}

/// Multi-head State-Space Dual (SSD) layer with ternary SIMD `LUT-GEMV` projections.
#[derive(Clone, Debug)]
pub struct SsdBlock {
    pub d_model: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub enable_induction_shift: bool,
    pub rms_weight: Vec<f32>,
    pub w_q: SimdPackedMatrix,
    pub w_k: SimdPackedMatrix,
    pub w_v: SimdPackedMatrix,
    pub w_o: SimdPackedMatrix,
    pub proj_scale: f32,
    pub w_gamma: Vec<f32>,
    pub b_gamma: Vec<f32>,
}

impl SsdBlock {
    /// Deterministically initializes a multi-head SSD block of width `d_model`.
    pub fn new(d_model: usize, num_heads: usize, seed: u64) -> Self {
        assert!(d_model > 0 && num_heads > 0 && d_model.is_multiple_of(num_heads));
        assert!(d_model.is_multiple_of(4));
        let head_dim = d_model / num_heads;
        let proj_scale = 1.0 / (d_model as f32).sqrt();

        let w_q = Self::init_ternary_matrix(d_model, d_model, seed.wrapping_add(1));
        let w_k = Self::init_ternary_matrix(d_model, d_model, seed.wrapping_add(2));
        let w_v = Self::init_ternary_matrix(d_model, d_model, seed.wrapping_add(3));
        let w_o = Self::init_ternary_matrix(d_model, d_model, seed.wrapping_add(4));

        let mut state = seed.wrapping_add(0x7F4A_7C15);
        let w_gamma: Vec<f32> = (0..num_heads * d_model)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = (z ^ (z >> 31)) as f64 / (u64::MAX as f64);
                ((u as f32) * 2.0 - 1.0) * 0.05
            })
            .collect();

        let b_gamma: Vec<f32> = (0..num_heads)
            .map(|h| 2.8 + (h as f32 / num_heads as f32) * 2.0)
            .collect();

        Self {
            d_model,
            num_heads,
            head_dim,
            enable_induction_shift: true,
            rms_weight: vec![1.0f32; d_model],
            w_q,
            w_k,
            w_v,
            w_o,
            proj_scale,
            w_gamma,
            b_gamma,
        }
    }

    fn init_ternary_matrix(rows: usize, cols: usize, mut state: u64) -> SimdPackedMatrix {
        let weights: Vec<i8> = (0..rows * cols)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z % 3) as i8) - 1
            })
            .collect();
        SimdPackedMatrix::from_ternary_weights(&weights, rows, cols)
    }

    #[inline]
    pub fn rms_norm(&self, x: &[f32], out: &mut [f32]) {
        let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
        let inv_rms = 1.0 / (mean_sq + 1e-5).sqrt();
        for ((o, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(self.rms_weight.iter()) {
            *o = xi * inv_rms * wi;
        }
    }

    /// Computes normalized `q_t`, `k_t` (with optional causal 1-token shift on odd heads),
    /// `v_t`, and per-head decay gates `gamma_t` using **zero heap allocations**.
    fn project_qkv_and_gates(&self, x: &[f32], prev_normed: &[f32], scratch: &mut SsdScratch) {
        self.rms_norm(x, &mut scratch.normed);

        scratch.qlut.populate(&scratch.normed);
        simd_lut_gemv(&self.w_q, &scratch.qlut, &mut scratch.q);
        simd_lut_gemv(&self.w_k, &scratch.qlut, &mut scratch.k);
        simd_lut_gemv(&self.w_v, &scratch.qlut, &mut scratch.v);

        if self.enable_induction_shift && self.num_heads > 1 {
            scratch.qlut.populate(prev_normed);
            simd_lut_gemv(&self.w_k, &scratch.qlut, &mut scratch.k_shifted);
            // Odd heads bind keys from `t - 1` (`prev_normed`) to values at `t`!
            for h in (1..self.num_heads).step_by(2) {
                let start = h * self.head_dim;
                let end = start + self.head_dim;
                scratch.k[start..end].copy_from_slice(&scratch.k_shifted[start..end]);
            }
        }

        for v in &mut scratch.v {
            *v *= self.proj_scale;
        }

        let inv_sqrt_dh = 1.0 / (self.head_dim as f32).sqrt();
        for (h, (gamma_slot, &b_g)) in scratch.gammas[..self.num_heads]
            .iter_mut()
            .zip(self.b_gamma.iter())
            .enumerate()
        {
            let start = h * self.head_dim;
            let end = start + self.head_dim;

            let q_slice = &mut scratch.q[start..end];
            let q_norm = q_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
            if q_norm > 1e-6 {
                let s = inv_sqrt_dh / q_norm;
                for qv in q_slice.iter_mut() {
                    *qv *= s;
                }
            }

            let k_slice = &mut scratch.k[start..end];
            let k_norm = k_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
            if k_norm > 1e-6 {
                let inv_kn = 1.0 / k_norm;
                for kv in k_slice.iter_mut() {
                    *kv *= inv_kn;
                }
            }

            let w_g = &self.w_gamma[h * self.d_model..(h + 1) * self.d_model];
            let logit: f32 = w_g
                .iter()
                .zip(scratch.normed.iter())
                .map(|(w, n)| w * n)
                .sum::<f32>()
                + b_g;
            *gamma_slot = 1.0 / (1.0 + (-logit).exp());
        }
    }

    /// Executes a single autoregressive step in constant $O(1)$ state memory with **zero heap allocations**.
    pub fn step_recurrent(
        &self,
        x: &[f32],
        state: &mut SsdState,
        scratch: &mut SsdScratch,
        delta_out: &mut [f32],
    ) {
        assert_eq!(x.len(), self.d_model);
        assert_eq!(delta_out.len(), self.d_model);
        assert_eq!(state.num_heads, self.num_heads);
        assert_eq!(state.head_dim, self.head_dim);

        self.project_qkv_and_gates(x, &state.prev_normed, scratch);
        state.prev_normed.copy_from_slice(&scratch.normed);

        let dh = self.head_dim;
        let head_mat_len = dh * dh;

        for (h, &gamma) in scratch.gammas[..self.num_heads].iter().enumerate() {
            let h_base = h * dh;
            let q_h = &scratch.q[h_base..h_base + dh];
            let k_h = &scratch.k[h_base..h_base + dh];
            let v_h = &scratch.v[h_base..h_base + dh];
            let y_h = &mut scratch.y_heads[h_base..h_base + dh];
            let s_mat = &mut state.fast_weights[h * head_mat_len..(h + 1) * head_mat_len];

            for (i, (&vi, yi)) in v_h.iter().zip(y_h.iter_mut()).enumerate() {
                let s_row = &mut s_mat[i * dh..(i + 1) * dh];
                let mut acc = 0.0f32;
                for ((s_ij, &kj), &qj) in s_row.iter_mut().zip(k_h.iter()).zip(q_h.iter()) {
                    let updated = gamma * (*s_ij) + vi * kj;
                    *s_ij = updated;
                    acc += updated * qj;
                }
                *yi = acc;
            }
        }

        state.tokens_seen += 1;

        scratch.qlut.populate(&scratch.y_heads);
        simd_lut_gemv(&self.w_o, &scratch.qlut, delta_out);
        for d in delta_out.iter_mut() {
            *d *= self.proj_scale;
        }
    }

    /// Reference causal parallel semi-separable attention dual over `xs` (`[T][d_model]`),
    /// proving exact mathematical duality with [`Self::step_recurrent`].
    pub fn forward_parallel_dual(&self, xs: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let seq_len = xs.len();
        let dh = self.head_dim;
        let mut scratch = SsdScratch::new(self.d_model, self.num_heads);
        let mut prev_normed = vec![0.0f32; self.d_model];

        let mut all_q = Vec::with_capacity(seq_len);
        let mut all_k = Vec::with_capacity(seq_len);
        let mut all_v = Vec::with_capacity(seq_len);
        let mut all_gamma = Vec::with_capacity(seq_len);

        for x in xs {
            self.project_qkv_and_gates(x, &prev_normed, &mut scratch);
            prev_normed.copy_from_slice(&scratch.normed);
            all_q.push(scratch.q.clone());
            all_k.push(scratch.k.clone());
            all_v.push(scratch.v.clone());
            all_gamma.push(scratch.gammas.clone());
        }

        let mut outputs = Vec::with_capacity(seq_len);
        for (t, q_t) in all_q.iter().enumerate() {
            let mut y_concat = vec![0.0f32; self.d_model];
            for h in 0..self.num_heads {
                let h_start = h * dh;
                let h_end = h_start + dh;
                let q_th = &q_t[h_start..h_end];

                for s in 0..=t {
                    let mut decay = 1.0f32;
                    for gamma_step in all_gamma.iter().take(t + 1).skip(s + 1) {
                        decay *= gamma_step[h];
                    }
                    let k_sh = &all_k[s][h_start..h_end];
                    let v_sh = &all_v[s][h_start..h_end];
                    let qk_dot: f32 = q_th.iter().zip(k_sh.iter()).map(|(q, k)| q * k).sum();
                    let attn_weight = decay * qk_dot;

                    for (yi, &vi) in y_concat[h_start..h_end].iter_mut().zip(v_sh.iter()) {
                        *yi += attn_weight * vi;
                    }
                }
            }

            scratch.qlut.populate(&y_concat);
            let mut delta_out = vec![0.0f32; self.d_model];
            simd_lut_gemv(&self.w_o, &scratch.qlut, &mut delta_out);
            for d in &mut delta_out {
                *d *= self.proj_scale;
            }
            outputs.push(delta_out);
        }

        outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_space_duality_recurrence_equals_parallel_attention() {
        let d_model = 128;
        let num_heads = 4;
        let seq_len = 32;

        let ssd = SsdBlock::new(d_model, num_heads, 2026);
        let xs: Vec<Vec<f32>> = (0..seq_len)
            .map(|t| {
                (0..d_model)
                    .map(|i| (((t * 17 + i * 7) as f32) * 0.09).sin())
                    .collect()
            })
            .collect();

        let dual_outputs = ssd.forward_parallel_dual(&xs);

        let mut state = SsdState::new(num_heads, d_model / num_heads);
        let initial_bytes = state.memory_bytes();
        let mut scratch = SsdScratch::new(d_model, num_heads);
        let mut rec_out = vec![0.0f32; d_model];

        for (t, x) in xs.iter().enumerate() {
            ssd.step_recurrent(x, &mut state, &mut scratch, &mut rec_out);
            assert_eq!(state.memory_bytes(), initial_bytes);

            for c in 0..d_model {
                let diff = (rec_out[c] - dual_outputs[t][c]).abs();
                assert!(
                    diff < 1e-4,
                    "step {t} channel {c} duality drift: rec={} dual={} diff={diff}",
                    rec_out[c],
                    dual_outputs[t][c]
                );
            }
        }
        assert_eq!(state.tokens_seen, seq_len);
    }

    #[test]
    fn test_constant_memory_stability_over_2000_tokens() {
        let d_model = 256;
        let num_heads = 8;
        let ssd = SsdBlock::new(d_model, num_heads, 999);
        let mut state = SsdState::new(num_heads, d_model / num_heads);
        let mut scratch = SsdScratch::new(d_model, num_heads);
        let mut out = vec![0.0f32; d_model];

        let fixed_bytes = state.memory_bytes();
        let token_vec: Vec<f32> = (0..d_model).map(|i| ((i as f32) * 0.05).cos()).collect();
        for _ in 0..2_000 {
            ssd.step_recurrent(&token_vec, &mut state, &mut scratch, &mut out);
        }

        assert_eq!(state.tokens_seen, 2_000);
        assert_eq!(state.memory_bytes(), fixed_bytes);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
