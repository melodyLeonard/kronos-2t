#![forbid(unsafe_code)]
//! Weight Compiler: BitNet $b1.58$ Ternary Quantizer & Multi-Rank Kronecker `.k2t` Factorizer.
//!
//! Bridges standard dense `FP32` neural network weights to `kronos-2t` execution primitives:
//! 1. [`compile_bitnet_ternary`]: AbsMean ternary quantization ($\{-1, 0, +1\} \times \beta$) into [`SimdPackedMatrix`].
//! 2. [`compile_kronecker_matrix`]: Alternating Coordinate Descent / Matching Pursuit factorizer
//!    decomposing any dense matrix $W \in \mathbb{R}^{R \times C}$ into multi-rank XOR-codebook seeds
//!    for zero-buffer register execution via [`crate::simd_lut::fused_kronecker_gemv`].

use crate::hadamard::{fwht64_signed_inplace, HADAMARD_BLOCK};
use crate::k2t_format::CompiledSwiGluExpert;
use crate::lut_gemv::GROUP_SIZE;
use crate::simd_lut::{
    bf16_to_f32, decode_nibble_code, default_register_codebook_64, f32_to_bf16,
    FusedBase3LutMatrix, FusedBipolarLutMatrix, MultiPlaneTernaryMatrix, SimdPackedMatrix,
    QUANT_CHUNK_BLOCKS, SIMD_ROW_STRIPE,
};

/// Output of multi-rank Kronecker compilation for a single `[rows, cols]` matrix.
#[derive(Clone, Debug)]
pub struct CompiledKroneckerMatrix {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub scales: Vec<f32>,
    pub row_seeds: Vec<u8>,
    pub col_seeds: Vec<u8>,
    pub relative_frobenius_error: f32,
}

/// Quantizes a dense `FP32` matrix `[rows, cols]` via BitNet $b1.58$ AbsMean scaling:
/// $\beta = \frac{1}{RC}\sum |W_{i,j}|$, $\tilde{W}_{i,j} = \text{round}(\text{clamp}(W_{i,j}/\beta, -1, 1))$.
pub fn compile_bitnet_ternary(weights: &[f32], rows: usize, cols: usize) -> (SimdPackedMatrix, f32) {
    assert_eq!(weights.len(), rows * cols);
    let mean_abs = weights.iter().map(|v| v.abs()).sum::<f32>() / (weights.len() as f32);
    if mean_abs <= f32::EPSILON {
        let zeros = vec![0i8; rows * cols];
        return (SimdPackedMatrix::from_ternary_weights(&zeros, rows, cols), 0.0);
    }

    let inv_beta = 1.0 / mean_abs;
    let ternary: Vec<i8> = weights
        .iter()
        .map(|&w| (w * inv_beta).round().clamp(-1.0, 1.0) as i8)
        .collect();

    // Refine least-squares scale beta* = <W, T> / <T, T>
    let mut dot_wt = 0.0f32;
    let mut dot_tt = 0.0f32;
    for (&w, &t) in weights.iter().zip(ternary.iter()) {
        let tf = t as f32;
        dot_wt += w * tf;
        dot_tt += tf * tf;
    }
    let optimal_beta = if dot_tt > 0.0 { dot_wt / dot_tt } else { 0.0 };

    (
        SimdPackedMatrix::from_ternary_weights(&ternary, rows, cols),
        optimal_beta,
    )
}

/// Wraps single tensor-wide [`compile_bitnet_ternary`] (`1.58`-bit global `AbsMean` PTQ) into a
/// 1-plane [`MultiPlaneTernaryMatrix`] and returns `(matrix, relative_frobenius_error)`.
pub fn compile_global_ternary_as_multiplane(
    weights: &[f32],
    rows: usize,
    cols: usize,
) -> (MultiPlaneTernaryMatrix, f32) {
    let (plane, beta) = compile_bitnet_ternary(weights, rows, cols);
    let beta_bf16 = f32_to_bf16(beta);
    let beta_q = bf16_to_f32(beta_bf16);

    let bpr = plane.blocks_per_row;
    let num_stripes = plane.num_stripes;
    let num_chunks = bpr.div_ceil(QUANT_CHUNK_BLOCKS);
    let mut bf16_chunk_scales = vec![0u16; num_stripes * num_chunks * SIMD_ROW_STRIPE];

    for r in 0..rows {
        let s = r / SIMD_ROW_STRIPE;
        let lane = r % SIMD_ROW_STRIPE;
        for c in 0..num_chunks {
            bf16_chunk_scales[(s * num_chunks + c) * SIMD_ROW_STRIPE + lane] = beta_bf16;
        }
    }

    // Compute exact relative Frobenius error of global 1-plane ternary approximation
    let orig_sq: f64 = weights.iter().map(|&w| (w as f64) * (w as f64)).sum();
    let inv_beta = if beta > f32::EPSILON { 1.0 / beta } else { 0.0 };
    // Re-evaluate against actual `compile_bitnet_ternary` trits (`mean_abs` rounding)
    let mean_abs = weights.iter().map(|v| v.abs()).sum::<f32>() / (weights.len() as f32);
    let inv_mean = if mean_abs > f32::EPSILON {
        1.0 / mean_abs
    } else {
        0.0
    };
    let _ = inv_beta;
    let err_sq: f64 = weights
        .iter()
        .map(|&w| {
            let t = (w * inv_mean).round().clamp(-1.0, 1.0);
            let diff = (w - beta_q * t) as f64;
            diff * diff
        })
        .sum();
    let rel_err = if orig_sq > 0.0 {
        (err_sq / orig_sq).sqrt() as f32
    } else {
        0.0
    };

    (
        MultiPlaneTernaryMatrix {
            rows,
            cols,
            num_planes: 1,
            blocks_per_row: bpr,
            num_stripes,
            num_chunks,
            planes: vec![plane],
            bf16_chunk_scales,
        },
        rel_err,
    )
}

/// Solves the **exact global $L_2$-optimal ternary projection** $\min_{\beta, t \in \{-1,0,1\}^L} \|r - \beta t\|_2^2$
/// in $O(L \log L)$ via sorted magnitude prefix-sum scanning, snapping $\beta^*$ to `BF16`.
#[inline]
fn solve_exact_l2_ternary_chunk(res: &[f32], out_trits: &mut [i8]) -> (u16, f32) {
    const MAX_CHUNK_ELEMS: usize = QUANT_CHUNK_BLOCKS * GROUP_SIZE;
    let len = res.len();
    debug_assert!(len <= MAX_CHUNK_ELEMS && len == out_trits.len());

    let mut pairs = [(0.0f32, 0usize); MAX_CHUNK_ELEMS];
    for (i, &val) in res.iter().enumerate() {
        pairs[i] = (val.abs(), i);
    }
    let active = &mut pairs[..len];
    active.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    if active[0].0 <= 1e-12 {
        out_trits.fill(0);
        return (0, 0.0);
    }

    let mut prefix_sum = 0.0f32;
    let mut best_score = -1.0f32;
    let mut best_sum = 0.0f32;
    let mut best_k = 1usize;

    for (idx, &(mag, _)) in active.iter().enumerate() {
        if mag <= 1e-12 {
            break;
        }
        let k = idx + 1;
        prefix_sum += mag;
        let score = (prefix_sum * prefix_sum) / (k as f32);
        if score > best_score {
            best_score = score;
            best_sum = prefix_sum;
            best_k = k;
        }
    }

    let beta_raw = best_sum / (best_k as f32);
    let beta_bf16 = f32_to_bf16(beta_raw);
    let beta_q = bf16_to_f32(beta_bf16);

    out_trits.fill(0);
    if beta_q > 0.0 {
        for &(_, pos) in &active[..best_k] {
            out_trits[pos] = if res[pos] >= 0.0 { 1 } else { -1 };
        }
    }

    (beta_bf16, beta_q)
}

/// Compiles any dense `FP32` weight matrix `[rows, cols]` into `num_planes` (`1..=4`) group-64
/// ternary planes (`MultiPlaneTernaryMatrix`) via Exact $L_2$ Residual Projection + Gauss-Seidel
/// Coordinate Refinement. Returns `(compiled_matrix, relative_frobenius_error)`.
pub fn compile_multiplane_ternary(
    weights: &[f32],
    rows: usize,
    cols: usize,
    num_planes: usize,
) -> (MultiPlaneTernaryMatrix, f32) {
    const CHUNK_ELEMS: usize = QUANT_CHUNK_BLOCKS * GROUP_SIZE;
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(GROUP_SIZE));
    assert!((1..=4).contains(&num_planes));

    let bpr = cols / GROUP_SIZE;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = bpr.div_ceil(QUANT_CHUNK_BLOCKS);

    let mut plane_trits: Vec<Vec<i8>> = (0..num_planes).map(|_| vec![0i8; rows * cols]).collect();
    let mut bf16_chunk_scales = vec![0u16; num_planes * num_stripes * num_chunks * SIMD_ROW_STRIPE];

    let mut orig_norm_sq = 0.0f64;
    let mut res_norm_sq = 0.0f64;

    let mut res_buf = [0.0f32; CHUNK_ELEMS];
    let mut local_trits = [[0i8; CHUNK_ELEMS]; 4];
    let mut local_bf16 = [0u16; 4];
    let mut local_betas = [0.0f32; 4];

    for r in 0..rows {
        let stripe = r / SIMD_ROW_STRIPE;
        let lane = r % SIMD_ROW_STRIPE;
        let row_w = &weights[r * cols..(r + 1) * cols];

        for c in 0..num_chunks {
            let col_start = c * CHUNK_ELEMS;
            let col_end = (col_start + CHUNK_ELEMS).min(cols);
            let len = col_end - col_start;
            let sub_w = &row_w[col_start..col_end];

            for &v in sub_w {
                orig_norm_sq += (v as f64) * (v as f64);
            }
            res_buf[..len].copy_from_slice(sub_w);

            // Greedy exact L2 pass across planes 0..num_planes
            for p in 0..num_planes {
                let (b16, bq) =
                    solve_exact_l2_ternary_chunk(&res_buf[..len], &mut local_trits[p][..len]);
                local_bf16[p] = b16;
                local_betas[p] = bq;
                if bq != 0.0 {
                    for i in 0..len {
                        res_buf[i] -= bq * (local_trits[p][i] as f32);
                    }
                }
            }

            // 2 sweeps of Gauss-Seidel coordinate refinement when num_planes >= 2
            if num_planes >= 2 {
                for _sweep in 0..2 {
                    for p in 0..num_planes {
                        let old_b = local_betas[p];
                        if old_b != 0.0 {
                            for i in 0..len {
                                res_buf[i] += old_b * (local_trits[p][i] as f32);
                            }
                        }
                        let (b16, bq) = solve_exact_l2_ternary_chunk(
                            &res_buf[..len],
                            &mut local_trits[p][..len],
                        );
                        local_bf16[p] = b16;
                        local_betas[p] = bq;
                        if bq != 0.0 {
                            for i in 0..len {
                                res_buf[i] -= bq * (local_trits[p][i] as f32);
                            }
                        }
                    }
                }
            }

            for &rv in &res_buf[..len] {
                res_norm_sq += (rv as f64) * (rv as f64);
            }

            for p in 0..num_planes {
                plane_trits[p][r * cols + col_start..r * cols + col_end]
                    .copy_from_slice(&local_trits[p][..len]);
                let scale_idx =
                    ((p * num_stripes + stripe) * num_chunks + c) * SIMD_ROW_STRIPE + lane;
                bf16_chunk_scales[scale_idx] = local_bf16[p];
            }
        }
    }

    let planes: Vec<SimdPackedMatrix> = plane_trits
        .iter()
        .map(|trits| SimdPackedMatrix::from_ternary_weights(trits, rows, cols))
        .collect();

    let rel_err = if orig_norm_sq > 0.0 {
        (res_norm_sq / orig_norm_sq).sqrt() as f32
    } else {
        0.0
    };

    (
        MultiPlaneTernaryMatrix {
            rows,
            cols,
            num_planes,
            blocks_per_row: bpr,
            num_stripes,
            num_chunks,
            planes,
            bf16_chunk_scales,
        },
        rel_err,
    )
}

/// Pre-factored upper-triangular Cholesky factor $U_c$ of $H_c^{-1} = U_c^T U_c$ for each 64-col
/// Hadamard-rotated activation block $c \in 0..\text{num\_chunks}$.
#[derive(Clone, Debug)]
pub struct Block64GptqHessian {
    pub cols: usize,
    pub num_chunks: usize,
    raw_h: Vec<[f64; HADAMARD_BLOCK * HADAMARD_BLOCK]>,
    pub h_inv_chol_upper: Vec<[f32; HADAMARD_BLOCK * HADAMARD_BLOCK]>,
    sample_count: usize,
}

impl Block64GptqHessian {
    pub fn new(cols: usize) -> Self {
        assert!(cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
        let num_chunks = cols / HADAMARD_BLOCK;
        Self {
            cols,
            num_chunks,
            raw_h: vec![[0.0f64; HADAMARD_BLOCK * HADAMARD_BLOCK]; num_chunks],
            h_inv_chol_upper: vec![[0.0f32; HADAMARD_BLOCK * HADAMARD_BLOCK]; num_chunks],
            sample_count: 0,
        }
    }

    /// Accumulates a single activation vector `x` (rotates a copy via `fwht64_signed_inplace` first!).
    pub fn observe_activation(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.cols);
        let mut x_rot = x.to_vec();
        fwht64_signed_inplace(&mut x_rot);

        for (c, h_blk) in self.raw_h.iter_mut().enumerate() {
            let sub = &x_rot[c * HADAMARD_BLOCK..(c + 1) * HADAMARD_BLOCK];
            for i in 0..HADAMARD_BLOCK {
                let xi = sub[i] as f64;
                let row = &mut h_blk[i * HADAMARD_BLOCK..(i + 1) * HADAMARD_BLOCK];
                for (slot, &xj) in row[i..].iter_mut().zip(sub[i..].iter()) {
                    *slot += xi * (xj as f64);
                }
            }
        }
        self.sample_count += 1;
    }

    /// Finalizes damped $H_c^{-1} = U_c^T U_c$ upper-triangular Cholesky factors across all 64-col blocks.
    pub fn finalize(&mut self, damp_ratio: f64) {
        let denom = (self.sample_count.max(1)) as f64;
        for (h_raw, u_out) in self.raw_h.iter().zip(self.h_inv_chol_upper.iter_mut()) {
            let mut h = [0.0f64; HADAMARD_BLOCK * HADAMARD_BLOCK];
            let mut trace = 0.0f64;
            for i in 0..HADAMARD_BLOCK {
                for j in i..HADAMARD_BLOCK {
                    let v = h_raw[i * HADAMARD_BLOCK + j] / denom;
                    h[i * HADAMARD_BLOCK + j] = v;
                    h[j * HADAMARD_BLOCK + i] = v;
                }
                trace += h[i * HADAMARD_BLOCK + i];
            }
            let damp = (damp_ratio * (trace / (HADAMARD_BLOCK as f64))).max(1e-6);
            for i in 0..HADAMARD_BLOCK {
                h[i * HADAMARD_BLOCK + i] += damp;
            }

            // Invert 64x64 SPD `h` via Gauss-Jordan in `f64`
            let mut inv = [0.0f64; HADAMARD_BLOCK * HADAMARD_BLOCK];
            for i in 0..HADAMARD_BLOCK {
                inv[i * HADAMARD_BLOCK + i] = 1.0;
            }
            let mut a = h;
            for k in 0..HADAMARD_BLOCK {
                let piv = a[k * HADAMARD_BLOCK + k].max(1e-15);
                let inv_piv = 1.0 / piv;
                for j in 0..HADAMARD_BLOCK {
                    a[k * HADAMARD_BLOCK + j] *= inv_piv;
                    inv[k * HADAMARD_BLOCK + j] *= inv_piv;
                }
                for i in 0..HADAMARD_BLOCK {
                    if i == k {
                        continue;
                    }
                    let factor = a[i * HADAMARD_BLOCK + k];
                    if factor == 0.0 {
                        continue;
                    }
                    for j in 0..HADAMARD_BLOCK {
                        a[i * HADAMARD_BLOCK + j] -= factor * a[k * HADAMARD_BLOCK + j];
                        inv[i * HADAMARD_BLOCK + j] -= factor * inv[k * HADAMARD_BLOCK + j];
                    }
                }
            }

            // Symmetrize `inv` and compute Cholesky `inv = L * L^T`, then `U = L^T`!
            let mut l_mat = [0.0f64; HADAMARD_BLOCK * HADAMARD_BLOCK];
            for i in 0..HADAMARD_BLOCK {
                for j in 0..=i {
                    let sym = 0.5 * (inv[i * HADAMARD_BLOCK + j] + inv[j * HADAMARD_BLOCK + i]);
                    let mut sum = sym;
                    for p in 0..j {
                        sum -= l_mat[i * HADAMARD_BLOCK + p] * l_mat[j * HADAMARD_BLOCK + p];
                    }
                    if i == j {
                        l_mat[i * HADAMARD_BLOCK + i] = sum.max(1e-15).sqrt();
                    } else {
                        l_mat[i * HADAMARD_BLOCK + j] = sum / l_mat[j * HADAMARD_BLOCK + j];
                    }
                }
            }

            u_out.fill(0.0);
            for i in 0..HADAMARD_BLOCK {
                for j in i..HADAMARD_BLOCK {
                    u_out[i * HADAMARD_BLOCK + j] = l_mat[j * HADAMARD_BLOCK + i] as f32;
                }
            }
        }
    }
}

/// Compiles dense weights `[rows, cols]` into a **Hadamard-Rotated + GPTQ Error-Compensated**
/// [`FusedBase3LutMatrix`] (`1` plane = `2.25b`, `2` planes = **`4.25b`**, `3` planes = **`6.25b`**)!
pub fn compile_hadamard_gptq_base3(
    weights: &[f32],
    rows: usize,
    cols: usize,
    num_planes: usize,
    hessian: Option<&Block64GptqHessian>,
) -> FusedBase3LutMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
    assert!((1..=3).contains(&num_planes));

    let bpr = cols / GROUP_SIZE;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / HADAMARD_BLOCK;
    let max_int: i32 = match num_planes {
        1 => 1,
        2 => 4,  // 3^2 = 9 levels: {-4..+4}
        3 => 13, // 3^3 = 27 levels: {-13..+13}
        _ => unreachable!(),
    };

    let mut plane_trits: Vec<Vec<i8>> = (0..num_planes).map(|_| vec![0i8; rows * cols]).collect();
    let mut bf16_chunk_scales = vec![0u16; num_stripes * num_chunks * SIMD_ROW_STRIPE];

    let mut row_rot = vec![0.0f32; cols];
    let mut work = [0.0f32; HADAMARD_BLOCK];
    let mut best_q = [0i32; HADAMARD_BLOCK];
    let mut cand_q = [0i32; HADAMARD_BLOCK];
    let ratios = [0.76f32, 0.83, 0.89, 0.95, 1.00];

    for r in 0..rows {
        let stripe = r / SIMD_ROW_STRIPE;
        let lane = r % SIMD_ROW_STRIPE;
        row_rot.copy_from_slice(&weights[r * cols..(r + 1) * cols]);
        fwht64_signed_inplace(&mut row_rot);

        for c in 0..num_chunks {
            let sub = &row_rot[c * HADAMARD_BLOCK..(c + 1) * HADAMARD_BLOCK];
            let max_abs = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= 1e-12 {
                continue;
            }

            let mut best_loss = f32::INFINITY;
            let mut best_bf16 = 0u16;
            let u_opt = hessian.map(|h| &h.h_inv_chol_upper[c]);

            for &rat in &ratios {
                let delta_raw = (max_abs * rat) / (max_int as f32);
                let d_bf16 = f32_to_bf16(delta_raw);
                let delta = bf16_to_f32(d_bf16);
                if delta <= 0.0 {
                    continue;
                }
                let inv_delta = 1.0 / delta;

                work.copy_from_slice(sub);
                let mut loss = 0.0f32;

                if let Some(u_mat) = u_opt {
                    for j in 0..HADAMARD_BLOCK {
                        let wj = work[j];
                        let q = (wj * inv_delta).round().clamp(-(max_int as f32), max_int as f32)
                            as i32;
                        cand_q[j] = q;
                        let w_hat = (q as f32) * delta;
                        let u_jj = u_mat[j * HADAMARD_BLOCK + j].max(1e-12);
                        let err = (wj - w_hat) / u_jj;
                        loss += err * err;

                        let u_row = &u_mat[j * HADAMARD_BLOCK..(j + 1) * HADAMARD_BLOCK];
                        for k in (j + 1)..HADAMARD_BLOCK {
                            work[k] -= err * u_row[k];
                        }
                    }
                } else {
                    for j in 0..HADAMARD_BLOCK {
                        let wj = work[j];
                        let q = (wj * inv_delta).round().clamp(-(max_int as f32), max_int as f32)
                            as i32;
                        cand_q[j] = q;
                        let diff = wj - (q as f32) * delta;
                        loss += diff * diff;
                    }
                }

                if loss < best_loss {
                    best_loss = loss;
                    best_bf16 = d_bf16;
                    best_q = cand_q;
                }
            }

            bf16_chunk_scales[(stripe * num_chunks + c) * SIMD_ROW_STRIPE + lane] = best_bf16;
            let col_base = r * cols + c * HADAMARD_BLOCK;
            for (j, &q) in best_q.iter().enumerate() {
                match num_planes {
                    1 => {
                        plane_trits[0][col_base + j] = q as i8;
                    }
                    2 => {
                        let u = q + 4; // 0..=8
                        plane_trits[0][col_base + j] = ((u / 3) - 1) as i8;
                        plane_trits[1][col_base + j] = ((u % 3) - 1) as i8;
                    }
                    3 => {
                        let u = q + 13; // 0..=26
                        plane_trits[0][col_base + j] = ((u / 9) - 1) as i8;
                        plane_trits[1][col_base + j] = (((u / 3) % 3) - 1) as i8;
                        plane_trits[2][col_base + j] = ((u % 3) - 1) as i8;
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    let planes: Vec<SimdPackedMatrix> = plane_trits
        .iter()
        .map(|trits| SimdPackedMatrix::from_ternary_weights(trits, rows, cols))
        .collect();

    FusedBase3LutMatrix {
        rows,
        cols,
        num_planes,
        blocks_per_row: bpr,
        num_stripes,
        num_chunks,
        planes,
        bf16_chunk_scales,
    }
}

/// Full-row ($C \times C$) Hadamard-rotated activation covariance & upper-triangular inverse-Cholesky
/// factor $U \in \mathbb{R}^{C \times C}$ ($U^T U = H^{-1}$) for exact **Lazy Block-64 Full-Row `GPTQ`**!
///
/// Uses the reversed-index identity $\tilde{H} = P H P = L L^T \implies U = P L^{-1} P$, computing
/// $U$ in $O(C^3 / 3)$ triangular flops while propagating quantization error across **all** chunks
/// $0 \dots \text{num\_chunks}-1$ of every row!
#[derive(Clone, Debug)]
pub struct FullRowGptqHessian {
    pub cols: usize,
    raw_h: Vec<f64>,
    pub h_inv_chol_upper: Vec<f32>,
    sample_count: usize,
    /// Rotated activations awaiting a batched `X^T X` flush (`pending_rows x cols`).
    pending: Vec<f32>,
}

/// Activations buffered per batched Hessian flush (`f32` partial sums stay exact enough at this size).
const HESSIAN_FLUSH_ROWS: usize = 256;

impl FullRowGptqHessian {
    pub fn new(cols: usize) -> Self {
        assert!(cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
        Self {
            cols,
            raw_h: vec![0.0f64; cols * cols],
            h_inv_chol_upper: Vec::new(),
            sample_count: 0,
            pending: Vec::with_capacity(HESSIAN_FLUSH_ROWS * cols),
        }
    }

    /// Accumulates a single input activation `x` after rotating it via `fwht64_signed_inplace`.
    pub fn observe_activation(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.cols);
        let mut x_rot = x.to_vec();
        fwht64_signed_inplace(&mut x_rot);
        self.observe_prerotated(&x_rot);
    }

    /// Accumulates an already-Hadamard-rotated activation vector `x_rot`.
    pub fn observe_prerotated(&mut self, x_rot: &[f32]) {
        assert_eq!(x_rot.len(), self.cols);
        self.pending.extend_from_slice(x_rot);
        self.sample_count += 1;
        if self.pending.len() >= HESSIAN_FLUSH_ROWS * self.cols {
            self.flush_pending();
        }
    }

    /// Adds the upper triangle of `X^T X` for all buffered rows into `raw_h`, splitting rows `i`
    /// round-robin across threads (balances the triangle) with contiguous `f32` SIMD inner loops.
    fn flush_pending(&mut self) {
        let c = self.cols;
        let n = self.pending.len() / c;
        if n == 0 {
            return;
        }
        let xs = &self.pending;
        let workers = std::thread::available_parallelism().map_or(1, |p| p.get()).clamp(1, 8);
        let partials: Vec<Vec<(usize, Vec<f32>)>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..workers)
                .map(|w| {
                    s.spawn(move || {
                        let mut out = Vec::new();
                        for i in (w..c).step_by(workers) {
                            let mut acc = vec![0.0f32; c - i];
                            for t in 0..n {
                                let x = &xs[t * c..(t + 1) * c];
                                let xi = x[i];
                                if xi != 0.0 {
                                    for (a, &xj) in acc.iter_mut().zip(x[i..].iter()) {
                                        *a += xi * xj;
                                    }
                                }
                            }
                            out.push((i, acc));
                        }
                        out
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (i, acc) in partials.into_iter().flatten() {
            for (slot, &v) in self.raw_h[i * c + i..(i + 1) * c].iter_mut().zip(acc.iter()) {
                *slot += v as f64;
            }
        }
        self.pending.clear();
    }

    /// Computes $U = P L^{-1} P$ (`U^T U = (H + damp * I)^{-1}`) in $O(C^3 / 3)$ `f64` flops.
    pub fn finalize(&mut self, damp_ratio: f64) {
        self.flush_pending();
        self.pending = Vec::new();
        let c = self.cols;
        let denom = (self.sample_count.max(1)) as f64;

        let mut trace = 0.0f64;
        for i in 0..c {
            trace += self.raw_h[i * c + i] / denom;
        }
        let mean_diag = (trace / (c as f64)).max(1e-8);
        // Ledoit-Wolf diagonal shrinkage prevents Marchenko-Pastur overfitting when `sample_count < cols`
        // while preserving exact per-channel diagonal energy `H[i, i]`!
        let lw_shrink = if self.sample_count < c {
            (damp_ratio * 3.0).clamp(0.08, 0.35)
        } else {
            damp_ratio.clamp(0.01, 0.15)
        };
        let off_diag_scale = 1.0 - lw_shrink;
        let iso_floor = (0.005 * mean_diag).max(1e-6);

        // Build reversed SPD matrix `h_rev[i, j] = H_lw[c - 1 - i, c - 1 - j]` (lower triangle needed)
        let mut l_mat = vec![0.0f64; c * c];
        for i in 0..c {
            let orig_i = c - 1 - i;
            for j in 0..=i {
                let orig_j = c - 1 - j;
                let (lo, hi) = if orig_i <= orig_j {
                    (orig_i, orig_j)
                } else {
                    (orig_j, orig_i)
                };
                let raw_val = self.raw_h[lo * c + hi] / denom;
                let val = if i == j {
                    raw_val + iso_floor
                } else {
                    raw_val * off_diag_scale
                };
                let sum_prev: f64 = l_mat[i * c..i * c + j]
                    .iter()
                    .zip(&l_mat[j * c..j * c + j])
                    .map(|(a, b)| a * b)
                    .sum();
                let rem = val - sum_prev;
                if i == j {
                    l_mat[i * c + i] = rem.max(1e-15).sqrt();
                } else {
                    l_mat[i * c + j] = rem / l_mat[j * c + j];
                }
            }
        }

        // Compute `mt[j * c + i] = (L^{-1})_{i, j}` (`j <= i`) via contiguous dot products!
        let mut mt = vec![0.0f64; c * c];
        for i in 0..c {
            let inv_lii = 1.0 / l_mat[i * c + i];
            mt[i * c + i] = inv_lii;
            for j in 0..i {
                let dot: f64 = l_mat[i * c + j..i * c + i]
                    .iter()
                    .zip(&mt[j * c + j..j * c + i])
                    .map(|(a, b)| a * b)
                    .sum();
                mt[j * c + i] = -inv_lii * dot;
            }
        }

        // Map `U[r, col] = M[c - 1 - r, c - 1 - col] = mt[(c - 1 - col) * c + (c - 1 - r)]` for `r <= col`
        let mut u_upper = vec![0.0f32; c * c];
        for r in 0..c {
            let rev_r = c - 1 - r;
            let dst_row = &mut u_upper[r * c..(r + 1) * c];
            for (col, slot) in dst_row.iter_mut().enumerate().skip(r) {
                let rev_col = c - 1 - col;
                *slot = mt[rev_col * c + rev_r] as f32;
            }
        }
        self.h_inv_chol_upper = u_upper;
        // Free `raw_h` after factorization to keep peak compile memory tiny!
        self.raw_h = Vec::new();
    }
}

/// Compiles dense weights `[rows, cols]` into a [`FusedBase3LutMatrix`] using **Full-Row Lazy
/// Block-64 Hadamard-GPTQ** parallelized across `num_threads` safe scoped OS threads!
pub fn compile_hadamard_gptq_full_row(
    weights: &[f32],
    rows: usize,
    cols: usize,
    num_planes: usize,
    hessian: Option<&FullRowGptqHessian>,
    num_threads: usize,
) -> FusedBase3LutMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
    assert!((1..=3).contains(&num_planes));
    if let Some(h) = hessian {
        assert_eq!(h.cols, cols);
        assert_eq!(h.h_inv_chol_upper.len(), cols * cols);
    }

    let bpr = cols / GROUP_SIZE;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / HADAMARD_BLOCK;
    let max_int: i32 = match num_planes {
        1 => 1,
        2 => 4,
        3 => 13,
        _ => unreachable!(),
    };

    let mut flat_q = vec![0i8; rows * cols];
    let mut bf16_chunk_scales = vec![0u16; num_stripes * num_chunks * SIMD_ROW_STRIPE];

    let workers = num_threads.max(1).min(num_stripes);
    let stripes_per_worker = num_stripes.div_ceil(workers);
    let u_slice = hessian.map(|h| h.h_inv_chol_upper.as_slice());

    std::thread::scope(|s| {
        let mut rem_w = weights;
        let mut rem_q = flat_q.as_mut_slice();
        let mut rem_scales = bf16_chunk_scales.as_mut_slice();
        let mut stripe_cursor = 0usize;

        while stripe_cursor < num_stripes && !rem_w.is_empty() {
            let take_stripes = stripes_per_worker.min(num_stripes - stripe_cursor);
            let row_start = stripe_cursor * SIMD_ROW_STRIPE;
            let row_end = ((stripe_cursor + take_stripes) * SIMD_ROW_STRIPE).min(rows);
            let take_rows = row_end - row_start;

            let (w_chunk, next_w) = rem_w.split_at(take_rows * cols);
            let (q_chunk, next_q) = rem_q.split_at_mut(take_rows * cols);
            let (sc_chunk, next_sc) =
                rem_scales.split_at_mut(take_stripes * num_chunks * SIMD_ROW_STRIPE);

            rem_w = next_w;
            rem_q = next_q;
            rem_scales = next_sc;
            stripe_cursor += take_stripes;

            s.spawn(move || {
                quantize_stripe_batch_full_row(
                    w_chunk,
                    q_chunk,
                    sc_chunk,
                    take_rows,
                    cols,
                    num_chunks,
                    max_int,
                    u_slice,
                );
            });
        }
    });

    let mut plane_trits: Vec<Vec<i8>> = (0..num_planes).map(|_| vec![0i8; rows * cols]).collect();
    for (idx, &qv) in flat_q.iter().enumerate() {
        let q = qv as i32;
        match num_planes {
            1 => {
                plane_trits[0][idx] = q as i8;
            }
            2 => {
                let u = q + 4;
                plane_trits[0][idx] = ((u / 3) - 1) as i8;
                plane_trits[1][idx] = ((u % 3) - 1) as i8;
            }
            3 => {
                let u = q + 13;
                plane_trits[0][idx] = ((u / 9) - 1) as i8;
                plane_trits[1][idx] = (((u / 3) % 3) - 1) as i8;
                plane_trits[2][idx] = ((u % 3) - 1) as i8;
            }
            _ => unreachable!(),
        }
    }

    let planes: Vec<SimdPackedMatrix> = plane_trits
        .iter()
        .map(|trits| SimdPackedMatrix::from_ternary_weights(trits, rows, cols))
        .collect();

    FusedBase3LutMatrix {
        rows,
        cols,
        num_planes,
        blocks_per_row: bpr,
        num_stripes,
        num_chunks,
        planes,
        bf16_chunk_scales,
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_stripe_batch_full_row(
    w_chunk: &[f32],
    q_chunk: &mut [i8],
    sc_chunk: &mut [u16],
    take_rows: usize,
    cols: usize,
    num_chunks: usize,
    max_int: i32,
    u_slice: Option<&[f32]>,
) {
    let ratios = [0.72f32, 0.80, 0.88, 0.96, 1.04, 1.14, 1.25];
    let mut row_work = vec![0.0f32; cols];
    let mut blk_work = [0.0f32; HADAMARD_BLOCK];
    let mut best_q = [0i8; HADAMARD_BLOCK];
    let mut cand_q = [0i8; HADAMARD_BLOCK];
    let mut best_err = [0.0f32; HADAMARD_BLOCK];
    let mut cand_err = [0.0f32; HADAMARD_BLOCK];

    for local_r in 0..take_rows {
        let local_stripe = local_r / SIMD_ROW_STRIPE;
        let lane = local_r % SIMD_ROW_STRIPE;

        row_work.copy_from_slice(&w_chunk[local_r * cols..(local_r + 1) * cols]);
        fwht64_signed_inplace(&mut row_work);

        for c in 0..num_chunks {
            let col_s = c * HADAMARD_BLOCK;
            let col_e = col_s + HADAMARD_BLOCK;
            let sub = &row_work[col_s..col_e];

            let max_abs = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= 1e-12 {
                continue;
            }

            let mut best_loss = f32::INFINITY;
            let mut best_bf16 = 0u16;

            for &rat in &ratios {
                let delta_raw = (max_abs * rat) / (max_int as f32);
                let d_bf16 = f32_to_bf16(delta_raw);
                let delta = bf16_to_f32(d_bf16);
                if delta <= 0.0 {
                    continue;
                }
                let inv_delta = 1.0 / delta;
                blk_work.copy_from_slice(sub);
                let mut loss = 0.0f32;

                if let Some(u_mat) = u_slice {
                    for j in 0..HADAMARD_BLOCK {
                        let g_col = col_s + j;
                        let wj = blk_work[j];
                        let q = (wj * inv_delta).round().clamp(-(max_int as f32), max_int as f32)
                            as i32;
                        cand_q[j] = q as i8;
                        let w_hat = (q as f32) * delta;
                        let u_jj = u_mat[g_col * cols + g_col].max(1e-12);
                        let err = (wj - w_hat) / u_jj;
                        cand_err[j] = err;
                        loss += err * err;

                        let u_sub = &u_mat[g_col * cols + g_col + 1..g_col * cols + col_e];
                        for (k_slot, &u_jk) in blk_work[j + 1..].iter_mut().zip(u_sub.iter()) {
                            *k_slot -= err * u_jk;
                        }
                    }
                } else {
                    for j in 0..HADAMARD_BLOCK {
                        let wj = blk_work[j];
                        let q = (wj * inv_delta).round().clamp(-(max_int as f32), max_int as f32)
                            as i32;
                        cand_q[j] = q as i8;
                        let diff = wj - (q as f32) * delta;
                        loss += diff * diff;
                    }
                }

                if loss < best_loss {
                    best_loss = loss;
                    best_bf16 = d_bf16;
                    best_q = cand_q;
                    best_err = cand_err;
                }
            }

            sc_chunk[(local_stripe * num_chunks + c) * SIMD_ROW_STRIPE + lane] = best_bf16;
            q_chunk[local_r * cols + col_s..local_r * cols + col_e].copy_from_slice(&best_q);

            // Propagate chunk `c` quantization errors across ALL remaining chunks `(col_e .. cols)`!
            if let (true, Some(u_mat)) = (col_e < cols, u_slice) {
                let rem_work = &mut row_work[col_e..cols];
                for (j, &err) in best_err.iter().enumerate() {
                    if err == 0.0 {
                        continue;
                    }
                    let g_col = col_s + j;
                    let u_tail = &u_mat[g_col * cols + col_e..(g_col + 1) * cols];
                    for (dst, &u_val) in rem_work.iter_mut().zip(u_tail.iter()) {
                        *dst -= err * u_val;
                    }
                }
            }
        }
    }
}

/// Compiles dense weights `[rows, cols]` into a **Full-16-Entry Bipolar Radix-2** [`FusedBipolarLutMatrix`]
/// (`3` planes = `3.25b`, **`4` planes = `4.25b` `16`-level**, **`5` planes = `5.25b` `32`-level**,
/// **`6` planes = `6.25b` `64`-level**) using **Full-Row Lazy Block-64 Hadamard-GPTQ**!
pub fn compile_hadamard_gptq_bipolar(
    weights: &[f32],
    rows: usize,
    cols: usize,
    num_planes: usize,
    hessian: Option<&FullRowGptqHessian>,
    num_threads: usize,
) -> FusedBipolarLutMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
    assert!((1..=6).contains(&num_planes));
    if let Some(h) = hessian {
        assert_eq!(h.cols, cols);
        assert_eq!(h.h_inv_chol_upper.len(), cols * cols);
    }

    let octets_per_row = cols / 8;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / HADAMARD_BLOCK;
    let max_odd = (1i32 << num_planes) - 1;

    let mut flat_u = vec![0u8; rows * cols];
    let mut bf16_chunk_scales = vec![0u16; num_stripes * num_chunks * SIMD_ROW_STRIPE];

    let workers = num_threads.max(1).min(num_stripes);
    let stripes_per_worker = num_stripes.div_ceil(workers);
    let u_slice = hessian.map(|h| h.h_inv_chol_upper.as_slice());

    std::thread::scope(|s| {
        let mut rem_w = weights;
        let mut rem_u = flat_u.as_mut_slice();
        let mut rem_scales = bf16_chunk_scales.as_mut_slice();
        let mut stripe_cursor = 0usize;

        while stripe_cursor < num_stripes && !rem_w.is_empty() {
            let take_stripes = stripes_per_worker.min(num_stripes - stripe_cursor);
            let row_start = stripe_cursor * SIMD_ROW_STRIPE;
            let row_end = ((stripe_cursor + take_stripes) * SIMD_ROW_STRIPE).min(rows);
            let take_rows = row_end - row_start;

            let (w_chunk, next_w) = rem_w.split_at(take_rows * cols);
            let (u_chunk, next_u) = rem_u.split_at_mut(take_rows * cols);
            let (sc_chunk, next_sc) =
                rem_scales.split_at_mut(take_stripes * num_chunks * SIMD_ROW_STRIPE);

            rem_w = next_w;
            rem_u = next_u;
            rem_scales = next_sc;
            stripe_cursor += take_stripes;

            s.spawn(move || {
                quantize_stripe_batch_bipolar(
                    w_chunk,
                    u_chunk,
                    sc_chunk,
                    take_rows,
                    cols,
                    num_chunks,
                    max_odd,
                    u_slice,
                );
            });
        }
    });

    let interleaved_data = FusedBipolarLutMatrix::pack_interleaved(&flat_u, rows, cols, num_planes);

    FusedBipolarLutMatrix {
        rows,
        cols,
        num_planes,
        octets_per_row,
        num_stripes,
        num_chunks,
        interleaved_data,
        bf16_chunk_scales,
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_stripe_batch_bipolar(
    w_chunk: &[f32],
    u_chunk: &mut [u8],
    sc_chunk: &mut [u16],
    take_rows: usize,
    cols: usize,
    num_chunks: usize,
    max_odd: i32,
    u_slice: Option<&[f32]>,
) {
    let ratios = [0.62f32, 0.67, 0.72, 0.77, 0.82, 0.87, 0.92, 0.97, 1.02, 1.07, 1.12, 1.18];
    let max_u_f = max_odd as f32;
    let half_off = 0.5 * max_u_f;

    let mut row_work = vec![0.0f32; cols];
    let mut blk_work = [0.0f32; HADAMARD_BLOCK];
    let mut best_u = [0u8; HADAMARD_BLOCK];
    let mut cand_u = [0u8; HADAMARD_BLOCK];
    let mut best_err = [0.0f32; HADAMARD_BLOCK];
    let mut cand_err = [0.0f32; HADAMARD_BLOCK];

    for local_r in 0..take_rows {
        let local_stripe = local_r / SIMD_ROW_STRIPE;
        let lane = local_r % SIMD_ROW_STRIPE;

        row_work.copy_from_slice(&w_chunk[local_r * cols..(local_r + 1) * cols]);
        fwht64_signed_inplace(&mut row_work);

        for c in 0..num_chunks {
            let col_s = c * HADAMARD_BLOCK;
            let col_e = col_s + HADAMARD_BLOCK;
            let sub = &row_work[col_s..col_e];

            let max_abs = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= 1e-12 {
                continue;
            }

            let mut best_loss = f32::INFINITY;
            let mut best_bf16 = 0u16;

            for &rat in &ratios {
                let delta_raw = (max_abs * rat) / max_u_f;
                let d_bf16 = f32_to_bf16(delta_raw);
                let delta = bf16_to_f32(d_bf16);
                if delta <= 0.0 {
                    continue;
                }
                let inv_2delta = 0.5 / delta;
                blk_work.copy_from_slice(sub);
                let mut loss = 0.0f32;

                if let Some(u_mat) = u_slice {
                    for j in 0..HADAMARD_BLOCK {
                        let g_col = col_s + j;
                        let wj = blk_work[j];
                        let u_idx = (wj * inv_2delta + half_off).round().clamp(0.0, max_u_f) as i32;
                        cand_u[j] = u_idx as u8;
                        let q_odd = 2 * u_idx - max_odd;
                        let w_hat = (q_odd as f32) * delta;
                        let u_jj = u_mat[g_col * cols + g_col].max(1e-12);
                        let err = (wj - w_hat) / u_jj;
                        cand_err[j] = err;
                        loss += err * err;

                        let u_sub = &u_mat[g_col * cols + g_col + 1..g_col * cols + col_e];
                        for (k_slot, &u_jk) in blk_work[j + 1..].iter_mut().zip(u_sub.iter()) {
                            *k_slot -= err * u_jk;
                        }
                    }
                } else {
                    for j in 0..HADAMARD_BLOCK {
                        let wj = blk_work[j];
                        let u_idx = (wj * inv_2delta + half_off).round().clamp(0.0, max_u_f) as i32;
                        cand_u[j] = u_idx as u8;
                        let q_odd = 2 * u_idx - max_odd;
                        let diff = wj - (q_odd as f32) * delta;
                        loss += diff * diff;
                    }
                }

                if loss < best_loss {
                    best_loss = loss;
                    best_bf16 = d_bf16;
                    best_u = cand_u;
                    best_err = cand_err;
                }
            }

            sc_chunk[(local_stripe * num_chunks + c) * SIMD_ROW_STRIPE + lane] = best_bf16;
            u_chunk[local_r * cols + col_s..local_r * cols + col_e].copy_from_slice(&best_u);

            if let (true, Some(u_mat)) = (col_e < cols, u_slice) {
                let rem_work = &mut row_work[col_e..cols];
                for (j, &err) in best_err.iter().enumerate() {
                    if err == 0.0 {
                        continue;
                    }
                    let g_col = col_s + j;
                    let u_tail = &u_mat[g_col * cols + col_e..(g_col + 1) * cols];
                    for (dst, &u_val) in rem_work.iter_mut().zip(u_tail.iter()) {
                        *dst -= err * u_val;
                    }
                }
            }
        }
    }
}

/// Factorizes a dense `FP32` matrix `[rows, cols]` into `rank` Kronecker codebook components
/// over `reg_codebook` (`[u8; 64]`) via Alternating Coordinate Descent + Orthogonal Residual Projection.
pub fn compile_kronecker_matrix(
    weights: &[f32],
    rows: usize,
    cols: usize,
    rank: usize,
    reg_codebook: &[u8; 64],
) -> CompiledKroneckerMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(GROUP_SIZE));
    assert!(rank > 0);

    let bpr = cols / GROUP_SIZE;
    let mut patterns = [[0.0f32; 4]; 64];
    let mut pat_energy = [0.0f32; 64];
    for p in 0..64 {
        let trits = decode_nibble_code(reg_codebook[p]);
        patterns[p] = [
            trits[0] as f32,
            trits[1] as f32,
            trits[2] as f32,
            trits[3] as f32,
        ];
        pat_energy[p] = patterns[p].iter().map(|v| v * v).sum();
    }

    let orig_norm_sq: f32 = weights.iter().map(|v| v * v).sum();
    let mut residual = weights.to_vec();

    let mut all_scales = Vec::with_capacity(rank);
    let mut all_row_seeds = Vec::with_capacity(rank * rows);
    let mut all_col_seeds = Vec::with_capacity(rank * bpr);

    for k in 0..rank {
        // Seed initial column pattern indices from highest-energy residual row
        let best_row = (0..rows)
            .max_by(|&ra, &rb| {
                let ea: f32 = residual[ra * cols..(ra + 1) * cols]
                    .iter()
                    .map(|v| v * v)
                    .sum();
                let eb: f32 = residual[rb * cols..(rb + 1) * cols]
                    .iter()
                    .map(|v| v * v)
                    .sum();
                ea.total_cmp(&eb)
            })
            .unwrap_or(0);

        let mut r_seeds = vec![0u8; rows];
        let mut c_eff = vec![((k * 13 + 1) & 0x3F) as u8; bpr];

        for (b, c_slot) in c_eff.iter_mut().enumerate() {
            let target = &residual[best_row * cols + b * 4..best_row * cols + (b + 1) * 4];
            let mut best_p = 1u8;
            let mut best_obj = f32::NEG_INFINITY;
            for p in 1..64u8 {
                let pat = &patterns[p as usize];
                let dot = target[0] * pat[0]
                    + target[1] * pat[1]
                    + target[2] * pat[2]
                    + target[3] * pat[3];
                let e = pat_energy[p as usize].max(1.0);
                let obj = (dot.abs()) / e.sqrt();
                if obj > best_obj {
                    best_obj = obj;
                    best_p = p;
                }
            }
            *c_slot = best_p;
        }

        // 4 rounds of Alternating Coordinate Descent maximizing |<residual, P>| / ||P||_F
        for _alt in 0..4 {
            // Optimize each row seed in 0..64
            for (r, r_slot) in r_seeds.iter_mut().enumerate() {
                let row_res = &residual[r * cols..(r + 1) * cols];
                let mut best_rv = 0u8;
                let mut best_score = f32::NEG_INFINITY;

                for rv in 0..64u8 {
                    let mut dot = 0.0f32;
                    let mut norm_sq = 0.0f32;
                    for (b, &ce) in c_eff.iter().enumerate() {
                        let p_idx = (rv ^ ce) as usize;
                        let pat = &patterns[p_idx];
                        let sub = &row_res[b * 4..b * 4 + 4];
                        dot +=
                            sub[0] * pat[0] + sub[1] * pat[1] + sub[2] * pat[2] + sub[3] * pat[3];
                        norm_sq += pat_energy[p_idx];
                    }
                    let score = if norm_sq > 0.0 {
                        dot / norm_sq.sqrt()
                    } else {
                        f32::NEG_INFINITY
                    };
                    if score > best_score {
                        best_score = score;
                        best_rv = rv;
                    }
                }
                *r_slot = best_rv;
            }

            // Optimize each column block seed in 0..64
            for (b, c_slot) in c_eff.iter_mut().enumerate() {
                let mut best_cv = 0u8;
                let mut best_score = f32::NEG_INFINITY;

                for cv in 0..64u8 {
                    let mut dot = 0.0f32;
                    let mut norm_sq = 0.0f32;
                    for (r, &rv) in r_seeds.iter().enumerate() {
                        let p_idx = (rv ^ cv) as usize;
                        let pat = &patterns[p_idx];
                        let sub = &residual[r * cols + b * 4..r * cols + b * 4 + 4];
                        dot +=
                            sub[0] * pat[0] + sub[1] * pat[1] + sub[2] * pat[2] + sub[3] * pat[3];
                        norm_sq += pat_energy[p_idx];
                    }
                    let score = if norm_sq > 0.0 {
                        dot / norm_sq.sqrt()
                    } else {
                        f32::NEG_INFINITY
                    };
                    if score > best_score {
                        best_score = score;
                        best_cv = cv;
                    }
                }
                *c_slot = best_cv;
            }
        }

        // Exact least-squares step size alpha_k = <residual, P_k> / ||P_k||_F^2
        let mut dot_rp = 0.0f32;
        let mut norm_p_sq = 0.0f32;
        for (r, &rv) in r_seeds.iter().enumerate() {
            let row_res = &residual[r * cols..(r + 1) * cols];
            for (b, &ce) in c_eff.iter().enumerate() {
                let p_idx = (rv ^ ce) as usize;
                let pat = &patterns[p_idx];
                let sub = &row_res[b * 4..b * 4 + 4];
                dot_rp += sub[0] * pat[0] + sub[1] * pat[1] + sub[2] * pat[2] + sub[3] * pat[3];
                norm_p_sq += pat_energy[p_idx];
            }
        }

        let alpha = if norm_p_sq > 0.0 {
            dot_rp / norm_p_sq
        } else {
            0.0
        };

        // Deflate residual
        if alpha != 0.0 {
            for (r, &rv) in r_seeds.iter().enumerate() {
                let row_res = &mut residual[r * cols..(r + 1) * cols];
                for (b, &ce) in c_eff.iter().enumerate() {
                    let pat = &patterns[(rv ^ ce) as usize];
                    let sub = &mut row_res[b * 4..b * 4 + 4];
                    sub[0] -= alpha * pat[0];
                    sub[1] -= alpha * pat[1];
                    sub[2] -= alpha * pat[2];
                    sub[3] -= alpha * pat[3];
                }
            }
        }

        all_scales.push(alpha);
        all_row_seeds.extend_from_slice(&r_seeds);
        // Invert `rotate_left(3)` applied during runtime decoding
        for &ce in &c_eff {
            all_col_seeds.push(ce.rotate_right(3));
        }
    }

    let res_norm_sq: f32 = residual.iter().map(|v| v * v).sum();
    let rel_err = if orig_norm_sq > 0.0 {
        (res_norm_sq / orig_norm_sq).sqrt()
    } else {
        0.0
    };

    CompiledKroneckerMatrix {
        rows,
        cols,
        rank,
        scales: all_scales,
        row_seeds: all_row_seeds,
        col_seeds: all_col_seeds,
        relative_frobenius_error: rel_err,
    }
}

/// Compiles three dense SwiGLU matrices (`w_gate`, `w_up`, `w_down`) into a 4KB-page-ready [`CompiledSwiGluExpert`].
#[allow(clippy::too_many_arguments)]
pub fn compile_swiglu_expert(
    layer_id: u32,
    expert_id: u64,
    d_model: usize,
    d_ff: usize,
    rank: usize,
    output_scale: f32,
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
) -> CompiledSwiGluExpert {
    let cb = default_register_codebook_64();
    let cg = compile_kronecker_matrix(w_gate, d_ff, d_model, rank, &cb);
    let cu = compile_kronecker_matrix(w_up, d_ff, d_model, rank, &cb);
    let cd = compile_kronecker_matrix(w_down, d_model, d_ff, rank, &cb);

    CompiledSwiGluExpert {
        layer_id,
        expert_id,
        d_model,
        d_ff,
        rank,
        output_scale,
        gate_scales: cg.scales,
        up_scales: cu.scales,
        down_scales: cd.scales,
        gate_row_seeds: cg.row_seeds,
        gate_col_seeds: cg.col_seeds,
        up_row_seeds: cu.row_seeds,
        up_col_seeds: cu.col_seeds,
        down_row_seeds: cd.row_seeds,
        down_col_seeds: cd.col_seeds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd_lut::{fused_kronecker_gemv, QuantizedActivationLut};

    #[test]
    fn test_multirank_kronecker_monotonic_error_reduction() {
        let rows = 32;
        let cols = 64;
        // Structured low-rank + block-patterned target matrix
        let target: Vec<f32> = (0..rows * cols)
            .map(|idx| {
                let r = (idx / cols) as f32;
                let c = (idx % cols) as f32;
                (r * 0.21).sin() * (c * 0.13).cos() + 0.4 * ((r + c) * 0.37).sin()
            })
            .collect();

        let cb = default_register_codebook_64();
        let c1 = compile_kronecker_matrix(&target, rows, cols, 1, &cb);
        let c2 = compile_kronecker_matrix(&target, rows, cols, 2, &cb);
        let c4 = compile_kronecker_matrix(&target, rows, cols, 4, &cb);
        let c8 = compile_kronecker_matrix(&target, rows, cols, 8, &cb);

        assert!(c2.relative_frobenius_error < c1.relative_frobenius_error);
        assert!(c4.relative_frobenius_error < c2.relative_frobenius_error);
        assert!(c8.relative_frobenius_error < c4.relative_frobenius_error);

        // Verify runtime fused_kronecker_gemv approximates dense target * x closely
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.11).cos()).collect();
        let qlut = QuantizedActivationLut::build(&x);
        let mut out_approx = vec![0.0f32; rows];
        fused_kronecker_gemv(
            &c8.row_seeds,
            &c8.col_seeds,
            &c8.scales,
            rows,
            &cb,
            &qlut,
            &mut out_approx,
        );

        let out_exact: Vec<f32> = (0..rows)
            .map(|r| {
                target[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();

        let dot: f32 = out_approx.iter().zip(out_exact.iter()).map(|(a, b)| a * b).sum();
        let na = out_approx.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = out_exact.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(dot / (na * nb) > 0.90);
    }

    #[test]
    fn test_multiplane_ternary_heavy_tailed_fidelity() {
        use crate::simd_lut::{simd_lut_gemv_multiplane, simd_lut_gemv_multiplane_scalar};

        let rows = 50; // Non-multiple of 16
        let cols = 192; // 3 chunks of 64
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| {
                let r = idx / cols;
                let c = idx % cols;
                let base = (((r * 17 + c * 31) as f32) * 0.13).sin() * 0.25;
                // Inject LLM-style heavy-tailed outlier channels!
                if c == 7 || c == 131 {
                    base * 14.0
                } else {
                    base
                }
            })
            .collect();

        let (m_glob, err_glob) = compile_global_ternary_as_multiplane(&weights, rows, cols);
        let (m1, err1) = compile_multiplane_ternary(&weights, rows, cols, 1);
        let (m2, err2) = compile_multiplane_ternary(&weights, rows, cols, 2);
        let (m3, err3) = compile_multiplane_ternary(&weights, rows, cols, 3);

        assert!(err1 < err_glob, "G=64 exact 1-plane ({err1}) must beat global ({err_glob})");
        assert!(err2 < err1, "2-plane ({err2}) must beat 1-plane ({err1})");
        assert!(err3 < err2, "3-plane ({err3}) must beat 2-plane ({err2})");
        let _ = m_glob;
        let _ = m1;

        let x: Vec<f32> = (0..cols)
            .map(|c| {
                let v = ((c as f32) * 0.09).cos();
                if c == 7 { v * 8.0 } else { v }
            })
            .collect();
        let qlut = QuantizedActivationLut::build(&x);

        let mut out_neon2 = vec![0.0f32; rows];
        let mut out_neon3 = vec![0.0f32; rows];
        let mut out_scalar3 = vec![0.0f32; rows];
        simd_lut_gemv_multiplane(&m2, &qlut, &mut out_neon2);
        simd_lut_gemv_multiplane(&m3, &qlut, &mut out_neon3);
        simd_lut_gemv_multiplane_scalar(&m3, &qlut, &mut out_scalar3);

        for r in 0..rows {
            assert!(
                (out_neon3[r] - out_scalar3[r]).abs() < 1e-4,
                "row {r} neon vs scalar mismatch: {} vs {}",
                out_neon3[r],
                out_scalar3[r]
            );
        }

        let out_fp32: Vec<f32> = (0..rows)
            .map(|r| {
                weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(w, v)| w * v)
                    .sum()
            })
            .collect();

        let cos = |a: &[f32], b: &[f32]| -> f32 {
            let d: f32 = a.iter().zip(b.iter()).map(|(u, v)| u * v).sum();
            let na = a.iter().map(|u| u * u).sum::<f32>().sqrt();
            let nb = b.iter().map(|u| u * u).sum::<f32>().sqrt();
            d / (na * nb)
        };
        assert!(cos(&out_neon2, &out_fp32) > 0.990);
        assert!(cos(&out_neon3, &out_fp32) > 0.999);
    }
}
