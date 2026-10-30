#![forbid(unsafe_code)]
//! Weight Compiler: BitNet $b1.58$ Ternary Quantizer & Multi-Rank Kronecker `.k2t` Factorizer.
//!
//! Bridges standard dense `FP32` neural network weights to `kronos-2t` execution primitives:
//! 1. [`compile_bitnet_ternary`]: AbsMean ternary quantization ($\{-1, 0, +1\} \times \beta$) into [`SimdPackedMatrix`].
//! 2. [`compile_kronecker_matrix`]: Alternating Coordinate Descent / Matching Pursuit factorizer
//!    decomposing any dense matrix $W \in \mathbb{R}^{R \times C}$ into multi-rank XOR-codebook seeds
//!    for zero-buffer register execution via [`crate::kernels::simd_lut::fused_kronecker_gemv`].

use crate::kernels::hadamard::{fwht64_signed_inplace, HADAMARD_BLOCK};
use crate::storage::k2t_format::CompiledSwiGluExpert;
use crate::kernels::lut_gemv::GROUP_SIZE;
use crate::kernels::simd_lut::{
    bf16_to_f32, decode_nibble_code, default_register_codebook_64, e8q2_decode_8d,
    e8q2_quantize_8d, f32_to_bf16, nf_expand_codebook32, FusedBase3LutMatrix,
    FusedBipolarLutMatrix, MultiPlaneTernaryMatrix, NfSdotMatrix, SimdPackedMatrix, Vq2LutMatrix,
    E8Q2_EVEN_TEMPLATES_I8, NF3_DEFAULT_CODEBOOK_I8, NF4_DEFAULT_CODEBOOK_I8,
    NF5_DEFAULT_POS_CODEBOOK_I8, QUANT_CHUNK_BLOCKS, SIMD_ROW_STRIPE, VQ2_GAUSSIAN_RING_16,
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
    /// Rotated activations awaiting a batched `X^T X` flush.
    pending: Vec<f32>,
}

/// Activations per batched Hessian flush.
const HESSIAN_FLUSH_ROWS: usize = 1024;

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

    /// Adds the upper triangle of `X^T X` for all buffered rows into `raw_h` (one `SYRK`).
    fn flush_pending(&mut self) {
        let c = self.cols;
        let n = self.pending.len() / c;
        if n == 0 {
            return;
        }
        let mut partial = vec![0.0f32; c * c];
        crate::kernels::accel::syrk_upper(&self.pending, n, c, &mut partial);
        for i in 0..c {
            for (slot, &v) in self.raw_h[i * c + i..(i + 1) * c]
                .iter_mut()
                .zip(&partial[i * c + i..(i + 1) * c])
            {
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

        // Factor the reversed Hessian and invert; retry with more damping if not positive definite.
        let mut l_mat = vec![0.0f64; c * c];
        let mut floor = iso_floor;
        loop {
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
                    l_mat[i * c + j] = if i == j {
                        raw_val + floor
                    } else {
                        raw_val * off_diag_scale
                    };
                }
            }
            if crate::kernels::accel::cholesky_lower(&mut l_mat, c) && crate::kernels::accel::invert_lower(&mut l_mat, c) {
                break;
            }
            floor *= 10.0;
        }

        // `U[r, col] = (L^{-1})[c - 1 - r, c - 1 - col]` for `r <= col`.
        let mut u_upper = vec![0.0f32; c * c];
        for r in 0..c {
            let src_row = &l_mat[(c - 1 - r) * c..(c - r) * c];
            for (col, slot) in u_upper[r * c..(r + 1) * c].iter_mut().enumerate().skip(r) {
                *slot = src_row[c - 1 - col] as f32;
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
    // Tail columns per propagation tile (keeps the stripe's rows in L1).
    const TAIL_TILE: usize = 512;

    let mut work = vec![0.0f32; SIMD_ROW_STRIPE * cols];
    let mut errs = vec![[0.0f32; HADAMARD_BLOCK]; SIMD_ROW_STRIPE];
    let mut blk_work = [0.0f32; HADAMARD_BLOCK];
    let mut best_u = [0u8; HADAMARD_BLOCK];
    let mut cand_u = [0u8; HADAMARD_BLOCK];
    let mut cand_err = [0.0f32; HADAMARD_BLOCK];

    for g0 in (0..take_rows).step_by(SIMD_ROW_STRIPE) {
        let g_rows = (take_rows - g0).min(SIMD_ROW_STRIPE);
        let local_stripe = g0 / SIMD_ROW_STRIPE;
        for r in 0..g_rows {
            let dst = &mut work[r * cols..(r + 1) * cols];
            dst.copy_from_slice(&w_chunk[(g0 + r) * cols..(g0 + r + 1) * cols]);
            fwht64_signed_inplace(dst);
        }

        for c in 0..num_chunks {
            let col_s = c * HADAMARD_BLOCK;
            let col_e = col_s + HADAMARD_BLOCK;

            for (r, err_out) in errs.iter_mut().enumerate().take(g_rows) {
                let sub = &work[r * cols + col_s..r * cols + col_e];
                err_out.fill(0.0);
                let max_abs = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                if max_abs <= 1e-12 {
                    continue;
                }

                let mut best_loss = f32::INFINITY;
                let mut best_bf16 = 0u16;
                for &rat in &ratios {
                    let d_bf16 = f32_to_bf16((max_abs * rat) / max_u_f);
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
                            let w_hat = ((2 * u_idx - max_odd) as f32) * delta;
                            let err = (wj - w_hat) / u_mat[g_col * cols + g_col].max(1e-12);
                            cand_err[j] = err;
                            loss += err * err;
                            let u_sub = &u_mat[g_col * cols + g_col + 1..g_col * cols + col_e];
                            for (k_slot, &u_jk) in blk_work[j + 1..].iter_mut().zip(u_sub) {
                                *k_slot -= err * u_jk;
                            }
                        }
                    } else {
                        for j in 0..HADAMARD_BLOCK {
                            let wj = blk_work[j];
                            let u_idx = (wj * inv_2delta + half_off).round().clamp(0.0, max_u_f) as i32;
                            cand_u[j] = u_idx as u8;
                            let diff = wj - ((2 * u_idx - max_odd) as f32) * delta;
                            loss += diff * diff;
                        }
                    }

                    if loss < best_loss {
                        best_loss = loss;
                        best_bf16 = d_bf16;
                        best_u = cand_u;
                        *err_out = cand_err;
                    }
                }

                let lane = r;
                sc_chunk[(local_stripe * num_chunks + c) * SIMD_ROW_STRIPE + lane] = best_bf16;
                let row = g0 + r;
                u_chunk[row * cols + col_s..row * cols + col_e].copy_from_slice(&best_u);
            }

            // Propagate this chunk's errors into all later columns for the whole stripe.
            if let (true, Some(u_mat)) = (col_e < cols, u_slice) {
                for t0 in (col_e..cols).step_by(TAIL_TILE) {
                    let t1 = (t0 + TAIL_TILE).min(cols);
                    for j in 0..HADAMARD_BLOCK {
                        let g_col = col_s + j;
                        let u_tail = &u_mat[g_col * cols + t0..g_col * cols + t1];
                        for (r, err) in errs.iter().enumerate().take(g_rows) {
                            let e = err[j];
                            if e != 0.0 {
                                for (dst, &u_val) in work[r * cols + t0..r * cols + t1].iter_mut().zip(u_tail) {
                                    *dst -= e * u_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Compiles `[rows, cols]` weights into a 2-bit Hadamard VQ [`Vq2LutMatrix`] with block-2 GPTQ.
/// `scale_group_chunks` 1, 2, 4 give 2.25, 2.125, 2.0625 bpw.
pub fn compile_hadamard_gptq_vq2(
    weights: &[f32],
    rows: usize,
    cols: usize,
    scale_group_chunks: usize,
    hessian: Option<&FullRowGptqHessian>,
    num_threads: usize,
) -> Vq2LutMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
    assert!((1..=8).contains(&scale_group_chunks));
    if let Some(h) = hessian {
        assert_eq!(h.cols, cols);
        assert_eq!(h.h_inv_chol_upper.len(), cols * cols);
    }

    let pairs_per_row = cols / 2;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / HADAMARD_BLOCK;
    let num_scale_groups = num_chunks.div_ceil(scale_group_chunks);

    let mut flat_pairs = vec![0u8; rows * pairs_per_row];
    let mut bf16_chunk_scales = vec![0u16; num_stripes * num_scale_groups * SIMD_ROW_STRIPE];

    let workers = num_threads.max(1).min(num_stripes);
    let stripes_per_worker = num_stripes.div_ceil(workers);
    let u_slice = hessian.map(|h| h.h_inv_chol_upper.as_slice());

    std::thread::scope(|s| {
        let mut rem_w = weights;
        let mut rem_p = flat_pairs.as_mut_slice();
        let mut rem_scales = bf16_chunk_scales.as_mut_slice();
        let mut stripe_cursor = 0usize;

        while stripe_cursor < num_stripes && !rem_w.is_empty() {
            let take_stripes = stripes_per_worker.min(num_stripes - stripe_cursor);
            let row_start = stripe_cursor * SIMD_ROW_STRIPE;
            let row_end = ((stripe_cursor + take_stripes) * SIMD_ROW_STRIPE).min(rows);
            let take_rows = row_end - row_start;

            let (w_chunk, next_w) = rem_w.split_at(take_rows * cols);
            let (p_chunk, next_p) = rem_p.split_at_mut(take_rows * pairs_per_row);
            let (sc_chunk, next_sc) =
                rem_scales.split_at_mut(take_stripes * num_scale_groups * SIMD_ROW_STRIPE);

            rem_w = next_w;
            rem_p = next_p;
            rem_scales = next_sc;
            stripe_cursor += take_stripes;

            s.spawn(move || {
                quantize_stripe_batch_vq2(
                    w_chunk,
                    p_chunk,
                    sc_chunk,
                    take_rows,
                    cols,
                    num_chunks,
                    scale_group_chunks,
                    num_scale_groups,
                    u_slice,
                );
            });
        }
    });

    let packed_data = Vq2LutMatrix::pack_interleaved(&flat_pairs, rows, cols);

    Vq2LutMatrix {
        rows,
        cols,
        scale_group_chunks,
        num_stripes,
        num_chunks,
        num_scale_groups,
        packed_data,
        bf16_chunk_scales,
    }
}

#[allow(clippy::too_many_arguments)]
fn quantize_stripe_batch_vq2(
    w_chunk: &[f32],
    p_chunk: &mut [u8],
    sc_chunk: &mut [u16],
    take_rows: usize,
    cols: usize,
    num_chunks: usize,
    sgc: usize,
    nsg: usize,
    u_slice: Option<&[f32]>,
) {
    let ratios = [0.58f32, 0.66, 0.74, 0.82, 0.90, 0.98, 1.06, 1.15, 1.25, 1.36, 1.48];
    const TAIL_TILE: usize = 512;
    let pairs_per_row = cols / 2;
    let max_group_len = sgc * HADAMARD_BLOCK;
    let max_group_pairs = max_group_len / 2;

    let mut work = vec![0.0f32; SIMD_ROW_STRIPE * cols];
    let mut errs = vec![vec![0.0f32; max_group_len]; SIMD_ROW_STRIPE];
    let mut blk_work = vec![0.0f32; max_group_len];
    let mut best_pairs = vec![0u8; max_group_pairs];
    let mut cand_pairs = vec![0u8; max_group_pairs];
    let mut cand_err = vec![0.0f32; max_group_len];

    for g0 in (0..take_rows).step_by(SIMD_ROW_STRIPE) {
        let g_rows = (take_rows - g0).min(SIMD_ROW_STRIPE);
        let local_stripe = g0 / SIMD_ROW_STRIPE;
        for r in 0..g_rows {
            let dst = &mut work[r * cols..(r + 1) * cols];
            dst.copy_from_slice(&w_chunk[(g0 + r) * cols..(g0 + r + 1) * cols]);
            fwht64_signed_inplace(dst);
        }

        for sg in 0..nsg {
            let c_start = sg * sgc;
            let c_end = (c_start + sgc).min(num_chunks);
            let col_s = c_start * HADAMARD_BLOCK;
            let col_e = c_end * HADAMARD_BLOCK;
            let group_len = col_e - col_s;
            let group_pairs = group_len / 2;

            for (r, err_out) in errs.iter_mut().enumerate().take(g_rows) {
                let sub = &work[r * cols + col_s..r * cols + col_e];
                err_out[..group_len].fill(0.0);
                let sum_sq: f32 = sub.iter().map(|&v| v * v).sum();
                let rms = (sum_sq / (group_len as f32)).sqrt();
                if rms <= 1e-12 {
                    p_chunk[(g0 + r) * pairs_per_row + col_s / 2..(g0 + r) * pairs_per_row + col_e / 2]
                        .fill(0);
                    sc_chunk[(local_stripe * nsg + sg) * SIMD_ROW_STRIPE + r] = 0;
                    continue;
                }

                let mut best_loss = f32::INFINITY;
                let mut best_bf16 = 0u16;

                let mut eval_delta = |d_bf16: u16,
                                      best_loss: &mut f32,
                                      best_bf16: &mut u16,
                                      best_pairs: &mut [u8],
                                      err_out: &mut [f32]| {
                    let delta = bf16_to_f32(d_bf16);
                    if delta <= 0.0 {
                        return;
                    }
                    let mut du = [0.0f32; 16];
                    let mut dv = [0.0f32; 16];
                    for k in 0..16 {
                        du[k] = VQ2_GAUSSIAN_RING_16[k].0 * delta;
                        dv[k] = VQ2_GAUSSIAN_RING_16[k].1 * delta;
                    }
                    blk_work[..group_len].copy_from_slice(sub);
                    let mut loss = 0.0f32;

                    if let Some(u_mat) = u_slice {
                        for (m, cp_slot) in cand_pairs[..group_pairs].iter_mut().enumerate() {
                            let j0 = 2 * m;
                            let j1 = j0 + 1;
                            let gc0 = col_s + j0;
                            let gc1 = col_s + j1;
                            let w0 = blk_work[j0];
                            let w1 = blk_work[j1];
                            let inv_u00 = 1.0 / u_mat[gc0 * cols + gc0].max(1e-12);
                            let u01 = u_mat[gc0 * cols + gc1];
                            let inv_u11 = 1.0 / u_mat[gc1 * cols + gc1].max(1e-12);

                            let mut b_k = 0usize;
                            let mut b_l = f32::INFINITY;
                            let mut b_e0 = 0.0f32;
                            let mut b_e1 = 0.0f32;
                            for k in 0..16 {
                                let e0 = (w0 - du[k]) * inv_u00;
                                let e1 = ((w1 - dv[k]) - e0 * u01) * inv_u11;
                                let lk = e0 * e0 + e1 * e1;
                                if lk < b_l {
                                    b_l = lk;
                                    b_k = k;
                                    b_e0 = e0;
                                    b_e1 = e1;
                                }
                            }
                            *cp_slot = b_k as u8;
                            cand_err[j0] = b_e0;
                            cand_err[j1] = b_e1;
                            loss += b_l;
                            if loss >= *best_loss {
                                return;
                            }

                            if j1 + 1 < group_len {
                                let u_sub0 = &u_mat[gc0 * cols + gc1 + 1..gc0 * cols + col_e];
                                let u_sub1 = &u_mat[gc1 * cols + gc1 + 1..gc1 * cols + col_e];
                                for (k_slot, (&u0k, &u1k)) in blk_work[j1 + 1..group_len]
                                    .iter_mut()
                                    .zip(u_sub0.iter().zip(u_sub1.iter()))
                                {
                                    *k_slot -= b_e0 * u0k + b_e1 * u1k;
                                }
                            }
                        }
                    } else {
                        for (m, cp_slot) in cand_pairs[..group_pairs].iter_mut().enumerate() {
                            let w0 = blk_work[2 * m];
                            let w1 = blk_work[2 * m + 1];
                            let mut b_k = 0usize;
                            let mut b_l = f32::INFINITY;
                            for k in 0..16 {
                                let d0 = w0 - du[k];
                                let d1 = w1 - dv[k];
                                let lk = d0 * d0 + d1 * d1;
                                if lk < b_l {
                                    b_l = lk;
                                    b_k = k;
                                }
                            }
                            *cp_slot = b_k as u8;
                            loss += b_l;
                        }
                    }

                    if loss < *best_loss {
                        *best_loss = loss;
                        *best_bf16 = d_bf16;
                        best_pairs[..group_pairs].copy_from_slice(&cand_pairs[..group_pairs]);
                        err_out[..group_len].copy_from_slice(&cand_err[..group_len]);
                    }
                };

                for &rat in &ratios {
                    eval_delta(
                        f32_to_bf16(rms * rat),
                        &mut best_loss,
                        &mut best_bf16,
                        &mut best_pairs,
                        err_out,
                    );
                }

                // Least-squares BF16 scale refit on the chosen points.
                let mut dot_wc = 0.0f32;
                let mut dot_cc = 0.0f32;
                for m in 0..group_pairs {
                    let (u, v) = VQ2_GAUSSIAN_RING_16[best_pairs[m] as usize];
                    dot_wc += sub[2 * m] * u + sub[2 * m + 1] * v;
                    dot_cc += u * u + v * v;
                }
                if dot_cc > 1e-12 && dot_wc > 0.0 {
                    let ls_bf16 = f32_to_bf16(dot_wc / dot_cc);
                    if ls_bf16 != best_bf16 {
                        eval_delta(
                            ls_bf16,
                            &mut best_loss,
                            &mut best_bf16,
                            &mut best_pairs,
                            err_out,
                        );
                    }
                }

                sc_chunk[(local_stripe * nsg + sg) * SIMD_ROW_STRIPE + r] = best_bf16;
                let row = g0 + r;
                p_chunk[row * pairs_per_row + col_s / 2..row * pairs_per_row + col_e / 2]
                    .copy_from_slice(&best_pairs[..group_pairs]);
            }

            // Propagate this group's error to the remaining columns.
            if let (true, Some(u_mat)) = (col_e < cols, u_slice) {
                for t0 in (col_e..cols).step_by(TAIL_TILE) {
                    let t1 = (t0 + TAIL_TILE).min(cols);
                    for j in 0..group_len {
                        let g_col = col_s + j;
                        let u_tail = &u_mat[g_col * cols + t0..g_col * cols + t1];
                        for (r, err) in errs.iter().enumerate().take(g_rows) {
                            let e = err[j];
                            if e != 0.0 {
                                for (dst, &u_val) in work[r * cols + t0..r * cols + t1]
                                    .iter_mut()
                                    .zip(u_tail)
                                {
                                    *dst -= e * u_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Fits a symmetric INT8 Lloyd-Max codebook (32 levels for NF5, 16 for NF4, 8 for NF3) to the
/// Hadamard-rotated weights. Returns `(codebook, top positive centroid in RMS units)`.
pub fn fit_empirical_nf_codebook(
    weights: &[f32],
    rows: usize,
    cols: usize,
    bits: usize,
    scale_group_chunks: usize,
) -> ([i8; 16], f32) {
    assert!(bits == 3 || bits == 4 || bits == 5);
    const HIST_BINS: usize = 2000;
    const Z_MAX: f32 = 5.0;
    const BIN_W: f32 = Z_MAX / (HIST_BINS as f32);

    let default_cb = match bits {
        5 => (NF5_DEFAULT_POS_CODEBOOK_I8, 3.2500f32),
        4 => (NF4_DEFAULT_CODEBOOK_I8, 2.7326f32),
        _ => (NF3_DEFAULT_CODEBOOK_I8, 2.1520f32),
    };

    let num_chunks = cols / HADAMARD_BLOCK;
    let sgc = scale_group_chunks.max(1);
    let nsg = num_chunks.div_ceil(sgc);
    let step = (rows / 512).max(1);

    let mut hist = vec![0u64; HIST_BINS];
    let mut total_samples = 0u64;
    let mut row_buf = vec![0.0f32; cols];

    for r in (0..rows).step_by(step) {
        row_buf.copy_from_slice(&weights[r * cols..(r + 1) * cols]);
        fwht64_signed_inplace(&mut row_buf);
        for sg in 0..nsg {
            let col_s = sg * sgc * HADAMARD_BLOCK;
            let col_e = ((sg + 1) * sgc * HADAMARD_BLOCK).min(cols);
            let sub = &row_buf[col_s..col_e];
            let sum_sq: f32 = sub.iter().map(|&v| v * v).sum();
            let rms = (sum_sq / (sub.len() as f32)).sqrt();
            if rms <= 1e-12 {
                continue;
            }
            let inv_rms = 1.0 / rms;
            for &v in sub {
                let z = v.abs() * inv_rms;
                let bin = ((z / BIN_W) as usize).min(HIST_BINS - 1);
                hist[bin] += 1;
                total_samples += 1;
            }
        }
    }

    if total_samples < 256 {
        return default_cb;
    }

    let half = 1usize << (bits - 1);
    let mut mu: Vec<f64> = match bits {
        5 => vec![
            0.068, 0.205, 0.343, 0.483, 0.626, 0.773, 0.925, 1.084, 1.252, 1.432, 1.628, 1.845,
            2.092, 2.382, 2.743, 3.250,
        ],
        4 => vec![0.1284, 0.3881, 0.6568, 0.9424, 1.2562, 1.6180, 2.0690, 2.7326],
        _ => vec![0.2451, 0.7560, 1.3439, 2.1520],
    };

    for _ in 0..24 {
        let mut bounds = vec![0usize; half + 1];
        bounds[0] = 0;
        bounds[half] = HIST_BINS;
        for m in 0..(half - 1) {
            let mid = 0.5 * (mu[m] + mu[m + 1]);
            bounds[m + 1] = ((mid / (BIN_W as f64)).round() as usize).clamp(bounds[m], HIST_BINS);
        }
        for m in 0..half {
            let mut w_sum = 0.0f64;
            let mut c_sum = 0u64;
            for (b, &cnt) in hist[bounds[m]..bounds[m + 1]].iter().enumerate() {
                if cnt > 0 {
                    let x = ((bounds[m] + b) as f64 + 0.5) * (BIN_W as f64);
                    w_sum += (cnt as f64) * x;
                    c_sum += cnt;
                }
            }
            if c_sum > 0 {
                mu[m] = w_sum / (c_sum as f64);
            }
        }
        for m in 1..half {
            if mu[m] <= mu[m - 1] + 1e-4 {
                mu[m] = mu[m - 1] + 0.05;
            }
        }
    }

    let mu_top = mu[half - 1].max(0.5);
    let mut q_pos = vec![0i32; half];
    for m in 0..half {
        q_pos[m] = ((127.0 * mu[m] / mu_top).round() as i32).clamp(1, 127);
    }
    q_pos[half - 1] = 127;
    // Keep entries strictly increasing so each is unique.
    for m in (0..half - 1).rev() {
        if q_pos[m] >= q_pos[m + 1] {
            q_pos[m] = (q_pos[m + 1] - 1).max(1);
        }
    }
    for m in 1..half {
        if q_pos[m] <= q_pos[m - 1] {
            q_pos[m] = (q_pos[m - 1] + 1).min(127);
        }
    }

    let mut cb = [0i8; 16];
    if bits == 5 {
        for m in 0..16 {
            cb[m] = q_pos[m] as i8;
        }
    } else {
        for m in 0..half {
            cb[half - 1 - m] = (-q_pos[m]) as i8;
            cb[half + m] = q_pos[m] as i8;
        }
    }
    (cb, mu_top as f32)
}

/// Compiles `[rows, cols]` weights into an [`NfSdotMatrix`] (`bits` 5, 4, 3, or 2-bit Gosset `E8Q2`)
/// with Hadamard GPTQ and least-squares BF16 scale refit.
pub fn compile_hadamard_gptq_nf(
    weights: &[f32],
    rows: usize,
    cols: usize,
    bits: usize,
    scale_group_chunks: usize,
    hessian: Option<&FullRowGptqHessian>,
    num_threads: usize,
) -> NfSdotMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
    assert!(
        (2..=5).contains(&bits),
        "NF-SDOT bits must be 2, 3, 4, or 5"
    );
    assert!((1..=8).contains(&scale_group_chunks));
    if bits == 2 {
        return compile_hadamard_gptq_e8q2(
            weights,
            rows,
            cols,
            scale_group_chunks,
            hessian,
            num_threads,
        );
    }
    if let Some(h) = hessian {
        assert_eq!(h.cols, cols);
        assert_eq!(h.h_inv_chol_upper.len(), cols * cols);
    }

    let (codebook, mu_top) =
        fit_empirical_nf_codebook(weights, rows, cols, bits, scale_group_chunks);

    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / HADAMARD_BLOCK;
    let num_scale_groups = num_chunks.div_ceil(scale_group_chunks);

    let mut flat_indices = vec![0u8; rows * cols];
    let mut bf16_scales = vec![0u16; num_stripes * num_scale_groups * SIMD_ROW_STRIPE];

    let workers = num_threads.max(1).min(num_stripes);
    let stripes_per_worker = num_stripes.div_ceil(workers);
    let u_slice = hessian.map(|h| h.h_inv_chol_upper.as_slice());

    std::thread::scope(|s| {
        let mut rem_w = weights;
        let mut rem_idx = flat_indices.as_mut_slice();
        let mut rem_scales = bf16_scales.as_mut_slice();
        let mut stripe_cursor = 0usize;

        while stripe_cursor < num_stripes && !rem_w.is_empty() {
            let take_stripes = stripes_per_worker.min(num_stripes - stripe_cursor);
            let row_start = stripe_cursor * SIMD_ROW_STRIPE;
            let row_end = ((stripe_cursor + take_stripes) * SIMD_ROW_STRIPE).min(rows);
            let take_rows = row_end - row_start;

            let (w_chunk, next_w) = rem_w.split_at(take_rows * cols);
            let (idx_chunk, next_idx) = rem_idx.split_at_mut(take_rows * cols);
            let (sc_chunk, next_sc) =
                rem_scales.split_at_mut(take_stripes * num_scale_groups * SIMD_ROW_STRIPE);

            rem_w = next_w;
            rem_idx = next_idx;
            rem_scales = next_sc;
            stripe_cursor += take_stripes;

            s.spawn(move || {
                quantize_stripe_batch_nf(
                    w_chunk,
                    idx_chunk,
                    sc_chunk,
                    take_rows,
                    cols,
                    num_chunks,
                    bits,
                    scale_group_chunks,
                    num_scale_groups,
                    &codebook,
                    mu_top,
                    u_slice,
                );
            });
        }
    });

    NfSdotMatrix::from_indices(
        &flat_indices,
        &bf16_scales,
        rows,
        cols,
        bits,
        scale_group_chunks,
        codebook,
    )
}

#[allow(clippy::too_many_arguments)]
fn quantize_stripe_batch_nf(
    w_chunk: &[f32],
    idx_chunk: &mut [u8],
    sc_chunk: &mut [u16],
    take_rows: usize,
    cols: usize,
    num_chunks: usize,
    bits: usize,
    sgc: usize,
    nsg: usize,
    codebook: &[i8; 16],
    mu_top: f32,
    u_slice: Option<&[f32]>,
) {
    let ratios = [
        0.74f32, 0.80, 0.86, 0.91, 0.96, 1.00, 1.04, 1.09, 1.15, 1.22, 1.30,
    ];
    const TAIL_TILE: usize = 512;
    let half = 1usize << (bits - 1);
    let levels = 1usize << bits;
    let cb32 = nf_expand_codebook32(bits, codebook);
    let mut cb_f = [0.0f32; 32];
    for k in 0..levels {
        cb_f[k] = cb32[k] as f32;
    }
    // Decision thresholds at codebook midpoints.
    let mut pos_th = [0.0f32; 15];
    for (m, th) in pos_th.iter_mut().enumerate().take(half - 1) {
        *th = 0.5 * (cb_f[half + m] + cb_f[half + m + 1]);
    }

    let quant_nearest = |w: f32, inv_delta: f32| -> usize {
        let aw = w.abs() * inv_delta;
        let mag = if bits == 5 {
            if aw < pos_th[7] {
                if aw < pos_th[3] {
                    if aw < pos_th[1] {
                        usize::from(aw >= pos_th[0])
                    } else {
                        2 + usize::from(aw >= pos_th[2])
                    }
                } else if aw < pos_th[5] {
                    4 + usize::from(aw >= pos_th[4])
                } else {
                    6 + usize::from(aw >= pos_th[6])
                }
            } else if aw < pos_th[11] {
                if aw < pos_th[9] {
                    8 + usize::from(aw >= pos_th[8])
                } else {
                    10 + usize::from(aw >= pos_th[10])
                }
            } else if aw < pos_th[13] {
                12 + usize::from(aw >= pos_th[12])
            } else {
                14 + usize::from(aw >= pos_th[14])
            }
        } else if bits == 4 {
            if aw < pos_th[3] {
                if aw < pos_th[1] {
                    usize::from(aw >= pos_th[0])
                } else {
                    2 + usize::from(aw >= pos_th[2])
                }
            } else if aw < pos_th[5] {
                4 + usize::from(aw >= pos_th[4])
            } else {
                6 + usize::from(aw >= pos_th[6])
            }
        } else if aw < pos_th[1] {
            usize::from(aw >= pos_th[0])
        } else {
            2 + usize::from(aw >= pos_th[2])
        };
        if w >= 0.0 {
            half + mag
        } else {
            half - 1 - mag
        }
    };

    let max_group_len = sgc * HADAMARD_BLOCK;
    let mut work = vec![0.0f32; SIMD_ROW_STRIPE * cols];
    let mut errs = vec![vec![0.0f32; max_group_len]; SIMD_ROW_STRIPE];
    let mut blk_work = vec![0.0f32; max_group_len];
    let mut best_idx = vec![0u8; max_group_len];
    let mut cand_idx = vec![0u8; max_group_len];
    let mut cand_err = vec![0.0f32; max_group_len];

    for g0 in (0..take_rows).step_by(SIMD_ROW_STRIPE) {
        let g_rows = (take_rows - g0).min(SIMD_ROW_STRIPE);
        let local_stripe = g0 / SIMD_ROW_STRIPE;
        for r in 0..g_rows {
            let dst = &mut work[r * cols..(r + 1) * cols];
            dst.copy_from_slice(&w_chunk[(g0 + r) * cols..(g0 + r + 1) * cols]);
            fwht64_signed_inplace(dst);
        }

        for sg in 0..nsg {
            let c_start = sg * sgc;
            let c_end = (c_start + sgc).min(num_chunks);
            let col_s = c_start * HADAMARD_BLOCK;
            let col_e = c_end * HADAMARD_BLOCK;
            let group_len = col_e - col_s;

            for (r, err_out) in errs.iter_mut().enumerate().take(g_rows) {
                let sub = &work[r * cols + col_s..r * cols + col_e];
                err_out[..group_len].fill(0.0);
                let sum_sq: f32 = sub.iter().map(|&v| v * v).sum();
                let rms = (sum_sq / (group_len as f32)).sqrt();
                if rms <= 1e-12 {
                    idx_chunk[(g0 + r) * cols + col_s..(g0 + r) * cols + col_e]
                        .fill((half - 1) as u8);
                    sc_chunk[(local_stripe * nsg + sg) * SIMD_ROW_STRIPE + r] = 0;
                    continue;
                }

                let base_delta = rms * (mu_top / 127.0);
                let mut best_loss = f32::INFINITY;
                let mut best_bf16 = 0u16;

                let mut eval_delta = |d_bf16: u16,
                                      best_loss: &mut f32,
                                      best_bf16: &mut u16,
                                      best_idx: &mut [u8],
                                      err_out: &mut [f32]| {
                    let delta = bf16_to_f32(d_bf16);
                    if delta <= 0.0 {
                        return;
                    }
                    let inv_delta = 1.0 / delta;
                    blk_work[..group_len].copy_from_slice(sub);
                    let mut loss = 0.0f32;

                    if let Some(u_mat) = u_slice {
                        for j in 0..group_len {
                            let g_col = col_s + j;
                            let wj = blk_work[j];
                            let k = quant_nearest(wj, inv_delta);
                            cand_idx[j] = k as u8;
                            let w_hat = cb_f[k] * delta;
                            let err = (wj - w_hat) / u_mat[g_col * cols + g_col].max(1e-12);
                            cand_err[j] = err;
                            loss += err * err;
                            if loss >= *best_loss {
                                return;
                            }
                            if j + 1 < group_len {
                                let u_sub = &u_mat[g_col * cols + g_col + 1..g_col * cols + col_e];
                                for (k_slot, &u_jk) in blk_work[j + 1..group_len].iter_mut().zip(u_sub) {
                                    *k_slot -= err * u_jk;
                                }
                            }
                        }
                    } else {
                        for j in 0..group_len {
                            let wj = blk_work[j];
                            let k = quant_nearest(wj, inv_delta);
                            cand_idx[j] = k as u8;
                            let diff = wj - cb_f[k] * delta;
                            loss += diff * diff;
                        }
                    }

                    if loss < *best_loss {
                        *best_loss = loss;
                        *best_bf16 = d_bf16;
                        best_idx[..group_len].copy_from_slice(&cand_idx[..group_len]);
                        err_out[..group_len].copy_from_slice(&cand_err[..group_len]);
                    }
                };

                for &rat in &ratios {
                    eval_delta(
                        f32_to_bf16(base_delta * rat),
                        &mut best_loss,
                        &mut best_bf16,
                        &mut best_idx,
                        err_out,
                    );
                }

                // Least-squares BF16 scale refit on the chosen centroids.
                let mut dot_wc = 0.0f32;
                let mut dot_cc = 0.0f32;
                for j in 0..group_len {
                    let c_val = cb_f[best_idx[j] as usize];
                    dot_wc += sub[j] * c_val;
                    dot_cc += c_val * c_val;
                }
                if dot_cc > 1e-12 && dot_wc > 0.0 {
                    let ls_bf16 = f32_to_bf16(dot_wc / dot_cc);
                    if ls_bf16 != best_bf16 {
                        eval_delta(
                            ls_bf16,
                            &mut best_loss,
                            &mut best_bf16,
                            &mut best_idx,
                            err_out,
                        );
                    }
                }

                sc_chunk[(local_stripe * nsg + sg) * SIMD_ROW_STRIPE + r] = best_bf16;
                let row = g0 + r;
                idx_chunk[row * cols + col_s..row * cols + col_e]
                    .copy_from_slice(&best_idx[..group_len]);
            }

            // Propagate this group's error to the remaining columns.
            if let (true, Some(u_mat)) = (col_e < cols, u_slice) {
                for t0 in (col_e..cols).step_by(TAIL_TILE) {
                    let t1 = (t0 + TAIL_TILE).min(cols);
                    for j in 0..group_len {
                        let g_col = col_s + j;
                        let u_tail = &u_mat[g_col * cols + t0..g_col * cols + t1];
                        for (r, err) in errs.iter().enumerate().take(g_rows) {
                            let e = err[j];
                            if e != 0.0 {
                                for (dst, &u_val) in work[r * cols + t0..r * cols + t1]
                                    .iter_mut()
                                    .zip(u_tail)
                                {
                                    *dst -= e * u_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct E8BlockChol {
    u: [[f32; 8]; 8],
    inv_diag: [f32; 8],
    anisotropic: bool,
}

impl E8BlockChol {
    fn from_upper(u_mat: &[f32], cols: usize, col0: usize) -> Self {
        let mut u = [[0.0f32; 8]; 8];
        let mut inv_diag = [1.0f32; 8];
        let mut off_energy = 0.0f32;
        let mut diag_energy = 0.0f32;
        let mut min_d = f32::INFINITY;
        let mut max_d = 0.0f32;
        for (a, (u_row, dst_inv)) in u.iter_mut().zip(inv_diag.iter_mut()).enumerate() {
            let ga = col0 + a;
            let d = u_mat[ga * cols + ga].max(1e-12);
            u_row[a] = d;
            *dst_inv = 1.0 / d;
            diag_energy += d * d;
            min_d = min_d.min(d);
            max_d = max_d.max(d);
            for (b, slot) in u_row.iter_mut().enumerate().skip(a + 1) {
                let v = u_mat[ga * cols + col0 + b];
                *slot = v;
                off_energy += v * v;
            }
        }
        Self {
            u,
            inv_diag,
            anisotropic: off_energy > 1e-6 * diag_energy || max_d > 1.01 * min_d,
        }
    }

    #[inline]
    fn solve_err_and_norm2(&self, y: &[f32; 8], q: &[i8; 8]) -> ([f32; 8], f32) {
        let mut e = [0.0f32; 8];
        let mut n2 = 0.0f32;
        for a in 0..8 {
            let mut r = y[a] - (q[a] as f32);
            for (p, &ep) in e.iter().enumerate().take(a) {
                r -= self.u[p][a] * ep;
            }
            let ea = r * self.inv_diag[a];
            e[a] = ea;
            n2 += ea * ea;
        }
        (e, n2)
    }

    #[inline]
    fn eval_half_u(&self, y: &[f32; 8], t_idx: usize) -> (u8, [f32; 4], f32) {
        let tp = E8Q2_EVEN_TEMPLATES_I8[t_idx];
        let mut sgn4 = 0u8;
        let mut e4 = [0.0f32; 4];
        let mut cost4 = 0.0f32;
        for i in 0..4 {
            let mut yt = y[i];
            for (p, &ep) in e4.iter().enumerate().take(i) {
                yt -= ep * self.u[p][i];
            }
            let mag = tp[i];
            let qi = if mag == 0 {
                0.0f32
            } else if yt < 0.0 {
                sgn4 |= 1u8 << i;
                -(mag as f32)
            } else {
                mag as f32
            };
            let ei = (yt - qi) * self.inv_diag[i];
            e4[i] = ei;
            cost4 += ei * ei;
        }
        (sgn4, e4, cost4)
    }

    #[inline]
    fn eval_half_v(&self, y_v: &[f32; 4], t_idx: usize) -> (u8, [f32; 4], f32) {
        let tp = E8Q2_EVEN_TEMPLATES_I8[t_idx];
        let mut sgn4 = 0u8;
        let mut e4 = [0.0f32; 4];
        let mut cost4 = 0.0f32;
        for i in 0..4 {
            let mut yt = y_v[i];
            for (p, &ep) in e4.iter().enumerate().take(i) {
                yt -= ep * self.u[4 + p][4 + i];
            }
            let mag = tp[i];
            let qi = if mag == 0 {
                0.0f32
            } else if yt < 0.0 {
                sgn4 |= 1u8 << i;
                -(mag as f32)
            } else {
                mag as f32
            };
            let ei = (yt - qi) * self.inv_diag[4 + i];
            e4[i] = ei;
            cost4 += ei * ei;
        }
        (sgn4, e4, cost4)
    }

    /// Cholesky decision-feedback Gosset `E8Q2` quantizer minimizing `||(y - q) U^{-1}||_2^2`.
    #[inline]
    fn quantize_8d_chol(&self, y: &[f32; 8]) -> (u8, u8, [f32; 8], f32) {
        let (mut best_tpl, mut best_sgn, q_iso, _) = e8q2_quantize_8d(y);
        let (mut best_e, mut best_n2) = self.solve_err_and_norm2(y, &q_iso);
        if !self.anisotropic {
            return (best_tpl, best_sgn, best_e, best_n2);
        }

        // 1. Odd-coset 2-state Cholesky-Wagner trellis (`popcount(tpl ^ sgn)` even).
        let mut st_cost = [0.0f32, f32::INFINITY];
        let mut st_tpl = [0u8; 2];
        let mut st_sgn = [0u8; 2];
        let mut st_e = [[0.0f32; 8]; 2];
        for a in 0..8 {
            let inv_d = self.inv_diag[a];
            let mut nx_cost = [f32::INFINITY; 2];
            let mut nx_tpl = [0u8; 2];
            let mut nx_sgn = [0u8; 2];
            let mut nx_e = [[0.0f32; 8]; 2];
            for p_prev in 0..2 {
                let base_c = st_cost[p_prev];
                if !base_c.is_finite() {
                    continue;
                }
                let mut yt = y[a];
                for (k, &ek) in st_e[p_prev].iter().enumerate().take(a) {
                    yt -= ek * self.u[k][a];
                }
                let (b0, s0, q0) = if yt >= -16.0 {
                    (0u8, 0u8, 16.0f32)
                } else {
                    (1u8, 1u8, -48.0f32)
                };
                let ea0 = (yt - q0) * inv_d;
                let c0 = base_c + ea0 * ea0;
                if c0 < nx_cost[p_prev] {
                    nx_cost[p_prev] = c0;
                    nx_tpl[p_prev] = st_tpl[p_prev] | (b0 << a);
                    nx_sgn[p_prev] = st_sgn[p_prev] | (s0 << a);
                    nx_e[p_prev] = st_e[p_prev];
                    nx_e[p_prev][a] = ea0;
                }
                let (b1, s1, q1) = if yt >= 16.0 {
                    (1u8, 0u8, 48.0f32)
                } else {
                    (0u8, 1u8, -16.0f32)
                };
                let ea1 = (yt - q1) * inv_d;
                let c1 = base_c + ea1 * ea1;
                let p_odd = p_prev ^ 1;
                if c1 < nx_cost[p_odd] {
                    nx_cost[p_odd] = c1;
                    nx_tpl[p_odd] = st_tpl[p_prev] | (b1 << a);
                    nx_sgn[p_odd] = st_sgn[p_prev] | (s1 << a);
                    nx_e[p_odd] = st_e[p_prev];
                    nx_e[p_odd][a] = ea1;
                }
            }
            st_cost = nx_cost;
            st_tpl = nx_tpl;
            st_sgn = nx_sgn;
            st_e = nx_e;
        }
        if st_cost[0] < best_n2 {
            best_n2 = st_cost[0];
            best_tpl = st_tpl[0];
            best_sgn = st_sgn[0];
            best_e = st_e[0];
        }

        // 2. All-32 even-coset `(11, 11)` (`0xBB`) 2-state Cholesky-Wagner trellis (`popcount(sgn)` odd).
        let mut bb_cost = [0.0f32, f32::INFINITY];
        let mut bb_sgn = [0u8; 2];
        let mut bb_e = [[0.0f32; 8]; 2];
        for a in 0..8 {
            let inv_d = self.inv_diag[a];
            let mut nx_cost = [f32::INFINITY; 2];
            let mut nx_sgn = [0u8; 2];
            let mut nx_e = [[0.0f32; 8]; 2];
            for p_prev in 0..2 {
                let base_c = bb_cost[p_prev];
                if !base_c.is_finite() {
                    continue;
                }
                let mut yt = y[a];
                for (k, &ek) in bb_e[p_prev].iter().enumerate().take(a) {
                    yt -= ek * self.u[k][a];
                }
                let ea_pos = (yt - 32.0) * inv_d;
                let c_pos = base_c + ea_pos * ea_pos;
                if c_pos < nx_cost[p_prev] {
                    nx_cost[p_prev] = c_pos;
                    nx_sgn[p_prev] = bb_sgn[p_prev];
                    nx_e[p_prev] = bb_e[p_prev];
                    nx_e[p_prev][a] = ea_pos;
                }
                let ea_neg = (yt + 32.0) * inv_d;
                let c_neg = base_c + ea_neg * ea_neg;
                let p_odd = p_prev ^ 1;
                if c_neg < nx_cost[p_odd] {
                    nx_cost[p_odd] = c_neg;
                    nx_sgn[p_odd] = bb_sgn[p_prev] | (1u8 << a);
                    nx_e[p_odd] = bb_e[p_prev];
                    nx_e[p_odd][a] = ea_neg;
                }
            }
            bb_cost = nx_cost;
            bb_sgn = nx_sgn;
            bb_e = nx_e;
        }
        if bb_cost[1] < best_n2 {
            best_n2 = bb_cost[1];
            best_tpl = 0xBB;
            best_sgn = bb_sgn[1];
            best_e = bb_e[1];
        }

        // 3. Factored even-coset Cholesky feedback (`u[0..4]` -> propagate `e_u` -> `v[4..8]`).
        let mut u_evals = [(0u8, [0.0f32; 4], f32::INFINITY); 28];
        for (t, slot) in u_evals.iter_mut().enumerate() {
            *slot = self.eval_half_u(y, t);
        }
        let best_two_in = |start: usize, end: usize| -> (usize, usize) {
            let mut b0 = start;
            let mut b1 = start;
            let mut c0 = f32::INFINITY;
            let mut c1 = f32::INFINITY;
            for (t, ev) in u_evals.iter().enumerate().take(end).skip(start) {
                if ev.2 < c0 {
                    c1 = c0;
                    b1 = b0;
                    c0 = ev.2;
                    b0 = t;
                } else if ev.2 < c1 {
                    c1 = ev.2;
                    b1 = t;
                }
            }
            (b0, b1)
        };
        let (u0_a, u0_b) = best_two_in(0, 11);
        let (up_a, up_b) = best_two_in(12, 16);
        let (ua_a, ua_b) = best_two_in(16, 28);

        let u_groups: [(usize, usize, usize); 7] = [
            (u0_a, 0, 12),
            (u0_b, 0, 12),
            (11, 0, 11),
            (up_a, 12, 28),
            (up_b, 12, 28),
            (ua_a, 12, 16),
            (ua_b, 12, 16),
        ];

        for (k, &(u_idx, v_start, v_end)) in u_groups.iter().enumerate() {
            if k > 0 && u_groups[..k].iter().any(|&(prev_u, _, _)| prev_u == u_idx) {
                continue;
            }
            let (sgn_u, e_u, cost_u) = u_evals[u_idx];
            if cost_u >= best_n2 {
                continue;
            }
            let tu = if u_idx >= 16 { u_idx - 16 } else { u_idx };
            let mut y_v = [0.0f32; 4];
            for b in 0..4 {
                y_v[b] = y[4 + b]
                    - (e_u[0] * self.u[0][4 + b]
                        + e_u[1] * self.u[1][4 + b]
                        + e_u[2] * self.u[2][4 + b]
                        + e_u[3] * self.u[3][4 + b]);
            }
            for (v_idx, &tv_arr) in E8Q2_EVEN_TEMPLATES_I8
                .iter()
                .enumerate()
                .take(v_end)
                .skip(v_start)
            {
                let (sgn_v, e_v, cost_v) = self.eval_half_v(&y_v, v_idx);
                let tot = cost_u + cost_v;
                if tot < best_n2 {
                    let tv = if v_idx >= 16 { v_idx - 16 } else { v_idx };
                    let tpl_ev = (tu as u8) | ((tv as u8) << 4);
                    let mut sgn_ev = sgn_u | (sgn_v << 4);
                    if ((tpl_ev ^ sgn_ev).count_ones() & 1) == 0 {
                        let tu_arr = E8Q2_EVEN_TEMPLATES_I8[u_idx];
                        let mut zero_i = 0usize;
                        for i in 0..4 {
                            if tu_arr[i] == 0 {
                                zero_i = i;
                                break;
                            }
                            if tv_arr[i] == 0 {
                                zero_i = 4 + i;
                                break;
                            }
                        }
                        sgn_ev ^= 1u8 << zero_i;
                    }
                    best_n2 = tot;
                    best_tpl = tpl_ev;
                    best_sgn = sgn_ev;
                    best_e[..4].copy_from_slice(&e_u);
                    best_e[4..].copy_from_slice(&e_v);
                }
            }
        }

        debug_assert!({
            let q_chk = e8q2_decode_8d(best_tpl, best_sgn);
            let (_, n2_chk) = self.solve_err_and_norm2(y, &q_chk);
            (n2_chk - best_n2).abs() <= 1e-3 * best_n2.max(1.0)
        });
        (best_tpl, best_sgn, best_e, best_n2)
    }
}

/// Compiles `[rows, cols]` weights into a 2-bit Walsh-Hadamard + Gosset `E8Q2` [`NfSdotMatrix`]
/// (`bits = 2`, `2.125 bpw` at `G=128`, `2.25 bpw` at `G=64`) using Block-8 Cholesky GPTQ.
pub fn compile_hadamard_gptq_e8q2(
    weights: &[f32],
    rows: usize,
    cols: usize,
    scale_group_chunks: usize,
    hessian: Option<&FullRowGptqHessian>,
    num_threads: usize,
) -> NfSdotMatrix {
    assert_eq!(weights.len(), rows * cols);
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(HADAMARD_BLOCK));
    assert!((1..=8).contains(&scale_group_chunks));
    if let Some(h) = hessian {
        assert_eq!(h.cols, cols);
        assert_eq!(h.h_inv_chol_upper.len(), cols * cols);
    }

    let blocks_per_row = cols / 8;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / HADAMARD_BLOCK;
    let num_scale_groups = num_chunks.div_ceil(scale_group_chunks);

    let mut tpl_codes = vec![0u8; rows * blocks_per_row];
    let mut sgn_codes = vec![1u8; rows * blocks_per_row];
    let mut bf16_scales = vec![0u16; num_stripes * num_scale_groups * SIMD_ROW_STRIPE];

    let u_slice = hessian.map(|h| h.h_inv_chol_upper.as_slice());
    let block_chols: Option<Vec<E8BlockChol>> = u_slice.map(|u_mat| {
        (0..blocks_per_row)
            .map(|eb| E8BlockChol::from_upper(u_mat, cols, eb * 8))
            .collect()
    });
    let chols_ref = block_chols.as_deref();

    let workers = num_threads.max(1).min(num_stripes);
    let stripes_per_worker = num_stripes.div_ceil(workers);

    std::thread::scope(|s| {
        let mut rem_w = weights;
        let mut rem_tpl = tpl_codes.as_mut_slice();
        let mut rem_sgn = sgn_codes.as_mut_slice();
        let mut rem_scales = bf16_scales.as_mut_slice();
        let mut stripe_cursor = 0usize;

        while stripe_cursor < num_stripes && !rem_w.is_empty() {
            let take_stripes = stripes_per_worker.min(num_stripes - stripe_cursor);
            let row_start = stripe_cursor * SIMD_ROW_STRIPE;
            let row_end = ((stripe_cursor + take_stripes) * SIMD_ROW_STRIPE).min(rows);
            let take_rows = row_end - row_start;

            let (w_chunk, next_w) = rem_w.split_at(take_rows * cols);
            let (tpl_chunk, next_tpl) = rem_tpl.split_at_mut(take_rows * blocks_per_row);
            let (sgn_chunk, next_sgn) = rem_sgn.split_at_mut(take_rows * blocks_per_row);
            let (sc_chunk, next_sc) =
                rem_scales.split_at_mut(take_stripes * num_scale_groups * SIMD_ROW_STRIPE);

            rem_w = next_w;
            rem_tpl = next_tpl;
            rem_sgn = next_sgn;
            rem_scales = next_sc;
            stripe_cursor += take_stripes;

            s.spawn(move || {
                quantize_stripe_batch_e8q2(
                    w_chunk,
                    tpl_chunk,
                    sgn_chunk,
                    sc_chunk,
                    take_rows,
                    cols,
                    num_chunks,
                    scale_group_chunks,
                    num_scale_groups,
                    u_slice,
                    chols_ref,
                );
            });
        }
    });

    NfSdotMatrix::from_e8q2_codes(
        &tpl_codes,
        &sgn_codes,
        &bf16_scales,
        rows,
        cols,
        scale_group_chunks,
    )
}

#[allow(clippy::too_many_arguments)]
fn quantize_stripe_batch_e8q2(
    w_chunk: &[f32],
    tpl_chunk: &mut [u8],
    sgn_chunk: &mut [u8],
    sc_chunk: &mut [u16],
    take_rows: usize,
    cols: usize,
    num_chunks: usize,
    sgc: usize,
    nsg: usize,
    u_slice: Option<&[f32]>,
    chols: Option<&[E8BlockChol]>,
) {
    let ratios = [
        1.00f32, 0.94, 1.07, 0.88, 1.15, 0.82, 1.24, 0.76, 1.35, 0.70, 1.48, 0.64, 1.62,
    ];
    const TAIL_TILE: usize = 512;
    let blocks_per_row = cols / 8;
    let max_group_len = sgc * HADAMARD_BLOCK;
    let max_group_blocks = max_group_len / 8;

    let mut work = vec![0.0f32; SIMD_ROW_STRIPE * cols];
    let mut errs = vec![vec![0.0f32; max_group_len]; SIMD_ROW_STRIPE];
    let mut blk_work = vec![0.0f32; max_group_len];
    let mut best_tpl = vec![0u8; max_group_blocks];
    let mut best_sgn = vec![1u8; max_group_blocks];
    let mut cand_tpl = vec![0u8; max_group_blocks];
    let mut cand_sgn = vec![1u8; max_group_blocks];
    let mut cand_err = vec![0.0f32; max_group_len];

    for g0 in (0..take_rows).step_by(SIMD_ROW_STRIPE) {
        let g_rows = (take_rows - g0).min(SIMD_ROW_STRIPE);
        let local_stripe = g0 / SIMD_ROW_STRIPE;
        for r in 0..g_rows {
            let dst = &mut work[r * cols..(r + 1) * cols];
            dst.copy_from_slice(&w_chunk[(g0 + r) * cols..(g0 + r + 1) * cols]);
            fwht64_signed_inplace(dst);
        }

        for sg in 0..nsg {
            let c_start = sg * sgc;
            let c_end = (c_start + sgc).min(num_chunks);
            let col_s = c_start * HADAMARD_BLOCK;
            let col_e = c_end * HADAMARD_BLOCK;
            let group_len = col_e - col_s;
            let eb_s = col_s / 8;
            let eb_e = col_e / 8;
            let group_blocks = eb_e - eb_s;

            for (r, err_out) in errs.iter_mut().enumerate().take(g_rows) {
                let sub = &work[r * cols + col_s..r * cols + col_e];
                err_out[..group_len].fill(0.0);
                let sum_sq: f32 = sub.iter().map(|&v| v * v).sum();
                let rms = (sum_sq / (group_len as f32)).sqrt();
                let row = g0 + r;
                if rms <= 1e-12 {
                    tpl_chunk[row * blocks_per_row + eb_s..row * blocks_per_row + eb_e].fill(0u8);
                    sgn_chunk[row * blocks_per_row + eb_s..row * blocks_per_row + eb_e].fill(1u8);
                    sc_chunk[(local_stripe * nsg + sg) * SIMD_ROW_STRIPE + r] = 0;
                    continue;
                }

                let base_delta = rms / 35.0;
                let mut best_loss = f32::INFINITY;
                let mut best_bf16 = 0u16;

                let mut eval_delta = |d_bf16: u16,
                                      best_loss: &mut f32,
                                      best_bf16: &mut u16,
                                      best_tpl: &mut [u8],
                                      best_sgn: &mut [u8],
                                      err_out: &mut [f32]| {
                    let delta = bf16_to_f32(d_bf16);
                    if delta <= 0.0 {
                        return;
                    }
                    let inv_delta = 1.0 / delta;
                    let delta2 = delta * delta;
                    blk_work[..group_len].copy_from_slice(sub);
                    let mut loss = 0.0f32;

                    if let (Some(u_mat), Some(chol_arr)) = (u_slice, chols) {
                        for b in 0..group_blocks {
                            let j0 = b * 8;
                            let mut y8 = [0.0f32; 8];
                            for (a, dst_y) in y8.iter_mut().enumerate() {
                                *dst_y = blk_work[j0 + a] * inv_delta;
                            }
                            let bc = &chol_arr[eb_s + b];
                            let (tpl, sgn, e_tilde, n2) = bc.quantize_8d_chol(&y8);

                            cand_tpl[b] = tpl;
                            cand_sgn[b] = sgn;
                            loss += delta2 * n2;
                            if loss >= *best_loss {
                                return;
                            }
                            for a in 0..8 {
                                cand_err[j0 + a] = delta * e_tilde[a];
                            }
                            let j_next = j0 + 8;
                            if j_next < group_len {
                                for a in 0..8 {
                                    let ea = cand_err[j0 + a];
                                    if ea == 0.0 {
                                        continue;
                                    }
                                    let g_col = col_s + j0 + a;
                                    let u_sub =
                                        &u_mat[g_col * cols + col_s + j_next..g_col * cols + col_e];
                                    for (k_slot, &u_jk) in
                                        blk_work[j_next..group_len].iter_mut().zip(u_sub)
                                    {
                                        *k_slot -= ea * u_jk;
                                    }
                                }
                            }
                        }
                    } else {
                        for b in 0..group_blocks {
                            let j0 = b * 8;
                            let mut y8 = [0.0f32; 8];
                            for (a, dst_y) in y8.iter_mut().enumerate() {
                                *dst_y = blk_work[j0 + a] * inv_delta;
                            }
                            let (tpl, sgn, _, sq_err) = e8q2_quantize_8d(&y8);
                            cand_tpl[b] = tpl;
                            cand_sgn[b] = sgn;
                            loss += delta2 * sq_err;
                            if loss >= *best_loss {
                                return;
                            }
                        }
                    }

                    if loss < *best_loss {
                        *best_loss = loss;
                        *best_bf16 = d_bf16;
                        best_tpl[..group_blocks].copy_from_slice(&cand_tpl[..group_blocks]);
                        best_sgn[..group_blocks].copy_from_slice(&cand_sgn[..group_blocks]);
                        err_out[..group_len].copy_from_slice(&cand_err[..group_len]);
                    }
                };

                for &rat in &ratios {
                    eval_delta(
                        f32_to_bf16(base_delta * rat),
                        &mut best_loss,
                        &mut best_bf16,
                        &mut best_tpl,
                        &mut best_sgn,
                        err_out,
                    );
                }

                // Least-squares BF16 scale refit on the chosen Gosset lattice vectors.
                let mut dot_wc = 0.0f32;
                let mut dot_cc = 0.0f32;
                for b in 0..group_blocks {
                    let q8 = e8q2_decode_8d(best_tpl[b], best_sgn[b]);
                    let j0 = b * 8;
                    for a in 0..8 {
                        let qf = q8[a] as f32;
                        dot_wc += sub[j0 + a] * qf;
                        dot_cc += qf * qf;
                    }
                }
                if dot_cc > 1e-12 && dot_wc > 0.0 {
                    let ls_bf16 = f32_to_bf16(dot_wc / dot_cc);
                    if ls_bf16 != best_bf16 {
                        eval_delta(
                            ls_bf16,
                            &mut best_loss,
                            &mut best_bf16,
                            &mut best_tpl,
                            &mut best_sgn,
                            err_out,
                        );
                    }
                }

                sc_chunk[(local_stripe * nsg + sg) * SIMD_ROW_STRIPE + r] = best_bf16;
                tpl_chunk[row * blocks_per_row + eb_s..row * blocks_per_row + eb_e]
                    .copy_from_slice(&best_tpl[..group_blocks]);
                sgn_chunk[row * blocks_per_row + eb_s..row * blocks_per_row + eb_e]
                    .copy_from_slice(&best_sgn[..group_blocks]);
            }

            if let (true, Some(u_mat)) = (col_e < cols, u_slice) {
                for t0 in (col_e..cols).step_by(TAIL_TILE) {
                    let t1 = (t0 + TAIL_TILE).min(cols);
                    for j in 0..group_len {
                        let g_col = col_s + j;
                        let u_tail = &u_mat[g_col * cols + t0..g_col * cols + t1];
                        for (r, err) in errs.iter().enumerate().take(g_rows) {
                            let e = err[j];
                            if e != 0.0 {
                                for (dst, &u_val) in work[r * cols + t0..r * cols + t1]
                                    .iter_mut()
                                    .zip(u_tail)
                                {
                                    *dst -= e * u_val;
                                }
                            }
                        }
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
    use crate::kernels::simd_lut::{fused_kronecker_gemv, QuantizedActivationLut};

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
        use crate::kernels::simd_lut::{simd_lut_gemv_multiplane, simd_lut_gemv_multiplane_scalar};

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

    #[test]
    fn test_full_row_hessian_inverse_cholesky() {
        let c = 128;
        let mut h = FullRowGptqHessian::new(c);
        let xs: Vec<Vec<f32>> = (0..1500)
            .map(|t| (0..c).map(|i| (((t * 131 + i * 71) % 97) as f32 / 97.0 - 0.5) * (1.0 + (i % 5) as f32)).collect())
            .collect();
        let mut raw = vec![0.0f64; c * c];
        for x in &xs {
            let mut xr = x.clone();
            fwht64_signed_inplace(&mut xr);
            for i in 0..c {
                for j in 0..c {
                    raw[i * c + j] += xr[i] as f64 * xr[j] as f64;
                }
            }
            h.observe_activation(x);
        }
        h.finalize(0.01);
        let u = &h.h_inv_chol_upper;
        assert!(u.iter().all(|v| v.is_finite()), "non-finite U");
        // (U^T U) * H_damped should be ~identity.
        let n = xs.len() as f64;
        let mean_diag = (0..c).map(|i| raw[i * c + i] / n).sum::<f64>() / c as f64;
        let hd = |i: usize, j: usize| {
            let v = raw[i * c + j] / n;
            if i == j { v + 0.005 * mean_diag } else { v * 0.99 }
        };
        let mut max_err = 0.0f64;
        for i in (0..c).step_by(7) {
            for j in (0..c).step_by(5) {
                let mut acc = 0.0f64;
                for k in 0..c {
                    let utu: f64 = (0..=i.min(k)).map(|r| u[r * c + i] as f64 * u[r * c + k] as f64).sum();
                    acc += utu * hd(k, j);
                }
                let want = if i == j { 1.0 } else { 0.0 };
                max_err = max_err.max((acc - want).abs());
            }
        }
        assert!(max_err < 1e-2, "U^T U H != I (max err {max_err})");
    }
}
