//! Hardware SIMD Register Lookup-Table GEMV (`aarch64` ARM NEON `vqtbl1q_s8` + `vqtbl4q_u8`).
//!
//! Provides two zero-multiply execution paths:
//! 1. **Static Ternary Stripe GEMV ([`simd_lut_gemv`])**: Executes pre-packed 16-row ternary stripes
//!    using 16-byte NEON register LUTs (`vqtbl1q_s8`).
//! 2. **Register-Fused Multi-Rank Kronecker GEMV ([`fused_kronecker_gemv`])**: Holds a 64-byte
//!    canonical pattern codebook inside 4 NEON registers (`uint8x16x4_t`, `vqtbl4q_u8`) and
//!    synthesizes weights **on the fly inside SIMD registers** immediately before `vqtbl1q_s8`—
//!    eliminating 100% of L1 tile-buffer memory reads and writes!

use crate::kernels::lut_gemv::{GROUP_SIZE, LUT_SIZE};
use crate::kernels::synthesis::KroneckerCompressedExpert;

/// Number of matrix rows processed simultaneously in a single 128-bit SIMD stripe.
pub const SIMD_ROW_STRIPE: usize = 16;

/// Number of 4-column blocks accumulated in `i16` before flushing to `f32` (`16 * 4 = 64` cols).
/// Group-64 (`G=64`) isolates activation & weight outliers in real LLMs (`SwiGLU` / `Attention`).
pub const QUANT_CHUNK_BLOCKS: usize = 16;

/// Compile-time map from base-3 4-trit code (`0..81`) to nibble-packed 2x2-trit byte:
/// `low_nibble = code % 9` (`0..9`), `high_nibble = code / 9` (`0..9`).
pub const BASE3_TO_NIBBLE: [u8; LUT_SIZE] = {
    let mut table = [0u8; LUT_SIZE];
    let mut c = 0;
    while c < LUT_SIZE {
        let lo = (c % 9) as u8;
        let hi = (c / 9) as u8;
        table[c] = lo | (hi << 4);
        c += 1;
    }
    table
};

/// Canonical 256-entry L1 codebook emitting nibble-packed `vqtbl1q_s8` codes.
pub fn default_simd_codebook() -> [u8; 256] {
    let mut cb = [0u8; 256];
    for (i, slot) in cb.iter_mut().enumerate() {
        *slot = BASE3_TO_NIBBLE[i % LUT_SIZE];
    }
    cb
}

/// Canonical 64-entry register-resident codebook (`uint8x16x4_t` for `vqtbl4q_u8`).
/// Entry 0 is `0x44` `(0, 0, 0, 0)`; entries `1..64` stride across all 81 4-trit patterns
/// (`(i * 5) % 81`, since `gcd(5, 81) == 1`) to maximize orthogonal motif coverage.
pub const fn default_register_codebook_64() -> [u8; 64] {
    let mut cb = [0u8; 64];
    cb[0] = BASE3_TO_NIBBLE[40]; // (0, 0, 0, 0) -> code 40 -> 0x44
    let mut i = 1;
    while i < 64 {
        let idx81 = (i * 5 + 40) % LUT_SIZE;
        cb[i] = BASE3_TO_NIBBLE[idx81];
        i += 1;
    }
    cb
}

/// Decodes a single nibble-packed byte (`lo | (hi << 4)`) back into 4 signed trits `{-1, 0, +1}`.
#[inline]
pub fn decode_nibble_code(code: u8) -> [i8; 4] {
    let lo = code & 0x0F;
    let hi = code >> 4;
    let w0 = ((lo % 3) as i8) - 1;
    let w1 = ((lo / 3) as i8) - 1;
    let w2 = ((hi % 3) as i8) - 1;
    let w3 = ((hi / 3) as i8) - 1;
    [w0, w1, w2, w3]
}

/// 16-row SIMD stripe ternary matrix (`[num_stripes, blocks_per_row, 16]`).
#[derive(Clone, Debug)]
pub struct SimdPackedMatrix {
    pub rows: usize,
    pub cols: usize,
    pub blocks_per_row: usize,
    pub num_stripes: usize,
    pub stripe_data: Vec<u8>,
}

impl SimdPackedMatrix {
    pub fn new_zeroed(rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "rows must be > 0");
        assert!(
            cols > 0 && cols.is_multiple_of(GROUP_SIZE),
            "cols ({cols}) must be > 0 and divisible by {GROUP_SIZE}"
        );
        let blocks_per_row = cols / GROUP_SIZE;
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        Self {
            rows,
            cols,
            blocks_per_row,
            num_stripes,
            stripe_data: vec![0u8; num_stripes * blocks_per_row * SIMD_ROW_STRIPE],
        }
    }

    pub fn from_ternary_weights(weights: &[i8], rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        let mut mat = Self::new_zeroed(rows, cols);
        let blocks_per_row = mat.blocks_per_row;

        for r in 0..rows {
            let stripe = r / SIMD_ROW_STRIPE;
            let lane = r % SIMD_ROW_STRIPE;
            let stripe_base = stripe * blocks_per_row * SIMD_ROW_STRIPE;
            let row_w = &weights[r * cols..(r + 1) * cols];

            for (b, w) in row_w.chunks_exact(GROUP_SIZE).enumerate() {
                for &val in w {
                    assert!(
                        (-1..=1).contains(&val),
                        "ternary weight must be in {{-1, 0, 1}}, got {val}"
                    );
                }
                let lo = ((w[0] + 1) + 3 * (w[1] + 1)) as u8;
                let hi = ((w[2] + 1) + 3 * (w[3] + 1)) as u8;
                mat.stripe_data[stripe_base + b * SIMD_ROW_STRIPE + lane] = lo | (hi << 4);
            }
        }
        mat
    }

    pub fn synthesize_from_seed_slices(
        &mut self,
        row_seeds: &[u8],
        col_seeds: &[u8],
        simd_codebook: &[u8; 256],
    ) {
        assert_eq!(self.rows, row_seeds.len());
        assert_eq!(self.blocks_per_row, col_seeds.len());
        let bpr = self.blocks_per_row;

        for s in 0..self.num_stripes {
            let row_start = s * SIMD_ROW_STRIPE;
            let valid_lanes = (self.rows - row_start).min(SIMD_ROW_STRIPE);
            let stripe_slice =
                &mut self.stripe_data[s * bpr * SIMD_ROW_STRIPE..(s + 1) * bpr * SIMD_ROW_STRIPE];

            if valid_lanes == SIMD_ROW_STRIPE {
                let r_seeds = &row_seeds[row_start..row_start + SIMD_ROW_STRIPE];
                for (col_vec, &c_seed) in stripe_slice
                    .chunks_exact_mut(SIMD_ROW_STRIPE)
                    .zip(col_seeds.iter())
                {
                    let rot_c = c_seed.rotate_left(3);
                    for lane in 0..SIMD_ROW_STRIPE {
                        col_vec[lane] = simd_codebook[(r_seeds[lane] ^ rot_c) as usize];
                    }
                }
            } else {
                for (b, &c_seed) in col_seeds.iter().enumerate() {
                    let rot_c = c_seed.rotate_left(3);
                    let col_vec =
                        &mut stripe_slice[b * SIMD_ROW_STRIPE..(b + 1) * SIMD_ROW_STRIPE];
                    for (slot, &r_seed) in col_vec[..valid_lanes]
                        .iter_mut()
                        .zip(&row_seeds[row_start..row_start + valid_lanes])
                    {
                        *slot = simd_codebook[(r_seed ^ rot_c) as usize];
                    }
                    for slot in &mut col_vec[valid_lanes..] {
                        *slot = 0x44;
                    }
                }
            }
        }
    }

    pub fn synthesize_from_kronecker(
        &mut self,
        expert: &KroneckerCompressedExpert,
        simd_codebook: &[u8; 256],
    ) {
        self.synthesize_from_seed_slices(&expert.row_seeds, &expert.col_seeds, simd_codebook);
    }
}

/// Precomputed register-sized `INT8` lookup tables for an activation vector `x`.
/// Supports zero-allocation reuse via [`Self::new_zeroed`] + [`Self::populate`].
#[derive(Clone, Debug)]
pub struct QuantizedActivationLut {
    pub num_blocks: usize,
    pub lut_lo: Vec<[i8; 16]>,
    pub lut_hi: Vec<[i8; 16]>,
    pub chunk_scales: Vec<f32>,
}

impl QuantizedActivationLut {
    /// Pre-allocates reusable LUT storage for vectors of length `cols`.
    pub fn new_zeroed(cols: usize) -> Self {
        assert!(cols > 0 && cols.is_multiple_of(GROUP_SIZE));
        let num_blocks = cols / GROUP_SIZE;
        let num_chunks = num_blocks.div_ceil(QUANT_CHUNK_BLOCKS);
        Self {
            num_blocks,
            lut_lo: vec![[0i8; 16]; num_blocks],
            lut_hi: vec![[0i8; 16]; num_blocks],
            chunk_scales: vec![0.0f32; num_chunks],
        }
    }

    /// Populates `self` in-place from `x` with **zero heap allocations**.
    /// Uses direct `FP32` pair-sum rounding over full `[-127, +127]` `INT8` range (`max(|x_a|+|x_b|) / 127`).
    pub fn populate(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.num_blocks * GROUP_SIZE);
        let chunk_elems = QUANT_CHUNK_BLOCKS * GROUP_SIZE;

        for (chunk_idx, (x_chunk, scale_slot)) in x
            .chunks(chunk_elems)
            .zip(self.chunk_scales.iter_mut())
            .enumerate()
        {
            let mut max_pair_l1 = 0.0f32;
            for vals in x_chunk.chunks_exact(GROUP_SIZE) {
                let p0 = vals[0].abs() + vals[1].abs();
                let p1 = vals[2].abs() + vals[3].abs();
                max_pair_l1 = max_pair_l1.max(p0).max(p1);
            }
            let base_block = chunk_idx * QUANT_CHUNK_BLOCKS;
            let block_count = x_chunk.len() / GROUP_SIZE;

            if max_pair_l1 <= f32::EPSILON {
                *scale_slot = 0.0;
                self.lut_lo[base_block..base_block + block_count].fill([0i8; 16]);
                self.lut_hi[base_block..base_block + block_count].fill([0i8; 16]);
                continue;
            }

            let scale = max_pair_l1 / 126.0;
            let inv_scale = 126.0 / max_pair_l1;
            *scale_slot = scale;

            for (local_b, vals) in x_chunk.chunks_exact(GROUP_SIZE).enumerate() {
                let b = base_block + local_b;
                let [q0, q1, q2, q3] = round_quad_to_i8(vals, inv_scale);
                self.lut_lo[b] = Self::build_pair_table_from_i8(q0, q1);
                self.lut_hi[b] = Self::build_pair_table_from_i8(q2, q3);
            }
        }
    }

    /// Allocates and builds a new `QuantizedActivationLut` from `x`.
    pub fn build(x: &[f32]) -> Self {
        let mut qlut = Self::new_zeroed(x.len());
        qlut.populate(x);
        qlut
    }

    #[inline]
    fn build_pair_table_from_i8(qa: i8, qb: i8) -> [i8; 16] {
        // Because `|va| + |vb| <= max_pair_l1`, `|sa| + |sb| <= 126.0`, so `|qa| + |qb| <= 127`!
        let sum = qa + qb;
        let diff = qa - qb;
        [
            -sum, -qb, diff, -qa, 0, qa, -diff, qb, sum, 0, 0, 0, 0, 0, 0, 0,
        ]
    }
}

/// Executes register-LUT GEMV on a static `SimdPackedMatrix`.
#[inline]
pub fn simd_lut_gemv(matrix: &SimdPackedMatrix, qlut: &QuantizedActivationLut, out: &mut [f32]) {
    assert_eq!(matrix.blocks_per_row, qlut.num_blocks);
    assert_eq!(out.len(), matrix.rows);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is mandatory on `aarch64` and slice bounds are checked above.
        unsafe { simd_lut_gemv_neon(matrix, qlut, out) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        simd_lut_gemv_scalar(matrix, qlut, out);
    }
}

/// Executes **Register-Fused Multi-Rank Kronecker Synthesis + `LUT-GEMV`** with ZERO L1 tile buffer!
///
/// - `row_seeds`: `[rank * rows]`
/// - `col_seeds`: `[rank * blocks_per_row]`
/// - `rank_scales`: `[rank]` (per-rank weight scale $\alpha_r$)
/// - `reg_codebook`: 64-byte register-resident pattern codebook (`uint8x16x4_t` in NEON)
#[inline]
pub fn fused_kronecker_gemv(
    row_seeds: &[u8],
    col_seeds: &[u8],
    rank_scales: &[f32],
    rows: usize,
    reg_codebook: &[u8; 64],
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    let rank = rank_scales.len();
    let bpr = qlut.num_blocks;
    assert!(rank > 0 && rows > 0);
    assert_eq!(row_seeds.len(), rank * rows);
    assert_eq!(col_seeds.len(), rank * bpr);
    assert_eq!(out.len(), rows);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`; `row_seeds`, `col_seeds`, `reg_codebook`,
        // `qlut`, and `out` lengths are all verified by assertions above.
        unsafe {
            fused_kronecker_gemv_neon(
                row_seeds,
                col_seeds,
                rank_scales,
                rows,
                reg_codebook,
                qlut,
                out,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        fused_kronecker_gemv_scalar(
            row_seeds,
            col_seeds,
            rank_scales,
            rows,
            reg_codebook,
            qlut,
            out,
        );
    }
}

/// Portable scalar reference for [`fused_kronecker_gemv`] with identical chunked rounding.
pub fn fused_kronecker_gemv_scalar(
    row_seeds: &[u8],
    col_seeds: &[u8],
    rank_scales: &[f32],
    rows: usize,
    reg_codebook: &[u8; 64],
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    let rank = rank_scales.len();
    let bpr = qlut.num_blocks;
    assert_eq!(row_seeds.len(), rank * rows);
    assert_eq!(col_seeds.len(), rank * bpr);
    assert_eq!(out.len(), rows);

    out.fill(0.0);
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);

    for (r_idx, &alpha) in rank_scales.iter().enumerate() {
        if alpha == 0.0 {
            continue;
        }
        let r_base = &row_seeds[r_idx * rows..(r_idx + 1) * rows];
        let c_base = &col_seeds[r_idx * bpr..(r_idx + 1) * bpr];

        for s in 0..num_stripes {
            let row_start = s * SIMD_ROW_STRIPE;
            let valid_lanes = (rows - row_start).min(SIMD_ROW_STRIPE);
            let mut r_stripe = [0u8; SIMD_ROW_STRIPE];
            r_stripe[..valid_lanes].copy_from_slice(&r_base[row_start..row_start + valid_lanes]);

            let mut facc = [0.0f32; SIMD_ROW_STRIPE];
            for (chunk_idx, &qscale) in qlut.chunk_scales.iter().enumerate() {
                if qscale == 0.0 {
                    continue;
                }
                let eff_scale = alpha * qscale;
                let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
                let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);
                let mut iacc = [0i16; SIMD_ROW_STRIPE];

                for (b, &c_seed) in c_base[b_start..b_end].iter().enumerate() {
                    let abs_b = b_start + b;
                    let c_rot = c_seed.rotate_left(3);
                    let t_lo = &qlut.lut_lo[abs_b];
                    let t_hi = &qlut.lut_hi[abs_b];
                    for lane in 0..valid_lanes {
                        let pat = ((r_stripe[lane] ^ c_rot) & 0x3F) as usize;
                        let code = reg_codebook[pat];
                        let sum_lo = t_lo[(code & 0x0F) as usize] as i16;
                        let sum_hi = t_hi[(code >> 4) as usize] as i16;
                        iacc[lane] += sum_lo + sum_hi;
                    }
                }

                for lane in 0..valid_lanes {
                    facc[lane] += (iacc[lane] as f32) * eff_scale;
                }
            }

            for lane in 0..valid_lanes {
                out[row_start + lane] += facc[lane];
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fused_kronecker_gemv_neon(
    row_seeds: &[u8],
    col_seeds: &[u8],
    rank_scales: &[f32],
    rows: usize,
    reg_codebook: &[u8; 64],
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    out.fill(0.0);
    let bpr = qlut.num_blocks;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let mask_3f = vdupq_n_u8(0x3F);
    let mask_0f = vdupq_n_u8(0x0F);

    // Pin the entire 64-byte pattern codebook into 4 NEON registers (`cb_regs`) ONCE!
    // SAFETY: `reg_codebook` has exact length 64 bytes (`4 * 16`).
    let cb_regs = unsafe {
        let p = reg_codebook.as_ptr();
        uint8x16x4_t(
            vld1q_u8(p),
            vld1q_u8(p.add(16)),
            vld1q_u8(p.add(32)),
            vld1q_u8(p.add(48)),
        )
    };

    for (r_idx, &alpha) in rank_scales.iter().enumerate() {
        if alpha == 0.0 {
            continue;
        }
        let r_base = &row_seeds[r_idx * rows..(r_idx + 1) * rows];
        let c_base = &col_seeds[r_idx * bpr..(r_idx + 1) * bpr];

        for s in 0..num_stripes {
            let row_start = s * SIMD_ROW_STRIPE;
            let valid_lanes = (rows - row_start).min(SIMD_ROW_STRIPE);

            let r_vec = if valid_lanes == SIMD_ROW_STRIPE {
                // SAFETY: `row_start + 16 <= r_base.len()`.
                unsafe { vld1q_u8(r_base.as_ptr().add(row_start)) }
            } else {
                let mut tmp_r = [0u8; SIMD_ROW_STRIPE];
                tmp_r[..valid_lanes].copy_from_slice(&r_base[row_start..row_start + valid_lanes]);
                // SAFETY: `tmp_r` is 16 bytes on the stack.
                unsafe { vld1q_u8(tmp_r.as_ptr()) }
            };

            let mut facc0 = vdupq_n_f32(0.0);
            let mut facc1 = vdupq_n_f32(0.0);
            let mut facc2 = vdupq_n_f32(0.0);
            let mut facc3 = vdupq_n_f32(0.0);

            for (chunk_idx, &qscale) in qlut.chunk_scales.iter().enumerate() {
                if qscale == 0.0 {
                    continue;
                }
                let eff_scale = alpha * qscale;
                let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
                let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);

                let mut iacc_lo_a = vdupq_n_s16(0);
                let mut iacc_hi_a = vdupq_n_s16(0);
                let mut iacc_lo_b = vdupq_n_s16(0);
                let mut iacc_hi_b = vdupq_n_s16(0);

                let mut b = b_start;
                while b + 2 <= b_end {
                    let c_rot0 = vdupq_n_u8(c_base[b].rotate_left(3));
                    let c_rot1 = vdupq_n_u8(c_base[b + 1].rotate_left(3));

                    // Synthesize 2 x 16 = 32 bytes (128 ternary weights!) directly inside NEON registers!
                    let pat0 = vandq_u8(veorq_u8(r_vec, c_rot0), mask_3f);
                    let pat1 = vandq_u8(veorq_u8(r_vec, c_rot1), mask_3f);
                    let codes0 = vqtbl4q_u8(cb_regs, pat0);
                    let codes1 = vqtbl4q_u8(cb_regs, pat1);

                    // SAFETY: `b + 1 < bpr`, each LUT entry is `[i8; 16]`.
                    unsafe {
                        let tbl_lo0 = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi0 = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let tbl_lo1 = vld1q_s8(qlut.lut_lo.as_ptr().add(b + 1).cast::<i8>());
                        let tbl_hi1 = vld1q_s8(qlut.lut_hi.as_ptr().add(b + 1).cast::<i8>());

                        let sum_lo0 = vqtbl1q_s8(tbl_lo0, vandq_u8(codes0, mask_0f));
                        let sum_hi0 = vqtbl1q_s8(tbl_hi0, vshrq_n_u8::<4>(codes0));
                        let sum_lo1 = vqtbl1q_s8(tbl_lo1, vandq_u8(codes1, mask_0f));
                        let sum_hi1 = vqtbl1q_s8(tbl_hi1, vshrq_n_u8::<4>(codes1));

                        iacc_lo_a = vaddq_s16(
                            iacc_lo_a,
                            vaddl_s8(vget_low_s8(sum_lo0), vget_low_s8(sum_hi0)),
                        );
                        iacc_hi_a = vaddq_s16(iacc_hi_a, vaddl_high_s8(sum_lo0, sum_hi0));
                        iacc_lo_b = vaddq_s16(
                            iacc_lo_b,
                            vaddl_s8(vget_low_s8(sum_lo1), vget_low_s8(sum_hi1)),
                        );
                        iacc_hi_b = vaddq_s16(iacc_hi_b, vaddl_high_s8(sum_lo1, sum_hi1));
                    }
                    b += 2;
                }

                while b < b_end {
                    let c_rot = vdupq_n_u8(c_base[b].rotate_left(3));
                    let pat = vandq_u8(veorq_u8(r_vec, c_rot), mask_3f);
                    let codes = vqtbl4q_u8(cb_regs, pat);

                    // SAFETY: `b < bpr`.
                    unsafe {
                        let tbl_lo = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let sum_lo = vqtbl1q_s8(tbl_lo, vandq_u8(codes, mask_0f));
                        let sum_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(codes));
                        iacc_lo_a = vaddq_s16(
                            iacc_lo_a,
                            vaddl_s8(vget_low_s8(sum_lo), vget_low_s8(sum_hi)),
                        );
                        iacc_hi_a = vaddq_s16(iacc_hi_a, vaddl_high_s8(sum_lo, sum_hi));
                    }
                    b += 1;
                }

                let iacc_lo = vaddq_s16(iacc_lo_a, iacc_lo_b);
                let iacc_hi = vaddq_s16(iacc_hi_a, iacc_hi_b);
                let vscale = vdupq_n_f32(eff_scale);

                let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iacc_lo)));
                let f1 = vcvtq_f32_s32(vmovl_high_s16(iacc_lo));
                let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iacc_hi)));
                let f3 = vcvtq_f32_s32(vmovl_high_s16(iacc_hi));

                facc0 = vfmaq_f32(facc0, f0, vscale);
                facc1 = vfmaq_f32(facc1, f1, vscale);
                facc2 = vfmaq_f32(facc2, f2, vscale);
                facc3 = vfmaq_f32(facc3, f3, vscale);
            }

            if valid_lanes == SIMD_ROW_STRIPE {
                // SAFETY: `row_start + 16 <= out.len()`.
                unsafe {
                    let dst = out.as_mut_ptr().add(row_start);
                    vst1q_f32(dst, vaddq_f32(vld1q_f32(dst), facc0));
                    vst1q_f32(dst.add(4), vaddq_f32(vld1q_f32(dst.add(4)), facc1));
                    vst1q_f32(dst.add(8), vaddq_f32(vld1q_f32(dst.add(8)), facc2));
                    vst1q_f32(dst.add(12), vaddq_f32(vld1q_f32(dst.add(12)), facc3));
                }
            } else {
                let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
                // SAFETY: `tmp` has length 16.
                unsafe {
                    let dst = tmp.as_mut_ptr();
                    vst1q_f32(dst, facc0);
                    vst1q_f32(dst.add(4), facc1);
                    vst1q_f32(dst.add(8), facc2);
                    vst1q_f32(dst.add(12), facc3);
                }
                for lane in 0..valid_lanes {
                    out[row_start + lane] += tmp[lane];
                }
            }
        }
    }
}

/// Portable scalar reference for [`simd_lut_gemv`].
pub fn simd_lut_gemv_scalar(
    matrix: &SimdPackedMatrix,
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    assert_eq!(matrix.blocks_per_row, qlut.num_blocks);
    assert_eq!(out.len(), matrix.rows);

    let bpr = matrix.blocks_per_row;

    for s in 0..matrix.num_stripes {
        let stripe_slice =
            &matrix.stripe_data[s * bpr * SIMD_ROW_STRIPE..(s + 1) * bpr * SIMD_ROW_STRIPE];
        let mut facc = [0.0f32; SIMD_ROW_STRIPE];

        for (chunk_idx, &scale) in qlut.chunk_scales.iter().enumerate() {
            if scale == 0.0 {
                continue;
            }
            let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
            let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);
            let mut iacc = [0i16; SIMD_ROW_STRIPE];

            for b in b_start..b_end {
                let codes = &stripe_slice[b * SIMD_ROW_STRIPE..(b + 1) * SIMD_ROW_STRIPE];
                let t_lo = &qlut.lut_lo[b];
                let t_hi = &qlut.lut_hi[b];
                for lane in 0..SIMD_ROW_STRIPE {
                    let code = codes[lane];
                    let sum_lo = t_lo[(code & 0x0F) as usize] as i16;
                    let sum_hi = t_hi[(code >> 4) as usize] as i16;
                    iacc[lane] += sum_lo + sum_hi;
                }
            }

            for lane in 0..SIMD_ROW_STRIPE {
                facc[lane] += (iacc[lane] as f32) * scale;
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        out[row_start..row_start + valid_lanes].copy_from_slice(&facc[..valid_lanes]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_lut_gemv_neon(
    matrix: &SimdPackedMatrix,
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    let bpr = matrix.blocks_per_row;
    let mask_0f = vdupq_n_u8(0x0F);

    for s in 0..matrix.num_stripes {
        // SAFETY: `s < matrix.num_stripes` and `stripe_data.len() == num_stripes * bpr * 16`.
        let stripe_ptr = unsafe { matrix.stripe_data.as_ptr().add(s * bpr * SIMD_ROW_STRIPE) };

        let mut facc0 = vdupq_n_f32(0.0);
        let mut facc1 = vdupq_n_f32(0.0);
        let mut facc2 = vdupq_n_f32(0.0);
        let mut facc3 = vdupq_n_f32(0.0);

        for (chunk_idx, &scale) in qlut.chunk_scales.iter().enumerate() {
            if scale == 0.0 {
                continue;
            }
            let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
            let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);

            let mut iacc_lo_a = vdupq_n_s16(0);
            let mut iacc_hi_a = vdupq_n_s16(0);
            let mut iacc_lo_b = vdupq_n_s16(0);
            let mut iacc_hi_b = vdupq_n_s16(0);

            let mut b = b_start;
            while b + 2 <= b_end {
                // SAFETY: `b + 1 < bpr`, valid 16-byte loads.
                unsafe {
                    let tbl_lo0 = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                    let tbl_hi0 = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                    let codes0 = vld1q_u8(stripe_ptr.add(b * SIMD_ROW_STRIPE));

                    let tbl_lo1 = vld1q_s8(qlut.lut_lo.as_ptr().add(b + 1).cast::<i8>());
                    let tbl_hi1 = vld1q_s8(qlut.lut_hi.as_ptr().add(b + 1).cast::<i8>());
                    let codes1 = vld1q_u8(stripe_ptr.add((b + 1) * SIMD_ROW_STRIPE));

                    let sum_lo0 = vqtbl1q_s8(tbl_lo0, vandq_u8(codes0, mask_0f));
                    let sum_hi0 = vqtbl1q_s8(tbl_hi0, vshrq_n_u8::<4>(codes0));
                    let sum_lo1 = vqtbl1q_s8(tbl_lo1, vandq_u8(codes1, mask_0f));
                    let sum_hi1 = vqtbl1q_s8(tbl_hi1, vshrq_n_u8::<4>(codes1));

                    iacc_lo_a = vaddq_s16(
                        iacc_lo_a,
                        vaddl_s8(vget_low_s8(sum_lo0), vget_low_s8(sum_hi0)),
                    );
                    iacc_hi_a = vaddq_s16(iacc_hi_a, vaddl_high_s8(sum_lo0, sum_hi0));

                    iacc_lo_b = vaddq_s16(
                        iacc_lo_b,
                        vaddl_s8(vget_low_s8(sum_lo1), vget_low_s8(sum_hi1)),
                    );
                    iacc_hi_b = vaddq_s16(iacc_hi_b, vaddl_high_s8(sum_lo1, sum_hi1));
                }
                b += 2;
            }

            while b < b_end {
                // SAFETY: `b < bpr`.
                unsafe {
                    let tbl_lo = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                    let tbl_hi = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                    let codes = vld1q_u8(stripe_ptr.add(b * SIMD_ROW_STRIPE));

                    let sum_lo = vqtbl1q_s8(tbl_lo, vandq_u8(codes, mask_0f));
                    let sum_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(codes));

                    iacc_lo_a = vaddq_s16(
                        iacc_lo_a,
                        vaddl_s8(vget_low_s8(sum_lo), vget_low_s8(sum_hi)),
                    );
                    iacc_hi_a = vaddq_s16(iacc_hi_a, vaddl_high_s8(sum_lo, sum_hi));
                }
                b += 1;
            }

            let iacc_lo = vaddq_s16(iacc_lo_a, iacc_lo_b);
            let iacc_hi = vaddq_s16(iacc_hi_a, iacc_hi_b);
            let vscale = vdupq_n_f32(scale);

            let f0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iacc_lo)));
            let f1 = vcvtq_f32_s32(vmovl_high_s16(iacc_lo));
            let f2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(iacc_hi)));
            let f3 = vcvtq_f32_s32(vmovl_high_s16(iacc_hi));

            facc0 = vfmaq_f32(facc0, f0, vscale);
            facc1 = vfmaq_f32(facc1, f1, vscale);
            facc2 = vfmaq_f32(facc2, f2, vscale);
            facc3 = vfmaq_f32(facc3, f3, vscale);
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        if valid_lanes == SIMD_ROW_STRIPE {
            // SAFETY: `row_start + 16 <= out.len()`.
            unsafe {
                let dst = out.as_mut_ptr().add(row_start);
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
            }
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            // SAFETY: `tmp` has exact length 16.
            unsafe {
                let dst = tmp.as_mut_ptr();
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
            }
            out[row_start..row_start + valid_lanes].copy_from_slice(&tmp[..valid_lanes]);
        }
    }
}

/// **Fused Base-3 Multi-Plane Ternary Matrix** sharing a single `BF16` scale per `(stripe, 64-col chunk)`:
/// - `num_planes = 1`: `{-1..+1} * scale` (`2.25` physical bits/weight)
/// - `num_planes = 2`: `(3 * T0 + T1) * scale = {-4..+4} * scale` (**`4.25` physical bits/weight**)
/// - `num_planes = 3`: `(9 * T0 + 3 * T1 + T2) * scale = {-13..+13} * scale` (**`6.25` physical bits/weight**)
///
/// Loads `QuantizedActivationLut` **once** per column block across all planes simultaneously and
/// flushes `BF16` chunk scales **once** per chunk—cutting LUT loads & epilogue ops by `2x–3x`!
#[derive(Clone, Debug)]
pub struct FusedBase3LutMatrix {
    pub rows: usize,
    pub cols: usize,
    pub num_planes: usize,
    pub blocks_per_row: usize,
    pub num_stripes: usize,
    pub num_chunks: usize,
    pub planes: Vec<SimdPackedMatrix>,
    /// Shared `BF16` scales: `[num_stripes, num_chunks, SIMD_ROW_STRIPE]`.
    pub bf16_chunk_scales: Vec<u16>,
}

impl FusedBase3LutMatrix {
    pub fn physical_bytes(&self) -> usize {
        let stripe_bytes: usize = self.planes.iter().map(|p| p.stripe_data.len()).sum();
        stripe_bytes + self.bf16_chunk_scales.len() * std::mem::size_of::<u16>()
    }

    pub fn physical_bits_per_weight(&self) -> f64 {
        (self.physical_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }
}

/// Executes [`FusedBase3LutMatrix`] on `out` (`0..num_stripes`). Also supports executing ONLY
/// Plane 0 (`draft_plane0_only = true`) at `2.25b` bandwidth for zero-memory self-speculative drafting!
#[inline]
pub fn simd_lut_base3_gemv(
    matrix: &FusedBase3LutMatrix,
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    assert_eq!(out.len(), matrix.rows);
    // SAFETY: `0..matrix.num_stripes` writes exclusively within `0..matrix.rows`.
    unsafe {
        simd_lut_base3_stripe_range(
            matrix,
            qlut,
            0,
            matrix.num_stripes,
            out.as_mut_ptr(),
            false,
        );
    }
}

/// Executes stripe sub-range `[stripe_start, stripe_end)` into `out_base` (disjoint across threads).
///
/// # Safety
/// Caller must ensure `stripe_end <= matrix.num_stripes`, `out_base` points to a valid allocation
/// of at least `matrix.rows` `f32` elements, and concurrent calls use strictly disjoint stripe ranges.
#[inline]
pub unsafe fn simd_lut_base3_stripe_range(
    matrix: &FusedBase3LutMatrix,
    qlut: &QuantizedActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    plane0_only: bool,
) {
    debug_assert_eq!(matrix.blocks_per_row, qlut.num_blocks);
    debug_assert_eq!(matrix.num_chunks, qlut.chunk_scales.len());
    debug_assert!(stripe_end <= matrix.num_stripes);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and preconditions are upheld by caller.
        unsafe {
            simd_lut_base3_stripe_range_neon(
                matrix,
                qlut,
                stripe_start,
                stripe_end,
                out_base,
                plane0_only,
            );
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // SAFETY: Preconditions upheld by caller.
        unsafe {
            simd_lut_base3_stripe_range_scalar(
                matrix,
                qlut,
                stripe_start,
                stripe_end,
                out_base,
                plane0_only,
            );
        }
    }
}

/// Portable scalar reference for [`simd_lut_base3_stripe_range`].
///
/// # Safety
/// Same preconditions as [`simd_lut_base3_stripe_range`].
pub unsafe fn simd_lut_base3_stripe_range_scalar(
    matrix: &FusedBase3LutMatrix,
    qlut: &QuantizedActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    plane0_only: bool,
) {
    let bpr = matrix.blocks_per_row;
    let num_chunks = matrix.num_chunks;
    let eff_planes = if plane0_only { 1 } else { matrix.num_planes };
    let top_mult: i32 = match matrix.num_planes {
        1 => 1,
        2 => 3,
        3 => 9,
        _ => unreachable!(),
    };

    for s in stripe_start..stripe_end {
        let mut facc = [0.0f32; SIMD_ROW_STRIPE];
        let scale_stripe_base = s * num_chunks * SIMD_ROW_STRIPE;

        for (chunk_idx, &qscale) in qlut.chunk_scales.iter().enumerate() {
            if qscale == 0.0 {
                continue;
            }
            let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
            let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);
            let mut p_acc = [[0i32; SIMD_ROW_STRIPE]; 3];

            for (p, acc_slot) in p_acc.iter_mut().enumerate().take(eff_planes) {
                let stripe_slice = &matrix.planes[p].stripe_data
                    [s * bpr * SIMD_ROW_STRIPE..(s + 1) * bpr * SIMD_ROW_STRIPE];
                for b in b_start..b_end {
                    let codes = &stripe_slice[b * SIMD_ROW_STRIPE..(b + 1) * SIMD_ROW_STRIPE];
                    let t_lo = &qlut.lut_lo[b];
                    let t_hi = &qlut.lut_hi[b];
                    for lane in 0..SIMD_ROW_STRIPE {
                        let code = codes[lane];
                        acc_slot[lane] += (t_lo[(code & 0x0F) as usize] as i32)
                            + (t_hi[(code >> 4) as usize] as i32);
                    }
                }
            }

            let ws_slice = &matrix.bf16_chunk_scales[scale_stripe_base
                + chunk_idx * SIMD_ROW_STRIPE
                ..scale_stripe_base + (chunk_idx + 1) * SIMD_ROW_STRIPE];
            for lane in 0..SIMD_ROW_STRIPE {
                let combined: i32 = if plane0_only {
                    p_acc[0][lane] * top_mult
                } else {
                    match eff_planes {
                        1 => p_acc[0][lane],
                        2 => 3 * p_acc[0][lane] + p_acc[1][lane],
                        3 => 9 * p_acc[0][lane] + 3 * p_acc[1][lane] + p_acc[2][lane],
                        _ => unreachable!(),
                    }
                };
                let wscale = bf16_to_f32(ws_slice[lane]);
                facc[lane] += ((combined as f32) * qscale) * wscale;
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        for (lane, &val) in facc.iter().enumerate().take(valid_lanes) {
            // SAFETY: `row_start + lane < matrix.rows` within disjoint stripe range.
            unsafe {
                *out_base.add(row_start + lane) = val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_lut_base3_stripe_range_neon(
    matrix: &FusedBase3LutMatrix,
    qlut: &QuantizedActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    plane0_only: bool,
) {
    use std::arch::aarch64::*;

    let bpr = matrix.blocks_per_row;
    let num_chunks = matrix.num_chunks;
    let mask_0f = vdupq_n_u8(0x0F);
    let eff_planes = if plane0_only { 1 } else { matrix.num_planes };
    let plane0_scale_mult: f32 = if plane0_only {
        match matrix.num_planes {
            1 => 1.0,
            2 => 3.0,
            3 => 9.0,
            _ => 1.0,
        }
    } else {
        1.0
    };

    for s in stripe_start..stripe_end {
        let mut facc0 = vdupq_n_f32(0.0);
        let mut facc1 = vdupq_n_f32(0.0);
        let mut facc2 = vdupq_n_f32(0.0);
        let mut facc3 = vdupq_n_f32(0.0);

        let scale_stripe_offset = s * num_chunks * SIMD_ROW_STRIPE;
        // SAFETY: `s < matrix.num_stripes`.
        let p0_ptr = unsafe { matrix.planes[0].stripe_data.as_ptr().add(s * bpr * SIMD_ROW_STRIPE) };
        let p1_ptr = if eff_planes >= 2 {
            unsafe { matrix.planes[1].stripe_data.as_ptr().add(s * bpr * SIMD_ROW_STRIPE) }
        } else {
            std::ptr::null()
        };
        let p2_ptr = if eff_planes >= 3 {
            unsafe { matrix.planes[2].stripe_data.as_ptr().add(s * bpr * SIMD_ROW_STRIPE) }
        } else {
            std::ptr::null()
        };

        for (chunk_idx, &qscale) in qlut.chunk_scales.iter().enumerate() {
            if qscale == 0.0 {
                continue;
            }
            let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
            let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);

            let (f0, f1, f2, f3) = if eff_planes == 2 {
                // FUSED 2-PLANE FAST PATH: loads `tbl_lo/tbl_hi` ONCE per block and accumulates
                // `3 * iacc0 + iacc1` directly in `i16` before a SINGLE float conversion!
                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                let mut a1_lo = vdupq_n_s16(0);
                let mut a1_hi = vdupq_n_s16(0);

                let mut b = b_start;
                while b + 2 <= b_end {
                    // SAFETY: `b + 1 < bpr`.
                    unsafe {
                        let tbl_lo0 = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi0 = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let tbl_lo1 = vld1q_s8(qlut.lut_lo.as_ptr().add(b + 1).cast::<i8>());
                        let tbl_hi1 = vld1q_s8(qlut.lut_hi.as_ptr().add(b + 1).cast::<i8>());

                        let c0_a = vld1q_u8(p0_ptr.add(b * SIMD_ROW_STRIPE));
                        let c0_b = vld1q_u8(p0_ptr.add((b + 1) * SIMD_ROW_STRIPE));
                        let c1_a = vld1q_u8(p1_ptr.add(b * SIMD_ROW_STRIPE));
                        let c1_b = vld1q_u8(p1_ptr.add((b + 1) * SIMD_ROW_STRIPE));

                        let s0_lo_a = vqtbl1q_s8(tbl_lo0, vandq_u8(c0_a, mask_0f));
                        let s0_hi_a = vqtbl1q_s8(tbl_hi0, vshrq_n_u8::<4>(c0_a));
                        let s0_lo_b = vqtbl1q_s8(tbl_lo1, vandq_u8(c0_b, mask_0f));
                        let s0_hi_b = vqtbl1q_s8(tbl_hi1, vshrq_n_u8::<4>(c0_b));

                        let s1_lo_a = vqtbl1q_s8(tbl_lo0, vandq_u8(c1_a, mask_0f));
                        let s1_hi_a = vqtbl1q_s8(tbl_hi0, vshrq_n_u8::<4>(c1_a));
                        let s1_lo_b = vqtbl1q_s8(tbl_lo1, vandq_u8(c1_b, mask_0f));
                        let s1_hi_b = vqtbl1q_s8(tbl_hi1, vshrq_n_u8::<4>(c1_b));

                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo_a), vget_low_s8(s0_hi_a)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo_a, s0_hi_a));
                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo_b), vget_low_s8(s0_hi_b)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo_b, s0_hi_b));

                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(s1_lo_a), vget_low_s8(s1_hi_a)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(s1_lo_a, s1_hi_a));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(s1_lo_b), vget_low_s8(s1_hi_b)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(s1_lo_b, s1_hi_b));
                    }
                    b += 2;
                }
                while b < b_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let c0 = vld1q_u8(p0_ptr.add(b * SIMD_ROW_STRIPE));
                        let c1 = vld1q_u8(p1_ptr.add(b * SIMD_ROW_STRIPE));
                        let s0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c0, mask_0f));
                        let s0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c0));
                        let s1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c1, mask_0f));
                        let s1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c1));
                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo), vget_low_s8(s0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo, s0_hi));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(s1_lo), vget_low_s8(s1_hi)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(s1_lo, s1_hi));
                    }
                    b += 1;
                }

                // `3 * a0 + a1` in `i16` (`max |3 * 4064 + 4064| = 16,256 <= 32,767`)!
                let tot_lo = vaddq_s16(a1_lo, vaddq_s16(a0_lo, vshlq_n_s16::<1>(a0_lo)));
                let tot_hi = vaddq_s16(a1_hi, vaddq_s16(a0_hi, vshlq_n_s16::<1>(a0_hi)));
                (
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(tot_lo))),
                    vcvtq_f32_s32(vmovl_high_s16(tot_lo)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(tot_hi))),
                    vcvtq_f32_s32(vmovl_high_s16(tot_hi)),
                )
            } else if eff_planes == 1 {
                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                for b in b_start..b_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let c0 = vld1q_u8(p0_ptr.add(b * SIMD_ROW_STRIPE));
                        let s0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c0, mask_0f));
                        let s0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c0));
                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo), vget_low_s8(s0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo, s0_hi));
                    }
                }
                (
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0_lo))),
                    vcvtq_f32_s32(vmovl_high_s16(a0_lo)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(a0_hi))),
                    vcvtq_f32_s32(vmovl_high_s16(a0_hi)),
                )
            } else {
                // FUSED 3-PLANE FAST PATH (`9 * a0 + 3 * a1 + a2` widened to `i32` before scale!)
                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                let mut a1_lo = vdupq_n_s16(0);
                let mut a1_hi = vdupq_n_s16(0);
                let mut a2_lo = vdupq_n_s16(0);
                let mut a2_hi = vdupq_n_s16(0);
                for b in b_start..b_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let c0 = vld1q_u8(p0_ptr.add(b * SIMD_ROW_STRIPE));
                        let c1 = vld1q_u8(p1_ptr.add(b * SIMD_ROW_STRIPE));
                        let c2 = vld1q_u8(p2_ptr.add(b * SIMD_ROW_STRIPE));
                        let s0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c0, mask_0f));
                        let s0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c0));
                        let s1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c1, mask_0f));
                        let s1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c1));
                        let s2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c2, mask_0f));
                        let s2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c2));
                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo), vget_low_s8(s0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo, s0_hi));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(s1_lo), vget_low_s8(s1_hi)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(s1_lo, s1_hi));
                        a2_lo = vaddq_s16(a2_lo, vaddl_s8(vget_low_s8(s2_lo), vget_low_s8(s2_hi)));
                        a2_hi = vaddq_s16(a2_hi, vaddl_high_s8(s2_lo, s2_hi));
                    }
                }
                let nine = vdupq_n_s32(9);
                let three = vdupq_n_s32(3);
                let comb = |v0: int32x4_t, v1: int32x4_t, v2: int32x4_t| -> float32x4_t {
                    vcvtq_f32_s32(vmlaq_s32(vmlaq_s32(v2, v1, three), v0, nine))
                };
                (
                    comb(
                        vmovl_s16(vget_low_s16(a0_lo)),
                        vmovl_s16(vget_low_s16(a1_lo)),
                        vmovl_s16(vget_low_s16(a2_lo)),
                    ),
                    comb(
                        vmovl_high_s16(a0_lo),
                        vmovl_high_s16(a1_lo),
                        vmovl_high_s16(a2_lo),
                    ),
                    comb(
                        vmovl_s16(vget_low_s16(a0_hi)),
                        vmovl_s16(vget_low_s16(a1_hi)),
                        vmovl_s16(vget_low_s16(a2_hi)),
                    ),
                    comb(
                        vmovl_high_s16(a0_hi),
                        vmovl_high_s16(a1_hi),
                        vmovl_high_s16(a2_hi),
                    ),
                )
            };

            let vq = vdupq_n_f32(qscale * plane0_scale_mult);
            // SAFETY: `scale_stripe_offset + (chunk_idx + 1) * 16 <= bf16_chunk_scales.len()`.
            unsafe {
                let ws_ptr = matrix
                    .bf16_chunk_scales
                    .as_ptr()
                    .add(scale_stripe_offset + chunk_idx * SIMD_ROW_STRIPE);
                let bf_lo = vld1q_u16(ws_ptr);
                let bf_hi = vld1q_u16(ws_ptr.add(8));
                let ws0 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf_lo))),
                    vq,
                );
                let ws1 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bf_lo))),
                    vq,
                );
                let ws2 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf_hi))),
                    vq,
                );
                let ws3 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bf_hi))),
                    vq,
                );

                facc0 = vfmaq_f32(facc0, f0, ws0);
                facc1 = vfmaq_f32(facc1, f1, ws1);
                facc2 = vfmaq_f32(facc2, f2, ws2);
                facc3 = vfmaq_f32(facc3, f3, ws3);
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        if valid_lanes == SIMD_ROW_STRIPE {
            // SAFETY: `row_start + 16 <= matrix.rows`.
            unsafe {
                let dst = out_base.add(row_start);
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
            }
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            unsafe {
                let dst = tmp.as_mut_ptr();
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
                for (lane, &val) in tmp.iter().enumerate().take(valid_lanes) {
                    *out_base.add(row_start + lane) = val;
                }
            }
        }
    }
}

/// Standard **`GGUF Q4_0` (`llama.cpp` `4.50` bits/weight, `G=32`)** reference quantized matrix
/// (`16` nibble-packed `INT4` weights `[-8, +7]` + `1` `BF16` scale per 32-column block) for
/// exact apples-to-apples Perplexity and SIMD speed benchmarking!
#[derive(Clone, Debug)]
pub struct GgufQ4_0Matrix {
    pub rows: usize,
    pub cols: usize,
    pub blocks_per_row: usize,
    /// Packed `u8` nibbles (`lo = (q0 + 8)`, `hi = (q1 + 8)`), `16` bytes per 32-weight block.
    pub qs: Vec<u8>,
    /// `BF16` per-block scale (`max_abs / -8.0` or `max_abs / 7.5`), `[rows * blocks_per_row]`.
    pub bf16_scales: Vec<u16>,
}

impl GgufQ4_0Matrix {
    pub fn from_f32(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols);
        assert!(cols.is_multiple_of(32));
        let bpr = cols / 32;
        let mut qs = vec![0u8; rows * bpr * 16];
        let mut bf16_scales = vec![0u16; rows * bpr];

        for r in 0..rows {
            for b in 0..bpr {
                let sub = &weights[r * cols + b * 32..r * cols + (b + 1) * 32];
                // Exact `ggml` `quantize_row_q4_0_ref`: `amax` signed extremum divided by `-8.0`
                let mut max_abs = 0.0f32;
                let mut amax = 0.0f32;
                for &v in sub {
                    let av = v.abs();
                    if av > max_abs {
                        max_abs = av;
                        amax = v;
                    }
                }
                let d_raw = amax / -8.0;
                let d_bf16 = f32_to_bf16(d_raw);
                let d = bf16_to_f32(d_bf16);
                bf16_scales[r * bpr + b] = d_bf16;

                let id = if d.abs() > f32::EPSILON { 1.0 / d } else { 0.0 };
                let dst = &mut qs[(r * bpr + b) * 16..(r * bpr + b + 1) * 16];
                for j in 0..16 {
                    let x0 = ((sub[j] * id + 8.5) as i32).clamp(0, 15) as u8;
                    let x1 = ((sub[j + 16] * id + 8.5) as i32).clamp(0, 15) as u8;
                    dst[j] = x0 | (x1 << 4);
                }
            }
        }
        Self {
            rows,
            cols,
            blocks_per_row: bpr,
            qs,
            bf16_scales,
        }
    }

    pub fn physical_bytes(&self) -> usize {
        self.qs.len() + self.bf16_scales.len() * std::mem::size_of::<u16>()
    }

    pub fn physical_bits_per_weight(&self) -> f64 {
        (self.physical_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Quantizes input `x` to `Q8_0` (`G=32` `INT8` + `f32` scale, exact `llama.cpp` `vec_dot_q4_0_q8_0`!)
    pub fn quantize_q8_0_activation(x: &[f32], q8_vals: &mut [i8], q8_scales: &mut [f32]) {
        assert!(x.len().is_multiple_of(32));
        let nb = x.len() / 32;
        assert_eq!(q8_vals.len(), x.len());
        assert_eq!(q8_scales.len(), nb);
        for b in 0..nb {
            let sub = &x[b * 32..(b + 1) * 32];
            let max_abs = sub.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= f32::EPSILON {
                q8_scales[b] = 0.0;
                q8_vals[b * 32..(b + 1) * 32].fill(0);
            } else {
                let d = max_abs / 127.0;
                let id = 127.0 / max_abs;
                q8_scales[b] = d;
                for (dst, &v) in q8_vals[b * 32..(b + 1) * 32].iter_mut().zip(sub.iter()) {
                    *dst = (v * id).round().clamp(-127.0, 127.0) as i8;
                }
            }
        }
    }

    /// Executes `Q4_0 x Q8_0` hardware NEON GEMV on row range `[row_start, row_end)`!
    ///
    /// # Safety
    /// `row_end <= self.rows` and `out_base` valid for `self.rows` elements on disjoint row ranges.
    #[inline]
    pub unsafe fn forward_row_range(
        &self,
        q8_vals: &[i8],
        q8_scales: &[f32],
        row_start: usize,
        row_end: usize,
        out_base: *mut f32,
    ) {
        let bpr = self.blocks_per_row;
        for r in row_start..row_end {
            let sum = dot_q4_0_q8_0_row(
                &self.qs[r * bpr * 16..(r + 1) * bpr * 16],
                &self.bf16_scales[r * bpr..(r + 1) * bpr],
                q8_vals,
                q8_scales,
            );
            // SAFETY: `r < self.rows` inside disjoint `[row_start, row_end)`.
            unsafe {
                *out_base.add(r) = sum;
            }
        }
    }
}

#[inline]
fn dot_q4_0_q8_0_row(qs: &[u8], w_scales: &[u16], q8_vals: &[i8], x_scales: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and slice lengths match `w_scales.len()`.
        unsafe { dot_q4_0_q8_0_neon(qs, w_scales, q8_vals, x_scales) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = 0.0f32;
        for (b, (&ws_bf16, &xs)) in w_scales.iter().zip(x_scales.iter()).enumerate() {
            let d = bf16_to_f32(ws_bf16) * xs;
            let qb = &qs[b * 16..(b + 1) * 16];
            let xb = &q8_vals[b * 32..(b + 1) * 32];
            let mut isum = 0i32;
            for j in 0..16 {
                let w0 = ((qb[j] & 0x0F) as i32) - 8;
                let w1 = ((qb[j] >> 4) as i32) - 8;
                isum += w0 * (xb[j] as i32) + w1 * (xb[j + 16] as i32);
            }
            acc += (isum as f32) * d;
        }
        acc
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_q4_0_q8_0_neon(
    qs: &[u8],
    w_scales: &[u16],
    q8_vals: &[i8],
    x_scales: &[f32],
) -> f32 {
    use std::arch::aarch64::*;
    let nb = w_scales.len();
    let mask_0f = vdupq_n_u8(0x0F);
    let minus_8 = vdupq_n_s8(-8);
    let mut sumf = 0.0f32;

    for b in 0..nb {
        let d = bf16_to_f32(w_scales[b]) * x_scales[b];
        if d == 0.0 {
            continue;
        }
        // SAFETY: `b < nb`, `qs` has `16 * nb` bytes and `q8_vals` has `32 * nb` bytes.
        unsafe {
            let qbyte = vld1q_u8(qs.as_ptr().add(b * 16));
            let w0 = vaddq_s8(vreinterpretq_s8_u8(vandq_u8(qbyte, mask_0f)), minus_8);
            let w1 = vaddq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(qbyte)), minus_8);

            let x0 = vld1q_s8(q8_vals.as_ptr().add(b * 32));
            let x1 = vld1q_s8(q8_vals.as_ptr().add(b * 32 + 16));

            let p0_lo = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
            let p0_hi = vmull_high_s8(w0, x0);
            let p1_lo = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
            let p1_hi = vmull_high_s8(w1, x1);

            let s16_0 = vaddq_s16(p0_lo, p1_lo);
            let s16_1 = vaddq_s16(p0_hi, p1_hi);
            let isum = vaddlvq_s16(s16_0) + vaddlvq_s16(s16_1);
            sumf += (isum as f32) * d;
        }
    }
    sumf
}

/// IEEE-754 round-to-nearest-even `f32` to `bfloat16` (`u16` bits).
#[inline]
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding_bias = 0x0000_7FFF + ((bits >> 16) & 1);
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

/// Lossless `bfloat16` (`u16` bits) to `f32` conversion.
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// Number of 8-column weight bytes inside a 64-column Hadamard/scale chunk (`8 * 8 = 64`).
pub const BIPOLAR_OCTETS_PER_CHUNK: usize = 8;

/// Zero-allocation 4-tuple Bipolar (`{-1, +1}^4`, `2^4 = 16` full NEON LUT entries!) activation table.
///
/// Utilizes **100% of every 16-byte `vqtbl1q_s8` register LUT** (`16 / 16` slots vs `9 / 16` in ternary!),
/// packing `8` weights per byte (`1.00` bit/weight/plane) while preserving exact integer linearity
/// across multi-bit power-of-two planes (`sum_{p=0..B-1} 2^p s_p = {-2^B+1, ..., -1, +1, ..., +2^B-1}`).
#[derive(Clone, Debug)]
pub struct BipolarActivationLut {
    pub cols: usize,
    pub num_quads: usize,
    pub num_chunks: usize,
    pub lut4: Vec<[i8; 16]>,
    pub chunk_scales: Vec<f32>,
    /// Build the 16-entry LUTs on `populate`.
    pub lut_enabled: bool,
    /// `int8` activations for [`Nibble4SdotMatrix`].
    pub int8: Option<Int8ChunkActivation>,
}

impl BipolarActivationLut {
    pub fn new_zeroed(cols: usize) -> Self {
        assert!(cols > 0 && cols.is_multiple_of(64));
        let num_quads = cols / 4;
        let num_chunks = cols / 64;
        Self {
            cols,
            num_quads,
            num_chunks,
            lut4: vec![[0i8; 16]; num_quads],
            chunk_scales: vec![0.0f32; num_chunks],
            lut_enabled: true,
            int8: None,
        }
    }

    pub fn build(x: &[f32]) -> Self {
        let mut lut = Self::new_zeroed(x.len());
        lut.populate(x);
        lut
    }

    /// Populates all 16-entry `{-1, +1}^4` tables in-place using **Conway-Sloane $D_4$ Checkerboard
    /// Lattice $2\times$-Resolution Quantization** (`inv_scale = 250.0 / max_quad_l1`).
    ///
    /// ### Number-Theoretic $D_4$ Parity Theorem
    /// For any $q \in \mathbb{Z}^4$ and any $s \in \{-1, +1\}^4$, $\sum_{k=0}^3 s_k q_k \equiv \sum_{k=0}^3 q_k \pmod 2$.
    /// Projecting $q$ onto the even-sum 4D checkerboard lattice $D_4 = \{q \in \mathbb{Z}^4 : \sum q_k \equiv 0 \pmod 2\}$
    /// makes **all 16 bipolar sums exact even integers in `[-252, +252]`**, so shifting right by 1 (`>> 1`)
    /// maps them into `[-126, +126]` (`i8`) with **zero remainder**, **zero carry error across bit-planes**,
    /// and **$2\times$ finer activation resolution (`>4x` lower MSE)**!
    #[inline]
    pub fn populate(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.cols);
        if let Some(a) = self.int8.as_mut() {
            a.populate(x);
        }
        if !self.lut_enabled {
            return;
        }
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on `aarch64` and `x.len() == self.cols`.
            unsafe { self.populate_neon_d4(x) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.populate_scalar_d4(x);
        }
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    fn populate_scalar_d4(&mut self, x: &[f32]) {
        for (c_idx, (x_chunk, scale_slot)) in x
            .chunks_exact(64)
            .zip(self.chunk_scales.iter_mut())
            .enumerate()
        {
            let mut max_quad_l1 = 0.0f32;
            for q in x_chunk.chunks_exact(4) {
                let l1 = q[0].abs() + q[1].abs() + q[2].abs() + q[3].abs();
                if l1 > max_quad_l1 {
                    max_quad_l1 = l1;
                }
            }
            let base_q = c_idx * 16;
            if max_quad_l1 <= f32::EPSILON {
                *scale_slot = 0.0;
                self.lut4[base_q..base_q + 16].fill([0i8; 16]);
                continue;
            }

            *scale_slot = max_quad_l1 / 125.0;
            let inv_scale = 250.0 / max_quad_l1;

            for (g, qv) in x_chunk.chunks_exact(4).enumerate() {
                let q = project_quad_d4(
                    [
                        qv[0] * inv_scale,
                        qv[1] * inv_scale,
                        qv[2] * inv_scale,
                        qv[3] * inv_scale,
                    ],
                    [
                        (qv[0] * inv_scale).round_ties_even() as i32,
                        (qv[1] * inv_scale).round_ties_even() as i32,
                        (qv[2] * inv_scale).round_ties_even() as i32,
                        (qv[3] * inv_scale).round_ties_even() as i32,
                    ],
                );
                let p01 = [-q[0] - q[1], q[0] - q[1], -q[0] + q[1], q[0] + q[1]];
                let p23 = [-q[2] - q[3], q[2] - q[3], -q[2] + q[3], q[2] + q[3]];
                let mut tbl = [0i8; 16];
                for m in 0..4 {
                    for l in 0..4 {
                        tbl[m * 4 + l] = ((p01[l] + p23[m]) >> 1) as i8;
                    }
                }
                self.lut4[base_q + g] = tbl;
            }
        }
    }

    /// Branch-free NEON populate, 4 quads per step; bit-exact with the scalar path.
    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn populate_neon_d4(&mut self, x: &[f32]) {
        use std::arch::aarch64::*;
        // `idx[j][e] = 16 * (e / 4) + 4 * (e % 4) + j`: entry `e` of quad `j` inside 4 packed vectors.
        const TIDX: [[u8; 16]; 4] = {
            let mut t = [[0u8; 16]; 4];
            let mut j = 0;
            while j < 4 {
                let mut e = 0;
                while e < 16 {
                    t[j][e] = (16 * (e / 4) + 4 * (e % 4) + j) as u8;
                    e += 1;
                }
                j += 1;
            }
            t
        };
        // SAFETY: constant tables of exactly 16 bytes.
        let tidx = unsafe {
            [
                vld1q_u8(TIDX[0].as_ptr()),
                vld1q_u8(TIDX[1].as_ptr()),
                vld1q_u8(TIDX[2].as_ptr()),
                vld1q_u8(TIDX[3].as_ptr()),
            ]
        };
        let one = vdupq_n_s32(1);
        let zero_f = vdupq_n_f32(0.0);

        for (c_idx, (x_chunk, scale_slot)) in x
            .chunks_exact(64)
            .zip(self.chunk_scales.iter_mut())
            .enumerate()
        {
            let x_ptr = x_chunk.as_ptr();
            // SAFETY: 4 groups x 16 floats = 64 floats in `x_chunk`.
            let groups = unsafe {
                [
                    vld4q_f32(x_ptr),
                    vld4q_f32(x_ptr.add(16)),
                    vld4q_f32(x_ptr.add(32)),
                    vld4q_f32(x_ptr.add(48)),
                ]
            };
            let mut vmax = zero_f;
            for v in &groups {
                // Same left-to-right summation order as the scalar reference.
                let l1 = vaddq_f32(
                    vaddq_f32(vaddq_f32(vabsq_f32(v.0), vabsq_f32(v.1)), vabsq_f32(v.2)),
                    vabsq_f32(v.3),
                );
                vmax = vmaxq_f32(vmax, l1);
            }
            let max_quad_l1 = vmaxvq_f32(vmax);
            let base_q = c_idx * 16;
            if max_quad_l1 <= f32::EPSILON {
                *scale_slot = 0.0;
                self.lut4[base_q..base_q + 16].fill([0i8; 16]);
                continue;
            }
            *scale_slot = max_quad_l1 / 125.0;
            let v_inv = vdupq_n_f32(250.0 / max_quad_l1);

            for (g, v) in groups.iter().enumerate() {
                let u = [
                    vmulq_f32(v.0, v_inv),
                    vmulq_f32(v.1, v_inv),
                    vmulq_f32(v.2, v_inv),
                    vmulq_f32(v.3, v_inv),
                ];
                let mut q = [
                    vcvtnq_s32_f32(u[0]),
                    vcvtnq_s32_f32(u[1]),
                    vcvtnq_s32_f32(u[2]),
                    vcvtnq_s32_f32(u[3]),
                ];
                let odd = vtstq_s32(vaddq_s32(vaddq_s32(q[0], q[1]), vaddq_s32(q[2], q[3])), one);
                // Odd-parity quads: move the coordinate with the largest rounding error (first max
                // wins) one step toward `u`, matching `project_quad_d4`'s tie rules exactly.
                let e = [
                    vsubq_f32(u[0], vcvtq_f32_s32(q[0])),
                    vsubq_f32(u[1], vcvtq_f32_s32(q[1])),
                    vsubq_f32(u[2], vcvtq_f32_s32(q[2])),
                    vsubq_f32(u[3], vcvtq_f32_s32(q[3])),
                ];
                let mut best = vabsq_f32(e[0]);
                let mut sel = [vdupq_n_u32(u32::MAX), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0)];
                for k in 1..4 {
                    let a = vabsq_f32(e[k]);
                    let gt = vcgtq_f32(a, best);
                    best = vbslq_f32(gt, a, best);
                    for prev in sel.iter_mut().take(k) {
                        *prev = vbicq_u32(*prev, gt);
                    }
                    sel[k] = gt;
                }
                for k in 0..4 {
                    let up = vorrq_u32(
                        vcgtq_f32(e[k], zero_f),
                        vandq_u32(vceqq_f32(e[k], zero_f), vcleq_s32(q[k], vdupq_n_s32(0))),
                    );
                    let step = vbslq_s32(up, one, vdupq_n_s32(-1));
                    let apply = vandq_u32(sel[k], odd);
                    q[k] = vaddq_s32(q[k], vandq_s32(step, vreinterpretq_s32_u32(apply)));
                }

                let a01 = vaddq_s32(q[0], q[1]);
                let b01 = vsubq_s32(q[0], q[1]);
                let a23 = vaddq_s32(q[2], q[3]);
                let b23 = vsubq_s32(q[2], q[3]);
                let p01 = [vnegq_s32(a01), b01, vnegq_s32(b01), a01];
                let p23 = [vnegq_s32(a23), b23, vnegq_s32(b23), a23];
                // `packed[m]` = entries `4m..4m+3` (`= (p01[l] + p23[m]) >> 1`) for the 4 quads.
                let mut packed = [vdupq_n_s8(0); 4];
                for (m, slot) in packed.iter_mut().enumerate() {
                    let e0 = vshrq_n_s32::<1>(vaddq_s32(p01[0], p23[m]));
                    let e1 = vshrq_n_s32::<1>(vaddq_s32(p01[1], p23[m]));
                    let e2 = vshrq_n_s32::<1>(vaddq_s32(p01[2], p23[m]));
                    let e3 = vshrq_n_s32::<1>(vaddq_s32(p01[3], p23[m]));
                    let h01 = vcombine_s16(vmovn_s32(e0), vmovn_s32(e1));
                    let h23 = vcombine_s16(vmovn_s32(e2), vmovn_s32(e3));
                    *slot = vcombine_s8(vmovn_s16(h01), vmovn_s16(h23));
                }
                let tbl = int8x16x4_t(packed[0], packed[1], packed[2], packed[3]);
                for (j, idx) in tidx.iter().enumerate() {
                    // SAFETY: `base_q + g * 4 + j < num_quads`.
                    unsafe {
                        vst1q_s8(
                            self.lut4.as_mut_ptr().add(base_q + g * 4 + j).cast::<i8>(),
                            vqtbl4q_s8(tbl, *idx),
                        );
                    }
                }
            }
        }
    }
}

#[cfg(any(not(target_arch = "aarch64"), test))]
#[inline]
fn project_quad_d4(u: [f32; 4], mut q: [i32; 4]) -> [i32; 4] {
    if ((q[0] + q[1] + q[2] + q[3]) & 1) != 0 {
        let mut best_k = 0usize;
        let mut best_abs_err = -1.0f32;
        for k in 0..4 {
            let err = (u[k] - (q[k] as f32)).abs();
            if err > best_abs_err {
                best_abs_err = err;
                best_k = k;
            }
        }
        let diff = u[best_k] - (q[best_k] as f32);
        if diff > 0.0 {
            q[best_k] += 1;
        } else if diff < 0.0 || q[best_k] > 0 {
            q[best_k] -= 1;
        } else {
            q[best_k] += 1;
        }
    }
    q
}

#[inline]
fn round_quad_to_i8(qv: &[f32], inv_scale: f32) -> [i8; 4] {
    debug_assert_eq!(qv.len(), 4);
    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;
        // SAFETY: NEON is baseline on `aarch64` and `qv.len() == 4`.
        unsafe {
            let v = vmulq_f32(vld1q_f32(qv.as_ptr()), vdupq_n_f32(inv_scale));
            let i32v = vcvtaq_s32_f32(v);
            let mut arr = [0i32; 4];
            vst1q_s32(arr.as_mut_ptr(), i32v);
            [arr[0] as i8, arr[1] as i8, arr[2] as i8, arr[3] as i8]
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        [
            (qv[0] * inv_scale).round() as i8,
            (qv[1] * inv_scale).round() as i8,
            (qv[2] * inv_scale).round() as i8,
            (qv[3] * inv_scale).round() as i8,
        ]
    }
}

/// 1-bit-per-weight 16-row SIMD stripe plane (`8` bipolar weights `{-1, +1}` per `u8` byte!).
#[derive(Clone, Debug)]
pub struct BipolarPackedPlane {
    pub rows: usize,
    pub cols: usize,
    pub octets_per_row: usize,
    pub num_stripes: usize,
    /// Layout: `[num_stripes, octets_per_row, SIMD_ROW_STRIPE]`.
    pub stripe_data: Vec<u8>,
}

impl BipolarPackedPlane {
    /// Packs bit-plane `p` (`bit = (u_idx >> p) & 1`) from unsigned quantized indices `u_grid` (`[rows, cols]`).
    pub fn from_unsigned_bitplane(u_grid: &[u8], rows: usize, cols: usize, bit_pos: usize) -> Self {
        assert_eq!(u_grid.len(), rows * cols);
        assert!(cols > 0 && cols.is_multiple_of(64));
        let octets_per_row = cols / 8;
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let mut stripe_data = vec![0u8; num_stripes * octets_per_row * SIMD_ROW_STRIPE];

        for r in 0..rows {
            let stripe = r / SIMD_ROW_STRIPE;
            let lane = r % SIMD_ROW_STRIPE;
            let s_base = stripe * octets_per_row * SIMD_ROW_STRIPE;
            let row_u = &u_grid[r * cols..(r + 1) * cols];
            for (o, oct) in row_u.chunks_exact(8).enumerate() {
                let mut byte = 0u8;
                for (k, &uv) in oct.iter().enumerate() {
                    byte |= ((uv >> bit_pos) & 1) << k;
                }
                stripe_data[s_base + o * SIMD_ROW_STRIPE + lane] = byte;
            }
        }
        Self {
            rows,
            cols,
            octets_per_row,
            num_stripes,
            stripe_data,
        }
    }
}

/// **Full-16-Entry Bipolar Radix-2 Zero-Multiply NEON `LUT-GEMV` Matrix**:
/// - `num_planes = 3`: `8` symmetric levels `{-7..+7}` at **`3.25` physical bits/weight**
/// - `num_planes = 4`: **`16` symmetric levels `{-15..+15}` at `4.25` physical bits/weight** (`68.1 MB`)
/// - `num_planes = 5`: **`32` symmetric levels `{-31..+31}` at `5.25` physical bits/weight** (`84.1 MB`)
/// - `num_planes = 6`: **`64` symmetric levels `{-63..+63}` at `6.25` physical bits/weight** (`100.2 MB`)
#[derive(Clone, Debug)]
pub struct FusedBipolarLutMatrix {
    pub rows: usize,
    pub cols: usize,
    pub num_planes: usize,
    pub octets_per_row: usize,
    pub num_stripes: usize,
    pub num_chunks: usize,
    /// Cache-line-interleaved bit-planes `[num_stripes, octets_per_row, num_planes, SIMD_ROW_STRIPE]`
    /// so all `num_planes` vectors of an 8-column octet sit contiguously in a single linear prefetch stream!
    pub interleaved_data: Vec<u8>,
    /// Shared `BF16` chunk scales `[num_stripes, num_chunks, SIMD_ROW_STRIPE]`.
    pub bf16_chunk_scales: Vec<u16>,
}

impl FusedBipolarLutMatrix {
    /// Packs unsigned quantized indices `u_grid` (`[rows, cols]`) directly into contiguous
    /// octet-interleaved 16-row SIMD stripes `[num_stripes, octets_per_row, num_planes, 16]`.
    pub fn pack_interleaved(u_grid: &[u8], rows: usize, cols: usize, num_planes: usize) -> Vec<u8> {
        assert_eq!(u_grid.len(), rows * cols);
        assert!(cols > 0 && cols.is_multiple_of(64));
        let opr = cols / 8;
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let mut data = vec![0u8; num_stripes * opr * num_planes * SIMD_ROW_STRIPE];

        for r in 0..rows {
            let stripe = r / SIMD_ROW_STRIPE;
            let lane = r % SIMD_ROW_STRIPE;
            let row_u = &u_grid[r * cols..(r + 1) * cols];
            for (o, oct) in row_u.chunks_exact(8).enumerate() {
                let base = (stripe * opr + o) * num_planes * SIMD_ROW_STRIPE + lane;
                for p in 0..num_planes {
                    let mut byte = 0u8;
                    for (k, &uv) in oct.iter().enumerate() {
                        byte |= ((uv >> p) & 1) << k;
                    }
                    data[base + p * SIMD_ROW_STRIPE] = byte;
                }
            }
        }
        data
    }

    pub fn physical_bytes(&self) -> usize {
        self.interleaved_data.len() + self.bf16_chunk_scales.len() * std::mem::size_of::<u16>()
    }

    pub fn physical_bits_per_weight(&self) -> f64 {
        (self.physical_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Unpacks row `r`'s plane codes `u` into `out`.
    pub fn unpack_row_codes(&self, r: usize, out: &mut [u8]) {
        assert!(r < self.rows);
        assert_eq!(out.len(), self.cols);
        let np = self.num_planes;
        let stripe = r / SIMD_ROW_STRIPE;
        let lane = r % SIMD_ROW_STRIPE;
        for (o, oct_out) in out.chunks_exact_mut(8).enumerate() {
            let base = (stripe * self.octets_per_row + o) * np * SIMD_ROW_STRIPE + lane;
            oct_out.fill(0);
            for p in 0..np {
                let byte = self.interleaved_data[base + p * SIMD_ROW_STRIPE];
                for (k, uk) in oct_out.iter_mut().enumerate() {
                    *uk |= ((byte >> k) & 1) << p;
                }
            }
        }
    }

    /// Dequantizes row `r` (Hadamard-rotated domain).
    pub fn dequantize_row_rotated(&self, r: usize, out: &mut [f32]) {
        let mut codes = vec![0u8; self.cols];
        self.unpack_row_codes(r, &mut codes);
        let max_odd = (1i32 << self.num_planes) - 1;
        let sc_base = (r / SIMD_ROW_STRIPE) * self.num_chunks * SIMD_ROW_STRIPE + r % SIMD_ROW_STRIPE;
        for (c, (dst, src)) in out.chunks_exact_mut(64).zip(codes.chunks_exact(64)).enumerate() {
            let delta = bf16_to_f32(self.bf16_chunk_scales[sc_base + c * SIMD_ROW_STRIPE]);
            for (d, &u) in dst.iter_mut().zip(src.iter()) {
                *d = ((2 * u as i32 - max_odd) as f32) * delta;
            }
        }
    }
}

/// Executes [`FusedBipolarLutMatrix`] across all stripes on a single thread.
#[inline]
pub fn simd_lut_bipolar_gemv(
    matrix: &FusedBipolarLutMatrix,
    blut: &BipolarActivationLut,
    out: &mut [f32],
) {
    assert_eq!(out.len(), matrix.rows);
    // SAFETY: `0..matrix.num_stripes` writes exclusively within `0..matrix.rows`.
    unsafe {
        simd_lut_bipolar_stripe_range(matrix, blut, 0, matrix.num_stripes, out.as_mut_ptr());
    }
}

/// Executes stripe sub-range `[stripe_start, stripe_end)` of [`FusedBipolarLutMatrix`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn simd_lut_bipolar_stripe_range(
    matrix: &FusedBipolarLutMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, blut.cols);
    debug_assert_eq!(matrix.num_chunks, blut.num_chunks);
    debug_assert!(stripe_end <= matrix.num_stripes);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and caller upholds bounds.
        unsafe {
            simd_lut_bipolar_stripe_range_neon(matrix, blut, stripe_start, stripe_end, out_base);
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        // SAFETY: Caller upholds bounds.
        unsafe {
            simd_lut_bipolar_stripe_range_scalar(matrix, blut, stripe_start, stripe_end, out_base);
        }
    }
}

/// Portable scalar reference for [`simd_lut_bipolar_stripe_range`].
///
/// # Safety
/// Same preconditions as [`simd_lut_bipolar_stripe_range`].
pub unsafe fn simd_lut_bipolar_stripe_range_scalar(
    matrix: &FusedBipolarLutMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    let opr = matrix.octets_per_row;
    let num_chunks = matrix.num_chunks;
    let np = matrix.num_planes;

    for s in stripe_start..stripe_end {
        let mut facc = [0.0f32; SIMD_ROW_STRIPE];
        let sc_base = s * num_chunks * SIMD_ROW_STRIPE;

        for (c_idx, &qscale) in blut.chunk_scales.iter().enumerate() {
            if qscale == 0.0 {
                continue;
            }
            let o_start = c_idx * BIPOLAR_OCTETS_PER_CHUNK;
            let o_end = o_start + BIPOLAR_OCTETS_PER_CHUNK;
            let mut tot_i32 = [0i32; SIMD_ROW_STRIPE];

            for o in o_start..o_end {
                let oct_base = (s * opr + o) * np * SIMD_ROW_STRIPE;
                let t_lo = &blut.lut4[2 * o];
                let t_hi = &blut.lut4[2 * o + 1];
                for p in 0..np {
                    let codes = &matrix.interleaved_data
                        [oct_base + p * SIMD_ROW_STRIPE..oct_base + (p + 1) * SIMD_ROW_STRIPE];
                    for lane in 0..SIMD_ROW_STRIPE {
                        let code = codes[lane];
                        let s_val = (t_lo[(code & 0x0F) as usize] as i32)
                            + (t_hi[(code >> 4) as usize] as i32);
                        tot_i32[lane] += s_val << p;
                    }
                }
            }

            let ws_slice = &matrix.bf16_chunk_scales
                [sc_base + c_idx * SIMD_ROW_STRIPE..sc_base + (c_idx + 1) * SIMD_ROW_STRIPE];
            for lane in 0..SIMD_ROW_STRIPE {
                let wscale = bf16_to_f32(ws_slice[lane]);
                facc[lane] += ((tot_i32[lane] as f32) * qscale) * wscale;
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        for (lane, &val) in facc.iter().enumerate().take(valid_lanes) {
            // SAFETY: `row_start + lane < matrix.rows` inside disjoint stripe range.
            unsafe {
                *out_base.add(row_start + lane) = val;
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_lut_bipolar_stripe_range_neon(
    matrix: &FusedBipolarLutMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;

    let opr = matrix.octets_per_row;
    let num_chunks = matrix.num_chunks;
    let np = matrix.num_planes;
    let mask_0f = vdupq_n_u8(0x0F);
    let data_base = matrix.interleaved_data.as_ptr();
    let stride_stripe = opr * np * SIMD_ROW_STRIPE;

    let mut s = stripe_start;

    // 2-STRIPE (`32`-ROW) UNROLLED FAST PATH FOR `np == 4`:
    // Shares every `tbl_lo` & `tbl_hi` load across 32 rows simultaneously while streaming
    // strictly sequential 64-byte contiguous cache-line blocks!
    if np == 4 {
        while s + 2 <= stripe_end {
            let mut fa0 = vdupq_n_f32(0.0);
            let mut fa1 = vdupq_n_f32(0.0);
            let mut fa2 = vdupq_n_f32(0.0);
            let mut fa3 = vdupq_n_f32(0.0);

            let mut fb0 = vdupq_n_f32(0.0);
            let mut fb1 = vdupq_n_f32(0.0);
            let mut fb2 = vdupq_n_f32(0.0);
            let mut fb3 = vdupq_n_f32(0.0);

            let sc_a = s * num_chunks * SIMD_ROW_STRIPE;
            let sc_b = (s + 1) * num_chunks * SIMD_ROW_STRIPE;
            // SAFETY: `s + 1 < matrix.num_stripes`.
            let ptr_sa = unsafe { data_base.add(s * stride_stripe) };
            let ptr_sb = unsafe { data_base.add((s + 1) * stride_stripe) };

            for (c_idx, &qscale) in blut.chunk_scales.iter().enumerate() {
                if qscale == 0.0 {
                    continue;
                }
                let o_start = c_idx * BIPOLAR_OCTETS_PER_CHUNK;
                let o_end = o_start + BIPOLAR_OCTETS_PER_CHUNK;

                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                let mut a1_lo = vdupq_n_s16(0);
                let mut a1_hi = vdupq_n_s16(0);
                let mut a2_lo = vdupq_n_s16(0);
                let mut a2_hi = vdupq_n_s16(0);
                let mut a3_lo = vdupq_n_s16(0);
                let mut a3_hi = vdupq_n_s16(0);

                let mut b0_lo = vdupq_n_s16(0);
                let mut b0_hi = vdupq_n_s16(0);
                let mut b1_lo = vdupq_n_s16(0);
                let mut b1_hi = vdupq_n_s16(0);
                let mut b2_lo = vdupq_n_s16(0);
                let mut b2_hi = vdupq_n_s16(0);
                let mut b3_lo = vdupq_n_s16(0);
                let mut b3_hi = vdupq_n_s16(0);

                for o in o_start..o_end {
                    // SAFETY: `2 * o + 1 < blut.num_quads` and `o < opr`.
                    unsafe {
                        let tbl_lo = vld1q_s8(blut.lut4.as_ptr().add(2 * o).cast::<i8>());
                        let tbl_hi = vld1q_s8(blut.lut4.as_ptr().add(2 * o + 1).cast::<i8>());
                        let off = o * 64;

                        let pa = ptr_sa.add(off);
                        let ca0 = vld1q_u8(pa);
                        let ca1 = vld1q_u8(pa.add(16));
                        let ca2 = vld1q_u8(pa.add(32));
                        let ca3 = vld1q_u8(pa.add(48));

                        let sa0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca0, mask_0f));
                        let sa0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca0));
                        let sa1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca1, mask_0f));
                        let sa1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca1));
                        let sa2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca2, mask_0f));
                        let sa2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca2));
                        let sa3_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca3, mask_0f));
                        let sa3_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca3));

                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(sa0_lo), vget_low_s8(sa0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(sa0_lo, sa0_hi));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(sa1_lo), vget_low_s8(sa1_hi)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(sa1_lo, sa1_hi));
                        a2_lo = vaddq_s16(a2_lo, vaddl_s8(vget_low_s8(sa2_lo), vget_low_s8(sa2_hi)));
                        a2_hi = vaddq_s16(a2_hi, vaddl_high_s8(sa2_lo, sa2_hi));
                        a3_lo = vaddq_s16(a3_lo, vaddl_s8(vget_low_s8(sa3_lo), vget_low_s8(sa3_hi)));
                        a3_hi = vaddq_s16(a3_hi, vaddl_high_s8(sa3_lo, sa3_hi));

                        let pb = ptr_sb.add(off);
                        let cb0 = vld1q_u8(pb);
                        let cb1 = vld1q_u8(pb.add(16));
                        let cb2 = vld1q_u8(pb.add(32));
                        let cb3 = vld1q_u8(pb.add(48));

                        let sb0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb0, mask_0f));
                        let sb0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb0));
                        let sb1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb1, mask_0f));
                        let sb1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb1));
                        let sb2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb2, mask_0f));
                        let sb2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb2));
                        let sb3_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb3, mask_0f));
                        let sb3_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb3));

                        b0_lo = vaddq_s16(b0_lo, vaddl_s8(vget_low_s8(sb0_lo), vget_low_s8(sb0_hi)));
                        b0_hi = vaddq_s16(b0_hi, vaddl_high_s8(sb0_lo, sb0_hi));
                        b1_lo = vaddq_s16(b1_lo, vaddl_s8(vget_low_s8(sb1_lo), vget_low_s8(sb1_hi)));
                        b1_hi = vaddq_s16(b1_hi, vaddl_high_s8(sb1_lo, sb1_hi));
                        b2_lo = vaddq_s16(b2_lo, vaddl_s8(vget_low_s8(sb2_lo), vget_low_s8(sb2_hi)));
                        b2_hi = vaddq_s16(b2_hi, vaddl_high_s8(sb2_lo, sb2_hi));
                        b3_lo = vaddq_s16(b3_lo, vaddl_s8(vget_low_s8(sb3_lo), vget_low_s8(sb3_hi)));
                        b3_hi = vaddq_s16(b3_hi, vaddl_high_s8(sb3_lo, sb3_hi));
                    }
                }

                let vq = vdupq_n_f32(qscale);
                let ta_lo = vaddq_s16(
                    vaddq_s16(a0_lo, vshlq_n_s16::<1>(a1_lo)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_lo), vshlq_n_s16::<3>(a3_lo)),
                );
                let ta_hi = vaddq_s16(
                    vaddq_s16(a0_hi, vshlq_n_s16::<1>(a1_hi)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_hi), vshlq_n_s16::<3>(a3_hi)),
                );
                let tb_lo = vaddq_s16(
                    vaddq_s16(b0_lo, vshlq_n_s16::<1>(b1_lo)),
                    vaddq_s16(vshlq_n_s16::<2>(b2_lo), vshlq_n_s16::<3>(b3_lo)),
                );
                let tb_hi = vaddq_s16(
                    vaddq_s16(b0_hi, vshlq_n_s16::<1>(b1_hi)),
                    vaddq_s16(vshlq_n_s16::<2>(b2_hi), vshlq_n_s16::<3>(b3_hi)),
                );

                unsafe {
                    let wsa = matrix.bf16_chunk_scales.as_ptr().add(sc_a + c_idx * 16);
                    let bfa_lo = vld1q_u16(wsa);
                    let bfa_hi = vld1q_u16(wsa.add(8));
                    fa0 = vfmaq_f32(
                        fa0,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ta_lo))), vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfa_lo))),
                    );
                    fa1 = vfmaq_f32(
                        fa1,
                        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(ta_lo)), vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfa_lo)),
                    );
                    fa2 = vfmaq_f32(
                        fa2,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ta_hi))), vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfa_hi))),
                    );
                    fa3 = vfmaq_f32(
                        fa3,
                        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(ta_hi)), vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfa_hi)),
                    );

                    let wsb = matrix.bf16_chunk_scales.as_ptr().add(sc_b + c_idx * 16);
                    let bfb_lo = vld1q_u16(wsb);
                    let bfb_hi = vld1q_u16(wsb.add(8));
                    fb0 = vfmaq_f32(
                        fb0,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(tb_lo))), vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfb_lo))),
                    );
                    fb1 = vfmaq_f32(
                        fb1,
                        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(tb_lo)), vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfb_lo)),
                    );
                    fb2 = vfmaq_f32(
                        fb2,
                        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(tb_hi))), vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfb_hi))),
                    );
                    fb3 = vfmaq_f32(
                        fb3,
                        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(tb_hi)), vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfb_hi)),
                    );
                }
            }

            // Store Stripe `s` and Stripe `s + 1`
            for (st_idx, (r0, r1, r2, r3)) in [(s, (fa0, fa1, fa2, fa3)), (s + 1, (fb0, fb1, fb2, fb3))] {
                let row_start = st_idx * SIMD_ROW_STRIPE;
                let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
                unsafe {
                    let dst = out_base.add(row_start);
                    if valid == SIMD_ROW_STRIPE {
                        vst1q_f32(dst, r0);
                        vst1q_f32(dst.add(4), r1);
                        vst1q_f32(dst.add(8), r2);
                        vst1q_f32(dst.add(12), r3);
                    } else {
                        let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
                        vst1q_f32(tmp.as_mut_ptr(), r0);
                        vst1q_f32(tmp.as_mut_ptr().add(4), r1);
                        vst1q_f32(tmp.as_mut_ptr().add(8), r2);
                        vst1q_f32(tmp.as_mut_ptr().add(12), r3);
                        for (lane, &val) in tmp.iter().enumerate().take(valid) {
                            *dst.add(lane) = val;
                        }
                    }
                }
            }
            s += 2;
        }
    } else if np == 5 {
        while s + 2 <= stripe_end {
            let mut fa0 = vdupq_n_f32(0.0);
            let mut fa1 = vdupq_n_f32(0.0);
            let mut fa2 = vdupq_n_f32(0.0);
            let mut fa3 = vdupq_n_f32(0.0);

            let mut fb0 = vdupq_n_f32(0.0);
            let mut fb1 = vdupq_n_f32(0.0);
            let mut fb2 = vdupq_n_f32(0.0);
            let mut fb3 = vdupq_n_f32(0.0);

            let sc_a = s * num_chunks * SIMD_ROW_STRIPE;
            let sc_b = (s + 1) * num_chunks * SIMD_ROW_STRIPE;
            let ptr_sa = unsafe { data_base.add(s * stride_stripe) };
            let ptr_sb = unsafe { data_base.add((s + 1) * stride_stripe) };

            for (c_idx, &qscale) in blut.chunk_scales.iter().enumerate() {
                if qscale == 0.0 {
                    continue;
                }
                let o_start = c_idx * BIPOLAR_OCTETS_PER_CHUNK;
                let o_end = o_start + BIPOLAR_OCTETS_PER_CHUNK;

                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                let mut a1_lo = vdupq_n_s16(0);
                let mut a1_hi = vdupq_n_s16(0);
                let mut a2_lo = vdupq_n_s16(0);
                let mut a2_hi = vdupq_n_s16(0);
                let mut a3_lo = vdupq_n_s16(0);
                let mut a3_hi = vdupq_n_s16(0);
                let mut a4_lo = vdupq_n_s16(0);
                let mut a4_hi = vdupq_n_s16(0);

                let mut b0_lo = vdupq_n_s16(0);
                let mut b0_hi = vdupq_n_s16(0);
                let mut b1_lo = vdupq_n_s16(0);
                let mut b1_hi = vdupq_n_s16(0);
                let mut b2_lo = vdupq_n_s16(0);
                let mut b2_hi = vdupq_n_s16(0);
                let mut b3_lo = vdupq_n_s16(0);
                let mut b3_hi = vdupq_n_s16(0);
                let mut b4_lo = vdupq_n_s16(0);
                let mut b4_hi = vdupq_n_s16(0);

                for o in o_start..o_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(blut.lut4.as_ptr().add(2 * o).cast::<i8>());
                        let tbl_hi = vld1q_s8(blut.lut4.as_ptr().add(2 * o + 1).cast::<i8>());
                        let off = o * 80;

                        let pa = ptr_sa.add(off);
                        let ca0 = vld1q_u8(pa);
                        let ca1 = vld1q_u8(pa.add(16));
                        let ca2 = vld1q_u8(pa.add(32));
                        let ca3 = vld1q_u8(pa.add(48));
                        let ca4 = vld1q_u8(pa.add(64));

                        let sa0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca0, mask_0f));
                        let sa0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca0));
                        let sa1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca1, mask_0f));
                        let sa1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca1));
                        let sa2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca2, mask_0f));
                        let sa2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca2));
                        let sa3_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca3, mask_0f));
                        let sa3_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca3));
                        let sa4_lo = vqtbl1q_s8(tbl_lo, vandq_u8(ca4, mask_0f));
                        let sa4_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(ca4));

                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(sa0_lo), vget_low_s8(sa0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(sa0_lo, sa0_hi));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(sa1_lo), vget_low_s8(sa1_hi)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(sa1_lo, sa1_hi));
                        a2_lo = vaddq_s16(a2_lo, vaddl_s8(vget_low_s8(sa2_lo), vget_low_s8(sa2_hi)));
                        a2_hi = vaddq_s16(a2_hi, vaddl_high_s8(sa2_lo, sa2_hi));
                        a3_lo = vaddq_s16(a3_lo, vaddl_s8(vget_low_s8(sa3_lo), vget_low_s8(sa3_hi)));
                        a3_hi = vaddq_s16(a3_hi, vaddl_high_s8(sa3_lo, sa3_hi));
                        a4_lo = vaddq_s16(a4_lo, vaddl_s8(vget_low_s8(sa4_lo), vget_low_s8(sa4_hi)));
                        a4_hi = vaddq_s16(a4_hi, vaddl_high_s8(sa4_lo, sa4_hi));

                        let pb = ptr_sb.add(off);
                        let cb0 = vld1q_u8(pb);
                        let cb1 = vld1q_u8(pb.add(16));
                        let cb2 = vld1q_u8(pb.add(32));
                        let cb3 = vld1q_u8(pb.add(48));
                        let cb4 = vld1q_u8(pb.add(64));

                        let sb0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb0, mask_0f));
                        let sb0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb0));
                        let sb1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb1, mask_0f));
                        let sb1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb1));
                        let sb2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb2, mask_0f));
                        let sb2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb2));
                        let sb3_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb3, mask_0f));
                        let sb3_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb3));
                        let sb4_lo = vqtbl1q_s8(tbl_lo, vandq_u8(cb4, mask_0f));
                        let sb4_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(cb4));

                        b0_lo = vaddq_s16(b0_lo, vaddl_s8(vget_low_s8(sb0_lo), vget_low_s8(sb0_hi)));
                        b0_hi = vaddq_s16(b0_hi, vaddl_high_s8(sb0_lo, sb0_hi));
                        b1_lo = vaddq_s16(b1_lo, vaddl_s8(vget_low_s8(sb1_lo), vget_low_s8(sb1_hi)));
                        b1_hi = vaddq_s16(b1_hi, vaddl_high_s8(sb1_lo, sb1_hi));
                        b2_lo = vaddq_s16(b2_lo, vaddl_s8(vget_low_s8(sb2_lo), vget_low_s8(sb2_hi)));
                        b2_hi = vaddq_s16(b2_hi, vaddl_high_s8(sb2_lo, sb2_hi));
                        b3_lo = vaddq_s16(b3_lo, vaddl_s8(vget_low_s8(sb3_lo), vget_low_s8(sb3_hi)));
                        b3_hi = vaddq_s16(b3_hi, vaddl_high_s8(sb3_lo, sb3_hi));
                        b4_lo = vaddq_s16(b4_lo, vaddl_s8(vget_low_s8(sb4_lo), vget_low_s8(sb4_hi)));
                        b4_hi = vaddq_s16(b4_hi, vaddl_high_s8(sb4_lo, sb4_hi));
                    }
                }

                let vq = vdupq_n_f32(qscale);
                let pa03_lo = vaddq_s16(
                    vaddq_s16(a0_lo, vshlq_n_s16::<1>(a1_lo)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_lo), vshlq_n_s16::<3>(a3_lo)),
                );
                let pa03_hi = vaddq_s16(
                    vaddq_s16(a0_hi, vshlq_n_s16::<1>(a1_hi)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_hi), vshlq_n_s16::<3>(a3_hi)),
                );
                let pb03_lo = vaddq_s16(
                    vaddq_s16(b0_lo, vshlq_n_s16::<1>(b1_lo)),
                    vaddq_s16(vshlq_n_s16::<2>(b2_lo), vshlq_n_s16::<3>(b3_lo)),
                );
                let pb03_hi = vaddq_s16(
                    vaddq_s16(b0_hi, vshlq_n_s16::<1>(b1_hi)),
                    vaddq_s16(vshlq_n_s16::<2>(b2_hi), vshlq_n_s16::<3>(b3_hi)),
                );

                let ta0 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_s16(vget_low_s16(pa03_lo)),
                    vshlq_n_s32::<4>(vmovl_s16(vget_low_s16(a4_lo))),
                ));
                let ta1 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_high_s16(pa03_lo),
                    vshlq_n_s32::<4>(vmovl_high_s16(a4_lo)),
                ));
                let ta2 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_s16(vget_low_s16(pa03_hi)),
                    vshlq_n_s32::<4>(vmovl_s16(vget_low_s16(a4_hi))),
                ));
                let ta3 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_high_s16(pa03_hi),
                    vshlq_n_s32::<4>(vmovl_high_s16(a4_hi)),
                ));

                let tb0 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_s16(vget_low_s16(pb03_lo)),
                    vshlq_n_s32::<4>(vmovl_s16(vget_low_s16(b4_lo))),
                ));
                let tb1 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_high_s16(pb03_lo),
                    vshlq_n_s32::<4>(vmovl_high_s16(b4_lo)),
                ));
                let tb2 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_s16(vget_low_s16(pb03_hi)),
                    vshlq_n_s32::<4>(vmovl_s16(vget_low_s16(b4_hi))),
                ));
                let tb3 = vcvtq_f32_s32(vaddq_s32(
                    vmovl_high_s16(pb03_hi),
                    vshlq_n_s32::<4>(vmovl_high_s16(b4_hi)),
                ));

                unsafe {
                    let wsa = matrix.bf16_chunk_scales.as_ptr().add(sc_a + c_idx * 16);
                    let bfa_lo = vld1q_u16(wsa);
                    let bfa_hi = vld1q_u16(wsa.add(8));
                    fa0 = vfmaq_f32(
                        fa0,
                        vmulq_f32(ta0, vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfa_lo))),
                    );
                    fa1 = vfmaq_f32(
                        fa1,
                        vmulq_f32(ta1, vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfa_lo)),
                    );
                    fa2 = vfmaq_f32(
                        fa2,
                        vmulq_f32(ta2, vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfa_hi))),
                    );
                    fa3 = vfmaq_f32(
                        fa3,
                        vmulq_f32(ta3, vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfa_hi)),
                    );

                    let wsb = matrix.bf16_chunk_scales.as_ptr().add(sc_b + c_idx * 16);
                    let bfb_lo = vld1q_u16(wsb);
                    let bfb_hi = vld1q_u16(wsb.add(8));
                    fb0 = vfmaq_f32(
                        fb0,
                        vmulq_f32(tb0, vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfb_lo))),
                    );
                    fb1 = vfmaq_f32(
                        fb1,
                        vmulq_f32(tb1, vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfb_lo)),
                    );
                    fb2 = vfmaq_f32(
                        fb2,
                        vmulq_f32(tb2, vq),
                        vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfb_hi))),
                    );
                    fb3 = vfmaq_f32(
                        fb3,
                        vmulq_f32(tb3, vq),
                        vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfb_hi)),
                    );
                }
            }

            for (st_idx, (r0, r1, r2, r3)) in [(s, (fa0, fa1, fa2, fa3)), (s + 1, (fb0, fb1, fb2, fb3))] {
                let row_start = st_idx * SIMD_ROW_STRIPE;
                let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
                unsafe {
                    let dst = out_base.add(row_start);
                    if valid == SIMD_ROW_STRIPE {
                        vst1q_f32(dst, r0);
                        vst1q_f32(dst.add(4), r1);
                        vst1q_f32(dst.add(8), r2);
                        vst1q_f32(dst.add(12), r3);
                    } else {
                        let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
                        vst1q_f32(tmp.as_mut_ptr(), r0);
                        vst1q_f32(tmp.as_mut_ptr().add(4), r1);
                        vst1q_f32(tmp.as_mut_ptr().add(8), r2);
                        vst1q_f32(tmp.as_mut_ptr().add(12), r3);
                        for (lane, &val) in tmp.iter().enumerate().take(valid) {
                            *dst.add(lane) = val;
                        }
                    }
                }
            }
            s += 2;
        }
    }

    while s < stripe_end {
        let mut facc0 = vdupq_n_f32(0.0);
        let mut facc1 = vdupq_n_f32(0.0);
        let mut facc2 = vdupq_n_f32(0.0);
        let mut facc3 = vdupq_n_f32(0.0);

        let sc_base = s * num_chunks * SIMD_ROW_STRIPE;
        // SAFETY: `s < matrix.num_stripes`.
        let ptr_s = unsafe { data_base.add(s * stride_stripe) };

        for (c_idx, &qscale) in blut.chunk_scales.iter().enumerate() {
            if qscale == 0.0 {
                continue;
            }
            let o_start = c_idx * BIPOLAR_OCTETS_PER_CHUNK;
            let o_end = o_start + BIPOLAR_OCTETS_PER_CHUNK;

            let (f0, f1, f2, f3) = if np == 4 {
                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                let mut a1_lo = vdupq_n_s16(0);
                let mut a1_hi = vdupq_n_s16(0);
                let mut a2_lo = vdupq_n_s16(0);
                let mut a2_hi = vdupq_n_s16(0);
                let mut a3_lo = vdupq_n_s16(0);
                let mut a3_hi = vdupq_n_s16(0);

                for o in o_start..o_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(blut.lut4.as_ptr().add(2 * o).cast::<i8>());
                        let tbl_hi = vld1q_s8(blut.lut4.as_ptr().add(2 * o + 1).cast::<i8>());
                        let pa = ptr_s.add(o * 64);

                        let c0 = vld1q_u8(pa);
                        let c1 = vld1q_u8(pa.add(16));
                        let c2 = vld1q_u8(pa.add(32));
                        let c3 = vld1q_u8(pa.add(48));

                        let s0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c0, mask_0f));
                        let s0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c0));
                        let s1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c1, mask_0f));
                        let s1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c1));
                        let s2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c2, mask_0f));
                        let s2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c2));
                        let s3_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c3, mask_0f));
                        let s3_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c3));

                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo), vget_low_s8(s0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo, s0_hi));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(s1_lo), vget_low_s8(s1_hi)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(s1_lo, s1_hi));
                        a2_lo = vaddq_s16(a2_lo, vaddl_s8(vget_low_s8(s2_lo), vget_low_s8(s2_hi)));
                        a2_hi = vaddq_s16(a2_hi, vaddl_high_s8(s2_lo, s2_hi));
                        a3_lo = vaddq_s16(a3_lo, vaddl_s8(vget_low_s8(s3_lo), vget_low_s8(s3_hi)));
                        a3_hi = vaddq_s16(a3_hi, vaddl_high_s8(s3_lo, s3_hi));
                    }
                }

                let tot_lo = vaddq_s16(
                    vaddq_s16(a0_lo, vshlq_n_s16::<1>(a1_lo)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_lo), vshlq_n_s16::<3>(a3_lo)),
                );
                let tot_hi = vaddq_s16(
                    vaddq_s16(a0_hi, vshlq_n_s16::<1>(a1_hi)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_hi), vshlq_n_s16::<3>(a3_hi)),
                );
                (
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(tot_lo))),
                    vcvtq_f32_s32(vmovl_high_s16(tot_lo)),
                    vcvtq_f32_s32(vmovl_s16(vget_low_s16(tot_hi))),
                    vcvtq_f32_s32(vmovl_high_s16(tot_hi)),
                )
            } else if np == 5 {
                let mut a0_lo = vdupq_n_s16(0);
                let mut a0_hi = vdupq_n_s16(0);
                let mut a1_lo = vdupq_n_s16(0);
                let mut a1_hi = vdupq_n_s16(0);
                let mut a2_lo = vdupq_n_s16(0);
                let mut a2_hi = vdupq_n_s16(0);
                let mut a3_lo = vdupq_n_s16(0);
                let mut a3_hi = vdupq_n_s16(0);
                let mut a4_lo = vdupq_n_s16(0);
                let mut a4_hi = vdupq_n_s16(0);

                for o in o_start..o_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(blut.lut4.as_ptr().add(2 * o).cast::<i8>());
                        let tbl_hi = vld1q_s8(blut.lut4.as_ptr().add(2 * o + 1).cast::<i8>());
                        let pa = ptr_s.add(o * 80);

                        let c0 = vld1q_u8(pa);
                        let c1 = vld1q_u8(pa.add(16));
                        let c2 = vld1q_u8(pa.add(32));
                        let c3 = vld1q_u8(pa.add(48));
                        let c4 = vld1q_u8(pa.add(64));

                        let s0_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c0, mask_0f));
                        let s0_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c0));
                        let s1_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c1, mask_0f));
                        let s1_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c1));
                        let s2_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c2, mask_0f));
                        let s2_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c2));
                        let s3_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c3, mask_0f));
                        let s3_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c3));
                        let s4_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c4, mask_0f));
                        let s4_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c4));

                        a0_lo = vaddq_s16(a0_lo, vaddl_s8(vget_low_s8(s0_lo), vget_low_s8(s0_hi)));
                        a0_hi = vaddq_s16(a0_hi, vaddl_high_s8(s0_lo, s0_hi));
                        a1_lo = vaddq_s16(a1_lo, vaddl_s8(vget_low_s8(s1_lo), vget_low_s8(s1_hi)));
                        a1_hi = vaddq_s16(a1_hi, vaddl_high_s8(s1_lo, s1_hi));
                        a2_lo = vaddq_s16(a2_lo, vaddl_s8(vget_low_s8(s2_lo), vget_low_s8(s2_hi)));
                        a2_hi = vaddq_s16(a2_hi, vaddl_high_s8(s2_lo, s2_hi));
                        a3_lo = vaddq_s16(a3_lo, vaddl_s8(vget_low_s8(s3_lo), vget_low_s8(s3_hi)));
                        a3_hi = vaddq_s16(a3_hi, vaddl_high_s8(s3_lo, s3_hi));
                        a4_lo = vaddq_s16(a4_lo, vaddl_s8(vget_low_s8(s4_lo), vget_low_s8(s4_hi)));
                        a4_hi = vaddq_s16(a4_hi, vaddl_high_s8(s4_lo, s4_hi));
                    }
                }

                let p03_lo = vaddq_s16(
                    vaddq_s16(a0_lo, vshlq_n_s16::<1>(a1_lo)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_lo), vshlq_n_s16::<3>(a3_lo)),
                );
                let p03_hi = vaddq_s16(
                    vaddq_s16(a0_hi, vshlq_n_s16::<1>(a1_hi)),
                    vaddq_s16(vshlq_n_s16::<2>(a2_hi), vshlq_n_s16::<3>(a3_hi)),
                );
                let t0 = vaddq_s32(
                    vmovl_s16(vget_low_s16(p03_lo)),
                    vshlq_n_s32::<4>(vmovl_s16(vget_low_s16(a4_lo))),
                );
                let t1 = vaddq_s32(
                    vmovl_high_s16(p03_lo),
                    vshlq_n_s32::<4>(vmovl_high_s16(a4_lo)),
                );
                let t2 = vaddq_s32(
                    vmovl_s16(vget_low_s16(p03_hi)),
                    vshlq_n_s32::<4>(vmovl_s16(vget_low_s16(a4_hi))),
                );
                let t3 = vaddq_s32(
                    vmovl_high_s16(p03_hi),
                    vshlq_n_s32::<4>(vmovl_high_s16(a4_hi)),
                );
                (
                    vcvtq_f32_s32(t0),
                    vcvtq_f32_s32(t1),
                    vcvtq_f32_s32(t2),
                    vcvtq_f32_s32(t3),
                )
            } else {
                let mut acc_lo = [vdupq_n_s16(0); 6];
                let mut acc_hi = [vdupq_n_s16(0); 6];
                for o in o_start..o_end {
                    unsafe {
                        let tbl_lo = vld1q_s8(blut.lut4.as_ptr().add(2 * o).cast::<i8>());
                        let tbl_hi = vld1q_s8(blut.lut4.as_ptr().add(2 * o + 1).cast::<i8>());
                        let pa = ptr_s.add(o * np * SIMD_ROW_STRIPE);
                        for p in 0..np {
                            let c = vld1q_u8(pa.add(p * SIMD_ROW_STRIPE));
                            let s_lo = vqtbl1q_s8(tbl_lo, vandq_u8(c, mask_0f));
                            let s_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(c));
                            acc_lo[p] = vaddq_s16(
                                acc_lo[p],
                                vaddl_s8(vget_low_s8(s_lo), vget_low_s8(s_hi)),
                            );
                            acc_hi[p] = vaddq_s16(acc_hi[p], vaddl_high_s8(s_lo, s_hi));
                        }
                    }
                }

                let mut t0 = vdupq_n_s32(0);
                let mut t1 = vdupq_n_s32(0);
                let mut t2 = vdupq_n_s32(0);
                let mut t3 = vdupq_n_s32(0);
                for p in 0..np {
                    let sh = vdupq_n_s32(p as i32);
                    t0 = vaddq_s32(t0, vshlq_s32(vmovl_s16(vget_low_s16(acc_lo[p])), sh));
                    t1 = vaddq_s32(t1, vshlq_s32(vmovl_high_s16(acc_lo[p]), sh));
                    t2 = vaddq_s32(t2, vshlq_s32(vmovl_s16(vget_low_s16(acc_hi[p])), sh));
                    t3 = vaddq_s32(t3, vshlq_s32(vmovl_high_s16(acc_hi[p]), sh));
                }
                (
                    vcvtq_f32_s32(t0),
                    vcvtq_f32_s32(t1),
                    vcvtq_f32_s32(t2),
                    vcvtq_f32_s32(t3),
                )
            };

            let vq = vdupq_n_f32(qscale);
            // SAFETY: `sc_base + (c_idx + 1) * 16 <= bf16_chunk_scales.len()`.
            unsafe {
                let ws_ptr = matrix
                    .bf16_chunk_scales
                    .as_ptr()
                    .add(sc_base + c_idx * SIMD_ROW_STRIPE);
                let bf_lo = vld1q_u16(ws_ptr);
                let bf_hi = vld1q_u16(ws_ptr.add(8));
                let ws0 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf_lo))),
                    vq,
                );
                let ws1 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bf_lo))),
                    vq,
                );
                let ws2 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf_hi))),
                    vq,
                );
                let ws3 = vmulq_f32(
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bf_hi))),
                    vq,
                );

                facc0 = vfmaq_f32(facc0, f0, ws0);
                facc1 = vfmaq_f32(facc1, f1, ws1);
                facc2 = vfmaq_f32(facc2, f2, ws2);
                facc3 = vfmaq_f32(facc3, f3, ws3);
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        if valid_lanes == SIMD_ROW_STRIPE {
            unsafe {
                let dst = out_base.add(row_start);
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
            }
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            unsafe {
                let dst = tmp.as_mut_ptr();
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
                for (lane, &val) in tmp.iter().enumerate().take(valid_lanes) {
                    *out_base.add(row_start + lane) = val;
                }
            }
        }
        s += 1;
    }
}

/// Group-scaled multi-plane ternary matrix (`W ≈ sum_{p=0..P-1} beta_{p,r,c} * T^{(p)}_{r,j}`).
///
/// Every plane executes via **zero-multiply** `vqtbl1q_s8` register LUTs, sharing a single
/// `QuantizedActivationLut` per input vector while applying `BF16` group-64 scales at chunk flush.
#[derive(Clone, Debug)]
pub struct MultiPlaneTernaryMatrix {
    pub rows: usize,
    pub cols: usize,
    pub num_planes: usize,
    pub blocks_per_row: usize,
    pub num_stripes: usize,
    pub num_chunks: usize,
    pub planes: Vec<SimdPackedMatrix>,
    /// Layout: `[num_planes, num_stripes, num_chunks, SIMD_ROW_STRIPE]` (`BF16` raw bits).
    pub bf16_chunk_scales: Vec<u16>,
}

impl MultiPlaneTernaryMatrix {
    /// Total physical RAM footprint in bytes (packed ternary stripes + `BF16` group-64 scales).
    pub fn physical_bytes(&self) -> usize {
        let stripe_bytes: usize = self.planes.iter().map(|p| p.stripe_data.len()).sum();
        stripe_bytes + self.bf16_chunk_scales.len() * std::mem::size_of::<u16>()
    }

    /// Physical bits per weight parameter (packed 2-bit trits + `BF16` scale overhead).
    pub fn physical_bits_per_weight(&self) -> f64 {
        (self.physical_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Information-theoretic bits per weight ($\log_2(3) \approx 1.585$ bits/plane + `0.25` scale bits/plane).
    pub fn info_bits_per_weight(&self) -> f64 {
        let scale_bits =
            (self.bf16_chunk_scales.len() as f64 * 16.0) / ((self.rows * self.cols) as f64);
        (self.num_planes as f64) * 3.0f64.log2() + scale_bits
    }
}

/// Executes zero-multiply multi-plane `LUT-GEMV` (`aarch64` NEON `vqtbl1q_s8` or scalar fallback).
#[inline]
pub fn simd_lut_gemv_multiplane(
    matrix: &MultiPlaneTernaryMatrix,
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    assert_eq!(matrix.blocks_per_row, qlut.num_blocks);
    assert_eq!(matrix.num_chunks, qlut.chunk_scales.len());
    assert_eq!(out.len(), matrix.rows);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and slice bounds are checked above.
        unsafe { simd_lut_gemv_multiplane_neon(matrix, qlut, out) };
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        simd_lut_gemv_multiplane_scalar(matrix, qlut, out);
    }
}

/// Portable scalar reference for [`simd_lut_gemv_multiplane`].
pub fn simd_lut_gemv_multiplane_scalar(
    matrix: &MultiPlaneTernaryMatrix,
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    assert_eq!(matrix.blocks_per_row, qlut.num_blocks);
    assert_eq!(matrix.num_chunks, qlut.chunk_scales.len());
    assert_eq!(out.len(), matrix.rows);

    let bpr = matrix.blocks_per_row;
    let num_stripes = matrix.num_stripes;
    let num_chunks = matrix.num_chunks;

    for s in 0..num_stripes {
        let mut facc = [0.0f32; SIMD_ROW_STRIPE];

        for p in 0..matrix.num_planes {
            let stripe_slice = &matrix.planes[p].stripe_data
                [s * bpr * SIMD_ROW_STRIPE..(s + 1) * bpr * SIMD_ROW_STRIPE];
            let scale_plane_base = ((p * num_stripes + s) * num_chunks) * SIMD_ROW_STRIPE;

            for (chunk_idx, &qscale) in qlut.chunk_scales.iter().enumerate() {
                if qscale == 0.0 {
                    continue;
                }
                let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
                let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);
                let mut iacc = [0i16; SIMD_ROW_STRIPE];

                for b in b_start..b_end {
                    let codes = &stripe_slice[b * SIMD_ROW_STRIPE..(b + 1) * SIMD_ROW_STRIPE];
                    let t_lo = &qlut.lut_lo[b];
                    let t_hi = &qlut.lut_hi[b];
                    for lane in 0..SIMD_ROW_STRIPE {
                        let code = codes[lane];
                        let sum_lo = t_lo[(code & 0x0F) as usize] as i16;
                        let sum_hi = t_hi[(code >> 4) as usize] as i16;
                        iacc[lane] += sum_lo + sum_hi;
                    }
                }

                let ws_slice = &matrix.bf16_chunk_scales[scale_plane_base
                    + chunk_idx * SIMD_ROW_STRIPE
                    ..scale_plane_base + (chunk_idx + 1) * SIMD_ROW_STRIPE];
                for lane in 0..SIMD_ROW_STRIPE {
                    let wscale = bf16_to_f32(ws_slice[lane]);
                    facc[lane] += ((iacc[lane] as f32) * qscale) * wscale;
                }
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        out[row_start..row_start + valid_lanes].copy_from_slice(&facc[..valid_lanes]);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn simd_lut_gemv_multiplane_neon(
    matrix: &MultiPlaneTernaryMatrix,
    qlut: &QuantizedActivationLut,
    out: &mut [f32],
) {
    use std::arch::aarch64::*;

    let bpr = matrix.blocks_per_row;
    let num_stripes = matrix.num_stripes;
    let num_chunks = matrix.num_chunks;
    let mask_0f = vdupq_n_u8(0x0F);

    for s in 0..num_stripes {
        let mut facc0 = vdupq_n_f32(0.0);
        let mut facc1 = vdupq_n_f32(0.0);
        let mut facc2 = vdupq_n_f32(0.0);
        let mut facc3 = vdupq_n_f32(0.0);

        for p in 0..matrix.num_planes {
            // SAFETY: `s < num_stripes` and `stripe_data.len() == num_stripes * bpr * 16`.
            let stripe_ptr =
                unsafe { matrix.planes[p].stripe_data.as_ptr().add(s * bpr * SIMD_ROW_STRIPE) };
            let scale_base_offset = ((p * num_stripes + s) * num_chunks) * SIMD_ROW_STRIPE;

            for (chunk_idx, &qscale) in qlut.chunk_scales.iter().enumerate() {
                if qscale == 0.0 {
                    continue;
                }
                let b_start = chunk_idx * QUANT_CHUNK_BLOCKS;
                let b_end = (b_start + QUANT_CHUNK_BLOCKS).min(bpr);

                let mut iacc_lo_a = vdupq_n_s16(0);
                let mut iacc_hi_a = vdupq_n_s16(0);
                let mut iacc_lo_b = vdupq_n_s16(0);
                let mut iacc_hi_b = vdupq_n_s16(0);

                let mut b = b_start;
                while b + 2 <= b_end {
                    // SAFETY: `b + 1 < bpr`, valid 16-byte loads.
                    unsafe {
                        let tbl_lo0 = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi0 = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let codes0 = vld1q_u8(stripe_ptr.add(b * SIMD_ROW_STRIPE));

                        let tbl_lo1 = vld1q_s8(qlut.lut_lo.as_ptr().add(b + 1).cast::<i8>());
                        let tbl_hi1 = vld1q_s8(qlut.lut_hi.as_ptr().add(b + 1).cast::<i8>());
                        let codes1 = vld1q_u8(stripe_ptr.add((b + 1) * SIMD_ROW_STRIPE));

                        let sum_lo0 = vqtbl1q_s8(tbl_lo0, vandq_u8(codes0, mask_0f));
                        let sum_hi0 = vqtbl1q_s8(tbl_hi0, vshrq_n_u8::<4>(codes0));
                        let sum_lo1 = vqtbl1q_s8(tbl_lo1, vandq_u8(codes1, mask_0f));
                        let sum_hi1 = vqtbl1q_s8(tbl_hi1, vshrq_n_u8::<4>(codes1));

                        iacc_lo_a = vaddq_s16(
                            iacc_lo_a,
                            vaddl_s8(vget_low_s8(sum_lo0), vget_low_s8(sum_hi0)),
                        );
                        iacc_hi_a = vaddq_s16(iacc_hi_a, vaddl_high_s8(sum_lo0, sum_hi0));

                        iacc_lo_b = vaddq_s16(
                            iacc_lo_b,
                            vaddl_s8(vget_low_s8(sum_lo1), vget_low_s8(sum_hi1)),
                        );
                        iacc_hi_b = vaddq_s16(iacc_hi_b, vaddl_high_s8(sum_lo1, sum_hi1));
                    }
                    b += 2;
                }

                while b < b_end {
                    // SAFETY: `b < bpr`.
                    unsafe {
                        let tbl_lo = vld1q_s8(qlut.lut_lo.as_ptr().add(b).cast::<i8>());
                        let tbl_hi = vld1q_s8(qlut.lut_hi.as_ptr().add(b).cast::<i8>());
                        let codes = vld1q_u8(stripe_ptr.add(b * SIMD_ROW_STRIPE));

                        let sum_lo = vqtbl1q_s8(tbl_lo, vandq_u8(codes, mask_0f));
                        let sum_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(codes));

                        iacc_lo_a = vaddq_s16(
                            iacc_lo_a,
                            vaddl_s8(vget_low_s8(sum_lo), vget_low_s8(sum_hi)),
                        );
                        iacc_hi_a = vaddq_s16(iacc_hi_a, vaddl_high_s8(sum_lo, sum_hi));
                    }
                    b += 1;
                }

                let iacc_lo = vaddq_s16(iacc_lo_a, iacc_lo_b);
                let iacc_hi = vaddq_s16(iacc_hi_a, iacc_hi_b);
                let vq = vdupq_n_f32(qscale);

                let f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(iacc_lo))), vq);
                let f1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(iacc_lo)), vq);
                let f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(iacc_hi))), vq);
                let f3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(iacc_hi)), vq);

                // Decode 16 `BF16` group-64 weight scales via single-instruction bit-shifts.
                // SAFETY: `scale_base_offset + (chunk_idx + 1) * 16 <= bf16_chunk_scales.len()`.
                unsafe {
                    let ws_ptr = matrix
                        .bf16_chunk_scales
                        .as_ptr()
                        .add(scale_base_offset + chunk_idx * SIMD_ROW_STRIPE);
                    let bf_lo = vld1q_u16(ws_ptr);
                    let bf_hi = vld1q_u16(ws_ptr.add(8));
                    let ws0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf_lo)));
                    let ws1 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bf_lo)));
                    let ws2 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf_hi)));
                    let ws3 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_high_u16(bf_hi)));

                    facc0 = vfmaq_f32(facc0, f0, ws0);
                    facc1 = vfmaq_f32(facc1, f1, ws1);
                    facc2 = vfmaq_f32(facc2, f2, ws2);
                    facc3 = vfmaq_f32(facc3, f3, ws3);
                }
            }
        }

        let row_start = s * SIMD_ROW_STRIPE;
        let valid_lanes = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        if valid_lanes == SIMD_ROW_STRIPE {
            // SAFETY: `row_start + 16 <= out.len()`.
            unsafe {
                let dst = out.as_mut_ptr().add(row_start);
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
            }
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            // SAFETY: `tmp` has exact length 16.
            unsafe {
                let dst = tmp.as_mut_ptr();
                vst1q_f32(dst, facc0);
                vst1q_f32(dst.add(4), facc1);
                vst1q_f32(dst.add(8), facc2);
                vst1q_f32(dst.add(12), facc3);
            }
            out[row_start..row_start + valid_lanes].copy_from_slice(&tmp[..valid_lanes]);
        }
    }
}

/// Per-64-column `int8` activations (`max|x| / 127` scale) with per-chunk sums.
#[derive(Clone, Debug)]
pub struct Int8ChunkActivation {
    pub cols: usize,
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
}

impl Int8ChunkActivation {
    pub fn new_zeroed(cols: usize) -> Self {
        assert!(cols > 0 && cols.is_multiple_of(64));
        Self {
            cols,
            q: vec![0i8; cols],
            scales: vec![0.0f32; cols / 64],
            sums: vec![0i32; cols / 64],
        }
    }

    pub fn populate(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.cols);
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on `aarch64` and `x.len() == self.cols`.
            unsafe { self.populate_neon(x) };
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.populate_scalar(x);
        }
    }

    /// Fuses `FWHT-64` + `INT8` chunk quantization completely in SIMD registers (`0` intermediate buffer writes).
    #[inline]
    pub fn populate_fwht(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.cols);
        let nc = self.cols / 64;
        let qp = self.q.as_mut_ptr();
        let sp = self.scales.as_mut_ptr();
        let sump = self.sums.as_mut_ptr();
        for c in 0..nc {
            // SAFETY: `c < nc` indexes valid disjoint 64-byte / 1-element chunk slots.
            unsafe {
                Self::populate_chunk_fwht_raw(qp, sp, sump, c, &x[c * 64..(c + 1) * 64], None);
            }
        }
    }

    /// Fuses `RMSNorm` scaling (`(x * inv_rms) * weight`) + `FWHT-64` + `INT8` chunk quantization in registers.
    #[inline]
    pub fn populate_rmsnorm_fwht(&mut self, x: &[f32], weight: &[f32], inv_rms: f32) {
        assert_eq!(x.len(), self.cols);
        assert_eq!(weight.len(), self.cols);
        let nc = self.cols / 64;
        let qp = self.q.as_mut_ptr();
        let sp = self.scales.as_mut_ptr();
        let sump = self.sums.as_mut_ptr();
        for c in 0..nc {
            // SAFETY: `c < nc` indexes valid disjoint 64-byte / 1-element chunk slots.
            unsafe {
                Self::populate_chunk_fwht_raw(
                    qp,
                    sp,
                    sump,
                    c,
                    &x[c * 64..(c + 1) * 64],
                    Some((&weight[c * 64..(c + 1) * 64], inv_rms)),
                );
            }
        }
    }

    /// Quantizes a single 64-float block `x64` at chunk index `c` with optional `(rms_weight64, inv_rms)`
    /// fused directly into in-register `FWHT-64` + `INT8` quantization.
    ///
    /// # Safety
    /// `q_base` must be valid for `(c + 1) * 64` `i8` writes (disjoint for distinct `c`), and
    /// `scales_base` / `sums_base` must be valid at offset `c`.
    #[inline]
    pub unsafe fn populate_chunk_fwht_raw(
        q_base: *mut i8,
        scales_base: *mut f32,
        sums_base: *mut i32,
        c: usize,
        x64: &[f32],
        rms_weight_inv: Option<(&[f32], f32)>,
    ) {
        debug_assert_eq!(x64.len(), 64);
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on `aarch64`; caller guarantees chunk pointer bounds.
            unsafe {
                Self::populate_chunk_fwht_neon(q_base, scales_base, sums_base, c, x64, rms_weight_inv);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            unsafe {
                Self::populate_chunk_fwht_scalar(q_base, scales_base, sums_base, c, x64, rms_weight_inv);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn populate_chunk_fwht_neon(
        q_base: *mut i8,
        scales_base: *mut f32,
        sums_base: *mut i32,
        c: usize,
        x64: &[f32],
        rms_weight_inv: Option<(&[f32], f32)>,
    ) {
        use crate::kernels::hadamard::{rademacher_sign_masks_64, RADEMACHER_TABLE_64};
        use std::arch::aarch64::*;

        let fallback_masks;
        let mptr = if c < RADEMACHER_TABLE_64.len() {
            RADEMACHER_TABLE_64[c].as_ptr()
        } else {
            fallback_masks = rademacher_sign_masks_64(c);
            fallback_masks.as_ptr()
        };
        let scale = vdupq_n_f32(0.125);
        let xp = x64.as_ptr();
        let mut v = [vdupq_n_f32(0.0); 16];

        if let Some((w64, inv_rms)) = rms_weight_inv {
            debug_assert_eq!(w64.len(), 64);
            let wp = w64.as_ptr();
            let vinv = vdupq_n_f32(inv_rms);
            for (k, slot) in v.iter_mut().enumerate() {
                // SAFETY: `k * 4 + 4 <= 64` for `x64`, `w64`, and `masks`.
                unsafe {
                    let xn = vmulq_f32(vmulq_f32(vld1q_f32(xp.add(k * 4)), vinv), vld1q_f32(wp.add(k * 4)));
                    let m = vld1q_u32(mptr.add(k * 4));
                    *slot = vmulq_f32(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(xn), m)), scale);
                }
            }
        } else {
            for (k, slot) in v.iter_mut().enumerate() {
                // SAFETY: `k * 4 + 4 <= 64` for `x64` and `masks`.
                unsafe {
                    let raw = vld1q_u32(xp.add(k * 4).cast::<u32>());
                    let m = vld1q_u32(mptr.add(k * 4));
                    *slot = vmulq_f32(vreinterpretq_f32_u32(veorq_u32(raw, m)), scale);
                }
            }
        }

        let neg_mask_1 = unsafe { vld1q_u32([0u32, 0x8000_0000, 0, 0x8000_0000].as_ptr()) };
        for slot in &mut v {
            let x = *slot;
            let sum1 = vtrn1q_f32(x, x);
            let sub1 = vtrn2q_f32(x, x);
            let b1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sub1), neg_mask_1));
            let s1 = vaddq_f32(sum1, b1);
            let lo = vget_low_f32(s1);
            let hi = vget_high_f32(s1);
            *slot = vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
        }

        for vec_h in [1usize, 2, 4, 8] {
            let step = vec_h * 2;
            let mut i = 0usize;
            while i < 16 {
                for j in i..i + vec_h {
                    let a = v[j];
                    let b = v[j + vec_h];
                    v[j] = vaddq_f32(a, b);
                    v[j + vec_h] = vsubq_f32(a, b);
                }
                i += step;
            }
        }

        let mut vmax0 = vmaxq_f32(vabsq_f32(v[0]), vabsq_f32(v[1]));
        let mut vmax1 = vmaxq_f32(vabsq_f32(v[2]), vabsq_f32(v[3]));
        for k in (4..16).step_by(4) {
            vmax0 = vmaxq_f32(vmax0, vmaxq_f32(vabsq_f32(v[k]), vabsq_f32(v[k + 1])));
            vmax1 = vmaxq_f32(vmax1, vmaxq_f32(vabsq_f32(v[k + 2]), vabsq_f32(v[k + 3])));
        }
        let max_abs = vmaxvq_f32(vmaxq_f32(vmax0, vmax1));
        // SAFETY: caller guarantees `q_base.add(c * 64)` (`64` bytes), `scales_base.add(c)`, `sums_base.add(c)`.
        unsafe {
            let qc = q_base.add(c * 64);
            if max_abs <= f32::EPSILON {
                let z = vdupq_n_s8(0);
                vst1q_s8(qc, z);
                vst1q_s8(qc.add(16), z);
                vst1q_s8(qc.add(32), z);
                vst1q_s8(qc.add(48), z);
                *scales_base.add(c) = 0.0;
                *sums_base.add(c) = 0;
                return;
            }
            let inv = vdupq_n_f32(127.0 / max_abs);
            let mut vsum = vdupq_n_s32(0);
            for b in 0..4 {
                let q0 = vcvtnq_s32_f32(vmulq_f32(v[b * 4], inv));
                let q1 = vcvtnq_s32_f32(vmulq_f32(v[b * 4 + 1], inv));
                let q2 = vcvtnq_s32_f32(vmulq_f32(v[b * 4 + 2], inv));
                let q3 = vcvtnq_s32_f32(vmulq_f32(v[b * 4 + 3], inv));
                vsum = vaddq_s32(vsum, vaddq_s32(vaddq_s32(q0, q1), vaddq_s32(q2, q3)));
                let h0 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
                let h1 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
                vst1q_s8(qc.add(b * 16), vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
            }
            *scales_base.add(c) = max_abs / 127.0;
            *sums_base.add(c) = vaddvq_s32(vsum);
        }
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    unsafe fn populate_chunk_fwht_scalar(
        q_base: *mut i8,
        scales_base: *mut f32,
        sums_base: *mut i32,
        c: usize,
        x64: &[f32],
        rms_weight_inv: Option<(&[f32], f32)>,
    ) {
        use crate::kernels::hadamard::{rademacher_sign_masks_64, RADEMACHER_TABLE_64};
        let fallback_masks;
        let masks = if c < RADEMACHER_TABLE_64.len() {
            &RADEMACHER_TABLE_64[c]
        } else {
            fallback_masks = rademacher_sign_masks_64(c);
            &fallback_masks
        };
        let mut block = [0.0f32; 64];
        if let Some((w64, inv_rms)) = rms_weight_inv {
            for i in 0..64 {
                let xn = (x64[i] * inv_rms) * w64[i];
                block[i] = f32::from_bits(xn.to_bits() ^ masks[i]) * 0.125;
            }
        } else {
            for i in 0..64 {
                block[i] = f32::from_bits(x64[i].to_bits() ^ masks[i]) * 0.125;
            }
        }
        let mut h = 1usize;
        while h < 64 {
            let step = h * 2;
            let mut i = 0usize;
            while i < 64 {
                for j in i..i + h {
                    let a = block[j];
                    let b = block[j + h];
                    block[j] = a + b;
                    block[j + h] = a - b;
                }
                i += step;
            }
            h *= 2;
        }
        let max_abs = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // SAFETY: caller guarantees valid chunk pointers at `c`.
        unsafe {
            let qc = std::slice::from_raw_parts_mut(q_base.add(c * 64), 64);
            if max_abs <= f32::EPSILON {
                qc.fill(0);
                *scales_base.add(c) = 0.0;
                *sums_base.add(c) = 0;
                return;
            }
            let inv = 127.0 / max_abs;
            let mut sum = 0i32;
            for (dst, &v) in qc.iter_mut().zip(block.iter()) {
                let qv = (v * inv).round_ties_even() as i32;
                sum += qv;
                *dst = qv as i8;
            }
            *scales_base.add(c) = max_abs / 127.0;
            *sums_base.add(c) = sum;
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn populate_neon(&mut self, x: &[f32]) {
        use std::arch::aarch64::*;
        for (c, (xc, qc)) in x.chunks_exact(64).zip(self.q.chunks_exact_mut(64)).enumerate() {
            let xp = xc.as_ptr();
            // SAFETY: 16 x 4 floats = 64 floats in `xc`; 4 x 16 bytes = 64 bytes in `qc`.
            unsafe {
                let mut vmax = vdupq_n_f32(0.0);
                for i in 0..16 {
                    vmax = vmaxq_f32(vmax, vabsq_f32(vld1q_f32(xp.add(i * 4))));
                }
                let max_abs = vmaxvq_f32(vmax);
                if max_abs <= f32::EPSILON {
                    qc.fill(0);
                    self.scales[c] = 0.0;
                    self.sums[c] = 0;
                    continue;
                }
                let inv = vdupq_n_f32(127.0 / max_abs);
                let mut vsum = vdupq_n_s32(0);
                for b in 0..4 {
                    let mut q32 = [vdupq_n_s32(0); 4];
                    for (k, slot) in q32.iter_mut().enumerate() {
                        *slot = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(xp.add(b * 16 + k * 4)), inv));
                        vsum = vaddq_s32(vsum, *slot);
                    }
                    let h0 = vcombine_s16(vmovn_s32(q32[0]), vmovn_s32(q32[1]));
                    let h1 = vcombine_s16(vmovn_s32(q32[2]), vmovn_s32(q32[3]));
                    vst1q_s8(qc.as_mut_ptr().add(b * 16), vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)));
                }
                self.scales[c] = max_abs / 127.0;
                self.sums[c] = vaddvq_s32(vsum);
            }
        }
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    fn populate_scalar(&mut self, x: &[f32]) {
        for (c, (xc, qc)) in x.chunks_exact(64).zip(self.q.chunks_exact_mut(64)).enumerate() {
            let max_abs = xc.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            if max_abs <= f32::EPSILON {
                qc.fill(0);
                self.scales[c] = 0.0;
                self.sums[c] = 0;
                continue;
            }
            let inv = 127.0 / max_abs;
            let mut sum = 0i32;
            for (dst, &v) in qc.iter_mut().zip(xc.iter()) {
                let qv = (v * inv).round_ties_even() as i32;
                sum += qv;
                *dst = qv as i8;
            }
            self.scales[c] = max_abs / 127.0;
            self.sums[c] = sum;
        }
    }
}

/// Pure-`SDOT` Bipolar weights (`B in 4..=6`): `q = 2 u - (2^B - 1)`, where `u in [0, 2^B - 1] <= 63`
/// has bit 7 = `0` and therefore executes in a single signed `sdot` pass with zero LUT fallback!
///
/// Stripe pairs (32 rows) are packed per 64-col chunk as one contiguous record (data, hi planes,
/// scales) so each worker reads a single sequential stream.
#[derive(Clone, Debug)]
pub struct Nibble4SdotMatrix {
    pub rows: usize,
    pub cols: usize,
    pub num_stripes: usize,
    pub num_chunks: usize,
    /// Bipolar planes `4..num_planes` (`0`, `1`, or `2`).
    pub hi_planes: usize,
    /// `num_stripes / 2`.
    pub num_pairs: usize,
    /// `[pair, chunk]` records of `1088 + 256 * hi_planes` bytes.
    pub pair_stream: Vec<u8>,
    /// `[chunk]` records for an odd last stripe; empty when `num_stripes` is even.
    pub tail_stream: Vec<u8>,
}

impl Nibble4SdotMatrix {
    #[inline]
    pub const fn pair_rec_bytes(hi_planes: usize) -> usize {
        1024 + 256 * hi_planes + 64
    }

    #[inline]
    pub const fn tail_rec_bytes(hi_planes: usize) -> usize {
        512 + 128 * hi_planes + 32
    }

    /// Repacks a 4-6 plane [`FusedBipolarLutMatrix`] for `SDOT`.
    pub fn from_bipolar(m: &FusedBipolarLutMatrix) -> Self {
        Self::from_bipolar_with_perm(m, false)
    }

    /// Like [`Self::from_bipolar`], optionally interleaving `gate_up_proj` into 64-row
    /// `[gate_64(ic), up_64(ic)]` blocks.
    pub fn from_bipolar_with_perm(m: &FusedBipolarLutMatrix, interleave_gate_up: bool) -> Self {
        assert!((4..=6).contains(&m.num_planes), "SDOT repack requires 4..=6 planes");
        if interleave_gate_up {
            assert!(
                m.rows.is_multiple_of(128),
                "gate_up interleaving requires rows divisible by 128"
            );
        }
        let hi_planes = m.num_planes - 4;
        let num_pairs = m.num_stripes / 2;
        let has_tail = (m.num_stripes & 1) != 0;
        let p_rec = Self::pair_rec_bytes(hi_planes);
        let t_rec = Self::tail_rec_bytes(hi_planes);
        let mut pair_stream = vec![0u8; num_pairs * m.num_chunks * p_rec];
        let mut tail_stream = vec![0u8; usize::from(has_tail) * m.num_chunks * t_rec];
        let inter_dim = m.rows / 2;
        let nc_i = inter_dim / 64;

        let map_row = |orig_r: usize| -> usize {
            if !interleave_gate_up {
                orig_r
            } else if orig_r < inter_dim {
                let ic = orig_r / 64;
                (ic * 2) * 64 + (orig_r % 64)
            } else {
                let up_r = orig_r - inter_dim;
                let ic = up_r / 64;
                debug_assert!(ic < nc_i);
                (ic * 2 + 1) * 64 + (up_r % 64)
            }
        };

        let mut codes = vec![0u8; m.cols];
        for orig_r in 0..m.rows {
            m.unpack_row_codes(orig_r, &mut codes);
            let r = map_row(orig_r);
            let orig_stripe = orig_r / SIMD_ROW_STRIPE;
            let orig_lane = orig_r % SIMD_ROW_STRIPE;
            let (stripe, lane) = (r / SIMD_ROW_STRIPE, r % SIMD_ROW_STRIPE);
            let (g, j) = (lane / 4, lane % 4);
            let in_pair = stripe < num_pairs * 2;
            let p = stripe / 2;
            let side = stripe & 1;

            for c in 0..m.num_chunks {
                let sc_u16 =
                    m.bf16_chunk_scales[(orig_stripe * m.num_chunks + c) * SIMD_ROW_STRIPE + orig_lane];
                let sc_bytes = sc_u16.to_le_bytes();
                let (rec_slice, data_off, hi_off, sc_off) = if in_pair {
                    let base = (p * m.num_chunks + c) * p_rec;
                    (
                        &mut pair_stream[base..base + p_rec],
                        side * 512,
                        1024 + side * 128 * hi_planes,
                        1024 + 256 * hi_planes + side * 32,
                    )
                } else {
                    let base = c * t_rec;
                    (
                        &mut tail_stream[base..base + t_rec],
                        0,
                        512,
                        512 + 128 * hi_planes,
                    )
                };
                rec_slice[sc_off + lane * 2..sc_off + lane * 2 + 2].copy_from_slice(&sc_bytes);

                let c_col = c * 64;
                for (sub_col, &u) in codes[c_col..c_col + 64].iter().enumerate() {
                    let (k, w) = (sub_col / 16, sub_col % 16);
                    let half = w / 8;
                    let sub = w / 4;
                    let t = w % 4;
                    let byte = data_off + k * 128 + g * 32 + half * 16 + j * 4 + t;
                    let lo = u & 0x0F;
                    rec_slice[byte] |= if (w % 8) < 4 { lo } else { lo << 4 };
                    if hi_planes == 1 {
                        let hi_byte = hi_off + (k / 2) * 64 + g * 16 + j * 4 + t;
                        let bit_pos = if (k & 1) == 0 { 4 + sub } else { sub };
                        rec_slice[hi_byte] |= ((u >> 4) & 1) << bit_pos;
                    } else if hi_planes == 2 {
                        let hi_byte = hi_off + k * 64 + g * 16 + j * 4 + t;
                        let shift = match sub {
                            0 => 4,
                            1 => 6,
                            2 => 2,
                            _ => 0,
                        };
                        rec_slice[hi_byte] |= ((u >> 4) & 0x03) << shift;
                    }
                }
            }
        }
        Self {
            rows: m.rows,
            cols: m.cols,
            num_stripes: m.num_stripes,
            num_chunks: m.num_chunks,
            hi_planes,
            num_pairs,
            pair_stream,
            tail_stream,
        }
    }

    pub fn num_planes(&self) -> usize {
        4 + self.hi_planes
    }

    pub fn physical_bytes(&self) -> usize {
        self.pair_stream.len() + self.tail_stream.len()
    }

    /// `(data, hi planes, BF16 scales)` pointers for `(stripe, chunk)`.
    #[inline(always)]
    fn stripe_chunk_ptrs(&self, stripe: usize, c: usize) -> (*const u8, *const u8, *const u8) {
        debug_assert!(stripe < self.num_stripes && c < self.num_chunks);
        let hp = self.hi_planes;
        if stripe < self.num_pairs * 2 {
            let p = stripe / 2;
            let side = stripe & 1;
            let p_rec = Self::pair_rec_bytes(hp);
            // SAFETY: `p < num_pairs` and `c < num_chunks`, so the record is inside `pair_stream`.
            unsafe {
                let base = self.pair_stream.as_ptr().add((p * self.num_chunks + c) * p_rec);
                (
                    base.add(side * 512),
                    base.add(1024 + side * 128 * hp),
                    base.add(1024 + 256 * hp + side * 32),
                )
            }
        } else {
            let t_rec = Self::tail_rec_bytes(hp);
            // SAFETY: Odd tail stripe implies `tail_stream.len() == num_chunks * t_rec`.
            unsafe {
                let base = self.tail_stream.as_ptr().add(c * t_rec);
                (base, base.add(512), base.add(512 + 128 * hp))
            }
        }
    }

    /// Dequantizes row `r` into `out` in the Hadamard-rotated domain.
    pub fn dequantize_row_rotated(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows);
        assert_eq!(out.len(), self.cols);
        let (stripe, lane) = (r / SIMD_ROW_STRIPE, r % SIMD_ROW_STRIPE);
        let (g, j) = (lane / 4, lane % 4);
        let max_odd = (1i32 << self.num_planes()) - 1;
        for c in 0..self.num_chunks {
            let (d_ptr, h_ptr, s_ptr) = self.stripe_chunk_ptrs(stripe, c);
            // SAFETY: `d_ptr` has 512 B, `h_ptr` `128 * hi_planes` B, `s_ptr` 32 B.
            let delta = unsafe {
                let b0 = *s_ptr.add(lane * 2);
                let b1 = *s_ptr.add(lane * 2 + 1);
                bf16_to_f32(u16::from_le_bytes([b0, b1]))
            };
            let dst_chunk = &mut out[c * 64..(c + 1) * 64];
            for (sub_col, dst) in dst_chunk.iter_mut().enumerate() {
                let (k, w) = (sub_col / 16, sub_col % 16);
                let sub = w / 4;
                let t = w % 4;
                // SAFETY: offsets stay within `d_ptr` and `h_ptr`.
                unsafe {
                    let byte = *d_ptr.add(k * 128 + g * 32 + (w / 8) * 16 + j * 4 + t);
                    let mut u = (if (w % 8) < 4 { byte & 0x0F } else { byte >> 4 }) as i32;
                    if self.hi_planes == 1 {
                        let hb = *h_ptr.add((k / 2) * 64 + g * 16 + j * 4 + t);
                        let bit_pos = if (k & 1) == 0 { 4 + sub } else { sub };
                        u |= (((hb >> bit_pos) & 1) as i32) << 4;
                    } else if self.hi_planes == 2 {
                        let hb = *h_ptr.add(k * 64 + g * 16 + j * 4 + t);
                        let shift = match sub {
                            0 => 4,
                            1 => 6,
                            2 => 2,
                            _ => 0,
                        };
                        u |= (((hb >> shift) & 0x03) as i32) << 4;
                    }
                    *dst = ((2 * u - max_odd) as f32) * delta;
                }
            }
        }
    }
}

/// Executes stripes `[stripe_start, stripe_end)` of a [`Nibble4SdotMatrix`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
/// `blut` must carry `int8` activations (`lut_enabled` is not required even when `hi_planes > 0`).
#[inline]
pub unsafe fn nibble4_sdot_stripe_range(
    matrix: &Nibble4SdotMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, blut.cols);
    let act = blut.int8.as_ref().expect("int8 activations required for Nibble4");
    // SAFETY: Caller upholds bounds.
    unsafe { nibble4_sdot_stripe_range_int8(matrix, act, stripe_start, stripe_end, out_base) };
}

/// Direct Pure-`SDOT` execution of stripes `[stripe_start, stripe_end)` from [`Int8ChunkActivation`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn nibble4_sdot_stripe_range_int8(
    matrix: &Nibble4SdotMatrix,
    act: &Int8ChunkActivation,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, act.cols);
    // SAFETY: `act` buffers have lengths `cols`, `cols/64`, `cols/64`.
    unsafe {
        nibble4_sdot_stripe_range_raw(
            matrix,
            act.q.as_ptr(),
            act.scales.as_ptr(),
            act.sums.as_ptr(),
            stripe_start,
            stripe_end,
            out_base,
        );
    }
}

/// Raw-pointer Pure-`SDOT` execution of stripes `[stripe_start, stripe_end)` (allows zero-barrier
/// worker-stack L1D activation quantization without heap allocations or cross-core cache traffic).
///
/// # Safety
/// `xq` valid for `matrix.cols` bytes, `xscales`/`xsums` valid for `matrix.num_chunks` elements,
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn nibble4_sdot_stripe_range_raw(
    matrix: &Nibble4SdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    xsums: *const i32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert!(stripe_end <= matrix.num_stripes);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Apple/ARMv8.4 targets have `dotprod`; caller upholds bounds.
        unsafe {
            match matrix.hi_planes {
                0 => sdot_decode_neon::<0>(matrix, xq, xscales, xsums, stripe_start, stripe_end, out_base),
                1 => sdot_decode_neon::<1>(matrix, xq, xscales, xsums, stripe_start, stripe_end, out_base),
                2 => sdot_decode_neon::<2>(matrix, xq, xscales, xsums, stripe_start, stripe_end, out_base),
                _ => unreachable!("hi_planes must be 0, 1, or 2"),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // SAFETY: Caller upholds bounds.
        let nc = matrix.num_chunks;
        let q_s = unsafe { std::slice::from_raw_parts(xq, matrix.cols) };
        let sc_s = unsafe { std::slice::from_raw_parts(xscales, nc) };
        let mut row = vec![0.0f32; matrix.cols];
        for s in stripe_start..stripe_end {
            for lane in 0..SIMD_ROW_STRIPE {
                let r = s * SIMD_ROW_STRIPE + lane;
                if r >= matrix.rows {
                    break;
                }
                matrix.dequantize_row_rotated(r, &mut row);
                let mut acc = 0.0f32;
                for (c, (wc, qc)) in row.chunks_exact(64).zip(q_s.chunks_exact(64)).enumerate() {
                    let dot: f32 = wc.iter().zip(qc.iter()).map(|(&w, &q)| w * q as f32).sum();
                    acc += dot * sc_s[c];
                }
                unsafe { *out_base.add(r) = acc };
            }
        }
        let _ = xsums;
    }
}

/// Portable scalar reference for [`nibble4_sdot_stripe_range`].
///
/// # Safety
/// Same preconditions as [`nibble4_sdot_stripe_range`].
pub unsafe fn nibble4_sdot_stripe_range_scalar(
    matrix: &Nibble4SdotMatrix,
    act: &Int8ChunkActivation,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    let mut row = vec![0.0f32; matrix.cols];
    for s in stripe_start..stripe_end {
        for lane in 0..SIMD_ROW_STRIPE {
            let r = s * SIMD_ROW_STRIPE + lane;
            if r >= matrix.rows {
                break;
            }
            matrix.dequantize_row_rotated(r, &mut row);
            let mut acc = 0.0f32;
            for (c, (wc, qc)) in row.chunks_exact(64).zip(act.q.chunks_exact(64)).enumerate() {
                let dot: f32 = wc.iter().zip(qc.iter()).map(|(&w, &q)| w * q as f32).sum();
                acc += dot * act.scales[c];
            }
            // SAFETY: `r < matrix.rows` inside the caller's disjoint stripe range.
            unsafe { *out_base.add(r) = acc };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_lane<const L: i32>(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::uint8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SAFETY: `sdot` (FEAT_DotProd) is baseline on every Apple-silicon / ARMv8.4+ target; `a` holds
    // unsigned codes `0..=63` (bit 7 == 0) so reading it as signed bytes is exact.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod\nsdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{l}]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            l = const L,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_s8_lane<const L: i32>(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::int8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SAFETY: `sdot` (FEAT_DotProd) is available on all Apple Silicon.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod\nsdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{l}]",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            l = const L,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn unpack_sdot_quad_local<const HP: usize>(
    va: std::arch::aarch64::uint8x16_t,
    vb: std::arch::aarch64::uint8x16_t,
    h_chunk_ptr: *const u8,
    k: usize,
    g: usize,
    mask_0f: std::arch::aarch64::uint8x16_t,
    mask_hi: std::arch::aarch64::uint8x16_t,
) -> (
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
    std::arch::aarch64::uint8x16_t,
) {
    use std::arch::aarch64::*;
    // SAFETY: NEON is baseline on `aarch64`; caller guarantees `h_chunk_ptr` has `128 * HP` bytes.
    unsafe {
        let la = vandq_u8(va, mask_0f);
        let ha = vshrq_n_u8::<4>(va);
        let lb = vandq_u8(vb, mask_0f);
        let hb = vshrq_n_u8::<4>(vb);
        if HP == 0 {
            (la, ha, lb, hb)
        } else if HP == 1 {
            let h = vld1q_u8(h_chunk_ptr.add((k >> 1) * 64 + g * 16));
            if (k & 1) == 0 {
                (
                    vbslq_u8(mask_hi, h, la),
                    vbslq_u8(mask_hi, vshrq_n_u8::<1>(h), ha),
                    vbslq_u8(mask_hi, vshrq_n_u8::<2>(h), lb),
                    vbslq_u8(mask_hi, vshrq_n_u8::<3>(h), hb),
                )
            } else {
                (
                    vbslq_u8(mask_hi, vshlq_n_u8::<4>(h), la),
                    vbslq_u8(mask_hi, vshlq_n_u8::<3>(h), ha),
                    vbslq_u8(mask_hi, vshlq_n_u8::<2>(h), lb),
                    vbslq_u8(mask_hi, vshlq_n_u8::<1>(h), hb),
                )
            }
        } else {
            let h = vld1q_u8(h_chunk_ptr.add(k * 64 + g * 16));
            (
                vbslq_u8(mask_hi, h, la),
                vbslq_u8(mask_hi, vshrq_n_u8::<2>(h), ha),
                vbslq_u8(mask_hi, vshlq_n_u8::<2>(h), lb),
                vbslq_u8(mask_hi, vshlq_n_u8::<4>(h), hb),
            )
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot_accum_16cols<const HP: usize, const K_ODD: bool>(
    mut acc: std::arch::aarch64::int32x4_t,
    va: std::arch::aarch64::uint8x16_t,
    vb: std::arch::aarch64::uint8x16_t,
    h: std::arch::aarch64::uint8x16_t,
    x16: std::arch::aarch64::int8x16_t,
    mask_0f: std::arch::aarch64::uint8x16_t,
    mask_hi: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: NEON + `sdot` are baseline on Apple Silicon; codes are in `0..=63`.
    unsafe {
        let la = vandq_u8(va, mask_0f);
        let ha = vshrq_n_u8::<4>(va);
        let lb = vandq_u8(vb, mask_0f);
        let hb = vshrq_n_u8::<4>(vb);
        let (ula, uha, ulb, uhb) = if HP == 0 {
            (la, ha, lb, hb)
        } else if HP == 1 {
            if !K_ODD {
                (
                    vbslq_u8(mask_hi, h, la),
                    vbslq_u8(mask_hi, vshrq_n_u8::<1>(h), ha),
                    vbslq_u8(mask_hi, vshrq_n_u8::<2>(h), lb),
                    vbslq_u8(mask_hi, vshrq_n_u8::<3>(h), hb),
                )
            } else {
                (
                    vbslq_u8(mask_hi, vshlq_n_u8::<4>(h), la),
                    vbslq_u8(mask_hi, vshlq_n_u8::<3>(h), ha),
                    vbslq_u8(mask_hi, vshlq_n_u8::<2>(h), lb),
                    vbslq_u8(mask_hi, vshlq_n_u8::<1>(h), hb),
                )
            }
        } else {
            (
                vbslq_u8(mask_hi, h, la),
                vbslq_u8(mask_hi, vshrq_n_u8::<2>(h), ha),
                vbslq_u8(mask_hi, vshlq_n_u8::<2>(h), lb),
                vbslq_u8(mask_hi, vshlq_n_u8::<4>(h), hb),
            )
        };
        acc = sdot_lane::<0>(acc, ula, x16);
        acc = sdot_lane::<1>(acc, uha, x16);
        acc = sdot_lane::<2>(acc, ulb, x16);
        acc = sdot_lane::<3>(acc, uhb, x16);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn sdot_one_stripe_neon<const HP: usize>(
    matrix: &Nibble4SdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    xsums: *const i32,
    s: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let max_odd: i32 = (1i32 << (4 + HP)) - 1;
    let nc = matrix.num_chunks;
    let mask_0f = vdupq_n_u8(0x0F);
    let mask_hi = vdupq_n_u8(if HP == 1 { 0x10 } else { 0x30 });
    let zero_u8 = vdupq_n_u8(0);
    let mut f = [vdupq_n_f32(0.0); 4];

    for c in 0..nc {
        let sx = unsafe { *xscales.add(c) };
        if sx == 0.0 {
            continue;
        }
        let (d_ptr, h_ptr, sc_ptr) = matrix.stripe_chunk_ptrs(s, c);
        let mut a = [vdupq_n_s32(0); 4];
        let mut b = [vdupq_n_s32(0); 4];
        for kp in 0..2 {
            let k0 = kp * 2;
            let k1 = k0 + 1;
            // SAFETY: `c * 64 + 64 <= cols`; pointers stay inside the `(s, c)` record.
            unsafe {
                let x0 = vld1q_s8(xq.add(c * 64 + k0 * 16));
                let x1 = vld1q_s8(xq.add(c * 64 + k1 * 16));
                let blk0 = d_ptr.add(k0 * 128);
                let blk1 = d_ptr.add(k1 * 128);
                for g in 0..4 {
                    let (h0, h1) = if HP == 0 {
                        (zero_u8, zero_u8)
                    } else if HP == 1 {
                        let h = vld1q_u8(h_ptr.add(kp * 64 + g * 16));
                        (h, h)
                    } else {
                        (
                            vld1q_u8(h_ptr.add(k0 * 64 + g * 16)),
                            vld1q_u8(h_ptr.add(k1 * 64 + g * 16)),
                        )
                    };
                    a[g] = sdot_accum_16cols::<HP, false>(
                        a[g],
                        vld1q_u8(blk0.add(g * 32)),
                        vld1q_u8(blk0.add(g * 32 + 16)),
                        h0,
                        x0,
                        mask_0f,
                        mask_hi,
                    );
                    b[g] = sdot_accum_16cols::<HP, true>(
                        b[g],
                        vld1q_u8(blk1.add(g * 32)),
                        vld1q_u8(blk1.add(g * 32 + 16)),
                        h1,
                        x1,
                        mask_0f,
                        mask_hi,
                    );
                }
            }
        }
        let offset = vdupq_n_s32(max_odd * unsafe { *xsums.add(c) });
        let vsx = vdupq_n_f32(sx);
        // SAFETY: `sc_ptr` holds `32` bytes (`16` little-endian `BF16` scales).
        unsafe {
            let bf0 = vreinterpretq_u16_u8(vld1q_u8(sc_ptr));
            let bf1 = vreinterpretq_u16_u8(vld1q_u8(sc_ptr.add(16)));
            let ws = [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf0))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf0)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf1))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf1)),
            ];
            for g in 0..4 {
                let dot = vsubq_s32(vshlq_n_s32::<1>(vaddq_s32(a[g], b[g])), offset);
                f[g] = vfmaq_f32(f[g], vmulq_f32(vcvtq_f32_s32(dot), vsx), ws[g]);
            }
        }
    }
    let row_start = s * SIMD_ROW_STRIPE;
    let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
    // SAFETY: `row_start + valid <= rows`.
    unsafe {
        let dst = out_base.add(row_start);
        if valid == SIMD_ROW_STRIPE {
            vst1q_f32(dst, f[0]);
            vst1q_f32(dst.add(4), f[1]);
            vst1q_f32(dst.add(8), f[2]);
            vst1q_f32(dst.add(12), f[3]);
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            for (g, fg) in f.iter().enumerate() {
                vst1q_f32(tmp.as_mut_ptr().add(g * 4), *fg);
            }
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, valid);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sdot_decode_neon<const HP: usize>(
    matrix: &Nibble4SdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    xsums: *const i32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let max_odd: i32 = (1i32 << (4 + HP)) - 1;
    let nc = matrix.num_chunks;
    let p_rec = Nibble4SdotMatrix::pair_rec_bytes(HP);
    let mask_0f = vdupq_n_u8(0x0F);
    let mask_hi = vdupq_n_u8(if HP == 1 { 0x10 } else { 0x30 });
    let zero_u8 = vdupq_n_u8(0);
    let pair_stream = matrix.pair_stream.as_ptr();

    let mut s = stripe_start;
    if (s & 1) != 0 && s < stripe_end {
        // SAFETY: `s < stripe_end <= matrix.num_stripes`.
        unsafe { sdot_one_stripe_neon::<HP>(matrix, xq, xscales, xsums, s, out_base) };
        s += 1;
    }

    // Stripe pairs: `s` is even here, so pair `p = s / 2` is one contiguous record per chunk.
    while s + 2 <= stripe_end {
        let p = s / 2;
        let pair_base = unsafe { pair_stream.add(p * nc * p_rec) };
        let mut fa = [vdupq_n_f32(0.0); 4];
        let mut fb = [vdupq_n_f32(0.0); 4];

        for c in 0..nc {
            let sx = unsafe { *xscales.add(c) };
            if sx == 0.0 {
                continue;
            }
            let rec = unsafe { pair_base.add(c * p_rec) };
            let mut aa = [vdupq_n_s32(0); 4];
            let mut ab = [vdupq_n_s32(0); 4];

            // SAFETY: `c * 64 + 64 <= cols`; `rec` holds `p_rec` contiguous bytes for `(p, c)`.
            unsafe {
                let x_base = xq.add(c * 64);
                let x0 = vld1q_s8(x_base);
                let x1 = vld1q_s8(x_base.add(16));
                let x2 = vld1q_s8(x_base.add(32));
                let x3 = vld1q_s8(x_base.add(48));

                let da = rec;
                let db = rec.add(512);
                let hia = rec.add(1024);
                let hib = rec.add(1024 + 128 * HP);

                for g in 0..4 {
                    let (ha0, ha1, hb0, hb1) = if HP == 0 {
                        (zero_u8, zero_u8, zero_u8, zero_u8)
                    } else if HP == 1 {
                        let ha = vld1q_u8(hia.add(g * 16));
                        let hb = vld1q_u8(hib.add(g * 16));
                        (ha, ha, hb, hb)
                    } else {
                        (
                            vld1q_u8(hia.add(g * 16)),
                            vld1q_u8(hia.add(64 + g * 16)),
                            vld1q_u8(hib.add(g * 16)),
                            vld1q_u8(hib.add(64 + g * 16)),
                        )
                    };
                    aa[g] = sdot_accum_16cols::<HP, false>(
                        aa[g],
                        vld1q_u8(da.add(g * 32)),
                        vld1q_u8(da.add(g * 32 + 16)),
                        ha0,
                        x0,
                        mask_0f,
                        mask_hi,
                    );
                    ab[g] = sdot_accum_16cols::<HP, false>(
                        ab[g],
                        vld1q_u8(db.add(g * 32)),
                        vld1q_u8(db.add(g * 32 + 16)),
                        hb0,
                        x0,
                        mask_0f,
                        mask_hi,
                    );
                    aa[g] = sdot_accum_16cols::<HP, true>(
                        aa[g],
                        vld1q_u8(da.add(128 + g * 32)),
                        vld1q_u8(da.add(128 + g * 32 + 16)),
                        ha1,
                        x1,
                        mask_0f,
                        mask_hi,
                    );
                    ab[g] = sdot_accum_16cols::<HP, true>(
                        ab[g],
                        vld1q_u8(db.add(128 + g * 32)),
                        vld1q_u8(db.add(128 + g * 32 + 16)),
                        hb1,
                        x1,
                        mask_0f,
                        mask_hi,
                    );

                    let (ha2, ha3, hb2, hb3) = if HP == 0 {
                        (zero_u8, zero_u8, zero_u8, zero_u8)
                    } else if HP == 1 {
                        let ha = vld1q_u8(hia.add(64 + g * 16));
                        let hb = vld1q_u8(hib.add(64 + g * 16));
                        (ha, ha, hb, hb)
                    } else {
                        (
                            vld1q_u8(hia.add(128 + g * 16)),
                            vld1q_u8(hia.add(192 + g * 16)),
                            vld1q_u8(hib.add(128 + g * 16)),
                            vld1q_u8(hib.add(192 + g * 16)),
                        )
                    };
                    aa[g] = sdot_accum_16cols::<HP, false>(
                        aa[g],
                        vld1q_u8(da.add(256 + g * 32)),
                        vld1q_u8(da.add(256 + g * 32 + 16)),
                        ha2,
                        x2,
                        mask_0f,
                        mask_hi,
                    );
                    ab[g] = sdot_accum_16cols::<HP, false>(
                        ab[g],
                        vld1q_u8(db.add(256 + g * 32)),
                        vld1q_u8(db.add(256 + g * 32 + 16)),
                        hb2,
                        x2,
                        mask_0f,
                        mask_hi,
                    );
                    aa[g] = sdot_accum_16cols::<HP, true>(
                        aa[g],
                        vld1q_u8(da.add(384 + g * 32)),
                        vld1q_u8(da.add(384 + g * 32 + 16)),
                        ha3,
                        x3,
                        mask_0f,
                        mask_hi,
                    );
                    ab[g] = sdot_accum_16cols::<HP, true>(
                        ab[g],
                        vld1q_u8(db.add(384 + g * 32)),
                        vld1q_u8(db.add(384 + g * 32 + 16)),
                        hb3,
                        x3,
                        mask_0f,
                        mask_hi,
                    );
                }

                let offset = vdupq_n_s32(max_odd * *xsums.add(c));
                let vsx = vdupq_n_f32(sx);
                let sca = rec.add(1024 + 256 * HP);
                let scb = sca.add(32);

                let bfa0 = vreinterpretq_u16_u8(vld1q_u8(sca));
                let bfa1 = vreinterpretq_u16_u8(vld1q_u8(sca.add(16)));
                let wsa = [
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfa0))),
                    vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfa0)),
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfa1))),
                    vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfa1)),
                ];
                for g in 0..4 {
                    let dot = vsubq_s32(vshlq_n_s32::<1>(aa[g]), offset);
                    fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(dot), vsx), wsa[g]);
                }

                let bfb0 = vreinterpretq_u16_u8(vld1q_u8(scb));
                let bfb1 = vreinterpretq_u16_u8(vld1q_u8(scb.add(16)));
                let wsb = [
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfb0))),
                    vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfb0)),
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bfb1))),
                    vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bfb1)),
                ];
                for g in 0..4 {
                    let dot = vsubq_s32(vshlq_n_s32::<1>(ab[g]), offset);
                    fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(dot), vsx), wsb[g]);
                }
            }
        }

        for (st_idx, f_arr) in [(s, &fa), (s + 1, &fb)] {
            let row_start = st_idx * SIMD_ROW_STRIPE;
            let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
            // SAFETY: `row_start + valid <= matrix.rows`.
            unsafe {
                let dst = out_base.add(row_start);
                if valid == SIMD_ROW_STRIPE {
                    vst1q_f32(dst, f_arr[0]);
                    vst1q_f32(dst.add(4), f_arr[1]);
                    vst1q_f32(dst.add(8), f_arr[2]);
                    vst1q_f32(dst.add(12), f_arr[3]);
                } else {
                    let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
                    for (g, fg) in f_arr.iter().enumerate() {
                        vst1q_f32(tmp.as_mut_ptr().add(g * 4), *fg);
                    }
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, valid);
                }
            }
        }
        s += 2;
    }

    while s < stripe_end {
        // SAFETY: `s < stripe_end <= matrix.num_stripes`.
        unsafe { sdot_one_stripe_neon::<HP>(matrix, xq, xscales, xsums, s, out_base) };
        s += 1;
    }
}

/// Executes stripes `[stripe_start, stripe_end)` for several tokens: row `r` of token `t` (with
/// activations `acts[t]`) goes to `out_base[t * out_stride + r]`.
///
/// Weights are tiled in L1 (`C_TILE` chunks) and reused across up to 8 tokens, so they are read from
/// memory once per call.
///
/// # Safety
/// As [`nibble4_sdot_stripe_range`], with `out_base` valid for `acts.len() * out_stride` floats.
pub unsafe fn nibble4_sdot_batch_stripe_range(
    matrix: &Nibble4SdotMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    if acts.is_empty() || stripe_start >= stripe_end {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: caller upholds bounds.
        unsafe {
            match matrix.hi_planes {
                0 => sdot_batch_stripe_outer_neon::<0>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
                1 => sdot_batch_stripe_outer_neon::<1>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
                2 => sdot_batch_stripe_outer_neon::<2>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
                _ => unreachable!("hi_planes must be 0, 1, or 2"),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (t, blut) in acts.iter().enumerate() {
            // SAFETY: caller upholds bounds.
            unsafe {
                nibble4_sdot_stripe_range(matrix, blut, stripe_start, stripe_end, out_base.add(t * out_stride))
            };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sdot_batch_stripe_outer_neon<const HP: usize>(
    matrix: &Nibble4SdotMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    const C_TILE: usize = 16;
    let nc = matrix.num_chunks;
    let n_tok = acts.len();

    let run_one = |s: usize, c_base: usize, c_end: usize| {
        let first_tile = c_base == 0;
        let mut t = 0usize;
        while t + 8 <= n_tok {
            // SAFETY: `t + 8 <= n_tok`, `s < num_stripes`, `c_end <= nc`.
            unsafe {
                sdot_batch_tile_neon::<HP, 8>(
                    matrix,
                    &acts[t..t + 8],
                    s,
                    c_base,
                    c_end,
                    first_tile,
                    out_base.add(t * out_stride),
                    out_stride,
                );
            }
            t += 8;
        }
        let rem = n_tok - t;
        if rem > 0 {
            // SAFETY: `rem in 1..=7` and `t + rem == n_tok`.
            unsafe {
                let sub = &acts[t..n_tok];
                let dst = out_base.add(t * out_stride);
                match rem {
                    7 => sdot_batch_tile_neon::<HP, 7>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    6 => sdot_batch_tile_neon::<HP, 6>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    5 => sdot_batch_tile_neon::<HP, 5>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    4 => sdot_batch_tile_neon::<HP, 4>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    3 => sdot_batch_tile_neon::<HP, 3>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    2 => sdot_batch_tile_neon::<HP, 2>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    _ => sdot_batch_tile_neon::<HP, 1>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                }
            }
        }
    };

    let mut s = stripe_start;
    if (s & 1) != 0 && s < stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            run_one(s, c_base, c_end);
            c_base = c_end;
        }
        s += 1;
    }
    // Tile `C_TILE` chunks so stripe `s + 1` reads its half of the pair record from L1.
    while s + 2 <= stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            run_one(s, c_base, c_end);
            run_one(s + 1, c_base, c_end);
            c_base = c_end;
        }
        s += 2;
    }
    while s < stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            run_one(s, c_base, c_end);
            c_base = c_end;
        }
        s += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn sdot_batch_tile_neon<const HP: usize, const NT: usize>(
    matrix: &Nibble4SdotMatrix,
    acts: &[BipolarActivationLut],
    s: usize,
    c_base: usize,
    c_end: usize,
    first_tile: bool,
    out_base: *mut f32,
    out_stride: usize,
) {
    use std::arch::aarch64::*;
    let i8a: [&Int8ChunkActivation; NT] =
        std::array::from_fn(|t| acts[t].int8.as_ref().expect("int8 activations required for Nibble4"));
    let max_odd: i32 = (1i32 << (4 + HP)) - 1;
    let mask_0f = vdupq_n_u8(0x0F);
    let mask_hi = vdupq_n_u8(if HP == 1 { 0x10 } else { 0x30 });
    let row_start = s * SIMD_ROW_STRIPE;
    let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
    let full_stripe = valid == SIMD_ROW_STRIPE;
    let num_g = valid.div_ceil(4);

    let mut facc: [[float32x4_t; NT]; 4] = [[vdupq_n_f32(0.0); NT]; 4];
    if !first_tile {
        for (g, fg) in facc.iter_mut().enumerate().take(num_g) {
            let g_off = g * 4;
            let g_valid = (valid - g_off).min(4);
            if full_stripe || g_valid == 4 {
                *fg = std::array::from_fn(|t| unsafe {
                    vld1q_f32(out_base.add(t * out_stride + row_start + g_off))
                });
            } else {
                *fg = std::array::from_fn(|t| unsafe {
                    let mut tmp = [0.0f32; 4];
                    std::ptr::copy_nonoverlapping(
                        out_base.add(t * out_stride + row_start + g_off),
                        tmp.as_mut_ptr(),
                        g_valid,
                    );
                    vld1q_f32(tmp.as_ptr())
                });
            }
        }
    }

    for c in c_base..c_end {
        let (d_ptr, h_ptr, sc_ptr) = matrix.stripe_chunk_ptrs(s, c);
        // SAFETY: `sc_ptr` holds `32` bytes (`16` BF16 scales).
        let ws: [float32x4_t; 4] = unsafe {
            let bf0 = vreinterpretq_u16_u8(vld1q_u8(sc_ptr));
            let bf1 = vreinterpretq_u16_u8(vld1q_u8(sc_ptr.add(16)));
            [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf0))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf0)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf1))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf1)),
            ]
        };
        for (g, fg) in facc.iter_mut().enumerate().take(num_g) {
            let mut acc = [vdupq_n_s32(0); NT];
            for k in 0..4 {
                // SAFETY: `c * 64 + k * 16 + 16 <= cols`; offsets stay within `(s, c)` record.
                unsafe {
                    let blk = d_ptr.add(k * 128 + g * 32);
                    let va = vld1q_u8(blk);
                    let vb = vld1q_u8(blk.add(16));
                    let (ula, uha, ulb, uhb) =
                        unpack_sdot_quad_local::<HP>(va, vb, h_ptr, k, g, mask_0f, mask_hi);
                    let col_off = c * 64 + k * 16;
                    for t in 0..NT {
                        let x16 = vld1q_s8(i8a[t].q.as_ptr().add(col_off));
                        let mut a = acc[t];
                        a = sdot_lane::<0>(a, ula, x16);
                        a = sdot_lane::<1>(a, uha, x16);
                        a = sdot_lane::<2>(a, ulb, x16);
                        a = sdot_lane::<3>(a, uhb, x16);
                        acc[t] = a;
                    }
                }
            }
            let wsg = ws[g];
            for t in 0..NT {
                let sx = i8a[t].scales[c];
                if sx != 0.0 {
                    let dot = vsubq_s32(vshlq_n_s32::<1>(acc[t]), vdupq_n_s32(max_odd * i8a[t].sums[c]));
                    let v = vmulq_f32(vcvtq_f32_s32(dot), vdupq_n_f32(sx));
                    fg[t] = vfmaq_f32(fg[t], v, wsg);
                }
            }
        }
    }

    for (g, fg) in facc.iter().enumerate().take(num_g) {
        let g_off = g * 4;
        let g_valid = (valid - g_off).min(4);
        for (t, &ft) in fg.iter().enumerate() {
            // SAFETY: `row_start + valid <= out_stride` for token `t`.
            unsafe {
                let dst = out_base.add(t * out_stride + row_start + g_off);
                if full_stripe || g_valid == 4 {
                    vst1q_f32(dst, ft);
                } else {
                    let mut tmp = [0.0f32; 4];
                    vst1q_f32(tmp.as_mut_ptr(), ft);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, g_valid);
                }
            }
        }
    }
}

// ============================================================================
// NF5 / NF4 / NF3 Lloyd-Max codebook matrices
// ============================================================================

/// Max (1960) 32-level symmetric Gaussian positive centroids (`q_0..q_15`, negative half is `-q_{15-m}`).
pub const NF5_DEFAULT_POS_CODEBOOK_I8: [i8; 16] = [
    3, 8, 13, 19, 24, 30, 36, 42, 49, 56, 64, 72, 82, 93, 107, 127,
];

/// Max (1960) 16-level symmetric Gaussian centroids scaled to signed `INT8` `[-127, +127]`.
pub const NF4_DEFAULT_CODEBOOK_I8: [i8; 16] = [
    -127, -96, -75, -58, -44, -31, -18, -6, 6, 18, 31, 44, 58, 75, 96, 127,
];

/// Max (1960) 8-level symmetric Gaussian centroids scaled to signed `INT8` `[-127, +127]`.
pub const NF3_DEFAULT_CODEBOOK_I8: [i8; 16] = [
    -127, -79, -45, -14, 14, 45, 79, 127, 0, 0, 0, 0, 0, 0, 0, 0,
];

/// Even-coset 4D magnitude templates (`16 * 2 D_8`) for 2-bit Gosset `E8Q2` lattice quantization:
/// `0..12` have `(sum / 32)` even, `12..16` and `16..28` have `(sum / 32)` odd.
pub const E8Q2_EVEN_TEMPLATES_I8: [[i8; 4]; 28] = [
    [0, 0, 0, 0],
    [32, 32, 0, 0],
    [32, 0, 32, 0],
    [32, 0, 0, 32],
    [0, 32, 32, 0],
    [0, 32, 0, 32],
    [0, 0, 32, 32],
    [64, 0, 0, 0],
    [0, 64, 0, 0],
    [0, 0, 64, 0],
    [0, 0, 0, 64],
    [32, 32, 32, 32],
    [32, 0, 0, 0],
    [0, 32, 0, 0],
    [0, 0, 32, 0],
    [0, 0, 0, 32],
    [32, 32, 32, 0],
    [32, 32, 0, 32],
    [32, 0, 32, 32],
    [0, 32, 32, 32],
    [96, 0, 0, 0],
    [0, 96, 0, 0],
    [0, 0, 96, 0],
    [0, 0, 0, 96],
    [64, 32, 32, 32],
    [32, 64, 32, 32],
    [32, 32, 64, 32],
    [32, 32, 32, 64],
];

/// Squared Euclidean norms of [`E8Q2_EVEN_TEMPLATES_I8`].
pub const E8Q2_EVEN_NORM2: [f32; 28] = {
    let mut out = [0.0f32; 28];
    let mut i = 0;
    while i < 28 {
        let t = E8Q2_EVEN_TEMPLATES_I8[i];
        let s = (t[0] as i32) * (t[0] as i32)
            + (t[1] as i32) * (t[1] as i32)
            + (t[2] as i32) * (t[2] as i32)
            + (t[3] as i32) * (t[3] as i32);
        out[i] = s as f32;
        i += 1;
    }
    out
};

/// Transposed 32-byte lookup tables for coordinates `0..4` of [`E8Q2_EVEN_TEMPLATES_I8`].
pub const E8Q2_EVEN_COL_TABLES_I8: [[i8; 32]; 4] = {
    let mut cols = [[0i8; 32]; 4];
    let mut c = 0;
    while c < 4 {
        let mut idx = 0;
        while idx < 28 {
            cols[c][idx] = E8Q2_EVEN_TEMPLATES_I8[idx][c];
            idx += 1;
        }
        c += 1;
    }
    cols
};

/// Packed 32-byte 2-bit-per-coordinate table (`q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)` where
/// `q_t = template[t] / 32`) so one 2-register `vqtbl2q_u8` decodes all 4 even-coset magnitudes.
pub const E8Q2_EVEN_PACKED_U8: [u8; 32] = {
    let mut out = [0u8; 32];
    let mut idx = 0;
    while idx < 28 {
        let t = E8Q2_EVEN_TEMPLATES_I8[idx];
        let q0 = (t[0] as u8) >> 5;
        let q1 = (t[1] as u8) >> 5;
        let q2 = (t[2] as u8) >> 5;
        let q3 = (t[3] as u8) >> 5;
        out[idx] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
        idx += 1;
    }
    out
};

/// Decodes a 16-bit `(tpl_byte, sgn_byte)` Gosset `E8Q2` codeword into 8 signed `INT8` coordinates
/// on `16 * 2 E_8` (`[-96, +96]`). `(0, 1)` decodes to `[0; 8]`.
#[inline]
pub fn e8q2_decode_8d(tpl_byte: u8, sgn_byte: u8) -> [i8; 8] {
    let is_even = ((tpl_byte ^ sgn_byte).count_ones() & 1) != 0;
    let mut out = [0i8; 8];
    if is_even {
        let tpl_u = (tpl_byte & 0x0F) as usize;
        let tpl_v = (tpl_byte >> 4) as usize;
        let u_idx = if tpl_u < 12 && tpl_v >= 12 { 16 + tpl_u } else { tpl_u };
        let v_idx = if tpl_v < 12 && tpl_u >= 12 { 16 + tpl_v } else { tpl_v };
        let u = E8Q2_EVEN_TEMPLATES_I8[u_idx];
        let v = E8Q2_EVEN_TEMPLATES_I8[v_idx];
        for i in 0..4 {
            let mu = u[i];
            out[i] = if ((sgn_byte >> i) & 1) != 0 { -mu } else { mu };
            let mv = v[i];
            out[4 + i] = if ((sgn_byte >> (4 + i)) & 1) != 0 { -mv } else { mv };
        }
    } else {
        for (i, dst) in out.iter_mut().enumerate() {
            let mag = if ((tpl_byte >> i) & 1) != 0 { 48i8 } else { 16i8 };
            *dst = if ((sgn_byte >> i) & 1) != 0 { -mag } else { mag };
        }
    }
    out
}

/// Finds the exact global `L2`-nearest 16-bit Gosset `E8Q2` codeword to pre-scaled `y` (`w / delta`).
/// Returns `(tpl_byte, sgn_byte, decoded_i8, squared_l2_error)`.
#[inline]
pub fn e8q2_quantize_8d(y: &[f32; 8]) -> (u8, u8, [i8; 8], f32) {
    // 1. Exact Wagner decoder over all 32,768 odd-coset points (`popcount(tpl ^ sgn)` even).
    let mut tpl_odd = 0u8;
    let mut sgn_odd = 0u8;
    let mut alt_b = [0u8; 8];
    let mut alt_s = [0u8; 8];
    let mut parity = 0u8;
    let mut min_pen = f32::INFINITY;
    let mut flip_i = 0usize;
    for (i, &yi) in y.iter().enumerate() {
        let (bp, sp, dp) = if yi >= -16.0 {
            let d = yi - 16.0;
            (0u8, 0u8, d * d)
        } else {
            let d = yi + 48.0;
            (1u8, 1u8, d * d)
        };
        let (bm, sm, dm) = if yi >= 16.0 {
            let d = yi - 48.0;
            (1u8, 0u8, d * d)
        } else {
            let d = yi + 16.0;
            (0u8, 1u8, d * d)
        };
        let pen = (dp - dm).abs();
        if pen < min_pen {
            min_pen = pen;
            flip_i = i;
        }
        if dp <= dm {
            tpl_odd |= bp << i;
            sgn_odd |= sp << i;
            alt_b[i] = bm;
            alt_s[i] = sm;
        } else {
            tpl_odd |= bm << i;
            sgn_odd |= sm << i;
            alt_b[i] = bp;
            alt_s[i] = sp;
            parity ^= 1;
        }
    }
    if parity != 0 {
        tpl_odd = (tpl_odd & !(1u8 << flip_i)) | (alt_b[flip_i] << flip_i);
        sgn_odd = (sgn_odd & !(1u8 << flip_i)) | (alt_s[flip_i] << flip_i);
    }
    let q_odd = e8q2_decode_8d(tpl_odd, sgn_odd);
    let mut odd_loss = 0.0f32;
    for i in 0..8 {
        let d = y[i] - (q_odd[i] as f32);
        odd_loss += d * d;
    }

    // 2. Factored exact search over all 32,768 even-coset points (`popcount(tpl ^ sgn)` odd).
    let mut ay = [0.0f32; 8];
    let mut sgn_nat = 0u8;
    let mut min_ay = f32::INFINITY;
    let mut min_ay_i = 0usize;
    for i in 0..8 {
        let a = y[i].abs();
        ay[i] = a;
        if y[i] < 0.0 {
            sgn_nat |= 1u8 << i;
        }
        if a < min_ay {
            min_ay = a;
            min_ay_i = i;
        }
    }

    let mut eu = [0.0f32; 28];
    let mut ev = [0.0f32; 28];
    for t in 0..28 {
        let tp = E8Q2_EVEN_TEMPLATES_I8[t];
        let n2 = E8Q2_EVEN_NORM2[t];
        let du = ay[0] * (tp[0] as f32)
            + ay[1] * (tp[1] as f32)
            + ay[2] * (tp[2] as f32)
            + ay[3] * (tp[3] as f32);
        let dv = ay[4] * (tp[0] as f32)
            + ay[5] * (tp[1] as f32)
            + ay[6] * (tp[2] as f32)
            + ay[7] * (tp[3] as f32);
        eu[t] = n2 - 2.0 * du;
        ev[t] = n2 - 2.0 * dv;
    }

    let argmin_range = |arr: &[f32; 28], start: usize, end: usize| -> usize {
        let mut best_i = start;
        let mut best_v = arr[start];
        for (i, &val) in arr.iter().enumerate().take(end).skip(start + 1) {
            if val < best_v {
                best_v = val;
                best_i = i;
            }
        }
        best_i
    };

    let u0 = argmin_range(&eu, 0, 11);
    let v0 = argmin_range(&ev, 0, 11);
    let u_p2 = argmin_range(&eu, 12, 16);
    let v_p2 = argmin_range(&ev, 12, 16);
    let u_alt = argmin_range(&eu, 16, 28) - 16;
    let v_alt = argmin_range(&ev, 16, 28) - 16;

    let pen_11 = if (sgn_nat.count_ones() & 1) == 0 {
        128.0 * min_ay
    } else {
        0.0
    };
    let cands: [(usize, usize, f32); 7] = [
        (u0, v0, eu[u0] + ev[v0]),
        (11, v0, eu[11] + ev[v0]),
        (u0, 11, eu[u0] + ev[11]),
        (11, 11, eu[11] + ev[11] + pen_11),
        (u_p2, v_p2, eu[u_p2] + ev[v_p2]),
        (u_alt, v_p2, eu[16 + u_alt] + ev[v_p2]),
        (u_p2, v_alt, eu[u_p2] + ev[16 + v_alt]),
    ];
    let mut best_tu = cands[0].0;
    let mut best_tv = cands[0].1;
    let mut best_e = cands[0].2;
    for &(tu, tv, cost) in &cands[1..] {
        if cost < best_e {
            best_e = cost;
            best_tu = tu;
            best_tv = tv;
        }
    }

    let tpl_even = (best_tu as u8) | ((best_tv as u8) << 4);
    let mut sgn_even = sgn_nat;
    if ((tpl_even ^ sgn_even).count_ones() & 1) == 0 {
        if tpl_even == 0xBB {
            sgn_even ^= 1u8 << min_ay_i;
        } else {
            let u_idx = if best_tu < 12 && best_tv >= 12 { 16 + best_tu } else { best_tu };
            let v_idx = if best_tv < 12 && best_tu >= 12 { 16 + best_tv } else { best_tv };
            let tu_arr = E8Q2_EVEN_TEMPLATES_I8[u_idx];
            let tv_arr = E8Q2_EVEN_TEMPLATES_I8[v_idx];
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
            sgn_even ^= 1u8 << zero_i;
        }
    }
    let q_even = e8q2_decode_8d(tpl_even, sgn_even);
    let mut even_loss = 0.0f32;
    for i in 0..8 {
        let d = y[i] - (q_even[i] as f32);
        even_loss += d * d;
    }

    if even_loss < odd_loss {
        (tpl_even, sgn_even, q_even, even_loss)
    } else {
        (tpl_odd, sgn_odd, q_odd, odd_loss)
    }
}

/// Expands the 16-byte serialized codebook into up to 32 signed INT8 centroids (`NF5` stores the 16
/// positive centroids `q_0..q_15` in `codebook[0..16]`, mirrored to `[-q_15..-q_0, q_0..q_15]`).
#[inline]
pub fn nf_expand_codebook32(bits: usize, codebook: &[i8; 16]) -> [i8; 32] {
    let mut out = [0i8; 32];
    if bits == 5 {
        for m in 0..16 {
            out[15 - m] = -codebook[m];
            out[16 + m] = codebook[m];
        }
    } else {
        out[..16].copy_from_slice(codebook);
    }
    out
}

/// 16 KiB VM-page-aligned byte buffer so Metal `newBufferWithBytesNoCopy` maps weights zero-copy.
pub struct PageAlignedBytes {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
    alloc_bytes: usize,
}

// SAFETY: `PageAlignedBytes` owns its heap allocation exclusively like `Vec<u8>`.
unsafe impl Send for PageAlignedBytes {}
// SAFETY: Shared references only allow immutable byte slice reads.
unsafe impl Sync for PageAlignedBytes {}

impl PageAlignedBytes {
    pub const PAGE_ALIGN: usize = 16_384;

    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: std::ptr::NonNull::dangling(),
                len: 0,
                alloc_bytes: 0,
            };
        }
        let alloc_bytes = len.div_ceil(Self::PAGE_ALIGN) * Self::PAGE_ALIGN;
        let layout = std::alloc::Layout::from_size_align(alloc_bytes, Self::PAGE_ALIGN)
            .expect("valid page-aligned layout");
        // SAFETY: `alloc_bytes > 0` and `PAGE_ALIGN` is a power of two.
        let raw = unsafe { std::alloc::alloc_zeroed(layout) };
        let Some(ptr) = std::ptr::NonNull::new(raw) else {
            std::alloc::handle_alloc_error(layout);
        };
        Self {
            ptr,
            len,
            alloc_bytes,
        }
    }

    pub fn from_slice(src: &[u8]) -> Self {
        let mut out = Self::zeroed(src.len());
        out.copy_from_slice(src);
        out
    }

    #[inline]
    pub fn as_page_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr.as_ptr().cast()
    }

    #[inline]
    pub fn page_alloc_bytes(&self) -> usize {
        self.alloc_bytes
    }
}

impl Default for PageAlignedBytes {
    fn default() -> Self {
        Self::zeroed(0)
    }
}

impl Clone for PageAlignedBytes {
    fn clone(&self) -> Self {
        Self::from_slice(self)
    }
}

impl std::fmt::Debug for PageAlignedBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PageAlignedBytes")
            .field("len", &self.len)
            .field("alloc_bytes", &self.alloc_bytes)
            .finish()
    }
}

impl PartialEq for PageAlignedBytes {
    fn eq(&self, other: &Self) -> bool {
        (**self) == (**other)
    }
}

impl Eq for PageAlignedBytes {}

impl std::ops::Deref for PageAlignedBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is valid for `len` initialized bytes (`dangling` when `len == 0`).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl std::ops::DerefMut for PageAlignedBytes {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: `&mut self` guarantees exclusive access to `len` initialized bytes.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for PageAlignedBytes {
    fn drop(&mut self) {
        if self.alloc_bytes > 0 {
            let layout = std::alloc::Layout::from_size_align(self.alloc_bytes, Self::PAGE_ALIGN)
                .expect("valid page-aligned layout");
            // SAFETY: Matches the exact `Layout` allocated in `PageAlignedBytes::zeroed`.
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

/// NF5/NF4/NF3 weights: 3-, 4-, or 5-bit indices into a per-matrix INT8 codebook, expanded with
/// `vqtbl1q_s8`/`vqtbl2q_s8` and multiplied with `SDOT`. Stored as stripe pairs like [`Nibble4SdotMatrix`].
#[derive(Clone, Debug)]
pub struct NfSdotMatrix {
    pub rows: usize,
    pub cols: usize,
    /// `3` (NF3), `4` (NF4), or `5` (NF5).
    pub bits: usize,
    /// 64-weight chunks per BF16 scale (1 or 2).
    pub scale_group_chunks: usize,
    pub num_stripes: usize,
    pub num_pairs: usize,
    pub num_chunks: usize,
    pub num_scale_groups: usize,
    /// INT8 centroids (16 positive centroids for NF5, 16 signed for NF4, 8 signed for NF3).
    pub codebook: [i8; 16],
    /// `[pair, scale_group]` records: 64 B of scales, then `c_len * 2 * chunk_bytes` of data.
    pub pair_stream: PageAlignedBytes,
    /// Odd last stripe: 32 B of scales plus data per scale group; empty when `num_stripes` is even.
    pub tail_stream: PageAlignedBytes,
}

impl NfSdotMatrix {
    /// Weight bytes per stripe and chunk: 640 for NF5, 512 for NF4, 384 for NF3, 256 for E8Q2.
    #[inline]
    pub const fn stripe_chunk_bytes(bits: usize) -> usize {
        match bits {
            5 => 640,
            4 => 512,
            3 => 384,
            2 => 256,
            _ => panic!("unsupported NfSdotMatrix bits"),
        }
    }

    /// Full 32-entry signed INT8 codebook (`0..32` for NF5, `0..16` for NF4, `0..8` for NF3).
    #[inline]
    pub fn codebook32(&self) -> [i8; 32] {
        nf_expand_codebook32(self.bits, &self.codebook)
    }

    #[inline]
    pub const fn pair_total_bytes(&self) -> usize {
        self.num_scale_groups * 64 + self.num_chunks * 2 * Self::stripe_chunk_bytes(self.bits)
    }

    #[inline]
    pub const fn tail_total_bytes(&self) -> usize {
        self.num_scale_groups * 32 + self.num_chunks * Self::stripe_chunk_bytes(self.bits)
    }

    pub fn physical_bytes(&self) -> usize {
        self.pair_stream.len() + self.tail_stream.len() + self.codebook.len()
    }

    /// Packs row-major `flat_indices` (`[rows, cols]`) and stripe-major `bf16_scales`
    /// (`[num_stripes, num_scale_groups, SIMD_ROW_STRIPE]`).
    pub fn from_indices(
        flat_indices: &[u8],
        bf16_scales: &[u16],
        rows: usize,
        cols: usize,
        bits: usize,
        scale_group_chunks: usize,
        codebook: [i8; 16],
    ) -> Self {
        assert!(
            bits == 3 || bits == 4 || bits == 5,
            "NfSdotMatrix bits must be 3, 4, or 5"
        );
        assert!(rows > 0 && cols > 0 && cols.is_multiple_of(64));
        assert!(scale_group_chunks >= 1);
        assert_eq!(flat_indices.len(), rows * cols);
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let num_pairs = num_stripes / 2;
        let has_tail = (num_stripes & 1) != 0;
        let num_chunks = cols / 64;
        let num_scale_groups = num_chunks.div_ceil(scale_group_chunks);
        assert_eq!(bf16_scales.len(), num_stripes * num_scale_groups * SIMD_ROW_STRIPE);

        let chunk_b = Self::stripe_chunk_bytes(bits);
        let pair_tot = num_scale_groups * 64 + num_chunks * 2 * chunk_b;
        let tail_tot = num_scale_groups * 32 + num_chunks * chunk_b;
        let mut pair_stream = PageAlignedBytes::zeroed(num_pairs * pair_tot);
        let mut tail_stream = PageAlignedBytes::zeroed(usize::from(has_tail) * tail_tot);

        let pack_one_stripe_chunk = |stripe: usize, c: usize, dst: &mut [u8]| {
            debug_assert_eq!(dst.len(), chunk_b);
            let r_base = stripe * SIMD_ROW_STRIPE;
            let c_col = c * 64;
            let idx_at = |lane: usize, sub_col: usize| -> u8 {
                let r = r_base + lane;
                if r < rows {
                    flat_indices[r * cols + c_col + sub_col]
                } else {
                    0
                }
            };
            if bits == 4 || bits == 5 {
                for k in 0..4 {
                    for g in 0..4 {
                        for j in 0..4 {
                            let lane = g * 4 + j;
                            for t in 0..4 {
                                let lo0 = idx_at(lane, k * 16 + t) & 0x0F;
                                let hi0 = idx_at(lane, k * 16 + 4 + t) & 0x0F;
                                dst[k * 128 + g * 32 + j * 4 + t] = lo0 | (hi0 << 4);

                                let lo1 = idx_at(lane, k * 16 + 8 + t) & 0x0F;
                                let hi1 = idx_at(lane, k * 16 + 12 + t) & 0x0F;
                                dst[k * 128 + g * 32 + 16 + j * 4 + t] = lo1 | (hi1 << 4);
                            }
                        }
                    }
                }
                if bits == 5 {
                    for g in 0..4 {
                        for kp in 0..2 {
                            let k0 = kp * 2;
                            let k1 = k0 + 1;
                            let blk = &mut dst[512 + g * 32 + kp * 16..512 + g * 32 + (kp + 1) * 16];
                            for j in 0..4 {
                                let lane = g * 4 + j;
                                for t in 0..4 {
                                    let b = j * 4 + t;
                                    let e0 = ((idx_at(lane, k0 * 16 + t) >> 4) & 1) << 4;
                                    let e1 = ((idx_at(lane, k0 * 16 + 4 + t) >> 4) & 1) << 5;
                                    let e2 = ((idx_at(lane, k0 * 16 + 8 + t) >> 4) & 1) << 6;
                                    let e3 = ((idx_at(lane, k0 * 16 + 12 + t) >> 4) & 1) << 7;
                                    let o0 = (idx_at(lane, k1 * 16 + t) >> 4) & 1;
                                    let o1 = ((idx_at(lane, k1 * 16 + 4 + t) >> 4) & 1) << 1;
                                    let o2 = ((idx_at(lane, k1 * 16 + 8 + t) >> 4) & 1) << 2;
                                    let o3 = ((idx_at(lane, k1 * 16 + 12 + t) >> 4) & 1) << 3;
                                    blk[b] = e0 | e1 | e2 | e3 | o0 | o1 | o2 | o3;
                                }
                            }
                        }
                    }
                }
            } else {
                // NF3: 128 3-bit indices pack into 48 bytes (`v0, v1, v2`).
                for kp in 0..2 {
                    let k0 = kp * 2;
                    let k1 = k0 + 1;
                    for g in 0..4 {
                        let blk = &mut dst[kp * 192 + g * 48..kp * 192 + (g + 1) * 48];
                        for j in 0..4 {
                            let lane = g * 4 + j;
                            for t in 0..4 {
                                let b = j * 4 + t;
                                let i0 = idx_at(lane, k0 * 16 + t) & 7;
                                let i1 = idx_at(lane, k0 * 16 + 4 + t) & 7;
                                let i2 = idx_at(lane, k0 * 16 + 8 + t) & 7;
                                let i3 = idx_at(lane, k0 * 16 + 12 + t) & 7;
                                let i4 = idx_at(lane, k1 * 16 + t) & 7;
                                let i5 = idx_at(lane, k1 * 16 + 4 + t) & 7;
                                let i6 = idx_at(lane, k1 * 16 + 8 + t) & 7;
                                let i7 = idx_at(lane, k1 * 16 + 12 + t) & 7;

                                blk[b] = i0 | (i1 << 3) | ((i2 & 3) << 6);
                                blk[16 + b] = i3 | (i4 << 3) | ((i5 & 3) << 6);
                                blk[32 + b] =
                                    i6 | (i7 << 3) | (((i2 >> 2) & 1) << 6) | (((i5 >> 2) & 1) << 7);
                            }
                        }
                    }
                }
            }
        };

        for p in 0..num_pairs {
            let sa = 2 * p;
            let sb = sa + 1;
            let mut cursor = p * pair_tot;
            for sg in 0..num_scale_groups {
                let sca = &bf16_scales[(sa * num_scale_groups + sg) * SIMD_ROW_STRIPE..][..SIMD_ROW_STRIPE];
                let scb = &bf16_scales[(sb * num_scale_groups + sg) * SIMD_ROW_STRIPE..][..SIMD_ROW_STRIPE];
                for (lane, (&ua, &ub)) in sca.iter().zip(scb.iter()).enumerate() {
                    pair_stream[cursor + lane * 2..cursor + lane * 2 + 2].copy_from_slice(&ua.to_le_bytes());
                    pair_stream[cursor + 32 + lane * 2..cursor + 32 + lane * 2 + 2]
                        .copy_from_slice(&ub.to_le_bytes());
                }
                cursor += 64;
                let c_start = sg * scale_group_chunks;
                let c_end = (c_start + scale_group_chunks).min(num_chunks);
                for c in c_start..c_end {
                    pack_one_stripe_chunk(sa, c, &mut pair_stream[cursor..cursor + chunk_b]);
                    pack_one_stripe_chunk(
                        sb,
                        c,
                        &mut pair_stream[cursor + chunk_b..cursor + 2 * chunk_b],
                    );
                    cursor += 2 * chunk_b;
                }
            }
            debug_assert_eq!(cursor, (p + 1) * pair_tot);
        }

        if has_tail {
            let st = num_pairs * 2;
            let mut cursor = 0usize;
            for sg in 0..num_scale_groups {
                let sct = &bf16_scales[(st * num_scale_groups + sg) * SIMD_ROW_STRIPE..][..SIMD_ROW_STRIPE];
                for (lane, &ut) in sct.iter().enumerate() {
                    tail_stream[cursor + lane * 2..cursor + lane * 2 + 2].copy_from_slice(&ut.to_le_bytes());
                }
                cursor += 32;
                let c_start = sg * scale_group_chunks;
                let c_end = (c_start + scale_group_chunks).min(num_chunks);
                for c in c_start..c_end {
                    pack_one_stripe_chunk(st, c, &mut tail_stream[cursor..cursor + chunk_b]);
                    cursor += chunk_b;
                }
            }
            debug_assert_eq!(cursor, tail_tot);
        }

        Self {
            rows,
            cols,
            bits,
            scale_group_chunks,
            num_stripes,
            num_pairs,
            num_chunks,
            num_scale_groups,
            codebook,
            pair_stream,
            tail_stream,
        }
    }

    /// Allocates a deterministic synthetic `NfSdotMatrix` (`bits` in `2..=5`) with finite BF16
    /// group scales and pseudo-random packed weights in O(bytes) time for hardware envelope benchmarks.
    pub fn new_synthetic(
        rows: usize,
        cols: usize,
        bits: usize,
        scale_group_chunks: usize,
        seed: u64,
    ) -> Self {
        assert!((2..=5).contains(&bits));
        assert!(rows > 0 && cols > 0 && cols.is_multiple_of(64));
        let scale_group_chunks = scale_group_chunks.max(1);
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let num_pairs = num_stripes / 2;
        let has_tail = (num_stripes & 1) != 0;
        let num_chunks = cols / 64;
        let num_scale_groups = num_chunks.div_ceil(scale_group_chunks);
        let chunk_b = Self::stripe_chunk_bytes(bits);
        let pair_tot = num_scale_groups * 64 + num_chunks * 2 * chunk_b;
        let tail_tot = num_scale_groups * 32 + num_chunks * chunk_b;
        let mut pair_stream = PageAlignedBytes::zeroed(num_pairs * pair_tot);
        let mut tail_stream = PageAlignedBytes::zeroed(usize::from(has_tail) * tail_tot);

        let codebook: [i8; 16] = match bits {
            2 => [0; 16],
            3 => NF3_DEFAULT_CODEBOOK_I8,
            5 => NF5_DEFAULT_POS_CODEBOOK_I8,
            _ => NF4_DEFAULT_CODEBOOK_I8,
        };

        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next_u64 = || -> u64 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };

        for p in 0..num_pairs {
            let mut cursor = p * pair_tot;
            for sg in 0..num_scale_groups {
                for lane in 0..(2 * SIMD_ROW_STRIPE) {
                    let bf16: u16 = 0x3B00 + (((p + sg + lane) as u16 * 13) & 0x007F);
                    pair_stream[cursor + lane * 2..cursor + lane * 2 + 2]
                        .copy_from_slice(&bf16.to_le_bytes());
                }
                cursor += 64;
                let c_start = sg * scale_group_chunks;
                let c_end = (c_start + scale_group_chunks).min(num_chunks);
                let data_len = (c_end - c_start) * 2 * chunk_b;
                for word in pair_stream[cursor..cursor + data_len].chunks_exact_mut(8) {
                    word.copy_from_slice(&next_u64().to_le_bytes());
                }
                cursor += data_len;
            }
        }

        if has_tail {
            let mut cursor = 0usize;
            for sg in 0..num_scale_groups {
                for lane in 0..SIMD_ROW_STRIPE {
                    let bf16: u16 = 0x3B00 + (((sg + lane) as u16 * 17) & 0x007F);
                    tail_stream[cursor + lane * 2..cursor + lane * 2 + 2]
                        .copy_from_slice(&bf16.to_le_bytes());
                }
                cursor += 32;
                let c_start = sg * scale_group_chunks;
                let c_end = (c_start + scale_group_chunks).min(num_chunks);
                let data_len = (c_end - c_start) * chunk_b;
                for word in tail_stream[cursor..cursor + data_len].chunks_exact_mut(8) {
                    word.copy_from_slice(&next_u64().to_le_bytes());
                }
                cursor += data_len;
            }
        }

        Self {
            rows,
            cols,
            bits,
            scale_group_chunks,
            num_stripes,
            num_pairs,
            num_chunks,
            num_scale_groups,
            codebook,
            pair_stream,
            tail_stream,
        }
    }

    /// Packs row-major 8D Gosset `E8Q2` `(tpl_codes, sgn_codes)` (`[rows, cols / 8]` each) and
    /// stripe-major `bf16_scales` (`[num_stripes, num_scale_groups, SIMD_ROW_STRIPE]`) into `bits = 2`.
    pub fn from_e8q2_codes(
        tpl_codes: &[u8],
        sgn_codes: &[u8],
        bf16_scales: &[u16],
        rows: usize,
        cols: usize,
        scale_group_chunks: usize,
    ) -> Self {
        assert!(rows > 0 && cols > 0 && cols.is_multiple_of(64));
        assert!(scale_group_chunks >= 1);
        let blocks_per_row = cols / 8;
        assert_eq!(tpl_codes.len(), rows * blocks_per_row);
        assert_eq!(sgn_codes.len(), rows * blocks_per_row);
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let num_pairs = num_stripes / 2;
        let has_tail = (num_stripes & 1) != 0;
        let num_chunks = cols / 64;
        let num_scale_groups = num_chunks.div_ceil(scale_group_chunks);
        assert_eq!(bf16_scales.len(), num_stripes * num_scale_groups * SIMD_ROW_STRIPE);

        let chunk_b = Self::stripe_chunk_bytes(2);
        let pair_tot = num_scale_groups * 64 + num_chunks * 2 * chunk_b;
        let tail_tot = num_scale_groups * 32 + num_chunks * chunk_b;
        let mut pair_stream = PageAlignedBytes::zeroed(num_pairs * pair_tot);
        let mut tail_stream = PageAlignedBytes::zeroed(usize::from(has_tail) * tail_tot);

        let pack_one_chunk = |stripe: usize, c: usize, dst: &mut [u8]| {
            debug_assert_eq!(dst.len(), chunk_b);
            let r_base = stripe * SIMD_ROW_STRIPE;
            let eb_base = c * 8;
            for kp in 0..2 {
                for g in 0..4 {
                    let blk = &mut dst[kp * 128 + g * 32..kp * 128 + (g + 1) * 32];
                    for eb in 0..4 {
                        let eb_col = eb_base + kp * 4 + eb;
                        for j in 0..4 {
                            let lane = g * 4 + j;
                            let r = r_base + lane;
                            let (tpl, sgn) = if r < rows {
                                let idx = r * blocks_per_row + eb_col;
                                (tpl_codes[idx], sgn_codes[idx])
                            } else {
                                (0u8, 1u8)
                            };
                            blk[eb * 4 + j] = tpl;
                            blk[16 + eb * 4 + j] = sgn;
                        }
                    }
                }
            }
        };

        for p in 0..num_pairs {
            let sa = 2 * p;
            let sb = sa + 1;
            let mut cursor = p * pair_tot;
            for sg in 0..num_scale_groups {
                let sca = &bf16_scales[(sa * num_scale_groups + sg) * SIMD_ROW_STRIPE..][..SIMD_ROW_STRIPE];
                let scb = &bf16_scales[(sb * num_scale_groups + sg) * SIMD_ROW_STRIPE..][..SIMD_ROW_STRIPE];
                for (lane, (&ua, &ub)) in sca.iter().zip(scb.iter()).enumerate() {
                    pair_stream[cursor + lane * 2..cursor + lane * 2 + 2].copy_from_slice(&ua.to_le_bytes());
                    pair_stream[cursor + 32 + lane * 2..cursor + 32 + lane * 2 + 2]
                        .copy_from_slice(&ub.to_le_bytes());
                }
                cursor += 64;
                let c_start = sg * scale_group_chunks;
                let c_end = (c_start + scale_group_chunks).min(num_chunks);
                for c in c_start..c_end {
                    pack_one_chunk(sa, c, &mut pair_stream[cursor..cursor + chunk_b]);
                    pack_one_chunk(sb, c, &mut pair_stream[cursor + chunk_b..cursor + 2 * chunk_b]);
                    cursor += 2 * chunk_b;
                }
            }
            debug_assert_eq!(cursor, (p + 1) * pair_tot);
        }

        if has_tail {
            let st = num_pairs * 2;
            let mut cursor = 0usize;
            for sg in 0..num_scale_groups {
                let sct = &bf16_scales[(st * num_scale_groups + sg) * SIMD_ROW_STRIPE..][..SIMD_ROW_STRIPE];
                for (lane, &ut) in sct.iter().enumerate() {
                    tail_stream[cursor + lane * 2..cursor + lane * 2 + 2].copy_from_slice(&ut.to_le_bytes());
                }
                cursor += 32;
                let c_start = sg * scale_group_chunks;
                let c_end = (c_start + scale_group_chunks).min(num_chunks);
                for c in c_start..c_end {
                    pack_one_chunk(st, c, &mut tail_stream[cursor..cursor + chunk_b]);
                    cursor += chunk_b;
                }
            }
            debug_assert_eq!(cursor, tail_tot);
        }

        Self {
            rows,
            cols,
            bits: 2,
            scale_group_chunks,
            num_stripes,
            num_pairs,
            num_chunks,
            num_scale_groups,
            codebook: [0i8; 16],
            pair_stream,
            tail_stream,
        }
    }

    /// Stream-quantizes a `[rows, cols]` little-endian `BF16` embedding table into 4-bit Hadamard `NF4`
    /// using bounded 32-row thread-local scratch (`< 6 MiB` total temporary RAM on `152k x 5120`).
    pub fn from_bf16_le_embeddings_nf4(
        bf16_le_bytes: &[u8],
        rows: usize,
        cols: usize,
        scale_group_chunks: usize,
        num_threads: usize,
    ) -> Self {
        assert_eq!(bf16_le_bytes.len(), rows * cols * 2);
        Self::build_nf4_from_bf16_getter(rows, cols, scale_group_chunks, num_threads, |r, c| {
            let off = (r * cols + c) * 2;
            u16::from_le_bytes([bf16_le_bytes[off], bf16_le_bytes[off + 1]])
        })
    }

    /// Stream-quantizes a `[rows, cols]` `BF16` (`u16`) embedding table into 4-bit Hadamard `NF4`.
    pub fn from_bf16_u16_embeddings_nf4(
        bf16_words: &[u16],
        rows: usize,
        cols: usize,
        scale_group_chunks: usize,
        num_threads: usize,
    ) -> Self {
        assert_eq!(bf16_words.len(), rows * cols);
        Self::build_nf4_from_bf16_getter(rows, cols, scale_group_chunks, num_threads, |r, c| {
            bf16_words[r * cols + c]
        })
    }

    fn build_nf4_from_bf16_getter<F>(
        rows: usize,
        cols: usize,
        scale_group_chunks: usize,
        num_threads: usize,
        get_bf16: F,
    ) -> Self
    where
        F: Fn(usize, usize) -> u16 + Sync,
    {
        assert!(rows > 0 && cols > 0 && cols.is_multiple_of(64));
        let sgc = scale_group_chunks.max(1);
        let codebook = NF4_DEFAULT_CODEBOOK_I8;
        let cb32 = nf_expand_codebook32(4, &codebook);
        let mut pos_th = [0.0f32; 7];
        for m in 0..7 {
            pos_th[m] = 0.5 * ((cb32[8 + m] as f32) + (cb32[9 + m] as f32));
        }
        let cb_top = (cb32[15] as f32).max(1.0);

        let quant4 = |w: f32, inv_delta: f32| -> u8 {
            let aw = w.abs() * inv_delta;
            let mag = if aw < pos_th[3] {
                if aw < pos_th[1] {
                    usize::from(aw >= pos_th[0])
                } else {
                    2 + usize::from(aw >= pos_th[2])
                }
            } else if aw < pos_th[5] {
                4 + usize::from(aw >= pos_th[4])
            } else {
                6 + usize::from(aw >= pos_th[6])
            };
            if w >= 0.0 {
                (8 + mag) as u8
            } else {
                (7 - mag) as u8
            }
        };

        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let num_pairs = num_stripes / 2;
        let has_tail = (num_stripes & 1) != 0;
        let num_chunks = cols / 64;
        let num_scale_groups = num_chunks.div_ceil(sgc);
        let chunk_b = Self::stripe_chunk_bytes(4);
        let pair_tot = num_scale_groups * 64 + num_chunks * 2 * chunk_b;
        let tail_tot = num_scale_groups * 32 + num_chunks * chunk_b;
        let mut pair_stream = PageAlignedBytes::zeroed(num_pairs * pair_tot);
        let mut tail_stream = PageAlignedBytes::zeroed(usize::from(has_tail) * tail_tot);

        let quantize_one_row = |global_r: usize,
                                row_buf: &mut [f32],
                                idx_dst: &mut [u8],
                                sc_dst: &mut [u16],
                                lane: usize| {
            if global_r >= rows {
                idx_dst.fill(8);
                for sg in 0..num_scale_groups {
                    sc_dst[sg * SIMD_ROW_STRIPE + lane] = 0;
                }
                return;
            }
            for (c, dst) in row_buf.iter_mut().enumerate() {
                *dst = bf16_to_f32(get_bf16(global_r, c));
            }
            crate::kernels::hadamard::fwht64_signed_inplace(row_buf);
            for sg in 0..num_scale_groups {
                let col0 = sg * sgc * 64;
                let col1 = (col0 + sgc * 64).min(cols);
                let grp = &row_buf[col0..col1];
                let mut max_abs = 0.0f32;
                for &v in grp {
                    if v.abs() > max_abs {
                        max_abs = v.abs();
                    }
                }
                if max_abs <= 1e-12 {
                    idx_dst[col0..col1].fill(8);
                    sc_dst[sg * SIMD_ROW_STRIPE + lane] = 0;
                    continue;
                }
                let inv0 = cb_top / max_abs;
                let mut dot_wq = 0.0f32;
                let mut dot_qq = 0.0f32;
                for &v in grp {
                    let q = quant4(v, inv0) as usize;
                    let qf = cb32[q] as f32;
                    dot_wq += v * qf;
                    dot_qq += qf * qf;
                }
                let delta_bf16 = f32_to_bf16(if dot_qq > 0.0 {
                    (dot_wq / dot_qq).max(0.0)
                } else {
                    max_abs / cb_top
                });
                let delta = bf16_to_f32(delta_bf16);
                let inv = if delta > 1e-12 { 1.0 / delta } else { 0.0 };
                for (sub, &v) in grp.iter().enumerate() {
                    idx_dst[col0 + sub] = quant4(v, inv);
                }
                sc_dst[sg * SIMD_ROW_STRIPE + lane] = delta_bf16;
            }
        };

        let pack_nibble_chunk = |idx_stripe: &[u8], c: usize, dst: &mut [u8]| {
            let c_col = c * 64;
            for k in 0..4 {
                for g in 0..4 {
                    for j in 0..4 {
                        let lane = g * 4 + j;
                        let row_off = lane * cols + c_col;
                        for t in 0..4 {
                            let lo0 = idx_stripe[row_off + k * 16 + t] & 0x0F;
                            let hi0 = idx_stripe[row_off + k * 16 + 4 + t] & 0x0F;
                            dst[k * 128 + g * 32 + j * 4 + t] = lo0 | (hi0 << 4);
                            let lo1 = idx_stripe[row_off + k * 16 + 8 + t] & 0x0F;
                            let hi1 = idx_stripe[row_off + k * 16 + 12 + t] & 0x0F;
                            dst[k * 128 + g * 32 + 16 + j * 4 + t] = lo1 | (hi1 << 4);
                        }
                    }
                }
            }
        };

        if num_pairs > 0 {
            let workers = num_threads.max(1).min(num_pairs);
            let pairs_per_worker = num_pairs.div_ceil(workers);
            std::thread::scope(|s| {
                let mut rem = &mut pair_stream[..];
                let mut p_start = 0usize;
                while p_start < num_pairs && !rem.is_empty() {
                    let take = pairs_per_worker.min(num_pairs - p_start);
                    let (chunk_bytes, next) = rem.split_at_mut(take * pair_tot);
                    rem = next;
                    let p0 = p_start;
                    p_start += take;
                    let q_row = &quantize_one_row;
                    let p_chunk = &pack_nibble_chunk;
                    s.spawn(move || {
                        let mut row_buf = vec![0.0f32; cols];
                        let mut idx_a = vec![8u8; SIMD_ROW_STRIPE * cols];
                        let mut idx_b = vec![8u8; SIMD_ROW_STRIPE * cols];
                        let mut sc_a = vec![0u16; num_scale_groups * SIMD_ROW_STRIPE];
                        let mut sc_b = vec![0u16; num_scale_groups * SIMD_ROW_STRIPE];
                        for local_p in 0..take {
                            let p = p0 + local_p;
                            let r_a = p * 2 * SIMD_ROW_STRIPE;
                            let r_b = r_a + SIMD_ROW_STRIPE;
                            for lane in 0..SIMD_ROW_STRIPE {
                                q_row(
                                    r_a + lane,
                                    &mut row_buf,
                                    &mut idx_a[lane * cols..(lane + 1) * cols],
                                    &mut sc_a,
                                    lane,
                                );
                                q_row(
                                    r_b + lane,
                                    &mut row_buf,
                                    &mut idx_b[lane * cols..(lane + 1) * cols],
                                    &mut sc_b,
                                    lane,
                                );
                            }
                            let dst_pair = &mut chunk_bytes[local_p * pair_tot..(local_p + 1) * pair_tot];
                            let mut cur = 0usize;
                            for sg in 0..num_scale_groups {
                                for lane in 0..SIMD_ROW_STRIPE {
                                    let ua = sc_a[sg * SIMD_ROW_STRIPE + lane];
                                    let ub = sc_b[sg * SIMD_ROW_STRIPE + lane];
                                    dst_pair[cur + lane * 2..cur + lane * 2 + 2]
                                        .copy_from_slice(&ua.to_le_bytes());
                                    dst_pair[cur + 32 + lane * 2..cur + 32 + lane * 2 + 2]
                                        .copy_from_slice(&ub.to_le_bytes());
                                }
                                cur += 64;
                                let c_start = sg * sgc;
                                let c_end = (c_start + sgc).min(num_chunks);
                                for c in c_start..c_end {
                                    p_chunk(&idx_a, c, &mut dst_pair[cur..cur + chunk_b]);
                                    p_chunk(
                                        &idx_b,
                                        c,
                                        &mut dst_pair[cur + chunk_b..cur + 2 * chunk_b],
                                    );
                                    cur += 2 * chunk_b;
                                }
                            }
                        }
                    });
                }
            });
        }

        if has_tail {
            let r_t = num_pairs * 2 * SIMD_ROW_STRIPE;
            let mut row_buf = vec![0.0f32; cols];
            let mut idx_t = vec![8u8; SIMD_ROW_STRIPE * cols];
            let mut sc_t = vec![0u16; num_scale_groups * SIMD_ROW_STRIPE];
            for lane in 0..SIMD_ROW_STRIPE {
                quantize_one_row(
                    r_t + lane,
                    &mut row_buf,
                    &mut idx_t[lane * cols..(lane + 1) * cols],
                    &mut sc_t,
                    lane,
                );
            }
            let mut cur = 0usize;
            for sg in 0..num_scale_groups {
                for lane in 0..SIMD_ROW_STRIPE {
                    let ut = sc_t[sg * SIMD_ROW_STRIPE + lane];
                    tail_stream[cur + lane * 2..cur + lane * 2 + 2]
                        .copy_from_slice(&ut.to_le_bytes());
                }
                cur += 32;
                let c_start = sg * sgc;
                let c_end = (c_start + sgc).min(num_chunks);
                for c in c_start..c_end {
                    pack_nibble_chunk(&idx_t, c, &mut tail_stream[cur..cur + chunk_b]);
                    cur += chunk_b;
                }
            }
        }

        Self {
            rows,
            cols,
            bits: 4,
            scale_group_chunks: sgc,
            num_stripes,
            num_pairs,
            num_chunks,
            num_scale_groups,
            codebook,
            pair_stream,
            tail_stream,
        }
    }

    /// Reorders `gate_up_proj` rows from `[gate; up]` into 64-row `[gate_64(ic), up_64(ic)]` blocks.
    pub fn to_gate_up_interleaved(&self) -> Self {
        assert!(
            self.rows.is_multiple_of(128) && self.tail_stream.is_empty(),
            "gate_up interleaving requires rows divisible by 128"
        );
        let nc_i = self.rows / 128;
        let block64_bytes = 2 * self.pair_total_bytes();
        debug_assert_eq!(self.pair_stream.len(), 2 * nc_i * block64_bytes);
        let mut new_pairs = PageAlignedBytes::zeroed(self.pair_stream.len());
        for ic in 0..nc_i {
            let g_src = ic * block64_bytes;
            let u_src = (nc_i + ic) * block64_bytes;
            let dst = (ic * 2) * block64_bytes;
            new_pairs[dst..dst + block64_bytes]
                .copy_from_slice(&self.pair_stream[g_src..g_src + block64_bytes]);
            new_pairs[dst + block64_bytes..dst + 2 * block64_bytes]
                .copy_from_slice(&self.pair_stream[u_src..u_src + block64_bytes]);
        }
        Self {
            pair_stream: new_pairs,
            tail_stream: self.tail_stream.clone(),
            ..*self
        }
    }

    /// `(data, BF16 scales)` pointers for `(stripe, chunk)`.
    #[inline(always)]
    fn stripe_chunk_ptrs(&self, stripe: usize, c: usize) -> (*const u8, *const u8) {
        debug_assert!(stripe < self.num_stripes && c < self.num_chunks);
        let sgc = self.scale_group_chunks;
        let sg = c / sgc;
        let sub_c = c % sgc;
        let c_start = sg * sgc;
        let chunk_b = Self::stripe_chunk_bytes(self.bits);
        if stripe < self.num_pairs * 2 {
            let p = stripe / 2;
            let side = stripe & 1;
            let pair_tot = self.pair_total_bytes();
            let sg_off = p * pair_tot + sg * 64 + c_start * (2 * chunk_b);
            // SAFETY: record offsets stay inside `pair_stream`.
            unsafe {
                let sg_ptr = self.pair_stream.as_ptr().add(sg_off);
                let sc_ptr = sg_ptr.add(side * 32);
                let d_ptr = sg_ptr.add(64 + sub_c * (2 * chunk_b) + side * chunk_b);
                (d_ptr, sc_ptr)
            }
        } else {
            let sg_off = sg * 32 + c_start * chunk_b;
            // SAFETY: odd tail stripe implies `tail_stream` holds `tail_total_bytes()`.
            unsafe {
                let sg_ptr = self.tail_stream.as_ptr().add(sg_off);
                let d_ptr = sg_ptr.add(32 + sub_c * chunk_b);
                (d_ptr, sg_ptr)
            }
        }
    }

    /// Dequantizes row `r` in the Hadamard-rotated domain (used for tied embeddings).
    pub fn dequantize_row_rotated(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows);
        assert_eq!(out.len(), self.cols);
        let (stripe, lane) = (r / SIMD_ROW_STRIPE, r % SIMD_ROW_STRIPE);
        let (g, j) = (lane / 4, lane % 4);
        let cb32 = self.codebook32();
        for c in 0..self.num_chunks {
            let (d_ptr, sc_ptr) = self.stripe_chunk_ptrs(stripe, c);
            // SAFETY: `sc_ptr` has 32 B, `d_ptr` 640 B (NF5), 512 B (NF4), or 384 B (NF3).
            let delta = unsafe {
                let b0 = *sc_ptr.add(lane * 2);
                let b1 = *sc_ptr.add(lane * 2 + 1);
                bf16_to_f32(u16::from_le_bytes([b0, b1]))
            };
            let dst_c = &mut out[c * 64..(c + 1) * 64];
            if self.bits == 4 || self.bits == 5 {
                for (sub_col, dst) in dst_c.iter_mut().enumerate() {
                    let (k, w) = (sub_col / 16, sub_col % 16);
                    let sub4 = w / 4;
                    let t = w % 4;
                    let b = j * 4 + t;
                    let byte = unsafe { *d_ptr.add(k * 128 + g * 32 + (sub4 / 2) * 16 + b) };
                    let lo4 = if (sub4 & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                    let idx = if self.bits == 5 {
                        let hi_byte = unsafe { *d_ptr.add(512 + g * 32 + (k / 2) * 16 + b) };
                        let shift = if (k & 1) == 0 { 4 + sub4 } else { sub4 };
                        (lo4 | (((hi_byte >> shift) & 1) << 4)) as usize
                    } else {
                        lo4 as usize
                    };
                    *dst = (cb32[idx] as f32) * delta;
                }
            } else if self.bits == 3 {
                for kp in 0..2 {
                    let k0 = kp * 2;
                    let k1 = k0 + 1;
                    for t in 0..4 {
                        let b = j * 4 + t;
                        let (v0, v1, v2) = unsafe {
                            let blk = d_ptr.add(kp * 192 + g * 48);
                            (*blk.add(b), *blk.add(16 + b), *blk.add(32 + b))
                        };
                        let idxs = [
                            (v0 & 7) as usize,
                            ((v0 >> 3) & 7) as usize,
                            (((v0 >> 6) & 3) | (((v2 >> 6) & 1) << 2)) as usize,
                            (v1 & 7) as usize,
                            ((v1 >> 3) & 7) as usize,
                            (((v1 >> 6) & 3) | (((v2 >> 7) & 1) << 2)) as usize,
                            (v2 & 7) as usize,
                            ((v2 >> 3) & 7) as usize,
                        ];
                        dst_c[k0 * 16 + t] = (cb32[idxs[0]] as f32) * delta;
                        dst_c[k0 * 16 + 4 + t] = (cb32[idxs[1]] as f32) * delta;
                        dst_c[k0 * 16 + 8 + t] = (cb32[idxs[2]] as f32) * delta;
                        dst_c[k0 * 16 + 12 + t] = (cb32[idxs[3]] as f32) * delta;
                        dst_c[k1 * 16 + t] = (cb32[idxs[4]] as f32) * delta;
                        dst_c[k1 * 16 + 4 + t] = (cb32[idxs[5]] as f32) * delta;
                        dst_c[k1 * 16 + 8 + t] = (cb32[idxs[6]] as f32) * delta;
                        dst_c[k1 * 16 + 12 + t] = (cb32[idxs[7]] as f32) * delta;
                    }
                }
            } else {
                for kp in 0..2 {
                    for eb in 0..4 {
                        let off = kp * 128 + g * 32 + eb * 4 + j;
                        let (tpl, sgn) = unsafe { (*d_ptr.add(off), *d_ptr.add(off + 16)) };
                        let q8 = e8q2_decode_8d(tpl, sgn);
                        let col0 = kp * 32 + eb * 8;
                        for (sub, &qv) in q8.iter().enumerate() {
                            dst_c[col0 + sub] = (qv as f32) * delta;
                        }
                    }
                }
            }
        }
    }
}

/// Runs stripes `[stripe_start, stripe_end)` of an [`NfSdotMatrix`] from a [`BipolarActivationLut`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn nf_sdot_stripe_range(
    matrix: &NfSdotMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, blut.cols);
    let act = blut.int8.as_ref().expect("int8 activations required for NfSdotMatrix");
    // SAFETY: Caller upholds bounds.
    unsafe { nf_sdot_stripe_range_int8(matrix, act, stripe_start, stripe_end, out_base) };
}

/// `SDOT` execution of stripes `[stripe_start, stripe_end)` of an [`NfSdotMatrix`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn nf_sdot_stripe_range_int8(
    matrix: &NfSdotMatrix,
    act: &Int8ChunkActivation,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, act.cols);
    // SAFETY: `act` buffers have lengths `cols` and `cols/64`.
    unsafe {
        nf_sdot_stripe_range_raw(
            matrix,
            act.q.as_ptr(),
            act.scales.as_ptr(),
            stripe_start,
            stripe_end,
            out_base,
        );
    }
}

/// Raw-pointer `SDOT` execution of stripes `[stripe_start, stripe_end)` of an [`NfSdotMatrix`].
///
/// # Safety
/// `xq` valid for `matrix.cols` bytes, `xscales` valid for `matrix.num_chunks` floats,
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn nf_sdot_stripe_range_raw(
    matrix: &NfSdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert!(stripe_end <= matrix.num_stripes);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON `vqtbl1q_s8`/`vqtbl2q_s8` and `sdot` are available; caller upholds bounds.
        unsafe {
            match matrix.bits {
                5 => nf_sdot_decode_neon::<5>(matrix, xq, xscales, stripe_start, stripe_end, out_base),
                4 => nf_sdot_decode_neon::<4>(matrix, xq, xscales, stripe_start, stripe_end, out_base),
                3 => nf_sdot_decode_neon::<3>(matrix, xq, xscales, stripe_start, stripe_end, out_base),
                _ => nf_sdot_decode_neon::<2>(matrix, xq, xscales, stripe_start, stripe_end, out_base),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // SAFETY: Caller upholds bounds.
        unsafe { nf_sdot_stripe_range_scalar_raw(matrix, xq, xscales, stripe_start, stripe_end, out_base) };
    }
}

/// Portable scalar reference for [`nf_sdot_stripe_range_raw`].
///
/// # Safety
/// Same preconditions as [`nf_sdot_stripe_range_raw`].
pub unsafe fn nf_sdot_stripe_range_scalar_raw(
    matrix: &NfSdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    let nc = matrix.num_chunks;
    // SAFETY: Caller guarantees `xq` (`cols`) and `xscales` (`nc`) bounds.
    let q_s = unsafe { std::slice::from_raw_parts(xq, matrix.cols) };
    let sc_s = unsafe { std::slice::from_raw_parts(xscales, nc) };
    let mut row = vec![0.0f32; matrix.cols];
    for s in stripe_start..stripe_end {
        for lane in 0..SIMD_ROW_STRIPE {
            let r = s * SIMD_ROW_STRIPE + lane;
            if r >= matrix.rows {
                break;
            }
            matrix.dequantize_row_rotated(r, &mut row);
            let mut acc = 0.0f32;
            for (c, (wc, qc)) in row.chunks_exact(64).zip(q_s.chunks_exact(64)).enumerate() {
                let dot: f32 = wc.iter().zip(qc.iter()).map(|(&w, &q)| w * (q as f32)).sum();
                acc += dot * sc_s[c];
            }
            // SAFETY: `r < matrix.rows` within disjoint stripe range.
            unsafe { *out_base.add(r) = acc };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn nf4_accum_16cols(
    mut acc: std::arch::aarch64::int32x4_t,
    va: std::arch::aarch64::uint8x16_t,
    vb: std::arch::aarch64::uint8x16_t,
    x16: std::arch::aarch64::int8x16_t,
    cb: std::arch::aarch64::int8x16_t,
    mask_0f: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: NEON `vqtbl1q_s8` and `sdot` are available on Apple Silicon.
    unsafe {
        let la = vqtbl1q_s8(cb, vandq_u8(va, mask_0f));
        let ha = vqtbl1q_s8(cb, vshrq_n_u8::<4>(va));
        let lb = vqtbl1q_s8(cb, vandq_u8(vb, mask_0f));
        let hb = vqtbl1q_s8(cb, vshrq_n_u8::<4>(vb));
        acc = sdot_s8_lane::<0>(acc, la, x16);
        acc = sdot_s8_lane::<1>(acc, ha, x16);
        acc = sdot_s8_lane::<2>(acc, lb, x16);
        acc = sdot_s8_lane::<3>(acc, hb, x16);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn nf4_pair_accum_16cols(
    mut aag: std::arch::aarch64::int32x4_t,
    mut abg: std::arch::aarch64::int32x4_t,
    da: *const u8,
    db: *const u8,
    x16: std::arch::aarch64::int8x16_t,
    cb: std::arch::aarch64::int8x16_t,
    mask_0f: std::arch::aarch64::uint8x16_t,
) -> (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t) {
    use std::arch::aarch64::*;
    // SAFETY: `da` and `db` hold 32 B; alternating `aag`/`abg` saturates dual SDOT pipes in <= 20 regs.
    unsafe {
        let va0 = vld1q_u8(da);
        let vb0 = vld1q_u8(db);
        let la0 = vqtbl1q_s8(cb, vandq_u8(va0, mask_0f));
        let lb0 = vqtbl1q_s8(cb, vandq_u8(vb0, mask_0f));
        let ha0 = vqtbl1q_s8(cb, vshrq_n_u8::<4>(va0));
        let hb0 = vqtbl1q_s8(cb, vshrq_n_u8::<4>(vb0));
        aag = sdot_s8_lane::<0>(aag, la0, x16);
        abg = sdot_s8_lane::<0>(abg, lb0, x16);
        aag = sdot_s8_lane::<1>(aag, ha0, x16);
        abg = sdot_s8_lane::<1>(abg, hb0, x16);

        let va1 = vld1q_u8(da.add(16));
        let vb1 = vld1q_u8(db.add(16));
        let la1 = vqtbl1q_s8(cb, vandq_u8(va1, mask_0f));
        let lb1 = vqtbl1q_s8(cb, vandq_u8(vb1, mask_0f));
        let ha1 = vqtbl1q_s8(cb, vshrq_n_u8::<4>(va1));
        let hb1 = vqtbl1q_s8(cb, vshrq_n_u8::<4>(vb1));
        aag = sdot_s8_lane::<2>(aag, la1, x16);
        abg = sdot_s8_lane::<2>(abg, lb1, x16);
        aag = sdot_s8_lane::<3>(aag, ha1, x16);
        abg = sdot_s8_lane::<3>(abg, hb1, x16);
    }
    (aag, abg)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn nf5_accum_16cols(
    mut acc: std::arch::aarch64::int32x4_t,
    va: std::arch::aarch64::uint8x16_t,
    vb: std::arch::aarch64::uint8x16_t,
    hi: std::arch::aarch64::uint8x16_t,
    x16: std::arch::aarch64::int8x16_t,
    cb2: std::arch::aarch64::int8x16x2_t,
    m0f: std::arch::aarch64::uint8x16_t,
    m10: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: NEON `vqtbl2q_s8` and `sdot` are available on Apple Silicon.
    unsafe {
        let la = vqtbl2q_s8(cb2, vbslq_u8(m10, hi, vandq_u8(va, m0f)));
        let ha = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<1>(hi), vshrq_n_u8::<4>(va)));
        let lb = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<2>(hi), vandq_u8(vb, m0f)));
        let hb = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<3>(hi), vshrq_n_u8::<4>(vb)));
        acc = sdot_s8_lane::<0>(acc, la, x16);
        acc = sdot_s8_lane::<1>(acc, ha, x16);
        acc = sdot_s8_lane::<2>(acc, lb, x16);
        acc = sdot_s8_lane::<3>(acc, hb, x16);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn nf5_pair_accum_16cols(
    mut aag: std::arch::aarch64::int32x4_t,
    mut abg: std::arch::aarch64::int32x4_t,
    da: *const u8,
    db: *const u8,
    hia: std::arch::aarch64::uint8x16_t,
    hib: std::arch::aarch64::uint8x16_t,
    x16: std::arch::aarch64::int8x16_t,
    cb2: std::arch::aarch64::int8x16x2_t,
    m0f: std::arch::aarch64::uint8x16_t,
    m10: std::arch::aarch64::uint8x16_t,
) -> (std::arch::aarch64::int32x4_t, std::arch::aarch64::int32x4_t) {
    use std::arch::aarch64::*;
    // SAFETY: `da` and `db` hold 32 B; splits low/high 8 cols so live vector regs stay <= 24.
    unsafe {
        let va0 = vld1q_u8(da);
        let vb0 = vld1q_u8(db);
        let la0 = vqtbl2q_s8(cb2, vbslq_u8(m10, hia, vandq_u8(va0, m0f)));
        let lb0 = vqtbl2q_s8(cb2, vbslq_u8(m10, hib, vandq_u8(vb0, m0f)));
        let ha0 = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<1>(hia), vshrq_n_u8::<4>(va0)));
        let hb0 = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<1>(hib), vshrq_n_u8::<4>(vb0)));
        aag = sdot_s8_lane::<0>(aag, la0, x16);
        abg = sdot_s8_lane::<0>(abg, lb0, x16);
        aag = sdot_s8_lane::<1>(aag, ha0, x16);
        abg = sdot_s8_lane::<1>(abg, hb0, x16);

        let va1 = vld1q_u8(da.add(16));
        let vb1 = vld1q_u8(db.add(16));
        let la1 = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<2>(hia), vandq_u8(va1, m0f)));
        let lb1 = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<2>(hib), vandq_u8(vb1, m0f)));
        let ha1 = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<3>(hia), vshrq_n_u8::<4>(va1)));
        let hb1 = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<3>(hib), vshrq_n_u8::<4>(vb1)));
        aag = sdot_s8_lane::<2>(aag, la1, x16);
        abg = sdot_s8_lane::<2>(abg, lb1, x16);
        aag = sdot_s8_lane::<3>(aag, ha1, x16);
        abg = sdot_s8_lane::<3>(abg, hb1, x16);
    }
    (aag, abg)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn nf3_accum_32cols(
    mut acc: std::arch::aarch64::int32x4_t,
    v0: std::arch::aarch64::uint8x16_t,
    v1: std::arch::aarch64::uint8x16_t,
    v2: std::arch::aarch64::uint8x16_t,
    x0: std::arch::aarch64::int8x16_t,
    x1: std::arch::aarch64::int8x16_t,
    cb: std::arch::aarch64::int8x16_t,
    m07: std::arch::aarch64::uint8x16_t,
    m04: std::arch::aarch64::uint8x16_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: NEON `vqtbl1q_s8` and `sdot` are available on Apple Silicon.
    unsafe {
        let i0 = vandq_u8(v0, m07);
        let i1 = vandq_u8(vshrq_n_u8::<3>(v0), m07);
        let i2 = vbslq_u8(m04, vshrq_n_u8::<4>(v2), vshrq_n_u8::<6>(v0));
        let i3 = vandq_u8(v1, m07);
        acc = sdot_s8_lane::<0>(acc, vqtbl1q_s8(cb, i0), x0);
        acc = sdot_s8_lane::<1>(acc, vqtbl1q_s8(cb, i1), x0);
        acc = sdot_s8_lane::<2>(acc, vqtbl1q_s8(cb, i2), x0);
        acc = sdot_s8_lane::<3>(acc, vqtbl1q_s8(cb, i3), x0);

        let i4 = vandq_u8(vshrq_n_u8::<3>(v1), m07);
        let i5 = vbslq_u8(m04, vshrq_n_u8::<5>(v2), vshrq_n_u8::<6>(v1));
        let i6 = vandq_u8(v2, m07);
        let i7 = vandq_u8(vshrq_n_u8::<3>(v2), m07);
        acc = sdot_s8_lane::<0>(acc, vqtbl1q_s8(cb, i4), x1);
        acc = sdot_s8_lane::<1>(acc, vqtbl1q_s8(cb, i5), x1);
        acc = sdot_s8_lane::<2>(acc, vqtbl1q_s8(cb, i6), x1);
        acc = sdot_s8_lane::<3>(acc, vqtbl1q_s8(cb, i7), x1);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn e8q2_load_packed_table() -> std::arch::aarch64::uint8x16x2_t {
    use std::arch::aarch64::*;
    // SAFETY: `E8Q2_EVEN_PACKED_U8` is a 32-byte static array.
    unsafe {
        let p = E8Q2_EVEN_PACKED_U8.as_ptr();
        uint8x16x2_t(vld1q_u8(p), vld1q_u8(p.add(16)))
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn e8q2_unpack_quad_transpose<const IS_V: bool>(
    vtpl: std::arch::aarch64::uint8x16_t,
    vsgn: std::arch::aarch64::uint8x16_t,
    is_even: std::arch::aarch64::uint8x16_t,
    ev_idx: std::arch::aarch64::uint8x16_t,
    tbl: std::arch::aarch64::uint8x16x2_t,
) -> (
    std::arch::aarch64::int8x16_t,
    std::arch::aarch64::int8x16_t,
    std::arch::aarch64::int8x16_t,
    std::arch::aarch64::int8x16_t,
) {
    use std::arch::aarch64::*;
    // SAFETY: NEON intrinsics are baseline on `aarch64`.
    unsafe {
        let m60 = vdupq_n_u8(0x60);
        let m20 = vdupq_n_u8(0x20);
        let m10 = vdupq_n_u8(0x10);
        let p = vqtbl2q_u8(tbl, ev_idx);
        let vsgn_s8 = vreinterpretq_s8_u8(vsgn);

        let c0 = {
            let ev = vandq_u8(vshlq_n_u8::<5>(p), m60);
            let od = if IS_V {
                vbslq_u8(m20, vshlq_n_u8::<1>(vtpl), m10)
            } else {
                vbslq_u8(m20, vshlq_n_u8::<5>(vtpl), m10)
            };
            let mag = vreinterpretq_s8_u8(vbslq_u8(is_even, ev, od));
            let sm = if IS_V {
                vshrq_n_s8::<7>(vshlq_n_s8::<3>(vsgn_s8))
            } else {
                vshrq_n_s8::<7>(vshlq_n_s8::<7>(vsgn_s8))
            };
            vsubq_s8(veorq_s8(mag, sm), sm)
        };
        let c1 = {
            let ev = vandq_u8(vshlq_n_u8::<3>(p), m60);
            let od = if IS_V {
                vbslq_u8(m20, vtpl, m10)
            } else {
                vbslq_u8(m20, vshlq_n_u8::<4>(vtpl), m10)
            };
            let mag = vreinterpretq_s8_u8(vbslq_u8(is_even, ev, od));
            let sm = if IS_V {
                vshrq_n_s8::<7>(vshlq_n_s8::<2>(vsgn_s8))
            } else {
                vshrq_n_s8::<7>(vshlq_n_s8::<6>(vsgn_s8))
            };
            vsubq_s8(veorq_s8(mag, sm), sm)
        };
        let c2 = {
            let ev = vandq_u8(vshlq_n_u8::<1>(p), m60);
            let od = if IS_V {
                vbslq_u8(m20, vshrq_n_u8::<1>(vtpl), m10)
            } else {
                vbslq_u8(m20, vshlq_n_u8::<3>(vtpl), m10)
            };
            let mag = vreinterpretq_s8_u8(vbslq_u8(is_even, ev, od));
            let sm = if IS_V {
                vshrq_n_s8::<7>(vshlq_n_s8::<1>(vsgn_s8))
            } else {
                vshrq_n_s8::<7>(vshlq_n_s8::<5>(vsgn_s8))
            };
            vsubq_s8(veorq_s8(mag, sm), sm)
        };
        let c3 = {
            let ev = vandq_u8(vshrq_n_u8::<1>(p), m60);
            let od = if IS_V {
                vbslq_u8(m20, vshrq_n_u8::<2>(vtpl), m10)
            } else {
                vbslq_u8(m20, vshlq_n_u8::<2>(vtpl), m10)
            };
            let mag = vreinterpretq_s8_u8(vbslq_u8(is_even, ev, od));
            let sm = if IS_V {
                vshrq_n_s8::<7>(vsgn_s8)
            } else {
                vshrq_n_s8::<7>(vshlq_n_s8::<4>(vsgn_s8))
            };
            vsubq_s8(veorq_s8(mag, sm), sm)
        };

        let z01_lo = vreinterpretq_s16_s8(vzip1q_s8(c0, c1));
        let z23_lo = vreinterpretq_s16_s8(vzip1q_s8(c2, c3));
        let z01_hi = vreinterpretq_s16_s8(vzip2q_s8(c0, c1));
        let z23_hi = vreinterpretq_s16_s8(vzip2q_s8(c2, c3));
        (
            vreinterpretq_s8_s16(vzip1q_s16(z01_lo, z23_lo)),
            vreinterpretq_s8_s16(vzip2q_s16(z01_lo, z23_lo)),
            vreinterpretq_s8_s16(vzip1q_s16(z01_hi, z23_hi)),
            vreinterpretq_s8_s16(vzip2q_s16(z01_hi, z23_hi)),
        )
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn e8q2_accum_32cols(
    mut acc: std::arch::aarch64::int32x4_t,
    blk: *const u8,
    x0: std::arch::aarch64::int8x16_t,
    x1: std::arch::aarch64::int8x16_t,
    tbl: std::arch::aarch64::uint8x16x2_t,
) -> std::arch::aarch64::int32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: `blk` holds 32 B (`16` tpl + `16` sgn); NEON + `sdot` are available on Apple Silicon.
    unsafe {
        let vtpl = vld1q_u8(blk);
        let vsgn = vld1q_u8(blk.add(16));
        let is_even = vtstq_u8(vcntq_u8(veorq_u8(vtpl, vsgn)), vdupq_n_u8(1));
        let tpl_u = vandq_u8(vtpl, vdupq_n_u8(0x0F));
        let tpl_v = vshrq_n_u8::<4>(vtpl);
        let v12 = vdupq_n_u8(12);
        let m16 = vdupq_n_u8(16);
        let u_lt12 = vcltq_u8(tpl_u, v12);
        let v_lt12 = vcltq_u8(tpl_v, v12);
        let u_idx = vorrq_u8(tpl_u, vandq_u8(vbicq_u8(u_lt12, v_lt12), m16));
        let v_idx = vorrq_u8(tpl_v, vandq_u8(vbicq_u8(v_lt12, u_lt12), m16));

        let (u0, u1, u2, u3) = e8q2_unpack_quad_transpose::<false>(vtpl, vsgn, is_even, u_idx, tbl);
        acc = sdot_s8_lane::<0>(acc, u0, x0);
        acc = sdot_s8_lane::<2>(acc, u1, x0);
        acc = sdot_s8_lane::<0>(acc, u2, x1);
        acc = sdot_s8_lane::<2>(acc, u3, x1);

        let (v0, v1, v2, v3) = e8q2_unpack_quad_transpose::<true>(vtpl, vsgn, is_even, v_idx, tbl);
        acc = sdot_s8_lane::<1>(acc, v0, x0);
        acc = sdot_s8_lane::<3>(acc, v1, x0);
        acc = sdot_s8_lane::<1>(acc, v2, x1);
        acc = sdot_s8_lane::<3>(acc, v3, x1);
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
unsafe fn nf_sdot_one_stripe_neon<const BITS: usize>(
    matrix: &NfSdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    s: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let nc = matrix.num_chunks;
    let cb32 = matrix.codebook32();
    let cb = if BITS >= 3 {
        unsafe { vld1q_s8(cb32.as_ptr()) }
    } else {
        vdupq_n_s8(0)
    };
    let cb2 = if BITS == 5 {
        unsafe { int8x16x2_t(cb, vld1q_s8(cb32.as_ptr().add(16))) }
    } else {
        int8x16x2_t(cb, cb)
    };
    let e8_tbl = if BITS == 2 {
        unsafe { e8q2_load_packed_table() }
    } else {
        uint8x16x2_t(vdupq_n_u8(0), vdupq_n_u8(0))
    };
    let mask_0f = vdupq_n_u8(0x0F);
    let m10 = vdupq_n_u8(0x10);
    let m07 = vdupq_n_u8(0x07);
    let m04 = vdupq_n_u8(0x04);
    let mut f = [vdupq_n_f32(0.0); 4];

    for c in 0..nc {
        let sx = unsafe { *xscales.add(c) };
        if sx == 0.0 {
            continue;
        }
        let (d_ptr, sc_ptr) = matrix.stripe_chunk_ptrs(s, c);
        let mut a = [vdupq_n_s32(0); 4];
        // SAFETY: `c * 64 + 64 <= cols`; `d_ptr` and `sc_ptr` stay inside the `(s, c)` record.
        unsafe {
            let x_base = xq.add(c * 64);
            let x0 = vld1q_s8(x_base);
            let x1 = vld1q_s8(x_base.add(16));
            let x2 = vld1q_s8(x_base.add(32));
            let x3 = vld1q_s8(x_base.add(48));
            if BITS == 5 {
                for (g, ag) in a.iter_mut().enumerate() {
                    let hi01 = vld1q_u8(d_ptr.add(512 + g * 32));
                    let hi23 = vld1q_u8(d_ptr.add(512 + g * 32 + 16));
                    *ag = nf5_accum_16cols(*ag, vld1q_u8(d_ptr.add(g * 32)), vld1q_u8(d_ptr.add(g * 32 + 16)), hi01, x0, cb2, mask_0f, m10);
                    *ag = nf5_accum_16cols(*ag, vld1q_u8(d_ptr.add(128 + g * 32)), vld1q_u8(d_ptr.add(128 + g * 32 + 16)), vshlq_n_u8::<4>(hi01), x1, cb2, mask_0f, m10);
                    *ag = nf5_accum_16cols(*ag, vld1q_u8(d_ptr.add(256 + g * 32)), vld1q_u8(d_ptr.add(256 + g * 32 + 16)), hi23, x2, cb2, mask_0f, m10);
                    *ag = nf5_accum_16cols(*ag, vld1q_u8(d_ptr.add(384 + g * 32)), vld1q_u8(d_ptr.add(384 + g * 32 + 16)), vshlq_n_u8::<4>(hi23), x3, cb2, mask_0f, m10);
                }
            } else if BITS == 4 {
                for (g, ag) in a.iter_mut().enumerate() {
                    *ag = nf4_accum_16cols(*ag, vld1q_u8(d_ptr.add(g * 32)), vld1q_u8(d_ptr.add(g * 32 + 16)), x0, cb, mask_0f);
                    *ag = nf4_accum_16cols(*ag, vld1q_u8(d_ptr.add(128 + g * 32)), vld1q_u8(d_ptr.add(128 + g * 32 + 16)), x1, cb, mask_0f);
                    *ag = nf4_accum_16cols(*ag, vld1q_u8(d_ptr.add(256 + g * 32)), vld1q_u8(d_ptr.add(256 + g * 32 + 16)), x2, cb, mask_0f);
                    *ag = nf4_accum_16cols(*ag, vld1q_u8(d_ptr.add(384 + g * 32)), vld1q_u8(d_ptr.add(384 + g * 32 + 16)), x3, cb, mask_0f);
                }
            } else if BITS == 3 {
                for (g, ag) in a.iter_mut().enumerate() {
                    let b0 = d_ptr.add(g * 48);
                    *ag = nf3_accum_32cols(*ag, vld1q_u8(b0), vld1q_u8(b0.add(16)), vld1q_u8(b0.add(32)), x0, x1, cb, m07, m04);
                    let b1 = d_ptr.add(192 + g * 48);
                    *ag = nf3_accum_32cols(*ag, vld1q_u8(b1), vld1q_u8(b1.add(16)), vld1q_u8(b1.add(32)), x2, x3, cb, m07, m04);
                }
            } else {
                for (g, ag) in a.iter_mut().enumerate() {
                    *ag = e8q2_accum_32cols(*ag, d_ptr.add(g * 32), x0, x1, e8_tbl);
                    *ag = e8q2_accum_32cols(*ag, d_ptr.add(128 + g * 32), x2, x3, e8_tbl);
                }
            }
            let vsx = vdupq_n_f32(sx);
            let bf0 = vreinterpretq_u16_u8(vld1q_u8(sc_ptr));
            let bf1 = vreinterpretq_u16_u8(vld1q_u8(sc_ptr.add(16)));
            let ws = [
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf0))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf0)),
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf1))),
                vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf1)),
            ];
            for g in 0..4 {
                f[g] = vfmaq_f32(f[g], vmulq_f32(vcvtq_f32_s32(a[g]), vsx), ws[g]);
            }
        }
    }

    let row_start = s * SIMD_ROW_STRIPE;
    let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
    // SAFETY: `row_start + valid <= matrix.rows`.
    unsafe {
        let dst = out_base.add(row_start);
        if valid == SIMD_ROW_STRIPE {
            vst1q_f32(dst, f[0]);
            vst1q_f32(dst.add(4), f[1]);
            vst1q_f32(dst.add(8), f[2]);
            vst1q_f32(dst.add(12), f[3]);
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            for (g, fg) in f.iter().enumerate() {
                vst1q_f32(tmp.as_mut_ptr().add(g * 4), *fg);
            }
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, valid);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load_bf16x4_as_f32(ptr: *const u8) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    // SAFETY: `ptr` holds 8 bytes (4 little-endian BF16 values); `ldr d` supports unaligned addresses on ARM64.
    unsafe {
        let raw = std::ptr::read_unaligned(ptr.cast::<u64>());
        let u16x4 = vreinterpret_u16_u64(vcreate_u64(raw));
        vreinterpretq_f32_u32(vshll_n_u16::<16>(u16x4))
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn nf_sdot_decode_neon<const BITS: usize>(
    matrix: &NfSdotMatrix,
    xq: *const i8,
    xscales: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let nc = matrix.num_chunks;
    let nsg = matrix.num_scale_groups;
    let sgc = matrix.scale_group_chunks;
    let chunk_b = NfSdotMatrix::stripe_chunk_bytes(BITS);
    let pair_tot = matrix.pair_total_bytes();
    let cb32 = matrix.codebook32();
    let cb = if BITS >= 3 {
        unsafe { vld1q_s8(cb32.as_ptr()) }
    } else {
        vdupq_n_s8(0)
    };
    let cb2 = if BITS == 5 {
        unsafe { int8x16x2_t(cb, vld1q_s8(cb32.as_ptr().add(16))) }
    } else {
        int8x16x2_t(cb, cb)
    };
    let e8_tbl = if BITS == 2 {
        unsafe { e8q2_load_packed_table() }
    } else {
        uint8x16x2_t(vdupq_n_u8(0), vdupq_n_u8(0))
    };
    let mask_0f = vdupq_n_u8(0x0F);
    let m10 = vdupq_n_u8(0x10);
    let m07 = vdupq_n_u8(0x07);
    let m04 = vdupq_n_u8(0x04);
    let pair_stream = matrix.pair_stream.as_ptr();

    let mut s = stripe_start;
    if (s & 1) != 0 && s < stripe_end {
        // SAFETY: `s < stripe_end <= matrix.num_stripes`.
        unsafe { nf_sdot_one_stripe_neon::<BITS>(matrix, xq, xscales, s, out_base) };
        s += 1;
    }

    // Keep sc_ptr in a GP register and (aag, abg) per g so live NEON registers stay <= 24.
    while s + 2 <= stripe_end {
        let p = s / 2;
        let mut cursor = unsafe { pair_stream.add(p * pair_tot) };
        let mut fa = [vdupq_n_f32(0.0); 4];
        let mut fb = [vdupq_n_f32(0.0); 4];

        for sg in 0..nsg {
            // SAFETY: `cursor` points to `64` B of paired BF16 scales followed by `c_len * 2 * chunk_b` data.
            unsafe {
                let sc_ptr = cursor;
                cursor = cursor.add(64);

                let c_start = sg * sgc;
                let c_end = (c_start + sgc).min(nc);
                for c in c_start..c_end {
                    let da = cursor;
                    let db = cursor.add(chunk_b);
                    cursor = cursor.add(2 * chunk_b);

                    let sx = *xscales.add(c);
                    if sx == 0.0 {
                        continue;
                    }
                    let vsx = vdupq_n_f32(sx);
                    let x_base = xq.add(c * 64);
                    let x0 = vld1q_s8(x_base);
                    let x1 = vld1q_s8(x_base.add(16));
                    let x2 = vld1q_s8(x_base.add(32));
                    let x3 = vld1q_s8(x_base.add(48));

                    if BITS == 5 {
                        for g in 0..4 {
                            let ga = da.add(g * 32);
                            let gb = db.add(g * 32);
                            let hia01 = vld1q_u8(da.add(512 + g * 32));
                            let hib01 = vld1q_u8(db.add(512 + g * 32));
                            let mut aag = vdupq_n_s32(0);
                            let mut abg = vdupq_n_s32(0);
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga, gb, hia01, hib01, x0, cb2, mask_0f, m10);
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga.add(128), gb.add(128), vshlq_n_u8::<4>(hia01), vshlq_n_u8::<4>(hib01), x1, cb2, mask_0f, m10);
                            let hia23 = vld1q_u8(da.add(512 + g * 32 + 16));
                            let hib23 = vld1q_u8(db.add(512 + g * 32 + 16));
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga.add(256), gb.add(256), hia23, hib23, x2, cb2, mask_0f, m10);
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga.add(384), gb.add(384), vshlq_n_u8::<4>(hia23), vshlq_n_u8::<4>(hib23), x3, cb2, mask_0f, m10);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    } else if BITS == 4 {
                        for g in 0..4 {
                            let ga = da.add(g * 32);
                            let gb = db.add(g * 32);
                            let mut aag = vdupq_n_s32(0);
                            let mut abg = vdupq_n_s32(0);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga, gb, x0, cb, mask_0f);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga.add(128), gb.add(128), x1, cb, mask_0f);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga.add(256), gb.add(256), x2, cb, mask_0f);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga.add(384), gb.add(384), x3, cb, mask_0f);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    } else if BITS == 3 {
                        for g in 0..4 {
                            let a0 = da.add(g * 48);
                            let b0 = db.add(g * 48);
                            let mut aag = nf3_accum_32cols(vdupq_n_s32(0), vld1q_u8(a0), vld1q_u8(a0.add(16)), vld1q_u8(a0.add(32)), x0, x1, cb, m07, m04);
                            let mut abg = nf3_accum_32cols(vdupq_n_s32(0), vld1q_u8(b0), vld1q_u8(b0.add(16)), vld1q_u8(b0.add(32)), x0, x1, cb, m07, m04);
                            let a1 = da.add(192 + g * 48);
                            let b1 = db.add(192 + g * 48);
                            aag = nf3_accum_32cols(aag, vld1q_u8(a1), vld1q_u8(a1.add(16)), vld1q_u8(a1.add(32)), x2, x3, cb, m07, m04);
                            abg = nf3_accum_32cols(abg, vld1q_u8(b1), vld1q_u8(b1.add(16)), vld1q_u8(b1.add(32)), x2, x3, cb, m07, m04);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    } else {
                        for g in 0..4 {
                            let mut aag = e8q2_accum_32cols(vdupq_n_s32(0), da.add(g * 32), x0, x1, e8_tbl);
                            aag = e8q2_accum_32cols(aag, da.add(128 + g * 32), x2, x3, e8_tbl);
                            let mut abg = e8q2_accum_32cols(vdupq_n_s32(0), db.add(g * 32), x0, x1, e8_tbl);
                            abg = e8q2_accum_32cols(abg, db.add(128 + g * 32), x2, x3, e8_tbl);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    }
                }
            }
        }

        for (st_idx, f_arr) in [(s, &fa), (s + 1, &fb)] {
            let row_start = st_idx * SIMD_ROW_STRIPE;
            let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
            // SAFETY: `row_start + valid <= matrix.rows`.
            unsafe {
                let dst = out_base.add(row_start);
                if valid == SIMD_ROW_STRIPE {
                    vst1q_f32(dst, f_arr[0]);
                    vst1q_f32(dst.add(4), f_arr[1]);
                    vst1q_f32(dst.add(8), f_arr[2]);
                    vst1q_f32(dst.add(12), f_arr[3]);
                } else {
                    let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
                    for (g, fg) in f_arr.iter().enumerate() {
                        vst1q_f32(tmp.as_mut_ptr().add(g * 4), *fg);
                    }
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, valid);
                }
            }
        }
        s += 2;
    }

    while s < stripe_end {
        // SAFETY: `s < stripe_end <= matrix.num_stripes`.
        unsafe { nf_sdot_one_stripe_neon::<BITS>(matrix, xq, xscales, s, out_base) };
        s += 1;
    }
}

/// Runs `SDOT` directly over a borrowed `pair_stream` slice (mapped MoE experts, `rows % 32 == 0`).
///
/// # Safety
/// `rows % 32 == 0`, `cols % 64 == 0`, `pair_end <= rows / 32`, `xq` valid for `cols` bytes,
/// `xscales` valid for `cols / 64` floats, `out_base` valid for `rows` floats on disjoint pair ranges.
#[allow(clippy::too_many_arguments)]
pub unsafe fn nf_sdot_pair_stream_range_slices(
    pair_stream: &[u8],
    codebook: &[i8; 16],
    rows: usize,
    cols: usize,
    bits: usize,
    scale_group_chunks: usize,
    xq: *const i8,
    xscales: *const f32,
    pair_start: usize,
    pair_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(rows % 32, 0);
    debug_assert_eq!(cols % 64, 0);
    let num_pairs = rows / 32;
    debug_assert!(pair_end <= num_pairs);
    if pair_start >= pair_end {
        return;
    }
    let nc = cols / 64;
    let sgc = scale_group_chunks.max(1);
    let nsg = nc.div_ceil(sgc);
    let chunk_b = NfSdotMatrix::stripe_chunk_bytes(bits);
    let pair_tot = nsg * 64 + nc * 2 * chunk_b;
    debug_assert!(pair_stream.len() >= num_pairs * pair_tot);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON and `sdot` are available; caller upholds bounds.
        unsafe {
            match bits {
                5 => nf_sdot_pairs_slice_neon::<5>(
                    pair_stream.as_ptr(),
                    codebook,
                    nc,
                    nsg,
                    sgc,
                    pair_tot,
                    xq,
                    xscales,
                    pair_start,
                    pair_end,
                    out_base,
                ),
                4 => nf_sdot_pairs_slice_neon::<4>(
                    pair_stream.as_ptr(),
                    codebook,
                    nc,
                    nsg,
                    sgc,
                    pair_tot,
                    xq,
                    xscales,
                    pair_start,
                    pair_end,
                    out_base,
                ),
                3 => nf_sdot_pairs_slice_neon::<3>(
                    pair_stream.as_ptr(),
                    codebook,
                    nc,
                    nsg,
                    sgc,
                    pair_tot,
                    xq,
                    xscales,
                    pair_start,
                    pair_end,
                    out_base,
                ),
                _ => nf_sdot_pairs_slice_neon::<2>(
                    pair_stream.as_ptr(),
                    codebook,
                    nc,
                    nsg,
                    sgc,
                    pair_tot,
                    xq,
                    xscales,
                    pair_start,
                    pair_end,
                    out_base,
                ),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let cb32 = nf_expand_codebook32(bits, codebook);
        let q_s = unsafe { std::slice::from_raw_parts(xq, cols) };
        let sc_s = unsafe { std::slice::from_raw_parts(xscales, nc) };
        for p in pair_start..pair_end {
            for sub in 0..2 {
                let s = p * 2 + sub;
                for lane in 0..SIMD_ROW_STRIPE {
                    let r = s * SIMD_ROW_STRIPE + lane;
                    let g = lane / 4;
                    let r_in_g = lane % 4;
                    let mut acc = 0.0f32;
                    for sg in 0..nsg {
                        let sg_off = p * pair_tot + sg * (64 + sgc * 2 * chunk_b);
                        let sc_off = sg_off + sub * 32 + lane * 2;
                        let bf = u16::from_le_bytes([pair_stream[sc_off], pair_stream[sc_off + 1]]);
                        let w_scale = f32::from_bits((bf as u32) << 16);
                        let c0 = sg * sgc;
                        let c1 = (c0 + sgc).min(nc);
                        for (ci, c) in (c0..c1).enumerate() {
                            let d_off = sg_off + 64 + (ci * 2 + sub) * chunk_b;
                            let d = &pair_stream[d_off..d_off + chunk_b];
                            let mut dot = 0i32;
                            if bits == 2 {
                                for kp in 0..2 {
                                    for eb in 0..4 {
                                        let off = kp * 128 + g * 32 + eb * 4 + r_in_g;
                                        let q8 = e8q2_decode_8d(d[off], d[off + 16]);
                                        let col0 = c * 64 + kp * 32 + eb * 8;
                                        for (sub_k, &qv) in q8.iter().enumerate() {
                                            dot += (qv as i32) * (q_s[col0 + sub_k] as i32);
                                        }
                                    }
                                }
                            } else {
                                for k in 0..64 {
                                    let idx = if bits == 4 || bits == 5 {
                                        let (kb, w) = (k / 16, k % 16);
                                        let sub4 = w / 4;
                                        let t = w % 4;
                                        let b = r_in_g * 4 + t;
                                        let byte = d[kb * 128 + g * 32 + (sub4 / 2) * 16 + b];
                                        let lo4 = if (sub4 & 1) == 0 { byte & 0x0F } else { byte >> 4 };
                                        if bits == 5 {
                                            let hi_byte = d[512 + g * 32 + (kb / 2) * 16 + b];
                                            let shift = if (kb & 1) == 0 { 4 + sub4 } else { sub4 };
                                            (lo4 | (((hi_byte >> shift) & 1) << 4)) as usize
                                        } else {
                                            lo4 as usize
                                        }
                                    } else {
                                        let half = k / 32;
                                        let k32 = k % 32;
                                        let sub8 = k32 / 4;
                                        let k_in_q = k32 % 4;
                                        let b = half * 192 + g * 48;
                                        let j = r_in_g * 4 + k_in_q;
                                        let (v0, v1, v2) = (d[b + j], d[b + 16 + j], d[b + 32 + j]);
                                        (match sub8 {
                                            0 => v0 & 0x07,
                                            1 => (v0 >> 3) & 0x07,
                                            2 => (v0 >> 6) | ((v2 >> 4) & 0x04),
                                            3 => v1 & 0x07,
                                            4 => (v1 >> 3) & 0x07,
                                            5 => (v1 >> 6) | ((v2 >> 5) & 0x04),
                                            6 => v2 & 0x07,
                                            _ => (v2 >> 3) & 0x07,
                                        }) as usize
                                    };
                                    dot += (cb32[idx] as i32) * (q_s[c * 64 + k] as i32);
                                }
                            }
                            acc += (dot as f32) * sc_s[c] * w_scale;
                        }
                    }
                    unsafe { *out_base.add(r) = acc };
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn nf_sdot_pairs_slice_neon<const BITS: usize>(
    pair_stream: *const u8,
    codebook: &[i8; 16],
    nc: usize,
    nsg: usize,
    sgc: usize,
    pair_tot: usize,
    xq: *const i8,
    xscales: *const f32,
    pair_start: usize,
    pair_end: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let chunk_b = NfSdotMatrix::stripe_chunk_bytes(BITS);
    let cb32 = nf_expand_codebook32(BITS, codebook);
    let cb = if BITS >= 3 {
        unsafe { vld1q_s8(cb32.as_ptr()) }
    } else {
        vdupq_n_s8(0)
    };
    let cb2 = if BITS == 5 {
        unsafe { int8x16x2_t(cb, vld1q_s8(cb32.as_ptr().add(16))) }
    } else {
        int8x16x2_t(cb, cb)
    };
    let e8_tbl = if BITS == 2 {
        unsafe { e8q2_load_packed_table() }
    } else {
        uint8x16x2_t(vdupq_n_u8(0), vdupq_n_u8(0))
    };
    let mask_0f = vdupq_n_u8(0x0F);
    let m10 = vdupq_n_u8(0x10);
    let m07 = vdupq_n_u8(0x07);
    let m04 = vdupq_n_u8(0x04);

    for p in pair_start..pair_end {
        let mut cursor = unsafe { pair_stream.add(p * pair_tot) };
        let mut fa = [vdupq_n_f32(0.0); 4];
        let mut fb = [vdupq_n_f32(0.0); 4];
        for sg in 0..nsg {
            unsafe {
                let sc_ptr = cursor;
                cursor = cursor.add(64);
                let c_start = sg * sgc;
                let c_end = (c_start + sgc).min(nc);
                for c in c_start..c_end {
                    let da = cursor;
                    let db = cursor.add(chunk_b);
                    cursor = cursor.add(2 * chunk_b);
                    let sx = *xscales.add(c);
                    if sx == 0.0 {
                        continue;
                    }
                    let vsx = vdupq_n_f32(sx);
                    let xb = xq.add(c * 64);
                    let x0 = vld1q_s8(xb);
                    let x1 = vld1q_s8(xb.add(16));
                    let x2 = vld1q_s8(xb.add(32));
                    let x3 = vld1q_s8(xb.add(48));
                    if BITS == 5 {
                        for g in 0..4 {
                            let ga = da.add(g * 32);
                            let gb = db.add(g * 32);
                            let hia01 = vld1q_u8(da.add(512 + g * 32));
                            let hib01 = vld1q_u8(db.add(512 + g * 32));
                            let mut aag = vdupq_n_s32(0);
                            let mut abg = vdupq_n_s32(0);
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga, gb, hia01, hib01, x0, cb2, mask_0f, m10);
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga.add(128), gb.add(128), vshlq_n_u8::<4>(hia01), vshlq_n_u8::<4>(hib01), x1, cb2, mask_0f, m10);
                            let hia23 = vld1q_u8(da.add(512 + g * 32 + 16));
                            let hib23 = vld1q_u8(db.add(512 + g * 32 + 16));
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga.add(256), gb.add(256), hia23, hib23, x2, cb2, mask_0f, m10);
                            (aag, abg) = nf5_pair_accum_16cols(aag, abg, ga.add(384), gb.add(384), vshlq_n_u8::<4>(hia23), vshlq_n_u8::<4>(hib23), x3, cb2, mask_0f, m10);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    } else if BITS == 4 {
                        for g in 0..4 {
                            let ga = da.add(g * 32);
                            let gb = db.add(g * 32);
                            let mut aag = vdupq_n_s32(0);
                            let mut abg = vdupq_n_s32(0);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga, gb, x0, cb, mask_0f);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga.add(128), gb.add(128), x1, cb, mask_0f);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga.add(256), gb.add(256), x2, cb, mask_0f);
                            (aag, abg) = nf4_pair_accum_16cols(aag, abg, ga.add(384), gb.add(384), x3, cb, mask_0f);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    } else if BITS == 3 {
                        for g in 0..4 {
                            let ga0 = da.add(g * 48);
                            let gb0 = db.add(g * 48);
                            let mut aag = nf3_accum_32cols(vdupq_n_s32(0), vld1q_u8(ga0), vld1q_u8(ga0.add(16)), vld1q_u8(ga0.add(32)), x0, x1, cb, m07, m04);
                            let mut abg = nf3_accum_32cols(vdupq_n_s32(0), vld1q_u8(gb0), vld1q_u8(gb0.add(16)), vld1q_u8(gb0.add(32)), x0, x1, cb, m07, m04);
                            let ga1 = da.add(192 + g * 48);
                            let gb1 = db.add(192 + g * 48);
                            aag = nf3_accum_32cols(aag, vld1q_u8(ga1), vld1q_u8(ga1.add(16)), vld1q_u8(ga1.add(32)), x2, x3, cb, m07, m04);
                            abg = nf3_accum_32cols(abg, vld1q_u8(gb1), vld1q_u8(gb1.add(16)), vld1q_u8(gb1.add(32)), x2, x3, cb, m07, m04);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    } else {
                        for g in 0..4 {
                            let mut aag = e8q2_accum_32cols(vdupq_n_s32(0), da.add(g * 32), x0, x1, e8_tbl);
                            aag = e8q2_accum_32cols(aag, da.add(128 + g * 32), x2, x3, e8_tbl);
                            let mut abg = e8q2_accum_32cols(vdupq_n_s32(0), db.add(g * 32), x0, x1, e8_tbl);
                            abg = e8q2_accum_32cols(abg, db.add(128 + g * 32), x2, x3, e8_tbl);
                            let wsa_g = load_bf16x4_as_f32(sc_ptr.add(g * 8));
                            let wsb_g = load_bf16x4_as_f32(sc_ptr.add(32 + g * 8));
                            fa[g] = vfmaq_f32(fa[g], vmulq_f32(vcvtq_f32_s32(aag), vsx), wsa_g);
                            fb[g] = vfmaq_f32(fb[g], vmulq_f32(vcvtq_f32_s32(abg), vsx), wsb_g);
                        }
                    }
                }
            }
        }
        unsafe {
            let dsta = out_base.add(p * 32);
            vst1q_f32(dsta, fa[0]);
            vst1q_f32(dsta.add(4), fa[1]);
            vst1q_f32(dsta.add(8), fa[2]);
            vst1q_f32(dsta.add(12), fa[3]);
            let dstb = dsta.add(16);
            vst1q_f32(dstb, fb[0]);
            vst1q_f32(dstb.add(4), fb[1]);
            vst1q_f32(dstb.add(8), fb[2]);
            vst1q_f32(dstb.add(12), fb[3]);
        }
    }
}

/// Runs stripes `[stripe_start, stripe_end)` of an [`NfSdotMatrix`] for a batch of tokens, reading
/// each weight tile once.
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `acts.len() * out_stride` floats.
pub unsafe fn nf_sdot_batch_stripe_range(
    matrix: &NfSdotMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    if acts.is_empty() || stripe_start >= stripe_end {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Caller upholds bounds.
        unsafe {
            match matrix.bits {
                5 => nf_sdot_batch_outer_neon::<5>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
                4 => nf_sdot_batch_outer_neon::<4>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
                3 => nf_sdot_batch_outer_neon::<3>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
                _ => nf_sdot_batch_outer_neon::<2>(matrix, acts, stripe_start, stripe_end, out_base, out_stride),
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for (t, blut) in acts.iter().enumerate() {
            // SAFETY: Caller upholds bounds.
            unsafe {
                nf_sdot_stripe_range(matrix, blut, stripe_start, stripe_end, out_base.add(t * out_stride));
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn nf_sdot_batch_outer_neon<const BITS: usize>(
    matrix: &NfSdotMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    const C_TILE: usize = 16;
    let nc = matrix.num_chunks;
    let n_tok = acts.len();

    let run_one = |s: usize, c_base: usize, c_end: usize| {
        let first_tile = c_base == 0;
        let mut t = 0usize;
        while t + 8 <= n_tok {
            // SAFETY: `t + 8 <= n_tok`, `s < num_stripes`, `c_end <= nc`.
            unsafe {
                nf_sdot_batch_tile_neon::<BITS, 8>(
                    matrix,
                    &acts[t..t + 8],
                    s,
                    c_base,
                    c_end,
                    first_tile,
                    out_base.add(t * out_stride),
                    out_stride,
                );
            }
            t += 8;
        }
        let rem = n_tok - t;
        if rem > 0 {
            // SAFETY: `rem in 1..=7` and `t + rem == n_tok`.
            unsafe {
                let sub = &acts[t..n_tok];
                let dst = out_base.add(t * out_stride);
                match rem {
                    7 => nf_sdot_batch_tile_neon::<BITS, 7>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    6 => nf_sdot_batch_tile_neon::<BITS, 6>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    5 => nf_sdot_batch_tile_neon::<BITS, 5>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    4 => nf_sdot_batch_tile_neon::<BITS, 4>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    3 => nf_sdot_batch_tile_neon::<BITS, 3>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    2 => nf_sdot_batch_tile_neon::<BITS, 2>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                    _ => nf_sdot_batch_tile_neon::<BITS, 1>(matrix, sub, s, c_base, c_end, first_tile, dst, out_stride),
                }
            }
        }
    };

    let mut s = stripe_start;
    if (s & 1) != 0 && s < stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            run_one(s, c_base, c_end);
            c_base = c_end;
        }
        s += 1;
    }
    while s + 2 <= stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            run_one(s, c_base, c_end);
            run_one(s + 1, c_base, c_end);
            c_base = c_end;
        }
        s += 2;
    }
    while s < stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            run_one(s, c_base, c_end);
            c_base = c_end;
        }
        s += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn nf_sdot_batch_tile_neon<const BITS: usize, const NT: usize>(
    matrix: &NfSdotMatrix,
    acts: &[BipolarActivationLut],
    s: usize,
    c_base: usize,
    c_end: usize,
    first_tile: bool,
    out_base: *mut f32,
    out_stride: usize,
) {
    use std::arch::aarch64::*;
    let i8a: [&Int8ChunkActivation; NT] =
        std::array::from_fn(|t| acts[t].int8.as_ref().expect("int8 activations required for NfSdotMatrix"));
    let cb32 = matrix.codebook32();
    let cb = if BITS >= 3 {
        unsafe { vld1q_s8(cb32.as_ptr()) }
    } else {
        vdupq_n_s8(0)
    };
    let cb2 = if BITS == 5 {
        unsafe { int8x16x2_t(cb, vld1q_s8(cb32.as_ptr().add(16))) }
    } else {
        int8x16x2_t(cb, cb)
    };
    let e8_tbl = if BITS == 2 {
        unsafe { e8q2_load_packed_table() }
    } else {
        uint8x16x2_t(vdupq_n_u8(0), vdupq_n_u8(0))
    };
    let mask_0f = vdupq_n_u8(0x0F);
    let m10 = vdupq_n_u8(0x10);
    let m07 = vdupq_n_u8(0x07);
    let m04 = vdupq_n_u8(0x04);
    let row_start = s * SIMD_ROW_STRIPE;
    let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
    let full_stripe = valid == SIMD_ROW_STRIPE;
    let num_g = valid.div_ceil(4);

    // Outer `g` keeps only `fg: [float32x4_t; NT]` (<=8 regs) + `acc` (<=8 regs) live across L1D-hot `c_base..c_end`.
    for g in 0..num_g {
        let g_off = g * 4;
        let g_valid = (valid - g_off).min(4);
        let mut fg: [float32x4_t; NT] = if first_tile {
            [vdupq_n_f32(0.0); NT]
        } else if full_stripe || g_valid == 4 {
            std::array::from_fn(|t| unsafe {
                vld1q_f32(out_base.add(t * out_stride + row_start + g_off))
            })
        } else {
            std::array::from_fn(|t| unsafe {
                let mut tmp = [0.0f32; 4];
                std::ptr::copy_nonoverlapping(
                    out_base.add(t * out_stride + row_start + g_off),
                    tmp.as_mut_ptr(),
                    g_valid,
                );
                vld1q_f32(tmp.as_ptr())
            })
        };

        for c in c_base..c_end {
            let (d_ptr, sc_ptr) = matrix.stripe_chunk_ptrs(s, c);
            let mut acc = [vdupq_n_s32(0); NT];
            if BITS == 5 {
                // SAFETY: `c * 64 + 64 <= cols`; `d_ptr` holds `640` B.
                unsafe {
                    let hi01 = vld1q_u8(d_ptr.add(512 + g * 32));
                    let hi23 = vld1q_u8(d_ptr.add(512 + g * 32 + 16));
                    let his = [hi01, vshlq_n_u8::<4>(hi01), hi23, vshlq_n_u8::<4>(hi23)];
                    for (k, &hi) in his.iter().enumerate() {
                        let blk = d_ptr.add(k * 128 + g * 32);
                        let va = vld1q_u8(blk);
                        let vb = vld1q_u8(blk.add(16));
                        let la = vqtbl2q_s8(cb2, vbslq_u8(m10, hi, vandq_u8(va, mask_0f)));
                        let ha = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<1>(hi), vshrq_n_u8::<4>(va)));
                        let lb = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<2>(hi), vandq_u8(vb, mask_0f)));
                        let hb = vqtbl2q_s8(cb2, vbslq_u8(m10, vshrq_n_u8::<3>(hi), vshrq_n_u8::<4>(vb)));
                        let col_off = c * 64 + k * 16;
                        for t in 0..NT {
                            let x16 = vld1q_s8(i8a[t].q.as_ptr().add(col_off));
                            let mut a = acc[t];
                            a = sdot_s8_lane::<0>(a, la, x16);
                            a = sdot_s8_lane::<1>(a, ha, x16);
                            a = sdot_s8_lane::<2>(a, lb, x16);
                            a = sdot_s8_lane::<3>(a, hb, x16);
                            acc[t] = a;
                        }
                    }
                }
            } else if BITS == 4 {
                for k in 0..4 {
                    // SAFETY: `c * 64 + k * 16 + 16 <= cols`; `d_ptr` holds `512` B.
                    unsafe {
                        let blk = d_ptr.add(k * 128 + g * 32);
                        let va = vld1q_u8(blk);
                        let vb = vld1q_u8(blk.add(16));
                        let la = vqtbl1q_s8(cb, vandq_u8(va, mask_0f));
                        let ha = vqtbl1q_s8(cb, vshrq_n_u8::<4>(va));
                        let lb = vqtbl1q_s8(cb, vandq_u8(vb, mask_0f));
                        let hb = vqtbl1q_s8(cb, vshrq_n_u8::<4>(vb));
                        let col_off = c * 64 + k * 16;
                        for t in 0..NT {
                            let x16 = vld1q_s8(i8a[t].q.as_ptr().add(col_off));
                            let mut a = acc[t];
                            a = sdot_s8_lane::<0>(a, la, x16);
                            a = sdot_s8_lane::<1>(a, ha, x16);
                            a = sdot_s8_lane::<2>(a, lb, x16);
                            a = sdot_s8_lane::<3>(a, hb, x16);
                            acc[t] = a;
                        }
                    }
                }
            } else if BITS == 3 {
                for kp in 0..2 {
                    let k0 = kp * 2;
                    let k1 = k0 + 1;
                    // SAFETY: `c * 64 + 64 <= cols`; `d_ptr` holds `384` B.
                    unsafe {
                        let blk = d_ptr.add(kp * 192 + g * 48);
                        let v0 = vld1q_u8(blk);
                        let v1 = vld1q_u8(blk.add(16));
                        let v2 = vld1q_u8(blk.add(32));
                        let w0 = vqtbl1q_s8(cb, vandq_u8(v0, m07));
                        let w1 = vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8::<3>(v0), m07));
                        let w2 = vqtbl1q_s8(cb, vbslq_u8(m04, vshrq_n_u8::<4>(v2), vshrq_n_u8::<6>(v0)));
                        let w3 = vqtbl1q_s8(cb, vandq_u8(v1, m07));
                        let w4 = vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8::<3>(v1), m07));
                        let w5 = vqtbl1q_s8(cb, vbslq_u8(m04, vshrq_n_u8::<5>(v2), vshrq_n_u8::<6>(v1)));
                        let w6 = vqtbl1q_s8(cb, vandq_u8(v2, m07));
                        let w7 = vqtbl1q_s8(cb, vandq_u8(vshrq_n_u8::<3>(v2), m07));
                        let off0 = c * 64 + k0 * 16;
                        let off1 = c * 64 + k1 * 16;
                        for t in 0..NT {
                            let x0 = vld1q_s8(i8a[t].q.as_ptr().add(off0));
                            let x1 = vld1q_s8(i8a[t].q.as_ptr().add(off1));
                            let mut a = acc[t];
                            a = sdot_s8_lane::<0>(a, w0, x0);
                            a = sdot_s8_lane::<1>(a, w1, x0);
                            a = sdot_s8_lane::<2>(a, w2, x0);
                            a = sdot_s8_lane::<3>(a, w3, x0);
                            a = sdot_s8_lane::<0>(a, w4, x1);
                            a = sdot_s8_lane::<1>(a, w5, x1);
                            a = sdot_s8_lane::<2>(a, w6, x1);
                            a = sdot_s8_lane::<3>(a, w7, x1);
                            acc[t] = a;
                        }
                    }
                }
            } else {
                for kp in 0..2 {
                    let k0 = kp * 2;
                    let k1 = k0 + 1;
                    // SAFETY: `c * 64 + 64 <= cols`; `d_ptr` holds `256` B.
                    unsafe {
                        let blk = d_ptr.add(kp * 128 + g * 32);
                        let vtpl = vld1q_u8(blk);
                        let vsgn = vld1q_u8(blk.add(16));
                        let is_even = vtstq_u8(vcntq_u8(veorq_u8(vtpl, vsgn)), vdupq_n_u8(1));
                        let tpl_u = vandq_u8(vtpl, mask_0f);
                        let tpl_v = vshrq_n_u8::<4>(vtpl);
                        let v12 = vdupq_n_u8(12);
                        let u_lt12 = vcltq_u8(tpl_u, v12);
                        let v_lt12 = vcltq_u8(tpl_v, v12);
                        let u_idx = vorrq_u8(tpl_u, vandq_u8(vbicq_u8(u_lt12, v_lt12), m10));
                        let v_idx = vorrq_u8(tpl_v, vandq_u8(vbicq_u8(v_lt12, u_lt12), m10));
                        let off0 = c * 64 + k0 * 16;
                        let off1 = c * 64 + k1 * 16;
                        let (u0, u1, u2, u3) =
                            e8q2_unpack_quad_transpose::<false>(vtpl, vsgn, is_even, u_idx, e8_tbl);
                        for t in 0..NT {
                            let x0 = vld1q_s8(i8a[t].q.as_ptr().add(off0));
                            let x1 = vld1q_s8(i8a[t].q.as_ptr().add(off1));
                            let mut a = acc[t];
                            a = sdot_s8_lane::<0>(a, u0, x0);
                            a = sdot_s8_lane::<2>(a, u1, x0);
                            a = sdot_s8_lane::<0>(a, u2, x1);
                            a = sdot_s8_lane::<2>(a, u3, x1);
                            acc[t] = a;
                        }
                        let (v0, v1, v2, v3) =
                            e8q2_unpack_quad_transpose::<true>(vtpl, vsgn, is_even, v_idx, e8_tbl);
                        for t in 0..NT {
                            let x0 = vld1q_s8(i8a[t].q.as_ptr().add(off0));
                            let x1 = vld1q_s8(i8a[t].q.as_ptr().add(off1));
                            let mut a = acc[t];
                            a = sdot_s8_lane::<1>(a, v0, x0);
                            a = sdot_s8_lane::<3>(a, v1, x0);
                            a = sdot_s8_lane::<1>(a, v2, x1);
                            a = sdot_s8_lane::<3>(a, v3, x1);
                            acc[t] = a;
                        }
                    }
                }
            }
            // SAFETY: `sc_ptr` holds `32` B (`16` BF16 scales).
            let wsg = unsafe { load_bf16x4_as_f32(sc_ptr.add(g * 8)) };
            for t in 0..NT {
                let sx = i8a[t].scales[c];
                if sx != 0.0 {
                    let v = vmulq_f32(vcvtq_f32_s32(acc[t]), vdupq_n_f32(sx));
                    fg[t] = vfmaq_f32(fg[t], v, wsg);
                }
            }
        }

        for (t, &ft) in fg.iter().enumerate() {
            // SAFETY: `row_start + valid <= out_stride` for token `t`.
            unsafe {
                let dst = out_base.add(t * out_stride + row_start + g_off);
                if full_stripe || g_valid == 4 {
                    vst1q_f32(dst, ft);
                } else {
                    let mut tmp = [0.0f32; 4];
                    vst1q_f32(tmp.as_mut_ptr(), ft);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, g_valid);
                }
            }
        }
    }
}

// ============================================================================
// 2-bit 2D vector quantization (`Vq2LutMatrix`)
// ============================================================================

/// Fixed-point scale of the VQ2 codebook.
pub const VQ2_CB_SCALE: f32 = 1.0 / 64.0;

/// 16-point 2D codebook, `u` coordinates (x 64): a 4-point inner square, an 8-point ring and 4 outer
/// corners.
pub const VQ2_CB_U_I8: [i8; 16] = [
    -26, 26, -26, 26, -96, 96, -96, 96, -31, 31, -31, 31, -80, 80, -80, 80,
];

/// 16-point 2D codebook, `v` coordinates (x 64).
pub const VQ2_CB_V_I8: [i8; 16] = [
    -26, -26, 26, 26, -31, -31, 31, 31, -96, -96, 96, 96, -80, -80, 80, 80,
];

/// Upper bound on the codebook radius (`sqrt(2) * 80 / 64`), so LUT entries fit in INT8.
pub const VQ2_MAX_RADIUS: f32 = 1.7690;

/// The VQ2 codebook as `(u, v)` floats.
pub const VQ2_GAUSSIAN_RING_16: [(f32, f32); 16] = {
    let mut cb = [(0.0f32, 0.0f32); 16];
    let mut k = 0;
    while k < 16 {
        cb[k] = (
            (VQ2_CB_U_I8[k] as f32) * VQ2_CB_SCALE,
            (VQ2_CB_V_I8[k] as f32) * VQ2_CB_SCALE,
        );
        k += 1;
    }
    cb
};

/// 2D vector-quantized weights: 4-bit codes for pairs of weights plus a BF16 scale per group of
/// `scale_group_chunks` 64-col chunks. `packed_data` is `[stripe, chunk, 4-col block, row lane]`.
#[derive(Clone, Debug)]
pub struct Vq2LutMatrix {
    pub rows: usize,
    pub cols: usize,
    pub scale_group_chunks: usize,
    pub num_stripes: usize,
    pub num_chunks: usize,
    pub num_scale_groups: usize,
    pub packed_data: Vec<u8>,
    /// `BF16` scales: `[num_stripes, num_scale_groups, SIMD_ROW_STRIPE]`.
    pub bf16_chunk_scales: Vec<u16>,
}

impl Vq2LutMatrix {
    /// Packs row-major 4-bit pair codes (`[rows, cols / 2]`) into `[stripe, chunk, block16, lane16]`.
    pub fn pack_interleaved(pair_codes: &[u8], rows: usize, cols: usize) -> Vec<u8> {
        assert!(rows > 0 && cols > 0 && cols.is_multiple_of(64));
        let pairs_per_row = cols / 2;
        assert_eq!(pair_codes.len(), rows * pairs_per_row);
        let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
        let num_chunks = cols / 64;
        let mut out = vec![0u8; num_stripes * num_chunks * 256];

        for r in 0..rows {
            let stripe = r / SIMD_ROW_STRIPE;
            let lane = r % SIMD_ROW_STRIPE;
            let row_pairs = &pair_codes[r * pairs_per_row..(r + 1) * pairs_per_row];
            for c in 0..num_chunks {
                let dst_chunk = &mut out[(stripe * num_chunks + c) * 256..(stripe * num_chunks + c + 1) * 256];
                let src_c = &row_pairs[c * 32..(c + 1) * 32];
                for b in 0..16 {
                    let lo = src_c[2 * b] & 0x0F;
                    let hi = src_c[2 * b + 1] & 0x0F;
                    dst_chunk[b * SIMD_ROW_STRIPE + lane] = lo | (hi << 4);
                }
            }
        }
        out
    }

    pub fn physical_bytes(&self) -> usize {
        self.packed_data.len() + self.bf16_chunk_scales.len() * std::mem::size_of::<u16>()
    }

    pub fn physical_bits_per_weight(&self) -> f64 {
        (self.physical_bytes() as f64 * 8.0) / ((self.rows * self.cols) as f64)
    }

    /// Unpacks the 4-bit pair codes of row `r`.
    pub fn unpack_row_pair_codes(&self, r: usize, out_pairs: &mut [u8]) {
        assert!(r < self.rows);
        assert_eq!(out_pairs.len(), self.cols / 2);
        let stripe = r / SIMD_ROW_STRIPE;
        let lane = r % SIMD_ROW_STRIPE;
        for c in 0..self.num_chunks {
            let chunk_bytes = &self.packed_data[(stripe * self.num_chunks + c) * 256..(stripe * self.num_chunks + c + 1) * 256];
            for b in 0..16 {
                let byte = chunk_bytes[b * SIMD_ROW_STRIPE + lane];
                out_pairs[c * 32 + 2 * b] = byte & 0x0F;
                out_pairs[c * 32 + 2 * b + 1] = byte >> 4;
            }
        }
    }

    /// Reorders `gate_up_proj` rows from `[gate; up]` into 64-row `[gate_64(ic), up_64(ic)]` blocks.
    pub fn to_gate_up_interleaved(&self) -> Self {
        assert!(
            self.rows.is_multiple_of(128),
            "gate_up interleaving requires rows divisible by 128"
        );
        let nc_i = self.rows / 128;
        let p_blk = 4 * self.num_chunks * 256;
        let s_blk = 4 * self.num_scale_groups * SIMD_ROW_STRIPE;
        let mut new_packed = vec![0u8; self.packed_data.len()];
        let mut new_scales = vec![0u16; self.bf16_chunk_scales.len()];
        for ic in 0..nc_i {
            let gp = ic * p_blk;
            let up = (nc_i + ic) * p_blk;
            let dp = (ic * 2) * p_blk;
            new_packed[dp..dp + p_blk].copy_from_slice(&self.packed_data[gp..gp + p_blk]);
            new_packed[dp + p_blk..dp + 2 * p_blk].copy_from_slice(&self.packed_data[up..up + p_blk]);

            let gs = ic * s_blk;
            let us = (nc_i + ic) * s_blk;
            let ds = (ic * 2) * s_blk;
            new_scales[ds..ds + s_blk].copy_from_slice(&self.bf16_chunk_scales[gs..gs + s_blk]);
            new_scales[ds + s_blk..ds + 2 * s_blk]
                .copy_from_slice(&self.bf16_chunk_scales[us..us + s_blk]);
        }
        let mut out = self.clone();
        out.packed_data = new_packed;
        out.bf16_chunk_scales = new_scales;
        out
    }

    /// Dequantizes row `r` in the Hadamard-rotated domain (used for tied embeddings).
    pub fn dequantize_row_rotated(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows);
        assert_eq!(out.len(), self.cols);
        let stripe = r / SIMD_ROW_STRIPE;
        let lane = r % SIMD_ROW_STRIPE;
        let sgc = self.scale_group_chunks.max(1);
        for c in 0..self.num_chunks {
            let sg = c / sgc;
            let delta = bf16_to_f32(
                self.bf16_chunk_scales[(stripe * self.num_scale_groups + sg) * SIMD_ROW_STRIPE + lane],
            );
            let chunk_bytes = &self.packed_data[(stripe * self.num_chunks + c) * 256..(stripe * self.num_chunks + c + 1) * 256];
            let dst = &mut out[c * 64..(c + 1) * 64];
            for b in 0..16 {
                let byte = chunk_bytes[b * SIMD_ROW_STRIPE + lane];
                let (u0, v0) = VQ2_GAUSSIAN_RING_16[(byte & 0x0F) as usize];
                let (u1, v1) = VQ2_GAUSSIAN_RING_16[(byte >> 4) as usize];
                dst[4 * b] = u0 * delta;
                dst[4 * b + 1] = v0 * delta;
                dst[4 * b + 2] = u1 * delta;
                dst[4 * b + 3] = v1 * delta;
            }
        }
    }
}

/// INT8 lookup tables for [`Vq2LutMatrix`] (32 tables per 64-col chunk), plus optional INT8
/// activations for batched `SDOT`.
#[derive(Clone, Debug)]
pub struct Vq2ActivationLut {
    pub cols: usize,
    pub num_chunks: usize,
    pub lut: Vec<[i8; 16]>,
    pub chunk_scales: Vec<f32>,
    pub int8: Option<Int8ChunkActivation>,
}

impl Vq2ActivationLut {
    pub fn new_zeroed(cols: usize) -> Self {
        assert!(cols > 0 && cols.is_multiple_of(64));
        let num_chunks = cols / 64;
        Self {
            cols,
            num_chunks,
            lut: vec![[0i8; 16]; num_chunks * 32],
            chunk_scales: vec![0.0f32; num_chunks],
            int8: None,
        }
    }

    pub fn build(x_rot: &[f32]) -> Self {
        let mut v = Self::new_zeroed(x_rot.len());
        v.populate(x_rot);
        v
    }

    /// Builds the LUTs from an already rotated `x_rot`.
    #[inline]
    pub fn populate(&mut self, x_rot: &[f32]) {
        assert_eq!(x_rot.len(), self.cols);
        #[cfg(target_arch = "aarch64")]
        {
            for c in 0..self.num_chunks {
                // SAFETY: `c < num_chunks`, `lut` holds `num_chunks * 32` 16-byte tables.
                unsafe {
                    Self::populate_rotated_chunk_neon(
                        self.lut.as_mut_ptr(),
                        self.chunk_scales.as_mut_ptr(),
                        c,
                        &x_rot[c * 64..(c + 1) * 64],
                    );
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.populate_scalar(x_rot);
        }
        if let Some(i8a) = self.int8.as_mut() {
            i8a.populate(x_rot);
        }
    }

    /// Builds the LUTs from unrotated `x_unrot`, applying the FWHT in registers.
    #[inline]
    pub fn populate_fwht(&mut self, x_unrot: &[f32]) {
        assert_eq!(x_unrot.len(), self.cols);
        for c in 0..self.num_chunks {
            // SAFETY: `c < num_chunks` and `x_unrot` length is `num_chunks * 64`.
            unsafe {
                Self::populate_chunk_fwht_raw(
                    self.lut.as_mut_ptr(),
                    self.chunk_scales.as_mut_ptr(),
                    c,
                    &x_unrot[c * 64..(c + 1) * 64],
                    None,
                );
            }
        }
    }

    /// Builds the LUTs from `x_raw`, applying RMSNorm and the FWHT in registers.
    #[inline]
    pub fn populate_rmsnorm_fwht(&mut self, x_raw: &[f32], weight: &[f32], inv_rms: f32) {
        assert_eq!(x_raw.len(), self.cols);
        assert_eq!(weight.len(), self.cols);
        for c in 0..self.num_chunks {
            // SAFETY: `c < num_chunks`, `x_raw` and `weight` have length `num_chunks * 64`.
            unsafe {
                Self::populate_chunk_fwht_raw(
                    self.lut.as_mut_ptr(),
                    self.chunk_scales.as_mut_ptr(),
                    c,
                    &x_raw[c * 64..(c + 1) * 64],
                    Some((&weight[c * 64..(c + 1) * 64], inv_rms)),
                );
            }
        }
    }

    /// Builds the LUT and scale for chunk `c`, optionally applying RMSNorm, through raw pointers so
    /// workers can fill disjoint chunks.
    ///
    /// # Safety
    /// `lut_base.add(c * 32)` must be valid for `32` `[i8; 16]` writes (`512` bytes), and
    /// `scales_base.add(c)` valid for `1` `f32` write.
    #[inline]
    pub unsafe fn populate_chunk_fwht_raw(
        lut_base: *mut [i8; 16],
        scales_base: *mut f32,
        c: usize,
        x64: &[f32],
        rms_weight_inv: Option<(&[f32], f32)>,
    ) {
        debug_assert_eq!(x64.len(), 64);
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on `aarch64`; caller upholds chunk pointer bounds.
            unsafe {
                Self::populate_chunk_fwht_neon(lut_base, scales_base, c, x64, rms_weight_inv);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            // SAFETY: caller upholds chunk pointer bounds.
            unsafe {
                Self::populate_chunk_fwht_scalar(lut_base, scales_base, c, x64, rms_weight_inv);
            }
        }
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    pub fn populate_scalar(&mut self, x_rot: &[f32]) {
        assert_eq!(x_rot.len(), self.cols);
        for c in 0..self.num_chunks {
            // SAFETY: `c < self.num_chunks`.
            unsafe {
                Self::populate_rotated_chunk_scalar(
                    self.lut.as_mut_ptr(),
                    self.chunk_scales.as_mut_ptr(),
                    c,
                    &x_rot[c * 64..(c + 1) * 64],
                );
            }
        }
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    unsafe fn populate_rotated_chunk_scalar(
        lut_base: *mut [i8; 16],
        scales_base: *mut f32,
        c: usize,
        x64: &[f32],
    ) {
        let mut max_pair_norm_sq = 0.0f32;
        for m in 0..32 {
            let x0 = x64[2 * m];
            let x1 = x64[2 * m + 1];
            max_pair_norm_sq = max_pair_norm_sq.max(x0 * x0 + x1 * x1);
        }
        let max_val = VQ2_MAX_RADIUS * max_pair_norm_sq.sqrt();
        // SAFETY: caller guarantees `lut_base.add(c * 32)` and `scales_base.add(c)`.
        unsafe {
            let luts = std::slice::from_raw_parts_mut(lut_base.add(c * 32), 32);
            if max_val <= f32::EPSILON {
                luts.fill([0i8; 16]);
                *scales_base.add(c) = 0.0;
                return;
            }
            let inv = 127.0 / max_val;
            *scales_base.add(c) = max_val / 127.0;
            for (m, tbl) in luts.iter_mut().enumerate() {
                let sx0 = x64[2 * m] * inv;
                let sx1 = x64[2 * m + 1] * inv;
                for (k, &(u, v)) in VQ2_GAUSSIAN_RING_16.iter().enumerate() {
                    tbl[k] = ((u * sx0) + (v * sx1)).round_ties_even() as i32 as i8;
                }
            }
        }
    }

    #[cfg(any(not(target_arch = "aarch64"), test))]
    unsafe fn populate_chunk_fwht_scalar(
        lut_base: *mut [i8; 16],
        scales_base: *mut f32,
        c: usize,
        x64: &[f32],
        rms_weight_inv: Option<(&[f32], f32)>,
    ) {
        use crate::kernels::hadamard::{rademacher_sign_masks_64, RADEMACHER_TABLE_64};
        let fallback_masks;
        let masks = if c < RADEMACHER_TABLE_64.len() {
            &RADEMACHER_TABLE_64[c]
        } else {
            fallback_masks = rademacher_sign_masks_64(c);
            &fallback_masks
        };
        let mut block = [0.0f32; 64];
        if let Some((w64, inv_rms)) = rms_weight_inv {
            for i in 0..64 {
                let xn = (x64[i] * inv_rms) * w64[i];
                block[i] = f32::from_bits(xn.to_bits() ^ masks[i]) * 0.125;
            }
        } else {
            for i in 0..64 {
                block[i] = f32::from_bits(x64[i].to_bits() ^ masks[i]) * 0.125;
            }
        }
        let mut h = 1usize;
        while h < 64 {
            let step = h * 2;
            let mut i = 0usize;
            while i < 64 {
                for j in i..i + h {
                    let a = block[j];
                    let b = block[j + h];
                    block[j] = a + b;
                    block[j + h] = a - b;
                }
                i += step;
            }
            h *= 2;
        }
        // SAFETY: caller guarantees valid `lut_base` and `scales_base` at chunk `c`.
        unsafe { Self::populate_rotated_chunk_scalar(lut_base, scales_base, c, &block) };
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn quantize_v16_to_vq2_luts_neon(
        v: &[std::arch::aarch64::float32x4_t; 16],
        lut_chunk_ptr: *mut i8,
        scale_slot: *mut f32,
    ) {
        use std::arch::aarch64::*;
        let mut vmax_sq = vdupq_n_f32(0.0);
        for pair in v.chunks_exact(2) {
            let sq0 = vmulq_f32(pair[0], pair[0]);
            let sq1 = vmulq_f32(pair[1], pair[1]);
            vmax_sq = vmaxq_f32(vmax_sq, vpaddq_f32(sq0, sq1));
        }
        let max_val = VQ2_MAX_RADIUS * vmaxvq_f32(vmax_sq).sqrt();
        // SAFETY: `lut_chunk_ptr` points to `32 * 16 = 512` bytes, `scale_slot` to `1` `f32`.
        unsafe {
            if max_val <= f32::EPSILON {
                let z = vdupq_n_s8(0);
                for m in 0..32 {
                    vst1q_s8(lut_chunk_ptr.add(m * 16), z);
                }
                *scale_slot = 0.0;
                return;
            }
            let inv = vdupq_n_f32(127.0 / max_val);
            *scale_slot = max_val / 127.0;

            let cu_arr: [[f32; 4]; 4] = std::array::from_fn(|g| {
                std::array::from_fn(|i| VQ2_GAUSSIAN_RING_16[g * 4 + i].0)
            });
            let cv_arr: [[f32; 4]; 4] = std::array::from_fn(|g| {
                std::array::from_fn(|i| VQ2_GAUSSIAN_RING_16[g * 4 + i].1)
            });
            let cu: [float32x4_t; 4] = std::array::from_fn(|g| vld1q_f32(cu_arr[g].as_ptr()));
            let cv: [float32x4_t; 4] = std::array::from_fn(|g| vld1q_f32(cv_arr[g].as_ptr()));

            for (k, &vk) in v.iter().enumerate() {
                let sv = vmulq_f32(vk, inv);
                let mut vals = [0.0f32; 4];
                vst1q_f32(vals.as_mut_ptr(), sv);
                for sub in 0..2 {
                    let sx0 = vdupq_n_f32(vals[2 * sub]);
                    let sx1 = vdupq_n_f32(vals[2 * sub + 1]);
                    let q0 = vcvtnq_s32_f32(vaddq_f32(vmulq_f32(cu[0], sx0), vmulq_f32(cv[0], sx1)));
                    let q1 = vcvtnq_s32_f32(vaddq_f32(vmulq_f32(cu[1], sx0), vmulq_f32(cv[1], sx1)));
                    let q2 = vcvtnq_s32_f32(vaddq_f32(vmulq_f32(cu[2], sx0), vmulq_f32(cv[2], sx1)));
                    let q3 = vcvtnq_s32_f32(vaddq_f32(vmulq_f32(cu[3], sx0), vmulq_f32(cv[3], sx1)));
                    let h0 = vcombine_s16(vmovn_s32(q0), vmovn_s32(q1));
                    let h1 = vcombine_s16(vmovn_s32(q2), vmovn_s32(q3));
                    vst1q_s8(
                        lut_chunk_ptr.add((2 * k + sub) * 16),
                        vcombine_s8(vmovn_s16(h0), vmovn_s16(h1)),
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn populate_rotated_chunk_neon(
        lut_base: *mut [i8; 16],
        scales_base: *mut f32,
        c: usize,
        x64: &[f32],
    ) {
        use std::arch::aarch64::*;
        let xp = x64.as_ptr();
        // SAFETY: `x64` holds 64 floats; caller guarantees `lut_base.add(c * 32)` and `scales_base.add(c)`.
        unsafe {
            let v: [float32x4_t; 16] = std::array::from_fn(|k| vld1q_f32(xp.add(k * 4)));
            Self::quantize_v16_to_vq2_luts_neon(
                &v,
                lut_base.add(c * 32).cast::<i8>(),
                scales_base.add(c),
            );
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn populate_chunk_fwht_neon(
        lut_base: *mut [i8; 16],
        scales_base: *mut f32,
        c: usize,
        x64: &[f32],
        rms_weight_inv: Option<(&[f32], f32)>,
    ) {
        use crate::kernels::hadamard::{rademacher_sign_masks_64, RADEMACHER_TABLE_64};
        use std::arch::aarch64::*;

        let fallback_masks;
        let mptr = if c < RADEMACHER_TABLE_64.len() {
            RADEMACHER_TABLE_64[c].as_ptr()
        } else {
            fallback_masks = rademacher_sign_masks_64(c);
            fallback_masks.as_ptr()
        };
        let scale = vdupq_n_f32(0.125);
        let xp = x64.as_ptr();
        let mut v = [vdupq_n_f32(0.0); 16];

        if let Some((w64, inv_rms)) = rms_weight_inv {
            debug_assert_eq!(w64.len(), 64);
            let wp = w64.as_ptr();
            let vinv = vdupq_n_f32(inv_rms);
            for (k, slot) in v.iter_mut().enumerate() {
                // SAFETY: `k * 4 + 4 <= 64`.
                unsafe {
                    let xn = vmulq_f32(vmulq_f32(vld1q_f32(xp.add(k * 4)), vinv), vld1q_f32(wp.add(k * 4)));
                    let m = vld1q_u32(mptr.add(k * 4));
                    *slot = vmulq_f32(vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(xn), m)), scale);
                }
            }
        } else {
            for (k, slot) in v.iter_mut().enumerate() {
                // SAFETY: `k * 4 + 4 <= 64`.
                unsafe {
                    let raw = vld1q_u32(xp.add(k * 4).cast::<u32>());
                    let m = vld1q_u32(mptr.add(k * 4));
                    *slot = vmulq_f32(vreinterpretq_f32_u32(veorq_u32(raw, m)), scale);
                }
            }
        }

        let neg_mask_1 = unsafe { vld1q_u32([0u32, 0x8000_0000, 0, 0x8000_0000].as_ptr()) };
        for slot in &mut v {
            let x = *slot;
            let sum1 = vtrn1q_f32(x, x);
            let sub1 = vtrn2q_f32(x, x);
            let b1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sub1), neg_mask_1));
            let s1 = vaddq_f32(sum1, b1);
            let lo = vget_low_f32(s1);
            let hi = vget_high_f32(s1);
            *slot = vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
        }

        for vec_h in [1usize, 2, 4, 8] {
            let step = vec_h * 2;
            let mut i = 0usize;
            while i < 16 {
                for j in i..i + vec_h {
                    let a = v[j];
                    let b = v[j + vec_h];
                    v[j] = vaddq_f32(a, b);
                    v[j + vec_h] = vsubq_f32(a, b);
                }
                i += step;
            }
        }

        // SAFETY: caller guarantees `lut_base.add(c * 32)` and `scales_base.add(c)`.
        unsafe {
            Self::quantize_v16_to_vq2_luts_neon(
                &v,
                lut_base.add(c * 32).cast::<i8>(),
                scales_base.add(c),
            );
        }
    }
}

/// Runs a [`Vq2LutMatrix`] over all rows.
#[inline]
pub fn vq2_lut_gemv(matrix: &Vq2LutMatrix, vlut: &Vq2ActivationLut, out: &mut [f32]) {
    assert_eq!(matrix.cols, vlut.cols);
    assert_eq!(out.len(), matrix.rows);
    // SAFETY: full stripe range writing `0..matrix.rows`.
    unsafe {
        vq2_lut_stripe_range(matrix, vlut, 0, matrix.num_stripes, out.as_mut_ptr());
    }
}

/// Runs stripes `[stripe_start, stripe_end)` of a [`Vq2LutMatrix`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn vq2_lut_stripe_range(
    matrix: &Vq2LutMatrix,
    vlut: &Vq2ActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, vlut.cols);
    // SAFETY: caller upholds bounds.
    unsafe {
        vq2_lut_stripe_range_raw(
            matrix,
            vlut.lut.as_ptr(),
            vlut.chunk_scales.as_ptr(),
            stripe_start,
            stripe_end,
            out_base,
        );
    }
}

/// Like [`vq2_lut_stripe_range`], with LUTs and scales passed as raw pointers.
///
/// # Safety
/// `lut_ptr` valid for `matrix.num_chunks * 32` 16-byte tables, `scales_ptr` valid for `matrix.num_chunks`
/// floats, `stripe_end <= matrix.num_stripes`, and `out_base` valid for `matrix.rows` floats on disjoint ranges.
#[inline]
pub unsafe fn vq2_lut_stripe_range_raw(
    matrix: &Vq2LutMatrix,
    lut_ptr: *const [i8; 16],
    scales_ptr: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    // SAFETY: `matrix` slices match `rows`, `cols`, `scale_group_chunks`, `num_stripes`, `num_chunks`.
    unsafe {
        vq2_lut_stripe_range_slices(
            &matrix.packed_data,
            &matrix.bf16_chunk_scales,
            matrix.rows,
            matrix.num_stripes,
            matrix.num_chunks,
            matrix.scale_group_chunks,
            lut_ptr,
            scales_ptr,
            stripe_start,
            stripe_end,
            out_base,
        );
    }
}

/// `VQ2` GEMV over borrowed slices (used for memory-mapped MoE experts).
///
/// # Safety
/// `packed_data.len() == num_stripes * num_chunks * 256`,
/// `bf16_scales` holds `[num_stripes, num_chunks.div_ceil(scale_group_chunks), 16]`,
/// `lut_ptr` holds `num_chunks * 32` 16-byte tables, `scales_ptr` holds `num_chunks` floats,
/// `stripe_end <= num_stripes`, and `out_base` is valid for `rows` floats on disjoint stripe ranges.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn vq2_lut_stripe_range_slices(
    packed_data: &[u8],
    bf16_scales: &[u16],
    rows: usize,
    num_stripes: usize,
    num_chunks: usize,
    scale_group_chunks: usize,
    lut_ptr: *const [i8; 16],
    scales_ptr: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert!(stripe_end <= num_stripes);
    let sgc = scale_group_chunks.max(1);
    let nsg = num_chunks.div_ceil(sgc);
    debug_assert_eq!(packed_data.len(), num_stripes * num_chunks * 256);
    debug_assert_eq!(bf16_scales.len(), num_stripes * nsg * SIMD_ROW_STRIPE);

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64`; caller upholds bounds.
        unsafe {
            vq2_decode_raw_neon(
                packed_data.as_ptr(),
                bf16_scales.as_ptr(),
                rows,
                num_chunks,
                sgc,
                nsg,
                lut_ptr.cast::<i8>(),
                scales_ptr,
                stripe_start,
                stripe_end,
                out_base,
            );
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // SAFETY: caller upholds bounds.
        unsafe {
            vq2_decode_raw_scalar(
                packed_data,
                bf16_scales,
                rows,
                num_chunks,
                sgc,
                nsg,
                lut_ptr,
                scales_ptr,
                stripe_start,
                stripe_end,
                out_base,
            );
        }
    }
}

/// Portable scalar reference for [`vq2_lut_stripe_range`].
///
/// # Safety
/// Same preconditions as [`vq2_lut_stripe_range`].
pub unsafe fn vq2_lut_stripe_range_scalar(
    matrix: &Vq2LutMatrix,
    vlut: &Vq2ActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    let sgc = matrix.scale_group_chunks.max(1);
    let nsg = matrix.num_scale_groups;
    // SAFETY: caller upholds bounds.
    unsafe {
        vq2_decode_raw_scalar(
            &matrix.packed_data,
            &matrix.bf16_chunk_scales,
            matrix.rows,
            matrix.num_chunks,
            sgc,
            nsg,
            vlut.lut.as_ptr(),
            vlut.chunk_scales.as_ptr(),
            stripe_start,
            stripe_end,
            out_base,
        );
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn vq2_decode_raw_scalar(
    packed_data: &[u8],
    bf16_scales: &[u16],
    rows: usize,
    num_chunks: usize,
    sgc: usize,
    nsg: usize,
    lut_ptr: *const [i8; 16],
    scales_ptr: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    // SAFETY: caller guarantees `lut_ptr` and `scales_ptr` lengths.
    let luts = unsafe { std::slice::from_raw_parts(lut_ptr, num_chunks * 32) };
    let xscales = unsafe { std::slice::from_raw_parts(scales_ptr, num_chunks) };

    for s in stripe_start..stripe_end {
        let mut facc = [0.0f32; SIMD_ROW_STRIPE];
        for (c, &sx) in xscales.iter().enumerate() {
            if sx == 0.0 {
                continue;
            }
            let chunk_bytes = &packed_data[(s * num_chunks + c) * 256..(s * num_chunks + c + 1) * 256];
            let mut iacc = [0i16; SIMD_ROW_STRIPE];
            for b in 0..16 {
                let t0 = &luts[c * 32 + 2 * b];
                let t1 = &luts[c * 32 + 2 * b + 1];
                let codes = &chunk_bytes[b * SIMD_ROW_STRIPE..(b + 1) * SIMD_ROW_STRIPE];
                for lane in 0..SIMD_ROW_STRIPE {
                    let code = codes[lane];
                    iacc[lane] += (t0[(code & 0x0F) as usize] as i16) + (t1[(code >> 4) as usize] as i16);
                }
            }
            let sg = c / sgc;
            let sc_slice = &bf16_scales[(s * nsg + sg) * SIMD_ROW_STRIPE..(s * nsg + sg + 1) * SIMD_ROW_STRIPE];
            for lane in 0..SIMD_ROW_STRIPE {
                let ws = bf16_to_f32(sc_slice[lane]);
                facc[lane] += ((iacc[lane] as f32) * sx) * ws;
            }
        }
        let row_start = s * SIMD_ROW_STRIPE;
        let valid = (rows - row_start).min(SIMD_ROW_STRIPE);
        for (lane, &val) in facc.iter().take(valid).enumerate() {
            // SAFETY: `row_start + lane < rows`.
            unsafe { *out_base.add(row_start + lane) = val };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vq2_epilogue_stripe_neon(
    f_arr: &mut [std::arch::aarch64::float32x4_t; 4],
    ilo: std::arch::aarch64::int16x8_t,
    ihi: std::arch::aarch64::int16x8_t,
    vsx: std::arch::aarch64::float32x4_t,
    sc_ptr: *const u16,
) {
    use std::arch::aarch64::*;
    // SAFETY: `sc_ptr` points to 16 BF16 scales for this chunk group.
    unsafe {
        let bf0 = vld1q_u16(sc_ptr);
        let bf1 = vld1q_u16(sc_ptr.add(8));
        let ws0 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf0)));
        let ws1 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf0));
        let ws2 = vreinterpretq_f32_u32(vshll_n_u16::<16>(vget_low_u16(bf1)));
        let ws3 = vreinterpretq_f32_u32(vshll_high_n_u16::<16>(bf1));

        let d0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ilo))), vsx);
        let d1 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(ilo)), vsx);
        let d2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(ihi))), vsx);
        let d3 = vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(ihi)), vsx);

        f_arr[0] = vfmaq_f32(f_arr[0], d0, ws0);
        f_arr[1] = vfmaq_f32(f_arr[1], d1, ws1);
        f_arr[2] = vfmaq_f32(f_arr[2], d2, ws2);
        f_arr[3] = vfmaq_f32(f_arr[3], d3, ws3);
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn vq2_store_stripe_neon(
    f_arr: &[std::arch::aarch64::float32x4_t; 4],
    st_idx: usize,
    rows: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let row_start = st_idx * SIMD_ROW_STRIPE;
    let valid = (rows - row_start).min(SIMD_ROW_STRIPE);
    // SAFETY: `row_start + valid <= rows`.
    unsafe {
        let dst = out_base.add(row_start);
        if valid == SIMD_ROW_STRIPE {
            vst1q_f32(dst, f_arr[0]);
            vst1q_f32(dst.add(4), f_arr[1]);
            vst1q_f32(dst.add(8), f_arr[2]);
            vst1q_f32(dst.add(12), f_arr[3]);
        } else {
            let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
            for (g, fg) in f_arr.iter().enumerate() {
                vst1q_f32(tmp.as_mut_ptr().add(g * 4), *fg);
            }
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, valid);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vq2_decode_raw_neon(
    data_ptr: *const u8,
    scales_ptr: *const u16,
    rows: usize,
    nc: usize,
    sgc: usize,
    nsg: usize,
    lut_i8_ptr: *const i8,
    xscales_ptr: *const f32,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let mask_0f = vdupq_n_u8(0x0F);
    let mut s = stripe_start;

    // Four stripes (64 rows) at a time, reusing each LUT across all of them; 24 accumulators plus
    // scratch fit in the 32 NEON registers.
    while s + 4 <= stripe_end {
        let mut f0 = [vdupq_n_f32(0.0); 4];
        let mut f1 = [vdupq_n_f32(0.0); 4];
        let mut f2 = [vdupq_n_f32(0.0); 4];
        let mut f3 = [vdupq_n_f32(0.0); 4];

        for c in 0..nc {
            let sx = unsafe { *xscales_ptr.add(c) };
            if sx == 0.0 {
                continue;
            }
            let p0 = unsafe { data_ptr.add((s * nc + c) * 256) };
            let p1 = unsafe { data_ptr.add(((s + 1) * nc + c) * 256) };
            let p2 = unsafe { data_ptr.add(((s + 2) * nc + c) * 256) };
            let p3 = unsafe { data_ptr.add(((s + 3) * nc + c) * 256) };
            let lp = unsafe { lut_i8_ptr.add(c * 512) };

            let mut i0_lo = vdupq_n_s16(0);
            let mut i0_hi = vdupq_n_s16(0);
            let mut i1_lo = vdupq_n_s16(0);
            let mut i1_hi = vdupq_n_s16(0);
            let mut i2_lo = vdupq_n_s16(0);
            let mut i2_hi = vdupq_n_s16(0);
            let mut i3_lo = vdupq_n_s16(0);
            let mut i3_hi = vdupq_n_s16(0);

            for b in 0..16 {
                // SAFETY: `b < 16`, each chunk holds `256` weight bytes and `512` LUT bytes.
                unsafe {
                    let tbl0 = vld1q_s8(lp.add(b * 32));
                    let tbl1 = vld1q_s8(lp.add(b * 32 + 16));
                    let off = b * 16;

                    let c0 = vld1q_u8(p0.add(off));
                    let a0 = vqtbl1q_s8(tbl0, vandq_u8(c0, mask_0f));
                    let b0 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c0));
                    i0_lo = vaddq_s16(i0_lo, vaddl_s8(vget_low_s8(a0), vget_low_s8(b0)));
                    i0_hi = vaddq_s16(i0_hi, vaddl_high_s8(a0, b0));

                    let c1 = vld1q_u8(p1.add(off));
                    let a1 = vqtbl1q_s8(tbl0, vandq_u8(c1, mask_0f));
                    let b1 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c1));
                    i1_lo = vaddq_s16(i1_lo, vaddl_s8(vget_low_s8(a1), vget_low_s8(b1)));
                    i1_hi = vaddq_s16(i1_hi, vaddl_high_s8(a1, b1));

                    let c2 = vld1q_u8(p2.add(off));
                    let a2 = vqtbl1q_s8(tbl0, vandq_u8(c2, mask_0f));
                    let b2 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c2));
                    i2_lo = vaddq_s16(i2_lo, vaddl_s8(vget_low_s8(a2), vget_low_s8(b2)));
                    i2_hi = vaddq_s16(i2_hi, vaddl_high_s8(a2, b2));

                    let c3 = vld1q_u8(p3.add(off));
                    let a3 = vqtbl1q_s8(tbl0, vandq_u8(c3, mask_0f));
                    let b3 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c3));
                    i3_lo = vaddq_s16(i3_lo, vaddl_s8(vget_low_s8(a3), vget_low_s8(b3)));
                    i3_hi = vaddq_s16(i3_hi, vaddl_high_s8(a3, b3));
                }
            }

            let sg = if sgc == 1 { c } else { c / sgc };
            let vsx = vdupq_n_f32(sx);
            unsafe {
                vq2_epilogue_stripe_neon(&mut f0, i0_lo, i0_hi, vsx, scales_ptr.add((s * nsg + sg) * 16));
                vq2_epilogue_stripe_neon(&mut f1, i1_lo, i1_hi, vsx, scales_ptr.add(((s + 1) * nsg + sg) * 16));
                vq2_epilogue_stripe_neon(&mut f2, i2_lo, i2_hi, vsx, scales_ptr.add(((s + 2) * nsg + sg) * 16));
                vq2_epilogue_stripe_neon(&mut f3, i3_lo, i3_hi, vsx, scales_ptr.add(((s + 3) * nsg + sg) * 16));
            }
        }

        unsafe {
            vq2_store_stripe_neon(&f0, s, rows, out_base);
            vq2_store_stripe_neon(&f1, s + 1, rows, out_base);
            vq2_store_stripe_neon(&f2, s + 2, rows, out_base);
            vq2_store_stripe_neon(&f3, s + 3, rows, out_base);
        }
        s += 4;
    }

    // Two stripes (32 rows).
    while s + 2 <= stripe_end {
        let mut f0 = [vdupq_n_f32(0.0); 4];
        let mut f1 = [vdupq_n_f32(0.0); 4];

        for c in 0..nc {
            let sx = unsafe { *xscales_ptr.add(c) };
            if sx == 0.0 {
                continue;
            }
            let p0 = unsafe { data_ptr.add((s * nc + c) * 256) };
            let p1 = unsafe { data_ptr.add(((s + 1) * nc + c) * 256) };
            let lp = unsafe { lut_i8_ptr.add(c * 512) };

            let mut i0_lo = vdupq_n_s16(0);
            let mut i0_hi = vdupq_n_s16(0);
            let mut i1_lo = vdupq_n_s16(0);
            let mut i1_hi = vdupq_n_s16(0);

            for b in 0..16 {
                unsafe {
                    let tbl0 = vld1q_s8(lp.add(b * 32));
                    let tbl1 = vld1q_s8(lp.add(b * 32 + 16));
                    let off = b * 16;

                    let c0 = vld1q_u8(p0.add(off));
                    let a0 = vqtbl1q_s8(tbl0, vandq_u8(c0, mask_0f));
                    let b0 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c0));
                    i0_lo = vaddq_s16(i0_lo, vaddl_s8(vget_low_s8(a0), vget_low_s8(b0)));
                    i0_hi = vaddq_s16(i0_hi, vaddl_high_s8(a0, b0));

                    let c1 = vld1q_u8(p1.add(off));
                    let a1 = vqtbl1q_s8(tbl0, vandq_u8(c1, mask_0f));
                    let b1 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c1));
                    i1_lo = vaddq_s16(i1_lo, vaddl_s8(vget_low_s8(a1), vget_low_s8(b1)));
                    i1_hi = vaddq_s16(i1_hi, vaddl_high_s8(a1, b1));
                }
            }

            let sg = if sgc == 1 { c } else { c / sgc };
            let vsx = vdupq_n_f32(sx);
            unsafe {
                vq2_epilogue_stripe_neon(&mut f0, i0_lo, i0_hi, vsx, scales_ptr.add((s * nsg + sg) * 16));
                vq2_epilogue_stripe_neon(&mut f1, i1_lo, i1_hi, vsx, scales_ptr.add(((s + 1) * nsg + sg) * 16));
            }
        }

        unsafe {
            vq2_store_stripe_neon(&f0, s, rows, out_base);
            vq2_store_stripe_neon(&f1, s + 1, rows, out_base);
        }
        s += 2;
    }

    // 1-STRIPE TAIL:
    while s < stripe_end {
        let mut f0 = [vdupq_n_f32(0.0); 4];
        for c in 0..nc {
            let sx = unsafe { *xscales_ptr.add(c) };
            if sx == 0.0 {
                continue;
            }
            let p0 = unsafe { data_ptr.add((s * nc + c) * 256) };
            let lp = unsafe { lut_i8_ptr.add(c * 512) };
            let mut i0_lo = vdupq_n_s16(0);
            let mut i0_hi = vdupq_n_s16(0);

            for b in 0..16 {
                unsafe {
                    let tbl0 = vld1q_s8(lp.add(b * 32));
                    let tbl1 = vld1q_s8(lp.add(b * 32 + 16));
                    let c0 = vld1q_u8(p0.add(b * 16));
                    let a0 = vqtbl1q_s8(tbl0, vandq_u8(c0, mask_0f));
                    let b0 = vqtbl1q_s8(tbl1, vshrq_n_u8::<4>(c0));
                    i0_lo = vaddq_s16(i0_lo, vaddl_s8(vget_low_s8(a0), vget_low_s8(b0)));
                    i0_hi = vaddq_s16(i0_hi, vaddl_high_s8(a0, b0));
                }
            }

            let sg = if sgc == 1 { c } else { c / sgc };
            let vsx = vdupq_n_f32(sx);
            unsafe {
                vq2_epilogue_stripe_neon(&mut f0, i0_lo, i0_hi, vsx, scales_ptr.add((s * nsg + sg) * 16));
            }
        }
        unsafe { vq2_store_stripe_neon(&f0, s, rows, out_base) };
        s += 1;
    }
}

/// Batched `SDOT` for [`Vq2LutMatrix`]: decodes codes to INT8 weights with `vqtbl1q_s8` and reuses
/// each weight tile across up to 8 tokens.
///
/// # Safety
/// `acts[t].int8` must be `Some` with `cols == matrix.cols`, `stripe_end <= matrix.num_stripes`,
/// and `out_base` valid for `acts.len() * out_stride` floats on disjoint stripe ranges.
pub unsafe fn vq2_sdot_batch_stripe_range(
    matrix: &Vq2LutMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    if acts.is_empty() || stripe_start >= stripe_end {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: caller upholds bounds.
        unsafe {
            vq2_batch_stripe_outer_neon(matrix, acts, stripe_start, stripe_end, out_base, out_stride);
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut row = vec![0.0f32; matrix.cols];
        for s in stripe_start..stripe_end {
            for lane in 0..SIMD_ROW_STRIPE {
                let r = s * SIMD_ROW_STRIPE + lane;
                if r >= matrix.rows {
                    break;
                }
                matrix.dequantize_row_rotated(r, &mut row);
                for (t, blut) in acts.iter().enumerate() {
                    let i8a = blut.int8.as_ref().expect("int8 activations required for Vq2 batch");
                    let mut acc = 0.0f32;
                    for (c, (wc, qc)) in row.chunks_exact(64).zip(i8a.q.chunks_exact(64)).enumerate() {
                        let dot: f32 = wc.iter().zip(qc.iter()).map(|(&w, &q)| w * (q as f32)).sum();
                        acc += dot * i8a.scales[c];
                    }
                    unsafe { *out_base.add(t * out_stride + r) = acc };
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn vq2_batch_stripe_outer_neon(
    matrix: &Vq2LutMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    const C_TILE: usize = 32; // 8 KB of weights per tile
    let nc = matrix.num_chunks;
    let n_tok = acts.len();

    for s in stripe_start..stripe_end {
        let mut c_base = 0usize;
        while c_base < nc {
            let c_end = (c_base + C_TILE).min(nc);
            let first_tile = c_base == 0;
            let mut t = 0usize;
            while t + 8 <= n_tok {
                unsafe {
                    vq2_batch_tile_neon::<8>(
                        matrix,
                        &acts[t..t + 8],
                        s,
                        c_base,
                        c_end,
                        first_tile,
                        out_base.add(t * out_stride),
                        out_stride,
                    );
                }
                t += 8;
            }
            if t + 4 <= n_tok {
                unsafe {
                    vq2_batch_tile_neon::<4>(
                        matrix,
                        &acts[t..t + 4],
                        s,
                        c_base,
                        c_end,
                        first_tile,
                        out_base.add(t * out_stride),
                        out_stride,
                    );
                }
                t += 4;
            }
            if t + 2 <= n_tok {
                unsafe {
                    vq2_batch_tile_neon::<2>(
                        matrix,
                        &acts[t..t + 2],
                        s,
                        c_base,
                        c_end,
                        first_tile,
                        out_base.add(t * out_stride),
                        out_stride,
                    );
                }
                t += 2;
            }
            if t < n_tok {
                unsafe {
                    vq2_batch_tile_neon::<1>(
                        matrix,
                        &acts[t..t + 1],
                        s,
                        c_base,
                        c_end,
                        first_tile,
                        out_base.add(t * out_stride),
                        out_stride,
                    );
                }
            }
            c_base = c_end;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
unsafe fn vq2_batch_tile_neon<const NT: usize>(
    matrix: &Vq2LutMatrix,
    acts: &[BipolarActivationLut],
    s: usize,
    c_base: usize,
    c_end: usize,
    first_tile: bool,
    out_base: *mut f32,
    out_stride: usize,
) {
    use std::arch::aarch64::*;
    let i8a: [&Int8ChunkActivation; NT] =
        std::array::from_fn(|t| acts[t].int8.as_ref().expect("int8 activations required for Vq2 batch"));
    let nc = matrix.num_chunks;
    let sgc = matrix.scale_group_chunks.max(1);
    let nsg = matrix.num_scale_groups;
    let mask_0f = vdupq_n_u8(0x0F);
    let cb_u = unsafe { vld1q_s8(VQ2_CB_U_I8.as_ptr()) };
    let cb_v = unsafe { vld1q_s8(VQ2_CB_V_I8.as_ptr()) };
    let data = matrix.packed_data.as_ptr();
    let scales = matrix.bf16_chunk_scales.as_ptr();
    let row_start = s * SIMD_ROW_STRIPE;
    let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
    let full_stripe = valid == SIMD_ROW_STRIPE;

    for g in 0..4 {
        let g_off = g * 4;
        if g_off >= valid {
            break;
        }
        let g_valid = (valid - g_off).min(4);
        let mut facc: [float32x4_t; NT] = if first_tile {
            [vdupq_n_f32(0.0); NT]
        } else if full_stripe || g_valid == 4 {
            std::array::from_fn(|t| unsafe { vld1q_f32(out_base.add(t * out_stride + row_start + g_off)) })
        } else {
            std::array::from_fn(|t| unsafe {
                let mut tmp = [0.0f32; 4];
                std::ptr::copy_nonoverlapping(
                    out_base.add(t * out_stride + row_start + g_off),
                    tmp.as_mut_ptr(),
                    g_valid,
                );
                vld1q_f32(tmp.as_ptr())
            })
        };

        for c in c_base..c_end {
            let chunk_ptr = unsafe { data.add((s * nc + c) * 256 + g * 4) };
            let mut acc = [vdupq_n_s32(0); NT];
            for k in 0..4 {
                // SAFETY: `(4 * k + 3) * 16 + g * 4 + 4 <= 256`; `c * 64 + k * 16 + 16 <= cols`.
                unsafe {
                    let b_ptr = chunk_ptr.add(k * 64).cast::<u32>();
                    let mut codes32 = vdupq_n_u32(0);
                    codes32 = vld1q_lane_u32::<0>(b_ptr, codes32);
                    codes32 = vld1q_lane_u32::<1>(b_ptr.add(4), codes32);
                    codes32 = vld1q_lane_u32::<2>(b_ptr.add(8), codes32);
                    codes32 = vld1q_lane_u32::<3>(b_ptr.add(12), codes32);
                    let codes = vreinterpretq_u8_u32(codes32);

                    let lo = vandq_u8(codes, mask_0f);
                    let hi = vshrq_n_u8::<4>(codes);
                    let c0 = vqtbl1q_s8(cb_u, lo);
                    let c1 = vqtbl1q_s8(cb_v, lo);
                    let c2 = vqtbl1q_s8(cb_u, hi);
                    let c3 = vqtbl1q_s8(cb_v, hi);

                    let p01_01 = vreinterpretq_s16_s8(vzip1q_s8(c0, c1));
                    let p23_01 = vreinterpretq_s16_s8(vzip1q_s8(c2, c3));
                    let w0 = vreinterpretq_u8_s16(vzip1q_s16(p01_01, p23_01));
                    let w1 = vreinterpretq_u8_s16(vzip2q_s16(p01_01, p23_01));

                    let p01_23 = vreinterpretq_s16_s8(vzip2q_s8(c0, c1));
                    let p23_23 = vreinterpretq_s16_s8(vzip2q_s8(c2, c3));
                    let w2 = vreinterpretq_u8_s16(vzip1q_s16(p01_23, p23_23));
                    let w3 = vreinterpretq_u8_s16(vzip2q_s16(p01_23, p23_23));

                    let col_off = c * 64 + k * 16;
                    for t in 0..NT {
                        let x16 = vld1q_s8(i8a[t].q.as_ptr().add(col_off));
                        let mut a = acc[t];
                        a = sdot_lane::<0>(a, w0, x16);
                        a = sdot_lane::<1>(a, w1, x16);
                        a = sdot_lane::<2>(a, w2, x16);
                        a = sdot_lane::<3>(a, w3, x16);
                        acc[t] = a;
                    }
                }
            }
            let sg = if sgc == 1 { c } else { c / sgc };
            let ws = unsafe {
                vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(
                    scales.add((s * nsg + sg) * SIMD_ROW_STRIPE + g * 4),
                )))
            };
            for t in 0..NT {
                let sx = i8a[t].scales[c] * VQ2_CB_SCALE;
                if sx != 0.0 {
                    let v = vmulq_f32(vcvtq_f32_s32(acc[t]), vdupq_n_f32(sx));
                    facc[t] = vfmaq_f32(facc[t], v, ws);
                }
            }
        }

        for (t, &ft) in facc.iter().enumerate() {
            unsafe {
                let dst = out_base.add(t * out_stride + row_start + g_off);
                if full_stripe || g_valid == 4 {
                    vst1q_f32(dst, ft);
                } else {
                    let mut tmp = [0.0f32; 4];
                    vst1q_f32(tmp.as_mut_ptr(), ft);
                    std::ptr::copy_nonoverlapping(tmp.as_ptr(), dst, g_valid);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn test_neon_matches_scalar_bit_exact_and_fp32_high_fidelity() {
        let rows = 70;
        let cols = 524;

        let mut state = 0xCAFE_BABE_1234_5678u64;
        let weights: Vec<i8> = (0..rows * cols)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z % 3) as i8) - 1
            })
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.041).sin() * 2.5).collect();

        let packed = SimdPackedMatrix::from_ternary_weights(&weights, rows, cols);
        let qlut = QuantizedActivationLut::build(&x);

        let mut out_simd = vec![0.0f32; rows];
        let mut out_scalar = vec![0.0f32; rows];
        simd_lut_gemv(&packed, &qlut, &mut out_simd);
        simd_lut_gemv_scalar(&packed, &qlut, &mut out_scalar);

        for r in 0..rows {
            assert!(
                (out_simd[r] - out_scalar[r]).abs() < 1e-4,
                "row {r} mismatch: simd={} scalar={}",
                out_simd[r],
                out_scalar[r]
            );
        }

        let out_fp32: Vec<f32> = (0..rows)
            .map(|r| {
                weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(x.iter())
                    .map(|(&w, &v)| (w as f32) * v)
                    .sum()
            })
            .collect();

        let cos = cosine_similarity(&out_simd, &out_fp32);
        assert!(cos > 0.9998, "expected >0.9998 cosine similarity, got {cos}");
    }

    #[test]
    fn test_fused_multirank_kronecker_neon_matches_scalar_exact() {
        let rows = 86; // Non-multiple of 16 to verify partial stripe tail handling!
        let cols = 520; // Spans multiple 256-col chunks + remainder!
        let bpr = cols / GROUP_SIZE;
        let rank = 4;

        let row_seeds: Vec<u8> = (0..rank * rows).map(|i| ((i * 37 + 13) & 0x3F) as u8).collect();
        let col_seeds: Vec<u8> = (0..rank * bpr).map(|i| ((i * 59 + 7) & 0xFF) as u8).collect();
        let rank_scales = [0.6f32, -0.35, 0.2, 0.1];
        let reg_cb = default_register_codebook_64();

        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.027).cos() * 1.8).collect();
        let qlut = QuantizedActivationLut::build(&x);

        let mut out_neon = vec![0.0f32; rows];
        let mut out_scalar = vec![0.0f32; rows];

        fused_kronecker_gemv(
            &row_seeds,
            &col_seeds,
            &rank_scales,
            rows,
            &reg_cb,
            &qlut,
            &mut out_neon,
        );
        fused_kronecker_gemv_scalar(
            &row_seeds,
            &col_seeds,
            &rank_scales,
            rows,
            &reg_cb,
            &qlut,
            &mut out_scalar,
        );

        for r in 0..rows {
            assert!(
                (out_neon[r] - out_scalar[r]).abs() < 1e-4,
                "fused row {r} mismatch: neon={} scalar={}",
                out_neon[r],
                out_scalar[r]
            );
        }
    }

    #[test]
    fn test_kronecker_direct_simd_synthesis_parity() {
        use crate::kernels::lut_gemv::build_activation_lut;
        use crate::kernels::synthesis::default_canonical_codebook;

        let rows = 128;
        let cols = 512;
        let expert = KroneckerCompressedExpert::new(732_705, rows, cols);
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.019).cos()).collect();

        let cb81 = default_canonical_codebook();
        let luts_fp32 = build_activation_lut(&x);
        let mut l1_buf = vec![0u8; rows * (cols / GROUP_SIZE)];
        let mut out_fp32 = vec![0.0f32; rows];
        expert.synthesize_and_execute(&cb81, &luts_fp32, &mut l1_buf, &mut out_fp32);

        let cb_simd = default_simd_codebook();
        let qlut = QuantizedActivationLut::build(&x);
        let mut simd_tile = SimdPackedMatrix::new_zeroed(rows, cols);
        simd_tile.synthesize_from_kronecker(&expert, &cb_simd);
        let mut out_simd = vec![0.0f32; rows];
        simd_lut_gemv(&simd_tile, &qlut, &mut out_simd);

        let cos = cosine_similarity(&out_simd, &out_fp32);
        assert!(cos > 0.9998, "Kronecker SIMD vs FP32 cosine similarity: {cos}");
    }

    #[test]
    fn test_fused_base3_and_gguf_q4_0_neon_parity() {
        use crate::model::compiler::{compile_hadamard_gptq_full_row, FullRowGptqHessian};
        use crate::kernels::hadamard::fwht64_signed_inplace;

        let rows = 50; // Partial last stripe!
        let cols = 128; // 2 Hadamard-64 chunks!
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| {
                let r = idx / cols;
                let c = idx % cols;
                let v = (((r * 19 + c * 37) as f32) * 0.11).sin() * 0.3;
                // Heavy-tailed weight outliers in channels disjoint from activation outlier `c == 13`
                // (the exact real-LLM failure mode where unrotated GGUF Q4_0 zeroes out `w[13]`!)
                if c == 29 || c == 77 {
                    v * 18.0
                } else {
                    v
                }
            })
            .collect();

        let sample_vec = |seed: u64| -> Vec<f32> {
            (0..cols)
                .map(|c| {
                    let mut z = (seed + 1)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    let u = ((z >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
                    if c == 13 { u * 12.0 } else { u }
                })
                .collect()
        };

        let mut hess = FullRowGptqHessian::new(cols);
        for s in 0..256u64 {
            hess.observe_activation(&sample_vec(s));
        }
        hess.finalize(0.02);

        let m2 = compile_hadamard_gptq_full_row(&weights, rows, cols, 2, Some(&hess), 2);
        let m2_rtn = compile_hadamard_gptq_full_row(&weights, rows, cols, 2, None, 2);
        let m3 = compile_hadamard_gptq_full_row(&weights, rows, cols, 3, Some(&hess), 2);
        let m3_rtn = compile_hadamard_gptq_full_row(&weights, rows, cols, 3, None, 2);
        let q4 = GgufQ4_0Matrix::from_f32(&weights, rows, cols);

        let padded_rows = m2.num_stripes * SIMD_ROW_STRIPE;
        assert_eq!(
            (m2.physical_bytes() as f64 * 8.0) / ((padded_rows * cols) as f64),
            4.25
        );
        assert_eq!(q4.physical_bits_per_weight(), 4.50);

        let x_test = sample_vec(999);
        let mut x_rot = x_test;
        fwht64_signed_inplace(&mut x_rot);
        let qlut = QuantizedActivationLut::build(&x_rot);

        let mut out_m2_neon = vec![0.0f32; rows];
        let mut out_m2_scalar = vec![0.0f32; rows];
        let mut out_m3_neon = vec![0.0f32; rows];
        let mut out_m3_scalar = vec![0.0f32; rows];
        simd_lut_base3_gemv(&m2, &qlut, &mut out_m2_neon);
        simd_lut_base3_gemv(&m3, &qlut, &mut out_m3_neon);
        unsafe {
            simd_lut_base3_stripe_range_scalar(
                &m2,
                &qlut,
                0,
                m2.num_stripes,
                out_m2_scalar.as_mut_ptr(),
                false,
            );
            simd_lut_base3_stripe_range_scalar(
                &m3,
                &qlut,
                0,
                m3.num_stripes,
                out_m3_scalar.as_mut_ptr(),
                false,
            );
        }
        for r in 0..rows {
            assert!(
                (out_m2_neon[r] - out_m2_scalar[r]).abs() < 1e-4,
                "m2 neon vs scalar mismatch at row {r}: {} vs {}",
                out_m2_neon[r],
                out_m2_scalar[r]
            );
            assert!(
                (out_m3_neon[r] - out_m3_scalar[r]).abs() < 1e-4,
                "m3 neon vs scalar mismatch at row {r}: {} vs {}",
                out_m3_neon[r],
                out_m3_scalar[r]
            );
        }

        // Evaluate over 32 held-out test activations (`seed = 1000..1032`, strictly disjoint from `0..256`)
        let mut sum_m2 = 0.0f32;
        let mut sum_m2_rtn = 0.0f32;
        let mut sum_m3 = 0.0f32;
        let mut sum_m3_rtn = 0.0f32;
        let mut sum_q4 = 0.0f32;

        for t_seed in 1000..1032u64 {
            let xt = sample_vec(t_seed);
            let y_fp: Vec<f32> = (0..rows)
                .map(|r| {
                    weights[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(xt.iter())
                        .map(|(w, x)| w * x)
                        .sum()
                })
                .collect();
            let mut xr = xt.clone();
            fwht64_signed_inplace(&mut xr);
            let lt = QuantizedActivationLut::build(&xr);

            let mut ym2 = vec![0.0f32; rows];
            let mut ym2r = vec![0.0f32; rows];
            let mut ym3 = vec![0.0f32; rows];
            let mut ym3r = vec![0.0f32; rows];
            simd_lut_base3_gemv(&m2, &lt, &mut ym2);
            simd_lut_base3_gemv(&m2_rtn, &lt, &mut ym2r);
            simd_lut_base3_gemv(&m3, &lt, &mut ym3);
            simd_lut_base3_gemv(&m3_rtn, &lt, &mut ym3r);

            let mut qv = vec![0i8; cols];
            let mut qs = vec![0.0f32; cols / 32];
            GgufQ4_0Matrix::quantize_q8_0_activation(&xt, &mut qv, &mut qs);
            let mut yq4 = vec![0.0f32; rows];
            unsafe {
                q4.forward_row_range(&qv, &qs, 0, rows, yq4.as_mut_ptr());
            }

            sum_m2 += cosine_similarity(&ym2, &y_fp);
            sum_m2_rtn += cosine_similarity(&ym2r, &y_fp);
            sum_m3 += cosine_similarity(&ym3, &y_fp);
            sum_m3_rtn += cosine_similarity(&ym3r, &y_fp);
            sum_q4 += cosine_similarity(&yq4, &y_fp);
        }

        let cos_m2 = sum_m2 / 32.0;
        let cos_m2_rtn = sum_m2_rtn / 32.0;
        let cos_m3 = sum_m3 / 32.0;
        let cos_m3_rtn = sum_m3_rtn / 32.0;
        let cos_q4 = sum_q4 / 32.0;
        assert!(
            cos_m2 > cos_m2_rtn && cos_m2 > cos_q4,
            "GPTQ m2 ({cos_m2}) must beat both RTN ({cos_m2_rtn}) and GGUF Q4_0 ({cos_q4})"
        );
        assert!(
            cos_m3 > cos_m3_rtn && cos_m3 > cos_m2 && cos_m3 > 0.985,
            "GPTQ m3 ({cos_m3}) must beat RTN ({cos_m3_rtn}), m2 ({cos_m2}), and exceed 0.985"
        );
    }

    #[test]
    fn test_bipolar_4plane_beats_base3_2plane_and_gguf_q4_0() {
        use crate::model::compiler::{
            compile_hadamard_gptq_bipolar, compile_hadamard_gptq_full_row, FullRowGptqHessian,
        };
        use crate::kernels::hadamard::fwht64_signed_inplace;

        let rows = 64;
        let cols = 128;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| {
                let z = (idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let base = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                if idx % 128 == 19 {
                    base * 14.0
                } else {
                    base * 0.4
                }
            })
            .collect();

        let sample_x = |s: u64| -> Vec<f32> {
            (0..cols)
                .map(|c| {
                    let z = (s + 1)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                    let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let v = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                    if c == 19 {
                        v * 18.0
                    } else {
                        v
                    }
                })
                .collect()
        };

        let mut hess = FullRowGptqHessian::new(cols);
        for s in 0..192u64 {
            hess.observe_activation(&sample_x(s));
        }
        hess.finalize(0.02);

        let b4 = compile_hadamard_gptq_bipolar(&weights, rows, cols, 4, Some(&hess), 2);
        let b5 = compile_hadamard_gptq_bipolar(&weights, rows, cols, 5, Some(&hess), 2);
        let m2 = compile_hadamard_gptq_full_row(&weights, rows, cols, 2, Some(&hess), 2);
        let q4 = GgufQ4_0Matrix::from_f32(&weights, rows, cols);

        // Exact same 4.25 physical bits/weight (beats GGUF Q4_0's 4.50b!), 16 levels vs 9 levels!
        assert_eq!(b4.physical_bits_per_weight(), 4.25);
        assert_eq!(b5.physical_bits_per_weight(), 5.25);
        assert_eq!(m2.physical_bits_per_weight(), 4.25);

        let xt = sample_x(9999);
        let y_fp: Vec<f32> = (0..rows)
            .map(|r| {
                weights[r * cols..(r + 1) * cols]
                    .iter()
                    .zip(xt.iter())
                    .map(|(w, x)| w * x)
                    .sum()
            })
            .collect();

        let mut qv = vec![0i8; cols];
        let mut qs = vec![0.0f32; cols / 32];
        GgufQ4_0Matrix::quantize_q8_0_activation(&xt, &mut qv, &mut qs);
        let mut y_q4 = vec![0.0f32; rows];
        unsafe {
            q4.forward_row_range(&qv, &qs, 0, rows, y_q4.as_mut_ptr());
        }

        let mut xr = xt;
        fwht64_signed_inplace(&mut xr);
        let blut = BipolarActivationLut::build(&xr);
        let mut blut_scalar = BipolarActivationLut::new_zeroed(xr.len());
        blut_scalar.populate_scalar_d4(&xr);
        assert_eq!(blut.lut4, blut_scalar.lut4, "D4 NEON vs scalar LUT bit-exact parity!");
        let qlut = QuantizedActivationLut::build(&xr);

        let mut yb4_neon = vec![0.0f32; rows];
        let mut yb4_scalar = vec![0.0f32; rows];
        let mut yb5_neon = vec![0.0f32; rows];
        let mut ym2_neon = vec![0.0f32; rows];

        simd_lut_bipolar_gemv(&b4, &blut, &mut yb4_neon);
        simd_lut_bipolar_gemv(&b5, &blut, &mut yb5_neon);
        simd_lut_base3_gemv(&m2, &qlut, &mut ym2_neon);
        unsafe {
            simd_lut_bipolar_stripe_range_scalar(
                &b4,
                &blut,
                0,
                b4.num_stripes,
                yb4_scalar.as_mut_ptr(),
            );
        }
        for r in 0..rows {
            assert!(
                (yb4_neon[r] - yb4_scalar[r]).abs() < 1e-4,
                "b4 neon vs scalar mismatch at row {r}"
            );
        }

        let cos_b4 = cosine_similarity(&yb4_neon, &y_fp);
        let cos_b5 = cosine_similarity(&yb5_neon, &y_fp);
        let cos_m2 = cosine_similarity(&ym2_neon, &y_fp);
        let cos_q4 = cosine_similarity(&y_q4, &y_fp);
        assert!(
            cos_b4 > cos_m2 && cos_b4 > cos_q4 && cos_b4 > 0.996,
            "16-level Bipolar-4 ({cos_b4}) must beat Base3-2P ({cos_m2}) and GGUF Q4_0 ({cos_q4})!"
        );
        assert!(cos_b5 > cos_b4 && cos_b5 > 0.999);
    }

    #[test]
    fn test_bipolar_dequantize_row_matches_kernel_and_weights() {
        use crate::model::compiler::compile_hadamard_gptq_bipolar;
        use crate::kernels::hadamard::{fwht64_signed_inplace, ifwht64_signed_inplace};

        let rows = 40;
        let cols = 192;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| (((idx * 7919 + 13) % 1009) as f32 / 1009.0 - 0.5) * 0.3)
            .collect();
        let x: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.37).sin()).collect();
        let mut x_rot = x.clone();
        fwht64_signed_inplace(&mut x_rot);

        for np in [4, 5] {
            let m = compile_hadamard_gptq_bipolar(&weights, rows, cols, np, None, 2);
            let mut kernel_out = vec![0.0f32; rows];
            simd_lut_bipolar_gemv(&m, &BipolarActivationLut::build(&x_rot), &mut kernel_out);

            let mut row = vec![0.0f32; cols];
            let mut dequant_out = vec![0.0f32; rows];
            let mut recon = vec![0.0f32; rows * cols];
            for r in 0..rows {
                m.dequantize_row_rotated(r, &mut row);
                dequant_out[r] = row.iter().zip(x_rot.iter()).map(|(a, b)| a * b).sum();
                ifwht64_signed_inplace(&mut row);
                recon[r * cols..(r + 1) * cols].copy_from_slice(&row);
            }
            let cos_kernel = cosine_similarity(&kernel_out, &dequant_out);
            let cos_weights = cosine_similarity(&weights, &recon);
            assert!(cos_kernel > 0.999, "planes {np}: kernel vs dequant cos {cos_kernel}");
            assert!(cos_weights > 0.99, "planes {np}: reconstruction cos {cos_weights}");
        }
    }

    #[test]
    fn test_nibble4_sdot_matches_bipolar_lut() {
        use crate::model::compiler::compile_hadamard_gptq_bipolar;
        use crate::kernels::hadamard::fwht64_signed_inplace;

        let rows = 50;
        let cols = 256;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| (((idx * 7919 + 13) % 1009) as f32 / 1009.0 - 0.5) * 0.3)
            .collect();
        let mut x_rot: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.37).sin()).collect();
        fwht64_signed_inplace(&mut x_rot);

        for np in 4..=6 {
            let m = compile_hadamard_gptq_bipolar(&weights, rows, cols, np, None, 2);
            let n = Nibble4SdotMatrix::from_bipolar(&m);
            assert_eq!(n.physical_bytes(), m.physical_bytes());
            let (mut row_b, mut row_n) = (vec![0.0f32; cols], vec![0.0f32; cols]);
            for r in 0..rows {
                m.dequantize_row_rotated(r, &mut row_b);
                n.dequantize_row_rotated(r, &mut row_n);
                assert_eq!(row_b, row_n, "planes {np}: repack changed row {r}");
            }

            let mut blut = BipolarActivationLut::build(&x_rot);
            blut.int8 = Some(Int8ChunkActivation::new_zeroed(cols));
            blut.populate(&x_rot);
            let mut lut_out = vec![0.0f32; rows];
            simd_lut_bipolar_gemv(&m, &blut, &mut lut_out);
            let (mut neon_out, mut scalar_out) = (vec![0.0f32; rows], vec![0.0f32; rows]);
            // SAFETY: full stripe range with `rows`-sized outputs.
            unsafe {
                nibble4_sdot_stripe_range(&n, &blut, 0, n.num_stripes, neon_out.as_mut_ptr());
                nibble4_sdot_stripe_range_scalar(
                    &n,
                    blut.int8.as_ref().unwrap(),
                    0,
                    n.num_stripes,
                    scalar_out.as_mut_ptr(),
                );
            }
            let cos_scalar = cosine_similarity(&neon_out, &scalar_out);
            let cos_lut = cosine_similarity(&neon_out, &lut_out);
            assert!(cos_scalar > 0.9999, "planes {np}: neon vs scalar cos {cos_scalar}");
            assert!(cos_lut > 0.9995, "planes {np}: sdot vs LUT cos {cos_lut}");
        }
    }

    #[test]
    #[ignore = "micro-benchmark: cargo test --release -- --ignored bench_activation_populate --nocapture"]
    fn bench_activation_populate() {
        for cols in [576usize, 1536] {
            let x: Vec<f32> = (0..cols).map(|c| ((c as f32) * 0.37).sin()).collect();
            let mut lut = BipolarActivationLut::new_zeroed(cols);
            let mut i8a = Int8ChunkActivation::new_zeroed(cols);
            let iters = 20_000;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                lut.populate(std::hint::black_box(&x));
            }
            let lut_ns = t.elapsed().as_nanos() as f64 / iters as f64;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                i8a.populate(std::hint::black_box(&x));
            }
            let i8_ns = t.elapsed().as_nanos() as f64 / iters as f64;
            println!("cols {cols}: D4 LUT populate {lut_ns:.0} ns | int8 populate {i8_ns:.0} ns");
        }
    }

    #[test]
    fn test_d4_populate_neon_matches_scalar() {
        let cols = 512;
        for seed in 0..40u64 {
            let x: Vec<f32> = (0..cols)
                .map(|c| {
                    let z = (seed * 1_000_003 + c as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    let h = (z ^ (z >> 29)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let v = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                    // Exercise exact zeros, integer-valued ties and an all-zero chunk.
                    if c / 64 == 3 || c % 17 == 0 {
                        0.0
                    } else if c % 29 == 0 {
                        8.0 * v.signum()
                    } else {
                        v * (1.0 + (c % 7) as f32)
                    }
                })
                .collect();
            let mut neon = BipolarActivationLut::new_zeroed(cols);
            let mut scalar = BipolarActivationLut::new_zeroed(cols);
            neon.populate(&x);
            scalar.populate_scalar_d4(&x);
            assert_eq!(neon.chunk_scales, scalar.chunk_scales, "seed {seed}");
            assert_eq!(neon.lut4, scalar.lut4, "seed {seed}");
        }
    }

    #[test]
    fn test_int8_populate_neon_matches_scalar() {
        let cols = 256;
        let x: Vec<f32> = (0..cols)
            .map(|c| if c / 64 == 2 { 0.0 } else { ((c as f32) * 0.71).sin() * (1.0 + (c % 5) as f32) })
            .collect();
        let mut a = Int8ChunkActivation::new_zeroed(cols);
        let mut b = Int8ChunkActivation::new_zeroed(cols);
        a.populate(&x);
        b.populate_scalar(&x);
        assert_eq!(a.q, b.q);
        assert_eq!(a.scales, b.scales);
        assert_eq!(a.sums, b.sums);
    }

    #[test]
    fn test_fused_rmsnorm_fwht_int8_bit_exact() {
        use crate::kernels::hadamard::fwht64_signed_inplace;
        let cols = 1536;
        let x: Vec<f32> = (0..cols)
            .map(|c| if c / 64 == 5 { 0.0 } else { ((c as f32) * 0.43).sin() * 2.1 })
            .collect();
        let w: Vec<f32> = (0..cols).map(|c| 0.75 + ((c as f32) * 0.13).cos() * 0.4).collect();
        let inv_rms = 0.8125f32;

        // 1. Direct FWHT + INT8
        let mut x_rot = x.clone();
        fwht64_signed_inplace(&mut x_rot);
        let mut ref_a = Int8ChunkActivation::new_zeroed(cols);
        ref_a.populate(&x_rot);

        let mut fused_a = Int8ChunkActivation::new_zeroed(cols);
        fused_a.populate_fwht(&x);
        assert_eq!(fused_a.q, ref_a.q);
        assert_eq!(fused_a.scales, ref_a.scales);
        assert_eq!(fused_a.sums, ref_a.sums);

        // 2. RMSNorm + FWHT + INT8 (+ scalar parity)
        let mut x_norm: Vec<f32> = x.iter().zip(&w).map(|(&xi, &wi)| (xi * inv_rms) * wi).collect();
        fwht64_signed_inplace(&mut x_norm);
        ref_a.populate(&x_norm);

        fused_a.populate_rmsnorm_fwht(&x, &w, inv_rms);
        assert_eq!(fused_a.q, ref_a.q);
        assert_eq!(fused_a.scales, ref_a.scales);
        assert_eq!(fused_a.sums, ref_a.sums);

        let mut scalar_a = Int8ChunkActivation::new_zeroed(cols);
        for c in 0..cols / 64 {
            unsafe {
                Int8ChunkActivation::populate_chunk_fwht_scalar(
                    scalar_a.q.as_mut_ptr(),
                    scalar_a.scales.as_mut_ptr(),
                    scalar_a.sums.as_mut_ptr(),
                    c,
                    &x[c * 64..(c + 1) * 64],
                    Some((&w[c * 64..(c + 1) * 64], inv_rms)),
                );
            }
        }
        assert_eq!(fused_a.q, scalar_a.q);
        assert_eq!(fused_a.scales, scalar_a.scales);
        assert_eq!(fused_a.sums, scalar_a.sums);
    }

    #[test]
    fn test_nibble4_batch_matches_single() {
        use crate::model::compiler::compile_hadamard_gptq_bipolar;
        // Exercise non-multiple-of-32 rows, multi-tile cols (`1536 > 1024`), every `NT` tail (`15 = 8+4+2+1`),
        // and `lut_enabled = false` for all `4, 5, 6` bit-plane counts!
        let (rows, cols) = (54, 1536);
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| (((idx * 7919 + 13) % 1009) as f32 / 1009.0 - 0.5) * 0.3)
            .collect();
        for np in [4, 5, 6] {
            let n = Nibble4SdotMatrix::from_bipolar(&compile_hadamard_gptq_bipolar(&weights, rows, cols, np, None, 4));
            let acts: Vec<BipolarActivationLut> = (0..15)
                .map(|t| {
                    let x: Vec<f32> = (0..cols).map(|c| ((c * (t + 3)) as f32 * 0.11).sin()).collect();
                    let mut b = BipolarActivationLut::new_zeroed(cols);
                    b.lut_enabled = false;
                    b.int8 = Some(Int8ChunkActivation::new_zeroed(cols));
                    b.populate(&x);
                    b
                })
                .collect();
            let mut batch = vec![0.0f32; acts.len() * rows];
            // SAFETY: `batch` holds `acts.len() * rows` floats.
            unsafe { nibble4_sdot_batch_stripe_range(&n, &acts, 0, n.num_stripes, batch.as_mut_ptr(), rows) };
            for (t, a) in acts.iter().enumerate() {
                let mut one = vec![0.0f32; rows];
                // SAFETY: `one` holds `rows` floats.
                unsafe { nibble4_sdot_stripe_range(&n, a, 0, n.num_stripes, one.as_mut_ptr()) };
                for (x, y) in batch[t * rows..(t + 1) * rows].iter().zip(&one) {
                    assert!((x - y).abs() <= 1e-4 * (1.0 + y.abs()), "planes {np} token {t}: {x} vs {y}");
                }
            }
        }
    }

    fn test_acts(cols: usize, n_tok: usize) -> Vec<BipolarActivationLut> {
        (0..n_tok)
            .map(|t| {
                let x: Vec<f32> = (0..cols).map(|c| ((c * (t + 3)) as f32 * 0.11).sin()).collect();
                let mut b = BipolarActivationLut::new_zeroed(cols);
                b.int8 = Some(Int8ChunkActivation::new_zeroed(cols));
                b.populate(&x);
                b
            })
            .collect()
    }

    #[test]
    #[ignore = "micro-benchmark: cargo test --release -- --ignored bench_batch_gemm --nocapture"]
    fn bench_batch_gemm() {
        use crate::model::compiler::compile_hadamard_gptq_bipolar;
        let (rows, cols, n_tok) = (3072, 576, 64);
        let weights: Vec<f32> = (0..rows * cols).map(|i| ((i * 7919 % 1009) as f32 / 1009.0 - 0.5) * 0.3).collect();
        let n = Nibble4SdotMatrix::from_bipolar(&compile_hadamard_gptq_bipolar(&weights, rows, cols, 4, None, 8));
        let acts = test_acts(cols, n_tok);
        let mut out = vec![0.0f32; n_tok * rows];
        let macs = (rows * cols * n_tok) as f64;
        let time = |f: &mut dyn FnMut()| {
            f();
            let t = std::time::Instant::now();
            for _ in 0..20 {
                f();
            }
            macs * 20.0 / t.elapsed().as_secs_f64() / 1e9
        };
        // SAFETY: `out` holds `n_tok * rows` floats.
        let sdot = time(&mut || unsafe { nibble4_sdot_batch_stripe_range(&n, &acts, 0, n.num_stripes, out.as_mut_ptr(), rows) });
        println!("batch GEMM 1T: {sdot:.1} GMAC/s");
    }

    #[test]
    fn test_vq2_lut_neon_matches_scalar_and_beats_scalar_2bit() {
        use crate::model::compiler::{
            compile_hadamard_gptq_bipolar, compile_hadamard_gptq_full_row,
            compile_hadamard_gptq_vq2, FullRowGptqHessian,
        };
        use crate::kernels::hadamard::fwht64_signed_inplace;

        // 118 rows covers the 4-, 2- and 1-stripe paths plus a partial stripe.
        let rows = 118;
        let cols = 256; // 4 Hadamard-64 chunks
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| {
                let z = (idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let base = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                if idx % 256 == 23 || idx % 256 == 141 {
                    base * 16.0
                } else {
                    base * 0.35
                }
            })
            .collect();

        let sample_x = |s: u64| -> Vec<f32> {
            (0..cols)
                .map(|c| {
                    let z = (s + 1)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                    let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let v = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                    if c == 23 { v * 14.0 } else { v }
                })
                .collect()
        };

        let mut hess = FullRowGptqHessian::new(cols);
        for s in 0..384u64 {
            hess.observe_activation(&sample_x(s));
        }
        hess.finalize(0.015);

        // 1. Bits per weight for each group size.
        let vq64_aligned = compile_hadamard_gptq_vq2(&weights[..112 * cols], 112, cols, 1, Some(&hess), 4);
        let vq128_aligned = compile_hadamard_gptq_vq2(&weights[..112 * cols], 112, cols, 2, Some(&hess), 4);
        let vq256_aligned = compile_hadamard_gptq_vq2(&weights[..112 * cols], 112, cols, 4, Some(&hess), 4);
        assert_eq!(vq64_aligned.physical_bits_per_weight(), 2.25);
        assert_eq!(vq128_aligned.physical_bits_per_weight(), 2.125);
        assert_eq!(vq256_aligned.physical_bits_per_weight(), 2.0625);

        // 2. `Vq2ActivationLut` NEON vs scalar.
        let x_raw = sample_x(7777);
        let rms_w: Vec<f32> = (0..cols).map(|c| 0.8 + ((c as f32) * 0.17).cos() * 0.3).collect();
        let inv_rms = 0.625f32;
        let mut vlut_neon = Vq2ActivationLut::new_zeroed(cols);
        let mut vlut_scalar = Vq2ActivationLut::new_zeroed(cols);
        vlut_neon.populate_rmsnorm_fwht(&x_raw, &rms_w, inv_rms);
        for c in 0..cols / 64 {
            unsafe {
                Vq2ActivationLut::populate_chunk_fwht_scalar(
                    vlut_scalar.lut.as_mut_ptr(),
                    vlut_scalar.chunk_scales.as_mut_ptr(),
                    c,
                    &x_raw[c * 64..(c + 1) * 64],
                    Some((&rms_w[c * 64..(c + 1) * 64], inv_rms)),
                );
            }
        }
        assert_eq!(vlut_neon.chunk_scales, vlut_scalar.chunk_scales);
        assert_eq!(vlut_neon.lut, vlut_scalar.lut);

        // 3. GEMV NEON vs scalar, and batched `SDOT` vs GEMV, for each group size.
        for sgc in [1, 2, 4] {
            let vq = compile_hadamard_gptq_vq2(&weights, rows, cols, sgc, Some(&hess), 4);
            let mut y_neon = vec![0.0f32; rows];
            let mut y_scalar = vec![0.0f32; rows];
            vq2_lut_gemv(&vq, &vlut_neon, &mut y_neon);
            unsafe {
                vq2_lut_stripe_range_scalar(&vq, &vlut_neon, 0, vq.num_stripes, y_scalar.as_mut_ptr());
            }
            for r in 0..rows {
                assert!(
                    (y_neon[r] - y_scalar[r]).abs() < 1e-4,
                    "sgc={sgc} row {r} neon vs scalar mismatch: {} vs {}",
                    y_neon[r],
                    y_scalar[r]
                );
            }

            // 15 tokens covers the 8, 4, 2 and 1 token paths.
            let acts: Vec<BipolarActivationLut> = (0..15)
                .map(|t| {
                    let mut xr = sample_x(9000 + t as u64);
                    fwht64_signed_inplace(&mut xr);
                    let mut b = BipolarActivationLut::new_zeroed(cols);
                    b.lut_enabled = false;
                    b.int8 = Some(Int8ChunkActivation::new_zeroed(cols));
                    b.populate(&xr);
                    b
                })
                .collect();
            let mut y_batch = vec![0.0f32; acts.len() * rows];
            unsafe {
                vq2_sdot_batch_stripe_range(&vq, &acts, 0, vq.num_stripes, y_batch.as_mut_ptr(), rows);
            }
            for t in 0..acts.len() {
                let mut xr = sample_x(9000 + t as u64);
                fwht64_signed_inplace(&mut xr);
                let vl = Vq2ActivationLut::build(&xr);
                let mut y_one = vec![0.0f32; rows];
                vq2_lut_gemv(&vq, &vl, &mut y_one);
                let cos = cosine_similarity(&y_batch[t * rows..(t + 1) * rows], &y_one);
                assert!(cos > 0.9997, "sgc={sgc} tok={t} batch SDOT vs LUT cos={cos}");
            }
        }

        // 4. VQ2 error is lower than 2-bit Bipolar and Base3-1P.
        let vq2 = compile_hadamard_gptq_vq2(&weights, rows, cols, 1, Some(&hess), 4);
        let bip2 = compile_hadamard_gptq_bipolar(&weights, rows, cols, 2, Some(&hess), 4);
        let ter1 = compile_hadamard_gptq_full_row(&weights, rows, cols, 1, Some(&hess), 4);

        let mut cos_vq2 = 0.0f32;
        let mut cos_bip2 = 0.0f32;
        let mut cos_ter1 = 0.0f32;
        for seed in 2000..2032u64 {
            let xt = sample_x(seed);
            let y_fp: Vec<f32> = (0..rows)
                .map(|r| {
                    weights[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(xt.iter())
                        .map(|(w, x)| w * x)
                        .sum()
                })
                .collect();
            let mut xr = xt;
            fwht64_signed_inplace(&mut xr);
            let vl = Vq2ActivationLut::build(&xr);
            let bl = BipolarActivationLut::build(&xr);
            let ql = QuantizedActivationLut::build(&xr);

            let (mut yv, mut yb, mut yt) = (vec![0.0f32; rows], vec![0.0f32; rows], vec![0.0f32; rows]);
            vq2_lut_gemv(&vq2, &vl, &mut yv);
            simd_lut_bipolar_gemv(&bip2, &bl, &mut yb);
            simd_lut_base3_gemv(&ter1, &ql, &mut yt);

            cos_vq2 += cosine_similarity(&yv, &y_fp);
            cos_bip2 += cosine_similarity(&yb, &y_fp);
            cos_ter1 += cosine_similarity(&yt, &y_fp);
        }
        cos_vq2 /= 32.0;
        cos_bip2 /= 32.0;
        cos_ter1 /= 32.0;
        assert!(
            cos_vq2 > cos_bip2 && cos_vq2 > cos_ter1 && cos_vq2 > 0.965,
            "2D-Gaussian Ring VQ2 ({cos_vq2:.4}) must beat Bipolar-2 ({cos_bip2:.4}) and Ternary-1P ({cos_ter1:.4}) at 2.25 bpw!"
        );
    }

    #[test]
    fn test_nf3_nf4_sdot_neon_scalar_parity_and_quality() {
        use crate::model::compiler::{
            compile_hadamard_gptq_bipolar, compile_hadamard_gptq_nf, FullRowGptqHessian,
        };
        use crate::kernels::hadamard::fwht64_signed_inplace;

        let cols = 256usize;
        let sample_x = |seed: u64| -> Vec<f32> {
            let mut s = seed ^ 0x9E37_79B9_7F4A_7C15;
            (0..cols)
                .map(|c| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    let u = ((s >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
                    let amp = if c % 23 == 0 { 6.0 } else { 1.0 };
                    u * amp
                })
                .collect()
        };

        let mut hess = FullRowGptqHessian::new(cols);
        for seed in 1..=96u64 {
            hess.observe_activation(&sample_x(seed));
        }
        hess.finalize(0.01);

        // 1. Odd (71) and even (128) row counts.
        for rows in [71usize, 128usize] {
            let mut ws = 0x517C_C1B7_2722_0A95u64;
            let weights: Vec<f32> = (0..(rows * cols))
                .map(|idx| {
                    ws = ws.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let z1 = ((ws >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
                    ws = ws.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let z2 = ((ws >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
                    // Heavy-tailed / Gaussian sum + outlier column
                    let spike = if idx % 64 == 7 { 3.5 } else { 1.0 };
                    (z1 + z2) * 0.4 * spike
                })
                .collect();

            for bits in [2usize, 3usize, 4usize, 5usize] {
                for sgc in [1usize, 2usize] {
                    let nf = compile_hadamard_gptq_nf(&weights, rows, cols, bits, sgc, Some(&hess), 4);
                    let mut x = sample_x(4242 + (bits * 10 + sgc) as u64);
                    fwht64_signed_inplace(&mut x);
                    let mut act = Int8ChunkActivation::new_zeroed(cols);
                    act.populate(&x);

                    let mut y_neon = vec![0.0f32; rows];
                    let mut y_odd = vec![0.0f32; rows];
                    let mut y_scalar = vec![0.0f32; rows];
                    unsafe {
                        nf_sdot_stripe_range_int8(&nf, &act, 0, nf.num_stripes, y_neon.as_mut_ptr());
                        // Odd start, even pairs and odd end.
                        nf_sdot_stripe_range_int8(&nf, &act, 0, 1, y_odd.as_mut_ptr());
                        nf_sdot_stripe_range_int8(&nf, &act, 1, nf.num_stripes, y_odd.as_mut_ptr());
                        nf_sdot_stripe_range_scalar_raw(
                            &nf,
                            act.q.as_ptr(),
                            act.scales.as_ptr(),
                            0,
                            nf.num_stripes,
                            y_scalar.as_mut_ptr(),
                        );
                    }
                    for r in 0..rows {
                        let d1 = (y_neon[r] - y_scalar[r]).abs();
                        let d2 = (y_odd[r] - y_scalar[r]).abs();
                        let tol = 1e-4 * y_scalar[r].abs().max(1.0);
                        assert!(
                            d1 <= tol && d2 <= tol,
                            "rows={rows} bits={bits} sgc={sgc} r={r}: neon={} odd={} scalar={}",
                            y_neon[r],
                            y_odd[r],
                            y_scalar[r]
                        );
                    }
                    if rows == 128 {
                        let mut y_slice = vec![0.0f32; rows];
                        unsafe {
                            nf_sdot_pair_stream_range_slices(
                                &nf.pair_stream,
                                &nf.codebook,
                                rows,
                                cols,
                                bits,
                                sgc,
                                act.q.as_ptr(),
                                act.scales.as_ptr(),
                                0,
                                rows / 32,
                                y_slice.as_mut_ptr(),
                            );
                        }
                        assert_eq!(y_slice, y_neon);
                    }

                    // 15 tokens covers the 8, 4, 2 and 1 token paths.
                    let acts: Vec<BipolarActivationLut> = (0..15)
                        .map(|t| {
                            let mut xr = sample_x(7700 + t as u64);
                            fwht64_signed_inplace(&mut xr);
                            let mut b = BipolarActivationLut::new_zeroed(cols);
                            b.lut_enabled = false;
                            b.int8 = Some(Int8ChunkActivation::new_zeroed(cols));
                            b.populate(&xr);
                            b
                        })
                        .collect();
                    let mut y_batch = vec![0.0f32; acts.len() * rows];
                    unsafe {
                        nf_sdot_batch_stripe_range(
                            &nf,
                            &acts,
                            0,
                            nf.num_stripes,
                            y_batch.as_mut_ptr(),
                            rows,
                        );
                    }
                    for (t, a) in acts.iter().enumerate() {
                        let mut y1 = vec![0.0f32; rows];
                        unsafe {
                            nf_sdot_stripe_range(&nf, a, 0, nf.num_stripes, y1.as_mut_ptr());
                        }
                        for r in 0..rows {
                            let diff = (y_batch[t * rows + r] - y1[r]).abs();
                            let tol = 1e-4 * y1[r].abs().max(1.0);
                            assert!(diff <= tol, "batch mismatch tok={t} r={r}: {diff}");
                        }
                    }

                    // `to_gate_up_interleaved` with `inter == 64`.
                    if rows == 128 {
                        let nf_il = nf.to_gate_up_interleaved();
                        let mut y_il = vec![0.0f32; rows];
                        unsafe {
                            nf_sdot_stripe_range_int8(
                                &nf_il,
                                &act,
                                0,
                                nf_il.num_stripes,
                                y_il.as_mut_ptr(),
                            );
                        }
                        assert_eq!(&y_il[0..64], &y_neon[0..64]);
                        assert_eq!(&y_il[64..128], &y_neon[64..128]);
                    }
                }
            }
        }

        // 2. `gate_up` interleaving with `inter = 128`.
        let rows_gu = 256usize;
        let weights_gu: Vec<f32> = (0..(rows_gu * cols))
            .map(|i| ((i as f32) * 0.137).sin() * 0.5)
            .collect();
        let bip4 = compile_hadamard_gptq_bipolar(&weights_gu, rows_gu, cols, 4, Some(&hess), 4);
        let nib_plain = Nibble4SdotMatrix::from_bipolar_with_perm(&bip4, false);
        let nib_il = Nibble4SdotMatrix::from_bipolar_with_perm(&bip4, true);
        let nf4_plain = compile_hadamard_gptq_nf(&weights_gu, rows_gu, cols, 4, 1, Some(&hess), 4);
        let nf4_il = nf4_plain.to_gate_up_interleaved();

        let mut xr = sample_x(9999);
        fwht64_signed_inplace(&mut xr);
        let mut act = Int8ChunkActivation::new_zeroed(cols);
        act.populate(&xr);
        let (mut yp_n, mut yi_n) = (vec![0.0f32; rows_gu], vec![0.0f32; rows_gu]);
        let (mut yp_f, mut yi_f) = (vec![0.0f32; rows_gu], vec![0.0f32; rows_gu]);
        unsafe {
            nibble4_sdot_stripe_range_int8(&nib_plain, &act, 0, nib_plain.num_stripes, yp_n.as_mut_ptr());
            nibble4_sdot_stripe_range_int8(&nib_il, &act, 0, nib_il.num_stripes, yi_n.as_mut_ptr());
            nf_sdot_stripe_range_int8(&nf4_plain, &act, 0, nf4_plain.num_stripes, yp_f.as_mut_ptr());
            nf_sdot_stripe_range_int8(&nf4_il, &act, 0, nf4_il.num_stripes, yi_f.as_mut_ptr());
        }
        for ic in 0..2 {
            assert_eq!(&yi_n[ic * 128..ic * 128 + 64], &yp_n[ic * 64..(ic + 1) * 64]);
            assert_eq!(&yi_n[ic * 128 + 64..ic * 128 + 128], &yp_n[128 + ic * 64..128 + (ic + 1) * 64]);
            assert_eq!(&yi_f[ic * 128..ic * 128 + 64], &yp_f[ic * 64..(ic + 1) * 64]);
            assert_eq!(&yi_f[ic * 128 + 64..ic * 128 + 128], &yp_f[128 + ic * 64..128 + (ic + 1) * 64]);
        }
    }

    #[test]
    fn test_e8q2_gosset_lattice_optimality_and_vq2_comparison() {
        use crate::model::compiler::{
            compile_hadamard_gptq_e8q2, compile_hadamard_gptq_vq2, FullRowGptqHessian,
        };
        use crate::kernels::hadamard::fwht64_signed_inplace;
        use std::collections::HashSet;

        // 1. Exhaustive check over all 65,536 16-bit `(tpl, sgn)` Gosset `E8Q2` codewords.
        assert_eq!(e8q2_decode_8d(0, 1), [0i8; 8]);
        let mut all_codewords = Vec::with_capacity(65_536);
        let mut distinct = HashSet::with_capacity(40_000);
        let mut shell1 = HashSet::new();
        let mut shell2 = HashSet::new();
        for code in 0..65_536u32 {
            let tpl = (code & 0xFF) as u8;
            let sgn = (code >> 8) as u8;
            let q = e8q2_decode_8d(tpl, sgn);
            all_codewords.push(q);

            let mut sum16 = 0i32;
            let mut norm2 = 0i32;
            let mut all_even32 = true;
            let mut all_odd16 = true;
            for &c in &q {
                let ci = c as i32;
                assert!((-96..=96).contains(&ci) && ci % 16 == 0);
                let u = ci / 16;
                if u.rem_euclid(2) != 0 {
                    all_even32 = false;
                } else {
                    all_odd16 = false;
                }
                sum16 += u;
                norm2 += ci * ci;
            }
            assert!(
                all_even32 || all_odd16,
                "code=({tpl},{sgn}) mixed coset parity: {q:?}"
            );
            assert_eq!(
                sum16.rem_euclid(4),
                0,
                "code=({tpl},{sgn}) violates Gosset 2*E_8 sum congruence: {q:?}"
            );
            distinct.insert(q);
            if norm2 == 2048 {
                shell1.insert(q);
            } else if norm2 == 4096 {
                shell2.insert(q);
            }
        }
        assert_eq!(distinct.len(), 36_769);
        assert_eq!(shell1.len(), 240, "must cover all 240 Gosset E_8 kissing roots");
        assert_eq!(shell2.len(), 2160, "must cover all 2,160 Gosset E_8 shell-2 vectors");

        // 2. `e8q2_quantize_8d` must match 65,536-point brute-force minimum L2 error.
        let mut rng = 0xD1B5_4A32_D192_ED03u64;
        for case_idx in 0..320usize {
            let mut y = [0.0f32; 8];
            let scale = match case_idx % 4 {
                0 => 14.0,
                1 => 28.0,
                2 => 52.0,
                _ => 88.0,
            };
            for slot in &mut y {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((rng >> 40) as f32) / ((1u32 << 24) as f32) - 0.5;
                *slot = u * 2.0 * scale;
            }
            if case_idx == 0 {
                y = [0.0; 8];
            }
            let (tpl, sgn, q, sq_err) = e8q2_quantize_8d(&y);
            assert_eq!(e8q2_decode_8d(tpl, sgn), q);
            let mut brute_best = f32::INFINITY;
            for cand in &all_codewords {
                let mut d2 = 0.0f32;
                for i in 0..8 {
                    let d = y[i] - (cand[i] as f32);
                    d2 += d * d;
                }
                if d2 < brute_best {
                    brute_best = d2;
                }
            }
            assert!(
                (sq_err - brute_best).abs() <= 1e-3 * brute_best.max(1.0),
                "case {case_idx}: e8q2_quantize_8d sq_err={sq_err} > brute_best={brute_best} for y={y:?}"
            );
        }

        // 3. `E8Q2` (`2.125 bpw` @ G=128) must strictly outperform `VQ2` (`2.125 bpw` @ G=128).
        let rows = 96usize;
        let cols = 256usize;
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| {
                let z = (idx as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                let base = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                if idx % 256 == 23 || idx % 256 == 141 {
                    base * 12.0
                } else {
                    base * 0.35
                }
            })
            .collect();
        let sample_x = |s: u64| -> Vec<f32> {
            (0..cols)
                .map(|c| {
                    let z = (s + 1)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((c as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
                    let h = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    let v = (((h >> 40) as f32) / ((1u32 << 24) as f32)) - 0.5;
                    if c == 23 { v * 8.0 } else { v }
                })
                .collect()
        };
        let mut hess = FullRowGptqHessian::new(cols);
        for s in 0..256u64 {
            hess.observe_activation(&sample_x(s));
        }
        hess.finalize(0.015);

        let e8 = compile_hadamard_gptq_e8q2(&weights, rows, cols, 2, Some(&hess), 4);
        let vq = compile_hadamard_gptq_vq2(&weights, rows, cols, 2, Some(&hess), 4);
        assert_eq!(e8.bits, 2);

        let mut mse_e8 = 0.0f64;
        let mut mse_vq = 0.0f64;
        let mut row_e8 = vec![0.0f32; cols];
        let mut row_vq = vec![0.0f32; cols];
        let mut row_ref = vec![0.0f32; cols];
        for r in 0..rows {
            row_ref.copy_from_slice(&weights[r * cols..(r + 1) * cols]);
            fwht64_signed_inplace(&mut row_ref);
            e8.dequantize_row_rotated(r, &mut row_e8);
            vq.dequantize_row_rotated(r, &mut row_vq);
            for c in 0..cols {
                let de = (row_ref[c] - row_e8[c]) as f64;
                let dv = (row_ref[c] - row_vq[c]) as f64;
                mse_e8 += de * de;
                mse_vq += dv * dv;
            }
        }

        let mut cos_e8 = 0.0f32;
        let mut cos_vq = 0.0f32;
        for seed in 3000..3032u64 {
            let xt = sample_x(seed);
            let y_fp: Vec<f32> = (0..rows)
                .map(|r| {
                    weights[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(xt.iter())
                        .map(|(w, x)| w * x)
                        .sum()
                })
                .collect();
            let mut xr = xt;
            fwht64_signed_inplace(&mut xr);
            let mut act = Int8ChunkActivation::new_zeroed(cols);
            act.populate(&xr);
            let vl = Vq2ActivationLut::build(&xr);
            let mut ye = vec![0.0f32; rows];
            let mut yv = vec![0.0f32; rows];
            unsafe {
                nf_sdot_stripe_range_int8(&e8, &act, 0, e8.num_stripes, ye.as_mut_ptr());
            }
            vq2_lut_gemv(&vq, &vl, &mut yv);
            cos_e8 += cosine_similarity(&ye, &y_fp);
            cos_vq += cosine_similarity(&yv, &y_fp);
        }
        cos_e8 /= 32.0;
        cos_vq /= 32.0;
        assert!(
            mse_e8 < mse_vq && cos_e8 > cos_vq && cos_e8 > 0.975,
            "Gosset E8Q2 (mse={mse_e8:.5}, cos={cos_e8:.5}) must beat VQ2 (mse={mse_vq:.5}, cos={cos_vq:.5}) at 2.125 bpw"
        );
    }
}
