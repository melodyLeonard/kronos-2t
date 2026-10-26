//! Hardware SIMD Register Lookup-Table GEMV (`aarch64` ARM NEON `vqtbl1q_s8` + `vqtbl4q_u8`).
//!
//! Provides two zero-multiply execution paths:
//! 1. **Static Ternary Stripe GEMV ([`simd_lut_gemv`])**: Executes pre-packed 16-row ternary stripes
//!    using 16-byte NEON register LUTs (`vqtbl1q_s8`).
//! 2. **Register-Fused Multi-Rank Kronecker GEMV ([`fused_kronecker_gemv`])**: Holds a 64-byte
//!    canonical pattern codebook inside 4 NEON registers (`uint8x16x4_t`, `vqtbl4q_u8`) and
//!    synthesizes weights **on the fly inside SIMD registers** immediately before `vqtbl1q_s8`—
//!    eliminating 100% of L1 tile-buffer memory reads and writes!

use crate::lut_gemv::{GROUP_SIZE, LUT_SIZE};
use crate::synthesis::KroneckerCompressedExpert;

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

/// Bipolar weights for `SDOT`: `q = (2 u_lo - 15) + sum_{p>=4} 2^p s_p`. The low 4 planes are
/// nibble-packed for `sdot`; planes `4..B` stay on the LUT path.
/// Nibble layout: `[stripe, chunk, col16, row4, 2 x 16 bytes]`, byte `4 * row + t` = cols `t`, `t + 4`.
#[derive(Clone, Debug)]
pub struct Nibble4SdotMatrix {
    pub rows: usize,
    pub cols: usize,
    pub num_stripes: usize,
    pub num_chunks: usize,
    pub data: Vec<u8>,
    /// Bipolar planes `4..num_planes`.
    pub hi_planes: usize,
    pub hi_data: Vec<u8>,
    /// `BF16` chunk scales `[num_stripes, num_chunks, SIMD_ROW_STRIPE]`.
    pub bf16_chunk_scales: Vec<u16>,
}

impl Nibble4SdotMatrix {
    /// Repacks a `4..=6`-plane [`FusedBipolarLutMatrix`].
    pub fn from_bipolar(m: &FusedBipolarLutMatrix) -> Self {
        assert!((4..=6).contains(&m.num_planes), "SDOT repack requires 4..=6 planes");
        let mut data = vec![0u8; m.num_stripes * m.num_chunks * 512];
        let mut codes = vec![0u8; m.cols];
        for r in 0..m.rows {
            m.unpack_row_codes(r, &mut codes);
            let (stripe, lane) = (r / SIMD_ROW_STRIPE, r % SIMD_ROW_STRIPE);
            let (g, j) = (lane / 4, lane % 4);
            for (col, &u) in codes.iter().enumerate() {
                let (c, k, w) = (col / 64, (col % 64) / 16, col % 16);
                let half = w / 8;
                let t = w % 4;
                let byte = ((stripe * m.num_chunks + c) * 4 + k) * 128 + g * 32 + half * 16 + j * 4 + t;
                let lo = u & 0x0F;
                data[byte] |= if (w % 8) < 4 { lo } else { lo << 4 };
            }
        }
        let hi_planes = m.num_planes - 4;
        let mut hi_data = vec![0u8; m.num_stripes * m.octets_per_row * hi_planes * SIMD_ROW_STRIPE];
        for so in 0..m.num_stripes * m.octets_per_row {
            for p in 0..hi_planes {
                let src = (so * m.num_planes + 4 + p) * SIMD_ROW_STRIPE;
                let dst = (so * hi_planes + p) * SIMD_ROW_STRIPE;
                hi_data[dst..dst + SIMD_ROW_STRIPE]
                    .copy_from_slice(&m.interleaved_data[src..src + SIMD_ROW_STRIPE]);
            }
        }
        Self {
            rows: m.rows,
            cols: m.cols,
            num_stripes: m.num_stripes,
            num_chunks: m.num_chunks,
            data,
            hi_planes,
            hi_data,
            bf16_chunk_scales: m.bf16_chunk_scales.clone(),
        }
    }

    pub fn num_planes(&self) -> usize {
        4 + self.hi_planes
    }

    pub fn physical_bytes(&self) -> usize {
        self.data.len() + self.hi_data.len() + self.bf16_chunk_scales.len() * std::mem::size_of::<u16>()
    }

    /// Dequantizes row `r` into `out` in the Hadamard-rotated domain.
    pub fn dequantize_row_rotated(&self, r: usize, out: &mut [f32]) {
        assert!(r < self.rows);
        assert_eq!(out.len(), self.cols);
        let (stripe, lane) = (r / SIMD_ROW_STRIPE, r % SIMD_ROW_STRIPE);
        let (g, j) = (lane / 4, lane % 4);
        let opr = self.cols / 8;
        let max_odd = (1i32 << self.num_planes()) - 1;
        for (col, dst) in out.iter_mut().enumerate() {
            let (c, k, w) = (col / 64, (col % 64) / 16, col % 16);
            let byte = self.data[((stripe * self.num_chunks + c) * 4 + k) * 128
                + g * 32
                + (w / 8) * 16
                + j * 4
                + w % 4];
            let mut u = (if (w % 8) < 4 { byte & 0x0F } else { byte >> 4 }) as i32;
            for p in 0..self.hi_planes {
                let hb = self.hi_data
                    [((stripe * opr + col / 8) * self.hi_planes + p) * SIMD_ROW_STRIPE + lane];
                u |= (((hb >> (col % 8)) & 1) as i32) << (4 + p);
            }
            let delta = bf16_to_f32(
                self.bf16_chunk_scales[(stripe * self.num_chunks + c) * SIMD_ROW_STRIPE + lane],
            );
            *dst = ((2 * u - max_odd) as f32) * delta;
        }
    }
}

/// Executes stripes `[stripe_start, stripe_end)` of a [`Nibble4SdotMatrix`].
///
/// # Safety
/// `stripe_end <= matrix.num_stripes`, `out_base` valid for `matrix.rows` floats on disjoint ranges.
/// `blut` must carry `int8` activations, plus populated LUTs when `matrix.hi_planes > 0`.
pub unsafe fn nibble4_sdot_stripe_range(
    matrix: &Nibble4SdotMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    debug_assert_eq!(matrix.cols, blut.cols);
    debug_assert!(stripe_end <= matrix.num_stripes);
    debug_assert!(matrix.hi_planes == 0 || blut.lut_enabled);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: Apple/ARMv8.4 targets have `dotprod`; caller upholds bounds.
        unsafe { nibble4_sdot_stripe_range_neon(matrix, blut, stripe_start, stripe_end, out_base) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        // SAFETY: Caller upholds bounds.
        unsafe {
            nibble4_sdot_stripe_range_scalar(
                matrix,
                blut.int8.as_ref().expect("int8 activations"),
                stripe_start,
                stripe_end,
                out_base,
            )
        }
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
    // nibbles `0..=15` so reading it as signed bytes is exact.
    unsafe {
        core::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.4b[{l}]",
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
#[target_feature(enable = "neon")]
unsafe fn nibble4_sdot_stripe_range_neon(
    matrix: &Nibble4SdotMatrix,
    blut: &BipolarActivationLut,
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
) {
    use std::arch::aarch64::*;
    let act = blut.int8.as_ref().expect("int8 activations required for Nibble4");
    let nc = matrix.num_chunks;
    let hp = matrix.hi_planes;
    let mask = vdupq_n_u8(0x0F);
    let data = matrix.data.as_ptr();
    let scales = matrix.bf16_chunk_scales.as_ptr();
    let xq = act.q.as_ptr();

    for s in stripe_start..stripe_end {
        let mut f = [vdupq_n_f32(0.0); 4];
        for c in 0..nc {
            let sx = act.scales[c];
            if sx == 0.0 {
                continue;
            }
            // Two accumulators per 4-row group break the `sdot` dependency chain.
            let mut a = [vdupq_n_s32(0); 4];
            let mut b = [vdupq_n_s32(0); 4];
            for k in 0..4 {
                // SAFETY: `c * 64 + k * 16 + 16 <= cols`; the block offset stays inside `data`.
                unsafe {
                    let x16 = vld1q_s8(xq.add(c * 64 + k * 16));
                    let blk = data.add(((s * nc + c) * 4 + k) * 128);
                    for g in 0..4 {
                        let va = vld1q_u8(blk.add(g * 32));
                        let vb = vld1q_u8(blk.add(g * 32 + 16));
                        a[g] = sdot_lane::<0>(a[g], vandq_u8(va, mask), x16);
                        b[g] = sdot_lane::<1>(b[g], vshrq_n_u8::<4>(va), x16);
                        a[g] = sdot_lane::<2>(a[g], vandq_u8(vb, mask), x16);
                        b[g] = sdot_lane::<3>(b[g], vshrq_n_u8::<4>(vb), x16);
                    }
                }
            }
            let offset = vdupq_n_s32(15 * act.sums[c]);
            let vsx = vdupq_n_f32(sx);
            let mut lo = [vdupq_n_f32(0.0); 4];
            for g in 0..4 {
                let dot = vsubq_s32(vshlq_n_s32::<1>(vaddq_s32(a[g], b[g])), offset);
                lo[g] = vmulq_f32(vcvtq_f32_s32(dot), vsx);
            }

            if hp > 0 {
                // SAFETY: `s < num_stripes`, `c < num_chunks`; LUTs populated by the caller.
                let hi = unsafe { nibble4_hi_planes_chunk(matrix, blut, s, c) };
                for g in 0..4 {
                    lo[g] = vaddq_f32(lo[g], hi[g]);
                }
            }

            for g in 0..4 {
                // SAFETY: scale index `(s * nc + c) * 16 + g * 4 + 4 <= len`.
                unsafe {
                    let ws = vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(
                        scales.add((s * nc + c) * SIMD_ROW_STRIPE + g * 4),
                    )));
                    f[g] = vfmaq_f32(f[g], lo[g], ws);
                }
            }
        }
        let row_start = s * SIMD_ROW_STRIPE;
        let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        let mut tmp = [0.0f32; SIMD_ROW_STRIPE];
        // SAFETY: `tmp` holds 16 floats; writes stay within `row_start + valid <= rows`.
        unsafe {
            for (g, fg) in f.iter().enumerate() {
                vst1q_f32(tmp.as_mut_ptr().add(g * 4), *fg);
            }
            std::ptr::copy_nonoverlapping(tmp.as_ptr(), out_base.add(row_start), valid);
        }
    }
}

/// High bipolar planes `4..B` of stripe `s`, chunk `c`: `16 * sum_p 2^(p-4) * sum(s_p * x)` per row.
///
/// # Safety
/// `s < matrix.num_stripes`, `c < matrix.num_chunks`, `matrix.hi_planes > 0`, `blut` LUTs populated.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn nibble4_hi_planes_chunk(
    matrix: &Nibble4SdotMatrix,
    blut: &BipolarActivationLut,
    s: usize,
    c: usize,
) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    let hp = matrix.hi_planes;
    let opr = matrix.cols / 8;
    let mask = vdupq_n_u8(0x0F);
    let hi_data = matrix.hi_data.as_ptr();
    let mut h_lo = vdupq_n_s16(0);
    let mut h_hi = vdupq_n_s16(0);
    for o in c * BIPOLAR_OCTETS_PER_CHUNK..(c + 1) * BIPOLAR_OCTETS_PER_CHUNK {
        // SAFETY: `2 * o + 1 < num_quads`; plane offsets stay inside `hi_data`.
        unsafe {
            let tbl_lo = vld1q_s8(blut.lut4.as_ptr().add(2 * o).cast::<i8>());
            let tbl_hi = vld1q_s8(blut.lut4.as_ptr().add(2 * o + 1).cast::<i8>());
            for p in 0..hp {
                let codes = vld1q_u8(hi_data.add(((s * opr + o) * hp + p) * SIMD_ROW_STRIPE));
                let v_lo = vqtbl1q_s8(tbl_lo, vandq_u8(codes, mask));
                let v_hi = vqtbl1q_s8(tbl_hi, vshrq_n_u8::<4>(codes));
                let sum_lo = vaddl_s8(vget_low_s8(v_lo), vget_low_s8(v_hi));
                let sum_hi = vaddl_high_s8(v_lo, v_hi);
                if p == 0 {
                    h_lo = vaddq_s16(h_lo, sum_lo);
                    h_hi = vaddq_s16(h_hi, sum_hi);
                } else {
                    h_lo = vaddq_s16(h_lo, vshlq_n_s16::<1>(sum_lo));
                    h_hi = vaddq_s16(h_hi, vshlq_n_s16::<1>(sum_hi));
                }
            }
        }
    }
    let vls = vdupq_n_f32(16.0 * blut.chunk_scales[c]);
    [
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h_lo))), vls),
        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(h_lo)), vls),
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(h_hi))), vls),
        vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(h_hi)), vls),
    ]
}

/// Executes stripes `[stripe_start, stripe_end)` for several tokens: row `r` of token `t` (with
/// activations `acts[t]`) goes to `out_base[t * out_stride + r]`.
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
    #[cfg(target_arch = "aarch64")]
    {
        let mut t = 0;
        while t < acts.len() {
            let n = (acts.len() - t).min(4);
            // SAFETY: caller upholds bounds; each group writes its own tokens' rows.
            unsafe {
                let out = out_base.add(t * out_stride);
                let a = &acts[t..t + n];
                match n {
                    4 => nibble4_batch_neon::<4>(matrix, a, stripe_start, stripe_end, out, out_stride),
                    3 => nibble4_batch_neon::<3>(matrix, a, stripe_start, stripe_end, out, out_stride),
                    2 => nibble4_batch_neon::<2>(matrix, a, stripe_start, stripe_end, out, out_stride),
                    _ => nibble4_batch_neon::<1>(matrix, a, stripe_start, stripe_end, out, out_stride),
                }
            }
            t += n;
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
unsafe fn nibble4_batch_neon<const NT: usize>(
    matrix: &Nibble4SdotMatrix,
    acts: &[BipolarActivationLut],
    stripe_start: usize,
    stripe_end: usize,
    out_base: *mut f32,
    out_stride: usize,
) {
    use std::arch::aarch64::*;
    let i8a: [&Int8ChunkActivation; NT] =
        std::array::from_fn(|t| acts[t].int8.as_ref().expect("int8 activations required for Nibble4"));
    let nc = matrix.num_chunks;
    let mask = vdupq_n_u8(0x0F);
    let data = matrix.data.as_ptr();
    let scales = matrix.bf16_chunk_scales.as_ptr();

    for s in stripe_start..stripe_end {
        let mut f = [[0.0f32; SIMD_ROW_STRIPE]; NT];
        for c in 0..nc {
            let hi: [Option<[float32x4_t; 4]>; NT] = std::array::from_fn(|t| {
                (matrix.hi_planes > 0 && i8a[t].scales[c] != 0.0)
                    // SAFETY: `s`, `c` in range; caller populated LUTs.
                    .then(|| unsafe { nibble4_hi_planes_chunk(matrix, &acts[t], s, c) })
            });
            for g in 0..4 {
                // One 4-row group at a time keeps the per-token accumulators in registers.
                let mut acc = [vdupq_n_s32(0); NT];
                for k in 0..4 {
                    // SAFETY: `c * 64 + k * 16 + 16 <= cols`; the block offset stays inside `data`.
                    unsafe {
                        let blk = data.add(((s * nc + c) * 4 + k) * 128 + g * 32);
                        let va = vld1q_u8(blk);
                        let vb = vld1q_u8(blk.add(16));
                        let (la, ha) = (vandq_u8(va, mask), vshrq_n_u8::<4>(va));
                        let (lb, hb) = (vandq_u8(vb, mask), vshrq_n_u8::<4>(vb));
                        for t in 0..NT {
                            let x16 = vld1q_s8(i8a[t].q.as_ptr().add(c * 64 + k * 16));
                            let mut a = acc[t];
                            a = sdot_lane::<0>(a, la, x16);
                            a = sdot_lane::<1>(a, ha, x16);
                            a = sdot_lane::<2>(a, lb, x16);
                            a = sdot_lane::<3>(a, hb, x16);
                            acc[t] = a;
                        }
                    }
                }
                // SAFETY: scale index `(s * nc + c) * 16 + g * 4 + 4 <= len`.
                let ws = unsafe {
                    vreinterpretq_f32_u32(vshll_n_u16::<16>(vld1_u16(
                        scales.add((s * nc + c) * SIMD_ROW_STRIPE + g * 4),
                    )))
                };
                for t in 0..NT {
                    let sx = i8a[t].scales[c];
                    if sx == 0.0 {
                        continue;
                    }
                    let dot = vsubq_s32(vshlq_n_s32::<1>(acc[t]), vdupq_n_s32(15 * i8a[t].sums[c]));
                    let mut v = vmulq_f32(vcvtq_f32_s32(dot), vdupq_n_f32(sx));
                    if let Some(h) = hi[t] {
                        v = vaddq_f32(v, h[g]);
                    }
                    // SAFETY: `f[t]` holds 16 floats.
                    unsafe {
                        let fp = f[t].as_mut_ptr().add(g * 4);
                        vst1q_f32(fp, vfmaq_f32(vld1q_f32(fp), v, ws));
                    }
                }
            }
        }
        let row_start = s * SIMD_ROW_STRIPE;
        let valid = (matrix.rows - row_start).min(SIMD_ROW_STRIPE);
        for (t, ft) in f.iter().enumerate() {
            // SAFETY: rows `row_start..row_start + valid` of token `t` are inside the caller's output.
            unsafe {
                std::ptr::copy_nonoverlapping(ft.as_ptr(), out_base.add(t * out_stride + row_start), valid);
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
        use crate::lut_gemv::build_activation_lut;
        use crate::synthesis::default_canonical_codebook;

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
        use crate::compiler::{compile_hadamard_gptq_full_row, FullRowGptqHessian};
        use crate::hadamard::fwht64_signed_inplace;

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
        use crate::compiler::{
            compile_hadamard_gptq_bipolar, compile_hadamard_gptq_full_row, FullRowGptqHessian,
        };
        use crate::hadamard::fwht64_signed_inplace;

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
        use crate::compiler::compile_hadamard_gptq_bipolar;
        use crate::hadamard::{fwht64_signed_inplace, ifwht64_signed_inplace};

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
        use crate::compiler::compile_hadamard_gptq_bipolar;
        use crate::hadamard::fwht64_signed_inplace;

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
    fn test_nibble4_batch_matches_single() {
        use crate::compiler::compile_hadamard_gptq_bipolar;
        let (rows, cols) = (40, 192);
        let weights: Vec<f32> = (0..rows * cols)
            .map(|idx| (((idx * 7919 + 13) % 1009) as f32 / 1009.0 - 0.5) * 0.3)
            .collect();
        for np in [4, 5, 6] {
            let n = Nibble4SdotMatrix::from_bipolar(&compile_hadamard_gptq_bipolar(&weights, rows, cols, np, None, 2));
            let acts: Vec<BipolarActivationLut> = (0..7)
                .map(|t| {
                    let x: Vec<f32> = (0..cols).map(|c| ((c * (t + 3)) as f32 * 0.11).sin()).collect();
                    let mut b = BipolarActivationLut::new_zeroed(cols);
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
        use crate::compiler::compile_hadamard_gptq_bipolar;
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
}
