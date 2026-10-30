//! Fast Block-64 Orthogonal Randomized Walsh-Hadamard Transform (`FWHT-64`).
//!
//! Computes the self-inverse orthogonal transformation $Q_b = \frac{1}{8} \tilde{H}_{64} D_b$
//! (with deterministic SplitMix64 Rademacher sign flips $D_b \in \{\pm 1\}^{64}$) in-place across
//! 64-element blocks using `aarch64` NEON butterfly additions and subtractions (**zero multiplies**
//! inside the butterfly stages).
//!
//! Because $(H_{64} D_b)$ is orthogonal, applying [`fwht64_signed_inplace`] to weight rows at
//! compile time and to input activations once before LUT construction preserves exact inner products:
//! $\langle H_{64} D_b w_b, \, H_{64} D_b x_b \rangle = \langle w_b, x_b \rangle$ while eliminating
//! massive channel outliers in both weights and activations!

/// Block size of the orthogonal Walsh-Hadamard rotation (matches `QUANT_CHUNK_BLOCKS * GROUP_SIZE = 64`).
pub const HADAMARD_BLOCK: usize = 64;

/// Compile-time Rademacher sign masks for up to 256 blocks (`cols <= 16,384`).
pub(crate) static RADEMACHER_TABLE_64: [[u32; HADAMARD_BLOCK]; 256] = {
    let mut table = [[0u32; HADAMARD_BLOCK]; 256];
    let mut b = 0;
    while b < 256 {
        table[b] = rademacher_sign_masks_64(b);
        b += 1;
    }
    table
};

/// Deterministic sign-flip masks (`0x0000_0000` for `+1`, `0x8000_0000` for `-1`) for block `block_idx`.
#[inline]
pub const fn rademacher_sign_masks_64(block_idx: usize) -> [u32; HADAMARD_BLOCK] {
    let mut masks = [0u32; HADAMARD_BLOCK];
    let mut state = 0x9E37_79B9_7F4A_7C15u64
        .wrapping_add((block_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut i = 0;
    while i < HADAMARD_BLOCK {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        masks[i] = if (z & 1) != 0 { 0x8000_0000 } else { 0 };
        i += 1;
    }
    masks
}

/// Applies $x_b \leftarrow \frac{1}{8} \tilde{H}_{64} D_b x_b$ in-place to every 64-element block of `data`.
#[inline]
pub fn fwht64_signed_inplace(data: &mut [f32]) {
    assert!(
        data.len().is_multiple_of(HADAMARD_BLOCK),
        "fwht64_signed_inplace length ({}) must be a multiple of 64",
        data.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        for (b_idx, block) in data.chunks_exact_mut(HADAMARD_BLOCK).enumerate() {
            // SAFETY: NEON is baseline on `aarch64` and `block.len() == 64`.
            unsafe { fwht64_block_neon(block, b_idx) };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        fwht64_signed_inplace_scalar(data);
    }
}

/// Portable scalar reference for [`fwht64_signed_inplace`].
pub fn fwht64_signed_inplace_scalar(data: &mut [f32]) {
    assert!(data.len().is_multiple_of(HADAMARD_BLOCK));
    for (b_idx, block) in data.chunks_exact_mut(HADAMARD_BLOCK).enumerate() {
        let fallback_masks;
        let masks: &[u32; HADAMARD_BLOCK] = if b_idx < RADEMACHER_TABLE_64.len() {
            &RADEMACHER_TABLE_64[b_idx]
        } else {
            fallback_masks = rademacher_sign_masks_64(b_idx);
            &fallback_masks
        };
        for (v, &m) in block.iter_mut().zip(masks.iter()) {
            *v = f32::from_bits(v.to_bits() ^ m) * 0.125;
        }
        let mut h = 1usize;
        while h < HADAMARD_BLOCK {
            let step = h * 2;
            let mut i = 0usize;
            while i < HADAMARD_BLOCK {
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
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fwht64_block_neon(block: &mut [f32], block_idx: usize) {
    use std::arch::aarch64::*;
    debug_assert_eq!(block.len(), HADAMARD_BLOCK);

    let fallback_masks;
    let mptr = if block_idx < RADEMACHER_TABLE_64.len() {
        RADEMACHER_TABLE_64[block_idx].as_ptr()
    } else {
        fallback_masks = rademacher_sign_masks_64(block_idx);
        fallback_masks.as_ptr()
    };
    let scale = vdupq_n_f32(0.125);
    let ptr = block.as_mut_ptr();

    let mut v = [vdupq_n_f32(0.0); 16];
    let mut k = 0usize;
    while k < 16 {
        // SAFETY: `k * 4 + 4 <= 64` for both `block` and `masks`.
        unsafe {
            let raw = vld1q_u32(ptr.add(k * 4).cast::<u32>());
            let m = vld1q_u32(mptr.add(k * 4));
            v[k] = vmulq_f32(vreinterpretq_f32_u32(veorq_u32(raw, m)), scale);
        }
        k += 1;
    }

    // Intra-vector stages: stride 1 and stride 2 inside each 4-lane vector
    let neg_mask_1 = unsafe { vld1q_u32([0u32, 0x8000_0000, 0, 0x8000_0000].as_ptr()) };
    for slot in &mut v {
        let x = *slot;
        // Stride 1: [x0+x1, x0-x1, x2+x3, x2-x3]
        let sum1 = vtrn1q_f32(x, x);
        let sub1 = vtrn2q_f32(x, x);
        let b1 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(sub1), neg_mask_1));
        let s1 = vaddq_f32(sum1, b1);

        // Stride 2: [s0+s2, s1+s3, s0-s2, s1-s3]
        let lo = vget_low_f32(s1);
        let hi = vget_high_f32(s1);
        *slot = vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
    }

    // Inter-vector stages: strides 4 (vec step 1), 8 (step 2), 16 (step 4), 32 (step 8)
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

    let mut out_k = 0usize;
    while out_k < 16 {
        // SAFETY: `out_k * 4 + 4 <= 64`.
        unsafe {
            vst1q_f32(ptr.add(out_k * 4), v[out_k]);
        }
        out_k += 1;
    }
}

/// Applies the exact orthogonal inverse $y_b \leftarrow Q_b^{-1} y_b = \frac{1}{8} D_b \tilde{H}_{64} y_b$
/// in-place to every 64-element block of `data`.
///
/// Used for zero-RAM tied-embedding token lookups (`embed_tokens` reconstructed on the fly from a single
/// Hadamard-quantized `lm_head` row in `< 0.3 µs` with `0` extra bytes of resident RAM).
#[inline]
pub fn ifwht64_signed_inplace(data: &mut [f32]) {
    assert!(
        data.len().is_multiple_of(HADAMARD_BLOCK),
        "ifwht64_signed_inplace length ({}) must be a multiple of 64",
        data.len()
    );

    #[cfg(target_arch = "aarch64")]
    {
        for (b_idx, block) in data.chunks_exact_mut(HADAMARD_BLOCK).enumerate() {
            // SAFETY: NEON is baseline on `aarch64` and `block.len() == 64`.
            unsafe { ifwht64_block_neon(block, b_idx) };
        }
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        ifwht64_signed_inplace_scalar(data);
    }
}

/// Portable scalar reference for [`ifwht64_signed_inplace`].
pub fn ifwht64_signed_inplace_scalar(data: &mut [f32]) {
    assert!(data.len().is_multiple_of(HADAMARD_BLOCK));
    for (b_idx, block) in data.chunks_exact_mut(HADAMARD_BLOCK).enumerate() {
        let mut h = 1usize;
        while h < HADAMARD_BLOCK {
            let step = h * 2;
            let mut i = 0usize;
            while i < HADAMARD_BLOCK {
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
        let fallback_masks;
        let masks: &[u32; HADAMARD_BLOCK] = if b_idx < RADEMACHER_TABLE_64.len() {
            &RADEMACHER_TABLE_64[b_idx]
        } else {
            fallback_masks = rademacher_sign_masks_64(b_idx);
            &fallback_masks
        };
        for (v, &m) in block.iter_mut().zip(masks.iter()) {
            let scaled = *v * 0.125;
            *v = f32::from_bits(scaled.to_bits() ^ m);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ifwht64_block_neon(block: &mut [f32], block_idx: usize) {
    use std::arch::aarch64::*;
    debug_assert_eq!(block.len(), HADAMARD_BLOCK);

    let fallback_masks;
    let mptr = if block_idx < RADEMACHER_TABLE_64.len() {
        RADEMACHER_TABLE_64[block_idx].as_ptr()
    } else {
        fallback_masks = rademacher_sign_masks_64(block_idx);
        fallback_masks.as_ptr()
    };
    let scale = vdupq_n_f32(0.125);
    let ptr = block.as_mut_ptr();

    let mut v = [vdupq_n_f32(0.0); 16];
    let mut k = 0usize;
    while k < 16 {
        // SAFETY: `k * 4 + 4 <= 64`.
        unsafe {
            v[k] = vld1q_f32(ptr.add(k * 4));
        }
        k += 1;
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

    let mut out_k = 0usize;
    while out_k < 16 {
        // SAFETY: `out_k * 4 + 4 <= 64` for both `block` and `masks`.
        unsafe {
            let scaled = vmulq_f32(v[out_k], scale);
            let m = vld1q_u32(mptr.add(out_k * 4));
            let restored = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(scaled), m));
            vst1q_f32(ptr.add(out_k * 4), restored);
        }
        out_k += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fwht64_orthogonality_and_neon_scalar_parity() {
        let mut x: Vec<f32> = (0..128)
            .map(|i| if i == 17 { 80.0 } else { ((i as f32) * 0.37).sin() })
            .collect();
        let mut w: Vec<f32> = (0..128)
            .map(|i| if i == 99 { -25.0 } else { ((i as f32) * 0.19).cos() })
            .collect();

        let dot_before: f32 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();

        let mut x_scalar = x.clone();
        fwht64_signed_inplace(&mut x);
        fwht64_signed_inplace_scalar(&mut x_scalar);
        fwht64_signed_inplace(&mut w);

        for (a, b) in x.iter().zip(x_scalar.iter()) {
            assert!((a - b).abs() < 1e-5, "NEON vs scalar mismatch: {a} vs {b}");
        }

        // Verify massive outlier (80.0) was flattened across all 64 channels (<= ~12.0)!
        let max_after = x.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        assert!(max_after < 15.0, "expected outlier suppression, got {max_after}");

        // Verify exact inner-product invariance <Q w, Q x> == <w, x>!
        let dot_after: f32 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (dot_before - dot_after).abs() < 1e-3,
            "inner product not preserved: {dot_before} vs {dot_after}"
        );
    }
}
