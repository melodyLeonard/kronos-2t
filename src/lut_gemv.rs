//! Multiplication-Free Four-Russians Lookup-Table GEMV (`LUT-GEMV`).
//!
//! Constrains weights to the ternary ring `W in {-1, 0, +1}`. Grouping the input
//! activation vector `x` into blocks of `g = 4` elements yields `3^4 = 81` possible
//! linear combinations per block. Precomputing those 81 sums once per layer strips
//! `y = W * x` of all floating-point multiplications in the inner loop.

/// `3^4 = 81` ternary combinations per 4-element block.
pub const LUT_SIZE: usize = 81;

/// Number of ternary weights packed into a single base-3 byte code.
pub const GROUP_SIZE: usize = 4;

/// Packs a slice of ternary weights (`-1, 0, +1`) of length `N` into `N / 4` base-3 LUT codes.
///
/// Each byte encodes 4 ternary weights `w_0, w_1, w_2, w_3 in {-1, 0, +1}` mapped to
/// unsigned trits `t_i = w_i + 1 in {0, 1, 2}`:
/// `code = t_0 + 3*t_1 + 9*t_2 + 27*t_3` (range `0..81`).
pub fn pack_ternary_row(weights: &[i8]) -> Vec<u8> {
    assert!(
        weights.len().is_multiple_of(GROUP_SIZE),
        "weights length ({}) must be a multiple of GROUP_SIZE ({})",
        weights.len(),
        GROUP_SIZE
    );
    weights
        .chunks_exact(GROUP_SIZE)
        .map(|w| {
            for &val in w {
                assert!(
                    (-1..=1).contains(&val),
                    "ternary weight must be in {{-1, 0, 1}}, got {val}"
                );
            }
            let t0 = (w[0] + 1) as u8;
            let t1 = (w[1] + 1) as u8;
            let t2 = (w[2] + 1) as u8;
            let t3 = (w[3] + 1) as u8;
            t0 + 3 * t1 + 9 * t2 + 27 * t3
        })
        .collect()
}

/// Unpacks base-3 LUT codes (`0..81`) back into signed ternary weights (`-1, 0, +1`).
pub fn unpack_ternary_row(packed: &[u8]) -> Vec<i8> {
    let mut out = Vec::with_capacity(packed.len() * GROUP_SIZE);
    for &code in packed {
        assert!(
            (code as usize) < LUT_SIZE,
            "invalid base-3 ternary code {code} (must be < {LUT_SIZE})"
        );
        let mut rem = code;
        for _ in 0..GROUP_SIZE {
            let trit = (rem % 3) as i8;
            out.push(trit - 1);
            rem /= 3;
        }
    }
    out
}

/// Precomputes all 81 possible ternary linear combinations for each 4-element
/// block of the input activation vector `x` using ONLY additions and negations.
pub fn build_activation_lut(x: &[f32]) -> Vec<[f32; LUT_SIZE]> {
    assert!(
        x.len().is_multiple_of(GROUP_SIZE),
        "activation length ({}) must be a multiple of GROUP_SIZE ({})",
        x.len(),
        GROUP_SIZE
    );
    let num_blocks = x.len() / GROUP_SIZE;
    let mut luts = vec![[0.0f32; LUT_SIZE]; num_blocks];

    for (b, chunk) in x.chunks_exact(GROUP_SIZE).enumerate() {
        let (x0, x1, x2, x3) = (chunk[0], chunk[1], chunk[2], chunk[3]);
        let vals0 = [-x0, 0.0, x0];
        let vals1 = [-x1, 0.0, x1];
        let vals2 = [-x2, 0.0, x2];
        let vals3 = [-x3, 0.0, x3];

        let table = &mut luts[b];
        for (t3, &s3) in vals3.iter().enumerate() {
            for (t2, &v2) in vals2.iter().enumerate() {
                let s23 = s3 + v2;
                for (t1, &v1) in vals1.iter().enumerate() {
                    let s123 = s23 + v1;
                    let base_idx = 27 * t3 + 9 * t2 + 3 * t1;
                    table[base_idx] = s123 + vals0[0];
                    table[base_idx + 1] = s123 + vals0[1];
                    table[base_idx + 2] = s123 + vals0[2];
                }
            }
        }
    }
    luts
}

/// Executes `out = W * x` with ZERO multiplications in the inner loop.
///
/// `packed_weights` has logical shape `[rows, cols / 4]`, where each `u8` is in `0..81`.
#[inline]
pub fn lut_gemv_execute(
    packed_weights: &[u8],
    luts: &[[f32; LUT_SIZE]],
    rows: usize,
    out: &mut [f32],
) {
    let blocks_per_row = luts.len();
    assert_eq!(packed_weights.len(), rows * blocks_per_row);
    assert_eq!(out.len(), rows);

    for (out_slot, row_codes) in out
        .iter_mut()
        .zip(packed_weights.chunks_exact(blocks_per_row))
    {
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        // Unroll by 4 via safe exact-sized chunks to break FMA dependency chains
        let code_chunks = row_codes.chunks_exact(4);
        let lut_chunks = luts.chunks_exact(4);
        let code_rem = code_chunks.remainder();
        let lut_rem = lut_chunks.remainder();

        for (codes, tables) in code_chunks.zip(lut_chunks) {
            acc0 += tables[0][codes[0] as usize];
            acc1 += tables[1][codes[1] as usize];
            acc2 += tables[2][codes[2] as usize];
            acc3 += tables[3][codes[3] as usize];
        }

        for (&code, table) in code_rem.iter().zip(lut_rem.iter()) {
            acc0 += table[code as usize];
        }

        *out_slot = (acc0 + acc1) + (acc2 + acc3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_all_81_combinations() {
        let mut all_weights = Vec::with_capacity(LUT_SIZE * GROUP_SIZE);
        for t3 in -1..=1 {
            for t2 in -1..=1 {
                for t1 in -1..=1 {
                    for t0 in -1..=1 {
                        all_weights.extend_from_slice(&[t0, t1, t2, t3]);
                    }
                }
            }
        }
        let packed = pack_ternary_row(&all_weights);
        assert_eq!(packed.len(), LUT_SIZE);
        for (expected_code, &actual_code) in packed.iter().enumerate() {
            assert_eq!(expected_code as u8, actual_code);
        }
        let unpacked = unpack_ternary_row(&packed);
        assert_eq!(unpacked, all_weights);
    }

    #[test]
    fn test_lut_gemv_exact_equivalence() {
        let rows = 64;
        let cols = 132; // Not divisible by 16 to exercise both unrolled loop and remainder!
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 - 32.0) * 0.125).collect();
        let weights: Vec<i8> = (0..rows * cols)
            .map(|i| (((i * 37 + 11) % 3) as i8) - 1)
            .collect();

        let packed: Vec<u8> = weights
            .chunks_exact(cols)
            .flat_map(pack_ternary_row)
            .collect();

        let luts = build_activation_lut(&x);
        let mut out_lut = vec![0.0f32; rows];
        lut_gemv_execute(&packed, &luts, rows, &mut out_lut);

        for (r, &actual) in out_lut.iter().enumerate() {
            let expected: f32 = weights[r * cols..(r + 1) * cols]
                .iter()
                .zip(x.iter())
                .map(|(&w, &val)| (w as f32) * val)
                .sum();
            assert!(
                (actual - expected).abs() < 1e-4,
                "row {r}: actual={actual}, expected={expected}"
            );
        }
    }
}
