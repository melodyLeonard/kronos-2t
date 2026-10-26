//! Sub-Bit Kronecker + Base-3 Ternary Codebook Weight Synthesis.
//!
//! Instead of storing dense weight matrices per micro-expert in Flash/DRAM,
//! experts are stored as compact Kronecker row/column seed factors indexing into
//! a shared canonical L1 codebook of 4-trit micro-patterns. Weights are synthesized
//! directly inside L1 cache right before zero-multiply `LUT-GEMV` execution.

use crate::lut_gemv::{lut_gemv_execute, GROUP_SIZE, LUT_SIZE};

/// Canonical 256-entry L1 codebook mapping 8-bit Kronecker ring indices to valid
/// base-3 4-trit LUT codes in `0..81`.
pub fn default_canonical_codebook() -> [u8; 256] {
    let mut codebook = [0u8; 256];
    for (i, slot) in codebook.iter_mut().enumerate() {
        *slot = (i % LUT_SIZE) as u8;
    }
    codebook
}

/// Represents a compressed micro-expert stored as Kronecker-indexed codebook seeds.
#[derive(Clone, Debug)]
pub struct KroneckerCompressedExpert {
    pub expert_id: usize,
    pub rows: usize,
    pub blocks_per_row: usize, // cols / 4
    pub row_seeds: Vec<u8>,    // [rows]
    pub col_seeds: Vec<u8>,    // [blocks_per_row]
}

impl KroneckerCompressedExpert {
    /// Constructs deterministic Kronecker seed factors for `expert_id` of shape `[rows, cols]`.
    pub fn new(expert_id: usize, rows: usize, cols: usize) -> Self {
        assert!(rows > 0, "rows must be > 0");
        assert!(
            cols > 0 && cols.is_multiple_of(GROUP_SIZE),
            "cols ({cols}) must be > 0 and divisible by {GROUP_SIZE}"
        );

        let blocks_per_row = cols / GROUP_SIZE;
        let row_seeds = (0..rows)
            .map(|r| ((r.wrapping_mul(31).wrapping_add(expert_id)) & 0xFF) as u8)
            .collect();
        let col_seeds = (0..blocks_per_row)
            .map(|c| ((c.wrapping_mul(17).wrapping_add(expert_id >> 3)) & 0xFF) as u8)
            .collect();

        Self {
            expert_id,
            rows,
            blocks_per_row,
            row_seeds,
            col_seeds,
        }
    }

    /// Total logical parameters (`rows * cols`) synthesized by this micro-expert.
    #[inline]
    pub fn logical_params(&self) -> usize {
        self.rows * (self.blocks_per_row * GROUP_SIZE)
    }

    /// Compressed storage footprint in bytes (`row_seeds + col_seeds`).
    #[inline]
    pub fn stored_bytes(&self) -> usize {
        self.row_seeds.len() + self.col_seeds.len()
    }

    /// Effective bits per logical parameter (`stored_bits / logical_params`).
    #[inline]
    pub fn effective_bits_per_param(&self) -> f64 {
        (self.stored_bytes() * 8) as f64 / (self.logical_params() as f64)
    }

    /// Synthesizes the packed base-3 LUT codes into `l1_tile_buffer` (`[rows * blocks_per_row]`).
    pub fn synthesize_codes(&self, codebook_map: &[u8; 256], l1_tile_buffer: &mut [u8]) {
        assert_eq!(
            l1_tile_buffer.len(),
            self.rows * self.blocks_per_row,
            "l1_tile_buffer length must equal rows * blocks_per_row"
        );

        for (row_slice, &r_seed) in l1_tile_buffer
            .chunks_exact_mut(self.blocks_per_row)
            .zip(self.row_seeds.iter())
        {
            for (slot, &c_seed) in row_slice.iter_mut().zip(self.col_seeds.iter()) {
                let idx = (r_seed ^ c_seed.rotate_left(3)) as usize;
                *slot = codebook_map[idx];
            }
        }
    }

    /// Synthesizes the packed base-3 LUT codes in L1 cache and immediately runs
    /// multiplication-free `LUT-GEMV` without touching main DRAM for dense weights.
    pub fn synthesize_and_execute(
        &self,
        codebook_map: &[u8; 256],
        luts: &[[f32; LUT_SIZE]],
        l1_tile_buffer: &mut [u8],
        out: &mut [f32],
    ) {
        self.synthesize_codes(codebook_map, l1_tile_buffer);
        lut_gemv_execute(l1_tile_buffer, luts, self.rows, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut_gemv::{build_activation_lut, unpack_ternary_row};

    #[test]
    fn test_sub_bit_footprint_and_exact_execution() {
        let rows = 512;
        let cols = 1024;
        let expert = KroneckerCompressedExpert::new(412_891, rows, cols);

        // Verify sub-bit storage budget (< 0.02 bits/param for rank-1 Kronecker seeds)
        assert!(expert.effective_bits_per_param() < 0.02);

        let codebook = default_canonical_codebook();
        let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.031).sin()).collect();
        let luts = build_activation_lut(&x);

        let mut tile_buf = vec![0u8; rows * (cols / GROUP_SIZE)];
        let mut out_synth = vec![0.0f32; rows];
        expert.synthesize_and_execute(&codebook, &luts, &mut tile_buf, &mut out_synth);

        // Unpack the synthesized codes to explicit {-1, 0, +1} weights and verify FP32 dot product
        let unpacked_weights = unpack_ternary_row(&tile_buf);
        for (r, &actual) in out_synth.iter().enumerate() {
            let expected: f32 = unpacked_weights[r * cols..(r + 1) * cols]
                .iter()
                .zip(x.iter())
                .map(|(&w, &val)| (w as f32) * val)
                .sum();
            assert!((actual - expected).abs() < 1e-4);
        }
    }
}
