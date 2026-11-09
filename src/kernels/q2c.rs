//! Tencent `q2_0c` 2-bit weights run as stored: codes `{-3,-1,1,3}` with one scale per 512 weights,
//! no Hadamard rotation (it would break the 4-level grid), so inputs are plain INT8 chunks.
//!
//! `w = d * (2c - 3)` with `c in 0..4`, so `w . x = d * (2 * (c . x) - 3 * sum(x))`: unsigned codes go
//! straight into `SDOT` and the `-3` term uses the per-chunk activation sums.

use crate::kernels::simd_lut::{BipolarActivationLut, Int8ChunkActivation, PageAlignedBytes};

pub const Q2C_BLOCK: usize = 512;
const CHUNKS_PER_BLOCK: usize = Q2C_BLOCK / 64;
/// Max `cols / 512` the decode kernel keeps on the stack.
const MAX_BLOCKS: usize = 64;

/// Row-major 2-bit codes, then LE `f32` scales (`[rows, cols / 512]`, exact from f16).
/// Each 64-weight chunk is 16 bytes; bits `2k..2k+2` of byte `i` hold weight `16k + i`.
#[derive(Clone, Debug)]
pub struct Q2cMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: PageAlignedBytes,
}

impl Q2cMatrix {
    #[inline]
    pub const fn code_bytes(rows: usize, cols: usize) -> usize {
        rows * cols / 4
    }

    #[inline]
    pub const fn total_bytes(rows: usize, cols: usize) -> usize {
        Self::code_bytes(rows, cols) + rows * (cols / Q2C_BLOCK) * 4
    }

    #[inline]
    pub const fn num_blocks(&self) -> usize {
        self.cols / Q2C_BLOCK
    }

    pub fn physical_bytes(&self) -> usize {
        self.data.len()
    }

    /// Wraps an existing stream (copied or mapped) laid out as [`Self::total_bytes`].
    pub fn from_stream(rows: usize, cols: usize, data: PageAlignedBytes) -> Result<Self, String> {
        if cols == 0 || !cols.is_multiple_of(Q2C_BLOCK) || cols / Q2C_BLOCK > MAX_BLOCKS {
            return Err(format!("q2c: bad cols {cols}"));
        }
        if data.len() != Self::total_bytes(rows, cols) {
            return Err(format!("q2c: stream is {} bytes, expected {}", data.len(), Self::total_bytes(rows, cols)));
        }
        Ok(Self { rows, cols, data })
    }

    /// From codes `0..4` (`[rows, cols]`) and per-512 scales (`[rows, cols / 512]`).
    pub fn from_codes(codes: &[u8], scales: &[f32], rows: usize, cols: usize) -> Self {
        assert!(cols.is_multiple_of(Q2C_BLOCK) && cols / Q2C_BLOCK <= MAX_BLOCKS);
        assert_eq!(codes.len(), rows * cols);
        assert_eq!(scales.len(), rows * cols / Q2C_BLOCK);
        let mut data = PageAlignedBytes::zeroed(Self::total_bytes(rows, cols));
        let cb = Self::code_bytes(rows, cols);
        for (chunk, dst) in codes.chunks_exact(64).zip(data[..cb].chunks_exact_mut(16)) {
            for (i, d) in dst.iter_mut().enumerate() {
                *d = (0..4).fold(0u8, |acc, k| acc | ((chunk[16 * k + i] & 3) << (2 * k)));
            }
        }
        for (s, dst) in scales.iter().zip(data[cb..].chunks_exact_mut(4)) {
            dst.copy_from_slice(&s.to_le_bytes());
        }
        Self { rows, cols, data }
    }

    /// From raw ggml `block_q2_0c` blocks (`f16 d`, 128 code bytes, weight `j` at bits
    /// `2 * (j % 4)` of byte `j / 4`), row-major `[rows, cols / 512]`.
    pub fn from_q2_0c_blocks(blocks: &[u8], rows: usize, cols: usize) -> Self {
        const BB: usize = 2 + Q2C_BLOCK / 4;
        assert!(cols.is_multiple_of(Q2C_BLOCK) && cols / Q2C_BLOCK <= MAX_BLOCKS);
        let nb = rows * cols / Q2C_BLOCK;
        assert_eq!(blocks.len(), nb * BB);
        let mut data = PageAlignedBytes::zeroed(Self::total_bytes(rows, cols));
        let cb = Self::code_bytes(rows, cols);
        let (codes, scales) = data.split_at_mut(cb);
        for (bi, blk) in blocks.chunks_exact(BB).enumerate() {
            let d = crate::storage::gguf::f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            scales[bi * 4..bi * 4 + 4].copy_from_slice(&d.to_le_bytes());
            let qs = &blk[2..];
            let code = |j: usize| (qs[j / 4] >> (2 * (j % 4))) & 3;
            let dst = &mut codes[bi * (Q2C_BLOCK / 4)..(bi + 1) * (Q2C_BLOCK / 4)];
            for c in 0..CHUNKS_PER_BLOCK {
                for i in 0..16 {
                    dst[c * 16 + i] = (0..4).fold(0u8, |acc, k| acc | (code(c * 64 + 16 * k + i) << (2 * k)));
                }
            }
        }
        Self { rows, cols, data }
    }

    #[inline]
    pub fn scale(&self, r: usize, b: usize) -> f32 {
        let off = Self::code_bytes(self.rows, self.cols) + (r * self.num_blocks() + b) * 4;
        f32::from_le_bytes(self.data[off..off + 4].try_into().unwrap())
    }

    #[inline]
    fn code(&self, r: usize, col: usize) -> u8 {
        let (c, w) = (col / 64, col % 64);
        (self.data[r * self.cols / 4 + c * 16 + w % 16] >> (2 * (w / 16))) & 3
    }

    pub fn dequantize_row(&self, r: usize, out: &mut [f32]) {
        assert_eq!(out.len(), self.cols);
        for (col, o) in out.iter_mut().enumerate() {
            *o = self.scale(r, col / Q2C_BLOCK) * (2.0 * self.code(r, col) as f32 - 3.0);
        }
    }

    /// Full-precision `f32` weights (tests, calibration).
    pub fn to_f32(&self) -> Vec<f32> {
        let mut w = vec![0.0f32; self.rows * self.cols];
        for (r, row) in w.chunks_exact_mut(self.cols).enumerate() {
            self.dequantize_row(r, row);
        }
        w
    }
}

/// `sum_{c in block b} scale[c] * sum[c]` per 512-block.
#[inline]
fn block_sums(act: &Int8ChunkActivation, out: &mut [f32]) {
    for (b, o) in out.iter_mut().enumerate() {
        let c0 = b * CHUNKS_PER_BLOCK;
        *o = (c0..c0 + CHUNKS_PER_BLOCK).map(|c| act.scales[c] * act.sums[c] as f32).sum();
    }
}

/// Rows `[r0, r1)` of `m · x` for one activation into `out[r]`.
///
/// # Safety
/// `r1 <= m.rows`; `out` valid for writes at `r0..r1` (disjoint across concurrent callers).
pub unsafe fn q2c_gemv_rows(m: &Q2cMatrix, act: &Int8ChunkActivation, r0: usize, r1: usize, out: *mut f32) {
    debug_assert_eq!(act.cols, m.cols);
    let mut tb = [0.0f32; MAX_BLOCKS];
    block_sums(act, &mut tb[..m.num_blocks()]);
    // SAFETY: caller upholds bounds.
    unsafe { q2c_rows::<1>(m, [act], [&tb[..]], r0, r1, [out]) };
}

/// Rows `[r0, r1)` for every token: `out[t * out_stride + r]`.
///
/// # Safety
/// `r1 <= m.rows`; `out` valid for `acts.len() * out_stride` floats, rows disjoint across callers.
pub unsafe fn q2c_batch_rows(
    m: &Q2cMatrix,
    acts: &[BipolarActivationLut],
    r0: usize,
    r1: usize,
    out: *mut f32,
    out_stride: usize,
) {
    let nb = m.num_blocks();
    let i8s: Vec<&Int8ChunkActivation> = acts.iter().map(|a| a.int8.as_ref().expect("int8 activations")).collect();
    let mut tb = vec![0.0f32; acts.len() * nb];
    for (a, t) in i8s.iter().zip(tb.chunks_exact_mut(nb)) {
        block_sums(a, t);
    }
    // 8-row groups keep the codes in L1 across token tiles.
    let mut rg = r0;
    while rg < r1 {
        let re = (rg + 8).min(r1);
        let mut t = 0;
        // SAFETY: caller upholds bounds; token `t` writes row `t` of `out`.
        unsafe {
            while t + 4 <= i8s.len() {
                q2c_rows::<4>(
                    m,
                    std::array::from_fn(|j| i8s[t + j]),
                    std::array::from_fn(|j| &tb[(t + j) * nb..(t + j + 1) * nb]),
                    rg,
                    re,
                    std::array::from_fn(|j| out.add((t + j) * out_stride)),
                );
                t += 4;
            }
            while t < i8s.len() {
                q2c_rows::<1>(m, [i8s[t]], [&tb[t * nb..(t + 1) * nb]], rg, re, [out.add(t * out_stride)]);
                t += 1;
            }
        }
        rg = re;
    }
}

#[inline]
unsafe fn q2c_rows<const NT: usize>(
    m: &Q2cMatrix,
    acts: [&Int8ChunkActivation; NT],
    tb: [&[f32]; NT],
    r0: usize,
    r1: usize,
    out: [*mut f32; NT],
) {
    debug_assert!(r1 <= m.rows);
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON + dotprod on every Apple / ARMv8.4 target; caller upholds bounds.
    unsafe {
        q2c_rows_neon::<NT>(m, acts, tb, r0, r1, out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    // SAFETY: caller upholds bounds.
    unsafe {
        q2c_rows_scalar::<NT>(m, acts, tb, r0, r1, out)
    }
}

/// Portable reference for [`q2c_rows_neon`].
///
/// # Safety
/// Same as [`q2c_gemv_rows`].
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
unsafe fn q2c_rows_scalar<const NT: usize>(
    m: &Q2cMatrix,
    acts: [&Int8ChunkActivation; NT],
    tb: [&[f32]; NT],
    r0: usize,
    r1: usize,
    out: [*mut f32; NT],
) {
    for r in r0..r1 {
        for t in 0..NT {
            let mut acc = 0.0f32;
            for b in 0..m.num_blocks() {
                let mut s = 0.0f32;
                for c in b * CHUNKS_PER_BLOCK..(b + 1) * CHUNKS_PER_BLOCK {
                    let dot: i32 = (0..64).map(|w| m.code(r, c * 64 + w) as i32 * acts[t].q[c * 64 + w] as i32).sum();
                    s += dot as f32 * acts[t].scales[c];
                }
                acc += m.scale(r, b) * (2.0 * s - 3.0 * tb[t][b]);
            }
            // SAFETY: caller upholds bounds.
            unsafe { *out[t].add(r) = acc };
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn sdot16(
    mut acc: std::arch::aarch64::int32x4_t,
    a: std::arch::aarch64::uint8x16_t,
    b: std::arch::aarch64::int8x16_t,
) -> std::arch::aarch64::int32x4_t {
    // SAFETY: `sdot` (FEAT_DotProd) is on all Apple silicon; codes `0..4` read as signed are exact.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod\nsdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn q2c_rows_neon<const NT: usize>(
    m: &Q2cMatrix,
    acts: [&Int8ChunkActivation; NT],
    tb: [&[f32]; NT],
    r0: usize,
    r1: usize,
    out: [*mut f32; NT],
) {
    use std::arch::aarch64::*;
    let nb = m.num_blocks();
    let row_bytes = m.cols / 4;
    let base = m.data.as_ptr();
    let scale_base = unsafe { base.add(Q2cMatrix::code_bytes(m.rows, m.cols)) };
    let m3 = vdupq_n_u8(3);
    let xq: [*const i8; NT] = std::array::from_fn(|t| acts[t].q.as_ptr());
    let xs: [*const f32; NT] = std::array::from_fn(|t| acts[t].scales.as_ptr());
    for r in r0..r1 {
        // SAFETY: `r < m.rows`, so the row's codes and scales are in bounds.
        unsafe {
            let codes = base.add(r * row_bytes);
            let scales = scale_base.add(r * nb * 4) as *const f32;
            let mut row_acc = [0.0f32; NT];
            for b in 0..nb {
                let mut facc = [vdupq_n_f32(0.0); NT];
                for cc in 0..CHUNKS_PER_BLOCK {
                    let c = b * CHUNKS_PER_BLOCK + cc;
                    let v = vld1q_u8(codes.add(c * 16));
                    let c0 = vandq_u8(v, m3);
                    let c1 = vandq_u8(vshrq_n_u8::<2>(v), m3);
                    let c2 = vandq_u8(vshrq_n_u8::<4>(v), m3);
                    let c3 = vshrq_n_u8::<6>(v);
                    for t in 0..NT {
                        let xp = xq[t].add(c * 64);
                        let a0 = sdot16(sdot16(vdupq_n_s32(0), c0, vld1q_s8(xp)), c1, vld1q_s8(xp.add(16)));
                        let a1 = sdot16(sdot16(vdupq_n_s32(0), c2, vld1q_s8(xp.add(32))), c3, vld1q_s8(xp.add(48)));
                        facc[t] = vfmaq_n_f32(facc[t], vcvtq_f32_s32(vaddq_s32(a0, a1)), *xs[t].add(c));
                    }
                }
                let d = scales.add(b).read_unaligned();
                for t in 0..NT {
                    row_acc[t] += d * (2.0 * vaddvq_f32(facc[t]) - 3.0 * tb[t][b]);
                }
            }
            for t in 0..NT {
                *out[t].add(r) = row_acc[t];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    #[test]
    fn q2_0c_blocks_roundtrip_and_gemv_matches_reference() {
        let (rows, cols) = (37, 1024);
        let mut seed = 7u64;
        // Raw blocks with known codes and scales.
        let mut blocks = Vec::new();
        let mut want = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for b in 0..cols / Q2C_BLOCK {
                let d = 0.01 + 0.02 * lcg(&mut seed).abs();
                let h = {
                    let mut o = [0u16];
                    crate::kernels::half::f32_to_f16(&[d], &mut o);
                    o[0]
                };
                let d = crate::storage::gguf::f16_to_f32(h);
                blocks.extend_from_slice(&h.to_le_bytes());
                let mut qs = [0u8; Q2C_BLOCK / 4];
                for j in 0..Q2C_BLOCK {
                    let c = ((lcg(&mut seed) + 1.0) * 2.0).min(3.0) as u8;
                    qs[j / 4] |= c << (2 * (j % 4));
                    want[r * cols + b * Q2C_BLOCK + j] = d * (2.0 * c as f32 - 3.0);
                }
                blocks.extend_from_slice(&qs);
            }
        }
        let m = Q2cMatrix::from_q2_0c_blocks(&blocks, rows, cols);
        assert_eq!(m.to_f32(), want);

        let x: Vec<f32> = (0..cols).map(|_| lcg(&mut seed)).collect();
        let mut act = Int8ChunkActivation::new_zeroed(cols);
        act.populate(&x);
        let xd: Vec<f32> = (0..cols).map(|i| act.q[i] as f32 * act.scales[i / 64]).collect();
        let mut got = vec![0.0f32; rows];
        unsafe { q2c_gemv_rows(&m, &act, 0, rows, got.as_mut_ptr()) };
        let mut scalar = vec![0.0f32; rows];
        let mut tb = vec![0.0f32; m.num_blocks()];
        block_sums(&act, &mut tb);
        unsafe { q2c_rows_scalar::<1>(&m, [&act], [&tb[..]], 0, rows, [scalar.as_mut_ptr()]) };
        for r in 0..rows {
            let exact: f32 = want[r * cols..(r + 1) * cols].iter().zip(&xd).map(|(w, x)| w * x).sum();
            assert!((got[r] - exact).abs() < 1e-3 * (1.0 + exact.abs()), "row {r}: {} vs {exact}", got[r]);
            assert!((scalar[r] - exact).abs() < 1e-3 * (1.0 + exact.abs()));
        }

        // Batch path equals per-token GEMV.
        let n = 6;
        let acts: Vec<BipolarActivationLut> = (0..n)
            .map(|_| {
                let xv: Vec<f32> = (0..cols).map(|_| lcg(&mut seed)).collect();
                let mut a = BipolarActivationLut::new_zeroed(cols);
                a.lut_enabled = false;
                a.int8 = Some(Int8ChunkActivation::new_zeroed(cols));
                a.populate(&xv);
                a
            })
            .collect();
        let mut batch = vec![0.0f32; n * rows];
        unsafe { q2c_batch_rows(&m, &acts, 0, rows, batch.as_mut_ptr(), rows) };
        for (t, a) in acts.iter().enumerate() {
            let mut one = vec![0.0f32; rows];
            unsafe { q2c_gemv_rows(&m, a.int8.as_ref().unwrap(), 0, rows, one.as_mut_ptr()) };
            assert_eq!(&batch[t * rows..(t + 1) * rows], &one[..]);
        }
    }
}
