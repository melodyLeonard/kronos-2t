#![deny(unsafe_op_in_unsafe_fn)]
//! Zero-Allocation $O(\sqrt{M})$ Cartesian Product-Key Router (`aarch64` NEON + Portable).
//!
//! Streams subspace scores through a stack-resident sorted top-$k$ buffer (`k <= 16`) with
//! **zero heap allocations** during token inference, using 4-way pipelined NEON `vfmaq_f32`.

const STACK_TOP_K_MAX: usize = 16;

/// Routes a token across `M = sub_keys^2` experts in `O(sqrt(M))` time.
#[derive(Clone, Debug)]
pub struct ProductKeyRouter {
    pub d_model: usize,
    pub sub_keys: usize,
    pub codebook_1: Vec<f32>, // [sub_keys * (d_model / 2)]
    pub codebook_2: Vec<f32>, // [sub_keys * (d_model / 2)]
}

impl ProductKeyRouter {
    /// Initializes a deterministic Product-Key Router using SplitMix64.
    pub fn new(d_model: usize, sub_keys: usize, seed: u64) -> Self {
        assert!(
            d_model > 0 && d_model.is_multiple_of(2),
            "d_model must be positive and even for Cartesian split"
        );
        assert!(sub_keys > 0, "sub_keys must be greater than zero");

        let half_d = d_model / 2;
        let codebook_1 = Self::deterministic_init(sub_keys * half_d, seed);
        let codebook_2 =
            Self::deterministic_init(sub_keys * half_d, seed.wrapping_add(0x9E37_79B9));

        Self {
            d_model,
            sub_keys,
            codebook_1,
            codebook_2,
        }
    }

    #[inline]
    pub fn total_experts(&self) -> usize {
        self.sub_keys * self.sub_keys
    }

    fn deterministic_init(len: usize, mut state: u64) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = (z ^ (z >> 31)) as f64 / (u64::MAX as f64);
                (u as f32 * 2.0 - 1.0) * 0.1
            })
            .collect()
    }

    /// Convenience wrapper returning a newly allocated `Vec` of top-`k` `(expert_id, score)`.
    pub fn route_top_k(&self, hidden_state: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut out = Vec::with_capacity(k);
        self.route_top_k_into(hidden_state, k, &mut out);
        out
    }

    /// Zero-allocation routing into a caller-reusable `out` buffer (when `k <= 16` and
    /// `out.capacity() >= k^2`).
    pub fn route_top_k_into(
        &self,
        hidden_state: &[f32],
        k: usize,
        out: &mut Vec<(usize, f32)>,
    ) {
        assert_eq!(
            hidden_state.len(),
            self.d_model,
            "hidden_state length must equal d_model"
        );
        out.clear();
        if k == 0 {
            return;
        }

        let half_d = self.d_model / 2;
        let (q1, q2) = hidden_state.split_at(half_d);
        let k_sub = k.min(self.sub_keys);

        if k_sub <= STACK_TOP_K_MAX {
            let mut top1 = [(0usize, f32::NEG_INFINITY); STACK_TOP_K_MAX];
            let mut top2 = [(0usize, f32::NEG_INFINITY); STACK_TOP_K_MAX];
            let n1 = Self::top_k_subspace_stack(&self.codebook_1, q1, k_sub, &mut top1);
            let n2 = Self::top_k_subspace_stack(&self.codebook_2, q2, k_sub, &mut top2);

            let mut merged = [(0usize, f32::NEG_INFINITY); STACK_TOP_K_MAX];
            let k_final = k.min(self.total_experts()).min(STACK_TOP_K_MAX);
            let mut m_len = 0usize;

            for &(idx1, score1) in &top1[..n1] {
                for &(idx2, score2) in &top2[..n2] {
                    let sum = score1 + score2;
                    let expert_id = idx1 * self.sub_keys + idx2;
                    Self::insert_top_k(&mut merged, &mut m_len, k_final, expert_id, sum);
                }
            }
            out.extend_from_slice(&merged[..m_len]);
        } else {
            let top1 = Self::top_k_subspace_Fallback(&self.codebook_1, q1, k_sub);
            let top2 = Self::top_k_subspace_Fallback(&self.codebook_2, q2, k_sub);
            out.reserve(top1.len() * top2.len());
            for &(idx1, score1) in &top1 {
                for &(idx2, score2) in &top2 {
                    out.push((idx1 * self.sub_keys + idx2, score1 + score2));
                }
            }
            out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            out.truncate(k.min(self.total_experts()));
        }
    }

    #[inline]
    fn insert_top_k(
        buf: &mut [(usize, f32); STACK_TOP_K_MAX],
        len: &mut usize,
        k: usize,
        id: usize,
        score: f32,
    ) {
        if *len == k {
            let worst = buf[k - 1];
            if score.total_cmp(&worst.1).then_with(|| worst.0.cmp(&id)).is_le() {
                return;
            }
        } else {
            *len += 1;
        }

        let mut pos = *len - 1;
        while pos > 0 {
            let prev = buf[pos - 1];
            if score.total_cmp(&prev.1).then_with(|| prev.0.cmp(&id)).is_gt() {
                buf[pos] = prev;
                pos -= 1;
            } else {
                break;
            }
        }
        buf[pos] = (id, score);
    }

    /// Scores codebook keys using pipelined 16-wide FMA, keeping top-`k` on the stack (`0` allocs).
    fn top_k_subspace_stack(
        codebook: &[f32],
        query: &[f32],
        k: usize,
        out_buf: &mut [(usize, f32); STACK_TOP_K_MAX],
    ) -> usize {
        let half_d = query.len();
        let mut len = 0usize;

        for (idx, key) in codebook.chunks_exact(half_d).enumerate() {
            let s = Self::dot_f32(key, query);
            Self::insert_top_k(out_buf, &mut len, k, idx, s);
        }

        len
    }

    #[allow(non_snake_case)]
    fn top_k_subspace_Fallback(codebook: &[f32], query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let half_d = query.len();
        let mut scores: Vec<(usize, f32)> = codebook
            .chunks_exact(half_d)
            .enumerate()
            .map(|(idx, key)| (idx, Self::dot_f32(key, query)))
            .collect();
        if k < scores.len() {
            scores.select_nth_unstable_by(k - 1, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            scores.truncate(k);
        }
        scores
    }

    #[inline]
    fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is baseline on `aarch64`, and `a.len() == b.len()` is enforced below.
            unsafe { Self::dot_f32_neon(a, b) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            Self::dot_f32_scalar16(a, b)
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[target_feature(enable = "neon")]
    unsafe fn dot_f32_neon(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::aarch64::*;
        let n = a.len().min(b.len());
        let ap = a.as_ptr();
        let bp = b.as_ptr();

        let mut acc0 = vdupq_n_f32(0.0);
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        let mut acc3 = vdupq_n_f32(0.0);

        let mut i = 0usize;
        while i + 16 <= n {
            // SAFETY: `i + 16 <= n <= a.len(), b.len()`.
            unsafe {
                acc0 = vfmaq_f32(acc0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
                acc1 = vfmaq_f32(acc1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
                acc2 = vfmaq_f32(acc2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
                acc3 = vfmaq_f32(acc3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
            }
            i += 16;
        }

        let sum_vec = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
        let mut total = vaddvq_f32(sum_vec);
        while i < n {
            total += a[i] * b[i];
            i += 1;
        }
        total
    }

    #[cfg(not(target_arch = "aarch64"))]
    #[inline]
    fn dot_f32_scalar16(a: &[f32], b: &[f32]) -> f32 {
        let a_chunks = a.chunks_exact(16);
        let b_chunks = b.chunks_exact(16);
        let a_rem = a_chunks.remainder();
        let b_rem = b_chunks.remainder();

        let mut acc0 = [0.0f32; 4];
        let mut acc1 = [0.0f32; 4];
        let mut acc2 = [0.0f32; 4];
        let mut acc3 = [0.0f32; 4];

        for (ac, bc) in a_chunks.zip(b_chunks) {
            acc0[0] += ac[0] * bc[0];
            acc0[1] += ac[1] * bc[1];
            acc0[2] += ac[2] * bc[2];
            acc0[3] += ac[3] * bc[3];
            acc1[0] += ac[4] * bc[4];
            acc1[1] += ac[5] * bc[5];
            acc1[2] += ac[6] * bc[6];
            acc1[3] += ac[7] * bc[7];
            acc2[0] += ac[8] * bc[8];
            acc2[1] += ac[9] * bc[9];
            acc2[2] += ac[10] * bc[10];
            acc2[3] += ac[11] * bc[11];
            acc3[0] += ac[12] * bc[12];
            acc3[1] += ac[13] * bc[13];
            acc3[2] += ac[14] * bc[14];
            acc3[3] += ac[15] * bc[15];
        }
        let mut tail = 0.0f32;
        for (&av, &bv) in a_rem.iter().zip(b_rem.iter()) {
            tail += av * bv;
        }
        let s0 = (acc0[0] + acc1[0]) + (acc2[0] + acc3[0]);
        let s1 = (acc0[1] + acc1[1]) + (acc2[1] + acc3[1]);
        let s2 = (acc0[2] + acc1[2]) + (acc2[2] + acc3[2]);
        let s3 = (acc0[3] + acc1[3]) + (acc2[3] + acc3[3]);
        ((s0 + s1) + (s2 + s3)) + tail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_key_exact_match_against_exhaustive_cartesian_search() {
        let d_model = 68; // Half-dim = 34 (non-multiple of 16) to test remainder paths!
        let sub_keys = 35; // Non-multiple of 4 to test 4-key remainder path!
        let k = 8;
        let router = ProductKeyRouter::new(d_model, sub_keys, 1337);
        let hidden: Vec<f32> = (0..d_model).map(|i| ((i as f32) * 0.17).cos()).collect();

        let fast_top_k = router.route_top_k(&hidden, k);

        let half_d = d_model / 2;
        let mut exhaustive = Vec::with_capacity(sub_keys * sub_keys);
        for i1 in 0..sub_keys {
            let k1 = &router.codebook_1[i1 * half_d..(i1 + 1) * half_d];
            for i2 in 0..sub_keys {
                let k2 = &router.codebook_2[i2 * half_d..(i2 + 1) * half_d];
                let dot1: f32 = k1.iter().zip(&hidden[..half_d]).map(|(a, b)| a * b).sum();
                let dot2: f32 = k2.iter().zip(&hidden[half_d..]).map(|(a, b)| a * b).sum();
                exhaustive.push((i1 * sub_keys + i2, dot1 + dot2));
            }
        }
        exhaustive.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        exhaustive.truncate(k);

        assert_eq!(fast_top_k.len(), k);
        for ((fast_id, fast_score), (ref_id, ref_score)) in
            fast_top_k.iter().zip(exhaustive.iter())
        {
            assert_eq!(fast_id, ref_id);
            assert!((fast_score - ref_score).abs() < 1e-5);
        }
    }

    #[test]
    fn test_router_edge_cases() {
        let router = ProductKeyRouter::new(8, 3, 7);
        let hidden = [0.5f32; 8];
        assert!(router.route_top_k(&hidden, 0).is_empty());
        let all = router.route_top_k(&hidden, 9);
        assert_eq!(all.len(), 9);
    }
}
