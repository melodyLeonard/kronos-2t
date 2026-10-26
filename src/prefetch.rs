//! 6-Layer Lookahead Subspace Router & Ring-Buffer Cache Prefetcher.
//!
//! Exploits smooth residual stream evolution (`h_{l+1} = h_l + f_l(h_l)`) by projecting
//! `h_l -> h_hat_{l+6}` via a lightweight low-rank subspace operator
//! `P_{l -> l+6}(h) = h + U * (V^T * h)` and pre-staging predicted micro-experts
//! into a bounded RAM ring-buffer before execution reaches layer `l + 6`.

use crate::router::ProductKeyRouter;
use std::collections::{HashSet, VecDeque};

/// Low-rank residual drift operator `P(h) = h + U * (V^T * h)` with rank `r << d_model`.
#[derive(Clone, Debug)]
pub struct SubspaceProjector {
    pub d_model: usize,
    pub rank: usize,
    pub u_proj: Vec<f32>, // [rank * d_model]
    pub v_proj: Vec<f32>, // [rank * d_model]
}

impl SubspaceProjector {
    /// Initializes a small-norm low-rank residual lookahead projector.
    pub fn new(d_model: usize, rank: usize, seed: u64) -> Self {
        assert!(d_model > 0 && rank > 0);
        let scale = 0.01 / (d_model as f32).sqrt();
        let u_proj = Self::init_matrix(rank * d_model, seed, scale);
        let v_proj = Self::init_matrix(rank * d_model, seed.wrapping_add(0x517C_C1B7), scale);
        Self {
            d_model,
            rank,
            u_proj,
            v_proj,
        }
    }

    fn init_matrix(len: usize, mut state: u64, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = (z ^ (z >> 31)) as f64 / (u64::MAX as f64);
                (u as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    /// Projects `h_l` into estimated future residual `h_hat_{l+delta}` in `O(2 * rank * d_model)`.
    pub fn project(&self, hidden: &[f32], out: &mut [f32]) {
        assert_eq!(hidden.len(), self.d_model);
        assert_eq!(out.len(), self.d_model);
        out.copy_from_slice(hidden);

        for (u_row, v_row) in self
            .u_proj
            .chunks_exact(self.d_model)
            .zip(self.v_proj.chunks_exact(self.d_model))
        {
            let latent: f32 = v_row.iter().zip(hidden.iter()).map(|(v, h)| v * h).sum();
            for (o, &u) in out.iter_mut().zip(u_row.iter()) {
                *o += latent * u;
            }
        }
    }
}

/// Predicts future expert activations `k` layers ahead and stages micro-experts
/// into a fixed-capacity RAM ring-buffer cache.
#[derive(Clone, Debug)]
pub struct LookaheadPrefetcher {
    pub capacity_experts: usize,
    pub ram_cache: HashSet<usize>,
    pub fifo_queue: VecDeque<usize>,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl LookaheadPrefetcher {
    pub fn new(capacity_experts: usize) -> Self {
        assert!(capacity_experts > 0, "capacity_experts must be > 0");
        Self {
            capacity_experts,
            ram_cache: HashSet::with_capacity(capacity_experts),
            fifo_queue: VecDeque::with_capacity(capacity_experts),
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Projects hidden state `h_l -> h_hat_{l+6}` (identity approximation) and prefetches
    /// predicted top-`k` experts into the RAM ring-buffer.
    pub fn prefetch_lookahead(
        &mut self,
        future_router: &ProductKeyRouter,
        current_hidden: &[f32],
        top_k: usize,
    ) {
        let predicted_experts = future_router.route_top_k(current_hidden, top_k);
        for (expert_id, _score) in predicted_experts {
            self.stage_into_ram(expert_id);
        }
    }

    /// Projects `h_l` via `SubspaceProjector` (`P_{l->l+6}`) before routing on `future_router`.
    pub fn prefetch_with_projector(
        &mut self,
        projector: &SubspaceProjector,
        future_router: &ProductKeyRouter,
        current_hidden: &[f32],
        scratch_hidden: &mut [f32],
        top_k: usize,
    ) {
        projector.project(current_hidden, scratch_hidden);
        self.prefetch_lookahead(future_router, scratch_hidden, top_k);
    }

    /// Stages `expert_id` into the bounded RAM ring-buffer, evicting the oldest entry if full.
    pub fn stage_into_ram(&mut self, expert_id: usize) {
        if !self.ram_cache.contains(&expert_id) {
            if self.ram_cache.len() >= self.capacity_experts {
                if let Some(evicted) = self.fifo_queue.pop_front() {
                    self.ram_cache.remove(&evicted);
                }
            }
            self.ram_cache.insert(expert_id);
            self.fifo_queue.push_back(expert_id);
        }
    }

    /// Accesses `expert_id` at execution time; returns `true` on RAM hit, `false` on Flash miss.
    pub fn access_expert(&mut self, expert_id: usize) -> bool {
        if self.ram_cache.contains(&expert_id) {
            self.cache_hits += 1;
            true
        } else {
            self.cache_misses += 1;
            self.stage_into_ram(expert_id);
            false
        }
    }

    /// Percentage of expert accesses served directly from RAM without stalling on Flash.
    pub fn hit_rate_percent(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_eviction_and_hit_rate() {
        let mut prefetcher = LookaheadPrefetcher::new(3);
        prefetcher.stage_into_ram(10);
        prefetcher.stage_into_ram(20);
        prefetcher.stage_into_ram(30);
        assert_eq!(prefetcher.ram_cache.len(), 3);

        // Staging a 4th distinct expert evicts 10 (FIFO)
        prefetcher.stage_into_ram(40);
        assert_eq!(prefetcher.ram_cache.len(), 3);
        assert!(prefetcher.access_expert(20)); // hit
        assert!(!prefetcher.access_expert(10)); // miss (was evicted)
        assert!((prefetcher.hit_rate_percent() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_subspace_projected_lookahead() {
        let d_model = 128;
        let router = ProductKeyRouter::new(d_model, 64, 99);
        let projector = SubspaceProjector::new(d_model, 8, 123);
        let mut prefetcher = LookaheadPrefetcher::new(32);

        let hidden: Vec<f32> = (0..d_model).map(|i| ((i as f32) * 0.05).sin()).collect();
        let mut evolved = vec![0.0f32; d_model];
        let mut scratch = vec![0.0f32; d_model];

        prefetcher.prefetch_with_projector(&projector, &router, &hidden, &mut scratch, 8);

        // Simulate actual future residual close to subspace projection
        projector.project(&hidden, &mut evolved);
        for val in &mut evolved {
            *val += 0.002;
        }

        let actual = router.route_top_k(&evolved, 4);
        for (exp_id, _) in actual {
            prefetcher.access_expert(exp_id);
        }
        assert!(prefetcher.hit_rate_percent() >= 75.0);
    }
}
