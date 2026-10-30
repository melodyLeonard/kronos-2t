#![forbid(unsafe_code)]
//! Zero-Allocation Multi-Layer `KronosEngine` Runtime.
//!
//! Unifies all five mathematical & systems pillars into an autoregressive token generation loop
//! with **zero heap allocations per token**:
//! 1. **Zero-Alloc $O(\sqrt{M})$ Cartesian Product-Key Routing** across $1{,}000{,}000$ experts/layer.
//! 2. **4KB Page-Aligned `.k2t` Zero-Copy `mmap` Pager** with `POSIX_MADV_WILLNEED` DMA hints.
//! 3. **6-Layer Manifold-Aligned Lookahead Subspace Prefetching** ($P_{\ell \to \ell+6}$).
//! 4. **Register-Fused Multi-Rank Kronecker ARM NEON `vqtbl4q_u8` + `vqtbl1q_s8` SwiGLU**
//!    (zero L1 tile buffers!).
//! 5. **Constant-Memory $O(1)$ State-Space Dual (SSD)** recurrent context blocks with induction shift.

use crate::storage::k2t_format::write_ranked_k2t_container;
use crate::storage::mmap_pager::{AsyncFlashPager, K2tMmap};
use crate::storage::prefetch::SubspaceProjector;
use crate::kernels::router::ProductKeyRouter;
use crate::kernels::simd_lut::{fused_kronecker_gemv, simd_lut_gemv, QuantizedActivationLut, SimdPackedMatrix};
use crate::kernels::ssd::{SsdBlock, SsdScratch, SsdState};
use std::io;
use std::path::Path;
use std::time::Instant;

/// Architecture & runtime configuration for [`KronosEngine`].
#[derive(Clone, Debug)]
pub struct KronosConfig {
    pub vocab_size: usize,
    pub num_layers: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub num_heads: usize,
    pub kronecker_rank: usize,
    pub router_sub_keys: usize,
    pub physical_experts_per_layer: usize,
    pub top_k_experts: usize,
    pub lookahead_layers: usize,
    pub lookahead_rank: usize,
    pub ram_ring_buffer_pages: usize,
    pub residual_depth_scale: Option<f32>,
    pub seed: u64,
}

impl KronosConfig {
    /// Default 12-layer `2.36 Trillion` virtual-parameter edge configuration.
    pub fn default_2t_prototype() -> Self {
        Self {
            vocab_size: 2048,
            num_layers: 12,
            d_model: 512,
            d_ff: 128,
            num_heads: 8,
            kronecker_rank: 1,
            router_sub_keys: 1000, // 1,000,000 virtual micro-experts per layer
            physical_experts_per_layer: 128,
            top_k_experts: 4,
            lookahead_layers: 6,
            lookahead_rank: 16,
            ram_ring_buffer_pages: 384,
            residual_depth_scale: None,
            seed: 0x2026_2000_0000_0000,
        }
    }

    pub fn virtual_total_parameters(&self) -> u128 {
        let experts_per_layer = (self.router_sub_keys as u128) * (self.router_sub_keys as u128);
        let params_per_expert = 3u128 * (self.d_model as u128) * (self.d_ff as u128);
        let sparse_params = (self.num_layers as u128) * experts_per_layer * params_per_expert;
        let ssd_params =
            (self.num_layers as u128) * 4u128 * (self.d_model as u128) * (self.d_model as u128);
        sparse_params + ssd_params
    }

    pub fn active_parameters_per_token(&self) -> usize {
        let params_per_expert = 3 * self.d_model * self.d_ff;
        let active_sparse = self.num_layers * self.top_k_experts * params_per_expert;
        let active_ssd = self.num_layers * 4 * self.d_model * self.d_model;
        let lm_head = self.vocab_size * self.d_model;
        active_sparse + active_ssd + lm_head
    }
}

/// Telemetry returned by an autoregressive generation run.
#[derive(Clone, Debug)]
pub struct GenerationReport {
    pub generated_tokens: Vec<usize>,
    pub tokens_generated: usize,
    pub elapsed_ms: f64,
    pub tokens_per_sec: f64,
    pub ssd_context_bytes: usize,
    pub equivalent_fp16_kv_bytes: usize,
    pub pager_hit_rate_percent: f64,
    pub willneed_syscalls: usize,
    pub dontneed_syscalls: usize,
}

struct EngineWorkspace {
    hidden: Vec<f32>,
    ssd_delta: Vec<f32>,
    ffn_normed: Vec<f32>,
    lookahead_scratch: Vec<f32>,
    gate_buf: Vec<f32>,
    up_buf: Vec<f32>,
    expert_out: Vec<f32>,
    final_normed: Vec<f32>,
    logits: Vec<f32>,
    routed_buf: Vec<(usize, f32)>,
    qlut_dmodel: QuantizedActivationLut,
    qlut_dff: QuantizedActivationLut,
    ssd_scratch: SsdScratch,
}

impl EngineWorkspace {
    fn new(cfg: &KronosConfig) -> Self {
        Self {
            hidden: vec![0.0f32; cfg.d_model],
            ssd_delta: vec![0.0f32; cfg.d_model],
            ffn_normed: vec![0.0f32; cfg.d_model],
            lookahead_scratch: vec![0.0f32; cfg.d_model],
            gate_buf: vec![0.0f32; cfg.d_ff],
            up_buf: vec![0.0f32; cfg.d_ff],
            expert_out: vec![0.0f32; cfg.d_model],
            final_normed: vec![0.0f32; cfg.d_model],
            logits: vec![0.0f32; cfg.vocab_size],
            routed_buf: Vec::with_capacity(cfg.top_k_experts.max(16)),
            qlut_dmodel: QuantizedActivationLut::new_zeroed(cfg.d_model),
            qlut_dff: QuantizedActivationLut::new_zeroed(cfg.d_ff),
            ssd_scratch: SsdScratch::new(cfg.d_model, cfg.num_heads),
        }
    }
}

/// Complete multi-layer sparse inference engine combining SSD, Product-Key Routing,
/// `.k2t` `mmap` lookahead paging, and register-fused ARM NEON `LUT-GEMV`.
pub struct KronosEngine {
    pub config: KronosConfig,
    pub pager: AsyncFlashPager,
    pub routers: Vec<ProductKeyRouter>,
    pub lookahead_projectors: Vec<SubspaceProjector>,
    pub entry_projectors: Vec<SubspaceProjector>,
    pub ssd_blocks: Vec<SsdBlock>,
    pub ssd_states: Vec<SsdState>,
    pub token_embeddings: Vec<f32>, // [vocab_size * d_model]
    pub final_rms_weight: Vec<f32>,
    pub lm_head: SimdPackedMatrix, // [vocab_size, d_model]
    pub lm_head_scale: f32,
    pub lm_head_bias: Vec<f32>, // [vocab_size]
    workspace: EngineWorkspace,
}

impl KronosEngine {
    /// Builds `.k2t` container at `k2t_path`, memory-maps it, and initializes all layers.
    pub fn build_and_open(cfg: KronosConfig, k2t_path: &Path) -> io::Result<Self> {
        let expert_scale = 0.15 / (cfg.d_model as f32).sqrt();
        write_ranked_k2t_container(
            k2t_path,
            cfg.num_layers as u32,
            cfg.physical_experts_per_layer as u32,
            cfg.d_model as u32,
            cfg.d_ff as u32,
            cfg.kronecker_rank as u32,
            expert_scale,
        )?;

        let mmap = K2tMmap::open(k2t_path)?;
        let pager = AsyncFlashPager::new(mmap, cfg.ram_ring_buffer_pages);

        let mut routers = Vec::with_capacity(cfg.num_layers);
        let mut lookahead_projectors = Vec::with_capacity(cfg.num_layers);
        let mut ssd_blocks = Vec::with_capacity(cfg.num_layers);

        for l in 0..cfg.num_layers {
            let l_seed = cfg.seed.wrapping_add((l as u64) * 10_007);
            routers.push(ProductKeyRouter::new(
                cfg.d_model,
                cfg.router_sub_keys,
                l_seed,
            ));
            lookahead_projectors.push(SubspaceProjector::new(
                cfg.d_model,
                cfg.lookahead_rank,
                l_seed.wrapping_add(503),
            ));
            ssd_blocks.push(SsdBlock::new(
                cfg.d_model,
                cfg.num_heads,
                l_seed.wrapping_add(1009),
            ));
        }

        let early_count = cfg.lookahead_layers.min(cfg.num_layers);
        let entry_projectors: Vec<SubspaceProjector> = (0..early_count)
            .map(|l| {
                SubspaceProjector::new(
                    cfg.d_model,
                    cfg.lookahead_rank,
                    cfg.seed.wrapping_add(90_001 + l as u64),
                )
            })
            .collect();

        let mut state = cfg.seed.wrapping_add(0xABCD_EF01);
        let token_embeddings: Vec<f32> = (0..cfg.vocab_size * cfg.d_model)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = (z ^ (z >> 31)) as f64 / (u64::MAX as f64);
                ((u as f32) * 2.0 - 1.0) * 0.25
            })
            .collect();

        let head_weights: Vec<i8> = (0..cfg.vocab_size * cfg.d_model)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z % 3) as i8) - 1
            })
            .collect();
        let lm_head =
            SimdPackedMatrix::from_ternary_weights(&head_weights, cfg.vocab_size, cfg.d_model);
        let lm_head_scale = 1.0 / (cfg.d_model as f32).sqrt();
        let lm_head_bias = vec![0.0f32; cfg.vocab_size];

        Ok(Self::from_compiled_parts(
            cfg,
            pager,
            routers,
            lookahead_projectors,
            entry_projectors,
            ssd_blocks,
            token_embeddings,
            lm_head,
            lm_head_scale,
            lm_head_bias,
        ))
    }

    /// Assembles a [`KronosEngine`] from trained/compiled components and an open `.k2t` [`AsyncFlashPager`].
    #[allow(clippy::too_many_arguments)]
    pub fn from_compiled_parts(
        cfg: KronosConfig,
        pager: AsyncFlashPager,
        routers: Vec<ProductKeyRouter>,
        lookahead_projectors: Vec<SubspaceProjector>,
        entry_projectors: Vec<SubspaceProjector>,
        ssd_blocks: Vec<SsdBlock>,
        token_embeddings: Vec<f32>,
        lm_head: SimdPackedMatrix,
        lm_head_scale: f32,
        lm_head_bias: Vec<f32>,
    ) -> Self {
        let ssd_states = (0..cfg.num_layers)
            .map(|_| SsdState::new(cfg.num_heads, cfg.d_model / cfg.num_heads))
            .collect();
        let workspace = EngineWorkspace::new(&cfg);
        Self {
            final_rms_weight: vec![1.0f32; cfg.d_model],
            config: cfg,
            pager,
            routers,
            lookahead_projectors,
            entry_projectors,
            ssd_blocks,
            ssd_states,
            token_embeddings,
            lm_head,
            lm_head_scale,
            lm_head_bias,
            workspace,
        }
    }

    /// Resets all recurrent SSD fast-weight states and pager telemetry counters.
    pub fn reset_context(&mut self) {
        for st in &mut self.ssd_states {
            st.reset();
        }
        self.pager.page_hits = 0;
        self.pager.page_misses = 0;
        self.pager.willneed_syscalls = 0;
        self.pager.dontneed_syscalls = 0;
    }

    pub fn total_ssd_state_bytes(&self) -> usize {
        self.ssd_states.iter().map(|s| s.memory_bytes()).sum()
    }

    pub fn equivalent_kv_cache_bytes(&self, seq_len: usize) -> usize {
        self.ssd_states
            .iter()
            .map(|s| s.equivalent_fp16_kv_cache_bytes(seq_len))
            .sum()
    }

    /// Returns the most recent post-layer RMS-normalized hidden representation (`[d_model]`).
    #[inline]
    pub fn last_final_normed(&self) -> &[f32] {
        &self.workspace.final_normed
    }

    /// Executes one full multi-layer sparse forward pass for `token_id` with **zero heap allocations**.
    pub fn forward_token(&mut self, token_id: usize) -> Result<&[f32], &'static str> {
        let _ = self.forward_token_with_trace(token_id, None)?;
        Ok(&self.workspace.logits)
    }

    /// Same as [`Self::forward_token`], optionally recording layer-0 routed expert IDs for live inspection.
    pub fn forward_token_with_trace(
        &mut self,
        token_id: usize,
        mut layer0_experts_out: Option<&mut Vec<usize>>,
    ) -> Result<&[f32], &'static str> {
        assert!(token_id < self.config.vocab_size, "token_id out of range");

        let dm = self.config.d_model;
        let dff = self.config.d_ff;
        let top_k = self.config.top_k_experts;
        let lookahead = self.config.lookahead_layers;
        let reg_cb = self.pager.mmap.header.reg_codebook_64;
        let act_scale = 1.0 / (dm as f32).sqrt();
        let down_scale = 1.0 / (dff as f32).sqrt();
        let residual_depth_scale = self
            .config
            .residual_depth_scale
            .unwrap_or_else(|| 0.25 / ((2 * self.config.num_layers) as f32).sqrt());

        // 1. Token embedding lookup
        let emb_slice = &self.token_embeddings[token_id * dm..(token_id + 1) * dm];
        self.workspace.hidden.copy_from_slice(emb_slice);

        // 2. Entry lookahead prefetch for initial layers `0..lookahead`
        for (early_l, proj) in self.entry_projectors.iter().enumerate() {
            self.pager.prefetch_future_layer(
                early_l,
                proj,
                &self.routers[early_l],
                &self.workspace.hidden,
                &mut self.workspace.lookahead_scratch,
                top_k * 3,
            );
        }

        // 3. Multi-layer SSD + Lookahead DMA + Register-Fused Kronecker NEON SwiGLU
        for l in 0..self.config.num_layers {
            let future_l = l + lookahead;
            if future_l < self.config.num_layers {
                self.pager.prefetch_future_layer(
                    future_l,
                    &self.lookahead_projectors[l],
                    &self.routers[future_l],
                    &self.workspace.hidden,
                    &mut self.workspace.lookahead_scratch,
                    top_k * 3,
                );
            }

            self.ssd_blocks[l].step_recurrent(
                &self.workspace.hidden,
                &mut self.ssd_states[l],
                &mut self.workspace.ssd_scratch,
                &mut self.workspace.ssd_delta,
            );
            for (h, &d) in self
                .workspace
                .hidden
                .iter_mut()
                .zip(self.workspace.ssd_delta.iter())
            {
                *h += residual_depth_scale * d;
            }

            rms_norm_slice(
                &self.workspace.hidden,
                &self.final_rms_weight,
                &mut self.workspace.ffn_normed,
            );
            self.routers[l].route_top_k_into(
                &self.workspace.ffn_normed,
                top_k,
                &mut self.workspace.routed_buf,
            );

            if l == 0 {
                if let Some(ref mut trace_vec) = layer0_experts_out {
                    trace_vec.clear();
                    for &(exp_id, _) in &self.workspace.routed_buf {
                        trace_vec.push(exp_id);
                    }
                }
            }

            let max_score = self
                .workspace
                .routed_buf
                .iter()
                .fold(f32::NEG_INFINITY, |m, &(_, s)| m.max(s));
            let mut weights = [0.0f32; 16];
            assert!(self.workspace.routed_buf.len() <= weights.len());
            let mut sum_exp = 0.0f32;
            for (w_slot, &(_, s)) in weights.iter_mut().zip(self.workspace.routed_buf.iter()) {
                let e = (s - max_score).exp();
                *w_slot = e;
                sum_exp += e;
            }

            // Populate reusable INT8 LUT in-place (0 heap allocations!)
            self.workspace
                .qlut_dmodel
                .populate(&self.workspace.ffn_normed);

            let num_routed = self.workspace.routed_buf.len();
            for (r_i, &raw_w) in weights[..num_routed].iter().enumerate() {
                let expert_id = self.workspace.routed_buf[r_i].0;
                let gate_weight = raw_w / sum_exp;
                let view = self.pager.access_expert(l, expert_id)?;

                // Register-fused Kronecker NEON synthesis + GEMV directly from mmap slice!
                fused_kronecker_gemv(
                    view.gate_row_seeds,
                    view.gate_col_seeds,
                    &view.gate_scales[..view.rank],
                    dff,
                    &reg_cb,
                    &self.workspace.qlut_dmodel,
                    &mut self.workspace.gate_buf,
                );
                fused_kronecker_gemv(
                    view.up_row_seeds,
                    view.up_col_seeds,
                    &view.up_scales[..view.rank],
                    dff,
                    &reg_cb,
                    &self.workspace.qlut_dmodel,
                    &mut self.workspace.up_buf,
                );

                for (g, &u) in self
                    .workspace
                    .gate_buf
                    .iter_mut()
                    .zip(self.workspace.up_buf.iter())
                {
                    let gs = (*g) * act_scale;
                    let us = u * act_scale;
                    let silu = gs / (1.0 + (-gs).exp());
                    *g = silu * us;
                }

                self.workspace.qlut_dff.populate(&self.workspace.gate_buf);
                fused_kronecker_gemv(
                    view.down_row_seeds,
                    view.down_col_seeds,
                    &view.down_scales[..view.rank],
                    dm,
                    &reg_cb,
                    &self.workspace.qlut_dff,
                    &mut self.workspace.expert_out,
                );

                let combined_scale =
                    residual_depth_scale * gate_weight * view.output_scale * down_scale;
                for (h, &eo) in self
                    .workspace
                    .hidden
                    .iter_mut()
                    .zip(self.workspace.expert_out.iter())
                {
                    *h += combined_scale * eo;
                }
            }
        }

        // 4. Final RMSNorm + Multiplication-Free NEON LM Head Projection
        rms_norm_slice(
            &self.workspace.hidden,
            &self.final_rms_weight,
            &mut self.workspace.final_normed,
        );
        self.workspace
            .qlut_dmodel
            .populate(&self.workspace.final_normed);
        simd_lut_gemv(
            &self.lm_head,
            &self.workspace.qlut_dmodel,
            &mut self.workspace.logits,
        );
        for (l, &b) in self
            .workspace
            .logits
            .iter_mut()
            .zip(self.lm_head_bias.iter())
        {
            *l = (*l) * self.lm_head_scale + b;
        }

        Ok(&self.workspace.logits)
    }

    /// Runs prefill over `prompt_tokens` and autoregressively decodes `max_new_tokens`.
    pub fn generate(
        &mut self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
    ) -> Result<GenerationReport, &'static str> {
        assert!(!prompt_tokens.is_empty(), "prompt_tokens cannot be empty");
        assert!(max_new_tokens > 0, "max_new_tokens must be > 0");

        let t_start = Instant::now();

        let mut next_token = 0usize;
        for &tok in prompt_tokens {
            let logits = self.forward_token(tok)?;
            next_token = argmax(logits);
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        generated.push(next_token);

        for _ in 1..max_new_tokens {
            let logits = self.forward_token(next_token)?;
            next_token = argmax(logits);
            generated.push(next_token);
        }

        let elapsed_ms = t_start.elapsed().as_secs_f64() * 1000.0;
        let total_passes = prompt_tokens.len() + max_new_tokens - 1;
        let tokens_per_sec = (total_passes as f64) / (elapsed_ms / 1000.0);
        let total_seq_len = self.ssd_states[0].tokens_seen;

        Ok(GenerationReport {
            generated_tokens: generated,
            tokens_generated: max_new_tokens,
            elapsed_ms,
            tokens_per_sec,
            ssd_context_bytes: self.total_ssd_state_bytes(),
            equivalent_fp16_kv_bytes: self.equivalent_kv_cache_bytes(total_seq_len),
            pager_hit_rate_percent: self.pager.hit_rate_percent(),
            willneed_syscalls: self.pager.willneed_syscalls,
            dontneed_syscalls: self.pager.dontneed_syscalls,
        })
    }
}

#[inline]
fn rms_norm_slice(x: &[f32], weight: &[f32], out: &mut [f32]) {
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + 1e-5).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *o = xi * inv_rms * wi;
    }
}

#[inline]
pub fn argmax(slice: &[f32]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_to_end_kronos_engine_deterministic_generation_and_constant_ram() {
        let tmp_path = std::env::temp_dir().join("kronos_2t_e2e_engine_test_v2.k2t");
        let cfg = KronosConfig {
            vocab_size: 512,
            num_layers: 8,
            d_model: 128,
            d_ff: 64,
            num_heads: 4,
            kronecker_rank: 2,
            router_sub_keys: 256,
            physical_experts_per_layer: 32,
            top_k_experts: 4,
            lookahead_layers: 4,
            lookahead_rank: 8,
            ram_ring_buffer_pages: 128,
            residual_depth_scale: None,
            seed: 42,
        };

        let mut engine =
            KronosEngine::build_and_open(cfg, &tmp_path).expect("engine initialization failed");
        let base_ram = engine.total_ssd_state_bytes();

        let prompt = [12, 45, 108, 255];
        let report_a = engine
            .generate(&prompt, 16)
            .expect("first generation run failed");

        assert_eq!(report_a.generated_tokens.len(), 16);
        assert_eq!(report_a.ssd_context_bytes, base_ram);
        assert!(report_a.tokens_per_sec > 0.0);
        assert!(report_a.willneed_syscalls > 0);

        engine.reset_context();
        let report_b = engine
            .generate(&prompt, 16)
            .expect("second generation run failed");
        assert_eq!(report_a.generated_tokens, report_b.generated_tokens);
        assert_eq!(engine.total_ssd_state_bytes(), base_ram);

        let _ = std::fs::remove_file(&tmp_path);
    }
}
