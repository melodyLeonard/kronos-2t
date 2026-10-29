#![deny(unsafe_op_in_unsafe_fn)]
//! Sparse-MoE router, calibrated cross-layer subspace lookahead, streaming `.k2tm` writer,
//! and hybrid `mmap` / `F_NOCACHE` NVMe SLRU runtime with speculative batch expert coalescing.

use crate::hadamard::fwht64_signed_inplace;
use crate::memory_budget::MemoryBudgetProfile;
use crate::mmap_pager::sys::RawMmap;
use crate::moe_800b::{
    nf_expert_byte_layout, nvme_io, pack_nf_expert_record, DirectNvmeShard, DirectNvmeSlruPager,
    K2tmShardHeader, MmapNfExpertView, NfExpertByteLayout,
};
use crate::prefetch::SubspaceProjector;
use crate::safetensors::SafeTensorsStore;
use crate::simd_lut::{
    nf_sdot_batch_stripe_range, nf_sdot_stripe_range_int8, BipolarActivationLut,
    Int8ChunkActivation, NfSdotMatrix,
};
use crate::thread_pool::SpinThreadPool;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::Ordering;

const K2TM_PAGE_BYTES: usize = 16_384;

/// Router probability scoring convention across Hugging Face MoE families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeRouterScoring {
    /// Full-expert softmax followed by top-k selection and optional renormalization (`Qwen2Moe`, `Qwen3Moe`).
    Softmax,
    /// Top-k raw logit selection followed by k-way softmax (`GptOss`).
    TopkSoftmax,
    /// Elementwise sigmoid followed by top-k selection and optional renormalization.
    Sigmoid,
}

impl MoeRouterScoring {
    pub fn to_tag(self) -> usize {
        match self {
            Self::Softmax => 0,
            Self::TopkSoftmax => 1,
            Self::Sigmoid => 2,
        }
    }

    pub fn from_tag(tag: usize) -> Self {
        match tag {
            1 => Self::TopkSoftmax,
            2 => Self::Sigmoid,
            _ => Self::Softmax,
        }
    }
}

/// Dense `[num_experts, hidden_size]` router with optional per-expert bias.
#[derive(Clone, Debug)]
pub struct DenseMoeRouter {
    pub num_experts: usize,
    pub hidden_size: usize,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl DenseMoeRouter {
    pub fn new(num_experts: usize, hidden_size: usize, weights: Vec<f32>, bias: Option<Vec<f32>>) -> Self {
        assert!(num_experts > 1 && hidden_size > 0);
        assert_eq!(weights.len(), num_experts * hidden_size);
        if let Some(ref b) = bias {
            assert_eq!(b.len(), num_experts);
        }
        Self {
            num_experts,
            hidden_size,
            weights,
            bias,
        }
    }

    pub fn physical_bytes(&self) -> usize {
        (self.weights.len() + self.bias.as_ref().map_or(0, |b| b.len())) * std::mem::size_of::<f32>()
    }

    /// Computes top-`k` `(expert_id, weight)` pairs sorted by ascending `expert_id`.
    /// Sorting by `expert_id` keeps `.k2tm` reads monotonic and makes single-token and
    /// multi-token coalesced accumulation bit-identical in floating-point addition order.
    pub fn route_sorted(
        &self,
        x: &[f32],
        top_k: usize,
        norm_topk_prob: bool,
        scoring: MoeRouterScoring,
        out: &mut [(usize, f32)],
    ) -> usize {
        debug_assert_eq!(x.len(), self.hidden_size);
        let k = top_k.min(self.num_experts).min(out.len());
        if k == 0 {
            return 0;
        }

        let mut logits_stack = [0.0f32; 256];
        let mut logits_heap = Vec::new();
        let logits: &mut [f32] = if self.num_experts <= logits_stack.len() {
            &mut logits_stack[..self.num_experts]
        } else {
            logits_heap.resize(self.num_experts, 0.0);
            &mut logits_heap
        };

        for (e, dst) in logits.iter_mut().enumerate() {
            let row = &self.weights[e * self.hidden_size..(e + 1) * self.hidden_size];
            let mut v = dot_f32_moe(row, x);
            if let Some(ref b) = self.bias {
                v += b[e];
            }
            *dst = v;
        }

        match scoring {
            MoeRouterScoring::TopkSoftmax => {
                select_topk_by_score(logits, k, out);
                let max_z = out[..k].iter().fold(f32::NEG_INFINITY, |m, &(_, z)| m.max(z));
                let mut sum = 0.0f32;
                for (_, w) in &mut out[..k] {
                    *w = (*w - max_z).exp();
                    sum += *w;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 1.0 / (k as f32) };
                for (_, w) in &mut out[..k] {
                    *w *= inv;
                }
            }
            MoeRouterScoring::Softmax => {
                let max_z = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for z in logits.iter_mut() {
                    *z = (*z - max_z).exp();
                    sum += *z;
                }
                let inv_all = if sum > 0.0 { 1.0 / sum } else { 1.0 / (self.num_experts as f32) };
                for z in logits.iter_mut() {
                    *z *= inv_all;
                }
                select_topk_by_score(logits, k, out);
                if norm_topk_prob {
                    let top_sum: f32 = out[..k].iter().map(|&(_, w)| w).sum();
                    if top_sum > 0.0 {
                        let inv_top = 1.0 / top_sum;
                        for (_, w) in &mut out[..k] {
                            *w *= inv_top;
                        }
                    }
                }
            }
            MoeRouterScoring::Sigmoid => {
                for z in logits.iter_mut() {
                    *z = 1.0 / (1.0 + (-*z).exp());
                }
                select_topk_by_score(logits, k, out);
                if norm_topk_prob {
                    let top_sum: f32 = out[..k].iter().map(|&(_, w)| w).sum();
                    if top_sum > 0.0 {
                        let inv_top = 1.0 / top_sum;
                        for (_, w) in &mut out[..k] {
                            *w *= inv_top;
                        }
                    }
                }
            }
        }

        // Insertion sort `k` items by ascending `expert_id` (`k` is typically 2..=8).
        for i in 1..k {
            let cur = out[i];
            let mut j = i;
            while j > 0 && out[j - 1].0 > cur.0 {
                out[j] = out[j - 1];
                j -= 1;
            }
            out[j] = cur;
        }
        k
    }
}

#[inline]
fn select_topk_by_score(scores: &[f32], k: usize, out: &mut [(usize, f32)]) {
    let mut len = 0usize;
    for (e, &s) in scores.iter().enumerate() {
        if len < k {
            let mut pos = len;
            while pos > 0 && s > out[pos - 1].1 {
                out[pos] = out[pos - 1];
                pos -= 1;
            }
            out[pos] = (e, s);
            len += 1;
        } else if s > out[k - 1].1 {
            let mut pos = k - 1;
            while pos > 0 && s > out[pos - 1].1 {
                out[pos] = out[pos - 1];
                pos -= 1;
            }
            out[pos] = (e, s);
        }
    }
}

/// Optional always-active shared SwiGLU expert (`Qwen2MoeForCausalLM`).
pub enum SharedExpertBackend {
    Fp32 {
        hidden_size: usize,
        inter_dim: usize,
        gate_up: Vec<f32>,
        down: Vec<f32>,
    },
    Nf {
        inter_dim: usize,
        gate_up: NfSdotMatrix,
        down: NfSdotMatrix,
    },
}

pub struct SharedMoeExpert {
    pub inter_dim: usize,
    pub backend: SharedExpertBackend,
    pub gate_weight: Option<Vec<f32>>,
}

impl SharedMoeExpert {
    pub fn physical_bytes(&self) -> usize {
        let gate_b = self.gate_weight.as_ref().map_or(0, |g| g.len() * 4);
        let w_b = match &self.backend {
            SharedExpertBackend::Fp32 { gate_up, down, .. } => (gate_up.len() + down.len()) * 4,
            SharedExpertBackend::Nf { gate_up, down, .. } => {
                gate_up.physical_bytes() + down.physical_bytes()
            }
        };
        gate_b + w_b
    }

    #[inline]
    fn gate_scale(&self, x_norm: &[f32]) -> f32 {
        match &self.gate_weight {
            Some(gw) => {
                let z = dot_f32_moe(gw, x_norm);
                1.0 / (1.0 + (-z).exp())
            }
            None => 1.0,
        }
    }

    pub fn forward_single(
        &self,
        x_norm: &[f32],
        act_in: &Int8ChunkActivation,
        gu_buf: &mut [f32],
        sw_buf: &mut [f32],
        act_mid: &mut Int8ChunkActivation,
        out: &mut [f32],
    ) {
        let inter = self.inter_dim;
        let scale = self.gate_scale(x_norm);
        match &self.backend {
            SharedExpertBackend::Fp32 {
                hidden_size,
                gate_up,
                down,
                ..
            } => {
                let h = *hidden_size;
                for (r, dst) in gu_buf[..2 * inter].iter_mut().enumerate() {
                    *dst = dot_f32_moe(&gate_up[r * h..(r + 1) * h], x_norm);
                }
                let (g, u) = gu_buf[..2 * inter].split_at(inter);
                for ((dst, &gv), &uv) in sw_buf[..inter].iter_mut().zip(g).zip(u) {
                    *dst = (gv / (1.0 + (-gv).exp())) * uv;
                }
                for (r, dst) in out.iter_mut().enumerate() {
                    *dst = scale * dot_f32_moe(&down[r * inter..(r + 1) * inter], &sw_buf[..inter]);
                }
            }
            SharedExpertBackend::Nf { gate_up, down, .. } => {
                debug_assert_eq!(act_in.cols, gate_up.cols);
                // SAFETY: `gu_buf` has at least `2 * inter == gate_up.rows` elements.
                unsafe {
                    nf_sdot_stripe_range_int8(
                        gate_up,
                        act_in,
                        0,
                        gate_up.num_stripes,
                        gu_buf.as_mut_ptr(),
                    );
                }
                let (g, u) = gu_buf[..2 * inter].split_at(inter);
                for ((dst, &gv), &uv) in sw_buf[..inter].iter_mut().zip(g).zip(u) {
                    *dst = (gv / (1.0 + (-gv).exp())) * uv;
                }
                fwht64_signed_inplace(&mut sw_buf[..inter]);
                if act_mid.cols != inter {
                    *act_mid = Int8ChunkActivation::new_zeroed(inter);
                }
                act_mid.populate(&sw_buf[..inter]);
                // SAFETY: `out` has `down.rows` elements.
                unsafe {
                    nf_sdot_stripe_range_int8(down, act_mid, 0, down.num_stripes, out.as_mut_ptr());
                }
                if (scale - 1.0).abs() > f32::EPSILON {
                    for v in out.iter_mut() {
                        *v *= scale;
                    }
                }
            }
        }
    }
}

/// Per-layer pinned MoE metadata (router, optional shared expert, optional 3-layer-ahead projector).
pub struct MoeLayerCore {
    pub moe_slot_idx: usize,
    pub router: DenseMoeRouter,
    pub shared_expert: Option<SharedMoeExpert>,
    pub lookahead: Option<(usize, SubspaceProjector)>,
}

impl MoeLayerCore {
    pub fn physical_bytes(&self) -> usize {
        self.router.physical_bytes()
            + self.shared_expert.as_ref().map_or(0, |s| s.physical_bytes())
            + self
                .lookahead
                .as_ref()
                .map_or(0, |(_, p)| (p.u_proj.len() + p.v_proj.len()) * 4)
    }
}

/// Fits a rank-`r` residual drift projector `P(h) = h + U (V^T h)` mapping `src_hiddens -> dst_hiddens`
/// via deflated orthogonal power iteration for `U` and exact dual Cholesky ridge regression for `V`.
pub fn fit_subspace_projector(
    src_hiddens: &[f32],
    dst_hiddens: &[f32],
    d_model: usize,
    rank: usize,
    ridge_lambda: f32,
) -> SubspaceProjector {
    assert!(d_model > 0 && rank > 0);
    assert_eq!(src_hiddens.len(), dst_hiddens.len());
    let total_n = src_hiddens.len() / d_model;
    let r_eff = rank.min(d_model).min(total_n.max(1));
    if total_n < 2 {
        return SubspaceProjector::new(d_model, r_eff.max(1), 0x5EED_CAFE);
    }

    // Subsample up to 256 representative tokens so dual Gram solve stays `O(M^3)` (`< 1 ms`).
    let m = total_n.min(256);
    let mut h_src = vec![0.0f32; m * d_model];
    let mut delta = vec![0.0f32; m * d_model];
    for i in 0..m {
        let idx = (i * total_n) / m;
        let s = &src_hiddens[idx * d_model..(idx + 1) * d_model];
        let d = &dst_hiddens[idx * d_model..(idx + 1) * d_model];
        h_src[i * d_model..(i + 1) * d_model].copy_from_slice(s);
        for (dst_v, (&dv, &sv)) in delta[i * d_model..(i + 1) * d_model]
            .iter_mut()
            .zip(d.iter().zip(s.iter()))
        {
            *dst_v = dv - sv;
        }
    }

    // 1. Deflated power iteration on `Delta^T Delta` for orthonormal rows `U` (`[r_eff, d_model]`).
    let mut u_proj = vec![0.0f32; r_eff * d_model];
    let mut residual = delta.clone();
    let mut scores = vec![0.0f32; m];
    for comp in 0..r_eff {
        let mut u = vec![0.0f32; d_model];
        for (j, v) in u.iter_mut().enumerate() {
            let phase = ((comp * 131 + j * 17 + 1) as f32) * 0.37;
            *v = phase.sin();
        }
        normalize_vec(&mut u);

        for _ in 0..10 {
            for (i, sc) in scores.iter_mut().enumerate() {
                *sc = dot_f32_moe(&residual[i * d_model..(i + 1) * d_model], &u);
            }
            u.fill(0.0);
            for (i, &sc) in scores.iter().enumerate() {
                let row = &residual[i * d_model..(i + 1) * d_model];
                for (uj, &rj) in u.iter_mut().zip(row.iter()) {
                    *uj += sc * rj;
                }
            }
            // Orthogonalize against previously extracted components.
            for prev in 0..comp {
                let pu = &u_proj[prev * d_model..(prev + 1) * d_model];
                let proj = dot_f32_moe(&u, pu);
                for (uj, &puj) in u.iter_mut().zip(pu.iter()) {
                    *uj -= proj * puj;
                }
            }
            if normalize_vec(&mut u) < 1e-8 {
                break;
            }
        }
        // Deflate `residual -= (residual * u) u^T`.
        for i in 0..m {
            let row = &mut residual[i * d_model..(i + 1) * d_model];
            let sc = dot_f32_moe(row, &u);
            for (rj, &uj) in row.iter_mut().zip(u.iter()) {
                *rj -= sc * uj;
            }
        }
        u_proj[comp * d_model..(comp + 1) * d_model].copy_from_slice(&u);
    }

    // 2. Target latent matrix `Z = Delta * U^T` (`[m, r_eff]`).
    let mut z_mat = vec![0.0f64; m * r_eff];
    for i in 0..m {
        let d_row = &delta[i * d_model..(i + 1) * d_model];
        for c in 0..r_eff {
            let u_row = &u_proj[c * d_model..(c + 1) * d_model];
            z_mat[i * r_eff + c] = dot_f32_moe(d_row, u_row) as f64;
        }
    }

    // 3. Dual ridge Gram matrix `G = H_src H_src^T + lambda * scale * I_m` (`[m, m]`).
    let mut gram = vec![0.0f64; m * m];
    let mut trace = 0.0f64;
    for i in 0..m {
        let hi = &h_src[i * d_model..(i + 1) * d_model];
        let diag = dot_f32_moe(hi, hi) as f64;
        gram[i * m + i] = diag;
        trace += diag;
        for j in (i + 1)..m {
            let hj = &h_src[j * d_model..(j + 1) * d_model];
            let v = dot_f32_moe(hi, hj) as f64;
            gram[i * m + j] = v;
            gram[j * m + i] = v;
        }
    }
    let reg = ((trace / (m as f64)).max(1e-6)) * (ridge_lambda.max(1e-4) as f64);
    for i in 0..m {
        gram[i * m + i] += reg;
    }

    // Cholesky `G = L L^T` and solve `G A = Z` for `A` (`[m, r_eff]`).
    let mut chol_l = vec![0.0f64; m * m];
    for i in 0..m {
        for j in 0..=i {
            let mut sum = gram[i * m + j];
            for p in 0..j {
                sum -= chol_l[i * m + p] * chol_l[j * m + p];
            }
            if i == j {
                chol_l[i * m + i] = sum.max(1e-12).sqrt();
            } else {
                chol_l[i * m + j] = sum / chol_l[j * m + j];
            }
        }
    }

    let mut alpha = vec![0.0f64; m * r_eff];
    let mut y_col = vec![0.0f64; m];
    for c in 0..r_eff {
        for i in 0..m {
            let mut s = z_mat[i * r_eff + c];
            for p in 0..i {
                s -= chol_l[i * m + p] * y_col[p];
            }
            y_col[i] = s / chol_l[i * m + i];
        }
        for i in (0..m).rev() {
            let mut s = y_col[i];
            for p in (i + 1)..m {
                s -= chol_l[p * m + i] * alpha[p * r_eff + c];
            }
            alpha[i * r_eff + c] = s / chol_l[i * m + i];
        }
    }

    // `V = A^T H_src` (`[r_eff, d_model]`).
    let mut v_proj = vec![0.0f32; r_eff * d_model];
    for c in 0..r_eff {
        let v_row = &mut v_proj[c * d_model..(c + 1) * d_model];
        for i in 0..m {
            let a = alpha[i * r_eff + c] as f32;
            let hi = &h_src[i * d_model..(i + 1) * d_model];
            for (vj, &hj) in v_row.iter_mut().zip(hi.iter()) {
                *vj += a * hj;
            }
        }
    }

    SubspaceProjector {
        d_model,
        rank: r_eff,
        u_proj,
        v_proj,
    }
}

fn normalize_vec(v: &mut [f32]) -> f32 {
    let n2 = dot_f32_moe(v, v);
    if n2 <= 1e-20 {
        return 0.0;
    }
    let norm = n2.sqrt();
    let inv = 1.0 / norm;
    for x in v.iter_mut() {
        *x *= inv;
    }
    norm
}

/// Streaming writer that appends 16 KiB-aligned quantized experts directly to `.k2tm` on disk
/// with `F_NOCACHE`, bounding compile-time expert RAM to a single expert.
pub struct StreamingK2tmWriter {
    writer: BufWriter<File>,
    pub header: K2tmShardHeader,
    pub layout: NfExpertByteLayout,
    record_buf: Vec<u8>,
    written_experts: usize,
}

impl StreamingK2tmWriter {
    pub fn create(
        path: &Path,
        num_moe_layers: usize,
        experts_per_layer: usize,
        d_model: usize,
        d_ff_expert: usize,
        bits: usize,
        scale_group_chunks: usize,
    ) -> io::Result<Self> {
        let layout = nf_expert_byte_layout(d_model, d_ff_expert, bits, scale_group_chunks);
        let header = K2tmShardHeader {
            num_layers: num_moe_layers,
            experts_per_layer,
            d_model,
            d_ff_expert,
            scale_group_chunks: layout.scale_group_chunks,
            aligned_expert_bytes: layout.aligned_expert_bytes,
            bits,
        };
        let file = File::create(path)?;
        #[cfg(unix)]
        let _ = nvme_io::set_file_nocache(&file);
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);
        writer.write_all(&header.encode_page())?;
        Ok(Self {
            writer,
            header,
            layout,
            record_buf: vec![0u8; layout.aligned_expert_bytes],
            written_experts: 0,
        })
    }

    pub fn write_expert(&mut self, gu: &NfSdotMatrix, dn: &NfSdotMatrix) -> io::Result<()> {
        pack_nf_expert_record(&self.layout, gu, dn, &mut self.record_buf);
        self.writer.write_all(&self.record_buf)?;
        self.written_experts += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<K2tmShardHeader> {
        let expected = self.header.num_layers * self.header.experts_per_layer;
        if self.written_experts != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "StreamingK2tmWriter expected {expected} experts, wrote {}",
                    self.written_experts
                ),
            ));
        }
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        Ok(self.header)
    }
}

/// Expert shard loading policy under the active [`MemoryBudgetProfile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeExpertOpenPolicy {
    /// Zero-copy `mmap` when the shard fits inside the active budget arena; `F_NOCACHE` NVMe SLRU otherwise.
    AutoBudget,
    /// Forces `DirectNvmeSlruPager` (`F_NOCACHE` physical `pread`) with `capacity_experts` DRAM slots.
    ForceNvmeSlru { capacity_experts: usize },
    /// Forces zero-copy read-only `mmap` (`Mobile5G` `e8q2` envelope).
    ForceZeroCopyMmap,
}

/// Live telemetry from [`SparseMoeRuntime`].
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeRuntimeStats {
    pub is_nvme_slru: bool,
    pub slru_capacity_experts: usize,
    pub slru_arena_bytes: usize,
    pub shard_disk_bytes: usize,
    pub dram_hits: usize,
    pub protected_hits: usize,
    pub async_prefetch_hits: usize,
    pub nvme_misses: usize,
    pub hit_rate_percent: f64,
    pub nvme_bytes_read: u64,
}

enum MoeExpertBackend {
    Fp32Ref {
        d_model: usize,
        d_ff: usize,
        /// `[moe_layer_slot][expert_id] -> (gate_up_f32, down_f32)`.
        layers: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    },
    ZeroCopyMmap {
        mmap: RawMmap,
        header: K2tmShardHeader,
        layout: NfExpertByteLayout,
        accesses: usize,
    },
    #[cfg(unix)]
    NvmeSlru {
        pager: Box<std::sync::Mutex<DirectNvmeSlruPager>>,
        layout: NfExpertByteLayout,
    },
}

/// Zero-allocation Sparse-MoE runtime supporting single-token decode, 3-layer-ahead async NVMe
/// prefetching, and `N = 1 + K` speculative-tree / batch expert coalescing.
pub struct SparseMoeRuntime {
    backend: MoeExpertBackend,
    pub d_model: usize,
    pub d_ff_expert: usize,
    pub num_experts: usize,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub router_scoring: MoeRouterScoring,
    act_in: Int8ChunkActivation,
    act_mid: Int8ChunkActivation,
    shared_act_mid: Int8ChunkActivation,
    gate_up_buf: Vec<f32>,
    sw_buf: Vec<f32>,
    expert_out: Vec<f32>,
    shared_gu_buf: Vec<f32>,
    shared_sw_buf: Vec<f32>,
    prefetch_proj_buf: Vec<f32>,
    prefetch_norm_buf: Vec<f32>,
    batch_in_acts: Vec<Int8ChunkActivation>,
    batch_mid_acts: Vec<Int8ChunkActivation>,
    batch_gu: Vec<f32>,
    batch_dn: Vec<f32>,
    expert_buckets: Vec<Vec<(usize, f32)>>,
}

impl SparseMoeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub fn new_fp32_reference(
        d_model: usize,
        d_ff_expert: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        router_scoring: MoeRouterScoring,
        shared_inter_dim: usize,
        layers: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    ) -> Self {
        Self::with_backend(
            MoeExpertBackend::Fp32Ref {
                d_model,
                d_ff: d_ff_expert,
                layers,
            },
            d_model,
            d_ff_expert,
            num_experts,
            top_k,
            norm_topk_prob,
            router_scoring,
            shared_inter_dim,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn open_k2tm(
        k2tm_path: &Path,
        d_model: usize,
        d_ff_expert: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        router_scoring: MoeRouterScoring,
        shared_inter_dim: usize,
        fallback_bits: usize,
        scale_group_chunks: usize,
        pinned_core_bytes: usize,
        kv_and_scratch_bytes: usize,
        draft_bytes: usize,
        policy: MoeExpertOpenPolicy,
    ) -> io::Result<Self> {
        let probe_file = File::open(k2tm_path)?;
        let file_len = probe_file.metadata()?.len() as usize;
        if file_len < K2TM_PAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2tm file shorter than 16 KiB header page",
            ));
        }
        let hdr_mmap = RawMmap::map_read_only(&probe_file, K2TM_PAGE_BYTES)?;
        let header = K2tmShardHeader::decode_page(hdr_mmap.as_slice())
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        drop(hdr_mmap);

        let bits = if (2..=5).contains(&header.bits) {
            header.bits
        } else {
            fallback_bits
        };
        let sgc = if header.scale_group_chunks > 0 {
            header.scale_group_chunks
        } else {
            scale_group_chunks.max(1)
        };
        let layout = nf_expert_byte_layout(d_model, d_ff_expert, bits, sgc);
        if header.aligned_expert_bytes != layout.aligned_expert_bytes
            || header.d_model != d_model
            || header.d_ff_expert != d_ff_expert
            || header.experts_per_layer != num_experts
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2tm header geometry mismatch against Sparse-MoE config",
            ));
        }

        let profile = MemoryBudgetProfile::active();
        let arena_budget =
            profile.slru_expert_arena_bytes(pinned_core_bytes, kv_and_scratch_bytes, draft_bytes);

        let backend = match policy {
            MoeExpertOpenPolicy::ForceZeroCopyMmap => {
                let _ = nvme_io::clear_file_nocache(&probe_file);
                let mmap = RawMmap::map_read_only(&probe_file, file_len)?;
                MoeExpertBackend::ZeroCopyMmap {
                    mmap,
                    header,
                    layout,
                    accesses: 0,
                }
            }
            MoeExpertOpenPolicy::AutoBudget if file_len <= arena_budget => {
                let _ = nvme_io::clear_file_nocache(&probe_file);
                let mmap = RawMmap::map_read_only(&probe_file, file_len)?;
                MoeExpertBackend::ZeroCopyMmap {
                    mmap,
                    header,
                    layout,
                    accesses: 0,
                }
            }
            #[cfg(unix)]
            MoeExpertOpenPolicy::ForceNvmeSlru { capacity_experts } => {
                drop(probe_file);
                let shard = DirectNvmeShard::open(k2tm_path, layout.aligned_expert_bytes)?;
                let total_experts = header.num_layers * header.experts_per_layer;
                let cap = capacity_experts.clamp(2, total_experts.max(2));
                let pager = Box::new(std::sync::Mutex::new(DirectNvmeSlruPager::new(
                    shard, cap, true, true,
                )?));
                MoeExpertBackend::NvmeSlru { pager, layout }
            }
            #[cfg(unix)]
            MoeExpertOpenPolicy::AutoBudget => {
                drop(probe_file);
                let shard = DirectNvmeShard::open(k2tm_path, layout.aligned_expert_bytes)?;
                let total_experts = header.num_layers * header.experts_per_layer;
                let min_cap = (top_k * 3).max(4);
                let cap = (arena_budget / layout.aligned_expert_bytes)
                    .max(min_cap)
                    .min(total_experts.max(min_cap));
                let pager = Box::new(std::sync::Mutex::new(DirectNvmeSlruPager::new(
                    shard, cap, true, true,
                )?));
                MoeExpertBackend::NvmeSlru { pager, layout }
            }
        };

        Ok(Self::with_backend(
            backend,
            d_model,
            d_ff_expert,
            num_experts,
            top_k,
            norm_topk_prob,
            router_scoring,
            shared_inter_dim,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn with_backend(
        backend: MoeExpertBackend,
        d_model: usize,
        d_ff_expert: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        router_scoring: MoeRouterScoring,
        shared_inter_dim: usize,
    ) -> Self {
        let s_inter = shared_inter_dim.max(64);
        Self {
            backend,
            d_model,
            d_ff_expert,
            num_experts,
            top_k,
            norm_topk_prob,
            router_scoring,
            act_in: Int8ChunkActivation::new_zeroed(d_model),
            act_mid: Int8ChunkActivation::new_zeroed(d_ff_expert),
            shared_act_mid: Int8ChunkActivation::new_zeroed(s_inter),
            gate_up_buf: vec![0.0f32; 2 * d_ff_expert],
            sw_buf: vec![0.0f32; d_ff_expert],
            expert_out: vec![0.0f32; d_model],
            shared_gu_buf: vec![0.0f32; 2 * s_inter],
            shared_sw_buf: vec![0.0f32; s_inter],
            prefetch_proj_buf: vec![0.0f32; d_model],
            prefetch_norm_buf: vec![0.0f32; d_model],
            batch_in_acts: Vec::new(),
            batch_mid_acts: Vec::new(),
            batch_gu: Vec::new(),
            batch_dn: Vec::new(),
            expert_buckets: (0..num_experts).map(|_| Vec::with_capacity(8)).collect(),
        }
    }

    /// Whether this runtime holds unquantized `FP32` reference experts.
    #[inline]
    pub fn is_fp32_reference(&self) -> bool {
        matches!(self.backend, MoeExpertBackend::Fp32Ref { .. })
    }

    /// Resident bytes occupied by the expert arena (SLRU DRAM slots, `mmap` file bytes, or FP32 vectors).
    pub fn resident_arena_bytes(&self) -> usize {
        match &self.backend {
            MoeExpertBackend::Fp32Ref { layers, .. } => layers
                .iter()
                .flat_map(|l| l.iter())
                .map(|(gu, dn)| (gu.len() + dn.len()) * 4)
                .sum(),
            MoeExpertBackend::ZeroCopyMmap { header, .. } => header.total_file_bytes(),
            #[cfg(unix)]
            MoeExpertBackend::NvmeSlru { pager, .. } => pager
                .lock()
                .expect("SLRU pager mutex poisoned")
                .arena_bytes(),
        }
    }

    pub fn stats(&self) -> MoeRuntimeStats {
        match &self.backend {
            MoeExpertBackend::Fp32Ref { .. } => MoeRuntimeStats::default(),
            MoeExpertBackend::ZeroCopyMmap {
                header, accesses, ..
            } => MoeRuntimeStats {
                is_nvme_slru: false,
                slru_capacity_experts: header.num_layers * header.experts_per_layer,
                slru_arena_bytes: header.total_file_bytes(),
                shard_disk_bytes: header.total_file_bytes(),
                dram_hits: *accesses,
                protected_hits: *accesses,
                async_prefetch_hits: 0,
                nvme_misses: 0,
                hit_rate_percent: if *accesses > 0 { 100.0 } else { 0.0 },
                nvme_bytes_read: 0,
            },
            #[cfg(unix)]
            MoeExpertBackend::NvmeSlru { pager, .. } => {
                let p = pager.lock().expect("SLRU pager mutex poisoned");
                MoeRuntimeStats {
                    is_nvme_slru: true,
                    slru_capacity_experts: p.total_capacity,
                    slru_arena_bytes: p.arena_bytes(),
                    shard_disk_bytes: p.shard.header.total_file_bytes(),
                    dram_hits: p.dram_hits,
                    protected_hits: p.protected_hits,
                    async_prefetch_hits: p.async_prefetch_hits,
                    nvme_misses: p.nvme_misses,
                    hit_rate_percent: p.hit_rate_percent(),
                    nvme_bytes_read: p.shard.nvme_bytes_read.load(Ordering::Relaxed),
                }
            }
        }
    }

    pub fn clear_slru_cache(&mut self) {
        #[cfg(unix)]
        if let MoeExpertBackend::NvmeSlru { pager, .. } = &mut self.backend {
            pager
                .get_mut()
                .expect("SLRU pager mutex poisoned")
                .clear_cache();
        }
    }

    /// Projects post-attention `hidden` 3 layers ahead and queues background `F_NOCACHE` `pread`s
    /// for predicted future experts.
    pub fn dispatch_lookahead_prefetch(
        &mut self,
        post_attn_hidden: &[f32],
        projector: &SubspaceProjector,
        future_post_attn_norm: &[f32],
        future_moe_slot: usize,
        future_router: &DenseMoeRouter,
        rms_eps: f32,
    ) {
        #[cfg(unix)]
        if let MoeExpertBackend::NvmeSlru { pager, .. } = &mut self.backend {
            let p = pager.get_mut().expect("SLRU pager mutex poisoned");
            projector.project(post_attn_hidden, &mut self.prefetch_proj_buf);
            rms_norm_moe(
                &self.prefetch_proj_buf,
                future_post_attn_norm,
                rms_eps,
                &mut self.prefetch_norm_buf,
            );
            let mut routes = [(0usize, 0.0f32); 16];
            let k = future_router.route_sorted(
                &self.prefetch_norm_buf,
                self.top_k.min(routes.len()),
                self.norm_topk_prob,
                self.router_scoring,
                &mut routes,
            );
            for &(exp_id, _) in &routes[..k] {
                p.prefetch_expert_async(future_moe_slot, exp_id);
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (
                post_attn_hidden,
                projector,
                future_post_attn_norm,
                future_moe_slot,
                future_router,
                rms_eps,
            );
        }
    }

    /// Executes one Sparse-MoE layer on a single post-RMSNorm token `x_norm` (`[d_model]`),
    /// writing the combined shared + routed expert residual contribution into `out` (`[d_model]`).
    pub fn forward_layer_single(
        &mut self,
        moe_core: &MoeLayerCore,
        x_norm: &[f32],
        pool: Option<&SpinThreadPool>,
        out: &mut [f32],
    ) {
        debug_assert_eq!(x_norm.len(), self.d_model);
        debug_assert_eq!(out.len(), self.d_model);

        let mut routes_stack = [(0usize, 0.0f32); 32];
        let mut routes_heap = Vec::new();
        let routes_buf: &mut [(usize, f32)] = if self.top_k <= routes_stack.len() {
            &mut routes_stack[..self.top_k]
        } else {
            routes_heap.resize(self.top_k, (0, 0.0));
            &mut routes_heap
        };
        let k = moe_core.router.route_sorted(
            x_norm,
            self.top_k,
            self.norm_topk_prob,
            self.router_scoring,
            routes_buf,
        );

        let need_int8 = !matches!(self.backend, MoeExpertBackend::Fp32Ref { .. })
            || moe_core
                .shared_expert
                .as_ref()
                .is_some_and(|s| matches!(s.backend, SharedExpertBackend::Nf { .. }));
        if need_int8 {
            self.act_in.populate_fwht(x_norm);
        }

        if let Some(ref shared) = moe_core.shared_expert {
            if self.shared_gu_buf.len() < 2 * shared.inter_dim {
                self.shared_gu_buf.resize(2 * shared.inter_dim, 0.0);
                self.shared_sw_buf.resize(shared.inter_dim, 0.0);
            }
            shared.forward_single(
                x_norm,
                &self.act_in,
                &mut self.shared_gu_buf,
                &mut self.shared_sw_buf,
                &mut self.shared_act_mid,
                out,
            );
        } else {
            out.fill(0.0);
        }

        let slot = moe_core.moe_slot_idx;
        match &mut self.backend {
            MoeExpertBackend::Fp32Ref { d_model, d_ff, layers } => {
                let h = *d_model;
                let inter = *d_ff;
                for &(exp_id, weight) in &routes_buf[..k] {
                    let (ref gu, ref dn) = layers[slot][exp_id];
                    for (r, dst) in self.gate_up_buf[..2 * inter].iter_mut().enumerate() {
                        *dst = dot_f32_moe(&gu[r * h..(r + 1) * h], x_norm);
                    }
                    let (g, u) = self.gate_up_buf[..2 * inter].split_at(inter);
                    for ((dst, &gv), &uv) in self.sw_buf[..inter].iter_mut().zip(g).zip(u) {
                        *dst = (gv / (1.0 + (-gv).exp())) * uv;
                    }
                    for (r, dst) in out.iter_mut().enumerate() {
                        *dst += weight * dot_f32_moe(&dn[r * inter..(r + 1) * inter], &self.sw_buf[..inter]);
                    }
                }
            }
            MoeExpertBackend::ZeroCopyMmap {
                mmap,
                header,
                layout,
                accesses,
            } => {
                let lay = *layout;
                for &(exp_id, weight) in &routes_buf[..k] {
                    let off = header.expert_byte_offset(slot, exp_id);
                    let slice = &mmap.as_slice()[off..off + lay.aligned_expert_bytes];
                    *accesses += 1;
                    let view = MmapNfExpertView::from_aligned_slice(slice, lay)
                        .expect("valid mapped NF expert slice");
                    view.forward_swiglu_zero_copy(
                        &self.act_in,
                        &mut self.gate_up_buf,
                        &mut self.act_mid,
                        &mut self.expert_out,
                        pool,
                    );
                    for (dst, &src) in out.iter_mut().zip(self.expert_out.iter()) {
                        *dst += weight * src;
                    }
                }
            }
            #[cfg(unix)]
            MoeExpertBackend::NvmeSlru { pager, layout } => {
                let lay = *layout;
                let p = pager.get_mut().expect("SLRU pager mutex poisoned");
                // Queue remaining selected experts on this layer so NVMe overlaps with the first expert's GEMVs.
                for &(exp_id, _) in routes_buf[..k].iter().skip(1) {
                    p.prefetch_expert_async(slot, exp_id);
                }
                for &(exp_id, weight) in &routes_buf[..k] {
                    let slice = p
                        .access_expert_slice(slot, exp_id)
                        .expect("NVMe SLRU expert pread failed");
                    let view = MmapNfExpertView::from_aligned_slice(slice, lay)
                        .expect("valid SLRU NF expert slice");
                    view.forward_swiglu_zero_copy(
                        &self.act_in,
                        &mut self.gate_up_buf,
                        &mut self.act_mid,
                        &mut self.expert_out,
                        pool,
                    );
                    for (dst, &src) in out.iter_mut().zip(self.expert_out.iter()) {
                        *dst += weight * src;
                    }
                }
            }
        }
    }

    /// Executes one Sparse-MoE layer across `n` tokens (`hidden_batch`: `[n, d_model]`) by grouping
    /// all `(token_idx, weight)` routes by unique `expert_id` so each active expert is fetched from
    /// `.k2tm` / SLRU and multiplied at most once across the entire batch (`1 + K` speculative tokens).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_layer_coalesced_batch(
        &mut self,
        moe_core: &MoeLayerCore,
        hidden_batch: &[f32],
        post_attn_norm: &[f32],
        rms_eps: f32,
        n: usize,
        pool: Option<&SpinThreadPool>,
        norm_batch_scratch: &mut [f32],
        out_batch: &mut [f32],
    ) {
        let h = self.d_model;
        assert!(n > 0);
        assert!(hidden_batch.len() >= n * h);
        assert!(norm_batch_scratch.len() >= n * h);
        assert!(out_batch.len() >= n * h);

        for t in 0..n {
            rms_norm_moe(
                &hidden_batch[t * h..(t + 1) * h],
                post_attn_norm,
                rms_eps,
                &mut norm_batch_scratch[t * h..(t + 1) * h],
            );
        }

        if n == 1 {
            self.forward_layer_single(
                moe_core,
                &norm_batch_scratch[..h],
                pool,
                &mut out_batch[..h],
            );
            return;
        }

        while self.batch_in_acts.len() < n {
            self.batch_in_acts.push(Int8ChunkActivation::new_zeroed(h));
        }
        while self.batch_mid_acts.len() < n {
            self.batch_mid_acts
                .push(Int8ChunkActivation::new_zeroed(self.d_ff_expert));
        }
        if self.batch_gu.len() < n * 2 * self.d_ff_expert {
            self.batch_gu.resize(n * 2 * self.d_ff_expert, 0.0);
        }
        if self.batch_dn.len() < n * h {
            self.batch_dn.resize(n * h, 0.0);
        }

        let need_int8 = !matches!(self.backend, MoeExpertBackend::Fp32Ref { .. })
            || moe_core
                .shared_expert
                .as_ref()
                .is_some_and(|s| matches!(s.backend, SharedExpertBackend::Nf { .. }));
        if need_int8 {
            for t in 0..n {
                self.batch_in_acts[t].populate_fwht(&norm_batch_scratch[t * h..(t + 1) * h]);
            }
        }

        if let Some(ref shared) = moe_core.shared_expert {
            let inter = shared.inter_dim;
            match &shared.backend {
                SharedExpertBackend::Nf { gate_up, down, .. } if n >= 2 => {
                    let mut acts_b: Vec<BipolarActivationLut> = (0..n)
                        .map(|t| {
                            let mut b = BipolarActivationLut::new_zeroed(h);
                            b.lut_enabled = false;
                            b.int8 = Some(self.batch_in_acts[t].clone());
                            b
                        })
                        .collect();
                    let mut s_gu = vec![0.0f32; n * 2 * inter];
                    // SAFETY: `s_gu` holds `n * gate_up.rows` floats.
                    unsafe {
                        nf_sdot_batch_stripe_range(
                            gate_up,
                            &acts_b,
                            0,
                            gate_up.num_stripes,
                            s_gu.as_mut_ptr(),
                            gate_up.rows,
                        );
                    }
                    let mut acts_mid_b: Vec<BipolarActivationLut> = (0..n)
                        .map(|t| {
                            let gu_t = &s_gu[t * 2 * inter..(t + 1) * 2 * inter];
                            let (g, u) = gu_t.split_at(inter);
                            let mut sw = vec![0.0f32; inter];
                            for ((dst, &gv), &uv) in sw.iter_mut().zip(g).zip(u) {
                                *dst = (gv / (1.0 + (-gv).exp())) * uv;
                            }
                            let mut i8m = Int8ChunkActivation::new_zeroed(inter);
                            i8m.populate_fwht(&sw);
                            let mut b = BipolarActivationLut::new_zeroed(inter);
                            b.lut_enabled = false;
                            b.int8 = Some(i8m);
                            b
                        })
                        .collect();
                    // SAFETY: `out_batch` holds `n * down.rows` floats.
                    unsafe {
                        nf_sdot_batch_stripe_range(
                            down,
                            &acts_mid_b,
                            0,
                            down.num_stripes,
                            out_batch.as_mut_ptr(),
                            down.rows,
                        );
                    }
                    if shared.gate_weight.is_some() {
                        for t in 0..n {
                            let sc = shared.gate_scale(&norm_batch_scratch[t * h..(t + 1) * h]);
                            for v in &mut out_batch[t * h..(t + 1) * h] {
                                *v *= sc;
                            }
                        }
                    }
                    acts_b.clear();
                    acts_mid_b.clear();
                }
                _ => {
                    if self.shared_gu_buf.len() < 2 * inter {
                        self.shared_gu_buf.resize(2 * inter, 0.0);
                        self.shared_sw_buf.resize(inter, 0.0);
                    }
                    for t in 0..n {
                        shared.forward_single(
                            &norm_batch_scratch[t * h..(t + 1) * h],
                            &self.batch_in_acts[t],
                            &mut self.shared_gu_buf,
                            &mut self.shared_sw_buf,
                            &mut self.shared_act_mid,
                            &mut out_batch[t * h..(t + 1) * h],
                        );
                    }
                }
            }
        } else {
            out_batch[..n * h].fill(0.0);
        }

        for b in &mut self.expert_buckets {
            b.clear();
        }
        let mut routes_stack = [(0usize, 0.0f32); 32];
        let mut routes_heap = Vec::new();
        let routes_buf: &mut [(usize, f32)] = if self.top_k <= routes_stack.len() {
            &mut routes_stack[..self.top_k]
        } else {
            routes_heap.resize(self.top_k, (0, 0.0));
            &mut routes_heap
        };

        for t in 0..n {
            let k = moe_core.router.route_sorted(
                &norm_batch_scratch[t * h..(t + 1) * h],
                self.top_k,
                self.norm_topk_prob,
                self.router_scoring,
                routes_buf,
            );
            for &(exp_id, weight) in &routes_buf[..k] {
                self.expert_buckets[exp_id].push((t, weight));
            }
        }

        let slot = moe_core.moe_slot_idx;
        #[cfg(unix)]
        if let MoeExpertBackend::NvmeSlru { pager, .. } = &mut self.backend {
            let p = pager.get_mut().expect("SLRU pager mutex poisoned");
            let mut queued = 0usize;
            for (exp_id, bucket) in self.expert_buckets.iter().enumerate() {
                if !bucket.is_empty() {
                    p.prefetch_expert_async(slot, exp_id);
                    queued += 1;
                    if queued >= 8 {
                        break;
                    }
                }
            }
        }

        let mut act_refs: Vec<&Int8ChunkActivation> = Vec::with_capacity(n);
        for (exp_id, bucket) in self.expert_buckets.iter().enumerate() {
            if bucket.is_empty() {
                continue;
            }
            let m = bucket.len();
            match &mut self.backend {
                MoeExpertBackend::Fp32Ref { d_model, d_ff, layers } => {
                    let hd = *d_model;
                    let inter = *d_ff;
                    let (ref gu, ref dn) = layers[slot][exp_id];
                    for &(tok_idx, weight) in bucket.iter() {
                        let x_t = &norm_batch_scratch[tok_idx * hd..(tok_idx + 1) * hd];
                        for (r, dst) in self.gate_up_buf[..2 * inter].iter_mut().enumerate() {
                            *dst = dot_f32_moe(&gu[r * hd..(r + 1) * hd], x_t);
                        }
                        let (g, u) = self.gate_up_buf[..2 * inter].split_at(inter);
                        for ((dst, &gv), &uv) in self.sw_buf[..inter].iter_mut().zip(g).zip(u) {
                            *dst = (gv / (1.0 + (-gv).exp())) * uv;
                        }
                        let out_t = &mut out_batch[tok_idx * hd..(tok_idx + 1) * hd];
                        for (r, dst) in out_t.iter_mut().enumerate() {
                            *dst += weight * dot_f32_moe(&dn[r * inter..(r + 1) * inter], &self.sw_buf[..inter]);
                        }
                    }
                }
                MoeExpertBackend::ZeroCopyMmap {
                    mmap,
                    header,
                    layout,
                    accesses,
                } => {
                    let lay = *layout;
                    let off = header.expert_byte_offset(slot, exp_id);
                    let slice = &mmap.as_slice()[off..off + lay.aligned_expert_bytes];
                    *accesses += 1;
                    let view = MmapNfExpertView::from_aligned_slice(slice, lay)
                        .expect("valid mapped NF expert slice");
                    act_refs.clear();
                    for &(tok_idx, _) in bucket.iter() {
                        act_refs.push(&self.batch_in_acts[tok_idx]);
                    }
                    view.forward_swiglu_coalesced(
                        &act_refs,
                        bucket,
                        &mut self.batch_gu[..m * lay.gu_rows],
                        &mut self.batch_mid_acts[..m],
                        &mut self.batch_dn[..m * h],
                        &mut out_batch[..n * h],
                        pool,
                    );
                }
                #[cfg(unix)]
                MoeExpertBackend::NvmeSlru { pager, layout } => {
                    let lay = *layout;
                    let p = pager.get_mut().expect("SLRU pager mutex poisoned");
                    let slice = p
                        .access_expert_slice(slot, exp_id)
                        .expect("NVMe SLRU expert pread failed");
                    let view = MmapNfExpertView::from_aligned_slice(slice, lay)
                        .expect("valid SLRU NF expert slice");
                    act_refs.clear();
                    for &(tok_idx, _) in bucket.iter() {
                        act_refs.push(&self.batch_in_acts[tok_idx]);
                    }
                    view.forward_swiglu_coalesced(
                        &act_refs,
                        bucket,
                        &mut self.batch_gu[..m * lay.gu_rows],
                        &mut self.batch_mid_acts[..m],
                        &mut self.batch_dn[..m * h],
                        &mut out_batch[..n * h],
                        pool,
                    );
                }
            }
        }
    }
}

/// Reads router weights (`[num_experts, hidden_size]`) and optional bias (`[num_experts]`) for layer `l`,
/// releasing SafeTensors VM pages when `release_pages` is true.
pub fn read_moe_router_f32(
    store: &SafeTensorsStore,
    layer_idx: usize,
    num_experts: usize,
    hidden_size: usize,
    release_pages: bool,
) -> io::Result<DenseMoeRouter> {
    let pfx = format!("model.layers.{layer_idx}.mlp");
    let w_candidates = [
        format!("{pfx}.gate.weight"),
        format!("{pfx}.router.weight"),
    ];
    let w_name = w_candidates
        .iter()
        .find(|n| store.has_tensor(n))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing MoE router weight at layer {layer_idx}"),
            )
        })?;
    let tv = store
        .tensor(w_name)
        .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
    let raw = tv.to_f32_vec();
    let weights = orient_2d_matrix(&raw, tv.shape, num_experts, hidden_size, w_name)?;
    if release_pages {
        store.release_tensor_pages(w_name);
    }

    let b_candidates = [format!("{pfx}.gate.bias"), format!("{pfx}.router.bias")];
    let bias = if let Some(b_name) = b_candidates.iter().find(|n| store.has_tensor(n)) {
        let bv = store
            .tensor(b_name)
            .map(|t| t.to_f32_vec())
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        if release_pages {
            store.release_tensor_pages(b_name);
        }
        Some(bv)
    } else {
        None
    };

    Ok(DenseMoeRouter::new(num_experts, hidden_size, weights, bias))
}

/// Reads one routed expert `(gate_up_f32: [2 * moe_inter, hidden_size], down_f32: [hidden_size, moe_inter])`
/// from either per-expert 2D tensors (`Qwen2Moe`/`Qwen3Moe`) or fused 3D tensors (`GptOss`/`Qwen3Moe`),
/// reclaiming interior SafeTensors VM pages immediately when `release_pages` is true.
pub fn read_moe_expert_f32(
    store: &SafeTensorsStore,
    layer_idx: usize,
    expert_id: usize,
    num_experts: usize,
    hidden_size: usize,
    moe_inter: usize,
    release_pages: bool,
) -> io::Result<(Vec<f32>, Vec<f32>)> {
    let pfx = format!("model.layers.{layer_idx}.mlp");

    // 1. Per-expert split `gate_proj` + `up_proj` (`Qwen2Moe` / `Qwen3Moe` / `Mixtral`).
    let gate_2d = format!("{pfx}.experts.{expert_id}.gate_proj.weight");
    let up_2d = format!("{pfx}.experts.{expert_id}.up_proj.weight");
    let fused_gu_2d = format!("{pfx}.experts.{expert_id}.gate_up_proj.weight");
    let gate_up = if store.has_tensor(&gate_2d) && store.has_tensor(&up_2d) {
        let g = read_2d_oriented(store, &gate_2d, moe_inter, hidden_size, release_pages)?;
        let u = read_2d_oriented(store, &up_2d, moe_inter, hidden_size, release_pages)?;
        let mut gu = Vec::with_capacity(2 * moe_inter * hidden_size);
        gu.extend_from_slice(&g);
        gu.extend_from_slice(&u);
        gu
    } else if store.has_tensor(&fused_gu_2d) {
        read_2d_oriented(store, &fused_gu_2d, 2 * moe_inter, hidden_size, release_pages)?
    } else {
        let gu_3d_candidates = [
            format!("{pfx}.experts.gate_up_proj"),
            format!("{pfx}.experts.gate_up_proj.weight"),
        ];
        let gu_3d = gu_3d_candidates
            .iter()
            .find(|n| store.has_tensor(n))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing expert {expert_id} gate_up tensor at layer {layer_idx}"),
                )
            })?;
        read_3d_slice_oriented(
            store,
            gu_3d,
            expert_id,
            num_experts,
            2 * moe_inter,
            hidden_size,
            release_pages,
        )?
    };

    // 2. Per-expert 2D or fused 3D `down_proj`.
    let down_2d = format!("{pfx}.experts.{expert_id}.down_proj.weight");
    let down = if store.has_tensor(&down_2d) {
        read_2d_oriented(store, &down_2d, hidden_size, moe_inter, release_pages)?
    } else {
        let dn_3d_candidates = [
            format!("{pfx}.experts.down_proj"),
            format!("{pfx}.experts.down_proj.weight"),
        ];
        let dn_3d = dn_3d_candidates
            .iter()
            .find(|n| store.has_tensor(n))
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing expert {expert_id} down_proj tensor at layer {layer_idx}"),
                )
            })?;
        read_3d_slice_oriented(
            store,
            dn_3d,
            expert_id,
            num_experts,
            hidden_size,
            moe_inter,
            release_pages,
        )?
    };

    Ok((gate_up, down))
}

/// `(inter_dim, gate_up_f32, down_f32, optional_gate_weight)` loaded from SafeTensors.
pub type RawSharedExpertF32 = (usize, Vec<f32>, Vec<f32>, Option<Vec<f32>>);

/// Reads optional shared expert `(gate_up_f32, down_f32, optional_gate_weight)` if present at `layer_idx`.
pub fn read_moe_shared_expert_f32(
    store: &SafeTensorsStore,
    layer_idx: usize,
    hidden_size: usize,
    shared_inter_hint: usize,
    release_pages: bool,
) -> io::Result<Option<RawSharedExpertF32>> {
    let pfx = format!("model.layers.{layer_idx}.mlp");
    let g_name = format!("{pfx}.shared_expert.gate_proj.weight");
    let u_name = format!("{pfx}.shared_expert.up_proj.weight");
    let d_name = format!("{pfx}.shared_expert.down_proj.weight");
    if !store.has_tensor(&g_name) || !store.has_tensor(&u_name) || !store.has_tensor(&d_name) {
        return Ok(None);
    }
    let g_tv = store
        .tensor(&g_name)
        .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
    let inter = if shared_inter_hint > 0 {
        shared_inter_hint
    } else if g_tv.shape.len() == 2 {
        if g_tv.shape[1] == hidden_size {
            g_tv.shape[0]
        } else {
            g_tv.shape[1]
        }
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "unexpected shared_expert.gate_proj rank",
        ));
    };
    let g = read_2d_oriented(store, &g_name, inter, hidden_size, release_pages)?;
    let u = read_2d_oriented(store, &u_name, inter, hidden_size, release_pages)?;
    let mut gu = Vec::with_capacity(2 * inter * hidden_size);
    gu.extend_from_slice(&g);
    gu.extend_from_slice(&u);
    let dn = read_2d_oriented(store, &d_name, hidden_size, inter, release_pages)?;

    let gate_name = format!("{pfx}.shared_expert_gate.weight");
    let gate_w = if store.has_tensor(&gate_name) {
        let gw = store
            .tensor(&gate_name)
            .map(|t| t.to_f32_vec())
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        if release_pages {
            store.release_tensor_pages(&gate_name);
        }
        Some(gw)
    } else {
        None
    };
    Ok(Some((inter, gu, dn, gate_w)))
}

fn read_2d_oriented(
    store: &SafeTensorsStore,
    name: &str,
    expected_rows: usize,
    expected_cols: usize,
    release_pages: bool,
) -> io::Result<Vec<f32>> {
    let tv = store
        .tensor(name)
        .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
    let raw = tv.to_f32_vec();
    let out = orient_2d_matrix(&raw, tv.shape, expected_rows, expected_cols, name)?;
    if release_pages {
        store.release_tensor_pages(name);
    }
    Ok(out)
}

fn read_3d_slice_oriented(
    store: &SafeTensorsStore,
    name: &str,
    expert_id: usize,
    num_experts: usize,
    expected_rows: usize,
    expected_cols: usize,
    release_pages: bool,
) -> io::Result<Vec<f32>> {
    let tv = store
        .tensor(name)
        .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
    if tv.shape.len() != 3 || tv.shape[0] != num_experts {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("tensor '{name}' expected 3D shape [{num_experts}, ..], got {:?}", tv.shape),
        ));
    }
    let slice = tv
        .slice_outer_to_f32(expert_id)
        .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
    let out = orient_2d_matrix(&slice, &tv.shape[1..], expected_rows, expected_cols, name)?;
    if release_pages {
        if expert_id + 1 == num_experts {
            store.release_tensor_pages(name);
        } else {
            store.release_tensor_outer_slice_pages(name, expert_id);
        }
    }
    Ok(out)
}

fn orient_2d_matrix(
    raw: &[f32],
    shape: &[usize],
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
) -> io::Result<Vec<f32>> {
    if shape.len() != 2 || raw.len() != expected_rows * expected_cols {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "tensor '{name}' shape {:?} incompatible with [{expected_rows}, {expected_cols}]",
                shape
            ),
        ));
    }
    if shape[0] == expected_rows && shape[1] == expected_cols {
        return Ok(raw.to_vec());
    }
    if shape[0] == expected_cols && shape[1] == expected_rows {
        let mut tr = vec![0.0f32; expected_rows * expected_cols];
        for c in 0..expected_cols {
            let src_row = &raw[c * expected_rows..(c + 1) * expected_rows];
            for (r, &v) in src_row.iter().enumerate() {
                tr[r * expected_cols + c] = v;
            }
        }
        return Ok(tr);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        format!(
            "tensor '{name}' shape {:?} does not match [{expected_rows}, {expected_cols}]",
            shape
        ),
    ))
}

#[inline]
fn rms_norm_moe(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let mean_sq = dot_f32_moe(x, x) / (x.len() as f32);
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    for ((dst, &xi), &wi) in out.iter_mut().zip(x.iter()).zip(weight.iter()) {
        *dst = xi * inv_rms * wi;
    }
}

#[inline]
fn dot_f32_moe(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is baseline on `aarch64` and `a.len() == b.len()`.
        unsafe { dot_f32_moe_neon(a, b) }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_f32_moe_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;
    let n = a.len();
    let pa = a.as_ptr();
    let pb = b.as_ptr();
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i + 16 <= n {
        // SAFETY: `i + 16 <= n`.
        unsafe {
            acc0 = vfmaq_f32(acc0, vld1q_f32(pa.add(i)), vld1q_f32(pb.add(i)));
            acc1 = vfmaq_f32(acc1, vld1q_f32(pa.add(i + 4)), vld1q_f32(pb.add(i + 4)));
            acc2 = vfmaq_f32(acc2, vld1q_f32(pa.add(i + 8)), vld1q_f32(pb.add(i + 8)));
            acc3 = vfmaq_f32(acc3, vld1q_f32(pa.add(i + 12)), vld1q_f32(pb.add(i + 12)));
        }
        i += 16;
    }
    let mut sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
    while i < n {
        // SAFETY: `i < n`.
        unsafe {
            sum += (*pa.add(i)) * (*pb.add(i));
        }
        i += 1;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_subspace_projector_reduces_drift_error_and_boosts_router_recall() {
        let d_model = 128;
        let num_experts = 16;
        let n_tokens = 96;
        let rank = 8;

        // Ground-truth rank-4 cross-layer drift `h_{l+3} = h_l + B (A h_l) + small_noise`.
        let true_r = 4;
        let a_true: Vec<f32> = (0..true_r * d_model)
            .map(|i| (((i * 19 + 3) as f32) * 0.13).sin() * 0.22)
            .collect();
        let b_true: Vec<f32> = (0..true_r * d_model)
            .map(|i| (((i * 29 + 7) as f32) * 0.17).cos() * 0.35)
            .collect();

        let total = n_tokens + 32;
        let mut src_all = vec![0.0f32; total * d_model];
        let mut dst_all = vec![0.0f32; total * d_model];
        for t in 0..total {
            for j in 0..d_model {
                let v = (((t * 37 + j * 11 + 5) as f32) * 0.21).sin()
                    + 0.4 * (((t * 13 + j * 7) as f32) * 0.09).cos();
                src_all[t * d_model + j] = v;
                dst_all[t * d_model + j] = v + 0.015 * (((t * 53 + j * 3) as f32) * 0.31).sin();
            }
            for r in 0..true_r {
                let lat = dot_f32_moe(
                    &a_true[r * d_model..(r + 1) * d_model],
                    &src_all[t * d_model..(t + 1) * d_model],
                );
                for j in 0..d_model {
                    dst_all[t * d_model + j] += lat * b_true[r * d_model + j];
                }
            }
        }

        let proj = fit_subspace_projector(
            &src_all[..n_tokens * d_model],
            &dst_all[..n_tokens * d_model],
            d_model,
            rank,
            0.01,
        );

        let router_w: Vec<f32> = (0..num_experts * d_model)
            .map(|i| (((i * 41 + 9) as f32) * 0.19).sin() * 0.25)
            .collect();
        let router = DenseMoeRouter::new(num_experts, d_model, router_w, None);

        let mut id_mse = 0.0f64;
        let mut proj_mse = 0.0f64;
        let mut id_hits = 0usize;
        let mut proj_hits = 0usize;
        let mut hat = vec![0.0f32; d_model];
        let mut r_true = [(0usize, 0.0f32); 4];
        let mut r_id = [(0usize, 0.0f32); 4];
        let mut r_pr = [(0usize, 0.0f32); 4];

        for t in n_tokens..total {
            let h_src = &src_all[t * d_model..(t + 1) * d_model];
            let h_dst = &dst_all[t * d_model..(t + 1) * d_model];
            proj.project(h_src, &mut hat);
            for j in 0..d_model {
                let d0 = (h_dst[j] - h_src[j]) as f64;
                let d1 = (h_dst[j] - hat[j]) as f64;
                id_mse += d0 * d0;
                proj_mse += d1 * d1;
            }
            router.route_sorted(h_dst, 4, true, MoeRouterScoring::Softmax, &mut r_true);
            router.route_sorted(h_src, 4, true, MoeRouterScoring::Softmax, &mut r_id);
            router.route_sorted(&hat, 4, true, MoeRouterScoring::Softmax, &mut r_pr);
            for &(eid, _) in &r_true {
                if r_id.iter().any(|&(x, _)| x == eid) {
                    id_hits += 1;
                }
                if r_pr.iter().any(|&(x, _)| x == eid) {
                    proj_hits += 1;
                }
            }
        }

        assert!(
            proj_mse < id_mse * 0.15,
            "held-out subspace MSE {proj_mse:.4} should be < 15% of identity MSE {id_mse:.4}"
        );
        assert!(
            proj_hits > id_hits,
            "projected router recall {proj_hits} should exceed identity recall {id_hits}"
        );
    }

    fn write_safetensors_shard(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (i, (name, shape, data)) in tensors.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let byte_len = data.len() * 4;
            let shape_str = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_str}],\"data_offsets\":[{offset},{}]}}",
                offset + byte_len
            ));
            offset += byte_len;
        }
        header.push('}');
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut f = File::create(path).expect("create safetensors shard");
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        for (_, _, data) in tensors {
            for &v in data {
                f.write_all(&v.to_le_bytes()).unwrap();
            }
        }
        f.flush().unwrap();
    }

    fn cosine_sim(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let xf = x as f64;
            let yf = y as f64;
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-12)
    }

    #[test]
    fn test_streaming_moe_safetensors_compile_slru_vs_mmap_and_coalesced_batch() {
        use crate::real_llm::{RealLlamaEngine, RealWeightMode};

        let tmp_dir = std::env::temp_dir().join(format!(
            "kronos_phase4_moe_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&tmp_dir).unwrap();

        let vocab = 256usize;
        let h_dim = 128usize;
        let num_layers = 6usize;
        let num_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 64usize; // q_dim = 256 != h_dim (128), exercising rectangular GptOss/Qwen3Moe attention
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let num_experts = 8usize;
        let top_k = 2usize;
        let moe_inter = 128usize;
        let shared_inter = 128usize;

        let config_json = format!(
            r#"{{
  "architectures": ["Qwen3MoeForCausalLM"],
  "vocab_size": {vocab},
  "hidden_size": {h_dim},
  "intermediate_size": 0,
  "moe_intermediate_size": {moe_inter},
  "shared_expert_intermediate_size": {shared_inter},
  "num_hidden_layers": {num_layers},
  "num_attention_heads": {num_heads},
  "num_key_value_heads": {num_kv_heads},
  "head_dim": {head_dim},
  "num_experts": {num_experts},
  "num_experts_per_tok": {top_k},
  "norm_topk_prob": true,
  "rms_norm_eps": 1e-5,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false
}}"#
        );
        std::fs::write(tmp_dir.join("config.json"), config_json).unwrap();

        let gen_vec = |len: usize, seed: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    let s1 = (((i.wrapping_mul(31).wrapping_add(seed * 97)) % 997) as f32 / 498.5)
                        - 1.0;
                    let s2 = (((i.wrapping_mul(17).wrapping_add(seed * 53)) % 389) as f32 / 194.5)
                        - 1.0;
                    (0.7 * s1 + 0.3 * s2) * scale
                })
                .collect()
        };
        let norm_vec = |len: usize, seed: usize| -> Vec<f32> {
            (0..len)
                .map(|i| 0.9 + 0.2 * (((i * 13 + seed * 7) % 101) as f32 / 101.0))
                .collect()
        };

        let mut shard1: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut shard2: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        shard1.push((
            "model.embed_tokens.weight".to_string(),
            vec![vocab, h_dim],
            gen_vec(vocab * h_dim, 1, 0.18),
        ));

        for l in 0..num_layers {
            let dst = if l < 3 { &mut shard1 } else { &mut shard2 };
            let pfx = format!("model.layers.{l}");
            let base = 100 + l * 71;
            dst.push((
                format!("{pfx}.input_layernorm.weight"),
                vec![h_dim],
                norm_vec(h_dim, base),
            ));
            dst.push((
                format!("{pfx}.post_attention_layernorm.weight"),
                vec![h_dim],
                norm_vec(h_dim, base + 1),
            ));
            dst.push((
                format!("{pfx}.self_attn.q_proj.weight"),
                vec![q_dim, h_dim],
                gen_vec(q_dim * h_dim, base + 2, 0.14),
            ));
            dst.push((
                format!("{pfx}.self_attn.k_proj.weight"),
                vec![kv_dim, h_dim],
                gen_vec(kv_dim * h_dim, base + 3, 0.14),
            ));
            dst.push((
                format!("{pfx}.self_attn.v_proj.weight"),
                vec![kv_dim, h_dim],
                gen_vec(kv_dim * h_dim, base + 4, 0.14),
            ));
            dst.push((
                format!("{pfx}.self_attn.o_proj.weight"),
                vec![h_dim, q_dim],
                gen_vec(h_dim * q_dim, base + 5, 0.11),
            ));

            if l % 2 == 0 {
                // Even layers: Qwen3Moe per-head QK-norm + per-expert 2D tensors (+ shared expert on layer 0).
                dst.push((
                    format!("{pfx}.self_attn.q_norm.weight"),
                    vec![head_dim],
                    norm_vec(head_dim, base + 6),
                ));
                dst.push((
                    format!("{pfx}.self_attn.k_norm.weight"),
                    vec![head_dim],
                    norm_vec(head_dim, base + 7),
                ));
                dst.push((
                    format!("{pfx}.mlp.gate.weight"),
                    vec![num_experts, h_dim],
                    gen_vec(num_experts * h_dim, base + 8, 0.30),
                ));
                for e in 0..num_experts {
                    let es = base + 10 + e * 3;
                    dst.push((
                        format!("{pfx}.mlp.experts.{e}.gate_proj.weight"),
                        vec![moe_inter, h_dim],
                        gen_vec(moe_inter * h_dim, es, 0.13),
                    ));
                    dst.push((
                        format!("{pfx}.mlp.experts.{e}.up_proj.weight"),
                        vec![moe_inter, h_dim],
                        gen_vec(moe_inter * h_dim, es + 1, 0.13),
                    ));
                    dst.push((
                        format!("{pfx}.mlp.experts.{e}.down_proj.weight"),
                        vec![h_dim, moe_inter],
                        gen_vec(h_dim * moe_inter, es + 2, 0.11),
                    ));
                }
                if l == 0 {
                    dst.push((
                        format!("{pfx}.mlp.shared_expert.gate_proj.weight"),
                        vec![shared_inter, h_dim],
                        gen_vec(shared_inter * h_dim, base + 50, 0.10),
                    ));
                    dst.push((
                        format!("{pfx}.mlp.shared_expert.up_proj.weight"),
                        vec![shared_inter, h_dim],
                        gen_vec(shared_inter * h_dim, base + 51, 0.10),
                    ));
                    dst.push((
                        format!("{pfx}.mlp.shared_expert.down_proj.weight"),
                        vec![h_dim, shared_inter],
                        gen_vec(h_dim * shared_inter, base + 52, 0.09),
                    ));
                    dst.push((
                        format!("{pfx}.mlp.shared_expert_gate.weight"),
                        vec![1, h_dim],
                        gen_vec(h_dim, base + 53, 0.25),
                    ));
                }
            } else {
                // Odd layers: GptOss fused 3D expert tensors + `router.weight`/`router.bias` + `o_proj.bias`.
                dst.push((
                    format!("{pfx}.self_attn.o_proj.bias"),
                    vec![h_dim],
                    gen_vec(h_dim, base + 6, 0.02),
                ));
                dst.push((
                    format!("{pfx}.mlp.router.weight"),
                    vec![num_experts, h_dim],
                    gen_vec(num_experts * h_dim, base + 7, 0.30),
                ));
                dst.push((
                    format!("{pfx}.mlp.router.bias"),
                    vec![num_experts],
                    gen_vec(num_experts, base + 8, 0.05),
                ));
                dst.push((
                    format!("{pfx}.mlp.experts.gate_up_proj"),
                    vec![num_experts, 2 * moe_inter, h_dim],
                    gen_vec(num_experts * 2 * moe_inter * h_dim, base + 9, 0.13),
                ));
                dst.push((
                    format!("{pfx}.mlp.experts.down_proj"),
                    vec![num_experts, h_dim, moe_inter],
                    gen_vec(num_experts * h_dim * moe_inter, base + 10, 0.11),
                ));
            }
        }

        shard2.push((
            "model.norm.weight".to_string(),
            vec![h_dim],
            norm_vec(h_dim, 901),
        ));
        shard2.push((
            "lm_head.weight".to_string(),
            vec![vocab, h_dim],
            gen_vec(vocab * h_dim, 902, 0.16),
        ));

        let s1_name = "model-00001-of-00002.safetensors";
        let s2_name = "model-00002-of-00002.safetensors";
        write_safetensors_shard(&tmp_dir.join(s1_name), &shard1);
        write_safetensors_shard(&tmp_dir.join(s2_name), &shard2);

        let mut wm_entries = Vec::new();
        for (name, _, _) in &shard1 {
            wm_entries.push(format!("\"{name}\": \"{s1_name}\""));
        }
        for (name, _, _) in &shard2 {
            wm_entries.push(format!("\"{name}\": \"{s2_name}\""));
        }
        let index_json = format!(
            "{{\"metadata\":{{\"total_size\":0}},\"weight_map\":{{{}}}}}",
            wm_entries.join(",")
        );
        std::fs::write(tmp_dir.join("model.safetensors.index.json"), index_json).unwrap();

        // 1. Exact FP32 reference over the sharded Sparse-MoE directory.
        let mut ref_eng = RealLlamaEngine::load_fp32_reference(&tmp_dir, 64).unwrap();
        assert!(ref_eng.is_moe());
        let test_seq = [3usize, 17, 42, 88, 129, 204];
        let mut ref_last = Vec::new();
        for &tok in &test_seq {
            ref_last = ref_eng.forward_token(tok).to_vec();
        }

        let calib_seqs: Vec<Vec<usize>> = (0..6)
            .map(|s| (0..24).map(|t| (s * 37 + t * 13 + 5) % vocab).collect())
            .collect();

        // 2. Compile and verify both `nf3nf4g128` (Edge8G) and `e8q2nf4g128` (Mobile5G).
        for (tag, ffn_bits, min_cos) in [("nf3nf4g128", 3usize, 0.96f64), ("e8q2nf4g128", 2usize, 0.90f64)] {
            let core_path = tmp_dir.join(format!("core-{tag}.k2tb"));
            let exp_path = tmp_dir.join(format!("experts-{tag}.k2tm"));
            let mode = RealWeightMode::KronosNfMixedGptq {
                attn_head_bits: 4,
                ffn_bits,
                scale_group_chunks: 2,
            };
            RealLlamaEngine::compile_streaming_moe_to_files(
                &tmp_dir,
                &core_path,
                &exp_path,
                mode,
                &calib_seqs,
                2,
            )
            .unwrap();

            let mut eng_mmap = RealLlamaEngine::load_moe_from_files(
                &tmp_dir,
                &core_path,
                &exp_path,
                64,
                MoeExpertOpenPolicy::ForceZeroCopyMmap,
            )
            .unwrap();
            assert!(eng_mmap.is_moe());
            assert!(eng_mmap.has_nf4_untied_embeddings());
            assert!(eng_mmap.is_all_sdot());

            // Force `DirectNvmeSlruPager` with only 6 DRAM expert slots (< 12 active per token)
            // so physical `F_NOCACHE` preads, SLRU evictions, and 3-layer-ahead async lookahead all fire.
            let mut eng_slru = RealLlamaEngine::load_moe_from_files(
                &tmp_dir,
                &core_path,
                &exp_path,
                64,
                MoeExpertOpenPolicy::ForceNvmeSlru {
                    capacity_experts: 6,
                },
            )
            .unwrap();
            assert!(eng_slru.moe_expert_arena_bytes() < eng_mmap.moe_expert_arena_bytes());

            // Disable Q8 KV cache on one pair first so FP32-KV comparison vs `ref_eng` is pure weight quantization,
            // then re-enable Q8 KV cache to test quantized KV + batch coalescing.
            eng_mmap.enable_q8_kv_cache(false);
            eng_slru.enable_q8_kv_cache(false);

            let mut mmap_last = Vec::new();
            for &tok in &test_seq {
                let l_mmap = eng_mmap.forward_token(tok).to_vec();
                let l_slru = eng_slru.forward_token(tok).to_vec();
                for (a, b) in l_mmap.iter().zip(l_slru.iter()) {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{tag}: NVMe F_NOCACHE SLRU must match zero-copy mmap bit-for-bit ({a} vs {b})"
                    );
                }
                mmap_last = l_mmap;
            }

            let cos = cosine_sim(&ref_last, &mmap_last);
            assert!(
                cos >= min_cos,
                "{tag}: Sparse-MoE logit cosine similarity {cos:.4} below {min_cos}"
            );

            let st = eng_slru.moe_stats().expect("moe_stats present");
            assert!(st.is_nvme_slru);
            assert_eq!(st.slru_capacity_experts, 6);
            assert!(st.nvme_bytes_read > 0, "{tag}: expected physical NVMe pread bytes > 0");
            assert!(
                st.dram_hits + st.async_prefetch_hits > 0,
                "{tag}: expected SLRU / 3-layer async lookahead hits > 0 (got {st:?})"
            );

            // Re-enable quantized KV cache and verify `N`-token `forward_batch_logits`
            // (which executes `SparseMoeRuntime::forward_layer_coalesced_batch`) matches sequential `forward_token`.
            eng_slru.enable_q8_kv_cache(true);
            let mut seq_all = vec![0.0f32; test_seq.len() * vocab];
            for (t, &tok) in test_seq.iter().enumerate() {
                let l = eng_slru.forward_token(tok);
                seq_all[t * vocab..(t + 1) * vocab].copy_from_slice(l);
            }

            eng_slru.reset_context();
            let mut batch_all = vec![0.0f32; test_seq.len() * vocab];
            eng_slru.forward_batch_logits(&test_seq, &mut batch_all);

            for t in 0..test_seq.len() {
                let s_row = &seq_all[t * vocab..(t + 1) * vocab];
                let b_row = &batch_all[t * vocab..(t + 1) * vocab];
                let max_diff = s_row
                    .iter()
                    .zip(b_row.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_diff < 1e-4,
                    "{tag} pos {t}: coalesced batch vs single-token max_diff = {max_diff}"
                );
                let top_s = s_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                let top_b = b_row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                assert_eq!(top_s, top_b, "{tag} pos {t}: top-1 token mismatch");
            }
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);
    }
}
