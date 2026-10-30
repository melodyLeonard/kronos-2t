#![deny(unsafe_op_in_unsafe_fn)]
//! 800B-class sparse-MoE inference with experts streamed from the SSD and a hot-expert DRAM cache.
//!
//! Experts live in a 16 KB-page-aligned `.k2tm` shard (VQ2 or NF3/NF4) and are read either
//! through `mmap` ([`K2tmShardMmap`], [`SlruExpertPager`]) or with `F_NOCACHE` `pread` into an
//! aligned DRAM pool ([`DirectNvmeShard`], [`DirectNvmeSlruPager`]). Attention uses an INT8 latent
//! KV cache ([`MlaLatentKvCache`]). [`Kronos800bMoEEngine`] runs decode and batched verification.

use crate::kernels::hadamard::{fwht64_signed_inplace, ifwht64_signed_inplace};
use crate::storage::mmap_pager::sys::RawMmap;
use crate::storage::prefetch::SubspaceProjector;
use crate::kernels::router::ProductKeyRouter;
use crate::kernels::simd_lut::{
    nf_sdot_pair_stream_range_slices, vq2_lut_stripe_range_slices, Int8ChunkActivation,
    NfSdotMatrix, Vq2ActivationLut, Vq2LutMatrix, SIMD_ROW_STRIPE,
};
use crate::kernels::thread_pool::{default_inference_threads, SpinThreadPool};
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};

/// `F_NOCACHE` file I/O and a 16 KB-aligned DRAM slot pool.
#[cfg(unix)]
pub mod nvme_io {
    use super::K2TM_VM_PAGE_BYTES;
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::ptr;

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    const F_NOCACHE: i32 = 48;

    unsafe extern "C" {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        fn fcntl(fd: i32, cmd: i32, ...) -> i32;
        fn pread(fd: i32, buf: *mut c_void, nbyte: usize, offset: i64) -> isize;
        fn posix_memalign(memptr: *mut *mut c_void, alignment: usize, size: usize) -> i32;
        fn free(ptr: *mut c_void);
    }

    /// Sets `F_NOCACHE` on `file` so its reads and writes skip the page cache.
    pub fn set_file_nocache(file: &File) -> io::Result<()> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            // SAFETY: valid fd; the variadic `fcntl` flag is passed as an int.
            let rc = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1i32) };
            if rc == -1 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let _ = file;
        }
        Ok(())
    }

    /// Clears `F_NOCACHE` on `file`; used before `mmap` so mapped pages stay cached.
    pub fn clear_file_nocache(file: &File) -> io::Result<()> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            // SAFETY: valid fd; the variadic `fcntl` flag is passed as an int.
            let rc = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 0i32) };
            if rc == -1 {
                return Err(io::Error::last_os_error());
            }
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let _ = file;
        }
        Ok(())
    }

    /// Reads exactly `len` bytes at `offset` into `dst`.
    ///
    /// # Safety
    /// `dst` must be valid for `len` writable bytes and disjoint from concurrent readers/writers.
    pub unsafe fn pread_exact_raw(
        file: &File,
        dst: *mut u8,
        len: usize,
        offset: usize,
    ) -> io::Result<()> {
        let fd = file.as_raw_fd();
        let mut done = 0usize;
        while done < len {
            // SAFETY: `dst.add(done)` is valid for `len - done` bytes.
            let n = unsafe {
                pread(
                    fd,
                    dst.add(done).cast::<c_void>(),
                    len - done,
                    (offset + done) as i64,
                )
            };
            if n < 0 {
                let err = io::Error::last_os_error();
                if err.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(err);
            }
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "unexpected EOF in physical NVMe pread",
                ));
            }
            done += n as usize;
        }
        Ok(())
    }

    /// Fixed number of 16 KB-aligned DRAM slots (`posix_memalign`), one expert per slot.
    pub struct PageAlignedDramArena {
        ptr: *mut u8,
        num_slots: usize,
        slot_bytes: usize,
    }

    // SAFETY: the arena owns the allocation; slot access is synchronized by `DirectNvmeSlruPager`.
    unsafe impl Send for PageAlignedDramArena {}
    // SAFETY: shared slot reads and disjoint slot writes are coordinated by `DirectNvmeSlruPager`.
    unsafe impl Sync for PageAlignedDramArena {}

    impl PageAlignedDramArena {
        pub fn new(num_slots: usize, slot_bytes: usize) -> io::Result<Self> {
            assert!(num_slots > 0 && slot_bytes > 0 && slot_bytes.is_multiple_of(K2TM_VM_PAGE_BYTES));
            let total = num_slots
                .checked_mul(slot_bytes)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "arena size overflow"))?;
            let mut raw: *mut c_void = ptr::null_mut();
            // SAFETY: Alignment `16384` is a power of two and multiple of `sizeof(void*)`.
            let rc = unsafe { posix_memalign(&mut raw, K2TM_VM_PAGE_BYTES, total) };
            if rc != 0 || raw.is_null() {
                return Err(io::Error::from_raw_os_error(rc));
            }
            let ptr = raw.cast::<u8>();
            // Touch every page so it is faulted in before timing.
            for off in (0..total).step_by(K2TM_VM_PAGE_BYTES) {
                // SAFETY: `off < total`.
                unsafe { ptr::write_volatile(ptr.add(off), 0) };
            }
            Ok(Self {
                ptr,
                num_slots,
                slot_bytes,
            })
        }

        #[inline]
        pub fn num_slots(&self) -> usize {
            self.num_slots
        }

        #[inline]
        pub fn slot_bytes(&self) -> usize {
            self.slot_bytes
        }

        #[inline]
        pub fn total_bytes(&self) -> usize {
            self.num_slots * self.slot_bytes
        }

        #[inline]
        pub fn slot_slice(&self, slot: usize) -> &[u8] {
            assert!(slot < self.num_slots);
            // SAFETY: `slot * slot_bytes + slot_bytes <= total_bytes` and `16 KB` aligned.
            unsafe { std::slice::from_raw_parts(self.ptr.add(slot * self.slot_bytes), self.slot_bytes) }
        }

        /// Raw pointer to `slot`, for `pread` fills.
        #[inline]
        pub fn slot_mut_ptr(&self, slot: usize) -> *mut u8 {
            assert!(slot < self.num_slots);
            // SAFETY: `slot < num_slots` within allocation bounds.
            unsafe { self.ptr.add(slot * self.slot_bytes) }
        }
    }

    impl Drop for PageAlignedDramArena {
        fn drop(&mut self) {
            if !self.ptr.is_null() {
                // SAFETY: `self.ptr` was allocated by `posix_memalign`.
                unsafe { free(self.ptr.cast::<c_void>()) };
            }
        }
    }
}

/// Apple Silicon VM page size.
pub const K2TM_VM_PAGE_BYTES: usize = 16_384;

/// `.k2tm` shard header magic (v1).
pub const K2TM_MAGIC: &[u8; 12] = b"K2TM800B_V1\0";

#[inline]
const fn align_up(val: usize, align: usize) -> usize {
    let rem = val % align;
    if rem == 0 {
        val
    } else {
        val + (align - rem)
    }
}

/// Geometry and memory settings for an 800B-class sparse-MoE model.
#[derive(Clone, Debug, PartialEq)]
pub struct Kronos800bConfig {
    pub num_layers: usize,
    pub d_model: usize,
    pub d_ff_expert: usize,
    pub router_sub_keys: usize,
    pub num_shared_experts: usize,
    pub top_k: usize,
    pub min_top_k: usize,
    pub confidence_tau: f32,
    pub min_tail_prob: f32,
    pub num_attn_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub vocab_size: usize,
    pub scale_group_chunks: usize,
    pub lookahead_layers: usize,
    pub slru_capacity_experts: usize,
    pub max_seq_len: usize,
}

/// Parameter, disk and RAM breakdown for [`Kronos800bConfig`].
#[derive(Clone, Debug)]
pub struct Kronos800bBudgetReport {
    pub routed_experts_per_layer: usize,
    pub params_per_expert: usize,
    pub bytes_per_expert_aligned: usize,
    pub total_routed_params: u64,
    pub total_shared_expert_params: u64,
    pub total_mla_params: u64,
    pub total_head_params: u64,
    pub total_params: u64,
    pub active_params_top_k: u64,
    pub total_disk_bytes: u64,
    pub pinned_shared_ram_bytes: u64,
    pub mla_kv_cache_ram_bytes: u64,
    pub slru_expert_cache_ram_bytes: u64,
    pub total_resident_ram_bytes: u64,
}

impl Kronos800bConfig {
    /// 812B-parameter VQ2 config: 80 layers, `d_model` 7168, 225 routed + 1 shared expert per layer,
    /// 128-head MLA, 8K context.
    pub fn frontier_800b() -> Self {
        Self {
            num_layers: 80,
            d_model: 7168,
            d_ff_expert: 2048,
            router_sub_keys: 15, // 15^2 = 225 routed experts
            num_shared_experts: 1,
            top_k: 8,
            min_top_k: 4,
            confidence_tau: 0.92,
            min_tail_prob: 0.035,
            num_attn_heads: 128,
            kv_lora_rank: 512,
            q_lora_rank: 1536,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            vocab_size: 128_256,
            scale_group_chunks: 2, // G = 128, 2.125 bpw
            lookahead_layers: 6,
            slru_capacity_experts: 1_328, // ~14.5 GiB
            max_seq_len: 8_192,
        }
    }

    /// [`Self::frontier_800b`] with NF3 experts (3.125 bpw) and a 768-expert cache.
    pub fn frontier_800b_nf3() -> Self {
        let mut c = Self::frontier_800b();
        c.slru_capacity_experts = 768;
        c
    }

    /// Byte layout of one E8Q2/NF3/NF4/NF5 expert in a `.k2tm` shard.
    /// `gu_rows` and `dn_rows` must be multiples of 32.
    pub fn nf_expert_layout(&self, bits: usize) -> NfExpertByteLayout {
        nf_expert_byte_layout(self.d_model, self.d_ff_expert, bits, self.scale_group_chunks)
    }

    /// Parameter, disk and RAM budget with NF3/NF4 experts.
    pub fn nf_budget_report(&self, bits: usize) -> Kronos800bBudgetReport {
        let mut rep = self.budget_report();
        let layout = self.nf_expert_layout(bits);
        let e_per_layer = self.routed_experts_per_layer();
        let bpw_nf = (bits as f64) + 0.25 / (self.scale_group_chunks.max(1) as f64);

        let mla_bytes = ((rep.total_mla_params as f64) * bpw_nf / 8.0).ceil() as u64;
        let shared_exp_bytes =
            (self.num_layers * self.num_shared_experts * layout.aligned_expert_bytes) as u64;
        let head_bytes = ((rep.total_head_params as f64) * 4.25 / 8.0).ceil() as u64;
        let router_proj_bytes =
            (self.num_layers * (self.router_sub_keys * self.d_model + 16 * self.d_model) * 4) as u64;
        let pinned_shared_ram_bytes = mla_bytes + shared_exp_bytes + head_bytes + router_proj_bytes;

        let routed_disk_bytes = (K2TM_VM_PAGE_BYTES as u64)
            + (self.num_layers as u64) * (e_per_layer as u64) * (layout.aligned_expert_bytes as u64);
        let slru_expert_cache_ram_bytes =
            (self.slru_capacity_experts * layout.aligned_expert_bytes) as u64;

        rep.bytes_per_expert_aligned = layout.aligned_expert_bytes;
        rep.pinned_shared_ram_bytes = pinned_shared_ram_bytes;
        rep.total_disk_bytes = routed_disk_bytes + pinned_shared_ram_bytes;
        rep.slru_expert_cache_ram_bytes = slru_expert_cache_ram_bytes;
        rep.total_resident_ram_bytes =
            pinned_shared_ram_bytes + rep.mla_kv_cache_ram_bytes + slru_expert_cache_ram_bytes;
        rep
    }

    /// Small config for unit tests.
    pub fn compact_test(num_layers: usize, d_model: usize, d_ff_expert: usize, sub_keys: usize) -> Self {
        Self {
            num_layers,
            d_model,
            d_ff_expert,
            router_sub_keys: sub_keys,
            num_shared_experts: 1,
            top_k: 4.min(sub_keys * sub_keys),
            min_top_k: 2.min(sub_keys * sub_keys),
            confidence_tau: 0.90,
            min_tail_prob: 0.04,
            num_attn_heads: (d_model / 64).max(2),
            kv_lora_rank: 128.min(d_model),
            q_lora_rank: 128.min(d_model),
            qk_nope_head_dim: 64,
            qk_rope_head_dim: 64,
            v_head_dim: 64,
            vocab_size: 1024,
            scale_group_chunks: 2,
            lookahead_layers: 2.min(num_layers.saturating_sub(1)),
            slru_capacity_experts: (num_layers * sub_keys * sub_keys / 3).max(4),
            max_seq_len: 128,
        }
    }

    #[inline]
    pub fn routed_experts_per_layer(&self) -> usize {
        self.router_sub_keys * self.router_sub_keys
    }

    #[inline]
    pub fn params_per_expert(&self) -> usize {
        3 * self.d_model * self.d_ff_expert
    }

    /// Byte layout of one VQ2 expert (`gate_up` `[2*d_ff, d_model]`, `down` `[d_model, d_ff]`) in a `.k2tm` shard.
    pub fn expert_layout(&self) -> ExpertByteLayout {
        assert!(self.d_model.is_multiple_of(64) && self.d_ff_expert.is_multiple_of(64));
        let sgc = self.scale_group_chunks.max(1);

        let gu_rows = 2 * self.d_ff_expert;
        let gu_stripes = gu_rows.div_ceil(SIMD_ROW_STRIPE);
        let gu_chunks = self.d_model / 64;
        let gu_sg = gu_chunks.div_ceil(sgc);
        let gu_packed_bytes = gu_stripes * gu_chunks * 256;
        let gu_scales_u16 = gu_stripes * gu_sg * SIMD_ROW_STRIPE;
        let gu_scales_bytes = gu_scales_u16 * 2;

        let dn_rows = self.d_model;
        let dn_stripes = dn_rows.div_ceil(SIMD_ROW_STRIPE);
        let dn_chunks = self.d_ff_expert / 64;
        let dn_sg = dn_chunks.div_ceil(sgc);
        let dn_packed_bytes = dn_stripes * dn_chunks * 256;
        let dn_scales_u16 = dn_stripes * dn_sg * SIMD_ROW_STRIPE;
        let dn_scales_bytes = dn_scales_u16 * 2;

        let gu_packed_off = 0usize;
        let gu_scales_off = align_up(gu_packed_off + gu_packed_bytes, 64);
        let dn_packed_off = align_up(gu_scales_off + gu_scales_bytes, 64);
        let dn_scales_off = align_up(dn_packed_off + dn_packed_bytes, 64);
        let raw_end = dn_scales_off + dn_scales_bytes;
        let aligned_expert_bytes = align_up(raw_end, K2TM_VM_PAGE_BYTES);

        ExpertByteLayout {
            d_model: self.d_model,
            d_ff: self.d_ff_expert,
            scale_group_chunks: sgc,
            gu_rows,
            gu_stripes,
            gu_chunks,
            gu_packed_off,
            gu_packed_bytes,
            gu_scales_off,
            gu_scales_u16,
            dn_rows,
            dn_stripes,
            dn_chunks,
            dn_packed_off,
            dn_packed_bytes,
            dn_scales_off,
            dn_scales_u16,
            aligned_expert_bytes,
        }
    }

    /// Parameter, disk and RAM budget with VQ2 experts.
    pub fn budget_report(&self) -> Kronos800bBudgetReport {
        let e_per_layer = self.routed_experts_per_layer();
        let p_exp = self.params_per_expert();
        let layout = self.expert_layout();

        let total_routed_params = (self.num_layers as u64) * (e_per_layer as u64) * (p_exp as u64);
        let total_shared_expert_params =
            (self.num_layers as u64) * (self.num_shared_experts as u64) * (p_exp as u64);

        let mla_layer_params = (self.q_lora_rank * self.d_model)
            + (self.num_attn_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim) * self.q_lora_rank)
            + ((self.kv_lora_rank + self.qk_rope_head_dim) * self.d_model)
            + (self.num_attn_heads * (self.qk_nope_head_dim + self.v_head_dim) * self.kv_lora_rank)
            + (self.d_model * self.num_attn_heads * self.v_head_dim);
        let total_mla_params = (self.num_layers as u64) * (mla_layer_params as u64);
        let total_head_params = (self.vocab_size as u64) * (self.d_model as u64);
        let total_params =
            total_routed_params + total_shared_expert_params + total_mla_params + total_head_params;

        let active_routed = (self.num_layers as u64) * (self.top_k as u64) * (p_exp as u64);
        let active_params_top_k =
            active_routed + total_shared_expert_params + total_mla_params + total_head_params;

        let bpw_vq2 = 2.0 + 0.25 / (self.scale_group_chunks.max(1) as f64);
        let mla_bytes = ((total_mla_params as f64) * bpw_vq2 / 8.0).ceil() as u64;
        let shared_exp_bytes =
            (self.num_layers * self.num_shared_experts * layout.aligned_expert_bytes) as u64;
        // Tied `lm_head` stored as 4-bit SDOT (4.25 bpw).
        let head_bytes = ((total_head_params as f64) * 4.25 / 8.0).ceil() as u64;
        let router_proj_bytes =
            (self.num_layers * (self.router_sub_keys * self.d_model + 16 * self.d_model) * 4) as u64;
        let pinned_shared_ram_bytes = mla_bytes + shared_exp_bytes + head_bytes + router_proj_bytes;

        let routed_disk_bytes =
            (K2TM_VM_PAGE_BYTES as u64) + (self.num_layers as u64) * (e_per_layer as u64) * (layout.aligned_expert_bytes as u64);
        let total_disk_bytes = routed_disk_bytes + pinned_shared_ram_bytes;

        // MLA latent cache: (kv_lora_rank + qk_rope_head_dim) i8 plus one f32 scale per 64 values.
        let mla_chunks = (self.kv_lora_rank + self.qk_rope_head_dim) / 64;
        let bytes_per_layer_tok = mla_chunks * (64 + 4);
        let mla_kv_cache_ram_bytes =
            (self.num_layers * self.max_seq_len * bytes_per_layer_tok) as u64;

        let slru_expert_cache_ram_bytes =
            (self.slru_capacity_experts * layout.aligned_expert_bytes) as u64;
        let total_resident_ram_bytes =
            pinned_shared_ram_bytes + mla_kv_cache_ram_bytes + slru_expert_cache_ram_bytes;

        Kronos800bBudgetReport {
            routed_experts_per_layer: e_per_layer,
            params_per_expert: p_exp,
            bytes_per_expert_aligned: layout.aligned_expert_bytes,
            total_routed_params,
            total_shared_expert_params,
            total_mla_params,
            total_head_params,
            total_params,
            active_params_top_k,
            total_disk_bytes,
            pinned_shared_ram_bytes,
            mla_kv_cache_ram_bytes,
            slru_expert_cache_ram_bytes,
            total_resident_ram_bytes,
        }
    }
}

/// Byte offsets and sizes within one VQ2 expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertByteLayout {
    pub d_model: usize,
    pub d_ff: usize,
    pub scale_group_chunks: usize,
    pub gu_rows: usize,
    pub gu_stripes: usize,
    pub gu_chunks: usize,
    pub gu_packed_off: usize,
    pub gu_packed_bytes: usize,
    pub gu_scales_off: usize,
    pub gu_scales_u16: usize,
    pub dn_rows: usize,
    pub dn_stripes: usize,
    pub dn_chunks: usize,
    pub dn_packed_off: usize,
    pub dn_packed_bytes: usize,
    pub dn_scales_off: usize,
    pub dn_scales_u16: usize,
    pub aligned_expert_bytes: usize,
}

/// Borrowed VQ2 expert inside a mapped or DRAM-resident shard slice.
#[derive(Clone, Copy)]
pub struct MmapVq2ExpertView<'a> {
    pub layout: ExpertByteLayout,
    pub gate_up_packed: &'a [u8],
    pub gate_up_scales: &'a [u16],
    pub down_packed: &'a [u8],
    pub down_scales: &'a [u16],
}

impl<'a> MmapVq2ExpertView<'a> {
    /// Wraps a 16 KB-aligned slice of `layout.aligned_expert_bytes` bytes.
    pub fn from_aligned_slice(slice: &'a [u8], layout: ExpertByteLayout) -> Result<Self, &'static str> {
        if slice.len() < layout.aligned_expert_bytes {
            return Err("expert slice shorter than aligned_expert_bytes");
        }
        let gu_packed = &slice[layout.gu_packed_off..layout.gu_packed_off + layout.gu_packed_bytes];
        let gu_s_bytes =
            &slice[layout.gu_scales_off..layout.gu_scales_off + layout.gu_scales_u16 * 2];
        let dn_packed = &slice[layout.dn_packed_off..layout.dn_packed_off + layout.dn_packed_bytes];
        let dn_s_bytes =
            &slice[layout.dn_scales_off..layout.dn_scales_off + layout.dn_scales_u16 * 2];

        if !(gu_s_bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>())
            || !(dn_s_bytes.as_ptr() as usize).is_multiple_of(std::mem::align_of::<u16>())
        {
            return Err("expert BF16 scale slice is not 2-byte aligned");
        }
        #[cfg(not(target_endian = "little"))]
        {
            return Err("zero-copy .k2tm u16 view requires little-endian target");
        }
        #[cfg(target_endian = "little")]
        {
            // SAFETY: pointers are at least 2-byte aligned and lengths match the `u16` counts.
            let gate_up_scales = unsafe {
                std::slice::from_raw_parts(gu_s_bytes.as_ptr().cast::<u16>(), layout.gu_scales_u16)
            };
            // SAFETY: As above for `dn_s_bytes`.
            let down_scales = unsafe {
                std::slice::from_raw_parts(dn_s_bytes.as_ptr().cast::<u16>(), layout.dn_scales_u16)
            };
            Ok(Self {
                layout,
                gate_up_packed: gu_packed,
                gate_up_scales,
                down_packed: dn_packed,
                down_scales,
            })
        }
    }

    /// SwiGLU forward (`[gate; up] -> SwiGLU -> FWHT-64 -> VQ2 LUT -> down`) on `pool`, reading weights in place.
    pub fn forward_swiglu_zero_copy(
        &self,
        vlut_in: &Vq2ActivationLut,
        gate_up_buf: &mut [f32],
        vlut_mid: &mut Vq2ActivationLut,
        out_buf: &mut [f32],
        pool: Option<&SpinThreadPool>,
    ) {
        let lay = self.layout;
        assert_eq!(vlut_in.cols, lay.d_model);
        assert_eq!(gate_up_buf.len(), lay.gu_rows);
        assert_eq!(vlut_mid.cols, lay.d_ff);
        assert_eq!(out_buf.len(), lay.d_model);

        let in_lut_addr = vlut_in.lut.as_ptr() as usize;
        let in_sc_addr = vlut_in.chunk_scales.as_ptr() as usize;
        let gu_addr = gate_up_buf.as_mut_ptr() as usize;

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.gu_stripes >= 16) {
            let grain = ((lay.gu_stripes / p.num_threads()) & !3).max(4);
            p.parallel_for(lay.gu_stripes, grain, |s0, s1| {
                // SAFETY: Disjoint 16-row stripes `[s0, s1)` write disjoint slices of `gate_up_buf`.
                unsafe {
                    vq2_lut_stripe_range_slices(
                        self.gate_up_packed,
                        self.gate_up_scales,
                        lay.gu_rows,
                        lay.gu_stripes,
                        lay.gu_chunks,
                        lay.scale_group_chunks,
                        in_lut_addr as *const [i8; 16],
                        in_sc_addr as *const f32,
                        s0,
                        s1,
                        gu_addr as *mut f32,
                    );
                }
            });
        } else {
            // SAFETY: Single-threaded full stripe range `0..gu_stripes`.
            unsafe {
                vq2_lut_stripe_range_slices(
                    self.gate_up_packed,
                    self.gate_up_scales,
                    lay.gu_rows,
                    lay.gu_stripes,
                    lay.gu_chunks,
                    lay.scale_group_chunks,
                    in_lut_addr as *const [i8; 16],
                    in_sc_addr as *const f32,
                    0,
                    lay.gu_stripes,
                    gu_addr as *mut f32,
                );
            }
        }

        // SwiGLU, FWHT-64 and LUT build fused per 64-wide chunk.
        let (gate_s, up_s) = gate_up_buf.split_at(lay.d_ff);
        let nc_mid = lay.d_ff / 64;
        let mut sw64 = [0.0f32; 64];
        for c in 0..nc_mid {
            let g64 = &gate_s[c * 64..(c + 1) * 64];
            let u64 = &up_s[c * 64..(c + 1) * 64];
            for ((dst, &g), &u) in sw64.iter_mut().zip(g64).zip(u64) {
                *dst = (g / (1.0 + (-g).exp())) * u;
            }
            // SAFETY: `c < nc_mid` and `vlut_mid` has capacity `nc_mid`.
            unsafe {
                Vq2ActivationLut::populate_chunk_fwht_raw(
                    vlut_mid.lut.as_mut_ptr(),
                    vlut_mid.chunk_scales.as_mut_ptr(),
                    c,
                    &sw64,
                    None,
                );
            }
        }

        let mid_lut_addr = vlut_mid.lut.as_ptr() as usize;
        let mid_sc_addr = vlut_mid.chunk_scales.as_ptr() as usize;
        let out_addr = out_buf.as_mut_ptr() as usize;

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.dn_stripes >= 16) {
            let grain = ((lay.dn_stripes / p.num_threads()) & !3).max(4);
            p.parallel_for(lay.dn_stripes, grain, |s0, s1| {
                // SAFETY: Disjoint 16-row stripes `[s0, s1)` write disjoint slices of `out_buf`.
                unsafe {
                    vq2_lut_stripe_range_slices(
                        self.down_packed,
                        self.down_scales,
                        lay.dn_rows,
                        lay.dn_stripes,
                        lay.dn_chunks,
                        lay.scale_group_chunks,
                        mid_lut_addr as *const [i8; 16],
                        mid_sc_addr as *const f32,
                        s0,
                        s1,
                        out_addr as *mut f32,
                    );
                }
            });
        } else {
            // SAFETY: Single-threaded full stripe range `0..dn_stripes`.
            unsafe {
                vq2_lut_stripe_range_slices(
                    self.down_packed,
                    self.down_scales,
                    lay.dn_rows,
                    lay.dn_stripes,
                    lay.dn_chunks,
                    lay.scale_group_chunks,
                    mid_lut_addr as *const [i8; 16],
                    mid_sc_addr as *const f32,
                    0,
                    lay.dn_stripes,
                    out_addr as *mut f32,
                );
            }
        }
    }

    /// Batched forward for `m` tokens routed to this expert. Each 64-row weight block stays in L1
    /// while all tokens use it, so the expert is read from memory once.
    pub fn forward_swiglu_coalesced_vluts(
        &self,
        in_vluts: &[&Vq2ActivationLut],
        gate_up_batch: &mut [f32],
        mid_vluts: &mut [Vq2ActivationLut],
        out_batch: &mut [f32],
        pool: Option<&SpinThreadPool>,
    ) {
        let m = in_vluts.len();
        assert!(m > 0 && mid_vluts.len() >= m);
        let lay = self.layout;
        assert!(gate_up_batch.len() >= m * lay.gu_rows);
        assert!(out_batch.len() >= m * lay.d_model);

        let gu_addr = gate_up_batch.as_mut_ptr() as usize;
        let run_gu_block = |s0: usize, s1: usize| {
            let mut sb = s0;
            while sb < s1 {
                let se = (sb + 4).min(s1);
                for (t, vlut) in in_vluts.iter().enumerate() {
                    // SAFETY: Each worker owns disjoint rows `[sb*16, se*16)` within every token's `gu_rows` slice.
                    unsafe {
                        vq2_lut_stripe_range_slices(
                            self.gate_up_packed,
                            self.gate_up_scales,
                            lay.gu_rows,
                            lay.gu_stripes,
                            lay.gu_chunks,
                            lay.scale_group_chunks,
                            vlut.lut.as_ptr(),
                            vlut.chunk_scales.as_ptr(),
                            sb,
                            se,
                            (gu_addr as *mut f32).add(t * lay.gu_rows),
                        );
                    }
                }
                sb = se;
            }
        };

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.gu_stripes >= 16) {
            p.parallel_for(lay.gu_stripes, 4, run_gu_block);
        } else {
            run_gu_block(0, lay.gu_stripes);
        }

        let nc_mid = lay.d_ff / 64;
        for (t, vlut_m) in mid_vluts[..m].iter_mut().enumerate() {
            let gu_t = &gate_up_batch[t * lay.gu_rows..(t + 1) * lay.gu_rows];
            let (gate_s, up_s) = gu_t.split_at(lay.d_ff);
            let mut sw64 = [0.0f32; 64];
            for c in 0..nc_mid {
                let g64 = &gate_s[c * 64..(c + 1) * 64];
                let u64 = &up_s[c * 64..(c + 1) * 64];
                for ((dst, &g), &u) in sw64.iter_mut().zip(g64).zip(u64) {
                    *dst = (g / (1.0 + (-g).exp())) * u;
                }
                // SAFETY: `c < nc_mid`.
                unsafe {
                    Vq2ActivationLut::populate_chunk_fwht_raw(
                        vlut_m.lut.as_mut_ptr(),
                        vlut_m.chunk_scales.as_mut_ptr(),
                        c,
                        &sw64,
                        None,
                    );
                }
            }
        }

        let mid_slice = &mid_vluts[..m];
        let out_addr = out_batch.as_mut_ptr() as usize;
        let run_dn_block = |s0: usize, s1: usize| {
            let mut sb = s0;
            while sb < s1 {
                let se = (sb + 4).min(s1);
                for (t, vlut_m) in mid_slice.iter().enumerate() {
                    // SAFETY: Each worker owns disjoint rows `[sb*16, se*16)` within every token's `d_model` slice.
                    unsafe {
                        vq2_lut_stripe_range_slices(
                            self.down_packed,
                            self.down_scales,
                            lay.dn_rows,
                            lay.dn_stripes,
                            lay.dn_chunks,
                            lay.scale_group_chunks,
                            vlut_m.lut.as_ptr(),
                            vlut_m.chunk_scales.as_ptr(),
                            sb,
                            se,
                            (out_addr as *mut f32).add(t * lay.d_model),
                        );
                    }
                }
                sb = se;
            }
        };

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.dn_stripes >= 16) {
            p.parallel_for(lay.dn_stripes, 4, run_dn_block);
        } else {
            run_dn_block(0, lay.dn_stripes);
        }
    }
}

/// Header stored in page 0 of a `.k2tm` shard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K2tmShardHeader {
    pub num_layers: usize,
    pub experts_per_layer: usize,
    pub d_model: usize,
    pub d_ff_expert: usize,
    pub scale_group_chunks: usize,
    pub aligned_expert_bytes: usize,
    pub bits: usize,
}

impl K2tmShardHeader {
    pub fn encode_page(&self) -> [u8; K2TM_VM_PAGE_BYTES] {
        let mut page = [0u8; K2TM_VM_PAGE_BYTES];
        page[0..12].copy_from_slice(K2TM_MAGIC);
        let words = [
            self.num_layers as u64,
            self.experts_per_layer as u64,
            self.d_model as u64,
            self.d_ff_expert as u64,
            self.scale_group_chunks as u64,
            self.aligned_expert_bytes as u64,
            self.bits as u64,
        ];
        for (i, &w) in words.iter().enumerate() {
            page[16 + i * 8..24 + i * 8].copy_from_slice(&w.to_le_bytes());
        }
        page
    }

    pub fn decode_page(page: &[u8]) -> Result<Self, &'static str> {
        if page.len() < K2TM_VM_PAGE_BYTES {
            return Err("k2tm header shorter than 16384 bytes");
        }
        if &page[0..12] != K2TM_MAGIC {
            return Err("bad k2tm header magic");
        }
        let rd = |i: usize| -> usize {
            u64::from_le_bytes(page[16 + i * 8..24 + i * 8].try_into().unwrap()) as usize
        };
        Ok(Self {
            num_layers: rd(0),
            experts_per_layer: rd(1),
            d_model: rd(2),
            d_ff_expert: rd(3),
            scale_group_chunks: rd(4),
            aligned_expert_bytes: rd(5),
            bits: rd(6),
        })
    }

    #[inline]
    pub fn total_file_bytes(&self) -> usize {
        K2TM_VM_PAGE_BYTES
            + self.num_layers * self.experts_per_layer * self.aligned_expert_bytes
    }

    #[inline]
    pub fn expert_byte_offset(&self, layer_idx: usize, expert_id: usize) -> usize {
        assert!(layer_idx < self.num_layers && expert_id < self.experts_per_layer);
        K2TM_VM_PAGE_BYTES
            + (layer_idx * self.experts_per_layer + expert_id) * self.aligned_expert_bytes
    }
}

/// Writes a `.k2tm` shard of `num_shard_layers * experts_per_layer` synthetic VQ2 experts to `path`.
pub fn write_k2tm_shard_file(
    path: &Path,
    cfg: &Kronos800bConfig,
    num_shard_layers: usize,
    experts_per_shard_layer: usize,
    seed: u64,
) -> io::Result<K2tmShardHeader> {
    let layout = cfg.expert_layout();
    let hdr = K2tmShardHeader {
        num_layers: num_shard_layers,
        experts_per_layer: experts_per_shard_layer,
        d_model: cfg.d_model,
        d_ff_expert: cfg.d_ff_expert,
        scale_group_chunks: layout.scale_group_chunks,
        aligned_expert_bytes: layout.aligned_expert_bytes,
        bits: 0,
    };

    let file = File::create(path)?;
    #[cfg(unix)]
    let _ = nvme_io::set_file_nocache(&file);
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, file);
    w.write_all(&hdr.encode_page())?;

    // One buffer of deterministic VQ2 data, refilled per expert.
    let mut expert_buf = vec![0u8; layout.aligned_expert_bytes];
    // BF16 scales in about [0.014, 0.019], varied per group.
    for (idx, chunk) in expert_buf[layout.gu_scales_off..layout.gu_scales_off + layout.gu_scales_u16 * 2]
        .chunks_exact_mut(2)
        .enumerate()
    {
        let bf16: u16 = 0x3C60 + ((idx as u16 * 13) & 0x003F);
        chunk.copy_from_slice(&bf16.to_le_bytes());
    }
    for (idx, chunk) in expert_buf[layout.dn_scales_off..layout.dn_scales_off + layout.dn_scales_u16 * 2]
        .chunks_exact_mut(2)
        .enumerate()
    {
        let bf16: u16 = 0x3C60 + ((idx as u16 * 17) & 0x003F);
        chunk.copy_from_slice(&bf16.to_le_bytes());
    }

    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for l in 0..num_shard_layers {
        for e in 0..experts_per_shard_layer {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(((l * 1024 + e) as u64) | 1);
            // Fill `gu_packed` and `dn_packed` with xorshift output, 8 bytes at a time.
            let mut s = state;
            for word in expert_buf[layout.gu_packed_off..layout.gu_packed_off + layout.gu_packed_bytes]
                .chunks_exact_mut(8)
            {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                word.copy_from_slice(&s.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
            }
            for word in expert_buf[layout.dn_packed_off..layout.dn_packed_off + layout.dn_packed_bytes]
                .chunks_exact_mut(8)
            {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                word.copy_from_slice(&s.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
            }
            w.write_all(&expert_buf)?;
        }
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(hdr)
}

/// Writes compiled [`Vq2LutMatrix`] experts (`gate_up`, `down`) to a `.k2tm` shard, for parity tests.
pub fn write_exact_experts_k2tm(
    path: &Path,
    cfg: &Kronos800bConfig,
    experts: &[(Vq2LutMatrix, Vq2LutMatrix)],
) -> io::Result<K2tmShardHeader> {
    let layout = cfg.expert_layout();
    let hdr = K2tmShardHeader {
        num_layers: 1,
        experts_per_layer: experts.len(),
        d_model: cfg.d_model,
        d_ff_expert: cfg.d_ff_expert,
        scale_group_chunks: layout.scale_group_chunks,
        aligned_expert_bytes: layout.aligned_expert_bytes,
        bits: 0,
    };
    let file = File::create(path)?;
    #[cfg(unix)]
    let _ = nvme_io::set_file_nocache(&file);
    let mut w = BufWriter::new(file);
    w.write_all(&hdr.encode_page())?;
    let mut buf = vec![0u8; layout.aligned_expert_bytes];
    for (gu, dn) in experts {
        buf.fill(0);
        assert_eq!(gu.packed_data.len(), layout.gu_packed_bytes);
        assert_eq!(gu.bf16_chunk_scales.len(), layout.gu_scales_u16);
        assert_eq!(dn.packed_data.len(), layout.dn_packed_bytes);
        assert_eq!(dn.bf16_chunk_scales.len(), layout.dn_scales_u16);
        buf[layout.gu_packed_off..layout.gu_packed_off + layout.gu_packed_bytes]
            .copy_from_slice(&gu.packed_data);
        for (dst, &sc) in buf[layout.gu_scales_off..layout.gu_scales_off + layout.gu_scales_u16 * 2]
            .chunks_exact_mut(2)
            .zip(gu.bf16_chunk_scales.iter())
        {
            dst.copy_from_slice(&sc.to_le_bytes());
        }
        buf[layout.dn_packed_off..layout.dn_packed_off + layout.dn_packed_bytes]
            .copy_from_slice(&dn.packed_data);
        for (dst, &sc) in buf[layout.dn_scales_off..layout.dn_scales_off + layout.dn_scales_u16 * 2]
            .chunks_exact_mut(2)
            .zip(dn.bf16_chunk_scales.iter())
        {
            dst.copy_from_slice(&sc.to_le_bytes());
        }
        w.write_all(&buf)?;
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(hdr)
}

/// Read-only `mmap` of a `.k2tm` shard.
pub struct K2tmShardMmap {
    raw: RawMmap,
    pub header: K2tmShardHeader,
    pub layout: ExpertByteLayout,
}

impl K2tmShardMmap {
    pub fn open(path: &Path, cfg: &Kronos800bConfig) -> io::Result<Self> {
        let file = File::open(path)?;
        #[cfg(unix)]
        let _ = nvme_io::clear_file_nocache(&file);
        let len = file.metadata()?.len() as usize;
        if len < K2TM_VM_PAGE_BYTES || !len.is_multiple_of(K2TM_VM_PAGE_BYTES) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2tm shard length must be a multiple of 16384 bytes",
            ));
        }
        let raw = RawMmap::map_read_only(&file, len)?;
        let header = K2tmShardHeader::decode_page(&raw.as_slice()[0..K2TM_VM_PAGE_BYTES])
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        let layout = cfg.expert_layout();
        if header.aligned_expert_bytes != layout.aligned_expert_bytes || len != header.total_file_bytes() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2tm shard geometry mismatch against Kronos800bConfig",
            ));
        }
        Ok(Self { raw, header, layout })
    }

    #[inline]
    pub fn byte_len(&self) -> usize {
        self.raw.as_slice().len()
    }

    /// Maps `(layer_idx, expert_id)` onto the shard; a shard with fewer layers wraps around.
    #[inline]
    fn physical_coords(&self, layer_idx: usize, expert_id: usize) -> (usize, usize) {
        (
            layer_idx % self.header.num_layers,
            expert_id % self.header.experts_per_layer,
        )
    }

    /// Asks the kernel to read an expert ahead (`POSIX_MADV_WILLNEED`).
    #[inline]
    pub fn advise_willneed_expert(&self, layer_idx: usize, expert_id: usize) {
        let (pl, pe) = self.physical_coords(layer_idx, expert_id);
        let off = self.header.expert_byte_offset(pl, pe);
        self.raw.advise_willneed(off, self.header.aligned_expert_bytes);
    }

    /// Releases an evicted expert's pages (`POSIX_MADV_DONTNEED`).
    #[inline]
    pub fn advise_dontneed_expert(&self, layer_idx: usize, expert_id: usize) {
        let (pl, pe) = self.physical_coords(layer_idx, expert_id);
        let off = self.header.expert_byte_offset(pl, pe);
        self.raw.advise_dontneed(off, self.header.aligned_expert_bytes);
    }

    /// Drops all expert pages (`POSIX_MADV_DONTNEED`) before cold-read benchmarks.
    pub fn evict_all_os_pages(&self) {
        let total_data = self.byte_len().saturating_sub(K2TM_VM_PAGE_BYTES);
        if total_data > 0 {
            self.raw.advise_dontneed(K2TM_VM_PAGE_BYTES, total_data);
        }
    }

    /// [`MmapVq2ExpertView`] into the mapping, without copying.
    #[inline]
    pub fn expert_view(
        &self,
        layer_idx: usize,
        expert_id: usize,
    ) -> Result<MmapVq2ExpertView<'_>, &'static str> {
        let (pl, pe) = self.physical_coords(layer_idx, expert_id);
        let off = self.header.expert_byte_offset(pl, pe);
        let slice = &self.raw.as_slice()[off..off + self.header.aligned_expert_bytes];
        MmapVq2ExpertView::from_aligned_slice(slice, self.layout)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlruTier {
    Probationary,
    Protected,
}

/// Segmented-LRU cache of `mmap`-backed experts (1/3 probationary, 2/3 protected).
///
/// New and prefetched experts enter probationary; a second access promotes them to protected.
/// Protected overflow demotes back to probationary, and only probationary eviction calls
/// `POSIX_MADV_DONTNEED`, so one-off experts do not push out hot ones.
pub struct SlruExpertPager {
    pub mmap: K2tmShardMmap,
    pub total_capacity: usize,
    pub prob_capacity: usize,
    pub prot_capacity: usize,
    prot_len: usize,
    prob_queue: VecDeque<(usize, u64)>,
    prot_queue: VecDeque<(usize, u64)>,
    state: HashMap<usize, (SlruTier, u64)>,
    tick: u64,
    predicted_buf: Vec<(usize, f32)>,
    pub dram_hits: usize,
    pub protected_hits: usize,
    pub nvme_misses: usize,
    pub willneed_hints: usize,
    pub dontneed_evictions: usize,
}

impl SlruExpertPager {
    pub fn new(mmap: K2tmShardMmap, total_capacity_experts: usize) -> Self {
        let cap = total_capacity_experts.max(2);
        // Probationary must hold one token's routed experts across all layers (about 416 at full size),
        // or a token would evict its own experts.
        let prob_capacity = (cap / 3).max(1);
        let prot_capacity = (cap - prob_capacity).max(1);
        Self {
            mmap,
            total_capacity: cap,
            prob_capacity,
            prot_capacity,
            prot_len: 0,
            prob_queue: VecDeque::with_capacity(cap),
            prot_queue: VecDeque::with_capacity(prot_capacity * 2),
            state: HashMap::with_capacity(cap),
            tick: 0,
            predicted_buf: Vec::with_capacity(16),
            dram_hits: 0,
            protected_hits: 0,
            nvme_misses: 0,
            willneed_hints: 0,
            dontneed_evictions: 0,
        }
    }

    pub fn reset_counters(&mut self) {
        self.dram_hits = 0;
        self.protected_hits = 0;
        self.nvme_misses = 0;
        self.willneed_hints = 0;
        self.dontneed_evictions = 0;
    }

    pub fn clear_cache_and_os_pages(&mut self) {
        self.prob_queue.clear();
        self.prot_queue.clear();
        self.state.clear();
        self.prot_len = 0;
        self.tick = 0;
        self.reset_counters();
        self.mmap.evict_all_os_pages();
    }

    #[inline]
    fn pack_key(&self, layer_idx: usize, expert_id: usize) -> usize {
        let (pl, pe) = self.mmap.physical_coords(layer_idx, expert_id);
        pl * self.mmap.header.experts_per_layer + pe
    }

    #[inline]
    fn unpack_key(&self, key: usize) -> (usize, usize) {
        let e = self.mmap.header.experts_per_layer;
        (key / e, key % e)
    }

    pub fn resident_count(&self) -> usize {
        self.state.len()
    }

    pub fn protected_count(&self) -> usize {
        self.prot_len
    }

    fn pop_valid_lru(
        queue: &mut VecDeque<(usize, u64)>,
        state: &HashMap<usize, (SlruTier, u64)>,
        expected_tier: SlruTier,
    ) -> Option<usize> {
        while let Some((k, stamp)) = queue.pop_front() {
            if let Some(&(tier, cur_stamp)) = state.get(&k) {
                if tier == expected_tier && cur_stamp == stamp {
                    return Some(k);
                }
            }
        }
        None
    }

    fn evict_probationary_if_needed(&mut self) {
        if self.state.len() < self.total_capacity {
            return;
        }
        if let Some(victim) =
            Self::pop_valid_lru(&mut self.prob_queue, &self.state, SlruTier::Probationary)
        {
            self.state.remove(&victim);
            let (pl, pe) = self.unpack_key(victim);
            self.mmap.advise_dontneed_expert(pl, pe);
            self.dontneed_evictions += 1;
        }
    }

    fn promote_to_protected(&mut self, key: usize) {
        if self.prot_len >= self.prot_capacity {
            if let Some(demoted) =
                Self::pop_valid_lru(&mut self.prot_queue, &self.state, SlruTier::Protected)
            {
                self.prot_len -= 1;
                self.tick += 1;
                let stamp = self.tick;
                self.state.insert(demoted, (SlruTier::Probationary, stamp));
                self.prob_queue.push_back((demoted, stamp));
            }
        }
        self.prot_len += 1;
        self.tick += 1;
        let stamp = self.tick;
        self.state.insert(key, (SlruTier::Protected, stamp));
        self.prot_queue.push_back((key, stamp));
    }

    /// Starts reading an expert into probationary (`POSIX_MADV_WILLNEED`) if not resident.
    /// Never demotes protected experts.
    pub fn prefetch_expert(&mut self, layer_idx: usize, expert_id: usize) {
        let key = self.pack_key(layer_idx, expert_id);
        if self.state.contains_key(&key) {
            return;
        }
        self.evict_probationary_if_needed();
        self.mmap.advise_willneed_expert(layer_idx, expert_id);
        self.willneed_hints += 1;
        self.tick += 1;
        let stamp = self.tick;
        self.state.insert(key, (SlruTier::Probationary, stamp));
        self.prob_queue.push_back((key, stamp));
    }

    /// Predicts experts `lookahead` layers ahead from `current_hidden`, prunes the tail and
    /// issues `POSIX_MADV_WILLNEED` for them.
    #[allow(clippy::too_many_arguments)]
    pub fn prefetch_lookahead_layer(
        &mut self,
        future_layer_idx: usize,
        projector: &SubspaceProjector,
        future_router: &ProductKeyRouter,
        current_hidden: &[f32],
        scratch_hidden: &mut [f32],
        top_k: usize,
        min_top_k: usize,
        confidence_tau: f32,
        min_tail_prob: f32,
    ) {
        projector.project(current_hidden, scratch_hidden);
        let mean_sq =
            scratch_hidden.iter().map(|&v| v * v).sum::<f32>() / (scratch_hidden.len() as f32);
        let inv_rms = 1.0 / (mean_sq + 1e-5).sqrt();
        for v in scratch_hidden.iter_mut() {
            *v *= inv_rms;
        }
        future_router.route_top_k_into(scratch_hidden, top_k, &mut self.predicted_buf);
        let mut weights = [(0usize, 0.0f32); 16];
        let k_eff = prune_routed_experts(
            &self.predicted_buf,
            min_top_k,
            confidence_tau,
            min_tail_prob,
            &mut weights,
        );
        for &(exp_id, _) in &weights[..k_eff] {
            self.prefetch_expert(future_layer_idx, exp_id);
        }
    }

    #[inline]
    pub fn is_resident(&self, layer_idx: usize, expert_id: usize) -> bool {
        let key = self.pack_key(layer_idx, expert_id);
        self.state.contains_key(&key)
    }

    /// Returns the expert for execution, promoting probationary to protected on a hit.
    pub fn access_expert(
        &mut self,
        layer_idx: usize,
        expert_id: usize,
    ) -> Result<MmapVq2ExpertView<'_>, &'static str> {
        let key = self.pack_key(layer_idx, expert_id);
        match self.state.get(&key).copied() {
            Some((SlruTier::Protected, _)) => {
                self.dram_hits += 1;
                self.protected_hits += 1;
                self.tick += 1;
                let stamp = self.tick;
                self.state.insert(key, (SlruTier::Protected, stamp));
                self.prot_queue.push_back((key, stamp));
            }
            Some((SlruTier::Probationary, _)) => {
                self.dram_hits += 1;
                self.promote_to_protected(key);
            }
            None => {
                self.nvme_misses += 1;
                self.evict_probationary_if_needed();
                self.mmap.advise_willneed_expert(layer_idx, expert_id);
                self.willneed_hints += 1;
                self.tick += 1;
                let stamp = self.tick;
                self.state.insert(key, (SlruTier::Probationary, stamp));
                self.prob_queue.push_back((key, stamp));
            }
        }
        self.mmap.expert_view(layer_idx, expert_id)
    }

    pub fn hit_rate_percent(&self) -> f64 {
        let tot = self.dram_hits + self.nvme_misses;
        if tot == 0 {
            0.0
        } else {
            (self.dram_hits as f64 / tot as f64) * 100.0
        }
    }
}

/// Byte offsets and sizes within one E8Q2/NF3/NF4/NF5 expert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NfExpertByteLayout {
    pub bits: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub scale_group_chunks: usize,
    pub gu_rows: usize,
    pub gu_pairs: usize,
    pub gu_chunks: usize,
    pub gu_codebook_off: usize,
    pub gu_pair_off: usize,
    pub gu_pair_bytes: usize,
    pub dn_rows: usize,
    pub dn_pairs: usize,
    pub dn_chunks: usize,
    pub dn_codebook_off: usize,
    pub dn_pair_off: usize,
    pub dn_pair_bytes: usize,
    pub aligned_expert_bytes: usize,
}

/// Computes the 16 KiB-page-aligned `.k2tm` byte layout for one SwiGLU expert (`bits` in `2..=5`).
pub fn nf_expert_byte_layout(
    d_model: usize,
    d_ff: usize,
    bits: usize,
    scale_group_chunks: usize,
) -> NfExpertByteLayout {
    assert!((2..=5).contains(&bits));
    assert!(d_model.is_multiple_of(64) && d_ff.is_multiple_of(64));
    let gu_rows = 2 * d_ff;
    let dn_rows = d_model;
    assert!(gu_rows.is_multiple_of(32) && dn_rows.is_multiple_of(32));
    let sgc = scale_group_chunks.max(1);
    let chunk_b = NfSdotMatrix::stripe_chunk_bytes(bits);

    let gu_chunks = d_model / 64;
    let gu_sg = gu_chunks.div_ceil(sgc);
    let gu_pairs = gu_rows / 32;
    let gu_pair_bytes = gu_pairs * (gu_sg * 64 + gu_chunks * 2 * chunk_b);

    let dn_chunks = d_ff / 64;
    let dn_sg = dn_chunks.div_ceil(sgc);
    let dn_pairs = dn_rows / 32;
    let dn_pair_bytes = dn_pairs * (dn_sg * 64 + dn_chunks * 2 * chunk_b);

    let gu_codebook_off = 0usize;
    let dn_codebook_off = 16usize;
    let gu_pair_off = 64usize;
    let dn_pair_off = align_up(gu_pair_off + gu_pair_bytes, 64);
    let aligned_expert_bytes = align_up(dn_pair_off + dn_pair_bytes, K2TM_VM_PAGE_BYTES);

    NfExpertByteLayout {
        bits,
        d_model,
        d_ff,
        scale_group_chunks: sgc,
        gu_rows,
        gu_pairs,
        gu_chunks,
        gu_codebook_off,
        gu_pair_off,
        gu_pair_bytes,
        dn_rows,
        dn_pairs,
        dn_chunks,
        dn_codebook_off,
        dn_pair_off,
        dn_pair_bytes,
        aligned_expert_bytes,
    }
}

/// Packs a compiled `(gate_up, down)` pair of [`NfSdotMatrix`] into a 16 KiB-aligned `.k2tm` record buffer.
pub fn pack_nf_expert_record(
    layout: &NfExpertByteLayout,
    gu: &NfSdotMatrix,
    dn: &NfSdotMatrix,
    buf: &mut [u8],
) {
    assert_eq!(buf.len(), layout.aligned_expert_bytes);
    buf.fill(0);
    assert_eq!(gu.bits, layout.bits);
    assert_eq!(dn.bits, layout.bits);
    assert!(gu.tail_stream.is_empty() && dn.tail_stream.is_empty());
    assert_eq!(gu.pair_stream.len(), layout.gu_pair_bytes);
    assert_eq!(dn.pair_stream.len(), layout.dn_pair_bytes);
    for (i, &c) in gu.codebook.iter().enumerate() {
        buf[layout.gu_codebook_off + i] = c as u8;
    }
    for (i, &c) in dn.codebook.iter().enumerate() {
        buf[layout.dn_codebook_off + i] = c as u8;
    }
    buf[layout.gu_pair_off..layout.gu_pair_off + layout.gu_pair_bytes]
        .copy_from_slice(&gu.pair_stream);
    buf[layout.dn_pair_off..layout.dn_pair_off + layout.dn_pair_bytes]
        .copy_from_slice(&dn.pair_stream);
}

/// Borrowed NF3/NF4 expert inside a DRAM slot or mapped shard.
#[derive(Clone, Copy)]
pub struct MmapNfExpertView<'a> {
    pub layout: NfExpertByteLayout,
    pub gate_up_codebook: [i8; 16],
    pub gate_up_pairs: &'a [u8],
    pub down_codebook: [i8; 16],
    pub down_pairs: &'a [u8],
}

impl<'a> MmapNfExpertView<'a> {
    pub fn from_aligned_slice(slice: &'a [u8], layout: NfExpertByteLayout) -> Result<Self, &'static str> {
        if slice.len() < layout.aligned_expert_bytes {
            return Err("NF expert slice shorter than aligned_expert_bytes");
        }
        let gu_cb: [i8; 16] =
            std::array::from_fn(|i| slice[layout.gu_codebook_off + i] as i8);
        let dn_cb: [i8; 16] =
            std::array::from_fn(|i| slice[layout.dn_codebook_off + i] as i8);
        let gate_up_pairs = &slice[layout.gu_pair_off..layout.gu_pair_off + layout.gu_pair_bytes];
        let down_pairs = &slice[layout.dn_pair_off..layout.dn_pair_off + layout.dn_pair_bytes];
        Ok(Self {
            layout,
            gate_up_codebook: gu_cb,
            gate_up_pairs,
            down_codebook: dn_cb,
            down_pairs,
        })
    }

    /// SwiGLU forward (`[gate; up] -> SwiGLU -> FWHT-64 -> INT8 -> down`) on `pool`, reading weights in place.
    pub fn forward_swiglu_zero_copy(
        &self,
        act_in: &Int8ChunkActivation,
        gate_up_buf: &mut [f32],
        act_mid: &mut Int8ChunkActivation,
        out_buf: &mut [f32],
        pool: Option<&SpinThreadPool>,
    ) {
        let lay = self.layout;
        assert_eq!(act_in.cols, lay.d_model);
        assert_eq!(gate_up_buf.len(), lay.gu_rows);
        assert_eq!(act_mid.cols, lay.d_ff);
        assert_eq!(out_buf.len(), lay.d_model);

        let in_q_addr = act_in.q.as_ptr() as usize;
        let in_sc_addr = act_in.scales.as_ptr() as usize;
        let gu_addr = gate_up_buf.as_mut_ptr() as usize;
        let gu_cb = &self.gate_up_codebook;

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.gu_pairs >= 8) {
            let grain = (lay.gu_pairs / (p.num_threads() * 2)).clamp(1, 8);
            p.parallel_for(lay.gu_pairs, grain, |p0, p1| {
                // SAFETY: Disjoint 32-row pairs `[p0, p1)` write disjoint slices of `gate_up_buf`.
                unsafe {
                    nf_sdot_pair_stream_range_slices(
                        self.gate_up_pairs,
                        gu_cb,
                        lay.gu_rows,
                        lay.d_model,
                        lay.bits,
                        lay.scale_group_chunks,
                        in_q_addr as *const i8,
                        in_sc_addr as *const f32,
                        p0,
                        p1,
                        gu_addr as *mut f32,
                    );
                }
            });
        } else {
            // SAFETY: Single-threaded full pair range `0..gu_pairs`.
            unsafe {
                nf_sdot_pair_stream_range_slices(
                    self.gate_up_pairs,
                    gu_cb,
                    lay.gu_rows,
                    lay.d_model,
                    lay.bits,
                    lay.scale_group_chunks,
                    in_q_addr as *const i8,
                    in_sc_addr as *const f32,
                    0,
                    lay.gu_pairs,
                    gu_addr as *mut f32,
                );
            }
        }

        let (gate_s, up_s) = gate_up_buf.split_at(lay.d_ff);
        let nc_mid = lay.d_ff / 64;
        let mut sw64 = [0.0f32; 64];
        for c in 0..nc_mid {
            let g64 = &gate_s[c * 64..(c + 1) * 64];
            let u64 = &up_s[c * 64..(c + 1) * 64];
            for ((dst, &g), &u) in sw64.iter_mut().zip(g64).zip(u64) {
                *dst = (g / (1.0 + (-g).exp())) * u;
            }
            // SAFETY: `c < nc_mid` and `act_mid` has capacity `nc_mid * 64`.
            unsafe {
                Int8ChunkActivation::populate_chunk_fwht_raw(
                    act_mid.q.as_mut_ptr(),
                    act_mid.scales.as_mut_ptr(),
                    act_mid.sums.as_mut_ptr(),
                    c,
                    &sw64,
                    None,
                );
            }
        }

        let mid_q_addr = act_mid.q.as_ptr() as usize;
        let mid_sc_addr = act_mid.scales.as_ptr() as usize;
        let out_addr = out_buf.as_mut_ptr() as usize;
        let dn_cb = &self.down_codebook;

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.dn_pairs >= 8) {
            let grain = (lay.dn_pairs / (p.num_threads() * 2)).clamp(1, 8);
            p.parallel_for(lay.dn_pairs, grain, |p0, p1| {
                // SAFETY: Disjoint 32-row pairs `[p0, p1)` write disjoint slices of `out_buf`.
                unsafe {
                    nf_sdot_pair_stream_range_slices(
                        self.down_pairs,
                        dn_cb,
                        lay.dn_rows,
                        lay.d_ff,
                        lay.bits,
                        lay.scale_group_chunks,
                        mid_q_addr as *const i8,
                        mid_sc_addr as *const f32,
                        p0,
                        p1,
                        out_addr as *mut f32,
                    );
                }
            });
        } else {
            // SAFETY: Single-threaded full pair range `0..dn_pairs`.
            unsafe {
                nf_sdot_pair_stream_range_slices(
                    self.down_pairs,
                    dn_cb,
                    lay.dn_rows,
                    lay.d_ff,
                    lay.bits,
                    lay.scale_group_chunks,
                    mid_q_addr as *const i8,
                    mid_sc_addr as *const f32,
                    0,
                    lay.dn_pairs,
                    out_addr as *mut f32,
                );
            }
        }
    }

    /// Coalesced SwiGLU forward across `m` tokens routed to this expert (`bits` in `2..=5`).
    /// Keeps each 128-row weight tile in L1D across all `m` tokens and accumulates weighted outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_swiglu_coalesced(
        &self,
        in_acts: &[&Int8ChunkActivation],
        routes: &[(usize, f32)],
        gate_up_batch: &mut [f32],
        mid_acts: &mut [Int8ChunkActivation],
        down_batch: &mut [f32],
        out_batch: &mut [f32],
        pool: Option<&SpinThreadPool>,
    ) {
        let m = in_acts.len();
        assert!(m > 0 && routes.len() == m && mid_acts.len() >= m);
        let lay = self.layout;
        assert!(gate_up_batch.len() >= m * lay.gu_rows);
        assert!(down_batch.len() >= m * lay.d_model);

        let gu_addr = gate_up_batch.as_mut_ptr() as usize;
        let gu_cb = &self.gate_up_codebook;
        let run_gu_block = |p0: usize, p1: usize| {
            let mut pb = p0;
            while pb < p1 {
                let pe = (pb + 4).min(p1);
                for (j, act) in in_acts.iter().enumerate() {
                    debug_assert_eq!(act.cols, lay.d_model);
                    // SAFETY: Disjoint 32-row pairs `[pb, pe)` within each token's `gu_rows` slice.
                    unsafe {
                        nf_sdot_pair_stream_range_slices(
                            self.gate_up_pairs,
                            gu_cb,
                            lay.gu_rows,
                            lay.d_model,
                            lay.bits,
                            lay.scale_group_chunks,
                            act.q.as_ptr(),
                            act.scales.as_ptr(),
                            pb,
                            pe,
                            (gu_addr as *mut f32).add(j * lay.gu_rows),
                        );
                    }
                }
                pb = pe;
            }
        };

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.gu_pairs >= 8) {
            let grain = (lay.gu_pairs / (p.num_threads() * 2)).clamp(1, 4);
            p.parallel_for(lay.gu_pairs, grain, run_gu_block);
        } else {
            run_gu_block(0, lay.gu_pairs);
        }

        let nc_mid = lay.d_ff / 64;
        for (j, mid) in mid_acts[..m].iter_mut().enumerate() {
            debug_assert_eq!(mid.cols, lay.d_ff);
            let gu_t = &gate_up_batch[j * lay.gu_rows..(j + 1) * lay.gu_rows];
            let (gate_s, up_s) = gu_t.split_at(lay.d_ff);
            let mut sw64 = [0.0f32; 64];
            for c in 0..nc_mid {
                let g64 = &gate_s[c * 64..(c + 1) * 64];
                let u64 = &up_s[c * 64..(c + 1) * 64];
                for ((dst, &g), &u) in sw64.iter_mut().zip(g64).zip(u64) {
                    *dst = (g / (1.0 + (-g).exp())) * u;
                }
                // SAFETY: `c < nc_mid` and `mid` has capacity `nc_mid * 64`.
                unsafe {
                    Int8ChunkActivation::populate_chunk_fwht_raw(
                        mid.q.as_mut_ptr(),
                        mid.scales.as_mut_ptr(),
                        mid.sums.as_mut_ptr(),
                        c,
                        &sw64,
                        None,
                    );
                }
            }
        }

        let mid_slice = &mid_acts[..m];
        let dn_addr = down_batch.as_mut_ptr() as usize;
        let dn_cb = &self.down_codebook;
        let run_dn_block = |p0: usize, p1: usize| {
            let mut pb = p0;
            while pb < p1 {
                let pe = (pb + 4).min(p1);
                for (j, mid) in mid_slice.iter().enumerate() {
                    // SAFETY: Disjoint 32-row pairs `[pb, pe)` within each token's `d_model` slice.
                    unsafe {
                        nf_sdot_pair_stream_range_slices(
                            self.down_pairs,
                            dn_cb,
                            lay.dn_rows,
                            lay.d_ff,
                            lay.bits,
                            lay.scale_group_chunks,
                            mid.q.as_ptr(),
                            mid.scales.as_ptr(),
                            pb,
                            pe,
                            (dn_addr as *mut f32).add(j * lay.d_model),
                        );
                    }
                }
                pb = pe;
            }
        };

        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && lay.dn_pairs >= 8) {
            let grain = (lay.dn_pairs / (p.num_threads() * 2)).clamp(1, 4);
            p.parallel_for(lay.dn_pairs, grain, run_dn_block);
        } else {
            run_dn_block(0, lay.dn_pairs);
        }

        for (j, &(tok_idx, weight)) in routes.iter().enumerate() {
            let dst = &mut out_batch[tok_idx * lay.d_model..(tok_idx + 1) * lay.d_model];
            let src = &down_batch[j * lay.d_model..(j + 1) * lay.d_model];
            for (d, &s) in dst.iter_mut().zip(src.iter()) {
                *d += weight * s;
            }
        }
    }
}

/// Writes compiled [`NfSdotMatrix`] experts to a `.k2tm` shard with `F_NOCACHE` and `fsync`,
/// so later reads come from the SSD.
pub fn write_exact_nf_experts_k2tm(
    path: &Path,
    cfg: &Kronos800bConfig,
    bits: usize,
    num_layers: usize,
    experts_per_layer: usize,
    experts: &[(NfSdotMatrix, NfSdotMatrix)],
) -> io::Result<K2tmShardHeader> {
    assert_eq!(experts.len(), num_layers * experts_per_layer);
    let layout = cfg.nf_expert_layout(bits);
    let hdr = K2tmShardHeader {
        num_layers,
        experts_per_layer,
        d_model: cfg.d_model,
        d_ff_expert: cfg.d_ff_expert,
        scale_group_chunks: layout.scale_group_chunks,
        aligned_expert_bytes: layout.aligned_expert_bytes,
        bits,
    };
    let file = File::create(path)?;
    #[cfg(unix)]
    let _ = nvme_io::set_file_nocache(&file);
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, file);
    w.write_all(&hdr.encode_page())?;

    let mut buf = vec![0u8; layout.aligned_expert_bytes];
    for (gu, dn) in experts {
        pack_nf_expert_record(&layout, gu, dn, &mut buf);
        w.write_all(&buf)?;
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(hdr)
}

/// Writes a `.k2tm` shard of synthetic E8Q2/NF3/NF4/NF5 experts with `F_NOCACHE` and `fsync`.
pub fn write_nf_k2tm_shard_file(
    path: &Path,
    cfg: &Kronos800bConfig,
    bits: usize,
    num_shard_layers: usize,
    experts_per_shard_layer: usize,
    seed: u64,
) -> io::Result<K2tmShardHeader> {
    let layout = cfg.nf_expert_layout(bits);
    let hdr = K2tmShardHeader {
        num_layers: num_shard_layers,
        experts_per_layer: experts_per_shard_layer,
        d_model: cfg.d_model,
        d_ff_expert: cfg.d_ff_expert,
        scale_group_chunks: layout.scale_group_chunks,
        aligned_expert_bytes: layout.aligned_expert_bytes,
        bits,
    };
    let file = File::create(path)?;
    #[cfg(unix)]
    let _ = nvme_io::set_file_nocache(&file);
    let mut w = BufWriter::with_capacity(4 * 1024 * 1024, file);
    w.write_all(&hdr.encode_page())?;

    let mut expert_buf = vec![0u8; layout.aligned_expert_bytes];
    let default_cb: [i8; 16] = match bits {
        2 => [0; 16],
        3 => [-127, -71, -39, -12, 12, 39, 71, 127, 0, 0, 0, 0, 0, 0, 0, 0],
        5 => [4, 11, 18, 25, 32, 39, 46, 53, 60, 68, 76, 85, 95, 106, 117, 127],
        _ => [-127, -92, -69, -52, -38, -25, -14, -4, 4, 14, 25, 38, 52, 69, 92, 127],
    };
    for (i, &c) in default_cb.iter().enumerate() {
        expert_buf[layout.gu_codebook_off + i] = c as u8;
        expert_buf[layout.dn_codebook_off + i] = c as u8;
    }

    let chunk_b = NfSdotMatrix::stripe_chunk_bytes(bits);
    let sgc = layout.scale_group_chunks;
    let init_pair_scales = |dst: &mut [u8], pairs: usize, chunks: usize| {
        let nsg = chunks.div_ceil(sgc);
        let pair_tot = nsg * 64 + chunks * 2 * chunk_b;
        for p in 0..pairs {
            let mut off = p * pair_tot;
            for sg in 0..nsg {
                for lane in 0..32 {
                    let bf16: u16 = 0x3980 + (((p + sg + lane) as u16 * 7) & 0x003F);
                    dst[off + lane * 2..off + lane * 2 + 2].copy_from_slice(&bf16.to_le_bytes());
                }
                let c_len = (sgc).min(chunks - sg * sgc);
                off += 64 + c_len * 2 * chunk_b;
            }
        }
    };
    init_pair_scales(
        &mut expert_buf[layout.gu_pair_off..layout.gu_pair_off + layout.gu_pair_bytes],
        layout.gu_pairs,
        layout.gu_chunks,
    );
    init_pair_scales(
        &mut expert_buf[layout.dn_pair_off..layout.dn_pair_off + layout.dn_pair_bytes],
        layout.dn_pairs,
        layout.dn_chunks,
    );

    let fill_pair_data = |dst: &mut [u8], pairs: usize, chunks: usize, mut s: u64| -> u64 {
        let nsg = chunks.div_ceil(sgc);
        let pair_tot = nsg * 64 + chunks * 2 * chunk_b;
        for p in 0..pairs {
            let mut off = p * pair_tot;
            for sg in 0..nsg {
                let c_len = sgc.min(chunks - sg * sgc);
                let data_bytes = c_len * 2 * chunk_b;
                for word in dst[off + 64..off + 64 + data_bytes].chunks_exact_mut(8) {
                    s ^= s >> 12;
                    s ^= s << 25;
                    s ^= s >> 27;
                    word.copy_from_slice(&s.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
                }
                off += 64 + data_bytes;
            }
        }
        s
    };

    let mut state = seed ^ 0xA5A5_5A5A_F0F0_0F0F;
    for l in 0..num_shard_layers {
        for e in 0..experts_per_shard_layer {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(((l * 1024 + e) as u64) | 1);
            let s = fill_pair_data(
                &mut expert_buf[layout.gu_pair_off..layout.gu_pair_off + layout.gu_pair_bytes],
                layout.gu_pairs,
                layout.gu_chunks,
                state,
            );
            let _ = fill_pair_data(
                &mut expert_buf[layout.dn_pair_off..layout.dn_pair_off + layout.dn_pair_bytes],
                layout.dn_pairs,
                layout.dn_chunks,
                s,
            );
            w.write_all(&expert_buf)?;
        }
    }
    w.flush()?;
    w.get_ref().sync_all()?;
    Ok(hdr)
}

/// `.k2tm` shard reader using `pread` on four `F_NOCACHE` descriptors, so data comes from the SSD,
/// not the page cache.
#[cfg(unix)]
pub struct DirectNvmeShard {
    fds: [File; 4],
    pub header: K2tmShardHeader,
    pub nvme_bytes_read: AtomicU64,
    pub nvme_io_ns: AtomicU64,
}

#[cfg(unix)]
impl DirectNvmeShard {
    pub fn open(path: &Path, expected_aligned_expert_bytes: usize) -> io::Result<Self> {
        let open_nc = || -> io::Result<File> {
            let f = File::open(path)?;
            nvme_io::set_file_nocache(&f)?;
            Ok(f)
        };
        let fds = [open_nc()?, open_nc()?, open_nc()?, open_nc()?];
        let len = fds[0].metadata()?.len() as usize;
        if len < K2TM_VM_PAGE_BYTES || !len.is_multiple_of(K2TM_VM_PAGE_BYTES) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2tm shard length must be a multiple of 16384 bytes",
            ));
        }
        let mut hdr_arena = nvme_io::PageAlignedDramArena::new(1, K2TM_VM_PAGE_BYTES)?;
        // SAFETY: `hdr_arena` slot 0 is `16384` bytes and `16 KB` page-aligned.
        unsafe {
            nvme_io::pread_exact_raw(&fds[0], hdr_arena.slot_mut_ptr(0), K2TM_VM_PAGE_BYTES, 0)?;
        }
        let _ = &mut hdr_arena;
        let header = K2tmShardHeader::decode_page(hdr_arena.slot_slice(0))
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;
        if header.aligned_expert_bytes != expected_aligned_expert_bytes
            || len != header.total_file_bytes()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2tm shard geometry mismatch against expected aligned_expert_bytes",
            ));
        }
        Ok(Self {
            fds,
            header,
            nvme_bytes_read: AtomicU64::new(0),
            nvme_io_ns: AtomicU64::new(0),
        })
    }

    #[inline]
    pub fn physical_coords(&self, layer_idx: usize, expert_id: usize) -> (usize, usize) {
        (
            layer_idx % self.header.num_layers,
            expert_id % self.header.experts_per_layer,
        )
    }

    pub fn reset_io_stats(&self) {
        self.nvme_bytes_read.store(0, Ordering::Relaxed);
        self.nvme_io_ns.store(0, Ordering::Relaxed);
    }

    pub fn physical_nvme_gbps(&self) -> f64 {
        let b = self.nvme_bytes_read.load(Ordering::Relaxed) as f64;
        let ns = self.nvme_io_ns.load(Ordering::Relaxed) as f64;
        if ns <= 0.0 {
            0.0
        } else {
            b / ns // bytes per ns = GB/s
        }
    }

    /// Reads an expert into `dst_ptr` with `pread`. With `qd4_striped`, splits the read into four
    /// parts on four descriptors to keep several requests in flight.
    ///
    /// # Safety
    /// `dst_ptr` must be `16 KB`-page-aligned and valid for `self.header.aligned_expert_bytes` exclusive writes.
    pub unsafe fn pread_expert_into(
        &self,
        layer_idx: usize,
        expert_id: usize,
        dst_ptr: *mut u8,
        qd4_striped: bool,
    ) -> io::Result<()> {
        let (pl, pe) = self.physical_coords(layer_idx, expert_id);
        let base_off = self.header.expert_byte_offset(pl, pe);
        let total = self.header.aligned_expert_bytes;
        let t0 = std::time::Instant::now();

        let total_pages = total / K2TM_VM_PAGE_BYTES;
        if qd4_striped && total_pages >= 4 {
            let dst_addr = dst_ptr as usize;
            let mut io_res: [io::Result<()>; 4] = [Ok(()), Ok(()), Ok(()), Ok(())];
            std::thread::scope(|s| {
                for (q, res_slot) in io_res.iter_mut().enumerate() {
                    let p0 = (total_pages * q) / 4;
                    let p1 = (total_pages * (q + 1)) / 4;
                    let sub_off = p0 * K2TM_VM_PAGE_BYTES;
                    let sub_len = (p1 - p0) * K2TM_VM_PAGE_BYTES;
                    let fd_ref = &self.fds[q];
                    s.spawn(move || {
                        if sub_len > 0 {
                            // SAFETY: Disjoint `[sub_off, sub_off + sub_len)` slices inside `dst_ptr[0..total]`.
                            *res_slot = unsafe {
                                nvme_io::pread_exact_raw(
                                    fd_ref,
                                    (dst_addr + sub_off) as *mut u8,
                                    sub_len,
                                    base_off + sub_off,
                                )
                            };
                        }
                    });
                }
            });
            for r in io_res {
                r?;
            }
        } else {
            // SAFETY: Caller guarantees `dst_ptr` is valid for `total` writable bytes.
            unsafe { nvme_io::pread_exact_raw(&self.fds[0], dst_ptr, total, base_off)? };
        }

        let ns = t0.elapsed().as_nanos() as u64;
        self.nvme_bytes_read.fetch_add(total as u64, Ordering::Relaxed);
        self.nvme_io_ns.fetch_add(ns, Ordering::Relaxed);
        Ok(())
    }
}

/// SLRU expert cache over a [`DirectNvmeShard`], with slots in a `PageAlignedDramArena`.
/// Hits return the slot directly; misses evict a probationary slot and `pread` into it.
/// An optional background thread runs prefetch reads.
#[cfg(unix)]
pub struct DirectNvmeSlruPager {
    pub shard: Arc<DirectNvmeShard>,
    arena: Arc<nvme_io::PageAlignedDramArena>,
    pub total_capacity: usize,
    pub prob_capacity: usize,
    pub prot_capacity: usize,
    pub qd4_striped: bool,
    prot_len: usize,
    free_slots: Vec<usize>,
    prob_queue: VecDeque<(usize, u64)>,
    prot_queue: VecDeque<(usize, u64)>,
    /// Packed key -> `(tier, stamp, slot)`.
    state: HashMap<usize, (SlruTier, u64, usize)>,
    /// Prefetches in flight: key -> slot.
    inflight: HashMap<usize, usize>,
    /// Finished prefetches not yet accessed.
    unread_prefetched: std::collections::HashSet<usize>,
    prefetch_tx: Option<mpsc::Sender<(usize, usize, usize)>>,
    prefetch_done_rx: mpsc::Receiver<(usize, io::Result<()>)>,
    tick: u64,
    pub dram_hits: usize,
    pub protected_hits: usize,
    pub async_prefetch_hits: usize,
    pub nvme_misses: usize,
    pub evictions: usize,
}

#[cfg(unix)]
impl DirectNvmeSlruPager {
    pub fn new(
        shard: DirectNvmeShard,
        total_capacity_experts: usize,
        qd4_striped: bool,
        enable_async_prefetch: bool,
    ) -> io::Result<Self> {
        let cap = total_capacity_experts.max(2);
        let prob_capacity = (cap / 3).max(1);
        let prot_capacity = (cap - prob_capacity).max(1);
        let arena = Arc::new(nvme_io::PageAlignedDramArena::new(
            cap,
            shard.header.aligned_expert_bytes,
        )?);
        let shard = Arc::new(shard);
        let (done_tx, prefetch_done_rx) = mpsc::channel();
        let prefetch_tx = if enable_async_prefetch {
            let (req_tx, req_rx) = mpsc::channel::<(usize, usize, usize)>();
            let shard_bg = Arc::clone(&shard);
            let arena_bg = Arc::clone(&arena);
            std::thread::spawn(move || {
                while let Ok((key, slot, use_qd4)) = req_rx.recv() {
                    let e = shard_bg.header.experts_per_layer;
                    let (pl, pe) = (key / e, key % e);
                    // SAFETY: `slot` is exclusively reserved in `inflight` until `done_tx.send` completes.
                    let res = unsafe {
                        shard_bg.pread_expert_into(
                            pl,
                            pe,
                            arena_bg.slot_mut_ptr(slot),
                            use_qd4 != 0,
                        )
                    };
                    if done_tx.send((key, res)).is_err() {
                        break;
                    }
                }
            });
            Some(req_tx)
        } else {
            None
        };

        Ok(Self {
            shard,
            arena,
            total_capacity: cap,
            prob_capacity,
            prot_capacity,
            qd4_striped,
            prot_len: 0,
            free_slots: (0..cap).rev().collect(),
            prob_queue: VecDeque::with_capacity(cap),
            prot_queue: VecDeque::with_capacity(prot_capacity * 2),
            state: HashMap::with_capacity(cap),
            inflight: HashMap::new(),
            unread_prefetched: std::collections::HashSet::with_capacity(32),
            prefetch_tx,
            prefetch_done_rx,
            tick: 0,
            dram_hits: 0,
            protected_hits: 0,
            async_prefetch_hits: 0,
            nvme_misses: 0,
            evictions: 0,
        })
    }

    fn drain_completed_prefetches(&mut self) {
        while let Ok((key, res)) = self.prefetch_done_rx.try_recv() {
            if let Some(slot) = self.inflight.remove(&key) {
                if res.is_ok() {
                    self.tick += 1;
                    let stamp = self.tick;
                    self.state.insert(key, (SlruTier::Probationary, stamp, slot));
                    self.prob_queue.push_back((key, stamp));
                    self.unread_prefetched.insert(key);
                } else {
                    self.free_slots.push(slot);
                }
            }
        }
    }

    fn wait_inflight_key(&mut self, wanted_key: usize) -> io::Result<()> {
        while self.inflight.contains_key(&wanted_key) {
            let (key, res) = self
                .prefetch_done_rx
                .recv()
                .map_err(|_| io::Error::other("async NVMe prefetch worker disconnected"))?;
            if let Some(slot) = self.inflight.remove(&key) {
                match res {
                    Ok(()) => {
                        self.tick += 1;
                        let stamp = self.tick;
                        self.state.insert(key, (SlruTier::Probationary, stamp, slot));
                        self.prob_queue.push_back((key, stamp));
                        self.unread_prefetched.insert(key);
                    }
                    Err(e) => {
                        self.free_slots.push(slot);
                        if key == wanted_key {
                            return Err(e);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        while !self.inflight.is_empty() {
            if let Ok((key, _)) = self.prefetch_done_rx.recv() {
                self.inflight.remove(&key);
            } else {
                self.inflight.clear();
                break;
            }
        }
        self.prob_queue.clear();
        self.prot_queue.clear();
        self.state.clear();
        self.unread_prefetched.clear();
        self.free_slots = (0..self.total_capacity).rev().collect();
        self.prot_len = 0;
        self.tick = 0;
        self.dram_hits = 0;
        self.protected_hits = 0;
        self.async_prefetch_hits = 0;
        self.nvme_misses = 0;
        self.evictions = 0;
        self.shard.reset_io_stats();
    }

    #[inline]
    fn pack_key(&self, layer_idx: usize, expert_id: usize) -> usize {
        let (pl, pe) = self.shard.physical_coords(layer_idx, expert_id);
        pl * self.shard.header.experts_per_layer + pe
    }

    fn pop_valid_tier_lru(
        queue: &mut VecDeque<(usize, u64)>,
        state: &HashMap<usize, (SlruTier, u64, usize)>,
        expected: SlruTier,
    ) -> Option<(usize, usize)> {
        while let Some((k, stamp)) = queue.pop_front() {
            if let Some(&(tier, cur_stamp, slot)) = state.get(&k) {
                if tier == expected && cur_stamp == stamp {
                    return Some((k, slot));
                }
            }
        }
        None
    }

    fn acquire_probationary_slot(&mut self) -> Option<usize> {
        if let Some(s) = self.free_slots.pop() {
            return Some(s);
        }
        if let Some((victim, slot)) =
            Self::pop_valid_tier_lru(&mut self.prob_queue, &self.state, SlruTier::Probationary)
        {
            self.state.remove(&victim);
            self.unread_prefetched.remove(&victim);
            self.evictions += 1;
            return Some(slot);
        }
        // Fallback if all slots are Protected + inflight: demote Protected LRU.
        if let Some((victim, slot)) =
            Self::pop_valid_tier_lru(&mut self.prot_queue, &self.state, SlruTier::Protected)
        {
            self.prot_len = self.prot_len.saturating_sub(1);
            self.state.remove(&victim);
            self.evictions += 1;
            return Some(slot);
        }
        None
    }

    fn promote_slot_to_protected(&mut self, key: usize, slot: usize) {
        if self.prot_len >= self.prot_capacity {
            if let Some((demoted_k, demoted_slot)) =
                Self::pop_valid_tier_lru(&mut self.prot_queue, &self.state, SlruTier::Protected)
            {
                self.prot_len -= 1;
                self.tick += 1;
                let stamp = self.tick;
                self.state
                    .insert(demoted_k, (SlruTier::Probationary, stamp, demoted_slot));
                self.prob_queue.push_back((demoted_k, stamp));
            }
        }
        self.prot_len += 1;
        self.tick += 1;
        let stamp = self.tick;
        self.state.insert(key, (SlruTier::Protected, stamp, slot));
        self.prot_queue.push_back((key, stamp));
    }

    /// Queues a background `pread` into a probationary slot.
    /// Does not evict protected experts or unread prefetches.
    pub fn prefetch_expert_async(&mut self, layer_idx: usize, expert_id: usize) {
        self.drain_completed_prefetches();
        let key = self.pack_key(layer_idx, expert_id);
        if self.state.contains_key(&key) || self.inflight.contains_key(&key) {
            return;
        }
        let prob_cap = self.total_capacity.saturating_sub(self.prot_capacity).max(2);
        let max_inflight = (prob_cap / 2).clamp(1, 8);
        if self.inflight.len() >= max_inflight {
            return;
        }
        // Don't take protected slots for a prefetch.
        if self.free_slots.is_empty() && self.state.len() <= self.prot_len {
            return;
        }
        let Some(tx) = self.prefetch_tx.as_ref().cloned() else {
            return;
        };
        if let Some(slot) = self.acquire_probationary_slot() {
            if tx.send((key, slot, usize::from(self.qd4_striped))).is_ok() {
                self.inflight.insert(key, slot);
            } else {
                self.free_slots.push(slot);
            }
        }
    }

    /// Returns the expert's slot, reading it with `pread` on a miss.
    pub fn access_expert_slice(&mut self, layer_idx: usize, expert_id: usize) -> io::Result<&[u8]> {
        self.drain_completed_prefetches();
        let key = self.pack_key(layer_idx, expert_id);
        let was_inflight = self.inflight.contains_key(&key);
        if was_inflight {
            self.wait_inflight_key(key)?;
        }

        let slot = match self.state.get(&key).copied() {
            Some((SlruTier::Protected, _, slot)) => {
                self.dram_hits += 1;
                self.protected_hits += 1;
                self.tick += 1;
                let stamp = self.tick;
                self.state.insert(key, (SlruTier::Protected, stamp, slot));
                self.prot_queue.push_back((key, stamp));
                slot
            }
            Some((SlruTier::Probationary, _, slot)) => {
                self.dram_hits += 1;
                if self.unread_prefetched.remove(&key) || was_inflight {
                    self.async_prefetch_hits += 1;
                }
                self.promote_slot_to_protected(key, slot);
                slot
            }
            None => {
                self.nvme_misses += 1;
                let slot = self
                    .acquire_probationary_slot()
                    .ok_or_else(|| io::Error::other("no SLRU DRAM slot available"))?;
                // SAFETY: `slot` is exclusively owned here and `16 KB` page-aligned.
                unsafe {
                    self.shard.pread_expert_into(
                        layer_idx,
                        expert_id,
                        self.arena.slot_mut_ptr(slot),
                        self.qd4_striped,
                    )?;
                }
                self.tick += 1;
                let stamp = self.tick;
                self.state.insert(key, (SlruTier::Probationary, stamp, slot));
                self.prob_queue.push_back((key, stamp));
                slot
            }
        };
        Ok(self.arena.slot_slice(slot))
    }

    #[inline]
    pub fn arena_bytes(&self) -> usize {
        self.arena.total_bytes()
    }

    #[inline]
    pub fn hit_rate_percent(&self) -> f64 {
        let tot = self.dram_hits + self.nvme_misses;
        if tot == 0 {
            0.0
        } else {
            (self.dram_hits as f64 / tot as f64) * 100.0
        }
    }

    /// Percent of accesses served from DRAM with no SSD read.
    #[inline]
    pub fn zero_io_dram_hit_percent(&self) -> f64 {
        let tot = self.dram_hits + self.nvme_misses;
        if tot == 0 {
            0.0
        } else {
            (self.dram_hits.saturating_sub(self.async_prefetch_hits) as f64 / tot as f64) * 100.0
        }
    }

    /// Percent of accesses served by a background prefetch.
    #[inline]
    pub fn async_prefetch_percent(&self) -> f64 {
        let tot = self.dram_hits + self.nvme_misses;
        if tot == 0 {
            0.0
        } else {
            (self.async_prefetch_hits as f64 / tot as f64) * 100.0
        }
    }

    #[inline]
    pub fn arena_ram_bytes(&self) -> usize {
        self.arena.total_bytes()
    }
}

/// Softmaxes the top-k router scores and keeps leading experts until their cumulative probability
/// reaches `confidence_tau` (at least `min_top_k`, none below `min_tail_prob`); renormalizes the rest.
pub fn prune_routed_experts(
    routed_scores: &[(usize, f32)],
    min_top_k: usize,
    confidence_tau: f32,
    min_tail_prob: f32,
    out_weights: &mut [(usize, f32); 16],
) -> usize {
    let k = routed_scores.len().min(16);
    if k == 0 {
        return 0;
    }
    let max_s = routed_scores[0].1;
    let mut probs = [0.0f32; 16];
    let mut sum_exp = 0.0f32;
    for (i, &(_, s)) in routed_scores[..k].iter().enumerate() {
        let e = (s - max_s).exp();
        probs[i] = e;
        sum_exp += e;
    }
    let inv_sum = 1.0 / sum_exp.max(1e-12);
    for p in &mut probs[..k] {
        *p *= inv_sum;
    }

    let k_min = min_top_k.clamp(1, k);
    let mut cum = 0.0f32;
    let mut k_eff = 0usize;
    for i in 0..k {
        if i >= k_min && (cum >= confidence_tau || probs[i] < min_tail_prob) {
            break;
        }
        out_weights[i] = (routed_scores[i].0, probs[i]);
        cum += probs[i];
        k_eff += 1;
    }

    let renorm = 1.0 / cum.max(1e-12);
    for (_, w) in &mut out_weights[..k_eff] {
        *w *= renorm;
    }
    k_eff
}

/// INT8 MLA KV cache: per layer and token, the `kv_lora_rank` latent and 64-dim RoPE key,
/// Hadamard-rotated, with one f32 scale per 64 values.
pub struct MlaLatentKvCache {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub latent_dim: usize,
    pub rope_dim: usize,
    pub total_chunks: usize,
    pub q8_data: Vec<i8>,
    pub scales: Vec<f32>,
}

impl MlaLatentKvCache {
    pub fn new(num_layers: usize, max_seq_len: usize, latent_dim: usize, rope_dim: usize) -> Self {
        assert!(latent_dim.is_multiple_of(64) && rope_dim == 64);
        let total_chunks = (latent_dim + rope_dim) / 64;
        let total_blocks = num_layers * max_seq_len * total_chunks;
        Self {
            num_layers,
            max_seq_len,
            latent_dim,
            rope_dim,
            total_chunks,
            q8_data: vec![0i8; total_blocks * 64],
            scales: vec![0.0f32; total_blocks],
        }
    }

    pub fn physical_bytes(&self) -> usize {
        self.q8_data.len() + self.scales.len() * std::mem::size_of::<f32>()
    }

    /// Rotates `c_kv` and `k_rope` with FWHT-64, quantizes to INT8 and stores them at `(layer_idx, pos)`.
    pub fn store_step(&mut self, layer_idx: usize, pos: usize, c_kv: &[f32], k_rope: &[f32]) {
        assert!(layer_idx < self.num_layers && pos < self.max_seq_len);
        assert_eq!(c_kv.len(), self.latent_dim);
        assert_eq!(k_rope.len(), 64);

        let lat_chunks = self.latent_dim / 64;
        let base_block = (layer_idx * self.max_seq_len + pos) * self.total_chunks;
        let mut tmp = [0.0f32; 64];

        for c in 0..lat_chunks {
            tmp.copy_from_slice(&c_kv[c * 64..(c + 1) * 64]);
            fwht64_signed_inplace(&mut tmp);
            let blk = base_block + c;
            quantize_block64(&tmp, &mut self.q8_data[blk * 64..(blk + 1) * 64], &mut self.scales[blk]);
        }
        tmp.copy_from_slice(k_rope);
        fwht64_signed_inplace(&mut tmp);
        let rblk = base_block + lat_chunks;
        quantize_block64(&tmp, &mut self.q8_data[rblk * 64..(rblk + 1) * 64], &mut self.scales[rblk]);
    }

    /// Absorbed MLA attention for one head over positions `0..=pos`; writes the result to `out_latent`.
    #[allow(clippy::too_many_arguments)]
    pub fn attend_absorbed_head(
        &self,
        layer_idx: usize,
        pos: usize,
        q_latent: &[f32],
        q_rope: &[f32],
        attn_scale: f32,
        scores_scratch: &mut [f32],
        out_latent: &mut [f32],
    ) {
        let seq_len = pos + 1;
        assert_eq!(q_latent.len(), self.latent_dim);
        assert_eq!(q_rope.len(), 64);
        assert!(scores_scratch.len() >= seq_len && out_latent.len() == self.latent_dim);

        let lat_chunks = self.latent_dim / 64;
        let mut q_i8_stack = [0i8; 16 * 64];
        let mut q_sc_stack = [0.0f32; 16];
        let mut q_i8_heap;
        let mut q_sc_heap;
        let (q_i8, q_sc): (&mut [i8], &mut [f32]) = if self.total_chunks <= 16 {
            (
                &mut q_i8_stack[..self.total_chunks * 64],
                &mut q_sc_stack[..self.total_chunks],
            )
        } else {
            q_i8_heap = vec![0i8; self.total_chunks * 64];
            q_sc_heap = vec![0.0f32; self.total_chunks];
            (&mut q_i8_heap, &mut q_sc_heap)
        };
        let mut tmp = [0.0f32; 64];
        for c in 0..lat_chunks {
            tmp.copy_from_slice(&q_latent[c * 64..(c + 1) * 64]);
            fwht64_signed_inplace(&mut tmp);
            quantize_block64(&tmp, &mut q_i8[c * 64..(c + 1) * 64], &mut q_sc[c]);
        }
        tmp.copy_from_slice(q_rope);
        fwht64_signed_inplace(&mut tmp);
        quantize_block64(
            &tmp,
            &mut q_i8[lat_chunks * 64..(lat_chunks + 1) * 64],
            &mut q_sc[lat_chunks],
        );

        let layer_base = layer_idx * self.max_seq_len * self.total_chunks;
        let sc = &mut scores_scratch[..seq_len];
        let mut max_s = f32::NEG_INFINITY;
        for (t, dst) in sc.iter_mut().enumerate() {
            let step_blk = layer_base + t * self.total_chunks;
            let mut sum = 0.0f32;
            for c in 0..self.total_chunks {
                let blk = step_blk + c;
                let dot = dot_i8_64(
                    &q_i8[c * 64..(c + 1) * 64],
                    &self.q8_data[blk * 64..(blk + 1) * 64],
                );
                sum += (dot as f32) * (q_sc[c] * self.scales[blk]);
            }
            let s = sum * attn_scale;
            *dst = s;
            if s > max_s {
                max_s = s;
            }
        }

        let mut sum_exp = 0.0f32;
        for s in sc.iter_mut() {
            *s = (*s - max_s).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp.max(1e-12);

        out_latent.fill(0.0);
        for (t, &e) in sc.iter().enumerate() {
            let p = e * inv_sum;
            let step_blk = layer_base + t * self.total_chunks;
            for c in 0..lat_chunks {
                let blk = step_blk + c;
                let w = p * self.scales[blk];
                let kq = &self.q8_data[blk * 64..(blk + 1) * 64];
                let dst = &mut out_latent[c * 64..(c + 1) * 64];
                for (d, &v) in dst.iter_mut().zip(kq.iter()) {
                    *d += w * (v as f32);
                }
            }
        }

        // Undo the FWHT-64 rotation on each 64-float chunk.
        for c in 0..lat_chunks {
            ifwht64_signed_inplace(&mut out_latent[c * 64..(c + 1) * 64]);
        }
    }
}

#[inline]
fn quantize_block64(rot: &[f32; 64], dst_q8: &mut [i8], dst_scale: &mut f32) {
    let max_abs = rot.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    if max_abs <= f32::EPSILON {
        dst_q8.fill(0);
        *dst_scale = 0.0;
        return;
    }
    let inv = 127.0 / max_abs;
    for (d, &v) in dst_q8.iter_mut().zip(rot.iter()) {
        *d = (v * inv).round_ties_even() as i32 as i8;
    }
    *dst_scale = max_abs / 127.0;
}

#[inline]
fn dot_i8_64(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), 64);
    debug_assert_eq!(b.len(), 64);
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: `sdot` is available on all Apple Silicon; both slices hold 64 `i8`s.
        unsafe {
            use std::arch::aarch64::*;
            let ap = a.as_ptr();
            let bp = b.as_ptr();
            let mut acc = vdupq_n_s32(0);
            for k in 0..4 {
                let va = vld1q_s8(ap.add(k * 16));
                let vb = vld1q_s8(bp.add(k * 16));
                core::arch::asm!(
                    ".arch_extension dotprod\nsdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
                    acc = inout(vreg) acc,
                    a = in(vreg) va,
                    b = in(vreg) vb,
                    options(pure, nomem, nostack, preserves_flags)
                );
            }
            vaddvq_s32(acc)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x as i32) * (y as i32)).sum()
    }
}

/// Deterministic synthetic [`Vq2LutMatrix`] for MLA and LM-head benchmarks.
pub fn synthesize_vq2_matrix(
    rows: usize,
    cols: usize,
    scale_group_chunks: usize,
    seed: u64,
) -> Vq2LutMatrix {
    assert!(rows > 0 && cols > 0 && cols.is_multiple_of(64));
    let sgc = scale_group_chunks.max(1);
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / 64;
    let num_scale_groups = num_chunks.div_ceil(sgc);
    let mut packed_data = vec![0u8; num_stripes * num_chunks * 256];
    let mut bf16_chunk_scales = vec![0u16; num_stripes * num_scale_groups * SIMD_ROW_STRIPE];

    for (idx, sc) in bf16_chunk_scales.iter_mut().enumerate() {
        *sc = 0x3C40 + ((idx as u16 * 19) & 0x003F);
    }
    let mut s = seed ^ 0xA5A5_5A5A_F0F0_0F0F;
    for word in packed_data.chunks_exact_mut(8) {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        word.copy_from_slice(&s.wrapping_mul(0x2545_F491_4F6C_DD1D).to_le_bytes());
    }
    Vq2LutMatrix {
        rows,
        cols,
        scale_group_chunks: sgc,
        num_stripes,
        num_chunks,
        num_scale_groups,
        packed_data,
        bf16_chunk_scales,
    }
}

/// Runs a [`Vq2LutMatrix`] on `m = in_vluts.len()` tokens with stripes in the outer loop,
/// so the weights are read once.
pub fn vq2_matrix_exec_l1d_coalesced(
    mat: &Vq2LutMatrix,
    in_vluts: &[&Vq2ActivationLut],
    out_batch: &mut [f32],
    pool: Option<&SpinThreadPool>,
) {
    let m = in_vluts.len();
    assert!(m > 0 && out_batch.len() >= m * mat.rows);
    let out_addr = out_batch.as_mut_ptr() as usize;
    let tile_stripes = if mat.num_chunks <= 224 { 4 } else { 2 };

    if m == 1 {
        let vlut = in_vluts[0];
        assert_eq!(vlut.cols, mat.cols);
        let run_one = |s0: usize, s1: usize| {
            // SAFETY: Disjoint stripe ranges `[s0, s1)` write disjoint slices of `out_batch[..mat.rows]`.
            unsafe {
                vq2_lut_stripe_range_slices(
                    &mat.packed_data,
                    &mat.bf16_chunk_scales,
                    mat.rows,
                    mat.num_stripes,
                    mat.num_chunks,
                    mat.scale_group_chunks,
                    vlut.lut.as_ptr(),
                    vlut.chunk_scales.as_ptr(),
                    s0,
                    s1,
                    out_addr as *mut f32,
                );
            }
        };
        if let Some(p) = pool.filter(|p| p.num_threads() > 1 && mat.num_stripes >= 16) {
            p.parallel_for(mat.num_stripes, tile_stripes, run_one);
        } else {
            run_one(0, mat.num_stripes);
        }
        return;
    }

    let run_block = |s0: usize, s1: usize| {
        let mut sb = s0;
        while sb < s1 {
            let se = (sb + tile_stripes).min(s1);
            for (t, vlut) in in_vluts.iter().enumerate() {
                // SAFETY: Disjoint stripe sub-ranges `[sb, se)` across workers and tokens.
                unsafe {
                    vq2_lut_stripe_range_slices(
                        &mat.packed_data,
                        &mat.bf16_chunk_scales,
                        mat.rows,
                        mat.num_stripes,
                        mat.num_chunks,
                        mat.scale_group_chunks,
                        vlut.lut.as_ptr(),
                        vlut.chunk_scales.as_ptr(),
                        sb,
                        se,
                        (out_addr as *mut f32).add(t * mat.rows),
                    );
                }
            }
            sb = se;
        }
    };
    if let Some(p) = pool.filter(|p| p.num_threads() > 1 && mat.num_stripes >= 16) {
        p.parallel_for(mat.num_stripes, tile_stripes, run_block);
    } else {
        run_block(0, mat.num_stripes);
    }
}

/// Synthetic VQ2 MLA projection weights for one layer.
pub struct PinnedMlaLayerVq2 {
    pub q_a_proj: Vq2LutMatrix,
    pub q_b_proj: Vq2LutMatrix,
    pub kv_a_proj: Vq2LutMatrix,
    pub kv_b_proj: Vq2LutMatrix,
    pub o_proj: Vq2LutMatrix,
}

impl PinnedMlaLayerVq2 {
    pub fn synthesize(cfg: &Kronos800bConfig, seed: u64) -> Self {
        let sgc = cfg.scale_group_chunks.max(1);
        let q_up_rows = cfg.num_attn_heads * (cfg.qk_nope_head_dim + cfg.qk_rope_head_dim);
        let kv_down_rows = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        let kv_up_rows = cfg.num_attn_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim);
        let v_concat = cfg.num_attn_heads * cfg.v_head_dim;
        Self {
            q_a_proj: synthesize_vq2_matrix(cfg.q_lora_rank, cfg.d_model, sgc, seed ^ 0x11),
            q_b_proj: synthesize_vq2_matrix(q_up_rows, cfg.q_lora_rank, sgc, seed ^ 0x22),
            kv_a_proj: synthesize_vq2_matrix(kv_down_rows, cfg.d_model, sgc, seed ^ 0x33),
            kv_b_proj: synthesize_vq2_matrix(kv_up_rows, cfg.kv_lora_rank, sgc, seed ^ 0x44),
            o_proj: synthesize_vq2_matrix(cfg.d_model, v_concat, sgc, seed ^ 0x55),
        }
    }

    pub fn params_count(&self) -> u64 {
        (self.q_a_proj.rows * self.q_a_proj.cols
            + self.q_b_proj.rows * self.q_b_proj.cols
            + self.kv_a_proj.rows * self.kv_a_proj.cols
            + self.kv_b_proj.rows * self.kv_b_proj.cols
            + self.o_proj.rows * self.o_proj.cols) as u64
    }

    pub fn physical_bytes(&self) -> usize {
        let b = |m: &Vq2LutMatrix| m.packed_data.len() + m.bf16_chunk_scales.len() * 2;
        b(&self.q_a_proj) + b(&self.q_b_proj) + b(&self.kv_a_proj) + b(&self.kv_b_proj) + b(&self.o_proj)
    }
}

struct PinnedFullStackState {
    mla_layers: Vec<PinnedMlaLayerVq2>,
    lm_head: Option<Vq2LutMatrix>,
    q_lora_buf: Vec<f32>,
    q_full_buf: Vec<f32>,
    kv_a_buf: Vec<f32>,
    kv_b_buf: Vec<f32>,
    mla_out_buf: Vec<f32>,
    logits_buf: Vec<f32>,
    vluts_q_lora: Vec<Vq2ActivationLut>,
    vluts_kv_lora: Vec<Vq2ActivationLut>,
    vluts_v_concat: Vec<Vq2ActivationLut>,
}

/// Stats from one token or one verification batch.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoEStepStats {
    pub tokens: usize,
    pub total_expert_routes: usize,
    pub unique_expert_loads: usize,
    pub pruned_tail_experts: usize,
    pub active_params_per_token: u64,
    pub weight_bytes_streamed: u64,
    pub elapsed_secs: f64,
}

/// 800B sparse-MoE runtime: MLA, shared and routed experts, LM head, [`SlruExpertPager`] with
/// lookahead prefetch, tail pruning and batch expert coalescing.
pub struct Kronos800bMoEEngine {
    pub cfg: Kronos800bConfig,
    pub pager: SlruExpertPager,
    pub routers: Vec<ProductKeyRouter>,
    pub projectors: Vec<SubspaceProjector>,
    pub mla_kv: MlaLatentKvCache,
    pool: SpinThreadPool,
    full_stack: Option<PinnedFullStackState>,
    vlut_in: Vq2ActivationLut,
    vlut_mid: Vq2ActivationLut,
    gate_up_buf: Vec<f32>,
    expert_out: Vec<f32>,
    lookahead_scratch: Vec<f32>,
    routed_buf: Vec<(usize, f32)>,
    batch_vluts_in: Vec<Vq2ActivationLut>,
    batch_vluts_mid: Vec<Vq2ActivationLut>,
    batch_gu: Vec<f32>,
    batch_out: Vec<f32>,
    pos: usize,
}

impl Kronos800bMoEEngine {
    pub fn new(cfg: Kronos800bConfig, mmap: K2tmShardMmap, num_threads: usize) -> Self {
        let threads = if num_threads == 0 {
            default_inference_threads()
        } else {
            num_threads
        };
        let routers: Vec<ProductKeyRouter> = (0..cfg.num_layers)
            .map(|l| ProductKeyRouter::new(cfg.d_model, cfg.router_sub_keys, 0x800B_0000 + l as u64))
            .collect();
        let projectors: Vec<SubspaceProjector> = (0..cfg.num_layers)
            .map(|l| SubspaceProjector::new(cfg.d_model, 8, 0x517C_0000 + l as u64))
            .collect();
        let mla_kv = MlaLatentKvCache::new(
            cfg.num_layers,
            cfg.max_seq_len,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
        );
        let pager = SlruExpertPager::new(mmap, cfg.slru_capacity_experts);
        let d_model = cfg.d_model;
        let d_ff = cfg.d_ff_expert;

        Self {
            cfg,
            pager,
            routers,
            projectors,
            mla_kv,
            pool: SpinThreadPool::new(threads),
            full_stack: None,
            vlut_in: Vq2ActivationLut::new_zeroed(d_model),
            vlut_mid: Vq2ActivationLut::new_zeroed(d_ff),
            gate_up_buf: vec![0.0f32; 2 * d_ff],
            expert_out: vec![0.0f32; d_model],
            lookahead_scratch: vec![0.0f32; d_model],
            routed_buf: Vec::with_capacity(16),
            batch_vluts_in: (0..8).map(|_| Vq2ActivationLut::new_zeroed(d_model)).collect(),
            batch_vluts_mid: (0..8).map(|_| Vq2ActivationLut::new_zeroed(d_ff)).collect(),
            batch_gu: vec![0.0f32; 8 * 2 * d_ff],
            batch_out: vec![0.0f32; 8 * d_model],
            pos: 0,
        }
    }

    #[inline]
    pub fn num_threads(&self) -> usize {
        self.pool.num_threads()
    }

    /// Allocates synthetic VQ2 MLA weights (`num_physical_mla_layers` copies, reused across layers)
    /// and optionally the LM head, so decode runs every active tensor.
    ///
    /// Use enough copies (e.g. 8, about 380 MiB) that they don't fit in L2.
    pub fn enable_pinned_mla_and_head(
        &mut self,
        num_physical_mla_layers: usize,
        include_lm_head: bool,
    ) {
        let n_phys = num_physical_mla_layers.max(1);
        let mla_layers: Vec<PinnedMlaLayerVq2> = (0..n_phys)
            .map(|i| PinnedMlaLayerVq2::synthesize(&self.cfg, 0x4D4C_4100 + i as u64))
            .collect();
        let lm_head = if include_lm_head {
            Some(synthesize_vq2_matrix(
                self.cfg.vocab_size,
                self.cfg.d_model,
                1, // G = 64, 2.25 bpw
                0x4845_4144_800B,
            ))
        } else {
            None
        };
        let q_up_rows =
            self.cfg.num_attn_heads * (self.cfg.qk_nope_head_dim + self.cfg.qk_rope_head_dim);
        let kv_down_rows = self.cfg.kv_lora_rank + self.cfg.qk_rope_head_dim;
        let kv_up_rows = self.cfg.num_attn_heads * (self.cfg.qk_nope_head_dim + self.cfg.v_head_dim);
        let v_concat = self.cfg.num_attn_heads * self.cfg.v_head_dim;

        self.full_stack = Some(PinnedFullStackState {
            mla_layers,
            lm_head,
            q_lora_buf: vec![0.0f32; 8 * self.cfg.q_lora_rank],
            q_full_buf: vec![0.0f32; 8 * q_up_rows],
            kv_a_buf: vec![0.0f32; 8 * kv_down_rows],
            kv_b_buf: vec![0.0f32; 8 * kv_up_rows],
            mla_out_buf: vec![0.0f32; 8 * self.cfg.d_model],
            logits_buf: vec![0.0f32; if include_lm_head { 8 * self.cfg.vocab_size } else { 0 }],
            vluts_q_lora: (0..8)
                .map(|_| Vq2ActivationLut::new_zeroed(self.cfg.q_lora_rank))
                .collect(),
            vluts_kv_lora: (0..8)
                .map(|_| Vq2ActivationLut::new_zeroed(self.cfg.kv_lora_rank))
                .collect(),
            vluts_v_concat: (0..8)
                .map(|_| Vq2ActivationLut::new_zeroed(v_concat))
                .collect(),
        });
    }

    pub fn reset_context(&mut self) {
        self.pos = 0;
    }

    fn populate_vlut_rmsnorm(
        pool: &SpinThreadPool,
        x: &[f32],
        vlut: &mut Vq2ActivationLut,
    ) {
        let cols = vlut.cols;
        debug_assert_eq!(x.len(), cols);
        let inv_rms =
            1.0 / (x.iter().map(|&v| v * v).sum::<f32>() / (cols as f32) + 1e-5).sqrt();
        let nc = cols / 64;
        let x_addr = x.as_ptr() as usize;
        let lut_addr = vlut.lut.as_mut_ptr() as usize;
        let sc_addr = vlut.chunk_scales.as_mut_ptr() as usize;
        let run_chunks = |c0: usize, c1: usize| {
            let mut scaled = [0.0f32; 64];
            for c in c0..c1 {
                // SAFETY: `c < nc`, `x` has `nc * 64` floats, and each chunk index `c` is disjoint.
                unsafe {
                    let src = std::slice::from_raw_parts((x_addr as *const f32).add(c * 64), 64);
                    for (d, &v) in scaled.iter_mut().zip(src.iter()) {
                        *d = v * inv_rms;
                    }
                    Vq2ActivationLut::populate_chunk_fwht_raw(
                        lut_addr as *mut [i8; 16],
                        sc_addr as *mut f32,
                        c,
                        &scaled,
                        None,
                    );
                }
            }
        };
        if pool.num_threads() > 1 && nc >= 64 {
            pool.parallel_for(nc, 8, run_chunks);
        } else {
            run_chunks(0, nc);
        }
    }

    fn populate_batch_vluts_rmsnorm(
        pool: &SpinThreadPool,
        batch_x: &[f32],
        stride: usize,
        cols: usize,
        batch_tokens: usize,
        vluts: &mut [Vq2ActivationLut],
    ) {
        let nc = cols / 64;
        let mut inv_rms = [0.0f32; 8];
        for (t, dst) in inv_rms[..batch_tokens].iter_mut().enumerate() {
            let xt = &batch_x[t * stride..t * stride + cols];
            *dst = 1.0 / (xt.iter().map(|&v| v * v).sum::<f32>() / (cols as f32) + 1e-5).sqrt();
        }
        let mut lut_ptrs = [0usize; 8];
        let mut sc_ptrs = [0usize; 8];
        for t in 0..batch_tokens {
            lut_ptrs[t] = vluts[t].lut.as_mut_ptr() as usize;
            sc_ptrs[t] = vluts[t].chunk_scales.as_mut_ptr() as usize;
        }
        let x_addr = batch_x.as_ptr() as usize;
        let total = batch_tokens * nc;
        let run_items = |i0: usize, i1: usize| {
            let mut scaled = [0.0f32; 64];
            for idx in i0..i1 {
                let t = idx / nc;
                let c = idx % nc;
                let ir = inv_rms[t];
                // SAFETY: Disjoint `(t, c)` pairs across workers.
                unsafe {
                    let src = std::slice::from_raw_parts(
                        (x_addr as *const f32).add(t * stride + c * 64),
                        64,
                    );
                    for (d, &v) in scaled.iter_mut().zip(src.iter()) {
                        *d = v * ir;
                    }
                    Vq2ActivationLut::populate_chunk_fwht_raw(
                        lut_ptrs[t] as *mut [i8; 16],
                        sc_ptrs[t] as *mut f32,
                        c,
                        &scaled,
                        None,
                    );
                }
            }
        };
        if pool.num_threads() > 1 && total >= 64 {
            pool.parallel_for(total, 8, run_items);
        } else {
            run_items(0, total);
        }
    }

    /// Prefetches layers `0..lookahead_layers` before generation starts.
    pub fn warmup_initial_layers(&mut self, initial_hidden: &[f32]) {
        let warm = self.cfg.lookahead_layers.min(self.cfg.num_layers);
        for l in 0..warm {
            self.routers[l].route_top_k_into(initial_hidden, self.cfg.top_k, &mut self.routed_buf);
            let mut weights = [(0usize, 0.0f32); 16];
            let k_eff = prune_routed_experts(
                &self.routed_buf,
                self.cfg.min_top_k,
                self.cfg.confidence_tau,
                self.cfg.min_tail_prob,
                &mut weights,
            );
            for &(exp_id, _) in &weights[..k_eff] {
                self.pager.prefetch_expert(l, exp_id);
            }
        }
    }

    /// Decodes `hidden` through `num_exec_layers` layers: lookahead prefetch, MLA, shared expert,
    /// pruned routed experts, and the LM head if enabled.
    pub fn forward_decode_layers(
        &mut self,
        hidden: &mut [f32],
        routed_override: Option<&[Vec<usize>]>,
        num_exec_layers: usize,
        enable_lookahead_dma: bool,
    ) -> MoEStepStats {
        let t0 = std::time::Instant::now();
        let nl = num_exec_layers.min(self.cfg.num_layers);
        let d_model = self.cfg.d_model;
        assert_eq!(hidden.len(), d_model);
        let pos = self.pos.min(self.cfg.max_seq_len - 1);
        let mut scores_scratch = [0.0f32; 256];
        let mut latent_out = vec![0.0f32; self.cfg.kv_lora_rank];
        let exp_bytes = self.cfg.expert_layout().aligned_expert_bytes as u64;
        let p_exp = self.cfg.params_per_expert() as u64;

        let mut total_routes = 0usize;
        let mut pruned_tail = 0usize;
        let mut active_params = 0u64;
        let mut bytes_streamed = 0u64;

        for l in 0..nl {
            // 1. Prefetch experts for upcoming layers.
            if enable_lookahead_dma && l + self.cfg.lookahead_layers < nl {
                let fl = l + self.cfg.lookahead_layers;
                if let Some(ov) = routed_override {
                    for &exp_id in &ov[fl] {
                        self.pager.prefetch_expert(fl, exp_id);
                    }
                } else {
                    self.pager.prefetch_lookahead_layer(
                        fl,
                        &self.projectors[l],
                        &self.routers[fl],
                        hidden,
                        &mut self.lookahead_scratch,
                        self.cfg.top_k,
                        self.cfg.min_top_k,
                        self.cfg.confidence_tau,
                        self.cfg.min_tail_prob,
                    );
                }
            }

            // 2. MLA attention.
            if let Some(fs) = self.full_stack.as_mut() {
                Self::populate_vlut_rmsnorm(&self.pool, hidden, &mut self.vlut_in);
                let mla = &fs.mla_layers[l % fs.mla_layers.len()];
                active_params += mla.params_count();
                bytes_streamed += mla.physical_bytes() as u64;

                let in_ref = [&self.vlut_in];
                vq2_matrix_exec_l1d_coalesced(
                    &mla.q_a_proj,
                    &in_ref,
                    &mut fs.q_lora_buf[..self.cfg.q_lora_rank],
                    Some(&self.pool),
                );
                vq2_matrix_exec_l1d_coalesced(
                    &mla.kv_a_proj,
                    &in_ref,
                    &mut fs.kv_a_buf[..self.cfg.kv_lora_rank + 64],
                    Some(&self.pool),
                );
                Self::populate_vlut_rmsnorm(
                    &self.pool,
                    &fs.q_lora_buf[..self.cfg.q_lora_rank],
                    &mut fs.vluts_q_lora[0],
                );
                let q_ref = [&fs.vluts_q_lora[0]];
                vq2_matrix_exec_l1d_coalesced(
                    &mla.q_b_proj,
                    &q_ref,
                    &mut fs.q_full_buf[..mla.q_b_proj.rows],
                    Some(&self.pool),
                );

                let (c_kv, k_rope) = fs.kv_a_buf[..self.cfg.kv_lora_rank + 64]
                    .split_at(self.cfg.kv_lora_rank);
                self.mla_kv.store_step(l, pos, c_kv, k_rope);
                if pos < scores_scratch.len() {
                    self.mla_kv.attend_absorbed_head(
                        l,
                        pos,
                        &fs.q_full_buf[..self.cfg.kv_lora_rank],
                        &fs.q_full_buf[self.cfg.kv_lora_rank..self.cfg.kv_lora_rank + 64],
                        0.125,
                        &mut scores_scratch[..pos + 1],
                        &mut latent_out,
                    );
                }
                Self::populate_vlut_rmsnorm(&self.pool, &latent_out, &mut fs.vluts_kv_lora[0]);
                let kv_ref = [&fs.vluts_kv_lora[0]];
                vq2_matrix_exec_l1d_coalesced(
                    &mla.kv_b_proj,
                    &kv_ref,
                    &mut fs.kv_b_buf[..mla.kv_b_proj.rows],
                    Some(&self.pool),
                );
                let v_concat = self.cfg.num_attn_heads * self.cfg.v_head_dim;
                Self::populate_vlut_rmsnorm(
                    &self.pool,
                    &fs.kv_b_buf[..v_concat],
                    &mut fs.vluts_v_concat[0],
                );
                let v_ref = [&fs.vluts_v_concat[0]];
                vq2_matrix_exec_l1d_coalesced(
                    &mla.o_proj,
                    &v_ref,
                    &mut fs.mla_out_buf[..d_model],
                    Some(&self.pool),
                );
                for (h, &mo) in hidden.iter_mut().zip(fs.mla_out_buf[..d_model].iter()) {
                    *h += mo * 0.125;
                }
            } else {
                self.mla_kv.store_step(
                    l,
                    pos,
                    &hidden[..self.cfg.kv_lora_rank],
                    &hidden[self.cfg.kv_lora_rank..self.cfg.kv_lora_rank + 64],
                );
                if pos < scores_scratch.len() {
                    self.mla_kv.attend_absorbed_head(
                        l,
                        pos,
                        &hidden[..self.cfg.kv_lora_rank],
                        &hidden[self.cfg.kv_lora_rank..self.cfg.kv_lora_rank + 64],
                        0.125,
                        &mut scores_scratch[..pos + 1],
                        &mut latent_out,
                    );
                    for (h, &lo) in
                        hidden[..self.cfg.kv_lora_rank].iter_mut().zip(latent_out.iter())
                    {
                        *h += lo * 0.05;
                    }
                }
            }

            // 3. RMSNorm + FWHT-64 + LUT, shared by all experts.
            Self::populate_vlut_rmsnorm(&self.pool, hidden, &mut self.vlut_in);

            // 4. Route and prune.
            let mut weights = [(0usize, 0.0f32); 16];
            let k_eff = if let Some(ov) = routed_override {
                let k = ov[l].len().min(16);
                let w = 1.0 / (k.max(1) as f32);
                for (i, &eid) in ov[l][..k].iter().enumerate() {
                    weights[i] = (eid, w);
                }
                k
            } else {
                self.routers[l].route_top_k_into(hidden, self.cfg.top_k, &mut self.routed_buf);
                let kept = prune_routed_experts(
                    &self.routed_buf,
                    self.cfg.min_top_k,
                    self.cfg.confidence_tau,
                    self.cfg.min_tail_prob,
                    &mut weights,
                );
                pruned_tail += self.cfg.top_k.saturating_sub(kept);
                kept
            };
            total_routes += k_eff;

            // 5. Shared expert, then routed experts.
            if self.cfg.num_shared_experts > 0 {
                let shared_view = self.pager.mmap.expert_view(l, 0).expect("valid shared view");
                shared_view.forward_swiglu_zero_copy(
                    &self.vlut_in,
                    &mut self.gate_up_buf,
                    &mut self.vlut_mid,
                    &mut self.expert_out,
                    Some(&self.pool),
                );
                for (h, &eo) in hidden.iter_mut().zip(self.expert_out.iter()) {
                    *h += eo * 0.25;
                }
                active_params += p_exp;
                bytes_streamed += exp_bytes;
            }

            for &(exp_id, weight) in &weights[..k_eff] {
                let view = self.pager.access_expert(l, exp_id).expect("valid expert view");
                view.forward_swiglu_zero_copy(
                    &self.vlut_in,
                    &mut self.gate_up_buf,
                    &mut self.vlut_mid,
                    &mut self.expert_out,
                    Some(&self.pool),
                );
                for (h, &eo) in hidden.iter_mut().zip(self.expert_out.iter()) {
                    *h += eo * weight;
                }
                active_params += p_exp;
                bytes_streamed += exp_bytes;
            }
        }

        // 6. LM head (optional).
        if let Some(fs) = self.full_stack.as_mut() {
            if let Some(head) = fs.lm_head.as_ref() {
                Self::populate_vlut_rmsnorm(&self.pool, hidden, &mut self.vlut_in);
                let in_ref = [&self.vlut_in];
                vq2_matrix_exec_l1d_coalesced(
                    head,
                    &in_ref,
                    &mut fs.logits_buf[..head.rows],
                    Some(&self.pool),
                );
                active_params += (head.rows * head.cols) as u64;
                bytes_streamed += (head.packed_data.len() + head.bf16_chunk_scales.len() * 2) as u64;
            }
        }

        self.pos = (self.pos + 1).min(self.cfg.max_seq_len - 1);
        MoEStepStats {
            tokens: 1,
            total_expert_routes: total_routes,
            unique_expert_loads: total_routes,
            pruned_tail_experts: pruned_tail,
            active_params_per_token: active_params,
            weight_bytes_streamed: bytes_streamed,
            elapsed_secs: t0.elapsed().as_secs_f64(),
        }
    }

    /// Verifies `B` draft tokens (`batch_hiddens.len() == B * d_model`) in one pass. MLA, shared
    /// experts, the LM head and each unique routed expert are read once for the whole batch.
    pub fn forward_speculative_coalesced_layers(
        &mut self,
        batch_hiddens: &mut [f32],
        batch_tokens: usize,
        routed_batch_override: Option<&[Vec<Vec<usize>>]>,
        num_exec_layers: usize,
    ) -> MoEStepStats {
        let t0 = std::time::Instant::now();
        assert!((1..=8).contains(&batch_tokens));
        let d_model = self.cfg.d_model;
        assert_eq!(batch_hiddens.len(), batch_tokens * d_model);
        let nl = num_exec_layers.min(self.cfg.num_layers);
        let exp_bytes = self.cfg.expert_layout().aligned_expert_bytes as u64;
        let p_exp = self.cfg.params_per_expert() as u64;

        let mut total_routes = 0usize;
        let mut unique_loads = 0usize;
        let mut pruned_tail = 0usize;
        let mut total_active_params_sum = 0u64;
        let mut bytes_streamed = 0u64;
        let mut coalesced: HashMap<usize, Vec<(usize, f32)>> = HashMap::with_capacity(64);
        let mut unique_eids: Vec<usize> = Vec::with_capacity(64);
        let mut scores_scratch = [0.0f32; 256];
        let mut latent_batch = vec![0.0f32; batch_tokens * self.cfg.kv_lora_rank];

        for l in 0..nl {
            // 1. MLA for all tokens (optional).
            if let Some(fs) = self.full_stack.as_mut() {
                Self::populate_batch_vluts_rmsnorm(
                    &self.pool,
                    batch_hiddens,
                    d_model,
                    d_model,
                    batch_tokens,
                    &mut self.batch_vluts_in[..batch_tokens],
                );
                let in_refs: Vec<&Vq2ActivationLut> =
                    self.batch_vluts_in[..batch_tokens].iter().collect();
                let mla = &fs.mla_layers[l % fs.mla_layers.len()];
                total_active_params_sum += mla.params_count() * (batch_tokens as u64);
                bytes_streamed += mla.physical_bytes() as u64;

                let q_lora = self.cfg.q_lora_rank;
                let kv_down = self.cfg.kv_lora_rank + 64;
                vq2_matrix_exec_l1d_coalesced(
                    &mla.q_a_proj,
                    &in_refs,
                    &mut fs.q_lora_buf[..batch_tokens * q_lora],
                    Some(&self.pool),
                );
                vq2_matrix_exec_l1d_coalesced(
                    &mla.kv_a_proj,
                    &in_refs,
                    &mut fs.kv_a_buf[..batch_tokens * kv_down],
                    Some(&self.pool),
                );
                Self::populate_batch_vluts_rmsnorm(
                    &self.pool,
                    &fs.q_lora_buf[..batch_tokens * q_lora],
                    q_lora,
                    q_lora,
                    batch_tokens,
                    &mut fs.vluts_q_lora[..batch_tokens],
                );
                let q_refs: Vec<&Vq2ActivationLut> =
                    fs.vluts_q_lora[..batch_tokens].iter().collect();
                vq2_matrix_exec_l1d_coalesced(
                    &mla.q_b_proj,
                    &q_refs,
                    &mut fs.q_full_buf[..batch_tokens * mla.q_b_proj.rows],
                    Some(&self.pool),
                );

                for t in 0..batch_tokens {
                    let pos_t = (self.pos + t).min(self.cfg.max_seq_len - 1);
                    let kv_t = &fs.kv_a_buf[t * kv_down..(t + 1) * kv_down];
                    let (c_kv, k_rope) = kv_t.split_at(self.cfg.kv_lora_rank);
                    self.mla_kv.store_step(l, pos_t, c_kv, k_rope);
                    if pos_t < scores_scratch.len() {
                        let q_t = &fs.q_full_buf[t * mla.q_b_proj.rows..(t + 1) * mla.q_b_proj.rows];
                        self.mla_kv.attend_absorbed_head(
                            l,
                            pos_t,
                            &q_t[..self.cfg.kv_lora_rank],
                            &q_t[self.cfg.kv_lora_rank..self.cfg.kv_lora_rank + 64],
                            0.125,
                            &mut scores_scratch[..pos_t + 1],
                            &mut latent_batch[t * self.cfg.kv_lora_rank..(t + 1) * self.cfg.kv_lora_rank],
                        );
                    }
                }

                Self::populate_batch_vluts_rmsnorm(
                    &self.pool,
                    &latent_batch,
                    self.cfg.kv_lora_rank,
                    self.cfg.kv_lora_rank,
                    batch_tokens,
                    &mut fs.vluts_kv_lora[..batch_tokens],
                );
                let kv_refs: Vec<&Vq2ActivationLut> =
                    fs.vluts_kv_lora[..batch_tokens].iter().collect();
                vq2_matrix_exec_l1d_coalesced(
                    &mla.kv_b_proj,
                    &kv_refs,
                    &mut fs.kv_b_buf[..batch_tokens * mla.kv_b_proj.rows],
                    Some(&self.pool),
                );
                let v_concat = self.cfg.num_attn_heads * self.cfg.v_head_dim;
                Self::populate_batch_vluts_rmsnorm(
                    &self.pool,
                    &fs.kv_b_buf[..batch_tokens * mla.kv_b_proj.rows],
                    mla.kv_b_proj.rows,
                    v_concat,
                    batch_tokens,
                    &mut fs.vluts_v_concat[..batch_tokens],
                );
                let v_refs: Vec<&Vq2ActivationLut> =
                    fs.vluts_v_concat[..batch_tokens].iter().collect();
                vq2_matrix_exec_l1d_coalesced(
                    &mla.o_proj,
                    &v_refs,
                    &mut fs.mla_out_buf[..batch_tokens * d_model],
                    Some(&self.pool),
                );
                for (h, &mo) in batch_hiddens
                    .iter_mut()
                    .zip(fs.mla_out_buf[..batch_tokens * d_model].iter())
                {
                    *h += mo * 0.125;
                }
            }

            // 2. FFN LUT per token.
            Self::populate_batch_vluts_rmsnorm(
                &self.pool,
                batch_hiddens,
                d_model,
                d_model,
                batch_tokens,
                &mut self.batch_vluts_in[..batch_tokens],
            );

            // 3. Shared expert for all tokens.
            if self.cfg.num_shared_experts > 0 {
                let refs: Vec<&Vq2ActivationLut> =
                    self.batch_vluts_in[..batch_tokens].iter().collect();
                let shared_view = self.pager.mmap.expert_view(l, 0).expect("valid shared view");
                shared_view.forward_swiglu_coalesced_vluts(
                    &refs,
                    &mut self.batch_gu[..batch_tokens * 2 * self.cfg.d_ff_expert],
                    &mut self.batch_vluts_mid[..batch_tokens],
                    &mut self.batch_out[..batch_tokens * d_model],
                    Some(&self.pool),
                );
                for (h, &v) in batch_hiddens
                    .iter_mut()
                    .zip(self.batch_out[..batch_tokens * d_model].iter())
                {
                    *h += v * 0.25;
                }
                total_active_params_sum += p_exp * (batch_tokens as u64);
                bytes_streamed += exp_bytes;
            }

            // 4. Route all tokens and group by expert.
            coalesced.clear();
            for t in 0..batch_tokens {
                if let Some(ov) = routed_batch_override {
                    let exps = &ov[l][t];
                    let w = 1.0 / (exps.len().max(1) as f32);
                    total_routes += exps.len();
                    for &eid in exps {
                        coalesced.entry(eid).or_default().push((t, w));
                    }
                } else {
                    let hid = &batch_hiddens[t * d_model..(t + 1) * d_model];
                    self.routers[l].route_top_k_into(hid, self.cfg.top_k, &mut self.routed_buf);
                    let mut weights = [(0usize, 0.0f32); 16];
                    let k_eff = prune_routed_experts(
                        &self.routed_buf,
                        self.cfg.min_top_k,
                        self.cfg.confidence_tau,
                        self.cfg.min_tail_prob,
                        &mut weights,
                    );
                    pruned_tail += self.cfg.top_k.saturating_sub(k_eff);
                    total_routes += k_eff;
                    for &(eid, w) in &weights[..k_eff] {
                        coalesced.entry(eid).or_default().push((t, w));
                    }
                }
            }
            unique_loads += coalesced.len();
            bytes_streamed += (coalesced.len() as u64) * exp_bytes;

            // Sorted for sequential page access.
            unique_eids.clear();
            unique_eids.extend(coalesced.keys().copied());
            unique_eids.sort_unstable();

            for &eid in &unique_eids {
                let members = &coalesced[&eid];
                let m = members.len();
                total_active_params_sum += p_exp * (m as u64);
                let view = self.pager.access_expert(l, eid).expect("valid expert view");
                if m == 1 {
                    let (t_idx, weight) = members[0];
                    view.forward_swiglu_zero_copy(
                        &self.batch_vluts_in[t_idx],
                        &mut self.gate_up_buf,
                        &mut self.vlut_mid,
                        &mut self.expert_out,
                        Some(&self.pool),
                    );
                    let hid = &mut batch_hiddens[t_idx * d_model..(t_idx + 1) * d_model];
                    for (h, &v) in hid.iter_mut().zip(self.expert_out.iter()) {
                        *h += v * weight;
                    }
                } else {
                    let refs: Vec<&Vq2ActivationLut> = members
                        .iter()
                        .map(|&(t_idx, _)| &self.batch_vluts_in[t_idx])
                        .collect();
                    view.forward_swiglu_coalesced_vluts(
                        &refs,
                        &mut self.batch_gu[..m * 2 * self.cfg.d_ff_expert],
                        &mut self.batch_vluts_mid[..m],
                        &mut self.batch_out[..m * d_model],
                        Some(&self.pool),
                    );
                    for (slot, &(t_idx, weight)) in members.iter().enumerate() {
                        let hid = &mut batch_hiddens[t_idx * d_model..(t_idx + 1) * d_model];
                        let eo = &self.batch_out[slot * d_model..(slot + 1) * d_model];
                        for (h, &v) in hid.iter_mut().zip(eo.iter()) {
                            *h += v * weight;
                        }
                    }
                }
            }
        }

        // 5. LM head for all tokens (optional).
        if let Some(fs) = self.full_stack.as_mut() {
            if let Some(head) = fs.lm_head.as_ref() {
                Self::populate_batch_vluts_rmsnorm(
                    &self.pool,
                    batch_hiddens,
                    d_model,
                    d_model,
                    batch_tokens,
                    &mut self.batch_vluts_in[..batch_tokens],
                );
                let in_refs: Vec<&Vq2ActivationLut> =
                    self.batch_vluts_in[..batch_tokens].iter().collect();
                vq2_matrix_exec_l1d_coalesced(
                    head,
                    &in_refs,
                    &mut fs.logits_buf[..batch_tokens * head.rows],
                    Some(&self.pool),
                );
                total_active_params_sum += (head.rows * head.cols * batch_tokens) as u64;
                bytes_streamed +=
                    (head.packed_data.len() + head.bf16_chunk_scales.len() * 2) as u64;
            }
        }

        self.pos = (self.pos + batch_tokens).min(self.cfg.max_seq_len - 1);
        MoEStepStats {
            tokens: batch_tokens,
            total_expert_routes: total_routes,
            unique_expert_loads: unique_loads,
            pruned_tail_experts: pruned_tail,
            active_params_per_token: total_active_params_sum / (batch_tokens as u64),
            weight_bytes_streamed: bytes_streamed,
            elapsed_secs: t0.elapsed().as_secs_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::compiler::compile_hadamard_gptq_vq2;
    use crate::kernels::simd_lut::vq2_lut_gemv;

    #[test]
    fn test_frontier_800b_budget_fits_24gb_m4() {
        let cfg = Kronos800bConfig::frontier_800b();
        let rep = cfg.budget_report();
        // Over 800B params, page-aligned experts, resident RAM within 20 GiB.
        assert!(
            rep.total_params >= 800_000_000_000,
            "expected >= 800B params, got {}",
            rep.total_params
        );
        assert_eq!(rep.bytes_per_expert_aligned, 11_698_176);
        assert_eq!(rep.bytes_per_expert_aligned % K2TM_VM_PAGE_BYTES, 0);
        let ram_gib = rep.total_resident_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            ram_gib <= 20.0,
            "800B resident RAM ({ram_gib:.2} GiB) must leave >= 4 GiB free on 24 GiB M4"
        );
    }

    #[test]
    fn test_mmap_zero_copy_expert_matches_in_ram_vq2_and_coalesced_batch() {
        let cfg = Kronos800bConfig::compact_test(2, 128, 128, 3);
        let gu_w: Vec<f32> = (0..(256 * 128))
            .map(|i| ((i as f32) * 0.037).sin() * 0.08)
            .collect();
        let dn_w: Vec<f32> = (0..(128 * 128))
            .map(|i| ((i as f32) * 0.053).cos() * 0.08)
            .collect();
        let gu_mat = compile_hadamard_gptq_vq2(&gu_w, 256, 128, 2, None, 2);
        let dn_mat = compile_hadamard_gptq_vq2(&dn_w, 128, 128, 2, None, 2);

        let tmp = std::env::temp_dir().join("kronos_test_800b_exact_parity.k2tm");
        write_exact_experts_k2tm(&tmp, &cfg, &[(gu_mat.clone(), dn_mat.clone())])
            .expect("write exact k2tm");
        let mmap = K2tmShardMmap::open(&tmp, &cfg).expect("open k2tm");
        let view = mmap.expert_view(0, 0).expect("expert view");

        let x0: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.11).sin()).collect();
        let x1: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.17).cos()).collect();
        let mut vlut0 = Vq2ActivationLut::new_zeroed(128);
        let mut vlut1 = Vq2ActivationLut::new_zeroed(128);
        vlut0.populate_fwht(&x0);
        vlut1.populate_fwht(&x1);

        // Reference via in-RAM `Vq2LutMatrix`
        let mut gu_ref = vec![0.0f32; 256];
        vq2_lut_gemv(&gu_mat, &vlut0, &mut gu_ref);
        let mut sw_ref = vec![0.0f32; 128];
        for i in 0..128 {
            let g = gu_ref[i];
            sw_ref[i] = (g / (1.0 + (-g).exp())) * gu_ref[128 + i];
        }
        let mut vlut_sw_ref = Vq2ActivationLut::new_zeroed(128);
        vlut_sw_ref.populate_fwht(&sw_ref);
        let mut out_ref = vec![0.0f32; 128];
        vq2_lut_gemv(&dn_mat, &vlut_sw_ref, &mut out_ref);

        // Single token from `mmap`.
        let mut gu_zc = vec![0.0f32; 256];
        let mut vlut_mid = Vq2ActivationLut::new_zeroed(128);
        let mut out_zc = vec![0.0f32; 128];
        view.forward_swiglu_zero_copy(&vlut0, &mut gu_zc, &mut vlut_mid, &mut out_zc, None);

        for (a, b) in out_ref.iter().zip(out_zc.iter()) {
            assert!((a - b).abs() < 1e-6, "zero-copy mmap mismatch: {a} vs {b}");
        }

        // Two-token batch.
        let mut batch_gu = vec![0.0f32; 2 * 256];
        let mut batch_mid = vec![Vq2ActivationLut::new_zeroed(128), Vq2ActivationLut::new_zeroed(128)];
        let mut batch_out = vec![0.0f32; 2 * 128];
        view.forward_swiglu_coalesced_vluts(
            &[&vlut0, &vlut1],
            &mut batch_gu,
            &mut batch_mid,
            &mut batch_out,
            None,
        );
        for (a, b) in out_ref.iter().zip(batch_out[..128].iter()) {
            assert!((a - b).abs() < 1e-6, "coalesced batch token 0 mismatch: {a} vs {b}");
        }
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_slru_scan_resistance_and_mla_hadamard_q8_accuracy() {
        let cfg = Kronos800bConfig::compact_test(2, 128, 128, 3);
        let tmp = std::env::temp_dir().join("kronos_test_800b_slru.k2tm");
        write_k2tm_shard_file(&tmp, &cfg, 2, 9, 42).expect("write shard");
        let mmap = K2tmShardMmap::open(&tmp, &cfg).expect("open shard");
        let mut pager = SlruExpertPager::new(mmap, 4); // prob = 1, prot = 3

        // Touch experts 0, 1, 2 twice to promote them.
        for eid in [0, 1, 2] {
            let _ = pager.access_expert(0, eid).unwrap();
            let _ = pager.access_expert(0, eid).unwrap();
        }
        assert_eq!(pager.protected_count(), 3);

        // Scan one-off experts 3..9; protected experts 0, 1, 2 must survive.
        for eid in 3..9 {
            let _ = pager.access_expert(0, eid).unwrap();
        }
        for eid in [0, 1, 2] {
            assert!(
                pager.is_resident(0, eid),
                "Protected hot expert {eid} must resist one-shot tail scan eviction!"
            );
        }

        // INT8 latent attention vs FP32 reference.
        let mut mla = MlaLatentKvCache::new(1, 16, 128, 64);
        let mut c_hist = Vec::new();
        let mut r_hist = Vec::new();
        for t in 0..8 {
            let c: Vec<f32> = (0..128)
                .map(|i| {
                    let v = (((t * 17 + i * 13) % 97) as f32 / 97.0 - 0.5) * 1.4;
                    if i == 11 { v * 10.0 } else { v }
                })
                .collect();
            let r: Vec<f32> = (0..64)
                .map(|i| (((t * 23 + i * 7) % 89) as f32 / 89.0 - 0.5) * 1.8)
                .collect();
            mla.store_step(0, t, &c, &r);
            c_hist.push(c);
            r_hist.push(r);
        }
        let q_lat: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.21).sin()).collect();
        let q_rope: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.31).cos()).collect();
        let mut sc = [0.0f32; 8];
        let mut out_q8 = vec![0.0f32; 128];
        mla.attend_absorbed_head(0, 7, &q_lat, &q_rope, 0.125, &mut sc, &mut out_q8);

        let mut ref_sc = [0.0f32; 8];
        for t in 0..8 {
            let d1: f32 = q_lat.iter().zip(&c_hist[t]).map(|(a, b)| a * b).sum();
            let d2: f32 = q_rope.iter().zip(&r_hist[t]).map(|(a, b)| a * b).sum();
            ref_sc[t] = (d1 + d2) * 0.125;
        }
        let m = ref_sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = ref_sc.iter().map(|s| (s - m).exp()).collect();
        let z: f32 = exps.iter().sum();
        let mut out_ref = vec![0.0f32; 128];
        for t in 0..8 {
            let p = exps[t] / z;
            for i in 0..128 {
                out_ref[i] += p * c_hist[t][i];
            }
        }
        let dot: f64 = out_ref.iter().zip(&out_q8).map(|(&a, &b)| (a as f64) * (b as f64)).sum();
        let na: f64 = out_ref.iter().map(|&a| (a as f64).powi(2)).sum();
        let nb: f64 = out_q8.iter().map(|&b| (b as f64).powi(2)).sum();
        let cos = dot / (na.sqrt() * nb.sqrt());
        assert!(cos > 0.9998, "MLA Hadamard INT8 cosine too low: {cos}");

        let _ = std::fs::remove_file(&tmp);
    }

    #[cfg(unix)]
    #[test]
    fn test_direct_nvme_fnocache_slru_and_nf3_expert_zero_copy_parity() {
        use crate::model::compiler::compile_hadamard_gptq_nf;
        use crate::kernels::simd_lut::nf_sdot_stripe_range_int8;

        let nf3_cfg = Kronos800bConfig::frontier_800b_nf3();
        let nf3_rep = nf3_cfg.nf_budget_report(3);
        let nf3_ram_gib = nf3_rep.total_resident_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!(
            nf3_ram_gib <= 20.0,
            "NF3 800B resident RAM ({nf3_ram_gib:.2} GiB) must fit <= 20 GiB on 24 GiB M4"
        );
        assert_eq!(nf3_rep.bytes_per_expert_aligned % K2TM_VM_PAGE_BYTES, 0);

        let cfg = Kronos800bConfig::compact_test(1, 128, 128, 2);
        let gu_w: Vec<f32> = (0..(256 * 128)).map(|i| ((i as f32) * 0.041).sin() * 0.12).collect();
        let dn_w: Vec<f32> = (0..(128 * 128)).map(|i| ((i as f32) * 0.067).cos() * 0.12).collect();
        let gu_nf3 = compile_hadamard_gptq_nf(&gu_w, 256, 128, 3, 2, None, 2);
        let dn_nf3 = compile_hadamard_gptq_nf(&dn_w, 128, 128, 3, 2, None, 2);

        let tmp = std::env::temp_dir().join("kronos_test_fnocache_nf3_parity.k2tm");
        write_exact_nf_experts_k2tm(
            &tmp,
            &cfg,
            3,
            1,
            2,
            &[(gu_nf3.clone(), dn_nf3.clone()), (gu_nf3.clone(), dn_nf3.clone())],
        )
        .expect("write NF3 k2tm");

        let lay = cfg.nf_expert_layout(3);
        let shard = DirectNvmeShard::open(&tmp, lay.aligned_expert_bytes).expect("open F_NOCACHE shard");
        let mut pager = DirectNvmeSlruPager::new(shard, 2, true, true).expect("create SLRU pager");

        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.13).sin()).collect();
        let mut act_in = Int8ChunkActivation::new_zeroed(128);
        act_in.populate_fwht(&x);

        // Reference via in-RAM `NfSdotMatrix`
        let mut gu_ref = vec![0.0f32; 256];
        unsafe {
            nf_sdot_stripe_range_int8(&gu_nf3, &act_in, 0, gu_nf3.num_stripes, gu_ref.as_mut_ptr());
        }
        let mut sw_ref = vec![0.0f32; 128];
        for i in 0..128 {
            let g = gu_ref[i];
            sw_ref[i] = (g / (1.0 + (-g).exp())) * gu_ref[128 + i];
        }
        let mut act_mid_ref = Int8ChunkActivation::new_zeroed(128);
        act_mid_ref.populate_fwht(&sw_ref);
        let mut out_ref = vec![0.0f32; 128];
        unsafe {
            nf_sdot_stripe_range_int8(&dn_nf3, &act_mid_ref, 0, dn_nf3.num_stripes, out_ref.as_mut_ptr());
        }

        // Cold `F_NOCACHE` read into `MmapNfExpertView`.
        let mut gu_zc = vec![0.0f32; 256];
        let mut act_mid = Int8ChunkActivation::new_zeroed(128);
        let mut out_zc = vec![0.0f32; 128];
        {
            let slice = pager.access_expert_slice(0, 0).expect("cold F_NOCACHE pread");
            let view = MmapNfExpertView::from_aligned_slice(slice, lay).expect("NF3 view");
            view.forward_swiglu_zero_copy(&act_in, &mut gu_zc, &mut act_mid, &mut out_zc, None);
        }
        assert_eq!(pager.nvme_misses, 1);
        assert!(pager.shard.nvme_bytes_read.load(Ordering::Relaxed) >= lay.aligned_expert_bytes as u64);
        for (a, b) in out_ref.iter().zip(out_zc.iter()) {
            assert!((a - b).abs() < 1e-6, "F_NOCACHE NF3 mismatch: {a} vs {b}");
        }

        // Prefetch expert 1, then expect a hit.
        pager.prefetch_expert_async(0, 1);
        let _ = pager.access_expert_slice(0, 1).expect("prefetched expert 1");
        assert_eq!(pager.dram_hits, 1);

        let _ = std::fs::remove_file(&tmp);
    }
}
