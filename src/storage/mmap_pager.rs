//! Zero-Copy `mmap` Flash Pager & Asynchronous Ring-Buffer Cache.
//!
//! Maps `.k2t` 4KB page-aligned model containers directly into virtual address space via
//! POSIX `mmap(PROT_READ, MAP_PRIVATE)` and wires 6-layer lookahead predictions to OS kernel
//! `posix_madvise(POSIX_MADV_WILLNEED)` / `POSIX_MADV_DONTNEED` hints.

use crate::storage::k2t_format::{K2tExpertPageView, K2tHeader, K2T_PAGE_SIZE};
use crate::storage::prefetch::SubspaceProjector;
use crate::kernels::router::ProductKeyRouter;
use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io;
use std::path::Path;

#[cfg(unix)]
pub(crate) mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::ptr;

    const PROT_READ: i32 = 0x01;
    const MAP_PRIVATE: i32 = 0x02;
    const POSIX_MADV_WILLNEED: i32 = 3;
    const POSIX_MADV_DONTNEED: i32 = 4;

    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: i32,
            flags: i32,
            fd: i32,
            offset: i64,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> i32;
        fn posix_madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
    }

    pub struct RawMmap {
        ptr: *mut c_void,
        len: usize,
    }

    // SAFETY: Mapping is strictly read-only (`PROT_READ`, `MAP_PRIVATE`) for its entire lifetime.
    unsafe impl Send for RawMmap {}
    // SAFETY: Concurrent shared reads and `posix_madvise` hints on a read-only mapping are thread-safe.
    unsafe impl Sync for RawMmap {}

    impl RawMmap {
        pub fn map_read_only(file: &File, len: usize) -> io::Result<Self> {
            // SAFETY: Valid open file descriptor, non-zero byte length, read-only private mapping.
            let ptr = unsafe {
                mmap(
                    ptr::null_mut(),
                    len,
                    PROT_READ,
                    MAP_PRIVATE,
                    file.as_raw_fd(),
                    0,
                )
            };
            let map_failed = (!0usize) as *mut c_void;
            if ptr == map_failed || ptr.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok(Self { ptr, len })
        }

        #[inline]
        pub fn as_slice(&self) -> &[u8] {
            // SAFETY: `self.ptr` points to a valid `self.len`-byte read-only kernel mapping.
            unsafe { std::slice::from_raw_parts(self.ptr.cast::<u8>(), self.len) }
        }

        #[inline]
        pub fn advise_willneed(&self, offset: usize, len: usize) {
            assert!(offset + len <= self.len);
            // SAFETY: Range `[offset, offset + len)` lies strictly inside `[0, self.len)` and is page-aligned.
            unsafe {
                let _ = posix_madvise(self.ptr.byte_add(offset), len, POSIX_MADV_WILLNEED);
            }
        }

        #[inline]
        pub fn advise_dontneed(&self, offset: usize, len: usize) {
            assert!(offset + len <= self.len);
            // SAFETY: Range `[offset, offset + len)` lies strictly inside `[0, self.len)` and is page-aligned.
            unsafe {
                let _ = posix_madvise(self.ptr.byte_add(offset), len, POSIX_MADV_DONTNEED);
            }
        }
    }

    impl Drop for RawMmap {
        fn drop(&mut self) {
            if !self.ptr.is_null() && self.len > 0 {
                // SAFETY: Unmaps the exact `[self.ptr, self.len)` region created in `map_read_only`.
                unsafe {
                    let _ = munmap(self.ptr, self.len);
                }
            }
        }
    }
}

#[cfg(not(unix))]
pub(crate) mod sys {
    use std::fs::File;
    use std::io::{self, Read};

    pub struct RawMmap {
        buf: Vec<u8>,
    }

    impl RawMmap {
        pub fn map_read_only(mut file: &File, len: usize) -> io::Result<Self> {
            let mut buf = Vec::with_capacity(len);
            file.read_to_end(&mut buf)?;
            Ok(Self { buf })
        }

        #[inline]
        pub fn as_slice(&self) -> &[u8] {
            &self.buf
        }

        #[inline]
        pub fn advise_willneed(&self, offset: usize, len: usize) {
            assert!(offset + len <= self.buf.len());
        }

        #[inline]
        pub fn advise_dontneed(&self, offset: usize, len: usize) {
            assert!(offset + len <= self.buf.len());
        }
    }
}

/// Read-only, 4KB page-aligned zero-copy `.k2t` memory mapping.
pub struct K2tMmap {
    raw: sys::RawMmap,
    pub header: K2tHeader,
}

impl K2tMmap {
    /// Opens `path`, maps it into virtual address space, and validates Page 0 header & size.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len < K2T_PAGE_SIZE || !len.is_multiple_of(K2T_PAGE_SIZE) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2t file length must be a non-zero multiple of 4096 bytes",
            ));
        }

        let raw = sys::RawMmap::map_read_only(&file, len)?;
        let header = K2tHeader::decode_page(&raw.as_slice()[0..K2T_PAGE_SIZE])
            .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;

        let expected_bytes = (header.total_pages as usize) * K2T_PAGE_SIZE;
        if len != expected_bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "k2t file byte size does not match header total_pages * 4096",
            ));
        }

        Ok(Self { raw, header })
    }

    /// Total mapped size in bytes.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.raw.as_slice().len()
    }

    /// Issues an asynchronous kernel read-ahead hint (`POSIX_MADV_WILLNEED`) for a 4KB page.
    #[inline]
    pub fn advise_willneed_page(&self, page_idx: usize) {
        assert!(page_idx > 0 && page_idx < self.header.total_pages as usize);
        self.raw
            .advise_willneed(page_idx * K2T_PAGE_SIZE, K2T_PAGE_SIZE);
    }

    /// Releases physical page cache pressure (`POSIX_MADV_DONTNEED`) when evicted from ring-buffer.
    #[inline]
    pub fn advise_dontneed_page(&self, page_idx: usize) {
        assert!(page_idx > 0 && page_idx < self.header.total_pages as usize);
        self.raw
            .advise_dontneed(page_idx * K2T_PAGE_SIZE, K2T_PAGE_SIZE);
    }

    /// Zero-copy slice borrow of an expert page (`4096` bytes) with $O(1)$ offset calculation.
    #[inline]
    pub fn expert_view(
        &self,
        layer_idx: usize,
        virtual_expert_id: usize,
    ) -> Result<K2tExpertPageView<'_>, &'static str> {
        let page_idx = self.header.expert_page_index(layer_idx, virtual_expert_id);
        let start = page_idx * K2T_PAGE_SIZE;
        let slice = &self.raw.as_slice()[start..start + K2T_PAGE_SIZE];
        K2tExpertPageView::from_page_slice(slice)
    }
}

/// Couples a [`K2tMmap`] container with a bounded RAM page ring-buffer driven by
/// 6-layer lookahead subspace projections.
pub struct AsyncFlashPager {
    pub mmap: K2tMmap,
    pub capacity_pages: usize,
    resident_pages: HashSet<usize>,
    fifo_queue: VecDeque<usize>,
    predicted_buf: Vec<(usize, f32)>,
    pub page_hits: usize,
    pub page_misses: usize,
    pub willneed_syscalls: usize,
    pub dontneed_syscalls: usize,
}

impl AsyncFlashPager {
    pub fn new(mmap: K2tMmap, capacity_pages: usize) -> Self {
        assert!(capacity_pages > 0, "capacity_pages must be > 0");
        Self {
            mmap,
            capacity_pages,
            resident_pages: HashSet::with_capacity(capacity_pages),
            fifo_queue: VecDeque::with_capacity(capacity_pages),
            predicted_buf: Vec::with_capacity(32),
            page_hits: 0,
            page_misses: 0,
            willneed_syscalls: 0,
            dontneed_syscalls: 0,
        }
    }

    /// Projects `current_hidden` 6 layers ahead via `projector`, aligns onto the unit-RMS
    /// manifold, routes on `future_router` with **zero heap allocations**, and triggers OS
    /// `POSIX_MADV_WILLNEED` DMA hints.
    pub fn prefetch_future_layer(
        &mut self,
        future_layer_idx: usize,
        projector: &SubspaceProjector,
        future_router: &ProductKeyRouter,
        current_hidden: &[f32],
        scratch_hidden: &mut [f32],
        prefetch_k: usize,
    ) {
        projector.project(current_hidden, scratch_hidden);
        let mean_sq =
            scratch_hidden.iter().map(|&v| v * v).sum::<f32>() / (scratch_hidden.len() as f32);
        let inv_rms = 1.0 / (mean_sq + 1e-5).sqrt();
        for v in scratch_hidden.iter_mut() {
            *v *= inv_rms;
        }
        future_router.route_top_k_into(scratch_hidden, prefetch_k, &mut self.predicted_buf);
        for i in 0..self.predicted_buf.len() {
            let virtual_id = self.predicted_buf[i].0;
            let page_idx = self
                .mmap
                .header
                .expert_page_index(future_layer_idx, virtual_id);
            self.stage_page(page_idx);
        }
    }

    /// Stages a specific `(layer_idx, virtual_expert_id)` page directly (e.g., cold-start warmup
    /// for layers `0..lookahead` before token generation begins).
    pub fn stage_expert(&mut self, layer_idx: usize, virtual_expert_id: usize) {
        let page_idx = self
            .mmap
            .header
            .expert_page_index(layer_idx, virtual_expert_id);
        self.stage_page(page_idx);
    }

    fn stage_page(&mut self, page_idx: usize) {
        if !self.resident_pages.contains(&page_idx) {
            if self.resident_pages.len() >= self.capacity_pages {
                if let Some(evicted_page) = self.fifo_queue.pop_front() {
                    self.resident_pages.remove(&evicted_page);
                    self.mmap.advise_dontneed_page(evicted_page);
                    self.dontneed_syscalls += 1;
                }
            }
            self.mmap.advise_willneed_page(page_idx);
            self.willneed_syscalls += 1;
            self.resident_pages.insert(page_idx);
            self.fifo_queue.push_back(page_idx);
        }
    }

    /// Records an execution-time expert access (updating hit/miss telemetry) and returns
    /// a zero-copy [`K2tExpertPageView`] directly into the memory-mapped 4KB page.
    pub fn access_expert(
        &mut self,
        layer_idx: usize,
        virtual_expert_id: usize,
    ) -> Result<K2tExpertPageView<'_>, &'static str> {
        let page_idx = self
            .mmap
            .header
            .expert_page_index(layer_idx, virtual_expert_id);
        if self.resident_pages.contains(&page_idx) {
            self.page_hits += 1;
        } else {
            self.page_misses += 1;
            self.stage_page(page_idx);
        }
        self.mmap.expert_view(layer_idx, virtual_expert_id)
    }

    /// Percentage of expert accesses already staged in the RAM ring-buffer prior to execution.
    pub fn hit_rate_percent(&self) -> f64 {
        let total = self.page_hits + self.page_misses;
        if total == 0 {
            0.0
        } else {
            (self.page_hits as f64 / total as f64) * 100.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::k2t_format::write_k2t_container;
    use crate::kernels::simd_lut::QuantizedActivationLut;

    #[test]
    fn test_k2t_write_mmap_prefetch_and_zero_copy_swiglu_exec() {
        let tmp_path = std::env::temp_dir().join("kronos_2t_test_pager.k2t");
        let num_layers = 8u32;
        let experts_per_layer = 16u32;
        let d_model = 256u32;
        let d_ff = 128u32;

        let written_hdr = write_k2t_container(
            &tmp_path,
            num_layers,
            experts_per_layer,
            d_model,
            d_ff,
            0.05,
        )
        .expect("failed to write .k2t file");

        let mmap = K2tMmap::open(&tmp_path).expect("failed to mmap .k2t file");
        assert_eq!(mmap.header, written_hdr);
        assert_eq!(
            mmap.byte_len(),
            ((1 + num_layers * experts_per_layer) as usize) * K2T_PAGE_SIZE
        );

        let mut pager = AsyncFlashPager::new(mmap, 24);
        let router = ProductKeyRouter::new(d_model as usize, 64, 777);
        let projector = SubspaceProjector::new(d_model as usize, 8, 888);

        let hidden: Vec<f32> = (0..d_model as usize)
            .map(|i| ((i as f32) * 0.03).sin())
            .collect();
        let mut scratch = vec![0.0f32; d_model as usize];
        let mut future_hidden = vec![0.0f32; d_model as usize];

        // Layer 0 prefetches for Layer 6 via `POSIX_MADV_WILLNEED`
        pager.prefetch_future_layer(6, &projector, &router, &hidden, &mut scratch, 8);
        assert!(pager.willneed_syscalls > 0);

        projector.project(&hidden, &mut future_hidden);
        let mean_sq =
            future_hidden.iter().map(|&v| v * v).sum::<f32>() / (future_hidden.len() as f32);
        let inv_rms = 1.0 / (mean_sq + 1e-5).sqrt();
        for v in &mut future_hidden {
            *v *= inv_rms;
        }
        let routed = router.route_top_k(&future_hidden, 4);

        let reg_cb = pager.mmap.header.reg_codebook_64;
        let qlut_in = QuantizedActivationLut::build(&future_hidden);
        let mut gate_out = vec![0.0f32; d_ff as usize];
        let mut up_out = vec![0.0f32; d_ff as usize];
        let mut expert_out = vec![0.0f32; d_model as usize];

        for &(exp_id, _) in &routed {
            let view = pager.access_expert(6, exp_id).expect("valid expert page");
            assert!(view.verify_checksum(), "FNV-1a checksum must match");

            crate::kernels::simd_lut::fused_kronecker_gemv(
                view.gate_row_seeds,
                view.gate_col_seeds,
                &view.gate_scales[..view.rank],
                d_ff as usize,
                &reg_cb,
                &qlut_in,
                &mut gate_out,
            );
            crate::kernels::simd_lut::fused_kronecker_gemv(
                view.up_row_seeds,
                view.up_col_seeds,
                &view.up_scales[..view.rank],
                d_ff as usize,
                &reg_cb,
                &qlut_in,
                &mut up_out,
            );

            for (g, &u) in gate_out.iter_mut().zip(up_out.iter()) {
                let silu = *g / (1.0 + (-*g).exp());
                *g = silu * u;
            }

            let qlut_mid = QuantizedActivationLut::build(&gate_out);
            crate::kernels::simd_lut::fused_kronecker_gemv(
                view.down_row_seeds,
                view.down_col_seeds,
                &view.down_scales[..view.rank],
                d_model as usize,
                &reg_cb,
                &qlut_mid,
                &mut expert_out,
            );
            assert!(expert_out.iter().all(|v| v.is_finite()));
        }

        assert!((pager.hit_rate_percent() - 100.0).abs() < 1e-6);
        let _ = std::fs::remove_file(&tmp_path);
    }
}
