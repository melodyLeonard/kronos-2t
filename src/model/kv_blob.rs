//! Token-major KV blobs: a backend-neutral byte layout for saving and restoring KV prefixes.

/// Describes one engine's KV slot encoding; blobs only round-trip between equal layouts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KvBlobLayout {
    /// `0` CPU, `1` Metal.
    pub backend: u8,
    /// `4`, `8`, or `32`.
    pub bits: u8,
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub bytes_per_pos: u32,
}

impl KvBlobLayout {
    pub const ENCODED_LEN: usize = 18;

    pub fn encode(&self) -> [u8; Self::ENCODED_LEN] {
        let mut b = [0u8; Self::ENCODED_LEN];
        b[0] = self.backend;
        b[1] = self.bits;
        b[2..6].copy_from_slice(&self.layers.to_le_bytes());
        b[6..10].copy_from_slice(&self.kv_heads.to_le_bytes());
        b[10..14].copy_from_slice(&self.head_dim.to_le_bytes());
        b[14..18].copy_from_slice(&self.bytes_per_pos.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Option<Self> {
        if b.len() < Self::ENCODED_LEN {
            return None;
        }
        let u = |i: usize| u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        Some(Self {
            backend: b[0],
            bits: b[1],
            layers: u(2),
            kv_heads: u(6),
            head_dim: u(10),
            bytes_per_pos: u(14),
        })
    }
}

/// Raw view of a head-major quantized KV cache (`slot = head * cap + pos`, no ring buffer).
pub(crate) struct KvSlotPtrs {
    pub k_data: *mut u8,
    pub v_data: *mut u8,
    pub k_scale: *mut u8,
    pub v_scale: *mut u8,
    pub data_bytes: usize,
    pub scale_bytes: usize,
    pub cap: usize,
    pub heads: usize,
}

impl KvSlotPtrs {
    #[inline]
    pub fn bytes_per_pos(&self) -> usize {
        2 * self.heads * (self.data_bytes + self.scale_bytes)
    }

    /// Appends positions `lo..hi` to `out`.
    ///
    /// # Safety
    /// Pointers must cover `heads * cap` slots and `hi <= cap`.
    pub unsafe fn export(&self, lo: usize, hi: usize, out: &mut Vec<u8>) {
        let (db, sb) = (self.data_bytes, self.scale_bytes);
        out.reserve((hi - lo) * self.bytes_per_pos());
        for pos in lo..hi {
            for h in 0..self.heads {
                let slot = h * self.cap + pos;
                // SAFETY: caller guarantees `slot < heads * cap`.
                unsafe {
                    out.extend_from_slice(std::slice::from_raw_parts(self.k_data.add(slot * db), db));
                    out.extend_from_slice(std::slice::from_raw_parts(self.k_scale.add(slot * sb), sb));
                    out.extend_from_slice(std::slice::from_raw_parts(self.v_data.add(slot * db), db));
                    out.extend_from_slice(std::slice::from_raw_parts(self.v_scale.add(slot * sb), sb));
                }
            }
        }
    }

    /// Writes positions `0..n` from a token-major blob.
    ///
    /// # Safety
    /// Pointers must cover `heads * cap` slots, `n <= cap`, and `data.len() >= n * bytes_per_pos()`.
    pub unsafe fn import(&self, data: &[u8], n: usize) {
        let (db, sb) = (self.data_bytes, self.scale_bytes);
        let mut src = 0usize;
        for pos in 0..n {
            for h in 0..self.heads {
                let slot = h * self.cap + pos;
                // SAFETY: caller guarantees `slot < heads * cap` and `data` holds `n` positions.
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr().add(src), self.k_data.add(slot * db), db);
                    src += db;
                    std::ptr::copy_nonoverlapping(data.as_ptr().add(src), self.k_scale.add(slot * sb), sb);
                    src += sb;
                    std::ptr::copy_nonoverlapping(data.as_ptr().add(src), self.v_data.add(slot * db), db);
                    src += db;
                    std::ptr::copy_nonoverlapping(data.as_ptr().add(src), self.v_scale.add(slot * sb), sb);
                    src += sb;
                }
            }
        }
    }
}
