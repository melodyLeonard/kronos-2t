#![forbid(unsafe_code)]
//! 4KB Page-Aligned `.k2t` Binary Model Container Format (Multi-Rank Kronecker Support).
//!
//! Aligns every SwiGLU Kronecker micro-expert (`W_gate`, `W_up`, `W_down` across `R` Kronecker ranks)
//! to an exact `4096`-byte OS / NVMe / UFS 4.0 page boundary:
//! - **Page 0 (`0..4096`)**: `K2tHeader` metadata, base-3 / 256-entry / 64-entry NEON register
//!   codebooks, and FNV-1a 64-bit integrity checksum.
//! - **Page `1 + layer * experts_per_layer + slot`**: Self-contained 4KB multi-rank SwiGLU seed page.

use crate::kernels::lut_gemv::GROUP_SIZE;
use crate::kernels::simd_lut::{default_register_codebook_64, default_simd_codebook};
use crate::kernels::synthesis::default_canonical_codebook;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Standard 4KB OS & Flash page size (`4096` bytes).
pub const K2T_PAGE_SIZE: usize = 4096;

/// Magic header bytes identifying a `.k2t` container.
pub const K2T_MAGIC: [u8; 4] = *b"K2T\x02";

/// Format version `2` (multi-rank register-codebook `.k2t`).
pub const K2T_VERSION: u32 = 2;

/// Byte size of the per-expert metadata prefix inside each 4KB page.
pub const EXPERT_PAGE_HEADER_BYTES: usize = 32;

/// Maximum Kronecker rank supported in fixed stack scale arrays without heap allocation.
pub const MAX_KRONECKER_RANK: usize = 8;

/// Parsed Page-0 header of a `.k2t` model container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct K2tHeader {
    pub version: u32,
    pub num_layers: u32,
    pub experts_per_layer: u32,
    pub d_model: u32,
    pub d_ff: u32,
    pub kronecker_rank: u32,
    pub total_pages: u64,
    pub base3_codebook: [u8; 256],
    pub simd_codebook: [u8; 256],
    pub reg_codebook_64: [u8; 64],
    pub header_checksum: u64,
}

impl K2tHeader {
    /// Computes exact byte size (scales + seeds) per SwiGLU micro-expert page for `rank`.
    pub fn seed_payload_bytes_ranked(d_model: usize, d_ff: usize, rank: usize) -> usize {
        assert!(d_model > 0 && d_model.is_multiple_of(GROUP_SIZE));
        assert!(d_ff > 0 && d_ff.is_multiple_of(GROUP_SIZE));
        assert!((1..=MAX_KRONECKER_RANK).contains(&rank));
        let scales_bytes = 3 * rank * 4; // gate_scales, up_scales, down_scales (f32 LE)
        let gate_seeds = rank * (d_ff + (d_model / GROUP_SIZE));
        let up_seeds = rank * (d_ff + (d_model / GROUP_SIZE));
        let down_seeds = rank * (d_model + (d_ff / GROUP_SIZE));
        scales_bytes + gate_seeds + up_seeds + down_seeds
    }

    /// Rank-1 seed byte count (excluding scales/header) for single-rank sizing reporting.
    pub fn seed_payload_bytes(d_model: usize, d_ff: usize) -> usize {
        let gate_bytes = d_ff + (d_model / GROUP_SIZE);
        let up_bytes = d_ff + (d_model / GROUP_SIZE);
        let down_bytes = d_model + (d_ff / GROUP_SIZE);
        gate_bytes + up_bytes + down_bytes
    }

    #[inline]
    pub fn logical_params_per_expert(&self) -> usize {
        3 * (self.d_model as usize) * (self.d_ff as usize)
    }

    #[inline]
    pub fn expert_page_index(&self, layer_idx: usize, virtual_expert_id: usize) -> usize {
        assert!(layer_idx < self.num_layers as usize);
        let epl = self.experts_per_layer as usize;
        let slot = virtual_expert_id % epl;
        1 + layer_idx * epl + slot
    }

    pub fn encode_page(&self) -> [u8; K2T_PAGE_SIZE] {
        let mut page = [0u8; K2T_PAGE_SIZE];
        page[0..4].copy_from_slice(&K2T_MAGIC);
        page[4..8].copy_from_slice(&self.version.to_le_bytes());
        page[8..12].copy_from_slice(&self.num_layers.to_le_bytes());
        page[12..16].copy_from_slice(&self.experts_per_layer.to_le_bytes());
        page[16..20].copy_from_slice(&self.d_model.to_le_bytes());
        page[20..24].copy_from_slice(&self.d_ff.to_le_bytes());
        page[24..28].copy_from_slice(&self.kronecker_rank.to_le_bytes());
        page[28..36].copy_from_slice(&self.total_pages.to_le_bytes());
        page[36..292].copy_from_slice(&self.base3_codebook);
        page[292..548].copy_from_slice(&self.simd_codebook);
        page[548..612].copy_from_slice(&self.reg_codebook_64);
        let checksum = fnv1a_64(&page[0..612]);
        page[612..620].copy_from_slice(&checksum.to_le_bytes());
        page
    }

    pub fn decode_page(page: &[u8]) -> Result<Self, &'static str> {
        if page.len() < K2T_PAGE_SIZE {
            return Err("k2t file smaller than 4KB header page");
        }
        if page[0..4] != K2T_MAGIC {
            return Err("invalid k2t magic bytes");
        }
        let version = u32::from_le_bytes(page[4..8].try_into().unwrap());
        if version != K2T_VERSION {
            return Err("unsupported k2t format version");
        }
        let num_layers = u32::from_le_bytes(page[8..12].try_into().unwrap());
        let experts_per_layer = u32::from_le_bytes(page[12..16].try_into().unwrap());
        let d_model = u32::from_le_bytes(page[16..20].try_into().unwrap());
        let d_ff = u32::from_le_bytes(page[20..24].try_into().unwrap());
        let kronecker_rank = u32::from_le_bytes(page[24..28].try_into().unwrap());
        let total_pages = u64::from_le_bytes(page[28..36].try_into().unwrap());

        if num_layers == 0
            || experts_per_layer == 0
            || d_model == 0
            || d_ff == 0
            || !(1..=MAX_KRONECKER_RANK as u32).contains(&kronecker_rank)
        {
            return Err("invalid k2t header dimensions or rank");
        }
        if !(d_model as usize).is_multiple_of(GROUP_SIZE)
            || !(d_ff as usize).is_multiple_of(GROUP_SIZE)
        {
            return Err("k2t dimensions must be multiples of 4");
        }
        let payload = Self::seed_payload_bytes_ranked(
            d_model as usize,
            d_ff as usize,
            kronecker_rank as usize,
        );
        if EXPERT_PAGE_HEADER_BYTES + payload > K2T_PAGE_SIZE {
            return Err("expert seed payload exceeds 4KB page boundary");
        }

        let expected_pages = 1u64 + (num_layers as u64) * (experts_per_layer as u64);
        if total_pages != expected_pages {
            return Err("k2t total_pages mismatch");
        }

        let mut base3_codebook = [0u8; 256];
        base3_codebook.copy_from_slice(&page[36..292]);
        let mut simd_codebook = [0u8; 256];
        simd_codebook.copy_from_slice(&page[292..548]);
        let mut reg_codebook_64 = [0u8; 64];
        reg_codebook_64.copy_from_slice(&page[548..612]);

        let stored_checksum = u64::from_le_bytes(page[612..620].try_into().unwrap());
        let actual_checksum = fnv1a_64(&page[0..612]);
        if stored_checksum != actual_checksum {
            return Err("k2t header FNV-1a checksum mismatch");
        }

        Ok(Self {
            version,
            num_layers,
            experts_per_layer,
            d_model,
            d_ff,
            kronecker_rank,
            total_pages,
            base3_codebook,
            simd_codebook,
            reg_codebook_64,
            header_checksum: stored_checksum,
        })
    }
}

/// Owned multi-rank SwiGLU expert definition produced by the Kronecker weight compiler.
#[derive(Clone, Debug)]
pub struct CompiledSwiGluExpert {
    pub layer_id: u32,
    pub expert_id: u64,
    pub d_model: usize,
    pub d_ff: usize,
    pub rank: usize,
    pub output_scale: f32,
    pub gate_scales: Vec<f32>,
    pub up_scales: Vec<f32>,
    pub down_scales: Vec<f32>,
    pub gate_row_seeds: Vec<u8>,
    pub gate_col_seeds: Vec<u8>,
    pub up_row_seeds: Vec<u8>,
    pub up_col_seeds: Vec<u8>,
    pub down_row_seeds: Vec<u8>,
    pub down_col_seeds: Vec<u8>,
}

impl CompiledSwiGluExpert {
    /// Encodes `self` into an exact `4096`-byte page with FNV-1a checksum.
    pub fn encode_page(&self) -> [u8; K2T_PAGE_SIZE] {
        let payload_len =
            K2tHeader::seed_payload_bytes_ranked(self.d_model, self.d_ff, self.rank);
        assert!(EXPERT_PAGE_HEADER_BYTES + payload_len <= K2T_PAGE_SIZE);

        let mut page = [0u8; K2T_PAGE_SIZE];
        page[0..8].copy_from_slice(&self.expert_id.to_le_bytes());
        page[8..12].copy_from_slice(&self.layer_id.to_le_bytes());
        page[12..16].copy_from_slice(&(self.d_model as u32).to_le_bytes());
        page[16..20].copy_from_slice(&(self.d_ff as u32).to_le_bytes());
        page[20..22].copy_from_slice(&(self.rank as u16).to_le_bytes());
        // bytes 22..24 reserved; we store output_scale folded or at offset 20..24?
        // Let's pack d_model (u16), d_ff (u16), rank (u16), reserved (u16) cleanly!
        page[12..14].copy_from_slice(&(self.d_model as u16).to_le_bytes());
        page[14..16].copy_from_slice(&(self.d_ff as u16).to_le_bytes());
        page[16..18].copy_from_slice(&(self.rank as u16).to_le_bytes());
        page[18..20].copy_from_slice(&0u16.to_le_bytes());
        page[20..24].copy_from_slice(&self.output_scale.to_le_bytes());

        let mut cursor = EXPERT_PAGE_HEADER_BYTES;
        for scales in [&self.gate_scales, &self.up_scales, &self.down_scales] {
            assert_eq!(scales.len(), self.rank);
            for &s in scales {
                page[cursor..cursor + 4].copy_from_slice(&s.to_le_bytes());
                cursor += 4;
            }
        }

        for slice in [
            self.gate_row_seeds.as_slice(),
            self.gate_col_seeds.as_slice(),
            self.up_row_seeds.as_slice(),
            self.up_col_seeds.as_slice(),
            self.down_row_seeds.as_slice(),
            self.down_col_seeds.as_slice(),
        ] {
            page[cursor..cursor + slice.len()].copy_from_slice(slice);
            cursor += slice.len();
        }
        assert_eq!(cursor, EXPERT_PAGE_HEADER_BYTES + payload_len);

        let checksum = fnv1a_64(&page[EXPERT_PAGE_HEADER_BYTES..cursor]);
        page[24..32].copy_from_slice(&checksum.to_le_bytes());
        page
    }
}

/// Zero-copy view into a single 4KB `.k2t` multi-rank micro-expert page.
/// Uses stack-allocated scale arrays (`[f32; MAX_KRONECKER_RANK]`) so decoding performs **zero heap allocations**.
#[derive(Clone, Copy, Debug)]
pub struct K2tExpertPageView<'a> {
    pub expert_id: u64,
    pub layer_id: u32,
    pub d_model: usize,
    pub d_ff: usize,
    pub rank: usize,
    pub output_scale: f32,
    pub payload_checksum: u64,
    pub gate_scales: [f32; MAX_KRONECKER_RANK],
    pub up_scales: [f32; MAX_KRONECKER_RANK],
    pub down_scales: [f32; MAX_KRONECKER_RANK],
    pub raw_payload: &'a [u8],
    pub gate_row_seeds: &'a [u8],
    pub gate_col_seeds: &'a [u8],
    pub up_row_seeds: &'a [u8],
    pub up_col_seeds: &'a [u8],
    pub down_row_seeds: &'a [u8],
    pub down_col_seeds: &'a [u8],
}

impl<'a> K2tExpertPageView<'a> {
    #[inline]
    pub fn from_page_slice(page: &'a [u8]) -> Result<Self, &'static str> {
        if page.len() != K2T_PAGE_SIZE {
            return Err("expert slice must be exactly 4096 bytes");
        }

        let expert_id = u64::from_le_bytes(page[0..8].try_into().unwrap());
        let layer_id = u32::from_le_bytes(page[8..12].try_into().unwrap());
        let d_model = u16::from_le_bytes(page[12..14].try_into().unwrap()) as usize;
        let d_ff = u16::from_le_bytes(page[14..16].try_into().unwrap()) as usize;
        let rank = u16::from_le_bytes(page[16..18].try_into().unwrap()) as usize;
        let output_scale = f32::from_le_bytes(page[20..24].try_into().unwrap());
        let payload_checksum = u64::from_le_bytes(page[24..32].try_into().unwrap());

        if d_model == 0
            || d_ff == 0
            || !(1..=MAX_KRONECKER_RANK).contains(&rank)
            || !d_model.is_multiple_of(GROUP_SIZE)
            || !d_ff.is_multiple_of(GROUP_SIZE)
        {
            return Err("invalid expert page dimensions or rank");
        }

        let payload_len = K2tHeader::seed_payload_bytes_ranked(d_model, d_ff, rank);
        if EXPERT_PAGE_HEADER_BYTES + payload_len > K2T_PAGE_SIZE {
            return Err("expert page payload exceeds 4096 bytes");
        }

        let raw_payload = &page[EXPERT_PAGE_HEADER_BYTES..EXPERT_PAGE_HEADER_BYTES + payload_len];
        let mut cursor = EXPERT_PAGE_HEADER_BYTES;

        let mut gate_scales = [0.0f32; MAX_KRONECKER_RANK];
        for slot in gate_scales.iter_mut().take(rank) {
            *slot = f32::from_le_bytes(page[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
        }
        let mut up_scales = [0.0f32; MAX_KRONECKER_RANK];
        for slot in up_scales.iter_mut().take(rank) {
            *slot = f32::from_le_bytes(page[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
        }
        let mut down_scales = [0.0f32; MAX_KRONECKER_RANK];
        for slot in down_scales.iter_mut().take(rank) {
            *slot = f32::from_le_bytes(page[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
        }

        let dm_blocks = d_model / GROUP_SIZE;
        let dff_blocks = d_ff / GROUP_SIZE;

        let gate_r_len = rank * d_ff;
        let gate_c_len = rank * dm_blocks;
        let down_r_len = rank * d_model;
        let down_c_len = rank * dff_blocks;

        let gate_row_seeds = &page[cursor..cursor + gate_r_len];
        cursor += gate_r_len;
        let gate_col_seeds = &page[cursor..cursor + gate_c_len];
        cursor += gate_c_len;

        let up_row_seeds = &page[cursor..cursor + gate_r_len];
        cursor += gate_r_len;
        let up_col_seeds = &page[cursor..cursor + gate_c_len];
        cursor += gate_c_len;

        let down_row_seeds = &page[cursor..cursor + down_r_len];
        cursor += down_r_len;
        let down_col_seeds = &page[cursor..cursor + down_c_len];

        Ok(Self {
            expert_id,
            layer_id,
            d_model,
            d_ff,
            rank,
            output_scale,
            payload_checksum,
            gate_scales,
            up_scales,
            down_scales,
            raw_payload,
            gate_row_seeds,
            gate_col_seeds,
            up_row_seeds,
            up_col_seeds,
            down_row_seeds,
            down_col_seeds,
        })
    }

    pub fn verify_checksum(&self) -> bool {
        fnv1a_64(self.raw_payload) == self.payload_checksum
    }
}

/// Encodes a deterministic rank-1 SwiGLU micro-expert into a `4096`-byte page buffer.
pub fn encode_expert_page(
    layer_id: u32,
    expert_id: u64,
    d_model: u32,
    d_ff: u32,
    output_scale: f32,
) -> [u8; K2T_PAGE_SIZE] {
    encode_ranked_expert_page(layer_id, expert_id, d_model, d_ff, 1, output_scale)
}

/// Encodes a deterministic multi-rank SwiGLU micro-expert into a `4096`-byte page buffer.
pub fn encode_ranked_expert_page(
    layer_id: u32,
    expert_id: u64,
    d_model: u32,
    d_ff: u32,
    rank: usize,
    output_scale: f32,
) -> [u8; K2T_PAGE_SIZE] {
    let dm = d_model as usize;
    let dff = d_ff as usize;
    let dm_blocks = dm / GROUP_SIZE;
    let dff_blocks = dff / GROUP_SIZE;

    let mut state = expert_id
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((layer_id as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
    let mut gen_bytes = |len: usize| -> Vec<u8> {
        (0..len)
            .map(|_| {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                ((z ^ (z >> 31)) & 0xFF) as u8
            })
            .collect()
    };

    let scales: Vec<f32> = (0..rank).map(|r| 1.0 / ((r + 1) as f32)).collect();
    let compiled = CompiledSwiGluExpert {
        layer_id,
        expert_id,
        d_model: dm,
        d_ff: dff,
        rank,
        output_scale,
        gate_scales: scales.clone(),
        up_scales: scales.clone(),
        down_scales: scales,
        gate_row_seeds: gen_bytes(rank * dff),
        gate_col_seeds: gen_bytes(rank * dm_blocks),
        up_row_seeds: gen_bytes(rank * dff),
        up_col_seeds: gen_bytes(rank * dm_blocks),
        down_row_seeds: gen_bytes(rank * dm),
        down_col_seeds: gen_bytes(rank * dff_blocks),
    };
    compiled.encode_page()
}

/// Writes a synthetic 4KB page-aligned `.k2t` model container (rank-1) to `path`.
pub fn write_k2t_container(
    path: &Path,
    num_layers: u32,
    experts_per_layer: u32,
    d_model: u32,
    d_ff: u32,
    output_scale: f32,
) -> io::Result<K2tHeader> {
    write_ranked_k2t_container(
        path,
        num_layers,
        experts_per_layer,
        d_model,
        d_ff,
        1,
        output_scale,
    )
}

/// Writes a synthetic multi-rank 4KB page-aligned `.k2t` model container to `path`.
pub fn write_ranked_k2t_container(
    path: &Path,
    num_layers: u32,
    experts_per_layer: u32,
    d_model: u32,
    d_ff: u32,
    kronecker_rank: u32,
    output_scale: f32,
) -> io::Result<K2tHeader> {
    let total_pages = 1u64 + (num_layers as u64) * (experts_per_layer as u64);
    let mut header = K2tHeader {
        version: K2T_VERSION,
        num_layers,
        experts_per_layer,
        d_model,
        d_ff,
        kronecker_rank,
        total_pages,
        base3_codebook: default_canonical_codebook(),
        simd_codebook: default_simd_codebook(),
        reg_codebook_64: default_register_codebook_64(),
        header_checksum: 0,
    };
    let page0 = header.encode_page();
    header.header_checksum = u64::from_le_bytes(page0[612..620].try_into().unwrap());

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * K2T_PAGE_SIZE, file);
    writer.write_all(&page0)?;

    for layer in 0..num_layers {
        for slot in 0..experts_per_layer {
            let page = encode_ranked_expert_page(
                layer,
                slot as u64,
                d_model,
                d_ff,
                kronecker_rank as usize,
                output_scale,
            );
            writer.write_all(&page)?;
        }
    }
    writer.flush()?;
    Ok(header)
}

/// Writes a `.k2t` container populated from **compiled learned SwiGLU experts** (`[num_layers * experts_per_layer]`).
pub fn write_compiled_k2t_container(
    path: &Path,
    num_layers: u32,
    experts_per_layer: u32,
    d_model: u32,
    d_ff: u32,
    kronecker_rank: u32,
    experts: &[CompiledSwiGluExpert],
) -> io::Result<K2tHeader> {
    let expected = (num_layers as usize) * (experts_per_layer as usize);
    assert_eq!(experts.len(), expected);
    let total_pages = 1u64 + (expected as u64);

    let mut header = K2tHeader {
        version: K2T_VERSION,
        num_layers,
        experts_per_layer,
        d_model,
        d_ff,
        kronecker_rank,
        total_pages,
        base3_codebook: default_canonical_codebook(),
        simd_codebook: default_simd_codebook(),
        reg_codebook_64: default_register_codebook_64(),
        header_checksum: 0,
    };
    let page0 = header.encode_page();
    header.header_checksum = u64::from_le_bytes(page0[612..620].try_into().unwrap());

    let file = File::create(path)?;
    let mut writer = BufWriter::with_capacity(64 * K2T_PAGE_SIZE, file);
    writer.write_all(&page0)?;
    for exp in experts {
        writer.write_all(&exp.encode_page())?;
    }
    writer.flush()?;
    Ok(header)
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}
