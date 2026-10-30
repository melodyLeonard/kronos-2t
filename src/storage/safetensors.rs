#![forbid(unsafe_code)]
//! Zero-Copy Pure-Rust SafeTensors Loader (`BF16` / `F16` / `F32`, Zero External Dependencies).
//!
//! Maps `.safetensors` files directly into virtual memory via [`crate::storage::mmap_pager::sys::RawMmap`]
//! and parses the 8-byte little-endian header length + JSON tensor index without allocating raw
//! weight copies until requested.

use crate::storage::mmap_pager::sys::RawMmap;
use crate::kernels::simd_lut::{bf16_to_f32, f32_to_bf16};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::path::Path;

/// Supported floating-point tensor storage formats inside `.safetensors`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    BF16,
    F16,
    F32,
}

impl TensorDtype {
    pub fn byte_size(self) -> usize {
        match self {
            Self::BF16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }
}

/// Metadata entry for a single tensor in the SafeTensors JSON header.
#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub dtype: TensorDtype,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

/// Borrowed zero-copy view of a tensor inside a memory-mapped `.safetensors` file.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub dtype: TensorDtype,
    pub shape: &'a [usize],
    pub raw_bytes: &'a [u8],
}

impl<'a> TensorView<'a> {
    /// Total number of scalar elements across all dimensions.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Decodes a single row `[row, 0..cols]` of a 2D `[rows, cols]` tensor into `dst` (`f32`).
    pub fn copy_row_f32(&self, row: usize, dst: &mut [f32]) {
        assert_eq!(self.shape.len(), 2, "copy_row_f32 expects a 2D tensor");
        let rows = self.shape[0];
        let cols = self.shape[1];
        assert!(row < rows && dst.len() == cols);
        let elem_bytes = self.dtype.byte_size();
        let start = row * cols * elem_bytes;
        let end = start + cols * elem_bytes;
        decode_bytes_to_f32(self.dtype, &self.raw_bytes[start..end], dst);
    }

    /// Decodes the entire tensor into a newly allocated `Vec<f32>`.
    pub fn to_f32_vec(&self) -> Vec<f32> {
        let n = self.numel();
        let mut out = vec![0.0f32; n];
        decode_bytes_to_f32(self.dtype, self.raw_bytes, &mut out);
        out
    }

    /// Decodes slice `outer_idx` along axis 0 (e.g., one expert from `[E, R, C]`) into `Vec<f32>`.
    pub fn slice_outer_to_f32(&self, outer_idx: usize) -> Result<Vec<f32>, String> {
        if self.shape.len() < 2 {
            return Err("slice_outer_to_f32 requires rank >= 2".to_string());
        }
        let outer = self.shape[0];
        if outer_idx >= outer {
            return Err(format!(
                "outer_idx {outer_idx} out of bounds for shape {:?}",
                self.shape
            ));
        }
        let inner_numel: usize = self.shape[1..].iter().product();
        let slice_bytes = inner_numel * self.dtype.byte_size();
        let start = outer_idx * slice_bytes;
        let end = start + slice_bytes;
        let mut out = vec![0.0f32; inner_numel];
        decode_bytes_to_f32(self.dtype, &self.raw_bytes[start..end], &mut out);
        Ok(out)
    }

    /// Decodes the entire tensor directly into `BF16` (`Vec<u16>`) without an intermediate `Vec<f32>`.
    pub fn to_bf16_vec(&self) -> Vec<u16> {
        let n = self.numel();
        let mut out = Vec::with_capacity(n);
        match self.dtype {
            TensorDtype::BF16 => {
                assert_eq!(self.raw_bytes.len(), n * 2);
                for chunk in self.raw_bytes.chunks_exact(2) {
                    out.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
            }
            TensorDtype::F16 => {
                assert_eq!(self.raw_bytes.len(), n * 2);
                for chunk in self.raw_bytes.chunks_exact(2) {
                    let v = f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
                    out.push(f32_to_bf16(v));
                }
            }
            TensorDtype::F32 => {
                assert_eq!(self.raw_bytes.len(), n * 4);
                for chunk in self.raw_bytes.chunks_exact(4) {
                    let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push(f32_to_bf16(v));
                }
            }
        }
        out
    }
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            f32::from_bits(sign)
        } else {
            let val = (mant as f32) * (2.0f32.powi(-24));
            if sign != 0 { -val } else { val }
        }
    } else if exp == 31 {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
    }
}

fn decode_bytes_to_f32(dtype: TensorDtype, src: &[u8], dst: &mut [f32]) {
    match dtype {
        TensorDtype::BF16 => {
            assert_eq!(src.len(), dst.len() * 2);
            for (slot, chunk) in dst.iter_mut().zip(src.chunks_exact(2)) {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                *slot = bf16_to_f32(u);
            }
        }
        TensorDtype::F16 => {
            assert_eq!(src.len(), dst.len() * 2);
            for (slot, chunk) in dst.iter_mut().zip(src.chunks_exact(2)) {
                let u = u16::from_le_bytes([chunk[0], chunk[1]]);
                *slot = f16_to_f32(u);
            }
        }
        TensorDtype::F32 => {
            assert_eq!(src.len(), dst.len() * 4);
            for (slot, chunk) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *slot = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
    }
}

/// Memory-mapped SafeTensors container.
pub struct SafeTensorsFile {
    mmap: RawMmap,
    data_section_offset: usize,
    pub tensors: HashMap<String, TensorMeta>,
}

impl SafeTensorsFile {
    /// Opens and memory-maps a `.safetensors` file, validating all tensor byte offsets.
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors file smaller than 8-byte header length",
            ));
        }

        let mmap = RawMmap::map_read_only(&file, len)?;
        let bytes = mmap.as_slice();
        let hdr_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let data_section_offset = 8usize.saturating_add(hdr_len);
        if data_section_offset > len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "safetensors header length exceeds file size",
            ));
        }

        let hdr_str = std::str::from_utf8(&bytes[8..data_section_offset])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let tensors = parse_safetensors_header(hdr_str, len - data_section_offset)
            .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;

        Ok(Self {
            mmap,
            data_section_offset,
            tensors,
        })
    }

    /// Total mapped file size in bytes.
    pub fn byte_len(&self) -> usize {
        self.mmap.as_slice().len()
    }

    /// Borrows a zero-copy [`TensorView`] by exact tensor name.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, String> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("missing tensor '{name}' in safetensors"))?;
        let abs_start = self.data_section_offset + meta.data_start;
        let abs_end = self.data_section_offset + meta.data_end;
        Ok(TensorView {
            dtype: meta.dtype,
            shape: &meta.shape,
            raw_bytes: &self.mmap.as_slice()[abs_start..abs_end],
        })
    }

    /// Hints the OS kernel (`POSIX_MADV_DONTNEED`) to reclaim interior 16 KiB VM pages of `name`.
    pub fn release_tensor_pages(&self, name: &str) {
        const VM_PAGE: usize = 16_384;
        if let Some(meta) = self.tensors.get(name) {
            let abs_start = self.data_section_offset + meta.data_start;
            let abs_end = self.data_section_offset + meta.data_end;
            let page_start = (abs_start + (VM_PAGE - 1)) & !(VM_PAGE - 1);
            let page_end = abs_end & !(VM_PAGE - 1);
            if page_end > page_start && page_end <= self.byte_len() {
                self.mmap.advise_dontneed(page_start, page_end - page_start);
            }
        }
    }

    /// Reclaims interior 16 KiB VM pages of axis-0 slice `outer_idx` within tensor `name`.
    pub fn release_tensor_outer_slice_pages(&self, name: &str, outer_idx: usize) {
        const VM_PAGE: usize = 16_384;
        if let Some(meta) = self.tensors.get(name) {
            if meta.shape.len() < 2 || outer_idx >= meta.shape[0] {
                return;
            }
            let inner_numel: usize = meta.shape[1..].iter().product();
            let slice_bytes = inner_numel * meta.dtype.byte_size();
            let abs_start = self.data_section_offset + meta.data_start + outer_idx * slice_bytes;
            let abs_end = abs_start + slice_bytes;
            let page_start = (abs_start + (VM_PAGE - 1)) & !(VM_PAGE - 1);
            let page_end = abs_end & !(VM_PAGE - 1);
            if page_end > page_start && page_end <= self.byte_len() {
                self.mmap.advise_dontneed(page_start, page_end - page_start);
            }
        }
    }
}

/// Unified single-file (`model.safetensors`) or multi-shard (`model.safetensors.index.json`) store.
pub struct SafeTensorsStore {
    shards: Vec<SafeTensorsFile>,
    weight_map: HashMap<String, usize>,
}

impl SafeTensorsStore {
    /// Opens `model_dir/model.safetensors` if present, otherwise `model_dir/model.safetensors.index.json`.
    pub fn open_dir(model_dir: &Path) -> io::Result<Self> {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            let st = SafeTensorsFile::open(&single)?;
            let weight_map = st.tensors.keys().map(|k| (k.clone(), 0usize)).collect();
            return Ok(Self {
                shards: vec![st],
                weight_map,
            });
        }

        let index_path = model_dir.join("model.safetensors.index.json");
        let text = fs::read_to_string(&index_path)?;
        let entries = parse_safetensors_index(&text)
            .map_err(|m| io::Error::new(io::ErrorKind::InvalidData, m))?;

        let mut shard_names: Vec<String> = Vec::new();
        let mut name_to_idx: HashMap<String, usize> = HashMap::new();
        let mut weight_map: HashMap<String, usize> = HashMap::with_capacity(entries.len());

        for (tensor_name, shard_file) in entries {
            let idx = if let Some(&i) = name_to_idx.get(&shard_file) {
                i
            } else {
                let i = shard_names.len();
                shard_names.push(shard_file.clone());
                name_to_idx.insert(shard_file, i);
                i
            };
            weight_map.insert(tensor_name, idx);
        }

        let mut shards = Vec::with_capacity(shard_names.len());
        for shard_file in &shard_names {
            shards.push(SafeTensorsFile::open(&model_dir.join(shard_file))?);
        }

        Ok(Self { shards, weight_map })
    }

    #[inline]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, String> {
        let &shard_idx = self
            .weight_map
            .get(name)
            .ok_or_else(|| format!("missing tensor '{name}' in safetensors store"))?;
        self.shards[shard_idx].tensor(name)
    }

    pub fn release_tensor_pages(&self, name: &str) {
        if let Some(&shard_idx) = self.weight_map.get(name) {
            self.shards[shard_idx].release_tensor_pages(name);
        }
    }

    pub fn release_tensor_outer_slice_pages(&self, name: &str, outer_idx: usize) {
        if let Some(&shard_idx) = self.weight_map.get(name) {
            self.shards[shard_idx].release_tensor_outer_slice_pages(name, outer_idx);
        }
    }
}

fn parse_safetensors_index(index_json: &str) -> Result<Vec<(String, String)>, String> {
    let mut cur = JsonCursor::new(index_json);
    cur.expect_byte(b'{')?;
    let mut out = Vec::new();
    if cur.peek() == Some(b'}') {
        cur.expect_byte(b'}')?;
        return Ok(out);
    }
    loop {
        let key = cur.parse_string()?;
        cur.expect_byte(b':')?;
        if key == "weight_map" {
            cur.expect_byte(b'{')?;
            if cur.peek() != Some(b'}') {
                loop {
                    let tensor = cur.parse_string()?;
                    cur.expect_byte(b':')?;
                    let shard = cur.parse_string()?;
                    out.push((tensor, shard));
                    if cur.peek() == Some(b',') {
                        cur.expect_byte(b',')?;
                    } else {
                        break;
                    }
                }
            }
            cur.expect_byte(b'}')?;
        } else {
            cur.skip_value()?;
        }
        if cur.peek() == Some(b',') {
            cur.expect_byte(b',')?;
        } else {
            break;
        }
    }
    cur.expect_byte(b'}')?;
    if out.is_empty() {
        return Err("model.safetensors.index.json contained empty weight_map".to_string());
    }
    Ok(out)
}

/// Minimal zero-dependency JSON cursor shared by `safetensors` and `bpe_tokenizer`.
pub(crate) struct JsonCursor<'a> {
    s: &'a [u8],
    pos: usize,
}

impl<'a> JsonCursor<'a> {
    pub fn new(text: &'a str) -> Self {
        Self {
            s: text.as_bytes(),
            pos: 0,
        }
    }

    pub fn skip_ws(&mut self) {
        while self.pos < self.s.len() && matches!(self.s[self.pos], b' ' | b'\t' | b'\r' | b'\n') {
            self.pos += 1;
        }
    }

    pub fn peek(&mut self) -> Option<u8> {
        self.skip_ws();
        self.s.get(self.pos).copied()
    }

    pub fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
        self.skip_ws();
        match self.s.get(self.pos) {
            Some(&b) if b == expected => {
                self.pos += 1;
                Ok(())
            }
            other => Err(format!(
                "expected '{}' at byte {}, got {:?}",
                expected as char, self.pos, other
            )),
        }
    }

    pub fn parse_string(&mut self) -> Result<String, String> {
        self.expect_byte(b'"')?;
        let mut out = String::new();
        while self.pos < self.s.len() {
            let b = self.s[self.pos];
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            } else if b == b'\\' {
                self.pos += 1;
                let esc = *self
                    .s
                    .get(self.pos)
                    .ok_or_else(|| "unterminated escape in JSON string".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000C}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let code = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&code) {
                            if self.s.get(self.pos..self.pos + 2) == Some(b"\\u") {
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                let combined = 0x10000
                                    + (((code as u32 - 0xD800) << 10) | (low as u32 - 0xDC00));
                                let ch = char::from_u32(combined)
                                    .ok_or_else(|| "invalid UTF-16 surrogate pair".to_string())?;
                                out.push(ch);
                            } else {
                                return Err("unpaired high surrogate in JSON".to_string());
                            }
                        } else {
                            let ch = char::from_u32(code as u32)
                                .ok_or_else(|| "invalid unicode scalar in JSON".to_string())?;
                            out.push(ch);
                        }
                    }
                    _ => return Err(format!("invalid JSON escape '\\{}'", esc as char)),
                }
            } else {
                // Consume valid UTF-8 chunk up to next '"' or '\\'
                let start = self.pos;
                while self.pos < self.s.len()
                    && self.s[self.pos] != b'"'
                    && self.s[self.pos] != b'\\'
                {
                    self.pos += 1;
                }
                let sub = std::str::from_utf8(&self.s[start..self.pos])
                    .map_err(|e| format!("invalid UTF-8 in JSON string: {e}"))?;
                out.push_str(sub);
            }
        }
        Err("unterminated JSON string".to_string())
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let slice = self
            .s
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| "truncated \\uXXXX escape".to_string())?;
        let hex_str = std::str::from_utf8(slice).map_err(|e| e.to_string())?;
        let val = u16::from_str_radix(hex_str, 16)
            .map_err(|e| format!("invalid hex in \\u escape: {e}"))?;
        self.pos += 4;
        Ok(val)
    }

    pub fn parse_u64(&mut self) -> Result<u64, String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.s.len() && self.s[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!("expected unsigned integer at byte {}", start));
        }
        std::str::from_utf8(&self.s[start..self.pos])
            .unwrap()
            .parse::<u64>()
            .map_err(|e| e.to_string())
    }

    pub fn parse_f64(&mut self) -> Result<f64, String> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.s.len()
            && matches!(
                self.s[self.pos],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.pos += 1;
        }
        if self.pos == start {
            return Err(format!("expected number at byte {}", start));
        }
        std::str::from_utf8(&self.s[start..self.pos])
            .unwrap()
            .parse::<f64>()
            .map_err(|e| e.to_string())
    }

    pub fn skip_value(&mut self) -> Result<(), String> {
        let b = self
            .peek()
            .ok_or_else(|| "unexpected EOF skipping JSON value".to_string())?;
        match b {
            b'"' => {
                let _ = self.parse_string()?;
                Ok(())
            }
            b'{' => {
                self.expect_byte(b'{')?;
                if self.peek() == Some(b'}') {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    let _ = self.parse_string()?;
                    self.expect_byte(b':')?;
                    self.skip_value()?;
                    if self.peek() == Some(b',') {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                self.expect_byte(b'}')
            }
            b'[' => {
                self.expect_byte(b'[')?;
                if self.peek() == Some(b']') {
                    self.pos += 1;
                    return Ok(());
                }
                loop {
                    self.skip_value()?;
                    if self.peek() == Some(b',') {
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                self.expect_byte(b']')
            }
            b't' => self.consume_literal(b"true"),
            b'f' => self.consume_literal(b"false"),
            b'n' => self.consume_literal(b"null"),
            _ => {
                let _ = self.parse_f64()?;
                Ok(())
            }
        }
    }

    pub fn consume_literal(&mut self, lit: &[u8]) -> Result<(), String> {
        self.skip_ws();
        if self.s.get(self.pos..self.pos + lit.len()) == Some(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err(format!("expected literal at byte {}", self.pos))
        }
    }
}

fn parse_safetensors_header(
    hdr_json: &str,
    data_section_len: usize,
) -> Result<HashMap<String, TensorMeta>, String> {
    let mut cur = JsonCursor::new(hdr_json);
    cur.expect_byte(b'{')?;
    let mut map = HashMap::new();

    if cur.peek() == Some(b'}') {
        cur.expect_byte(b'}')?;
        return Ok(map);
    }

    loop {
        let key = cur.parse_string()?;
        cur.expect_byte(b':')?;
        if key == "__metadata__" {
            cur.skip_value()?;
        } else {
            let meta = parse_tensor_meta(&mut cur, data_section_len, &key)?;
            map.insert(key, meta);
        }

        if cur.peek() == Some(b',') {
            cur.expect_byte(b',')?;
        } else {
            break;
        }
    }

    cur.expect_byte(b'}')?;
    Ok(map)
}

fn parse_tensor_meta(
    cur: &mut JsonCursor<'_>,
    data_section_len: usize,
    tensor_name: &str,
) -> Result<TensorMeta, String> {
    cur.expect_byte(b'{')?;
    let mut dtype: Option<TensorDtype> = None;
    let mut shape: Option<Vec<usize>> = None;
    let mut offsets: Option<(usize, usize)> = None;

    loop {
        let field = cur.parse_string()?;
        cur.expect_byte(b':')?;
        match field.as_str() {
            "dtype" => {
                let dt = cur.parse_string()?;
                dtype = Some(match dt.as_str() {
                    "BF16" => TensorDtype::BF16,
                    "F16" => TensorDtype::F16,
                    "F32" => TensorDtype::F32,
                    other => {
                        return Err(format!("unsupported dtype '{other}' on '{tensor_name}'"));
                    }
                });
            }
            "shape" => {
                cur.expect_byte(b'[')?;
                let mut dims = Vec::new();
                if cur.peek() != Some(b']') {
                    loop {
                        dims.push(cur.parse_u64()? as usize);
                        if cur.peek() == Some(b',') {
                            cur.expect_byte(b',')?;
                        } else {
                            break;
                        }
                    }
                }
                cur.expect_byte(b']')?;
                shape = Some(dims);
            }
            "data_offsets" => {
                cur.expect_byte(b'[')?;
                let s = cur.parse_u64()? as usize;
                cur.expect_byte(b',')?;
                let e = cur.parse_u64()? as usize;
                cur.expect_byte(b']')?;
                offsets = Some((s, e));
            }
            _ => cur.skip_value()?,
        }

        if cur.peek() == Some(b',') {
            cur.expect_byte(b',')?;
        } else {
            break;
        }
    }
    cur.expect_byte(b'}')?;

    let dtype = dtype.ok_or_else(|| format!("missing dtype for '{tensor_name}'"))?;
    let shape = shape.ok_or_else(|| format!("missing shape for '{tensor_name}'"))?;
    let (data_start, data_end) =
        offsets.ok_or_else(|| format!("missing data_offsets for '{tensor_name}'"))?;

    let expected_bytes = shape.iter().product::<usize>() * dtype.byte_size();
    if data_end < data_start || (data_end - data_start) != expected_bytes {
        return Err(format!(
            "tensor '{tensor_name}' byte span {}..{} != expected {}",
            data_start, data_end, expected_bytes
        ));
    }
    if data_end > data_section_len {
        return Err(format!(
            "tensor '{tensor_name}' offset {data_end} exceeds data section {data_section_len}"
        ));
    }

    Ok(TensorMeta {
        dtype,
        shape,
        data_start,
        data_end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::simd_lut::f32_to_bf16;
    use std::io::Write;

    #[test]
    fn test_safetensors_bf16_roundtrip_and_row_view() {
        let tmp = std::env::temp_dir().join("kronos_test_mini.safetensors");
        let vals = [1.5f32, -2.25, 0.0, 42.0, -0.125, 8.0];
        let mut raw = Vec::new();
        for &v in &vals {
            raw.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
        }
        let hdr = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut f = File::create(&tmp).unwrap();
        f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
        f.write_all(hdr.as_bytes()).unwrap();
        f.write_all(&raw).unwrap();
        drop(f);

        let st = SafeTensorsFile::open(&tmp).unwrap();
        let tv = st.tensor("w").unwrap();
        assert_eq!(tv.shape, &[2, 3]);
        let all = tv.to_f32_vec();
        for (a, b) in all.iter().zip(vals.iter()) {
            assert!((a - b).abs() < 1e-3);
        }
        let mut row1 = [0.0f32; 3];
        tv.copy_row_f32(1, &mut row1);
        assert_eq!(&row1, &all[3..6]);
        let bf = tv.to_bf16_vec();
        assert_eq!(bf.len(), 6);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_sharded_safetensors_store() {
        let dir = std::env::temp_dir().join("kronos_test_sharded_st");
        let _ = fs::create_dir_all(&dir);
        let write_shard = |file_name: &str, tensor_name: &str, val: f32| {
            let raw = f32_to_bf16(val).to_le_bytes();
            let hdr = format!(
                r#"{{"{tensor_name}":{{"dtype":"BF16","shape":[1,1],"data_offsets":[0,2]}}}}"#
            );
            let mut f = File::create(dir.join(file_name)).unwrap();
            f.write_all(&(hdr.len() as u64).to_le_bytes()).unwrap();
            f.write_all(hdr.as_bytes()).unwrap();
            f.write_all(&raw).unwrap();
        };
        write_shard("model-00001-of-00002.safetensors", "a.weight", 3.5);
        write_shard("model-00002-of-00002.safetensors", "b.bias", -1.25);
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata":{"total_size":4},"weight_map":{"a.weight":"model-00001-of-00002.safetensors","b.bias":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let store = SafeTensorsStore::open_dir(&dir).unwrap();
        assert!(store.has_tensor("a.weight"));
        assert!(store.has_tensor("b.bias"));
        assert!(!store.has_tensor("c.bias"));
        assert_eq!(store.tensor("a.weight").unwrap().to_f32_vec(), vec![3.5]);
        assert_eq!(store.tensor("b.bias").unwrap().to_f32_vec(), vec![-1.25]);
        let _ = fs::remove_dir_all(&dir);
    }
}
