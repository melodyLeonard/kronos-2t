#![forbid(unsafe_code)]
//! Zero-Copy Pure-Rust SafeTensors Loader (`BF16` / `F16` / `F32`, Zero External Dependencies).
//!
//! Maps `.safetensors` files directly into virtual memory via [`crate::mmap_pager::sys::RawMmap`]
//! and parses the 8-byte little-endian header length + JSON tensor index without allocating raw
//! weight copies until requested.

use crate::mmap_pager::sys::RawMmap;
use crate::simd_lut::bf16_to_f32;
use std::collections::HashMap;
use std::fs::File;
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
    use crate::simd_lut::f32_to_bf16;
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
        let _ = std::fs::remove_file(&tmp);
    }
}
