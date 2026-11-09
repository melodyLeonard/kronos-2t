//! Minimal GGUF v2/v3 reader (header, metadata, tensor table) per ggml's `gguf.md`.

use std::collections::BTreeMap;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";

/// Tencent `block_q2_0c` (not in upstream ggml): 512 weights, f16 scale, codes `{-3,-1,1,3}`.
pub const GGML_TYPE_Q2_0C: u32 = 40;
pub const Q2_0C_BLOCK: usize = 512;
pub const Q2_0C_BLOCK_BYTES: usize = 2 + Q2_0C_BLOCK / 4;

#[derive(Debug, Clone)]
pub enum Val {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Arr(Vec<Val>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Val {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Val::U8(v) => Some(v as u64),
            Val::U16(v) => Some(v as u64),
            Val::U32(v) => Some(v as u64),
            Val::U64(v) => Some(v),
            Val::I32(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn short(&self) -> String {
        match self {
            Val::Str(s) => format!("{:?}", s.chars().take(60).collect::<String>()),
            Val::Arr(a) => format!("[{} items]", a.len()),
            other => format!("{other:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Innermost first (ggml order).
    pub dims: Vec<u64>,
    pub ty: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.p..self.p + n).ok_or("truncated gguf header")?;
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> { Ok(self.take(1)?[0]) }
    fn u16(&mut self) -> Result<u16, String> { Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap())) }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn str(&mut self) -> Result<String, String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn val(&mut self, ty: u32) -> Result<Val, String> {
        Ok(match ty {
            0 => Val::U8(self.u8()?),
            1 => Val::I8(self.u8()? as i8),
            2 => Val::U16(self.u16()?),
            3 => Val::I16(self.u16()? as i16),
            4 => Val::U32(self.u32()?),
            5 => Val::I32(self.u32()? as i32),
            6 => Val::F32(f32::from_bits(self.u32()?)),
            7 => Val::Bool(self.u8()? != 0),
            8 => Val::Str(self.str()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                Val::Arr((0..n).map(|_| self.val(et)).collect::<Result<_, _>>()?)
            }
            10 => Val::U64(self.u64()?),
            11 => Val::I64(self.u64()? as i64),
            12 => Val::F64(f64::from_bits(self.u64()?)),
            t => return Err(format!("unknown gguf value type {t}")),
        })
    }
}

pub struct Gguf<'a> {
    pub kv: BTreeMap<String, Val>,
    pub tensors: Vec<TensorInfo>,
    pub data: &'a [u8],
}

impl<'a> Gguf<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, String> {
        if bytes.get(..4) != Some(GGUF_MAGIC) {
            return Err("not a GGUF file".into());
        }
        let mut c = Cursor { b: bytes, p: 4 };
        let version = c.u32()?;
        if version < 2 {
            return Err(format!("gguf v{version} unsupported"));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let k = c.str()?;
            let t = c.u32()?;
            kv.insert(k, c.val(t)?);
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.str()?;
            let nd = c.u32()? as usize;
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<_, _>>()?;
            let ty = c.u32()?;
            let offset = c.u64()?;
            tensors.push(TensorInfo { name, dims, ty, offset });
        }
        let align = kv.get("general.alignment").and_then(Val::as_u64).unwrap_or(32) as usize;
        let data_start = c.p.div_ceil(align) * align;
        let data = bytes.get(data_start..).ok_or("truncated gguf data")?;
        Ok(Self { kv, tensors, data })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Bytes from the tensor's offset to the end of the data section.
    pub fn tensor_bytes(&self, t: &TensorInfo) -> &'a [u8] {
        &self.data[t.offset as usize..]
    }

    /// Raw `block_q2_0c` blocks of a 2D `[cols, rows]` tensor: `rows * cols / 512` blocks, row-major.
    pub fn q2_0c_blocks(&self, name: &str) -> Result<(&'a [u8], usize, usize), String> {
        let t = self.tensor(name).ok_or_else(|| format!("gguf: missing tensor {name}"))?;
        if t.ty != GGML_TYPE_Q2_0C || t.dims.len() != 2 {
            return Err(format!("gguf: {name} is not a 2D q2_0c tensor (type {})", t.ty));
        }
        let (cols, rows) = (t.dims[0] as usize, t.dims[1] as usize);
        if !cols.is_multiple_of(Q2_0C_BLOCK) {
            return Err(format!("gguf: {name} cols {cols} not a multiple of 512"));
        }
        let n = rows * cols / Q2_0C_BLOCK * Q2_0C_BLOCK_BYTES;
        let b = self.tensor_bytes(t).get(..n).ok_or_else(|| format!("gguf: {name} truncated"))?;
        Ok((b, rows, cols))
    }
}

/// ggml type ids (ggml.h `ggml_type`).
pub fn type_name(t: u32) -> &'static str {
    match t {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1",
        10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K",
        16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S", 22 => "IQ2_S",
        23 => "IQ4_XS", 24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64", 28 => "F64", 29 => "IQ1_M", 30 => "BF16",
        34 => "TQ1_0", 35 => "TQ2_0", 39 => "MXFP4", 40 => "Q2_0C",
        _ => "?",
    }
}

pub fn f16_to_f32(h: u16) -> f32 {
    let mut out = [0.0f32];
    crate::kernels::half::f16_to_f32(&[h], &mut out);
    out[0]
}
