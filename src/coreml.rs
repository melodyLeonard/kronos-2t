//! Minimal Core ML runner (macOS, `ane` feature): loads a compiled `.mlmodelc` and runs it with
//! one fp16 input, returning named outputs as `f32`. Talks to Core ML through the Objective-C
//! runtime, so it needs no crates.

use std::ffi::{c_char, c_void, CString};
use std::path::Path;

type Id = *mut c_void;
type Sel = *mut c_void;

#[link(name = "objc")]
unsafe extern "C" {
    fn objc_getClass(name: *const c_char) -> Id;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
    fn objc_autoreleasePoolPush() -> *mut c_void;
    fn objc_autoreleasePoolPop(pool: *mut c_void);
}

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "CoreML", kind = "framework")]
unsafe extern "C" {}

const ML_FLOAT16: isize = 0x10000 | 16;
const ML_FLOAT32: isize = 0x10000 | 32;

/// Core ML compute units (`MLComputeUnits`).
#[derive(Clone, Copy, Debug)]
pub enum ComputeUnits {
    CpuOnly = 0,
    CpuAndGpu = 1,
    All = 2,
    CpuAndNeuralEngine = 3,
}

fn sel(name: &str) -> Sel {
    let c = CString::new(name).unwrap();
    // SAFETY: valid NUL-terminated selector name.
    unsafe { sel_registerName(c.as_ptr()) }
}

fn class(name: &str) -> Id {
    let c = CString::new(name).unwrap();
    // SAFETY: valid NUL-terminated class name.
    let cls = unsafe { objc_getClass(c.as_ptr()) };
    assert!(!cls.is_null(), "Objective-C class {name} not found");
    cls
}

/// `objc_msgSend` cast to a concrete signature (arm64 passes arguments like a normal call).
macro_rules! msg {
    ($obj:expr, $sel:expr $(, $arg:expr => $ty:ty)* ; $ret:ty) => {{
        let f: unsafe extern "C" fn(Id, Sel $(, $ty)*) -> $ret = std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f($obj, sel($sel) $(, $arg)*)
    }};
}

unsafe fn nsstring(s: &str) -> Id {
    let c = CString::new(s).unwrap();
    unsafe { msg!(class("NSString"), "stringWithUTF8String:", c.as_ptr() => *const c_char; Id) }
}

unsafe fn error_text(err: Id) -> String {
    if err.is_null() {
        return "unknown Core ML error".into();
    }
    unsafe {
        let desc: Id = msg!(err, "localizedDescription"; Id);
        let p: *const c_char = msg!(desc, "UTF8String"; *const c_char);
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

unsafe fn nsarray_of_longs(v: &[usize]) -> Id {
    unsafe {
        let nums: Vec<Id> = v.iter().map(|&n| msg!(class("NSNumber"), "numberWithLong:", n as isize => isize; Id)).collect();
        msg!(class("NSArray"), "arrayWithObjects:count:", nums.as_ptr() => *const Id, nums.len() => usize; Id)
    }
}

unsafe fn longs_of_nsarray(a: Id) -> Vec<usize> {
    unsafe {
        let n: usize = msg!(a, "count"; usize);
        (0..n).map(|i| msg!(msg!(a, "objectAtIndex:", i => usize; Id), "longValue"; isize) as usize).collect()
    }
}

/// A loaded Core ML model.
pub struct CoreMlModel {
    model: Id,
}

// SAFETY: `MLModel` prediction is thread-safe; callers serialize access through `&mut self`.
unsafe impl Send for CoreMlModel {}

impl CoreMlModel {
    /// Loads a compiled `.mlmodelc` directory.
    pub fn load(path: &Path, units: ComputeUnits) -> Result<Self, String> {
        let p = path.to_str().ok_or("non-UTF-8 path")?;
        // SAFETY: Objective-C calls with matching signatures; returned objects are retained.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let url: Id = msg!(class("NSURL"), "fileURLWithPath:", nsstring(p) => Id; Id);
            let config: Id = msg!(class("MLModelConfiguration"), "new"; Id);
            msg!(config, "setComputeUnits:", units as isize => isize; ());
            let mut err: Id = std::ptr::null_mut();
            let model: Id = msg!(class("MLModel"), "modelWithContentsOfURL:configuration:error:",
                url => Id, config => Id, &mut err => *mut Id; Id);
            let result = if model.is_null() {
                Err(error_text(err))
            } else {
                let _: Id = msg!(model, "retain"; Id);
                Ok(Self { model })
            };
            msg!(config, "release"; ());
            objc_autoreleasePoolPop(pool);
            result
        }
    }

    /// Runs the model on fp16 input `input_name` with `shape`, returning the requested outputs as
    /// contiguous `f32` (row-major in each output's logical shape).
    pub fn predict(&mut self, input_name: &str, shape: &[usize], input: &[f32], outputs: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        assert_eq!(input.len(), shape.iter().product::<usize>());
        let mut half = vec![0u16; input.len()];
        f32_to_f16(input, &mut half);
        // SAFETY: as in `load`; `half` outlives the prediction and the MLMultiArray does not
        // free it (null deallocator).
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let result = (|| {
                let mut strides = vec![1usize; shape.len()];
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1];
                }
                let mut err: Id = std::ptr::null_mut();
                let arr: Id = msg!(class("MLMultiArray"), "alloc"; Id);
                let arr: Id = msg!(arr, "initWithDataPointer:shape:dataType:strides:deallocator:error:",
                    half.as_mut_ptr() as *mut c_void => *mut c_void, nsarray_of_longs(shape) => Id,
                    ML_FLOAT16 => isize, nsarray_of_longs(&strides) => Id,
                    std::ptr::null_mut::<c_void>() => *mut c_void, &mut err => *mut Id; Id);
                if arr.is_null() {
                    return Err(error_text(err));
                }
                let value: Id = msg!(class("MLFeatureValue"), "featureValueWithMultiArray:", arr => Id; Id);
                msg!(arr, "release"; ());
                let dict: Id = msg!(class("NSDictionary"), "dictionaryWithObject:forKey:",
                    value => Id, nsstring(input_name) => Id; Id);
                let provider: Id = msg!(class("MLDictionaryFeatureProvider"), "alloc"; Id);
                let provider: Id = msg!(provider, "initWithDictionary:error:", dict => Id, &mut err => *mut Id; Id);
                if provider.is_null() {
                    return Err(error_text(err));
                }
                let out: Id = msg!(self.model, "predictionFromFeatures:error:", provider => Id, &mut err => *mut Id; Id);
                msg!(provider, "release"; ());
                if out.is_null() {
                    return Err(error_text(err));
                }
                outputs.iter().map(|name| read_output(out, name)).collect()
            })();
            objc_autoreleasePoolPop(pool);
            result
        }
    }
}

impl Drop for CoreMlModel {
    fn drop(&mut self) {
        // SAFETY: balances the `retain` in `load`.
        unsafe { msg!(self.model, "release"; ()) };
    }
}

unsafe fn read_output(provider: Id, name: &str) -> Result<Vec<f32>, String> {
    unsafe {
        let fv: Id = msg!(provider, "featureValueForName:", nsstring(name) => Id; Id);
        if fv.is_null() {
            return Err(format!("model has no output {name}"));
        }
        let arr: Id = msg!(fv, "multiArrayValue"; Id);
        let shape = longs_of_nsarray(msg!(arr, "shape"; Id));
        let strides = longs_of_nsarray(msg!(arr, "strides"; Id));
        let dtype: isize = msg!(arr, "dataType"; isize);
        let ptr: *const c_void = msg!(arr, "dataPointer"; *const c_void);
        let n: usize = shape.iter().product();
        let mut out = vec![0.0f32; n];
        let rank = shape.len();
        let inner = shape[rank - 1];
        let contiguous_inner = strides[rank - 1] == 1;
        // Walk all leading indices; copy the innermost dimension in one go when contiguous.
        let rows = n / inner;
        let mut idx = vec![0usize; rank - 1];
        for r in 0..rows {
            let off: usize = idx.iter().zip(&strides).map(|(i, s)| i * s).sum();
            let dst = &mut out[r * inner..(r + 1) * inner];
            match dtype {
                ML_FLOAT16 if contiguous_inner => {
                    f16_to_f32(std::slice::from_raw_parts((ptr as *const u16).add(off), inner), dst)
                }
                ML_FLOAT16 => {
                    for (j, d) in dst.iter_mut().enumerate() {
                        let mut one = [0.0f32];
                        f16_to_f32(&[*(ptr as *const u16).add(off + j * strides[rank - 1])], &mut one);
                        *d = one[0];
                    }
                }
                ML_FLOAT32 => {
                    for (j, d) in dst.iter_mut().enumerate() {
                        *d = *(ptr as *const f32).add(off + j * strides[rank - 1]);
                    }
                }
                other => return Err(format!("unsupported output dtype {other:#x}")),
            }
            for k in (0..rank - 1).rev() {
                idx[k] += 1;
                if idx[k] < shape[k] {
                    break;
                }
                idx[k] = 0;
            }
        }
        Ok(out)
    }
}

/// IEEE half -> single conversion (NEON `fcvtl` for full blocks of 4).
pub fn f16_to_f32(src: &[u16], dst: &mut [f32]) {
    assert_eq!(src.len(), dst.len());
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 4 <= src.len() {
        // SAFETY: 4 halves in, 4 floats out.
        unsafe {
            let h = std::arch::aarch64::vld1_u16(src.as_ptr().add(i));
            let f: std::arch::aarch64::float32x4_t;
            core::arch::asm!("fcvtl {o:v}.4s, {h:v}.4h", o = out(vreg) f, h = in(vreg) h, options(pure, nomem, nostack));
            std::arch::aarch64::vst1q_f32(dst.as_mut_ptr().add(i), f);
        }
        i += 4;
    }
    for (d, &h) in dst[i..].iter_mut().zip(&src[i..]) {
        let (sign, exp, man) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
        let bits = match exp {
            0 if man == 0 => sign << 31,
            0 => {
                let lz = man.leading_zeros() - 21;
                (sign << 31) | ((113 - lz) << 23) | (((man << lz) & 0x3ff) << 13)
            }
            31 => (sign << 31) | (0xff << 23) | (man << 13),
            e => (sign << 31) | ((e + 112) << 23) | (man << 13),
        };
        *d = f32::from_bits(bits);
    }
}

/// IEEE single -> half conversion (round to nearest even, NEON `fcvtn`).
pub fn f32_to_f16(src: &[f32], dst: &mut [u16]) {
    assert_eq!(src.len(), dst.len());
    let mut i = 0;
    #[cfg(target_arch = "aarch64")]
    while i + 4 <= src.len() {
        // SAFETY: 4 floats in, 4 halves out.
        unsafe {
            let f = std::arch::aarch64::vld1q_f32(src.as_ptr().add(i));
            let h: std::arch::aarch64::uint16x4_t;
            core::arch::asm!("fcvtn {h:v}.4h, {f:v}.4s", h = out(vreg) h, f = in(vreg) f, options(pure, nomem, nostack));
            std::arch::aarch64::vst1_u16(dst.as_mut_ptr().add(i), h);
        }
        i += 4;
    }
    for (d, &x) in dst[i..].iter_mut().zip(&src[i..]) {
        let b = x.to_bits();
        let (sign, exp, man) = ((b >> 16) & 0x8000, ((b >> 23) & 0xff) as i32 - 127 + 15, b & 0x7f_ffff);
        *d = if exp >= 31 {
            (sign | 0x7c00) as u16
        } else if exp <= 0 {
            sign as u16
        } else {
            let mut h = sign | ((exp as u32) << 10) | (man >> 13);
            if (man & 0x1fff) > 0x1000 || ((man & 0x1fff) == 0x1000 && (h & 1) == 1) {
                h += 1;
            }
            h as u16
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        let x: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.731).chain([65504.0, 1e-5, -0.0]).collect();
        let mut h = vec![0u16; x.len()];
        let mut y = vec![0.0f32; x.len()];
        f32_to_f16(&x, &mut h);
        f16_to_f32(&h, &mut y);
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() <= a.abs() * 1e-3 + 1e-4, "{a} vs {b}");
        }
    }
}
