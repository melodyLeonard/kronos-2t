//! Zero-crate Metal.framework GPU inference backend for NF4 models on Apple Silicon.

use crate::hadamard::rademacher_sign_masks_64;
use crate::memory_budget::MemoryBudgetProfile;
use crate::real_llm::LlamaConfig;
use crate::simd_lut::{
    bf16_to_f32, f32_to_bf16, NfSdotMatrix, PageAlignedBytes, NF4_DEFAULT_CODEBOOK_I8,
};
use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::OnceLock;

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

#[link(name = "Metal", kind = "framework")]
unsafe extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

const MTL_RESOURCE_STORAGE_MODE_SHARED: usize = 0;
const MTL_CMD_STATUS_COMPLETED: usize = 4;
const MAX_PROMPT_CHUNK_GPU: usize = 1536;
const GEMV_MAX_TOKENS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct MtlSize {
    width: usize,
    height: usize,
    depth: usize,
}

impl MtlSize {
    #[inline]
    const fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

struct Selectors {
    retain: Sel,
    release: Sel,
    str_utf8: Sel,
    utf8_string: Sel,
    loc_desc: Sel,
    new_cmd_queue: Sel,
    new_lib_src: Sel,
    new_fn_name: Sel,
    new_cps_fn: Sel,
    new_buf_len: Sel,
    new_buf_bytes: Sel,
    new_buf_nocopy: Sel,
    buf_contents: Sel,
    buf_length: Sel,
    cmd_buffer: Sel,
    compute_enc: Sel,
    set_cps: Sel,
    set_buf: Sel,
    set_bytes: Sel,
    dispatch_tg: Sel,
    end_enc: Sel,
    commit: Sel,
    wait_done: Sel,
    cmd_status: Sel,
    cmd_error: Sel,
    rec_max_ws: Sel,
}

// SAFETY: Objective-C `SEL` values are immutable process-wide interned pointers.
unsafe impl Send for Selectors {}
// SAFETY: Objective-C `SEL` values are immutable process-wide interned pointers.
unsafe impl Sync for Selectors {}

fn sels() -> &'static Selectors {
    static SELS: OnceLock<Selectors> = OnceLock::new();
    SELS.get_or_init(|| {
        let reg = |s: &CStr| -> Sel {
            // SAFETY: `s` is a valid NUL-terminated C string literal.
            unsafe { sel_registerName(s.as_ptr()) }
        };
        Selectors {
            retain: reg(c"retain"),
            release: reg(c"release"),
            str_utf8: reg(c"stringWithUTF8String:"),
            utf8_string: reg(c"UTF8String"),
            loc_desc: reg(c"localizedDescription"),
            new_cmd_queue: reg(c"newCommandQueue"),
            new_lib_src: reg(c"newLibraryWithSource:options:error:"),
            new_fn_name: reg(c"newFunctionWithName:"),
            new_cps_fn: reg(c"newComputePipelineStateWithFunction:error:"),
            new_buf_len: reg(c"newBufferWithLength:options:"),
            new_buf_bytes: reg(c"newBufferWithBytes:length:options:"),
            new_buf_nocopy: reg(c"newBufferWithBytesNoCopy:length:options:deallocator:"),
            buf_contents: reg(c"contents"),
            buf_length: reg(c"length"),
            cmd_buffer: reg(c"commandBuffer"),
            compute_enc: reg(c"computeCommandEncoder"),
            set_cps: reg(c"setComputePipelineState:"),
            set_buf: reg(c"setBuffer:offset:atIndex:"),
            set_bytes: reg(c"setBytes:length:atIndex:"),
            dispatch_tg: reg(c"dispatchThreadgroups:threadsPerThreadgroup:"),
            end_enc: reg(c"endEncoding"),
            commit: reg(c"commit"),
            wait_done: reg(c"waitUntilCompleted"),
            cmd_status: reg(c"status"),
            cmd_error: reg(c"error"),
            rec_max_ws: reg(c"recommendedMaxWorkingSetSize"),
        }
    })
}

macro_rules! msg_sel {
    ($obj:expr, $sel:expr $(, $arg:expr => $ty:ty)* ; $ret:ty) => {{
        let f: unsafe extern "C" fn(Id, Sel $(, $ty)*) -> $ret =
            std::mem::transmute(objc_msgSend as unsafe extern "C" fn());
        f($obj, $sel $(, $arg)*)
    }};
}

fn nsstring_class() -> Id {
    static CLS: OnceLock<usize> = OnceLock::new();
    let raw = *CLS.get_or_init(|| {
        // SAFETY: Valid NUL-terminated class name.
        let c = unsafe { objc_getClass(c"NSString".as_ptr()) };
        assert!(!c.is_null(), "NSString class missing");
        c as usize
    });
    raw as Id
}

unsafe fn nsstring(s: &str) -> Result<Id, String> {
    let c = CString::new(s).map_err(|_| "NUL byte in string".to_string())?;
    let st = sels();
    // SAFETY: Valid `NSString` class and UTF-8 pointer inside active autorelease pool.
    let obj = unsafe { msg_sel!(nsstring_class(), st.str_utf8, c.as_ptr() => *const c_char; Id) };
    if obj.is_null() {
        Err("NSString allocation failed".to_string())
    } else {
        Ok(obj)
    }
}

unsafe fn ns_error_string(err: Id) -> String {
    if err.is_null() {
        return "unknown Metal error".to_string();
    }
    let st = sels();
    // SAFETY: `err` is a valid `NSError*`.
    unsafe {
        let desc: Id = msg_sel!(err, st.loc_desc; Id);
        if desc.is_null() {
            return "unknown Metal error".to_string();
        }
        let ptr: *const c_char = msg_sel!(desc, st.utf8_string; *const c_char);
        if ptr.is_null() {
            "unknown Metal error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

struct MtlObj(Id);

// SAFETY: Metal handles are thread-safe refcounted objects; mutable execution uses `&mut self`.
unsafe impl Send for MtlObj {}
// SAFETY: Shared references only inspect immutable metadata/lengths.
unsafe impl Sync for MtlObj {}

impl MtlObj {
    fn from_retained(ptr: Id, what: &str) -> Result<Self, String> {
        if ptr.is_null() {
            Err(format!("Metal allocation failed: {what}"))
        } else {
            Ok(Self(ptr))
        }
    }

    #[inline]
    fn id(&self) -> Id {
        self.0
    }

    #[inline]
    fn contents_ptr(&self) -> *mut u8 {
        let st = sels();
        // SAFETY: Valid `MTLBuffer` with `MTLResourceStorageModeShared`.
        unsafe { msg_sel!(self.0, st.buf_contents; *mut u8) }
    }

    #[inline]
    fn byte_len(&self) -> usize {
        let st = sels();
        // SAFETY: Valid `MTLBuffer`.
        unsafe { msg_sel!(self.0, st.buf_length; usize) }
    }
}

impl Drop for MtlObj {
    fn drop(&mut self) {
        if !self.0.is_null() {
            let st = sels();
            // SAFETY: Balances the `+1` retain count from `new*` / `MTLCreateSystemDefaultDevice`.
            unsafe { msg_sel!(self.0, st.release; ()) };
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NormFwhtParams {
    dim: u32,
    n_tok: u32,
    eps: f32,
    mode: u32,
    interleaved: u32,
    write_f16: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Nf4MatParams {
    cb0: u32,
    cb1: u32,
    cb2: u32,
    cb3: u32,
    rows: u32,
    cols: u32,
    num_stripes: u32,
    num_pairs: u32,
    num_chunks: u32,
    scale_group_chunks: u32,
    num_scale_groups: u32,
    pair_total_bytes: u32,
    n_tok: u32,
    has_bias: u32,
    has_residual: u32,
    n_offset: u32,
    bits: u32,
    chunk_bytes: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RopeKvParams {
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    qkv_rows: u32,
    pos0: u32,
    n_tok: u32,
    layer_idx: u32,
    kv_cap: u32,
    kv_bits: u32,
    window_size: u32,
    sink_tokens: u32,
    ring_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GqaFlashParams {
    num_q_heads: u32,
    num_kv_heads: u32,
    kv_group: u32,
    head_dim: u32,
    qkv_rows: u32,
    h_dim: u32,
    pos0: u32,
    n_tok: u32,
    layer_idx: u32,
    kv_cap: u32,
    attn_scale: f32,
    kv_bits: u32,
    window_size: u32,
    sink_tokens: u32,
    ring_len: u32,
    _pad: u32,
}

pub(crate) struct GpuNf4Matrix {
    pair_buf: MtlObj,
    tail_buf: MtlObj,
    cb_packed: [u32; 4],
    rows: u32,
    cols: u32,
    num_stripes: u32,
    num_pairs: u32,
    num_chunks: u32,
    scale_group_chunks: u32,
    num_scale_groups: u32,
    pair_total_bytes: u32,
    bits: u32,
    chunk_bytes: u32,
}

pub(crate) struct GpuLayerWeights {
    input_norm: MtlObj,
    post_attn_norm: MtlObj,
    qkv_bias: Option<MtlObj>,
    qkv_proj: GpuNf4Matrix,
    o_proj: GpuNf4Matrix,
    gate_up_proj: GpuNf4Matrix,
    gate_up_interleaved: bool,
    down_proj: GpuNf4Matrix,
}

pub(crate) struct LayerBorrow<'a> {
    pub input_norm: &'a [f32],
    pub post_attn_norm: &'a [f32],
    pub qkv_bias: Option<&'a [f32]>,
    pub qkv_proj: &'a NfSdotMatrix,
    pub o_proj: &'a NfSdotMatrix,
    pub gate_up_proj: &'a NfSdotMatrix,
    pub gate_up_interleaved: bool,
    pub down_proj: &'a NfSdotMatrix,
}

struct GpuKvCache {
    num_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    groups_per_head: usize,
    bits: u8,
    window_size: usize,
    sink_tokens: usize,
    ring_len: usize,
    max_seq_len: usize,
    cap_seq_len: usize,
    k_data: MtlObj,
    v_data: MtlObj,
    k_scales: MtlObj,
    v_scales: MtlObj,
}

impl GpuKvCache {
    fn new(
        device: Id,
        cfg: &LlamaConfig,
        bits: u8,
        sliding_window: Option<usize>,
        sink_tokens: usize,
        prompt_chunk_gpu: usize,
        init_cap_tokens: usize,
    ) -> Result<Self, String> {
        if bits != 4 && bits != 8 {
            return Err(format!("Metal KV cache bits must be 4 or 8 (got {bits})"));
        }
        let head_dim = cfg.head_dim();
        if head_dim != 64 && head_dim != 128 {
            return Err(format!("Metal KV cache requires head_dim 64 or 128 (got {head_dim})"));
        }
        let groups_per_head = head_dim / 32;
        let slack = prompt_chunk_gpu.max(512);
        let (window_size, eff_sinks, ring_len, cap_seq_len) =
            if let Some(w) = sliding_window.filter(|&w| w >= 8 && w + slack < cfg.max_seq_len) {
                let s = sink_tokens.min(w / 2);
                let rl = (w - s) + slack;
                (w, s, rl, s + rl)
            } else {
                let init_cap = init_cap_tokens.clamp(64, 2048);
                (0, 0, 0, cfg.max_seq_len.clamp(1, init_cap))
            };
        let slots = cfg.num_hidden_layers * cfg.num_key_value_heads * cap_seq_len;
        let slot_data_bytes = if bits == 4 { head_dim / 2 } else { head_dim };
        let slot_scale_bytes = groups_per_head
            * if bits == 4 {
                std::mem::size_of::<u16>()
            } else {
                std::mem::size_of::<f32>()
            };
        let k_data = alloc_shared_buffer(device, slots * slot_data_bytes, "kv.k_data")?;
        let v_data = alloc_shared_buffer(device, slots * slot_data_bytes, "kv.v_data")?;
        let k_scales = alloc_shared_buffer(device, slots * slot_scale_bytes, "kv.k_scales")?;
        let v_scales = alloc_shared_buffer(device, slots * slot_scale_bytes, "kv.v_scales")?;
        Ok(Self {
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            groups_per_head,
            bits,
            window_size,
            sink_tokens: eff_sinks,
            ring_len,
            max_seq_len: cfg.max_seq_len,
            cap_seq_len,
            k_data,
            v_data,
            k_scales,
            v_scales,
        })
    }

    #[inline]
    fn slot_of_pos(&self, pos: usize) -> usize {
        if self.ring_len == 0 || pos < self.sink_tokens {
            pos
        } else {
            self.sink_tokens + ((pos - self.sink_tokens) % self.ring_len)
        }
    }

    fn physical_bytes(&self) -> usize {
        self.k_data.byte_len()
            + self.v_data.byte_len()
            + self.k_scales.byte_len()
            + self.v_scales.byte_len()
    }

    fn ensure_seq_capacity(&mut self, device: Id, needed_len: usize, live_len: usize) -> Result<(), String> {
        assert!(needed_len <= self.max_seq_len, "sequence exceeds max_seq_len");
        if self.ring_len > 0 || needed_len <= self.cap_seq_len {
            return Ok(());
        }
        let mut new_cap = self.cap_seq_len.max(1);
        while new_cap < needed_len {
            new_cap = (new_cap * 2).min(self.max_seq_len);
        }
        let heads = self.num_layers * self.num_kv_heads;
        let data_bytes = if self.bits == 4 {
            self.head_dim / 2
        } else {
            self.head_dim
        };
        let sc_bytes = self.groups_per_head
            * if self.bits == 4 {
                std::mem::size_of::<u16>()
            } else {
                std::mem::size_of::<f32>()
            };
        let slots = heads * new_cap;

        let new_k = alloc_shared_buffer(device, slots * data_bytes, "kv.k_data_grow")?;
        let new_v = alloc_shared_buffer(device, slots * data_bytes, "kv.v_data_grow")?;
        let new_ks = alloc_shared_buffer(device, slots * sc_bytes, "kv.k_scales_grow")?;
        let new_vs = alloc_shared_buffer(device, slots * sc_bytes, "kv.v_scales_grow")?;

        let copy_len = live_len.min(self.cap_seq_len);
        if copy_len > 0 {
            let old_cap = self.cap_seq_len;
            for h in 0..heads {
                let so = h * old_cap;
                let dn = h * new_cap;
                // SAFETY: Shared unified-memory buffers are idle between synchronous command buffers.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.k_data.contents_ptr().add(so * data_bytes),
                        new_k.contents_ptr().add(dn * data_bytes),
                        copy_len * data_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.v_data.contents_ptr().add(so * data_bytes),
                        new_v.contents_ptr().add(dn * data_bytes),
                        copy_len * data_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.k_scales.contents_ptr().add(so * sc_bytes),
                        new_ks.contents_ptr().add(dn * sc_bytes),
                        copy_len * sc_bytes,
                    );
                    std::ptr::copy_nonoverlapping(
                        self.v_scales.contents_ptr().add(so * sc_bytes),
                        new_vs.contents_ptr().add(dn * sc_bytes),
                        copy_len * sc_bytes,
                    );
                }
            }
        }
        self.k_data = new_k;
        self.v_data = new_v;
        self.k_scales = new_ks;
        self.v_scales = new_vs;
        self.cap_seq_len = new_cap;
        Ok(())
    }

    fn load_prefix_kv(&mut self, device: Id, keys: &[f32], values: &[f32], rows: usize, n: usize) -> Result<(), String> {
        self.ensure_seq_capacity(device, n, 0)?;
        let hd = self.head_dim;
        let gph = self.groups_per_head;
        let nkvh = self.num_kv_heads;
        let kvd = nkvh * hd;
        let cap = self.cap_seq_len;
        if self.bits == 8 {
            let kp = self.k_data.contents_ptr().cast::<i8>();
            let vp = self.v_data.contents_ptr().cast::<i8>();
            let ksp = self.k_scales.contents_ptr().cast::<f32>();
            let vsp = self.v_scales.contents_ptr().cast::<f32>();
            for l in 0..self.num_layers {
                let src_l = l * rows * kvd;
                for pos in 0..n {
                    let s_pos = self.slot_of_pos(pos);
                    for kvh in 0..nkvh {
                        let slot = (l * nkvh + kvh) * cap + s_pos;
                        let off = src_l + pos * kvd + kvh * hd;
                        let kh = &keys[off..off + hd];
                        let vh = &values[off..off + hd];
                        for g in 0..gph {
                            let kg = &kh[g * 32..(g + 1) * 32];
                            let vg = &vh[g * 32..(g + 1) * 32];
                            // SAFETY: `slot < num_layers * nkvh * cap`.
                            unsafe {
                                quantize_q8_group32(kg, kp.add(slot * hd + g * 32), ksp.add(slot * gph + g));
                                quantize_q8_group32(vg, vp.add(slot * hd + g * 32), vsp.add(slot * gph + g));
                            }
                        }
                    }
                }
            }
        } else {
            let half_hd = hd / 2;
            let kp = self.k_data.contents_ptr();
            let vp = self.v_data.contents_ptr();
            let ksp = self.k_scales.contents_ptr().cast::<u16>();
            let vsp = self.v_scales.contents_ptr().cast::<u16>();
            for l in 0..self.num_layers {
                let src_l = l * rows * kvd;
                for pos in 0..n {
                    let s_pos = self.slot_of_pos(pos);
                    for kvh in 0..nkvh {
                        let slot = (l * nkvh + kvh) * cap + s_pos;
                        let off = src_l + pos * kvd + kvh * hd;
                        let kh = &keys[off..off + hd];
                        let vh = &values[off..off + hd];
                        for g in 0..gph {
                            let kg = &kh[g * 32..(g + 1) * 32];
                            let vg = &vh[g * 32..(g + 1) * 32];
                            // SAFETY: `slot < num_layers * nkvh * cap`.
                            unsafe {
                                quantize_q4_hadamard_group32(
                                    kg,
                                    kp.add(slot * half_hd + g * 16),
                                    ksp.add(slot * gph + g),
                                );
                                quantize_q4_hadamard_group32(
                                    vg,
                                    vp.add(slot * half_hd + g * 16),
                                    vsp.add(slot * gph + g),
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

const KV_RAD32_MASK_CPU: u32 = 0xA5C3_9669;

#[inline]
fn kv_nf4_nearest_cpu(y: f32) -> u8 {
    let mut best = 0u8;
    let mut best_d = f32::INFINITY;
    for (idx, &c) in NF4_DEFAULT_CODEBOOK_I8.iter().enumerate() {
        let d = (y - (c as f32)).abs();
        if d < best_d {
            best_d = d;
            best = idx as u8;
        }
    }
    best
}

#[inline]
unsafe fn quantize_q4_hadamard_group32(src: &[f32], dst_q4: *mut u8, dst_s_bf16: *mut u16) {
    debug_assert_eq!(src.len(), 32);
    let inv_sqrt32 = 1.0f32 / (32.0f32).sqrt();
    let mut rot = [0.0f32; 32];
    for (i, &v) in src.iter().enumerate() {
        let sign = if ((KV_RAD32_MASK_CPU >> i) & 1) != 0 { -1.0 } else { 1.0 };
        rot[i] = v * sign * inv_sqrt32;
    }
    let mut d = 1usize;
    while d <= 16 {
        for base in (0..32).step_by(d * 2) {
            for j in 0..d {
                let a = rot[base + j];
                let b = rot[base + j + d];
                rot[base + j] = a + b;
                rot[base + j + d] = a - b;
            }
        }
        d <<= 1;
    }
    let mut m = 0.0f32;
    for &v in &rot {
        m = m.max(v.abs());
    }
    // SAFETY: Caller provides 16 `u8` slots and 1 `u16` slot.
    unsafe {
        if m <= 1e-12 {
            std::ptr::write_bytes(dst_q4, 0x88, 16);
            *dst_s_bf16 = 0;
            return;
        }
        let inv0 = 127.0 / m;
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for &v in &rot {
            let c = NF4_DEFAULT_CODEBOOK_I8[kv_nf4_nearest_cpu(v * inv0) as usize] as f32;
            num += v * c;
            den += c * c;
        }
        let s_ref = if den > 1e-12 && num > 0.0 {
            num / den
        } else {
            m / 127.0
        };
        let sb = f32_to_bf16(s_ref);
        let sf = bf16_to_f32(sb);
        let inv = if sf > 1e-20 { 1.0 / sf } else { 0.0 };
        *dst_s_bf16 = sb;
        for i in 0..16 {
            let lo = kv_nf4_nearest_cpu(rot[i] * inv);
            let hi = kv_nf4_nearest_cpu(rot[i + 16] * inv);
            *dst_q4.add(i) = lo | (hi << 4);
        }
    }
}

#[inline]
unsafe fn quantize_q8_group32(src: &[f32], dst_q: *mut i8, dst_s: *mut f32) {
    let mut m = 0.0f32;
    for &v in src {
        m = m.max(v.abs());
    }
    if m <= 1e-12 {
        // SAFETY: Caller provides 32 `i8` slots and 1 `f32` slot.
        unsafe {
            std::ptr::write_bytes(dst_q, 0, 32);
            *dst_s = 0.0;
        }
        return;
    }
    let sc = m / 127.0;
    let inv = 127.0 / m;
    // SAFETY: Caller provides 32 `i8` slots and 1 `f32` slot.
    unsafe {
        *dst_s = sc;
        for (i, &v) in src.iter().enumerate() {
            *dst_q.add(i) = (v * inv).round().clamp(-127.0, 127.0) as i8;
        }
    }
}

struct GpuScratch {
    cap_tokens: usize,
    logits_cap_tokens: usize,
    hidden: MtlObj,
    x_rot_f32: MtlObj,
    x_rot_f16: MtlObj,
    x_tok_scale: MtlObj,
    qkv: MtlObj,
    attn_out: MtlObj,
    gate_up: MtlObj,
    logits: MtlObj,
}

impl GpuScratch {
    fn new(device: Id, cfg: &LlamaConfig, init_tokens: usize, init_logits_tokens: usize) -> Result<Self, String> {
        let n = init_tokens.max(1);
        let n_f16 = n.div_ceil(128) * 128;
        let nl = init_logits_tokens.max(1);
        let max_rot = cfg.hidden_size.max(cfg.intermediate_size);
        let qkv_rows = cfg.hidden_size + 2 * cfg.kv_dim();
        let f32_b = std::mem::size_of::<f32>();
        let f16_b = std::mem::size_of::<u16>();
        Ok(Self {
            cap_tokens: n,
            logits_cap_tokens: nl,
            hidden: alloc_shared_buffer(device, n * cfg.hidden_size * f32_b, "sc.hidden")?,
            x_rot_f32: alloc_shared_buffer(device, n * max_rot * f32_b, "sc.x_rot_f32")?,
            x_rot_f16: alloc_shared_buffer(device, n_f16 * max_rot * f16_b, "sc.x_rot_f16")?,
            x_tok_scale: alloc_shared_buffer(device, n * f32_b, "sc.x_tok_scale")?,
            qkv: alloc_shared_buffer(device, n * qkv_rows * f32_b, "sc.qkv")?,
            attn_out: alloc_shared_buffer(device, n * cfg.hidden_size * f32_b, "sc.attn_out")?,
            gate_up: alloc_shared_buffer(device, n * 2 * cfg.intermediate_size * f32_b, "sc.gate_up")?,
            logits: alloc_shared_buffer(device, nl * cfg.vocab_size * f32_b, "sc.logits")?,
        })
    }

    fn ensure_capacity(
        &mut self,
        device: Id,
        cfg: &LlamaConfig,
        n_tok: usize,
        logits_tokens: usize,
        max_chunk_tokens: usize,
    ) -> Result<(), String> {
        let f32_b = std::mem::size_of::<f32>();
        let f16_b = std::mem::size_of::<u16>();
        if n_tok > self.cap_tokens {
            let n = n_tok.next_power_of_two().min(max_chunk_tokens).max(n_tok);
            let n_f16 = n.div_ceil(128) * 128;
            let max_rot = cfg.hidden_size.max(cfg.intermediate_size);
            let qkv_rows = cfg.hidden_size + 2 * cfg.kv_dim();
            self.hidden = alloc_shared_buffer(device, n * cfg.hidden_size * f32_b, "sc.hidden")?;
            self.x_rot_f32 = alloc_shared_buffer(device, n * max_rot * f32_b, "sc.x_rot_f32")?;
            self.x_rot_f16 = alloc_shared_buffer(device, n_f16 * max_rot * f16_b, "sc.x_rot_f16")?;
            self.x_tok_scale = alloc_shared_buffer(device, n * f32_b, "sc.x_tok_scale")?;
            self.qkv = alloc_shared_buffer(device, n * qkv_rows * f32_b, "sc.qkv")?;
            self.attn_out = alloc_shared_buffer(device, n * cfg.hidden_size * f32_b, "sc.attn_out")?;
            self.gate_up = alloc_shared_buffer(device, n * 2 * cfg.intermediate_size * f32_b, "sc.gate_up")?;
            self.cap_tokens = n;
        }
        if logits_tokens > self.logits_cap_tokens {
            let nl = logits_tokens.max(8);
            self.logits = alloc_shared_buffer(device, nl * cfg.vocab_size * f32_b, "sc.logits")?;
            self.logits_cap_tokens = nl;
        }
        Ok(())
    }
}

pub struct MetalLlamaEngine {
    device: MtlObj,
    queue: MtlObj,
    prompt_chunk_gpu: usize,
    init_kv_cap_tokens: usize,
    pso_norm_fwht: MtlObj,
    pso_transpose_f16: MtlObj,
    pso_gemv_decode1: MtlObj,
    pso_gemv_batch: MtlObj,
    pso_gemm_mma: MtlObj,
    pso_rope_kv: MtlObj,
    pso_gqa_decode: MtlObj,
    pso_gqa_prefill: MtlObj,
    dummy_buf: MtlObj,
    rademacher_buf: MtlObj,
    rope_buf: MtlObj,
    layers: Vec<GpuLayerWeights>,
    final_norm: MtlObj,
    lm_head: GpuNf4Matrix,
    kv: GpuKvCache,
    scratch: GpuScratch,
}

fn alloc_shared_buffer(device: Id, byte_len: usize, what: &str) -> Result<MtlObj, String> {
    let len = byte_len.max(16);
    let st = sels();
    // SAFETY: Valid `MTLDevice` and byte length.
    let ptr = unsafe {
        msg_sel!(
            device,
            st.new_buf_len,
            len => usize,
            MTL_RESOURCE_STORAGE_MODE_SHARED => usize;
            Id
        )
    };
    let obj = MtlObj::from_retained(ptr, what)?;
    // Pre-fault 16 KiB shared pages on CPU so first GPU dispatch never stalls on VM faults.
    unsafe {
        let base = obj.contents_ptr();
        if !base.is_null() {
            let mut off = 0usize;
            while off < len {
                std::ptr::write_volatile(base.add(off), 0);
                off += 16384;
            }
        }
    }
    Ok(obj)
}

fn upload_slice_buffer<T: Copy>(device: Id, slice: &[T], what: &str) -> Result<MtlObj, String> {
    let byte_len = std::mem::size_of_val(slice);
    if byte_len == 0 {
        return alloc_shared_buffer(device, 16, what);
    }
    let st = sels();
    // SAFETY: `slice.as_ptr()` is valid for `byte_len` bytes during synchronous copy construction.
    let ptr = unsafe {
        msg_sel!(
            device,
            st.new_buf_bytes,
            slice.as_ptr().cast::<c_void>() => *const c_void,
            byte_len => usize,
            MTL_RESOURCE_STORAGE_MODE_SHARED => usize;
            Id
        )
    };
    MtlObj::from_retained(ptr, what)
}

fn wrap_page_aligned_nocopy(device: Id, bytes: &PageAlignedBytes, dummy: Id, what: &str) -> Result<MtlObj, String> {
    let alloc_len = bytes.page_alloc_bytes();
    let st = sels();
    if alloc_len == 0 {
        // SAFETY: Retain shared 16-byte dummy buffer for empty tail stream.
        let retained = unsafe { msg_sel!(dummy, st.retain; Id) };
        return MtlObj::from_retained(retained, what);
    }
    // SAFETY: `PageAlignedBytes` is 16 KiB aligned and `MetalLlamaEngine` drops before weight matrices.
    let ptr = unsafe {
        msg_sel!(
            device,
            st.new_buf_nocopy,
            bytes.as_page_ptr() => *mut c_void,
            alloc_len => usize,
            MTL_RESOURCE_STORAGE_MODE_SHARED => usize,
            std::ptr::null_mut::<c_void>() => *mut c_void;
            Id
        )
    };
    MtlObj::from_retained(ptr, what)
}

fn wrap_nf4_matrix(device: Id, mat: &NfSdotMatrix, dummy: Id, label: &str) -> Result<GpuNf4Matrix, String> {
    if !(2..=5).contains(&mat.bits) {
        return Err(format!(
            "{label}: Metal GPU backend supports 2..=5-bit NfSdot (got {}-bit)",
            mat.bits
        ));
    }
    let pair_buf = wrap_page_aligned_nocopy(device, &mat.pair_stream, dummy, label)?;
    let tail_buf = wrap_page_aligned_nocopy(device, &mat.tail_stream, dummy, label)?;
    let pack4 = |base: usize| -> u32 {
        (mat.codebook[base] as u8 as u32)
            | ((mat.codebook[base + 1] as u8 as u32) << 8)
            | ((mat.codebook[base + 2] as u8 as u32) << 16)
            | ((mat.codebook[base + 3] as u8 as u32) << 24)
    };
    let cb_packed = [pack4(0), pack4(4), pack4(8), pack4(12)];
    Ok(GpuNf4Matrix {
        pair_buf,
        tail_buf,
        cb_packed,
        rows: mat.rows as u32,
        cols: mat.cols as u32,
        num_stripes: mat.num_stripes as u32,
        num_pairs: mat.num_pairs as u32,
        num_chunks: mat.num_chunks as u32,
        scale_group_chunks: mat.scale_group_chunks as u32,
        num_scale_groups: mat.num_scale_groups as u32,
        pair_total_bytes: mat.pair_total_bytes() as u32,
        bits: mat.bits as u32,
        chunk_bytes: NfSdotMatrix::stripe_chunk_bytes(mat.bits) as u32,
    })
}

fn build_pipeline(device: Id, library: Id, fn_name: &str) -> Result<MtlObj, String> {
    let st = sels();
    // SAFETY: Valid `MTLDevice` and `MTLLibrary` inside active autorelease pool.
    unsafe {
        let ns_name = nsstring(fn_name)?;
        let func: Id = msg_sel!(library, st.new_fn_name, ns_name => Id; Id);
        if func.is_null() {
            return Err(format!("Metal kernel function `{fn_name}` not found"));
        }
        let func_guard = MtlObj(func);
        let mut err: Id = std::ptr::null_mut();
        let pso: Id = msg_sel!(device, st.new_cps_fn, func_guard.id() => Id, &mut err => *mut Id; Id);
        if pso.is_null() {
            Err(format!("Metal pipeline `{fn_name}` failed: {}", ns_error_string(err)))
        } else {
            Ok(MtlObj(pso))
        }
    }
}

impl MetalLlamaEngine {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn try_new(
        cfg: &LlamaConfig,
        rope_cos_sin: &[f32],
        layers_in: &[LayerBorrow<'_>],
        final_norm_in: &[f32],
        lm_head_in: &NfSdotMatrix,
        kv_bits: u8,
        kv_window: Option<usize>,
        kv_sinks: usize,
    ) -> Result<Self, String> {
        if !cfg.hidden_size.is_multiple_of(64) || !cfg.intermediate_size.is_multiple_of(64) {
            return Err("Metal backend requires hidden_size and intermediate_size divisible by 64".to_string());
        }
        let hd = cfg.head_dim();
        if hd != 64 && hd != 128 {
            return Err(format!("Metal backend requires head_dim 64 or 128 (got {hd})"));
        }
        // SAFETY: Objective-C calls wrapped in an explicit autorelease pool.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<Self, String> {
                let raw_dev = MTLCreateSystemDefaultDevice();
                let device = MtlObj::from_retained(raw_dev, "MTLCreateSystemDefaultDevice")?;
                let st = sels();
                let rec_ws: u64 = msg_sel!(device.id(), st.rec_max_ws; u64);
                let budget = MemoryBudgetProfile::active_with_metal_working_set(
                    (rec_ws > 0).then_some(rec_ws as usize),
                );
                let prompt_chunk_gpu = budget.prompt_chunk_gpu.clamp(128, MAX_PROMPT_CHUNK_GPU);
                let init_kv_cap_tokens = budget.initial_kv_cap_tokens.clamp(64, 2048);

                let raw_q: Id = msg_sel!(device.id(), st.new_cmd_queue; Id);
                let queue = MtlObj::from_retained(raw_q, "newCommandQueue")?;

                let src_ns = nsstring(METAL_SHADER_SOURCE)?;
                let mut err: Id = std::ptr::null_mut();
                let raw_lib: Id = msg_sel!(
                    device.id(),
                    st.new_lib_src,
                    src_ns => Id,
                    std::ptr::null_mut::<c_void>() => Id,
                    &mut err => *mut Id;
                    Id
                );
                if raw_lib.is_null() {
                    return Err(format!("Metal shader compile error: {}", ns_error_string(err)));
                }
                let library = MtlObj(raw_lib);

                let pso_norm_fwht = build_pipeline(device.id(), library.id(), "rmsnorm_fwht64")?;
                let pso_transpose_f16 = build_pipeline(device.id(), library.id(), "transpose_x_f16")?;
                let pso_gemv_decode1 = build_pipeline(device.id(), library.id(), "nf4_gemv_decode1")?;
                let pso_gemv_batch = build_pipeline(device.id(), library.id(), "nf4_gemv_batch")?;
                let pso_gemm_mma = build_pipeline(device.id(), library.id(), "nf4_gemm_mma")?;
                let pso_rope_kv = build_pipeline(device.id(), library.id(), "qkv_rope_kv_store")?;
                let pso_gqa_decode = build_pipeline(device.id(), library.id(), "causal_gqa_flash_q8_decode")?;
                let pso_gqa_prefill = build_pipeline(device.id(), library.id(), "causal_gqa_flash_q8_prefill")?;

                let dummy_buf = alloc_shared_buffer(device.id(), 64, "dummy")?;
                let max_blocks = cfg.hidden_size.max(cfg.intermediate_size) / 64;
                let mut rad_masks = Vec::with_capacity(max_blocks * 64);
                for b in 0..max_blocks {
                    rad_masks.extend_from_slice(&rademacher_sign_masks_64(b));
                }
                let rademacher_buf = upload_slice_buffer(device.id(), &rad_masks, "rademacher")?;
                let rope_buf = upload_slice_buffer(device.id(), rope_cos_sin, "rope_cos_sin")?;

                let mut layers = Vec::with_capacity(layers_in.len());
                for (idx, l) in layers_in.iter().enumerate() {
                    let input_norm = upload_slice_buffer(device.id(), l.input_norm, "input_norm")?;
                    let post_attn_norm = upload_slice_buffer(device.id(), l.post_attn_norm, "post_attn_norm")?;
                    let qkv_bias = match l.qkv_bias {
                        Some(b) => Some(upload_slice_buffer(device.id(), b, "qkv_bias")?),
                        None => None,
                    };
                    let pfx = format!("layer.{idx}");
                    layers.push(GpuLayerWeights {
                        input_norm,
                        post_attn_norm,
                        qkv_bias,
                        qkv_proj: wrap_nf4_matrix(device.id(), l.qkv_proj, dummy_buf.id(), &format!("{pfx}.qkv"))?,
                        o_proj: wrap_nf4_matrix(device.id(), l.o_proj, dummy_buf.id(), &format!("{pfx}.o"))?,
                        gate_up_proj: wrap_nf4_matrix(
                            device.id(),
                            l.gate_up_proj,
                            dummy_buf.id(),
                            &format!("{pfx}.gate_up"),
                        )?,
                        gate_up_interleaved: l.gate_up_interleaved,
                        down_proj: wrap_nf4_matrix(device.id(), l.down_proj, dummy_buf.id(), &format!("{pfx}.down"))?,
                    });
                }
                let final_norm = upload_slice_buffer(device.id(), final_norm_in, "final_norm")?;
                let lm_head = wrap_nf4_matrix(device.id(), lm_head_in, dummy_buf.id(), "lm_head")?;
                let kv = GpuKvCache::new(
                    device.id(),
                    cfg,
                    kv_bits,
                    kv_window,
                    kv_sinks,
                    prompt_chunk_gpu,
                    init_kv_cap_tokens,
                )?;
                let init_scratch_toks = cfg.max_seq_len.clamp(16, prompt_chunk_gpu.min(256));
                let scratch = GpuScratch::new(device.id(), cfg, init_scratch_toks, 8)?;

                Ok(Self {
                    device,
                    queue,
                    prompt_chunk_gpu,
                    init_kv_cap_tokens,
                    pso_norm_fwht,
                    pso_transpose_f16,
                    pso_gemv_decode1,
                    pso_gemv_batch,
                    pso_gemm_mma,
                    pso_rope_kv,
                    pso_gqa_decode,
                    pso_gqa_prefill,
                    dummy_buf,
                    rademacher_buf,
                    rope_buf,
                    layers,
                    final_norm,
                    lm_head,
                    kv,
                    scratch,
                })
            })();
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    #[inline]
    pub fn prompt_chunk_gpu(&self) -> usize {
        self.prompt_chunk_gpu
    }

    #[inline]
    pub fn kv_bits(&self) -> u8 {
        self.kv.bits
    }

    #[inline]
    pub fn kv_sliding_window(&self) -> Option<(usize, usize)> {
        (self.kv.window_size > 0).then_some((self.kv.window_size, self.kv.sink_tokens))
    }

    #[inline]
    pub fn can_truncate_kv_to(&self, cur_pos: usize, target_len: usize) -> bool {
        if target_len >= cur_pos || self.kv.ring_len == 0 {
            return true;
        }
        let slack = self
            .kv
            .ring_len
            .saturating_sub(self.kv.window_size.saturating_sub(self.kv.sink_tokens));
        target_len <= self.kv.sink_tokens || (cur_pos - target_len) <= slack
    }

    pub fn reconfigure_kv(
        &mut self,
        cfg: &LlamaConfig,
        bits: u8,
        window: Option<usize>,
        sinks: usize,
    ) -> Result<(), String> {
        self.kv = GpuKvCache::new(
            self.device.id(),
            cfg,
            bits,
            window,
            sinks,
            self.prompt_chunk_gpu,
            self.init_kv_cap_tokens,
        )?;
        Ok(())
    }

    #[inline]
    pub fn kv_cache_bytes(&self) -> usize {
        self.kv.physical_bytes()
    }

    pub fn load_prefix_kv(&mut self, keys: &[f32], values: &[f32], rows: usize, n: usize) -> Result<(), String> {
        self.kv.load_prefix_kv(self.device.id(), keys, values, rows, n)
    }

    pub fn forward_tokens<F>(
        &mut self,
        cfg: &LlamaConfig,
        tokens: &[usize],
        pos_start: usize,
        mut embed_fn: F,
        mut all_logits_out: Option<&mut [f32]>,
        last_logits_out: &mut [f32],
    ) -> Result<(), String>
    where
        F: FnMut(usize, &mut [f32]),
    {
        assert!(!tokens.is_empty());
        assert!(pos_start + tokens.len() <= cfg.max_seq_len, "sequence exceeds max_seq_len");
        let vocab = cfg.vocab_size;
        assert_eq!(last_logits_out.len(), vocab);
        if let Some(ref all) = all_logits_out {
            assert!(all.len() >= tokens.len() * vocab);
        }

        let want_all = all_logits_out.is_some();
        let max_chunk = self.prompt_chunk_gpu;
        let mut offset = 0usize;
        while offset < tokens.len() {
            let chunk_len = (tokens.len() - offset).min(max_chunk);
            let is_last_chunk = offset + chunk_len == tokens.len();
            let chunk = &tokens[offset..offset + chunk_len];
            let chunk_pos0 = pos_start + offset;
            let logits_toks = if want_all {
                chunk_len
            } else if is_last_chunk {
                1
            } else {
                0
            };

            self.run_chunk(cfg, chunk, chunk_pos0, &mut embed_fn, logits_toks)?;

            if let Some(all_dst) = all_logits_out.as_deref_mut() {
                let count = chunk_len * vocab;
                // SAFETY: `scratch.logits` holds at least `count` floats after `waitUntilCompleted`.
                let src = unsafe { std::slice::from_raw_parts(self.scratch.logits.contents_ptr().cast::<f32>(), count) };
                all_dst[offset * vocab..(offset + chunk_len) * vocab].copy_from_slice(src);
                if is_last_chunk {
                    last_logits_out.copy_from_slice(&src[(chunk_len - 1) * vocab..chunk_len * vocab]);
                }
            } else if is_last_chunk {
                // SAFETY: `scratch.logits` holds `vocab` floats for the final prompt token.
                let src = unsafe { std::slice::from_raw_parts(self.scratch.logits.contents_ptr().cast::<f32>(), vocab) };
                last_logits_out.copy_from_slice(src);
            }

            offset += chunk_len;
        }
        Ok(())
    }

    fn run_chunk<F>(
        &mut self,
        cfg: &LlamaConfig,
        tokens: &[usize],
        pos0: usize,
        embed_fn: &mut F,
        logits_tokens: usize,
    ) -> Result<(), String>
    where
        F: FnMut(usize, &mut [f32]),
    {
        let n_tok = tokens.len();
        self.kv.ensure_seq_capacity(self.device.id(), pos0 + n_tok, pos0)?;
        self.scratch.ensure_capacity(
            self.device.id(),
            cfg,
            n_tok,
            logits_tokens.max(1),
            self.prompt_chunk_gpu,
        )?;

        let h_dim = cfg.hidden_size;
        let inter_dim = cfg.intermediate_size;
        let qkv_rows = h_dim + 2 * cfg.kv_dim();
        let head_dim = cfg.head_dim();
        let num_q_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let kv_group = num_q_heads / num_kv_heads;
        let attn_scale = 1.0f32 / (head_dim as f32).sqrt();
        let use_mma = n_tok >= 2;

        // Populate input embeddings directly into unified-RAM `scratch.hidden`.
        {
            // SAFETY: `scratch.hidden` has capacity `>= n_tok * h_dim` floats and GPU is idle.
            let hid_slice =
                unsafe { std::slice::from_raw_parts_mut(self.scratch.hidden.contents_ptr().cast::<f32>(), n_tok * h_dim) };
            for (t, &tok) in tokens.iter().enumerate() {
                embed_fn(tok, &mut hid_slice[t * h_dim..(t + 1) * h_dim]);
            }
        }

        // SAFETY: All Metal encoder bindings stay alive inside `self` until `waitUntilCompleted` returns.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<(), String> {
                let st = sels();
                let cmd: Id = msg_sel!(self.queue.id(), st.cmd_buffer; Id);
                if cmd.is_null() {
                    return Err("MTLCommandQueue commandBuffer returned nil".to_string());
                }
                let enc: Id = msg_sel!(cmd, st.compute_enc; Id);
                if enc.is_null() {
                    return Err("MTLCommandBuffer computeCommandEncoder returned nil".to_string());
                }

                let kv_cap = self.kv.cap_seq_len as u32;

                for (l_idx, layer) in self.layers.iter().enumerate() {
                    // 1. Pre-attention RMSNorm + signed FWHT-64.
                    self.encode_norm_fwht(
                        enc,
                        self.scratch.hidden.id(),
                        0,
                        layer.input_norm.id(),
                        h_dim,
                        n_tok,
                        cfg.rms_norm_eps,
                        0,
                        false,
                        use_mma,
                    );

                    // 2. Fused QKV projection (+ optional Qwen2 QKV bias).
                    self.encode_nf4_proj(
                        enc,
                        &layer.qkv_proj,
                        layer.qkv_bias.as_ref().map(|b| b.id()),
                        None,
                        self.scratch.qkv.id(),
                        n_tok,
                    );

                    // 3. NeoX RoPE on Q/K + Q8_0 / Hadamard-NF4 group-32 KV cache write.
                    {
                        msg_sel!(enc, st.set_cps, self.pso_rope_kv.id() => Id; ());
                        msg_sel!(enc, st.set_buf, self.scratch.qkv.id() => Id, 0usize => usize, 0usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.rope_buf.id() => Id, 0usize => usize, 1usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.k_data.id() => Id, 0usize => usize, 2usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.k_scales.id() => Id, 0usize => usize, 3usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.v_data.id() => Id, 0usize => usize, 4usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.v_scales.id() => Id, 0usize => usize, 5usize => usize; ());
                        let rp = RopeKvParams {
                            num_q_heads: num_q_heads as u32,
                            num_kv_heads: num_kv_heads as u32,
                            head_dim: head_dim as u32,
                            qkv_rows: qkv_rows as u32,
                            pos0: pos0 as u32,
                            n_tok: n_tok as u32,
                            layer_idx: l_idx as u32,
                            kv_cap,
                            kv_bits: self.kv.bits as u32,
                            window_size: self.kv.window_size as u32,
                            sink_tokens: self.kv.sink_tokens as u32,
                            ring_len: self.kv.ring_len as u32,
                        };
                        msg_sel!(
                            enc,
                            st.set_bytes,
                            (&rp as *const RopeKvParams).cast::<c_void>() => *const c_void,
                            std::mem::size_of::<RopeKvParams>() => usize,
                            6usize => usize;
                            ()
                        );
                        let total_heads = num_q_heads + 2 * num_kv_heads;
                        msg_sel!(
                            enc,
                            st.dispatch_tg,
                            MtlSize::new(total_heads.div_ceil(4), n_tok, 1) => MtlSize,
                            MtlSize::new(128, 1, 1) => MtlSize;
                            ()
                        );
                    }

                    // 4. Causal GQA online-softmax FlashAttention / FlashDecoding over Q8_0 / Q4 KV cache.
                    {
                        let pso = if n_tok <= GEMV_MAX_TOKENS {
                            self.pso_gqa_decode.id()
                        } else {
                            self.pso_gqa_prefill.id()
                        };
                        msg_sel!(enc, st.set_cps, pso => Id; ());
                        msg_sel!(enc, st.set_buf, self.scratch.qkv.id() => Id, 0usize => usize, 0usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.k_data.id() => Id, 0usize => usize, 1usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.k_scales.id() => Id, 0usize => usize, 2usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.v_data.id() => Id, 0usize => usize, 3usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.kv.v_scales.id() => Id, 0usize => usize, 4usize => usize; ());
                        msg_sel!(enc, st.set_buf, self.scratch.attn_out.id() => Id, 0usize => usize, 5usize => usize; ());
                        let gp = GqaFlashParams {
                            num_q_heads: num_q_heads as u32,
                            num_kv_heads: num_kv_heads as u32,
                            kv_group: kv_group as u32,
                            head_dim: head_dim as u32,
                            qkv_rows: qkv_rows as u32,
                            h_dim: h_dim as u32,
                            pos0: pos0 as u32,
                            n_tok: n_tok as u32,
                            layer_idx: l_idx as u32,
                            kv_cap,
                            attn_scale,
                            kv_bits: self.kv.bits as u32,
                            window_size: self.kv.window_size as u32,
                            sink_tokens: self.kv.sink_tokens as u32,
                            ring_len: self.kv.ring_len as u32,
                            _pad: 0,
                        };
                        msg_sel!(
                            enc,
                            st.set_bytes,
                            (&gp as *const GqaFlashParams).cast::<c_void>() => *const c_void,
                            std::mem::size_of::<GqaFlashParams>() => usize,
                            6usize => usize;
                            ()
                        );
                        let (grid_x, grid_y) = if n_tok <= GEMV_MAX_TOKENS {
                            (num_q_heads, n_tok)
                        } else {
                            (num_q_heads, n_tok.div_ceil(32))
                        };
                        msg_sel!(
                            enc,
                            st.dispatch_tg,
                            MtlSize::new(grid_x, grid_y, 1) => MtlSize,
                            MtlSize::new(128, 1, 1) => MtlSize;
                            ()
                        );
                    }

                    // 5. Signed FWHT-64 on `attn_out` + `o_proj` fused with `+ hidden` residual.
                    self.encode_norm_fwht(
                        enc,
                        self.scratch.attn_out.id(),
                        0,
                        self.dummy_buf.id(),
                        h_dim,
                        n_tok,
                        cfg.rms_norm_eps,
                        1,
                        false,
                        use_mma,
                    );
                    self.encode_nf4_proj(
                        enc,
                        &layer.o_proj,
                        None,
                        Some(self.scratch.hidden.id()),
                        self.scratch.hidden.id(),
                        n_tok,
                    );

                    // 6. Post-attention RMSNorm + signed FWHT-64 -> `gate_up_proj`.
                    self.encode_norm_fwht(
                        enc,
                        self.scratch.hidden.id(),
                        0,
                        layer.post_attn_norm.id(),
                        h_dim,
                        n_tok,
                        cfg.rms_norm_eps,
                        0,
                        false,
                        use_mma,
                    );
                    self.encode_nf4_proj(
                        enc,
                        &layer.gate_up_proj,
                        None,
                        None,
                        self.scratch.gate_up.id(),
                        n_tok,
                    );

                    // 7. Fused SwiGLU + signed FWHT-64 -> `down_proj` fused with `+ hidden` residual.
                    self.encode_norm_fwht(
                        enc,
                        self.scratch.gate_up.id(),
                        0,
                        self.dummy_buf.id(),
                        inter_dim,
                        n_tok,
                        cfg.rms_norm_eps,
                        2,
                        layer.gate_up_interleaved,
                        use_mma,
                    );
                    self.encode_nf4_proj(
                        enc,
                        &layer.down_proj,
                        None,
                        Some(self.scratch.hidden.id()),
                        self.scratch.hidden.id(),
                        n_tok,
                    );
                }

                // 8. Final RMSNorm + signed FWHT-64 + `lm_head` (last token only or full verification batch).
                if logits_tokens > 0 {
                    let src_tok_off = if logits_tokens == 1 {
                        (n_tok - 1) * h_dim * std::mem::size_of::<f32>()
                    } else {
                        debug_assert_eq!(logits_tokens, n_tok);
                        0
                    };
                    let head_mma = logits_tokens >= 2;
                    self.encode_norm_fwht(
                        enc,
                        self.scratch.hidden.id(),
                        src_tok_off,
                        self.final_norm.id(),
                        h_dim,
                        logits_tokens,
                        cfg.rms_norm_eps,
                        0,
                        false,
                        head_mma,
                    );
                    self.encode_nf4_proj(
                        enc,
                        &self.lm_head,
                        None,
                        None,
                        self.scratch.logits.id(),
                        logits_tokens,
                    );
                }

                msg_sel!(enc, st.end_enc; ());
                msg_sel!(cmd, st.commit; ());
                msg_sel!(cmd, st.wait_done; ());

                let status: usize = msg_sel!(cmd, st.cmd_status; usize);
                if status != MTL_CMD_STATUS_COMPLETED {
                    let err: Id = msg_sel!(cmd, st.cmd_error; Id);
                    return Err(format!(
                        "Metal command buffer failed (status {status}): {}",
                        ns_error_string(err)
                    ));
                }
                Ok(())
            })();
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn encode_norm_fwht(
        &self,
        enc: Id,
        x_in: Id,
        x_offset_bytes: usize,
        norm_w: Id,
        dim: usize,
        n_tok: usize,
        eps: f32,
        mode: u32,
        interleaved: bool,
        write_f16: bool,
    ) {
        let st = sels();
        let params = NormFwhtParams {
            dim: dim as u32,
            n_tok: n_tok as u32,
            eps,
            mode,
            interleaved: u32::from(interleaved),
            write_f16: u32::from(write_f16),
        };
        // SAFETY: Valid active `MTLComputeCommandEncoder` and buffers owned by `self`.
        unsafe {
            msg_sel!(enc, st.set_cps, self.pso_norm_fwht.id() => Id; ());
            msg_sel!(enc, st.set_buf, x_in => Id, x_offset_bytes => usize, 0usize => usize; ());
            msg_sel!(enc, st.set_buf, norm_w => Id, 0usize => usize, 1usize => usize; ());
            msg_sel!(enc, st.set_buf, self.rademacher_buf.id() => Id, 0usize => usize, 2usize => usize; ());
            msg_sel!(enc, st.set_buf, self.scratch.x_rot_f32.id() => Id, 0usize => usize, 3usize => usize; ());
            msg_sel!(enc, st.set_buf, self.scratch.x_rot_f16.id() => Id, 0usize => usize, 4usize => usize; ());
            msg_sel!(enc, st.set_buf, self.scratch.x_tok_scale.id() => Id, 0usize => usize, 5usize => usize; ());
            msg_sel!(
                enc,
                st.set_bytes,
                (&params as *const NormFwhtParams).cast::<c_void>() => *const c_void,
                std::mem::size_of::<NormFwhtParams>() => usize,
                6usize => usize;
                ()
            );
            msg_sel!(
                enc,
                st.dispatch_tg,
                MtlSize::new(n_tok, 1, 1) => MtlSize,
                MtlSize::new(256, 1, 1) => MtlSize;
                ()
            );
            if write_f16 {
                let n_stride = n_tok.div_ceil(128) * 128;
                msg_sel!(enc, st.set_cps, self.pso_transpose_f16.id() => Id; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_rot_f32.id() => Id, 0usize => usize, 0usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_tok_scale.id() => Id, 0usize => usize, 1usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_rot_f16.id() => Id, 0usize => usize, 2usize => usize; ());
                msg_sel!(
                    enc,
                    st.set_bytes,
                    (&params as *const NormFwhtParams).cast::<c_void>() => *const c_void,
                    std::mem::size_of::<NormFwhtParams>() => usize,
                    3usize => usize;
                    ()
                );
                msg_sel!(
                    enc,
                    st.dispatch_tg,
                    MtlSize::new(n_stride / 32, dim / 32, 1) => MtlSize,
                    MtlSize::new(256, 1, 1) => MtlSize;
                    ()
                );
            }
        }
    }

    unsafe fn encode_nf4_proj(
        &self,
        enc: Id,
        mat: &GpuNf4Matrix,
        bias: Option<Id>,
        residual: Option<Id>,
        out_buf: Id,
        n_tok: usize,
    ) {
        let st = sels();
        let params = Nf4MatParams {
            cb0: mat.cb_packed[0],
            cb1: mat.cb_packed[1],
            cb2: mat.cb_packed[2],
            cb3: mat.cb_packed[3],
            rows: mat.rows,
            cols: mat.cols,
            num_stripes: mat.num_stripes,
            num_pairs: mat.num_pairs,
            num_chunks: mat.num_chunks,
            scale_group_chunks: mat.scale_group_chunks,
            num_scale_groups: mat.num_scale_groups,
            pair_total_bytes: mat.pair_total_bytes,
            n_tok: n_tok as u32,
            has_bias: u32::from(bias.is_some()),
            has_residual: u32::from(residual.is_some()),
            n_offset: 0,
            bits: mat.bits,
            chunk_bytes: mat.chunk_bytes,
        };
        let bias_id = bias.unwrap_or_else(|| self.dummy_buf.id());
        let res_id = residual.unwrap_or_else(|| self.dummy_buf.id());

        // SAFETY: Valid active `MTLComputeCommandEncoder` and zero-copy weight buffers owned by `self`.
        unsafe {
            if n_tok == 1 {
                msg_sel!(enc, st.set_cps, self.pso_gemv_decode1.id() => Id; ());
                msg_sel!(enc, st.set_buf, mat.pair_buf.id() => Id, 0usize => usize, 0usize => usize; ());
                msg_sel!(enc, st.set_buf, mat.tail_buf.id() => Id, 0usize => usize, 1usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_rot_f32.id() => Id, 0usize => usize, 2usize => usize; ());
                msg_sel!(enc, st.set_buf, bias_id => Id, 0usize => usize, 3usize => usize; ());
                msg_sel!(enc, st.set_buf, res_id => Id, 0usize => usize, 4usize => usize; ());
                msg_sel!(enc, st.set_buf, out_buf => Id, 0usize => usize, 5usize => usize; ());
                msg_sel!(
                    enc,
                    st.set_bytes,
                    (&params as *const Nf4MatParams).cast::<c_void>() => *const c_void,
                    std::mem::size_of::<Nf4MatParams>() => usize,
                    6usize => usize;
                    ()
                );
                let pair_blocks = (mat.num_stripes as usize).div_ceil(2);
                msg_sel!(
                    enc,
                    st.dispatch_tg,
                    MtlSize::new(pair_blocks, 1, 1) => MtlSize,
                    MtlSize::new(128, 1, 1) => MtlSize;
                    ()
                );
            } else if n_tok <= GEMV_MAX_TOKENS {
                msg_sel!(enc, st.set_cps, self.pso_gemv_batch.id() => Id; ());
                msg_sel!(enc, st.set_buf, mat.pair_buf.id() => Id, 0usize => usize, 0usize => usize; ());
                msg_sel!(enc, st.set_buf, mat.tail_buf.id() => Id, 0usize => usize, 1usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_rot_f16.id() => Id, 0usize => usize, 2usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_tok_scale.id() => Id, 0usize => usize, 3usize => usize; ());
                msg_sel!(enc, st.set_buf, bias_id => Id, 0usize => usize, 4usize => usize; ());
                msg_sel!(enc, st.set_buf, res_id => Id, 0usize => usize, 5usize => usize; ());
                msg_sel!(enc, st.set_buf, out_buf => Id, 0usize => usize, 6usize => usize; ());
                msg_sel!(
                    enc,
                    st.set_bytes,
                    (&params as *const Nf4MatParams).cast::<c_void>() => *const c_void,
                    std::mem::size_of::<Nf4MatParams>() => usize,
                    7usize => usize;
                    ()
                );
                let pair_blocks = (mat.num_stripes as usize).div_ceil(2);
                msg_sel!(
                    enc,
                    st.dispatch_tg,
                    MtlSize::new(pair_blocks, 1, 1) => MtlSize,
                    MtlSize::new(128, 1, 1) => MtlSize;
                    ()
                );
            } else {
                let m_tiles = (mat.rows as usize).div_ceil(32);
                let n_tiles = n_tok.div_ceil(128);

                msg_sel!(enc, st.set_cps, self.pso_gemm_mma.id() => Id; ());
                msg_sel!(enc, st.set_buf, mat.pair_buf.id() => Id, 0usize => usize, 0usize => usize; ());
                msg_sel!(enc, st.set_buf, mat.tail_buf.id() => Id, 0usize => usize, 1usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_rot_f16.id() => Id, 0usize => usize, 2usize => usize; ());
                msg_sel!(enc, st.set_buf, self.scratch.x_tok_scale.id() => Id, 0usize => usize, 3usize => usize; ());
                msg_sel!(enc, st.set_buf, bias_id => Id, 0usize => usize, 4usize => usize; ());
                msg_sel!(enc, st.set_buf, res_id => Id, 0usize => usize, 5usize => usize; ());
                msg_sel!(enc, st.set_buf, out_buf => Id, 0usize => usize, 6usize => usize; ());
                msg_sel!(
                    enc,
                    st.set_bytes,
                    (&params as *const Nf4MatParams).cast::<c_void>() => *const c_void,
                    std::mem::size_of::<Nf4MatParams>() => usize,
                    7usize => usize;
                    ()
                );
                msg_sel!(
                    enc,
                    st.dispatch_tg,
                    MtlSize::new(n_tiles, m_tiles, 1) => MtlSize,
                    MtlSize::new(128, 1, 1) => MtlSize;
                    ()
                );
            }
        }
    }
}

const METAL_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct NormFwhtParams {
    uint dim;
    uint n_tok;
    float eps;
    uint mode;        // 0 = RMSNorm + FWHT64, 1 = FWHT64, 2 = SwiGLU + FWHT64
    uint interleaved; // 1 = [gate_64(ic), up_64(ic)] blocks
    uint write_f16;
};

struct Nf4MatParams {
    uint cb0;
    uint cb1;
    uint cb2;
    uint cb3;
    uint rows;
    uint cols;
    uint num_stripes;
    uint num_pairs;
    uint num_chunks;
    uint scale_group_chunks;
    uint num_scale_groups;
    uint pair_total_bytes;
    uint n_tok;
    uint has_bias;
    uint has_residual;
    uint n_offset;
    uint bits;
    uint chunk_bytes;
};

struct RopeKvParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint head_dim;
    uint qkv_rows;
    uint pos0;
    uint n_tok;
    uint layer_idx;
    uint kv_cap;
    uint kv_bits;
    uint window_size;
    uint sink_tokens;
    uint ring_len;
};

struct GqaFlashParams {
    uint num_q_heads;
    uint num_kv_heads;
    uint kv_group;
    uint head_dim;
    uint qkv_rows;
    uint h_dim;
    uint pos0;
    uint n_tok;
    uint layer_idx;
    uint kv_cap;
    float attn_scale;
    uint kv_bits;
    uint window_size;
    uint sink_tokens;
    uint ring_len;
    uint _pad;
};

constant float KV_INV_SQRT32 = 0.17677669529663689f;
constant uint  KV_RAD32_MASK = 0xA5C39669u;
constant float KV_NF4_CB[16] = {
    -127.0f, -96.0f, -75.0f, -58.0f, -44.0f, -31.0f, -18.0f, -6.0f,
       6.0f,  18.0f,  31.0f,  44.0f,  58.0f,  75.0f,  96.0f, 127.0f
};

inline float bf16_u16_to_f32(ushort b) {
    return as_type<float>(uint(b) << 16u);
}

inline ushort f32_to_bf16_u16(float v) {
    uint bits = as_type<uint>(v);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        return ushort((bits >> 16u) | 0x0040u);
    }
    uint bias = 0x00007FFFu + ((bits >> 16u) & 1u);
    return ushort((bits + bias) >> 16u);
}

inline uint kv_slot_of_pos(uint pos, uint sink_tokens, uint ring_len) {
    return (ring_len == 0u || pos < sink_tokens)
        ? pos
        : (sink_tokens + ((pos - sink_tokens) % ring_len));
}

inline uint kv_nf4_idx(float y) {
    float a = fabs(y);
    uint p;
    if (a < 37.5f) {
        p = (a < 12.0f) ? 0u : ((a < 24.5f) ? 1u : 2u);
    } else if (a < 85.5f) {
        p = (a < 51.0f) ? 3u : ((a < 66.5f) ? 4u : 5u);
    } else {
        p = (a < 111.5f) ? 6u : 7u;
    }
    return (y >= 0.0f) ? (8u + p) : (7u - p);
}

inline float simd_fwht32(float v, uint lane) {
    float sign = (((KV_RAD32_MASK >> lane) & 1u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32;
    v *= sign;
    for (uint d = 1u; d <= 16u; d <<= 1u) {
        float p = simd_shuffle_xor(v, ushort(d));
        v = ((lane & d) == 0u) ? (v + p) : (p - v);
    }
    return v;
}

inline float simd_ifwht32(float v, uint lane) {
    for (uint d = 1u; d <= 16u; d <<= 1u) {
        float p = simd_shuffle_xor(v, ushort(d));
        v = ((lane & d) == 0u) ? (v + p) : (p - v);
    }
    float sign = (((KV_RAD32_MASK >> lane) & 1u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32;
    return v * sign;
}

inline void ifwht32_pair4(thread float4& v0, thread float4& v1, uint qc) {
    v0 = float4(v0.x + v0.y, v0.x - v0.y, v0.z + v0.w, v0.z - v0.w);
    v1 = float4(v1.x + v1.y, v1.x - v1.y, v1.z + v1.w, v1.z - v1.w);
    v0 = float4(v0.x + v0.z, v0.y + v0.w, v0.x - v0.z, v0.y - v0.w);
    v1 = float4(v1.x + v1.z, v1.y + v1.w, v1.x - v1.z, v1.y - v1.w);
    float4 p0 = simd_shuffle_xor(v0, ushort(1));
    float4 p1 = simd_shuffle_xor(v1, ushort(1));
    bool low4 = ((qc & 1u) == 0u);
    v0 = low4 ? (v0 + p0) : (p0 - v0);
    v1 = low4 ? (v1 + p1) : (p1 - v1);
    float4 q0 = simd_shuffle_xor(v0, ushort(2));
    float4 q1 = simd_shuffle_xor(v1, ushort(2));
    bool low8 = ((qc & 2u) == 0u);
    v0 = low8 ? (v0 + q0) : (q0 - v0);
    v1 = low8 ? (v1 + q1) : (q1 - v1);
    float4 s0 = v0 + v1;
    float4 s1 = v0 - v1;
    uint m0 = KV_RAD32_MASK >> (qc * 4u);
    uint m1 = KV_RAD32_MASK >> (16u + qc * 4u);
    v0 = s0 * float4(
        ((m0 & 1u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32,
        ((m0 & 2u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32,
        ((m0 & 4u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32,
        ((m0 & 8u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32);
    v1 = s1 * float4(
        ((m1 & 1u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32,
        ((m1 & 2u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32,
        ((m1 & 4u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32,
        ((m1 & 8u) != 0u) ? -KV_INV_SQRT32 : KV_INV_SQRT32);
}

inline float bf16_bytes_to_f32(device const uchar* p) {
    uint u = uint(p[0]) | (uint(p[1]) << 8);
    return as_type<float>(u << 16);
}

inline float nf4_byte_lut(uint cb0, uint cb1, uint cb2, uint cb3, uint idx) {
    uint p0 = ((idx & 4u) == 0u) ? cb0 : cb1;
    uint p1 = ((idx & 4u) == 0u) ? cb2 : cb3;
    uint w  = ((idx & 8u) == 0u) ? p0 : p1;
    uint shift = ((idx ^ 3u) & 3u) << 3u;
    return float(int(w << shift) >> 24);
}

inline float4 nf4_lut4_lo(uint cb0, uint cb1, uint cb2, uint cb3, uint w) {
    return float4(
        nf4_byte_lut(cb0, cb1, cb2, cb3, w & 0xFu),
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 8u) & 0xFu),
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 16u) & 0xFu),
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 24u) & 0xFu));
}

inline float4 nf4_lut4_hi(uint cb0, uint cb1, uint cb2, uint cb3, uint w) {
    return float4(
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 4u) & 0xFu),
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 12u) & 0xFu),
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 20u) & 0xFu),
        nf4_byte_lut(cb0, cb1, cb2, cb3, (w >> 28u) & 0xFu));
}

inline void fwht64_simd(thread float& a, thread float& b, uint m0, uint m1, uint lane) {
    a = as_type<float>(as_type<uint>(a) ^ m0) * 0.125f;
    b = as_type<float>(as_type<uint>(b) ^ m1) * 0.125f;
    for (uint d = 1u; d <= 16u; d <<= 1u) {
        float pa = simd_shuffle_xor(a, ushort(d));
        float pb = simd_shuffle_xor(b, ushort(d));
        bool low = ((lane & d) == 0u);
        a = low ? (a + pa) : (pa - a);
        b = low ? (b + pb) : (pb - b);
    }
    float out0 = a + b;
    float out1 = a - b;
    a = out0;
    b = out1;
}

kernel void rmsnorm_fwht64(
    device const float* x_in        [[buffer(0)]],
    device const float* norm_w      [[buffer(1)]],
    device const uint*  rademacher  [[buffer(2)]],
    device float*       x_rot_f32   [[buffer(3)]],
    device half*        x_rot_f16   [[buffer(4)]],
    device float*       x_tok_scale [[buffer(5)]],
    constant NormFwhtParams& p      [[buffer(6)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint tid   [[thread_index_in_threadgroup]])
{
    if (tg_id >= p.n_tok) return;
    const uint dim = p.dim;
    const uint num_blocks = dim >> 6u;
    threadgroup float sg_buf[8];
    (void)x_rot_f16;

    float inv_rms = 1.0f;
    if (p.mode == 0u) {
        device const float* row = x_in + ulong(tg_id) * ulong(dim);
        float sq = 0.0f;
        for (uint i = tid; i < dim; i += 256u) {
            float v = row[i];
            sq += v * v;
        }
        sq = simd_sum(sq);
        if (lane == 0u) sg_buf[sg_id] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = (lane < 8u) ? sg_buf[lane] : 0.0f;
        tot = simd_sum(tot);
        inv_rms = rsqrt(tot / float(dim) + p.eps);
        if (p.write_f16 != 0u) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    device float* dst32 = x_rot_f32 + ulong(tg_id) * ulong(dim);
    float local_max = 0.0f;

    if (p.mode == 2u) {
        device const float* gu = x_in + ulong(tg_id) * ulong(dim * 2u);
        for (uint b = sg_id; b < num_blocks; b += 8u) {
            uint i0 = (b << 6u) + lane;
            uint i1 = i0 + 32u;
            float g0, u0, g1, u1;
            if (p.interleaved != 0u) {
                uint base = b * 128u;
                g0 = gu[base + lane];
                g1 = gu[base + 32u + lane];
                u0 = gu[base + 64u + lane];
                u1 = gu[base + 96u + lane];
            } else {
                g0 = gu[i0];
                g1 = gu[i1];
                u0 = gu[dim + i0];
                u1 = gu[dim + i1];
            }
            float a = (g0 / (1.0f + exp(-g0))) * u0;
            float c = (g1 / (1.0f + exp(-g1))) * u1;
            fwht64_simd(a, c, rademacher[i0], rademacher[i1], lane);
            dst32[i0] = a;
            dst32[i1] = c;
            if (p.write_f16 != 0u) {
                local_max = max(local_max, max(fabs(a), fabs(c)));
            }
        }
    } else {
        device const float* row = x_in + ulong(tg_id) * ulong(dim);
        for (uint b = sg_id; b < num_blocks; b += 8u) {
            uint i0 = (b << 6u) + lane;
            uint i1 = i0 + 32u;
            float a = row[i0];
            float c = row[i1];
            if (p.mode == 0u) {
                a = (a * inv_rms) * norm_w[i0];
                c = (c * inv_rms) * norm_w[i1];
            }
            fwht64_simd(a, c, rademacher[i0], rademacher[i1], lane);
            dst32[i0] = a;
            dst32[i1] = c;
            if (p.write_f16 != 0u) {
                local_max = max(local_max, max(fabs(a), fabs(c)));
            }
        }
    }

    if (p.write_f16 != 0u) {
        float sm = simd_max(local_max);
        if (lane == 0u) sg_buf[sg_id] = sm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg_id == 0u) {
            float tok_max = (lane < 8u) ? sg_buf[lane] : 0.0f;
            tok_max = simd_max(tok_max);
            if (lane == 0u) {
                x_tok_scale[tg_id] = (tok_max > 1e-12f) ? (tok_max * (1.0f / 16.0f)) : 1.0f;
            }
        }
    }
}

kernel void transpose_x_f16(
    device const float* x_rot_f32   [[buffer(0)]],
    device const float* x_tok_scale [[buffer(1)]],
    device half*        x_rot_f16   [[buffer(2)]],
    constant NormFwhtParams& p      [[buffer(3)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint tid     [[thread_index_in_threadgroup]])
{
    const uint tok_base = tg_pos.x * 32u;
    const uint col_base = tg_pos.y * 32u;
    const uint n_stride = ((p.n_tok + 127u) >> 7u) << 7u;
    const uint tx = tid & 31u;
    const uint ty = tid >> 5u;

    threadgroup half tile[32][33];
    for (uint r = ty; r < 32u; r += 8u) {
        uint tok = tok_base + r;
        half h = 0.0h;
        if (tok < p.n_tok) {
            float inv_s = 1.0f / x_tok_scale[tok];
            h = half(x_rot_f32[ulong(tok) * ulong(p.dim) + ulong(col_base + tx)] * inv_s);
        }
        tile[tx][r] = h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = ty; r < 32u; r += 8u) {
        uint col = col_base + r;
        x_rot_f16[ulong(col) * ulong(n_stride) + ulong(tok_base + tx)] = tile[r][tx];
    }
}

inline half2 make_nf4_pair(uint cb0, uint cb1, uint cb2, uint cb3, uint byte_idx) {
    float lo = nf4_byte_lut(cb0, cb1, cb2, cb3, byte_idx & 0xFu);
    float hi = nf4_byte_lut(cb0, cb1, cb2, cb3, (byte_idx >> 4u) & 0xFu);
    return half2(half(lo), half(hi));
}

constant char4 E8Q2_EVEN_TPL[28] = {
    char4(0, 0, 0, 0),
    char4(32, 32, 0, 0),
    char4(32, 0, 32, 0),
    char4(32, 0, 0, 32),
    char4(0, 32, 32, 0),
    char4(0, 32, 0, 32),
    char4(0, 0, 32, 32),
    char4(64, 0, 0, 0),
    char4(0, 64, 0, 0),
    char4(0, 0, 64, 0),
    char4(0, 0, 0, 64),
    char4(32, 32, 32, 32),
    char4(32, 0, 0, 0),
    char4(0, 32, 0, 0),
    char4(0, 0, 32, 0),
    char4(0, 0, 0, 32),
    char4(32, 32, 32, 0),
    char4(32, 32, 0, 32),
    char4(32, 0, 32, 32),
    char4(0, 32, 32, 32),
    char4(96, 0, 0, 0),
    char4(0, 96, 0, 0),
    char4(0, 0, 96, 0),
    char4(0, 0, 0, 96),
    char4(64, 32, 32, 32),
    char4(32, 64, 32, 32),
    char4(32, 32, 64, 32),
    char4(32, 32, 32, 64)
};

inline void init_nf_tables(
    constant Nf4MatParams& p,
    uint tid_tg,
    threadgroup half2* l_pair,
    threadgroup float* cb32_s)
{
    if (p.bits == 4u) {
        l_pair[tid_tg] = make_nf4_pair(p.cb0, p.cb1, p.cb2, p.cb3, tid_tg);
        l_pair[tid_tg + 128u] = make_nf4_pair(p.cb0, p.cb1, p.cb2, p.cb3, tid_tg + 128u);
    } else if (tid_tg < 16u) {
        float c = nf4_byte_lut(p.cb0, p.cb1, p.cb2, p.cb3, tid_tg);
        if (p.bits == 5u) {
            cb32_s[15u - tid_tg] = -c;
            cb32_s[16u + tid_tg] = c;
        } else {
            cb32_s[tid_tg] = c;
            cb32_s[16u + tid_tg] = 0.0f;
        }
    }
}

inline void unpack_nf_8cols(
    device const uchar* d_ptr,
    uint bits,
    uint k,
    uint tid,
    threadgroup const half2* l_pair,
    threadgroup const float* cb32_s,
    thread float4& w_lo,
    thread float4& w_hi)
{
    if (bits == 4u) {
        uchar4 b = ((device const uchar4*)d_ptr)[k * 32u + tid];
        half2 p0 = l_pair[b.x], p1 = l_pair[b.y], p2 = l_pair[b.z], p3 = l_pair[b.w];
        w_lo = float4(p0.x, p1.x, p2.x, p3.x);
        w_hi = float4(p0.y, p1.y, p2.y, p3.y);
    } else if (bits == 2u) {
        uint g = tid >> 3u;
        uint half16 = (tid >> 2u) & 1u;
        uint j = tid & 3u;
        uint eb = ((k & 1u) << 1u) | half16;
        uint off = (k >> 1u) * 128u + g * 32u + eb * 4u + j;
        uint tpl = uint(d_ptr[off]);
        uint sgn = uint(d_ptr[off + 16u]);
        float4 mu, mv;
        if ((popcount(tpl ^ sgn) & 1u) != 0u) {
            uint tpl_u = tpl & 0xFu;
            uint tpl_v = tpl >> 4u;
            uint u_idx = (tpl_u < 12u && tpl_v >= 12u) ? (16u + tpl_u) : tpl_u;
            uint v_idx = (tpl_v < 12u && tpl_u >= 12u) ? (16u + tpl_v) : tpl_v;
            mu = float4(E8Q2_EVEN_TPL[u_idx]);
            mv = float4(E8Q2_EVEN_TPL[v_idx]);
        } else {
            mu = float4(
                ((tpl & 1u) != 0u) ? 48.0f : 16.0f,
                ((tpl & 2u) != 0u) ? 48.0f : 16.0f,
                ((tpl & 4u) != 0u) ? 48.0f : 16.0f,
                ((tpl & 8u) != 0u) ? 48.0f : 16.0f);
            mv = float4(
                ((tpl & 16u) != 0u) ? 48.0f : 16.0f,
                ((tpl & 32u) != 0u) ? 48.0f : 16.0f,
                ((tpl & 64u) != 0u) ? 48.0f : 16.0f,
                ((tpl & 128u) != 0u) ? 48.0f : 16.0f);
        }
        w_lo = float4(
            ((sgn & 1u) != 0u) ? -mu.x : mu.x,
            ((sgn & 2u) != 0u) ? -mu.y : mu.y,
            ((sgn & 4u) != 0u) ? -mu.z : mu.z,
            ((sgn & 8u) != 0u) ? -mu.w : mu.w);
        w_hi = float4(
            ((sgn & 16u) != 0u) ? -mv.x : mv.x,
            ((sgn & 32u) != 0u) ? -mv.y : mv.y,
            ((sgn & 64u) != 0u) ? -mv.z : mv.z,
            ((sgn & 128u) != 0u) ? -mv.w : mv.w);
    } else if (bits == 3u) {
        uint g = tid >> 3u;
        uint half16 = (tid >> 2u) & 1u;
        uint j = tid & 3u;
        device const uchar4* blk4 = (device const uchar4*)(d_ptr + (k >> 1u) * 192u + g * 48u);
        uint sub_pair = ((k & 1u) << 1u) | half16;
        uint4 idx_lo, idx_hi;
        if (sub_pair == 0u) {
            uint4 v0 = uint4(blk4[j]);
            idx_lo = v0 & 7u;
            idx_hi = (v0 >> 3u) & 7u;
        } else if (sub_pair == 1u) {
            uint4 v0 = uint4(blk4[j]);
            uint4 v1 = uint4(blk4[4u + j]);
            uint4 v2 = uint4(blk4[8u + j]);
            idx_lo = ((v0 >> 6u) & 3u) | (((v2 >> 6u) & 1u) << 2u);
            idx_hi = v1 & 7u;
        } else if (sub_pair == 2u) {
            uint4 v1 = uint4(blk4[4u + j]);
            uint4 v2 = uint4(blk4[8u + j]);
            idx_lo = (v1 >> 3u) & 7u;
            idx_hi = ((v1 >> 6u) & 3u) | (((v2 >> 7u) & 1u) << 2u);
        } else {
            uint4 v2 = uint4(blk4[8u + j]);
            idx_lo = v2 & 7u;
            idx_hi = (v2 >> 3u) & 7u;
        }
        w_lo = float4(cb32_s[idx_lo.x], cb32_s[idx_lo.y], cb32_s[idx_lo.z], cb32_s[idx_lo.w]);
        w_hi = float4(cb32_s[idx_hi.x], cb32_s[idx_hi.y], cb32_s[idx_hi.z], cb32_s[idx_hi.w]);
    } else {
        uint g = tid >> 3u;
        uint half16 = (tid >> 2u) & 1u;
        uint j = tid & 3u;
        uint4 b = uint4(((device const uchar4*)d_ptr)[k * 32u + tid]);
        uint4 hb = uint4(((device const uchar4*)(d_ptr + 512u))[g * 8u + (k >> 1u) * 4u + j]);
        uint sh0 = (((k & 1u) == 0u) ? 4u : 0u) + half16 * 2u;
        uint4 idx_lo = (b & 0xFu) | (((hb >> sh0) & 1u) << 4u);
        uint4 idx_hi = (b >> 4u) | (((hb >> (sh0 + 1u)) & 1u) << 4u);
        w_lo = float4(cb32_s[idx_lo.x], cb32_s[idx_lo.y], cb32_s[idx_lo.z], cb32_s[idx_lo.w]);
        w_hi = float4(cb32_s[idx_hi.x], cb32_s[idx_hi.y], cb32_s[idx_hi.z], cb32_s[idx_hi.w]);
    }
}

kernel void nf4_gemv_decode1(
    device const uchar* pair_stream [[buffer(0)]],
    device const uchar* tail_stream [[buffer(1)]],
    device const float* x_rot       [[buffer(2)]],
    device const float* bias        [[buffer(3)]],
    device const float* residual    [[buffer(4)]],
    device float*       out         [[buffer(5)]],
    constant Nf4MatParams& p        [[buffer(6)]],
    uint pair_idx [[threadgroup_position_in_grid]],
    uint sg_id    [[simdgroup_index_in_threadgroup]],
    uint tid      [[thread_index_in_simdgroup]],
    uint tid_tg   [[thread_index_in_threadgroup]])
{
    const uint sa = pair_idx * 2u;
    if (sa >= p.num_stripes) return;
    const bool is_tail = (pair_idx >= p.num_pairs);
    const bool has_b = (!is_tail) && ((sa + 1u) < p.num_stripes);

    threadgroup half2 l_pair[256];
    threadgroup float cb32_s[32];
    init_nf_tables(p, tid_tg, l_pair, cb32_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint g = tid >> 3u;
    const uint half16 = (tid >> 2u) & 1u;
    const uint j = tid & 3u;
    const uint lane = (g << 2u) | j;
    const uint chunk_b = p.chunk_bytes;
    const uint pair_chunk_b = 2u * chunk_b;

    float acc_a = 0.0f;
    float acc_b = 0.0f;

    if (!is_tail) {
        device const uchar* p_base = pair_stream + ulong(pair_idx) * ulong(p.pair_total_bytes);
        const uint sg_stride = 64u + p.scale_group_chunks * pair_chunk_b;
        for (uint sg = sg_id; sg < p.num_scale_groups; sg += 4u) {
            device const uchar* sg_ptr = p_base + ulong(sg) * ulong(sg_stride);
            float wsa = bf16_bytes_to_f32(sg_ptr + lane * 2u);
            float wsb = bf16_bytes_to_f32(sg_ptr + 32u + lane * 2u);
            uint c0 = sg * p.scale_group_chunks;
            uint c1 = min(c0 + p.scale_group_chunks, p.num_chunks);
            float sga = 0.0f;
            float sgb = 0.0f;
            for (uint c = c0; c < c1; ++c) {
                uint sub_c = c - c0;
                device const uchar* da = sg_ptr + 64u + sub_c * pair_chunk_b;
                device const uchar* db = da + chunk_b;
                for (uint k = 0u; k < 4u; ++k) {
                    uint col = c * 64u + k * 16u + half16 * 8u;
                    device const float4* xp = (device const float4*)(x_rot + col);
                    float4 x0 = xp[0];
                    float4 x1 = xp[1];

                    float4 wa0, wa1, wb0, wb1;
                    unpack_nf_8cols(da, p.bits, k, tid, l_pair, cb32_s, wa0, wa1);
                    sga += dot(wa0, x0) + dot(wa1, x1);

                    unpack_nf_8cols(db, p.bits, k, tid, l_pair, cb32_s, wb0, wb1);
                    sgb += dot(wb0, x0) + dot(wb1, x1);
                }
            }
            acc_a += sga * wsa;
            acc_b += sgb * wsb;
        }
    } else {
        const uint sg_stride = 32u + p.scale_group_chunks * chunk_b;
        for (uint sg = sg_id; sg < p.num_scale_groups; sg += 4u) {
            device const uchar* sg_ptr = tail_stream + ulong(sg) * ulong(sg_stride);
            float wsa = bf16_bytes_to_f32(sg_ptr + lane * 2u);
            uint c0 = sg * p.scale_group_chunks;
            uint c1 = min(c0 + p.scale_group_chunks, p.num_chunks);
            float sga = 0.0f;
            for (uint c = c0; c < c1; ++c) {
                uint sub_c = c - c0;
                device const uchar* da = sg_ptr + 32u + sub_c * chunk_b;
                for (uint k = 0u; k < 4u; ++k) {
                    uint col = c * 64u + k * 16u + half16 * 8u;
                    device const float4* xp = (device const float4*)(x_rot + col);
                    float4 wa0, wa1;
                    unpack_nf_8cols(da, p.bits, k, tid, l_pair, cb32_s, wa0, wa1);
                    sga += dot(wa0, xp[0]) + dot(wa1, xp[1]);
                }
            }
            acc_a += sga * wsa;
        }
    }

    acc_a += simd_shuffle_xor(acc_a, ushort(4));
    acc_b += simd_shuffle_xor(acc_b, ushort(4));

    threadgroup float tg_red[4][32];
    if (half16 == 0u) {
        tg_red[sg_id][lane] = acc_a;
        tg_red[sg_id][16u + lane] = acc_b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        bool second = (tid >= 16u);
        if (second && !has_b) return;
        uint row = sa * 16u + tid;
        if (row < p.rows) {
            float sum = tg_red[0][tid] + tg_red[1][tid] + tg_red[2][tid] + tg_red[3][tid];
            if (p.has_bias != 0u) sum += bias[row];
            if (p.has_residual != 0u) sum += residual[row];
            out[row] = sum;
        }
    }
}

kernel void nf4_gemv_batch(
    device const uchar* pair_stream [[buffer(0)]],
    device const uchar* tail_stream [[buffer(1)]],
    device const half*  x_rot_f16   [[buffer(2)]],
    device const float* x_tok_scale [[buffer(3)]],
    device const float* bias        [[buffer(4)]],
    device const float* residual    [[buffer(5)]],
    device float*       out         [[buffer(6)]],
    constant Nf4MatParams& p        [[buffer(7)]],
    uint pair_idx [[threadgroup_position_in_grid]],
    uint sg_id    [[simdgroup_index_in_threadgroup]],
    uint tid      [[thread_index_in_simdgroup]],
    uint tid_tg   [[thread_index_in_threadgroup]])
{
    const uint sa = pair_idx * 2u;
    if (sa >= p.num_stripes) return;
    const bool is_tail = (pair_idx >= p.num_pairs);
    const bool has_b = (!is_tail) && ((sa + 1u) < p.num_stripes);

    threadgroup half2 l_pair[256];
    threadgroup float cb32_s[32];
    threadgroup float tg_buf[4][256];
    init_nf_tables(p, tid_tg, l_pair, cb32_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint g = tid >> 3u;
    const uint half16 = (tid >> 2u) & 1u;
    const uint j = tid & 3u;
    const uint lane = (g << 2u) | j;
    const uint wc = half16 * 8u;
    const uint chunk_b = p.chunk_bytes;
    const uint pair_chunk_b = 2u * chunk_b;
    threadgroup half* ws_sg = (threadgroup half*)(&tg_buf[sg_id][0]);

    simdgroup_float8x8 acc_a0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_a1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_b0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 acc_b1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    if (!is_tail) {
        device const uchar* p_base = pair_stream + ulong(pair_idx) * ulong(p.pair_total_bytes);
        const uint sg_stride = 64u + p.scale_group_chunks * pair_chunk_b;
        for (uint sg = sg_id; sg < p.num_scale_groups; sg += 4u) {
            device const uchar* sg_ptr = p_base + ulong(sg) * ulong(sg_stride);
            float wsa = bf16_bytes_to_f32(sg_ptr + lane * 2u);
            float wsb = bf16_bytes_to_f32(sg_ptr + 32u + lane * 2u);
            uint c0 = sg * p.scale_group_chunks;
            uint c1 = min(c0 + p.scale_group_chunks, p.num_chunks);
            for (uint c = c0; c < c1; ++c) {
                uint sub_c = c - c0;
                device const uchar* da = sg_ptr + 64u + sub_c * pair_chunk_b;
                device const uchar* db = da + chunk_b;
                for (uint k = 0u; k < 4u; ++k) {
                    uint col = c * 64u + k * 16u;
                    simdgroup_half8x8 x_f0, x_f1;
                    simdgroup_load(x_f0, x_rot_f16 + ulong(col) * 128ul, 128);
                    simdgroup_load(x_f1, x_rot_f16 + ulong(col + 8u) * 128ul, 128);

                    float4 wa0, wa1;
                    unpack_nf_8cols(da, p.bits, k, tid, l_pair, cb32_s, wa0, wa1);
                    *(threadgroup half4*)(ws_sg + lane * 20u + wc) = half4(wa0 * wsa);
                    *(threadgroup half4*)(ws_sg + lane * 20u + wc + 4u) = half4(wa1 * wsa);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_half8x8 wa00, wa01, wa10, wa11;
                    simdgroup_load(wa00, ws_sg + 0u, 20);
                    simdgroup_load(wa01, ws_sg + 8u, 20);
                    simdgroup_load(wa10, ws_sg + 160u, 20);
                    simdgroup_load(wa11, ws_sg + 168u, 20);
                    simdgroup_multiply_accumulate(acc_a0, wa00, x_f0, acc_a0);
                    simdgroup_multiply_accumulate(acc_a0, wa01, x_f1, acc_a0);
                    simdgroup_multiply_accumulate(acc_a1, wa10, x_f0, acc_a1);
                    simdgroup_multiply_accumulate(acc_a1, wa11, x_f1, acc_a1);
                    simdgroup_barrier(mem_flags::mem_threadgroup);

                    float4 wb0, wb1;
                    unpack_nf_8cols(db, p.bits, k, tid, l_pair, cb32_s, wb0, wb1);
                    *(threadgroup half4*)(ws_sg + lane * 20u + wc) = half4(wb0 * wsb);
                    *(threadgroup half4*)(ws_sg + lane * 20u + wc + 4u) = half4(wb1 * wsb);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_half8x8 wb00, wb01, wb10, wb11;
                    simdgroup_load(wb00, ws_sg + 0u, 20);
                    simdgroup_load(wb01, ws_sg + 8u, 20);
                    simdgroup_load(wb10, ws_sg + 160u, 20);
                    simdgroup_load(wb11, ws_sg + 168u, 20);
                    simdgroup_multiply_accumulate(acc_b0, wb00, x_f0, acc_b0);
                    simdgroup_multiply_accumulate(acc_b0, wb01, x_f1, acc_b0);
                    simdgroup_multiply_accumulate(acc_b1, wb10, x_f0, acc_b1);
                    simdgroup_multiply_accumulate(acc_b1, wb11, x_f1, acc_b1);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    } else {
        const uint sg_stride = 32u + p.scale_group_chunks * chunk_b;
        for (uint sg = sg_id; sg < p.num_scale_groups; sg += 4u) {
            device const uchar* sg_ptr = tail_stream + ulong(sg) * ulong(sg_stride);
            float wsa = bf16_bytes_to_f32(sg_ptr + lane * 2u);
            uint c0 = sg * p.scale_group_chunks;
            uint c1 = min(c0 + p.scale_group_chunks, p.num_chunks);
            for (uint c = c0; c < c1; ++c) {
                uint sub_c = c - c0;
                device const uchar* da = sg_ptr + 32u + sub_c * chunk_b;
                for (uint k = 0u; k < 4u; ++k) {
                    uint col = c * 64u + k * 16u;
                    simdgroup_half8x8 x_f0, x_f1;
                    simdgroup_load(x_f0, x_rot_f16 + ulong(col) * 128ul, 128);
                    simdgroup_load(x_f1, x_rot_f16 + ulong(col + 8u) * 128ul, 128);

                    float4 wa0, wa1;
                    unpack_nf_8cols(da, p.bits, k, tid, l_pair, cb32_s, wa0, wa1);
                    *(threadgroup half4*)(ws_sg + lane * 20u + wc) = half4(wa0 * wsa);
                    *(threadgroup half4*)(ws_sg + lane * 20u + wc + 4u) = half4(wa1 * wsa);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                    simdgroup_half8x8 wa00, wa01, wa10, wa11;
                    simdgroup_load(wa00, ws_sg + 0u, 20);
                    simdgroup_load(wa01, ws_sg + 8u, 20);
                    simdgroup_load(wa10, ws_sg + 160u, 20);
                    simdgroup_load(wa11, ws_sg + 168u, 20);
                    simdgroup_multiply_accumulate(acc_a0, wa00, x_f0, acc_a0);
                    simdgroup_multiply_accumulate(acc_a0, wa01, x_f1, acc_a0);
                    simdgroup_multiply_accumulate(acc_a1, wa10, x_f0, acc_a1);
                    simdgroup_multiply_accumulate(acc_a1, wa11, x_f1, acc_a1);
                    simdgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
        }
    }

    simdgroup_store(acc_a0, &tg_buf[sg_id][0], 8);
    simdgroup_store(acc_a1, &tg_buf[sg_id][64], 8);
    simdgroup_store(acc_b0, &tg_buf[sg_id][128], 8);
    simdgroup_store(acc_b1, &tg_buf[sg_id][192], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint nt = min(8u, p.n_tok);
    bool second = (tid >= 16u);
    if (!(second && !has_b)) {
        uint row = sa * 16u + tid;
        if (row < p.rows) {
            float b_val = (p.has_bias != 0u) ? bias[row] : 0.0f;
            for (uint m = sg_id; m < nt; m += 4u) {
                uint off = tid * 8u + m;
                float sum = (tg_buf[0][off] + tg_buf[1][off] + tg_buf[2][off] + tg_buf[3][off]) * x_tok_scale[m] + b_val;
                ulong dst_idx = ulong(m) * ulong(p.rows) + ulong(row);
                if (p.has_residual != 0u) sum += residual[dst_idx];
                out[dst_idx] = sum;
            }
        }
    }
}

kernel void nf4_gemm_mma(
    device const uchar* pair_stream [[buffer(0)]],
    device const uchar* tail_stream [[buffer(1)]],
    device const half*  x_rot_f16   [[buffer(2)]],
    device const float* x_tok_scale [[buffer(3)]],
    device const float* bias        [[buffer(4)]],
    device const float* residual    [[buffer(5)]],
    device float*       out         [[buffer(6)]],
    constant Nf4MatParams& p        [[buffer(7)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint tid     [[thread_index_in_simdgroup]],
    uint tid_tg  [[thread_index_in_threadgroup]])
{
    const uint n_base = tg_pos.x * 128u;
    const uint pair_idx = tg_pos.y;
    const uint row_base = pair_idx * 32u;
    if (row_base >= p.rows || n_base >= p.n_tok) return;

    const uint n_stride = ((p.n_tok + 127u) >> 7u) << 7u;

    threadgroup half Ws[32][68];
    threadgroup half2 l_pair[256];
    threadgroup float cb32_s[32];
    init_nf_tables(p, tid_tg, l_pair, cb32_s);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 acc[4][4];
    for (uint mi = 0u; mi < 4u; ++mi) {
        for (uint ni = 0u; ni < 4u; ++ni) {
            acc[mi][ni] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        }
    }

    const uint side = sg_id >> 1u;
    const uint k_sub = sg_id & 1u;
    const uint st = pair_idx * 2u + side;
    const bool st_valid = (st < p.num_stripes);
    const bool st_tail = (pair_idx >= p.num_pairs);

    const uint g = tid >> 3u;
    const uint half16 = (tid >> 2u) & 1u;
    const uint j = tid & 3u;
    const uint lane = (g << 2u) | j;
    const uint m_row = side * 16u + lane;
    const uint w_col = k_sub * 16u + half16 * 8u;
    const uint sg_n = sg_id * 8u;
    const uint n_ni = min(4u, (p.n_tok - n_base + 31u) >> 5u);
    const uint chunk_b = p.chunk_bytes;
    const uint pair_chunk_b = 2u * chunk_b;

    const uint pair_sg_stride = 64u + p.scale_group_chunks * pair_chunk_b;
    const uint tail_sg_stride = 32u + p.scale_group_chunks * chunk_b;
    const device uchar* pair_base = pair_stream + ulong(pair_idx) * ulong(p.pair_total_bytes);

    uint sg = 0u;
    uint sub_c = 0u;
    float w_scale = 0.0f;
    device const uchar* sg_ptr = nullptr;

    for (uint c = 0u; c < p.num_chunks; ++c) {
        if (sub_c == 0u && st_valid) {
            if (!st_tail) {
                sg_ptr = pair_base + ulong(sg) * ulong(pair_sg_stride);
                w_scale = bf16_bytes_to_f32(sg_ptr + side * 32u + lane * 2u);
            } else {
                sg_ptr = tail_stream + ulong(sg) * ulong(tail_sg_stride);
                w_scale = bf16_bytes_to_f32(sg_ptr + lane * 2u);
            }
        }
        if (st_valid) {
            device const uchar* d_ptr = (!st_tail)
                ? (sg_ptr + 64u + sub_c * pair_chunk_b + side * chunk_b)
                : (sg_ptr + 32u + sub_c * chunk_b);
            for (uint hc = 0u; hc < 2u; ++hc) {
                float4 w0, w1;
                unpack_nf_8cols(d_ptr, p.bits, hc * 2u + k_sub, tid, l_pair, cb32_s, w0, w1);
                uint wc = hc * 32u + w_col;
                *(threadgroup half4*)(&Ws[m_row][wc]) = half4(w0 * w_scale);
                *(threadgroup half4*)(&Ws[m_row][wc + 4u]) = half4(w1 * w_scale);
            }
        } else {
            for (uint hc = 0u; hc < 2u; ++hc) {
                uint wc = hc * 32u + w_col;
                *(threadgroup half4*)(&Ws[m_row][wc]) = half4(0.0h);
                *(threadgroup half4*)(&Ws[m_row][wc + 4u]) = half4(0.0h);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        device const half* x_base = x_rot_f16 + ulong(c * 64u) * ulong(n_stride) + ulong(n_base + sg_n);
        if (n_ni == 4u) {
            for (uint ks = 0u; ks < 64u; ks += 8u) {
                simdgroup_half8x8 a_frag[4];
                simdgroup_half8x8 b_frag[4];
                #pragma clang loop unroll(full)
                for (uint mi = 0u; mi < 4u; ++mi) {
                    simdgroup_load(a_frag[mi], &Ws[mi * 8u][ks], 68);
                }
                device const half* x_ks = x_base + ulong(ks) * ulong(n_stride);
                #pragma clang loop unroll(full)
                for (uint ni = 0u; ni < 4u; ++ni) {
                    simdgroup_load(b_frag[ni], x_ks + ni * 32u, n_stride);
                }
                #pragma clang loop unroll(full)
                for (uint mi = 0u; mi < 4u; ++mi) {
                    #pragma clang loop unroll(full)
                    for (uint ni = 0u; ni < 4u; ++ni) {
                        simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                    }
                }
            }
        } else {
            for (uint ks = 0u; ks < 64u; ks += 8u) {
                simdgroup_half8x8 a_frag[4];
                #pragma clang loop unroll(full)
                for (uint mi = 0u; mi < 4u; ++mi) {
                    simdgroup_load(a_frag[mi], &Ws[mi * 8u][ks], 68);
                }
                device const half* x_ks = x_base + ulong(ks) * ulong(n_stride);
                #pragma clang loop unroll(full)
                for (uint ni = 0u; ni < 4u; ++ni) {
                    if (ni < n_ni) {
                        simdgroup_half8x8 b_frag;
                        simdgroup_load(b_frag, x_ks + ni * 32u, n_stride);
                        #pragma clang loop unroll(full)
                        for (uint mi = 0u; mi < 4u; ++mi) {
                            simdgroup_multiply_accumulate(acc[mi][ni], a_frag[mi], b_frag, acc[mi][ni]);
                        }
                    }
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        ++sub_c;
        if (sub_c == p.scale_group_chunks) {
            sub_c = 0u;
            ++sg;
        }
    }

    threadgroup float* cs_sg = ((threadgroup float*)&Ws[0][0]) + sg_id * 64u;
    const uint r_in = tid >> 2u;
    const uint c_in = (tid & 3u) * 2u;
    #pragma clang loop unroll(full)
    for (uint mi = 0u; mi < 4u; ++mi) {
        uint out_row = row_base + mi * 8u + r_in;
        float b_val = (p.has_bias != 0u && out_row < p.rows) ? bias[out_row] : 0.0f;
        #pragma clang loop unroll(full)
        for (uint ni = 0u; ni < 4u; ++ni) {
            if (ni < n_ni) {
                simdgroup_store(acc[mi][ni], cs_sg, 8);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                if (out_row < p.rows) {
                    for (uint sub = 0u; sub < 2u; ++sub) {
                        uint tok = n_base + ni * 32u + sg_n + c_in + sub;
                        if (tok < p.n_tok) {
                            float val = cs_sg[r_in * 8u + c_in + sub] * x_tok_scale[tok] + b_val;
                            ulong dst_idx = ulong(tok) * ulong(p.rows) + ulong(out_row);
                            if (p.has_residual != 0u) val += residual[dst_idx];
                            out[dst_idx] = val;
                        }
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

inline void store_q8_group32(
    float val,
    uint lane,
    device char* dst_q,
    device float* dst_scale)
{
    float m = simd_max(fabs(val));
    float sc = (m > 1e-12f) ? (m * (1.0f / 127.0f)) : 0.0f;
    float inv = (m > 1e-12f) ? (127.0f / m) : 0.0f;
    int q = clamp(int(rint(val * inv)), -127, 127);
    dst_q[lane] = char(q);
    if (lane == 0u) *dst_scale = sc;
}

inline void store_q4_group32(
    float val,
    uint lane,
    device uchar* dst_q4,
    device ushort* dst_scale_bf16)
{
    float r = simd_fwht32(val, lane);
    float m = simd_max(fabs(r));
    if (m <= 1e-12f) {
        if (lane < 16u) dst_q4[lane] = uchar(0x88u);
        if (lane == 0u) *dst_scale_bf16 = ushort(0u);
        return;
    }
    float inv0 = 127.0f / m;
    float c0 = KV_NF4_CB[kv_nf4_idx(r * inv0)];
    float num = simd_sum(r * c0);
    float den = simd_sum(c0 * c0);
    float s_ref = (den > 1e-12f && num > 0.0f) ? (num / den) : (m * (1.0f / 127.0f));
    ushort sb = f32_to_bf16_u16(s_ref);
    float sf = bf16_u16_to_f32(sb);
    float inv = (sf > 1e-20f) ? (1.0f / sf) : 0.0f;
    uint q_lo = kv_nf4_idx(r * inv);
    uint q_hi = simd_shuffle_xor(q_lo, ushort(16));
    if (lane < 16u) {
        dst_q4[lane] = uchar(q_lo | (q_hi << 4u));
    }
    if (lane == 0u) {
        *dst_scale_bf16 = sb;
    }
}

kernel void qkv_rope_kv_store(
    device float*       qkv      [[buffer(0)]],
    device const float* rope_tab [[buffer(1)]],
    device uchar*       k_raw    [[buffer(2)]],
    device uchar*       ks_raw   [[buffer(3)]],
    device uchar*       v_raw    [[buffer(4)]],
    device uchar*       vs_raw   [[buffer(5)]],
    constant RopeKvParams& p     [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    const uint t = tg_pos.y;
    const uint b = tg_pos.x * 4u + sg_id;
    const uint total_heads = p.num_q_heads + 2u * p.num_kv_heads;
    if (t >= p.n_tok || b >= total_heads) return;

    const uint hd = p.head_dim;
    const uint half_d = hd >> 1u;
    const uint gph = hd >> 5u;
    const uint pos = p.pos0 + t;
    device float* head = qkv + ulong(t) * ulong(p.qkv_rows) + ulong(b * hd);
    device const float* rope = rope_tab + ulong(pos) * ulong(hd);

    if (b < p.num_q_heads) {
        for (uint e = 0u; e < (half_d >> 5u); ++e) {
            uint i = e * 32u + lane;
            float x0 = head[i];
            float x1 = head[half_d + i];
            float c = rope[i];
            float s = rope[half_d + i];
            float rx0 = x0 * c - x1 * s;
            float rx1 = x1 * c + x0 * s;
            if (p.kv_bits == 4u) {
                rx0 = simd_fwht32(rx0, lane);
                rx1 = simd_fwht32(rx1, lane);
            }
            head[i] = rx0;
            head[half_d + i] = rx1;
        }
    } else if (b < p.num_q_heads + p.num_kv_heads) {
        uint kvh = b - p.num_q_heads;
        uint s_pos = kv_slot_of_pos(pos, p.sink_tokens, p.ring_len);
        ulong slot = (ulong(p.layer_idx * p.num_kv_heads + kvh) * ulong(p.kv_cap)) + ulong(s_pos);
        uint half_groups = half_d >> 5u;
        if (p.kv_bits == 4u) {
            device uchar* kq = k_raw + slot * ulong(half_d);
            device ushort* ks = ((device ushort*)ks_raw) + slot * ulong(gph);
            for (uint e = 0u; e < half_groups; ++e) {
                uint i = e * 32u + lane;
                float x0 = head[i];
                float x1 = head[half_d + i];
                float c = rope[i];
                float s = rope[half_d + i];
                float rx0 = x0 * c - x1 * s;
                float rx1 = x1 * c + x0 * s;
                head[i] = rx0;
                head[half_d + i] = rx1;
                store_q4_group32(rx0, lane, kq + e * 16u, ks + e);
                store_q4_group32(rx1, lane, kq + (half_groups + e) * 16u, ks + half_groups + e);
            }
        } else {
            device char* kq = ((device char*)k_raw) + slot * ulong(hd);
            device float* ks = ((device float*)ks_raw) + slot * ulong(gph);
            for (uint e = 0u; e < half_groups; ++e) {
                uint i = e * 32u + lane;
                float x0 = head[i];
                float x1 = head[half_d + i];
                float c = rope[i];
                float s = rope[half_d + i];
                float rx0 = x0 * c - x1 * s;
                float rx1 = x1 * c + x0 * s;
                head[i] = rx0;
                head[half_d + i] = rx1;
                store_q8_group32(rx0, lane, kq + e * 32u, ks + e);
                store_q8_group32(rx1, lane, kq + half_d + e * 32u, ks + half_groups + e);
            }
        }
    } else {
        uint kvh = b - p.num_q_heads - p.num_kv_heads;
        uint s_pos = kv_slot_of_pos(pos, p.sink_tokens, p.ring_len);
        ulong slot = (ulong(p.layer_idx * p.num_kv_heads + kvh) * ulong(p.kv_cap)) + ulong(s_pos);
        if (p.kv_bits == 4u) {
            device uchar* vq = v_raw + slot * ulong(half_d);
            device ushort* vs = ((device ushort*)vs_raw) + slot * ulong(gph);
            for (uint g = 0u; g < gph; ++g) {
                float v = head[g * 32u + lane];
                store_q4_group32(v, lane, vq + g * 16u, vs + g);
            }
        } else {
            device char* vq = ((device char*)v_raw) + slot * ulong(hd);
            device float* vs = ((device float*)vs_raw) + slot * ulong(gph);
            for (uint g = 0u; g < gph; ++g) {
                float v = head[g * 32u + lane];
                store_q8_group32(v, lane, vq + g * 32u, vs + g);
            }
        }
    }
}

kernel void causal_gqa_flash_q8_decode(
    device const float* qkv      [[buffer(0)]],
    device const uchar* k_raw    [[buffer(1)]],
    device const uchar* ks_raw   [[buffer(2)]],
    device const uchar* v_raw    [[buffer(3)]],
    device const uchar* vs_raw   [[buffer(4)]],
    device float*       attn_out [[buffer(5)]],
    constant GqaFlashParams& p   [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]])
{
    const uint qh = tg_pos.x;
    const uint t = tg_pos.y;
    if (qh >= p.num_q_heads || t >= p.n_tok) return;

    const uint hd = p.head_dim;
    const uint kvh = qh / p.kv_group;
    const uint seq_len = p.pos0 + t + 1u;
    const uint act_len = (p.window_size > 0u && seq_len > p.window_size) ? p.window_size : seq_len;
    const uint tail_start = (p.window_size > 0u && seq_len > p.window_size)
        ? (seq_len - (p.window_size - p.sink_tokens))
        : p.sink_tokens;
    const ulong head_base_slot = ulong(p.layer_idx * p.num_kv_heads + kvh) * ulong(p.kv_cap);
    device const float* q_ptr = qkv + ulong(t) * ulong(p.qkv_rows) + ulong(qh * hd);

    float q0 = q_ptr[lane] * p.attn_scale;
    float q1 = q_ptr[32u + lane] * p.attn_scale;
    float q2 = (hd == 128u) ? (q_ptr[64u + lane] * p.attn_scale) : 0.0f;
    float q3 = (hd == 128u) ? (q_ptr[96u + lane] * p.attn_scale) : 0.0f;

    float m_val = -1e30f;
    float l_val = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    if (p.kv_bits == 4u) {
        const uint b_idx = lane & 15u;
        const uint shift = (lane >> 4u) << 2u;
        device const ushort* ksp = (device const ushort*)ks_raw;
        device const ushort* vsp = (device const ushort*)vs_raw;
        if (hd == 128u) {
            for (uint idx = sg_id; idx < act_len; idx += 4u) {
                uint pos = (idx < p.sink_tokens) ? idx : (tail_start + idx - p.sink_tokens);
                uint s_pos = kv_slot_of_pos(pos, p.sink_tokens, p.ring_len);
                ulong slot = head_base_slot + ulong(s_pos);
                device const uchar* kp = k_raw + slot * 64ul;
                ushort4 ksb = *(device const ushort4*)(ksp + slot * 4ul);
                float4 ks = float4(
                    bf16_u16_to_f32(ksb.x),
                    bf16_u16_to_f32(ksb.y),
                    bf16_u16_to_f32(ksb.z),
                    bf16_u16_to_f32(ksb.w));
                float k0 = KV_NF4_CB[(uint(kp[b_idx]) >> shift) & 0xFu];
                float k1 = KV_NF4_CB[(uint(kp[16u + b_idx]) >> shift) & 0xFu];
                float k2 = KV_NF4_CB[(uint(kp[32u + b_idx]) >> shift) & 0xFu];
                float k3 = KV_NF4_CB[(uint(kp[48u + b_idx]) >> shift) & 0xFu];
                float dot_p = q0 * (k0 * ks.x) + q1 * (k1 * ks.y) + q2 * (k2 * ks.z) + q3 * (k3 * ks.w);
                float score = simd_sum(dot_p);
                float m_new = (l_val == 0.0f) ? score : max(m_val, score);
                float alpha = (l_val == 0.0f) ? 0.0f : exp(m_val - m_new);
                float pw = exp(score - m_new);
                l_val = l_val * alpha + pw;
                m_val = m_new;

                device const uchar* vp = v_raw + slot * 64ul;
                ushort4 vsb = *(device const ushort4*)(vsp + slot * 4ul);
                float4 vs = float4(
                    bf16_u16_to_f32(vsb.x),
                    bf16_u16_to_f32(vsb.y),
                    bf16_u16_to_f32(vsb.z),
                    bf16_u16_to_f32(vsb.w));
                o0 = o0 * alpha + (pw * vs.x) * KV_NF4_CB[(uint(vp[b_idx]) >> shift) & 0xFu];
                o1 = o1 * alpha + (pw * vs.y) * KV_NF4_CB[(uint(vp[16u + b_idx]) >> shift) & 0xFu];
                o2 = o2 * alpha + (pw * vs.z) * KV_NF4_CB[(uint(vp[32u + b_idx]) >> shift) & 0xFu];
                o3 = o3 * alpha + (pw * vs.w) * KV_NF4_CB[(uint(vp[48u + b_idx]) >> shift) & 0xFu];
            }
        } else {
            for (uint idx = sg_id; idx < act_len; idx += 4u) {
                uint pos = (idx < p.sink_tokens) ? idx : (tail_start + idx - p.sink_tokens);
                uint s_pos = kv_slot_of_pos(pos, p.sink_tokens, p.ring_len);
                ulong slot = head_base_slot + ulong(s_pos);
                device const uchar* kp = k_raw + slot * 32ul;
                ushort2 ksb = *(device const ushort2*)(ksp + slot * 2ul);
                float2 ks = float2(bf16_u16_to_f32(ksb.x), bf16_u16_to_f32(ksb.y));
                float k0 = KV_NF4_CB[(uint(kp[b_idx]) >> shift) & 0xFu];
                float k1 = KV_NF4_CB[(uint(kp[16u + b_idx]) >> shift) & 0xFu];
                float score = simd_sum(q0 * (k0 * ks.x) + q1 * (k1 * ks.y));
                float m_new = (l_val == 0.0f) ? score : max(m_val, score);
                float alpha = (l_val == 0.0f) ? 0.0f : exp(m_val - m_new);
                float pw = exp(score - m_new);
                l_val = l_val * alpha + pw;
                m_val = m_new;

                device const uchar* vp = v_raw + slot * 32ul;
                ushort2 vsb = *(device const ushort2*)(vsp + slot * 2ul);
                float2 vs = float2(bf16_u16_to_f32(vsb.x), bf16_u16_to_f32(vsb.y));
                o0 = o0 * alpha + (pw * vs.x) * KV_NF4_CB[(uint(vp[b_idx]) >> shift) & 0xFu];
                o1 = o1 * alpha + (pw * vs.y) * KV_NF4_CB[(uint(vp[16u + b_idx]) >> shift) & 0xFu];
            }
        }
    } else {
        device const char* k_q8 = (device const char*)k_raw;
        device const float* k_scales = (device const float*)ks_raw;
        device const char* v_q8 = (device const char*)v_raw;
        device const float* v_scales = (device const float*)vs_raw;
        if (hd == 128u) {
            for (uint idx = sg_id; idx < act_len; idx += 4u) {
                uint pos = (idx < p.sink_tokens) ? idx : (tail_start + idx - p.sink_tokens);
                uint s_pos = kv_slot_of_pos(pos, p.sink_tokens, p.ring_len);
                ulong slot = head_base_slot + ulong(s_pos);
                device const char* kp = k_q8 + slot * 128ul;
                float4 ks = *(device const float4*)(k_scales + slot * 4ul);
                float dot_p = q0 * (float(kp[lane]) * ks.x)
                            + q1 * (float(kp[32u + lane]) * ks.y)
                            + q2 * (float(kp[64u + lane]) * ks.z)
                            + q3 * (float(kp[96u + lane]) * ks.w);
                float score = simd_sum(dot_p);
                float m_new = (l_val == 0.0f) ? score : max(m_val, score);
                float alpha = (l_val == 0.0f) ? 0.0f : exp(m_val - m_new);
                float pw = exp(score - m_new);
                l_val = l_val * alpha + pw;
                m_val = m_new;

                device const char* vp = v_q8 + slot * 128ul;
                float4 vs = *(device const float4*)(v_scales + slot * 4ul);
                o0 = o0 * alpha + (pw * vs.x) * float(vp[lane]);
                o1 = o1 * alpha + (pw * vs.y) * float(vp[32u + lane]);
                o2 = o2 * alpha + (pw * vs.z) * float(vp[64u + lane]);
                o3 = o3 * alpha + (pw * vs.w) * float(vp[96u + lane]);
            }
        } else {
            for (uint idx = sg_id; idx < act_len; idx += 4u) {
                uint pos = (idx < p.sink_tokens) ? idx : (tail_start + idx - p.sink_tokens);
                uint s_pos = kv_slot_of_pos(pos, p.sink_tokens, p.ring_len);
                ulong slot = head_base_slot + ulong(s_pos);
                device const char* kp = k_q8 + slot * 64ul;
                float2 ks = *(device const float2*)(k_scales + slot * 2ul);
                float dot_p = q0 * (float(kp[lane]) * ks.x)
                            + q1 * (float(kp[32u + lane]) * ks.y);
                float score = simd_sum(dot_p);
                float m_new = (l_val == 0.0f) ? score : max(m_val, score);
                float alpha = (l_val == 0.0f) ? 0.0f : exp(m_val - m_new);
                float pw = exp(score - m_new);
                l_val = l_val * alpha + pw;
                m_val = m_new;

                device const char* vp = v_q8 + slot * 64ul;
                float2 vs = *(device const float2*)(v_scales + slot * 2ul);
                o0 = o0 * alpha + (pw * vs.x) * float(vp[lane]);
                o1 = o1 * alpha + (pw * vs.y) * float(vp[32u + lane]);
            }
        }
    }

    threadgroup float tg_m[4];
    threadgroup float tg_l[4];
    threadgroup float tg_o[4][128];
    if (lane == 0u) {
        tg_m[sg_id] = m_val;
        tg_l[sg_id] = l_val;
    }
    tg_o[sg_id][lane] = o0;
    tg_o[sg_id][32u + lane] = o1;
    if (hd == 128u) {
        tg_o[sg_id][64u + lane] = o2;
        tg_o[sg_id][96u + lane] = o3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg_id == 0u) {
        float m_glob = tg_m[0];
        if (tg_l[1] > 0.0f) m_glob = max(m_glob, tg_m[1]);
        if (tg_l[2] > 0.0f) m_glob = max(m_glob, tg_m[2]);
        if (tg_l[3] > 0.0f) m_glob = max(m_glob, tg_m[3]);
        float w0 = (tg_l[0] > 0.0f) ? exp(tg_m[0] - m_glob) : 0.0f;
        float w1 = (tg_l[1] > 0.0f) ? exp(tg_m[1] - m_glob) : 0.0f;
        float w2 = (tg_l[2] > 0.0f) ? exp(tg_m[2] - m_glob) : 0.0f;
        float w3 = (tg_l[3] > 0.0f) ? exp(tg_m[3] - m_glob) : 0.0f;
        float inv_l = 1.0f / (tg_l[0] * w0 + tg_l[1] * w1 + tg_l[2] * w2 + tg_l[3] * w3);

        device float* dst = attn_out + ulong(t) * ulong(p.h_dim) + ulong(qh * hd);
        float r0 = (tg_o[0][lane] * w0 + tg_o[1][lane] * w1 + tg_o[2][lane] * w2 + tg_o[3][lane] * w3) * inv_l;
        uint i1 = 32u + lane;
        float r1 = (tg_o[0][i1] * w0 + tg_o[1][i1] * w1 + tg_o[2][i1] * w2 + tg_o[3][i1] * w3) * inv_l;
        if (p.kv_bits == 4u) {
            r0 = simd_ifwht32(r0, lane);
            r1 = simd_ifwht32(r1, lane);
        }
        dst[lane] = r0;
        dst[i1] = r1;
        if (hd == 128u) {
            uint i2 = 64u + lane;
            uint i3 = 96u + lane;
            float r2 = (tg_o[0][i2] * w0 + tg_o[1][i2] * w1 + tg_o[2][i2] * w2 + tg_o[3][i2] * w3) * inv_l;
            float r3 = (tg_o[0][i3] * w0 + tg_o[1][i3] * w1 + tg_o[2][i3] * w2 + tg_o[3][i3] * w3) * inv_l;
            if (p.kv_bits == 4u) {
                r2 = simd_ifwht32(r2, lane);
                r3 = simd_ifwht32(r3, lane);
            }
            dst[i2] = r2;
            dst[i3] = r3;
        }
    }
}

kernel void causal_gqa_flash_q8_prefill(
    device const float* qkv      [[buffer(0)]],
    device const uchar* k_raw    [[buffer(1)]],
    device const uchar* ks_raw   [[buffer(2)]],
    device const uchar* v_raw    [[buffer(3)]],
    device const uchar* vs_raw   [[buffer(4)]],
    device float*       attn_out [[buffer(5)]],
    constant GqaFlashParams& p   [[buffer(6)]],
    uint2 tg_pos [[threadgroup_position_in_grid]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint tid_tg  [[thread_index_in_threadgroup]])
{
    const uint qh = tg_pos.x;
    const uint tg_t0 = tg_pos.y * 32u;
    if (qh >= p.num_q_heads || tg_t0 >= p.n_tok) return;

    const uint hd = p.head_dim;
    const uint n_kt = hd >> 3u;
    const uint n_dp = hd >> 4u;
    const uint kvh = qh / p.kv_group;
    const uint max_seq_len = p.pos0 + min(p.n_tok, tg_t0 + 32u);
    const uint min_row_seq = p.pos0 + tg_t0 + 1u;
    const uint gap_end = (p.window_size > 0u && min_row_seq > p.window_size)
        ? max(p.sink_tokens, min_row_seq - (p.window_size - p.sink_tokens))
        : p.sink_tokens;
    const uint gap_len = gap_end - p.sink_tokens;
    const uint total_union = max_seq_len - gap_len;
    const ulong head_base_slot = ulong(p.layer_idx * p.num_kv_heads + kvh) * ulong(p.kv_cap);

    const uint qr = lane >> 2u;
    const uint qc = lane & 3u;
    const uint c0 = qc * 4u;

    threadgroup half Ks[128][20];
    threadgroup half Vs[16][132];
    threadgroup half Ps[4][8][20];
    threadgroup float Sc[4][8][16];

    simdgroup_half8x8 q_frag[16];
    for (uint half_sg = 0u; half_sg < 2u; ++half_sg) {
        if ((sg_id >> 1u) == half_sg) {
            uint r_local = (sg_id & 1u) * 8u + qr;
            uint q_tok = tg_t0 + sg_id * 8u + qr;
            for (uint d = c0; d < hd; d += 16u) {
                half4 qv = half4(0.0h);
                if (q_tok < p.n_tok) {
                    float4 fq = *(device const float4*)(qkv + ulong(q_tok) * ulong(p.qkv_rows) + ulong(qh * hd + d));
                    qv = half4(fq * p.attn_scale);
                }
                *(threadgroup half4*)(&Vs[r_local][d]) = qv;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if ((sg_id >> 1u) == half_sg) {
            uint r_local = (sg_id & 1u) * 8u;
            #pragma clang loop unroll(full)
            for (uint kt = 0u; kt < 16u; ++kt) {
                if (kt < n_kt) {
                    simdgroup_load(q_frag[kt], &Vs[r_local][kt * 8u], 132);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float4 o_acc[8];
    #pragma clang loop unroll(full)
    for (uint dp = 0u; dp < 8u; ++dp) {
        o_acc[dp] = float4(0.0f);
    }
    float m_row = -1e30f;
    float l_row = 0.0f;

    const uint q_tok = tg_t0 + sg_id * 8u + qr;
    const uint row_seq_len = (q_tok < p.n_tok) ? (p.pos0 + q_tok + 1u) : 0u;
    const uint row_win_start = (p.window_size > 0u && row_seq_len > p.window_size)
        ? (row_seq_len - (p.window_size - p.sink_tokens))
        : p.sink_tokens;
    const uint kb = tid_tg >> 3u;
    const uint sub = tid_tg & 7u;

    for (uint u0 = 0u; u0 < total_union; u0 += 16u) {
        uint u_kv = u0 + kb;
        uint pos_kv = (u_kv < p.sink_tokens) ? u_kv : (u_kv + gap_len);
        uint s_pos = kv_slot_of_pos(pos_kv, p.sink_tokens, p.ring_len);
        if (hd == 128u) {
            uint d0 = sub * 16u;
            if (u_kv < total_union) {
                ulong slot = head_base_slot + ulong(s_pos);
                uint g32 = sub >> 1u;
                if (p.kv_bits == 4u) {
                    uint shift = (sub & 1u) * 4u;
                    float ks = bf16_u16_to_f32(((device const ushort*)ks_raw)[slot * 4ul + ulong(g32)]);
                    float vs = bf16_u16_to_f32(((device const ushort*)vs_raw)[slot * 4ul + ulong(g32)]);
                    device const uchar4* kp4 = (device const uchar4*)(k_raw + slot * 64ul + ulong(g32 * 16u));
                    device const uchar4* vp4 = (device const uchar4*)(v_raw + slot * 64ul + ulong(g32 * 16u));
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                        uchar4 kb4 = kp4[j];
                        uchar4 vb4 = vp4[j];
                        uint dj = d0 + j * 4u;
                        Ks[dj + 0u][kb] = half(KV_NF4_CB[(uint(kb4.x) >> shift) & 0xFu] * ks);
                        Ks[dj + 1u][kb] = half(KV_NF4_CB[(uint(kb4.y) >> shift) & 0xFu] * ks);
                        Ks[dj + 2u][kb] = half(KV_NF4_CB[(uint(kb4.z) >> shift) & 0xFu] * ks);
                        Ks[dj + 3u][kb] = half(KV_NF4_CB[(uint(kb4.w) >> shift) & 0xFu] * ks);
                        *(threadgroup half4*)(&Vs[kb][dj]) = half4(
                            half(KV_NF4_CB[(uint(vb4.x) >> shift) & 0xFu] * vs),
                            half(KV_NF4_CB[(uint(vb4.y) >> shift) & 0xFu] * vs),
                            half(KV_NF4_CB[(uint(vb4.z) >> shift) & 0xFu] * vs),
                            half(KV_NF4_CB[(uint(vb4.w) >> shift) & 0xFu] * vs));
                    }
                } else {
                    float ks = ((device const float*)ks_raw)[slot * 4ul + ulong(g32)];
                    float vs = ((device const float*)vs_raw)[slot * 4ul + ulong(g32)];
                    device const char4* kp4 = (device const char4*)(((device const char*)k_raw) + slot * 128ul + ulong(d0));
                    device const char4* vp4 = (device const char4*)(((device const char*)v_raw) + slot * 128ul + ulong(d0));
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 4u; ++j) {
                        char4 kq = kp4[j];
                        char4 vq = vp4[j];
                        uint dj = d0 + j * 4u;
                        Ks[dj + 0u][kb] = half(float(kq.x) * ks);
                        Ks[dj + 1u][kb] = half(float(kq.y) * ks);
                        Ks[dj + 2u][kb] = half(float(kq.z) * ks);
                        Ks[dj + 3u][kb] = half(float(kq.w) * ks);
                        *(threadgroup half4*)(&Vs[kb][dj]) = half4(
                            half(float(vq.x) * vs),
                            half(float(vq.y) * vs),
                            half(float(vq.z) * vs),
                            half(float(vq.w) * vs));
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 4u; ++j) {
                    uint dj = d0 + j * 4u;
                    Ks[dj + 0u][kb] = 0.0h;
                    Ks[dj + 1u][kb] = 0.0h;
                    Ks[dj + 2u][kb] = 0.0h;
                    Ks[dj + 3u][kb] = 0.0h;
                    *(threadgroup half4*)(&Vs[kb][dj]) = half4(0.0h);
                }
            }
        } else {
            uint d0 = sub * 8u;
            if (u_kv < total_union) {
                ulong slot = head_base_slot + ulong(s_pos);
                uint g32 = sub >> 2u;
                if (p.kv_bits == 4u) {
                    uint byte_off = (sub & 1u) * 8u;
                    uint shift = ((sub >> 1u) & 1u) * 4u;
                    float ks = bf16_u16_to_f32(((device const ushort*)ks_raw)[slot * 2ul + ulong(g32)]);
                    float vs = bf16_u16_to_f32(((device const ushort*)vs_raw)[slot * 2ul + ulong(g32)]);
                    device const uchar4* kp4 = (device const uchar4*)(k_raw + slot * 32ul + ulong(g32 * 16u + byte_off));
                    device const uchar4* vp4 = (device const uchar4*)(v_raw + slot * 32ul + ulong(g32 * 16u + byte_off));
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 2u; ++j) {
                        uchar4 kb4 = kp4[j];
                        uchar4 vb4 = vp4[j];
                        uint dj = d0 + j * 4u;
                        Ks[dj + 0u][kb] = half(KV_NF4_CB[(uint(kb4.x) >> shift) & 0xFu] * ks);
                        Ks[dj + 1u][kb] = half(KV_NF4_CB[(uint(kb4.y) >> shift) & 0xFu] * ks);
                        Ks[dj + 2u][kb] = half(KV_NF4_CB[(uint(kb4.z) >> shift) & 0xFu] * ks);
                        Ks[dj + 3u][kb] = half(KV_NF4_CB[(uint(kb4.w) >> shift) & 0xFu] * ks);
                        *(threadgroup half4*)(&Vs[kb][dj]) = half4(
                            half(KV_NF4_CB[(uint(vb4.x) >> shift) & 0xFu] * vs),
                            half(KV_NF4_CB[(uint(vb4.y) >> shift) & 0xFu] * vs),
                            half(KV_NF4_CB[(uint(vb4.z) >> shift) & 0xFu] * vs),
                            half(KV_NF4_CB[(uint(vb4.w) >> shift) & 0xFu] * vs));
                    }
                } else {
                    float ks = ((device const float*)ks_raw)[slot * 2ul + ulong(g32)];
                    float vs = ((device const float*)vs_raw)[slot * 2ul + ulong(g32)];
                    device const char4* kp4 = (device const char4*)(((device const char*)k_raw) + slot * 64ul + ulong(d0));
                    device const char4* vp4 = (device const char4*)(((device const char*)v_raw) + slot * 64ul + ulong(d0));
                    #pragma clang loop unroll(full)
                    for (uint j = 0u; j < 2u; ++j) {
                        char4 kq = kp4[j];
                        char4 vq = vp4[j];
                        uint dj = d0 + j * 4u;
                        Ks[dj + 0u][kb] = half(float(kq.x) * ks);
                        Ks[dj + 1u][kb] = half(float(kq.y) * ks);
                        Ks[dj + 2u][kb] = half(float(kq.z) * ks);
                        Ks[dj + 3u][kb] = half(float(kq.w) * ks);
                        *(threadgroup half4*)(&Vs[kb][dj]) = half4(
                            half(float(vq.x) * vs),
                            half(float(vq.y) * vs),
                            half(float(vq.z) * vs),
                            half(float(vq.w) * vs));
                    }
                }
            } else {
                #pragma clang loop unroll(full)
                for (uint j = 0u; j < 2u; ++j) {
                    uint dj = d0 + j * 4u;
                    Ks[dj + 0u][kb] = 0.0h;
                    Ks[dj + 1u][kb] = 0.0h;
                    Ks[dj + 2u][kb] = 0.0h;
                    Ks[dj + 3u][kb] = 0.0h;
                    *(threadgroup half4*)(&Vs[kb][dj]) = half4(0.0h);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 s0_mat = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        simdgroup_float8x8 s1_mat = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        #pragma clang loop unroll(full)
        for (uint kt = 0u; kt < 16u; ++kt) {
            if (kt < n_kt) {
                simdgroup_half8x8 kf0, kf1;
                simdgroup_load(kf0, &Ks[kt * 8u][0], 20);
                simdgroup_load(kf1, &Ks[kt * 8u][8], 20);
                simdgroup_multiply_accumulate(s0_mat, q_frag[kt], kf0, s0_mat);
                simdgroup_multiply_accumulate(s1_mat, q_frag[kt], kf1, s1_mat);
            }
        }
        simdgroup_store(s0_mat, &Sc[sg_id][0][0], 16);
        simdgroup_store(s1_mat, &Sc[sg_id][0][8], 16);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float4 sc = *(threadgroup const float4*)(&Sc[sg_id][qr][c0]);
        uint u0c = u0 + c0;
        uint kp0 = (u0c + 0u < p.sink_tokens) ? (u0c + 0u) : (u0c + 0u + gap_len);
        uint kp1 = (u0c + 1u < p.sink_tokens) ? (u0c + 1u) : (u0c + 1u + gap_len);
        uint kp2 = (u0c + 2u < p.sink_tokens) ? (u0c + 2u) : (u0c + 2u + gap_len);
        uint kp3 = (u0c + 3u < p.sink_tokens) ? (u0c + 3u) : (u0c + 3u + gap_len);
        bool4 valid = bool4(
            (u0c + 0u < total_union) && (kp0 < row_seq_len) && (kp0 < p.sink_tokens || kp0 >= row_win_start),
            (u0c + 1u < total_union) && (kp1 < row_seq_len) && (kp1 < p.sink_tokens || kp1 >= row_win_start),
            (u0c + 2u < total_union) && (kp2 < row_seq_len) && (kp2 < p.sink_tokens || kp2 >= row_win_start),
            (u0c + 3u < total_union) && (kp3 < row_seq_len) && (kp3 < p.sink_tokens || kp3 >= row_win_start));
        if (!valid.x) sc.x = -1e30f;
        if (!valid.y) sc.y = -1e30f;
        if (!valid.z) sc.z = -1e30f;
        if (!valid.w) sc.w = -1e30f;

        uint any_v = uint(valid.x || valid.y || valid.z || valid.w);
        any_v |= simd_shuffle_xor(any_v, ushort(1));
        any_v |= simd_shuffle_xor(any_v, ushort(2));

        float rmax = max(max(sc.x, sc.y), max(sc.z, sc.w));
        rmax = max(rmax, simd_shuffle_xor(rmax, ushort(1)));
        rmax = max(rmax, simd_shuffle_xor(rmax, ushort(2)));

        float alpha = 1.0f;
        float4 pw = float4(0.0f);
        if (any_v != 0u) {
            float m_new = (l_row == 0.0f) ? rmax : max(m_row, rmax);
            alpha = (l_row == 0.0f) ? 0.0f : exp(m_row - m_new);
            pw = float4(
                valid.x ? exp(sc.x - m_new) : 0.0f,
                valid.y ? exp(sc.y - m_new) : 0.0f,
                valid.z ? exp(sc.z - m_new) : 0.0f,
                valid.w ? exp(sc.w - m_new) : 0.0f);
            float psum = pw.x + pw.y + pw.z + pw.w;
            psum += simd_shuffle_xor(psum, ushort(1));
            psum += simd_shuffle_xor(psum, ushort(2));
            l_row = l_row * alpha + psum;
            m_row = m_new;
        }
        *(threadgroup half4*)(&Ps[sg_id][qr][c0]) = half4(pw);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 pf0, pf1;
        simdgroup_load(pf0, &Ps[sg_id][0][0], 20);
        simdgroup_load(pf1, &Ps[sg_id][0][8], 20);

        #pragma clang loop unroll(full)
        for (uint dp = 0u; dp < 8u; ++dp) {
            if (dp < n_dp) {
                uint d0 = dp * 16u;
                simdgroup_half8x8 v00, v01, v10, v11;
                simdgroup_load(v00, &Vs[0][d0], 132);
                simdgroup_load(v01, &Vs[0][d0 + 8u], 132);
                simdgroup_load(v10, &Vs[8][d0], 132);
                simdgroup_load(v11, &Vs[8][d0 + 8u], 132);
                simdgroup_float8x8 pv0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_float8x8 pv1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                simdgroup_multiply_accumulate(pv0, pf0, v00, pv0);
                simdgroup_multiply_accumulate(pv0, pf1, v10, pv0);
                simdgroup_multiply_accumulate(pv1, pf0, v01, pv1);
                simdgroup_multiply_accumulate(pv1, pf1, v11, pv1);
                simdgroup_store(pv0, &Sc[sg_id][0][0], 16);
                simdgroup_store(pv1, &Sc[sg_id][0][8], 16);
                simdgroup_barrier(mem_flags::mem_threadgroup);
                float4 dv = *(threadgroup const float4*)(&Sc[sg_id][qr][c0]);
                o_acc[dp] = o_acc[dp] * alpha + dv;
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (p.kv_bits == 4u) {
        const uint n_g32 = hd >> 5u;
        for (uint g = 0u; g < n_g32; ++g) {
            ifwht32_pair4(o_acc[2u * g], o_acc[2u * g + 1u], qc);
        }
    }

    if (q_tok < p.n_tok && l_row > 0.0f) {
        float inv_l = 1.0f / l_row;
        device float* dst = attn_out + ulong(q_tok) * ulong(p.h_dim) + ulong(qh * hd);
        #pragma clang loop unroll(full)
        for (uint dp = 0u; dp < 8u; ++dp) {
            if (dp < n_dp) {
                *(device float4*)(dst + dp * 16u + c0) = o_acc[dp] * inv_l;
            }
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use crate::real_llm::{argmax, RealLlamaEngine, RealWeightMode};
    use crate::simd_lut::f32_to_bf16;
    use std::fs;
    use std::path::Path;

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

    fn kl_divergence_nats(ref_logits: &[f32], cand_logits: &[f32]) -> f64 {
        let lse = |s: &[f32]| -> f64 {
            let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = s.iter().map(|&v| ((v as f64) - m).exp()).sum();
            m + sum.ln()
        };
        let ref_lse = lse(ref_logits);
        let cand_lse = lse(cand_logits);
        let mut kl = 0.0f64;
        for (&rl, &cl) in ref_logits.iter().zip(cand_logits.iter()) {
            let log_p = (rl as f64) - ref_lse;
            if log_p > -40.0 {
                let p = log_p.exp();
                let log_q = (cl as f64) - cand_lse;
                kl += p * (log_p - log_q);
            }
        }
        kl.max(0.0)
    }

    #[test]
    fn test_metal_nf4_decode_batch_and_mma_prefill_parity() {
        for (hd, nh, nkv, vocab) in [(64usize, 4usize, 2usize, 136usize), (128usize, 4usize, 2usize, 152usize)] {
            let h = nh * hd;
            let inter = h * 2;
            let nl = 2usize;
            let kvd = nkv * hd;
            let dir = std::env::temp_dir().join(format!(
                "kronos_metal_test_hd{hd}_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();

            fs::write(
                dir.join("config.json"),
                format!(
                    r#"{{"vocab_size":{vocab},"hidden_size":{h},"intermediate_size":{inter},"num_hidden_layers":{nl},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":false}}"#
                ),
            )
            .unwrap();

            let mk_bf16 = |n: usize, seed: usize, scale: f32| -> Vec<u8> {
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let mut z = (i as u64)
                        .wrapping_add((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
                    let v = (2.0 * u - 1.0) * scale;
                    out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
                }
                out
            };
            let write_safetensors = |path: &Path, entries: Vec<(String, Vec<usize>, Vec<u8>)>| {
                let mut data = Vec::new();
                let mut hdr_items = Vec::new();
                for (name, shape, raw) in entries {
                    let s = data.len();
                    data.extend_from_slice(&raw);
                    let e = data.len();
                    let sh_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                    hdr_items.push(format!(
                        r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{s},{e}]}}"#,
                        sh_str.join(",")
                    ));
                }
                let hdr = format!("{{{}}}", hdr_items.join(","));
                let mut file_bytes = (hdr.len() as u64).to_le_bytes().to_vec();
                file_bytes.extend_from_slice(hdr.as_bytes());
                file_bytes.extend_from_slice(&data);
                fs::write(path, file_bytes).unwrap();
            };

            let mut tensors = vec![(
                "model.embed_tokens.weight".to_string(),
                vec![vocab, h],
                mk_bf16(vocab * h, 31, 0.4),
            )];
            for l in 0..nl {
                let p = format!("model.layers.{l}");
                tensors.extend([
                    (format!("{p}.input_layernorm.weight"), vec![h], mk_bf16(h, 7 + l, 1.0)),
                    (format!("{p}.post_attention_layernorm.weight"), vec![h], mk_bf16(h, 11 + l, 1.0)),
                    (format!("{p}.self_attn.q_proj.weight"), vec![h, h], mk_bf16(h * h, 13 + l, 0.08)),
                    (format!("{p}.self_attn.k_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 17 + l, 0.08)),
                    (format!("{p}.self_attn.v_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 19 + l, 0.08)),
                    (format!("{p}.self_attn.q_proj.bias"), vec![h], mk_bf16(h, 23 + l, 0.25)),
                    (format!("{p}.self_attn.k_proj.bias"), vec![kvd], mk_bf16(kvd, 29 + l, 0.25)),
                    (format!("{p}.self_attn.v_proj.bias"), vec![kvd], mk_bf16(kvd, 37 + l, 0.25)),
                    (format!("{p}.self_attn.o_proj.weight"), vec![h, h], mk_bf16(h * h, 41 + l, 0.08)),
                    (format!("{p}.mlp.gate_proj.weight"), vec![inter, h], mk_bf16(inter * h, 43 + l, 0.08)),
                    (format!("{p}.mlp.up_proj.weight"), vec![inter, h], mk_bf16(inter * h, 47 + l, 0.08)),
                    (format!("{p}.mlp.down_proj.weight"), vec![h, inter], mk_bf16(h * inter, 53 + l, 0.055)),
                ]);
            }
            tensors.push(("model.norm.weight".to_string(), vec![h], mk_bf16(h, 59, 1.0)));
            tensors.push(("lm_head.weight".to_string(), vec![vocab, h], mk_bf16(vocab * h, 61, 0.3)));
            write_safetensors(&dir.join("model.safetensors"), tensors);

            let mode = RealWeightMode::KronosNfGptq {
                bits: 4,
                scale_group_chunks: 2,
            };
            let calib = vec![
                (0..16).map(|i| (i * 5 + 1) % vocab).collect::<Vec<_>>(),
                (0..16).map(|i| (i * 7 + 3) % vocab).collect::<Vec<_>>(),
            ];
            let compiled = RealLlamaEngine::compile_streaming_gptq_from_dir(&dir, mode, &calib, 2).unwrap();
            let cache_path = dir.join("kronos-nf4g128.k2tb");
            compiled.save_bipolar(&cache_path).unwrap();

            let mut cpu_eng = RealLlamaEngine::load_bipolar(&dir, &cache_path, 256).unwrap();
            cpu_eng.repack_sdot();
            cpu_eng.enable_q8_kv_cache(true);
            cpu_eng.set_num_threads(2);

            let mut gpu_eng = RealLlamaEngine::load_bipolar(&dir, &cache_path, 256).unwrap();
            gpu_eng.repack_sdot();
            gpu_eng.enable_metal().expect("Metal GPU unavailable on macOS test host");
            assert!(gpu_eng.is_metal_enabled());

            // 1. MMA prefill path (`n_tok = 140 > 128`, 2 BN=128 tiles + odd tail stripe on `lm_head`).
            let prefill_toks: Vec<usize> = (0..140).map(|i| (i * 11 + 2) % vocab).collect();
            let cpu_p = cpu_eng.forward_prompt(&prefill_toks).to_vec();
            let gpu_p = gpu_eng.forward_prompt(&prefill_toks).to_vec();
            assert!(
                gpu_p.iter().all(|v| v.is_finite()),
                "non-finite logits in Metal MMA prefill (hd={hd})"
            );
            let cos_p = cosine_sim(&cpu_p, &gpu_p);
            let kl_p = kl_divergence_nats(&cpu_p, &gpu_p);
            assert!(
                cos_p > 0.9995 && kl_p < 0.005,
                "MMA prefill mismatch (hd={hd}): cos={cos_p:.6}, kl={kl_p:.6}"
            );
            assert_eq!(argmax(&cpu_p), argmax(&gpu_p));

            // 2. Batched speculative verification path (`n_tok = 5 <= GEMV_MAX_TOKENS`).
            let batch_toks: Vec<usize> = (0..5).map(|i| (i * 13 + 7) % vocab).collect();
            let mut cpu_b = vec![0.0f32; 5 * vocab];
            let mut gpu_b = vec![0.0f32; 5 * vocab];
            cpu_eng.forward_batch_logits(&batch_toks, &mut cpu_b);
            gpu_eng.forward_batch_logits(&batch_toks, &mut gpu_b);
            for t in 0..5 {
                let cs = &cpu_b[t * vocab..(t + 1) * vocab];
                let gs = &gpu_b[t * vocab..(t + 1) * vocab];
                let cos_b = cosine_sim(cs, gs);
                let kl_b = kl_divergence_nats(cs, gs);
                assert!(
                    cos_b > 0.9995 && kl_b < 0.005,
                    "batch verify token {t} mismatch (hd={hd}): cos={cos_b:.6}, kl={kl_b:.6}"
                );
            }

            // 3. Single-token decode path (`n_tok = 1`).
            for step in 0..4 {
                let tok = (step * 17 + 9) % vocab;
                let cl = cpu_eng.forward_token(tok).to_vec();
                let gl = gpu_eng.forward_token(tok).to_vec();
                let cos_d = cosine_sim(&cl, &gl);
                let kl_d = kl_divergence_nats(&cl, &gl);
                assert!(
                    cos_d > 0.9995 && kl_d < 0.005,
                    "decode1 step {step} mismatch (hd={hd}): cos={cos_d:.6}, kl={kl_d:.6}"
                );
                let ca = argmax(&cl);
                let ga = argmax(&gl);
                assert!(
                    (cl[ca] - cl[ga]).abs() < 0.1,
                    "decode1 step {step} top logit gap too large: cl[{ca}]={} vs cl[{ga}]={}",
                    cl[ca],
                    cl[ga]
                );
            }

            // 4. Follow-up turn MMA prefill (`n_tok = 32`, `pos0 > 0` non-multiple of 16).
            let follow_toks: Vec<usize> = (0..32).map(|i| (i * 19 + 5) % vocab).collect();
            let cpu_f = cpu_eng.forward_prompt(&follow_toks).to_vec();
            let gpu_f = gpu_eng.forward_prompt(&follow_toks).to_vec();
            let cos_f = cosine_sim(&cpu_f, &gpu_f);
            let kl_f = kl_divergence_nats(&cpu_f, &gpu_f);
            assert!(
                cos_f > 0.9995 && kl_f < 0.005,
                "follow-up n_tok=32 MMA prefill mismatch (hd={hd}): cos={cos_f:.6}, kl={kl_f:.6}"
            );

            cpu_eng.set_num_threads(1);
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn test_metal_q4_kv_and_sliding_window_sink_parity() {
        for (hd, nh, nkv, vocab) in [(64usize, 4usize, 2usize, 136usize), (128usize, 4usize, 2usize, 152usize)] {
            let h = nh * hd;
            let inter = h * 2;
            let nl = 2usize;
            let kvd = nkv * hd;
            let dir = std::env::temp_dir().join(format!(
                "kronos_metal_q4win_hd{hd}_{}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();

            fs::write(
                dir.join("config.json"),
                format!(
                    r#"{{"vocab_size":{vocab},"hidden_size":{h},"intermediate_size":{inter},"num_hidden_layers":{nl},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":false}}"#
                ),
            )
            .unwrap();

            let mk_bf16 = |n: usize, seed: usize, scale: f32| -> Vec<u8> {
                let mut out = Vec::with_capacity(n * 2);
                for i in 0..n {
                    let mut z = (i as u64)
                        .wrapping_add((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                    z ^= z >> 31;
                    let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
                    let v = (2.0 * u - 1.0) * scale;
                    out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
                }
                out
            };
            let write_safetensors = |path: &Path, entries: Vec<(String, Vec<usize>, Vec<u8>)>| {
                let mut data = Vec::new();
                let mut hdr_items = Vec::new();
                for (name, shape, raw) in entries {
                    let s = data.len();
                    data.extend_from_slice(&raw);
                    let e = data.len();
                    let sh_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                    hdr_items.push(format!(
                        r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{s},{e}]}}"#,
                        sh_str.join(",")
                    ));
                }
                let hdr = format!("{{{}}}", hdr_items.join(","));
                let mut file_bytes = (hdr.len() as u64).to_le_bytes().to_vec();
                file_bytes.extend_from_slice(hdr.as_bytes());
                file_bytes.extend_from_slice(&data);
                fs::write(path, file_bytes).unwrap();
            };

            let mut tensors = vec![(
                "model.embed_tokens.weight".to_string(),
                vec![vocab, h],
                mk_bf16(vocab * h, 31, 0.4),
            )];
            for l in 0..nl {
                let p = format!("model.layers.{l}");
                tensors.extend([
                    (format!("{p}.input_layernorm.weight"), vec![h], mk_bf16(h, 7 + l, 1.0)),
                    (format!("{p}.post_attention_layernorm.weight"), vec![h], mk_bf16(h, 11 + l, 1.0)),
                    (format!("{p}.self_attn.q_proj.weight"), vec![h, h], mk_bf16(h * h, 13 + l, 0.08)),
                    (format!("{p}.self_attn.k_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 17 + l, 0.08)),
                    (format!("{p}.self_attn.v_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 19 + l, 0.08)),
                    (format!("{p}.self_attn.o_proj.weight"), vec![h, h], mk_bf16(h * h, 41 + l, 0.08)),
                    (format!("{p}.mlp.gate_proj.weight"), vec![inter, h], mk_bf16(inter * h, 43 + l, 0.08)),
                    (format!("{p}.mlp.up_proj.weight"), vec![inter, h], mk_bf16(inter * h, 47 + l, 0.08)),
                    (format!("{p}.mlp.down_proj.weight"), vec![h, inter], mk_bf16(h * inter, 53 + l, 0.055)),
                ]);
            }
            tensors.push(("model.norm.weight".to_string(), vec![h], mk_bf16(h, 59, 1.0)));
            tensors.push(("lm_head.weight".to_string(), vec![vocab, h], mk_bf16(vocab * h, 61, 0.3)));
            write_safetensors(&dir.join("model.safetensors"), tensors);

            let mode = RealWeightMode::KronosNfGptq {
                bits: 4,
                scale_group_chunks: 2,
            };
            let calib = vec![(0..16).map(|i| (i * 5 + 1) % vocab).collect::<Vec<_>>()];
            let mut compiled = RealLlamaEngine::compile_streaming_gptq_from_dir(&dir, mode, &calib, 2).unwrap();
            compiled.compress_untied_embeddings_nf4();
            let cache_path = dir.join("kronos-nf4g128.k2tb");
            compiled.save_bipolar(&cache_path).unwrap();

            let mut cpu_q8 = RealLlamaEngine::load_bipolar(&dir, &cache_path, 4096).unwrap();
            cpu_q8.repack_sdot();
            cpu_q8.set_kv_bits(8);
            cpu_q8.set_num_threads(2);

            let mut gpu_eng = RealLlamaEngine::load_bipolar(&dir, &cache_path, 4096).unwrap();
            gpu_eng.repack_sdot();
            gpu_eng.enable_metal().unwrap();
            gpu_eng.set_kv_bits(8);
            let q8_bytes = gpu_eng.kv_cache_bytes();
            gpu_eng.set_kv_bits(4);
            let q4_bytes = gpu_eng.kv_cache_bytes();
            assert_eq!(q4_bytes * 2, q8_bytes, "Metal Q4-KV must be exact 2.0x smaller than Q8-KV");

            let mut gpu_step = RealLlamaEngine::load_bipolar(&dir, &cache_path, 4096).unwrap();
            gpu_step.repack_sdot();
            gpu_step.enable_metal().unwrap();
            gpu_step.set_kv_bits(4);

            // 1. Q4-KV MMA prefill (`n_tok = 68`) vs Metal decode1 (`cos > 0.9998`) and CPU Q8 reference (`cos > 0.998`).
            let prefill_toks: Vec<usize> = (0..68).map(|i| (i * 11 + 2) % vocab).collect();
            let cp = cpu_q8.forward_prompt(&prefill_toks).to_vec();
            let gp = gpu_eng.forward_prompt(&prefill_toks).to_vec();
            let mut gs_last = Vec::new();
            for &t in &prefill_toks {
                gs_last = gpu_step.forward_token(t).to_vec();
            }
            let cos_self = cosine_sim(&gp, &gs_last);
            assert!(
                cos_self > 0.9997,
                "Metal Q4-KV MMA prefill vs decode1 mismatch (hd={hd}): cos={cos_self:.6}"
            );
            let cos_p = cosine_sim(&cp, &gp);
            let kl_p = kl_divergence_nats(&cp, &gp);
            assert!(
                cos_p > 0.998 && kl_p < 0.015,
                "Metal Q4-KV MMA prefill vs CPU Q8-KV mismatch (hd={hd}): cos={cos_p:.6}, kl={kl_p:.6}"
            );

            let batch_toks: Vec<usize> = (0..5).map(|i| (i * 13 + 7) % vocab).collect();
            let mut cb = vec![0.0f32; 5 * vocab];
            let mut gb = vec![0.0f32; 5 * vocab];
            cpu_q8.forward_batch_logits(&batch_toks, &mut cb);
            gpu_eng.forward_batch_logits(&batch_toks, &mut gb);
            for (t, &bt) in batch_toks.iter().enumerate() {
                let cs = &cb[t * vocab..(t + 1) * vocab];
                let gs = &gb[t * vocab..(t + 1) * vocab];
                let gs_step = gpu_step.forward_token(bt);
                let cos_self_b = cosine_sim(gs, gs_step);
                assert!(
                    cos_self_b > 0.9997,
                    "Metal Q4-KV batch vs decode1 token {t} mismatch (hd={hd}): cos={cos_self_b:.6}"
                );
                let cos_b = cosine_sim(cs, gs);
                let kl_b = kl_divergence_nats(cs, gs);
                assert!(
                    cos_b > 0.997 && kl_b < 0.015,
                    "Metal Q4-KV batch verify token {t} mismatch (hd={hd}): cos={cos_b:.6}, kl={kl_b:.6}"
                );
            }

            for step in 0..3 {
                let tok = (step * 17 + 9) % vocab;
                let cl = cpu_q8.forward_token(tok).to_vec();
                let gl = gpu_eng.forward_token(tok).to_vec();
                let gs_step = gpu_step.forward_token(tok);
                assert!(
                    cosine_sim(&gl, gs_step) > 0.9999,
                    "Metal Q4-KV decode1 determinism mismatch (hd={hd})"
                );
                let cos_d = cosine_sim(&cl, &gl);
                assert!(
                    cos_d > 0.997,
                    "Metal Q4-KV decode1 step {step} mismatch (hd={hd}): cos={cos_d:.6}"
                );
            }

            // 2. Sliding-window (`W=24`, `sinks=4`) + Q4-KV across ring wraparound (`1650` tokens).
            cpu_q8.set_kv_sliding_window(Some(24), 4);
            gpu_eng.set_kv_sliding_window(Some(24), 4);
            gpu_step.set_kv_sliding_window(Some(24), 4);
            let ring_toks: Vec<usize> = (0..1650).map(|i| (i * 19 + 3) % vocab).collect();
            let cw = cpu_q8.forward_prompt(&ring_toks).to_vec();
            let gw = gpu_eng.forward_prompt(&ring_toks).to_vec();
            let mut gsw_last = Vec::new();
            for &t in &ring_toks {
                gsw_last = gpu_step.forward_token(t).to_vec();
            }
            let cos_self_w = cosine_sim(&gw, &gsw_last);
            assert!(
                cos_self_w > 0.9997,
                "Metal Q4-KV sliding-window MMA prefill vs decode1 mismatch (hd={hd}): cos={cos_self_w:.6}"
            );
            let cos_w = cosine_sim(&cw, &gw);
            let kl_w = kl_divergence_nats(&cw, &gw);
            assert!(
                cos_w > 0.996 && kl_w < 0.015,
                "Metal Q4-KV sliding-window wraparound prefill mismatch (hd={hd}): cos={cos_w:.6}, kl={kl_w:.6}"
            );
            assert!(gpu_eng.can_truncate_kv_to(0));
            assert!(!gpu_eng.can_truncate_kv_to(8));
            assert!(gpu_eng.can_truncate_kv_to(1648));

            for step in 0..3 {
                let tok = (step * 23 + 1) % vocab;
                let cl = cpu_q8.forward_token(tok).to_vec();
                let gl = gpu_eng.forward_token(tok).to_vec();
                let gs = gpu_step.forward_token(tok);
                let cos_self_dw = cosine_sim(&gl, gs);
                assert!(
                    cos_self_dw > 0.9997,
                    "Metal Q4-KV sliding-window decode1 self-consistency mismatch (hd={hd}): cos={cos_self_dw:.6}"
                );
                let cos_dw = cosine_sim(&cl, &gl);
                let kl_dw = kl_divergence_nats(&cl, &gl);
                assert!(
                    cos_dw > 0.996 && kl_dw < 0.015,
                    "Metal Q4-KV sliding-window decode1 step {step} mismatch (hd={hd}): cos={cos_dw:.6}, kl={kl_dw:.6}"
                );
            }

            cpu_q8.set_num_threads(1);
            let _ = fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn test_metal_e8q2_nf3_nf5_and_mixed_parity() {
        let hd = 64usize;
        let nh = 4usize;
        let nkv = 2usize;
        let vocab = 136usize; // 9 stripes (4 pairs + 1 odd tail stripe)
        let h = nh * hd;
        let inter = h * 2;
        let nl = 2usize;
        let kvd = nkv * hd;
        let dir = std::env::temp_dir().join(format!(
            "kronos_metal_lowbit_{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"vocab_size":{vocab},"hidden_size":{h},"intermediate_size":{inter},"num_hidden_layers":{nl},"num_attention_heads":{nh},"num_key_value_heads":{nkv},"rms_norm_eps":1e-6,"rope_theta":1000000.0,"tie_word_embeddings":false}}"#
            ),
        )
        .unwrap();

        let mk_bf16 = |n: usize, seed: usize, scale: f32| -> Vec<u8> {
            let mut out = Vec::with_capacity(n * 2);
            for i in 0..n {
                let mut z = (i as u64)
                    .wrapping_add((seed as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = ((z >> 40) as f32) / ((1u32 << 24) as f32);
                let v = (2.0 * u - 1.0) * scale;
                out.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
            }
            out
        };
        let write_safetensors = |path: &Path, entries: Vec<(String, Vec<usize>, Vec<u8>)>| {
            let mut data = Vec::new();
            let mut hdr_items = Vec::new();
            for (name, shape, raw) in entries {
                let s = data.len();
                data.extend_from_slice(&raw);
                let e = data.len();
                let sh_str: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
                hdr_items.push(format!(
                    r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{s},{e}]}}"#,
                    sh_str.join(",")
                ));
            }
            let hdr = format!("{{{}}}", hdr_items.join(","));
            let mut file_bytes = (hdr.len() as u64).to_le_bytes().to_vec();
            file_bytes.extend_from_slice(hdr.as_bytes());
            file_bytes.extend_from_slice(&data);
            fs::write(path, file_bytes).unwrap();
        };

        let mut tensors = vec![(
            "model.embed_tokens.weight".to_string(),
            vec![vocab, h],
            mk_bf16(vocab * h, 31, 0.4),
        )];
        for l in 0..nl {
            let p = format!("model.layers.{l}");
            tensors.extend([
                (format!("{p}.input_layernorm.weight"), vec![h], mk_bf16(h, 7 + l, 1.0)),
                (format!("{p}.post_attention_layernorm.weight"), vec![h], mk_bf16(h, 11 + l, 1.0)),
                (format!("{p}.self_attn.q_proj.weight"), vec![h, h], mk_bf16(h * h, 13 + l, 0.08)),
                (format!("{p}.self_attn.k_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 17 + l, 0.08)),
                (format!("{p}.self_attn.v_proj.weight"), vec![kvd, h], mk_bf16(kvd * h, 19 + l, 0.08)),
                (format!("{p}.self_attn.o_proj.weight"), vec![h, h], mk_bf16(h * h, 41 + l, 0.08)),
                (format!("{p}.mlp.gate_proj.weight"), vec![inter, h], mk_bf16(inter * h, 43 + l, 0.08)),
                (format!("{p}.mlp.up_proj.weight"), vec![inter, h], mk_bf16(inter * h, 47 + l, 0.08)),
                (format!("{p}.mlp.down_proj.weight"), vec![h, inter], mk_bf16(h * inter, 53 + l, 0.055)),
            ]);
        }
        tensors.push(("model.norm.weight".to_string(), vec![h], mk_bf16(h, 59, 1.0)));
        tensors.push(("lm_head.weight".to_string(), vec![vocab, h], mk_bf16(vocab * h, 61, 0.3)));
        write_safetensors(&dir.join("model.safetensors"), tensors);

        let calib = vec![(0..16).map(|i| (i * 5 + 1) % vocab).collect::<Vec<_>>()];
        let modes = [
            (
                "e8q2g128",
                RealWeightMode::KronosNfGptq {
                    bits: 2,
                    scale_group_chunks: 2,
                },
            ),
            (
                "nf3g128",
                RealWeightMode::KronosNfGptq {
                    bits: 3,
                    scale_group_chunks: 2,
                },
            ),
            (
                "e8q2nf4g128",
                RealWeightMode::KronosNfMixedGptq {
                    attn_head_bits: 4,
                    ffn_bits: 2,
                    scale_group_chunks: 2,
                },
            ),
            (
                "nfg54324",
                RealWeightMode::KronosNfPlanesGptq {
                    qkv: 5,
                    o: 4,
                    gate_up: 3,
                    down: 2,
                    head: 4,
                    scale_group_chunks: 2,
                },
            ),
        ];

        let prefill_toks: Vec<usize> = (0..68).map(|i| (i * 11 + 2) % vocab).collect();
        let batch_toks: Vec<usize> = (0..5).map(|i| (i * 13 + 7) % vocab).collect();

        for (tag, mode) in modes {
            let mut compiled =
                RealLlamaEngine::compile_streaming_gptq_from_dir(&dir, mode, &calib, 2).unwrap();
            compiled.compress_untied_embeddings_nf4();
            let cache_path = dir.join(format!("kronos-{tag}.k2tb"));
            compiled.save_bipolar(&cache_path).unwrap();

            let mut cpu = RealLlamaEngine::load_bipolar(&dir, &cache_path, 512).unwrap();
            cpu.repack_sdot();
            cpu.set_kv_bits(8);
            cpu.set_num_threads(2);

            let mut gpu = RealLlamaEngine::load_bipolar(&dir, &cache_path, 512).unwrap();
            gpu.repack_sdot();
            gpu.enable_metal().unwrap();
            gpu.set_kv_bits(8);

            let mut gpu_step = RealLlamaEngine::load_bipolar(&dir, &cache_path, 512).unwrap();
            gpu_step.repack_sdot();
            gpu_step.enable_metal().unwrap();
            gpu_step.set_kv_bits(8);

            let cp = cpu.forward_prompt(&prefill_toks).to_vec();
            let gp = gpu.forward_prompt(&prefill_toks).to_vec();
            let mut gs_last = Vec::new();
            for &t in &prefill_toks {
                gs_last = gpu_step.forward_token(t).to_vec();
            }
            let cos_self_p = cosine_sim(&gp, &gs_last);
            let cos_cpu_p = cosine_sim(&cp, &gp);
            let kl_p = kl_divergence_nats(&cp, &gp);
            assert!(
                cos_self_p > 0.9997 && cos_cpu_p > 0.999 && kl_p < 0.005,
                "Metal {tag} MMA prefill mismatch: self_cos={cos_self_p:.6}, cpu_cos={cos_cpu_p:.6}, kl={kl_p:.6}"
            );

            let mut cb = vec![0.0f32; 5 * vocab];
            let mut gb = vec![0.0f32; 5 * vocab];
            cpu.forward_batch_logits(&batch_toks, &mut cb);
            gpu.forward_batch_logits(&batch_toks, &mut gb);
            for (t, &bt) in batch_toks.iter().enumerate() {
                let cs = &cb[t * vocab..(t + 1) * vocab];
                let gs = &gb[t * vocab..(t + 1) * vocab];
                let gs_step = gpu_step.forward_token(bt);
                let cos_self_b = cosine_sim(gs, gs_step);
                let cos_cpu_b = cosine_sim(cs, gs);
                assert!(
                    cos_self_b > 0.9997 && cos_cpu_b > 0.999,
                    "Metal {tag} batch tok {t} mismatch: self_cos={cos_self_b:.6}, cpu_cos={cos_cpu_b:.6}"
                );
            }

            for step in 0..3 {
                let tok = (step * 17 + 9) % vocab;
                let cl = cpu.forward_token(tok).to_vec();
                let gl = gpu.forward_token(tok).to_vec();
                let gs_s = gpu_step.forward_token(tok);
                let cos_self_d = cosine_sim(&gl, gs_s);
                let cos_cpu_d = cosine_sim(&cl, &gl);
                assert!(
                    cos_self_d > 0.9999 && cos_cpu_d > 0.999,
                    "Metal {tag} decode1 step {step} mismatch: self_cos={cos_self_d:.6}, cpu_cos={cos_cpu_d:.6}"
                );
            }

            cpu.set_num_threads(1);
        }

        let _ = fs::remove_dir_all(&dir);
    }
}
