//! Metal GPTQ compiler: Hadamard-rotated NF3/4/5 quantization and the calibration forwards that
//! feed the next layer, run on the GPU with IEEE float semantics so results match the CPU path.
//! Matrices of one shape are compiled in batches to keep the GPU occupied.

use super::metal_gpu::{
    alloc_shared_buffer, build_pipeline, msg_sel, ns_error_string, nsstring, objc_autoreleasePoolPop,
    objc_autoreleasePoolPush, objc_getClass, objc_msgSend, sel_registerName, sels, upload_slice_buffer,
    Id, MtlObj, MtlSize, Sel, MTLCreateSystemDefaultDevice, MTL_CMD_STATUS_COMPLETED,
};
use crate::kernels::hadamard::{rademacher_sign_masks_64, HADAMARD_BLOCK};
use crate::kernels::simd_lut::{nf_expand_codebook32, Int8ChunkActivation, NfSdotMatrix, SIMD_ROW_STRIPE};
use crate::model::compiler::{fit_empirical_nf_codebook, FullRowGptqHessian};
use std::ffi::c_void;

const MAX_GROUP_LEN: usize = 128;
const NUM_RATIOS: usize = 11;
const CAND_SLOTS: usize = NUM_RATIOS + 1;
/// Rows quantized per SIMD group; must match `ROWS_PER_SG` in the shader.
const ROWS_PER_SG: usize = 8;
/// Floats per matrix in the tables buffer: 32 codebook, 15 thresholds, `mu_top` at 47.
const TABLE_FLOATS: usize = 48;

/// Which compiled matrices a forward call reads.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Slot {
    GateUp,
    Down,
}

/// Hessians for a batch of matrices.
#[derive(Clone, Copy)]
pub enum HessianSet<'a> {
    None,
    /// One Hessian shared by every matrix (same input activations).
    Shared(&'a FullRowGptqHessian),
    /// One Hessian per matrix.
    PerMatrix(&'a [&'a FullRowGptqHessian]),
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FwhtParams {
    rows: u32,
    cols: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct GptqParams {
    rows: u32,
    cols: u32,
    col_s: u32,
    col_e: u32,
    nsg: u32,
    sg: u32,
    bits: u32,
    nhalf: u32,
    has_u: u32,
    rows_per_mat: u32,
    u_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TailParams {
    rows: u32,
    cols: u32,
    col_s: u32,
    gl: u32,
    t0: u32,
    rows_per_mat: u32,
    u_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MmParams {
    rows: u32,
    cols: u32,
    n_tok: u32,
    nsg: u32,
    sgc: u32,
    mat: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct SwParams {
    n_tok: u32,
    inter: u32,
}

struct Buf {
    obj: Option<MtlObj>,
    cap: usize,
}

impl Buf {
    const fn empty() -> Self {
        Self { obj: None, cap: 0 }
    }

    fn ensure(&mut self, device: Id, bytes: usize, what: &str) -> Result<Id, String> {
        if self.obj.is_none() || self.cap < bytes {
            let cap = bytes.max(16).next_power_of_two();
            self.obj = Some(alloc_shared_buffer(device, cap, what)?);
            self.cap = cap;
        }
        Ok(self.obj.as_ref().unwrap().id())
    }

    fn upload<T: Copy>(&mut self, device: Id, data: &[T], what: &str) -> Result<Id, String> {
        let bytes = std::mem::size_of_val(data);
        let id = self.ensure(device, bytes, what)?;
        // SAFETY: shared-mode buffer of at least `bytes` bytes.
        unsafe {
            let dst: *mut u8 = msg_sel!(id, sels().buf_contents; *mut u8);
            std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), dst, bytes);
        }
        Ok(id)
    }

    /// Uploads several slices back to back.
    fn upload_concat<T: Copy>(&mut self, device: Id, parts: &[&[T]], what: &str) -> Result<Id, String> {
        let bytes: usize = parts.iter().map(|p| std::mem::size_of_val(*p)).sum();
        let id = self.ensure(device, bytes, what)?;
        // SAFETY: shared-mode buffer of at least `bytes` bytes.
        unsafe {
            let mut dst: *mut u8 = msg_sel!(id, sels().buf_contents; *mut u8);
            for p in parts {
                let n = std::mem::size_of_val(*p);
                std::ptr::copy_nonoverlapping(p.as_ptr().cast::<u8>(), dst, n);
                dst = dst.add(n);
            }
        }
        Ok(id)
    }

    fn read<T: Copy>(&self, out: &mut [T]) {
        let bytes = std::mem::size_of_val(out);
        let obj = self.obj.as_ref().expect("buffer allocated");
        assert!(bytes <= self.cap);
        // SAFETY: shared-mode buffer with at least `bytes` valid bytes after the GPU completed.
        unsafe { std::ptr::copy_nonoverlapping(obj.contents_ptr(), out.as_mut_ptr().cast::<u8>(), bytes) };
    }

    fn id(&self) -> Id {
        self.obj.as_ref().expect("buffer allocated").id()
    }
}

/// A batch of compiled matrices of one shape, kept on the GPU for the forward kernels.
struct CompiledSlot {
    idx: Buf,
    scales: Buf,
    cb: Buf,
    count: usize,
    rows: usize,
    cols: usize,
    nsg: usize,
    sgc: usize,
}

impl CompiledSlot {
    const fn empty() -> Self {
        Self {
            idx: Buf::empty(),
            scales: Buf::empty(),
            cb: Buf::empty(),
            count: 0,
            rows: 0,
            cols: 0,
            nsg: 0,
            sgc: 1,
        }
    }
}

pub struct MetalGptqCompiler {
    device: MtlObj,
    queue: MtlObj,
    pso_fwht: MtlObj,
    pso_cand: MtlObj,
    pso_pick: MtlObj,
    pso_tail: MtlObj,
    pso_matmul: MtlObj,
    pso_swiglu: MtlObj,
    sel_dispatch_threads: Sel,
    masks: MtlObj,
    w: Buf,
    u: Buf,
    u_key: (usize, usize),
    tables: Buf,
    cand_loss: Buf,
    cand_idx: Buf,
    cand_err: Buf,
    err: Buf,
    gu: CompiledSlot,
    dn: CompiledSlot,
    act_q: Buf,
    act_s: Buf,
    act_cols: usize,
    tok_ids: Buf,
    ident: Buf,
    ident_len: usize,
    mm_out: Buf,
    sw_out: Buf,
    /// Per gate_up matrix: rotated SwiGLU outputs as INT8 chunks, input to `forward_down`.
    mid: Vec<(Buf, Buf, usize)>,
}

// SAFETY: Metal objects are thread-safe; the compiler is driven from one thread at a time.
unsafe impl Send for MetalGptqCompiler {}

impl MetalGptqCompiler {
    /// Returns `Err` when no Metal device exists or the kernels fail to compile.
    pub fn new() -> Result<Self, String> {
        // SAFETY: Objective-C calls inside an autorelease pool; every object we keep is retained.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<Self, String> {
                let st = sels();
                let device = MtlObj::from_retained(MTLCreateSystemDefaultDevice(), "MTLCreateSystemDefaultDevice")?;
                let queue = MtlObj::from_retained(msg_sel!(device.id(), st.new_cmd_queue; Id), "newCommandQueue")?;
                let opts_cls = objc_getClass(c"MTLCompileOptions".as_ptr());
                if opts_cls.is_null() {
                    return Err("MTLCompileOptions class missing".into());
                }
                let sel_alloc = sel_registerName(c"alloc".as_ptr());
                let sel_init = sel_registerName(c"init".as_ptr());
                let sel_fast = sel_registerName(c"setFastMathEnabled:".as_ptr());
                let opts: Id = msg_sel!(msg_sel!(opts_cls, sel_alloc; Id), sel_init; Id);
                let opts = MtlObj::from_retained(opts, "MTLCompileOptions")?;
                // IEEE semantics (no fast math, no contraction) so GPU results match the CPU compiler.
                msg_sel!(opts.id(), sel_fast, false => bool; ());
                let src = nsstring(COMPILE_SHADER_SOURCE)?;
                let mut err: Id = std::ptr::null_mut();
                let lib: Id = msg_sel!(device.id(), st.new_lib_src, src => Id, opts.id() => Id, &mut err => *mut Id; Id);
                if lib.is_null() {
                    return Err(format!("Metal compile-shader error: {}", ns_error_string(err)));
                }
                let lib = MtlObj(lib);
                let p = |n: &str| build_pipeline(device.id(), lib.id(), n);
                let pso_fwht = p("gptq_fwht_rows")?;
                let pso_cand = p("gptq_candidates")?;
                let pso_pick = p("gptq_pick")?;
                let pso_tail = p("gptq_tail_update")?;
                let pso_matmul = p("nf_idx_matmul_i8")?;
                let pso_swiglu = p("swiglu_fwht_quant")?;
                let mut masks_v = Vec::with_capacity(256 * HADAMARD_BLOCK);
                for b in 0..256 {
                    masks_v.extend_from_slice(&rademacher_sign_masks_64(b));
                }
                let masks = upload_slice_buffer(device.id(), &masks_v, "rademacher")?;
                Ok(Self {
                    device,
                    queue,
                    pso_fwht,
                    pso_cand,
                    pso_pick,
                    pso_tail,
                    pso_matmul,
                    pso_swiglu,
                    sel_dispatch_threads: sel_registerName(c"dispatchThreads:threadsPerThreadgroup:".as_ptr()),
                    masks,
                    w: Buf::empty(),
                    u: Buf::empty(),
                    u_key: (0, 0),
                    tables: Buf::empty(),
                    cand_loss: Buf::empty(),
                    cand_idx: Buf::empty(),
                    cand_err: Buf::empty(),
                    err: Buf::empty(),
                    gu: CompiledSlot::empty(),
                    dn: CompiledSlot::empty(),
                    act_q: Buf::empty(),
                    act_s: Buf::empty(),
                    act_cols: 0,
                    tok_ids: Buf::empty(),
                    ident: Buf::empty(),
                    ident_len: 0,
                    mm_out: Buf::empty(),
                    sw_out: Buf::empty(),
                    mid: Vec::new(),
                })
            })();
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    /// Runs `f` inside one command buffer and waits for completion.
    fn run<F: FnOnce(&Self, Id) -> Result<(), String>>(&self, f: F) -> Result<(), String> {
        // SAFETY: all bound buffers live in `self` until `waitUntilCompleted` returns.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<(), String> {
                let st = sels();
                let cmd: Id = msg_sel!(self.queue.id(), st.cmd_buffer; Id);
                if cmd.is_null() {
                    return Err("commandBuffer returned nil".into());
                }
                let enc: Id = msg_sel!(cmd, st.compute_enc; Id);
                if enc.is_null() {
                    return Err("computeCommandEncoder returned nil".into());
                }
                f(self, enc)?;
                msg_sel!(enc, st.end_enc; ());
                msg_sel!(cmd, st.commit; ());
                msg_sel!(cmd, st.wait_done; ());
                let status: usize = msg_sel!(cmd, st.cmd_status; usize);
                if status != MTL_CMD_STATUS_COMPLETED {
                    let err: Id = msg_sel!(cmd, st.cmd_error; Id);
                    return Err(format!("Metal compile command failed (status {status}): {}", ns_error_string(err)));
                }
                Ok(())
            })();
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    /// Binds `bufs` at indices `0..`, `params` at the next index, and dispatches `grid` threads.
    unsafe fn dispatch<P: Copy>(&self, enc: Id, pso: &MtlObj, bufs: &[Id], params: &P, grid: MtlSize, tg: MtlSize) {
        let st = sels();
        unsafe {
            msg_sel!(enc, st.set_cps, pso.id() => Id; ());
            for (i, &b) in bufs.iter().enumerate() {
                msg_sel!(enc, st.set_buf, b => Id, 0usize => usize, i => usize; ());
            }
            msg_sel!(
                enc,
                st.set_bytes,
                (params as *const P).cast::<c_void>() => *const c_void,
                std::mem::size_of::<P>() => usize,
                bufs.len() => usize;
                ()
            );
            msg_sel!(enc, self.sel_dispatch_threads, grid => MtlSize, tg => MtlSize; ());
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// GPU twin of `compile_hadamard_gptq_nf` for a batch of `[rows, cols]` matrices (3..=5 bits,
    /// `scale_group_chunks <= 2`, `rows % 16 == 0`). The results stay on the GPU in `slot` for the forwards.
    pub fn compile_nf_batch(
        &mut self,
        slot: Slot,
        weights: &[&[f32]],
        rows: usize,
        cols: usize,
        bits: usize,
        sgc: usize,
        hess: HessianSet<'_>,
    ) -> Result<Vec<NfSdotMatrix>, String> {
        let b = weights.len();
        assert!(b > 0 && weights.iter().all(|w| w.len() == rows * cols));
        assert!((3..=5).contains(&bits) && (1..=2).contains(&sgc) && cols.is_multiple_of(HADAMARD_BLOCK));
        assert!(rows.is_multiple_of(SIMD_ROW_STRIPE), "batched GPTQ needs rows % 16 == 0");
        assert!(cols / HADAMARD_BLOCK <= 256, "cols must be <= 16384");
        let prof = std::env::var_os("KRONOS_COMPILE_PROF").is_some();
        let t0 = std::time::Instant::now();

        let mut codebooks = vec![([0i8; 16], 0.0f32); b];
        std::thread::scope(|s| {
            for (w, out) in weights.iter().zip(codebooks.iter_mut()) {
                s.spawn(move || *out = fit_empirical_nf_codebook(w, rows, cols, bits, sgc));
            }
        });
        let half = 1usize << (bits - 1);
        let mut tables_v = vec![0.0f32; b * TABLE_FLOATS];
        let mut cb_i = vec![0i32; b * 32];
        for (m, (codebook, mu_top)) in codebooks.iter().enumerate() {
            let cb32 = nf_expand_codebook32(bits, codebook);
            let t = &mut tables_v[m * TABLE_FLOATS..(m + 1) * TABLE_FLOATS];
            for k in 0..(1usize << bits) {
                t[k] = cb32[k] as f32;
                cb_i[m * 32 + k] = cb32[k] as i32;
            }
            for i in 0..half - 1 {
                t[32 + i] = 0.5 * (t[half + i] + t[half + i + 1]);
            }
            t[47] = *mu_top;
        }
        let t_cb = t0.elapsed().as_secs_f64();

        let num_chunks = cols / HADAMARD_BLOCK;
        let nsg = num_chunks.div_ceil(sgc);
        let num_stripes = rows / SIMD_ROW_STRIPE;
        let all_rows = b * rows;
        let dev = self.device.id();

        self.w.upload_concat(dev, weights, "gptq weights")?;
        let (has_u, u_stride) = match hess {
            HessianSet::None => {
                self.u.ensure(dev, 16, "gptq hessian")?;
                self.u_key = (0, 0);
                (0u32, 0u32)
            }
            HessianSet::Shared(h) => {
                assert_eq!(h.cols, cols);
                let u = h.h_inv_chol_upper.as_slice();
                let key = (u.as_ptr() as usize, u.len());
                if self.u_key != key || self.u.obj.is_none() {
                    self.u.upload(dev, u, "gptq hessian")?;
                    self.u_key = key;
                }
                (1u32, 0u32)
            }
            HessianSet::PerMatrix(hs) => {
                assert_eq!(hs.len(), b);
                let parts: Vec<&[f32]> = hs
                    .iter()
                    .map(|h| {
                        assert_eq!(h.cols, cols);
                        h.h_inv_chol_upper.as_slice()
                    })
                    .collect();
                self.u.upload_concat(dev, &parts, "gptq hessians")?;
                self.u_key = (0, 0);
                (1u32, (cols * cols) as u32)
            }
        };
        self.tables.upload(dev, &tables_v, "gptq tables")?;
        self.cand_loss.ensure(dev, all_rows * CAND_SLOTS * 4, "cand loss")?;
        self.cand_idx.ensure(dev, all_rows * CAND_SLOTS * MAX_GROUP_LEN, "cand idx")?;
        self.cand_err.ensure(dev, all_rows * CAND_SLOTS * MAX_GROUP_LEN * 4, "cand err")?;
        self.err.ensure(dev, all_rows * MAX_GROUP_LEN * 4, "group err")?;
        let cs = match slot {
            Slot::GateUp => &mut self.gu,
            Slot::Down => &mut self.dn,
        };
        cs.idx.ensure(dev, all_rows * cols, "nf idx")?;
        cs.scales.ensure(dev, b * num_stripes * nsg * SIMD_ROW_STRIPE * 2, "nf scales")?;
        cs.cb.upload(dev, &cb_i, "nf codebooks")?;
        cs.count = b;
        cs.rows = rows;
        cs.cols = cols;
        cs.nsg = nsg;
        cs.sgc = sgc;
        let (idx_id, sc_id) = (cs.idx.id(), cs.scales.id());
        let t_up = t0.elapsed().as_secs_f64();

        let groups: Vec<(usize, usize, GptqParams)> = (0..nsg)
            .map(|sg| {
                let col_s = sg * sgc * HADAMARD_BLOCK;
                let col_e = ((sg + 1) * sgc * HADAMARD_BLOCK).min(cols);
                (
                    col_s,
                    col_e,
                    GptqParams {
                        rows: all_rows as u32,
                        cols: cols as u32,
                        col_s: col_s as u32,
                        col_e: col_e as u32,
                        nsg: nsg as u32,
                        sg: sg as u32,
                        bits: bits as u32,
                        nhalf: half as u32,
                        has_u,
                        rows_per_mat: rows as u32,
                        u_stride,
                    },
                )
            })
            .collect();
        // SAFETY (all closures below): every bound buffer is owned by `s` and outlives the wait.
        let enc_fwht = |s: &Self, enc: Id| unsafe {
            s.dispatch(
                enc,
                &s.pso_fwht,
                &[s.w.id(), s.masks.id()],
                &FwhtParams { rows: all_rows as u32, cols: cols as u32 },
                MtlSize::new(num_chunks, all_rows, 1),
                MtlSize::new(8, 8, 1),
            );
        };
        let enc_cand = |s: &Self, enc: Id, gp: &GptqParams| unsafe {
            s.dispatch(
                enc,
                &s.pso_cand,
                &[s.w.id(), s.u.id(), s.cand_loss.id(), s.cand_idx.id(), s.cand_err.id(), s.tables.id()],
                gp,
                MtlSize::new(32 * NUM_RATIOS, all_rows.div_ceil(ROWS_PER_SG), 1),
                MtlSize::new(32 * NUM_RATIOS, 1, 1),
            );
        };
        let enc_pick = |s: &Self, enc: Id, gp: &GptqParams| unsafe {
            s.dispatch(
                enc,
                &s.pso_pick,
                &[s.w.id(), s.u.id(), s.cand_loss.id(), s.cand_idx.id(), s.cand_err.id(), s.tables.id(), idx_id, sc_id, s.err.id()],
                gp,
                MtlSize::new(32, all_rows.div_ceil(ROWS_PER_SG), 1),
                MtlSize::new(32, 8, 1),
            );
        };
        let enc_tail = |s: &Self, enc: Id, col_s: usize, col_e: usize| unsafe {
            s.dispatch(
                enc,
                &s.pso_tail,
                &[s.w.id(), s.u.id(), s.err.id()],
                &TailParams {
                    rows: all_rows as u32,
                    cols: cols as u32,
                    col_s: col_s as u32,
                    gl: (col_e - col_s) as u32,
                    t0: col_e as u32,
                    rows_per_mat: rows as u32,
                    u_stride,
                },
                MtlSize::new((cols - col_e).div_ceil(64) * 64, all_rows.div_ceil(16) * 2, 1),
                MtlSize::new(64, 2, 1),
            );
        };
        if prof {
            let mut kt = [0.0f64; 4];
            let mut timed = |k: usize, f: &dyn Fn(&Self, Id)| -> Result<(), String> {
                let t = std::time::Instant::now();
                self.run(|s, enc| {
                    f(s, enc);
                    Ok(())
                })?;
                kt[k] += t.elapsed().as_secs_f64();
                Ok(())
            };
            timed(0, &enc_fwht)?;
            for (col_s, col_e, gp) in &groups {
                timed(1, &|s, enc| enc_cand(s, enc, gp))?;
                timed(2, &|s, enc| enc_pick(s, enc, gp))?;
                if *col_e < cols && has_u == 1 {
                    timed(3, &|s, enc| enc_tail(s, enc, *col_s, *col_e))?;
                }
            }
            eprintln!(
                "  kernels: fwht {:.1}ms candidates {:.1}ms pick {:.1}ms tail {:.1}ms",
                kt[0] * 1e3, kt[1] * 1e3, kt[2] * 1e3, kt[3] * 1e3
            );
        } else {
            self.run(|s, enc| {
                enc_fwht(s, enc);
                for (col_s, col_e, gp) in &groups {
                    enc_cand(s, enc, gp);
                    enc_pick(s, enc, gp);
                    if *col_e < cols && has_u == 1 {
                        enc_tail(s, enc, *col_s, *col_e);
                    }
                }
                Ok(())
            })?;
        }
        let t_gpu = t0.elapsed().as_secs_f64();

        let mut flat_idx = vec![0u8; all_rows * cols];
        let scales_per = num_stripes * nsg * SIMD_ROW_STRIPE;
        let mut scales = vec![0u16; b * scales_per];
        let cs = match slot {
            Slot::GateUp => &self.gu,
            Slot::Down => &self.dn,
        };
        cs.idx.read(&mut flat_idx);
        cs.scales.read(&mut scales);
        let mut mats: Vec<Option<NfSdotMatrix>> = (0..b).map(|_| None).collect();
        std::thread::scope(|s| {
            for (m, out) in mats.iter_mut().enumerate() {
                let idx = &flat_idx[m * rows * cols..(m + 1) * rows * cols];
                let sc = &scales[m * scales_per..(m + 1) * scales_per];
                let cb = codebooks[m].0;
                s.spawn(move || *out = Some(NfSdotMatrix::from_indices(idx, sc, rows, cols, bits, sgc, cb)));
            }
        });
        if prof {
            let t_all = t0.elapsed().as_secs_f64();
            eprintln!(
                "gptq {b}x[{rows}x{cols}] nf{bits}: codebook {:.1}ms upload {:.1}ms gpu {:.1}ms pack {:.1}ms",
                t_cb * 1e3, (t_up - t_cb) * 1e3, (t_gpu - t_up) * 1e3, (t_all - t_gpu) * 1e3
            );
        }
        Ok(mats.into_iter().map(|m| m.unwrap()).collect())
    }

    #[allow(clippy::too_many_arguments)]
    /// Single-matrix convenience over [`Self::compile_nf_batch`].
    pub fn compile_nf(
        &mut self,
        slot: Slot,
        weights: &[f32],
        rows: usize,
        cols: usize,
        bits: usize,
        sgc: usize,
        hessian: Option<&FullRowGptqHessian>,
    ) -> Result<NfSdotMatrix, String> {
        let hs = match hessian {
            Some(h) => HessianSet::Shared(h),
            None => HessianSet::None,
        };
        Ok(self.compile_nf_batch(slot, &[weights], rows, cols, bits, sgc, hs)?.pop().unwrap())
    }

    /// Uploads a layer's rotated INT8 input activations once; experts index into them by token.
    pub fn upload_layer_acts(&mut self, acts: &[Int8ChunkActivation]) -> Result<(), String> {
        let cols = acts.first().map_or(0, |a| a.cols);
        let nc = cols / 64;
        let mut q = Vec::with_capacity(acts.len() * cols);
        let mut s = Vec::with_capacity(acts.len() * nc);
        for a in acts {
            q.extend_from_slice(&a.q);
            s.extend_from_slice(&a.scales);
        }
        let dev = self.device.id();
        self.act_q.upload(dev, &q, "layer act q")?;
        self.act_s.upload(dev, &s, "layer act scales")?;
        self.act_cols = cols;
        Ok(())
    }

    /// `rotate(SwiGLU(gate_up[mat](x_t)))` for the given tokens of the uploaded layer activations.
    /// Returns `[n_tok, inter]` rotated outputs; their INT8 form stays on the GPU for `forward_down(mat)`.
    pub fn forward_gate_up_swiglu(&mut self, mat: usize, tok_ids: &[u32]) -> Result<Vec<f32>, String> {
        let n = tok_ids.len();
        assert!(mat < self.gu.count);
        let (rows, cols) = (self.gu.rows, self.gu.cols);
        assert_eq!(cols, self.act_cols, "layer activations must be uploaded for this width");
        let inter = rows / 2;
        let dev = self.device.id();
        self.tok_ids.upload(dev, tok_ids, "tok ids")?;
        self.mm_out.ensure(dev, n * rows * 4, "matmul out")?;
        self.sw_out.ensure(dev, n * inter * 4, "swiglu out")?;
        while self.mid.len() <= mat {
            self.mid.push((Buf::empty(), Buf::empty(), 0));
        }
        self.mid[mat].0.ensure(dev, n * inter, "mid q")?;
        self.mid[mat].1.ensure(dev, n * (inter / 64) * 4, "mid scales")?;
        self.mid[mat].2 = n;
        self.run(|s, enc| {
            // SAFETY: buffers owned by `s`.
            unsafe {
                s.dispatch(
                    enc,
                    &s.pso_matmul,
                    &[s.gu.idx.id(), s.gu.scales.id(), s.gu.cb.id(), s.act_q.id(), s.act_s.id(), s.tok_ids.id(), s.mm_out.id()],
                    &MmParams { rows: rows as u32, cols: cols as u32, n_tok: n as u32, nsg: s.gu.nsg as u32, sgc: s.gu.sgc as u32, mat: mat as u32 },
                    MtlSize::new(rows, n, 1),
                    MtlSize::new(32, 4, 1),
                );
                s.dispatch(
                    enc,
                    &s.pso_swiglu,
                    &[s.mm_out.id(), s.masks.id(), s.sw_out.id(), s.mid[mat].0.id(), s.mid[mat].1.id()],
                    &SwParams { n_tok: n as u32, inter: inter as u32 },
                    MtlSize::new(inter / 64, n, 1),
                    MtlSize::new(4, 16, 1),
                );
            }
            Ok(())
        })?;
        let mut out = vec![0.0f32; n * inter];
        self.sw_out.read(&mut out);
        Ok(out)
    }

    /// `down[mat](mid)` for every token of the last `forward_gate_up_swiglu(mat, ..)`; returns `[n_tok, hidden]`.
    pub fn forward_down(&mut self, mat: usize) -> Result<Vec<f32>, String> {
        assert!(mat < self.dn.count && mat < self.mid.len());
        let n = self.mid[mat].2;
        let (rows, cols) = (self.dn.rows, self.dn.cols);
        let dev = self.device.id();
        if self.ident_len < n {
            let ids: Vec<u32> = (0..n as u32).collect();
            self.ident.upload(dev, &ids, "identity ids")?;
            self.ident_len = n;
        }
        self.mm_out.ensure(dev, n * rows * 4, "matmul out")?;
        self.run(|s, enc| {
            // SAFETY: buffers owned by `s`.
            unsafe {
                s.dispatch(
                    enc,
                    &s.pso_matmul,
                    &[s.dn.idx.id(), s.dn.scales.id(), s.dn.cb.id(), s.mid[mat].0.id(), s.mid[mat].1.id(), s.ident.id(), s.mm_out.id()],
                    &MmParams { rows: rows as u32, cols: cols as u32, n_tok: n as u32, nsg: s.dn.nsg as u32, sgc: s.dn.sgc as u32, mat: mat as u32 },
                    MtlSize::new(rows, n, 1),
                    MtlSize::new(32, 4, 1),
                );
            }
            Ok(())
        })?;
        let mut out = vec![0.0f32; n * rows];
        self.mm_out.read(&mut out);
        Ok(out)
    }
}

const COMPILE_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
#pragma METAL fp contract(off)
using namespace metal;

constant uint MAX_GL = 128u;
constant uint NUM_RATIOS = 11u;
constant uint CAND_SLOTS = 12u;
constant uint TABLE_FLOATS = 48u;
#define ROWS_PER_SG 8u
constant float RATIOS[11] = {0.74f, 0.80f, 0.86f, 0.91f, 0.96f, 1.00f, 1.04f, 1.09f, 1.15f, 1.22f, 1.30f};

struct FwhtParams { uint rows; uint cols; };
struct GptqParams { uint rows; uint cols; uint col_s; uint col_e; uint nsg; uint sg; uint bits; uint nhalf; uint has_u; uint rows_per_mat; uint u_stride; };
struct TailParams { uint rows; uint cols; uint col_s; uint gl; uint t0; uint rows_per_mat; uint u_stride; };
struct MmParams { uint rows; uint cols; uint n_tok; uint nsg; uint sgc; uint mat; };
struct SwParams { uint n_tok; uint inter; };

inline ushort f32_to_bf16(float v) {
    uint bits = as_type<uint>(v);
    if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u) {
        return ushort((bits >> 16) | 0x0040u);
    }
    uint bias = 0x00007FFFu + ((bits >> 16) & 1u);
    return ushort((bits + bias) >> 16);
}
inline float bf16_to_f32(ushort b) { return as_type<float>(uint(b) << 16); }

inline void fwht64(thread float* v, device const uint* m) {
    for (uint i = 0; i < 64u; i++) v[i] = as_type<float>(as_type<uint>(v[i]) ^ m[i]) * 0.125f;
    for (uint h = 1u; h < 64u; h <<= 1u) {
        for (uint i = 0; i < 64u; i += 2u * h) {
            for (uint j = i; j < i + h; j++) {
                float a = v[j]; float b = v[j + h];
                v[j] = a + b; v[j + h] = a - b;
            }
        }
    }
}

kernel void gptq_fwht_rows(device float* w [[buffer(0)]], device const uint* masks [[buffer(1)]],
                           constant FwhtParams& p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    uint blk = gid.x, row = gid.y;
    if (row >= p.rows || blk * 64u >= p.cols) return;
    device float* b = w + row * p.cols + blk * 64u;
    float v[64];
    for (uint i = 0; i < 64u; i++) v[i] = b[i];
    fwht64(v, masks + blk * 64u);
    for (uint i = 0; i < 64u; i++) b[i] = v[i];
}

struct QTables { float4 cb[8]; float4 th[4]; };

inline QTables load_tables(constant float* tables) {
    QTables t;
    for (uint i = 0; i < 8u; i++) t.cb[i] = float4(tables[4u * i], tables[4u * i + 1u], tables[4u * i + 2u], tables[4u * i + 3u]);
    for (uint i = 0; i < 4u; i++) t.th[i] = float4(tables[32u + 4u * i], tables[33u + 4u * i], tables[34u + 4u * i], tables[35u + 4u * i]);
    return t;
}

inline float comp4(float4 v, uint c) { return (c == 0u) ? v.x : (c == 1u) ? v.y : (c == 2u) ? v.z : v.w; }

// Codebook value for a (SIMD-uniform) index without a dynamic memory access.
inline float cb_at(thread const QTables& t, uint k) {
    uint q = k >> 2u;
    float4 v = (q < 4u) ? ((q < 2u) ? ((q == 0u) ? t.cb[0] : t.cb[1]) : ((q == 2u) ? t.cb[2] : t.cb[3]))
                        : ((q < 6u) ? ((q == 4u) ? t.cb[4] : t.cb[5]) : ((q == 6u) ? t.cb[6] : t.cb[7]));
    return comp4(v, k & 3u);
}

// Nearest codebook level: thresholds are ascending midpoints, so the level is the count of thresholds below |w|.
inline uint quant_nearest(float w, float inv_delta, uint bits, uint nhalf, thread const QTables& t) {
    float aw = fabs(w) * inv_delta;
    uint mag = (aw >= t.th[0].x ? 1u : 0u) + (aw >= t.th[0].y ? 1u : 0u) + (aw >= t.th[0].z ? 1u : 0u);
    if (bits >= 4u) {
        mag += (aw >= t.th[0].w ? 1u : 0u) + (aw >= t.th[1].x ? 1u : 0u) + (aw >= t.th[1].y ? 1u : 0u) + (aw >= t.th[1].z ? 1u : 0u);
    }
    if (bits == 5u) {
        mag += (aw >= t.th[1].w ? 1u : 0u) + (aw >= t.th[2].x ? 1u : 0u) + (aw >= t.th[2].y ? 1u : 0u) + (aw >= t.th[2].z ? 1u : 0u)
             + (aw >= t.th[2].w ? 1u : 0u) + (aw >= t.th[3].x ? 1u : 0u) + (aw >= t.th[3].y ? 1u : 0u) + (aw >= t.th[3].z ? 1u : 0u);
    }
    return (w >= 0.0f) ? (nhalf + mag) : (nhalf - 1u - mag);
}

// Lanes per row and columns per lane for one scale group of `gl` columns.
#define LPR (32u / ROWS_PER_SG)

inline float comp_sel(thread const float4* b, uint nq, uint c) {
    uint q = c >> 2u;
    float4 v = b[0];
    for (uint i = 1u; i < nq; i++) v = (q == i) ? b[i] : v;
    return comp4(v, c & 3u);
}

inline void upd4(thread float4& b, uint col0, uint j, float err, float4 u4) {
    b.x = (col0 + 0u > j) ? b.x - err * u4.x : b.x;
    b.y = (col0 + 1u > j) ? b.y - err * u4.y : b.y;
    b.z = (col0 + 2u > j) ? b.z - err * u4.z : b.z;
    b.w = (col0 + 3u > j) ? b.w - err * u4.w : b.w;
}

// A SIMD group quantizes ROWS_PER_SG rows' scale groups at once: LPR lanes per row, each owning
// `gl / LPR` consecutive columns. The column loop is sequential (GPTQ error feedback) and every float
// op happens in the CPU order, so the result is bit-identical. Lanes with `ok == false` only keep the
// shuffles alive. Returns the loss, or INFINITY once it reaches `best` (CPU early exit).
inline float eval_delta_sg(device const float* sub, device const float* u, uint cols, uint col_s, uint gl, uint has_u,
                           float delta, uint bits, uint nhalf, thread const QTables& t, float best, uint lane, bool ok,
                           device uchar* out_idx, device float* out_err) {
    uint sl = lane % LPR;
    uint cpl = gl / LPR;
    uint nq = cpl >> 2u;
    uint base = sl * cpl;
    ok = ok && delta > 0.0f;
    float inv_delta = ok ? 1.0f / delta : 0.0f;
    float4 b[MAX_GL / 4u / LPR];
    for (uint q = 0; q < MAX_GL / 4u / LPR; q++) b[q] = float4(0.0f);
    if (ok) {
        for (uint q = 0; q < MAX_GL / 4u / LPR; q++) if (q < nq) b[q] = *(device const float4*)(sub + base + 4u * q);
    }
    device const float* udiag = u + col_s * cols + col_s;
    device const float* urow = u + col_s * cols + col_s + base;
    float loss = 0.0f;
    bool done = !ok;
    for (uint j = 0; j < gl; j++) {
        uint owner_sl = j / cpl;
        float wj = simd_shuffle(comp_sel(b, nq, j % cpl), ushort((lane / LPR) * LPR + owner_sl));
        uint k = quant_nearest(wj, inv_delta, bits, nhalf, t);
        float w_hat = cb_at(t, k) * delta;
        float err = 0.0f;
        if (has_u != 0u) {
            err = (wj - w_hat) / fmax(udiag[j * (cols + 1u)], 1e-12f);
            if (!done) {
                loss += err * err;
                if (loss >= best) { done = true; loss = INFINITY; }
            }
            if (!done) {
                device const float* ur = urow + j * cols;
                for (uint q = 0; q < MAX_GL / 4u / LPR; q++) {
                    if (q < nq) upd4(b[q], base + 4u * q, j, err, *(device const float4*)(ur + 4u * q));
                }
            }
        } else if (!done) {
            float d = wj - w_hat;
            loss += d * d;
        }
        if (!done && sl == owner_sl) { out_idx[j] = uchar(k); out_err[j] = err; }
    }
    if (has_u == 0u && !done && loss >= best) loss = INFINITY;
    return done ? INFINITY : loss;
}

inline float group_rms(device const float* sub, uint gl) {
    float sum_sq = 0.0f;
    for (uint j = 0; j < gl; j++) sum_sq += sub[j] * sub[j];
    return sqrt(sum_sq / float(gl));
}

// One SIMD group per (candidate ratio, 4 rows): full loss for that scale candidate. Slot NUM_RATIOS
// is scratch for the least-squares refit in `gptq_pick`.
kernel void gptq_candidates(device const float* w [[buffer(0)]], device const float* u [[buffer(1)]],
                            device float* cand_loss [[buffer(2)]], device uchar* cand_idx [[buffer(3)]],
                            device float* cand_err [[buffer(4)]], constant float* tables [[buffer(5)]],
                            constant GptqParams& p [[buffer(6)]], uint2 gid [[thread_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint c = gid.x / 32u;
    if (c >= NUM_RATIOS) return;
    uint r = gid.y * ROWS_PER_SG + lane / LPR;
    bool row_ok = r < p.rows;
    uint mat = (row_ok ? r : 0u) / p.rows_per_mat;
    constant float* tb = tables + mat * TABLE_FLOATS;
    QTables t = load_tables(tb);
    device const float* um = u + mat * p.u_stride;
    uint gl = p.col_e - p.col_s;
    device const float* sub = w + (row_ok ? r : 0u) * p.cols + p.col_s;
    float rms = row_ok ? group_rms(sub, gl) : 0.0f;
    bool ok = row_ok && rms > 1e-12f;
    uint slot = (row_ok ? r : 0u) * CAND_SLOTS + c;
    float delta = bf16_to_f32(f32_to_bf16(rms * (tb[47] / 127.0f) * RATIOS[c]));
    float loss = eval_delta_sg(sub, um, p.cols, p.col_s, gl, p.has_u, delta, p.bits, p.nhalf, t, INFINITY, lane, ok,
                               cand_idx + slot * MAX_GL, cand_err + slot * MAX_GL);
    if (row_ok && (lane % LPR) == 0u) cand_loss[slot] = ok ? loss : INFINITY;
}

inline void store_cols(device uchar* idx_dst, device float* err_dst, device const uchar* idx_src, device const float* err_src, uint cpl) {
    for (uint q = 0; q < cpl; q += 4u) {
        *(device uchar4*)(idx_dst + q) = *(device const uchar4*)(idx_src + q);
        *(device float4*)(err_dst + q) = *(device const float4*)(err_src + q);
    }
}

// One SIMD group per 4 rows: pick the best ratio (ties -> lowest), try the least-squares refit, emit results.
kernel void gptq_pick(device const float* w [[buffer(0)]], device const float* u [[buffer(1)]],
                      device const float* cand_loss [[buffer(2)]], device uchar* cand_idx [[buffer(3)]],
                      device float* cand_err [[buffer(4)]], constant float* tables [[buffer(5)]],
                      device uchar* out_idx [[buffer(6)]], device ushort* out_scales [[buffer(7)]],
                      device float* out_err [[buffer(8)]], constant GptqParams& p [[buffer(9)]],
                      uint2 gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint r = gid.y * ROWS_PER_SG + lane / LPR;
    bool row_ok = r < p.rows;
    uint sl = lane % LPR;
    uint rr = row_ok ? r : 0u;
    uint mat = rr / p.rows_per_mat;
    constant float* tb = tables + mat * TABLE_FLOATS;
    QTables t = load_tables(tb);
    device const float* um = u + mat * p.u_stride;
    uint gl = p.col_e - p.col_s;
    uint cpl = gl / LPR;
    uint stripe = rr / 16u, ln = rr % 16u;
    device float* err_row = out_err + rr * MAX_GL;
    device uchar* idx_row = out_idx + rr * p.cols + p.col_s;
    device const float* sub = w + rr * p.cols + p.col_s;
    float rms = row_ok ? group_rms(sub, gl) : 0.0f;
    bool ok = row_ok && rms > 1e-12f;
    float base_delta = rms * (tb[47] / 127.0f);
    float best_loss = INFINITY;
    uint best_c = 0u;
    for (uint c = 0; c < NUM_RATIOS; c++) {
        float l = ok ? cand_loss[rr * CAND_SLOTS + c] : INFINITY;
        if (l < best_loss) { best_loss = l; best_c = c; }
    }
    ushort best_bf16 = (best_loss < INFINITY) ? f32_to_bf16(base_delta * RATIOS[best_c]) : ushort(0);
    uint best_slot = rr * CAND_SLOTS + best_c;
    device const uchar* bidx = cand_idx + best_slot * MAX_GL;
    // Least-squares refit in CPU summation order.
    float dot_wc = 0.0f, dot_cc = 0.0f;
    if (ok) {
        for (uint j = 0; j < gl; j++) {
            float c_val = cb_at(t, (best_loss < INFINITY) ? uint(bidx[j]) : 0u);
            dot_wc += sub[j] * c_val;
            dot_cc += c_val * c_val;
        }
    }
    ushort ls_bf16 = f32_to_bf16(dot_wc / dot_cc);
    bool need_ls = ok && dot_cc > 1e-12f && dot_wc > 0.0f && ls_bf16 != best_bf16;
    uint ls_slot = rr * CAND_SLOTS + NUM_RATIOS;
    float l = eval_delta_sg(sub, um, p.cols, p.col_s, gl, p.has_u, bf16_to_f32(ls_bf16), p.bits, p.nhalf, t, best_loss, lane, need_ls,
                            cand_idx + ls_slot * MAX_GL, cand_err + ls_slot * MAX_GL);
    if (need_ls && l < best_loss) { best_loss = l; best_bf16 = ls_bf16; best_slot = ls_slot; }
    if (!row_ok) return;
    if (sl == 0u) out_scales[(stripe * p.nsg + p.sg) * 16u + ln] = ok ? best_bf16 : ushort(0);
    uint base = sl * cpl;
    if (!ok) {
        for (uint q = 0; q < cpl; q += 4u) {
            *(device uchar4*)(idx_row + base + q) = uchar4(uchar(p.nhalf - 1u));
            *(device float4*)(err_row + base + q) = float4(0.0f);
        }
    } else if (best_loss < INFINITY) {
        store_cols(idx_row + base, err_row + base, cand_idx + best_slot * MAX_GL + base, cand_err + best_slot * MAX_GL + base, cpl);
    } else {
        for (uint q = 0; q < cpl; q += 4u) {
            *(device uchar4*)(idx_row + base + q) = uchar4(0);
            *(device float4*)(err_row + base + q) = float4(0.0f);
        }
    }
}

// w[r, t] -= sum_j err[r, j] * U[col_s + j, t] for t >= t0, in CPU order.
// Threadgroup = 64 columns x 16 rows; each thread owns 8 rows of one column, err staged in threadgroup memory.
kernel void gptq_tail_update(device float* w [[buffer(0)]], device const float* u [[buffer(1)]],
                             device const float* err [[buffer(2)]], constant TailParams& p [[buffer(3)]],
                             uint2 tgid [[threadgroup_position_in_grid]], uint2 lid [[thread_position_in_threadgroup]],
                             uint tidx [[thread_index_in_threadgroup]]) {
    threadgroup float etile[16 * MAX_GL];
    uint r_base = tgid.y * 16u;
    u += (r_base / p.rows_per_mat) * p.u_stride;
    for (uint i = tidx; i < 16u * MAX_GL; i += 128u) {
        uint rr = r_base + i / MAX_GL, j = i % MAX_GL;
        etile[i] = (rr < p.rows && j < p.gl) ? err[rr * MAX_GL + j] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t = p.t0 + tgid.x * 64u + lid.x;
    if (t >= p.cols) return;
    uint r0 = r_base + lid.y * 8u;
    float v[8];
    for (uint i = 0; i < 8u; i++) v[i] = (r0 + i < p.rows) ? w[(r0 + i) * p.cols + t] : 0.0f;
    threadgroup const float* e = etile + lid.y * 8u * MAX_GL;
    for (uint j = 0; j < p.gl; j++) {
        float uj = u[(p.col_s + j) * p.cols + t];
        for (uint i = 0; i < 8u; i++) {
            float ej = e[i * MAX_GL + j];
            if (ej != 0.0f) v[i] -= ej * uj;
        }
    }
    for (uint i = 0; i < 8u; i++) if (r0 + i < p.rows) w[(r0 + i) * p.cols + t] = v[i];
}

// out[i, r] = sum_c act_scale[t, c] * w_scale[r, sg(c)] * sum_k cb[idx[r, c*64+k]] * q[t, c*64+k], t = tok_ids[i].
kernel void nf_idx_matmul_i8(device const uchar* idx [[buffer(0)]], device const ushort* scales [[buffer(1)]],
                             constant int* cb [[buffer(2)]], device const char* actq [[buffer(3)]],
                             device const float* acts [[buffer(4)]], device const uint* tok_ids [[buffer(5)]],
                             device float* out [[buffer(6)]], constant MmParams& p [[buffer(7)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint r = gid.x, i = gid.y;
    if (r >= p.rows || i >= p.n_tok) return;
    uint t = tok_ids[i];
    uint nc = p.cols / 64u;
    uint stripe = r / 16u, lane = r % 16u;
    idx += p.mat * p.rows * p.cols;
    scales += p.mat * (p.rows / 16u) * p.nsg * 16u;
    cb += p.mat * 32u;
    device const uchar* wrow = idx + r * p.cols;
    device const char* arow = actq + t * p.cols;
    float acc = 0.0f;
    for (uint c = 0; c < nc; c++) {
        float ws = bf16_to_f32(scales[(stripe * p.nsg + c / p.sgc) * 16u + lane]);
        int dot = 0;
        for (uint k = 0; k < 64u; k += 4u) {
            uchar4 w4 = *(device const uchar4*)(wrow + c * 64u + k);
            char4 a4 = *(device const char4*)(arow + c * 64u + k);
            dot += cb[w4.x] * int(a4.x) + cb[w4.y] * int(a4.y) + cb[w4.z] * int(a4.z) + cb[w4.w] * int(a4.w);
        }
        acc += float(dot) * ws * acts[t * nc + c];
    }
    out[i * p.rows + r] = acc;
}

// Per (token, 64-block): SwiGLU, rotate, store f32 and the INT8 chunk quantization used by `populate`.
kernel void swiglu_fwht_quant(device const float* gu [[buffer(0)]], device const uint* masks [[buffer(1)]],
                              device float* sw_out [[buffer(2)]], device char* q_out [[buffer(3)]],
                              device float* qs_out [[buffer(4)]], constant SwParams& p [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint blk = gid.x, i = gid.y;
    if (i >= p.n_tok || blk * 64u >= p.inter) return;
    device const float* g = gu + i * 2u * p.inter + blk * 64u;
    device const float* up = g + p.inter;
    float v[64];
    for (uint k = 0; k < 64u; k++) { float gv = g[k]; v[k] = (gv / (1.0f + exp(-gv))) * up[k]; }
    fwht64(v, masks + blk * 64u);
    float max_abs = 0.0f;
    for (uint k = 0; k < 64u; k++) { sw_out[i * p.inter + blk * 64u + k] = v[k]; max_abs = fmax(max_abs, fabs(v[k])); }
    device char* q = q_out + i * p.inter + blk * 64u;
    uint nc = p.inter / 64u;
    if (max_abs <= 1.1920929e-7f) {
        for (uint k = 0; k < 64u; k++) q[k] = 0;
        qs_out[i * nc + blk] = 0.0f;
        return;
    }
    float inv = 127.0f / max_abs;
    for (uint k = 0; k < 64u; k++) q[k] = char(int(rint(v[k] * inv)));
    qs_out[i * nc + blk] = max_abs / 127.0f;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::hadamard::fwht64_signed_inplace;
    use crate::kernels::simd_lut::nf_sdot_stripe_range_int8;
    use crate::model::compiler::compile_hadamard_gptq_nf;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }

    fn dequant_all(m: &NfSdotMatrix) -> Vec<f32> {
        let mut out = vec![0.0f32; m.rows * m.cols];
        let mut row = vec![0.0f32; m.cols];
        for r in 0..m.rows {
            m.dequantize_row_rotated(r, &mut row);
            out[r * m.cols..(r + 1) * m.cols].copy_from_slice(&row);
        }
        out
    }

    #[test]
    fn gpu_gptq_matches_cpu_bit_exact() {
        let mut gpu = MetalGptqCompiler::new().expect("Metal device");
        let mut seed = 11u64;
        for &(rows, cols, bits, sgc) in &[(96usize, 512usize, 4usize, 2usize), (128, 256, 3, 2), (80, 384, 5, 1)] {
            let w: Vec<f32> = (0..rows * cols).map(|_| 0.05 * lcg(&mut seed)).collect();
            let mut hess = FullRowGptqHessian::new(cols);
            for _ in 0..cols + 64 {
                let x: Vec<f32> = (0..cols).map(|i| lcg(&mut seed) * (1.0 + (i % 7) as f32)).collect();
                hess.observe_activation(&x);
            }
            hess.finalize(0.01);
            let cpu = compile_hadamard_gptq_nf(&w, rows, cols, bits, sgc, Some(&hess), 4);
            let gpu_m = gpu.compile_nf(Slot::GateUp, &w, rows, cols, bits, sgc, Some(&hess)).unwrap();
            assert_eq!(cpu.codebook, gpu_m.codebook);
            let (a, b) = (dequant_all(&cpu), dequant_all(&gpu_m));
            let mism = a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
            assert_eq!(mism, 0, "{rows}x{cols} nf{bits} sgc{sgc}: {mism} mismatching weights");
        }
    }

    #[test]
    fn gpu_batch_matches_single_per_matrix() {
        let mut gpu = MetalGptqCompiler::new().expect("Metal device");
        let (rows, cols) = (64usize, 256usize);
        let mut seed = 3u64;
        let ws: Vec<Vec<f32>> = (0..3).map(|_| (0..rows * cols).map(|_| 0.05 * lcg(&mut seed)).collect()).collect();
        let hs: Vec<FullRowGptqHessian> = (0..3)
            .map(|_| {
                let mut h = FullRowGptqHessian::new(cols);
                for _ in 0..cols + 32 {
                    let x: Vec<f32> = (0..cols).map(|i| lcg(&mut seed) * (1.0 + (i % 5) as f32)).collect();
                    h.observe_activation(&x);
                }
                h.finalize(0.01);
                h
            })
            .collect();
        let wrefs: Vec<&[f32]> = ws.iter().map(|w| w.as_slice()).collect();
        let hrefs: Vec<&FullRowGptqHessian> = hs.iter().collect();
        let batch = gpu.compile_nf_batch(Slot::Down, &wrefs, rows, cols, 4, 2, HessianSet::PerMatrix(&hrefs)).unwrap();
        for m in 0..3 {
            let cpu = compile_hadamard_gptq_nf(&ws[m], rows, cols, 4, 2, Some(&hs[m]), 4);
            let (a, b) = (dequant_all(&cpu), dequant_all(&batch[m]));
            assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "matrix {m} differs from CPU");
        }
    }

    #[test]
    fn gpu_forward_matches_cpu_sdot() {
        let mut gpu = MetalGptqCompiler::new().expect("Metal device");
        let (h, inter, n) = (256usize, 128usize, 37usize);
        let mut seed = 5u64;
        let gu: Vec<f32> = (0..2 * inter * h).map(|_| 0.05 * lcg(&mut seed)).collect();
        let dn: Vec<f32> = (0..h * inter).map(|_| 0.05 * lcg(&mut seed)).collect();
        let gu_m = gpu.compile_nf(Slot::GateUp, &gu, 2 * inter, h, 4, 2, None).unwrap();
        let dn_m = gpu.compile_nf(Slot::Down, &dn, h, inter, 4, 2, None).unwrap();
        let acts: Vec<Int8ChunkActivation> = (0..n)
            .map(|_| {
                let mut x: Vec<f32> = (0..h).map(|_| lcg(&mut seed)).collect();
                fwht64_signed_inplace(&mut x);
                let mut a = Int8ChunkActivation::new_zeroed(h);
                a.populate(&x);
                a
            })
            .collect();
        gpu.upload_layer_acts(&acts).unwrap();
        let ids: Vec<u32> = (0..n as u32).rev().collect();
        let sw_gpu = gpu.forward_gate_up_swiglu(0, &ids).unwrap();
        let out_gpu = gpu.forward_down(0).unwrap();
        let mut gu_tmp = vec![0.0f32; 2 * inter];
        let mut sw = vec![0.0f32; inter];
        let mut dn_tmp = vec![0.0f32; h];
        let mut mid = Int8ChunkActivation::new_zeroed(inter);
        for (i, &t) in ids.iter().enumerate() {
            unsafe { nf_sdot_stripe_range_int8(&gu_m, &acts[t as usize], 0, gu_m.num_stripes, gu_tmp.as_mut_ptr()) };
            let (g, u) = gu_tmp.split_at(inter);
            for ((s, &gv), &uv) in sw.iter_mut().zip(g).zip(u) {
                *s = gv / (1.0 + (-gv).exp()) * uv;
            }
            fwht64_signed_inplace(&mut sw);
            let (mut num, mut da, mut db) = (0.0f64, 0.0f64, 0.0f64);
            for (a, b) in sw.iter().zip(&sw_gpu[i * inter..(i + 1) * inter]) {
                num += (*a as f64) * (*b as f64);
                da += (*a as f64).powi(2);
                db += (*b as f64).powi(2);
            }
            assert!(num / (da.sqrt() * db.sqrt()) > 0.9999, "swiglu cosine too low at token {i}");
            mid.populate(&sw);
            unsafe { nf_sdot_stripe_range_int8(&dn_m, &mid, 0, dn_m.num_stripes, dn_tmp.as_mut_ptr()) };
            let (mut num, mut da, mut db) = (0.0f64, 0.0f64, 0.0f64);
            for (a, b) in dn_tmp.iter().zip(&out_gpu[i * h..(i + 1) * h]) {
                num += (*a as f64) * (*b as f64);
                da += (*a as f64).powi(2);
                db += (*b as f64).powi(2);
            }
            assert!(num / (da.sqrt() * db.sqrt()) > 0.999, "down cosine too low at token {i}");
        }
    }
}
