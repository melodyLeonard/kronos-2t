//! Qwen3.5 hybrid dense models: Gated DeltaNet linear-attention layers interleaved with gated
//! full-attention layers (`layer_types`), SwiGLU MLPs and zero-centred `(1 + w)` RMSNorms.
//!
//! The CPU path is sequence-batched and runs layer by layer, which serves three jobs: an
//! unquantized BF16 reference, streaming Hadamard-GPTQ compilation, and a quantized parity check
//! for the Metal engine (`backend::metal_qwen35`).

use crate::kernels::accel::sgemm_xwt;
use crate::kernels::hadamard::{fwht64_signed_inplace, ifwht64_signed_inplace};
use crate::kernels::simd_lut::{NfSdotMatrix, PageAlignedBytes, SIMD_ROW_STRIPE};
use crate::model::compiler::{compile_hadamard_gptq_nf, FullRowGptqHessian};
use crate::storage::safetensors::SafeTensorsStore;
use std::io::{self, Read, Write};
use std::path::Path;

const CACHE_MAGIC: &[u8] = b"K2TQWEN35\n";
const GPTQ_DAMP: f64 = 0.01;
/// `l2norm` epsilon inside the gated delta rule.
const L2_EPS: f32 = 1e-6;
/// Vocabulary rows per streamed `lm_head` block when scoring.
const VOCAB_BLOCK: usize = 16_384;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

/// Text-model hyperparameters from `config.json` (`text_config` when nested).
#[derive(Clone, Debug)]
pub struct Qwen35Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    /// `true` for `full_attention` layers, `false` for Gated DeltaNet layers.
    pub full_attn: Vec<bool>,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub tie_word_embeddings: bool,
    pub eos_token_id: usize,
}

/// Value text after `"key":` inside `src`, or `None`.
fn json_raw<'a>(src: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{key}\"");
    let at = src.find(&pat)? + pat.len();
    let rest = src[at..].trim_start().strip_prefix(':')?;
    Some(rest.trim_start())
}

fn json_num(src: &str, key: &str) -> Option<f64> {
    let v = json_raw(src, key)?;
    let end = v.find([',', '}', ']', '\n']).unwrap_or(v.len());
    v[..end].trim().parse().ok()
}

fn json_bool(src: &str, key: &str) -> Option<bool> {
    let v = json_raw(src, key)?;
    if v.starts_with("true") {
        Some(true)
    } else if v.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

/// The balanced `{...}` object that follows `"key":`.
fn json_object<'a>(src: &'a str, key: &str) -> Option<&'a str> {
    let v = json_raw(src, key)?;
    if !v.starts_with('{') {
        return None;
    }
    let mut depth = 0i32;
    for (i, c) in v.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&v[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

impl Qwen35Config {
    pub fn from_model_dir(model_dir: &Path) -> io::Result<Self> {
        let raw = std::fs::read_to_string(model_dir.join("config.json"))?;
        let tc = json_object(&raw, "text_config").unwrap_or(&raw);
        let need = |k: &str| json_num(tc, k).ok_or_else(|| invalid(format!("config.json: missing {k}")));
        let hidden_size = need("hidden_size")? as usize;
        let num_heads = need("num_attention_heads")? as usize;
        let head_dim = json_num(tc, "head_dim").map_or(hidden_size / num_heads, |v| v as usize);
        let partial = json_num(tc, "partial_rotary_factor").unwrap_or(1.0);
        let num_layers = need("num_hidden_layers")? as usize;
        let lt = json_raw(tc, "layer_types").ok_or_else(|| invalid("config.json: missing layer_types"))?;
        let lt = &lt[..lt.find(']').unwrap_or(lt.len())];
        let full_attn: Vec<bool> = lt
            .split(',')
            .filter(|s| s.contains("attention"))
            .map(|s| s.contains("full_attention"))
            .collect();
        if full_attn.len() != num_layers {
            return Err(invalid(format!("layer_types has {} entries for {num_layers} layers", full_attn.len())));
        }
        let cfg = Self {
            vocab_size: need("vocab_size")? as usize,
            hidden_size,
            intermediate_size: need("intermediate_size")? as usize,
            num_layers,
            num_heads,
            num_kv_heads: need("num_key_value_heads")? as usize,
            head_dim,
            rotary_dim: ((head_dim as f64 * partial) as usize) & !1,
            rope_theta: json_num(tc, "rope_theta").unwrap_or(10_000_000.0) as f32,
            rms_norm_eps: json_num(tc, "rms_norm_eps").unwrap_or(1e-6) as f32,
            full_attn,
            lin_k_heads: need("linear_num_key_heads")? as usize,
            lin_v_heads: need("linear_num_value_heads")? as usize,
            lin_k_dim: need("linear_key_head_dim")? as usize,
            lin_v_dim: need("linear_value_head_dim")? as usize,
            conv_kernel: need("linear_conv_kernel_dim")? as usize,
            tie_word_embeddings: json_bool(&raw, "tie_word_embeddings").unwrap_or(false),
            eos_token_id: need("eos_token_id")? as usize,
        };
        if !hidden_size.is_multiple_of(64) || !cfg.intermediate_size.is_multiple_of(64) || !cfg.lin_v_dim().is_multiple_of(64) {
            return Err(invalid("Qwen3.5: hidden, intermediate and value dims must be multiples of 64"));
        }
        if cfg.lin_v_heads % cfg.lin_k_heads != 0 || num_heads % cfg.num_kv_heads != 0 {
            return Err(invalid("Qwen3.5: head counts must divide evenly"));
        }
        Ok(cfg)
    }

    #[inline]
    pub fn lin_k_total(&self) -> usize {
        self.lin_k_heads * self.lin_k_dim
    }
    #[inline]
    pub fn lin_v_dim(&self) -> usize {
        self.lin_v_heads * self.lin_v_dim
    }
    /// Causal-conv channels: `[q | k | v]`.
    #[inline]
    pub fn conv_dim(&self) -> usize {
        2 * self.lin_k_total() + self.lin_v_dim()
    }
    /// Rows of the fused linear-attention input projection: `[q | k | v | z | b | a]`.
    #[inline]
    pub fn lin_in_rows(&self) -> usize {
        self.conv_dim() + self.lin_v_dim() + 2 * self.lin_v_heads
    }
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }
    /// Rows of the fused full-attention projection: `[q+gate (per head) | k | v]`.
    #[inline]
    pub fn attn_in_rows(&self) -> usize {
        2 * self.q_dim() + 2 * self.kv_dim()
    }
    pub fn num_full_layers(&self) -> usize {
        self.full_attn.iter().filter(|&&f| f).count()
    }
}

/// Tensor-name prefix of the text decoder (`model.language_model.` for the multimodal checkpoint).
fn text_prefix(store: &SafeTensorsStore) -> &'static str {
    if store.has_tensor("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    }
}

fn load_f32(store: &SafeTensorsStore, name: &str) -> io::Result<Vec<f32>> {
    let v = store.tensor(name).map(|t| t.to_f32_vec()).map_err(invalid)?;
    store.release_tensor_pages(name);
    Ok(v)
}

/// Zero-centred RMSNorm weight `1 + w`.
fn load_norm(store: &SafeTensorsStore, name: &str) -> io::Result<Vec<f32>> {
    Ok(load_f32(store, name)?.into_iter().map(|w| 1.0 + w).collect())
}

/// Row-major weights for `y = x W^T`; `rotated` weights expect FWHT-64 rotated inputs.
pub struct Lin {
    pub rows: usize,
    pub cols: usize,
    pub w: Vec<f32>,
    pub rotated: bool,
}

impl Lin {
    fn apply(&self, x: &[f32], t: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; t * self.rows];
        if self.rotated {
            let mut xr = x.to_vec();
            xr.chunks_exact_mut(self.cols).for_each(fwht64_signed_inplace);
            sgemm_xwt(&xr, &self.w, t, self.cols, self.rows, &mut y);
        } else {
            sgemm_xwt(x, &self.w, t, self.cols, self.rows, &mut y);
        }
        y
    }
}

/// Dequantizes every row of `m` into the rotated domain.
fn dequant_rotated(m: &NfSdotMatrix) -> Lin {
    let mut w = vec![0.0f32; m.rows * m.cols];
    let threads = std::thread::available_parallelism().map_or(1, |p| p.get()).clamp(1, 12);
    let per = m.rows.div_ceil(threads).div_ceil(SIMD_ROW_STRIPE) * SIMD_ROW_STRIPE;
    std::thread::scope(|s| {
        for (i, chunk) in w.chunks_mut(per * m.cols).enumerate() {
            s.spawn(move || {
                for (j, row) in chunk.chunks_exact_mut(m.cols).enumerate() {
                    m.dequantize_row_rotated(i * per + j, row);
                }
            });
        }
    });
    Lin { rows: m.rows, cols: m.cols, w, rotated: true }
}

fn rms_norm_rows(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let d = w.len();
    let mut out = vec![0.0f32; x.len()];
    for (xr, o) in x.chunks_exact(d).zip(out.chunks_exact_mut(d)) {
        let ms = xr.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for ((o, &v), &g) in o.iter_mut().zip(xr).zip(w) {
            *o = v * inv * g;
        }
    }
    out
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// Small per-layer tensors of a Gated DeltaNet layer.
#[derive(Clone)]
pub struct LinearParams {
    /// `[conv_dim, conv_kernel]` depthwise causal-conv taps.
    pub conv_w: Vec<f32>,
    pub dt_bias: Vec<f32>,
    /// `-exp(A_log)` per value head.
    pub a_neg: Vec<f32>,
    /// Gated RMSNorm weight over one value head (plain `w`, not `1 + w`).
    pub norm_w: Vec<f32>,
}

/// Small per-layer tensors of a gated full-attention layer.
#[derive(Clone)]
pub struct FullParams {
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
}

/// Gated DeltaNet over one sequence: `proj` is `[t, lin_in_rows]`; returns the gated-norm output
/// `[t, lin_v_dim]` that feeds `out_proj`.
pub fn deltanet_seq(cfg: &Qwen35Config, p: &LinearParams, proj: &[f32], t: usize) -> Vec<f32> {
    let rows = cfg.lin_in_rows();
    let cd = cfg.conv_dim();
    let kk = cfg.conv_kernel;
    let (nk, nv, dk, dv) = (cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim);
    let ktot = cfg.lin_k_total();
    let vdim = cfg.lin_v_dim();

    // Causal depthwise conv + SiLU over the q/k/v channels.
    let mut conv = vec![0.0f32; t * cd];
    for c in 0..cd {
        let taps = &p.conv_w[c * kk..(c + 1) * kk];
        for ti in 0..t {
            let mut acc = 0.0f32;
            for (j, &w) in taps.iter().enumerate() {
                let src = ti as isize + j as isize - (kk as isize - 1);
                if src >= 0 {
                    acc += w * proj[src as usize * rows + c];
                }
            }
            conv[ti * cd + c] = silu(acc);
        }
    }

    let rep = nv / nk;
    let q_scale = 1.0 / (dk as f32).sqrt();
    let mut out = vec![0.0f32; t * vdim];
    let mut state = vec![0.0f32; dk * dv];
    let mut q = vec![0.0f32; dk];
    let mut k = vec![0.0f32; dk];
    let mut o = vec![0.0f32; dv];
    for h in 0..nv {
        state.iter_mut().for_each(|s| *s = 0.0);
        let kh = h / rep;
        for ti in 0..t {
            let row = &conv[ti * cd..(ti + 1) * cd];
            q.copy_from_slice(&row[kh * dk..(kh + 1) * dk]);
            k.copy_from_slice(&row[ktot + kh * dk..ktot + (kh + 1) * dk]);
            let v = &row[2 * ktot + h * dv..2 * ktot + (h + 1) * dv];
            let qn = 1.0 / (q.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
            let kn = 1.0 / (k.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
            q.iter_mut().for_each(|x| *x *= qn * q_scale);
            k.iter_mut().for_each(|x| *x *= kn);
            let prow = &proj[ti * rows..(ti + 1) * rows];
            let b = prow[cd + vdim + h];
            let a = prow[cd + vdim + nv + h];
            let beta = 1.0 / (1.0 + (-b).exp());
            let decay = (p.a_neg[h] * softplus(a + p.dt_bias[h])).exp();
            // S <- decay * S;  delta = beta * (v - S^T k);  S <- S + k delta^T;  o = S^T q
            for j in 0..dv {
                let mut kv = 0.0f32;
                for r in 0..dk {
                    kv += state[r * dv + j] * k[r];
                }
                let delta = beta * (v[j] - decay * kv);
                let mut acc = 0.0f32;
                for r in 0..dk {
                    let s = decay * state[r * dv + j] + k[r] * delta;
                    state[r * dv + j] = s;
                    acc += s * q[r];
                }
                o[j] = acc;
            }
            // Gated RMSNorm: norm(o) * w * silu(z).
            let z = &prow[cd + h * dv..cd + (h + 1) * dv];
            let ms = o.iter().map(|x| x * x).sum::<f32>() / dv as f32;
            let inv = 1.0 / (ms + cfg.rms_norm_eps).sqrt();
            let dst = &mut out[ti * vdim + h * dv..ti * vdim + (h + 1) * dv];
            for j in 0..dv {
                dst[j] = o[j] * inv * p.norm_w[j] * silu(z[j]);
            }
        }
    }
    out
}

/// `[cos_0..cos_{r/2}, sin_0..sin_{r/2}]` for position `pos` over the rotary dims.
pub fn rope_row(cfg: &Qwen35Config, pos: usize, out: &mut [f32]) {
    let half = cfg.rotary_dim / 2;
    for i in 0..half {
        let inv = cfg.rope_theta.powf(-((2 * i) as f32) / cfg.rotary_dim as f32);
        let ang = pos as f32 * inv;
        out[i] = ang.cos();
        out[half + i] = ang.sin();
    }
}

/// Builds the `[max_pos, rotary_dim]` cos/sin table.
pub fn rope_table(cfg: &Qwen35Config, max_pos: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; max_pos * cfg.rotary_dim];
    for (p, row) in t.chunks_exact_mut(cfg.rotary_dim).enumerate() {
        rope_row(cfg, p, row);
    }
    t
}

fn head_rms(x: &mut [f32], w: &[f32], eps: f32) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter_mut().zip(w).for_each(|(v, &g)| *v *= inv * g);
}

fn apply_partial_rope(x: &mut [f32], cs: &[f32]) {
    let half = cs.len() / 2;
    for i in 0..half {
        let (c, s) = (cs[i], cs[half + i]);
        let (x1, x2) = (x[i], x[i + half]);
        x[i] = x1 * c - x2 * s;
        x[i + half] = x2 * c + x1 * s;
    }
}

/// Gated causal GQA over one sequence: `proj` is `[t, attn_in_rows]`; returns `[t, q_dim]`.
pub fn full_attn_seq(cfg: &Qwen35Config, p: &FullParams, proj: &[f32], t: usize) -> Vec<f32> {
    let (nh, nkv, hd) = (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim);
    let rows = cfg.attn_in_rows();
    let qg = 2 * cfg.q_dim();
    let kvd = cfg.kv_dim();
    let group = nh / nkv;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut cs = vec![0.0f32; cfg.rotary_dim];

    let mut qs = vec![0.0f32; t * cfg.q_dim()];
    let mut gates = vec![0.0f32; t * cfg.q_dim()];
    let mut ks = vec![0.0f32; t * kvd];
    let mut vs = vec![0.0f32; t * kvd];
    for ti in 0..t {
        rope_row(cfg, ti, &mut cs);
        let row = &proj[ti * rows..(ti + 1) * rows];
        for h in 0..nh {
            let q = &mut qs[ti * cfg.q_dim() + h * hd..ti * cfg.q_dim() + (h + 1) * hd];
            q.copy_from_slice(&row[h * 2 * hd..h * 2 * hd + hd]);
            gates[ti * cfg.q_dim() + h * hd..ti * cfg.q_dim() + (h + 1) * hd]
                .copy_from_slice(&row[h * 2 * hd + hd..(h + 1) * 2 * hd]);
            head_rms(q, &p.q_norm, cfg.rms_norm_eps);
            apply_partial_rope(&mut q[..cfg.rotary_dim], &cs);
        }
        for h in 0..nkv {
            let k = &mut ks[ti * kvd + h * hd..ti * kvd + (h + 1) * hd];
            k.copy_from_slice(&row[qg + h * hd..qg + (h + 1) * hd]);
            head_rms(k, &p.k_norm, cfg.rms_norm_eps);
            apply_partial_rope(&mut k[..cfg.rotary_dim], &cs);
            vs[ti * kvd + h * hd..ti * kvd + (h + 1) * hd]
                .copy_from_slice(&row[qg + kvd + h * hd..qg + kvd + (h + 1) * hd]);
        }
    }

    let mut out = vec![0.0f32; t * cfg.q_dim()];
    let mut scores = vec![0.0f32; t];
    for ti in 0..t {
        for h in 0..nh {
            let kvh = h / group;
            let q = &qs[ti * cfg.q_dim() + h * hd..ti * cfg.q_dim() + (h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            for (tj, s) in scores[..=ti].iter_mut().enumerate() {
                let k = &ks[tj * kvd + kvh * hd..tj * kvd + (kvh + 1) * hd];
                *s = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                mx = mx.max(*s);
            }
            let mut den = 0.0f32;
            for s in scores[..=ti].iter_mut() {
                *s = (*s - mx).exp();
                den += *s;
            }
            let o = &mut out[ti * cfg.q_dim() + h * hd..ti * cfg.q_dim() + (h + 1) * hd];
            for (tj, &s) in scores[..=ti].iter().enumerate() {
                let v = &vs[tj * kvd + kvh * hd..tj * kvd + (kvh + 1) * hd];
                let wgt = s / den;
                o.iter_mut().zip(v).for_each(|(a, &b)| *a += wgt * b);
            }
            let g = &gates[ti * cfg.q_dim() + h * hd..ti * cfg.q_dim() + (h + 1) * hd];
            o.iter_mut().zip(g).for_each(|(a, &gv)| *a *= 1.0 / (1.0 + (-gv).exp()));
        }
    }
    out
}

/// `silu(gate) * up` from a `[t, 2 * inter]` buffer laid out `[gate | up]` per token.
fn swiglu_rows(gu: &[f32], inter: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; gu.len() / 2];
    for (g, o) in gu.chunks_exact(2 * inter).zip(out.chunks_exact_mut(inter)) {
        for i in 0..inter {
            o[i] = silu(g[i]) * g[inter + i];
        }
    }
    out
}

/// Runs `f` on each sequence slice of a `[sum(lens), width]` buffer and concatenates the results.
fn per_seq(lens: &[usize], x: &[f32], width: usize, mut f: impl FnMut(&[f32], usize) -> Vec<f32>) -> Vec<f32> {
    let mut out = Vec::new();
    let mut off = 0;
    for &n in lens {
        out.extend(f(&x[off * width..(off + n) * width], n));
        off += n;
    }
    out
}

fn add_into(dst: &mut [f32], src: &[f32]) {
    dst.iter_mut().zip(src).for_each(|(a, b)| *a += b);
}

/// One quantized decoder layer.
pub struct QLayer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub mixer: QMixer,
    /// `[gate | up]` rows, `2 * intermediate_size` x `hidden_size`.
    pub gate_up: NfSdotMatrix,
    pub down: NfSdotMatrix,
}

pub enum QMixer {
    Linear { in_proj: NfSdotMatrix, params: LinearParams, out_proj: NfSdotMatrix },
    Full { in_proj: NfSdotMatrix, params: FullParams, out_proj: NfSdotMatrix },
}

impl QMixer {
    pub fn in_proj(&self) -> &NfSdotMatrix {
        match self {
            QMixer::Linear { in_proj, .. } | QMixer::Full { in_proj, .. } => in_proj,
        }
    }
    pub fn out_proj(&self) -> &NfSdotMatrix {
        match self {
            QMixer::Linear { out_proj, .. } | QMixer::Full { out_proj, .. } => out_proj,
        }
    }
}

/// A compiled Qwen3.5 text model: NF-quantized projections plus small FP32 tensors.
pub struct Qwen35Model {
    pub cfg: Qwen35Config,
    /// NF4 embedding rows in the rotated domain (`dequantize_row_rotated` + `ifwht64`).
    pub embed: NfSdotMatrix,
    pub layers: Vec<QLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: NfSdotMatrix,
}

/// Quantization recipe: bits for layer projections and for `lm_head`.
#[derive(Clone, Copy, Debug)]
pub struct QuantSpec {
    pub bits: usize,
    pub head_bits: usize,
    pub scale_group_chunks: usize,
    pub gptq: bool,
}

fn quantize(w: &[f32], rows: usize, cols: usize, bits: usize, sgc: usize, hess: Option<&FullRowGptqHessian>, threads: usize) -> NfSdotMatrix {
    compile_hadamard_gptq_nf(w, rows, cols, bits, sgc, hess, threads)
}

fn hessian_of(x: &[f32], cols: usize) -> FullRowGptqHessian {
    let mut h = FullRowGptqHessian::new(cols);
    let mut buf = vec![0.0f32; cols];
    for row in x.chunks_exact(cols) {
        buf.copy_from_slice(row);
        fwht64_signed_inplace(&mut buf);
        h.observe_prerotated(&buf);
    }
    h.finalize(GPTQ_DAMP);
    h
}

/// Loads and concatenates row blocks of several projections into one matrix.
fn load_rows(store: &SafeTensorsStore, names: &[String]) -> io::Result<Vec<f32>> {
    let mut w = Vec::new();
    for n in names {
        w.extend(load_f32(store, n)?);
    }
    Ok(w)
}

fn embed_rows_bf16(store: &SafeTensorsStore, pfx: &str, seqs_flat: &[usize], h: usize) -> io::Result<Vec<f32>> {
    let name = format!("{pfx}embed_tokens.weight");
    let t = store.tensor(&name).map_err(invalid)?;
    let mut x = vec![0.0f32; seqs_flat.len() * h];
    for (i, &tok) in seqs_flat.iter().enumerate() {
        t.copy_row_f32(tok, &mut x[i * h..(i + 1) * h]);
    }
    Ok(x)
}

/// Mixer input-projection names, in fused row order.
fn mixer_in_names(cfg: &Qwen35Config, pfx: &str, l: usize) -> Vec<String> {
    let lp = format!("{pfx}layers.{l}.");
    if cfg.full_attn[l] {
        ["q_proj", "k_proj", "v_proj"].iter().map(|n| format!("{lp}self_attn.{n}.weight")).collect()
    } else {
        ["in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"]
            .iter()
            .map(|n| format!("{lp}linear_attn.{n}.weight"))
            .collect()
    }
}

fn mixer_out_name(cfg: &Qwen35Config, pfx: &str, l: usize) -> String {
    if cfg.full_attn[l] {
        format!("{pfx}layers.{l}.self_attn.o_proj.weight")
    } else {
        format!("{pfx}layers.{l}.linear_attn.out_proj.weight")
    }
}

fn load_linear_params(cfg: &Qwen35Config, store: &SafeTensorsStore, pfx: &str, l: usize) -> io::Result<LinearParams> {
    let lp = format!("{pfx}layers.{l}.linear_attn.");
    let conv_w = load_f32(store, &format!("{lp}conv1d.weight"))?;
    if conv_w.len() != cfg.conv_dim() * cfg.conv_kernel {
        return Err(invalid(format!("layer {l}: conv1d has {} weights", conv_w.len())));
    }
    Ok(LinearParams {
        conv_w,
        dt_bias: load_f32(store, &format!("{lp}dt_bias"))?,
        a_neg: load_f32(store, &format!("{lp}A_log"))?.into_iter().map(|a| -a.exp()).collect(),
        norm_w: load_f32(store, &format!("{lp}norm.weight"))?,
    })
}

fn load_full_params(store: &SafeTensorsStore, pfx: &str, l: usize) -> io::Result<FullParams> {
    let lp = format!("{pfx}layers.{l}.self_attn.");
    Ok(FullParams {
        q_norm: load_norm(store, &format!("{lp}q_norm.weight"))?,
        k_norm: load_norm(store, &format!("{lp}k_norm.weight"))?,
    })
}

/// Mixer core (everything between the input and output projections) per sequence.
fn mixer_core(cfg: &Qwen35Config, l: usize, lin: Option<&LinearParams>, full: Option<&FullParams>, lens: &[usize], proj: &[f32]) -> Vec<f32> {
    if cfg.full_attn[l] {
        let p = full.expect("full-attention params");
        per_seq(lens, proj, cfg.attn_in_rows(), |x, n| full_attn_seq(cfg, p, x, n))
    } else {
        let p = lin.expect("linear-attention params");
        per_seq(lens, proj, cfg.lin_in_rows(), |x, n| deltanet_seq(cfg, p, x, n))
    }
}

/// Mean negative log-likelihood of `targets[i]` given normalized hidden rows `hn`, streaming the
/// vocabulary through `rows_fn(start, end) -> Lin`.
fn nll_streamed(hn: &[f32], h: usize, targets: &[Option<usize>], vocab: usize, mut rows_fn: impl FnMut(usize, usize) -> Lin) -> (f64, usize) {
    let t = targets.len();
    let mut mx = vec![f32::NEG_INFINITY; t];
    let mut den = vec![0.0f64; t];
    let mut tgt = vec![0.0f32; t];
    let mut start = 0;
    while start < vocab {
        let end = (start + VOCAB_BLOCK).min(vocab);
        let lin = rows_fn(start, end);
        let logits = lin.apply(hn, t);
        let n = end - start;
        for i in 0..t {
            let row = &logits[i * n..(i + 1) * n];
            let bmax = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if bmax > mx[i] {
                den[i] *= ((mx[i] - bmax) as f64).exp();
                mx[i] = bmax;
            }
            den[i] += row.iter().map(|&v| ((v - mx[i]) as f64).exp()).sum::<f64>();
            match targets[i] {
                Some(tok) if (start..end).contains(&tok) => tgt[i] = row[tok - start],
                _ => {}
            }
        }
        start = end;
    }
    let mut nll = 0.0f64;
    let mut count = 0;
    for i in 0..t {
        if targets[i].is_some() {
            nll += mx[i] as f64 + den[i].ln() - tgt[i] as f64;
            count += 1;
        }
    }
    let _ = h;
    (nll, count)
}

fn next_targets(seqs: &[Vec<usize>]) -> Vec<Option<usize>> {
    seqs.iter()
        .flat_map(|s| (0..s.len()).map(move |i| s.get(i + 1).copied()))
        .collect()
}

/// Perplexity of the unquantized BF16 checkpoint on `seqs` (each scored from position 0).
pub fn reference_ppl(model_dir: &Path, seqs: &[Vec<usize>], log: &mut dyn FnMut(&str)) -> io::Result<f64> {
    let cfg = Qwen35Config::from_model_dir(model_dir)?;
    let store = SafeTensorsStore::open_dir(model_dir)?;
    let pfx = text_prefix(&store);
    let h = cfg.hidden_size;
    let lens: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
    let flat: Vec<usize> = seqs.concat();
    let t = flat.len();
    let mut x = embed_rows_bf16(&store, pfx, &flat, h)?;
    for l in 0..cfg.num_layers {
        let t0 = std::time::Instant::now();
        let lp = format!("{pfx}layers.{l}.");
        let xn = rms_norm_rows(&x, &load_norm(&store, &format!("{lp}input_layernorm.weight"))?, cfg.rms_norm_eps);
        let in_w = load_rows(&store, &mixer_in_names(&cfg, pfx, l))?;
        let in_lin = Lin { rows: in_w.len() / h, cols: h, w: in_w, rotated: false };
        let proj = in_lin.apply(&xn, t);
        drop(in_lin);
        let (lin_p, full_p) = if cfg.full_attn[l] {
            (None, Some(load_full_params(&store, pfx, l)?))
        } else {
            (Some(load_linear_params(&cfg, &store, pfx, l)?), None)
        };
        let core = mixer_core(&cfg, l, lin_p.as_ref(), full_p.as_ref(), &lens, &proj);
        let out_w = load_f32(&store, &mixer_out_name(&cfg, pfx, l))?;
        let cols = out_w.len() / h;
        add_into(&mut x, &Lin { rows: h, cols, w: out_w, rotated: false }.apply(&core, t));

        let xn = rms_norm_rows(&x, &load_norm(&store, &format!("{lp}post_attention_layernorm.weight"))?, cfg.rms_norm_eps);
        let gu_w = load_rows(&store, &[format!("{lp}mlp.gate_proj.weight"), format!("{lp}mlp.up_proj.weight")])?;
        let gu = Lin { rows: 2 * cfg.intermediate_size, cols: h, w: gu_w, rotated: false }.apply(&xn, t);
        let act = swiglu_rows(&gu, cfg.intermediate_size);
        let dn = load_f32(&store, &format!("{lp}mlp.down_proj.weight"))?;
        add_into(&mut x, &Lin { rows: h, cols: cfg.intermediate_size, w: dn, rotated: false }.apply(&act, t));
        log(&format!("reference layer {l}/{} ({:.1}s)", cfg.num_layers, t0.elapsed().as_secs_f64()));
    }
    let hn = rms_norm_rows(&x, &load_norm(&store, &format!("{pfx}norm.weight"))?, cfg.rms_norm_eps);
    let head_name = if cfg.tie_word_embeddings { format!("{pfx}embed_tokens.weight") } else { "lm_head.weight".to_string() };
    let head = store.tensor(&head_name).map_err(invalid)?;
    let (nll, n) = nll_streamed(&hn, h, &next_targets(seqs), cfg.vocab_size, |s, e| {
        let mut w = vec![0.0f32; (e - s) * h];
        for r in s..e {
            head.copy_row_f32(r, &mut w[(r - s) * h..(r - s + 1) * h]);
        }
        Lin { rows: e - s, cols: h, w, rotated: false }
    });
    Ok((nll / n.max(1) as f64).exp())
}

/// Compiles `model_dir` layer by layer with Hadamard GPTQ, propagating calibration activations
/// through the already-quantized layers.
pub fn compile(model_dir: &Path, spec: QuantSpec, calib: &[Vec<usize>], threads: usize, log: &mut dyn FnMut(&str)) -> io::Result<Qwen35Model> {
    let cfg = Qwen35Config::from_model_dir(model_dir)?;
    let store = SafeTensorsStore::open_dir(model_dir)?;
    let pfx = text_prefix(&store);
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let (bits, sgc) = (spec.bits, spec.scale_group_chunks);
    let lens: Vec<usize> = calib.iter().map(|s| s.len()).collect();
    let flat: Vec<usize> = calib.concat();
    let t = flat.len();
    let hess = |x: &[f32], cols: usize| spec.gptq.then(|| hessian_of(x, cols));

    let embed = {
        let name = format!("{pfx}embed_tokens.weight");
        let words = store.tensor(&name).map(|v| v.to_bf16_vec()).map_err(invalid)?;
        store.release_tensor_pages(&name);
        NfSdotMatrix::from_bf16_u16_embeddings_nf4(&words, cfg.vocab_size, h, sgc, threads)
    };
    let mut x = embed_rows_bf16(&store, pfx, &flat, h)?;
    let mut layers = Vec::with_capacity(cfg.num_layers);
    for l in 0..cfg.num_layers {
        let t0 = std::time::Instant::now();
        let lp = format!("{pfx}layers.{l}.");
        let input_norm = load_norm(&store, &format!("{lp}input_layernorm.weight"))?;
        let post_norm = load_norm(&store, &format!("{lp}post_attention_layernorm.weight"))?;

        let xn = rms_norm_rows(&x, &input_norm, cfg.rms_norm_eps);
        let in_w = load_rows(&store, &mixer_in_names(&cfg, pfx, l))?;
        let in_rows = in_w.len() / h;
        let in_q = quantize(&in_w, in_rows, h, bits, sgc, hess(&xn, h).as_ref(), threads);
        drop(in_w);
        let proj = dequant_rotated(&in_q).apply(&xn, t);
        let (lin_p, full_p) = if cfg.full_attn[l] {
            (None, Some(load_full_params(&store, pfx, l)?))
        } else {
            (Some(load_linear_params(&cfg, &store, pfx, l)?), None)
        };
        let core = mixer_core(&cfg, l, lin_p.as_ref(), full_p.as_ref(), &lens, &proj);
        drop(proj);
        let out_w = load_f32(&store, &mixer_out_name(&cfg, pfx, l))?;
        let core_cols = out_w.len() / h;
        let out_q = quantize(&out_w, h, core_cols, bits, sgc, hess(&core, core_cols).as_ref(), threads);
        drop(out_w);
        add_into(&mut x, &dequant_rotated(&out_q).apply(&core, t));
        drop(core);

        let xn = rms_norm_rows(&x, &post_norm, cfg.rms_norm_eps);
        let gu_w = load_rows(&store, &[format!("{lp}mlp.gate_proj.weight"), format!("{lp}mlp.up_proj.weight")])?;
        let gu_q = quantize(&gu_w, 2 * inter, h, bits, sgc, hess(&xn, h).as_ref(), threads);
        drop(gu_w);
        let act = swiglu_rows(&dequant_rotated(&gu_q).apply(&xn, t), inter);
        let dn_w = load_f32(&store, &format!("{lp}mlp.down_proj.weight"))?;
        let dn_q = quantize(&dn_w, h, inter, bits, sgc, hess(&act, inter).as_ref(), threads);
        drop(dn_w);
        add_into(&mut x, &dequant_rotated(&dn_q).apply(&act, t));

        let mixer = match (lin_p, full_p) {
            (Some(params), _) => QMixer::Linear { in_proj: in_q, params, out_proj: out_q },
            (_, Some(params)) => QMixer::Full { in_proj: in_q, params, out_proj: out_q },
            _ => unreachable!(),
        };
        layers.push(QLayer { input_norm, post_norm, mixer, gate_up: gu_q, down: dn_q });
        log(&format!(
            "compiled layer {}/{} ({}) in {:.1}s",
            l + 1,
            cfg.num_layers,
            if cfg.full_attn[l] { "attention" } else { "deltanet" },
            t0.elapsed().as_secs_f64()
        ));
    }
    let final_norm = load_norm(&store, &format!("{pfx}norm.weight"))?;
    let hn = rms_norm_rows(&x, &final_norm, cfg.rms_norm_eps);
    let head_name = if cfg.tie_word_embeddings { format!("{pfx}embed_tokens.weight") } else { "lm_head.weight".to_string() };
    let t0 = std::time::Instant::now();
    let head_w = load_f32(&store, &head_name)?;
    let lm_head = quantize(&head_w, cfg.vocab_size, h, spec.head_bits, sgc, hess(&hn, h).as_ref(), threads);
    log(&format!("compiled lm_head in {:.1}s", t0.elapsed().as_secs_f64()));
    Ok(Qwen35Model { cfg, embed, layers, final_norm, lm_head })
}

impl Qwen35Model {
    /// Embedding row of `tok` in the normal (unrotated) domain.
    pub fn embed_into(&self, tok: usize, out: &mut [f32]) {
        self.embed.dequantize_row_rotated(tok, out);
        ifwht64_signed_inplace(out);
    }

    /// Perplexity of the quantized model on the CPU (dequantized weights, FP32 activations).
    pub fn ppl_cpu(&self, seqs: &[Vec<usize>]) -> f64 {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let lens: Vec<usize> = seqs.iter().map(|s| s.len()).collect();
        let flat: Vec<usize> = seqs.concat();
        let t = flat.len();
        let mut x = vec![0.0f32; t * h];
        for (i, &tok) in flat.iter().enumerate() {
            self.embed_into(tok, &mut x[i * h..(i + 1) * h]);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let xn = rms_norm_rows(&x, &layer.input_norm, cfg.rms_norm_eps);
            let proj = dequant_rotated(layer.mixer.in_proj()).apply(&xn, t);
            let core = match &layer.mixer {
                QMixer::Linear { params, .. } => mixer_core(cfg, l, Some(params), None, &lens, &proj),
                QMixer::Full { params, .. } => mixer_core(cfg, l, None, Some(params), &lens, &proj),
            };
            add_into(&mut x, &dequant_rotated(layer.mixer.out_proj()).apply(&core, t));
            let xn = rms_norm_rows(&x, &layer.post_norm, cfg.rms_norm_eps);
            let act = swiglu_rows(&dequant_rotated(&layer.gate_up).apply(&xn, t), cfg.intermediate_size);
            add_into(&mut x, &dequant_rotated(&layer.down).apply(&act, t));
        }
        let hn = rms_norm_rows(&x, &self.final_norm, cfg.rms_norm_eps);
        let (nll, n) = nll_streamed(&hn, h, &next_targets(seqs), cfg.vocab_size, |s, e| {
            let mut w = vec![0.0f32; (e - s) * h];
            for r in s..e {
                self.lm_head.dequantize_row_rotated(r, &mut w[(r - s) * h..(r - s + 1) * h]);
            }
            Lin { rows: e - s, cols: h, w, rotated: true }
        });
        (nll / n.max(1) as f64).exp()
    }

    /// Logits `[t, vocab]` of one sequence on the CPU (small models and tests).
    pub fn logits_cpu(&self, seq: &[usize]) -> Vec<f32> {
        let cfg = &self.cfg;
        let h = cfg.hidden_size;
        let t = seq.len();
        let lens = [t];
        let mut x = vec![0.0f32; t * h];
        for (i, &tok) in seq.iter().enumerate() {
            self.embed_into(tok, &mut x[i * h..(i + 1) * h]);
        }
        for (l, layer) in self.layers.iter().enumerate() {
            let xn = rms_norm_rows(&x, &layer.input_norm, cfg.rms_norm_eps);
            let proj = dequant_rotated(layer.mixer.in_proj()).apply(&xn, t);
            let core = match &layer.mixer {
                QMixer::Linear { params, .. } => mixer_core(cfg, l, Some(params), None, &lens, &proj),
                QMixer::Full { params, .. } => mixer_core(cfg, l, None, Some(params), &lens, &proj),
            };
            add_into(&mut x, &dequant_rotated(layer.mixer.out_proj()).apply(&core, t));
            let xn = rms_norm_rows(&x, &layer.post_norm, cfg.rms_norm_eps);
            let act = swiglu_rows(&dequant_rotated(&layer.gate_up).apply(&xn, t), cfg.intermediate_size);
            add_into(&mut x, &dequant_rotated(&layer.down).apply(&act, t));
        }
        let hn = rms_norm_rows(&x, &self.final_norm, cfg.rms_norm_eps);
        dequant_rotated(&self.lm_head).apply(&hn, t)
    }

    /// Random model with the given shapes, for kernel parity tests.
    pub fn synthetic(cfg: Qwen35Config, seed: u64) -> Self {
        let mut s = seed;
        let mut rnd = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * scale
                })
                .collect()
        };
        let h = cfg.hidden_size;
        let q = |w: Vec<f32>, rows: usize, cols: usize| compile_hadamard_gptq_nf(&w, rows, cols, 4, 2, None, 4);
        let ws = 1.0 / (h as f32).sqrt();
        let mut layers = Vec::new();
        for l in 0..cfg.num_layers {
            let mixer = if cfg.full_attn[l] {
                QMixer::Full {
                    in_proj: q(rnd(cfg.attn_in_rows() * h, 2.0 * ws), cfg.attn_in_rows(), h),
                    params: FullParams {
                        q_norm: rnd(cfg.head_dim, 0.4).into_iter().map(|v| 1.0 + v).collect(),
                        k_norm: rnd(cfg.head_dim, 0.4).into_iter().map(|v| 1.0 + v).collect(),
                    },
                    out_proj: q(rnd(h * cfg.q_dim(), 2.0 / (cfg.q_dim() as f32).sqrt()), h, cfg.q_dim()),
                }
            } else {
                QMixer::Linear {
                    in_proj: q(rnd(cfg.lin_in_rows() * h, 2.0 * ws), cfg.lin_in_rows(), h),
                    params: LinearParams {
                        conv_w: rnd(cfg.conv_dim() * cfg.conv_kernel, 1.0),
                        dt_bias: rnd(cfg.lin_v_heads, 1.0),
                        a_neg: rnd(cfg.lin_v_heads, 1.0).into_iter().map(|v| -(v + 0.6).exp()).collect(),
                        norm_w: rnd(cfg.lin_v_dim, 0.4).into_iter().map(|v| 1.0 + v).collect(),
                    },
                    out_proj: q(rnd(h * cfg.lin_v_dim(), 2.0 / (cfg.lin_v_dim() as f32).sqrt()), h, cfg.lin_v_dim()),
                }
            };
            layers.push(QLayer {
                input_norm: rnd(h, 0.4).into_iter().map(|v| 1.0 + v).collect(),
                post_norm: rnd(h, 0.4).into_iter().map(|v| 1.0 + v).collect(),
                mixer,
                gate_up: q(rnd(2 * cfg.intermediate_size * h, 2.0 * ws), 2 * cfg.intermediate_size, h),
                down: q(rnd(h * cfg.intermediate_size, 2.0 / (cfg.intermediate_size as f32).sqrt()), h, cfg.intermediate_size),
            });
        }
        let emb: Vec<u16> = rnd(cfg.vocab_size * h, 2.0)
            .into_iter()
            .map(crate::kernels::simd_lut::f32_to_bf16)
            .collect();
        let embed = NfSdotMatrix::from_bf16_u16_embeddings_nf4(&emb, cfg.vocab_size, h, 2, 4);
        let lm_head = q(rnd(cfg.vocab_size * h, 2.0 * ws), cfg.vocab_size, h);
        Self { final_norm: vec![1.0; h], embed, layers, lm_head, cfg }
    }

    /// Bytes of quantized weights the GPU reads per decoded token (all layers plus `lm_head`).
    pub fn weight_bytes_per_token(&self) -> usize {
        let m = |x: &NfSdotMatrix| x.pair_stream.len() + x.tail_stream.len();
        self.layers
            .iter()
            .map(|l| m(l.mixer.in_proj()) + m(l.mixer.out_proj()) + m(&l.gate_up) + m(&l.down))
            .sum::<usize>()
            + m(&self.lm_head)
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let tmp = path.with_extension("k2tb.partial");
        let mut w = io::BufWriter::new(std::fs::File::create(&tmp)?);
        w.write_all(CACHE_MAGIC)?;
        w_u64(&mut w, self.layers.len())?;
        w_mat(&mut w, &self.embed)?;
        for l in &self.layers {
            w_f32s(&mut w, &l.input_norm)?;
            w_f32s(&mut w, &l.post_norm)?;
            match &l.mixer {
                QMixer::Linear { in_proj, params, out_proj } => {
                    w_u64(&mut w, 0)?;
                    w_f32s(&mut w, &params.conv_w)?;
                    w_f32s(&mut w, &params.dt_bias)?;
                    w_f32s(&mut w, &params.a_neg)?;
                    w_f32s(&mut w, &params.norm_w)?;
                    w_mat(&mut w, in_proj)?;
                    w_mat(&mut w, out_proj)?;
                }
                QMixer::Full { in_proj, params, out_proj } => {
                    w_u64(&mut w, 1)?;
                    w_f32s(&mut w, &params.q_norm)?;
                    w_f32s(&mut w, &params.k_norm)?;
                    w_mat(&mut w, in_proj)?;
                    w_mat(&mut w, out_proj)?;
                }
            }
            w_mat(&mut w, &l.gate_up)?;
            w_mat(&mut w, &l.down)?;
        }
        w_f32s(&mut w, &self.final_norm)?;
        w_mat(&mut w, &self.lm_head)?;
        w.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        std::fs::rename(&tmp, path)
    }

    pub fn load(model_dir: &Path, path: &Path) -> io::Result<Self> {
        let cfg = Qwen35Config::from_model_dir(model_dir)?;
        let mut r = io::BufReader::with_capacity(1 << 20, std::fs::File::open(path)?);
        let mut magic = vec![0u8; CACHE_MAGIC.len()];
        r.read_exact(&mut magic)?;
        if magic != CACHE_MAGIC {
            return Err(invalid("not a Qwen3.5 Kronos cache"));
        }
        let n = r_u64(&mut r)?;
        if n != cfg.num_layers {
            return Err(invalid(format!("cache has {n} layers, config has {}", cfg.num_layers)));
        }
        let embed = r_mat(&mut r)?;
        let mut layers = Vec::with_capacity(n);
        for l in 0..n {
            let input_norm = r_f32s(&mut r)?;
            let post_norm = r_f32s(&mut r)?;
            let tag = r_u64(&mut r)?;
            if (tag == 1) != cfg.full_attn[l] {
                return Err(invalid(format!("layer {l}: cache layer type does not match config")));
            }
            let mixer = if tag == 0 {
                let params = LinearParams {
                    conv_w: r_f32s(&mut r)?,
                    dt_bias: r_f32s(&mut r)?,
                    a_neg: r_f32s(&mut r)?,
                    norm_w: r_f32s(&mut r)?,
                };
                QMixer::Linear { in_proj: r_mat(&mut r)?, params, out_proj: r_mat(&mut r)? }
            } else {
                let params = FullParams { q_norm: r_f32s(&mut r)?, k_norm: r_f32s(&mut r)? };
                QMixer::Full { in_proj: r_mat(&mut r)?, params, out_proj: r_mat(&mut r)? }
            };
            layers.push(QLayer { input_norm, post_norm, mixer, gate_up: r_mat(&mut r)?, down: r_mat(&mut r)? });
        }
        let final_norm = r_f32s(&mut r)?;
        let lm_head = r_mat(&mut r)?;
        Ok(Self { cfg, embed, layers, final_norm, lm_head })
    }
}

fn w_u64(w: &mut impl Write, v: usize) -> io::Result<()> {
    w.write_all(&(v as u64).to_le_bytes())
}

fn w_f32s(w: &mut impl Write, v: &[f32]) -> io::Result<()> {
    w_u64(w, v.len())?;
    v.iter().try_for_each(|x| w.write_all(&x.to_le_bytes()))
}

fn w_mat(w: &mut impl Write, m: &NfSdotMatrix) -> io::Result<()> {
    for v in [m.rows, m.cols, m.bits, m.scale_group_chunks] {
        w_u64(w, v)?;
    }
    let cb: [u8; 16] = std::array::from_fn(|i| m.codebook[i] as u8);
    w.write_all(&cb)?;
    w_u64(w, m.pair_stream.len())?;
    w.write_all(&m.pair_stream)?;
    w_u64(w, m.tail_stream.len())?;
    w.write_all(&m.tail_stream)
}

fn r_u64(r: &mut impl Read) -> io::Result<usize> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b) as usize)
}

fn r_f32s(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let n = r_u64(r)?;
    let mut b = vec![0u8; n * 4];
    r.read_exact(&mut b)?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn r_bytes(r: &mut impl Read) -> io::Result<PageAlignedBytes> {
    let n = r_u64(r)?;
    let mut buf = PageAlignedBytes::zeroed(n);
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn r_mat(r: &mut impl Read) -> io::Result<NfSdotMatrix> {
    let rows = r_u64(r)?;
    let cols = r_u64(r)?;
    let bits = r_u64(r)?;
    let scale_group_chunks = r_u64(r)?;
    let mut cb = [0u8; 16];
    r.read_exact(&mut cb)?;
    let pair_stream = r_bytes(r)?;
    let tail_stream = r_bytes(r)?;
    let num_stripes = rows.div_ceil(SIMD_ROW_STRIPE);
    let num_chunks = cols / 64;
    Ok(NfSdotMatrix {
        rows,
        cols,
        bits,
        scale_group_chunks,
        num_stripes,
        num_pairs: num_stripes / 2,
        num_chunks,
        num_scale_groups: num_chunks.div_ceil(scale_group_chunks),
        codebook: std::array::from_fn(|i| cb[i] as i8),
        pair_stream,
        tail_stream,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 32,
            hidden_size: 64,
            intermediate_size: 128,
            num_layers: 2,
            num_heads: 2,
            num_kv_heads: 1,
            head_dim: 64,
            rotary_dim: 16,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-6,
            full_attn: vec![false, true],
            lin_k_heads: 1,
            lin_v_heads: 2,
            lin_k_dim: 16,
            lin_v_dim: 32,
            conv_kernel: 4,
            tie_word_embeddings: false,
            eos_token_id: 0,
        }
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// A token's output never depends on later tokens.
    #[test]
    fn deltanet_is_causal_and_finite() {
        let cfg = tiny_cfg();
        let t = 5;
        let p = LinearParams {
            conv_w: rand_vec(cfg.conv_dim() * 4, 1),
            dt_bias: rand_vec(2, 2),
            a_neg: vec![-0.5, -1.5],
            norm_w: vec![1.0; cfg.lin_v_dim],
        };
        let proj = rand_vec(t * cfg.lin_in_rows(), 3);
        let got = deltanet_seq(&cfg, &p, &proj, t);
        assert_eq!(got.len(), t * cfg.lin_v_dim());
        assert!(got.iter().all(|v| v.is_finite()));
        // Causality: the first token's output does not depend on later tokens.
        let mut proj2 = proj.clone();
        for v in &mut proj2[cfg.lin_in_rows()..] {
            *v += 1.0;
        }
        let got2 = deltanet_seq(&cfg, &p, &proj2, t);
        let d0 = cfg.lin_v_dim();
        assert!(got[..d0].iter().zip(&got2[..d0]).all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn partial_rope_leaves_tail_dims() {
        let cfg = tiny_cfg();
        let mut cs = vec![0.0; cfg.rotary_dim];
        rope_row(&cfg, 7, &mut cs);
        let mut x = rand_vec(cfg.head_dim, 9);
        let before = x.clone();
        apply_partial_rope(&mut x[..cfg.rotary_dim], &cs);
        assert_eq!(&x[cfg.rotary_dim..], &before[cfg.rotary_dim..]);
        let n0: f32 = before[..cfg.rotary_dim].iter().map(|v| v * v).sum();
        let n1: f32 = x[..cfg.rotary_dim].iter().map(|v| v * v).sum();
        assert!((n0 - n1).abs() < 1e-4);
    }

    #[test]
    fn config_reads_nested_text_config() {
        let dir = std::env::temp_dir().join(format!("qwen35_cfg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("config.json"),
            r#"{"model_type":"qwen3_5","text_config":{"eos_token_id":7,"head_dim":256,"hidden_size":4096,
            "intermediate_size":12288,"layer_types":["linear_attention","full_attention"],
            "linear_conv_kernel_dim":4,"linear_key_head_dim":128,"linear_num_key_heads":16,
            "linear_num_value_heads":32,"linear_value_head_dim":128,"num_attention_heads":16,
            "num_hidden_layers":2,"num_key_value_heads":4,"rms_norm_eps":1e-06,"vocab_size":248320,
            "rope_parameters":{"rope_theta":10000000,"partial_rotary_factor":0.25}},"tie_word_embeddings":false}"#,
        )
        .unwrap();
        let c = Qwen35Config::from_model_dir(&dir).unwrap();
        assert_eq!(c.rotary_dim, 64);
        assert_eq!(c.full_attn, vec![false, true]);
        assert_eq!(c.lin_in_rows(), 2 * 2048 + 4096 + 4096 + 64);
        assert_eq!(c.attn_in_rows(), 2 * 4096 + 2 * 1024);
        assert!((c.rope_theta - 1e7).abs() < 1.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
