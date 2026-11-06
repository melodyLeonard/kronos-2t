//! Metal engine for Qwen3.5 hybrid models (`model::qwen35`).
//!
//! Projections reuse the NF GEMV/GEMM kernels of `metal_gpu`; this file adds the causal conv,
//! the one-pass gated delta rule, gated partial-RoPE attention over an FP16 KV cache, and the
//! per-chunk encoding. Recurrent state lives on the GPU, so decode and prefill share one path.

use crate::backend::metal_gpu::{
    alloc_shared_buffer, build_pipeline, gemv_splits, msg_sel, ns_error_string, nsstring, objc_autoreleasePoolPop,
    objc_autoreleasePoolPush, objc_msgSend, sels, upload_slice_buffer, wrap_nf4_matrix, GpuNf4Matrix, Id, MtlObj, MtlSize,
    Nf4MatParams, NormFwhtParams, Sel, GEMV_MAX_SPLITS, GEMV_MAX_TOKENS, METAL_SHADER_SOURCE, MTLCreateSystemDefaultDevice,
    MTL_CMD_STATUS_COMPLETED,
};
use crate::kernels::hadamard::rademacher_sign_masks_64;
use crate::model::qwen35::{QMixer, Qwen35Model};
use std::ffi::c_void;

/// Max tokens per command buffer; longer prompts run in chunks.
pub const MAX_CHUNK: usize = 512;
const ATTN_MAX_SPLITS: usize = 32;
/// Keys per split before decode attention fans out over more threadgroups.
const ATTN_SPLIT_KEYS: usize = 256;

#[repr(C)]
#[derive(Clone, Copy)]
struct ConvParams {
    conv_dim: u32,
    in_stride: u32,
    n_tok: u32,
    /// Floats between per-token state checkpoints; 0 = no checkpoints.
    ckpt_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DeltaParams {
    n_tok: u32,
    nk: u32,
    nv: u32,
    conv_dim: u32,
    in_stride: u32,
    ktot: u32,
    z_off: u32,
    b_off: u32,
    a_off: u32,
    vdim: u32,
    eps: f32,
    ckpt_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnPrepParams {
    nh: u32,
    nkv: u32,
    hd: u32,
    rot: u32,
    in_stride: u32,
    pos0: u32,
    n_tok: u32,
    kv_cap: u32,
    eps: f32,
    theta: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MtpPrepParams {
    h: u32,
    n_tok: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct AttnParams {
    nh: u32,
    nkv: u32,
    hd: u32,
    pos0: u32,
    n_tok: u32,
    kv_cap: u32,
    n_splits: u32,
    split_len: u32,
    scale: f32,
    _pad: u32,
}

const QWEN35_SHADER_SOURCE: &str = r#"
struct ConvParams { uint conv_dim; uint in_stride; uint n_tok; uint ckpt_stride; };
struct DeltaParams {
    uint n_tok; uint nk; uint nv; uint conv_dim; uint in_stride; uint ktot;
    uint z_off; uint b_off; uint a_off; uint vdim; float eps; uint ckpt_stride;
};
struct AttnPrepParams {
    uint nh; uint nkv; uint hd; uint rot; uint in_stride; uint pos0; uint n_tok; uint kv_cap;
    float eps; float theta;
};
struct AttnParams {
    uint nh; uint nkv; uint hd; uint pos0; uint n_tok; uint kv_cap; uint n_splits; uint split_len;
    float scale; uint _pad;
};

inline float q35_silu(float x) { return x / (1.0f + exp(-x)); }

// Causal depthwise conv (kernel 4) + SiLU over the q/k/v channels; `state` keeps the last 3 inputs.
kernel void q35_conv_silu(
    device const float* proj  [[buffer(0)]],
    device const float* w     [[buffer(1)]],
    device float*       state [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant ConvParams& p    [[buffer(4)]],
    device float*       ckpt  [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= p.conv_dim) return;
    float4 wc = float4(w[c * 4u], w[c * 4u + 1u], w[c * 4u + 2u], w[c * 4u + 3u]);
    float h0 = state[c * 3u], h1 = state[c * 3u + 1u], h2 = state[c * 3u + 2u];
    for (uint t = 0u; t < p.n_tok; t++) {
        float x = proj[ulong(t) * p.in_stride + c];
        float acc = wc.x * h0 + wc.y * h1 + wc.z * h2 + wc.w * x;
        out[ulong(t) * p.conv_dim + c] = q35_silu(acc);
        h0 = h1; h1 = h2; h2 = x;
        if (p.ckpt_stride != 0u) {
            device float* ck = ckpt + ulong(t) * p.ckpt_stride + c * 3u;
            ck[0] = h0; ck[1] = h1; ck[2] = h2;
        }
    }
    state[c * 3u] = h0; state[c * 3u + 1u] = h1; state[c * 3u + 2u] = h2;
}

// One threadgroup per value head (dk = dv = 128, 256 threads). Thread (col, part) keeps rows
// part*64 .. part*64+63 of state column `col` in registers across the whole chunk.
// Per token: o = d S^T q + delta (k.q), delta = beta (v - d S^T k), S <- d S + k delta^T,
// then out = rmsnorm(o) * w * silu(z).
kernel void q35_deltanet(
    device const float* conv   [[buffer(0)]],
    device const float* proj   [[buffer(1)]],
    device const float* dt_bias[[buffer(2)]],
    device const float* a_neg  [[buffer(3)]],
    device const float* norm_w [[buffer(4)]],
    device float*       state  [[buffer(5)]],
    device float*       out    [[buffer(6)]],
    constant DeltaParams& p    [[buffer(7)]],
    device float*       ckpt   [[buffer(8)]],
    uint h    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint DK = 128u, DV = 128u;
    threadgroup float tk[128];
    threadgroup float tq[128];
    threadgroup float red[8];
    const uint col = tid >> 1u;
    const uint part = tid & 1u;
    const uint r0 = part * 64u;
    const uint kh = h / (p.nv / p.nk);
    device float* S = state + ulong(h) * DK * DV;

    float s[64];
    for (uint i = 0u; i < 64u; i++) s[i] = S[(r0 + i) * DV + col];
    const float q_scale = rsqrt(float(DK));
    const float dtb = dt_bias[h];
    const float an = a_neg[h];
    const float nw = norm_w[col];

    for (uint t = 0u; t < p.n_tok; t++) {
        device const float* row = conv + ulong(t) * p.conv_dim;
        device const float* prow = proj + ulong(t) * p.in_stride;
        float x = (tid < 128u) ? row[p.ktot + kh * DK + tid] : row[kh * DK + (tid - 128u)];
        float sq = simd_sum(x * x);
        if (lane == 0u) red[sg] = sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ksq = red[0] + red[1] + red[2] + red[3];
        float qsq = red[4] + red[5] + red[6] + red[7];
        if (tid < 128u) tk[tid] = x * rsqrt(ksq + 1e-6f);
        else tq[tid - 128u] = x * rsqrt(qsq + 1e-6f) * q_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kq_part = (tid < 128u) ? tk[tid] * tq[tid] : 0.0f;
        kq_part = simd_sum(kq_part);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) red[sg] = kq_part;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float kq = red[0] + red[1] + red[2] + red[3];

        float beta = 1.0f / (1.0f + exp(-prow[p.b_off + h]));
        float av = prow[p.a_off + h] + dtb;
        float sp = (av > 20.0f) ? av : log(1.0f + exp(av));
        float decay = exp(an * sp);

        float ak = 0.0f, bq = 0.0f;
        for (uint i = 0u; i < 64u; i++) {
            ak += s[i] * tk[r0 + i];
            bq += s[i] * tq[r0 + i];
        }
        ak += simd_shuffle_xor(ak, 1u);
        bq += simd_shuffle_xor(bq, 1u);
        float v = row[2u * p.ktot + h * DV + col];
        float delta = beta * (v - decay * ak);
        float o = decay * bq + delta * kq;
        for (uint i = 0u; i < 64u; i++) s[i] = decay * s[i] + tk[r0 + i] * delta;

        float osq = simd_sum(part == 0u ? o * o : 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) red[sg] = osq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ss = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        if (part == 0u) {
            float z = prow[p.z_off + h * DV + col];
            out[ulong(t) * p.vdim + h * DV + col] = o * rsqrt(ss / float(DV) + p.eps) * nw * q35_silu(z);
        }
        if (p.ckpt_stride != 0u) {
            device float* ck = ckpt + ulong(t) * p.ckpt_stride + ulong(h) * DK * DV;
            for (uint i = 0u; i < 64u; i++) ck[(r0 + i) * DV + col] = s[i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0u; i < 64u; i++) S[(r0 + i) * DV + col] = s[i];
}

// One threadgroup per (token, head); threads = head_dim (256). Q heads write normed, rotated q and
// the raw gate; KV heads write normed, rotated k and v into the FP16 cache at their position.
kernel void q35_attn_prep(
    device const float* proj   [[buffer(0)]],
    device const float* q_norm [[buffer(1)]],
    device const float* k_norm [[buffer(2)]],
    device float*       q_out  [[buffer(3)]],
    device float*       g_out  [[buffer(4)]],
    device half*        k_cache[[buffer(5)]],
    device half*        v_cache[[buffer(6)]],
    constant AttnPrepParams& p [[buffer(7)]],
    uint2 tg   [[threadgroup_position_in_grid]],
    uint d     [[thread_index_in_threadgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint t = tg.x, hidx = tg.y;
    const uint hd = p.hd;
    threadgroup float buf[256];
    threadgroup float red[8];
    device const float* row = proj + ulong(t) * p.in_stride;
    const bool is_q = hidx < p.nh;
    const uint qdim = p.nh * hd, kvd = p.nkv * hd;
    float x;
    if (is_q) x = row[hidx * 2u * hd + d];
    else x = row[2u * qdim + (hidx - p.nh) * hd + d];
    float sq = simd_sum(x * x);
    if (lane == 0u) red[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0u; i < hd / 32u; i++) tot += red[i];
    x = x * rsqrt(tot / float(hd) + p.eps) * (is_q ? q_norm[d] : k_norm[d]);
    buf[d] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pos = p.pos0 + t;
    if (d < p.rot) {
        uint half_r = p.rot / 2u;
        uint i = d % half_r;
        float ang = float(pos) * pow(p.theta, -float(2u * i) / float(p.rot));
        float c = cos(ang), s = sin(ang);
        x = (d < half_r) ? (buf[d] * c - buf[d + half_r] * s) : (buf[d] * c + buf[d - half_r] * s);
    }
    if (is_q) {
        q_out[ulong(t) * qdim + hidx * hd + d] = x;
        g_out[ulong(t) * qdim + hidx * hd + d] = row[hidx * 2u * hd + hd + d];
    } else {
        uint g = hidx - p.nh;
        ulong slot = (ulong(g) * p.kv_cap + pos) * hd + d;
        k_cache[slot] = half(x);
        v_cache[slot] = half(row[2u * qdim + kvd + g * hd + d]);
    }
}

// Flash attention partials: grid (split, head, token), 8 simdgroups each walking every 8th key;
// lane holds 8 of the 256 dims. Writes (m, l, o[256]) per (token, head, split).
kernel void q35_attn(
    device const float* q      [[buffer(0)]],
    device const half*  k_cache[[buffer(1)]],
    device const half*  v_cache[[buffer(2)]],
    device float*       part   [[buffer(3)]],
    constant AttnParams& p     [[buffer(4)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint split = tg.x, h = tg.y, t = tg.z;
    const uint hd = 256u;
    const uint g = h / (p.nh / p.nkv);
    const uint last = p.pos0 + t;
    const uint k0 = split * p.split_len;
    const uint k1 = min(k0 + p.split_len, last + 1u);
    threadgroup float tm[8], tl[8];
    threadgroup float to[8][256];

    device const float* qrow = q + (ulong(t) * p.nh + h) * hd + lane * 8u;
    float qv[8];
    for (uint i = 0u; i < 8u; i++) qv[i] = qrow[i] * p.scale;
    float m = -INFINITY, l = 0.0f;
    float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint j = k0 + sg; j < k1; j += 8u) {
        device const half* kr = k_cache + (ulong(g) * p.kv_cap + j) * hd + lane * 8u;
        float sc = 0.0f;
        for (uint i = 0u; i < 8u; i++) sc += qv[i] * float(kr[i]);
        sc = simd_sum(sc);
        float mn = max(m, sc);
        float corr = exp(m - mn);
        float pj = exp(sc - mn);
        l = l * corr + pj;
        device const half* vr = v_cache + (ulong(g) * p.kv_cap + j) * hd + lane * 8u;
        for (uint i = 0u; i < 8u; i++) o[i] = o[i] * corr + pj * float(vr[i]);
        m = mn;
    }
    if (lane == 0u) { tm[sg] = m; tl[sg] = l; }
    for (uint i = 0u; i < 8u; i++) to[sg][lane * 8u + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float mx = -INFINITY;
        for (uint s2 = 0u; s2 < 8u; s2++) mx = max(mx, tm[s2]);
        float lt = 0.0f;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint s2 = 0u; s2 < 8u; s2++) {
            float w = (tm[s2] == -INFINITY) ? 0.0f : exp(tm[s2] - mx);
            lt += tl[s2] * w;
            for (uint i = 0u; i < 8u; i++) acc[i] += to[s2][lane * 8u + i] * w;
        }
        device float* dst = part + ((ulong(t) * p.nh + h) * p.n_splits + split) * (hd + 2u);
        if (lane == 0u) { dst[0] = mx; dst[1] = lt; }
        for (uint i = 0u; i < 8u; i++) dst[2u + lane * 8u + i] = acc[i];
    }
}

// Grouped-query flash attention partials: grid (split, kv head, token). Each 8-key K/V tile is
// loaded into threadgroup memory once and shared by the 4 query heads of the group; simdgroup
// `s` serves head `s & 3` on keys `(s >> 2) * 4 ..` of the tile. Same partial layout as `q35_attn`.
kernel void q35_attn_gqa(
    device const float* q      [[buffer(0)]],
    device const half*  k_cache[[buffer(1)]],
    device const half*  v_cache[[buffer(2)]],
    device float*       part   [[buffer(3)]],
    constant AttnParams& p     [[buffer(4)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint tid   [[thread_index_in_threadgroup]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]])
{
    const uint split = tg.x, g = tg.y, t = tg.z;
    const uint hd = 256u;
    const uint hl = sg & 3u, sub = sg >> 2u;
    const uint h = g * 4u + hl;
    const uint last = p.pos0 + t;
    const uint k0 = split * p.split_len;
    const uint k1 = min(k0 + p.split_len, last + 1u);
    threadgroup half kt[8][256];
    threadgroup half vt[8][256];
    threadgroup float tm[8], tl[8];
    threadgroup float to[8][256];

    device const float* qrow = q + (ulong(t) * p.nh + h) * hd + lane * 8u;
    float qv[8];
    for (uint i = 0u; i < 8u; i++) qv[i] = qrow[i] * p.scale;
    float m = -INFINITY, l = 0.0f;
    float o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    device const half* kb = k_cache + ulong(g) * p.kv_cap * hd;
    device const half* vb = v_cache + ulong(g) * p.kv_cap * hd;
    for (uint j0 = k0; j0 < k1; j0 += 8u) {
        uint nk = min(8u, k1 - j0);
        // 256 threads load 8 keys x 256 dims of K and V, 8 halves each per tensor.
        uint r = tid >> 5u, c = (tid & 31u) * 8u;
        if (r < nk) {
            *(threadgroup half4*)&kt[r][c] = *(device const half4*)(kb + ulong(j0 + r) * hd + c);
            *(threadgroup half4*)&kt[r][c + 4u] = *(device const half4*)(kb + ulong(j0 + r) * hd + c + 4u);
            *(threadgroup half4*)&vt[r][c] = *(device const half4*)(vb + ulong(j0 + r) * hd + c);
            *(threadgroup half4*)&vt[r][c + 4u] = *(device const half4*)(vb + ulong(j0 + r) * hd + c + 4u);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint jj = sub * 4u; jj < min(sub * 4u + 4u, nk); jj++) {
            float sc = 0.0f;
            for (uint i = 0u; i < 8u; i++) sc += qv[i] * float(kt[jj][lane * 8u + i]);
            sc = simd_sum(sc);
            float mn = max(m, sc);
            float corr = exp(m - mn);
            float pj = exp(sc - mn);
            l = l * corr + pj;
            for (uint i = 0u; i < 8u; i++) o[i] = o[i] * corr + pj * float(vt[jj][lane * 8u + i]);
            m = mn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) { tm[sg] = m; tl[sg] = l; }
    for (uint i = 0u; i < 8u; i++) to[sg][lane * 8u + i] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sub == 0u) {
        float m2 = tm[sg + 4u];
        float mx = max(m, m2);
        float w0 = (m == -INFINITY) ? 0.0f : exp(m - mx);
        float w1 = (m2 == -INFINITY) ? 0.0f : exp(m2 - mx);
        device float* dst = part + ((ulong(t) * p.nh + h) * p.n_splits + split) * (hd + 2u);
        if (lane == 0u) { dst[0] = mx; dst[1] = tl[sg] * w0 + tl[sg + 4u] * w1; }
        for (uint i = 0u; i < 8u; i++) {
            uint d = lane * 8u + i;
            dst[2u + d] = to[sg][d] * w0 + to[sg + 4u][d] * w1;
        }
    }
}

// Merges split partials, normalizes, applies sigmoid(gate). Grid (head, token), 256 threads.
kernel void q35_attn_reduce(
    device const float* part [[buffer(0)]],
    device const float* gate [[buffer(1)]],
    device float*       out  [[buffer(2)]],
    constant AttnParams& p   [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint d   [[thread_index_in_threadgroup]])
{
    const uint h = tg.x, t = tg.y;
    const uint hd = 256u;
    device const float* base = part + (ulong(t) * p.nh + h) * p.n_splits * (hd + 2u);
    float mx = -INFINITY;
    for (uint s = 0u; s < p.n_splits; s++) mx = max(mx, base[s * (hd + 2u)]);
    float lt = 0.0f, acc = 0.0f;
    for (uint s = 0u; s < p.n_splits; s++) {
        device const float* ps = base + s * (hd + 2u);
        float w = (ps[0] == -INFINITY) ? 0.0f : exp(ps[0] - mx);
        lt += ps[1] * w;
        acc += ps[2u + d] * w;
    }
    ulong idx = (ulong(t) * p.nh + h) * hd + d;
    float gv = gate[idx];
    out[idx] = (acc / lt) * (1.0f / (1.0f + exp(-gv)));
}

// `nf4_gemv_batch` (8-token simdgroup-matrix tile) with column splits so small-row matrices still
// fill the GPU; partials go to `q35_gemv_multi_reduce`.
kernel void q35_gemv_batch_split(
    device const uchar* pair_stream [[buffer(0)]],
    device const uchar* tail_stream [[buffer(1)]],
    device const half*  x_rot_f16   [[buffer(2)]],
    device const float* x_tok_scale [[buffer(3)]],
    device const float* bias        [[buffer(4)]],
    device const float* residual    [[buffer(5)]],
    device float*       out         [[buffer(6)]],
    constant Nf4MatParams& p        [[buffer(7)]],
    device float*       part        [[buffer(8)]],
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint sg_id    [[simdgroup_index_in_threadgroup]],
    uint tid      [[thread_index_in_simdgroup]],
    uint tid_tg   [[thread_index_in_threadgroup]])
{
    const uint pair_idx = tg_pos.x;
    const uint split = tg_pos.y;
    const uint per_split = (p.num_scale_groups + p.n_splits - 1u) / p.n_splits;
    const uint sg_lo = split * per_split;
    const uint sg_hi = min(p.num_scale_groups, sg_lo + per_split);
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
        for (uint sg = sg_lo + sg_id; sg < sg_hi; sg += 4u) {
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
        for (uint sg = sg_lo + sg_id; sg < sg_hi; sg += 4u) {
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
                float sum = (tg_buf[0][off] + tg_buf[1][off] + tg_buf[2][off] + tg_buf[3][off]) * x_tok_scale[m];
                if (p.n_splits > 1u) {
                    part[(ulong(split) * p.n_tok + m) * p.rows + row] = sum;
                    continue;
                }
                sum += b_val;
                ulong dst_idx = ulong(m) * ulong(p.rows) + ulong(row);
                if (p.has_residual != 0u) sum += residual[dst_idx];
                out[dst_idx] = sum;
            }
        }
    }
}

kernel void q35_gemv_multi_reduce(
    device const float* part     [[buffer(0)]],
    device const float* residual [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    constant Nf4MatParams& p     [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint row = gid.x, m = gid.y;
    if (row >= p.rows || m >= p.n_tok) return;
    float sum = 0.0f;
    for (uint s = 0u; s < p.n_splits; ++s) sum += part[(ulong(s) * p.n_tok + m) * p.rows + row];
    ulong idx = ulong(m) * p.rows + row;
    if (p.has_residual != 0u) sum += residual[idx];
    out[idx] = sum;
}

struct MtpPrepParams { uint h; uint n_tok; float eps; uint _pad; };

inline float q35_block_sum(float v, threadgroup float* red, uint sg, uint lane) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (uint i = 0u; i < 8u; i++) s += red[i];
    return s;
}

// MTP input row per token: `[rms(e) * w_pe | rms(rms(h) * w_h) * w_ph]`, 256 threads.
kernel void q35_mtp_prep(
    device const float* emb  [[buffer(0)]],
    device const float* hid  [[buffer(1)]],
    device const float* w_h  [[buffer(2)]],
    device const float* w_pe [[buffer(3)]],
    device const float* w_ph [[buffer(4)]],
    device float*       u    [[buffer(5)]],
    constant MtpPrepParams& p[[buffer(6)]],
    uint t    [[threadgroup_position_in_grid]],
    uint tid  [[thread_index_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    const uint h = p.h;
    device const float* e = emb + ulong(t) * h;
    device const float* x = hid + ulong(t) * h;
    float se = 0.0f, sx = 0.0f;
    for (uint i = tid; i < h; i += 256u) { se += e[i] * e[i]; sx += x[i] * x[i]; }
    float ie = rsqrt(q35_block_sum(se, red, sg, lane) / float(h) + p.eps);
    float ix = rsqrt(q35_block_sum(sx, red, sg, lane) / float(h) + p.eps);
    float sy = 0.0f;
    for (uint i = tid; i < h; i += 256u) { float y = x[i] * ix * w_h[i]; sy += y * y; }
    float iy = rsqrt(q35_block_sum(sy, red, sg, lane) / float(h) + p.eps);
    device float* dst = u + ulong(t) * 2u * h;
    for (uint i = tid; i < h; i += 256u) {
        dst[i] = e[i] * ie * w_pe[i];
        dst[h + i] = x[i] * ix * w_h[i] * iy * w_ph[i];
    }
}
"#;

struct GpuLinear {
    in_proj: GpuNf4Matrix,
    out_proj: GpuNf4Matrix,
    conv_w: MtlObj,
    dt_bias: MtlObj,
    a_neg: MtlObj,
    norm_w: MtlObj,
    /// Index among linear-attention layers (state slot).
    slot: usize,
}

struct GpuFull {
    in_proj: GpuNf4Matrix,
    out_proj: GpuNf4Matrix,
    q_norm: MtlObj,
    k_norm: MtlObj,
    /// Index among full-attention layers (KV slot).
    slot: usize,
}

enum GpuMixer {
    Linear(GpuLinear),
    Full(GpuFull),
}

struct GpuLayer {
    input_norm: MtlObj,
    post_norm: MtlObj,
    mixer: GpuMixer,
    gate_up: GpuNf4Matrix,
    down: GpuNf4Matrix,
}

/// MTP draft head on the GPU; its attention layer uses the KV slot after the main layers.
struct GpuMtp {
    pre_emb: MtlObj,
    pre_hid: MtlObj,
    fc: GpuNf4Matrix,
    layer: GpuLayer,
    norm: MtlObj,
    /// `[MAX_CHUNK, h]` token embeddings, `[.., 2h]` fc inputs, `[.., h]` residual stream.
    emb: MtlObj,
    u: MtlObj,
    x: MtlObj,
    /// Pre-norm MTP hidden of the last drafted row, input to the next chained step.
    last: MtlObj,
}

struct Pipes {
    norm_fwht: MtlObj,
    transpose_f16: MtlObj,
    gemv1: MtlObj,
    gemv_batch: MtlObj,
    gemm_mma: MtlObj,
    gemv_reduce: MtlObj,
    conv: MtlObj,
    delta: MtlObj,
    attn_prep: MtlObj,
    attn: MtlObj,
    attn_gqa: MtlObj,
    attn_reduce: MtlObj,
    mtp_prep: MtlObj,
    gemv_batch_split: MtlObj,
    gemv_multi_reduce: MtlObj,
}

struct Scratch {
    hidden: MtlObj,
    x_rot_f32: MtlObj,
    x_rot_f16: MtlObj,
    x_tok_scale: MtlObj,
    gemv_part: MtlObj,
    proj: MtlObj,
    conv: MtlObj,
    core: MtlObj,
    q: MtlObj,
    gate: MtlObj,
    attn_part: MtlObj,
    gate_up: MtlObj,
    logits: MtlObj,
}

/// Where per-stage GPU time goes, from `KRONOS_DECODE_PROF` (decode steps only).
struct Prof {
    sel_start: usize,
    sel_end: usize,
    acc: Vec<(&'static str, f64)>,
    steps: usize,
}

pub struct MetalQwen35 {
    _device: MtlObj,
    queue: MtlObj,
    _library: MtlObj,
    pipes: Pipes,
    dummy: MtlObj,
    rademacher: MtlObj,
    layers: Vec<GpuLayer>,
    final_norm: MtlObj,
    lm_head: GpuNf4Matrix,
    conv_state: MtlObj,
    delta_state: MtlObj,
    k_cache: MtlObj,
    v_cache: MtlObj,
    sc: Scratch,
    kv_cap: usize,
    pos: usize,
    /// Token rows the `logits` scratch holds.
    logits_cap: usize,
    /// Per-token `(conv, delta)` state checkpoints for speculative verification.
    ckpt: Option<(MtlObj, MtlObj)>,
    /// `(pos0, n)` of the last `verify` call, until `accept` settles it.
    pending: Option<(usize, usize)>,
    mtp: Option<GpuMtp>,
    /// Row-prefix view of `lm_head` scored by MTP drafts (low BPE ids are the frequent tokens).
    draft_head: Option<GpuNf4Matrix>,
    draft_vocab: usize,
    // Shapes.
    h: usize,
    inter: usize,
    vocab: usize,
    eps: f32,
    nh: usize,
    nkv: usize,
    hd: usize,
    rot: usize,
    theta: f32,
    nk: usize,
    nv: usize,
    ktot: usize,
    vdim: usize,
    conv_dim: usize,
    lin_rows: usize,
    attn_rows: usize,
    n_lin: usize,
    prof: Option<std::cell::RefCell<Prof>>,
}

unsafe fn set_buf(enc: Id, buf: Id, offset: usize, idx: usize) {
    let st = sels();
    unsafe { msg_sel!(enc, st.set_buf, buf => Id, offset => usize, idx => usize; ()) };
}

unsafe fn set_bytes<T>(enc: Id, v: &T, idx: usize) {
    let st = sels();
    unsafe {
        msg_sel!(
            enc,
            st.set_bytes,
            (v as *const T).cast::<c_void>() => *const c_void,
            std::mem::size_of::<T>() => usize,
            idx => usize;
            ()
        )
    };
}

unsafe fn dispatch(enc: Id, pso: &MtlObj, grid: MtlSize, threads: MtlSize) {
    let st = sels();
    unsafe {
        msg_sel!(enc, st.set_cps, pso.id() => Id; ());
        msg_sel!(enc, st.dispatch_tg, grid => MtlSize, threads => MtlSize; ());
    }
}

unsafe fn set_cps(enc: Id, pso: &MtlObj) {
    let st = sels();
    unsafe { msg_sel!(enc, st.set_cps, pso.id() => Id; ()) };
}

unsafe fn dispatch_only(enc: Id, grid: MtlSize, threads: MtlSize) {
    let st = sels();
    unsafe { msg_sel!(enc, st.dispatch_tg, grid => MtlSize, threads => MtlSize; ()) };
}

impl MetalQwen35 {
    /// Uploads `model` (zero-copy for quantized matrices) with room for `max_ctx` positions.
    pub fn new(model: &Qwen35Model, max_ctx: usize) -> Result<Self, String> {
        let cfg = &model.cfg;
        if cfg.head_dim != 256 || cfg.lin_k_dim != 128 || cfg.lin_v_dim != 128 || cfg.conv_kernel != 4 {
            return Err("Metal Qwen3.5 kernels expect head_dim 256, linear head dims 128, conv kernel 4".into());
        }
        // SAFETY: Objective-C calls wrapped in an explicit autorelease pool.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<Self, String> {
                let st = sels();
                let device = MtlObj::from_retained(MTLCreateSystemDefaultDevice(), "MTLCreateSystemDefaultDevice")?;
                let queue = MtlObj::from_retained(msg_sel!(device.id(), st.new_cmd_queue; Id), "newCommandQueue")?;
                let exp = std::env::var("KRONOS_GEMV_EXP").unwrap_or_else(|_| "0".to_string());
                let src = format!("#define GEMV_EXP {exp}\n{METAL_SHADER_SOURCE}\n{QWEN35_SHADER_SOURCE}");
                let src_ns = nsstring(&src)?;
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
                let bp = |name: &str| build_pipeline(device.id(), library.id(), name);
                let pipes = Pipes {
                    norm_fwht: bp("rmsnorm_fwht64")?,
                    transpose_f16: bp("transpose_x_f16")?,
                    gemv1: bp("nf4_gemv_decode1")?,
                    gemv_batch: bp("nf4_gemv_batch")?,
                    gemm_mma: bp("nf4_gemm_mma")?,
                    gemv_reduce: bp("gemv_splitk_reduce")?,
                    conv: bp("q35_conv_silu")?,
                    delta: bp("q35_deltanet")?,
                    attn_prep: bp("q35_attn_prep")?,
                    attn: bp("q35_attn")?,
                    attn_gqa: bp("q35_attn_gqa")?,
                    attn_reduce: bp("q35_attn_reduce")?,
                    mtp_prep: bp("q35_mtp_prep")?,
                    gemv_batch_split: bp("q35_gemv_batch_split")?,
                    gemv_multi_reduce: bp("q35_gemv_multi_reduce")?,
                };
                let dummy = alloc_shared_buffer(device.id(), 64, "dummy")?;
                let mtp_cols = if model.mtp.is_some() { 2 * cfg.hidden_size } else { 0 };
                let max_rot = cfg.hidden_size.max(cfg.intermediate_size).max(cfg.q_dim()).max(cfg.lin_v_dim()).max(mtp_cols);
                let mut masks = Vec::with_capacity(max_rot);
                for b in 0..max_rot / 64 {
                    masks.extend_from_slice(&rademacher_sign_masks_64(b));
                }
                let rademacher = upload_slice_buffer(device.id(), &masks, "rademacher")?;

                let wrap = |m, label: &str| wrap_nf4_matrix(device.id(), m, dummy.id(), label);
                let up = |v: &[f32], label: &str| upload_slice_buffer(device.id(), v, label);
                let (mut n_lin, mut n_full) = (0usize, 0usize);
                let mut layers = Vec::with_capacity(model.layers.len());
                let mut max_rows = 0usize;
                for (i, l) in model.layers.iter().enumerate() {
                    let mixer = match &l.mixer {
                        QMixer::Linear { in_proj, params, out_proj } => {
                            n_lin += 1;
                            GpuMixer::Linear(GpuLinear {
                                in_proj: wrap(in_proj, &format!("l{i}.in"))?,
                                out_proj: wrap(out_proj, &format!("l{i}.out"))?,
                                conv_w: up(&params.conv_w, "conv_w")?,
                                dt_bias: up(&params.dt_bias, "dt_bias")?,
                                a_neg: up(&params.a_neg, "a_neg")?,
                                norm_w: up(&params.norm_w, "norm_w")?,
                                slot: n_lin - 1,
                            })
                        }
                        QMixer::Full { in_proj, params, out_proj } => {
                            n_full += 1;
                            GpuMixer::Full(GpuFull {
                                in_proj: wrap(in_proj, &format!("l{i}.in"))?,
                                out_proj: wrap(out_proj, &format!("l{i}.out"))?,
                                q_norm: up(&params.q_norm, "q_norm")?,
                                k_norm: up(&params.k_norm, "k_norm")?,
                                slot: n_full - 1,
                            })
                        }
                    };
                    max_rows = max_rows.max(l.mixer.in_proj().rows).max(l.gate_up.rows);
                    layers.push(GpuLayer {
                        input_norm: up(&l.input_norm, "input_norm")?,
                        post_norm: up(&l.post_norm, "post_norm")?,
                        mixer,
                        gate_up: wrap(&l.gate_up, &format!("l{i}.gate_up"))?,
                        down: wrap(&l.down, &format!("l{i}.down"))?,
                    });
                }
                let f32b = 4usize;
                let n = MAX_CHUNK;
                let n_f16 = n.div_ceil(128) * 128;
                let sc = Scratch {
                    hidden: alloc_shared_buffer(device.id(), n * cfg.hidden_size * f32b, "hidden")?,
                    x_rot_f32: alloc_shared_buffer(device.id(), n * max_rot * f32b, "x_rot_f32")?,
                    x_rot_f16: alloc_shared_buffer(device.id(), n_f16 * max_rot * 2, "x_rot_f16")?,
                    x_tok_scale: alloc_shared_buffer(device.id(), n * f32b, "x_tok_scale")?,
                    gemv_part: alloc_shared_buffer(device.id(), max_rows.max(cfg.hidden_size) * GEMV_MAX_SPLITS * GEMV_MAX_TOKENS * f32b, "gemv_part")?,
                    proj: alloc_shared_buffer(device.id(), n * cfg.lin_in_rows().max(cfg.attn_in_rows()) * f32b, "proj")?,
                    conv: alloc_shared_buffer(device.id(), n * cfg.conv_dim() * f32b, "conv")?,
                    core: alloc_shared_buffer(device.id(), n * cfg.lin_v_dim().max(cfg.q_dim()) * f32b, "core")?,
                    q: alloc_shared_buffer(device.id(), n * cfg.q_dim() * f32b, "q")?,
                    gate: alloc_shared_buffer(device.id(), n * cfg.q_dim() * f32b, "gate")?,
                    attn_part: alloc_shared_buffer(
                        device.id(),
                        n.max(GEMV_MAX_TOKENS * ATTN_MAX_SPLITS) * cfg.num_heads * (cfg.head_dim + 2) * f32b,
                        "attn_part",
                    )?,
                    gate_up: alloc_shared_buffer(device.id(), n * 2 * cfg.intermediate_size * f32b, "gate_up")?,
                    logits: alloc_shared_buffer(device.id(), cfg.vocab_size * f32b, "logits")?,
                };
                let conv_state = alloc_shared_buffer(device.id(), n_lin * cfg.conv_dim() * 3 * f32b, "conv_state")?;
                let delta_state = alloc_shared_buffer(
                    device.id(),
                    n_lin * cfg.lin_v_heads * cfg.lin_k_dim * cfg.lin_v_dim * f32b,
                    "delta_state",
                )?;
                let mtp = match &model.mtp {
                    Some(m) => {
                        let QMixer::Full { in_proj, params, out_proj } = &m.layer.mixer else {
                            return Err("MTP layer must be full attention".into());
                        };
                        let rows = MAX_CHUNK;
                        Some(GpuMtp {
                            pre_emb: up(&m.pre_emb, "mtp.pre_emb")?,
                            pre_hid: up(&m.pre_hid, "mtp.pre_hid")?,
                            fc: wrap(&m.fc, "mtp.fc")?,
                            layer: GpuLayer {
                                input_norm: up(&m.layer.input_norm, "mtp.input_norm")?,
                                post_norm: up(&m.layer.post_norm, "mtp.post_norm")?,
                                mixer: GpuMixer::Full(GpuFull {
                                    in_proj: wrap(in_proj, "mtp.in")?,
                                    out_proj: wrap(out_proj, "mtp.out")?,
                                    q_norm: up(&params.q_norm, "mtp.q_norm")?,
                                    k_norm: up(&params.k_norm, "mtp.k_norm")?,
                                    slot: n_full,
                                }),
                                gate_up: wrap(&m.layer.gate_up, "mtp.gate_up")?,
                                down: wrap(&m.layer.down, "mtp.down")?,
                            },
                            norm: up(&m.norm, "mtp.norm")?,
                            emb: alloc_shared_buffer(device.id(), rows * cfg.hidden_size * f32b, "mtp.emb")?,
                            u: alloc_shared_buffer(device.id(), rows * 2 * cfg.hidden_size * f32b, "mtp.u")?,
                            x: alloc_shared_buffer(device.id(), rows * cfg.hidden_size * f32b, "mtp.x")?,
                            last: alloc_shared_buffer(device.id(), cfg.hidden_size * f32b, "mtp.last")?,
                        })
                    }
                    None => None,
                };
                let kv_slots = n_full + usize::from(mtp.is_some());
                let kv_bytes = kv_slots.max(1) * cfg.num_kv_heads * max_ctx * cfg.head_dim * 2;
                let k_cache = alloc_shared_buffer(device.id(), kv_bytes, "k_cache")?;
                let v_cache = alloc_shared_buffer(device.id(), kv_bytes, "v_cache")?;
                let prof = std::env::var_os("KRONOS_DECODE_PROF").map(|_| std::cell::RefCell::new(Prof {
                    sel_start: crate::backend::metal_gpu::sel_registerName(c"GPUStartTime".as_ptr()) as usize,
                    sel_end: crate::backend::metal_gpu::sel_registerName(c"GPUEndTime".as_ptr()) as usize,
                    acc: Vec::new(),
                    steps: 0,
                }));
                let mut eng = Self {
                    final_norm: up(&model.final_norm, "final_norm")?,
                    lm_head: wrap(&model.lm_head, "lm_head")?,
                    _device: device,
                    queue,
                    _library: library,
                    pipes,
                    dummy,
                    rademacher,
                    layers,
                    conv_state,
                    delta_state,
                    k_cache,
                    v_cache,
                    sc,
                    kv_cap: max_ctx,
                    pos: 0,
                    logits_cap: 1,
                    ckpt: None,
                    pending: None,
                    draft_head: None,
                    draft_vocab: 0,
                    mtp,
                    h: cfg.hidden_size,
                    inter: cfg.intermediate_size,
                    vocab: cfg.vocab_size,
                    eps: cfg.rms_norm_eps,
                    nh: cfg.num_heads,
                    nkv: cfg.num_kv_heads,
                    hd: cfg.head_dim,
                    rot: cfg.rotary_dim,
                    theta: cfg.rope_theta,
                    nk: cfg.lin_k_heads,
                    nv: cfg.lin_v_heads,
                    ktot: cfg.lin_k_total(),
                    vdim: cfg.lin_v_dim(),
                    conv_dim: cfg.conv_dim(),
                    lin_rows: cfg.lin_in_rows(),
                    attn_rows: cfg.attn_in_rows(),
                    n_lin,
                    prof,
                };
                eng.reset();
                Ok(eng)
            })();
            objc_autoreleasePoolPop(pool);
            res
        }
    }

    /// Clears the recurrent state and the KV position.
    pub fn reset(&mut self) {
        for b in [&self.conv_state, &self.delta_state] {
            // SAFETY: shared buffer, GPU idle between calls.
            unsafe { std::ptr::write_bytes(b.contents_ptr(), 0, b.byte_len()) };
        }
        self.pos = 0;
    }

    #[inline]
    pub fn pos(&self) -> usize {
        self.pos
    }

    #[inline]
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    #[inline]
    pub fn n_linear_layers(&self) -> usize {
        self.n_lin
    }

    /// Feeds `tokens` at the current position. `on_logits(i, logits)` is called for every token
    /// index in `want` (relative to `tokens`), in order.
    pub fn forward(
        &mut self,
        model: &Qwen35Model,
        tokens: &[usize],
        want: &[usize],
        on_logits: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<(), String> {
        if self.pos + tokens.len() > self.kv_cap {
            return Err(format!("context {} exceeds capacity {}", self.pos + tokens.len(), self.kv_cap));
        }
        let mut start = 0;
        while start < tokens.len() {
            let end = (start + MAX_CHUNK).min(tokens.len());
            let chunk_want: Vec<usize> = want.iter().filter(|&&w| w >= start && w < end).map(|&w| w - start).collect();
            self.run_chunk(model, &tokens[start..end], &chunk_want, &mut |i, l| on_logits(start + i, l), false)?;
            start = end;
        }
        Ok(())
    }

    /// Speculative verification: feeds up to `GEMV_MAX_TOKENS` tokens in one batched pass, reports
    /// logits for each, and keeps per-token recurrent checkpoints until [`Self::accept`].
    pub fn verify(
        &mut self,
        model: &Qwen35Model,
        tokens: &[usize],
        on_logits: &mut dyn FnMut(usize, &[f32]),
    ) -> Result<(), String> {
        let n = tokens.len();
        if n == 0 || n > GEMV_MAX_TOKENS {
            return Err(format!("verify takes 1..={GEMV_MAX_TOKENS} tokens (got {n})"));
        }
        if self.pos + n > self.kv_cap {
            return Err(format!("context {} exceeds capacity {}", self.pos + n, self.kv_cap));
        }
        let pos0 = self.pos;
        let want: Vec<usize> = (0..n).collect();
        self.run_chunk(model, tokens, &want, on_logits, true)?;
        self.pending = Some((pos0, n));
        Ok(())
    }

    /// Keeps the first `keep` tokens of the last [`Self::verify`] and rolls the rest back.
    pub fn accept(&mut self, keep: usize) {
        let Some((pos0, n)) = self.pending.take() else { return };
        let keep = keep.clamp(1, n);
        if keep == n {
            return;
        }
        let (conv_ck, delta_ck) = self.ckpt.as_ref().expect("verify allocates checkpoints");
        for (live, ck) in [(&self.conv_state, conv_ck), (&self.delta_state, delta_ck)] {
            let slot = live.byte_len();
            // SAFETY: checkpoint slot `keep - 1` holds one full copy of the live state; GPU idle.
            unsafe { std::ptr::copy_nonoverlapping(ck.contents_ptr().add((keep - 1) * slot), live.contents_ptr(), slot) };
        }
        self.pos = pos0 + keep;
    }

    /// Scores MTP drafts over the first `n` vocabulary rows only (`0` = full vocabulary).
    /// Verification always uses the full head, so outputs do not change, only draft cost.
    pub fn set_draft_vocab(&mut self, n: usize) {
        let n = (n / 32) * 32;
        if n == 0 || n >= self.vocab {
            self.draft_head = None;
            return;
        }
        let m = &self.lm_head;
        let st = sels();
        // SAFETY: `retain` balances the release in `MtlObj::drop`; the view shares `lm_head`'s buffers.
        let (pair, tail) = unsafe { (msg_sel!(m.pair_buf.id(), st.retain; Id), msg_sel!(self.dummy.id(), st.retain; Id)) };
        self.draft_head = Some(GpuNf4Matrix {
            pair_buf: MtlObj(pair),
            tail_buf: MtlObj(tail),
            cb_packed: m.cb_packed,
            rows: n as u32,
            cols: m.cols,
            num_stripes: (n / 16) as u32,
            num_pairs: (n / 32) as u32,
            num_chunks: m.num_chunks,
            scale_group_chunks: m.scale_group_chunks,
            num_scale_groups: m.num_scale_groups,
            pair_total_bytes: m.pair_total_bytes,
            bits: m.bits,
            chunk_bytes: m.chunk_bytes,
        });
        self.draft_vocab = n;
    }

    #[inline]
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// MTP pass pairing `next_toks[i]` with main hidden row `hid_row0 + i` of the last
    /// forward/verify chunk, at MTP positions `pos0..`. Fills the MTP KV cache and, when
    /// `want_draft`, returns the greedy draft that follows the last row.
    pub fn mtp_fill(
        &mut self,
        model: &Qwen35Model,
        next_toks: &[usize],
        hid_row0: usize,
        pos0: usize,
        want_draft: bool,
    ) -> Result<Option<usize>, String> {
        self.mtp_pass(model, next_toks, Some(hid_row0), pos0, want_draft)
    }

    /// One chained MTP step from the previous MTP hidden: drafts the token after `tok`.
    pub fn mtp_chain(&mut self, model: &Qwen35Model, tok: usize, pos: usize) -> Result<usize, String> {
        Ok(self.mtp_pass(model, &[tok], None, pos, true)?.expect("draft"))
    }

    fn mtp_pass(
        &mut self,
        model: &Qwen35Model,
        toks: &[usize],
        hid_row0: Option<usize>,
        pos0: usize,
        want_draft: bool,
    ) -> Result<Option<usize>, String> {
        let n = toks.len();
        let h = self.h;
        let Some(m) = &self.mtp else { return Err("model has no MTP head".into()) };
        if n == 0 || n > MAX_CHUNK || pos0 + n > self.kv_cap {
            return Err(format!("MTP pass of {n} rows at {pos0} is out of range"));
        }
        {
            // SAFETY: `m.emb` holds `MAX_CHUNK * h` floats and the GPU is idle.
            let emb = unsafe { std::slice::from_raw_parts_mut(m.emb.contents_ptr().cast::<f32>(), n * h) };
            for (i, &t) in toks.iter().enumerate() {
                model.embed_into(t, &mut emb[i * h..(i + 1) * h]);
            }
        }
        let mm = n >= 2;
        let GpuMixer::Full(full) = &m.layer.mixer else { unreachable!() };
        // SAFETY: every bound buffer is owned by `self` and outlives `waitUntilCompleted`.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<(), String> {
                let st = sels();
                let cmd: Id = msg_sel!(self.queue.id(), st.cmd_buffer; Id);
                let enc: Id = msg_sel!(cmd, st.compute_enc; Id);
                let (hid, hid_off, w_h) = match hid_row0 {
                    Some(r) => (self.sc.hidden.id(), r * h * 4, self.final_norm.id()),
                    None => (m.last.id(), 0, m.norm.id()),
                };
                let pp = MtpPrepParams { h: h as u32, n_tok: n as u32, eps: self.eps, _pad: 0 };
                set_buf(enc, m.emb.id(), 0, 0);
                set_buf(enc, hid, hid_off, 1);
                set_buf(enc, w_h, 0, 2);
                set_buf(enc, m.pre_emb.id(), 0, 3);
                set_buf(enc, m.pre_hid.id(), 0, 4);
                set_buf(enc, m.u.id(), 0, 5);
                set_bytes(enc, &pp, 6);
                dispatch(enc, &self.pipes.mtp_prep, MtlSize::new(n, 1, 1), MtlSize::new(256, 1, 1));
                self.norm_fwht(enc, m.u.id(), self.dummy.id(), 2 * h, n, 1, mm);
                self.proj(enc, &m.fc, None, m.x.id(), n);

                let layer = &m.layer;
                self.norm_fwht(enc, m.x.id(), layer.input_norm.id(), h, n, 0, mm);
                self.proj(enc, &full.in_proj, None, self.sc.proj.id(), n);
                self.encode_attention(enc, full, n, pos0);
                self.norm_fwht(enc, self.sc.core.id(), self.dummy.id(), self.nh * self.hd, n, 1, mm);
                self.proj(enc, &full.out_proj, Some(m.x.id()), m.x.id(), n);
                self.norm_fwht(enc, m.x.id(), layer.post_norm.id(), h, n, 0, mm);
                self.proj(enc, &layer.gate_up, None, self.sc.gate_up.id(), n);
                self.norm_fwht(enc, self.sc.gate_up.id(), self.dummy.id(), self.inter, n, 2, mm);
                self.proj(enc, &layer.down, Some(m.x.id()), m.x.id(), n);
                if want_draft {
                    self.norm_fwht_at(enc, m.x.id(), (n - 1) * h * 4, m.norm.id(), h, 1, 0, false);
                    self.proj(enc, self.draft_head.as_ref().unwrap_or(&self.lm_head), None, self.sc.logits.id(), 1);
                }
                msg_sel!(enc, st.end_enc; ());
                msg_sel!(cmd, st.commit; ());
                msg_sel!(cmd, st.wait_done; ());
                let status: usize = msg_sel!(cmd, st.cmd_status; usize);
                if status != MTL_CMD_STATUS_COMPLETED {
                    let e: Id = msg_sel!(cmd, st.cmd_error; Id);
                    return Err(format!("Metal MTP pass failed: {}", ns_error_string(e)));
                }
                Ok(())
            })();
            objc_autoreleasePoolPop(pool);
            res?;
            std::ptr::copy_nonoverlapping(m.x.contents_ptr().add((n - 1) * h * 4), m.last.contents_ptr(), h * 4);
        }
        if !want_draft {
            return Ok(None);
        }
        // SAFETY: `logits` holds at least one row of `vocab` floats; GPU idle.
        let rows = if self.draft_head.is_some() { self.draft_vocab } else { self.vocab };
        let logits = unsafe { std::slice::from_raw_parts(self.sc.logits.contents_ptr().cast::<f32>(), rows) };
        Ok(Some(crate::model::real_llm::argmax(logits)))
    }

    fn run_chunk(
        &mut self,
        model: &Qwen35Model,
        tokens: &[usize],
        want: &[usize],
        on_logits: &mut dyn FnMut(usize, &[f32]),
        ckpt: bool,
    ) -> Result<(), String> {
        if ckpt && self.ckpt.is_none() {
            let conv = alloc_shared_buffer(self._device.id(), GEMV_MAX_TOKENS * self.n_lin * self.conv_dim * 3 * 4, "conv_ckpt")?;
            let delta = alloc_shared_buffer(self._device.id(), GEMV_MAX_TOKENS * self.n_lin * self.nv * 128 * 128 * 4, "delta_ckpt")?;
            self.ckpt = Some((conv, delta));
        }
        let ckpt_bufs = if ckpt { self.ckpt.take() } else { None };
        let res = self.run_chunk_inner(model, tokens, want, on_logits, ckpt_bufs.as_ref());
        if ckpt {
            self.ckpt = ckpt_bufs;
        }
        res
    }

    fn run_chunk_inner(
        &mut self,
        model: &Qwen35Model,
        tokens: &[usize],
        want: &[usize],
        on_logits: &mut dyn FnMut(usize, &[f32]),
        ckpt: Option<&(MtlObj, MtlObj)>,
    ) -> Result<(), String> {
        let n = tokens.len();
        let h = self.h;
        {
            // SAFETY: `hidden` holds `MAX_CHUNK * h` floats and the GPU is idle.
            let hid = unsafe { std::slice::from_raw_parts_mut(self.sc.hidden.contents_ptr().cast::<f32>(), n * h) };
            for (t, &tok) in tokens.iter().enumerate() {
                model.embed_into(tok, &mut hid[t * h..(t + 1) * h]);
            }
        }
        if let (Some(&lo), Some(&hi)) = (want.iter().min(), want.iter().max()) {
            let m = hi - lo + 1;
            if m > self.logits_cap {
                self.sc.logits = alloc_shared_buffer(self._device.id(), m * self.vocab * 4, "logits")?;
                self.logits_cap = m;
            }
        }
        let pos0 = self.pos;
        // SAFETY: every buffer bound below is owned by `self` and outlives `waitUntilCompleted`.
        unsafe {
            let pool = objc_autoreleasePoolPush();
            let res = (|| -> Result<(), String> {
                let st = sels();
                let mut cmd: Id = msg_sel!(self.queue.id(), st.cmd_buffer; Id);
                let mut enc: Id = msg_sel!(cmd, st.compute_enc; Id);
                if cmd.is_null() || enc.is_null() {
                    return Err("Metal command buffer allocation failed".into());
                }
                let prof_on = self.prof.is_some() && n == 1;
                let mm = n >= 2;
                for layer in &self.layers {
                    self.norm_fwht(enc, self.sc.hidden.id(), layer.input_norm.id(), h, n, 0, mm);
                    match &layer.mixer {
                        GpuMixer::Linear(lin) => {
                            self.proj(enc, &lin.in_proj, None, self.sc.proj.id(), n);
                            if prof_on { self.cut(&mut cmd, &mut enc, "lin_in_proj"); }
                            self.encode_conv(enc, lin, n, ckpt.map(|c| &c.0));
                            self.encode_delta(enc, lin, n, ckpt.map(|c| &c.1));
                            if prof_on { self.cut(&mut cmd, &mut enc, "deltanet"); }
                            self.norm_fwht(enc, self.sc.core.id(), self.dummy.id(), self.vdim, n, 1, mm);
                            self.proj(enc, &lin.out_proj, Some(self.sc.hidden.id()), self.sc.hidden.id(), n);
                            if prof_on { self.cut(&mut cmd, &mut enc, "lin_out_proj"); }
                        }
                        GpuMixer::Full(full) => {
                            self.proj(enc, &full.in_proj, None, self.sc.proj.id(), n);
                            if prof_on { self.cut(&mut cmd, &mut enc, "attn_in_proj"); }
                            self.encode_attention(enc, full, n, pos0);
                            if prof_on { self.cut(&mut cmd, &mut enc, "attention"); }
                            self.norm_fwht(enc, self.sc.core.id(), self.dummy.id(), self.nh * self.hd, n, 1, mm);
                            self.proj(enc, &full.out_proj, Some(self.sc.hidden.id()), self.sc.hidden.id(), n);
                            if prof_on { self.cut(&mut cmd, &mut enc, "attn_out_proj"); }
                        }
                    }
                    self.norm_fwht(enc, self.sc.hidden.id(), layer.post_norm.id(), h, n, 0, mm);
                    self.proj(enc, &layer.gate_up, None, self.sc.gate_up.id(), n);
                    self.norm_fwht(enc, self.sc.gate_up.id(), self.dummy.id(), self.inter, n, 2, mm);
                    self.proj(enc, &layer.down, Some(self.sc.hidden.id()), self.sc.hidden.id(), n);
                    if prof_on { self.cut(&mut cmd, &mut enc, "mlp"); }
                }
                // Logits for the contiguous row range covering `want`, batched like any projection.
                if let (Some(&lo), Some(&hi)) = (want.iter().min(), want.iter().max()) {
                    let m = hi - lo + 1;
                    self.norm_fwht_at(enc, self.sc.hidden.id(), lo * h * 4, self.final_norm.id(), h, m, 0, m >= 2);
                    self.proj(enc, &self.lm_head, None, self.sc.logits.id(), m);
                    if prof_on { self.cut(&mut cmd, &mut enc, "lm_head"); }
                }
                msg_sel!(enc, st.end_enc; ());
                msg_sel!(cmd, st.commit; ());
                msg_sel!(cmd, st.wait_done; ());
                let status: usize = msg_sel!(cmd, st.cmd_status; usize);
                if status != MTL_CMD_STATUS_COMPLETED {
                    let e: Id = msg_sel!(cmd, st.cmd_error; Id);
                    return Err(format!("Metal command buffer failed: {}", ns_error_string(e)));
                }
                if let Some(&lo) = want.iter().min() {
                    let all = std::slice::from_raw_parts(self.sc.logits.contents_ptr().cast::<f32>(), self.logits_cap * self.vocab);
                    for &i in want {
                        on_logits(i, &all[(i - lo) * self.vocab..(i - lo + 1) * self.vocab]);
                    }
                }
                if prof_on {
                    if let Some(cell) = &self.prof {
                        let mut p = cell.borrow_mut();
                        p.steps += 1;
                        if p.steps.is_multiple_of(32) {
                            let tot: f64 = p.acc.iter().map(|x| x.1).sum();
                            let mut line = format!("decode profile over {} steps:", p.steps);
                            for (k, v) in &p.acc {
                                line.push_str(&format!(" {k} {:.2}ms ({:.0}%)", v * 1e3 / p.steps as f64, 100.0 * v / tot));
                            }
                            eprintln!("{line}");
                        }
                    }
                }
                Ok(())
            })();
            objc_autoreleasePoolPop(pool);
            res?;
        }
        self.pos += n;
        Ok(())
    }

    unsafe fn cut(&self, cmd: &mut Id, enc: &mut Id, label: &'static str) {
        let st = sels();
        let Some(cell) = &self.prof else { return };
        let mut p = cell.borrow_mut();
        unsafe {
            msg_sel!(*enc, st.end_enc; ());
            msg_sel!(*cmd, st.commit; ());
            msg_sel!(*cmd, st.wait_done; ());
            let t0: f64 = msg_sel!(*cmd, p.sel_start as Sel; f64);
            let t1: f64 = msg_sel!(*cmd, p.sel_end as Sel; f64);
            match p.acc.iter_mut().find(|e| e.0 == label) {
                Some(e) => e.1 += (t1 - t0).max(0.0),
                None => p.acc.push((label, (t1 - t0).max(0.0))),
            }
            *cmd = msg_sel!(self.queue.id(), st.cmd_buffer; Id);
            *enc = msg_sel!(*cmd, st.compute_enc; Id);
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn norm_fwht(&self, enc: Id, x: Id, w: Id, dim: usize, n: usize, mode: u32, f16: bool) {
        unsafe { self.norm_fwht_at(enc, x, 0, w, dim, n, mode, f16) };
    }

    /// `rmsnorm_fwht64` (mode 0 norm, 1 rotate only, 2 SwiGLU) into `x_rot_*`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn norm_fwht_at(&self, enc: Id, x: Id, x_off: usize, w: Id, dim: usize, n: usize, mode: u32, f16: bool) {
        let p = NormFwhtParams { dim: dim as u32, n_tok: n as u32, eps: self.eps, mode, interleaved: 0, write_f16: u32::from(f16) };
        unsafe {
            set_cps(enc, &self.pipes.norm_fwht);
            set_buf(enc, x, x_off, 0);
            set_buf(enc, w, 0, 1);
            set_buf(enc, self.rademacher.id(), 0, 2);
            set_buf(enc, self.sc.x_rot_f32.id(), 0, 3);
            set_buf(enc, self.sc.x_rot_f16.id(), 0, 4);
            set_buf(enc, self.sc.x_tok_scale.id(), 0, 5);
            set_bytes(enc, &p, 6);
            dispatch_only(enc, MtlSize::new(n, 1, 1), MtlSize::new(256, 1, 1));
            if f16 {
                let n_stride = n.div_ceil(128) * 128;
                set_cps(enc, &self.pipes.transpose_f16);
                set_buf(enc, self.sc.x_rot_f32.id(), 0, 0);
                set_buf(enc, self.sc.x_tok_scale.id(), 0, 1);
                set_buf(enc, self.sc.x_rot_f16.id(), 0, 2);
                set_bytes(enc, &p, 3);
                dispatch_only(enc, MtlSize::new(n_stride / 32, dim / 32, 1), MtlSize::new(256, 1, 1));
            }
        }
    }

    /// NF projection of the rotated activations into `out` (optionally `+= residual`).
    unsafe fn proj(&self, enc: Id, mat: &GpuNf4Matrix, residual: Option<Id>, out: Id, n: usize) {
        let pair_blocks = (mat.num_stripes as usize).div_ceil(2);
        let splits = if n <= GEMV_MAX_TOKENS { gemv_splits(pair_blocks, mat.num_scale_groups as usize) } else { 1 };
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
            n_tok: n as u32,
            has_bias: 0,
            has_residual: u32::from(residual.is_some()),
            n_offset: 0,
            bits: mat.bits,
            chunk_bytes: mat.chunk_bytes,
            n_splits: splits as u32,
            _pad: 0,
        };
        let res = residual.unwrap_or_else(|| self.dummy.id());
        let dummy = self.dummy.id();
        unsafe {
            if n == 1 {
                set_cps(enc, &self.pipes.gemv1);
                set_buf(enc, mat.pair_buf.id(), 0, 0);
                set_buf(enc, mat.tail_buf.id(), 0, 1);
                set_buf(enc, self.sc.x_rot_f32.id(), 0, 2);
                set_buf(enc, dummy, 0, 3);
                set_buf(enc, res, 0, 4);
                set_buf(enc, out, 0, 5);
                set_bytes(enc, &params, 6);
                set_buf(enc, self.sc.gemv_part.id(), 0, 7);
                dispatch_only(enc, MtlSize::new(pair_blocks, splits, 1), MtlSize::new(128, 1, 1));
                if splits > 1 {
                    set_cps(enc, &self.pipes.gemv_reduce);
                    set_buf(enc, self.sc.gemv_part.id(), 0, 0);
                    set_buf(enc, dummy, 0, 1);
                    set_buf(enc, res, 0, 2);
                    set_buf(enc, out, 0, 3);
                    set_bytes(enc, &params, 4);
                    dispatch_only(enc, MtlSize::new((mat.rows as usize).div_ceil(256), 1, 1), MtlSize::new(256, 1, 1));
                }
            } else if n <= GEMV_MAX_TOKENS {
                set_cps(enc, &self.pipes.gemv_batch_split);
                set_buf(enc, mat.pair_buf.id(), 0, 0);
                set_buf(enc, mat.tail_buf.id(), 0, 1);
                set_buf(enc, self.sc.x_rot_f16.id(), 0, 2);
                set_buf(enc, self.sc.x_tok_scale.id(), 0, 3);
                set_buf(enc, dummy, 0, 4);
                set_buf(enc, res, 0, 5);
                set_buf(enc, out, 0, 6);
                set_bytes(enc, &params, 7);
                set_buf(enc, self.sc.gemv_part.id(), 0, 8);
                dispatch_only(enc, MtlSize::new(pair_blocks, splits, 1), MtlSize::new(128, 1, 1));
                if splits > 1 {
                    set_cps(enc, &self.pipes.gemv_multi_reduce);
                    set_buf(enc, self.sc.gemv_part.id(), 0, 0);
                    set_buf(enc, res, 0, 2);
                    set_buf(enc, out, 0, 3);
                    set_bytes(enc, &params, 4);
                    dispatch_only(enc, MtlSize::new((mat.rows as usize).div_ceil(256), n, 1), MtlSize::new(256, 1, 1));
                }
            } else {
                let mma = n > GEMV_MAX_TOKENS;
                set_cps(enc, if mma { &self.pipes.gemm_mma } else { &self.pipes.gemv_batch });
                set_buf(enc, mat.pair_buf.id(), 0, 0);
                set_buf(enc, mat.tail_buf.id(), 0, 1);
                set_buf(enc, self.sc.x_rot_f16.id(), 0, 2);
                set_buf(enc, self.sc.x_tok_scale.id(), 0, 3);
                set_buf(enc, dummy, 0, 4);
                set_buf(enc, res, 0, 5);
                set_buf(enc, out, 0, 6);
                set_bytes(enc, &params, 7);
                let grid = if mma {
                    MtlSize::new(n.div_ceil(128), (mat.rows as usize).div_ceil(32), 1)
                } else {
                    MtlSize::new(pair_blocks, 1, 1)
                };
                dispatch_only(enc, grid, MtlSize::new(128, 1, 1));
            }
        }
    }

    unsafe fn encode_conv(&self, enc: Id, lin: &GpuLinear, n: usize, ckpt: Option<&MtlObj>) {
        let per = self.conv_dim * 3;
        let p = ConvParams {
            conv_dim: self.conv_dim as u32,
            in_stride: self.lin_rows as u32,
            n_tok: n as u32,
            ckpt_stride: if ckpt.is_some() { (self.n_lin * per) as u32 } else { 0 },
        };
        let state_off = lin.slot * per * 4;
        unsafe {
            set_buf(enc, self.sc.proj.id(), 0, 0);
            set_buf(enc, lin.conv_w.id(), 0, 1);
            set_buf(enc, self.conv_state.id(), state_off, 2);
            set_buf(enc, self.sc.conv.id(), 0, 3);
            set_bytes(enc, &p, 4);
            set_buf(enc, ckpt.map_or(self.dummy.id(), |c| c.id()), if ckpt.is_some() { state_off } else { 0 }, 5);
            dispatch(enc, &self.pipes.conv, MtlSize::new(self.conv_dim.div_ceil(256), 1, 1), MtlSize::new(256, 1, 1));
        }
    }

    unsafe fn encode_delta(&self, enc: Id, lin: &GpuLinear, n: usize, ckpt: Option<&MtlObj>) {
        let per = self.nv * 128 * 128;
        let p = DeltaParams {
            n_tok: n as u32,
            nk: self.nk as u32,
            nv: self.nv as u32,
            conv_dim: self.conv_dim as u32,
            in_stride: self.lin_rows as u32,
            ktot: self.ktot as u32,
            z_off: self.conv_dim as u32,
            b_off: (self.conv_dim + self.vdim) as u32,
            a_off: (self.conv_dim + self.vdim + self.nv) as u32,
            vdim: self.vdim as u32,
            eps: self.eps,
            ckpt_stride: if ckpt.is_some() { (self.n_lin * per) as u32 } else { 0 },
        };
        let state_off = lin.slot * per * 4;
        unsafe {
            set_buf(enc, self.sc.conv.id(), 0, 0);
            set_buf(enc, self.sc.proj.id(), 0, 1);
            set_buf(enc, lin.dt_bias.id(), 0, 2);
            set_buf(enc, lin.a_neg.id(), 0, 3);
            set_buf(enc, lin.norm_w.id(), 0, 4);
            set_buf(enc, self.delta_state.id(), state_off, 5);
            set_buf(enc, self.sc.core.id(), 0, 6);
            set_bytes(enc, &p, 7);
            set_buf(enc, ckpt.map_or(self.dummy.id(), |c| c.id()), if ckpt.is_some() { state_off } else { 0 }, 8);
            dispatch(enc, &self.pipes.delta, MtlSize::new(self.nv, 1, 1), MtlSize::new(256, 1, 1));
        }
    }

    unsafe fn encode_attention(&self, enc: Id, full: &GpuFull, n: usize, pos0: usize) {
        let pp = AttnPrepParams {
            nh: self.nh as u32,
            nkv: self.nkv as u32,
            hd: self.hd as u32,
            rot: self.rot as u32,
            in_stride: self.attn_rows as u32,
            pos0: pos0 as u32,
            n_tok: n as u32,
            kv_cap: self.kv_cap as u32,
            eps: self.eps,
            theta: self.theta,
        };
        let kv_off = full.slot * self.nkv * self.kv_cap * self.hd * 2;
        let total = pos0 + n;
        let n_splits = if n <= GEMV_MAX_TOKENS { total.div_ceil(ATTN_SPLIT_KEYS).clamp(1, ATTN_MAX_SPLITS) } else { 1 };
        let split_len = total.div_ceil(n_splits);
        let ap = AttnParams {
            nh: self.nh as u32,
            nkv: self.nkv as u32,
            hd: self.hd as u32,
            pos0: pos0 as u32,
            n_tok: n as u32,
            kv_cap: self.kv_cap as u32,
            n_splits: n_splits as u32,
            split_len: split_len as u32,
            scale: 1.0 / (self.hd as f32).sqrt(),
            _pad: 0,
        };
        unsafe {
            set_buf(enc, self.sc.proj.id(), 0, 0);
            set_buf(enc, full.q_norm.id(), 0, 1);
            set_buf(enc, full.k_norm.id(), 0, 2);
            set_buf(enc, self.sc.q.id(), 0, 3);
            set_buf(enc, self.sc.gate.id(), 0, 4);
            set_buf(enc, self.k_cache.id(), kv_off, 5);
            set_buf(enc, self.v_cache.id(), kv_off, 6);
            set_bytes(enc, &pp, 7);
            dispatch(enc, &self.pipes.attn_prep, MtlSize::new(n, self.nh + self.nkv, 1), MtlSize::new(self.hd, 1, 1));

            set_buf(enc, self.sc.q.id(), 0, 0);
            set_buf(enc, self.k_cache.id(), kv_off, 1);
            set_buf(enc, self.v_cache.id(), kv_off, 2);
            set_buf(enc, self.sc.attn_part.id(), 0, 3);
            set_bytes(enc, &ap, 4);
            if self.nh == 4 * self.nkv {
                dispatch(enc, &self.pipes.attn_gqa, MtlSize::new(n_splits, self.nkv, n), MtlSize::new(256, 1, 1));
            } else {
                dispatch(enc, &self.pipes.attn, MtlSize::new(n_splits, self.nh, n), MtlSize::new(256, 1, 1));
            }

            set_buf(enc, self.sc.attn_part.id(), 0, 0);
            set_buf(enc, self.sc.gate.id(), 0, 1);
            set_buf(enc, self.sc.core.id(), 0, 2);
            set_bytes(enc, &ap, 3);
            dispatch(enc, &self.pipes.attn_reduce, MtlSize::new(self.nh, n, 1), MtlSize::new(256, 1, 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::qwen35::Qwen35Config;

    fn small_cfg() -> Qwen35Config {
        Qwen35Config {
            vocab_size: 512,
            hidden_size: 256,
            intermediate_size: 512,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 1,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 10_000_000.0,
            rms_norm_eps: 1e-6,
            full_attn: vec![false, true, false, true],
            lin_k_heads: 1,
            lin_v_heads: 2,
            lin_k_dim: 128,
            lin_v_dim: 128,
            conv_kernel: 4,
            tie_word_embeddings: false,
            eos_token_id: 0,
        }
    }

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        let scale = b.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1e-6);
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) / scale
    }

    /// GPU prefill, GPU token-by-token decode and the CPU reference agree on every position.
    #[test]
    fn gpu_matches_cpu_prefill_and_decode() {
        let model = Qwen35Model::synthetic(small_cfg(), 7, false);
        let seq: Vec<usize> = (0..40).map(|i| (i * 37 + 11) % 512).collect();
        let cpu = model.logits_cpu(&seq);
        let v = model.cfg.vocab_size;
        let mut eng = MetalQwen35::new(&model, 128).unwrap();

        let want: Vec<usize> = (0..seq.len()).collect();
        let mut gpu = vec![0.0f32; seq.len() * v];
        eng.forward(&model, &seq, &want, &mut |i, l| gpu[i * v..(i + 1) * v].copy_from_slice(l)).unwrap();
        let e_prefill = max_rel(&gpu, &cpu);

        eng.reset();
        let mut dec = vec![0.0f32; seq.len() * v];
        // Mixed chunk sizes exercise the batch GEMV, MMA and single-token paths with carried state.
        let mut i = 0;
        for n in [1usize, 3, 9, 1, 1, 25] {
            let chunk = &seq[i..i + n];
            let w: Vec<usize> = (0..n).collect();
            eng.forward(&model, chunk, &w, &mut |j, l| dec[(i + j) * v..(i + j + 1) * v].copy_from_slice(l)).unwrap();
            i += n;
        }
        let e_decode = max_rel(&dec, &cpu);
        let agree = (0..seq.len())
            .filter(|&t| {
                let am = |x: &[f32]| crate::model::real_llm::argmax(&x[t * v..(t + 1) * v]);
                am(&gpu) == am(&cpu)
            })
            .count();
        eprintln!("prefill max rel err {e_prefill:.4}, decode {e_decode:.4}, argmax agree {agree}/{}", seq.len());
        assert!(e_prefill < 0.05 && e_decode < 0.05, "GPU/CPU mismatch: {e_prefill} {e_decode}");
    }

    /// GPU MTP drafts match the CPU MTP head row by row, whether filled singly or in a batch.
    #[test]
    fn mtp_matches_cpu() {
        let model = Qwen35Model::synthetic(small_cfg(), 13, true);
        let seq: Vec<usize> = (0..20).map(|i| (i * 29 + 3) % 512).collect();
        let cpu = model.mtp_logits_cpu(&seq);
        let v = model.cfg.vocab_size;
        let mut eng = MetalQwen35::new(&model, 128).unwrap();
        eng.forward(&model, &seq, &[], &mut |_, _| {}).unwrap();
        let mut agree = 0;
        for i in 0..seq.len() - 1 {
            let d = eng.mtp_fill(&model, &seq[i + 1..i + 2], i, i, true).unwrap().unwrap();
            agree += usize::from(d == crate::model::real_llm::argmax(&cpu[i * v..(i + 1) * v]));
        }
        // Batched fill over all rows must leave the same KV, so the last draft agrees too.
        let last = eng.mtp_fill(&model, &seq[1..], 0, 0, true).unwrap().unwrap();
        let want = crate::model::real_llm::argmax(&cpu[(seq.len() - 2) * v..]);
        eprintln!("MTP argmax agree {agree}/{}, batched last {}", seq.len() - 1, last == want);
        assert!(agree + 1 >= seq.len() - 1 && last == want);
    }

    /// Rejected draft tokens leave no trace: verify junk, keep a prefix, then continue normally.
    #[test]
    fn verify_rollback_matches_plain_decode() {
        let model = Qwen35Model::synthetic(small_cfg(), 11, false);
        let seq: Vec<usize> = (0..24).map(|i| (i * 53 + 5) % 512).collect();
        let cpu = model.logits_cpu(&seq);
        let v = model.cfg.vocab_size;
        let mut eng = MetalQwen35::new(&model, 128).unwrap();
        eng.forward(&model, &seq[..10], &[], &mut |_, _| {}).unwrap();
        let mut got = vec![0.0f32; v];
        eng.verify(&model, &[seq[10], seq[11], 300, 301, 302], &mut |i, l| {
            if i == 1 {
                got.copy_from_slice(l);
            }
        })
        .unwrap();
        let e_verify = max_rel(&got, &cpu[11 * v..12 * v]);
        eng.accept(2);
        assert_eq!(eng.pos(), 12);
        let mut rest = vec![0.0f32; (seq.len() - 12) * v];
        let want: Vec<usize> = (0..seq.len() - 12).collect();
        eng.forward(&model, &seq[12..], &want, &mut |i, l| rest[i * v..(i + 1) * v].copy_from_slice(l)).unwrap();
        let e_after = max_rel(&rest, &cpu[12 * v..]);
        eprintln!("verify err {e_verify:.4}, after rollback {e_after:.4}");
        assert!(e_verify < 0.05 && e_after < 0.05);
    }
}
