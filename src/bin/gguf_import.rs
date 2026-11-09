//! Imports a GGUF checkpoint into a Kronos model directory.
//!
//! `gguf_import <file.gguf> --inspect` lists metadata and tensors (name, dims, ggml type).
//! `gguf_import <file.gguf> --out <dir>` dequantizes every tensor to BF16 and writes
//! `model.safetensors` with HuggingFace tensor names, so `kronos_chat --mode …` can compile it.
//! Tencent's `q2_0c` (GGML type 40): 512 weights per block, one f16 scale, codes `{-3,-1,1,3}`.
//!
//! Header parsing lives in `kronos_2t::storage::gguf`.

use kronos_2t::storage::gguf::{f16_to_f32, type_name, Gguf, TensorInfo};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn bf16(v: f32) -> u16 {
    let b = v.to_bits();
    // round to nearest even
    let round = ((b >> 16) & 1) + 0x7fff;
    ((b.wrapping_add(round)) >> 16) as u16
}

/// Dequantizes one tensor to f32 (row-major, dims[0] fastest as in ggml).
fn dequant(g: &Gguf, t: &TensorInfo, q2c_levels: &[f32; 4]) -> Vec<f32> {
    let n: usize = t.dims.iter().product::<u64>() as usize;
    let base = &g.data[t.offset as usize..];
    match t.ty {
        0 => base[..n * 4].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        1 => base[..n * 2].chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        30 => base[..n * 2].chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        8 => {
            // Q8_0: blocks of 32: f16 scale + 32 int8
            let mut out = Vec::with_capacity(n);
            for blk in base[..n / 32 * 34].chunks_exact(34) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                out.extend(blk[2..].iter().map(|&q| q as i8 as f32 * d));
            }
            out
        }
        2 => {
            // Q4_0: blocks of 32: f16 scale + 16 bytes of nibbles (low nibbles first 16, high next 16), value = (q - 8) * d
            let mut out = Vec::with_capacity(n);
            for blk in base[..n / 32 * 18].chunks_exact(18) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qs = &blk[2..];
                for i in 0..16 { out.push(((qs[i] & 0x0f) as f32 - 8.0) * d); }
                for i in 0..16 { out.push(((qs[i] >> 4) as f32 - 8.0) * d); }
            }
            out
        }
        14 => {
            // Q6_K: 256-weight super-blocks: ql[128] (low 4 bits), qh[64] (high 2 bits), scales[16] i8, d f16.
            const QK: usize = 256;
            let mut out = Vec::with_capacity(n);
            for blk in base[..n / QK * 210].chunks_exact(210) {
                let ql = &blk[..128];
                let qh = &blk[128..192];
                let sc = &blk[192..208];
                let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
                let mut vals = [0f32; QK];
                for half in 0..2 {
                    let (ql, qh, sc) = (&ql[half * 64..], &qh[half * 32..], &sc[half * 8..]);
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        let o = half * 128;
                        vals[o + l] = d * sc[is] as i8 as f32 * q1 as f32;
                        vals[o + l + 32] = d * sc[is + 2] as i8 as f32 * q2 as f32;
                        vals[o + l + 64] = d * sc[is + 4] as i8 as f32 * q3 as f32;
                        vals[o + l + 96] = d * sc[is + 6] as i8 as f32 * q4 as f32;
                    }
                }
                out.extend_from_slice(&vals);
            }
            out
        }
        ty if is_q2_0c(g, ty) => {
            // block_q2_0c: 512 weights, f16 scale + 128 bytes of 2-bit codes (little-endian pairs within a byte).
            const QK: usize = 512;
            let mut out = Vec::with_capacity(n);
            for blk in base[..n / QK * (2 + QK / 4)].chunks_exact(2 + QK / 4) {
                let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                let qs = &blk[2..];
                for i in 0..QK {
                    let (byte, shift) = q2_0c_index(i);
                    let code = (qs[byte] >> shift) & 3;
                    out.push(q2c_levels[code as usize] * d);
                }
            }
            out
        }
        other => panic!("tensor {} has unsupported ggml type {} ({})", t.name, other, type_name(other)),
    }
}

/// `block_q2_0c` packing (llama.cpp PR #19357 `dequantize_row_q2_0c`): four consecutive codes per
/// byte, element `j` in bits `2*(j%4)` of `qs[j/4]`; values `{-3,-1,1,3} * d`.
fn q2_0c_index(i: usize) -> (usize, u32) {
    (i / 4, 2 * (i % 4) as u32)
}

fn is_q2_0c(g: &Gguf, ty: u32) -> bool {
    // Not in upstream ggml: the file's `general.file_type` and the unknown id identify it.
    ty == 40 || (type_name(ty) == "?" && g.kv.contains_key("general.architecture"))
}

/// llama.cpp tensor name → HuggingFace name for HunYuan dense / Llama-style models.
fn hf_name(name: &str) -> Option<String> {
    if name == "token_embd.weight" { return Some("model.embed_tokens.weight".into()); }
    if name == "output_norm.weight" { return Some("model.norm.weight".into()); }
    if name == "output.weight" { return Some("lm_head.weight".into()); }
    let rest = name.strip_prefix("blk.")?;
    let (idx, tail) = rest.split_once('.')?;
    let pfx = format!("model.layers.{idx}");
    let mapped = match tail {
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "attn_q.bias" => "self_attn.q_proj.bias",
        "attn_k.bias" => "self_attn.k_proj.bias",
        "attn_v.bias" => "self_attn.v_proj.bias",
        "attn_q_norm.weight" => "self_attn.query_layernorm.weight",
        "attn_k_norm.weight" => "self_attn.key_layernorm.weight",
        "attn_norm.weight" => "input_layernorm.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("{pfx}.{mapped}"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gguf_import <file.gguf> (--inspect | --out DIR) [--levels -3,-1,1,3]");
        std::process::exit(2);
    }
    let path = PathBuf::from(&args[1]);
    let bytes = fs::read(&path).expect("read gguf");
    let g = Gguf::parse(&bytes).expect("parse gguf");
    let levels: [f32; 4] = args.iter().position(|a| a == "--levels").and_then(|i| args.get(i + 1)).map(|s| {
        let v: Vec<f32> = s.split(',').map(|x| x.trim().parse().expect("level")).collect();
        [v[0], v[1], v[2], v[3]]
    }).unwrap_or([-3.0, -1.0, 1.0, 3.0]);

    if args.iter().any(|a| a == "--inspect") {
        for (k, v) in &g.kv {
            if !k.starts_with("tokenizer.ggml.") || k.ends_with("model") || k.ends_with("pre") { println!("{k} = {}", v.short()); }
        }
        let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
        for t in &g.tensors {
            let n: u64 = t.dims.iter().product();
            let e = by_type.entry(t.ty).or_insert((0, 0));
            e.0 += 1;
            e.1 += n;
        }
        println!("--- {} tensors, data {} MB", g.tensors.len(), g.data.len() >> 20);
        for (ty, (cnt, n)) in &by_type { println!("type {ty:>3} {:<8} {cnt:>4} tensors {:>8} M params", type_name(*ty), n / 1_000_000); }
        for t in g.tensors.iter().take(24) { println!("{:<40} {:?} {}", t.name, t.dims, type_name(t.ty)); }
        return;
    }

    let out = PathBuf::from(args.iter().position(|a| a == "--out").and_then(|i| args.get(i + 1)).expect("--out DIR"));
    fs::create_dir_all(&out).expect("out dir");
    write_safetensors(&g, &out.join("model.safetensors"), &levels);
    eprintln!("wrote {}", out.join("model.safetensors").display());
}

/// Writes every tensor as BF16 with HF names: 8-byte header length, JSON header, raw data.
fn write_safetensors(g: &Gguf, path: &Path, levels: &[f32; 4]) {
    let mut header = String::from("{");
    let mut blobs: Vec<Vec<u8>> = Vec::new();
    let mut off = 0usize;
    for t in &g.tensors {
        let Some(name) = hf_name(&t.name) else { eprintln!("skip {}", t.name); continue };
        let vals = dequant(g, t, levels);
        let mut buf = Vec::with_capacity(vals.len() * 2);
        for v in &vals { buf.extend_from_slice(&bf16(*v).to_le_bytes()); }
        // ggml dims are innermost-first; HF shape is outermost-first.
        let shape: Vec<u64> = t.dims.iter().rev().copied().collect();
        if header.len() > 1 { header.push(','); }
        header.push_str(&format!("\"{name}\":{{\"dtype\":\"BF16\",\"shape\":{shape:?},\"data_offsets\":[{off},{}]}}", off + buf.len()));
        off += buf.len();
        blobs.push(buf);
        eprintln!("{name} {:?} {}", shape, type_name(t.ty));
    }
    header.push('}');
    while header.len() % 8 != 0 { header.push(' '); }
    let mut f = fs::File::create(path).expect("create safetensors");
    f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    for b in &blobs { f.write_all(b).unwrap(); }
}
