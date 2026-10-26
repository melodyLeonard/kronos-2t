"""Minimal SmolLM2 (LlamaForCausalLM, tied embeddings) safetensors -> BF16 GGUF converter.

Tokenizer and general metadata are copied from a reference GGUF of the same tokenizer family;
tensor names and the q/k RoPE permutation follow llama.cpp's convert_hf_to_gguf.py.
usage: st2gguf.py <gguf-py dir> <model_dir> <reference.gguf> <out.gguf> <name> <size_label>
"""
import sys, json, struct
import numpy as np
sys.path.insert(0, sys.argv[1])
import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType

model_dir, ref_path, out_path, name, size_label = sys.argv[2:7]
cfg = json.load(open(f"{model_dir}/config.json"))
assert cfg.get("tie_word_embeddings", True), "only tied-embedding models supported"
raw = open(f"{model_dir}/model.safetensors", "rb").read()
n = struct.unpack("<Q", raw[:8])[0]; hdr = json.loads(raw[8:8 + n]); base = 8 + n

def st(name):
    m = hdr[name]; a, b = m["data_offsets"]; assert m["dtype"] == "BF16"
    return np.frombuffer(raw[base + a:base + b], dtype=np.uint16).reshape(m["shape"])

def permute(w, n_head):  # llama.cpp LlamaModel.permute
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)

ref = GGUFReader(ref_path)
w = GGUFWriter(out_path, "llama")
hp = {
    "llama.block_count": cfg["num_hidden_layers"],
    "llama.context_length": cfg["max_position_embeddings"],
    "llama.embedding_length": cfg["hidden_size"],
    "llama.feed_forward_length": cfg["intermediate_size"],
    "llama.attention.head_count": cfg["num_attention_heads"],
    "llama.attention.head_count_kv": cfg["num_key_value_heads"],
    "llama.rope.freq_base": float(cfg["rope_theta"]),
    "llama.attention.layer_norm_rms_epsilon": float(cfg["rms_norm_eps"]),
    "llama.attention.key_length": cfg["hidden_size"] // cfg["num_attention_heads"],
    "llama.attention.value_length": cfg["hidden_size"] // cfg["num_attention_heads"],
    "llama.vocab_size": cfg["vocab_size"],
    "llama.rope.dimension_count": cfg["hidden_size"] // cfg["num_attention_heads"],
    "general.name": name,
    "general.size_label": size_label,
}
for key, f in ref.fields.items():
    if key.startswith("GGUF.") or key == "general.architecture":
        continue
    vtype = f.types[0]
    if key in hp:
        val = hp[key]
    else:
        val = f.contents()
    sub = f.types[1] if vtype == GGUFValueType.ARRAY else None
    w.add_key_value(key, val, vtype, sub)

def add_bf16(gname, arr):
    w.add_tensor(gname, np.ascontiguousarray(arr).view(np.uint8), raw_dtype=GGMLQuantizationType.BF16)

def add_f32(gname, arr):
    w.add_tensor(gname, (arr.astype(np.uint32) << 16).view(np.float32))

nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
add_bf16("token_embd.weight", st("model.embed_tokens.weight"))
for i in range(cfg["num_hidden_layers"]):
    p = f"model.layers.{i}."
    add_f32(f"blk.{i}.attn_norm.weight", st(p + "input_layernorm.weight"))
    add_bf16(f"blk.{i}.attn_q.weight", permute(st(p + "self_attn.q_proj.weight"), nh))
    add_bf16(f"blk.{i}.attn_k.weight", permute(st(p + "self_attn.k_proj.weight"), nkv))
    add_bf16(f"blk.{i}.attn_v.weight", st(p + "self_attn.v_proj.weight"))
    add_bf16(f"blk.{i}.attn_output.weight", st(p + "self_attn.o_proj.weight"))
    add_f32(f"blk.{i}.ffn_norm.weight", st(p + "post_attention_layernorm.weight"))
    add_bf16(f"blk.{i}.ffn_gate.weight", st(p + "mlp.gate_proj.weight"))
    add_bf16(f"blk.{i}.ffn_up.weight", st(p + "mlp.up_proj.weight"))
    add_bf16(f"blk.{i}.ffn_down.weight", st(p + "mlp.down_proj.weight"))
add_f32("output_norm.weight", st("model.norm.weight"))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("wrote", out_path)
