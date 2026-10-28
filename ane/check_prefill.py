"""Compares a Core ML prefill model with an fp32 numpy reference on random embeddings.

usage: python ane/check_prefill.py <model_dir> <prefill_sN.mlpackage> [--units ANE|CPU]
"""
import argparse, json, os, re
import numpy as np
import coremltools as ct
from export_prefill import load_safetensors, rope_tables


def reference(model_dir, x):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    H, L, nh, nkv = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd, eps, S = H // nh, cfg["rms_norm_eps"], x.shape[0]
    get = load_safetensors(os.path.join(model_dir, "model.safetensors"))
    cos, sin = (t.astype(np.float32) for t in rope_tables(S, hd, cfg["rope_theta"]))
    rms = lambda t, w: t / np.sqrt((t * t).mean(-1, keepdims=True) + eps) * w
    rot = lambda t: np.concatenate([-t[..., hd // 2:], t[..., :hd // 2]], -1)
    mask = np.triu(np.full((S, S), -np.inf), 1)
    ks, vs = [], []
    for i in range(L):
        g = lambda n: get(f"model.layers.{i}.{n}")
        h = rms(x, g("input_layernorm.weight"))
        q = (h @ g("self_attn.q_proj.weight").T).reshape(S, nh, hd).transpose(1, 0, 2)
        k = (h @ g("self_attn.k_proj.weight").T).reshape(S, nkv, hd).transpose(1, 0, 2)
        v = (h @ g("self_attn.v_proj.weight").T).reshape(S, nkv, hd).transpose(1, 0, 2)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        ks.append(k.transpose(1, 0, 2).reshape(S, -1)); vs.append(v.transpose(1, 0, 2).reshape(S, -1))
        kr, vr = np.repeat(k, nh // nkv, 0), np.repeat(v, nh // nkv, 0)
        s = q @ kr.transpose(0, 2, 1) / np.sqrt(hd) + mask
        p = np.exp(s - s.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        o = (p @ vr).transpose(1, 0, 2).reshape(S, H)
        x = x + o @ g("self_attn.o_proj.weight").T
        h = rms(x, g("post_attention_layernorm.weight"))
        a = h @ g("mlp.gate_proj.weight").T
        x = x + ((a / (1 + np.exp(-a))) * (h @ g("mlp.up_proj.weight").T)) @ g("mlp.down_proj.weight").T
    return x, np.stack(ks), np.stack(vs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("mlpackage"); ap.add_argument("--units", default="ANE")
    a = ap.parse_args()
    S = int(re.search(r"_s(\d+)", a.mlpackage).group(1))
    H = json.load(open(os.path.join(a.model_dir, "config.json")))["hidden_size"]
    emb = load_safetensors(os.path.join(a.model_dir, "model.safetensors"))("model.embed_tokens.weight")
    x = emb[np.random.default_rng(0).integers(0, emb.shape[0], S)]
    unit = {"ANE": ct.ComputeUnit.CPU_AND_NE, "CPU": ct.ComputeUnit.CPU_ONLY}[a.units]
    out = ct.models.MLModel(a.mlpackage, compute_units=unit).predict({"x": x[None].astype(np.float16)})
    rh, rk, rv = reference(a.model_dir, x.astype(np.float32))
    for name, got, ref in [("hidden", out["hidden"][0], rh), ("keys", out["keys"], rk), ("values", out["values"], rv)]:
        got = got.astype(np.float32).reshape(ref.shape)
        cos = (got * ref).sum() / np.sqrt((got * got).sum() * (ref * ref).sum())
        print(f"{name:<7} cosine {cos:.5f}  max|diff| {np.abs(got - ref).max():.3f}  max|ref| {np.abs(ref).max():.1f}")


if __name__ == "__main__":
    main()
