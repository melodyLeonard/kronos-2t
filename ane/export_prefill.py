"""Exports a Llama-architecture model's prompt prefill to Core ML for the Neural Engine.

The model takes the embeddings of `seq` prompt tokens (positions 0..seq-1, causal) and returns
the hidden state after the last layer plus every layer's K (after RoPE) and V, which Kronos
loads into its KV cache before decoding on the CPU. Weights are int8 per output channel.

usage: python ane/export_prefill.py <model_dir> --seq 128 [--out <model_dir>/ane]
"""
import argparse, json, os, struct, subprocess, time
import numpy as np
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types


def load_safetensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    hdr = json.loads(raw[8:8 + n])
    base = 8 + n

    def get(name):
        m = hdr[name]
        a, b = m["data_offsets"]
        assert m["dtype"] == "BF16", m["dtype"]
        u16 = np.frombuffer(raw[base + a:base + b], dtype=np.uint16).reshape(m["shape"])
        return (u16.astype(np.uint32) << 16).view(np.float32)
    return get


def rope_tables(seq, head_dim, theta):
    inv = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    ang = np.arange(seq)[:, None] * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, axis=1)
    sin = np.concatenate([np.sin(ang)] * 2, axis=1)
    return cos.astype(np.float16), sin.astype(np.float16)


def build(model_dir, seq):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    H, F, L = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = H // nh
    group = nh // nkv
    get = load_safetensors(os.path.join(model_dir, "model.safetensors"))
    f16 = lambda a: a.astype(np.float16)
    cos, sin = rope_tables(seq, hd, cfg["rope_theta"])
    mask = np.triu(np.full((seq, seq), -1e4, dtype=np.float16), k=1)

    def rms(x, w):
        # sqrt(mean(x^2)) via the L2 norm of x / sqrt(H), keeping fp16 sums in range.
        n = mb.reduce_l2_norm(x=mb.mul(x=x, y=np.float16(1.0 / np.sqrt(H))), axes=[-1], keep_dims=True)
        return mb.mul(x=mb.real_div(x=x, y=mb.maximum(x=n, y=np.float16(1e-3))), y=f16(w))

    def heads(t, n):  # [1, S, n*hd] -> [n, S, hd]
        return mb.transpose(x=mb.reshape(x=t, shape=[seq, n, hd]), perm=[1, 0, 2])

    def rope(t):  # rotate halves, as in HF Llama
        x1 = mb.slice_by_index(x=t, begin=[0, 0, 0], end=[0, seq, hd // 2], end_mask=[True, True, False])
        x2 = mb.slice_by_index(x=t, begin=[0, 0, hd // 2], end=[0, seq, hd], end_mask=[True, True, True])
        rot = mb.concat(values=[mb.mul(x=x2, y=np.float16(-1.0)), x1], axis=-1)
        return mb.add(x=mb.mul(x=t, y=cos), y=mb.mul(x=rot, y=sin))

    layers = []
    for i in range(L):
        p = f"model.layers.{i}."
        layers.append({k: get(p + n) for k, n in [
            ("in", "input_layernorm.weight"), ("post", "post_attention_layernorm.weight"),
            ("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"),
            ("v", "self_attn.v_proj.weight"), ("o", "self_attn.o_proj.weight"),
            ("g", "mlp.gate_proj.weight"), ("u", "mlp.up_proj.weight"), ("d", "mlp.down_proj.weight")]})

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, seq, H), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def prog(x):
        ks, vs = [], []
        for w in layers:
            h = rms(x, w["in"])
            q = rope(heads(mb.linear(x=h, weight=f16(w["q"] / np.sqrt(hd))), nh))
            k = rope(heads(mb.linear(x=h, weight=f16(w["k"])), nkv))
            v = heads(mb.linear(x=h, weight=f16(w["v"])), nkv)
            ks.append(mb.reshape(x=mb.transpose(x=k, perm=[1, 0, 2]), shape=[1, seq, nkv * hd]))
            vs.append(mb.reshape(x=mb.transpose(x=v, perm=[1, 0, 2]), shape=[1, seq, nkv * hd]))
            # Grouped-query attention: [nkv, group*S, hd] x [nkv, S, hd]^T.
            qg = mb.reshape(x=q, shape=[nkv, group * seq, hd])
            s = mb.matmul(x=qg, y=k, transpose_y=True)
            s = mb.add(x=mb.reshape(x=s, shape=[nkv, group, seq, seq]), y=mask)
            pr = mb.reshape(x=mb.softmax(x=s, axis=-1), shape=[nkv, group * seq, seq])
            o = mb.reshape(x=mb.matmul(x=pr, y=v), shape=[nh, seq, hd])
            o = mb.reshape(x=mb.transpose(x=o, perm=[1, 0, 2]), shape=[1, seq, H])
            x = mb.add(x=x, y=mb.linear(x=o, weight=f16(w["o"])))
            h = rms(x, w["post"])
            m = mb.mul(x=mb.silu(x=mb.linear(x=h, weight=f16(w["g"]))), y=mb.linear(x=h, weight=f16(w["u"])))
            x = mb.add(x=x, y=mb.linear(x=m, weight=f16(w["d"])))
        return (mb.identity(x=x, name="hidden"),
                mb.concat(values=ks, axis=0, name="keys"),
                mb.concat(values=vs, axis=0, name="values"))

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
    cfg8 = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=4096))
    return cto.coreml.linear_quantize_weights(model, cfg8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(a.model_dir, "ane")
    os.makedirs(out, exist_ok=True)
    t = time.perf_counter()
    path = os.path.join(out, f"prefill_s{a.seq}.mlpackage")
    build(a.model_dir, a.seq).save(path)
    subprocess.run(["xcrun", "coremlcompiler", "compile", path, out], check=True, stdout=subprocess.DEVNULL)
    print(f"saved {path} and compiled .mlmodelc in {time.perf_counter() - t:.0f}s")


if __name__ == "__main__":
    main()
