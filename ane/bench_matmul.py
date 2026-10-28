"""Core ML throughput of SmolLM2-1.7B-shaped projections on Neural Engine, GPU and CPU.

Builds 24 layers of q/k/v/o and gate/up/down linears (2048 hidden, 8192 MLP) with random
weights, compresses them to 4-bit palettized (16-entry lookup table per 16 output channels;
blockwise int4 is not supported on the Neural Engine and falls back to CPU), then times 1-, 64- and 256-token
passes on each compute unit. Attention, norms and the LM head are left out on purpose: this
isolates the matrix-multiply cost that dominates decode and prefill.

usage: python ane/bench_matmul.py [--layers 24] [--out /tmp/ane]
"""
import argparse, os, time
import numpy as np
import coremltools as ct
import coremltools.optimize as cto
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

H, F = 2048, 8192


def build(seq, layers, rng):
    shapes = [(H, H)] * 4 + [(F, H), (F, H), (H, F)]  # q k v o gate up down, [out, in]
    weights = [[(rng.standard_normal(s, dtype=np.float32) * 0.02).astype(np.float16) for s in shapes]
               for _ in range(layers)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, seq, H), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def prog(x):
        for w in weights:
            q, k, v, o, g, u, d = w
            a = mb.add(x=mb.add(x=mb.linear(x=x, weight=q), y=mb.linear(x=x, weight=k)), y=mb.linear(x=x, weight=v))
            x = mb.add(x=x, y=mb.linear(x=a, weight=o))
            m = mb.mul(x=mb.silu(x=mb.linear(x=x, weight=g)), y=mb.linear(x=x, weight=u))
            x = mb.add(x=x, y=mb.linear(x=m, weight=d))
        return x

    model = ct.convert(prog, minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16)
    cfg = cto.coreml.OptimizationConfig(global_config=cto.coreml.OpPalettizerConfig(
        mode="uniform", nbits=4, granularity="per_grouped_channel", group_size=16, weight_threshold=1024))
    return cto.coreml.palettize_weights(model, cfg)


def bench(path, seq, unit, reps):
    m = ct.models.MLModel(path, compute_units=unit)
    x = {"x": np.random.default_rng(1).standard_normal((1, seq, H)).astype(np.float16)}
    for _ in range(3):
        m.predict(x)
    t = time.perf_counter()
    for _ in range(reps):
        m.predict(x)
    return (time.perf_counter() - t) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--out", default="/tmp/ane")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)
    params = a.layers * (4 * H * H + 3 * H * F)
    print(f"{a.layers} layers, {params / 1e9:.2f}B weights at 4-bit (~{params * 0.5 * 1.06 / 2**30:.2f} GiB)")
    units = {"ANE": ct.ComputeUnit.CPU_AND_NE, "GPU": ct.ComputeUnit.CPU_AND_GPU, "CPU": ct.ComputeUnit.CPU_ONLY}
    for seq in (1, 64, 256):
        path = os.path.join(a.out, f"proj_lut4_s{seq}_l{a.layers}.mlpackage")
        if not os.path.exists(path):
            t = time.perf_counter()
            build(seq, a.layers, rng).save(path)
            print(f"  built seq={seq} in {time.perf_counter() - t:.0f}s")
        for name, unit in units.items():
            dt = bench(path, seq, unit, reps=10 if seq == 1 else 5)
            gbs = params * 0.53 / dt / 1e9
            print(f"seq {seq:>3} {name}: {dt * 1e3:8.2f} ms/pass  {seq / dt:8.1f} tok/s  ({gbs:6.1f} GB/s weights, "
                  f"{2 * params * seq / dt / 1e12:5.2f} TOPS)")


if __name__ == "__main__":
    main()
