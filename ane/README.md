# Neural Engine experiments

Core ML benchmarks of SmolLM2-1.7B-shaped work on the M4 Pro Neural Engine. Separate from the
Kronos engine; needs `pip install coremltools numpy`.

```bash
python ane/bench_matmul.py --layers 24 --out /tmp/ane   # q/k/v/o + MLP projections, 1.61B weights
python ane/placement.py /tmp/ane/proj_lut4_s1_l24.mlpackage   # which unit each op runs on
```

## Results (projections only, 4-bit palettized, M4 Pro)

| Tokens per pass | Neural Engine | Core ML GPU | Core ML CPU |
|---:|---:|---:|---:|
| 1 | 88.6 tok/s (11.3 ms, 76 GB/s) | 31.8 | 33.4 |
| 64 | 5322 tok/s (12.0 ms, 17 TOPS) | 1132 | 1275 |
| 256 | 4110 tok/s (62.3 ms, 13 TOPS) | 1699 | 1610 |

For comparison, Kronos on the CPU decodes the full 1.7B model at ~183 tok/s and prefills at ~434.

- Single-token decode is bandwidth-bound and the Neural Engine reaches only ~76 GB/s, so the
  Kronos CPU path is ~2x faster for one conversation.
- A 64-token pass costs about the same as a 1-token pass (12.0 vs 11.3 ms). Batched work
  (prompt prefill, many concurrent conversations, speculative-decoding verification) runs
  10x+ faster than on the CPU.
- Blockwise int4 weights fall back to the CPU; 4-bit palettized and int8 weights run on the
  Neural Engine.

Not yet measured: attention, KV cache, norms and the LM head, which a full model needs.
