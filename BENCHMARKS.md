# Verified Benchmarks: Kronos vs llama.cpp and Ollama

All numbers below were measured on one Apple M4 Pro (10 performance + 4 efficiency cores) with
both engines running the **same base-model weights**, scored on the **same tokens**, and calibrated
on the **same data**. Kronos numbers come from `src/bin/wikitext_ppl.rs`; llama.cpp numbers from
`llama-perplexity` / `llama-bench`; Ollama numbers from its HTTP API.

## Headline

On SmolLM2-135M and SmolLM2-360M, at equal or better quality than llama.cpp:

- **Kronos files are 12–23% smaller** at every quality tier, from IQ4_XS-class to Q6_K-class.
- **Single-core decode is 1.6–2.0× faster** than llama.cpp CPU at tiers up to 5.25 bits. The
  exception is Bipolar-6 vs Q6_K, where Kronos is 7–8% slower.
- **Multi-core CPU decode is 1.7–2.2× faster** than llama.cpp's best CPU thread count.
- **Multi-core CPU decode is 1.9–3.8× faster** than llama.cpp on the Metal GPU, and 2.0–6.0×
  faster than Ollama.

## Setup

| | |
|---|---|
| Hardware | Apple M4 Pro, macOS |
| llama.cpp | 0.7.0 (build 11669, Homebrew, BLAS + Metal) |
| Ollama | 0.35.0 (llama.cpp Metal backend) |
| Models | `HuggingFaceTB/SmolLM2-135M`, `HuggingFaceTB/SmolLM2-360M` (base, not Instruct) |
| Eval text | WikiText-2 raw **test**: first 64 chunks × 256 tokens, second half of each chunk scored (8,128 tokens), as in `llama-perplexity -c 256` |
| Calibration | WikiText-2 raw **train**, first 32,768 tokens for **both** engines (Kronos GPTQ Hessians; llama.cpp `llama-imatrix --chunks 128 -c 256`) |
| Metrics | Perplexity; KL divergence and top-1 agreement vs the full-precision model's logits at the same positions |
| Size | All weight bytes, including the embedding table (Kronos decodes tied embeddings from the quantized `lm_head`; llama.cpp size as reported by `llama-bench`) |
| Speed | Decode `tg32` from an empty context, mean of 5 runs (`llama-bench -n 32 -p 0 -r 5`; `wikitext_ppl --speed`). Kronos: median of 3 suite repetitions. Ollama: `raw` API, median of 6 `eval_count / eval_duration` |

### How the numbers were checked
- **Same weights.** The 135M GGUF is `ggml-org/SmolLM2-135M-GGUF` BF16. All 272 of its tensors
  match the safetensors bit-for-bit (q/k in llama.cpp's RoPE permutation). The 360M GGUF was
  written by a numpy converter (`scripts/st2gguf.py`). Run on 135M, that converter reproduces the
  official file's metadata and tensors exactly.
- **Same tokens.** Both engines read token IDs produced by `llama-tokenize`. Kronos FP32 and
  llama.cpp F16 perplexity agree: **17.4040 vs 17.4040** (135M) and 12.77 vs 12.77 (360M).
- **Mismatched-tensor fallback.** SmolLM2's hidden sizes (576 and 960) are not multiples of 256, so
  llama.cpp's K-quants fall back to other types on most tensors. Its "Q4_K_M" is mostly Q5_0 with a
  Q8_0 embedding table (about 6.2 bits/weight here). Comparisons below therefore use **file size**,
  not quant names.

## Quality vs size (WikiText-2, lower KL = closer to full precision)

### SmolLM2-135M (full precision: perplexity 17.40)

| Engine / mode | Size | Perplexity | KL (nats) | Top-1 |
|---|---:|---:|---:|---:|
| **Kronos Bipolar-4** (4.25b) | **68.1 MiB** | **18.84** | **0.0771** | **84.5%** |
| llama.cpp IQ4_XS (pure) | 71.5 MiB | 20.32 | 0.1443 | 77.4% |
| **Kronos planes 55454** (4.64b) | **74.5 MiB** | **18.22** | **0.0532** | **86.7%** |
| **Kronos Bipolar-5** (5.25b) | **84.2 MiB** | **17.82** | **0.0211** | **91.3%** |
| llama.cpp IQ4_XS | 85.0 MiB | 18.96 | 0.0792 | 84.1% |
| llama.cpp Q4_K_M (pure) | 85.1 MiB | 18.62 | 0.0668 | 83.8% |
| llama.cpp Q4_1 | 92.1 MiB | 18.20 | 0.0521 | 87.2% |
| llama.cpp Q5_1 (pure) | 96.3 MiB | 17.76 | 0.0248 | 89.8% |
| llama.cpp Q4_K_M | 98.9 MiB | 17.86 | 0.0337 | 89.2% |
| **Kronos Bipolar-6** (6.25b) | **100.2 MiB** | **17.48** | **0.0063** | **94.9%** |
| llama.cpp Q5_1 | 104.8 MiB | 17.58 | 0.0126 | 92.9% |
| llama.cpp Q6_K | 130.3 MiB | 17.56 | 0.0071 | 94.6% |
| llama.cpp Q8_0 | 136.4 MiB | 17.47 | 0.0038 | 96.0% |

### SmolLM2-360M (full precision: perplexity 12.77)

| Engine / mode | Size | Perplexity | KL (nats) | Top-1 |
|---|---:|---:|---:|---:|
| **Kronos Bipolar-4** (4.25b) | **183.3 MiB** | **13.47** | **0.0535** | **88.0%** |
| llama.cpp IQ4_XS (pure) | 192.0 MiB | 14.75 | 0.1439 | 78.5% |
| **Kronos planes 55454** (4.68b) | **202.0 MiB** | **13.24** | **0.0351** | **90.1%** |
| llama.cpp IQ4_XS | 214.5 MiB | 13.53 | 0.0555 | 87.4% |
| **Kronos Bipolar-5** (5.25b) | **226.4 MiB** | **12.95** | **0.0152** | **93.3%** |
| llama.cpp Q4_1 | 235.6 MiB | 13.26 | 0.0397 | 89.6% |
| llama.cpp Q4_K_M | 256.4 MiB | 12.96 | 0.0224 | 91.8% |
| **Kronos Bipolar-6** (6.25b) | **269.5 MiB** | **12.84** | **0.0047** | **96.1%** |
| llama.cpp Q5_1 | 273.1 MiB | 12.89 | 0.0107 | 93.8% |
| llama.cpp Q6_K | 348.6 MiB | 12.86 | 0.0067 | 94.8% |
| llama.cpp Q8_0 | 366.8 MiB | 12.80 | 0.0038 | 95.9% |

Plane mode `pQOGDH` gives the bit-planes for qkv, o, gate_up, down and lm_head; `55454` gives the
extra plane to qkv, o and down. An ablation found those to be the most KL-efficient places for
extra bits.

## Decode speed at matched quality (tokens/s, higher is better)

Each Kronos tier is paired with the llama.cpp quant of about the same KL. "Best CPU" is
llama.cpp's fastest of 1, 4 and 8 threads.

### SmolLM2-135M

| Pair (Kronos vs llama.cpp) | Kronos 1T | llama.cpp 1T | Kronos 8T | llama.cpp best CPU | llama.cpp Metal | Ollama |
|---|---:|---:|---:|---:|---:|---:|
| Bipolar-4 vs IQ4_XS | **665** | 417 | **1,834** | 820 | 483 | 304 |
| 55454 vs Q4_1 | **565** | 290 | **1,485** | 802 | 502 | 334 |
| Bipolar-5 vs Q4_K_M | **467** | 253 | **1,392** | 723 | 446 | 437 |
| Bipolar-6 vs Q6_K | 357 | **384** | **1,166** | 631 | 447 | 429 |

### SmolLM2-360M

| Pair (Kronos vs llama.cpp) | Kronos 1T | llama.cpp 1T | Kronos 8T | llama.cpp best CPU | llama.cpp Metal | Ollama |
|---|---:|---:|---:|---:|---:|---:|
| Bipolar-4 vs IQ4_XS | **265** | 160 | **840** | 438 | 341 | 357 |
| 55454 vs Q4_1 | **215** | 107 | **689** | 409 | 364 | — |
| Bipolar-5 vs Q4_K_M | **179** | 96 | **637** | 351 | 311 | 302 |
| Bipolar-6 vs Q6_K | 136 | **148** | **561** | 334 | 299 | 284 |

Kronos's 8-thread numbers vary by up to ±10% between runs (the M4 Pro's two performance clusters).
The tables show the median of three runs.

## What changed to get here

These changes are on branch `improve/beat-cutting-edge`, relative to tag `baseline`:

1. **Real calibration data.** GPTQ now runs on 32k tokens instead of 187. WikiText-2 KL for
   Bipolar-4 dropped from 0.185 to 0.077.
2. **Bit allocation per matrix.** An ablation of KL saved per extra bit ranked qkv > down > o ≈
   lm_head > gate_up, which gave the `55454` mix.
3. **SDOT kernel.** Odd-grid weights `q = 2u − (2^B − 1)` split into
   `2·Σu_lo·x − 15·Σx + Σ_{p≥4} 2^p·Σs_p·x`. The low 4 planes run on int8 `sdot` over nibble codes;
   high planes stay on the bipolar LUT inside the same loop. This is bit-identical storage at about
   1.8× fewer instructions. Bipolar-4 single-core went from 366 to 665 tok/s.
4. **Honest memory.** Tied embeddings are decoded from the quantized `lm_head` rows. The baseline
   kept a separate 108 MiB FP32 table that its "Weight RAM" figure did not count.
5. **Branch-free activation quantizers.** The D4 LUT build (bit-exact with the scalar version)
   and the int8 quantizer are vectorized. This lifted 5/6-plane multi-core decode by about 30%.
6. **Faster GPTQ build.** The Hessian is accumulated in batched, parallel f32 (2× faster compile),
   with a 12-point scale search and lower damping.

## Limitations

- **Small models only.** Results cover 135M and 360M parameters. At this size, the GPU's fixed
  per-token overhead dominates Metal and Ollama decode. Larger models will narrow or reverse the
  GPU comparison.
- **Prompt processing is not compared.** Kronos processes prompts one token at a time;
  llama.cpp batches prefill and will be far faster on long prompts.
- **Calibration domain.** Calibration and evaluation both use WikiText-2 (train vs test), and
  both engines use the same calibration data.
- **Compile time.** The GPTQ compile takes about 2–4 minutes for 135M and about 8 minutes for 360M
  per mode. llama.cpp quantization takes seconds.
- **Bipolar-6 is slower than Q6_K on one core** (7–8%).
- **Ollama speeds are noisier** (the tg32 median ranged ±15% between runs). Ollama's 135M
  Q4_K_M figure comes from a quant without imatrix; imatrix does not change decode speed.

## Reproduce

```bash
# llama.cpp tokens (same tokenizer for both SmolLM2 sizes)
llama-tokenize -m SmolLM2-135M-F16.gguf -f wiki.test.raw  --ids --no-bos | tr -d '[] ' | tr ',' '\n' > wiki.ids
llama-tokenize -m SmolLM2-135M-F16.gguf -f wiki.train.raw --ids --no-bos | tr -d '[] ' | tr ',' '\n' > train.ids

# Kronos quality + speed (compiled engines are cached in --cache-dir)
cargo run --release --bin wikitext_ppl -- models/SmolLM2-135M wiki.ids \
  --modes b4,p55454,b5,p66666 --calib-ids train.ids --calib-tokens 32768 \
  --cache-dir k2tb --sdot --speed

# llama.cpp quality (KL vs F16) and speed
llama-perplexity -m F16.gguf -f wiki.test.raw -c 256 --chunks 64 -b 256 --kl-divergence-base f16.kld
llama-imatrix -m F16.gguf -f wiki.train.raw -c 256 --chunks 128 -o imatrix.gguf
llama-quantize --imatrix imatrix.gguf BF16.gguf IQ4_XS.gguf IQ4_XS
llama-perplexity -m IQ4_XS.gguf -c 256 --chunks 64 -b 256 --kl-divergence-base f16.kld --kl-divergence
llama-bench -m IQ4_XS.gguf -ngl 0 -t 1,4,8 -p 0 -n 32 -r 5
```
