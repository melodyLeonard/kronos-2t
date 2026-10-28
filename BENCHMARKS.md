# Benchmarks

Kronos vs llama.cpp 0.7.0 (build 11669) and Ollama 0.35.0 on an Apple M4 Pro (10P + 4E cores, 24 GB).

- Models: SmolLM2-135M, -360M and -1.7B base. Both engines load the same weights.
- Quality: WikiText-2 test, 64 chunks of 256 tokens, second half of each chunk scored, as
  `llama-perplexity -c 256 --chunks 64`. KL and top-1 are measured against full-precision logits.
- Calibration: the same WikiText-2 train tokens for both engines (Kronos GPTQ, llama.cpp
  imatrix). 32k tokens for 135M/360M, 16k for 1.7B.
- Size: all weight bytes, including embeddings.
- Speed: `tg32` decode and `pp256` prompt processing, mean of 5 runs. Kronos values are the
  median of 3 repetitions. "Best CPU" is the fastest of 1, 4 and 8 threads.

Each Kronos tier is compared with the llama.cpp quant closest in quality. Kronos tiers:
Bipolar-4 (4.25 bits), p55454 (4.6–4.7 bits, extra plane on qkv/o/down), Bipolar-5 (5.25 bits),
Bipolar-6 (6.25 bits).

## Quality

| Model | Kronos tier | Size | KL | llama.cpp | Size | KL |
|---|---|---:|---:|---|---:|---:|
| 135M | Bipolar-4 | 68.1 MiB | 0.076 | IQ4_XS | 85.0 MiB | 0.079 |
| 135M | p55454 | 74.5 MiB | 0.053 | Q4_1 | 92.1 MiB | 0.052 |
| 135M | Bipolar-5 | 84.2 MiB | 0.021 | Q4_K_M | 98.9 MiB | 0.034 |
| 135M | Bipolar-6 | 100.2 MiB | 0.006 | Q6_K | 130.3 MiB | 0.007 |
| 360M | Bipolar-4 | 183.3 MiB | 0.052 | IQ4_XS | 214.5 MiB | 0.055 |
| 360M | p55454 | 202.0 MiB | 0.036 | Q4_1 | 235.6 MiB | 0.040 |
| 360M | Bipolar-5 | 226.4 MiB | 0.016 | Q4_K_M | 256.4 MiB | 0.022 |
| 360M | Bipolar-6 | 269.5 MiB | 0.005 | Q6_K | 348.6 MiB | 0.007 |
| 1.7B | Bipolar-4 | 867 MiB | 0.050 | IQ4_XS | 895 MiB | 0.048 |
| 1.7B | p55454 | 963 MiB | 0.033 | Q4_K_M | 1005 MiB | 0.036 |
| 1.7B | Bipolar-5 | 1071 MiB | 0.014 | Q5_K_M | 1167 MiB | 0.020 |
| 1.7B | Bipolar-6 | 1275 MiB | 0.004 | Q6_K | 1339 MiB | 0.017 |

The hidden sizes of 135M and 360M (576, 960) are not multiples of 256, so llama.cpp falls back
from K-quants on most tensors there. At 1.7B (2048) K-quants apply fully, and the 4-bit tier is
a tie.

### Calibration domain (135M, KL on WikiText / C4)

Both engines calibrated on a 50/50 mix of WikiText-2 train and C4 text:

| Kronos | Size | KL | llama.cpp | Size | KL |
|---|---:|---:|---|---:|---:|
| Bipolar-4 | 68.1 MiB | 0.085 / 0.101 | IQ4_XS | 85.0 MiB | 0.080 / 0.084 |
| p55454 | 74.5 MiB | 0.059 / 0.070 | Q4_1 | 92.1 MiB | 0.052 / 0.054 |
| Bipolar-5 | 84.2 MiB | 0.023 / 0.029 | Q4_K_M | 98.9 MiB | 0.034 / 0.036 |
| Bipolar-6 | 100.2 MiB | 0.007 / 0.008 | Q6_K | 130.3 MiB | 0.007 / 0.008 |

GPTQ fits its calibration data more tightly than llama.cpp's imatrix. With generic calibration,
the 4-bit tiers fall behind the next-larger llama.cpp quant in KL; 5 and 6 bits still match or
beat it at 15–23% less size.

## Decode (tg32, tok/s)

| Model | Pair | Kronos 1T | llama.cpp 1T | Kronos best CPU | llama.cpp best CPU | llama.cpp Metal | Ollama* |
|---|---|---:|---:|---:|---:|---:|---:|
| 135M | B4 / IQ4_XS | **692** | 397 | **1856** | 790 | 464 | 416 |
| 135M | p55454 / Q4_1 | **561** | 281 | **1468** | 784 | 477 | 399 |
| 135M | B5 / Q4_K_M | **458** | 247 | **1333** | 683 | 436 | 419 |
| 135M | B6 / Q6_K | 352 | **377** | **1185** | 612 | 433 | 380 |
| 360M | B4 / IQ4_XS | **265** | 155 | **759** | 419 | 344 | 304 |
| 360M | p55454 / Q4_1 | **214** | 106 | **665** | 388 | 354 | – |
| 360M | B5 / Q4_K_M | **176** | 92 | **620** | 313 | 307 | 293 |
| 360M | B6 / Q6_K | 134 | **144** | **525** | 311 | 297 | 302 |
| 1.7B | B4 / IQ4_XS | **57** | 28 | **183** | 139 | 174 | 147 |
| 1.7B | p55454 / Q4_K_M | **45** | 22 | **177** | 118 | 171 | 157 |
| 1.7B | B5 / Q5_K_M | **38** | 20 | **159** | 104 | 132 | – |
| 1.7B | B6 / Q6_K | **29** | 20 | **127** | 100 | 139 | 125 |

\* Ollama: `tg128` median of 10 runs. Interquartile ranges are within 8% of the median, except
135M IQ4_XS (311–459) and 1.7B IQ4_XS (136–171).

## Prompt processing (pp256, tok/s)

| Model | Pair | Kronos 1T | llama.cpp 1T | Kronos best CPU | llama.cpp best CPU | llama.cpp Metal |
|---|---|---:|---:|---:|---:|---:|
| 135M | B4 / IQ4_XS | **935** | 660 | **4576** | 2611 | 18918 |
| 135M | p55454 / Q4_1 | 678 | **1038** | **3211** | 2340 | 19105 |
| 135M | B5 / Q4_K_M | 549 | **681** | **2873** | 2114 | 17961 |
| 135M | B6 / Q6_K | 434 | **562** | 2459 | **2533** | 18891 |
| 360M | B4 / IQ4_XS | **355** | 288 | **1936** | 1414 | 7995 |
| 360M | B6 / Q6_K | 158 | **240** | 963 | **1075** | 7946 |
| 1.7B | B4 / IQ4_XS | 76 | **183** | 434 | 437 | 1930 |
| 1.7B | p55454 / Q4_K_M | **53** | 33 | **309** | 215 | 1882 |
| 1.7B | B6 / Q6_K | 32 | **46** | 190 | **315** | 1838 |

Prefill is batched (4 tokens per weight load, `sdot`) with NEON attention. The batched GEMM
runs at about 133 GMAC/s per core, 85% of `sdot` peak; an `i8mm` (`SMMLA`) version measured the
same, since `SMMLA` peaks only 1.2x higher on the M4. The Metal GPU is 4–10x faster than any CPU
result for prefill.

## Chat model: SmolLM2-1.7B-Instruct

Measured on the current code. Quality: 32 WikiText-2 chunks, 16k WikiText train tokens for
calibration, int8 KV cache. The full-precision reference scores 12.81 PPL.

| Mode | Bits | Weights | KV cache (2048 ctx) | PPL | KL | Top-1 | pp256, 8T |
|---|---:|---:|---:|---:|---:|---:|---:|
| p55454 | 4.72 | 963 MB | 25.5 MB | 13.29 | 0.051 | 89.0% | 509 tok/s |
| nf4g128 | 4.13 | 842 MB | 25.5 MB | 13.52 | 0.066 | 88.0% | 606 tok/s |

`nf4g128` uses a 16-level Lloyd-Max codebook with one scale per 128 weights: 13% smaller and
19% faster at prefill than p55454, with slightly higher KL. The FP32 KV cache takes 96 MB.

Chat decode through `kronos_server` (median of 3 interleaved runs, tok/s):

| Workload | Previous release, p55454 | p55454 | nf4g128 |
|---|---:|---:|---:|
| Summarize a 290-token passage | 100 | 125 | 143 |
| 250-word essay on that passage | 100 | 121 | 141 |
| 250-word story | 110 | 131 | 147 |

With `--ane`, prompt prefill runs on the Neural Engine: the first token of a 290-token prompt
arrives in about 340 ms instead of 650–850 ms, with the same output quality (WikiText PPL
12.05 vs 12.06 for CPU prefill).

## MoE streaming experiment

`bench_800b` models an 812B-parameter mixture-of-experts layout (80 layers, 7168 hidden,
top-8 of 225 experts) with synthetic weights, reading experts from the SSD with `F_NOCACHE`.
Measured on the same M4 Pro:

| Case | tok/s | Notes |
|---|---:|---|
| All experts read from SSD | 0.78 | 5.9 GB/s SSD reads |
| Zipf-distributed routing, RAM cache + prefetch | 1.64 | 47% of experts hit the cache |
| All active experts already in RAM | 13–24 | Upper bound; does not fit in 24 GB at 3 bits |

Real MoE models are trained to spread tokens evenly across experts, so the cache hit rate
will be lower than with Zipf routing. On FFN layers from SmolLM2-1.7B, 3-bit NF3 experts reach
0.96 output cosine to FP32 and 2-bit VQ2 0.86. On the full 1.7B model, 2-bit VQ2 scores
35 PPL against 12.8, too lossy to use.

## Remaining limitations

- GPU prefill is far ahead; CPU prefill trails llama.cpp for some quant types.
- 4-bit quality depends on the calibration domain (see above).
- Bipolar-6 single-core decode is ~7% slower than Q6_K at 135M/360M (43% faster at 1.7B).
- GPTQ compile: 34 s (135M), 90 s (360M), ~12 min (1.7B) per tier.
- Apple Silicon only (NEON + `sdot`).

## Reproduce

```bash
# token ids: one per line, from llama-tokenize
llama-tokenize -m F16.gguf -f wiki.test.raw  --ids --no-bos | tr -d '[] ' | tr ',' '\n' > wiki.ids
llama-tokenize -m F16.gguf -f wiki.train.raw --ids --no-bos | tr -d '[] ' | tr ',' '\n' > train.ids

# Kronos: quality, decode and prefill
cargo run --release --bin wikitext_ppl -- models/SmolLM2-135M wiki.ids \
  --modes b4,p55454,b5,p66666 --calib-ids train.ids --calib-tokens 32768 \
  --cache-dir k2tb --sdot --speed --prefill

# llama.cpp
llama-perplexity -m F16.gguf -f wiki.test.raw -c 256 --chunks 64 --kl-divergence-base f16.kld
llama-imatrix -m F16.gguf -f wiki.train.raw -c 256 --chunks 128 -o imatrix.gguf
llama-quantize --imatrix imatrix.gguf BF16.gguf IQ4_XS.gguf IQ4_XS
llama-perplexity -m IQ4_XS.gguf -c 256 --chunks 64 --kl-divergence-base f16.kld --kl-divergence
llama-bench -m IQ4_XS.gguf -ngl 0 -t 1,4,8 -p 256 -n 32 -r 5
```

`scripts/st2gguf.py` converts SmolLM2 safetensors to GGUF without torch.
