# Benchmarks

Kronos vs llama.cpp 0.7.0 and Ollama 0.35.0 on an Apple M4 Pro.

- Models: SmolLM2-135M, SmolLM2-360M (base). Same weights in both engines.
- Quality: WikiText-2 test, 64 × 256-token chunks, second half scored (as `llama-perplexity -c 256`).
  KL and top-1 are measured against full-precision logits.
- Calibration: the first 32,768 WikiText-2 train tokens for both engines (Kronos GPTQ, llama.cpp imatrix).
- Size: all weight bytes, including embeddings.
- Speed: `tg32` decode, mean of 5 runs.

## Quality vs size

SmolLM2-135M (FP32 perplexity 17.40)

| Model | Size | PPL | KL | Top-1 |
|---|---:|---:|---:|---:|
| **Kronos Bipolar-4** | **68.1 MiB** | **18.84** | **0.077** | **84.5%** |
| llama.cpp IQ4_XS (pure) | 71.5 MiB | 20.32 | 0.144 | 77.4% |
| **Kronos p55454** | **74.5 MiB** | **18.22** | **0.053** | **86.7%** |
| **Kronos Bipolar-5** | **84.2 MiB** | **17.82** | **0.021** | **91.3%** |
| llama.cpp IQ4_XS | 85.0 MiB | 18.96 | 0.079 | 84.1% |
| llama.cpp Q4_1 | 92.1 MiB | 18.20 | 0.052 | 87.2% |
| llama.cpp Q4_K_M | 98.9 MiB | 17.86 | 0.034 | 89.2% |
| **Kronos Bipolar-6** | **100.2 MiB** | **17.48** | **0.006** | **94.9%** |
| llama.cpp Q5_1 | 104.8 MiB | 17.58 | 0.013 | 92.9% |
| llama.cpp Q6_K | 130.3 MiB | 17.56 | 0.007 | 94.6% |

SmolLM2-360M (FP32 perplexity 12.77)

| Model | Size | PPL | KL | Top-1 |
|---|---:|---:|---:|---:|
| **Kronos Bipolar-4** | **183.3 MiB** | **13.47** | **0.054** | **88.0%** |
| **Kronos p55454** | **202.0 MiB** | **13.24** | **0.035** | **90.1%** |
| llama.cpp IQ4_XS | 214.5 MiB | 13.53 | 0.056 | 87.4% |
| **Kronos Bipolar-5** | **226.4 MiB** | **12.95** | **0.015** | **93.3%** |
| llama.cpp Q4_1 | 235.6 MiB | 13.26 | 0.040 | 89.6% |
| llama.cpp Q4_K_M | 256.4 MiB | 12.96 | 0.022 | 91.8% |
| **Kronos Bipolar-6** | **269.5 MiB** | **12.84** | **0.005** | **96.1%** |
| llama.cpp Q5_1 | 273.1 MiB | 12.89 | 0.011 | 93.8% |
| llama.cpp Q6_K | 348.6 MiB | 12.86 | 0.007 | 94.8% |

The hidden sizes (576, 960) are not multiples of 256, so llama.cpp's K-quants fall back to other
types on most tensors. Compare by size, not by quant name.

## Decode speed at matched quality (tok/s)

| Model | Pair | Kronos 1T | llama.cpp 1T | Kronos 8T | llama.cpp best CPU | llama.cpp Metal | Ollama |
|---|---|---:|---:|---:|---:|---:|---:|
| 135M | Bipolar-4 / IQ4_XS | **665** | 417 | **1834** | 820 | 483 | 304 |
| 135M | p55454 / Q4_1 | **565** | 290 | **1485** | 802 | 502 | 334 |
| 135M | Bipolar-5 / Q4_K_M | **467** | 253 | **1392** | 723 | 446 | 437 |
| 135M | Bipolar-6 / Q6_K | 357 | **384** | **1166** | 631 | 447 | 429 |
| 360M | Bipolar-4 / IQ4_XS | **265** | 160 | **840** | 438 | 341 | 357 |
| 360M | p55454 / Q4_1 | **215** | 107 | **689** | 409 | 364 | – |
| 360M | Bipolar-5 / Q4_K_M | **179** | 96 | **637** | 351 | 311 | 302 |
| 360M | Bipolar-6 / Q6_K | 136 | **148** | **561** | 334 | 299 | 284 |

## Limitations

- Models up to 360M only.
- Prompt processing (prefill) is not compared.
- Calibration and evaluation are both from WikiText-2.
- GPTQ compile takes minutes; llama.cpp quantizes in seconds.
- Bipolar-6 is slower than Q6_K on one core.

## Reproduce

```bash
cargo run --release --bin wikitext_ppl -- models/SmolLM2-135M wiki.ids \
  --modes b4,p55454,b5,p66666 --calib-ids train.ids --calib-tokens 32768 --cache-dir k2tb --sdot --speed

llama-perplexity -m F16.gguf -f wiki.test.raw -c 256 --chunks 64 --kl-divergence-base f16.kld
llama-imatrix -m F16.gguf -f wiki.train.raw -c 256 --chunks 128 -o imatrix.gguf
llama-quantize --imatrix imatrix.gguf BF16.gguf IQ4_XS.gguf IQ4_XS
llama-perplexity -m IQ4_XS.gguf -c 256 --chunks 64 --kl-divergence-base f16.kld --kl-divergence
llama-bench -m IQ4_XS.gguf -ngl 0 -t 1,4,8 -p 0 -n 32 -r 5
```

`wiki.ids` / `train.ids` are `llama-tokenize --ids --no-bos` output, one id per line.
`scripts/st2gguf.py` converts SmolLM2 safetensors to GGUF without torch.
