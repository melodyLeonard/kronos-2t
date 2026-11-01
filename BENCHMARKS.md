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

## Qwen2.5-Coder-14B-Instruct (32k context)

Measured on the same 24 GB M4 Pro (20-core GPU) with `--mode nf4g128 --ctx 32768` and the
Hadamard int8 KV cache. When `--draft models/Qwen2.5-Coder-0.5B-Instruct` (`nf4g128`, 243 MiB)
is attached, the 14B target runs on the Metal GPU using zero-copy `MTLStorageModeShared` weight
views while the 0.5B draft runs on CPU P-cores, overlapping draft prompt sync with GPU prefill.

Across 10 coding prompts, max last-token KL divergence between the Metal GPU path and the CPU
`nf4g128` path is 0.000109 (9/10 exact 32-token greedy matches), and greedy speculative decoding
matches greedy target decoding token-for-token.

Coding workload (median of 3 runs on a 1,451-token prompt + 400-token reply, normalized to a
1,300-token read + 400-token write step alongside `llama-bench -p 1300 -n 400 -r 3`, plus a
2-step `kronos_agent` 3-file to-do web app build with syntax validation):

| Configuration | Weights | Max RSS (32k ctx) | Prefill | Decode | 1300 + 400 step | 3-file app build |
|---|---:|---:|---:|---:|---:|---:|
| Kronos nf4g128, CPU | 8.17 GiB | 10.16 GiB | 57.0 tok/s | 21.5 tok/s | 41.4 s | 259.4 s |
| Kronos nf4g128, CPU + 0.5B draft | 8.41 GiB | 10.21 GiB | 57.0 tok/s | 30.1 tok/s | 36.1 s | 67.6 s |
| Kronos nf4g128, Metal | 8.17 GiB | 9.32 GiB | 252.9 tok/s | 25.1 tok/s | 21.1 s | – |
| Kronos nf4g128, Metal + 0.5B draft | 8.41 GiB | 10.21 GiB | 249.0 tok/s | 48.5 tok/s | 13.4 s | 22.7 s |
| llama.cpp Q4_K_M, Metal (`llama-bench`) | 8.37 GiB | – | 229.8 tok/s | 26.0 tok/s | 21.1 s | – |

Notes:
- Draft acceptance is 74.1% (320/432) on the 400-token coding benchmark and 77.5% (595/768) on
  Step 1 of the 3-file web app build (50.8 tok/s decode).
- The 1,451 + 400 coding benchmark completes in 14.1 s on Metal + 0.5B draft (5.83 s prefill +
  8.25 s decode) with zero swap activity.
- Pure 14B Metal prefill reaches 262.6 tok/s at 512 tokens (`simdgroup_matrix` FP32 accumulate
  with on-the-fly NF4 codebook dequantization in threadgroup memory).

## Prompt lookup, per-session KV, and 4-bit KV

Measured on the same M4 Pro through `kronos_server` (`--mode nf4g128`, int8 KV unless noted,
greedy, median of 3 runs). "Code edit" sends a 157-line Python module (1,595 prompt tokens) and
asks for one method rename with the complete file echoed back (1,207 reply tokens);
"free-form" is a 300-token essay. "Before" is commit f1a605a (5-token lookup only when no draft
is attached); "after" tries lookup first (up to 16 tokens), then the draft model.

| Model | Workload | Before | After | Lookup accepted |
|---|---|---:|---:|---:|
| 0.5B, Metal, no draft | code edit | 249 tok/s | 290 tok/s | 1043 / 1273 |
| 0.5B, Metal, no draft | free-form | 176 tok/s | 175 tok/s | 61 / 123 |
| 14B + 0.5B draft, Metal* | code edit | 15.4 tok/s | 14.7 tok/s | draft 1064 / 1077 before; lookup+draft 1102 / 1293 after |
| 14B + 0.5B draft, Metal* | free-form | 10.5 tok/s | 9.6 tok/s | 142 / 361 before; 141 / 369 after |
| 14B, Metal, no draft | code edit, free-form | pending | pending | – |

\* Measured (n=2) while another process ran an 8-core WikiText battery, so the CPU draft had no
free P-cores; the 48 tok/s reference above was measured on an idle machine. Under that load the
two binaries are within noise of each other on the 14B. Re-measure on an idle machine before
drawing a 14B conclusion.

Greedy output is identical in all modes (`test_tiered_speculation_matches_plain_greedy_05b`
compares plain decoding, lookup, and lookup + draft token-for-token on the 0.5B).

### Per-session KV cache (`--kv-sessions 3`)

Two conversations of about 5,530 tokens each, alternating one new turn per request. Prefill
time is the server's `prefill_secs` for that request, which here is the time to the first token.

| Model | Switch between A and B | Resume after restart | Without `session_id` |
|---|---:|---:|---:|
| 0.5B, Metal, int8 KV | 47–84 ms (11 new tokens) | 66 ms (from `state-dir/kv`, 36 MiB) | 1.3–1.6 s (5,195 tokens) |
| 0.5B, Metal, Q4 KV | 49–96 ms | 97 ms | – |
| 0.5B, CPU, int8 KV | 0.46–0.64 s | 1.1 s | 52–54 s cold |
| 14B + 0.5B draft, Metal | pending | pending | pending |

Replies after a restore match the replies from a full reprocess byte for byte (same KV bytes).
The CPU rows were measured while another process used all P-cores.

### 4-bit KV (`--q4-kv`)

SmolLM2-135M, `nf4g128`, WikiText-2 test as in the tables above (64 chunks of 256 tokens,
GPTQ calibration on 32k WikiText train tokens), plus 16 chunks of 2,048 tokens for longer
contexts. KV bytes are the measured allocation after filling 32,768 positions.

| KV cache | PPL, 256 ctx | KL | Top-1 | PPL, 2048 ctx | KL | Top-1 | KV bytes/token | KV at 32k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 reference weights, FP32 KV | 17.40 | – | – | 13.39 | – | – | 73,728 | 2,304 MiB |
| int8 Hadamard (default) | 18.47 | 0.067 | 85.2% | 14.22 | 0.065 | 86.2% | 12,240 | 382.5 MiB |
| 4-bit Hadamard (`--q4-kv`) | 19.09 | 0.099 | 82.4% | 14.67 | 0.098 | 83.2% | 6,120 | 191.2 MiB |

For Qwen2.5-Coder-14B on Metal (48 layers, 8 KV heads of 128, group-32 scales) the same slot
layout gives 110,592 B/token with int8 KV (3.38 GiB at 32k) and 55,296 B/token with Q4
(1.69 GiB at 32k). Q4 KV costs about 0.03 nat of KL and 3 points of top-1 agreement on the 135M;
on the 0.5B it changed one of the two 5,500-token test answers above, so it is a memory trade,
not a free default.

## Agent context growth (`eval/`, Qwen2.5-Coder-14B + 0.5B draft, Metal)

`eval/run_eval.py` runs the 10 hidden-check tasks in `eval/tasks/` through the live
`kronos_server` agent loop (`--ctx 32768`, temperature 0). Two full runs on the same M4 Pro 24 GB;
the baseline shared the machine with a 3 GB model process and a `cargo test`, the after run did not.
Per-task tables: `eval/results/baseline.md`, `eval/results/after.md`, `eval/results/before_after.md`.

| metric | baseline `f1a605a` | after `fc146c3` |
|---|---:|---:|
| pass rate | 3/10 | 2/10 |
| median step time | 16.66 s | 6.71 s |
| median prompt tokens per step | 5,746 | 4,825 |
| total agent steps / prompt tokens | 92 / 658,463 | 152 / 752,015 |
| peak server RSS | 8,428 MiB | 6,698 MiB |
| follow-up first token over 20 turns, median | 11.46 s | 5.18 s |
| follow-up first token, p90 / max | 14.65 / 21.72 s | 49.90 / 51.06 s |
| follow-up prompt tokens at turn 20 | 14,937 | 4,808 |
| follow-up turns ending without a final reply | 0 | 13 |

`fc146c3` replaced the inlined per-turn file contents (up to 12 KB) with a file list plus files
changed since the last turn, capped `run_command` / `search_files` / `list_dir` output at 2 KB
(full text under `.kronos_tool_output/`), ended the loop at the first passing verification after
the last edit, and compacted history past an 8,192-token budget. Step size and time fell; two
regressions cost the follow-up task: `edit_file` retried `old_text` that carried pasted `  6: `
line-number prefixes until `max_steps`, and compaction folded the 24 KB `read_file` result of the
step just taken, causing re-read loops and the 51 s re-prefill outliers. `c67d265` strips pasted
line-number prefixes, never compacts the most recent step, and raises the budget to 12,288; it is
not yet measured.

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

## Edge-20B Sparse-MoE envelopes, Gosset E8 2-bit quantization, and Kronos-Jev (`bench_edge_budget`)

Measured on the same Apple Silicon machine with `cargo run --release --bin bench_edge_budget`:

| Suite / Configuration | Active / Total Params | Weight Format | Resident RSS / Envelope | Throughput / Latency | Quality / Calibration |
|---|---|---|---:|---:|---|
| Edge8G Sparse-MoE (SLRU `F_NOCACHE`) | 3.6B / 20.91B (top-4 of 32) | `e8q2g128` experts + `nf4` trunk + `Q4` sink KV | 3.832 GiB / 5.000 GiB | 69.7 tok/s decode | zero swap, deterministic cap |
| Mobile5G Sparse-MoE (SLRU `F_NOCACHE`) | 3.6B / 20.91B (top-4 of 32) | `e8q2g128` experts + `nf4` trunk + `Q4` sink KV | 4.006 GiB / 4.900 GiB | 32.4 tok/s decode | iOS 5 GB envelope compliant |
| Mobile5G Sparse-MoE (zero-copy `mmap`) | 2.8B / 15.90B (top-4 of 24) | `e8q2g128` experts + `nf4` trunk + `Q4` sink KV | 3.953 GiB / 4.900 GiB | 40.9 tok/s decode | zero-copy `MADV_PAGEOUT` |
| SmolLM2-135M Gosset `e8q2g128` | 135M dense | 8D Gosset $E_8$ Cholesky-Wagner GPTQ (`2.136 bpw`) | 36.0 MiB | 562.4 tok/s decode | beats scalar `vq2g128` at `2.13 bpw` |
| Kronos-Jev single-pass decision engine | 0.5B / structural + neural | Bayes-factor log-odds + KV prefix cache | 0 B extra weights | 13.72 ms neural / 0.016 ms structural | 94.4% acc, Brier 0.0307, ECE 0.0672 (36x faster than 497 ms generative judge; -55.4% cold tool tokens) |

## Remaining limitations

- Metal GPU acceleration covers `nf4`, `nf4g128`, `nf3`, `nf5`, and `e8q2g128`; bipolar bit-plane tiers (`b4`, `p55454`, `b5`, `b6`) run on CPU (or `--ane` for prefill).
- 4-bit quality depends on the calibration domain (see above).
- Bipolar-6 single-core decode is ~7% slower than Q6_K at 135M/360M (43% faster at 1.7B).
- GPTQ compile: 34 s (135M), 90 s (360M), ~12 min (1.7B) per tier on CPU; the 30B-A3B MoE experts compile on the Metal GPU in 12.3 min (48 min on CPU). Dense models still compile on CPU.
- Apple Silicon only (`aarch64` NEON + `sdot`, `Metal.framework` on macOS, `libkronos_2t.a` C FFI on `aarch64-apple-ios`).

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
