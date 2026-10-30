# kronos-2t

kronos-2t is an inference engine for Llama and Qwen2 language models, written in Rust with
ARM NEON and `Metal.framework` kernels and no crate dependencies. It quantizes weights to 3–6 bits
with a Hadamard rotation and GPTQ, stores them as bit-planes or NF4 codebook stripes, and runs
them on CPU (`sdot`) or zero-copy Metal GPU buffers on Apple Silicon.

On SmolLM2 models from 135M to 1.7B parameters, compared with llama.cpp on the same machine:

- Files are 3–23% smaller at matched quality. KL divergence is equal or lower in every tier
  except 4-bit at 1.7B, which is a tie.
- Decode is 1.3–2.3x faster than llama.cpp's best CPU configuration, and 0.9–4x the speed of
  llama.cpp on the Metal GPU.

Numbers, method and limitations: [BENCHMARKS.md](BENCHMARKS.md).

| SmolLM2-135M | Size | KL | Decode, 1 core | Decode, best CPU |
|---|---:|---:|---:|---:|
| Kronos Bipolar-4 | 68.1 MiB | 0.076 | 692 tok/s | 1856 tok/s |
| llama.cpp IQ4_XS | 85.0 MiB | 0.079 | 397 tok/s | 790 tok/s |
| Kronos Bipolar-6 | 100.2 MiB | 0.006 | 352 tok/s | 1185 tok/s |
| llama.cpp Q6_K | 130.3 MiB | 0.007 | 377 tok/s | 612 tok/s |

### How it works

- **Rotation.** Weights and activations are rotated with a signed 64-point Walsh-Hadamard
  transform to spread outliers before quantization.
- **GPTQ.** Each layer is quantized against a Hessian built from calibration activations that
  have already passed through the quantized earlier layers.
- **Bit-planes.** A B-bit weight is stored as B planes of ±1. The odd integer
  `q = 2u - (2^B - 1)` lets the low 4 planes run as nibbles on `sdot`; planes 5 and 6 use a
  16-entry lookup table in the same loop.
- **Mixed precision.** Plane counts are set per matrix type. An ablation put the extra plane on
  the qkv, o and down projections.
- **Tied embeddings.** The embedding table is decoded from the quantized output head, so there
  is no separate FP32 copy.
- **Batched prefill.** Prompts are processed 4 tokens per weight load.

### Engineering iterations and mathematical formulation

1. Iteration 1 — Multiplication-free Four-Russians NEON LUT-GEMV and Cartesian product-key routing (`src/lut_gemv.rs`, `src/simd_lut.rs`, `src/router.rs`, `src/synthesis.rs`):
   - Precomputes $3^4 = 81$ (padded to 16-entry nibble tables) linear combinations of 4 input activations so ternary inner products $\sum_j w_{ij} x_j$ with $w_{ij} \in \{-1, 0, +1\}$ reduce to single-cycle ARM NEON `vqtbl1q_s8` register lookups with zero multipliers.
   - Multi-rank Kronecker codebook synthesis (`vqtbl4q_u8`) reconstructs sub-bit expert tiles inside SIMD registers without L1 spill.
   - Product-key routing decomposes $M$ experts into $C_1 \times C_2$ sub-keys ($\sqrt{M} + \sqrt{M}$ dot products), cutting routing complexity from $O(M)$ to $O(\sqrt{M})$.

2. Iteration 2 — Page-aligned `.k2t` container, zero-copy `mmap` pager, and $O(1)$ State-Space Dual recurrence (`src/k2t_format.rs`, `src/mmap_pager.rs`, `src/ssd.rs`):
   - Aligns weight stripes to 4 KB / 16 KB virtual-memory pages mapped directly via POSIX `mmap` + `POSIX_MADV_WILLNEED` lookahead prefetching (`src/prefetch.rs` rank-$r$ subspace residual projection $P_l x_t$).
   - Replaces $O(L)$ growing attention state in recurrent blocks with constant-memory fast-weight recurrence $S_t^{(h)} = \gamma_t^{(h)} S_{t-1}^{(h)} + v_t^{(h)} (k_t^{(h)})^T$, $y_t^{(h)} = S_t^{(h)} q_t^{(h)}$, using a 1-token causal induction shift on odd heads for bigram associative recall.

3. Iteration 3 — Block-64 randomized Walsh-Hadamard incoherence, Ledoit-Wolf Cholesky GPTQ, and $D_4$ lattice activation LUTs (`src/hadamard.rs`, `src/accel.rs`, `src/real_llm.rs`):
   - Heavy-tailed LLM weight and activation outliers destroy uniform low-bit grids. Applying the orthogonal signed Hadamard transform $Q_b = \frac{1}{8} H_{64} D_b$ ($D_b \in \{\pm 1\}^{64}$ Rademacher signs, $Q_b^T Q_b = I_{64}$) preserves exact inner products $\langle Q_b w_b, Q_b x_b \rangle = \langle w_b, x_b \rangle$ while bounding worst-case channel magnitude to $O(\sqrt{\log d / d})$ sub-Gaussian concentration.
   - Layer-wise GPTQ solves $\min_{\hat{W}} \| W X - \hat{W} X \|_F^2 = \operatorname{tr}((W - \hat{W}) H (W - \hat{W})^T)$ using damped Hessian $H = \frac{2}{N} X X^T + \lambda I$ and triangular inverse $H^{-1} = L L^T$ (via Cholesky factorization), propagating column quantization error $\delta_j = (w_j - \hat{w}_j) / [H^{-1}]_{jj}$ across remaining columns $w_{j:} \leftarrow w_{j:} - \delta_j [H^{-1}]_{j, j:}$.
   - Bipolar bit-plane decomposition $q = 2u - (2^B - 1) = \sum_{p=0}^{B-1} 2^p s_p$ ($s_p \in \{\pm 1\}$) executes low planes on `aarch64` `sdot` and higher planes (`p55454`, `b6`) via register LUTs; 4D checkerboard $D_4$ lattice rounding ($\sum_{i=1}^4 z_i \equiv 0 \pmod 2$) improves sphere-packing gain on activations.

4. Iteration 4 — Spherical Lloyd-Max `NF3`/`NF4`/`NF5` codebooks, Hadamard `Q8`/`Q4` KV cache, and Core ML Neural Engine prefill (`src/real_llm.rs`, `src/coreml.rs`):
   - Under Hadamard rotation, group-normalized coordinates follow a spherical Gaussian distribution $w \sim \mathcal{N}(0, \sigma^2)$. Non-uniform Lloyd-Max centroids $c_k = \mathbb{E}[w \mid w \in [t_k, t_{k+1})]$ minimize $\mathbb{E}[(w - c_{Q(w)})^2]$ for 8, 16, and 32 levels (`NF3`, `NF4`, `NF5`) with 128-element group scales (`4.125 bpw` for `nf4g128`).
   - Rotating keys and values by `FWHT-64` before `INT8` (`Q8`) or `INT4` (`Q4`) quantization cuts KV memory 4x–8x; sliding-window + attention-sink eviction (`sink + sliding`) bounds KV memory strictly at $O(W_{\text{sink}} + W_{\text{slide}})$ regardless of sequence length.

5. Iteration 5 — Zero-copy `Metal.framework` GPU `simdgroup_matrix` engine and lossless heterogeneous CPU-draft speculative decoding (`src/metal_gpu.rs`, `src/chat.rs`):
   - Maps `.k2tb` weights into unified Apple Silicon RAM with `MTLStorageModeShared` (`0` copy bytes) and dequantizes `NF3`/`NF4`/`NF5`/`E8Q2` codebooks in threadgroup SRAM feeding `simdgroup_matrix` FP32 accumulators.
   - Runs a `0.5B` draft model on CPU P-cores while the `14B` target verifies $K+1$ tokens in a single batched Metal pass, accepting drafted token $x_i$ with probability $\min(1, p_{\text{target}}(x_i) / p_{\text{draft}}(x_i))$ so the output distribution equals the exact target distribution while doubling decode throughput (`48.5 tok/s` on `Qwen2.5-Coder-14B`).

6. Iteration 6 — Physical `F_NOCACHE` NVMe SLRU expert streaming and `k2di` sub-millisecond device/voice runtime (`src/moe_800b.rs`, `src/device.rs`, `src/speech.rs`):
   - Streams 16 KB-aligned `.k2tm` MoE experts directly from NVMe Flash via `fcntl(fd, F_NOCACHE, 1)` + `pread` into a bounded probationary/protected SLRU DRAM pool, bypassing macOS unified buffer-cache double-buffering so OS page caches cannot evict active attention weights.
   - Indexes local macOS apps, projects, `.venv` toolchains, and documents into an FNV-1a word + 3-gram posting list (`device_index.k2di`) queried in `< 2 ms`.

7. Iteration 7 — Gosset $E_8$ 8-dimensional lattice 2-bit quantization (`e8q2g128`, `2.125 bpw`) with Cholesky-Wagner GPTQ and `Edge8G` / `Mobile5G` `20.91B` Sparse-MoE budgets (`src/hadamard.rs`, `src/sparse_moe.rs`, `src/memory_budget.rs`, `src/mobile_ffi.rs`):
   - Scalar 2-bit quantization (`VQ2`) collapses at `2.1 bpw` (`35 PPL` vs `12.8`). By Viazovska's theorem, the 8D Gosset lattice $E_8 = D_8 \cup (D_8 + \frac{1}{2}\mathbf{1}_8)$ (where $\sum_{i=1}^8 z_i \equiv 0 \pmod 2$) achieves optimal 8D sphere packing ($\Delta_8 = \pi^4 / 384$) and minimal normalized second moment $G(E_8) = \frac{929}{12960} \approx 0.07168$ vs $G(\mathbb{Z}) = \frac{1}{12} \approx 0.08333$.
   - `e8q2g128` packs 256 codebook vectors ($1$ zero + $240$ minimal-norm shell-1 roots $\|v\|^2 = 2$ + $15$ shell-2 axes) into `2 bytes` per 8 weights (`2.0 bpw` + FP16/128 scale = `2.125 bpw`), using Wagner shortest-parity rounding ($O(8)$) inside block-8 Cholesky GPTQ ($8 \times 8$ block error feedback $H_{B,B}^{-1}$).
   - Combined with `NF4` untied embeddings, `Q4` sliding-sink KV, and bounded SLRU/mmap expert paging, a `20.91B`-parameter Sparse-MoE (`3.6B` active, top-4 of 32 experts) runs inside `3.83 GiB <= 5.00 GiB` RSS on 8 GB Macs (`69.7 tok/s`) and `4.01 GiB <= 4.90 GiB` RSS under iOS `os_proc_available_memory` limits (`32.4 tok/s`, exported as `aarch64-apple-ios` `libkronos_2t.a`).

8. Iteration 8 — `Kronos-Jev` single-forward-pass calibrated decision primitives and monotonic rubric verification (`src/jev.rs`, `src/agent.rs`):
   - Generative multi-token judges spend hundreds of decode tokens (`~497 ms` on `0.5B`, seconds on `14B`) with uncalibrated verbal confidence. `Kronos-Jev` scores anchor token sets $\mathcal{A}_k$ in one forward pass (`13.72 ms` neural with cached prefix KV snapshot/restore, `0.016 ms` structural) and fuses structural log-odds $l_0$ with neural Bayes-factor log-odds via $l_{\text{post}} = l_0 + \alpha \log \frac{P(Y)}{P(N)}$, where $\alpha = w_{\text{neural}} \cdot \operatorname{clamp}((m - 0.05)/0.55, 0, 1)^2$ shrinks automatically to $0$ when anchor probability mass $m$ is diffuse or when deterministic compiler/syntax/DOM/file oracles give exact ground truth.
   - Pre-turn KV-aware schema pruning cuts cold tool-schema tokens by `55.4%` while preserving `0 ms` hot KV prefix reuse, and monotonic checkpointing rolls back regressions if post-edit verification drops below the verified high-water mark.

### Current constraints

- Metal GPU acceleration supports `nf4`, `nf4g128`, `nf3`, `nf5`, and `e8q2g128`; bipolar bit-plane tiers (`b4`, `p55454`, `b5`, `b6`) run on CPU (or `--ane` for prefill).
- With generic calibration data, 4-bit tiers lose part of their quality margin over domain-matched GPTQ calibration.
- GPTQ compilation takes minutes on multi-billion-parameter checkpoints and is cached to `.k2tb` / `.k2tm` for instant subsequent loads.
- Targets Apple Silicon (`aarch64` NEON `dotprod`, macOS `Metal.framework`, and `aarch64-apple-ios` static library FFI).
- Processes one generation stream per server instance with stateful KV-prefix reuse.

### Building

```bash
cargo build --release
cargo test --release
```

### Chat and API server

Download an Instruct model from Hugging Face (`config.json`, `model.safetensors`, `vocab.json`,
`merges.txt`, `tokenizer.json`) into `models/<name>`, then chat in the terminal:

```bash
cargo run --release --bin kronos_chat -- models/SmolLM2-1.7B-Instruct --calib calib.txt
```

The first run compiles the model with GPTQ on `calib.txt` (any few hundred KB of plain text;
a mix of encyclopedic and web text works well) and caches it as
`models/<name>/kronos-<mode>.k2tb`. Later runs load the cache in seconds and no longer need
`--calib`. `--mode` picks the precision (`b4`, `nf4g128`, `p55454` (default), `b5`, `b6`); see
[BENCHMARKS.md](BENCHMARKS.md) for the size, quality and speed of each.

The KV cache is stored as int8 after a Hadamard rotation, about 4x smaller than FP32;
`--fp32-kv` turns this off. On macOS, `nf4` and `nf4g128` models run on the Metal GPU by default
(`--cpu` forces CPU execution). Pass `--draft models/<small-model>` to enable lossless speculative
decoding with a smaller same-family draft model on CPU while the target runs on Metal:

```bash
cargo run --release --bin kronos_chat -- models/Qwen2.5-Coder-14B-Instruct \
  --mode nf4g128 --ctx 32768 --draft models/Qwen2.5-Coder-0.5B-Instruct
```

Built with `--features ane`, `--ane` runs prompt prefill on the Neural Engine (see
[ane/README.md](ane/README.md)).

### Sandboxed workspace tools and web UI

Pass `--agent --workdir <dir>` to `kronos_chat` (or toggle `/agent` interactively) to enable six
jailed workspace tools (`write_file`, `edit_file`, `read_file`, `list_dir`, `search_files`,
`run_command`) with post-edit syntax checks for Rust, JS/TS, JSON, HTML, CSS and Python. For
non-interactive single-goal runs, use `kronos_agent`:

```bash
cargo run --release --bin kronos_agent -- --model models/Qwen2.5-Coder-14B-Instruct \
  --draft models/Qwen2.5-Coder-0.5B-Instruct --mode nf4g128 --ctx 32768 \
  --workdir ./workspace --prompt "Create index.html and app.js for a counter widget."
```

`kronos_server` serves the same model over an OpenAI-compatible HTTP API (including `tools` and
`tool_calls`) alongside a split-pane web chat and live workspace preview:

```bash
cargo run --release --bin kronos_server -- models/Qwen2.5-Coder-14B-Instruct \
  --mode nf4g128 --ctx 32768 --draft models/Qwen2.5-Coder-0.5B-Instruct \
  --port 18080 --workdir ./workspace
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:18080/v1", api_key="secret")
reply = client.chat.completions.create(
    model="kronos", messages=[{"role": "user", "content": "Summarize GPTQ in one line."}])
print(reply.choices[0].message.content)
```

Open `http://127.0.0.1:18080/` for the web chat and workspace UI served by the same process.

Endpoints: `POST /v1/chat/completions` (`stream: true` sends server-sent events),
`POST /v1/agent/chat` (streaming tool loop), `GET /v1/agent/files`, `GET /v1/agent/file?path=...`,
`POST /v1/agent/clear`, `GET /preview/<path>`, `GET /v1/models`, `GET /health`. Supported request
fields: `messages`, `tools`, `tool_choice`, `temperature`, `top_p`, `top_k`, `max_tokens`,
`repetition_penalty`, `seed`, `stream`. The server binds to `127.0.0.1` by default; use
`--host 0.0.0.0` and `--api-key` to expose it on a network. It runs one generation at a time and
reuses the KV cache when a request extends the previous conversation.

To embed Kronos in another Rust program, use `kronos_2t::chat::ChatModel` directly.

### Benchmarking

`wikitext_ppl` compiles a model, measures quality the same way as `llama-perplexity`, and
optionally measures decode and prefill speed:

```bash
cargo run --release --bin wikitext_ppl -- models/SmolLM2-135M wiki.ids \
  --modes b4,p55454,b5,p66666 --calib-ids train.ids --calib-tokens 32768 \
  --cache-dir k2tb --sdot --speed --prefill
```

Models are read from Hugging Face safetensors (`config.json`, `model.safetensors`,
`vocab.json`, `merges.txt`). `wiki.ids` and `train.ids` hold one token id per line.

### Sparse-MoE, Edge-20B budget suite, and iOS static library

`src/sparse_moe.rs`, `src/memory_budget.rs`, `src/jev.rs`, and `src/mobile_ffi.rs` compile and run Hugging Face SafeTensors Sparse-MoE models (`Qwen2MoeForCausalLM`, `MixtralForCausalLM`, `GptOssForCausalLM`) into `.k2tb` trunk + 16 KB-aligned `.k2tm` expert shards under `Edge8G` (`5.00 GiB`) and `Mobile5G` (`4.90 GiB`) RAM envelopes:

```bash
cargo run --release --bin bench_edge_budget
cargo build --release --lib --target aarch64-apple-ios
```

`src/router.rs`, `src/synthesis.rs`, `src/ssd.rs`, `src/k2t_format.rs`, and `src/mmap_pager.rs` provide the sub-bit Kronecker LUT and $O(1)$ SSD components exercised by `cargo run --release --bin kronos-2t`, while `src/moe_800b.rs` and `bench_800b` benchmark `F_NOCACHE` NVMe SLRU expert streaming up to 812B scale.
