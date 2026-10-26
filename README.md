# KRONOS-2T (v0.5)

**Kronecker-Routed Non-multiplying On-device Sparse Engine**

`kronos-2t` is a zero-dependency, zero-allocation-per-token Rust & ARM NEON inference engine proving the mathematical and systems primitives required to execute ultra-sparse Trillion-parameter ($2 \times 10^{12}$) architectures. On SmolLM2-135M/360M it is smaller and faster than llama.cpp at matched quality ([BENCHMARKS.md](BENCHMARKS.md)).

---

## Core Architecture

1. **Zero-Allocation 4-Key SIMD Product-Key Router (`src/router.rs`)**: Scores 4 subspace keys simultaneously per query scan into stack-resident top-$k$ buffers across $1{,}000{,}000$ Cartesian micro-experts with **0 heap allocations**.
2. **Full-16-Entry Bipolar Radix-2 + Conway-Sloane $D_4$ Lattice NEON `LUT-GEMV` (`src/simd_lut.rs`)**:
   - **100% Register-LUT Density (`FusedBipolarLutMatrix`)**: Packs 4 bipolar weights $s \in \{-1, +1\}^4$ per 4-bit nibble (`1.00` bit/weight/plane), utilizing all $2^4 = 16$ slots of ARM NEON `vqtbl1q_s8` inside cache-line-interleaved 32-row unrolled zero-multiply shift-add kernels (`4.25` bits/weight = `16` symmetric levels; `5.25` bits/weight = `32` symmetric levels).
   - **Conway-Sloane $D_4$ Checkerboard Lattice Activation Quantizer (`BipolarActivationLut`)**: Exploits invariant parity $\sum_{k=0}^3 s_k q_k \equiv \sum_{k=0}^3 q_k \pmod 2$ across all 16 bipolar entries to project 4-tuples onto $D_4 = \{q \in \mathbb{Z}^4 : \sum q_k \equiv 0 \pmod 2\}$ and shift right by 1 (`>> 1`), doubling `INT8` activation resolution with zero carry error across bit-planes.
   - **Register-Fused Multi-Rank Kronecker (`fused_kronecker_gemv`)**: Pins a 64-byte canonical pattern codebook inside 4 NEON registers (`uint8x16x4_t`, `vqtbl4q_u8`) and synthesizes weights **inside SIMD registers** with **0 bytes** of L1 tile buffer.
3. **Orthogonal Block-64 Walsh-Hadamard (`FWHT-64`) + Sequential Full-Row Cholesky `GPTQ` Compiler (`src/hadamard.rs`, `src/compiler.rs`, `src/real_llm.rs`)**: Rotates weights and activations in-place via compile-time Rademacher-signed `FWHT-64` butterflies ($\langle H_{64} D w, H_{64} D x \rangle = \langle w, x \rangle$) to suppress LLM outlier channels, paired with Ledoit-Wolf shrunk Cholesky `GPTQ` sequential error feedback across all 30 layers + `49,152`-row `lm_head`.
4. **128-Byte Cache-Padded Persistent P-Core Spin Pool (`src/thread_pool.rs`)**: Zero-atomic-contention deterministic core-to-stripe partitioning (`QOS_CLASS_USER_INTERACTIVE`), keeping each P-core's weight slice resident in private L1/L2 cache across tokens, paired with `< 1 ULP` Cephes NEON `exp_f32_neon` `SwiGLU` & `Softmax`.
5. **4KB Page-Aligned `.k2t` Container & POSIX `mmap` Pager (`src/k2t_format.rs`, `src/mmap_pager.rs`)**: Zero-copy virtual memory mapping with 6-layer manifold-aligned lookahead prefetching (`POSIX_MADV_WILLNEED`, `99.8%` hit rate).
6. **Constant-Memory $O(1)$ State-Space Dual + Causal Induction Shift (`src/ssd.rs`)**: Enables 1,200+ token in-context associative recall with **0 bytes** of KV-cache growth.

---

## Benchmark (SmolLM2-135M, Apple M4 Pro)

WikiText-2, same weights and calibration as llama.cpp (imatrix). Details and 360M: [BENCHMARKS.md](BENCHMARKS.md).

| Kronos tier | Size | KL | 1-core tok/s | 8-core tok/s | `llama.cpp` match | Size | KL | 1-core | best CPU | Metal |
| :--- | ---: | ---: | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| **Bipolar-4** (4.25b) | **68.1 MiB** | **0.077** | **665** | **1,834** | IQ4_XS | 85.0 MiB | 0.079 | 417 | 820 | 483 |
| **planes 55454** (4.64b) | **74.5 MiB** | **0.053** | **565** | **1,485** | Q4_1 | 92.1 MiB | 0.052 | 290 | 802 | 502 |
| **Bipolar-5** (5.25b) | **84.2 MiB** | **0.021** | **467** | **1,392** | Q4_K_M | 98.9 MiB | 0.034 | 253 | 723 | 446 |
| **Bipolar-6** (6.25b) | **100.2 MiB** | **0.006** | 357 | **1,166** | Q6_K | 130.3 MiB | 0.007 | **384** | 631 | 447 |

---

## Build, Test & Run

```bash
cargo clippy --all-targets -- -D warnings
cargo test --release

# 1. Core 2.36T-virtual sparse engine & 1,200-token Needle-in-a-Haystack suite
cargo run --release --bin kronos-2t

# 2. Real pretrained LLM (`HuggingFaceTB/SmolLM2-135M`) quality & speed audit
cargo run --release --bin verify_real_weights

# 3. WikiText-2 quality (matched 1:1 to llama-perplexity) + decode speed; see BENCHMARKS.md
cargo run --release --bin wikitext_ppl -- models/SmolLM2-135M wiki.ids \
  --modes b4,p55454,b5,p66666 --calib-ids train.ids --calib-tokens 32768 --cache-dir k2tb --sdot --speed
```
