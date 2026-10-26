#![deny(unsafe_op_in_unsafe_fn)]
//! `kronos-2t`: Kronecker-Routed Non-multiplying On-device Sparse Engine.
//!
//! Core mathematical & systems primitives for sub-bit, sub-linear sparse edge inference:
//! - [`router`]: Zero-allocation $O(\sqrt{M})$ Cartesian Product-Key Router (`1,000,000+` experts).
//! - [`lut_gemv`]: Multiplication-free Four-Russians Ternary Lookup-Table GEMV (`FP32`).
//! - [`simd_lut`]: Hardware ARM NEON (`vqtbl1q_s8` + `vqtbl4q_u8`) register-fused LUT-GEMV.
//! - [`synthesis`]: Sub-bit Kronecker + canonical codebook L1 weight synthesis.
//! - [`prefetch`]: 6-layer lookahead subspace residual projection & RAM ring-buffer cache.
//! - [`k2t_format`]: 4KB page-aligned `.k2t` v2 multi-rank container & zero-copy views.
//! - [`mmap_pager`]: Zero-copy POSIX `mmap` + `POSIX_MADV_WILLNEED` asynchronous Flash pager.
//! - [`ssd`]: Constant-memory $O(1)$ State-Space Dual (SSD) block with induction shift.
//! - [`accel`]: Parallel SYRK / Cholesky / triangular inverse for GPTQ.
//! - [`compiler`]: BitNet $b1.58$ ternary quantizer & Multi-Rank Kronecker `.k2t` factorizer.
//! - [`engine`]: Zero-allocation multi-layer sparse inference runtime.
//! - [`demo_model`]: Trainable `.k2t`-compiled language & 1,200-token Needle-in-a-Haystack suite.
//! - [`safetensors`]: Zero-copy pure-Rust `.safetensors` (`BF16`/`F16`/`F32`) `mmap` loader.
//! - [`bpe_tokenizer`]: Pure-Rust `SmolLM2`/`GPT-2` byte-level BPE tokenizer (`vocab.json` + `merges.txt`).
//! - [`real_llm`]: Real pretrained `LlamaForCausalLM` (`SmolLM2-135M`) `FP32` vs Multi-Plane NEON LUT verifier.

pub mod accel;
pub mod bpe_tokenizer;
pub mod compiler;
pub mod demo_model;
pub mod engine;
pub mod hadamard;
pub mod k2t_format;
pub mod lut_gemv;
pub mod mmap_pager;
pub mod prefetch;
pub mod real_llm;
pub mod router;
pub mod safetensors;
pub mod simd_lut;
pub mod ssd;
pub mod synthesis;
pub mod thread_pool;
