#![deny(unsafe_op_in_unsafe_fn)]
//! `kronos-2t`: Kronecker-Routed Non-multiplying On-device Sparse Engine.
//!
//! Layout:
//! - [`kernels`]: LUT GEMV, NEON bit-plane kernels, Hadamard, routing, SSD, and thread pool.
//! - [`storage`]: `.k2t` containers, `mmap` paging, prefetch, and SafeTensors loading.
//! - [`model`]: Llama/Qwen engines, Sparse-MoE runtimes, quantization compiler, tokenizer, memory budgets.
//! - [`backend`]: Metal GPU and Core ML backends.
//! - [`runtime`]: Chat, agent loop, decisions, conversations, device context, vision, and speech.
//! - [`ffi`]: C ABI for iOS hosts.
//! - [`config`]: Shared constants: prompts, markup tokens, paths, server defaults, and limits.
//! - [`util`]: Shared utilities.

pub mod backend;
pub mod config;
pub mod ffi;
pub mod kernels;
pub mod model;
pub mod runtime;
pub mod storage;
pub mod util;
