# kronos-2t

kronos-2t is a CPU inference engine for Llama-architecture language models, written in Rust
with ARM NEON kernels and no crate dependencies. It quantizes weights to 4–6 bits with a
Hadamard rotation and GPTQ, stores them as bit-planes, and runs them with integer dot-product
(`sdot`) kernels. It targets Apple Silicon.

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

### Why not use it

- The Metal GPU processes prompts 4–10x faster than any CPU path, including this one.
- With generic calibration data, the 4-bit tiers lose their quality edge over llama.cpp.
- Compiling a model takes minutes (GPTQ), not seconds.
- Only Llama-architecture models with tied embeddings, on ARM64 with `dotprod` (Apple Silicon).
- One request at a time; it serves a single model on one machine, not a fleet.

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
`--calib`. `--mode` picks the precision (`b4`, `p55454` (default), `b5`, `b6`); see
[BENCHMARKS.md](BENCHMARKS.md) for the size, quality and speed of each.

`kronos_server` serves the same model over an OpenAI-compatible HTTP API, so any OpenAI client
library works by changing its base URL:

```bash
cargo run --release --bin kronos_server -- models/SmolLM2-1.7B-Instruct --port 8080 --api-key secret
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="secret")
reply = client.chat.completions.create(
    model="kronos", messages=[{"role": "user", "content": "Summarize GPTQ in one line."}])
print(reply.choices[0].message.content)
```

Endpoints: `POST /v1/chat/completions` (`stream: true` sends server-sent events), `GET /v1/models`,
`GET /health`. Supported request fields: `messages`, `temperature`, `top_p`, `top_k`,
`max_tokens`, `repetition_penalty`, `seed`, `stream`. The server binds to `127.0.0.1` by
default; use `--host 0.0.0.0` and `--api-key` to expose it on a network. It runs one
generation at a time and reuses the KV cache when a request extends the previous conversation.

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

### Other code

`src/router.rs`, `src/synthesis.rs`, `src/ssd.rs`, `src/k2t_format.rs` and `src/mmap_pager.rs`
are experimental sparse-expert and storage components, run by
`cargo run --release --bin kronos-2t`. The LLM path above does not use them.
