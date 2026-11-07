# `/translate`: Qwen3.5-9B vs Hy-MT2-1.8B

2026-11-07 · M4 Pro 24 GB · both `nf4g128` on Metal, Hadamard Q8 KV · `kronos_server` from commit `26bf0f3`.
Probe: `translate_eval` (`src/bin/translate_eval.rs`), ten EN chat sentences to FR/ES/DE/YO, each with
1–3 context lines and a tone. Latency is end to end over HTTP on localhost (best of 2 for the quality
table). Hy-MT2 ran with its model-card sampling (temperature 0.7, top_p 0.6, top_k 20, repetition 1.05),
Qwen with greedy decoding.

| | Qwen3.5-9B | Hy-MT2-1.8B |
|---|---|---|
| NF4 g128 file (`kronos-*.k2tb`) | 4.0 GiB | 881 MiB |
| resident weights (server) | 4.30 GiB | 0.86 GiB |
| quality probe, 10 sentences, total | 7.67 s | 1.86 s |
| first request with a new context, p50 / p95 | 715 / 1664 ms | 267 / 344 ms |
| follow-ups sharing context + tone (5–15 words), p50 / p95, n=30 | 727 / 1109 ms | 197 / 408 ms |
| FR/ES/DE fidelity (8 sentences) | 8/8 usable, best register | 7/8 usable, 2 wording slips |
| Yoruba (2 sentences) | wrong but Yoruba-shaped | gibberish (not a supported language) |

Hy-MT2 reuses the prompt prefix between requests that only change the source text (the context and
tone sit before `[Source Text]` in its single user turn), so follow-ups are close to decode-only. The
Qwen3.5 path does not get faster on follow-ups: its hybrid recurrent state cannot be truncated to a
prefix, so each request re-prefills the system turn.

## Outputs

| # | to / tone | input | Qwen3.5-9B | Hy-MT2-1.8B |
|---|---|---|---|---|
| 1 | Spanish, casual | see you tomorrow | te veo mañana | ¿Nos vemos mañana? |
| 2 | French, casual | I'm running late, start without me | Je suis en retard, commence sans moi ! | Je suis en retard, on commence sans moi |
| 3 | German, formal | Could you send me the invoice by Friday? | Könnten Sie mir die Rechnung bis Freitag senden? | Könnten Sie mir die Rechnung bis Freitag zusenden? |
| 4 | French, casual | lol no way, he actually said that?? | lol impossible, il a vraiment dit ça ?? | lol non, il a vraiment dit ça ??? |
| 5 | Yoruba, casual | Happy birthday! Hope you have a great day 🎉 | Ojọ ibi-ọjọ́ rẹ! Níwọ̀n bí ẹni pé ọjọ́ yìí yóò jẹ́ ọjọ́ tí ó tọ́ fún ẹni. 🎉 | Biyi o ṣe! Fẹ́n ọrọ ọpọ ọdẹ́ lẹ́yin 🎉 |
| 6 | Spanish, formal | We regret to inform you that the position has been filled. | Lamentamos informarle que la posición ha sido ocupada. | Lamentamos informarle que el puesto ya ha sido ocupado. |
| 7 | German, casual | Can you pick up the kids? I'm stuck at work | Kannst du die Kids abholen? Ich stecke hier im Job fest. | Kannst du die Kinder abholen? Ich bin bei der Arbeit festgefangen. |
| 8 | French, formal | The package arrived broken, I'd like a refund. | Le colis est arrivé endommagé ; je souhaiterais obtenir un remboursement. | Le colis est arrivé endommagé, je souhaite un remboursement. |
| 9 | Yoruba, casual | Good morning! Did you sleep well? | Oye! Ṣe o nira? | Good morning! Nigbari, dua ọmọ ti ṣe ọmọ? |
| 10 | Spanish, formal | Please find attached the revised contract for your review. | Adjunto encontrará el contrato revisado para su revisión. | Por favor, tenga a su disposición el contrato revisado para su revisión. |

Judgement. Both models honour tone (tu/vous, du/Sie, usted) and keep emoji and punctuation. Hy-MT2 is
idiomatic on 6, 3 and 4 (`el puesto`, `zusenden`, keeps `lol`), slips on 7 (`festgefangen` is not
German for "stuck") and 10 (`tenga a su disposición` drifts from "please find attached"), and the
sampling adds variation between identical requests (1 gained question marks from the question in its
context; a greedy run gave `Nos vemos mañana`). Qwen3.5-9B is more careful on the formal sentences (8,
10) and more colloquial on 7, at 3.5–4x the latency and 4.7x the size. Neither handles Yoruba: Qwen's
output is wrong Yoruba, Hy-MT2's is noise, so `/translate` routes Yoruba, Igbo, Hausa and Swahili to
the general model (`hy_mt2_supports` in `src/config/prompts.rs`).

## Running it

```bash
kronos_server models/Qwen3.5-9B --mode nf4g128 --translate-model models/Hy-MT2-1.8B \
  --host 0.0.0.0 --port 18085
```

`/translate` picks Hy-MT2 for the 38 languages on its card and Qwen3.5 otherwise. Hy-MT2 alone
(`kronos_server models/Hy-MT2-1.8B --mode nf4g128 --calib models/calib.txt`) compiles in about 6 min
on the CPU the first time. Probe: `translate_eval --url http://127.0.0.1:18085 [--latency]`.

## The 1.25-bit variant (`tencent/Hy-MT2-1.8B-1.25Bit-GGUF`, 440 MB)

It is a GGUF whose 224 linear tensors use ggml type 42, `STQ1_0`, the Sherry format (AngelSlim;
llama.cpp PR ggml-org/llama.cpp#22836, not merged as of this note). Each weight is ternary
`{-d, 0, +d}` with 3:4 structured sparsity: in every 64-weight chunk the lanes `w, w+16, w+32, w+48`
form a group of four of which exactly one is zero, stored as a 4-bit index into a 32-entry codebook
plus one sign bit, with an fp16 scale per 256-weight block: 42 B / 256 weights = 1.3125 bits/weight.
`token_embd` (tied with the output head) is Q6_K; norms are F32. The GGUF header also records the
stretched RoPE base `11158840`, which is what Kronos now computes from `rope_scaling.alpha = 1000`.

Kronos cannot ingest it today: the compiler reads SafeTensors only (`src/storage/safetensors.rs`) and
there is no GGUF reader or STQ1_0/Q6_K decoder. It is a small, exact import rather than a
re-quantization, because the model was trained in this format (Sherry QAT), so no calibration or GPTQ
is involved:

1. a GGUF header/tensor reader (the header is 16 MB because of the 120k-token vocab);
2. an STQ1_0 block decoder (codebook and lane order from the PR) and a Q6_K decoder for the embeddings;
3. packing into an existing Kronos ternary tier: `MultiPlaneTernaryMatrix` with one plane holds exactly
   `{-1, 0, +1} x scale` at 2 bits/weight, so the 1.54 B linear weights become about 386 MB plus the
   embeddings at NF4 (about 125 MB), roughly 510 MB resident versus the 440 MB GGUF; keeping the 440 MB
   needs a native STQ1_0 kernel (NEON `vqtbl2q` + `sdot`, as in the PR), which fits the LUT-GEMV
   design in `src/kernels/simd_lut.rs`.

Either way the weights are far above the 60 MB a keyboard extension gets, so the on-device plan is the
host app (or an mmap-streamed file) serving the extension, not the extension itself.
