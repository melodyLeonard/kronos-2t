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
| quality probe, 10 sentences, total | 7.67 s | 1.58 s |
| first request with a new context, p50 / p95 | 715 / 1664 ms | 255 / 300 ms |
| follow-ups sharing context + tone (5–15 words), p50 / p95, n=30 | 727 / 1109 ms | 183 / 330 ms |
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
| 2 | French, casual | I'm running late, start without me | Je suis en retard, commence sans moi ! | Je suis en retard, commence sans moi |
| 3 | German, formal | Could you send me the invoice by Friday? | Könnten Sie mir die Rechnung bis Freitag senden? | Könnten Sie mir den Rechnungsstück am Freitag zusenden? |
| 4 | French, casual | lol no way, he actually said that?? | lol impossible, il a vraiment dit ça ?? | lol non, vraiment il a dit ça?? |
| 5 | Yoruba, casual | Happy birthday! Hope you have a great day 🎉 | Ojọ ibi-ọjọ́ rẹ! Níwọ̀n bí ẹni pé ọjọ́ yìí yóò jẹ́ ọjọ́ tí ó tọ́ fún ẹni. 🎉 | Blessed birthday! Hope you have a great day 🎉 |
| 6 | Spanish, formal | We regret to inform you that the position has been filled. | Lamentamos informarle que la posición ha sido ocupada. | Lamentamos informarle que el puesto ya ha sido cubierto. |
| 7 | German, casual | Can you pick up the kids? I'm stuck at work | Kannst du die Kids abholen? Ich stecke hier im Job fest. | Kannst du die Kinder abholen? Ich bin bei der Arbeit festgefangen. |
| 8 | French, formal | The package arrived broken, I'd like a refund. | Le colis est arrivé endommagé ; je souhaiterais obtenir un remboursement. | Le colis est arrivé endommagé, je souhaiterais un remboursement. |
| 9 | Yoruba, casual | Good morning! Did you sleep well? | Oye! Ṣe o nira? | Gud morning! Ní ba lọ sí? |
| 10 | Spanish, formal | Please find attached the revised contract for your review. | Adjunto encontrará el contrato revisado para su revisión. | Por favor, tenga a su disposición el contrato revisado para su revisión. |

Judgement. Both models honour tone (tu/vous, du/Sie, usted) and keep emoji and punctuation. Hy-MT2 is
idiomatic on 2, 4, 6 and 8 (`el puesto ... cubierto`, keeps `lol`, `je souhaiterais`), slips on 3
(`den Rechnungsstück` is not German), 7 (`festgefangen` is not German for "stuck") and 10 (`tenga a
su disposición` drifts from "please find attached"), and the sampling adds variation between identical
requests (1 gained question marks from the question in its context; a greedy run gave `Nos vemos
mañana`). Qwen3.5-9B is more careful on the formal sentences (3, 10) and more colloquial on 7, at
3.5–4x the latency and 4.7x the size. Neither handles Yoruba (Hy-MT2 now mostly leaves it in English),
so `/translate` routes Yoruba, Igbo, Hausa and Swahili to the general model (`hy_mt2_supports` in
`src/config/prompts.rs`).

Prompt wording matters for Hy-MT2. The first wording of the background block ("Earlier messages from
the same sender ... do not translate them") made it echo a French context line: `yes! see you there,
can't wait` with context `On se voit demain ? J'ai réservé pour 19h` came back as `Oui ! On se voit
demain ? J'ai réservé pour 19h`. A grid over three headings and three task lines (five context cases,
card sampling) found only the short heading `Previous messages in this chat (context only, never
translate or repeat them):` free of echoes; it is now combined with an explicit `Translate only the
[Source Text].` and covered by `hunyuan_prompt_keeps_context_apart_from_source`. That case now gives
`Oui ! À demain, je te vois là-bas, j'attends avec impatience.`

Cold calls. A first request after the server has sat idle took 3.45 s: on this 24 GB machine with the
Qwen3.5-9B server, a second Qwen3.5 copy and a `cargo test` build running, the idle Hy-MT2 process had
been paged out (RSS 31 MB, 8 GB of pageouts), so the call re-faulted the 0.86 GiB of weights from swap.
Warm calls on the same server measured 0.21–0.38 s right after. Keeping the model resident is a matter
of memory headroom on the host; Kronos has no `mlock` option.

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
