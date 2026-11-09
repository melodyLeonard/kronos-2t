# Agent eval: qwen35-9b-ckpt

Commit `d107774` · 2026-11-06T11:34:18 · model Qwen3.5-9B · server peak RSS 6419.6 MiB

Pass rate: 7/10 · median step time 2.73 s · median prompt tokens/step 5658 · total prompt tokens 719588 · follow-up first-token median 1.985 s (max 13.640 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 18.5 | 3.88 | 3638 / 4015 | 660 | 3.345 / 3.345 | 0 |
| 2 | 02_bugfix_js | FAIL | 5 | 22.3 | 5.77 | 3665 / 4874 | 867 | 4.197 / 4.197 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 12 | 108.3 | 7.93 | 5988 / 7625 | 3260 | 4.633 / 4.633 | 0 |
| 4 | 04_rename_refactor | pass | 11 | 33.7 | 1.65 | 4505 / 5663 | 1004 | 11.827 / 11.827 | 0 |
| 5 | 05_extract_module | pass | 5 | 49.5 | 11.58 | 4575 / 6779 | 1379 | 4.209 / 4.209 | 0 |
| 6 | 06_csv_report | FAIL | 2 | 24.9 | 11.64 | 3215 / 3554 | 666 | 3.566 / 3.566 | 0 |
| 7 | 07_web_page | pass | 2 | 35.0 | 15.79 | 3146 / 3622 | 965 | 2.497 / 2.497 | 0 |
| 8 | 08_followups | pass | 43 | 146.1 | 2.09 | 6219 / 12164 | 2119 | 2.025 / 13.640 | 1 |
| 9 | 09_build_fix_loop | pass | 22 | 96.5 | 2.93 | 8899 / 11975 | 2878 | 8.617 / 8.617 | 0 |
| 10 | 10_long_context | pass | 7 | 134.6 | 9.19 | 3748 / 13931 | 566 | 10.687 / 10.687 | 2 |
