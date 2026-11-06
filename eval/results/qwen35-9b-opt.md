# Agent eval: qwen35-9b-opt

Commit `7bbf1b2` · 2026-11-06T12:33:28 · model Qwen3.5-9B · server peak RSS 3368.3 MiB

Pass rate: 5/10 · median step time 2.79 s · median prompt tokens/step 5242 · total prompt tokens 598420 · follow-up first-token median 1.997 s (max 14.053 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 13.0 | 2.76 | 3637 / 4014 | 660 | 2.307 / 2.307 | 0 |
| 2 | 02_bugfix_js | FAIL | 5 | 21.4 | 5.55 | 3664 / 4873 | 867 | 3.699 / 3.699 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 9 | 101.1 | 8.35 | 5720 / 6493 | 2917 | 4.376 / 4.376 | 0 |
| 4 | 04_rename_refactor | pass | 11 | 33.8 | 1.86 | 4504 / 5662 | 1004 | 9.608 / 9.608 | 0 |
| 5 | 05_extract_module | pass | 5 | 47.0 | 10.97 | 4574 / 6778 | 1379 | 4.421 / 4.421 | 0 |
| 6 | 06_csv_report | FAIL | 2 | 23.5 | 11.04 | 3214 / 3553 | 666 | 3.034 / 3.034 | 0 |
| 7 | 07_web_page | FAIL | 2 | 35.2 | 16.17 | 3178 / 3686 | 1019 | 2.437 / 2.437 | 0 |
| 8 | 08_followups | pass | 43 | 124.0 | 2.22 | 6211 / 12146 | 2119 | 2.029 / 14.053 | 1 |
| 9 | 09_build_fix_loop | pass | 15 | 81.8 | 4.48 | 7847 / 10596 | 2159 | 8.409 / 8.409 | 0 |
| 10 | 10_long_context | FAIL | 6 | 50.6 | 6.03 | 3804 / 4203 | 867 | 10.424 / 10.424 | 0 |
