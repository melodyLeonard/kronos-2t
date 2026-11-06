# Agent eval: qwen35-9b-a

Commit `aa11e8d` · 2026-11-06T11:09:18 · model Qwen3.5-9B · server peak RSS 2906.8 MiB

Pass rate: 7/10 · median step time 2.38 s · median prompt tokens/step 5602 · total prompt tokens 714621 · follow-up first-token median 1.776 s (max 12.601 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 13.1 | 2.75 | 3637 / 4014 | 660 | 2.369 / 2.369 | 0 |
| 2 | 02_bugfix_js | FAIL | 5 | 25.0 | 4.73 | 3664 / 4873 | 867 | 8.627 / 8.627 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 12 | 134.0 | 12.12 | 6684 / 8738 | 4315 | 9.477 / 9.477 | 0 |
| 4 | 04_rename_refactor | pass | 11 | 33.7 | 1.70 | 4504 / 5662 | 1004 | 12.346 / 12.346 | 0 |
| 5 | 05_extract_module | pass | 5 | 47.2 | 9.95 | 4574 / 6778 | 1379 | 9.679 / 9.679 | 0 |
| 6 | 06_csv_report | FAIL | 2 | 25.5 | 12.20 | 3215 / 3555 | 668 | 8.187 / 8.187 | 0 |
| 7 | 07_web_page | pass | 2 | 34.5 | 16.02 | 3144 / 3620 | 964 | 7.610 / 7.610 | 0 |
| 8 | 08_followups | pass | 43 | 110.6 | 1.95 | 6211 / 12146 | 2113 | 1.824 / 12.601 | 1 |
| 9 | 09_build_fix_loop | pass | 21 | 81.7 | 2.86 | 8675 / 11499 | 2585 | 10.646 / 10.646 | 0 |
| 10 | 10_long_context | pass | 7 | 130.6 | 10.64 | 3747 / 13930 | 566 | 9.489 / 9.489 | 2 |
