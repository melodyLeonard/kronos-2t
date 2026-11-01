# Agent eval: nf3-hot-nf4-b

Commit `af722ed` · 2026-11-01T10:42:26 · server peak RSS - MiB

Pass rate: 5/10 · median step time 3.20 s · median prompt tokens/step 6142 · total prompt tokens 596548 · follow-up first-token median 2.902 s (max 7.969 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 15.2 | 3.35 | 3482 / 3824 | 590 | 4.215 / 4.215 | 0 |
| 2 | 02_bugfix_js | FAIL | 3 | 28.0 | 10.20 | 4642 / 5022 | 1161 | 2.458 / 2.458 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 6 | 30.3 | 3.82 | 4614 / 5232 | 1128 | 3.454 / 3.454 | 0 |
| 4 | 04_rename_refactor | pass | 7 | 46.6 | 8.02 | 5542 / 6677 | 2124 | 4.822 / 4.822 | 0 |
| 5 | 05_extract_module | pass | 4 | 26.3 | 7.41 | 4449 / 5510 | 1105 | 3.156 / 3.156 | 0 |
| 6 | 06_csv_report | pass | 2 | 13.9 | 6.86 | 3396 / 3700 | 597 | 2.341 / 2.341 | 0 |
| 7 | 07_web_page | pass | 2 | 17.0 | 7.71 | 3204 / 3518 | 714 | 1.640 / 1.640 | 0 |
| 8 | 08_followups | FAIL | 51 | 150.5 | 2.04 | 7601 / 11852 | 2981 | 2.833 / 7.969 | 1 |
| 9 | 09_build_fix_loop | FAIL | 13 | 77.7 | 5.26 | 7401 / 10239 | 2494 | 5.262 / 5.262 | 0 |
| 10 | 10_long_context | FAIL | 2 | 9.2 | 3.66 | 3468 / 3519 | 185 | 3.945 / 3.945 | 0 |
