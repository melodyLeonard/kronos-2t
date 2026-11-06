# Agent eval: prev-iso

Commit `462dbae` · 2026-11-06T17:06:33 · model Qwen3.5-9B · server peak RSS 3678.6 MiB

Pass rate: 4/10 · median step time 2.68 s · median prompt tokens/step 5859 · total prompt tokens 720055 · follow-up first-token median 1.862 s (max 6.952 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 13.3 | 2.78 | 3633 / 4010 | 660 | 2.351 / 2.351 | 0 |
| 2 | 02_bugfix_js | FAIL | 5 | 18.9 | 4.42 | 3660 / 4869 | 867 | 2.190 / 2.190 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 9 | 100.9 | 12.92 | 5861 / 8135 | 3864 | 2.950 / 2.950 | 0 |
| 4 | 04_rename_refactor | pass | 11 | 31.4 | 1.83 | 4701 / 5859 | 1004 | 6.004 / 6.004 | 0 |
| 5 | 05_extract_module | FAIL | 11 | 161.0 | 14.14 | 8408 / 12123 | 4089 | 3.589 / 3.589 | 1 |
| 6 | 06_csv_report | FAIL | 2 | 29.5 | 14.16 | 3539 / 3883 | 677 | 9.828 / 9.828 | 0 |
| 7 | 07_web_page | FAIL | 2 | 32.1 | 14.56 | 3482 / 3974 | 1017 | 1.566 / 1.566 | 0 |
| 8 | 08_followups | pass | 42 | 100.4 | 1.81 | 6252 / 11960 | 2057 | 1.856 / 6.952 | 1 |
| 9 | 09_build_fix_loop | FAIL | 19 | 85.7 | 3.05 | 8976 / 11414 | 2305 | 4.392 / 4.392 | 0 |
| 10 | 10_long_context | pass | 6 | 28.7 | 2.60 | 4284 / 5041 | 812 | 3.391 / 3.391 | 0 |
