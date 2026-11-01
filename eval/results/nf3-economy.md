# Agent eval: nf3-economy

Commit `24b7d5a` · 2026-11-01T08:41:04 · server peak RSS - MiB

Pass rate: 6/10 · median step time 3.37 s · median prompt tokens/step 5825 · total prompt tokens 669589 · follow-up first-token median 2.642 s (max 10.185 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 1 | 17.7 | 12.55 | 3184 / 3184 | 593 | 2.917 / 2.917 | 0 |
| 2 | 02_bugfix_js | FAIL | 3 | 37.6 | 10.86 | 4637 / 5049 | 1158 | 6.716 / 6.716 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 2 | 26.4 | 10.92 | 3752 / 4120 | 852 | 6.504 / 6.504 | 0 |
| 4 | 04_rename_refactor | pass | 8 | 66.0 | 7.64 | 5668 / 7334 | 2489 | 8.248 / 8.248 | 0 |
| 5 | 05_extract_module | pass | 7 | 43.8 | 5.17 | 5263 / 5825 | 1247 | 7.895 / 7.895 | 0 |
| 6 | 06_csv_report | pass | 2 | 24.4 | 9.72 | 3390 / 3695 | 600 | 5.804 / 5.804 | 0 |
| 7 | 07_web_page | pass | 2 | 28.5 | 11.68 | 3211 / 3540 | 767 | 4.882 / 4.882 | 0 |
| 8 | 08_followups | pass | 50 | 234.0 | 1.85 | 6512 / 11931 | 2159 | 2.623 / 10.185 | 1 |
| 9 | 09_build_fix_loop | FAIL | 18 | 152.0 | 4.97 | 8110 / 11847 | 3553 | 8.979 / 8.979 | 0 |
| 10 | 10_long_context | FAIL | 12 | 78.5 | 3.02 | 5098 / 6572 | 1291 | 5.777 / 5.777 | 0 |
