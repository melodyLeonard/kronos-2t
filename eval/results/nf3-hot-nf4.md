# Agent eval: nf3-hot-nf4

Commit `af722ed` · 2026-11-01T10:34:46 · server peak RSS - MiB

Pass rate: 6/10 · median step time 2.71 s · median prompt tokens/step 6022 · total prompt tokens 738084 · follow-up first-token median 2.693 s (max 9.083 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 1 | 13.1 | 12.92 | 3184 / 3184 | 593 | 3.026 / 3.026 | 0 |
| 2 | 02_bugfix_js | FAIL | 3 | 28.2 | 10.30 | 4659 / 5039 | 1183 | 2.477 / 2.477 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 10 | 38.6 | 3.26 | 5042 / 6026 | 1497 | 3.425 / 3.425 | 0 |
| 4 | 04_rename_refactor | pass | 7 | 46.5 | 8.09 | 5512 / 6647 | 2114 | 4.729 / 4.729 | 0 |
| 5 | 05_extract_module | pass | 7 | 29.9 | 5.11 | 5263 / 5825 | 1247 | 3.125 / 3.125 | 0 |
| 6 | 06_csv_report | pass | 2 | 14.1 | 6.95 | 3392 / 3699 | 604 | 2.315 / 2.315 | 0 |
| 7 | 07_web_page | pass | 3 | 20.5 | 3.01 | 3574 / 3742 | 936 | 1.606 / 1.606 | 0 |
| 8 | 08_followups | pass | 69 | 172.0 | 2.07 | 7462 / 11963 | 4195 | 2.632 / 9.083 | 1 |
| 9 | 09_build_fix_loop | FAIL | 10 | 52.0 | 4.55 | 6818 / 8538 | 1670 | 4.823 / 4.823 | 0 |
| 10 | 10_long_context | FAIL | 2 | 8.4 | 3.37 | 3468 / 3519 | 185 | 3.516 / 3.516 | 0 |
