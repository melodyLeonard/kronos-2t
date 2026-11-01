# Agent eval: refs-a

Commit `ef82056` · 2026-11-01T12:27:37 · server peak RSS - MiB

Pass rate: 8/10 · median step time 2.32 s · median prompt tokens/step 6260 · total prompt tokens 828132 · follow-up first-token median 2.426 s (max 2.850 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 1 | 13.4 | 13.05 | 3184 / 3184 | 593 | 3.124 / 3.124 | 0 |
| 2 | 02_bugfix_js | pass | 6 | 81.4 | 10.52 | 6966 / 9027 | 3361 | 2.475 / 2.475 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 10 | 38.4 | 3.35 | 5042 / 6078 | 1482 | 3.456 / 3.456 | 0 |
| 4 | 04_rename_refactor | pass | 8 | 65.7 | 8.43 | 5787 / 7855 | 2999 | 4.748 / 4.748 | 0 |
| 5 | 05_extract_module | pass | 12 | 72.9 | 5.74 | 8402 / 10609 | 2456 | 3.134 / 3.134 | 0 |
| 6 | 06_csv_report | pass | 2 | 14.2 | 6.94 | 3392 / 3699 | 604 | 2.323 / 2.323 | 0 |
| 7 | 07_web_page | pass | 3 | 20.8 | 3.02 | 3574 / 3742 | 936 | 1.624 / 1.624 | 0 |
| 8 | 08_followups | pass | 68 | 137.3 | 1.33 | 6850 / 12267 | 2982 | 2.407 / 2.850 | 1 |
| 9 | 09_build_fix_loop | FAIL | 10 | 52.0 | 4.52 | 6818 / 8538 | 1670 | 4.833 / 4.833 | 0 |
| 10 | 10_long_context | pass | 6 | 120.9 | 3.36 | 3698 / 14168 | 644 | 3.482 / 3.482 | 2 |
