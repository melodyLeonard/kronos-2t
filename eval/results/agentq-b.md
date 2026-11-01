# Agent eval: agentq-b

Commit `e57653b` · 2026-11-01T11:08:12 · server peak RSS - MiB

Pass rate: 8/10 · median step time 2.38 s · median prompt tokens/step 6460 · total prompt tokens 886028 · follow-up first-token median 2.735 s (max 9.525 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 15.6 | 3.95 | 3482 / 3824 | 590 | 2.981 / 2.981 | 0 |
| 2 | 02_bugfix_js | pass | 6 | 66.5 | 11.74 | 6420 / 8036 | 2334 | 2.812 / 2.812 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 6 | 31.7 | 4.24 | 4614 / 5232 | 1047 | 3.850 / 3.850 | 0 |
| 4 | 04_rename_refactor | pass | 8 | 72.2 | 9.34 | 5795 / 7765 | 2947 | 5.542 / 5.542 | 0 |
| 5 | 05_extract_module | pass | 13 | 83.7 | 6.33 | 8502 / 10775 | 2511 | 3.525 / 3.525 | 0 |
| 6 | 06_csv_report | pass | 2 | 15.2 | 7.49 | 3396 / 3700 | 597 | 2.575 / 2.575 | 0 |
| 7 | 07_web_page | pass | 3 | 22.8 | 3.34 | 3574 / 3742 | 936 | 1.848 / 1.848 | 0 |
| 8 | 08_followups | pass | 75 | 161.0 | 1.33 | 7014 / 11849 | 3123 | 2.712 / 9.525 | 1 |
| 9 | 09_build_fix_loop | FAIL | 11 | 71.4 | 5.72 | 7065 / 9239 | 2244 | 5.579 / 5.579 | 0 |
| 10 | 10_long_context | pass | 6 | 132.7 | 3.74 | 3703 / 14173 | 601 | 3.993 / 3.993 | 2 |
