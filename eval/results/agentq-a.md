# Agent eval: agentq-a

Commit `e57653b` · 2026-11-01T10:56:46 · server peak RSS - MiB

Pass rate: 8/10 · median step time 2.99 s · median prompt tokens/step 6144 · total prompt tokens 876477 · follow-up first-token median 2.597 s (max 9.818 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 1 | 12.9 | 12.60 | 3184 / 3184 | 593 | 2.824 / 2.824 | 0 |
| 2 | 02_bugfix_js | pass | 6 | 80.4 | 10.37 | 6966 / 9027 | 3361 | 2.445 / 2.445 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 10 | 38.5 | 3.25 | 5042 / 6026 | 1497 | 3.413 / 3.413 | 0 |
| 4 | 04_rename_refactor | pass | 7 | 53.2 | 8.96 | 5523 / 6658 | 2124 | 4.998 / 4.998 | 0 |
| 5 | 05_extract_module | pass | 12 | 87.3 | 7.10 | 8402 / 10514 | 2456 | 4.064 / 4.064 | 0 |
| 6 | 06_csv_report | pass | 2 | 15.8 | 7.79 | 3392 / 3699 | 604 | 2.673 / 2.673 | 0 |
| 7 | 07_web_page | pass | 12 | 55.3 | 3.66 | 4883 / 6606 | 3252 | 1.812 / 1.812 | 0 |
| 8 | 08_followups | pass | 69 | 151.9 | 1.48 | 6838 / 11834 | 3093 | 2.579 / 9.818 | 1 |
| 9 | 09_build_fix_loop | FAIL | 10 | 57.8 | 5.11 | 6818 / 8538 | 1670 | 5.533 / 5.533 | 0 |
| 10 | 10_long_context | pass | 6 | 132.8 | 3.74 | 3698 / 14168 | 644 | 4.038 / 4.038 | 2 |
