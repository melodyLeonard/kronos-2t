# Agent eval: new-iso

Commit `462dbae` · 2026-11-06T17:16:53 · model Qwen3.5-9B · server peak RSS 6453.5 MiB

Pass rate: 4/10 · median step time 2.97 s · median prompt tokens/step 6192 · total prompt tokens 912130 · follow-up first-token median 1.868 s (max 6.972 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 16.9 | 3.80 | 3632 / 4009 | 660 | 2.827 / 2.827 | 0 |
| 2 | 02_bugfix_js | FAIL | 5 | 23.9 | 5.34 | 3659 / 4868 | 867 | 2.587 / 2.587 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 9 | 79.7 | 7.96 | 5823 / 6963 | 2346 | 3.455 / 3.455 | 0 |
| 4 | 04_rename_refactor | pass | 11 | 28.4 | 1.51 | 4700 / 5858 | 1004 | 5.198 / 5.198 | 0 |
| 5 | 05_extract_module | FAIL | 14 | 198.7 | 14.59 | 8524 / 12274 | 5046 | 2.929 / 2.929 | 1 |
| 6 | 06_csv_report | FAIL | 2 | 23.8 | 11.24 | 3538 / 3882 | 677 | 3.122 / 3.122 | 0 |
| 7 | 07_web_page | FAIL | 2 | 32.9 | 14.95 | 3480 / 3973 | 1017 | 1.627 / 1.627 | 0 |
| 8 | 08_followups | pass | 42 | 102.8 | 1.83 | 6247 / 11943 | 2057 | 1.844 / 6.972 | 1 |
| 9 | 09_build_fix_loop | FAIL | 40 | 146.7 | 3.21 | 8107 / 12245 | 4612 | 4.511 / 4.511 | 2 |
| 10 | 10_long_context | pass | 6 | 31.3 | 2.73 | 4283 / 5040 | 812 | 3.745 / 3.745 | 0 |
