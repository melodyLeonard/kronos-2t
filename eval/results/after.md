# Agent eval: after

Commit `be6a2d6` · 2026-10-31T09:19:41 · server peak RSS 6698.2 MiB

Pass rate: 2/10 · median step time 6.71 s · median prompt tokens/step 4825 · total prompt tokens 752015 · follow-up first-token median 5.177 s (max 51.062 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | FAIL | 9 | 135.2 | 9.93 | 5571 / 7699 | 1181 | 5.492 / 5.492 | 0 |
| 2 | 02_bugfix_js | FAIL | 2 | 44.2 | 21.30 | 3068 / 3438 | 599 | 4.485 / 4.485 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 2 | 50.7 | 24.46 | 3434 / 3906 | 449 | 7.639 / 7.639 | 0 |
| 4 | 04_rename_refactor | FAIL | 2 | 38.7 | 19.32 | 3886 / 4469 | 289 | 12.902 / 12.902 | 0 |
| 5 | 05_extract_module | pass | 9 | 816.3 | 8.82 | 6018 / 7537 | 2539 | 6.434 / 6.434 | 0 |
| 6 | 06_csv_report | FAIL | 2 | 35.6 | 16.79 | 2901 / 3141 | 500 | 4.881 / 4.881 | 0 |
| 7 | 07_web_page | pass | 2 | 31.4 | 14.27 | 2710 / 2961 | 495 | 3.429 / 3.429 | 0 |
| 8 | 08_followups | FAIL | 103 | 953.2 | 5.76 | 4951 / 8094 | 8084 | 5.073 / 51.062 | 6 |
| 9 | 09_build_fix_loop | FAIL | 9 | 405.7 | 29.40 | 5888 / 7914 | 4261 | 9.254 / 9.254 | 2 |
| 10 | 10_long_context | FAIL | 12 | 63.3 | 3.39 | 4126 / 4880 | 616 | 7.598 / 7.598 | 1 |
