# Agent eval: refs-b

Commit `ef82056` · 2026-11-01T12:37:54 · server peak RSS - MiB

Pass rate: 8/10 · median step time 2.52 s · median prompt tokens/step 6637 · total prompt tokens 871885 · follow-up first-token median 2.635 s (max 9.642 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | pass | 4 | 14.2 | 3.49 | 3482 / 3875 | 590 | 2.759 / 2.759 | 0 |
| 2 | 02_bugfix_js | pass | 6 | 62.7 | 11.23 | 6420 / 8036 | 2334 | 2.643 / 2.643 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 6 | 31.5 | 4.11 | 4639 / 5282 | 1100 | 3.642 / 3.642 | 0 |
| 4 | 04_rename_refactor | pass | 8 | 68.1 | 8.89 | 5795 / 7823 | 2947 | 5.330 / 5.330 | 0 |
| 5 | 05_extract_module | pass | 12 | 77.3 | 6.36 | 8415 / 10622 | 2456 | 3.366 / 3.366 | 0 |
| 6 | 06_csv_report | pass | 2 | 14.6 | 7.18 | 3396 / 3700 | 597 | 2.492 / 2.492 | 0 |
| 7 | 07_web_page | pass | 4 | 21.4 | 2.54 | 3566 / 3655 | 836 | 1.779 / 1.779 | 0 |
| 8 | 08_followups | pass | 71 | 163.4 | 1.45 | 7278 / 11990 | 3341 | 2.618 / 9.642 | 1 |
| 9 | 09_build_fix_loop | FAIL | 12 | 72.9 | 5.32 | 7106 / 9782 | 2218 | 5.428 / 5.428 | 0 |
| 10 | 10_long_context | pass | 6 | 132.0 | 3.71 | 3703 / 14173 | 601 | 4.012 / 4.012 | 2 |
