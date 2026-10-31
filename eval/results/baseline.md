# Agent eval: baseline

Commit `f1a605a` · 2026-10-31T08:34:41 · server peak RSS 8427.9 MiB

Pass rate: 3/10 · median step time 16.66 s · median prompt tokens/step 5746 · total prompt tokens 658463 · follow-up first-token median 11.458 s (max 21.720 s) over 19 turns

| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 01_bugfix_py | FAIL | 5 | 157.9 | 19.31 | 3672 / 5094 | 926 | 13.117 / 13.117 | 0 |
| 2 | 02_bugfix_js | FAIL | 2 | 90.7 | 42.31 | 3371 / 3587 | 544 | 10.801 / 10.801 | 0 |
| 3 | 03_bugfix_ledger | FAIL | 6 | 255.9 | 44.95 | 3700 / 3700 | 1176 | 17.052 / 17.052 | 0 |
| 4 | 04_rename_refactor | FAIL | 5 | 211.5 | 40.32 | 4611 / 5262 | 717 | 27.139 / 27.139 | 0 |
| 5 | 05_extract_module | pass | 14 | 456.5 | 15.97 | 5758 / 7439 | 2689 | 15.791 / 15.791 | 0 |
| 6 | 06_csv_report | FAIL | 3 | 187.5 | 73.59 | 4586 / 4853 | 827 | 24.061 / 24.061 | 0 |
| 7 | 07_web_page | pass | 2 | 135.6 | 65.63 | 2673 / 2931 | 511 | 7.756 / 7.756 | 0 |
| 8 | 08_followups | pass | 47 | 700.3 | 10.92 | 8958 / 15839 | 2077 | 11.366 / 21.720 | 0 |
| 9 | 09_build_fix_loop | FAIL | 4 | 258.0 | 74.72 | 6538 / 10313 | 1987 | 21.125 / 21.125 | 0 |
| 10 | 10_long_context | FAIL | 4 | 165.3 | 11.64 | 8222 / 13479 | 204 | 6.415 / 6.415 | 0 |
