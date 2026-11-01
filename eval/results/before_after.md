# baseline (f1a605a) vs after (be6a2d6)

| metric | baseline | after |
|---|---:|---:|
| pass rate | 3/10 | 2/10 |
| median step time (s) | 16.66 | 6.71 |
| median prompt tokens / step | 5746 | 4825 |
| total prompt tokens | 658463 | 752015 |
| total completion tokens | 11658 | 19013 |
| total agent steps | 92 | 152 |
| total wall (s) | 2619.9 | 2575.1 |
| peak server RSS (MiB) | 8427.9 | 6698.2 |
| follow-up first token, median (s) | 11.46 | 5.18 |
| follow-up first token, p90 / max (s) | 14.65 / 21.72 | 49.90 / 51.06 |
| follow-up prompt tokens at turn 20 (step 1) | 14937 | 4808 |
| follow-up steps over 20 turns / turns ending in error | 47 / 0 | 103 / 13 |

| # | task | baseline | after | steps baseline / after | prompt tok/step baseline / after |
|---|---|---|---|---:|---:|
| 1 | 01_bugfix_py | FAIL | FAIL | 5 / 9 | 3672 / 5571 |
| 2 | 02_bugfix_js | FAIL | FAIL | 2 / 2 | 3371 / 3068 |
| 3 | 03_bugfix_ledger | FAIL | FAIL | 6 / 2 | 3700 / 3434 |
| 4 | 04_rename_refactor | FAIL | FAIL | 5 / 2 | 4611 / 3886 |
| 5 | 05_extract_module | pass | pass | 14 / 9 | 5758 / 6018 |
| 6 | 06_csv_report | FAIL | FAIL | 3 / 2 | 4586 / 2901 |
| 7 | 07_web_page | pass | pass | 2 / 2 | 2673 / 2710 |
| 8 | 08_followups | pass | FAIL | 47 / 103 | 8958 / 4951 |
| 9 | 09_build_fix_loop | FAIL | FAIL | 4 / 9 | 6538 / 5888 |
| 10 | 10_long_context | FAIL | FAIL | 4 / 12 | 8222 / 4126 |
