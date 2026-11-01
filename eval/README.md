# Agent evaluation suite

Ten coding tasks run through the live `kronos_server` agent loop (`POST /v1/agent/chat`,
the same loop the workspace UI uses). The model sees only `workspace/` and `prompt.txt`;
the checks in `hidden/` never enter the workspace.

| # | task | kind | hidden check |
|---|---|---|---|
| 1 | `01_bugfix_py` | 3 planted bugs in an 80-line Python module | unit assertions |
| 2 | `02_bugfix_js` | 3 planted bugs in a 70-line CommonJS module | node assertions |
| 3 | `03_bugfix_ledger` | 3 planted bugs in a 90-line Python class | unit assertions |
| 4 | `04_rename_refactor` | rename a function across 4 files | import + grep + run |
| 5 | `05_extract_module` | extract 4 functions into a new module | AST + behavior |
| 6 | `06_csv_report` | CSV to exact text report | exact output on 2 CSVs |
| 7 | `07_web_page` | single-file HTML/JS to-do page | headless DOM shim in node |
| 8 | `08_followups` | 20 follow-up turns, one small edit each | final file contents |
| 9 | `09_build_fix_loop` | 5 failing unit tests, 30-minute budget | tests + hidden extras + test file hash |
| 10 | `10_long_context` | fix one function in a 1,510-line file | behavior + rest of file unchanged |

Run (starts and stops the server itself, ports 18090-18099):

```bash
python3 -I eval/run_eval.py --label baseline
```

Per task the runner records pass/fail, wall time, steps, prompt and completion tokens per step
(from `step_stats`), first-token latency per turn (request send to first `text`/`tool_start`
event), compactions, and the server's peak RSS from `ps -o rss`. Output: `results/<label>.md`
(tracked) and `results/<label>.json` with the full transcripts (ignored by git). `--tasks 01,08`
selects tasks; `--server-url http://127.0.0.1:18085` reuses a running server.
