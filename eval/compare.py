#!/usr/bin/env python3
"""Side-by-side markdown table for two eval runs: python3 -I eval/compare.py baseline after > eval/results/before_after.md"""

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def load(label):
    with open(os.path.join(HERE, "results", f"{label}.json"), encoding="utf-8") as f:
        return json.load(f)


def fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def followup(run):
    conv = [r for r in run["tasks"] if r["kind"] == "conversation"]
    turns = [t for r in conv for t in r["turns"]]
    ftl = [t["first_token_s"] for t in turns[1:] if t.get("first_token_s") is not None]
    first_prompt = [t["steps"][0]["prompt_tokens"] for t in turns if t["steps"] and t["steps"][0].get("prompt_tokens")]
    return {
        "turns": len(turns),
        "ftl_median": round(statistics.median(ftl), 2) if ftl else None,
        "ftl_p90": round(sorted(ftl)[int(0.9 * (len(ftl) - 1))], 2) if ftl else None,
        "ftl_max": max(ftl) if ftl else None,
        "prompt_last": first_prompt[-1] if first_prompt else None,
        "steps": sum(len(t["steps"]) for t in turns),
        "errors": sum(1 for t in turns if t.get("error")),
    }


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    fa, fb = followup(a), followup(b)
    aa, ab = a["aggregate"], b["aggregate"]
    rows = [
        ("pass rate", f"{aa['passed']}/{aa['total']}", f"{ab['passed']}/{ab['total']}"),
        ("median step time (s)", fmt(aa["step_secs_median"], 2), fmt(ab["step_secs_median"], 2)),
        ("median prompt tokens / step", fmt(aa["prompt_tokens_per_step_median"], 0), fmt(ab["prompt_tokens_per_step_median"], 0)),
        ("total prompt tokens", aa["prompt_tokens_total"], ab["prompt_tokens_total"]),
        ("total completion tokens", aa["completion_tokens_total"], ab["completion_tokens_total"]),
        ("total agent steps", sum(r["summary"]["steps"] for r in a["tasks"]), sum(r["summary"]["steps"] for r in b["tasks"])),
        ("total wall (s)", fmt(aa["wall_s"]), fmt(ab["wall_s"])),
        ("peak server RSS (MiB)", fmt(aa["peak_rss_mib"]), fmt(ab["peak_rss_mib"])),
        (f"follow-up first token, median (s)", fmt(fa["ftl_median"], 2), fmt(fb["ftl_median"], 2)),
        (f"follow-up first token, p90 / max (s)", f"{fmt(fa['ftl_p90'], 2)} / {fmt(fa['ftl_max'], 2)}", f"{fmt(fb['ftl_p90'], 2)} / {fmt(fb['ftl_max'], 2)}"),
        ("follow-up prompt tokens at turn 20 (step 1)", fa["prompt_last"], fb["prompt_last"]),
        ("follow-up steps over 20 turns / turns ending in error", f"{fa['steps']} / {fa['errors']}", f"{fb['steps']} / {fb['errors']}"),
    ]
    print(f"# {a['label']} ({a['commit']}) vs {b['label']} ({b['commit']})\n")
    print(f"| metric | {a['label']} | {b['label']} |\n|---|---:|---:|")
    for name, x, y in rows:
        print(f"| {name} | {x} | {y} |")
    print("\n| # | task | " + f"{a['label']} | {b['label']} | " + f"steps {a['label']} / {b['label']} | prompt tok/step {a['label']} / {b['label']} |")
    print("|---|---|---|---|---:|---:|")
    bt = {r["name"]: r for r in b["tasks"]}
    for i, r in enumerate(a["tasks"], 1):
        o = bt.get(r["name"])
        sa, sb = r["summary"], (o or {}).get("summary", {})
        print(f"| {i} | {r['name']} | {'pass' if r['passed'] else 'FAIL'} | {('pass' if o['passed'] else 'FAIL') if o else '-'} | "
              f"{sa['steps']} / {sb.get('steps', '-')} | {fmt(sa['prompt_tokens_per_step_median'], 0)} / {fmt(sb.get('prompt_tokens_per_step_median'), 0)} |")


if __name__ == "__main__":
    main()
