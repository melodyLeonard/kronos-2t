#!/usr/bin/env python3
"""Agent evaluation runner: starts kronos_server, drives each task, scores hidden checks.

Usage:
  python3 -I eval/run_eval.py --label baseline [--tasks 01,02] [--port 18090]
         [--server-bin target/release/kronos_server] [--model DIR] [--draft DIR]
         [--server-url http://127.0.0.1:18090]   # reuse a running server instead

Writes eval/results/<label>.json and eval/results/<label>.md.
"""

import argparse
import http.client
import json
import os
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TASKS_DIR = os.path.join(HERE, "tasks")
RESULTS_DIR = os.path.join(HERE, "results")
MODELS = os.environ.get("KRONOS_MODELS") or next((d for d in (os.path.join(ROOT, "models"), "/Users/melodyleonard/Documents/project/kronos-2t/models") if os.path.isdir(d)), os.path.join(ROOT, "models"))


class RssMonitor(threading.Thread):
    """Samples `ps -o rss=` for a pid; keeps the peak in KiB."""

    def __init__(self, pid, interval=0.5):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak_kib, self._stop = pid, interval, 0, threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["ps", "-o", "rss=", "-p", str(self.pid)], capture_output=True, text=True, timeout=5).stdout.strip()
                if out:
                    self.peak_kib = max(self.peak_kib, int(out))
            except (ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()


class Server:
    def __init__(self, args, workroot):
        self.args, self.workroot, self.proc, self.rss = args, workroot, None, None
        self.url = args.server_url or f"http://127.0.0.1:{args.port}"

    def start(self):
        if self.args.server_url:
            self._wait_health(30)
            return
        cmd = [self.args.server_bin, self.args.model, "--mode", "nf4g128", "--ctx", str(self.args.ctx), "--metal",
               "--draft", self.args.draft, "--port", str(self.args.port), "--workdir", self.workroot,
               "--state-dir", os.path.join(self.workroot, ".state")]
        log = open(os.path.join(self.workroot, "server.log"), "ab")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        self.rss = RssMonitor(self.proc.pid)
        self.rss.start()
        self._wait_health(600)

    def _wait_health(self, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                raise SystemExit(f"server exited early (code {self.proc.returncode}); see server.log")
            try:
                st, body = http_request(self.url, "GET", "/health", None, timeout=5)
                if st == 200:
                    return
            except OSError:
                pass
            time.sleep(1)
        raise SystemExit("server did not become healthy")

    def peak_rss_mib(self):
        return round(self.rss.peak_kib / 1024, 1) if self.rss else None

    def stop(self):
        if self.rss:
            self.rss.stop()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def http_request(base, method, path, body, timeout=60):
    u = urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    data = json.dumps(body).encode() if body is not None else None
    conn.request(method, path, body=data, headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    out = resp.read().decode("utf-8", "replace")
    conn.close()
    return resp.status, out


def sse_agent_turn(base, payload, deadline):
    """POST /v1/agent/chat and collect per-step metrics until done/error/deadline."""
    u = urlparse(base)
    body = json.dumps(payload).encode()
    sock = socket.create_connection((u.hostname, u.port), timeout=30)
    req = (f"POST /v1/agent/chat HTTP/1.1\r\nHost: {u.hostname}:{u.port}\r\nContent-Type: application/json\r\n"
           f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode() + body
    t_send = time.monotonic()
    sock.sendall(req)
    turn = {"status": None, "steps": [], "first_token_s": None, "first_event_s": None, "final_text": "", "messages": None,
            "error": None, "compactions": [], "tool_calls": [], "timed_out": False}
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(65536)
        if not chunk:
            break
        head += chunk
    head, buf = head.split(b"\r\n\r\n", 1) if b"\r\n\r\n" in head else (head, b"")
    status_line = head.split(b"\r\n", 1)[0].decode("utf-8", "replace")
    turn["status"] = int(status_line.split()[1]) if len(status_line.split()) > 1 else None
    if turn["status"] != 200:
        rest = b""
        try:
            while True:
                c = sock.recv(65536)
                if not c:
                    break
                rest += c
        except OSError:
            pass
        turn["error"] = f"HTTP {turn['status']}: {(buf + rest).decode('utf-8', 'replace')[:500]}"
        sock.close()
        return turn
    cur = None
    sock.settimeout(1.0)
    while True:
        if time.monotonic() > deadline:
            turn["timed_out"] = True
            break
        try:
            chunk = sock.recv(65536)
        except (TimeoutError, socket.timeout):
            continue
        except OSError as e:
            turn["error"] = turn["error"] or f"socket: {e}"
            break
        if not chunk:
            break
        buf += chunk
        done = False
        while b"\n\n" in buf:
            frame, buf = buf.split(b"\n\n", 1)
            for line in frame.split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].decode("utf-8", "replace")
                if data == "[DONE]":
                    done = True
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic() - t_send
                kind = ev.get("type")
                if turn["first_event_s"] is None:
                    turn["first_event_s"] = round(now, 3)
                if kind == "step_start":
                    cur = {"step": ev["step"], "t_start": now, "first_token_s": None, "tools": []}
                    turn["steps"].append(cur)
                elif kind in ("text", "tool_progress", "tool_start"):
                    if turn["first_token_s"] is None:
                        turn["first_token_s"] = round(now, 3)
                    if cur is not None and cur["first_token_s"] is None:
                        cur["first_token_s"] = round(now - cur["t_start"], 3)
                    if kind == "tool_start":
                        cur["tools"].append(ev.get("name"))
                        turn["tool_calls"].append(ev.get("name"))
                        cur.setdefault("tool_args", []).append((ev.get("arguments") or "")[:200])
                elif kind == "tool_done" and cur is not None:
                    cur.setdefault("tool_results", []).append({"name": ev.get("name"), "ok": ev.get("ok"), "output": (ev.get("output") or "")[:300]})
                elif kind == "step_stats" and cur is not None:
                    us = ev.get("usage", {})
                    cur.update({"prompt_tokens": us.get("prompt_tokens"), "prefilled_tokens": us.get("prefilled_tokens"),
                                "completion_tokens": us.get("completion_tokens"), "prefill_secs": us.get("prefill_secs"),
                                "decode_secs": us.get("decode_secs"), "step_secs": round(now - cur["t_start"], 3)})
                elif kind == "compacted":
                    turn["compactions"].append({k: ev.get(k) for k in ("before_tokens", "after_tokens", "compacted_messages")})
                elif kind == "done":
                    turn["final_text"] = ev.get("text", "")
                    turn["messages"] = ev.get("messages")
                    turn["context_tokens"] = ev.get("context_tokens")
                elif kind == "error":
                    turn["error"] = ev.get("message")
            if done:
                break
        if done:
            break
    turn["wall_s"] = round(time.monotonic() - t_send, 3)
    for s in turn["steps"]:
        s.pop("t_start", None)
    sock.close()
    return turn


def run_check(task_dir, workdir, check_cmd):
    cmd = list(check_cmd) + [workdir]
    try:
        r = subprocess.run(cmd, cwd=task_dir, capture_output=True, text=True, timeout=300)
        return r.returncode == 0, (r.stdout + r.stderr).strip()[-2000:]
    except subprocess.TimeoutExpired:
        return False, "check timed out"


def prepare_workdir(task_dir, run_root, name):
    wd = os.path.join(run_root, name)
    if os.path.exists(wd):
        shutil.rmtree(wd)
    shutil.copytree(os.path.join(task_dir, "workspace"), wd)
    return os.path.realpath(wd)


def run_agent_task(server, task, run_root):
    wd = prepare_workdir(task["dir"], run_root, task["name"])
    t0 = time.monotonic()
    turn = sse_agent_turn(server.url, {
        "messages": [{"role": "user", "content": task["prompt"]}],
        "workdir": wd, "max_steps": task["max_steps"], "cmd_timeout": task["cmd_timeout"], "temperature": 0.0, "seed": 0,
    }, deadline=t0 + task["timeout_secs"])
    passed, detail = run_check(task["dir"], wd, task["check"])
    return {"turns": [turn], "passed": passed, "check_output": detail, "workdir": wd}


def run_conversation_task(server, task, run_root):
    wd = prepare_workdir(task["dir"], run_root, task["name"])
    with open(os.path.join(task["dir"], "turns.json"), encoding="utf-8") as f:
        prompts = json.load(f)
    history = []
    turns = []
    deadline = time.monotonic() + task["timeout_secs"]
    for i, p in enumerate(prompts, 1):
        history.append({"role": "user", "content": p})
        turn = sse_agent_turn(server.url, {
            "messages": history, "workdir": wd, "max_steps": task["max_steps"], "cmd_timeout": task["cmd_timeout"],
            "temperature": 0.0, "seed": 0,
        }, deadline=deadline)
        turn["turn"] = i
        turns.append(turn)
        print(f"    turn {i:2d}: first_token {turn['first_token_s']}s wall {turn['wall_s']}s steps {len(turn['steps'])} prompt_tok {[s.get('prompt_tokens') for s in turn['steps']]}", flush=True)
        if turn["messages"] is None or turn["error"]:
            # Keep the history consistent even if the server failed this turn.
            history.append({"role": "assistant", "content": turn.get("final_text") or "(no reply)"})
            if time.monotonic() > deadline:
                break
            continue
        history = turn["messages"]
    passed, detail = run_check(task["dir"], wd, task["check"])
    return {"turns": turns, "passed": passed, "check_output": detail, "workdir": wd}


def load_tasks(selected):
    tasks = []
    for name in sorted(os.listdir(TASKS_DIR)):
        d = os.path.join(TASKS_DIR, name)
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "task.json")):
            continue
        if selected and not any(name.startswith(s) for s in selected):
            continue
        with open(os.path.join(d, "task.json"), encoding="utf-8") as f:
            meta = json.load(f)
        prompt = ""
        if os.path.exists(os.path.join(d, "prompt.txt")):
            with open(os.path.join(d, "prompt.txt"), encoding="utf-8") as f:
                prompt = f.read().strip()
        tasks.append({"name": name, "dir": d, "prompt": prompt, **meta})
    return tasks


def summarize(result):
    steps = [s for t in result["turns"] for s in t["steps"] if s.get("prompt_tokens") is not None]
    step_secs = [s["step_secs"] for s in steps if s.get("step_secs") is not None]
    ftl = [t["first_token_s"] for t in result["turns"] if t.get("first_token_s") is not None]
    return {
        "steps": sum(len(t["steps"]) for t in result["turns"]),
        "wall_s": round(sum(t["wall_s"] for t in result["turns"]), 1),
        "prompt_tokens_total": sum(s["prompt_tokens"] for s in steps),
        "completion_tokens_total": sum(s["completion_tokens"] or 0 for s in steps),
        "prompt_tokens_per_step_median": statistics.median([s["prompt_tokens"] for s in steps]) if steps else None,
        "prompt_tokens_max": max((s["prompt_tokens"] for s in steps), default=None),
        "step_secs_median": round(statistics.median(step_secs), 2) if step_secs else None,
        "first_token_s_median": round(statistics.median(ftl), 3) if ftl else None,
        "first_token_s_max": max(ftl) if ftl else None,
        "compactions": sum(len(t["compactions"]) for t in result["turns"]),
        "errors": [t["error"] for t in result["turns"] if t.get("error")],
        "timed_out": any(t.get("timed_out") for t in result["turns"]),
    }


def fmt(v, nd=1):
    if v is None:
        return "-"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def write_markdown(path, run):
    rows = ["| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(run["tasks"], 1):
        s = r["summary"]
        rows.append(f"| {i} | {r['name']} | {'pass' if r['passed'] else 'FAIL'} | {s['steps']} | {fmt(s['wall_s'])} | {fmt(s['step_secs_median'], 2)} | "
                    f"{fmt(s['prompt_tokens_per_step_median'], 0)} / {fmt(s['prompt_tokens_max'])} | {s['completion_tokens_total']} | "
                    f"{fmt(s['first_token_s_median'], 3)} / {fmt(s['first_token_s_max'], 3)} | {s['compactions']} |")
    a = run["aggregate"]
    head = [f"# Agent eval: {run['label']}", "", f"Commit `{run['commit']}` · {run['started_at']} · server peak RSS {fmt(a['peak_rss_mib'])} MiB",
            "", f"Pass rate: {a['passed']}/{a['total']} · median step time {fmt(a['step_secs_median'], 2)} s · "
            f"median prompt tokens/step {fmt(a['prompt_tokens_per_step_median'], 0)} · total prompt tokens {a['prompt_tokens_total']} · "
            f"follow-up first-token median {fmt(a['followup_first_token_s_median'], 3)} s (max {fmt(a['followup_first_token_s_max'], 3)} s) over {a['followup_turns']} turns", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--tasks", default="", help="comma-separated task name prefixes, e.g. 01,02")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--server-bin", default=os.path.join(ROOT, "target", "release", "kronos_server"))
    ap.add_argument("--model", default=os.path.join(MODELS, "Qwen2.5-Coder-14B-Instruct"))
    ap.add_argument("--draft", default=os.path.join(MODELS, "Qwen2.5-Coder-0.5B-Instruct"))
    ap.add_argument("--server-url", default="")
    ap.add_argument("--workroot", default="")
    args = ap.parse_args()

    selected = [s.strip() for s in args.tasks.split(",") if s.strip()]
    tasks = load_tasks(selected)
    if not tasks:
        raise SystemExit("no tasks selected")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    workroot = args.workroot or os.path.join(ROOT, "target", "eval-work", args.label)
    os.makedirs(workroot, exist_ok=True)
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    run = {"label": args.label, "commit": commit, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "tasks": []}

    server = Server(args, workroot)
    print(f"starting server on {server.url} (workroot {workroot})", flush=True)
    server.start()
    try:
        for task in tasks:
            print(f"== {task['name']} ({task['kind']})", flush=True)
            t0 = time.monotonic()
            if task["kind"] == "conversation":
                res = run_conversation_task(server, task, workroot)
            else:
                res = run_agent_task(server, task, workroot)
            res["name"] = task["name"]
            res["kind"] = task["kind"]
            res["summary"] = summarize(res)
            res["summary"]["total_wall_s"] = round(time.monotonic() - t0, 1)
            run["tasks"].append(res)
            s = res["summary"]
            print(f"   {'PASS' if res['passed'] else 'FAIL'} steps={s['steps']} wall={s['wall_s']}s prompt_tok/step={s['prompt_tokens_per_step_median']} "
                  f"first_token={s['first_token_s_median']}s errors={s['errors']}\n   check: {res['check_output'].splitlines()[-1] if res['check_output'] else ''}", flush=True)
            partial = os.path.join(RESULTS_DIR, f"{args.label}.partial.json")
            with open(partial, "w", encoding="utf-8") as f:
                json.dump(run, f, indent=1)
    finally:
        peak = server.peak_rss_mib()
        server.stop()

    all_steps = [s for r in run["tasks"] for t in r["turns"] for s in t["steps"] if s.get("prompt_tokens") is not None]
    conv = [r for r in run["tasks"] if r["kind"] == "conversation"]
    ftl = [t["first_token_s"] for r in conv for t in r["turns"][1:] if t.get("first_token_s") is not None]
    run["aggregate"] = {
        "passed": sum(1 for r in run["tasks"] if r["passed"]), "total": len(run["tasks"]),
        "step_secs_median": round(statistics.median([s["step_secs"] for s in all_steps]), 2) if all_steps else None,
        "prompt_tokens_per_step_median": statistics.median([s["prompt_tokens"] for s in all_steps]) if all_steps else None,
        "prompt_tokens_total": sum(s["prompt_tokens"] for s in all_steps),
        "completion_tokens_total": sum(s["completion_tokens"] or 0 for s in all_steps),
        "followup_turns": len(ftl),
        "followup_first_token_s_median": round(statistics.median(ftl), 3) if ftl else None,
        "followup_first_token_s_max": max(ftl) if ftl else None,
        "peak_rss_mib": peak,
        "wall_s": round(sum(r["summary"]["total_wall_s"] for r in run["tasks"]), 1),
    }
    out_json = os.path.join(RESULTS_DIR, f"{args.label}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1)
    write_markdown(os.path.join(RESULTS_DIR, f"{args.label}.md"), run)
    partial = os.path.join(RESULTS_DIR, f"{args.label}.partial.json")
    if os.path.exists(partial):
        os.remove(partial)
    print(json.dumps(run["aggregate"], indent=1))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
