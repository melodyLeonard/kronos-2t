//! Agent evaluation runner: starts `kronos_server`, drives each task in `eval/tasks` through
//! `POST /v1/agent/chat`, and scores the hidden checks.
//!
//! Usage:
//!   `kronos_eval run --label L [--tasks 01,02] [--port 18090] [--ctx 32768] [--model DIR]
//!                    [--mode nf4g128] [--draft DIR] [--server-bin PATH] [--server-url URL] [--workroot DIR]`
//!   `kronos_eval compare A B`  side-by-side markdown for two runs
//!
//! Writes `eval/results/<label>.json` (full transcripts) and `eval/results/<label>.md`.

use kronos_2t::util::json::Json;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ROOT: &str = env!("CARGO_MANIFEST_DIR");

fn eval_dir() -> PathBuf {
    Path::new(ROOT).join("eval")
}

fn models_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("KRONOS_MODELS") {
        return PathBuf::from(d);
    }
    // Worktrees under `target/` share the main checkout's `models/`.
    let mut p = Path::new(ROOT).to_path_buf();
    loop {
        if p.join("models").is_dir() {
            return p.join("models");
        }
        if !p.pop() {
            return Path::new(ROOT).join("models");
        }
    }
}

struct Args {
    cmd: String,
    pos: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse() -> Self {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_else(|| usage());
        let (mut pos, mut flags) = (Vec::new(), Vec::new());
        let rest: Vec<String> = it.collect();
        let mut i = 0;
        while i < rest.len() {
            if let Some(k) = rest[i].strip_prefix("--") {
                let v = rest.get(i + 1).cloned().unwrap_or_default();
                flags.push((k.to_string(), v));
                i += 2;
            } else {
                pos.push(rest[i].clone());
                i += 1;
            }
        }
        Self { cmd, pos, flags }
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.flags.iter().find(|f| f.0 == k).map(|f| f.1.as_str())
    }
}

fn usage() -> ! {
    eprintln!("usage: kronos_eval run --label L [flags] | kronos_eval compare A B");
    std::process::exit(2)
}

// ---------- JSON helpers ----------

fn obj(kv: Vec<(&str, Json)>) -> Json {
    Json::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn num(v: f64) -> Json {
    Json::Num(v)
}

fn opt(v: Option<f64>) -> Json {
    v.map_or(Json::Null, Json::Num)
}

fn s(v: &str) -> Json {
    Json::Str(v.to_string())
}

fn f(j: &Json, k: &str) -> Option<f64> {
    j.get(k).and_then(Json::as_f64)
}

fn round(v: f64, nd: i32) -> f64 {
    let m = 10f64.powi(nd);
    (v * m).round() / m
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    Some(if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 })
}

/// `-` for missing, else `nd` decimals.
fn fmt(v: Option<f64>, nd: usize) -> String {
    v.map_or_else(|| "-".to_string(), |x| format!("{x:.nd$}"))
}

// ---------- server ----------

struct Server {
    url_host: String,
    port: u16,
    child: Option<Child>,
    peak_rss_kib: Arc<AtomicU64>,
    stop_rss: Arc<AtomicBool>,
}

impl Server {
    fn start(a: &Args, workroot: &Path) -> Self {
        let (host, port) = match a.get("server-url") {
            Some(u) => {
                let hp = u.trim_start_matches("http://").trim_end_matches('/');
                let (h, p) = hp.split_once(':').unwrap_or((hp, "80"));
                (h.to_string(), p.parse().expect("--server-url port"))
            }
            None => ("127.0.0.1".to_string(), a.get("port").map_or(18090, |p| p.parse().expect("--port"))),
        };
        let mut srv = Server {
            url_host: host,
            port,
            child: None,
            peak_rss_kib: Arc::new(AtomicU64::new(0)),
            stop_rss: Arc::new(AtomicBool::new(false)),
        };
        if a.get("server-url").is_some() {
            srv.wait_health(30);
            return srv;
        }
        let bin = a.get("server-bin").map_or_else(|| Path::new(ROOT).join("target/release/kronos_server"), PathBuf::from);
        let model = a.get("model").map_or_else(|| models_dir().join("Qwen3.5-9B"), PathBuf::from);
        let mut cmd = Command::new(&bin);
        cmd.arg(&model)
            .args(["--mode", a.get("mode").unwrap_or("nf4g128")])
            .args(["--ctx", a.get("ctx").unwrap_or("32768")])
            .arg("--metal")
            .args(["--port", &port.to_string()])
            .arg("--workdir")
            .arg(workroot)
            .arg("--state-dir")
            .arg(workroot.join(".state"));
        if let Some(d) = a.get("draft") {
            cmd.args(["--draft", d]);
        }
        let log = std::fs::OpenOptions::new().create(true).append(true).open(workroot.join("server.log")).expect("server.log");
        cmd.stdout(Stdio::from(log.try_clone().expect("log"))).stderr(Stdio::from(log));
        let child = cmd.spawn().unwrap_or_else(|e| panic!("{}: {e}", bin.display()));
        let pid = child.id();
        let (peak, stop) = (srv.peak_rss_kib.clone(), srv.stop_rss.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                if let Ok(out) = Command::new("ps").args(["-o", "rss=", "-p", &pid.to_string()]).output() {
                    if let Ok(kib) = String::from_utf8_lossy(&out.stdout).trim().parse::<u64>() {
                        peak.fetch_max(kib, Ordering::Relaxed);
                    }
                }
                std::thread::sleep(Duration::from_millis(500));
            }
        });
        srv.child = Some(child);
        srv.wait_health(600);
        srv
    }

    fn url(&self) -> String {
        format!("http://{}:{}", self.url_host, self.port)
    }

    fn wait_health(&mut self, timeout_s: u64) {
        let t0 = Instant::now();
        while t0.elapsed().as_secs() < timeout_s {
            if let Some(c) = self.child.as_mut() {
                if let Ok(Some(st)) = c.try_wait() {
                    eprintln!("server exited early ({st}); see server.log");
                    std::process::exit(1);
                }
            }
            if http_get(&self.url_host, self.port, "/health").is_some_and(|(st, _)| st == 200) {
                return;
            }
            std::thread::sleep(Duration::from_secs(1));
        }
        eprintln!("server did not become healthy");
        std::process::exit(1);
    }

    fn peak_rss_mib(&self) -> Option<f64> {
        self.child.as_ref().map(|_| round(self.peak_rss_kib.load(Ordering::Relaxed) as f64 / 1024.0, 1))
    }

    fn stop(&mut self) {
        self.stop_rss.store(true, Ordering::Relaxed);
        if let Some(c) = self.child.as_mut() {
            // SAFETY: plain `kill(2)` on our own child's pid.
            unsafe { libc_kill(c.id() as i32, 15) };
            let t0 = Instant::now();
            while t0.elapsed() < Duration::from_secs(20) {
                if matches!(c.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
            let _ = c.kill();
        }
    }
}

unsafe extern "C" {
    #[link_name = "kill"]
    fn libc_kill(pid: i32, sig: i32) -> i32;
}

fn http_get(host: &str, port: u16, path: &str) -> Option<(u16, String)> {
    let mut s = TcpStream::connect((host, port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok()?;
    write!(s, "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n").ok()?;
    let mut out = String::new();
    s.read_to_string(&mut out).ok()?;
    let status = out.split_whitespace().nth(1)?.parse().ok()?;
    Some((status, out.split_once("\r\n\r\n").map_or(String::new(), |x| x.1.to_string())))
}

// ---------- agent turns ----------

/// One `POST /v1/agent/chat` turn with per-step metrics, as JSON in the runner's result format.
fn sse_agent_turn(srv: &Server, payload: &Json, deadline: Instant) -> Json {
    let body = payload.stringify();
    let t_send = Instant::now();
    let since = |t: Instant| round(t.duration_since(t_send).as_secs_f64(), 3);
    let mut steps: Vec<Vec<(String, Json)>> = Vec::new();
    let mut step_t0: Vec<Instant> = Vec::new();
    let (mut first_token, mut first_event) = (None, None);
    let (mut final_text, mut messages, mut error, mut context_tokens) = (String::new(), Json::Null, None::<String>, Json::Null);
    let mut compactions = Vec::new();
    let mut tool_calls = Vec::new();
    let mut timed_out = false;
    let mut status = None;

    let mut sock = match TcpStream::connect((srv.url_host.as_str(), srv.port)) {
        Ok(s) => s,
        Err(e) => return obj(vec![("error", s(&format!("connect: {e}"))), ("steps", Json::Arr(vec![])), ("wall_s", num(0.0))]),
    };
    let req = format!(
        "POST /v1/agent/chat HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        srv.url_host,
        srv.port,
        body.len()
    );
    let _ = sock.write_all(req.as_bytes());
    let _ = sock.set_read_timeout(Some(Duration::from_secs(1)));
    let mut buf: Vec<u8> = Vec::new();
    let mut header_done = false;
    let mut chunk = vec![0u8; 65536];
    'read: loop {
        if Instant::now() > deadline {
            timed_out = true;
            break;
        }
        let n = match sock.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => continue,
            Err(e) => {
                error.get_or_insert(format!("socket: {e}"));
                break;
            }
        };
        buf.extend_from_slice(&chunk[..n]);
        if !header_done {
            let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") else { continue };
            let head = String::from_utf8_lossy(&buf[..at]).to_string();
            status = head.split_whitespace().nth(1).and_then(|v| v.parse::<u16>().ok());
            buf.drain(..at + 4);
            header_done = true;
            if status != Some(200) {
                let _ = sock.read_to_end(&mut buf);
                error = Some(format!("HTTP {status:?}: {}", String::from_utf8_lossy(&buf).chars().take(500).collect::<String>()));
                break;
            }
        }
        while let Some(at) = buf.windows(2).position(|w| w == b"\n\n") {
            let frame = String::from_utf8_lossy(&buf[..at]).to_string();
            buf.drain(..at + 2);
            for line in frame.lines() {
                let Some(data) = line.strip_prefix("data: ") else { continue };
                if data == "[DONE]" {
                    break 'read;
                }
                let Ok(ev) = Json::parse(data) else { continue };
                let now = Instant::now();
                let kind = ev.get("type").and_then(Json::as_str).unwrap_or("");
                first_event.get_or_insert(since(now));
                match kind {
                    "step_start" => {
                        steps.push(vec![
                            ("step".to_string(), ev.get("step").cloned().unwrap_or(Json::Null)),
                            ("first_token_s".to_string(), Json::Null),
                            ("tools".to_string(), Json::Arr(vec![])),
                        ]);
                        step_t0.push(now);
                    }
                    "text" | "tool_progress" | "tool_start" => {
                        first_token.get_or_insert(since(now));
                        if let (Some(cur), Some(&t0)) = (steps.last_mut(), step_t0.last()) {
                            if cur[1].1 == Json::Null {
                                cur[1].1 = num(round(now.duration_since(t0).as_secs_f64(), 3));
                            }
                            if kind == "tool_start" {
                                let name = ev.get("name").cloned().unwrap_or(Json::Null);
                                if let Json::Arr(t) = &mut cur[2].1 {
                                    t.push(name.clone());
                                }
                                tool_calls.push(name);
                            }
                        }
                    }
                    "tool_done" => {
                        if let Some(cur) = steps.last_mut() {
                            let out: String = ev.get("output").and_then(Json::as_str).unwrap_or("").chars().take(300).collect();
                            let rec = obj(vec![
                                ("name", ev.get("name").cloned().unwrap_or(Json::Null)),
                                ("ok", ev.get("ok").cloned().unwrap_or(Json::Null)),
                                ("output", s(&out)),
                            ]);
                            match cur.iter_mut().find(|e| e.0 == "tool_results") {
                                Some((_, Json::Arr(v))) => v.push(rec),
                                _ => cur.push(("tool_results".to_string(), Json::Arr(vec![rec]))),
                            }
                        }
                    }
                    "step_stats" => {
                        if let (Some(cur), Some(&t0)) = (steps.last_mut(), step_t0.last()) {
                            let us = ev.get("usage").cloned().unwrap_or(Json::Null);
                            for k in ["prompt_tokens", "prefilled_tokens", "completion_tokens", "prefill_secs", "decode_secs"] {
                                cur.push((k.to_string(), us.get(k).cloned().unwrap_or(Json::Null)));
                            }
                            cur.push(("step_secs".to_string(), num(round(now.duration_since(t0).as_secs_f64(), 3))));
                        }
                    }
                    "compacted" => compactions.push(obj(vec![
                        ("before_tokens", ev.get("before_tokens").cloned().unwrap_or(Json::Null)),
                        ("after_tokens", ev.get("after_tokens").cloned().unwrap_or(Json::Null)),
                        ("compacted_messages", ev.get("compacted_messages").cloned().unwrap_or(Json::Null)),
                    ])),
                    "done" => {
                        final_text = ev.get("text").and_then(Json::as_str).unwrap_or("").to_string();
                        messages = ev.get("messages").cloned().unwrap_or(Json::Null);
                        context_tokens = ev.get("context_tokens").cloned().unwrap_or(Json::Null);
                    }
                    "error" => {
                        error = ev.get("message").and_then(Json::as_str).map(str::to_string);
                        if let Some(m) = ev.get("messages") {
                            messages = m.clone();
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    obj(vec![
        ("status", status.map_or(Json::Null, |v| num(v as f64))),
        ("steps", Json::Arr(steps.into_iter().map(Json::Obj).collect())),
        ("first_token_s", opt(first_token)),
        ("first_event_s", opt(first_event)),
        ("final_text", Json::Str(final_text)),
        ("messages", messages),
        ("context_tokens", context_tokens),
        ("error", error.map_or(Json::Null, Json::Str)),
        ("compactions", Json::Arr(compactions)),
        ("tool_calls", Json::Arr(tool_calls)),
        ("timed_out", Json::Bool(timed_out)),
        ("wall_s", num(round(t_send.elapsed().as_secs_f64(), 3))),
    ])
}

// ---------- tasks ----------

struct Task {
    name: String,
    dir: PathBuf,
    prompt: String,
    kind: String,
    max_steps: f64,
    cmd_timeout: f64,
    timeout_secs: f64,
    check: Vec<String>,
}

fn load_tasks(selected: &[String]) -> Vec<Task> {
    let tasks_dir = eval_dir().join("tasks");
    let mut names: Vec<String> = std::fs::read_dir(&tasks_dir)
        .expect("eval/tasks")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
        .collect();
    names.sort();
    let mut out = Vec::new();
    for name in names {
        let dir = tasks_dir.join(&name);
        let Ok(meta_text) = std::fs::read_to_string(dir.join("task.json")) else { continue };
        if !selected.is_empty() && !selected.iter().any(|p| name.starts_with(p.as_str())) {
            continue;
        }
        let meta = Json::parse(&meta_text).unwrap_or_else(|e| panic!("{name}/task.json: {e}"));
        let prompt = std::fs::read_to_string(dir.join("prompt.txt")).map(|p| p.trim().to_string()).unwrap_or_default();
        out.push(Task {
            kind: meta.get("kind").and_then(Json::as_str).unwrap_or("agent").to_string(),
            max_steps: f(&meta, "max_steps").unwrap_or(12.0),
            cmd_timeout: f(&meta, "cmd_timeout").unwrap_or(30.0),
            timeout_secs: f(&meta, "timeout_secs").unwrap_or(600.0),
            check: meta
                .get("check")
                .and_then(Json::as_arr)
                .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
                .unwrap_or_default(),
            name,
            dir,
            prompt,
        });
    }
    out
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_dir(&e.path(), &to)?;
        } else {
            std::fs::copy(e.path(), to)?;
        }
    }
    Ok(())
}

fn prepare_workdir(task: &Task, run_root: &Path) -> PathBuf {
    let wd = run_root.join(&task.name);
    let _ = std::fs::remove_dir_all(&wd);
    copy_dir(&task.dir.join("workspace"), &wd).expect("copy workspace");
    std::fs::canonicalize(&wd).expect("workdir")
}

/// Runs the hidden check with the workdir appended; returns (passed, last 2000 chars of output).
fn run_check(task: &Task, wd: &Path) -> (bool, String) {
    let Some((prog, rest)) = task.check.split_first() else { return (false, "no check".into()) };
    let mut child = match Command::new(prog).args(rest).arg(wd).current_dir(&task.dir).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn() {
        Ok(c) => c,
        Err(e) => return (false, format!("{prog}: {e}")),
    };
    let t0 = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(st)) => {
                let mut out = String::new();
                if let Some(mut o) = child.stdout.take() {
                    let _ = o.read_to_string(&mut out);
                }
                if let Some(mut e) = child.stderr.take() {
                    let _ = e.read_to_string(&mut out);
                }
                let out = out.trim().to_string();
                let tail: String = out.chars().rev().take(2000).collect::<Vec<_>>().into_iter().rev().collect();
                return (st.success(), tail);
            }
            Ok(None) if t0.elapsed() > Duration::from_secs(300) => {
                let _ = child.kill();
                return (false, "check timed out".into());
            }
            _ => std::thread::sleep(Duration::from_millis(100)),
        }
    }
}

fn payload(messages: Json, wd: &Path, task: &Task) -> Json {
    obj(vec![
        ("messages", messages),
        ("workdir", s(&wd.to_string_lossy())),
        ("max_steps", num(task.max_steps)),
        ("cmd_timeout", num(task.cmd_timeout)),
        ("temperature", num(0.0)),
        ("seed", num(0.0)),
    ])
}

fn user(text: &str) -> Json {
    obj(vec![("role", s("user")), ("content", s(text))])
}

fn run_task(srv: &Server, task: &Task, run_root: &Path) -> Json {
    let wd = prepare_workdir(task, run_root);
    let deadline = Instant::now() + Duration::from_secs_f64(task.timeout_secs);
    let mut turns = Vec::new();
    if task.kind == "conversation" {
        let prompts = Json::parse(&std::fs::read_to_string(task.dir.join("turns.json")).expect("turns.json")).expect("turns.json");
        let mut history: Vec<Json> = Vec::new();
        for (i, p) in prompts.as_arr().unwrap_or(&[]).iter().enumerate() {
            history.push(user(p.as_str().unwrap_or("")));
            let turn = sse_agent_turn(srv, &payload(Json::Arr(history.clone()), &wd, task), deadline);
            let prompt_toks: Vec<String> = turn
                .get("steps")
                .and_then(Json::as_arr)
                .unwrap_or(&[])
                .iter()
                .map(|s| fmt(f(s, "prompt_tokens"), 0))
                .collect();
            println!(
                "    turn {:2}: first_token {}s wall {}s steps {} prompt_tok [{}]",
                i + 1,
                fmt(f(&turn, "first_token_s"), 3),
                fmt(f(&turn, "wall_s"), 3),
                prompt_toks.len(),
                prompt_toks.join(", ")
            );
            match turn.get("messages").and_then(Json::as_arr) {
                Some(m) => history = m.to_vec(),
                None => {}
            }
            let failed = turn.get("messages").is_none_or(|m| *m == Json::Null) || turn.get("error").is_some_and(|e| *e != Json::Null);
            if failed && history.last().and_then(|m| m.get("role")).and_then(Json::as_str) != Some("assistant") {
                let text = turn.get("final_text").and_then(Json::as_str).filter(|t| !t.is_empty()).unwrap_or("(no reply)");
                history.push(obj(vec![("role", s("assistant")), ("content", s(text))]));
            }
            let mut t = turn;
            if let Json::Obj(kv) = &mut t {
                kv.push(("turn".to_string(), num((i + 1) as f64)));
            }
            turns.push(t);
            if failed && Instant::now() > deadline {
                break;
            }
        }
    } else {
        turns.push(sse_agent_turn(srv, &payload(Json::Arr(vec![user(&task.prompt)]), &wd, task), deadline));
    }
    let (passed, detail) = run_check(task, &wd);
    obj(vec![
        ("turns", Json::Arr(turns)),
        ("passed", Json::Bool(passed)),
        ("check_output", Json::Str(detail)),
        ("workdir", s(&wd.to_string_lossy())),
    ])
}

fn all_steps(turns: &[Json]) -> Vec<&Json> {
    turns
        .iter()
        .flat_map(|t| t.get("steps").and_then(Json::as_arr).unwrap_or(&[]))
        .filter(|s| f(s, "prompt_tokens").is_some())
        .collect()
}

fn summarize(result: &Json) -> Json {
    let turns = result.get("turns").and_then(Json::as_arr).unwrap_or(&[]);
    let steps = all_steps(turns);
    let pt: Vec<f64> = steps.iter().filter_map(|s| f(s, "prompt_tokens")).collect();
    let ftl: Vec<f64> = turns.iter().filter_map(|t| f(t, "first_token_s")).collect();
    let errors: Vec<Json> = turns.iter().filter_map(|t| t.get("error").filter(|e| **e != Json::Null).cloned()).collect();
    obj(vec![
        ("steps", num(turns.iter().map(|t| t.get("steps").and_then(Json::as_arr).map_or(0, <[Json]>::len)).sum::<usize>() as f64)),
        ("wall_s", num(round(turns.iter().filter_map(|t| f(t, "wall_s")).sum(), 1))),
        ("prompt_tokens_total", num(pt.iter().sum())),
        ("completion_tokens_total", num(steps.iter().filter_map(|s| f(s, "completion_tokens")).sum())),
        ("prompt_tokens_per_step_median", opt(median(pt.clone()))),
        ("prompt_tokens_max", opt(pt.iter().cloned().reduce(f64::max))),
        ("step_secs_median", opt(median(steps.iter().filter_map(|s| f(s, "step_secs")).collect()).map(|v| round(v, 2)))),
        ("first_token_s_median", opt(median(ftl.clone()).map(|v| round(v, 3)))),
        ("first_token_s_max", opt(ftl.iter().cloned().reduce(f64::max))),
        ("compactions", num(turns.iter().map(|t| t.get("compactions").and_then(Json::as_arr).map_or(0, <[Json]>::len)).sum::<usize>() as f64)),
        ("errors", Json::Arr(errors)),
        ("timed_out", Json::Bool(turns.iter().any(|t| t.get("timed_out").and_then(Json::as_bool) == Some(true)))),
    ])
}

fn write_markdown(path: &Path, run: &Json) {
    let mut out = String::new();
    let a = run.get("aggregate").cloned().unwrap_or(Json::Null);
    let get = |k: &str| run.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    out.push_str(&format!("# Agent eval: {}\n\n", get("label")));
    out.push_str(&format!(
        "Commit `{}` · {} · model {} · server peak RSS {} MiB\n\n",
        get("commit"),
        get("started_at"),
        get("model"),
        fmt(f(&a, "peak_rss_mib"), 1)
    ));
    out.push_str(&format!(
        "Pass rate: {}/{} · median step time {} s · median prompt tokens/step {} · total prompt tokens {} · follow-up first-token median {} s (max {} s) over {} turns\n\n",
        fmt(f(&a, "passed"), 0),
        fmt(f(&a, "total"), 0),
        fmt(f(&a, "step_secs_median"), 2),
        fmt(f(&a, "prompt_tokens_per_step_median"), 0),
        fmt(f(&a, "prompt_tokens_total"), 0),
        fmt(f(&a, "followup_first_token_s_median"), 3),
        fmt(f(&a, "followup_first_token_s_max"), 3),
        fmt(f(&a, "followup_turns"), 0),
    ));
    out.push_str("| # | task | pass | steps | wall s | median step s | prompt tok/step (median / max) | completion tok | first token s (median / max) | compactions |\n");
    out.push_str("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for (i, r) in run.get("tasks").and_then(Json::as_arr).unwrap_or(&[]).iter().enumerate() {
        let sm = r.get("summary").cloned().unwrap_or(Json::Null);
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} / {} | {} | {} / {} | {} |\n",
            i + 1,
            r.get("name").and_then(Json::as_str).unwrap_or(""),
            if r.get("passed").and_then(Json::as_bool) == Some(true) { "pass" } else { "FAIL" },
            fmt(f(&sm, "steps"), 0),
            fmt(f(&sm, "wall_s"), 1),
            fmt(f(&sm, "step_secs_median"), 2),
            fmt(f(&sm, "prompt_tokens_per_step_median"), 0),
            fmt(f(&sm, "prompt_tokens_max"), 0),
            fmt(f(&sm, "completion_tokens_total"), 0),
            fmt(f(&sm, "first_token_s_median"), 3),
            fmt(f(&sm, "first_token_s_max"), 3),
            fmt(f(&sm, "compactions"), 0),
        ));
    }
    std::fs::write(path, out).expect("write markdown");
}

fn now_iso() -> String {
    let out = Command::new("date").arg("+%Y-%m-%dT%H:%M:%S").output().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    out.unwrap_or_default()
}

fn cmd_run(a: &Args) {
    let label = a.get("label").unwrap_or_else(|| usage()).to_string();
    let selected: Vec<String> = a.get("tasks").unwrap_or("").split(',').map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).collect();
    let tasks = load_tasks(&selected);
    if tasks.is_empty() {
        eprintln!("no tasks selected");
        std::process::exit(2);
    }
    let results = eval_dir().join("results");
    std::fs::create_dir_all(&results).expect("results dir");
    let workroot = a.get("workroot").map_or_else(|| Path::new(ROOT).join("target/eval-work").join(&label), PathBuf::from);
    std::fs::create_dir_all(&workroot).expect("workroot");
    let workroot = std::fs::canonicalize(&workroot).expect("workroot");
    let commit = Command::new("git")
        .args(["-C", ROOT, "rev-parse", "--short", "HEAD"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let model = a.get("model").map_or_else(|| "Qwen3.5-9B".to_string(), |m| Path::new(m).file_name().map_or(m.to_string(), |n| n.to_string_lossy().into_owned()));
    let started_at = now_iso();

    let mut srv = Server::start(a, &workroot);
    println!("server on {} (workroot {})", srv.url(), workroot.display());
    let mut done = Vec::new();
    let partial = results.join(format!("{label}.partial.json"));
    for task in &tasks {
        println!("== {} ({})", task.name, task.kind);
        let t0 = Instant::now();
        let mut res = run_task(&srv, task, &workroot);
        let mut summary = summarize(&res);
        if let Json::Obj(kv) = &mut summary {
            kv.push(("total_wall_s".to_string(), num(round(t0.elapsed().as_secs_f64(), 1))));
        }
        if let Json::Obj(kv) = &mut res {
            kv.push(("name".to_string(), s(&task.name)));
            kv.push(("kind".to_string(), s(&task.kind)));
            kv.push(("summary".to_string(), summary.clone()));
        }
        let check_last = res.get("check_output").and_then(Json::as_str).unwrap_or("").lines().last().unwrap_or("").to_string();
        println!(
            "   {} steps={} wall={}s prompt_tok/step={} first_token={}s errors={}\n   check: {check_last}",
            if res.get("passed").and_then(Json::as_bool) == Some(true) { "PASS" } else { "FAIL" },
            fmt(f(&summary, "steps"), 0),
            fmt(f(&summary, "wall_s"), 1),
            fmt(f(&summary, "prompt_tokens_per_step_median"), 0),
            fmt(f(&summary, "first_token_s_median"), 3),
            summary.get("errors").map_or(String::new(), Json::stringify),
        );
        done.push(res);
        let snap = obj(vec![("label", s(&label)), ("commit", s(&commit)), ("tasks", Json::Arr(done.clone()))]);
        let _ = std::fs::write(&partial, snap.stringify());
    }
    let peak = srv.peak_rss_mib();
    srv.stop();

    let turns_of = |r: &Json| r.get("turns").and_then(Json::as_arr).map(<[Json]>::to_vec).unwrap_or_default();
    let all_turns: Vec<Json> = done.iter().flat_map(turns_of).collect();
    let steps = all_steps(&all_turns);
    let ftl: Vec<f64> = done
        .iter()
        .filter(|r| r.get("kind").and_then(Json::as_str) == Some("conversation"))
        .flat_map(|r| turns_of(r).into_iter().skip(1))
        .filter_map(|t| f(&t, "first_token_s"))
        .collect();
    let aggregate = obj(vec![
        ("passed", num(done.iter().filter(|r| r.get("passed").and_then(Json::as_bool) == Some(true)).count() as f64)),
        ("total", num(done.len() as f64)),
        ("step_secs_median", opt(median(steps.iter().filter_map(|s| f(s, "step_secs")).collect()).map(|v| round(v, 2)))),
        ("prompt_tokens_per_step_median", opt(median(steps.iter().filter_map(|s| f(s, "prompt_tokens")).collect()))),
        ("prompt_tokens_total", num(steps.iter().filter_map(|s| f(s, "prompt_tokens")).sum())),
        ("completion_tokens_total", num(steps.iter().filter_map(|s| f(s, "completion_tokens")).sum())),
        ("followup_turns", num(ftl.len() as f64)),
        ("followup_first_token_s_median", opt(median(ftl.clone()).map(|v| round(v, 3)))),
        ("followup_first_token_s_max", opt(ftl.iter().cloned().reduce(f64::max))),
        ("peak_rss_mib", opt(peak)),
        ("wall_s", num(round(done.iter().filter_map(|r| r.get("summary").and_then(|s| f(s, "total_wall_s"))).sum(), 1))),
    ]);
    let run = obj(vec![
        ("label", s(&label)),
        ("commit", s(&commit)),
        ("model", s(&model)),
        ("started_at", s(&started_at)),
        ("tasks", Json::Arr(done)),
        ("aggregate", aggregate.clone()),
    ]);
    let out_json = results.join(format!("{label}.json"));
    std::fs::write(&out_json, run.stringify()).expect("write results");
    write_markdown(&results.join(format!("{label}.md")), &run);
    let _ = std::fs::remove_file(&partial);
    println!("{}\nwrote {}", aggregate.stringify(), out_json.display());
}

fn followup(run: &Json) -> (usize, Option<f64>, Option<f64>, Option<f64>, Option<f64>, usize, usize) {
    let turns: Vec<Json> = run
        .get("tasks")
        .and_then(Json::as_arr)
        .unwrap_or(&[])
        .iter()
        .filter(|r| r.get("kind").and_then(Json::as_str) == Some("conversation"))
        .flat_map(|r| r.get("turns").and_then(Json::as_arr).map(<[Json]>::to_vec).unwrap_or_default())
        .collect();
    let mut ftl: Vec<f64> = turns.iter().skip(1).filter_map(|t| f(t, "first_token_s")).collect();
    let first_prompt: Vec<f64> = turns
        .iter()
        .filter_map(|t| t.get("steps").and_then(Json::as_arr).and_then(|s| s.first()).and_then(|s| f(s, "prompt_tokens")))
        .collect();
    ftl.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p90 = (!ftl.is_empty()).then(|| round(ftl[(0.9 * (ftl.len() - 1) as f64) as usize], 2));
    (
        turns.len(),
        median(ftl.clone()).map(|v| round(v, 2)),
        p90,
        ftl.last().copied(),
        first_prompt.last().copied(),
        turns.iter().map(|t| t.get("steps").and_then(Json::as_arr).map_or(0, <[Json]>::len)).sum(),
        turns.iter().filter(|t| t.get("error").is_some_and(|e| *e != Json::Null)).count(),
    )
}

fn cmd_compare(a: &Args) {
    let [la, lb] = [&a.pos[0], &a.pos[1]];
    let load = |l: &str| -> Json {
        let p = eval_dir().join("results").join(format!("{l}.json"));
        Json::parse(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).expect("results json")
    };
    let (ra, rb) = (load(la), load(lb));
    let (aa, ab) = (ra.get("aggregate").cloned().unwrap_or(Json::Null), rb.get("aggregate").cloned().unwrap_or(Json::Null));
    let (fa, fb) = (followup(&ra), followup(&rb));
    let total_steps = |r: &Json| -> f64 {
        r.get("tasks").and_then(Json::as_arr).unwrap_or(&[]).iter().filter_map(|t| t.get("summary").and_then(|s| f(s, "steps"))).sum()
    };
    let rows: Vec<(String, String, String)> = vec![
        ("pass rate".into(), format!("{}/{}", fmt(f(&aa, "passed"), 0), fmt(f(&aa, "total"), 0)), format!("{}/{}", fmt(f(&ab, "passed"), 0), fmt(f(&ab, "total"), 0))),
        ("median step time (s)".into(), fmt(f(&aa, "step_secs_median"), 2), fmt(f(&ab, "step_secs_median"), 2)),
        ("median prompt tokens / step".into(), fmt(f(&aa, "prompt_tokens_per_step_median"), 0), fmt(f(&ab, "prompt_tokens_per_step_median"), 0)),
        ("total prompt tokens".into(), fmt(f(&aa, "prompt_tokens_total"), 0), fmt(f(&ab, "prompt_tokens_total"), 0)),
        ("total completion tokens".into(), fmt(f(&aa, "completion_tokens_total"), 0), fmt(f(&ab, "completion_tokens_total"), 0)),
        ("total agent steps".into(), fmt(Some(total_steps(&ra)), 0), fmt(Some(total_steps(&rb)), 0)),
        ("total wall (s)".into(), fmt(f(&aa, "wall_s"), 1), fmt(f(&ab, "wall_s"), 1)),
        ("peak server RSS (MiB)".into(), fmt(f(&aa, "peak_rss_mib"), 1), fmt(f(&ab, "peak_rss_mib"), 1)),
        ("follow-up first token, median (s)".into(), fmt(fa.1, 2), fmt(fb.1, 2)),
        ("follow-up first token, p90 / max (s)".into(), format!("{} / {}", fmt(fa.2, 2), fmt(fa.3, 2)), format!("{} / {}", fmt(fb.2, 2), fmt(fb.3, 2))),
        ("follow-up prompt tokens at last turn (step 1)".into(), fmt(fa.4, 0), fmt(fb.4, 0)),
        ("follow-up steps / turns ending in error".into(), format!("{} / {}", fa.5, fa.6), format!("{} / {}", fb.5, fb.6)),
    ];
    let label = |r: &Json, k: &str| r.get(k).and_then(Json::as_str).unwrap_or("").to_string();
    println!("# {la} ({}) vs {lb} ({})\n", label(&ra, "commit"), label(&rb, "commit"));
    println!("| metric | {la} | {lb} |\n|---|---:|---:|");
    for (n, x, y) in rows {
        println!("| {n} | {x} | {y} |");
    }
    println!("\n| # | task | {la} | {lb} | steps {la} / {lb} | prompt tok/step {la} / {lb} |\n|---|---|---|---|---:|---:|");
    let tb = rb.get("tasks").and_then(Json::as_arr).unwrap_or(&[]);
    let pass = |r: &Json| if r.get("passed").and_then(Json::as_bool) == Some(true) { "pass" } else { "FAIL" };
    for (i, r) in ra.get("tasks").and_then(Json::as_arr).unwrap_or(&[]).iter().enumerate() {
        let name = r.get("name").and_then(Json::as_str).unwrap_or("");
        let o = tb.iter().find(|x| x.get("name").and_then(Json::as_str) == Some(name));
        let sa = r.get("summary").cloned().unwrap_or(Json::Null);
        let sb = o.and_then(|x| x.get("summary")).cloned().unwrap_or(Json::Null);
        println!(
            "| {} | {name} | {} | {} | {} / {} | {} / {} |",
            i + 1,
            pass(r),
            o.map_or("-", pass),
            fmt(f(&sa, "steps"), 0),
            fmt(f(&sb, "steps"), 0),
            fmt(f(&sa, "prompt_tokens_per_step_median"), 0),
            fmt(f(&sb, "prompt_tokens_per_step_median"), 0),
        );
    }
}

fn main() {
    let a = Args::parse();
    match a.cmd.as_str() {
        "run" => cmd_run(&a),
        "compare" if a.pos.len() == 2 => cmd_compare(&a),
        _ => usage(),
    }
}
