//! Bridge between Kronos Mobile and this Mac, behind a bearer token. Serves device status,
//! conversation lookup, the paired iPhone's app list, chat through the local model server,
//! agent tasks that need approval on the Mac, and the phone model files for the Android app.
//! Every other server route stays private.
//!
//!   kronos_device_bridge [--host 127.0.0.1] [--port 18080] [--upstream 127.0.0.1:18085]
//!                        [--model-dir DIR] [--approval-seconds 120] [--no-confirm]
//!
//! Pass `--host 0.0.0.0` so the phone can reach it over Wi-Fi.
//! `--no-confirm` runs tasks without the Mac prompt: anyone with the token can then run commands.
//! Token: `~/.kronos/device_token` (created 0600 on first run).

use kronos_2t::runtime::conversations::retrieve_local_conversation_json;
use kronos_2t::runtime::device::global_device_manager;
use kronos_2t::util::json::{quote, Json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

static LOOKUP: Mutex<()> = Mutex::new(());
static JSON_SEQ: AtomicU64 = AtomicU64::new(0);

const MAX_BODY: usize = 8 << 20;
const MAX_TASK_CHARS: usize = 1000;
const MAX_PENDING_APPROVALS: usize = 3;
const MAX_TASKS_KEPT: usize = 20;
const PHONE_MODEL_FILES: [&str; 7] = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "kronos-nf4g128.k2tb",
];

#[derive(Clone, Default)]
struct Task {
    id: u64,
    prompt: String,
    state: &'static str,
    turn_id: u64,
    detail: String,
    text: String,
    deliverables: Vec<String>,
}

struct Bridge {
    token: String,
    upstream: String,
    model_dir: PathBuf,
    confirm: bool,
    approval_seconds: u64,
    tasks: Mutex<Vec<Task>>,
    next_id: Mutex<u64>,
}

/// One request to the local model server. Returns (status, body).
fn upstream_call(addr: &str, method: &str, target: &str, body: &[u8], timeout: Duration) -> Result<(u16, Vec<u8>), String> {
    let sock = addr.parse().map_err(|e| format!("{e}"))?;
    let mut up = TcpStream::connect_timeout(&sock, Duration::from_secs(2)).map_err(|e| e.to_string())?;
    let _ = up.set_read_timeout(Some(timeout));
    write!(
        up,
        "{method} {target} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .and_then(|_| up.write_all(body))
    .map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    up.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").ok_or("bad upstream response")?;
    let code = String::from_utf8_lossy(&raw[..split]).split(' ').nth(1).and_then(|c| c.parse().ok()).unwrap_or(502);
    Ok((code, raw[split + 4..].to_vec()))
}

fn dialog_approved(osascript_stdout: &str) -> bool {
    osascript_stdout.contains("button returned:Run") && !osascript_stdout.contains("gave up:true")
}

/// The prompt goes in as argv so it cannot alter the script. Silence, Deny or failure means no.
fn approved_on_mac(prompt: &str, seconds: u64) -> bool {
    let msg = format!("Run this task from your iPhone?\n\n{prompt}\n\nKronos can edit files and run commands in its workspace.");
    let line = format!(
        "display dialog (item 1 of argv) with title \"Kronos\" buttons {{\"Deny\", \"Run\"}} default button \"Deny\" cancel button \"Deny\" giving up after {seconds}"
    );
    Command::new("/usr/bin/osascript")
        .args(["-e", "on run argv", "-e", &line, "-e", "end run", "--", &msg])
        .output()
        .is_ok_and(|o| o.status.success() && dialog_approved(&String::from_utf8_lossy(&o.stdout)))
}

fn with_task(b: &Bridge, id: u64, f: impl FnOnce(&mut Task)) {
    let mut tasks = b.tasks.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(t) = tasks.iter_mut().find(|t| t.id == id) {
        f(t);
    }
}

fn run_task(b: &Bridge, id: u64, prompt: &str) {
    if b.confirm && !approved_on_mac(prompt, b.approval_seconds) {
        eprintln!("task {id}: not approved");
        return with_task(b, id, |t| {
            if t.state == "awaiting_approval" {
                t.state = "denied";
            }
        });
    }
    let mut cancelled = false;
    with_task(b, id, |t| cancelled = t.state == "cancelled");
    if cancelled {
        return;
    }
    eprintln!("task {id}: approved: {prompt}");
    let body = format!(r#"{{"prompt":{},"label":"iPhone"}}"#, quote(prompt));
    let turn = upstream_call(&b.upstream, "POST", "/v1/agents/spawn", body.as_bytes(), Duration::from_secs(10))
        .ok()
        .filter(|(c, _)| *c == 200)
        .and_then(|(_, raw)| Json::parse(&String::from_utf8_lossy(&raw)).ok())
        .and_then(|j| j.get("agents")?.as_arr()?.first()?.get("turn_id")?.as_f64());
    with_task(b, id, |t| match turn {
        Some(tid) => {
            t.state = "running";
            t.turn_id = tid as u64;
        }
        None => {
            t.state = "failed";
            t.detail = "The Kronos server on your Mac did not accept the task.".into();
        }
    });
}

/// Pulls a running task's progress from the model server.
fn refresh_task(b: &Bridge, t: &mut Task) {
    if t.state != "running" {
        return;
    }
    let target = format!("/v1/turn/live?turn_id={}", t.turn_id);
    let Ok((200, raw)) = upstream_call(&b.upstream, "GET", &target, b"", Duration::from_secs(5)) else {
        t.detail = "The Kronos server on your Mac is not responding.".into();
        return;
    };
    let Ok(live) = Json::parse(&String::from_utf8_lossy(&raw)) else { return };
    if live.get("turn_id").and_then(Json::as_f64).map(|v| v as u64) != Some(t.turn_id) {
        t.state = "failed";
        t.detail = "The Kronos server on your Mac restarted and lost the task.".into();
        return;
    }
    t.detail = live.get("status_detail").and_then(Json::as_str).unwrap_or("").to_string();
    if live.get("active").and_then(Json::as_bool) == Some(true) {
        return;
    }
    t.deliverables = live
        .get("deliverables")
        .and_then(Json::as_arr)
        .map(|a| a.iter().filter_map(|d| d.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    let events = live.get("events").and_then(Json::as_arr);
    let done = events.and_then(|ev| ev.iter().rev().find(|e| e.get("type").and_then(Json::as_str) == Some("done")));
    match done.and_then(|d| d.get("text")).and_then(Json::as_str) {
        Some(text) => {
            t.state = "done";
            t.text = text.to_string();
        }
        None => t.state = "failed",
    }
}

fn task_json(t: &Task) -> String {
    let items: Vec<String> = t.deliverables.iter().map(|d| quote(d)).collect();
    format!(
        r#"{{"id":{},"prompt":{},"state":{},"detail":{},"text":{},"deliverables":[{}]}}"#,
        t.id,
        quote(&t.prompt),
        quote(t.state),
        quote(&t.detail),
        quote(&t.text),
        items.join(",")
    )
}

fn task_create(b: &Arc<Bridge>, stream: &mut TcpStream, body: &[u8]) {
    let parsed = Json::parse(&String::from_utf8_lossy(body)).ok();
    let prompt = parsed.as_ref().and_then(|j| j.get("prompt")).and_then(Json::as_str).unwrap_or("").trim().to_string();
    // A longer task could not be read in full in the approval prompt.
    if prompt.is_empty() || prompt.chars().count() > MAX_TASK_CHARS {
        return reply(stream, 400, "Bad Request", &format!(r#"{{"error":{}}}"#, quote(&format!("A task must be 1 to {MAX_TASK_CHARS} characters."))));
    }
    if !matches!(upstream_call(&b.upstream, "GET", "/health", b"", Duration::from_secs(3)), Ok((200, _))) {
        return reply(stream, 503, "Service Unavailable", r#"{"error":"The Kronos server is not running on your Mac. Start it, then ask again."}"#);
    }
    let mut tasks = b.tasks.lock().unwrap_or_else(|p| p.into_inner());
    if tasks.iter().filter(|t| t.state == "awaiting_approval").count() >= MAX_PENDING_APPROVALS {
        return reply(stream, 429, "Too Many Requests", r#"{"error":"Other tasks are already waiting for approval on your Mac."}"#);
    }
    let id = {
        let mut n = b.next_id.lock().unwrap_or_else(|p| p.into_inner());
        *n += 1;
        *n
    };
    let task = Task { id, prompt: prompt.clone(), state: "awaiting_approval", ..Task::default() };
    let out = task_json(&task);
    tasks.push(task);
    if tasks.len() > MAX_TASKS_KEPT {
        tasks.remove(0);
    }
    drop(tasks);
    let bg = Arc::clone(b);
    std::thread::spawn(move || run_task(&bg, id, &prompt));
    reply(stream, 200, "OK", &out);
}

fn task_list(b: &Bridge, stream: &mut TcpStream, id: Option<u64>) {
    let mut snapshot: Vec<Task> = b.tasks.lock().unwrap_or_else(|p| p.into_inner()).clone();
    snapshot.retain(|t| id.is_none_or(|want| t.id == want));
    for t in &mut snapshot {
        refresh_task(b, t);
        let fresh = t.clone();
        with_task(b, t.id, |live| {
            if live.state == "running" {
                *live = fresh;
            }
        });
    }
    let items: Vec<String> = snapshot.iter().rev().map(task_json).collect();
    reply(stream, 200, "OK", &format!(r#"{{"tasks":[{}]}}"#, items.join(",")));
}

fn task_cancel(b: &Bridge, stream: &mut TcpStream, body: &[u8]) {
    let parsed = Json::parse(&String::from_utf8_lossy(body)).ok();
    let id = parsed.as_ref().and_then(|j| j.get("id")).and_then(Json::as_f64).unwrap_or(0.0) as u64;
    let mut turn = 0;
    with_task(b, id, |t| {
        if matches!(t.state, "awaiting_approval" | "running") {
            turn = t.turn_id;
            t.state = "cancelled";
        }
    });
    if turn != 0 {
        let _ = upstream_call(&b.upstream, "POST", "/v1/agent/cancel", format!(r#"{{"turn_id":{turn}}}"#).as_bytes(), Duration::from_secs(5));
    }
    reply(stream, 200, "OK", r#"{"status":"ok"}"#);
}

/// Runs devicectl and returns its JSON output. iOS hides this list from apps; the paired Mac can read it.
fn devicectl_json(args: &[&str]) -> Result<Json, String> {
    let seq = JSON_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("kronos_devicectl_{}_{seq}.json", std::process::id()));
    let out = Command::new("/usr/bin/xcrun")
        .arg("devicectl")
        .args(args)
        .args(["--timeout", "12", "--json-output"])
        .arg(&path)
        .output()
        .map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(&path).unwrap_or_default();
    let _ = std::fs::remove_file(&path);
    if !out.status.success() && text.is_empty() {
        return Err(String::from_utf8_lossy(&out.stderr).lines().next().unwrap_or("devicectl failed").to_string());
    }
    Json::parse(&text)
}

/// First physical iPhone: one with a live connection when any has it.
fn pick_iphone(devices: &[Json]) -> Option<(String, String)> {
    let phones: Vec<&Json> = devices
        .iter()
        .filter(|d| {
            let hw = d.get("hardwareProperties");
            hw.and_then(|h| h.get("reality")).and_then(Json::as_str) == Some("physical")
                && hw.and_then(|h| h.get("deviceType")).and_then(Json::as_str) == Some("iPhone")
        })
        .collect();
    let connected = |d: &&&Json| d.get("connectionProperties").and_then(|c| c.get("tunnelState")).and_then(Json::as_str) == Some("connected");
    let d = phones.iter().find(connected).or(phones.first())?;
    let udid = d.get("hardwareProperties")?.get("udid")?.as_str()?.to_string();
    let name = d.get("deviceProperties").and_then(|p| p.get("name")).and_then(Json::as_str).unwrap_or("iPhone").to_string();
    Some((udid, name))
}

fn iphone() -> Result<(String, String), String> {
    let devices = devicectl_json(&["list", "devices"])?;
    let list = devices.get("result").and_then(|r| r.get("devices")).and_then(Json::as_arr).ok_or("no device list")?;
    pick_iphone(list).ok_or_else(|| "no iPhone is paired with this Mac".to_string())
}

/// (name, bundle id) for each visible app.
fn list_apps(udid: &str) -> Result<Vec<(String, String)>, String> {
    let apps = devicectl_json(&["device", "info", "apps", "--device", udid, "--include-all-apps"])?;
    let arr = apps.get("result").and_then(|r| r.get("apps")).and_then(Json::as_arr).ok_or("the iPhone is not reachable from this Mac")?;
    Ok(arr
        .iter()
        .filter(|a| a.get("hidden").and_then(Json::as_bool) != Some(true))
        .filter_map(|a| Some((a.get("name")?.as_str()?.to_string(), a.get("bundleIdentifier")?.as_str()?.to_string())))
        .collect())
}

fn phone_apps(stream: &mut TcpStream) {
    let out = iphone().and_then(|(udid, name)| {
        let items: Vec<String> = list_apps(&udid)?
            .iter()
            .map(|(n, b)| format!(r#"{{"name":{},"bundle":{}}}"#, quote(n), quote(b)))
            .collect();
        Ok(format!(r#"{{"device":{},"apps":[{}]}}"#, quote(&name), items.join(",")))
    });
    match out {
        Ok(body) => reply(stream, 200, "OK", &body),
        Err(e) => reply(stream, 503, "Service Unavailable", &format!(r#"{{"error":{}}}"#, quote(&e))),
    }
}

/// Opens an installed app on the iPhone. Only bundle ids from its own app list are accepted.
fn phone_app_open(stream: &mut TcpStream, body: &[u8]) {
    let parsed = Json::parse(&String::from_utf8_lossy(body)).ok();
    let bundle = parsed.as_ref().and_then(|j| j.get("bundle")).and_then(Json::as_str).unwrap_or("").to_string();
    let out = iphone().and_then(|(udid, _)| {
        if !list_apps(&udid)?.iter().any(|(_, b)| *b == bundle) {
            return Err("that app is not on the iPhone".to_string());
        }
        let run = Command::new("/usr/bin/xcrun")
            .args(["devicectl", "device", "process", "launch", "--timeout", "12", "--device", &udid, &bundle])
            .output()
            .map_err(|e| e.to_string())?;
        if run.status.success() {
            Ok(())
        } else {
            Err("the iPhone refused. It may be locked.".to_string())
        }
    });
    match out {
        Ok(()) => reply(stream, 200, "OK", r#"{"status":"ok"}"#),
        Err(e) => reply(stream, 503, "Service Unavailable", &format!(r#"{{"error":{}}}"#, quote(&e))),
    }
}

fn forward_chat(b: &Bridge, stream: &mut TcpStream, body: &[u8]) {
    match upstream_call(&b.upstream, "POST", "/v1/chat/completions", body, Duration::from_secs(300)) {
        Ok((code, raw)) => reply(stream, code, "Upstream", &String::from_utf8_lossy(&raw)),
        Err(_) => reply(stream, 503, "Service Unavailable", r#"{"error":"model server not running"}"#),
    }
}

fn model_manifest(b: &Bridge, stream: &mut TcpStream) {
    let files: Option<Vec<String>> = PHONE_MODEL_FILES
        .iter()
        .map(|n| {
            let bytes = std::fs::metadata(b.model_dir.join(n)).ok()?.len();
            Some(format!(r#"{{"name":{},"bytes":{bytes}}}"#, quote(n)))
        })
        .collect();
    match files {
        Some(f) => reply(stream, 200, "OK", &format!(r#"{{"mode":"nf4g128","files":[{}]}}"#, f.join(","))),
        None => reply(stream, 404, "Not Found", r#"{"error":"phone model is not on this Mac"}"#),
    }
}

fn model_file(b: &Bridge, stream: &mut TcpStream, name: &str) {
    if !PHONE_MODEL_FILES.contains(&name) {
        return reply(stream, 404, "Not Found", r#"{"error":"not a model file"}"#);
    }
    let Ok(mut file) = std::fs::File::open(b.model_dir.join(name)) else {
        return reply(stream, 404, "Not Found", r#"{"error":"phone model is not on this Mac"}"#);
    };
    let len = file.metadata().map(|m| m.len()).unwrap_or(0);
    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n");
    if stream.write_all(head.as_bytes()).is_ok() {
        let _ = std::io::copy(&mut file, stream);
    }
}

fn default_model_dir() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default()).join("Documents/project/kronos-2t/models/Qwen3-0.6B")
}

fn load_token() -> std::io::Result<String> {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    let path = home.join(".kronos/device_token");
    if let Ok(t) = std::fs::read_to_string(&path) {
        let t = t.trim().to_string();
        if t.len() >= 32 {
            return Ok(t);
        }
    }
    let mut raw = [0u8; 24];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut raw)?;
    let token: String = raw.iter().map(|b| format!("{b:02x}")).collect();
    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&path)?;
    f.write_all(token.as_bytes())?;
    Ok(token)
}

fn token_matches(given: &str, want: &str) -> bool {
    given.len() == want.len() && given.bytes().zip(want.bytes()).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        let hex = (b[i] == b'%' && i + 2 < b.len())
            .then(|| std::str::from_utf8(&b[i + 1..i + 3]).ok().and_then(|h| u8::from_str_radix(h, 16).ok()))
            .flatten();
        match (b[i], hex) {
            (_, Some(v)) => {
                out.push(v);
                i += 3;
            }
            (b'+', _) => {
                out.push(b' ');
                i += 1;
            }
            (c, _) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn reply(stream: &mut TcpStream, code: u16, reason: &str, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
}

fn handle(mut stream: TcpStream, b: &Arc<Bridge>) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
    let mut buf = Vec::with_capacity(4096);
    let mut chunk = [0u8; 2048];
    while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
        match stream.read(&mut chunk) {
            Ok(n) if n > 0 && buf.len() < 65536 => buf.extend_from_slice(&chunk[..n]),
            _ => return,
        }
    }
    let head = String::from_utf8_lossy(&buf).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split(' ');
    let (method, target) = (first.next().unwrap_or(""), first.next().unwrap_or(""));
    let authed = lines.any(|l| {
        l.split_once(':').is_some_and(|(k, v)| {
            k.eq_ignore_ascii_case("authorization")
                && v.trim().strip_prefix("Bearer ").is_some_and(|t| token_matches(t, &b.token))
        })
    });
    if !authed {
        std::thread::sleep(Duration::from_secs(1));
        return reply(&mut stream, 401, "Unauthorized", r#"{"error":"invalid token"}"#);
    }
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let split = buf.windows(4).position(|w| w == b"\r\n\r\n").unwrap_or(0) + 4;
    let want = head[..split.min(head.len())]
        .split("\r\n")
        .find_map(|l| l.split_once(':').filter(|(k, _)| k.eq_ignore_ascii_case("content-length")).and_then(|(_, v)| v.trim().parse::<usize>().ok()))
        .unwrap_or(0);
    if want > MAX_BODY {
        return reply(&mut stream, 413, "Too Large", r#"{"error":"body too large"}"#);
    }
    while buf.len() < split + want {
        match stream.read(&mut chunk) {
            Ok(n) if n > 0 => buf.extend_from_slice(&chunk[..n]),
            _ => return,
        }
    }
    let body = &buf[split..split + want];
    match (method, path) {
        ("POST", "/v1/chat/completions") => forward_chat(b, &mut stream, body),
        ("GET", "/v1/device/model") => model_manifest(b, &mut stream),
        ("GET", p) if p.starts_with("/v1/device/model/") => model_file(b, &mut stream, &p["/v1/device/model/".len()..]),
        ("GET", "/v1/device/phone_apps") => phone_apps(&mut stream),
        ("POST", "/v1/device/phone_apps/open") => phone_app_open(&mut stream, body),
        ("POST", "/v1/device/tasks") => task_create(b, &mut stream, body),
        ("POST", "/v1/device/tasks/cancel") => task_cancel(b, &mut stream, body),
        ("GET", "/v1/device/tasks") => {
            let id = query.split('&').find_map(|p| p.strip_prefix("id=")).and_then(|v| v.parse().ok());
            task_list(b, &mut stream, id);
        }
        ("GET", "/v1/device/status") => reply(&mut stream, 200, "OK", &global_device_manager().status_json()),
        ("GET", "/v1/device/conversations") => {
            let param = |n: &str| query.split('&').find_map(|p| p.strip_prefix(n)).map(percent_decode);
            let q = param("q=").unwrap_or_default();
            // Message times are rendered by child sqlite3 in TZ: use the phone's zone.
            let _guard = LOOKUP.lock().unwrap_or_else(|p| p.into_inner());
            match param("tz=").filter(|t| t.chars().all(|c| c.is_ascii_alphanumeric() || "/_+-".contains(c))) {
                Some(tz) => std::env::set_var("TZ", tz),
                None => std::env::remove_var("TZ"),
            }
            reply(&mut stream, 200, "OK", &retrieve_local_conversation_json(&q));
        }
        _ => reply(&mut stream, 404, "Not Found", r#"{"error":"not served by the bridge"}"#),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let addr = format!(
        "{}:{}",
        flag("--host").unwrap_or_else(|| "127.0.0.1".into()),
        flag("--port").unwrap_or_else(|| "18080".into())
    );
    let token = match load_token() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("token: {e}");
            std::process::exit(1);
        }
    };
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    let bridge = Arc::new(Bridge {
        token,
        upstream: flag("--upstream").unwrap_or_else(|| "127.0.0.1:18085".into()),
        model_dir: flag("--model-dir").map(PathBuf::from).unwrap_or_else(default_model_dir),
        confirm: !args.iter().any(|a| a == "--no-confirm"),
        approval_seconds: flag("--approval-seconds").and_then(|v| v.parse().ok()).unwrap_or(120),
        tasks: Mutex::new(Vec::new()),
        next_id: Mutex::new(0),
    });
    eprintln!("device bridge on {addr}, model server at {}, task approval {}", bridge.upstream, if bridge.confirm { "on" } else { "OFF" });
    for stream in listener.incoming().flatten() {
        let b = Arc::clone(&bridge);
        std::thread::spawn(move || handle(stream, &b));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_clicked_run_counts_as_approval() {
        assert!(dialog_approved("button returned:Run, gave up:false"));
        assert!(!dialog_approved("button returned:, gave up:true"));
        assert!(!dialog_approved("button returned:Run, gave up:true"));
        assert!(!dialog_approved(""));
    }

    #[test]
    fn token_must_match_exactly() {
        assert!(token_matches("abc", "abc"));
        assert!(!token_matches("abd", "abc"));
        assert!(!token_matches("ab", "abc"));
    }
}
