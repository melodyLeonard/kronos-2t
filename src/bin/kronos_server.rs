//! OpenAI-compatible HTTP API for a Kronos-compiled SmolLM2-Instruct model.
//!
//! Usage: `kronos_server <model_dir> [flags]`
//!   `--host H`            bind address (default 127.0.0.1)
//!   `--port P`            (default 8080)
//!   `--api-key KEY`       require `Authorization: Bearer KEY`
//!   `--ane`               prefill on the Neural Engine (`ane` feature, models in `<model_dir>/ane`)
//!   `--mode M`, `--calib FILE`, `--calib-tokens N`, `--ctx N`, `--threads N` as in `kronos_chat`
//!
//! Endpoints: `POST /v1/chat/completions` (with `"stream": true` for server-sent events),
//! `GET /v1/models`, `GET /health`, and a web chat page at `/`. Requests are served one generation
//! at a time.

use kronos_2t::chat::{ChatModel, Completion, GenParams, Message};
use kronos_2t::json::{quote, Json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MAX_BODY: usize = 4 << 20;
const CHAT_PAGE: &str = include_str!("chat.html");
const MAX_HEADER: usize = 64 << 10;

struct Server {
    model: Mutex<ChatModel>,
    name: String,
    api_key: Option<String>,
    next_id: AtomicU64,
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Request, (u16, String)> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| (400, e.to_string()))?;
    let mut parts = line.split_whitespace();
    let (method, path) = match (parts.next(), parts.next()) {
        (Some(m), Some(p)) => (m.to_string(), p.split('?').next().unwrap_or("").to_string()),
        _ => return Err((400, "malformed request line".into())),
    };
    let mut headers = Vec::new();
    let mut header_bytes = 0;
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h).map_err(|e| (400, e.to_string()))?;
        header_bytes += n;
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        if header_bytes > MAX_HEADER {
            return Err((431, "headers too large".into()));
        }
        if let Some((k, v)) = h.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    let mut req = Request { method, path, headers, body: Vec::new() };
    let len: usize = req.header("content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
    if len > MAX_BODY {
        return Err((413, "request body too large".into()));
    }
    req.body = vec![0; len];
    reader.read_exact(&mut req.body).map_err(|e| (400, e.to_string()))?;
    Ok(req)
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    }
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{CORS}Connection: close\r\n\r\n",
        reason(status),
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).and_then(|_| stream.write_all(body.as_bytes()));
}

fn error(stream: &mut TcpStream, status: u16, msg: &str) {
    let kind = if status == 401 { "authentication_error" } else { "invalid_request_error" };
    respond(stream, status, "application/json", &format!(r#"{{"error":{{"message":{},"type":"{kind}"}}}}"#, quote(msg)));
}

/// `content` may be a string or an array of `{"type":"text","text":...}` parts.
fn message_text(content: &Json) -> Option<String> {
    match content {
        Json::Str(s) => Some(s.clone()),
        Json::Arr(parts) => Some(parts.iter().filter_map(|p| p.get("text").and_then(Json::as_str)).collect()),
        Json::Null => Some(String::new()),
        _ => None,
    }
}

fn parse_chat(body: &Json, defaults: &GenParams) -> Result<(Vec<Message>, GenParams, bool), String> {
    let msgs = body.get("messages").and_then(Json::as_arr).ok_or("`messages` must be an array")?;
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = m.get("role").and_then(Json::as_str).ok_or("each message needs a `role`")?;
        if !matches!(role, "system" | "user" | "assistant") {
            return Err(format!("unsupported role `{role}`"));
        }
        let text = m.get("content").and_then(message_text).ok_or("message `content` must be text")?;
        out.push(Message::new(role, &text));
    }
    if out.last().is_none_or(|m| m.role == "assistant") {
        return Err("the last message must be from `user` or `system`".into());
    }
    let num = |k: &str| body.get(k).and_then(Json::as_f64);
    let p = GenParams {
        temperature: num("temperature").map_or(defaults.temperature, |v| v as f32),
        top_k: num("top_k").map_or(defaults.top_k, |v| v.max(1.0) as usize),
        top_p: num("top_p").map_or(defaults.top_p, |v| v as f32),
        repeat_penalty: num("repetition_penalty").map_or(defaults.repeat_penalty, |v| v as f32),
        max_tokens: num("max_completion_tokens").or(num("max_tokens")).map_or(defaults.max_tokens, |v| v.max(1.0) as usize),
        seed: num("seed").map_or(defaults.seed, |v| v as u64),
    };
    Ok((out, p, body.get("stream").and_then(Json::as_bool).unwrap_or(false)))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn usage(c: &Completion) -> String {
    format!(
        r#"{{"prompt_tokens":{},"completion_tokens":{},"total_tokens":{}}}"#,
        c.prompt_tokens,
        c.completion_tokens,
        c.prompt_tokens + c.completion_tokens
    )
}

fn chat_completions(srv: &Server, stream: &mut TcpStream, req: &Request) {
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return error(stream, 400, &format!("invalid JSON: {e}")),
    };
    let (messages, params, streaming) = match parse_chat(&body, &GenParams::default()) {
        Ok(v) => v,
        Err(e) => return error(stream, 400, &e),
    };
    let id = format!("chatcmpl-{}-{}", now(), srv.next_id.fetch_add(1, Ordering::Relaxed));
    let created = now();
    let mut model = srv.model.lock().unwrap_or_else(|p| p.into_inner());

    if !streaming {
        match model.generate(&messages, &params, |_| true) {
            Ok(c) => respond(
                stream,
                200,
                "application/json",
                &format!(
                    r#"{{"id":"{id}","object":"chat.completion","created":{created},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":"{}"}}],"usage":{}}}"#,
                    quote(&srv.name),
                    quote(&c.text),
                    c.finish.as_str(),
                    usage(&c)
                ),
            ),
            Err(e) => error(stream, 400, &e),
        }
        return;
    }

    let head = format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n");
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }
    let chunk = |delta: &str, finish: &str, extra: &str| {
        format!(
            "data: {{\"id\":\"{id}\",\"object\":\"chat.completion.chunk\",\"created\":{created},\"model\":{},\"choices\":[{{\"index\":0,\"delta\":{delta},\"finish_reason\":{finish}}}]{extra}}}\n\n",
            quote(&srv.name)
        )
    };
    let mut ok = stream.write_all(chunk(r#"{"role":"assistant","content":""}"#, "null", "").as_bytes()).is_ok();
    let result = model.generate(&messages, &params, |piece| {
        ok = ok && stream.write_all(chunk(&format!(r#"{{"content":{}}}"#, quote(piece)), "null", "").as_bytes()).is_ok();
        ok
    });
    let tail = match result {
        Ok(c) => chunk("{}", &quote(c.finish.as_str()), &format!(r#","usage":{}"#, usage(&c))),
        Err(e) => format!("data: {{\"error\":{{\"message\":{}}}}}\n\n", quote(&e)),
    };
    let _ = stream.write_all(tail.as_bytes()).and_then(|_| stream.write_all(b"data: [DONE]\n\n"));
}

fn handle(srv: &Server, mut stream: TcpStream) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let req = match read_request(&mut stream) {
        Ok(r) => r,
        Err((status, msg)) => return error(&mut stream, status, &msg),
    };
    if req.method == "OPTIONS" {
        return respond(&mut stream, 204, "text/plain", "");
    }
    if req.method == "GET" && (req.path == "/" || req.path == "/index.html") {
        return respond(&mut stream, 200, "text/html; charset=utf-8", CHAT_PAGE);
    }
    if let Some(key) = &srv.api_key {
        let ok = req.header("authorization").and_then(|v| v.strip_prefix("Bearer ")) == Some(key.as_str());
        if !ok {
            return error(&mut stream, 401, "missing or invalid API key");
        }
    }
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => respond(&mut stream, 200, "application/json", r#"{"status":"ok"}"#),
        ("GET", "/v1/models") => respond(
            &mut stream,
            200,
            "application/json",
            &format!(r#"{{"object":"list","data":[{{"id":{},"object":"model","owned_by":"kronos"}}]}}"#, quote(&srv.name)),
        ),
        ("POST", "/v1/chat/completions") => chat_completions(srv, &mut stream, &req),
        _ => error(&mut stream, 404, &format!("no route for {} {}", req.method, req.path)),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!("usage: {} <model_dir> [--host 127.0.0.1] [--port 8080] [--api-key KEY] [--mode p55454] ...", args[0]);
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let num = |name: &str, default: f64| flag(name).map_or(default, |v| v.parse().expect("numeric flag"));
    let model_dir = Path::new(&args[1]);
    let mode = flag("--mode").unwrap_or_else(|| "p55454".to_string());
    let calib = flag("--calib");
    let default_threads = kronos_2t::thread_pool::default_inference_threads() as f64;
    let calib_opt = calib.as_deref().map(|c| (Path::new(c), num("--calib-tokens", 16384.0) as usize));
    let mut model = ChatModel::load(
        model_dir,
        &mode,
        num("--ctx", 2048.0) as usize,
        num("--threads", default_threads) as usize,
        calib_opt,
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if let Some(draft_dir) = flag("--draft") {
        let draft_mode = flag("--draft-mode").unwrap_or_else(|| mode.clone());
        eprintln!("attaching speculative draft {} ({draft_mode})...", draft_dir);
        model.attach_draft(Path::new(&draft_dir), &draft_mode, calib_opt).unwrap_or_else(|e| {
            eprintln!("draft error: {e}");
            std::process::exit(1);
        });
    }
    if args.iter().any(|a| a == "--ane") {
        #[cfg(all(feature = "ane", target_os = "macos"))]
        match model.enable_ane(&model_dir.join("ane")) {
            Ok(b) => eprintln!("Neural Engine prefill buckets: {b:?}"),
            Err(e) => eprintln!("Neural Engine disabled: {e}"),
        }
        #[cfg(not(all(feature = "ane", target_os = "macos")))]
        {
            let _ = &mut model;
            eprintln!("built without the `ane` feature; prefilling on CPU");
        }
    }
    let addr = format!("{}:{}", flag("--host").unwrap_or_else(|| "127.0.0.1".into()), num("--port", 8080.0) as u16);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    let srv = Arc::new(Server {
        name: model.name.clone(),
        model: Mutex::new(model),
        api_key: flag("--api-key"),
        next_id: AtomicU64::new(1),
    });
    eprintln!("serving {} on http://{addr}/v1", srv.name);
    for conn in listener.incoming().flatten() {
        let srv = Arc::clone(&srv);
        std::thread::spawn(move || handle(&srv, conn));
    }
}
