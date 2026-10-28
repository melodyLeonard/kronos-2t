//! OpenAI-compatible HTTP server (chat completions + tool calling) for compiled Kronos models.
//!
//! Usage: `kronos_server <model_dir> [flags]`
//!   `--host H`            bind address (default 127.0.0.1)
//!   `--port P`            (default 18080)
//!   `--api-key KEY`       require `Authorization: Bearer KEY`
//!   `--cache-dir DIR`     optional directory to read/write compiled `.k2tb` caches
//!   `--fp32-kv`           use an `FP32` KV cache instead of the default INT8 one
//!   `--ane`               prefill on the Neural Engine (`ane` feature, models in `<model_dir>/ane`)
//!   `--mode M`, `--calib FILE`, `--calib-tokens N`, `--ctx N`, `--threads N` as in `kronos_chat`
//!
//! Endpoints: `POST /v1/chat/completions` (supports `"tools"`, `"tool_calls"`, `"role": "tool"`, and
//! `"stream": true` server-sent events), `GET /v1/models`, `GET /health`, and a web chat page at `/`.
//! Requests are served one generation at a time.

use kronos_2t::chat::{ChatModel, Completion, GenParams, Message, ToolCall};
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

/// `content` may be a string, `null`, or an array of `{"type":"text","text":...}` parts.
fn message_text(content: Option<&Json>) -> Result<String, String> {
    match content {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => Ok(parts.iter().filter_map(|p| p.get("text").and_then(Json::as_str)).collect()),
        Some(_) => Err("message `content` must be text, array, or null".to_string()),
    }
}

fn parse_tool_calls_array(arr: &[Json]) -> Result<Vec<ToolCall>, String> {
    let mut out = Vec::with_capacity(arr.len());
    for (idx, tc) in arr.iter().enumerate() {
        let id = tc
            .get("id")
            .and_then(Json::as_str)
            .map_or_else(|| format!("call_{idx}"), str::to_string);
        let fn_obj = tc.get("function").unwrap_or(tc);
        let name = fn_obj
            .get("name")
            .and_then(Json::as_str)
            .ok_or("assistant `tool_calls` entry missing `function.name`")?
            .to_string();
        let arguments = match fn_obj.get("arguments") {
            Some(Json::Str(s)) => s.clone(),
            Some(other) => other.stringify(),
            None => "{}".to_string(),
        };
        out.push(ToolCall { id, name, arguments });
    }
    Ok(out)
}

fn parse_chat(
    body: &Json,
    defaults: &GenParams,
) -> Result<(Vec<Message>, Vec<String>, GenParams, bool), String> {
    let msgs = body.get("messages").and_then(Json::as_arr).ok_or("`messages` must be an array")?;
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = m.get("role").and_then(Json::as_str).ok_or("each message needs a `role`")?;
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err(format!("unsupported role `{role}`"));
        }
        let text = message_text(m.get("content"))?;
        match role {
            "assistant" => {
                let tcs = match m.get("tool_calls").and_then(Json::as_arr) {
                    Some(arr) if !arr.is_empty() => parse_tool_calls_array(arr)?,
                    _ => Vec::new(),
                };
                if tcs.is_empty() {
                    out.push(Message::new("assistant", &text));
                } else {
                    out.push(Message::assistant_tool_calls(&text, tcs));
                }
            }
            "tool" => {
                let id = m.get("tool_call_id").and_then(Json::as_str).unwrap_or("call_0");
                out.push(Message::tool_result(id, &text));
            }
            _ => out.push(Message::new(role, &text)),
        }
    }
    if out.last().is_none_or(|m| m.role == "assistant") {
        return Err("the last message must be from `user`, `system`, or `tool`".into());
    }

    let tool_choice_none = body
        .get("tool_choice")
        .and_then(Json::as_str)
        .is_some_and(|s| s == "none");
    let tools_json = if tool_choice_none {
        Vec::new()
    } else {
        body.get("tools")
            .and_then(Json::as_arr)
            .map(|arr| arr.iter().map(Json::stringify).collect())
            .unwrap_or_default()
    };

    let num = |k: &str| body.get(k).and_then(Json::as_f64);
    let p = GenParams {
        temperature: num("temperature").map_or(defaults.temperature, |v| v as f32),
        top_k: num("top_k").map_or(defaults.top_k, |v| v.max(1.0) as usize),
        top_p: num("top_p").map_or(defaults.top_p, |v| v as f32),
        repeat_penalty: num("repetition_penalty").map_or(defaults.repeat_penalty, |v| v as f32),
        max_tokens: num("max_completion_tokens")
            .or(num("max_tokens"))
            .map_or(defaults.max_tokens, |v| v.max(1.0) as usize),
        seed: num("seed").map_or(defaults.seed, |v| v as u64),
    };
    Ok((out, tools_json, p, body.get("stream").and_then(Json::as_bool).unwrap_or(false)))
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

fn format_tool_calls_json(calls: &[ToolCall], include_index: bool) -> String {
    let mut items = Vec::with_capacity(calls.len());
    for (idx, tc) in calls.iter().enumerate() {
        let idx_field = if include_index { format!(r#""index":{idx},"#) } else { String::new() };
        items.push(format!(
            r#"{{{idx_field}"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#,
            quote(&tc.id),
            quote(&tc.name),
            quote(&tc.arguments),
        ));
    }
    format!("[{}]", items.join(","))
}

fn format_completion_response(id: &str, created: u64, model_name: &str, c: &Completion) -> String {
    let msg_json = if c.tool_calls.is_empty() {
        format!(r#"{{"role":"assistant","content":{}}}"#, quote(&c.text))
    } else {
        let content_val = if c.text.is_empty() { "null".to_string() } else { quote(&c.text) };
        format!(
            r#"{{"role":"assistant","content":{content_val},"tool_calls":{}}}"#,
            format_tool_calls_json(&c.tool_calls, false)
        )
    };
    format!(
        r#"{{"id":"{id}","object":"chat.completion","created":{created},"model":{},"choices":[{{"index":0,"message":{msg_json},"finish_reason":"{}"}}],"usage":{}}}"#,
        quote(model_name),
        c.finish.as_str(),
        usage(c)
    )
}

fn chat_completions(srv: &Server, stream: &mut TcpStream, req: &Request) {
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return error(stream, 400, &format!("invalid JSON: {e}")),
    };
    let (messages, tools_json, params, streaming) = match parse_chat(&body, &GenParams::default()) {
        Ok(v) => v,
        Err(e) => return error(stream, 400, &e),
    };
    let id = format!("chatcmpl-{}-{}", now(), srv.next_id.fetch_add(1, Ordering::Relaxed));
    let created = now();
    let mut model = srv.model.lock().unwrap_or_else(|p| p.into_inner());

    if !streaming {
        match model.generate_with_tools(&messages, &tools_json, &params, |_| true) {
            Ok(c) => respond(
                stream,
                200,
                "application/json",
                &format_completion_response(&id, created, &srv.name, &c),
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

    if !tools_json.is_empty() {
        let _ = stream.write_all(chunk(r#"{"role":"assistant","content":null}"#, "null", "").as_bytes());
        match model.generate_with_tools(&messages, &tools_json, &params, |_| true) {
            Ok(c) => {
                if !c.text.is_empty() {
                    let _ = stream.write_all(
                        chunk(&format!(r#"{{"content":{}}}"#, quote(&c.text)), "null", "").as_bytes(),
                    );
                }
                if !c.tool_calls.is_empty() {
                    let _ = stream.write_all(
                        chunk(
                            &format!(r#"{{"tool_calls":{}}}"#, format_tool_calls_json(&c.tool_calls, true)),
                            "null",
                            "",
                        )
                        .as_bytes(),
                    );
                }
                let tail = chunk("{}", &quote(c.finish.as_str()), &format!(r#","usage":{}"#, usage(&c)));
                let _ = stream.write_all(tail.as_bytes()).and_then(|_| stream.write_all(b"data: [DONE]\n\n"));
            }
            Err(e) => {
                let err = format!("data: {{\"error\":{{\"message\":{}}}}}\n\n", quote(&e));
                let _ = stream.write_all(err.as_bytes()).and_then(|_| stream.write_all(b"data: [DONE]\n\n"));
            }
        }
        return;
    }

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
        eprintln!(
            "usage: {} <model_dir> [--cache-dir DIR] [--host 127.0.0.1] [--port 18080] [--api-key KEY] [--mode nf4g128] ...",
            args[0]
        );
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let num = |name: &str, default: f64| flag(name).map_or(default, |v| v.parse().expect("numeric flag"));
    let model_dir = Path::new(&args[1]);
    let mode = flag("--mode").unwrap_or_else(|| "p55454".to_string());
    let calib = flag("--calib");
    let default_threads = kronos_2t::thread_pool::default_inference_threads() as f64;
    let calib_opt = calib.as_deref().map(|c| (Path::new(c), num("--calib-tokens", 16384.0) as usize));
    let cache_dir_flag = flag("--cache-dir");
    let cache_dir_opt = cache_dir_flag.as_deref().map(Path::new);
    let mut model = ChatModel::load_with_cache_dir(
        model_dir,
        cache_dir_opt,
        &mode,
        num("--ctx", 2048.0) as usize,
        num("--threads", default_threads) as usize,
        calib_opt,
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    if args.iter().any(|a| a == "--fp32-kv") {
        model.enable_q8_kv_cache(false);
    } else if args.iter().any(|a| a == "--q8-kv") {
        model.enable_q8_kv_cache(true);
    }
    if let Some(draft_dir) = flag("--draft") {
        let draft_mode = flag("--draft-mode").unwrap_or_else(|| mode.clone());
        eprintln!("attaching speculative draft {} ({draft_mode})...", draft_dir);
        model
            .attach_draft_with_cache_dir(Path::new(&draft_dir), cache_dir_opt, &draft_mode, calib_opt)
            .unwrap_or_else(|e| {
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
    let addr = format!("{}:{}", flag("--host").unwrap_or_else(|| "127.0.0.1".into()), num("--port", 18080.0) as u16);
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    let weights_gib = model.resident_weight_bytes() as f64 / (1u64 << 30) as f64;
    let kv_gib = model.kv_cache_bytes() as f64 / (1u64 << 30) as f64;
    let kv_tag = if model.is_q8_kv_enabled() { "Q8-KV" } else { "FP32-KV" };
    let ctx_sz = model.context_size();
    let srv = Arc::new(Server {
        name: model.name.clone(),
        model: Mutex::new(model),
        api_key: flag("--api-key"),
        next_id: AtomicU64::new(1),
    });
    eprintln!(
        "serving {} on http://{addr}/v1 (weights {weights_gib:.2} GiB + {kv_tag} {kv_gib:.2} GiB @ ctx={ctx_sz})",
        srv.name
    );
    for conn in listener.incoming().flatten() {
        let srv = Arc::clone(&srv);
        std::thread::spawn(move || handle(&srv, conn));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kronos_2t::chat::FinishReason;

    #[test]
    fn test_openai_tool_request_and_response_wire_format() {
        let req_json = Json::parse(
            r#"{
                "model": "qwen2.5-coder",
                "messages": [
                    {"role": "user", "content": "Create index.html"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": "{\"path\":\"index.html\",\"content\":\"<h1>ok</h1>\"}"
                                }
                            }
                        ]
                    },
                    {"role": "tool", "tool_call_id": "call_0", "content": "wrote 11 bytes"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "parameters": {"type": "object"}
                        }
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 1024
            }"#,
        )
        .unwrap();

        let (msgs, tools, params, stream) = parse_chat(&req_json, &GenParams::default()).unwrap();
        assert!(!stream);
        assert_eq!(params.temperature, 0.0);
        assert_eq!(params.max_tokens, 1024);
        assert_eq!(tools.len(), 1);
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1].tool_calls.len(), 1);
        assert_eq!(msgs[1].tool_calls[0].name, "write_file");
        assert_eq!(msgs[1].tool_calls[0].arguments, r#"{"path":"index.html","content":"<h1>ok</h1>"}"#);
        assert_eq!(msgs[2].role, "tool");
        assert_eq!(msgs[2].tool_call_id.as_deref(), Some("call_0"));

        let comp = Completion {
            text: String::new(),
            tool_calls: vec![ToolCall {
                id: "call_0".to_string(),
                name: "write_file".to_string(),
                arguments: r#"{"path":"index.html","content":"<h1>ok</h1>"}"#.to_string(),
            }],
            prompt_tokens: 120,
            prefilled_tokens: 120,
            completion_tokens: 24,
            finish: FinishReason::ToolCalls,
            prefill_secs: 0.2,
            decode_secs: 1.0,
            prefill_unit: "cpu",
        };
        let wire = format_completion_response("chatcmpl-1", 1700000000, "qwen2.5-coder", &comp);
        let parsed = Json::parse(&wire).unwrap();
        let choice0 = &parsed.get("choices").unwrap().as_arr().unwrap()[0];
        assert_eq!(choice0.get("finish_reason").and_then(Json::as_str), Some("tool_calls"));
        let msg = choice0.get("message").unwrap();
        assert_eq!(msg.get("content"), Some(&Json::Null));
        let tc0 = &msg.get("tool_calls").unwrap().as_arr().unwrap()[0];
        assert_eq!(tc0.get("function").unwrap().get("name").and_then(Json::as_str), Some("write_file"));
        assert_eq!(
            tc0.get("function").unwrap().get("arguments").and_then(Json::as_str),
            Some(r#"{"path":"index.html","content":"<h1>ok</h1>"}"#)
        );
    }
}
