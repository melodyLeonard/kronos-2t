//! OpenAI-compatible HTTP server, persistent session store, and sandboxed workspace agent.
//!
//! Usage: `kronos_server <model_dir> [flags]`
//!   `--host H`            bind address (default 127.0.0.1)
//!   `--port P`            (default 18080)
//!   `--api-key KEY`       require `Authorization: Bearer KEY`
//!   `--workdir DIR`       workspace root for web agent tools and `/preview/*` (default temp dir)
//!   `--state-dir DIR`     directory for persistent conversation and project state
//!   `--cache-dir DIR`     optional directory to read/write compiled `.k2tb` caches
//!   `--fp32-kv`           use an `FP32` KV cache instead of the default INT8 one
//!   `--cpu`               force CPU inference (disables default Metal GPU backend for NF4)
//!   `--metal`             require Metal GPU inference
//!   `--draft DIR`         draft model for speculative decoding (`--draft-mode M`)
//!   `--ane`               prefill on the Neural Engine (`ane` feature, models in `<model_dir>/ane`)
//!   `--mode M`, `--calib FILE`, `--calib-tokens N`, `--ctx N`, `--threads N` as in `kronos_chat`
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /v1/agent/chat`, `POST /v1/turn`,
//! `POST /v1/compact`, `GET /v1/sessions`, `GET /v1/sessions/get?id=...`, `POST /v1/sessions/save`,
//! `POST /v1/sessions/delete`, `POST /v1/projects/select`, `POST /v1/workspace/revert`,
//! `POST /v1/screencapture`, `GET /v1/agent/files`, `GET /v1/agent/file?path=...`,
//! `POST /v1/agent/clear`, `GET /preview/<path>`, `GET /v1/models`, `GET /health`,
//! the workspace UI at `/`, and the floating quick-prompt UI at `/quick`.

use kronos_2t::agent::{
    build_unified_system_prompt, classify_turn_intent, compact_conversation_history,
    list_workspace_files, mime_type_for_path, resolve_jailed_path, run_agent_turn,
    serialize_messages_json, tool_definitions_json, truncate_utf8, validate_file, AgentEvent,
    AgentTurnConfig, FileChangeDiff, TurnRoute, AGENT_SYSTEM_PROMPT, MAX_READ_BYTES,
    UNIFIED_SYSTEM_PROMPT,
};
use kronos_2t::attachments::{ingest_attachment, AttachmentIngestion};
use kronos_2t::chat::{ChatModel, Completion, GenParams, Message, ToolCall};
use kronos_2t::device::{global_device_manager, use_computer_tool_definition_json};
use kronos_2t::json::{quote, Json};
use kronos_2t::vision::analyze_image_payload;
use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_BODY: usize = 32 << 20;
const CHAT_PAGE: &str = include_str!("chat.html");
const QUICK_PAGE: &str = include_str!("quick.html");
const MAX_HEADER: usize = 64 << 10;
const MAX_REVERT_SNAPSHOT_BYTES: usize = 128 << 10;

#[derive(Clone, Debug)]
struct SessionSummary {
    id: String,
    title: String,
    workdir: String,
    created_at: u64,
    updated_at: u64,
    context_tokens: usize,
    added_lines: usize,
    removed_lines: usize,
    changed_files: usize,
}

struct UiStateIndex {
    projects: Vec<String>,
    sessions: Vec<SessionSummary>,
}

struct Server {
    model: Mutex<ChatModel>,
    name: String,
    api_key: Option<String>,
    workdir: Mutex<PathBuf>,
    state_dir: PathBuf,
    ui_index: Mutex<UiStateIndex>,
    ctx_limit: usize,
    next_id: AtomicU64,
    media_seq: AtomicU64,
}

impl Server {
    fn active_workdir(&self) -> PathBuf {
        self.workdir.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

struct Request {
    method: String,
    path: String,
    query: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }

    fn query_param(&self, key: &str) -> Option<String> {
        for pair in self.query.split('&') {
            if pair.is_empty() {
                continue;
            }
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            if percent_decode(k) == key {
                return Some(percent_decode(v));
            }
        }
        None
    }
}

fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
        } else if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hi = (bytes[i + 1] as char).to_digit(16);
            let lo = (bytes[i + 2] as char).to_digit(16);
            if let (Some(h), Some(l)) = (hi, lo) {
                out.push(((h << 4) | l) as u8);
                i += 3;
                continue;
            }
            out.push(bytes[i]);
            i += 1;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn encode_base64(bytes: &[u8]) -> String {
    const LUT: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for ch in bytes.chunks(3) {
        let b0 = ch[0];
        let b1 = if ch.len() > 1 { ch[1] } else { 0 };
        let b2 = if ch.len() > 2 { ch[2] } else { 0 };
        s.push(LUT[(b0 >> 2) as usize] as char);
        s.push(LUT[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize] as char);
        s.push(if ch.len() > 1 {
            LUT[(((b1 & 0x0F) << 2) | (b2 >> 6)) as usize] as char
        } else {
            '='
        });
        s.push(if ch.len() > 2 {
            LUT[(b2 & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    s
}

fn read_request(stream: &mut TcpStream) -> Result<Request, (u16, String)> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| (400, e.to_string()))?;
    let mut parts = line.split_whitespace();
    let (method, path, query) = match (parts.next(), parts.next()) {
        (Some(m), Some(target)) => {
            let (p, q) = target.split_once('?').unwrap_or((target, ""));
            (m.to_string(), p.to_string(), q.to_string())
        }
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
    let mut req = Request {
        method,
        path,
        query,
        headers,
        body: Vec::new(),
    };
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
        403 => "Forbidden",
        404 => "Not Found",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        _ => "Internal Server Error",
    }
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Authorization, Content-Type\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n";

fn respond_bytes(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    extra_headers: &str,
    body: &[u8],
) {
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n{extra_headers}{CORS}Connection: close\r\n\r\n",
        reason(status),
        body.len()
    );
    let _ = stream.write_all(head.as_bytes()).and_then(|_| stream.write_all(body));
}

fn respond(stream: &mut TcpStream, status: u16, content_type: &str, body: &str) {
    respond_bytes(stream, status, content_type, "", body.as_bytes());
}

fn error(stream: &mut TcpStream, status: u16, msg: &str) {
    let kind = if status == 401 { "authentication_error" } else { "invalid_request_error" };
    respond(
        stream,
        status,
        "application/json",
        &format!(r#"{{"error":{{"message":{},"type":"{kind}"}}}}"#, quote(msg)),
    );
}

/// `content` may be a string, `null`, or an OpenAI multimodal array (`text` + `image_url` parts).
fn message_text(content: Option<&Json>) -> Result<String, String> {
    match content {
        None | Some(Json::Null) => Ok(String::new()),
        Some(Json::Str(s)) => Ok(s.clone()),
        Some(Json::Arr(parts)) => {
            let mut blocks = Vec::new();
            let mut text_buf = String::new();
            let mut img_idx = 0usize;
            for p in parts {
                let kind = p.get("type").and_then(Json::as_str).unwrap_or("text");
                if kind == "image_url" || p.get("image_url").is_some() {
                    let url_opt = p
                        .get("image_url")
                        .and_then(|u| u.as_str().or_else(|| u.get("url").and_then(Json::as_str)))
                        .or_else(|| p.get("url").and_then(Json::as_str));
                    if let Some(raw_url) = url_opt {
                        img_idx += 1;
                        let perc = analyze_image_payload(raw_url, None, img_idx)
                            .map_err(|e| format!("image #{img_idx}: {e}"))?;
                        blocks.push(perc.to_prompt_context(img_idx));
                    }
                } else if let Some(t) = p.get("text").and_then(Json::as_str) {
                    text_buf.push_str(t);
                }
            }
            if blocks.is_empty() {
                Ok(text_buf)
            } else {
                let mut combined = blocks.join("\n");
                if !text_buf.trim().is_empty() {
                    combined.push('\n');
                    combined.push_str(&text_buf);
                }
                Ok(combined)
            }
        }
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

fn parse_messages_field(body: &Json) -> Result<Vec<Message>, String> {
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
    Ok(out)
}

fn parse_gen_params(body: &Json, defaults: &GenParams) -> GenParams {
    let num = |k: &str| body.get(k).and_then(Json::as_f64);
    GenParams {
        temperature: num("temperature").map_or(defaults.temperature, |v| v as f32),
        top_k: num("top_k").map_or(defaults.top_k, |v| v.max(1.0) as usize),
        top_p: num("top_p").map_or(defaults.top_p, |v| v as f32),
        repeat_penalty: num("repetition_penalty").map_or(defaults.repeat_penalty, |v| v as f32),
        max_tokens: num("max_completion_tokens")
            .or(num("max_tokens"))
            .map_or(defaults.max_tokens, |v| v.max(1.0) as usize),
        seed: num("seed").map_or(defaults.seed, |v| v as u64),
    }
}

fn parse_chat(
    body: &Json,
    defaults: &GenParams,
) -> Result<(Vec<Message>, Vec<String>, GenParams, bool), String> {
    let out = parse_messages_field(body)?;
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
    let p = parse_gen_params(body, defaults);
    Ok((out, tools_json, p, body.get("stream").and_then(Json::as_bool).unwrap_or(false)))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn usage(c: &Completion) -> String {
    format!(
        r#"{{"prompt_tokens":{},"prefilled_tokens":{},"completion_tokens":{},"total_tokens":{},"drafted_tokens":{},"accepted_tokens":{},"prefill_secs":{:.6},"decode_secs":{:.6},"prefill_unit":{}}}"#,
        c.prompt_tokens,
        c.prefilled_tokens,
        c.completion_tokens,
        c.prompt_tokens + c.completion_tokens,
        c.drafted_tokens,
        c.accepted_tokens,
        c.prefill_secs,
        c.decode_secs,
        quote(c.prefill_unit),
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

    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n"
    );
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

fn attach_request_media(
    body: &Json,
    history: &mut [Message],
    workdir: &Path,
    media_seq: &AtomicU64,
) -> Result<(Vec<AttachmentIngestion>, f64), String> {
    let mut raw_entries: Vec<(String, String, String)> = Vec::new();
    if let Some(arr) = body.get("attachments").and_then(Json::as_arr) {
        for item in arr {
            if let Some(raw_str) = item.as_str() {
                raw_entries.push((String::new(), String::new(), raw_str.to_string()));
            } else {
                let name = item.get("name").and_then(Json::as_str).unwrap_or("").to_string();
                let mime = item.get("mime").and_then(Json::as_str).unwrap_or("").to_string();
                let data = item
                    .get("data")
                    .or_else(|| item.get("data_url"))
                    .or_else(|| item.get("url"))
                    .and_then(Json::as_str)
                    .ok_or("each entry in `attachments` requires a `data` string")?
                    .to_string();
                raw_entries.push((name, mime, data));
            }
        }
    }
    if let Some(arr) = body.get("images").and_then(Json::as_arr) {
        for item in arr {
            let raw = item
                .as_str()
                .or_else(|| item.get("data_url").and_then(Json::as_str))
                .or_else(|| item.get("url").and_then(Json::as_str))
                .ok_or("each entry in `images` must be a base64 string or `{data_url}` object")?;
            let name = item.get("name").and_then(Json::as_str).unwrap_or("clipboard.png");
            raw_entries.push((name.to_string(), "image/png".to_string(), raw.to_string()));
        }
    }
    if raw_entries.is_empty() {
        return Ok((Vec::new(), 0.0));
    }

    let t0 = Instant::now();
    let mut ingested = Vec::with_capacity(raw_entries.len());
    let mut blocks = Vec::with_capacity(raw_entries.len());
    let mut all_images = true;
    for (idx, (name, mime, payload)) in raw_entries.iter().enumerate() {
        let slot = media_seq.fetch_add(1, Ordering::Relaxed) as usize;
        let item = ingest_attachment(name, mime, payload, Some(workdir), slot)
            .map_err(|e| format!("attachment #{}: {e}", idx + 1))?;
        if !item.is_image {
            all_images = false;
        }
        blocks.push(item.prompt_context.clone());
        ingested.push(item);
    }

    if let Some(last_user) = history.iter_mut().rev().find(|m| m.role == "user") {
        let fallback_prompt = if all_images {
            "Analyze the attached image concisely, covering visible text, layout structure, and color palette."
        } else {
            "Summarize and analyze the attached file contents concisely."
        };
        let user_prompt = if last_user.content.trim().is_empty() {
            fallback_prompt
        } else {
            last_user.content.as_str()
        };
        last_user.content = format!("{}\n{}", blocks.join("\n"), user_prompt);
    }
    Ok((ingested, t0.elapsed().as_secs_f64() * 1000.0))
}

fn format_diff_json(diff: Option<&FileChangeDiff>) -> String {
    let Some(d) = diff else {
        return "null".to_string();
    };
    let can_revert = d.before_content.len() <= MAX_REVERT_SNAPSHOT_BYTES;
    let before = if can_revert { d.before_content.as_str() } else { "" };
    format!(
        r#"{{"path":{},"existed_before":{},"can_revert":{can_revert},"before_content":{},"added_lines":{},"removed_lines":{},"unified_diff":{}}}"#,
        quote(&d.path),
        d.existed_before,
        quote(before),
        d.added_lines,
        d.removed_lines,
        quote(&d.unified_diff),
    )
}

struct StreamTurnSpec {
    history: Vec<Message>,
    params: GenParams,
    max_steps: usize,
    cmd_timeout: u64,
    initial_route: TurnRoute,
    workdir: PathBuf,
    attachments: Vec<AttachmentIngestion>,
    media_ms: f64,
}

fn stream_agent_loop_sse(srv: &Server, stream: &mut TcpStream, spec: StreamTurnSpec) {
    let StreamTurnSpec {
        mut history,
        params,
        max_steps,
        cmd_timeout,
        initial_route,
        workdir,
        attachments,
        media_ms,
    } = spec;
    let head = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n{CORS}Connection: close\r\n\r\n"
    );
    if stream.write_all(head.as_bytes()).is_err() {
        return;
    }

    let mut write_sse = |payload: &str| -> bool {
        let frame = format!("data: {payload}\n\n");
        stream.write_all(frame.as_bytes()).and_then(|_| stream.flush()).is_ok()
    };

    let mut active_route = initial_route;
    let _ = write_sse(&format!(r#"{{"type":"route","mode":{}}}"#, quote(active_route.as_str())));

    if !attachments.is_empty() {
        let items_json: Vec<String> = attachments
            .iter()
            .map(|a| {
                format!(
                    r#"{{"kind":{},"name":{},"badge":{},"saved_path":{},"is_image":{},"elapsed_ms":{:.1}}}"#,
                    quote(&a.kind),
                    quote(&a.name),
                    quote(&a.badge),
                    quote(a.saved_rel_path.as_deref().unwrap_or("")),
                    a.is_image,
                    a.elapsed_ms,
                )
            })
            .collect();
        let _ = write_sse(&format!(
            r#"{{"type":"attachments","count":{},"elapsed_ms":{:.1},"items":[{}]}}"#,
            attachments.len(),
            media_ms,
            items_json.join(",")
        ));

        let vision_badges: Vec<String> = attachments
            .iter()
            .filter(|a| a.is_image)
            .map(|a| quote(a.badge.strip_prefix("vision · ").unwrap_or(&a.badge)))
            .collect();
        if !vision_badges.is_empty() {
            let _ = write_sse(&format!(
                r#"{{"type":"vision","count":{},"elapsed_ms":{:.1},"badges":[{}]}}"#,
                vision_badges.len(),
                media_ms,
                vision_badges.join(",")
            ));
        }
    }

    if history.first().is_none_or(|m| m.role != "system") {
        let sys = build_unified_system_prompt(&workdir);
        history.insert(0, Message::new("system", &sys));
    } else if history[0].content.starts_with(UNIFIED_SYSTEM_PROMPT)
        || history[0].content.starts_with(AGENT_SYSTEM_PROMPT)
        || history[0].content.starts_with("You are Kronos")
        || history[0].content.starts_with("Your name is Kronos")
    {
        history[0].content = build_unified_system_prompt(&workdir);
    }

    let mut effective_route = initial_route;
    let last_user_text = history
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    if !last_user_text.contains("[Device Context") {
        if let Some(dev_ctx) = global_device_manager().query_for_turn(&last_user_text) {
            if dev_ctx.wants_computer_action && effective_route == TurnRoute::Chat {
                effective_route = TurnRoute::Computer;
                active_route = TurnRoute::Computer;
                let _ = write_sse(r#"{"type":"route","mode":"computer"}"#);
            }
            let items_json: Vec<String> = dev_ctx
                .hits
                .iter()
                .take(6)
                .map(|h| {
                    format!(
                        r#"{{"kind":{},"name":{},"path":{},"exists_on_disk":{}}}"#,
                        quote(h.kind.label()),
                        quote(&h.name),
                        quote(&h.path),
                        h.exists_on_disk
                    )
                })
                .collect();
            let _ = write_sse(&format!(
                r#"{{"type":"device_context","hits":{},"elapsed_ms":{:.2},"items":[{}]}}"#,
                dev_ctx.hits.len(),
                dev_ctx.elapsed_ms,
                items_json.join(",")
            ));
            if let Some(last_u) = history.iter_mut().rev().find(|m| m.role == "user") {
                last_u.content = format!("{}\n\n{}", dev_ctx.prompt_block, last_u.content);
            }
        }
    }

    let tools_json = match effective_route {
        TurnRoute::Chat => Vec::new(),
        TurnRoute::Computer => vec![use_computer_tool_definition_json()],
        TurnRoute::Agent => {
            let mut t = tool_definitions_json();
            if global_device_manager().current_config().allow_computer_control {
                t.push(use_computer_tool_definition_json());
            }
            t
        }
    };
    let cfg = AgentTurnConfig {
        workdir,
        params,
        max_steps,
        cmd_timeout_secs: cmd_timeout,
        expect_tool_call: matches!(effective_route, TurnRoute::Agent | TurnRoute::Computer),
    };

    let mut model = srv.model.lock().unwrap_or_else(|p| p.into_inner());
    let run_res = run_agent_turn(&mut model, &mut history, &tools_json, &cfg, |ev| match ev {
        AgentEvent::ContextCompacted {
            before_tokens,
            after_tokens,
            compacted_messages,
        } => write_sse(&format!(
            r#"{{"type":"compacted","before_tokens":{before_tokens},"after_tokens":{after_tokens},"compacted_messages":{compacted_messages}}}"#
        )),
        AgentEvent::StepStart { step, max_steps } => write_sse(&format!(
            r#"{{"type":"step_start","step":{step},"max_steps":{max_steps}}}"#
        )),
        AgentEvent::TextDelta(delta) => {
            if delta.is_empty() {
                true
            } else {
                write_sse(&format!(r#"{{"type":"text","delta":{}}}"#, quote(delta)))
            }
        }
        AgentEvent::ToolProgress {
            step,
            name,
            target,
            bytes,
        } => {
            let next_r = if name == "use_computer" {
                TurnRoute::Computer
            } else {
                TurnRoute::Agent
            };
            if active_route == TurnRoute::Chat {
                active_route = next_r;
                let _ = write_sse(&format!(r#"{{"type":"route","mode":{}}}"#, quote(active_route.as_str())));
            }
            write_sse(&format!(
                r#"{{"type":"tool_progress","step":{step},"name":{},"target":{},"bytes":{bytes}}}"#,
                quote(name),
                quote(target),
            ))
        }
        AgentEvent::StepStats { step, completion } => write_sse(&format!(
            r#"{{"type":"step_stats","step":{step},"usage":{}}}"#,
            usage(completion)
        )),
        AgentEvent::ToolStart { step, call } => {
            let next_r = if call.name == "use_computer" {
                TurnRoute::Computer
            } else {
                TurnRoute::Agent
            };
            if active_route == TurnRoute::Chat {
                active_route = next_r;
                let _ = write_sse(&format!(r#"{{"type":"route","mode":{}}}"#, quote(active_route.as_str())));
            }
            write_sse(&format!(
                r#"{{"type":"tool_start","step":{step},"id":{},"name":{},"arguments":{}}}"#,
                quote(&call.id),
                quote(&call.name),
                quote(&call.arguments),
            ))
        }
        AgentEvent::ToolDone { step, call, outcome } => write_sse(&format!(
            r#"{{"type":"tool_done","step":{step},"id":{},"name":{},"ok":{},"validation_ok":{},"validation_msg":{},"output":{},"diff":{}}}"#,
            quote(&call.id),
            quote(&call.name),
            outcome.ok,
            outcome.validation_ok,
            quote(&outcome.validation_msg),
            quote(&truncate_utf8(&outcome.output, 4096)),
            format_diff_json(outcome.diff.as_ref()),
        )),
        AgentEvent::ValidationRepair { step, report } => write_sse(&format!(
            r#"{{"type":"validation_repair","step":{step},"report":{}}}"#,
            quote(report)
        )),
        AgentEvent::Verified { step, report } => {
            let details: Vec<String> = report.details.iter().map(|d| quote(d)).collect();
            write_sse(&format!(
                r#"{{"type":"verified","step":{step},"checks_run":{},"elapsed_ms":{:.1},"summary":{},"details":[{}]}}"#,
                report.checks_run,
                report.elapsed_ms,
                quote(&report.summary),
                details.join(",")
            ))
        }
        AgentEvent::Done { .. } => true,
    });

    match run_res {
        Ok(final_text) => {
            let ctx_tokens = model.count_prompt_tokens(&history, &tools_json);
            let ctx_limit = model.context_size();
            let done_json = format!(
                r#"{{"type":"done","route":{},"context_tokens":{ctx_tokens},"context_limit":{ctx_limit},"text":{},"messages":{}}}"#,
                quote(active_route.as_str()),
                quote(&final_text),
                serialize_messages_json(&history),
            );
            let _ = write_sse(&done_json);
        }
        Err(e) => {
            let err_json = format!(r#"{{"type":"error","message":{}}}"#, quote(&e));
            let _ = write_sse(&err_json);
        }
    }
    let _ = stream.write_all(b"data: [DONE]\n\n");
}

fn agent_chat_sse(srv: &Server, stream: &mut TcpStream, req: &Request) {
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return error(stream, 400, &format!("invalid JSON: {e}")),
    };
    let mut history = match parse_messages_field(&body) {
        Ok(m) => m,
        Err(e) => return error(stream, 400, &e),
    };
    let workdir = srv.active_workdir();
    let (attachments, media_ms) =
        match attach_request_media(&body, &mut history, &workdir, &srv.media_seq) {
            Ok(v) => v,
            Err(e) => return error(stream, 400, &e),
        };
    let agent_defaults = GenParams {
        temperature: 0.0,
        top_k: 40,
        top_p: 0.9,
        repeat_penalty: 1.05,
        max_tokens: 4096,
        seed: 0,
    };
    let params = parse_gen_params(&body, &agent_defaults);
    let max_steps = body
        .get("max_steps")
        .and_then(Json::as_f64)
        .map_or(10, |v| (v as usize).clamp(1, 24));
    let cmd_timeout = body
        .get("cmd_timeout")
        .and_then(Json::as_f64)
        .map_or(20, |v| (v as u64).clamp(1, 120));
    stream_agent_loop_sse(
        srv,
        stream,
        StreamTurnSpec {
            history,
            params,
            max_steps,
            cmd_timeout,
            initial_route: TurnRoute::Agent,
            workdir,
            attachments,
            media_ms,
        },
    );
}

fn unified_turn_sse(srv: &Server, stream: &mut TcpStream, req: &Request) {
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return error(stream, 400, &format!("invalid JSON: {e}")),
    };
    let mut history = match parse_messages_field(&body) {
        Ok(m) => m,
        Err(e) => return error(stream, 400, &e),
    };
    let raw_user_text = history
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let workdir = srv.active_workdir();
    let (attachments, media_ms) =
        match attach_request_media(&body, &mut history, &workdir, &srv.media_seq) {
            Ok(v) => v,
            Err(e) => return error(stream, 400, &e),
        };
    let workspace_has_files = !list_workspace_files(&workdir)
        .into_iter()
        .filter(|e| !e.path.starts_with(".kronos_media/"))
        .count()
        .eq(&0);
    let history_used_tools = history.iter().any(|m| m.role == "tool" || !m.tool_calls.is_empty());
    let has_images = attachments.iter().any(|a| a.is_image);
    let route = match body.get("route").and_then(Json::as_str).unwrap_or("auto") {
        "chat" => TurnRoute::Chat,
        "agent" => TurnRoute::Agent,
        "computer" => TurnRoute::Computer,
        _ => classify_turn_intent(
            &raw_user_text,
            has_images,
            workspace_has_files,
            history_used_tools,
        ),
    };
    let defaults = match route {
        TurnRoute::Chat => GenParams {
            temperature: 0.35,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.06,
            max_tokens: 1536,
            seed: 0,
        },
        TurnRoute::Computer => GenParams {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.05,
            max_tokens: 1024,
            seed: 0,
        },
        TurnRoute::Agent => GenParams {
            temperature: 0.0,
            top_k: 40,
            top_p: 0.9,
            repeat_penalty: 1.05,
            max_tokens: 4096,
            seed: 0,
        },
    };
    let params = parse_gen_params(&body, &defaults);
    let default_steps = match route {
        TurnRoute::Chat => 4,
        TurnRoute::Computer => 4,
        TurnRoute::Agent => 10,
    };
    let max_steps = body
        .get("max_steps")
        .and_then(Json::as_f64)
        .map_or(default_steps, |v| (v as usize).clamp(1, 24));
    let cmd_timeout = body
        .get("cmd_timeout")
        .and_then(Json::as_f64)
        .map_or(20, |v| (v as u64).clamp(1, 120));
    stream_agent_loop_sse(
        srv,
        stream,
        StreamTurnSpec {
            history,
            params,
            max_steps,
            cmd_timeout,
            initial_route: route,
            workdir,
            attachments,
            media_ms,
        },
    );
}

fn compact_history_http(srv: &Server, stream: &mut TcpStream, req: &Request) {
    let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
        Ok(b) => b,
        Err(e) => return error(stream, 400, &format!("invalid JSON: {e}")),
    };
    let Some(msgs_arr) = body.get("messages").and_then(Json::as_arr) else {
        return error(stream, 400, "`messages` must be an array");
    };
    if msgs_arr.is_empty() {
        return respond(
            stream,
            200,
            "application/json",
            &format!(
                r#"{{"compacted":false,"before_tokens":0,"after_tokens":0,"context_limit":{},"compacted_messages":0,"messages":[]}}"#,
                srv.ctx_limit
            ),
        );
    }
    let mut history = Vec::with_capacity(msgs_arr.len());
    for m in msgs_arr {
        let Some(role) = m.get("role").and_then(Json::as_str) else {
            return error(stream, 400, "each message needs a `role`");
        };
        let text = match message_text(m.get("content")) {
            Ok(t) => t,
            Err(e) => return error(stream, 400, &e),
        };
        if role == "assistant" {
            let tcs = match m.get("tool_calls").and_then(Json::as_arr) {
                Some(arr) if !arr.is_empty() => match parse_tool_calls_array(arr) {
                    Ok(v) => v,
                    Err(e) => return error(stream, 400, &e),
                },
                _ => Vec::new(),
            };
            if tcs.is_empty() {
                history.push(Message::new("assistant", &text));
            } else {
                history.push(Message::assistant_tool_calls(&text, tcs));
            }
        } else if role == "tool" {
            let id = m.get("tool_call_id").and_then(Json::as_str).unwrap_or("call_0");
            history.push(Message::tool_result(id, &text));
        } else {
            history.push(Message::new(role, &text));
        }
    }
    let workdir = srv.active_workdir();
    if history.first().is_none_or(|m| m.role != "system") {
        let sys = build_unified_system_prompt(&workdir);
        history.insert(0, Message::new("system", &sys));
    }
    let tools_json = tool_definitions_json();
    let model = srv.model.lock().unwrap_or_else(|p| p.into_inner());
    let rep = compact_conversation_history(&model, &mut history, &tools_json, &workdir, true);
    let ctx_limit = model.context_size();
    drop(model);
    respond(
        stream,
        200,
        "application/json",
        &format!(
            r#"{{"compacted":{},"before_tokens":{},"after_tokens":{},"context_limit":{ctx_limit},"compacted_messages":{},"messages":{}}}"#,
            rep.compacted,
            rep.before_tokens,
            rep.after_tokens,
            rep.compacted_messages,
            serialize_messages_json(&history),
        ),
    );
}

fn sanitize_session_id(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.len() > 80 {
        return Err("session `id` must be 1..=80 characters".to_string());
    }
    if !trimmed
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("session `id` may only contain ASCII alphanumeric, `_`, or `-`".to_string());
    }
    Ok(trimmed.to_string())
}

fn parse_session_summary(j: &Json) -> Option<SessionSummary> {
    let id = sanitize_session_id(j.get("id")?.as_str()?).ok()?;
    let title = j
        .get("title")
        .and_then(Json::as_str)
        .unwrap_or("Untitled Conversation")
        .to_string();
    let workdir = j
        .get("workdir")
        .and_then(Json::as_str)
        .unwrap_or("")
        .to_string();
    let created_at = j.get("created_at").and_then(Json::as_f64).unwrap_or(0.0) as u64;
    let updated_at = j
        .get("updated_at")
        .and_then(Json::as_f64)
        .map_or(created_at, |v| v as u64);
    let context_tokens = j.get("context_tokens").and_then(Json::as_f64).unwrap_or(0.0) as usize;
    let added_lines = j.get("added_lines").and_then(Json::as_f64).unwrap_or(0.0) as usize;
    let removed_lines = j.get("removed_lines").and_then(Json::as_f64).unwrap_or(0.0) as usize;
    let changed_files = j.get("changed_files").and_then(Json::as_f64).unwrap_or(0.0) as usize;
    Some(SessionSummary {
        id,
        title,
        workdir,
        created_at,
        updated_at,
        context_tokens,
        added_lines,
        removed_lines,
        changed_files,
    })
}

fn persist_projects_file(state_dir: &Path, projects: &[String]) {
    let items: Vec<String> = projects.iter().map(|p| quote(p)).collect();
    let payload = format!("[{}]\n", items.join(","));
    let _ = fs::write(state_dir.join("projects.json"), payload);
}

fn load_ui_state_index(state_dir: &Path, initial_workdir: &Path) -> UiStateIndex {
    let _ = fs::create_dir_all(state_dir.join("sessions"));
    let mut projects = Vec::new();
    let init_str = initial_workdir.to_string_lossy().into_owned();
    if !init_str.is_empty() {
        projects.push(init_str);
    }
    if let Ok(raw) = fs::read_to_string(state_dir.join("projects.json")) {
        if let Ok(parsed) = Json::parse(&raw) {
            if let Some(arr) = parsed.as_arr() {
                for item in arr {
                    if let Some(p) = item.as_str() {
                        if !p.is_empty() && !projects.iter().any(|x| x == p) {
                            projects.push(p.to_string());
                        }
                    }
                }
            }
        }
    }
    let mut sessions = Vec::new();
    if let Ok(rd) = fs::read_dir(state_dir.join("sessions")) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(raw) = fs::read_to_string(&path) {
                if let Ok(parsed) = Json::parse(&raw) {
                    if let Some(sum) = parse_session_summary(&parsed) {
                        if !sum.workdir.is_empty() && !projects.iter().any(|p| p == &sum.workdir) {
                            projects.push(sum.workdir.clone());
                        }
                        sessions.push(sum);
                    }
                }
            }
        }
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    persist_projects_file(state_dir, &projects);
    UiStateIndex { projects, sessions }
}

fn format_sessions_list_json(active_workdir: &Path, idx: &UiStateIndex) -> String {
    let proj_items: Vec<String> = idx.projects.iter().map(|p| quote(p)).collect();
    let sess_items: Vec<String> = idx
        .sessions
        .iter()
        .map(|s| {
            format!(
                r#"{{"id":{},"title":{},"workdir":{},"created_at":{},"updated_at":{},"context_tokens":{},"added_lines":{},"removed_lines":{},"changed_files":{}}}"#,
                quote(&s.id),
                quote(&s.title),
                quote(&s.workdir),
                s.created_at,
                s.updated_at,
                s.context_tokens,
                s.added_lines,
                s.removed_lines,
                s.changed_files,
            )
        })
        .collect();
    format!(
        r#"{{"active_workdir":{},"projects":[{}],"sessions":[{}]}}"#,
        quote(&active_workdir.to_string_lossy()),
        proj_items.join(","),
        sess_items.join(","),
    )
}

fn save_session_record(
    state_dir: &Path,
    idx: &mut UiStateIndex,
    raw_body: &[u8],
) -> Result<String, String> {
    let body_str = std::str::from_utf8(raw_body).map_err(|e| format!("invalid UTF-8: {e}"))?;
    let parsed = Json::parse(body_str).map_err(|e| format!("invalid JSON: {e}"))?;
    let mut summary = parse_session_summary(&parsed)
        .ok_or("session payload requires a valid `id` string")?;
    if summary.updated_at == 0 {
        summary.updated_at = now();
    }
    if summary.created_at == 0 {
        summary.created_at = summary.updated_at;
    }
    let sess_dir = state_dir.join("sessions");
    fs::create_dir_all(&sess_dir).map_err(|e| format!("create sessions dir: {e}"))?;
    let tmp_path = sess_dir.join(format!("{}.json.tmp", summary.id));
    let dst_path = sess_dir.join(format!("{}.json", summary.id));
    fs::write(&tmp_path, raw_body).map_err(|e| format!("write session: {e}"))?;
    fs::rename(&tmp_path, &dst_path).map_err(|e| format!("commit session: {e}"))?;

    if !summary.workdir.is_empty() && !idx.projects.iter().any(|p| p == &summary.workdir) {
        idx.projects.push(summary.workdir.clone());
        persist_projects_file(state_dir, &idx.projects);
    }
    let saved_id = summary.id.clone();
    if let Some(existing) = idx.sessions.iter_mut().find(|s| s.id == summary.id) {
        *existing = summary;
    } else {
        idx.sessions.push(summary);
    }
    idx.sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(saved_id)
}

fn delete_session_record(
    state_dir: &Path,
    idx: &mut UiStateIndex,
    raw_id: &str,
) -> Result<(), String> {
    let id = sanitize_session_id(raw_id)?;
    let dst_path = state_dir.join("sessions").join(format!("{id}.json"));
    let _ = fs::remove_file(dst_path);
    idx.sessions.retain(|s| s.id != id);
    Ok(())
}

fn expand_tilde_path(raw: &str) -> PathBuf {
    let trimmed = raw.trim();
    if let Some(rest) = trimmed.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    } else if trimmed == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(trimmed)
}

fn revert_workspace_change(
    workdir: &Path,
    rel_path: &str,
    existed_before: bool,
    before_content: &str,
) -> Result<String, (u16, String)> {
    let target = resolve_jailed_path(workdir, rel_path).map_err(|e| (400, e))?;
    if !existed_before {
        if target.is_file() {
            fs::remove_file(&target)
                .map_err(|e| (500, format!("remove `{rel_path}`: {e}")))?;
        }
        return Ok(format!(
            r#"{{"status":"ok","path":{},"action":"removed","validation_ok":true,"validation_msg":"removed"}}"#,
            quote(rel_path)
        ));
    }
    if let Some(parent) = target.parent() {
        fs::create_dir_all(parent).map_err(|e| (500, e.to_string()))?;
    }
    fs::write(&target, before_content)
        .map_err(|e| (500, format!("restore `{rel_path}`: {e}")))?;
    let (v_ok, v_msg) = match validate_file(workdir, &target, true) {
        Ok(note) => (true, if note.is_empty() { "ok".to_string() } else { note }),
        Err(err) => (false, err),
    };
    Ok(format!(
        r#"{{"status":"ok","path":{},"action":"restored","validation_ok":{v_ok},"validation_msg":{}}}"#,
        quote(rel_path),
        quote(&v_msg),
    ))
}

fn handle_screencapture(stream: &mut TcpStream, req: &Request) {
    let is_loopback = stream.peer_addr().map(|a| a.ip().is_loopback()).unwrap_or(false);
    if !is_loopback {
        return error(stream, 403, "screen capture is restricted to local loopback connections");
    }
    let mode = std::str::from_utf8(&req.body)
        .ok()
        .and_then(|s| Json::parse(s).ok())
        .and_then(|j| j.get("mode").and_then(Json::as_str).map(str::to_string))
        .unwrap_or_else(|| "interactive".to_string());
    let stamp = now();
    let tmp = std::env::temp_dir().join(format!(
        "kronos_shot_{}_{}.png",
        std::process::id(),
        stamp
    ));
    let _ = fs::remove_file(&tmp);
    let mut cmd = Command::new("/usr/sbin/screencapture");
    if mode != "full" {
        cmd.arg("-i");
    }
    cmd.arg("-x").arg(&tmp);
    let status = match cmd.status() {
        Ok(s) => s,
        Err(e) => return error(stream, 500, &format!("screencapture failed: {e}")),
    };
    if !status.success() || !tmp.is_file() {
        let _ = fs::remove_file(&tmp);
        return respond(stream, 200, "application/json", r#"{"status":"cancelled"}"#);
    }
    let bytes = match fs::read(&tmp) {
        Ok(b) if !b.is_empty() => b,
        _ => {
            let _ = fs::remove_file(&tmp);
            return respond(stream, 200, "application/json", r#"{"status":"cancelled"}"#);
        }
    };
    let _ = fs::remove_file(&tmp);
    let data_url = format!("data:image/png;base64,{}", encode_base64(&bytes));
    let fname = format!("screenshot_{stamp}.png");
    respond(
        stream,
        200,
        "application/json",
        &format!(
            r#"{{"status":"ok","name":{},"data":{}}}"#,
            quote(&fname),
            quote(&data_url)
        ),
    );
}

#[cfg(test)]
fn format_agent_files_json(workdir: &Path) -> String {
    format_agent_files_with_ctx(workdir, 16384)
}

fn format_agent_files_with_ctx(workdir: &Path, ctx_limit: usize) -> String {
    let entries: Vec<_> = list_workspace_files(workdir)
        .into_iter()
        .filter(|e| !e.path.starts_with(".kronos_media/"))
        .collect();
    let has_index = entries.iter().any(|e| e.path == "index.html")
        || entries.iter().any(|e| e.path.ends_with(".html"));
    let preview_entry = entries
        .iter()
        .find(|e| e.path == "index.html")
        .or_else(|| entries.iter().find(|e| e.path.ends_with(".html")))
        .map(|e| e.path.as_str())
        .unwrap_or("");
    let items: Vec<String> = entries
        .iter()
        .map(|e| {
            format!(
                r#"{{"path":{},"bytes":{},"validation_ok":{},"validation_msg":{}}}"#,
                quote(&e.path),
                e.bytes,
                e.validation_ok,
                quote(&e.validation_msg),
            )
        })
        .collect();
    format!(
        r#"{{"workdir":{},"context_limit":{ctx_limit},"has_html":{has_index},"preview_path":{},"files":[{}]}}"#,
        quote(&workdir.to_string_lossy()),
        quote(preview_entry),
        items.join(",")
    )
}

fn format_agent_file_detail_json(workdir: &Path, rel_path: &str) -> Result<String, (u16, String)> {
    let target = resolve_jailed_path(workdir, rel_path).map_err(|e| (400, e))?;
    if !target.is_file() {
        return Err((404, format!("file `{rel_path}` not found")));
    }
    let bytes = fs::read(&target).map_err(|e| (500, format!("read `{rel_path}`: {e}")))?;
    let total_bytes = bytes.len();
    let content = truncate_utf8(&String::from_utf8_lossy(&bytes), MAX_READ_BYTES * 2);
    let (v_ok, v_msg) = match validate_file(workdir, &target, true) {
        Ok(note) => (true, if note.is_empty() { "ok".to_string() } else { note }),
        Err(err) => (false, err),
    };
    Ok(format!(
        r#"{{"path":{},"bytes":{total_bytes},"validation_ok":{v_ok},"validation_msg":{},"content":{}}}"#,
        quote(rel_path),
        quote(&v_msg),
        quote(&content),
    ))
}

fn clear_workspace_dir(workdir: &Path) -> Result<usize, String> {
    let Ok(rd) = fs::read_dir(workdir) else {
        return Ok(0);
    };
    let mut removed = 0usize;
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name == ".git" {
            continue;
        }
        let p = entry.path();
        if p.is_dir() {
            fs::remove_dir_all(&p).map_err(|e| e.to_string())?;
        } else {
            fs::remove_file(&p).map_err(|e| e.to_string())?;
        }
        removed += 1;
    }
    Ok(removed)
}

fn serve_preview(workdir: &Path, stream: &mut TcpStream, req_path: &str) {
    let raw_rel = req_path
        .strip_prefix("/preview")
        .unwrap_or("")
        .trim_start_matches('/');
    let decoded = percent_decode(raw_rel);
    let rel = if decoded.is_empty() { "index.html" } else { decoded.as_str() };
    let mut target = match resolve_jailed_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return error(stream, 400, &e),
    };
    if target.is_dir() {
        target = target.join("index.html");
    }
    match fs::read(&target) {
        Ok(bytes) => {
            let mime = mime_type_for_path(&target);
            respond_bytes(
                stream,
                200,
                mime,
                "Cache-Control: no-store, no-cache, must-revalidate\r\nPragma: no-cache\r\nExpires: 0\r\n",
                &bytes,
            );
        }
        Err(_) => respond(
            stream,
            404,
            "text/html; charset=utf-8",
            "<!doctype html><meta charset=\"utf-8\"><body style=\"font:14px sans-serif;color:#78716c;padding:24px\">No preview file generated yet.</body>",
        ),
    }
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
    if req.method == "GET" && req.path == "/quick" {
        return respond(&mut stream, 200, "text/html; charset=utf-8", QUICK_PAGE);
    }
    if req.method == "GET" && (req.path == "/preview" || req.path.starts_with("/preview/")) {
        let wd = srv.active_workdir();
        return serve_preview(&wd, &mut stream, &req.path);
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
            &format!(
                r#"{{"object":"list","data":[{{"id":{},"object":"model","owned_by":"kronos"}}]}}"#,
                quote(&srv.name)
            ),
        ),
        ("POST", "/v1/chat/completions") => chat_completions(srv, &mut stream, &req),
        ("POST", "/v1/agent/chat") => agent_chat_sse(srv, &mut stream, &req),
        ("POST", "/v1/turn") => unified_turn_sse(srv, &mut stream, &req),
        ("POST", "/v1/compact") => compact_history_http(srv, &mut stream, &req),
        ("GET", "/v1/sessions") => {
            let wd = srv.active_workdir();
            let idx = srv.ui_index.lock().unwrap_or_else(|p| p.into_inner());
            let body = format_sessions_list_json(&wd, &idx);
            drop(idx);
            respond(&mut stream, 200, "application/json", &body);
        }
        ("GET", "/v1/sessions/get") => {
            let Some(raw_id) = req.query_param("id") else {
                return error(&mut stream, 400, "missing `id` query parameter");
            };
            let id = match sanitize_session_id(&raw_id) {
                Ok(v) => v,
                Err(e) => return error(&mut stream, 400, &e),
            };
            let path = srv.state_dir.join("sessions").join(format!("{id}.json"));
            match fs::read(&path) {
                Ok(bytes) => respond_bytes(&mut stream, 200, "application/json", "", &bytes),
                Err(_) => error(&mut stream, 404, "session not found"),
            }
        }
        ("POST", "/v1/sessions/save") => {
            let mut idx = srv.ui_index.lock().unwrap_or_else(|p| p.into_inner());
            match save_session_record(&srv.state_dir, &mut idx, &req.body) {
                Ok(id) => {
                    drop(idx);
                    respond(
                        &mut stream,
                        200,
                        "application/json",
                        &format!(r#"{{"status":"ok","id":{}}}"#, quote(&id)),
                    );
                }
                Err(e) => error(&mut stream, 400, &e),
            }
        }
        ("POST", "/v1/sessions/delete") => {
            let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
                Ok(b) => b,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            let Some(id) = body.get("id").and_then(Json::as_str) else {
                return error(&mut stream, 400, "missing `id` field");
            };
            let mut idx = srv.ui_index.lock().unwrap_or_else(|p| p.into_inner());
            match delete_session_record(&srv.state_dir, &mut idx, id) {
                Ok(()) => {
                    drop(idx);
                    respond(&mut stream, 200, "application/json", r#"{"status":"ok"}"#);
                }
                Err(e) => error(&mut stream, 400, &e),
            }
        }
        ("POST", "/v1/projects/select") => {
            let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
                Ok(b) => b,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            let Some(raw_wd) = body.get("workdir").and_then(Json::as_str) else {
                return error(&mut stream, 400, "missing `workdir` field");
            };
            let create = body.get("create").and_then(Json::as_bool).unwrap_or(true);
            let candidate = expand_tilde_path(raw_wd);
            if candidate.as_os_str().is_empty() {
                return error(&mut stream, 400, "`workdir` cannot be empty");
            }
            if create {
                if let Err(e) = fs::create_dir_all(&candidate) {
                    return error(&mut stream, 400, &format!("create workdir: {e}"));
                }
            } else if !candidate.is_dir() {
                return error(&mut stream, 404, "project directory does not exist");
            }
            let canon = match candidate.canonicalize() {
                Ok(c) if c.is_dir() => c,
                _ => return error(&mut stream, 400, "workdir is not an accessible directory"),
            };
            {
                let mut wd_guard = srv.workdir.lock().unwrap_or_else(|p| p.into_inner());
                *wd_guard = canon.clone();
            }
            {
                let mut idx = srv.ui_index.lock().unwrap_or_else(|p| p.into_inner());
                let canon_str = canon.to_string_lossy().into_owned();
                if !idx.projects.iter().any(|p| p == &canon_str) {
                    idx.projects.push(canon_str);
                    persist_projects_file(&srv.state_dir, &idx.projects);
                }
            }
            respond(
                &mut stream,
                200,
                "application/json",
                &format_agent_files_with_ctx(&canon, srv.ctx_limit),
            );
        }
        ("POST", "/v1/workspace/revert") => {
            let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
                Ok(b) => b,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            let Some(path) = body.get("path").and_then(Json::as_str) else {
                return error(&mut stream, 400, "missing `path` field");
            };
            let existed_before = body.get("existed_before").and_then(Json::as_bool).unwrap_or(false);
            let before_content = body.get("before_content").and_then(Json::as_str).unwrap_or("");
            let wd = srv.active_workdir();
            match revert_workspace_change(&wd, path, existed_before, before_content) {
                Ok(json_body) => respond(&mut stream, 200, "application/json", &json_body),
                Err((code, msg)) => error(&mut stream, code, &msg),
            }
        }
        ("POST", "/v1/screencapture") => handle_screencapture(&mut stream, &req),
        ("GET", "/v1/device/status") => {
            respond(
                &mut stream,
                200,
                "application/json",
                &global_device_manager().status_json(),
            );
        }
        ("POST", "/v1/device/config") => {
            let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
                Ok(b) => b,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            let en = body.get("enabled").and_then(Json::as_bool);
            let cc = body.get("allow_computer_control").and_then(Json::as_bool);
            let rebuild = body.get("rebuild").and_then(Json::as_bool).unwrap_or(false);
            let _ = global_device_manager().update_config(en, cc, rebuild);
            respond(
                &mut stream,
                200,
                "application/json",
                &global_device_manager().status_json(),
            );
        }
        ("GET", "/v1/device/profile") => {
            let mgr = global_device_manager();
            let md = mgr.profile_markdown();
            let body = format!(
                r#"{{"markdown":{},"status":{}}}"#,
                quote(&md),
                mgr.status_json()
            );
            respond(&mut stream, 200, "application/json", &body);
        }
        ("POST", "/v1/device/open") => {
            let body = match std::str::from_utf8(&req.body).map_err(|e| e.to_string()).and_then(Json::parse) {
                Ok(b) => b,
                Err(e) => return error(&mut stream, 400, &format!("invalid JSON: {e}")),
            };
            match global_device_manager().execute_computer_action(&body) {
                Ok(msg) => respond(
                    &mut stream,
                    200,
                    "application/json",
                    &format!(r#"{{"status":"ok","message":{}}}"#, quote(&msg)),
                ),
                Err(e) => error(&mut stream, 400, &e),
            }
        }
        ("GET", "/v1/agent/files") => {
            let wd = srv.active_workdir();
            respond(
                &mut stream,
                200,
                "application/json",
                &format_agent_files_with_ctx(&wd, srv.ctx_limit),
            );
        }
        ("GET", "/v1/agent/file") => {
            let Some(rel) = req.query_param("path") else {
                return error(&mut stream, 400, "missing `path` query parameter");
            };
            let wd = srv.active_workdir();
            match format_agent_file_detail_json(&wd, &rel) {
                Ok(json_body) => respond(&mut stream, 200, "application/json", &json_body),
                Err((code, msg)) => error(&mut stream, code, &msg),
            }
        }
        ("POST", "/v1/agent/clear") => {
            let wd = srv.active_workdir();
            match clear_workspace_dir(&wd) {
                Ok(n) => respond(
                    &mut stream,
                    200,
                    "application/json",
                    &format!(r#"{{"status":"ok","removed":{n}}}"#),
                ),
                Err(e) => error(&mut stream, 500, &e),
            }
        }
        _ => error(&mut stream, 404, &format!("no route for {} {}", req.method, req.path)),
    }
}

fn default_state_dir() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join(".kronos").join("ui_state");
        }
    }
    std::env::temp_dir().join("kronos_ui_state")
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!(
            "usage: {} <model_dir> [--draft DIR] [--workdir DIR] [--state-dir DIR] [--cache-dir DIR] [--host 127.0.0.1] [--port 18080] [--api-key KEY] [--mode nf4g128] ...",
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
    let requested_ctx = flag("--ctx")
        .or_else(|| flag("--max-seq-len"))
        .map_or(16384usize, |v| v.parse().expect("numeric --ctx"));
    let mut model = ChatModel::load_with_cache_dir(
        model_dir,
        cache_dir_opt,
        &mode,
        requested_ctx,
        num("--threads", default_threads) as usize,
        calib_opt,
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let force_cpu = args.iter().any(|a| a == "--cpu" || a == "--fp32-kv" || a == "--ane");
    let require_metal = args.iter().any(|a| a == "--metal");
    if args.iter().any(|a| a == "--fp32-kv") {
        model.enable_q8_kv_cache(false);
    } else if args.iter().any(|a| a == "--q8-kv") {
        model.enable_q8_kv_cache(true);
    }
    if require_metal {
        model.enable_metal().unwrap_or_else(|e| {
            eprintln!("metal error: {e}");
            std::process::exit(1);
        });
    } else if !force_cpu {
        let _ = model.enable_metal();
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

    let raw_workdir = match flag("--workdir") {
        Some(w) => expand_tilde_path(&w),
        None => std::env::temp_dir().join("kronos_server_workspace"),
    };
    fs::create_dir_all(&raw_workdir).unwrap_or_else(|e| {
        eprintln!("create workdir {}: {e}", raw_workdir.display());
        std::process::exit(1);
    });
    let workdir = raw_workdir.canonicalize().unwrap_or(raw_workdir);

    let state_dir = match flag("--state-dir") {
        Some(s) => expand_tilde_path(&s),
        None => default_state_dir(),
    };
    fs::create_dir_all(&state_dir).unwrap_or_else(|e| {
        eprintln!("create state-dir {}: {e}", state_dir.display());
        std::process::exit(1);
    });
    let ui_index = load_ui_state_index(&state_dir, &workdir);

    let enable_dev_ctx = args.iter().any(|a| a == "--device-context");
    let t_warm = Instant::now();
    let warm_sys = build_unified_system_prompt(&workdir);
    let warm_tools = tool_definitions_json();
    let mut prewarmed_tokens = 0usize;
    std::thread::scope(|s| {
        let vj = s.spawn(kronos_2t::vision::prewarm_vision);
        let dj = s.spawn(move || {
            let mgr = global_device_manager();
            if enable_dev_ctx {
                mgr.update_config(Some(true), None, false);
            }
        });
        prewarmed_tokens = model.prewarm_prefix(&warm_sys, &warm_tools);
        let _ = vj.join();
        let _ = dj.join();
    });
    eprintln!(
        "pre-warmed {prewarmed_tokens} system+tool KV tokens and Apple Vision pipeline in {:.2}s",
        t_warm.elapsed().as_secs_f64()
    );

    let addr = format!(
        "{}:{}",
        flag("--host").unwrap_or_else(|| "127.0.0.1".into()),
        num("--port", 18080.0) as u16
    );
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("bind {addr}: {e}");
        std::process::exit(1);
    });
    let weights_gib = model.resident_weight_bytes() as f64 / (1u64 << 30) as f64;
    let kv_gib = model.kv_cache_bytes() as f64 / (1u64 << 30) as f64;
    let kv_tag = if model.is_metal_enabled() {
        "Metal-Q8-KV"
    } else if model.is_q8_kv_enabled() {
        "Q8-KV"
    } else {
        "FP32-KV"
    };
    let ctx_sz = model.context_size();
    let srv = Arc::new(Server {
        name: model.name.clone(),
        model: Mutex::new(model),
        api_key: flag("--api-key"),
        workdir: Mutex::new(workdir.clone()),
        state_dir,
        ui_index: Mutex::new(ui_index),
        ctx_limit: ctx_sz,
        next_id: AtomicU64::new(1),
        media_seq: AtomicU64::new(1),
    });
    eprintln!(
        "serving {} on http://{addr}/ (quick HUD http://{addr}/quick · weights {weights_gib:.2} GiB + {kv_tag} {kv_gib:.2} GiB @ ctx={ctx_sz} · workdir={})",
        srv.name,
        workdir.display()
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
            token_ids: Vec::new(),
            tool_calls: vec![ToolCall {
                id: "call_0".to_string(),
                name: "write_file".to_string(),
                arguments: r#"{"path":"index.html","content":"<h1>ok</h1>"}"#.to_string(),
            }],
            prompt_tokens: 120,
            prefilled_tokens: 120,
            completion_tokens: 24,
            drafted_tokens: 0,
            accepted_tokens: 0,
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

    #[test]
    fn test_agent_workspace_sessions_revert_and_pages() {
        let html_check = kronos_2t::agent::validate_content_by_ext(
            Path::new("."),
            Path::new("chat.html"),
            "html",
            CHAT_PAGE,
            false,
        );
        assert!(html_check.is_ok(), "chat.html validation failed: {html_check:?}");

        let quick_check = kronos_2t::agent::validate_content_by_ext(
            Path::new("."),
            Path::new("quick.html"),
            "html",
            QUICK_PAGE,
            false,
        );
        assert!(quick_check.is_ok(), "quick.html validation failed: {quick_check:?}");

        let req = Request {
            method: "GET".into(),
            path: "/v1/agent/file".into(),
            query: "path=src%2Fapp.js&t=123".into(),
            headers: Vec::new(),
            body: Vec::new(),
        };
        assert_eq!(req.query_param("path").as_deref(), Some("src/app.js"));
        assert_eq!(req.query_param("t").as_deref(), Some("123"));

        let dir = std::env::temp_dir().join(format!("kronos_srv_agent_test_{}", std::process::id()));
        let state_dir = std::env::temp_dir().join(format!("kronos_srv_state_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&state_dir);
        fs::create_dir_all(dir.join("src")).unwrap();
        fs::write(dir.join("index.html"), "<!doctype html><html><body><script src=\"src/app.js\"></script></body></html>").unwrap();
        fs::write(dir.join("src/app.js"), "const ok = 1;\n").unwrap();

        let files_json = Json::parse(&format_agent_files_json(&dir)).unwrap();
        assert_eq!(files_json.get("has_html").and_then(Json::as_bool), Some(true));
        assert_eq!(files_json.get("preview_path").and_then(Json::as_str), Some("index.html"));
        let arr = files_json.get("files").and_then(Json::as_arr).unwrap();
        assert_eq!(arr.len(), 2);

        let detail = Json::parse(&format_agent_file_detail_json(&dir, "src/app.js").unwrap()).unwrap();
        assert_eq!(detail.get("validation_ok").and_then(Json::as_bool), Some(true));
        assert_eq!(detail.get("content").and_then(Json::as_str), Some("const ok = 1;\n"));
        assert!(format_agent_file_detail_json(&dir, "../escape.js").is_err());

        // Revert check: restore older contents and remove newly created file.
        fs::write(dir.join("src/app.js"), "const ok = 2;\n").unwrap();
        let rev = Json::parse(&revert_workspace_change(&dir, "src/app.js", true, "const ok = 1;\n").unwrap()).unwrap();
        assert_eq!(rev.get("action").and_then(Json::as_str), Some("restored"));
        assert_eq!(fs::read_to_string(dir.join("src/app.js")).unwrap(), "const ok = 1;\n");

        fs::write(dir.join("src/temp.js"), "const t = 1;\n").unwrap();
        let rev_rm = Json::parse(&revert_workspace_change(&dir, "src/temp.js", false, "").unwrap()).unwrap();
        assert_eq!(rev_rm.get("action").and_then(Json::as_str), Some("removed"));
        assert!(!dir.join("src/temp.js").exists());

        // Session store CRUD check.
        let mut idx = load_ui_state_index(&state_dir, &dir);
        assert!(sanitize_session_id("../bad").is_err());
        let saved_id = save_session_record(
            &state_dir,
            &mut idx,
            br#"{"id":"sess_unit_1","title":"Landing Page","workdir":"/tmp/demo","created_at":100,"updated_at":120,"context_tokens":512,"added_lines":14,"removed_lines":3,"changed_files":2,"messages":[]}"#,
        )
        .unwrap();
        assert_eq!(saved_id, "sess_unit_1");
        assert_eq!(idx.sessions.len(), 1);
        let reloaded = load_ui_state_index(&state_dir, &dir);
        assert_eq!(reloaded.sessions.len(), 1);
        assert_eq!(reloaded.sessions[0].added_lines, 14);
        delete_session_record(&state_dir, &mut idx, "sess_unit_1").unwrap();
        assert_eq!(idx.sessions.len(), 0);

        let removed = clear_workspace_dir(&dir).unwrap();
        assert_eq!(removed, 2);
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&state_dir);
    }
}
