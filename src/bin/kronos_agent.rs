#![forbid(unsafe_code)]
//! Sandboxed tool-calling coding runner (`kronos_agent --model <dir> --workdir <dir> --prompt "..."` or `--server 127.0.0.1:18080`).

use kronos_2t::chat::{ChatModel, FinishReason, GenParams, Message, ToolCall};
use kronos_2t::json::{quote, Json};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const MAX_READ_BYTES: usize = 16 * 1024;
const MAX_CMD_OUTPUT_BYTES: usize = 8 * 1024;

const AGENT_SYSTEM_PROMPT: &str = concat!(
    "You are Kronos Agent, a local software engineering runner on Apple Silicon. ",
    "Use the provided workspace tools (`write_file`, `read_file`, `list_dir`, `run_command`) to build, ",
    "run, and verify complete working solutions directly in the workspace directory. ",
    "Do not write out file drafts in markdown prose before calling tools; emit `<tool_call>` ",
    "function calls immediately to create files and execute commands on disk, then verify them (when ",
    "starting a background local HTTP server, redirect stdout/stderr to `server.log` and `sleep 1` so it ",
    "binds its port before running `curl -fsSI`), and give your final summary once every file and ",
    "check has succeeded."
);

fn tool_definitions_json() -> Vec<String> {
    vec![
        r#"{"type":"function","function":{"name":"write_file","description":"Create or overwrite a UTF-8 file inside the workspace directory (parent directories are created automatically).","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative file path inside the workspace, e.g. index.html or src/app.js"},"content":{"type":"string","description":"Complete file contents to write"}},"required":["path","content"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 text file inside the workspace directory.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative file path inside the workspace"}},"required":["path"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"list_dir","description":"List files and subdirectories at a relative path inside the workspace directory.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative directory path (use '.' for workspace root)"}}},"required":[]}}"#.to_string(),
        r#"{"type":"function","function":{"name":"run_command","description":"Execute a shell command (`sh -c`) inside the workspace directory with a timeout. Redirect stdout/stderr if leaving a background server running (e.g. `nohup python3 -m http.server 18123 > server.log 2>&1 &`).","parameters":{"type":"object","properties":{"cmd":{"type":"string","description":"Shell command line to run in the workspace"}},"required":["cmd"]}}}"#.to_string(),
    ]
}

/// Resolves `user_rel` inside `workdir`, rejecting any `..` traversal or path outside `workdir`.
fn resolve_jailed_path(workdir: &Path, user_rel: &str) -> Result<PathBuf, String> {
    let trimmed = user_rel.trim();
    if trimmed.is_empty() {
        return Ok(workdir.to_path_buf());
    }
    let candidate = Path::new(trimmed);
    if candidate.is_absolute() {
        let canon_work = workdir.canonicalize().unwrap_or_else(|_| workdir.to_path_buf());
        if let Ok( stripped) = candidate.strip_prefix(&canon_work) {
            return resolve_jailed_path(workdir, &stripped.to_string_lossy());
        }
        if let Ok(stripped) = candidate.strip_prefix(workdir) {
            return resolve_jailed_path(workdir, &stripped.to_string_lossy());
        }
        return Err(format!("path `{trimmed}` escapes workspace directory"));
    }

    let mut clean = PathBuf::new();
    for comp in candidate.components() {
        match comp {
            Component::CurDir => {}
            Component::Normal(seg) => clean.push(seg),
            Component::ParentDir => {
                if !clean.pop() {
                    return Err(format!("path `{trimmed}` escapes workspace directory via `..`"));
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!("path `{trimmed}` must be relative to workspace"));
            }
        }
    }
    Ok(workdir.join(clean))
}

fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n...[truncated {} bytes]", &s[..end], s.len() - end)
}

fn execute_tool(workdir: &Path, call: &ToolCall, cmd_timeout_secs: u64, step_tag: usize) -> String {
    let args = match Json::parse(&call.arguments) {
        Ok(v) => v,
        Err(e) => return format!("error: invalid JSON arguments `{}`: {e}", call.arguments),
    };

    match call.name.as_str() {
        "write_file" => {
            let Some(rel) = args.get("path").and_then(Json::as_str) else {
                return "error: `write_file` requires string argument `path`".to_string();
            };
            let Some(content) = args.get("content").and_then(Json::as_str) else {
                return "error: `write_file` requires string argument `content`".to_string();
            };
            let target = match resolve_jailed_path(workdir, rel) {
                Ok(p) => p,
                Err(e) => return format!("error: {e}"),
            };
            if let Some(parent) = target.parent() {
                if let Err(e) = fs::create_dir_all(parent) {
                    return format!("error creating directory `{}`: {e}", parent.display());
                }
            }
            match fs::write(&target, content.as_bytes()) {
                Ok(()) => format!(
                    "ok: wrote {} bytes ({} lines) to `{rel}`",
                    content.len(),
                    content.lines().count()
                ),
                Err(e) => format!("error writing `{rel}`: {e}"),
            }
        }
        "read_file" => {
            let Some(rel) = args.get("path").and_then(Json::as_str) else {
                return "error: `read_file` requires string argument `path`".to_string();
            };
            let target = match resolve_jailed_path(workdir, rel) {
                Ok(p) => p,
                Err(e) => return format!("error: {e}"),
            };
            match fs::read_to_string(&target) {
                Ok(text) => truncate_utf8(&text, MAX_READ_BYTES),
                Err(e) => format!("error reading `{rel}`: {e}"),
            }
        }
        "list_dir" => {
            let rel = args.get("path").and_then(Json::as_str).unwrap_or(".");
            let target = match resolve_jailed_path(workdir, rel) {
                Ok(p) => p,
                Err(e) => return format!("error: {e}"),
            };
            let entries = match fs::read_dir(&target) {
                Ok(rd) => rd,
                Err(e) => return format!("error listing `{rel}`: {e}"),
            };
            let mut rows = Vec::new();
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                if name.starts_with(".kronos_cmd_") {
                    continue;
                }
                if let Ok(meta) = e.metadata() {
                    if meta.is_dir() {
                        rows.push(format!("{name}/"));
                    } else {
                        rows.push(format!("{name} ({} B)", meta.len()));
                    }
                } else {
                    rows.push(name);
                }
            }
            rows.sort();
            if rows.is_empty() {
                "(empty directory)".to_string()
            } else {
                rows.join("\n")
            }
        }
        "run_command" => {
            let Some(cmd) = args.get("cmd").and_then(Json::as_str) else {
                return "error: `run_command` requires string argument `cmd`".to_string();
            };
            run_shell_command_with_timeout(workdir, cmd, cmd_timeout_secs, step_tag)
        }
        other => format!("error: unknown tool `{other}`"),
    }
}

/// Redirects stdout/stderr to a file rather than a pipe so backgrounded child processes do not block wait.
fn run_shell_command_with_timeout(
    workdir: &Path,
    cmd: &str,
    timeout_secs: u64,
    step_tag: usize,
) -> String {
    let log_path = workdir.join(format!(".kronos_cmd_{}_{}.log", std::process::id(), step_tag));
    let out_file = match File::create(&log_path) {
        Ok(f) => f,
        Err(e) => return format!("error creating command output file: {e}"),
    };
    let err_file = match out_file.try_clone() {
        Ok(f) => f,
        Err(e) => {
            let _ = fs::remove_file(&log_path);
            return format!("error cloning command output file: {e}");
        }
    };

    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out_file))
        .stderr(Stdio::from(err_file))
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let _ = fs::remove_file(&log_path);
            return format!("error spawning shell: {e}");
        }
    };

    let deadline = Instant::now() + Duration::from_secs(timeout_secs.max(1));
    let mut exit_code: Option<i32> = None;
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                exit_code = Some(status.code().unwrap_or(-1));
                break;
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    timed_out = true;
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => {
                let _ = fs::remove_file(&log_path);
                return format!("error waiting on command: {e}");
            }
        }
    }

    let raw_out = fs::read(&log_path).unwrap_or_default();
    let _ = fs::remove_file(&log_path);
    let text_out = truncate_utf8(&String::from_utf8_lossy(&raw_out), MAX_CMD_OUTPUT_BYTES);

    if timed_out {
        format!(
            "timed_out after {timeout_secs}s (process killed; start long-running servers in background with `> server.log 2>&1 &`)\noutput:\n{text_out}"
        )
    } else {
        let code = exit_code.unwrap_or(0);
        if text_out.trim().is_empty() {
            format!("exit_code={code} (no stdout/stderr)")
        } else {
            format!("exit_code={code}\n{text_out}")
        }
    }
}

enum AgentBackend {
    Local(Box<ChatModel>),
    Http { addr: String, api_key: Option<String> },
}

struct StepReply {
    text: String,
    tool_calls: Vec<ToolCall>,
    finish: FinishReason,
    telemetry: String,
}

impl AgentBackend {
    fn step(
        &mut self,
        messages: &[Message],
        tools_json: &[String],
        params: &GenParams,
        stream_stdout: bool,
    ) -> Result<StepReply, String> {
        match self {
            AgentBackend::Local(model) => {
                let mut out = io::stdout();
                let c = model.generate_with_tools(messages, tools_json, params, |piece| {
                    if stream_stdout {
                        eprint!("{piece}");
                        let _ = out.flush();
                    }
                    true
                })?;
                if stream_stdout && (!c.text.is_empty() || !c.tool_calls.is_empty()) {
                    eprintln!();
                }
                let telemetry = format!(
                    "{} prompt tok ({} new) @ {:.0} tok/s · {} gen tok @ {:.1} tok/s",
                    c.prompt_tokens,
                    c.prefilled_tokens,
                    c.prefilled_tokens as f64 / c.prefill_secs.max(1e-9),
                    c.completion_tokens,
                    c.completion_tokens as f64 / c.decode_secs.max(1e-9),
                );
                Ok(StepReply {
                    text: c.text,
                    tool_calls: c.tool_calls,
                    finish: c.finish,
                    telemetry,
                })
            }
            AgentBackend::Http { addr, api_key } => {
                http_chat_step(addr, api_key.as_deref(), messages, tools_json, params)
            }
        }
    }
}

fn http_chat_step(
    addr: &str,
    api_key: Option<&str>,
    messages: &[Message],
    tools_json: &[String],
    params: &GenParams,
) -> Result<StepReply, String> {
    let host_port = addr
        .strip_prefix("http://")
        .unwrap_or(addr)
        .trim_end_matches('/')
        .trim_end_matches("/v1");

    let mut msg_items = Vec::with_capacity(messages.len());
    for m in messages {
        if m.role == "assistant" && !m.tool_calls.is_empty() {
            let content_val = if m.content.is_empty() {
                "null".to_string()
            } else {
                quote(&m.content)
            };
            let tcs: Vec<String> = m
                .tool_calls
                .iter()
                .map(|tc| {
                    format!(
                        r#"{{"id":{},"type":"function","function":{{"name":{},"arguments":{}}}}}"#,
                        quote(&tc.id),
                        quote(&tc.name),
                        quote(&tc.arguments),
                    )
                })
                .collect();
            msg_items.push(format!(
                r#"{{"role":"assistant","content":{content_val},"tool_calls":[{}]}}"#,
                tcs.join(",")
            ));
        } else if m.role == "tool" {
            let id = m.tool_call_id.as_deref().unwrap_or("call_0");
            msg_items.push(format!(
                r#"{{"role":"tool","tool_call_id":{},"content":{}}}"#,
                quote(id),
                quote(&m.content)
            ));
        } else {
            msg_items.push(format!(
                r#"{{"role":{},"content":{}}}"#,
                quote(&m.role),
                quote(&m.content)
            ));
        }
    }

    let tools_field = if tools_json.is_empty() {
        String::new()
    } else {
        format!(r#","tools":[{}]"#, tools_json.join(","))
    };
    let body = format!(
        r#"{{"model":"kronos","messages":[{}]{tools_field},"temperature":{},"top_k":{},"top_p":{},"repetition_penalty":{},"max_tokens":{},"seed":{}}}"#,
        msg_items.join(","),
        params.temperature,
        params.top_k,
        params.top_p,
        params.repeat_penalty,
        params.max_tokens,
        params.seed,
    );

    let t0 = Instant::now();
    let mut stream = TcpStream::connect(host_port).map_err(|e| format!("connect {host_port}: {e}"))?;
    let _ = stream.set_read_timeout(Some(Duration::from_secs(600)));
    let auth = api_key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\n{auth}Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(req.as_bytes()).map_err(|e| format!("send HTTP request: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).map_err(|e| format!("read HTTP status: {e}"))?;
    loop {
        let mut h = String::new();
        let n = reader.read_line(&mut h).map_err(|e| format!("read HTTP header: {e}"))?;
        if n == 0 || h == "\r\n" || h == "\n" {
            break;
        }
    }
    let mut resp_bytes = Vec::new();
    reader.read_to_end(&mut resp_bytes).map_err(|e| format!("read HTTP body: {e}"))?;
    let dt = t0.elapsed().as_secs_f64();
    let resp_str = String::from_utf8_lossy(&resp_bytes);
    let parsed = Json::parse(&resp_str).map_err(|e| format!("parse server JSON ({status_line}): {e}: {resp_str}"))?;
    if let Some(err) = parsed.get("error") {
        return Err(format!("server error: {}", err.stringify()));
    }
    let choice0 = parsed
        .get("choices")
        .and_then(Json::as_arr)
        .and_then(|a| a.first())
        .ok_or("missing choices[0] in server response")?;
    let finish = match choice0.get("finish_reason").and_then(Json::as_str).unwrap_or("stop") {
        "tool_calls" => FinishReason::ToolCalls,
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    };
    let msg = choice0.get("message").ok_or("missing message in choices[0]")?;
    let text = msg.get("content").and_then(Json::as_str).unwrap_or("").to_string();
    let mut tool_calls = Vec::new();
    if let Some(arr) = msg.get("tool_calls").and_then(Json::as_arr) {
        for (i, tc) in arr.iter().enumerate() {
            let id = tc
                .get("id")
                .and_then(Json::as_str)
                .map_or_else(|| format!("call_{i}"), str::to_string);
            let fn_obj = tc.get("function").unwrap_or(tc);
            let name = fn_obj.get("name").and_then(Json::as_str).unwrap_or("").to_string();
            let arguments = match fn_obj.get("arguments") {
                Some(Json::Str(s)) => s.clone(),
                Some(other) => other.stringify(),
                None => "{}".to_string(),
            };
            tool_calls.push(ToolCall { id, name, arguments });
        }
    }
    let comp_tok = parsed
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Json::as_f64)
        .unwrap_or(0.0) as usize;
    let prompt_tok = parsed
        .get("usage")
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Json::as_f64)
        .unwrap_or(0.0) as usize;
    let telemetry = format!("HTTP {prompt_tok} prompt tok · {comp_tok} gen tok in {dt:.2}s");
    Ok(StepReply { text, tool_calls, finish, telemetry })
}

struct AgentConfig {
    workdir: PathBuf,
    params: GenParams,
    max_steps: usize,
    cmd_timeout_secs: u64,
    quiet_stream: bool,
}

fn run_agent_goal(
    backend: &mut AgentBackend,
    history: &mut Vec<Message>,
    tools_json: &[String],
    cfg: &AgentConfig,
) -> Result<String, String> {
    let t_goal = Instant::now();
    for step in 1..=cfg.max_steps {
        eprintln!("\n=== [Agent Step {step}/{}] ===", cfg.max_steps);
        let mut reply = backend.step(history, tools_json, &cfg.params, !cfg.quiet_stream)?;
        eprintln!("  [{}]", reply.telemetry);

        if reply.tool_calls.is_empty() {
            history.push(Message::new("assistant", &reply.text));
            eprintln!(
                "=== [Completed in {step} step(s), {:.2}s total] ===",
                t_goal.elapsed().as_secs_f64()
            );
            return Ok(reply.text);
        }

        // Assign globally unique tool_call IDs across steps.
        for (idx, tc) in reply.tool_calls.iter_mut().enumerate() {
            tc.id = format!("call_s{step}_{idx}");
        }
        history.push(Message::assistant_tool_calls(&reply.text, reply.tool_calls.clone()));

        for (idx, tc) in reply.tool_calls.iter().enumerate() {
            let arg_preview = truncate_utf8(&tc.arguments, 140).replace('\n', "\\n");
            eprintln!("  -> tool `{}`({arg_preview})", tc.name);
            let result = execute_tool(&cfg.workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx);
            let res_preview = truncate_utf8(&result, 200).replace('\n', " | ");
            eprintln!("     <- {res_preview}");
            history.push(Message::tool_result(&tc.id, &result));
        }

        if reply.finish == FinishReason::Length {
            eprintln!("  (warning: step hit max_tokens; continuing with executed tool calls)");
        }
    }
    Err(format!("reached --max-steps {} without final response", cfg.max_steps))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let num = |name: &str, default: f64| flag(name).map_or(default, |v| v.parse().expect("numeric flag"));

    let model_flag = flag("--model").or_else(|| {
        args.get(1)
            .filter(|a| !a.starts_with("--"))
            .cloned()
    });
    let server_flag = flag("--server");
    if model_flag.is_none() && server_flag.is_none() {
        eprintln!(
            "usage:\n  {} --model <model_dir> [--cache-dir DIR] [--mode nf4g128] [--ctx 32768] --workdir <DIR> [--prompt \"...\"]\n  {} --server 127.0.0.1:18080 --workdir <DIR> [--prompt \"...\"]",
            args[0], args[0]
        );
        std::process::exit(2);
    }

    let workdir_str = flag("--workdir").unwrap_or_else(|| ".".to_string());
    let workdir = PathBuf::from(&workdir_str);
    fs::create_dir_all(&workdir).unwrap_or_else(|e| {
        eprintln!("create workdir {}: {e}", workdir.display());
        std::process::exit(1);
    });
    let workdir = workdir.canonicalize().unwrap_or(workdir);

    let mut backend = if let Some(srv) = server_flag {
        eprintln!("connecting Kronos Agent to HTTP server {srv} (workdir: {})...", workdir.display());
        AgentBackend::Http {
            addr: srv,
            api_key: flag("--api-key"),
        }
    } else {
        let model_dir_str = model_flag.unwrap();
        let model_dir = Path::new(&model_dir_str);
        let mode = flag("--mode").unwrap_or_else(|| "nf4g128".to_string());
        let calib = flag("--calib");
        let calib_tokens = num("--calib-tokens", 2048.0) as usize;
        let calib_opt = calib.as_deref().map(|c| (Path::new(c), calib_tokens));
        let cache_dir_flag = flag("--cache-dir");
        let cache_dir_opt = cache_dir_flag.as_deref().map(Path::new);
        let default_threads = kronos_2t::thread_pool::default_inference_threads() as f64;
        let mut model = ChatModel::load_with_cache_dir(
            model_dir,
            cache_dir_opt,
            &mode,
            num("--ctx", 32768.0) as usize,
            num("--threads", default_threads) as usize,
            calib_opt,
        )
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
        if args.iter().any(|a| a == "--fp32-kv") {
            model.enable_q8_kv_cache(false);
        }
        eprintln!(
            "loaded {} (weights {:.2} GiB + {} {:.2} GiB @ ctx={}) · workdir={}",
            model.name,
            model.resident_weight_bytes() as f64 / (1u64 << 30) as f64,
            if model.is_q8_kv_enabled() { "Q8-KV" } else { "FP32-KV" },
            model.kv_cache_bytes() as f64 / (1u64 << 30) as f64,
            model.context_size(),
            workdir.display(),
        );
        AgentBackend::Local(Box::new(model))
    };

    let params = GenParams {
        temperature: num("--temp", 0.0) as f32,
        top_k: num("--top-k", 40.0) as usize,
        top_p: num("--top-p", 0.9) as f32,
        repeat_penalty: num("--repeat-penalty", 1.05) as f32,
        max_tokens: num("--max-new", 4096.0) as usize,
        seed: num("--seed", 0.0) as u64,
    };
    let max_steps = num("--max-steps", 12.0) as usize;
    let cmd_timeout = num("--cmd-timeout", 20.0) as u64;
    let quiet_stream = args.iter().any(|a| a == "--quiet-stream");
    let agent_cfg = AgentConfig {
        workdir,
        params,
        max_steps,
        cmd_timeout_secs: cmd_timeout,
        quiet_stream,
    };
    let sys_prompt = flag("--system").unwrap_or_else(|| AGENT_SYSTEM_PROMPT.to_string());
    let tools_json = tool_definitions_json();
    let mut history = vec![Message::new("system", &sys_prompt)];

    if let Some(prompt) = flag("--prompt") {
        history.push(Message::new("user", &prompt));
        match run_agent_goal(&mut backend, &mut history, &tools_json, &agent_cfg) {
            Ok(answer) => {
                println!("{answer}");
            }
            Err(e) => {
                eprintln!("agent error: {e}");
                std::process::exit(1);
            }
        }
        return;
    }

    eprintln!("Kronos Agent interactive mode · /reset /exit");
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    loop {
        print!("\nagent> ");
        let _ = stdout.flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history = vec![Message::new("system", &sys_prompt)];
                println!("(agent history reset)");
                continue;
            }
            goal => history.push(Message::new("user", goal)),
        }
        match run_agent_goal(&mut backend, &mut history, &tools_json, &agent_cfg) {
            Ok(answer) => println!("\n{answer}"),
            Err(e) => eprintln!("\n(agent error: {e})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workspace_jail_and_tools_and_nonblocking_background_command() {
        let dir = std::env::temp_dir().join(format!("kronos_agent_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        assert!(resolve_jailed_path(&dir, "../escape.txt").is_err());
        assert!(resolve_jailed_path(&dir, "sub/../../escape.txt").is_err());
        assert!(resolve_jailed_path(&dir, "/etc/passwd").is_err());
        assert_eq!(resolve_jailed_path(&dir, "web/index.html").unwrap(), dir.join("web/index.html"));

        let w = execute_tool(
            &dir,
            &ToolCall {
                id: "c0".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"web/index.html","content":"<h1>Kronos</h1>"}"#.into(),
            },
            5,
            1,
        );
        assert!(w.starts_with("ok: wrote 15 bytes"));

        let r = execute_tool(
            &dir,
            &ToolCall {
                id: "c1".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"web/index.html"}"#.into(),
            },
            5,
            2,
        );
        assert_eq!(r, "<h1>Kronos</h1>");

        let l = execute_tool(
            &dir,
            &ToolCall {
                id: "c2".into(),
                name: "list_dir".into(),
                arguments: r#"{"path":"web"}"#.into(),
            },
            5,
            3,
        );
        assert!(l.contains("index.html"));

        // File-backed output ensures backgrounded jobs return immediately.
        let t0 = Instant::now();
        let bg = execute_tool(
            &dir,
            &ToolCall {
                id: "c3".into(),
                name: "run_command".into(),
                arguments: r#"{"cmd":"echo launched && sleep 3 &"}"#.into(),
            },
            5,
            4,
        );
        assert!(t0.elapsed() < Duration::from_secs(2), "background command blocked: {:?}", t0.elapsed());
        assert!(bg.contains("exit_code=0"));

        // Foreground command exceeding timeout is terminated.
        let fg = execute_tool(
            &dir,
            &ToolCall {
                id: "c4".into(),
                name: "run_command".into(),
                arguments: r#"{"cmd":"echo starting && sleep 5"}"#.into(),
            },
            1,
            5,
        );
        assert!(fg.starts_with("timed_out after 1s"));
        assert!(fg.contains("starting"));

        let _ = fs::remove_dir_all(&dir);
    }
}
