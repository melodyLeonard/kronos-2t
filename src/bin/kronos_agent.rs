#![forbid(unsafe_code)]
//! Sandboxed tool-calling coding runner (`kronos_agent --model <dir> --workdir <dir> --prompt "..."` or `--server 127.0.0.1:18080`).

use kronos_2t::runtime::agent::{
    execute_tool, run_agent_turn, serialize_messages_json, tool_definitions_json, truncate_utf8,
    validate_workspace, AgentEvent, AgentTurnConfig, AGENT_SYSTEM_PROMPT,
};
use kronos_2t::runtime::chat::{ChatModel, FinishReason, GenParams, Message, ToolCall};
use kronos_2t::util::json::Json;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

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

    let messages_json = serialize_messages_json(messages);
    let tools_field = if tools_json.is_empty() {
        String::new()
    } else {
        format!(r#","tools":[{}]"#, tools_json.join(","))
    };
    let body = format!(
        r#"{{"model":"kronos","messages":{messages_json}{tools_field},"temperature":{},"top_k":{},"top_p":{},"repetition_penalty":{},"max_tokens":{},"seed":{}}}"#,
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
    match backend {
        AgentBackend::Local(model) => {
            let turn_cfg = AgentTurnConfig {
                workdir: cfg.workdir.clone(),
                params: cfg.params.clone(),
                max_steps: cfg.max_steps,
                cmd_timeout_secs: cfg.cmd_timeout_secs,
                expect_tool_call: true,
            };
            let mut printed_prose = false;
            run_agent_turn(model, history, tools_json, &turn_cfg, |ev| {
                match ev {
                    AgentEvent::JevDecision { plan } => {
                        if !plan.prefetched_files.is_empty() || plan.pruned_tool_count > 0 {
                            eprintln!(
                                "  [jev pre-pass {:.2} ms: route={} conf={:.2} prefetched={} pruned_tools={} rubric={}]",
                                plan.elapsed_ms,
                                plan.route.as_str(),
                                plan.route_choice.confidence,
                                plan.prefetched_files.len(),
                                plan.pruned_tool_count,
                                plan.rubric.len()
                            );
                        }
                    }
                    AgentEvent::ContextCompacted {
                        before_tokens,
                        after_tokens,
                        compacted_messages,
                    } => {
                        eprintln!(
                            "  [context compacted: {before_tokens} -> {after_tokens} tokens ({compacted_messages} items pruned)]"
                        );
                    }
                    AgentEvent::StepStart { step, max_steps } => {
                        printed_prose = false;
                        eprintln!("\n=== [Agent Step {step}/{max_steps}] ===");
                    }
                    AgentEvent::TextDelta(piece) => {
                        if !cfg.quiet_stream && !piece.is_empty() {
                            printed_prose = true;
                            eprint!("{piece}");
                            let _ = io::stderr().flush();
                        }
                    }
                    AgentEvent::ToolProgress { .. } => {}
                    AgentEvent::StepStats { completion: c, .. } => {
                        if printed_prose {
                            eprintln!();
                            printed_prose = false;
                        }
                        let draft_info = match c.acceptance_rate() {
                            Some(r) => format!(
                                " · accept {:.1}% ({}/{})",
                                r * 100.0,
                                c.accepted_tokens,
                                c.drafted_tokens
                            ),
                            None => String::new(),
                        };
                        eprintln!(
                            "  [{} prompt tok ({} new, {}) @ {:.0} tok/s · {} gen tok @ {:.1} tok/s{draft_info}]",
                            c.prompt_tokens,
                            c.prefilled_tokens,
                            c.prefill_unit,
                            c.prefilled_tokens as f64 / c.prefill_secs.max(1e-9),
                            c.completion_tokens,
                            c.completion_tokens as f64 / c.decode_secs.max(1e-9),
                        );
                    }
                    AgentEvent::ToolStart { call, .. } => {
                        let arg_preview = truncate_utf8(&call.arguments, 140).replace('\n', "\\n");
                        eprintln!("  -> tool `{}`({arg_preview})", call.name);
                    }
                    AgentEvent::ToolDone { outcome, .. } => {
                        let res_preview = truncate_utf8(&outcome.output, 200).replace('\n', " | ");
                        eprintln!("     <- {res_preview}");
                    }
                    AgentEvent::WorkspaceChanged { workdir } => {
                        eprintln!("  [workspace switched to {}]", workdir.display());
                    }
                    AgentEvent::ValidationRepair { step, report } => {
                        eprintln!("  [step {step} validation repair]\n{report}");
                    }
                    AgentEvent::Verified { step, report } => {
                        eprintln!(
                            "  [step {step} verified in {:.1} ms: {}]",
                            report.elapsed_ms, report.summary
                        );
                    }
                    AgentEvent::JevVerification {
                        step,
                        eval,
                        rolled_back,
                    } => {
                        let rb = if rolled_back { " · rolled_back" } else { "" };
                        eprintln!(
                            "  [step {step} jev rubric {:.1} ms: score={:.2} passed={}{rb}]",
                            eval.elapsed_ms, eval.score.expected, eval.passed
                        );
                    }
                    AgentEvent::Done { steps, total_secs, .. } => {
                        eprintln!("=== [Completed in {steps} step(s), {total_secs:.2}s total] ===");
                    }
                }
                true
            })
        }
        AgentBackend::Http { addr, api_key } => {
            let t_goal = Instant::now();
            let mut touched_files = false;
            let mut repair_attempts = 0usize;
            for step in 1..=cfg.max_steps {
                eprintln!("\n=== [Agent Step {step}/{}] ===", cfg.max_steps);
                let mut reply = http_chat_step(addr, api_key.as_deref(), history, tools_json, &cfg.params)?;
                eprintln!("  [{}]", reply.telemetry);

                if reply.tool_calls.is_empty() {
                    history.push(Message::new("assistant", &reply.text));
                    if touched_files && repair_attempts < 2 && step < cfg.max_steps {
                        if let Err(report) = validate_workspace(&cfg.workdir) {
                            repair_attempts += 1;
                            eprintln!("  [step {step} validation repair]\n{report}");
                            history.push(Message::new(
                                "user",
                                &format!(
                                    "Workspace validation failed:\n{report}\nFix every reported file using `edit_file` or `write_file` before finishing."
                                ),
                            ));
                            continue;
                        }
                    }
                    eprintln!(
                        "=== [Completed in {step} step(s), {:.2}s total] ===",
                        t_goal.elapsed().as_secs_f64()
                    );
                    return Ok(reply.text);
                }

                for (idx, tc) in reply.tool_calls.iter_mut().enumerate() {
                    tc.id = format!("call_s{step}_{idx}");
                }
                history.push(Message::assistant_tool_calls(&reply.text, reply.tool_calls.clone()));

                let mut active_workdir = cfg.workdir.clone();
                for (idx, tc) in reply.tool_calls.iter().enumerate() {
                    if matches!(tc.name.as_str(), "write_file" | "edit_file") {
                        touched_files = true;
                    }
                    let arg_preview = truncate_utf8(&tc.arguments, 140).replace('\n', "\\n");
                    eprintln!("  -> tool `{}`({arg_preview})", tc.name);
                    let outcome =
                        execute_tool(&active_workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx);
                    if let Some(ref next_dir) = outcome.new_workdir {
                        active_workdir.clone_from(next_dir);
                    }
                    let res_preview = truncate_utf8(&outcome.output, 200).replace('\n', " | ");
                    eprintln!("     <- {res_preview}");
                    history.push(Message::tool_result(&tc.id, &outcome.output));
                }

                if reply.finish == FinishReason::Length {
                    eprintln!("  (warning: step hit max_tokens; continuing with executed tool calls)");
                }
            }
            Err(format!("reached --max-steps {} without final response", cfg.max_steps))
        }
    }
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
            "usage:\n  {} --model <model_dir> [--draft DIR] [--draft-mode M] [--cpu|--metal|--ane] [--cache-dir DIR] [--mode nf4g128] [--ctx 32768] --workdir <DIR> [--prompt \"...\"]\n  {} --server 127.0.0.1:18080 --workdir <DIR> [--prompt \"...\"]",
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
        if let Some(mb_str) = flag("--memory-budget-mb") {
            let mb: usize = mb_str.parse().expect("numeric --memory-budget-mb");
            kronos_2t::model::memory_budget::MemoryBudgetProfile::set_global_budget_override_mb(mb);
        }
        let default_threads = kronos_2t::kernels::thread_pool::default_inference_threads() as f64;
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
        let force_cpu = args.iter().any(|a| a == "--cpu" || a == "--fp32-kv" || a == "--ane");
        let require_metal = args.iter().any(|a| a == "--metal");
        if args.iter().any(|a| a == "--fp32-kv") {
            model.enable_q8_kv_cache(false);
        } else if args.iter().any(|a| a == "--q8-kv") {
            model.set_kv_bits(8);
        } else if args.iter().any(|a| a == "--q4-kv") {
            model.set_kv_bits(4);
        }
        if let Some(bits_str) = flag("--kv-bits") {
            let bits: u8 = bits_str.parse().expect("numeric --kv-bits (4, 8, or 32)");
            model.set_kv_bits(bits);
        }
        if let Some(win_str) = flag("--kv-window") {
            let win: usize = win_str.parse().expect("numeric --kv-window");
            let sinks = num("--kv-sinks", 4.0) as usize;
            model.set_kv_sliding_window((win > 0).then_some(win), sinks);
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
                Ok(b) => eprintln!("Neural Engine prefill: {b:?}"),
                Err(e) => eprintln!("Neural Engine disabled: {e}"),
            }
            #[cfg(not(all(feature = "ane", target_os = "macos")))]
            {
                let _ = &mut model;
                eprintln!("built without the `ane` feature; prefilling on CPU");
            }
        }
        let kv_tag = {
            let base = match (model.is_metal_enabled(), model.kv_bits()) {
                (true, 4) => "Metal-Q4-KV",
                (true, _) => "Metal-Q8-KV",
                (false, 4) => "Q4-KV",
                (false, 8) => "Q8-KV",
                _ => "FP32-KV",
            };
            match model.kv_sliding_window() {
                Some(w) => format!("{base}/W{w}"),
                None => base.to_string(),
            }
        };
        eprintln!(
            "loaded {} (weights {:.2} GiB + {} {:.2} GiB @ ctx={}) · workdir={}",
            model.name,
            model.resident_weight_bytes() as f64 / (1u64 << 30) as f64,
            kv_tag,
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
    use kronos_2t::runtime::agent::resolve_jailed_path;

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
        assert!(w.ok && w.validation_ok);
        assert!(w.output.starts_with("ok: wrote 15 bytes"));

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
        assert_eq!(r.output, "1: <h1>Kronos</h1>");

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
        assert!(l.output.contains("index.html"));

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
        assert!(bg.output.contains("exit_code=0"));

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
        assert!(fg.output.starts_with("timed_out after 1s"));
        assert!(fg.output.contains("starting"));

        let _ = fs::remove_dir_all(&dir);
    }
}
