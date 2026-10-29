//! Terminal chat and workspace agent on a compiled Kronos model.
//!
//! Usage: `kronos_chat <model_dir> [flags]`
//!   `--mode M`            `nf4`, `nf4g128`, `nf3nf4`, `nf3`, `b4`, `b5`, `b6` or `pQOGDH` (default `p55454`)
//!   `--calib FILE`        calibration text, needed once to compile each mode
//!   `--calib-tokens N`    (default 16384)
//!   `--cache-dir DIR`     optional directory to read/write compiled `.k2tb` caches
//!   `--ctx N`             context window in tokens (default 2048)
//!   `--threads N`         (default 8)
//!   `--fp32-kv`           use an `FP32` KV cache instead of the default INT8 one
//!   `--cpu`               force CPU inference (disables default Metal GPU backend for NF4)
//!   `--metal`             require Metal GPU inference
//!   `--draft DIR`         draft model for speculative decoding (`--draft-mode M`)
//!   `--agent`, `--tools`  start with workspace tools and post-edit validation enabled
//!   `--workdir DIR`       workspace root for `--agent` (default temp workspace)
//!   `--max-steps N`       max tool steps per turn in agent mode (default 10)
//!   `--cmd-timeout S`     per-command timeout in seconds in agent mode (default 20)
//!   `--temp T`, `--top-k K`, `--top-p P`, `--repeat-penalty R`, `--max-new N`, `--seed N`
//!   `--system TEXT`       system prompt
//!   `--ane`               prefill on the Neural Engine (`ane` feature, models in `<model_dir>/ane`)
//!
//! Commands: `/agent` toggles workspace tools, `/workdir [dir]` views or sets workspace,
//! `/files` lists workspace files, `/validate` validates workspace syntax, `/reset` clears history, `/exit` quits.

use kronos_2t::agent::{
    list_workspace_files, run_agent_turn, tool_definitions_json, truncate_utf8, validate_workspace,
    AgentEvent, AgentTurnConfig, AGENT_SYSTEM_PROMPT,
};
use kronos_2t::chat::{ChatModel, GenParams, Message};
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};

fn prepare_workdir(path: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    Ok(path.canonicalize().unwrap_or_else(|_| path.to_path_buf()))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!(
            "usage: {} <model_dir> [--mode p55454] [--draft DIR] [--agent] [--workdir DIR] [--ctx 2048] [--temp 0.7] ...",
            args[0]
        );
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let num = |name: &str, default: f64| flag(name).map_or(default, |v| v.parse().expect("numeric flag"));
    let model_dir = Path::new(&args[1]);
    let mode = flag("--mode").unwrap_or_else(|| "p55454".to_string());
    let calib = flag("--calib");
    let calib_tokens = num("--calib-tokens", 16384.0) as usize;
    let mut agent_mode = args.iter().any(|a| a == "--agent" || a == "--tools");

    eprintln!("loading {} ({mode})...", model_dir.display());
    let default_threads = kronos_2t::thread_pool::default_inference_threads() as f64;
    let calib_opt = calib.as_deref().map(|c| (Path::new(c), calib_tokens));
    let cache_dir_flag = flag("--cache-dir");
    let cache_dir_opt = cache_dir_flag.as_deref().map(Path::new);
    let default_ctx = if agent_mode { 32768.0 } else { 2048.0 };
    let mut model = ChatModel::load_with_cache_dir(
        model_dir,
        cache_dir_opt,
        &mode,
        num("--ctx", default_ctx) as usize,
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
    let default_temp = if agent_mode { 0.0 } else { 0.7 };
    let default_max_new = if agent_mode { 4096.0 } else { 512.0 };
    let mut params = GenParams {
        temperature: num("--temp", default_temp) as f32,
        top_k: num("--top-k", 40.0) as usize,
        top_p: num("--top-p", 0.9) as f32,
        repeat_penalty: num("--repeat-penalty", 1.1) as f32,
        max_tokens: num("--max-new", default_max_new) as usize,
        seed: num("--seed", 0.0) as u64,
    };
    let max_steps = num("--max-steps", 10.0) as usize;
    let cmd_timeout = num("--cmd-timeout", 20.0) as u64;
    let mut workdir = match flag("--workdir") {
        Some(w) => PathBuf::from(w),
        None => std::env::temp_dir().join("kronos_chat_workspace"),
    };
    if agent_mode {
        workdir = prepare_workdir(&workdir).unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    }

    let explicit_system = flag("--system");
    let build_base_history = |is_agent: bool| -> Vec<Message> {
        if let Some(ref s) = explicit_system {
            vec![Message::new("system", s)]
        } else if is_agent {
            vec![Message::new("system", AGENT_SYSTEM_PROMPT)]
        } else {
            Vec::new()
        }
    };
    let mut history = build_base_history(agent_mode);
    let tools_json = tool_definitions_json();

    let kv_tag = if model.is_metal_enabled() {
        "Metal-Q8-KV"
    } else if model.is_q8_kv_enabled() {
        "Q8-KV"
    } else {
        "FP32-KV"
    };
    eprintln!(
        "{} ready (weights {:.2} GiB + {} {:.2} GiB @ ctx={}) · /agent /workdir /files /validate /reset /exit",
        model.name,
        model.resident_weight_bytes() as f64 / (1u64 << 30) as f64,
        kv_tag,
        model.kv_cache_bytes() as f64 / (1u64 << 30) as f64,
        model.context_size(),
    );
    if agent_mode {
        eprintln!("agent mode active · workdir={}", workdir.display());
    }
    let stdin = io::stdin();
    let mut out = io::stdout();
    loop {
        if agent_mode {
            print!("\nagent> ");
        } else {
            print!("\n> ");
        }
        out.flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if matches!(trimmed, "/exit" | "/quit") {
            break;
        }
        if trimmed == "/reset" {
            history = build_base_history(agent_mode);
            model.reset_cache();
            println!("(conversation cleared)");
            continue;
        }
        if trimmed == "/agent" {
            agent_mode = !agent_mode;
            if agent_mode {
                match prepare_workdir(&workdir) {
                    Ok(p) => workdir = p,
                    Err(e) => {
                        println!("(workdir error: {e})");
                        agent_mode = false;
                        continue;
                    }
                }
                params.max_tokens = params.max_tokens.max(4096);
            }
            history = build_base_history(agent_mode);
            model.reset_cache();
            println!(
                "(agent mode {} · workdir={})",
                if agent_mode { "enabled" } else { "disabled" },
                workdir.display()
            );
            continue;
        }
        if trimmed == "/workdir" || trimmed.starts_with("/workdir ") {
            let arg = trimmed.strip_prefix("/workdir").unwrap_or("").trim();
            if arg.is_empty() {
                println!("(workdir={})", workdir.display());
            } else {
                match prepare_workdir(Path::new(arg)) {
                    Ok(p) => {
                        workdir = p;
                        println!("(workdir={})", workdir.display());
                    }
                    Err(e) => println!("(workdir error: {e})"),
                }
            }
            continue;
        }
        if trimmed == "/files" {
            let files = list_workspace_files(&workdir);
            if files.is_empty() {
                println!("(no files in {})", workdir.display());
            } else {
                for f in files {
                    let status = if f.validation_ok { "valid" } else { &f.validation_msg };
                    println!("  {} ({} B) [{}]", f.path, f.bytes, status);
                }
            }
            continue;
        }
        if trimmed == "/validate" {
            match validate_workspace(&workdir) {
                Ok(()) => println!("(workspace validation ok)"),
                Err(report) => println!("(workspace validation errors)\n{report}"),
            }
            continue;
        }

        history.push(Message::new("user", trimmed));
        if agent_mode {
            let cfg = AgentTurnConfig {
                workdir: workdir.clone(),
                params: params.clone(),
                max_steps,
                cmd_timeout_secs: cmd_timeout,
                expect_tool_call: true,
            };
            let mut printed_text = false;
            let res = run_agent_turn(&mut model, &mut history, &tools_json, &cfg, |ev| {
                match ev {
                    AgentEvent::ContextCompacted {
                        before_tokens,
                        after_tokens,
                        compacted_messages,
                    } => {
                        println!(
                            "[context compacted: {before_tokens} -> {after_tokens} tokens ({compacted_messages} items pruned)]"
                        );
                    }
                    AgentEvent::StepStart { .. } => {
                        printed_text = false;
                    }
                    AgentEvent::TextDelta(piece) => {
                        if !piece.is_empty() {
                            printed_text = true;
                            print!("{piece}");
                            out.flush().ok();
                        }
                    }
                    AgentEvent::ToolProgress { .. } => {}
                    AgentEvent::StepStats { step, completion: c } => {
                        if printed_text {
                            println!();
                            printed_text = false;
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
                        println!(
                            "[step {step} · {} prompt tok ({} new, {}) @ {:.0} tok/s · {} gen tok @ {:.1} tok/s{draft_info}]",
                            c.prompt_tokens,
                            c.prefilled_tokens,
                            c.prefill_unit,
                            c.prefilled_tokens as f64 / c.prefill_secs.max(1e-9),
                            c.completion_tokens,
                            c.completion_tokens as f64 / c.decode_secs.max(1e-9),
                        );
                    }
                    AgentEvent::ToolStart { call, .. } => {
                        let preview = truncate_utf8(&call.arguments, 100).replace('\n', "\\n");
                        println!("  -> {}({preview})", call.name);
                    }
                    AgentEvent::ToolDone { outcome, .. } => {
                        let preview = truncate_utf8(&outcome.output, 160).replace('\n', " | ");
                        println!("  <- {preview}");
                    }
                    AgentEvent::ValidationRepair { step, report } => {
                        println!("  [step {step} validation repair]\n{report}");
                    }
                    AgentEvent::Verified { step, report } => {
                        println!(
                            "  [step {step} verified in {:.1} ms: {}]",
                            report.elapsed_ms, report.summary
                        );
                    }
                    AgentEvent::Done { steps, total_secs, .. } => {
                        println!("[completed in {steps} step(s), {total_secs:.2}s]");
                    }
                }
                true
            });
            params.seed = params.seed.wrapping_add(1);
            if let Err(e) = res {
                println!("\n(agent error: {e})");
            }
            continue;
        }

        let result = model.generate(&history, &params, |piece| {
            print!("{piece}");
            out.flush().ok();
            true
        });
        params.seed = params.seed.wrapping_add(1);
        match result {
            Ok(c) => {
                let draft_info = match c.acceptance_rate() {
                    Some(r) => format!(" · accept {:.1}% ({}/{})", r * 100.0, c.accepted_tokens, c.drafted_tokens),
                    None => String::new(),
                };
                println!(
                    "\n[{} prompt tok ({} new, {}) @ {:.0} tok/s · {} reply tok @ {:.1} tok/s{draft_info}]",
                    c.prompt_tokens,
                    c.prefilled_tokens,
                    c.prefill_unit,
                    c.prefilled_tokens as f64 / c.prefill_secs.max(1e-9),
                    c.completion_tokens,
                    c.completion_tokens as f64 / c.decode_secs.max(1e-9)
                );
                history.push(Message::new("assistant", &c.text));
            }
            Err(e) => {
                println!("\n(error: {e})");
                history.pop();
            }
        }
    }
}
