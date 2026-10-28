//! Terminal chat on a Kronos-compiled SmolLM2-Instruct model.
//!
//! Usage: `kronos_chat <model_dir> [flags]`
//!   `--mode M`            `b4`, `b5`, `b6` or `pQOGDH` (default `p55454`)
//!   `--calib FILE`        calibration text, needed once to compile each mode
//!   `--calib-tokens N`    (default 16384)
//!   `--ctx N`             context window in tokens (default 2048)
//!   `--threads N`         (default 8)
//!   `--temp T`, `--top-k K`, `--top-p P`, `--repeat-penalty R`, `--max-new N`, `--seed N`
//!   `--system TEXT`       system prompt
//!
//! Commands: `/reset` clears the conversation, `/exit` quits.

use kronos_2t::chat::{ChatModel, GenParams, Message};
use std::io::{self, BufRead, Write};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1].starts_with("--") {
        eprintln!("usage: {} <model_dir> [--mode p55454] [--calib FILE] [--ctx 2048] [--temp 0.7] ...", args[0]);
        std::process::exit(2);
    }
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned());
    let num = |name: &str, default: f64| flag(name).map_or(default, |v| v.parse().expect("numeric flag"));
    let model_dir = Path::new(&args[1]);
    let mode = flag("--mode").unwrap_or_else(|| "p55454".to_string());
    let calib = flag("--calib");
    let calib_tokens = num("--calib-tokens", 16384.0) as usize;

    eprintln!("loading {} ({mode})...", model_dir.display());
    let mut model = ChatModel::load(
        model_dir,
        &mode,
        num("--ctx", 2048.0) as usize,
        num("--threads", 8.0) as usize,
        calib.as_deref().map(|c| (Path::new(c), calib_tokens)),
    )
    .unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let mut params = GenParams {
        temperature: num("--temp", 0.7) as f32,
        top_k: num("--top-k", 40.0) as usize,
        top_p: num("--top-p", 0.9) as f32,
        repeat_penalty: num("--repeat-penalty", 1.1) as f32,
        max_tokens: num("--max-new", 512.0) as usize,
        seed: num("--seed", 0.0) as u64,
    };
    let system: Vec<Message> = flag("--system").map(|s| Message::new("system", &s)).into_iter().collect();
    let mut history = system.clone();

    eprintln!("{} ready · /reset /exit", model.name);
    let stdin = io::stdin();
    let mut out = io::stdout();
    loop {
        print!("\n> ");
        out.flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history = system.clone();
                println!("(conversation cleared)");
                continue;
            }
            msg => history.push(Message::new("user", msg)),
        }
        let result = model.generate(&history, &params, |piece| {
            print!("{piece}");
            out.flush().ok();
            true
        });
        params.seed = params.seed.wrapping_add(1);
        match result {
            Ok(c) => {
                println!(
                    "\n[{} prompt tok ({} new) @ {:.0} tok/s · {} reply tok @ {:.1} tok/s]",
                    c.prompt_tokens,
                    c.prefilled_tokens,
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
