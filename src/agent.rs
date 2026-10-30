#![forbid(unsafe_code)]
//! Sandboxed workspace tools, deterministic post-edit syntax validation, and KV-preserving agent loop.

use crate::chat::{ChatModel, Completion, GenParams, Message, ToolCall};
use crate::json::{quote, Json};
use std::fs::{self, File};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const MAX_READ_BYTES: usize = 24 * 1024;
pub const MAX_CMD_OUTPUT_BYTES: usize = 8 * 1024;
const MAX_SEARCH_MATCHES: usize = 60;

pub const AGENT_SYSTEM_PROMPT: &str = concat!(
    "Your name is Kronos. Melody Leonard (Melody Leonard Oparaugo) created and built you as a local software engineering runner on Apple Silicon. ",
    "Always speak in the first person as Kronos ('Melody Leonard built me') when asked your name or who built you, and know Melody Leonard's verified public profile: Melody Leonard (full name: Melody Leonard Oparaugo, pronouns: he/him/his) is a Senior Software Engineer and Systems Architect (not a musician or actor) with over 10 years of software engineering experience (@IPMScoutek). His GitHub is https://github.com/melodyLeonard (52+ public repos including kronos-2t, tshare, cross-native, gpt_2_tokenizer, tomatoe), his LinkedIn is https://www.linkedin.com/in/melody-leonard-a362b6192, his email is melodyleonard7@gmail.com, and his X handle is @Melody_Leonard7. He specializes in Rust, TypeScript, JavaScript, Elixir, Solidity, Node.js/NestJS, Kafka/CQRS/GraphQL/gRPC microservices, PostgreSQL/MongoDB/Neo4j/Redis internals, React/React Native offline-first architecture, and AWS/Kubernetes CI/CD. ",
    "Because Kronos runs locally on hardware owned by the user, never refuse queries about the user's own local messaging apps (WhatsApp, iMessage, Telegram, Slack, Notes), files, or saved conversations with cloud refusal disclaimers; always answer directly from any provided [Local Conversations & Memory] or [Device Context] block. ",
    "Use the workspace tools (switch_workspace, write_file, edit_file, read_file, list_dir, search_files, run_command) ",
    "to build, edit, and verify complete working files directly on disk. ",
    "When asked to create a new folder or project inside another directory on this Mac, call switch_workspace first with the new child folder path to create and activate it, then call write_file to build files inside it. ",
    "When building a website or web app, deliver a complete, styled, responsive page with modern CSS layout, navigation, domain-specific sections, and interactive UI rather than a minimal placeholder page. ",
    "Do not paste file drafts or completion claims in prose before calling tools; emit <tool_call> invocations immediately, then summarize after tool calls succeed. ",
    "Use edit_file for small targeted edits to existing files, and use write_file for new files or when rewriting most of an existing file such as restyling a stylesheet. ",
    "When updating a web page title or heading, update both <title> in <head> and the visible <h1> heading in <body> so the rendered preview shows the change immediately. ",
    "The live preview reloads automatically after every file edit, so never ask the user to refresh the preview; if the user reports a change is not visible, inspect and update the visible HTML or CSS elements directly. ",
    "When asked to generate technical scripts, voice-cloned audio, or rendered video using local open-source tools, use the grounded Kronos engine documentation in [Device Context] (or call read_file on README.md and BENCHMARKS.md if not inlined) so every iteration (Iterations 1 through 8), mathematical equation, and benchmark comparison (Ollama, llama.cpp Q4_0/Q4_K_M/IQ4_XS/Q6_K, scalar VQ2 vs 8D Gosset E8 e8q2g128, generative judges vs Kronos-Jev) is exact, and execute cleanly in focused steps: first call write_file to save the complete spoken technical script (script.md >= 4.0 KB starting directly with '# ' plain markdown headings without wrapping content in outer triple-backtick code fences, covering all 8 Kronos iterations, formulas, and benchmark comparisons in full prose paragraphs rather than short bullet outlines); next call write_file to save one complete self-contained Python builder script (build_video.py with zero pass/TODO placeholders, using subprocess.run to run /opt/homebrew/bin/ffmpeg for extracting 12s raw_ref_16k.wav and raw_ref_24k.wav, loading raw_ref_16k.wav via wave + np.frombuffer into faster_whisper segs, _ = wm.transcribe(pcm), slicing raw_ref_24k.wav strictly to int(end_sec * 24000) int16 frames into ref_24k.wav so ref_24k.wav matches non-empty ref_text word-for-word, rendering 6 to 8 1920x1080 dark-slate PNG slides slides/slide_01.png..slide_0N.png with PIL.ImageDraw + /System/Library/Fonts/Helvetica.ttc + draw.textbbox, synthesizing one single-line symbol-free spoken English paragraph of 130 to 210 chars per slide via mlx_audio.tts.generate.generate_audio(..., ref_audio='ref_24k.wav', ref_text=ref_text, lang_code='en', temperature=0.55, top_p=0.9, repetition_penalty=1.05, output_path='audio', file_prefix=f'seg_{idx:02d}', audio_format='wav', join_audio=True), ASR-verifying EVERY audio/seg_XX.wav with faster_whisper checking overall word_recall >= 0.72 and both first_half_recall >= 0.65 and second_half_recall >= 0.65 with temperature=0.4 retry, writing audio/coherence_report.json, muxing segments/seg_XX.mp4, and concatenating into workspace-root kronos_explained.mp4); then run build_video.py with the matched .venv python3 via run_command and verify kronos_explained.mp4 with /opt/homebrew/bin/ffprobe before summarizing. ",
    "On multi-step modularization, refactoring, styling, or self-testing tasks, write 2 to 4 complete non-empty files per step using write_file <tool_call> blocks (never write 0-byte empty stubs and never paste file contents inside Markdown code fences in prose) so every tool call closes cleanly within max_tokens, and keep iterating with tool calls until every module is created and wired up, inline monolith blocks are extracted, requested visual styles and interactive behaviors are complete, and you have written and executed an automated verification command via run_command (fixing any non-zero exit or assertion failure before summarizing). ",
    "If a tool response reports an error or validation failure, fix it with a tool call immediately before finishing."
);

#[derive(Clone, Debug)]
pub struct WorkspaceFileEntry {
    pub path: String,
    pub bytes: u64,
    pub validation_ok: bool,
    pub validation_msg: String,
}

#[derive(Clone, Debug)]
pub struct FileChangeDiff {
    pub path: String,
    pub existed_before: bool,
    pub before_content: String,
    pub added_lines: usize,
    pub removed_lines: usize,
    pub unified_diff: String,
}

#[derive(Clone, Debug)]
pub struct VerificationReport {
    pub checks_run: usize,
    pub summary: String,
    pub details: Vec<String>,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ToolExecOutcome {
    pub ok: bool,
    pub validation_ok: bool,
    pub validation_msg: String,
    pub output: String,
    pub diff: Option<FileChangeDiff>,
    pub new_workdir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnRoute {
    Chat,
    Agent,
    Computer,
}

impl TurnRoute {
    pub fn as_str(self) -> &'static str {
        match self {
            TurnRoute::Chat => "chat",
            TurnRoute::Agent => "agent",
            TurnRoute::Computer => "computer",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CompactionReport {
    pub compacted: bool,
    pub before_tokens: usize,
    pub after_tokens: usize,
    pub compacted_messages: usize,
}

pub enum AgentEvent<'a> {
    JevDecision {
        plan: &'a crate::jev::JevPrepassPlan,
    },
    ContextCompacted {
        before_tokens: usize,
        after_tokens: usize,
        compacted_messages: usize,
    },
    StepStart {
        step: usize,
        max_steps: usize,
    },
    TextDelta(&'a str),
    ToolProgress {
        step: usize,
        name: &'a str,
        target: &'a str,
        bytes: usize,
    },
    StepStats {
        step: usize,
        completion: &'a Completion,
    },
    ToolStart {
        step: usize,
        call: &'a ToolCall,
    },
    ToolDone {
        step: usize,
        call: &'a ToolCall,
        outcome: &'a ToolExecOutcome,
    },
    WorkspaceChanged {
        workdir: &'a Path,
    },
    ValidationRepair {
        step: usize,
        report: &'a str,
    },
    Verified {
        step: usize,
        report: &'a VerificationReport,
    },
    JevVerification {
        step: usize,
        eval: &'a crate::jev::JevRubricEvaluation,
        rolled_back: bool,
    },
    Done {
        steps: usize,
        total_secs: f64,
        final_text: &'a str,
    },
}

#[derive(Clone)]
pub struct AgentTurnConfig {
    pub workdir: PathBuf,
    pub params: GenParams,
    pub max_steps: usize,
    pub cmd_timeout_secs: u64,
    pub expect_tool_call: bool,
}

pub type SubAgentSpawner =
    dyn Fn(&Path, &str, &str, bool) -> Result<String, String> + Send + Sync;

pub fn spawn_agent_tool_definition_json() -> String {
    r#"{"type":"function","function":{"name":"spawn_agent","description":"Spawn an independent Kronos worker agent thread in its own workspace directory to execute a parallel sub-task concurrently while other tasks continue.","parameters":{"type":"object","properties":{"workdir":{"type":"string","description":"Target workspace directory path for the worker agent"},"label":{"type":"string","description":"Short descriptive role label for the worker agent"},"task":{"type":"string","description":"Complete actionable task prompt for the worker agent"},"wait":{"type":"boolean","description":"If true, wait for the worker agent to finish and return its summary; if false (default), run concurrently in the background"}},"required":["workdir","task"]}}}"#.to_string()
}

pub fn is_multi_agent_prompt(text: &str) -> bool {
    let l = text.to_ascii_lowercase();
    l.contains("spawn_agent")
        || l.contains("subagent")
        || l.contains("sub-agent")
        || l.contains("multiple task")
        || l.contains("multiple agent")
        || l.contains("parallel agent")
        || l.contains("parallel task")
        || l.contains("two agent")
        || l.contains("own agents")
        || (l.contains("in parallel") && (l.contains("agent") || l.contains("task")))
}

pub fn tool_definitions_json() -> Vec<String> {
    vec![
        r#"{"type":"function","function":{"name":"switch_workspace","description":"Create or switch Kronos's active project workspace directory on disk (for example creating a new project folder inside ~/Documents/orgProj/<new-folder-name>) so subsequent write_file, edit_file, read_file, and run_command calls operate inside it.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Target project folder path (e.g. '/Users/.../Documents/orgProj/emr-portal' or '~/Documents/orgProj/emr-portal')"},"subfolder":{"type":"string","description":"Optional new child folder name to create inside `path` when `path` is a parent directory"}},"required":["path"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"write_file","description":"Write or overwrite a UTF-8 workspace file and validate syntax.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative workspace path"},"content":{"type":"string","description":"Full file contents"}},"required":["path","content"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"edit_file","description":"Replace a unique text span in an existing workspace file and validate syntax.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative workspace path"},"old_text":{"type":"string","description":"Existing unique text block"},"new_text":{"type":"string","description":"Replacement text block"}},"required":["path","old_text","new_text"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"read_file","description":"Read a workspace file (or a local file path from [Device Context]) with 1-based line numbers.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative workspace path or local file path"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}},"required":["path"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"list_dir","description":"List directory entries inside workspace.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative directory (default '.')"}},"required":[]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"search_files","description":"Search workspace files for a substring pattern.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"case_insensitive":{"type":"boolean"}},"required":["pattern"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"run_command","description":"Run sh -c command inside workspace with timeout.","parameters":{"type":"object","properties":{"cmd":{"type":"string"}},"required":["cmd"]}}}"#.to_string(),
    ]
}

/// Fast latent-feature intent gate routing user prompts between direct streaming chat and workspace tools.
pub fn classify_turn_intent(
    user_text: &str,
    has_images: bool,
    workspace_has_files: bool,
    history_used_tools: bool,
) -> TurnRoute {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return TurnRoute::Chat;
    }

    let greetings = [
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
        "who are you",
        "what can you do",
        "help",
    ];
    let stripped = lower.trim_end_matches(['?', '.', ',']);
    if greetings.contains(&stripped) {
        return TurnRoute::Chat;
    }

    if crate::device::is_computer_control_prompt(&lower)
        && crate::device::global_device_manager()
            .current_config()
            .allow_computer_control
    {
        return TurnRoute::Computer;
    }

    let mut score = 0i32;

    let file_markers = [
        ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".py", ".rs", ".json", ".svg", ".md",
        ".toml", "index.", "style.", "app.", "src/", "./",
    ];
    if file_markers.iter().any(|m| lower.contains(m)) {
        score += 6;
    }

    let action_phrases = [
        "build ",
        "build a",
        "create ",
        "make a ",
        "make an ",
        "make it ",
        "make the ",
        "write file",
        "write a file",
        "write code to",
        "save to ",
        "edit ",
        "update ",
        "modify ",
        "change the ",
        "change it",
        "style ",
        "restyle",
        "beautify",
        "redesign",
        "refactor",
        "modularize",
        "split ",
        "extract ",
        "separate ",
        "scaffold",
        "implement ",
        "fix ",
        "repair ",
        "debug ",
        "test ",
        "verify ",
        "add a ",
        "add an ",
        "add the ",
        "remove the ",
        "delete the ",
        "rename ",
        "run ",
        "execute ",
        "compile ",
        "center the ",
        "align ",
    ];
    for p in action_phrases {
        if lower.starts_with(p) || lower.contains(&format!(" {p}")) || lower.contains(&format!(", {p}")) {
            score += 4;
            break;
        }
    }

    let artifact_nouns = [
        "webpage",
        "web page",
        "website",
        "web app",
        "landing page",
        "dashboard",
        "todo",
        "to-do",
        "tracker",
        "calculator",
        "timer",
        "game",
        "header",
        "navbar",
        "sidebar",
        "footer",
        "button",
        "buttons",
        "input",
        "inputs",
        "form",
        "modal",
        "stylesheet",
        "preview",
        "workspace",
        "title",
        "heading",
        "dark mode",
        "light mode",
    ];
    if artifact_nouns.iter().any(|n| lower.contains(n)) {
        score += 2;
    }

    if workspace_has_files || history_used_tools {
        let tweak_terms = [
            "darker", "lighter", "bigger", "smaller", "rounded", "modern", "cleaner", "gradient",
            "shadow", "padding", "margin", "font", "color", "colors", "theme", "layout",
            "responsive", "match this", "like this", "still the same", "not working",
            "doesn't work", "doesnt work", "didn't change", "didnt change", "nothing changed",
            "don't see", "dont see",
        ];
        if score >= 2 || tweak_terms.iter().any(|t| lower.contains(t)) {
            score += 3;
        }
    }

    if has_images {
        let img_build_terms = [
            "build", "clone", "recreate", "replicate", "match", "style", "like this", "fix",
            "embed", "use this", "turn this into", "implement",
        ];
        if img_build_terms.iter().any(|t| lower.contains(t)) {
            score += 5;
        }
    }

    let qa_prefixes = [
        "what ",
        "which ",
        "who ",
        "who's ",
        "whos ",
        "where ",
        "where's ",
        "wheres ",
        "when ",
        "what's ",
        "whats ",
        "how's ",
        "hows ",
        "why's ",
        "whys ",
        "is there ",
        "are there ",
        "describe ",
        "read this",
        "read the text",
        "can you read",
        "can you explain",
        "explain ",
        "how does ",
        "how do ",
        "why is ",
        "why does ",
        "compare ",
        "summarize ",
        "analyze ",
        "translate ",
        "write a haiku",
        "write a poem",
        "write a story",
    ];
    let has_followup_mutation = lower.contains(" and fix")
        || lower.contains(" and build")
        || lower.contains(" and update")
        || lower.contains(" and edit")
        || lower.contains(" and change")
        || lower.contains(" and style");
    if !has_followup_mutation && qa_prefixes.iter().any(|q| lower.starts_with(q)) {
        score -= 6;
    }

    if score > 0 {
        TurnRoute::Agent
    } else {
        TurnRoute::Chat
    }
}

/// Resolves `user_rel` inside `workdir`, rejecting `..` traversal or paths outside `workdir`.
pub fn resolve_jailed_path(workdir: &Path, user_rel: &str) -> Result<PathBuf, String> {
    let trimmed = user_rel.trim();
    if trimmed.is_empty() || trimmed == "." {
        return Ok(workdir.to_path_buf());
    }
    let candidate = Path::new(trimmed);
    if candidate.is_absolute() {
        let canon_work = workdir.canonicalize().unwrap_or_else(|_| workdir.to_path_buf());
        if let Ok(stripped) = candidate.strip_prefix(&canon_work) {
            return resolve_jailed_path(workdir, &stripped.to_string_lossy());
        }
        if let Ok(stripped) = candidate.strip_prefix(workdir) {
            return resolve_jailed_path(workdir, &stripped.to_string_lossy());
        }
        return Err(format!(
            "path `{trimmed}` is outside current workspace `{}`; call `switch_workspace` first to create or switch to that project directory",
            workdir.display()
        ));
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

/// Resolves a read-only path: prefers `workdir`, and also allows reading existing non-sensitive
/// files under the user's home directory or `/tmp` (such as `[Device Context]` documentation)
/// while keeping all mutations (`write_file`, `edit_file`) strictly jailed to `workdir`.
pub fn resolve_readable_path(workdir: &Path, user_rel: &str) -> Result<PathBuf, String> {
    let trimmed = user_rel.trim();
    if trimmed.is_empty() {
        return Err("path cannot be empty".to_string());
    }
    if trimmed != "~" && !trimmed.starts_with("~/") {
        if let Ok(p) = resolve_jailed_path(workdir, trimmed) {
            return Ok(p);
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let expanded = if let Some(rest) = trimmed.strip_prefix("~/") {
        let Some(ref h) = home else {
            return Err("HOME is not set".to_string());
        };
        h.join(rest)
    } else {
        PathBuf::from(trimmed)
    };
    if !expanded.is_absolute() {
        return resolve_jailed_path(workdir, user_rel);
    }
    let norm = normalize_lexical_path(&expanded);
    let canon = norm.canonicalize().unwrap_or_else(|_| norm.clone());
    let raw_tmp = std::env::temp_dir();
    let norm_tmp = normalize_lexical_path(&raw_tmp);
    let canon_tmp = raw_tmp.canonicalize().unwrap_or_else(|_| norm_tmp.clone());
    let in_tmp = canon.starts_with("/tmp/")
        || canon.starts_with("/private/tmp/")
        || canon.starts_with(&canon_tmp)
        || canon.starts_with(&norm_tmp);
    let in_safe_home = if let Some(ref h) = home {
        let norm_home = normalize_lexical_path(h);
        let canon_home = h.canonicalize().unwrap_or_else(|_| norm_home.clone());
        let rel_opt = canon
            .strip_prefix(&canon_home)
            .ok()
            .or_else(|| canon.strip_prefix(&norm_home).ok());
        if let Some(rel) = rel_opt {
            let comps: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            let blocked_first = comps.first().is_some_and(|first| {
                matches!(
                    first.as_str(),
                    ".ssh"
                        | ".gnupg"
                        | ".aws"
                        | ".kube"
                        | ".docker"
                        | ".netrc"
                        | ".npmrc"
                        | ".pypirc"
                )
            });
            let blocked_keychain = comps.len() >= 2
                && comps[0].eq_ignore_ascii_case("Library")
                && comps[1].eq_ignore_ascii_case("Keychains");
            !blocked_first && !blocked_keychain
        } else {
            false
        }
    } else {
        false
    };
    if !in_tmp && !in_safe_home {
        return Err(format!(
            "read path `{trimmed}` is outside workspace and permitted user directories"
        ));
    }
    if !canon.is_file() {
        return Err(format!("file `{trimmed}` does not exist on disk"));
    }
    Ok(canon)
}

pub fn truncate_utf8(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n...[truncated {} bytes]", &s[..end], s.len() - end)
}

fn truncate_utf8_head_tail(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes || max_bytes < 512 {
        return truncate_utf8(s, max_bytes);
    }
    let head_budget = (max_bytes * 7) / 16;
    let tail_budget = max_bytes.saturating_sub(head_budget + 64);
    let mut head_end = head_budget.min(s.len());
    while head_end > 0 && !s.is_char_boundary(head_end) {
        head_end -= 1;
    }
    let mut tail_start = s.len().saturating_sub(tail_budget).max(head_end);
    while tail_start < s.len() && !s.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    if tail_start <= head_end {
        return s.to_string();
    }
    format!(
        "{}\n...[omitted {} middle bytes]...\n{}",
        &s[..head_end],
        tail_start - head_end,
        &s[tail_start..]
    )
}

pub fn mime_type_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "md" | "markdown" => "text/markdown; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
        "mp4" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "wav" => "audio/wav",
        "m4a" => "audio/mp4",
        "mp3" => "audio/mpeg",
        _ => "text/plain; charset=utf-8",
    }
}

fn should_skip_entry(name: &str) -> bool {
    name == ".git"
        || name == "node_modules"
        || name == "target"
        || name == "__pycache__"
        || name == ".DS_Store"
        || name == ".kronos_media"
        || name == ".kronos"
        || name.starts_with(".kronos_cmd_")
}

pub fn list_workspace_files(workdir: &Path) -> Vec<WorkspaceFileEntry> {
    let mut out = Vec::new();
    collect_files_rec(workdir, workdir, &mut out);
    out.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

fn collect_files_rec(root: &Path, cur: &Path, out: &mut Vec<WorkspaceFileEntry>) {
    let Ok(rd) = fs::read_dir(cur) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if should_skip_entry(&name) {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if meta.is_dir() {
            collect_files_rec(root, &path, out);
        } else if meta.is_file() {
            if let Ok(rel) = path.strip_prefix(root) {
                let rel_str = rel.to_string_lossy().replace('\\', "/");
                let (v_ok, v_msg) = match validate_file(root, &path, true) {
                    Ok(note) => (true, if note.is_empty() { "ok".to_string() } else { note }),
                    Err(err) => (false, err),
                };
                out.push(WorkspaceFileEntry {
                    path: rel_str,
                    bytes: meta.len(),
                    validation_ok: v_ok,
                    validation_msg: v_msg,
                });
            }
        }
    }
}

/// Validates syntax and local asset links of a single file.
pub fn validate_file(workdir: &Path, file_path: &Path, strict_assets: bool) -> Result<String, String> {
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(
        ext.as_str(),
        "json" | "js" | "mjs" | "ts" | "css" | "html" | "htm" | "py" | "md"
    ) {
        return Ok(String::new());
    }
    let content = fs::read_to_string(file_path).map_err(|e| format!("read error: {e}"))?;
    validate_content_by_ext(workdir, file_path, &ext, &content, strict_assets)
}

pub fn validate_content_by_ext(
    workdir: &Path,
    file_path: &Path,
    ext: &str,
    content: &str,
    strict_assets: bool,
) -> Result<String, String> {
    if matches!(
        ext,
        "json" | "js" | "mjs" | "ts" | "css" | "html" | "htm" | "py" | "md"
    ) && content.trim().is_empty()
    {
        return Err(
            "file is empty (0 bytes) — pass the complete source code in `write_file` `content`"
                .to_string(),
        );
    }
    match ext {
        "json" => {
            Json::parse(content).map_err(|e| format!("JSON syntax error: {e}"))?;
            Ok(String::new())
        }
        "js" | "mjs" | "ts" => {
            validate_js_like(content, 1)?;
            validate_esm_commonjs_mix(ext, content)?;
            Ok(String::new())
        }
        "css" => {
            validate_css(content, 1)?;
            Ok(String::new())
        }
        "html" | "htm" => validate_html(workdir, file_path, content, strict_assets),
        "py" => {
            validate_python_lexical(content)?;
            validate_python_runtime_pitfalls(file_path, content)?;
            Ok(String::new())
        }
        "md" => {
            validate_markdown_script(file_path, content)?;
            Ok(String::new())
        }
        _ => Ok(String::new()),
    }
}

fn validate_markdown_script(file_path: &Path, content: &str) -> Result<(), String> {
    let fname = file_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let is_script_md = matches!(fname.as_str(), "script.md" | "narration.md" | "video_script.md");
    if !is_script_md {
        return Ok(());
    }
    let lower = content.to_ascii_lowercase();
    let is_kronos_tech_script = lower.contains("kronos")
        && (lower.contains("iteration 1")
            || lower.contains("sub-2-bit")
            || lower.contains("nf4")
            || lower.contains("e8"));
    if !is_kronos_tech_script {
        return Ok(());
    }
    if content.len() < 3200 {
        return Err(format!(
            "`{fname}` is truncated ({} bytes < 3200 minimum) — write the complete technical narration script covering all 8 Kronos iterations (Iteration 1 through Iteration 8 / Kronos-Jev), explicit mathematical equations, and quantitative benchmarks vs llama.cpp and Ollama without wrapping the file in outer markdown fences",
            content.len()
        ));
    }
    let has_iter8 = lower.contains("iteration 8")
        || lower.contains("kronos-jev")
        || lower.contains("joint embedding");
    if !has_iter8 {
        return Err(format!(
            "`{fname}` stops before Iteration 8 (Kronos-Jev Joint Embedding Plan-Execute-Verify) — include all 8 iterations from Iteration 1 through Iteration 8"
        ));
    }
    if !lower.contains("llama.cpp") && !lower.contains("ollama") {
        return Err(format!(
            "`{fname}` is missing quantitative benchmark comparisons against llama.cpp and Ollama — include the benchmark comparison section"
        ));
    }
    Ok(())
}

fn python_imports_symbol(content: &str, sym: &str) -> bool {
    for raw in content.lines() {
        let code = raw.split('#').next().unwrap_or("").trim();
        if let Some(rest) = code.strip_prefix("import ") {
            for clause in rest.split(',') {
                let c = clause.trim();
                if c == sym {
                    return true;
                }
                if let Some((lhs, rhs)) = c.split_once(" as ") {
                    if lhs.trim() == sym || rhs.trim() == sym {
                        return true;
                    }
                } else if c.split('.').next().map(str::trim) == Some(sym) {
                    return true;
                }
            }
        } else if let Some(rest) = code.strip_prefix("from ") {
            if let Some((mod_part, imp_part)) = rest.split_once(" import ") {
                if mod_part.trim().split('.').next() == Some(sym) {
                    return true;
                }
                for clause in imp_part.trim_matches(|ch| ch == '(' || ch == ')').split(',') {
                    let c = clause.trim();
                    if c == sym {
                        return true;
                    }
                    if let Some((lhs, rhs)) = c.split_once(" as ") {
                        if lhs.trim() == sym || rhs.trim() == sym {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

/// Catches deterministic Python library API pitfalls (Pillow 10+ `textsize`, missing macOS font paths,
/// `wave.readframes()` raw `bytes` passed to `np.array`, `faster_whisper` tuple return values indexed as
/// dicts, and misaligned zero-shot voice-cloning reference audio).
fn validate_python_runtime_pitfalls(file_path: &Path, content: &str) -> Result<(), String> {
    let file_stem = file_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let uses_faster_whisper = content.contains("faster_whisper") || content.contains("WhisperModel");
    let uses_mlx_audio = content.contains("mlx_audio") || content.contains("generate_audio");
    let uses_imagedraw = content.contains("ImageDraw");
    let has_makedirs = content.contains("makedirs(") || content.contains("mkdir(");
    let has_import_re = python_imports_symbol(content, "re");
    let has_import_json = python_imports_symbol(content, "json");
    let has_import_wave = python_imports_symbol(content, "wave");
    let has_import_textwrap = python_imports_symbol(content, "textwrap");
    let has_import_numpy = python_imports_symbol(content, "numpy") || python_imports_symbol(content, "np");
    let has_import_subprocess = python_imports_symbol(content, "subprocess");
    let has_import_shutil = python_imports_symbol(content, "shutil");
    let has_import_math = python_imports_symbol(content, "math");
    let has_import_sys = python_imports_symbol(content, "sys");
    let has_import_time = python_imports_symbol(content, "time");
    let has_import_glob = python_imports_symbol(content, "glob");
    let has_import_mlx_audio_pkg = python_imports_symbol(content, "mlx_audio");
    let has_from_mlx_audio = content.contains("from mlx_audio");
    let has_load_model = content.contains("load_model(");
    let splits_markdown_script =
        content.contains("script_content.split(") || content.contains(".read().split('###");
    let uses_ffmpeg_concat = content.contains("'-f', 'concat'")
        || content.contains("\"-f\", \"concat\"")
        || content.contains("-f concat");
    let defines_run_command = content.contains("def run_command");
    let has_list_coherence_var = content.contains("coherence_report = []")
        || content.contains("coherence_report=[]");
    let lines: Vec<&str> = content.lines().collect();
    let mut in_loop_indent: Option<usize> = None;
    let mut in_slides_block = false;
    for (idx, raw_line) in lines.iter().enumerate() {
        let lower_raw = raw_line.to_ascii_lowercase();
        if lower_raw.contains("# placeholder")
            || lower_raw.contains("your transcribed reference text here")
            || lower_raw.contains("clean single line spoken english narration of 130")
            || lower_raw.contains("'slide title 1'")
            || lower_raw.contains("\"slide title 1\"")
            || lower_raw.contains("'formula / bullet 1'")
            || lower_raw.contains("\"formula / bullet 1\"")
        {
            return Err(format!(
                "Python line {}: template placeholder string is not allowed — write grounded technical Kronos titles, bullets, and spoken narration sentences for all 6 to 8 slides",
                idx + 1
            ));
        }
        let code = raw_line.split('#').next().unwrap_or("").trim();
        if code.is_empty() {
            continue;
        }
        let indent = raw_line.chars().take_while(|c| c.is_whitespace()).count();
        if let Some(loop_ind) = in_loop_indent {
            if indent <= loop_ind && !code.starts_with("elif ") && !code.starts_with("else:") {
                in_loop_indent = None;
            }
        }
        if (code.starts_with("for ") || code.starts_with("while ")) && code.ends_with(':') {
            in_loop_indent = Some(indent);
        }
        if code.starts_with("SLIDES = [") || code.starts_with("SLIDES=[") {
            in_slides_block = true;
        }
        if !defines_run_command && (code.starts_with("run_command(") || code.contains(" run_command(")) {
            return Err(format!(
                "Python line {}: `run_command(...)` is a Kronos tool, not a Python function — inside `.py` scripts use `import subprocess; subprocess.run(['/opt/homebrew/bin/ffmpeg', ...], check=True)` for CLI commands and call `from mlx_audio.tts.utils import load_model; from mlx_audio.tts.generate import generate_audio` directly in Python",
                idx + 1
            ));
        }
        if !has_import_re
            && (code.contains("re.findall(")
                || code.contains("re.sub(")
                || code.contains("re.search(")
                || code.contains("re.split(")
                || code.contains("re.compile("))
        {
            return Err(format!(
                "Python line {}: `re` module used without `import re` at top of script",
                idx + 1
            ));
        }
        if !has_import_json && (code.contains("json.dump(") || code.contains("json.dumps(") || code.contains("json.loads(") || code.contains("json.load(")) {
            return Err(format!(
                "Python line {}: `json` module used without `import json` at top of script",
                idx + 1
            ));
        }
        if !has_import_wave && code.contains("wave.open(") {
            return Err(format!(
                "Python line {}: `wave` module used without `import wave` at top of script",
                idx + 1
            ));
        }
        if !has_import_textwrap && code.contains("textwrap.") {
            return Err(format!(
                "Python line {}: `textwrap` module used without `import textwrap` at top of script",
                idx + 1
            ));
        }
        if !has_import_numpy && code.contains("np.") {
            return Err(format!(
                "Python line {}: `np` used without `import numpy as np` at top of script",
                idx + 1
            ));
        }
        if !has_import_subprocess && code.contains("subprocess.") {
            return Err(format!(
                "Python line {}: `subprocess` module used without `import subprocess` at top of script",
                idx + 1
            ));
        }
        if !has_import_shutil && code.contains("shutil.") {
            return Err(format!(
                "Python line {}: `shutil` module used without `import shutil` at top of script",
                idx + 1
            ));
        }
        if !has_import_math && code.contains("math.") {
            return Err(format!(
                "Python line {}: `math` module used without `import math` at top of script",
                idx + 1
            ));
        }
        if !has_import_sys && code.contains("sys.") {
            return Err(format!(
                "Python line {}: `sys` module used without `import sys` at top of script",
                idx + 1
            ));
        }
        if !has_import_time && (code.contains("time.time(") || code.contains("time.sleep(")) {
            return Err(format!(
                "Python line {}: `time` module used without `import time` at top of script",
                idx + 1
            ));
        }
        if !has_import_glob && code.contains("glob.glob(") {
            return Err(format!(
                "Python line {}: `glob` module used without `import glob` at top of script",
                idx + 1
            ));
        }
        if in_loop_indent.is_some()
            && (code.contains("load_model(") || code.contains("WhisperModel("))
        {
            return Err(format!(
                "Python line {}: calling `load_model(...)` or `WhisperModel(...)` inside a loop reloads multi-gigabyte weights on every slide — load `wm` and `m` once at top level before the loop",
                idx + 1
            ));
        }
        if uses_mlx_audio && in_slides_block {
            if raw_line.contains('$')
                || raw_line.contains("\\min")
                || raw_line.contains("\\text{")
                || raw_line.contains("\\frac")
                || raw_line.contains("\\operatorname")
                || raw_line.contains("\\langle")
                || raw_line.contains("||")
            {
                return Err(format!(
                    "Python line {}: `SLIDES` contains raw LaTeX or symbolic operators (`$`, `\\min`, `\\text`, `\\frac`, `||`) which print raw syntax on PIL slides and cause Qwen3-TTS to mispronounce symbols — write 2 to 4 clean ASCII-math bullet strings in each slide's `bullets` list (`O(sqrt(M))`, `Q_b = (1/8) H_64 D_b`, `G(E8) = 929/12960`), and write each slide's `spoken` narration string in pure symbol-free spoken English words (140 to 280 chars)",
                    idx + 1
                ));
            }
            if let Some(last_comma) = code.rfind("], ") {
                let spoken_part = code[last_comma + 3..].trim().trim_end_matches([',', ')']);
                if spoken_part.len() > 340 {
                    return Err(format!(
                        "Python line {}: slide `spoken` narration string is {} chars (> 340 chars), producing 60 to 120s audio clips with degraded prosody — put detailed formulas into 2 to 4 slide `bullets`, and keep each slide's `spoken` string between 140 and 280 characters of concise spoken English words",
                        idx + 1,
                        spoken_part.len()
                    ));
                }
            }
            if indent == 0 && code == "]" {
                in_slides_block = false;
            }
        }
        if uses_imagedraw
            && (code.contains("'\\n'.join(textwrap.wrap(bullet")
                || code.contains("\"\\n\".join(textwrap.wrap(bullet"))
        {
            let next_line = lines
                .get(idx + 1)
                .map(|l| l.split('#').next().unwrap_or("").trim())
                .unwrap_or("");
            if next_line == "y += 40" || next_line == "y += 36" || next_line == "y += 32" {
                return Err(format!(
                    "Python line {}: drawing `'\\n'.join(textwrap.wrap(bullet, ...))` and advancing `y` by a single line height overlaps wrapped lines — loop `for wl in textwrap.wrap('- ' + bullet, width=76): draw.text((96, y), wl, font=font_body, fill='#f8fafc'); y += 44` followed by `y += 18`",
                    idx + 1
                ));
            }
        }
        if uses_imagedraw
            && (raw_line.contains("\x09imes ")
                || raw_line.contains("\\times")
                || raw_line.contains("\\frac{")
                || raw_line.contains("\\operatorname{")
                || raw_line.contains("\\mathbb{"))
        {
            return Err(format!(
                "Python line {}: PIL `ImageDraw.text` draws raw characters rather than typesetting LaTeX (`\\times` turns into a tab character `imes` and `\\frac` prints raw backslashes on slides) — keep LaTeX `$...$` in `script.md`, and write slide bullets in clean readable ASCII/Unicode math (`O(sqrt(M))`, `Q_b = (1/8) H_64 D_b`, `G(E8) = 929/12960 = 0.07168`, `l_post = l_0 + alpha * log(P(Y)/P(N))`) wrapped with `textwrap.wrap(..., width=82)`",
                idx + 1
            ));
        }
        if uses_mlx_audio
            && (code.contains("enumerate(segs)")
                || code.contains("text = seg.text")
                || code.contains("text=seg.text"))
        {
            return Err(format!(
                "Python line {}: `segs` holds Whisper segments of the 12s reference voice clip (`raw_ref_16k.wav`), not the video slides — convert `segs = list(segs)` once to compute `ref_text` and `end_sec`, then define a `SLIDES = [(title, bullets, spoken_sentence), ...]` list of 6 to 8 technical Kronos slides covering Iterations 1..8 and llama.cpp/Ollama benchmarks, and loop `for idx, (title, bullets, spoken) in enumerate(SLIDES, 1):`",
                idx + 1
            ));
        }
        if (code.contains("tw[:len(tw)//2]") || code.contains("w[:len(w)//2]"))
            && (code.contains("for tw in ") || code.contains("for w in "))
        {
            return Err(format!(
                "Python line {}: slice the word list into first-half words and second-half words (`mid = max(1, len(t_words) // 2); h1 = t_words[:mid]; h2 = t_words[mid:]`), not characters of a single word",
                idx + 1
            ));
        }
        if has_list_coherence_var && code.contains("json.dump(coherence_report,") {
            return Err(format!(
                "Python line {}: `audio/coherence_report.json` must be a JSON object `{{'verified': True, 'ref_text': ref_text, 'segments': seg_reports}}`, not a bare JSON array",
                idx + 1
            ));
        }
        if code.starts_with("def ") && code.ends_with(':') {
            for next_raw in lines.iter().skip(idx + 1) {
                let next_code = next_raw.split('#').next().unwrap_or("").trim();
                if next_code.is_empty() {
                    continue;
                }
                if next_code == "pass" || next_code == "..." {
                    return Err(format!(
                        "Python line {}: stub function `{code}` with `pass` body is not allowed — implement the full function logic",
                        idx + 1
                    ));
                }
                break;
            }
        }
        if !has_makedirs
            && (code.contains(".save('slides/")
                || code.contains(".save(\"slides/")
                || code.contains(".save(f'slides/")
                || code.contains(".save(f\"slides/"))
        {
            return Err(format!(
                "Python line {}: saving into `slides/` without `os.makedirs('slides', exist_ok=True)` raises `FileNotFoundError`",
                idx + 1
            ));
        }
        if code.contains(".textsize(") {
            return Err(format!(
                "Python line {}: Pillow 10+ removed `draw.textsize(...)`; use `bx = draw.textbbox((0, 0), text, font=font); w, h = bx[2] - bx[0], bx[3] - bx[1]`",
                idx + 1
            ));
        }
        if code.contains("SFNSDisplay") && !content.contains("except ") {
            return Err(format!(
                "Python line {}: `/System/Library/Fonts/SFNSDisplay-Regular.ttf` does not exist on macOS; load `/System/Library/Fonts/Helvetica.ttc` with fallback to `ImageFont.load_default()`",
                idx + 1
            ));
        }
        if code.contains("np.array(") && code.contains("readframes(") {
            return Err(format!(
                "Python line {}: `wave.readframes(...)` returns raw `bytes`, so `np.array(wav.readframes(...), dtype=np.int16)` raises `ValueError`; use `np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0`",
                idx + 1
            ));
        }
        if uses_faster_whisper
            && (code.contains("['segments']") || code.contains("[\"segments\"]"))
        {
            return Err(format!(
                "Python line {}: `faster_whisper` `WhisperModel.transcribe(pcm)` returns `(segments_iter, info)` tuple where each item `s` has `s.start`, `s.end`, `s.text` attributes (not a `['segments']` dict); write `segs, _ = wm.transcribe(pcm); segs = list(segs)`",
                idx + 1
            ));
        }
        if (code.contains("mlx_audio.tts.generate'")
            || code.contains("mlx_audio.tts.generate\""))
            && (code.contains("subprocess.") || content.contains("subprocess.run"))
        {
            return Err(format!(
                "Python line {}: do not invoke the `mlx_audio.tts.generate` CLI binary inside `subprocess.run` (it reloads 3.4 GB weights per slide and lacks `--text` / `--model` arguments) — load the local TTS directory once in Python via `from mlx_audio.tts.utils import load_model; from mlx_audio.tts.generate import generate_audio; m = load_model('/Users/melodyleonard/Documents/Kronos-video/tts-model')` and call `generate_audio(model=m, text=spoken, ref_audio='ref_24k.wav', ref_text=ref_text, ...)`",
                idx + 1
            ));
        }
        if !code.starts_with("from mlx_audio")
            && !code.starts_with("import mlx_audio")
            && code.contains("mlx_audio.")
            && !has_import_mlx_audio_pkg
        {
            return Err(format!(
                "Python line {}: `mlx_audio.` module attribute is used without `import mlx_audio` — either `from mlx_audio.tts.utils import load_model; from mlx_audio.tts.generate import generate_audio` and call `generate_audio(model=m, text=spoken, ...)` directly, or add `import mlx_audio`",
                idx + 1
            ));
        }
        if uses_mlx_audio
            && (code.starts_with("ref_text = text")
                || code.starts_with("ref_text=text")
                || code.starts_with("ref_text = section")
                || code.starts_with("ref_text = spoken"))
        {
            return Err(format!(
                "Python line {}: `ref_text` must be the `faster_whisper` transcript of the trimmed 4..8.5s reference voice clip `ref_24k.wav`, never the target slide text (`ref_text = text` corrupts Qwen3-TTS voice cloning)",
                idx + 1
            ));
        }
        if uses_ffmpeg_concat
            && (code.contains("f\"file '{") || code.contains("f'file \"{") || code.contains("file '"))
            && (code.contains("slide") || code.contains("audio") || code.contains(".png") || code.contains(".wav"))
            && !code.contains(".mp4")
        {
            return Err(format!(
                "Python line {}: `ffmpeg -f concat` cannot mux raw `.png` images and `.wav` audio listed in a concat text file — first mux each `(slide_XX.png, seg_XX.wav)` pair into `segments/seg_XX.mp4` via `ffmpeg -y -loop 1 -i slide_XX.png -i seg_XX.wav -c:v libx264 -tune stillimage -pix_fmt yuv420p -c:a aac -b:a 192k -shortest segments/seg_XX.mp4`, then list `file 'segments/seg_XX.mp4'` in `concat.txt`",
                idx + 1
            ));
        }
        if uses_mlx_audio && code.contains("generate_audio(") && !code.starts_with("def ") {
            let call_end = (idx + 16).min(lines.len());
            let call_window: String = lines[idx..call_end]
                .iter()
                .map(|l| l.split('#').next().unwrap_or(""))
                .collect::<Vec<_>>()
                .join(" ");
            let compact: String = call_window.chars().filter(|c| !c.is_whitespace()).collect();
            if content.contains("mlx_audio") && (!has_load_model || !compact.contains("model=")) {
                return Err(format!(
                    "Python line {}: `mlx_audio.tts.generate.generate_audio(...)` requires loading the local TTS directory once via `from mlx_audio.tts.utils import load_model; m = load_model('<tts_model_dir>')` and passing `model=m, text=spoken, ref_audio='ref_24k.wav', ref_text=ref_text`",
                    idx + 1
                ));
            }
            if content.contains("mlx_audio")
                && !(compact.contains("(text=") || compact.contains(",text="))
            {
                return Err(format!(
                    "Python line {}: `mlx_audio.tts.generate.generate_audio(...)` is missing `text=spoken` (the target narration sentence to synthesize)",
                    idx + 1
                ));
            }
            if splits_markdown_script
                && (compact.contains("(text=section")
                    || compact.contains(",text=section")
                    || compact.contains("(section,"))
            {
                return Err(format!(
                    "Python line {}: passing raw Markdown `script.md` sections (with `###` headings, LaTeX `$` formulas, backticks, or tables) directly to `generate_audio` causes incoherent second-half TTS audio — define clean single-line symbol-free spoken English strings of 130 to 210 characters per slide (speaking equations in plain words)",
                    idx + 1
                ));
            }
            if compact.contains("(text=ref_text,")
                || compact.contains(",text=ref_text,")
                || compact.contains(",text=ref_text)")
                || compact.contains("(text=ref_text)")
            {
                return Err(format!(
                    "Python line {}: `generate_audio(..., text=ref_text, ...)` synthesizes the reference transcript instead of the slide narration; pass each slide's clean single-line spoken narration string as `text=...` and the trimmed reference sentence as `ref_text=...`",
                    idx + 1
                ));
            }
            if compact.contains("ref_audio='raw_ref_24k.wav'")
                || compact.contains("ref_audio=\"raw_ref_24k.wav\"")
            {
                return Err(format!(
                    "Python line {}: passing untrimmed `raw_ref_24k.wav` as `ref_audio` causes trailing words beyond `ref_text` to bleed into synthesized speech; slice `raw_ref_24k.wav` strictly to `int(end_sec * 24000)` samples into `ref_24k.wav` so `ref_audio` matches `ref_text` word-for-word",
                    idx + 1
                ));
            }
        }
    }
    if file_stem.contains("video")
        && (content.contains("slides/") || content.contains("'slides'") || content.contains("\"slides\""))
        && (!uses_mlx_audio || !content.contains(".mp4"))
    {
        return Err(
            "Python video-pipeline check: `build_video.py` currently only renders slides and omits `mlx_audio` voice-cloned audio synthesis (`from mlx_audio.tts.utils import load_model; from mlx_audio.tts.generate import generate_audio`), `faster_whisper` `audio/coherence_report.json` verification, or `ffmpeg` `.mp4` muxing — include the complete end-to-end slide + voice-cloning + ASR coherence check + MP4 muxing pipeline in `build_video.py`"
                .to_string(),
        );
    }
    if uses_mlx_audio && (has_import_mlx_audio_pkg || has_from_mlx_audio || content.contains("mlx_audio")) {
        if content.contains("ref_24k.wav")
            && !content.contains("'wb'")
            && !content.contains("\"wb\"")
            && !content.contains("ffmpeg")
        {
            return Err(
                "Python voice-cloning check: `'ref_24k.wav'` is referenced without extracting and trimming it from the source voice `.m4a` via `ffmpeg` + `wave.open('ref_24k.wav', 'wb')`".to_string(),
            );
        }
        if !content.contains("coherence_report.json") {
            return Err(
                "Python voice-cloning check: `mlx_audio` synthesis script must verify every generated `audio/seg_XX.wav` with `faster_whisper` (`word_recall`, `first_half_recall`, `second_half_recall`, retrying once at `temperature=0.4` if below threshold) and write `audio/coherence_report.json`".to_string(),
            );
        }
        if uses_faster_whisper {
            if content.contains("wm.transcribe(audio_path")
                || content.contains("wm.transcribe(f'")
                || content.contains("wm.transcribe(f\"")
                || content.contains("wm.transcribe('audio/")
                || content.contains("wm.transcribe(\"audio/")
            {
                return Err(
                    "Python Whisper check: `wm.transcribe(audio_path)` fails on PyAV `metadata_errors` in this environment — load `int16 -> float32 / 32768.0` via `wave` + `np.frombuffer`, resample 24 kHz TTS WAVs to 16 kHz via `p16 = np.interp(np.linspace(0, len(p24) - 1, len(p24) * 2 // 3), np.arange(len(p24)), p24).astype(np.float32)`, and pass `p16` to `wm.transcribe(p16)`".to_string(),
                );
            }
            if (content.contains("audio/seg_") || content.contains("file_prefix"))
                && content.contains("coherence_report.json")
                && !content.contains("np.interp(")
                && !content.contains("resample_poly")
                && !content.contains("16000 / 24000")
                && !content.contains("* 2 // 3")
            {
                return Err(
                    "Python Whisper sample-rate check: `mlx_audio` saves 24 kHz WAVs (`audio/seg_XX.wav`) while `faster_whisper` expects 16 kHz float32 numpy PCM — resample `p24` from 24 kHz to 16 kHz via `p16 = np.interp(np.linspace(0, len(p24) - 1, len(p24) * 2 // 3), np.arange(len(p24)), p24).astype(np.float32)` before calling `wm.transcribe(p16)`".to_string(),
                );
            }
        }
    }
    Ok(())
}

/// Catches CommonJS (`require(...)`, `__dirname`, `__filename`) inside ES modules (`.mjs` or files
/// using top-level `import` / `export`) and unquoted macOS application paths inside `exec(...)`.
fn validate_esm_commonjs_mix(ext: &str, content: &str) -> Result<(), String> {
    let uses_esm_syntax = ext == "mjs"
        || content.lines().any(|raw| {
            let l = raw.trim_start();
            l.starts_with("import ") || l.starts_with("export ")
        });
    if !uses_esm_syntax {
        return Ok(());
    }
    let has_create_require = content.contains("createRequire");
    let has_file_url_dirname = content.contains("fileURLToPath") || content.contains("const __dirname");
    for (idx, raw_line) in content.lines().enumerate() {
        let code = raw_line.split("//").next().unwrap_or("").trim();
        if code.is_empty() || code.starts_with('*') || code.starts_with("/*") {
            continue;
        }
        if !has_create_require && (code.contains("= require(") || code.starts_with("require(")) {
            return Err(format!(
                "ES module line {}: CommonJS `require()` is not defined in ES modules — use top-level `import fs from 'node:fs'`, `import http from 'node:http'`, and `import {{ execFileSync }} from 'node:child_process'`",
                idx + 1
            ));
        }
        if !has_file_url_dirname && (code.contains("__dirname") || code.contains("__filename")) {
            return Err(format!(
                "ES module line {}: `__dirname` / `__filename` is not defined in ES modules — use `process.cwd()` or `import.meta.url`",
                idx + 1
            ));
        }
        if (code.contains("exec(") || code.contains("execSync("))
            && code.contains("/Applications/Google Chrome.app/")
            && !code.contains("\"/Applications/Google Chrome.app/")
            && !code.contains("'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [")
        {
            return Err(format!(
                "ES module line {}: `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` contains spaces and must be double-quoted inside shell `exec(...)` or invoked via `execFileSync('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [...args])`",
                idx + 1
            ));
        }
    }
    Ok(())
}

/// Validates all source files in `workdir`, returning `Ok(())` or a combined diagnostic report.
pub fn validate_workspace(workdir: &Path) -> Result<(), String> {
    let entries = list_workspace_files(workdir);
    let mut errors = Vec::new();
    for e in entries {
        if !e.validation_ok {
            errors.push(format!("{}: {}", e.path, e.validation_msg));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn line_col_at(s: &str, byte_idx: usize, base_line: usize) -> (usize, usize) {
    let clamped = byte_idx.min(s.len());
    let mut line = base_line;
    let mut line_start = 0usize;
    for (i, b) in s.bytes().enumerate().take(clamped) {
        if b == b'\n' {
            line += 1;
            line_start = i + 1;
        }
    }
    let col = s[line_start..clamped].chars().count() + 1;
    (line, col)
}

fn matching_open(close: u8) -> u8 {
    match close {
        b')' => b'(',
        b']' => b'[',
        b'}' => b'{',
        _ => b'?',
    }
}

fn can_start_regex(prev_sig: Option<u8>, prev_word: &str) -> bool {
    if matches!(
        prev_word,
        "return" | "throw" | "case" | "delete" | "void" | "typeof" | "instanceof" | "in" | "of" | "new" | "yield" | "await"
    ) {
        return true;
    }
    match prev_sig {
        None => true,
        Some(b) => matches!(
            b,
            b'(' | b'[' | b'{' | b',' | b';' | b':' | b'=' | b'?' | b'!' | b'&' | b'|' | b'^' | b'~' | b'+' | b'-' | b'*' | b'%' | b'<' | b'>'
        ),
    }
}

/// Fast lexical scanner checking balanced delimiters, strings, template expressions, regexes, and comments.
fn validate_js_like(src: &str, base_line: usize) -> Result<(), String> {
    #[derive(Clone, Copy)]
    enum Frame {
        Delim { ch: u8, pos: usize },
        TemplateExpr { brace_pos: usize, tmpl_pos: usize },
    }

    enum TemplateStop {
        Closed,
        EnteredExpr,
    }

    fn scan_template_body(
        src: &str,
        i: &mut usize,
        tmpl_pos: usize,
        stack: &mut Vec<Frame>,
        base_line: usize,
    ) -> Result<TemplateStop, String> {
        let bytes = src.as_bytes();
        let len = bytes.len();
        while *i < len {
            let c = bytes[*i];
            if c == b'\\' {
                *i += 2;
                continue;
            }
            if c == b'`' {
                *i += 1;
                return Ok(TemplateStop::Closed);
            }
            if c == b'$' && *i + 1 < len && bytes[*i + 1] == b'{' {
                let brace_pos = *i;
                *i += 2;
                stack.push(Frame::TemplateExpr { brace_pos, tmpl_pos });
                return Ok(TemplateStop::EnteredExpr);
            }
            *i += 1;
        }
        let (l, c) = line_col_at(src, tmpl_pos, base_line);
        Err(format!("line {l} col {c}: unterminated template literal `` ` ``"))
    }

    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    let mut stack: Vec<Frame> = Vec::new();
    let mut prev_sig: Option<u8> = None;
    let mut prev_word_range: Option<(usize, usize)> = None;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            i += 2;
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            let mut closed = false;
            while i + 1 < len {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                let (l, c) = line_col_at(src, start, base_line);
                return Err(format!("line {l} col {c}: unterminated block comment `/*`"));
            }
            continue;
        }
        if b == b'\'' || b == b'"' {
            let quote_ch = b;
            let start = i;
            i += 1;
            let mut closed = false;
            while i < len {
                let c = bytes[i];
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == b'\n' || c == b'\r' {
                    break;
                }
                if c == quote_ch {
                    i += 1;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                let (l, c) = line_col_at(src, start, base_line);
                return Err(format!("line {l} col {c}: unterminated string literal"));
            }
            prev_sig = Some(quote_ch);
            prev_word_range = None;
            continue;
        }
        if b == b'`' {
            let tmpl_start = i;
            i += 1;
            match scan_template_body(src, &mut i, tmpl_start, &mut stack, base_line)? {
                TemplateStop::Closed => {
                    prev_sig = Some(b'`');
                    prev_word_range = None;
                }
                TemplateStop::EnteredExpr => {
                    prev_sig = Some(b'{');
                    prev_word_range = None;
                }
            }
            continue;
        }
        if b == b'/' {
            let prev_word = prev_word_range.map_or("", |(s, e)| &src[s..e]);
            if can_start_regex(prev_sig, prev_word) {
                let start = i;
                i += 1;
                let mut in_class = false;
                let mut closed = false;
                while i < len {
                    let c = bytes[i];
                    if c == b'\\' {
                        i += 2;
                        continue;
                    }
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    if c == b'[' {
                        in_class = true;
                    } else if c == b']' {
                        in_class = false;
                    } else if c == b'/' && !in_class {
                        i += 1;
                        while i < len && bytes[i].is_ascii_alphabetic() {
                            i += 1;
                        }
                        closed = true;
                        break;
                    }
                    i += 1;
                }
                if !closed {
                    let (l, c) = line_col_at(src, start, base_line);
                    return Err(format!("line {l} col {c}: unterminated regex literal"));
                }
                prev_sig = Some(b'/');
                prev_word_range = None;
                continue;
            }
        }
        if b == b'(' || b == b'[' || b == b'{' {
            stack.push(Frame::Delim { ch: b, pos: i });
            prev_sig = Some(b);
            prev_word_range = None;
            i += 1;
            continue;
        }
        if b == b')' || b == b']' || b == b'}' {
            let Some(top) = stack.pop() else {
                let (l, c) = line_col_at(src, i, base_line);
                return Err(format!("line {l} col {c}: unexpected closing `{}`", b as char));
            };
            match top {
                Frame::Delim { ch, pos } => {
                    if ch != matching_open(b) {
                        let (l, c) = line_col_at(src, i, base_line);
                        let (ol, oc) = line_col_at(src, pos, base_line);
                        return Err(format!(
                            "line {l} col {c}: mismatched `{}` (opened `{}` at line {ol} col {oc})",
                            b as char, ch as char
                        ));
                    }
                    prev_sig = Some(b);
                    prev_word_range = None;
                    i += 1;
                }
                Frame::TemplateExpr { brace_pos, tmpl_pos } => {
                    if b != b'}' {
                        let (l, c) = line_col_at(src, i, base_line);
                        let (ol, oc) = line_col_at(src, brace_pos, base_line);
                        return Err(format!(
                            "line {l} col {c}: mismatched `{}` inside `${{...}}` opened at line {ol} col {oc}",
                            b as char
                        ));
                    }
                    i += 1;
                    match scan_template_body(src, &mut i, tmpl_pos, &mut stack, base_line)? {
                        TemplateStop::Closed => {
                            prev_sig = Some(b'`');
                            prev_word_range = None;
                        }
                        TemplateStop::EnteredExpr => {
                            prev_sig = Some(b'{');
                            prev_word_range = None;
                        }
                    }
                }
            }
            continue;
        }
        if b.is_ascii_alphabetic() || b == b'_' || b == b'$' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'$') {
                i += 1;
            }
            prev_sig = Some(b'a');
            prev_word_range = Some((start, i));
            continue;
        }
        prev_sig = Some(b);
        prev_word_range = None;
        i += 1;
    }

    if let Some(unclosed) = stack.last() {
        match *unclosed {
            Frame::Delim { ch, pos } => {
                let (l, c) = line_col_at(src, pos, base_line);
                return Err(format!("line {l} col {c}: unclosed `{}`", ch as char));
            }
            Frame::TemplateExpr { brace_pos, .. } => {
                let (l, c) = line_col_at(src, brace_pos, base_line);
                return Err(format!("line {l} col {c}: unclosed template expression `${{`"));
            }
        }
    }
    Ok(())
}

fn validate_css(src: &str, base_line: usize) -> Result<(), String> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    let mut stack: Vec<(u8, usize)> = Vec::new();
    while i < len {
        let b = bytes[i];
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            let mut closed = false;
            while i + 1 < len {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                let (l, c) = line_col_at(src, start, base_line);
                return Err(format!("CSS line {l} col {c}: unterminated comment `/*`"));
            }
            continue;
        }
        if b == b'\'' || b == b'"' {
            let q = b;
            let start = i;
            i += 1;
            let mut closed = false;
            while i < len {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'\n' {
                    break;
                }
                if bytes[i] == q {
                    i += 1;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                let (l, c) = line_col_at(src, start, base_line);
                return Err(format!("CSS line {l} col {c}: unterminated string"));
            }
            continue;
        }
        if matches!(b, b'{' | b'(' | b'[') {
            stack.push((b, i));
        } else if matches!(b, b'}' | b')' | b']') {
            let Some((open, pos)) = stack.pop() else {
                let (l, c) = line_col_at(src, i, base_line);
                return Err(format!("CSS line {l} col {c}: unexpected `{}`", b as char));
            };
            if open != matching_open(b) {
                let (l, c) = line_col_at(src, i, base_line);
                let (ol, oc) = line_col_at(src, pos, base_line);
                return Err(format!(
                    "CSS line {l} col {c}: mismatched `{}` (opened at line {ol} col {oc})",
                    b as char
                ));
            }
        }
        i += 1;
    }
    if let Some(&(open, pos)) = stack.last() {
        let (l, c) = line_col_at(src, pos, base_line);
        return Err(format!("CSS line {l} col {c}: unclosed `{}`", open as char));
    }
    Ok(())
}

fn validate_python_lexical(src: &str) -> Result<(), String> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    let mut stack: Vec<(u8, usize)> = Vec::new();
    while i < len {
        let b = bytes[i];
        if b == b'#' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'\'' || b == b'"' {
            let q = b;
            let start = i;
            let triple = i + 2 < len && bytes[i + 1] == q && bytes[i + 2] == q;
            i += if triple { 3 } else { 1 };
            let mut closed = false;
            while i < len {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if triple {
                    if i + 2 < len && bytes[i] == q && bytes[i + 1] == q && bytes[i + 2] == q {
                        i += 3;
                        closed = true;
                        break;
                    }
                } else {
                    if bytes[i] == b'\n' {
                        break;
                    }
                    if bytes[i] == q {
                        i += 1;
                        closed = true;
                        break;
                    }
                }
                i += 1;
            }
            if !closed {
                let (l, c) = line_col_at(src, start, 1);
                return Err(format!("Python line {l} col {c}: unterminated string literal"));
            }
            continue;
        }
        if matches!(b, b'(' | b'[' | b'{') {
            stack.push((b, i));
        } else if matches!(b, b')' | b']' | b'}') {
            let Some((open, pos)) = stack.pop() else {
                let (l, c) = line_col_at(src, i, 1);
                return Err(format!("Python line {l} col {c}: unexpected `{}`", b as char));
            };
            if open != matching_open(b) {
                let (l, c) = line_col_at(src, i, 1);
                let (ol, oc) = line_col_at(src, pos, 1);
                return Err(format!(
                    "Python line {l} col {c}: mismatched `{}` (opened at line {ol} col {oc})",
                    b as char
                ));
            }
        }
        i += 1;
    }
    if let Some(&(open, pos)) = stack.last() {
        let (l, c) = line_col_at(src, pos, 1);
        return Err(format!("Python line {l} col {c}: unclosed `{}`", open as char));
    }
    Ok(())
}

fn is_void_html_tag(tag: &str) -> bool {
    matches!(
        tag,
        "area" | "base" | "br" | "col" | "embed" | "hr" | "img" | "input" | "link" | "meta" | "param" | "source" | "track" | "wbr"
    )
}

fn find_case_insensitive_end_tag(src: &str, from: usize, tag: &str) -> Option<usize> {
    let needle = format!("</{tag}");
    let bytes = src.as_bytes();
    let nbytes = needle.as_bytes();
    if from + nbytes.len() >= bytes.len() {
        return None;
    }
    let mut pos = from;
    while pos + nbytes.len() < bytes.len() {
        if bytes[pos] == b'<'
            && bytes[pos + 1] == b'/'
            && bytes[pos..pos + nbytes.len()].eq_ignore_ascii_case(nbytes)
        {
            let next = bytes[pos + nbytes.len()];
            if next == b'>' || next.is_ascii_whitespace() {
                return Some(pos);
            }
        }
        pos += 1;
    }
    None
}

fn extract_html_attr(attrs: &str, attr_name: &str) -> Option<String> {
    let bytes = attrs.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    while i < len {
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let name_start = i;
        while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'-' || bytes[i] == b'_') {
            i += 1;
        }
        let name = &attrs[name_start..i];
        while i < len && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i < len && bytes[i] == b'=' {
            i += 1;
            while i < len && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < len && (bytes[i] == b'"' || bytes[i] == b'\'') {
                let q = bytes[i];
                i += 1;
                let v_start = i;
                while i < len && bytes[i] != q {
                    i += 1;
                }
                let val = &attrs[v_start..i];
                if i < len {
                    i += 1;
                }
                if name.eq_ignore_ascii_case(attr_name) {
                    return Some(val.to_string());
                }
            } else {
                let v_start = i;
                while i < len && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' {
                    i += 1;
                }
                let val = &attrs[v_start..i];
                if name.eq_ignore_ascii_case(attr_name) {
                    return Some(val.to_string());
                }
            }
        } else if name_start == i {
            i += 1;
        }
    }
    None
}

fn is_external_url(u: &str) -> bool {
    let t = u.trim();
    t.is_empty()
        || t.starts_with('#')
        || t.starts_with("http://")
        || t.starts_with("https://")
        || t.starts_with("//")
        || t.starts_with("data:")
        || t.starts_with("mailto:")
        || t.starts_with("javascript:")
}

fn validate_html(workdir: &Path, html_file: &Path, src: &str, strict_assets: bool) -> Result<String, String> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    let mut tag_stack: Vec<(String, usize)> = Vec::new();
    let mut missing_assets: Vec<String> = Vec::new();
    let html_dir = html_file.parent().unwrap_or(workdir);

    while i < len {
        if bytes[i] != b'<' {
            i += 1;
            continue;
        }
        if src[i..].starts_with("<!--") {
            let start = i;
            let Some(rel_end) = src[i + 4..].find("-->") else {
                let (l, c) = line_col_at(src, start, 1);
                return Err(format!("HTML line {l} col {c}: unterminated comment `<!--`"));
            };
            i += 4 + rel_end + 3;
            continue;
        }
        if i + 1 < len && (bytes[i + 1] == b'!' || bytes[i + 1] == b'?') {
            let Some(rel_gt) = src[i + 2..].find('>') else {
                let (l, c) = line_col_at(src, i, 1);
                return Err(format!("HTML line {l} col {c}: unclosed declaration tag"));
            };
            i += 2 + rel_gt + 1;
            continue;
        }
        if i + 1 >= len || !(bytes[i + 1].is_ascii_alphabetic() || bytes[i + 1] == b'/') {
            i += 1;
            continue;
        }

        let tag_pos = i;
        let is_closing = bytes[i + 1] == b'/';
        let mut p = if is_closing { i + 2 } else { i + 1 };
        let name_start = p;
        while p < len && (bytes[p].is_ascii_alphanumeric() || bytes[p] == b'-') {
            p += 1;
        }
        if p == name_start {
            i += 1;
            continue;
        }
        let tag_name = src[name_start..p].to_ascii_lowercase();

        // Scan until `>` respecting quoted attribute values.
        let attr_start = p;
        let mut in_quote: Option<u8> = None;
        while p < len {
            let ch = bytes[p];
            if let Some(q) = in_quote {
                if ch == q {
                    in_quote = None;
                }
            } else if ch == b'"' || ch == b'\'' {
                in_quote = Some(ch);
            } else if ch == b'>' {
                break;
            }
            p += 1;
        }
        if p >= len {
            let (l, c) = line_col_at(src, tag_pos, 1);
            return Err(format!("HTML line {l} col {c}: unclosed `<{tag_name}` tag or attribute quote"));
        }
        let raw_attrs = &src[attr_start..p];
        let self_closing = raw_attrs.trim_end().ends_with('/');
        i = p + 1;

        if is_closing {
            if is_void_html_tag(&tag_name) {
                continue;
            }
            if let Some(pos) = tag_stack.iter().rposition(|(open, _)| open == &tag_name) {
                // Allow optional HTML closures like <p> or <li> inside the matched container.
                for (unclosed, upos) in tag_stack.drain(pos + 1..) {
                    if !matches!(unclosed.as_str(), "p" | "li" | "dt" | "dd" | "option" | "tr" | "td" | "th") {
                        let (l, c) = line_col_at(src, upos, 1);
                        return Err(format!("HTML line {l} col {c}: unclosed `<{unclosed}>` before `</{tag_name}>`"));
                    }
                }
                tag_stack.pop();
            } else {
                let (l, c) = line_col_at(src, tag_pos, 1);
                return Err(format!("HTML line {l} col {c}: unexpected closing `</{tag_name}>`"));
            }
            continue;
        }

        // Check referenced local JS / CSS assets.
        if tag_name == "script" {
            if let Some(src_attr) = extract_html_attr(raw_attrs, "src") {
                check_local_asset(workdir, html_dir, &src_attr, &mut missing_assets);
            }
        } else if tag_name == "link" {
            let rel_val = extract_html_attr(raw_attrs, "rel").unwrap_or_default().to_ascii_lowercase();
            if rel_val.contains("stylesheet") {
                if let Some(href_attr) = extract_html_attr(raw_attrs, "href") {
                    check_local_asset(workdir, html_dir, &href_attr, &mut missing_assets);
                }
            }
        }

        if self_closing || is_void_html_tag(&tag_name) {
            continue;
        }

        if tag_name == "script" || tag_name == "style" {
            let body_start = i;
            let Some(close_start) = find_case_insensitive_end_tag(src, body_start, &tag_name) else {
                let (l, c) = line_col_at(src, tag_pos, 1);
                return Err(format!("HTML line {l} col {c}: unclosed `<{tag_name}>` block"));
            };
            let body = &src[body_start..close_start];
            let (base_line, _) = line_col_at(src, body_start, 1);
            if tag_name == "script" {
                let script_type = extract_html_attr(raw_attrs, "type").unwrap_or_default().to_ascii_lowercase();
                if script_type == "application/json" || script_type == "importmap" {
                    if !body.trim().is_empty() {
                        Json::parse(body).map_err(|e| format!("inline JSON `<script>` at line {base_line}: {e}"))?;
                    }
                } else if script_type.is_empty()
                    || script_type == "module"
                    || script_type.contains("javascript")
                {
                    validate_js_like(body, base_line)
                        .map_err(|e| format!("inline `<script>` {e}"))?;
                }
            } else {
                validate_css(body, base_line).map_err(|e| format!("inline `<style>` {e}"))?;
            }
            let Some(rel_gt) = src[close_start..].find('>') else {
                let (l, c) = line_col_at(src, close_start, 1);
                return Err(format!("HTML line {l} col {c}: unclosed `</{tag_name}`"));
            };
            i = close_start + rel_gt + 1;
            continue;
        }

        tag_stack.push((tag_name, tag_pos));
    }

    for (unclosed, upos) in tag_stack {
        if !matches!(unclosed.as_str(), "html" | "head" | "body" | "p" | "li" | "dt" | "dd" | "option") {
            let (l, c) = line_col_at(src, upos, 1);
            return Err(format!("HTML line {l} col {c}: unclosed `<{unclosed}>` tag"));
        }
    }

    if !missing_assets.is_empty() {
        let joined = missing_assets.join(", ");
        if strict_assets {
            return Err(format!("missing local asset(s) referenced in HTML: {joined}"));
        }
        return Ok(format!("pending referenced asset(s): {joined}"));
    }
    Ok(String::new())
}

fn check_local_asset(workdir: &Path, html_dir: &Path, raw_ref: &str, missing: &mut Vec<String>) {
    if is_external_url(raw_ref) {
        return;
    }
    let clean_ref = raw_ref.split(['?', '#']).next().unwrap_or("").trim();
    if clean_ref.is_empty() {
        return;
    }
    let target = if let Some(stripped) = clean_ref.strip_prefix('/') {
        resolve_jailed_path(workdir, stripped)
    } else {
        let base_rel = html_dir.strip_prefix(workdir).unwrap_or_else(|_| Path::new(""));
        let joined = base_rel.join(clean_ref);
        resolve_jailed_path(workdir, &joined.to_string_lossy())
    };
    match target {
        Ok(p) => {
            if !p.is_file() && !missing.iter().any(|m| m == clean_ref) {
                missing.push(clean_ref.to_string());
            }
        }
        Err(_) => {
            if !missing.iter().any(|m| m == clean_ref) {
                missing.push(format!("{clean_ref} (outside workspace)"));
            }
        }
    }
}

pub fn execute_tool(
    workdir: &Path,
    call: &ToolCall,
    cmd_timeout_secs: u64,
    step_tag: usize,
) -> ToolExecOutcome {
    let args = match Json::parse(&call.arguments) {
        Ok(v) => v,
        Err(e) => {
            return ToolExecOutcome {
                ok: false,
                validation_ok: false,
                validation_msg: "invalid tool JSON".to_string(),
                output: format!("error: invalid JSON arguments `{}`: {e}", call.arguments),
                diff: None,
                new_workdir: None,
            };
        }
    };

    match call.name.as_str() {
        "switch_workspace" => exec_switch_workspace(workdir, &args),
        "write_file" => exec_write_file(workdir, &args),
        "edit_file" => exec_edit_file(workdir, &args),
        "read_file" => exec_read_file(workdir, &args),
        "list_dir" => exec_list_dir(workdir, &args),
        "search_files" => exec_search_files(workdir, &args),
        "run_command" => {
            let Some(cmd) = args.get("cmd").and_then(Json::as_str) else {
                return ToolExecOutcome {
                    ok: false,
                    validation_ok: true,
                    validation_msg: "ok".to_string(),
                    output: "error: `run_command` requires string argument `cmd`".to_string(),
                    diff: None,
                    new_workdir: None,
                };
            };
            let out = run_shell_command_with_timeout(workdir, cmd, cmd_timeout_secs, step_tag);
            let ok = out.starts_with("exit_code=0");
            ToolExecOutcome {
                ok,
                validation_ok: true,
                validation_msg: "ok".to_string(),
                output: out,
                diff: None,
                new_workdir: None,
            }
        }
        "use_computer" => match crate::device::global_device_manager().execute_computer_action(&args) {
            Ok(msg) => ToolExecOutcome {
                ok: true,
                validation_ok: true,
                validation_msg: "ok".to_string(),
                output: msg,
                diff: None,
                new_workdir: None,
            },
            Err(err) => ToolExecOutcome {
                ok: false,
                validation_ok: true,
                validation_msg: "ok".to_string(),
                output: format!("error: {err}"),
                diff: None,
                new_workdir: None,
            },
        },
        other => ToolExecOutcome {
            ok: false,
            validation_ok: true,
            validation_msg: "ok".to_string(),
            output: format!("error: unknown tool `{other}`"),
            diff: None,
            new_workdir: None,
        },
    }
}

fn normalize_lexical_path(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                let _ = out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn is_umbrella_parent_directory(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    const PROJECT_MARKERS: &[&str] = &[
        ".git",
        "Cargo.toml",
        "package.json",
        "pyproject.toml",
        "go.mod",
        "mix.exs",
        "index.html",
        "Makefile",
    ];
    if PROJECT_MARKERS.iter().any(|m| dir.join(m).exists()) {
        return false;
    }
    let Ok(rd) = fs::read_dir(dir) else {
        return false;
    };
    let mut subdirs = 0usize;
    let mut code_files = 0usize;
    for entry in rd.flatten() {
        let name = entry.file_name();
        let s = name.to_string_lossy();
        if s.starts_with('.') {
            continue;
        }
        if let Ok(ft) = entry.file_type() {
            if ft.is_dir() {
                subdirs += 1;
            } else if ft.is_file() {
                let ext = entry
                    .path()
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                if matches!(
                    ext.as_str(),
                    "html" | "css" | "js" | "ts" | "rs" | "py" | "go" | "c" | "cpp" | "swift"
                ) {
                    code_files += 1;
                }
            }
        }
    }
    subdirs >= 2 && code_files == 0
}

pub fn resolve_safe_project_workdir(
    cur_workdir: &Path,
    raw_path: &str,
    subfolder: Option<&str>,
) -> Result<(PathBuf, bool), String> {
    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return Err("error: workspace `path` cannot be empty".to_string());
    }

    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut candidate = if trimmed == "~" {
        return Err(
            "error: cannot switch workspace directly to home directory; specify a project subfolder"
                .to_string(),
        );
    } else if let Some(rest) = trimmed.strip_prefix("~/") {
        let Some(ref h) = home else {
            return Err("error: HOME is not set; provide an absolute path".to_string());
        };
        h.join(rest)
    } else {
        let p = PathBuf::from(trimmed);
        if p.is_absolute() {
            p
        } else if let Some(ref h) = home {
            if trimmed.starts_with("Documents/")
                || trimmed.starts_with("Desktop/")
                || trimmed.starts_with("Downloads/")
                || trimmed == "Documents"
                || trimmed == "Desktop"
                || trimmed == "Downloads"
            {
                h.join(trimmed)
            } else {
                cur_workdir.join(p)
            }
        } else {
            cur_workdir.join(p)
        }
    };

    if let Some(sub) = subfolder.map(str::trim).filter(|s| !s.is_empty()) {
        if sub.contains("..") || sub.starts_with('/') {
            return Err("error: `subfolder` must be a clean child folder name without `..`".to_string());
        }
        candidate = candidate.join(sub.trim_matches('/'));
    }

    let norm = normalize_lexical_path(&candidate);
    let tmp_sys = normalize_lexical_path(&std::env::temp_dir());
    let in_tmp = norm.starts_with("/tmp/")
        || norm.starts_with("/private/tmp/")
        || (norm.starts_with(&tmp_sys) && norm != tmp_sys);

    let in_safe_home = if let Some(ref h) = home {
        let norm_home = normalize_lexical_path(h);
        let canon_home = h.canonicalize().unwrap_or_else(|_| norm_home.clone());
        let rel_opt = norm
            .strip_prefix(&norm_home)
            .ok()
            .or_else(|| norm.strip_prefix(&canon_home).ok());
        if let Some(rel) = rel_opt {
            let comps: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            if comps.is_empty() {
                return Err(
                    "error: cannot switch workspace directly to home directory; pass a child project directory"
                        .to_string(),
                );
            }
            let first = comps[0].as_str();
            if first.starts_with('.') || first.eq_ignore_ascii_case("Library") {
                return Err(format!(
                    "error: switching workspace into protected system/hidden folder `~/{first}` is not permitted"
                ));
            }
            if comps.len() == 1
                && matches!(
                    first,
                    "Documents"
                        | "Downloads"
                        | "Desktop"
                        | "Movies"
                        | "Music"
                        | "Pictures"
                        | "Applications"
                        | "Public"
                )
            {
                return Err(format!(
                    "error: `{}` is a top-level personal folder; pass a new or specific child project folder such as `{}/<project-folder>`",
                    norm.display(),
                    norm.display()
                ));
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    if !in_tmp && !in_safe_home {
        return Err(format!(
            "error: workspace path `{}` must reside inside a project folder under the user home directory or /tmp",
            norm.display()
        ));
    }

    if is_umbrella_parent_directory(&norm) {
        return Err(format!(
            "error: `{}` is an existing parent directory containing multiple folders; pass a new child project folder path such as `{}/<new-folder-name>` (or set `subfolder`)",
            norm.display(),
            norm.display()
        ));
    }

    let existed = norm.exists();
    if let Err(e) = fs::create_dir_all(&norm) {
        return Err(format!(
            "error creating workspace directory `{}`: {e}",
            norm.display()
        ));
    }
    let canon = norm.canonicalize().unwrap_or(norm);
    Ok((canon, existed))
}

fn exec_switch_workspace(cur_workdir: &Path, args: &Json) -> ToolExecOutcome {
    let Some(raw_path) = args.get("path").and_then(Json::as_str) else {
        return err_outcome("error: `switch_workspace` requires string argument `path`");
    };
    let sub = args.get("subfolder").and_then(Json::as_str);
    let (canon, existed) = match resolve_safe_project_workdir(cur_workdir, raw_path, sub) {
        Ok(v) => v,
        Err(e) => return err_outcome(&e),
    };
    let file_count = list_workspace_files(&canon).len();
    let action_verb = if existed {
        "switched active workspace to"
    } else {
        "created and switched active workspace to"
    };
    ToolExecOutcome {
        ok: true,
        validation_ok: true,
        validation_msg: "ok".to_string(),
        output: format!(
            "ok: {action_verb} `{}` ({file_count} existing files); subsequent write_file and edit_file calls now write inside this folder",
            canon.display()
        ),
        diff: None,
        new_workdir: Some(canon),
    }
}

pub fn compute_unified_diff(path: &str, before: &str, after: &str) -> (usize, usize, String) {
    if before == after {
        return (0, 0, String::new());
    }
    let old_lines: Vec<&str> = before.lines().collect();
    let new_lines: Vec<&str> = after.lines().collect();
    let mut pre = 0usize;
    while pre < old_lines.len() && pre < new_lines.len() && old_lines[pre] == new_lines[pre] {
        pre += 1;
    }
    let mut suf = 0usize;
    while pre + suf < old_lines.len()
        && pre + suf < new_lines.len()
        && old_lines[old_lines.len() - 1 - suf] == new_lines[new_lines.len() - 1 - suf]
    {
        suf += 1;
    }
    let old_mid = &old_lines[pre..old_lines.len() - suf];
    let new_mid = &new_lines[pre..new_lines.len() - suf];

    let mut ops: Vec<(char, &str)> = Vec::new();
    let mut added = 0usize;
    let mut removed = 0usize;

    if old_mid.len() <= 320 && new_mid.len() <= 320 && old_mid.len() * new_mid.len() <= 64_000 {
        let m = old_mid.len();
        let n = new_mid.len();
        let stride = n + 1;
        let mut dp = vec![0u16; (m + 1) * stride];
        for i in (0..m).rev() {
            for j in (0..n).rev() {
                dp[i * stride + j] = if old_mid[i] == new_mid[j] {
                    dp[(i + 1) * stride + j + 1] + 1
                } else {
                    dp[(i + 1) * stride + j].max(dp[i * stride + j + 1])
                };
            }
        }
        let (mut i, mut j) = (0usize, 0usize);
        while i < m && j < n {
            if old_mid[i] == new_mid[j] {
                ops.push((' ', old_mid[i]));
                i += 1;
                j += 1;
            } else if dp[(i + 1) * stride + j] >= dp[i * stride + j + 1] {
                ops.push(('-', old_mid[i]));
                removed += 1;
                i += 1;
            } else {
                ops.push(('+', new_mid[j]));
                added += 1;
                j += 1;
            }
        }
        while i < m {
            ops.push(('-', old_mid[i]));
            removed += 1;
            i += 1;
        }
        while j < n {
            ops.push(('+', new_mid[j]));
            added += 1;
            j += 1;
        }
    } else {
        for &l in old_mid {
            ops.push(('-', l));
            removed += 1;
        }
        for &l in new_mid {
            ops.push(('+', l));
            added += 1;
        }
    }

    let ctx_before_start = pre.saturating_sub(2);
    let ctx_after_end = (new_lines.len() - suf + 2).min(new_lines.len());
    let old_hunk_start = if old_lines.is_empty() { 0 } else { ctx_before_start + 1 };
    let new_hunk_start = if new_lines.is_empty() { 0 } else { ctx_before_start + 1 };
    let pre_ctx_count = pre - ctx_before_start;
    let post_ctx_count = ctx_after_end.saturating_sub(new_lines.len() - suf);
    let old_hunk_len = pre_ctx_count + old_mid.len() + post_ctx_count;
    let new_hunk_len = pre_ctx_count + new_mid.len() + post_ctx_count;

    let mut out = format!(
        "--- a/{path}\n+++ b/{path}\n@@ -{old_hunk_start},{old_hunk_len} +{new_hunk_start},{new_hunk_len} @@\n"
    );
    for &l in &old_lines[ctx_before_start..pre] {
        out.push(' ');
        out.push_str(l);
        out.push('\n');
    }
    let max_diff_lines = 180usize;
    for (idx, &(tag, l)) in ops.iter().enumerate() {
        if idx >= max_diff_lines {
            out.push_str(&format!("... ({} more diff lines)\n", ops.len() - max_diff_lines));
            break;
        }
        out.push(tag);
        out.push_str(l);
        out.push('\n');
    }
    for &l in &new_lines[new_lines.len() - suf..ctx_after_end] {
        out.push(' ');
        out.push_str(l);
        out.push('\n');
    }
    (added, removed, out)
}

fn build_file_diff(rel: &str, prev_opt: Option<String>, after: &str) -> Option<FileChangeDiff> {
    let existed_before = prev_opt.is_some();
    let before_content = prev_opt.unwrap_or_default();
    if existed_before && before_content == after {
        return None;
    }
    let (added_lines, removed_lines, unified_diff) =
        compute_unified_diff(rel, &before_content, after);
    Some(FileChangeDiff {
        path: rel.to_string(),
        existed_before,
        before_content,
        added_lines,
        removed_lines,
        unified_diff,
    })
}

fn attach_validation_status(
    workdir: &Path,
    target: &Path,
    rel: &str,
    base_msg: String,
    diff: Option<FileChangeDiff>,
) -> ToolExecOutcome {
    match validate_file(workdir, target, false) {
        Ok(note) => {
            let v_msg = if note.is_empty() { "ok".to_string() } else { note.clone() };
            let suffix = if note.is_empty() {
                " [validation: ok]".to_string()
            } else {
                format!(" [validation: ok; {note}]")
            };
            ToolExecOutcome {
                ok: true,
                validation_ok: true,
                validation_msg: v_msg,
                output: format!("{base_msg}{suffix}"),
                diff,
                new_workdir: None,
            }
        }
        Err(err) => {
            let snippet = extract_validation_line_snippet(target, &err);
            ToolExecOutcome {
                ok: true,
                validation_ok: false,
                validation_msg: err.clone(),
                output: format!(
                    "{base_msg}\n[validation error in `{rel}`: {err} — fix with `edit_file` or `write_file`]{snippet}"
                ),
                diff,
                new_workdir: None,
            }
        }
    }
}

fn extract_validation_line_snippet(target: &Path, err: &str) -> String {
    let Some(pos) = err.find("line ") else {
        return String::new();
    };
    let digits: String = err[pos + 5..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let Ok(line_no) = digits.parse::<usize>() else {
        return String::new();
    };
    if line_no == 0 {
        return String::new();
    }
    let Ok(src) = fs::read_to_string(target) else {
        return String::new();
    };
    let all_lines: Vec<&str> = src.lines().collect();
    if line_no > all_lines.len() {
        return String::new();
    }
    let start = line_no.saturating_sub(1).max(1);
    let end = (line_no + 1).min(all_lines.len());
    let mut out = String::from("\nContext around offending line:");
    for ln in start..=end {
        out.push_str(&format!("\n  {ln}: {}", all_lines[ln - 1]));
    }
    out
}

fn relative_display_in_workdir(workdir: &Path, target: &Path, fallback: &str) -> String {
    if let Ok(rel) = target.strip_prefix(workdir) {
        let s = rel.to_string_lossy().replace('\\', "/");
        if !s.is_empty() {
            return s;
        }
    }
    let canon_work = workdir.canonicalize().unwrap_or_else(|_| workdir.to_path_buf());
    if let Ok(rel) = target.strip_prefix(&canon_work) {
        let s = rel.to_string_lossy().replace('\\', "/");
        if !s.is_empty() {
            return s;
        }
    }
    fallback.to_string()
}

fn extract_plain_html_tag_inner(html: &str, tag: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let open_pat = format!("<{tag}");
    let close_pat = format!("</{tag}>");
    let open = lower.find(&open_pat)?;
    if lower[open + open_pat.len()..].contains(&open_pat) {
        return None;
    }
    let gt_rel = html[open..].find('>')?;
    let inner_start = open + gt_rel + 1;
    let close_rel = lower[inner_start..].find(&close_pat)?;
    let inner = html[inner_start..inner_start + close_rel].trim();
    if inner.is_empty() || inner.contains('<') {
        None
    } else {
        Some(inner.to_string())
    }
}

fn strip_outer_markdown_code_fence(lower_rel: &str, raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if !trimmed.starts_with("```") {
        return None;
    }
    let first_nl = trimmed.find('\n')?;
    let open_fence = trimmed[..first_nl].trim_end_matches('\r');
    let lang = open_fence.strip_prefix("```")?.trim();
    if !lang
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return None;
    }
    let is_md_file = lower_rel.ends_with(".md") || lower_rel.ends_with(".markdown");
    if is_md_file && !lang.eq_ignore_ascii_case("markdown") && !lang.eq_ignore_ascii_case("md") {
        return None;
    }
    let body_and_close = &trimmed[first_nl + 1..];
    let last_nl = body_and_close.rfind('\n')?;
    if body_and_close[last_nl + 1..].trim() != "```" {
        return None;
    }
    let mut inner = body_and_close[..last_nl].to_string();
    inner.push('\n');
    Some(inner)
}

fn exec_write_file(workdir: &Path, args: &Json) -> ToolExecOutcome {
    let Some(rel) = args.get("path").and_then(Json::as_str) else {
        return err_outcome("error: `write_file` requires string argument `path`");
    };
    let Some(content) = args.get("content").and_then(Json::as_str) else {
        return err_outcome(
            "error: `write_file` requires string arguments `path` and `content` (pass full file text in `content`)",
        );
    };
    let lower_rel_check = rel.trim().trim_start_matches("./").to_ascii_lowercase();
    if lower_rel_check == ".kronos_media" || lower_rel_check.starts_with(".kronos_media/") {
        return err_outcome(
            "error: `.kronos_media/` holds read-only attached reference files — call `write_file` on project source files (such as `styles/*.css`, `js/*.js`, `index.html`, or `tests/*.mjs`) instead",
        );
    }
    if content.trim().is_empty() && !lower_rel_check.ends_with(".gitkeep") {
        return err_outcome(&format!(
            "error: refusing to write empty (0-byte) file `{rel}` — pass the complete non-empty source code in `write_file` `content` (write 2 to 3 complete files per step so output never truncates)"
        ));
    }
    let target = match resolve_jailed_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return err_outcome(&format!("error: {e}")),
    };
    if let Some(parent) = target.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return err_outcome(&format!("error creating directory `{}`: {e}", parent.display()));
        }
    }
    let prev_opt = fs::read_to_string(&target).ok();
    let lower_rel = rel.to_ascii_lowercase();
    let mut final_content =
        strip_outer_markdown_code_fence(&lower_rel, content).unwrap_or_else(|| content.to_string());
    let mut sync_note = String::new();
    if lower_rel.ends_with(".html") || lower_rel.ends_with(".htm") {
        if let Some(prev) = &prev_opt {
            if let (Some(old_t), Some(old_h1), Some(new_t), Some(new_h1)) = (
                extract_plain_html_tag_inner(prev, "title"),
                extract_plain_html_tag_inner(prev, "h1"),
                extract_plain_html_tag_inner(&final_content, "title"),
                extract_plain_html_tag_inner(&final_content, "h1"),
            ) {
                if old_t != new_t && new_h1 == old_h1 && new_h1 != new_t {
                    let new_title_tag = format!("<title>{new_t}</title>");
                    let (synced, h1_opt) =
                        sync_html_title_and_h1(rel, "<title>", &new_title_tag, final_content);
                    final_content = synced;
                    if let Some(t) = h1_opt {
                        sync_note = format!("; synced visible <h1> to `{t}`");
                    }
                }
            }
        }
    }
    match fs::write(&target, final_content.as_bytes()) {
        Ok(()) => {
            let clean_rel = relative_display_in_workdir(workdir, &target, rel);
            let diff = build_file_diff(&clean_rel, prev_opt, &final_content);
            let msg = format!(
                "ok: wrote {} bytes ({} lines{sync_note}) to `{clean_rel}`",
                final_content.len(),
                final_content.lines().count()
            );
            attach_validation_status(workdir, &target, &clean_rel, msg, diff)
        }
        Err(e) => err_outcome(&format!("error writing `{rel}`: {e}")),
    }
}

/// Matches `norm_old` against `norm_existing` by comparing contiguous non-empty trimmed lines.
fn match_unique_trimmed_span(
    norm_existing: &str,
    norm_old: &str,
    new_text: &str,
) -> Option<(usize, String)> {
    let old_nonempty: Vec<&str> = norm_old
        .split('\n')
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if old_nonempty.is_empty() {
        return None;
    }

    let mut ex_nonempty: Vec<(usize, usize, &str)> = Vec::new();
    let mut cursor = 0usize;
    for raw_line in norm_existing.split('\n') {
        let start = cursor;
        let end = start + raw_line.len();
        let trimmed = raw_line.trim();
        if !trimmed.is_empty() {
            ex_nonempty.push((start, end, trimmed));
        }
        cursor = end + 1;
    }
    if ex_nonempty.len() < old_nonempty.len() {
        return None;
    }

    let m = old_nonempty.len();
    let mut found_range: Option<(usize, usize)> = None;
    for k in 0..=(ex_nonempty.len() - m) {
        if (0..m).all(|j| ex_nonempty[k + j].2 == old_nonempty[j]) {
            if found_range.is_some() {
                return None;
            }
            found_range = Some((ex_nonempty[k].0, ex_nonempty[k + m - 1].1));
        }
    }

    let (start_byte, mut end_byte) = found_range?;
    let norm_new = new_text.replace("\r\n", "\n");
    if norm_new.ends_with('\n') && norm_existing.as_bytes().get(end_byte) == Some(&b'\n') {
        end_byte += 1;
    }
    let updated = format!(
        "{}{}{}",
        &norm_existing[..start_byte],
        norm_new,
        &norm_existing[end_byte..]
    );
    Some((start_byte, updated))
}

/// Keeps visible `<h1>` synchronized when `edit_file` updates `<title>...</title>` in an HTML document.
fn sync_html_title_and_h1(
    rel: &str,
    old_text: &str,
    new_text: &str,
    updated: String,
) -> (String, Option<String>) {
    let lower_rel = rel.to_ascii_lowercase();
    if !(lower_rel.ends_with(".html") || lower_rel.ends_with(".htm")) {
        return (updated, None);
    }
    let lower_new = new_text.to_ascii_lowercase();
    let lower_old = old_text.to_ascii_lowercase();
    if !lower_new.contains("<title>") && !lower_old.contains("<title>") {
        return (updated, None);
    }
    if lower_new.contains("<h1") {
        return (updated, None);
    }
    let Some(t_open) = lower_new.find("<title>") else {
        return (updated, None);
    };
    let after_t = t_open + "<title>".len();
    let Some(t_close_rel) = lower_new[after_t..].find("</title>") else {
        return (updated, None);
    };
    let new_title = new_text[after_t..after_t + t_close_rel].trim();
    if new_title.is_empty() || new_title.contains('<') {
        return (updated, None);
    }

    let lower_upd = updated.to_ascii_lowercase();
    let Some(h1_open) = lower_upd.find("<h1") else {
        return (updated, None);
    };
    // Only auto-sync when there is a single `<h1>` element with plain text content.
    if lower_upd[h1_open + 3..].contains("<h1") {
        return (updated, None);
    }
    let Some(gt_rel) = updated[h1_open..].find('>') else {
        return (updated, None);
    };
    let inner_start = h1_open + gt_rel + 1;
    let Some(h1_close_rel) = lower_upd[inner_start..].find("</h1>") else {
        return (updated, None);
    };
    let inner_end = inner_start + h1_close_rel;
    let cur_h1 = updated[inner_start..inner_end].trim();
    if cur_h1.is_empty() || cur_h1.contains('<') || cur_h1 == new_title {
        return (updated, None);
    }
    let synced = format!(
        "{}{}{}",
        &updated[..inner_start],
        new_title,
        &updated[inner_end..]
    );
    (synced, Some(new_title.to_string()))
}

fn write_edited_content(
    workdir: &Path,
    target: &Path,
    rel: &str,
    line_no: usize,
    old_text: &str,
    new_text: &str,
    raw_updated: String,
) -> ToolExecOutcome {
    let prev_opt = fs::read_to_string(target).ok();
    let clean_rel = relative_display_in_workdir(workdir, target, rel);
    let (updated, h1_synced) = sync_html_title_and_h1(&clean_rel, old_text, new_text, raw_updated);
    match fs::write(target, updated.as_bytes()) {
        Ok(()) => {
            let diff = build_file_diff(&clean_rel, prev_opt, &updated);
            let sync_note = match h1_synced {
                Some(t) => format!("; synced visible <h1> to `{t}`"),
                None => String::new(),
            };
            let msg = format!(
                "ok: edited `{clean_rel}` at line {line_no} ({} bytes, {} lines{sync_note})",
                updated.len(),
                updated.lines().count()
            );
            attach_validation_status(workdir, target, &clean_rel, msg, diff)
        }
        Err(e) => err_outcome(&format!("error writing `{clean_rel}`: {e}")),
    }
}

fn exec_edit_file(workdir: &Path, args: &Json) -> ToolExecOutcome {
    let Some(rel) = args.get("path").and_then(Json::as_str) else {
        return err_outcome("error: `edit_file` requires string argument `path`");
    };
    let Some(old_text) = args.get("old_text").and_then(Json::as_str) else {
        return err_outcome("error: `edit_file` requires string argument `old_text`");
    };
    let Some(new_text) = args.get("new_text").and_then(Json::as_str) else {
        return err_outcome("error: `edit_file` requires string argument `new_text`");
    };
    let target = match resolve_jailed_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return err_outcome(&format!("error: {e}")),
    };
    if !old_text.trim().is_empty() && old_text == new_text {
        let mut snippet = String::new();
        if let Ok(existing) = fs::read_to_string(&target) {
            if let Some(pos) = existing.find(old_text) {
                let (ln, _) = line_col_at(&existing, pos, 1);
                let all_lines: Vec<&str> = existing.lines().collect();
                let s = ln.saturating_sub(1).max(1);
                let e = (ln + 2).min(all_lines.len());
                snippet.push_str("\nExact lines on disk at that location:");
                for l in s..=e {
                    snippet.push_str(&format!("\n  {l}: {}", all_lines[l - 1]));
                }
            }
        }
        return err_outcome(&format!(
            "error: `edit_file` `new_text` is identical to `old_text` (no-op edit) — include the full line on disk in `old_text` and provide a different `new_text`, or rewrite cleanly with `write_file`{snippet}"
        ));
    }
    if old_text.trim().is_empty() {
        if new_text.trim().is_empty() {
            return err_outcome("error: `edit_file` `old_text` and `new_text` cannot both be empty");
        }
        if let Some(parent) = target.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return err_outcome(&format!("error creating parent dirs for `{rel}`: {e}"));
            }
        }
        return write_edited_content(workdir, &target, rel, 1, old_text, new_text, new_text.to_string());
    }
    let existing = match fs::read_to_string(&target) {
        Ok(s) => s,
        Err(e) => return err_outcome(&format!("error reading `{rel}`: {e}")),
    };

    let mut match_positions = Vec::new();
    let mut search_from = 0usize;
    while let Some(pos) = existing[search_from..].find(old_text) {
        let abs = search_from + pos;
        match_positions.push(abs);
        search_from = abs + old_text.len().max(1);
    }

    if match_positions.is_empty() {
        // Fallback 1: CRLF vs LF mismatch.
        let norm_existing = existing.replace("\r\n", "\n");
        let norm_old = old_text.replace("\r\n", "\n");
        if let Some(pos) = norm_existing.find(&norm_old) {
            if !norm_existing[pos + norm_old.len()..].contains(&norm_old) {
                let norm_new = new_text.replace("\r\n", "\n");
                let updated = format!(
                    "{}{}{}",
                    &norm_existing[..pos],
                    norm_new,
                    &norm_existing[pos + norm_old.len()..]
                );
                let (line_no, _) = line_col_at(&norm_existing, pos, 1);
                return write_edited_content(workdir, &target, rel, line_no, old_text, new_text, updated);
            }
        }
        // Fallback 2: unique contiguous trimmed-line match across indentation or blank-line differences.
        if let Some((pos, updated)) = match_unique_trimmed_span(&norm_existing, &norm_old, new_text) {
            let (line_no, _) = line_col_at(&norm_existing, pos, 1);
            return write_edited_content(workdir, &target, rel, line_no, old_text, new_text, updated);
        }
        // Fallback 3: idempotent no-op when a prior edit in the same turn already applied new_text.
        let norm_new_trim = new_text.replace("\r\n", "\n");
        let norm_new_trim = norm_new_trim.trim();
        if !norm_new_trim.is_empty() && norm_existing.contains(norm_new_trim) {
            let msg = format!("ok: `{rel}` already contains the requested text");
            return attach_validation_status(workdir, &target, rel, msg, None);
        }
        let first_old_line = old_text
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("")
            .trim();
        let mut hint = String::new();
        if !first_old_line.is_empty() {
            for (idx, line) in existing.lines().enumerate() {
                if line.contains(first_old_line) {
                    hint = format!(" (nearest match for `{first_old_line}` is at line {})", idx + 1);
                    break;
                }
            }
        }
        if existing.len() <= 6144 {
            let mut numbered = String::new();
            for (idx, line) in existing.lines().enumerate() {
                numbered.push_str(&format!("{:>4} | {}\n", idx + 1, line));
            }
            return err_outcome(&format!(
                "error: `old_text` not found in `{rel}`{hint}. Use `write_file` to replace `{rel}` completely, or pass exact lines from `{rel}` below to `edit_file`:\n{numbered}"
            ));
        }
        return err_outcome(&format!(
            "error: `old_text` not found in `{rel}`{hint}; call `read_file` to inspect exact text or `write_file` to replace the file"
        ));
    }

    if match_positions.len() > 1 {
        let lines: Vec<String> = match_positions
            .iter()
            .take(6)
            .map(|&p| line_col_at(&existing, p, 1).0.to_string())
            .collect();
        return err_outcome(&format!(
            "error: `old_text` matched {} locations in `{rel}` (lines {}); include more surrounding lines so `old_text` is unique",
            match_positions.len(),
            lines.join(", ")
        ));
    }

    let pos = match_positions[0];
    let (line_no, _) = line_col_at(&existing, pos, 1);
    let updated = format!("{}{}{}", &existing[..pos], new_text, &existing[pos + old_text.len()..]);
    write_edited_content(workdir, &target, rel, line_no, old_text, new_text, updated)
}

fn exec_read_file(workdir: &Path, args: &Json) -> ToolExecOutcome {
    let Some(rel) = args.get("path").and_then(Json::as_str) else {
        return err_outcome("error: `read_file` requires string argument `path`");
    };
    let target = match resolve_readable_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return err_outcome(&format!("error: {e}")),
    };
    let text = match fs::read_to_string(&target) {
        Ok(t) => t,
        Err(e) => return err_outcome(&format!("error reading `{rel}`: {e}")),
    };

    let start_opt = args.get("start_line").and_then(Json::as_f64).map(|v| (v.max(1.0)) as usize);
    let end_opt = args.get("end_line").and_then(Json::as_f64).map(|v| (v.max(1.0)) as usize);
    let all_lines: Vec<&str> = text.lines().collect();
    let total = all_lines.len();
    if total == 0 {
        return ok_outcome("(empty file)".to_string());
    }

    let s_line = start_opt.unwrap_or(1).clamp(1, total);
    let e_line = end_opt.unwrap_or(total).clamp(s_line, total);
    let mut formatted = String::new();
    if start_opt.is_some() || end_opt.is_some() {
        formatted.push_str(&format!("[`{rel}` lines {s_line}..{e_line} of {total}]\n"));
    }
    for (offset, line) in all_lines[s_line - 1..e_line].iter().enumerate() {
        if offset > 0 {
            formatted.push('\n');
        }
        formatted.push_str(&format!("{}: {line}", s_line + offset));
    }
    ok_outcome(truncate_utf8(&formatted, MAX_READ_BYTES))
}

fn exec_list_dir(workdir: &Path, args: &Json) -> ToolExecOutcome {
    let rel = args.get("path").and_then(Json::as_str).unwrap_or(".");
    let target = match resolve_jailed_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return err_outcome(&format!("error: {e}")),
    };
    let entries = match fs::read_dir(&target) {
        Ok(rd) => rd,
        Err(e) => return err_outcome(&format!("error listing `{rel}`: {e}")),
    };
    let mut rows = Vec::new();
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if should_skip_entry(&name) {
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
        ok_outcome("(empty directory)".to_string())
    } else {
        ok_outcome(rows.join("\n"))
    }
}

fn exec_search_files(workdir: &Path, args: &Json) -> ToolExecOutcome {
    let Some(pattern) = args.get("pattern").and_then(Json::as_str) else {
        return err_outcome("error: `search_files` requires string argument `pattern`");
    };
    if pattern.is_empty() {
        return err_outcome("error: `search_files` `pattern` must not be empty");
    }
    let rel = args.get("path").and_then(Json::as_str).unwrap_or(".");
    let ci = args.get("case_insensitive").and_then(Json::as_bool).unwrap_or(false);
    let target = match resolve_jailed_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return err_outcome(&format!("error: {e}")),
    };
    let pat_cmp = if ci { pattern.to_lowercase() } else { String::new() };
    let mut hits = Vec::new();
    search_rec(workdir, &target, pattern, &pat_cmp, ci, &mut hits);
    if hits.is_empty() {
        ok_outcome(format!("no matches for `{pattern}` in `{rel}`"))
    } else {
        let truncated_note = if hits.len() >= MAX_SEARCH_MATCHES {
            format!("\n...[capped at {MAX_SEARCH_MATCHES} matches]")
        } else {
            String::new()
        };
        ok_outcome(truncate_utf8(&format!("{}{truncated_note}", hits.join("\n")), MAX_READ_BYTES))
    }
}

fn search_rec(
    root: &Path,
    cur: &Path,
    pattern: &str,
    pat_lower: &str,
    ci: bool,
    hits: &mut Vec<String>,
) {
    if hits.len() >= MAX_SEARCH_MATCHES {
        return;
    }
    if cur.is_file() {
        search_single_file(root, cur, pattern, pat_lower, ci, hits);
        return;
    }
    let Ok(rd) = fs::read_dir(cur) else {
        return;
    };
    let mut paths: Vec<PathBuf> = rd
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if should_skip_entry(&name) {
                None
            } else {
                Some(e.path())
            }
        })
        .collect();
    paths.sort();
    for p in paths {
        if hits.len() >= MAX_SEARCH_MATCHES {
            break;
        }
        if p.is_dir() {
            search_rec(root, &p, pattern, pat_lower, ci, hits);
        } else if p.is_file() {
            search_single_file(root, &p, pattern, pat_lower, ci, hits);
        }
    }
}

fn search_single_file(
    root: &Path,
    file: &Path,
    pattern: &str,
    pat_lower: &str,
    ci: bool,
    hits: &mut Vec<String>,
) {
    let Ok(bytes) = fs::read(file) else {
        return;
    };
    if bytes.iter().take(512).any(|&b| b == 0) {
        return;
    }
    let Ok(text) = String::from_utf8(bytes) else {
        return;
    };
    let rel = file
        .strip_prefix(root)
        .unwrap_or(file)
        .to_string_lossy()
        .replace('\\', "/");
    for (idx, line) in text.lines().enumerate() {
        let matched = if ci {
            line.to_lowercase().contains(pat_lower)
        } else {
            line.contains(pattern)
        };
        if matched {
            let snippet = truncate_utf8(line.trim(), 180);
            hits.push(format!("{rel}:{}: {snippet}", idx + 1));
            if hits.len() >= MAX_SEARCH_MATCHES {
                break;
            }
        }
    }
}

fn ok_outcome(output: String) -> ToolExecOutcome {
    ToolExecOutcome {
        ok: true,
        validation_ok: true,
        validation_msg: "ok".to_string(),
        output,
        diff: None,
        new_workdir: None,
    }
}

fn err_outcome(msg: &str) -> ToolExecOutcome {
    ToolExecOutcome {
        ok: false,
        validation_ok: true,
        validation_msg: "ok".to_string(),
        output: msg.to_string(),
        diff: None,
        new_workdir: None,
    }
}

#[derive(Clone, Copy)]
struct ShellWordSpan {
    start: usize,
    end: usize,
    unquoted: bool,
    has_backslash: bool,
}

/// Automatically wraps unquoted space-containing filesystem paths that physically exist on disk
/// (such as `/Users/.../Downloads/James Iroha Close 4.m4a` or `/Applications/Google Chrome.app/...`)
/// in single quotes so `sh -c` does not split them on spaces.
fn auto_quote_unquoted_existing_paths(workdir: &Path, cmd: &str) -> String {
    if !cmd.contains(' ') {
        return cmd.to_string();
    }
    let bytes = cmd.as_bytes();
    let n = bytes.len();
    let mut spans: Vec<ShellWordSpan> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let b = bytes[i];
        if b.is_ascii_whitespace() || matches!(b, b';' | b'|' | b'&' | b'<' | b'>' | b'(' | b')') {
            i += 1;
            continue;
        }
        let start = i;
        let mut unquoted = true;
        let mut has_backslash = false;
        while i < n {
            let c = bytes[i];
            if c.is_ascii_whitespace() || matches!(c, b';' | b'|' | b'&' | b'<' | b'>' | b'(' | b')') {
                break;
            }
            if c == b'\'' {
                unquoted = false;
                i += 1;
                while i < n && bytes[i] != b'\'' {
                    i += 1;
                }
                if i < n {
                    i += 1;
                }
            } else if c == b'"' {
                unquoted = false;
                i += 1;
                while i < n && bytes[i] != b'"' {
                    if bytes[i] == b'\\' && i + 1 < n {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if i < n {
                    i += 1;
                }
            } else if c == b'`' {
                unquoted = false;
                i += 1;
                while i < n && bytes[i] != b'`' {
                    if bytes[i] == b'\\' && i + 1 < n {
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
                if i < n {
                    i += 1;
                }
            } else if c == b'\\' && i + 1 < n {
                has_backslash = true;
                i += 2;
            } else {
                i += 1;
            }
        }
        spans.push(ShellWordSpan {
            start,
            end: i,
            unquoted,
            has_backslash,
        });
    }

    if spans.len() < 2 {
        return cmd.to_string();
    }

    let mut out = String::with_capacity(cmd.len() + 16);
    let mut cursor = 0usize;
    let mut idx = 0usize;
    while idx < spans.len() {
        let sp = spans[idx];
        let tok = &cmd[sp.start..sp.end];
        let mut matched: Option<(usize, String)> = None;
        if sp.unquoted && !sp.has_backslash {
            let (prefix, head, is_abs) =
                if tok.starts_with('/') && tok.len() > 1 && !tok.starts_with("//") {
                    ("", tok, true)
                } else if let Some(eq_pos) = tok.find("=/") {
                    let h = &tok[eq_pos + 1..];
                    if h.len() > 1 && !h.starts_with("//") {
                        (&tok[..eq_pos + 1], h, true)
                    } else {
                        ("", "", false)
                    }
                } else if !tok.starts_with('-') && idx > 0 {
                    ("", tok, false)
                } else {
                    ("", "", false)
                };

            if !head.is_empty() {
                let mut max_k = 0usize;
                for k in 1..=8 {
                    if idx + k >= spans.len() {
                        break;
                    }
                    let prev = spans[idx + k - 1];
                    let next = spans[idx + k];
                    let sep = &cmd[prev.end..next.start];
                    if !next.unquoted
                        || next.has_backslash
                        || sep.is_empty()
                        || !sep.bytes().all(|b| b == b' ')
                    {
                        break;
                    }
                    max_k = k;
                }
                for k in (1..=max_k).rev() {
                    let tail_end = spans[idx + k].end;
                    let candidate = format!("{head}{}", &cmd[sp.end..tail_end]);
                    let exists = if is_abs {
                        Path::new(&candidate).exists()
                    } else {
                        workdir.join(&candidate).exists()
                    };
                    if exists {
                        let escaped = candidate.replace('\'', "'\\''");
                        matched = Some((idx + k, format!("{prefix}'{escaped}'")));
                        break;
                    }
                }
            }
        }

        if let Some((last_idx, replacement)) = matched {
            out.push_str(&cmd[cursor..sp.start]);
            out.push_str(&replacement);
            cursor = spans[last_idx].end;
            idx = last_idx + 1;
        } else {
            idx += 1;
        }
    }
    if cursor == 0 {
        return cmd.to_string();
    }
    out.push_str(&cmd[cursor..]);
    out
}

/// Extracts third-party Python package roots (`mlx_audio`, `faster_whisper`, `PIL`, etc.) from a
/// bare `python3 <script>.py` invocation and binds it to an indexed local `.venv/bin/python3`
/// when `workdir` has no local `.venv` of its own.
fn resolve_device_venv_python_command(workdir: &Path, cmd: &str) -> String {
    let trimmed = cmd.trim_start();
    let lead_ws = &cmd[..cmd.len() - trimmed.len()];
    let (py_bin, after_py) = if let Some(rest) = trimmed.strip_prefix("python3 ") {
        ("python3", rest)
    } else if let Some(rest) = trimmed.strip_prefix("python ") {
        ("python", rest)
    } else {
        return cmd.to_string();
    };
    let after_trimmed = after_py.trim_start();
    if after_trimmed.starts_with("-m py_compile")
        || after_trimmed.starts_with("-m pip")
        || workdir.join(".venv/bin/python3").is_file()
    {
        return cmd.to_string();
    }
    let mut py_source = String::new();
    for tok in after_trimmed.split_whitespace() {
        if tok == "-c" {
            py_source.push_str(after_trimmed);
            break;
        }
        if tok.starts_with('-') {
            continue;
        }
        let clean = tok.trim_matches(|c| c == '\'' || c == '"');
        if clean.ends_with(".py") {
            if let Ok(full) = resolve_jailed_path(workdir, clean) {
                if let Ok(src) = fs::read_to_string(&full) {
                    py_source = src;
                }
            }
        }
        break;
    }
    if py_source.is_empty() {
        return cmd.to_string();
    }
    const SPECIALIZED_PKGS: &[&str] = &[
        "mlx_audio",
        "faster_whisper",
        "mlx",
        "soundfile",
        "librosa",
        "scipy",
        "moviepy",
        "cv2",
        "torch",
    ];
    let mut needed: Vec<String> = Vec::new();
    for line in py_source.lines() {
        let l = line.trim();
        let mod_spec = if let Some(r) = l.strip_prefix("from ") {
            r.split_whitespace().next()
        } else if let Some(r) = l.strip_prefix("import ") {
            r.split(|c: char| c.is_ascii_whitespace() || c == ',').next()
        } else {
            None
        };
        if let Some(m) = mod_spec {
            let root = m.split('.').next().unwrap_or("").trim();
            if SPECIALIZED_PKGS.iter().any(|p| p.eq_ignore_ascii_case(root))
                && !needed.iter().any(|e| e.eq_ignore_ascii_case(root))
            {
                needed.push(root.to_string());
            }
        }
    }
    if needed.is_empty() {
        return cmd.to_string();
    }
    if let Some(venv_py) = crate::device::global_device_manager().find_venv_python_for_packages(&needed) {
        let rest_after_bin = &trimmed[py_bin.len()..];
        let quoted_py = format!("'{}'", venv_py.replace('\'', "'\\''"));
        return format!("{lead_ws}{quoted_py}{rest_after_bin}");
    }
    cmd.to_string()
}

/// Redirects stdout/stderr to a file rather than a pipe so backgrounded child processes do not block wait.
pub fn run_shell_command_with_timeout(
    workdir: &Path,
    cmd: &str,
    timeout_secs: u64,
    step_tag: usize,
) -> String {
    let quoted_cmd = auto_quote_unquoted_existing_paths(workdir, cmd);
    let normalized_cmd = resolve_device_venv_python_command(workdir, &quoted_cmd);
    let cmd = normalized_cmd.as_str();
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

    let inherited_path = std::env::var("PATH").unwrap_or_default();
    let full_path = if inherited_path.contains("/opt/homebrew/bin") {
        inherited_path
    } else if inherited_path.is_empty() {
        "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin".to_string()
    } else {
        format!("/opt/homebrew/bin:/usr/local/bin:{inherited_path}")
    };

    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(workdir)
        .env("PATH", &full_path)
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
    let text_out = truncate_utf8_head_tail(&String::from_utf8_lossy(&raw_out), MAX_CMD_OUTPUT_BYTES);

    if timed_out {
        format!(
            "timed_out after {timeout_secs}s (process killed; start long-running servers in background with `> server.log 2>&1 &`)\noutput:\n{text_out}"
        )
    } else {
        let code = exit_code.unwrap_or(0);
        let py_inline_hint = if code != 0
            && (cmd.contains("python -c") || cmd.contains("python3 -c"))
            && (text_out.contains("SyntaxError") || cmd.contains("; for ") || cmd.contains("; with "))
        {
            "\n[hint: Python `-c` one-liners reject `; for` / `; with` compound statements after semicolons — use `write_file` to save a `.py` script file and execute `python3 <script>.py` with `run_command`]"
        } else {
            ""
        };
        if text_out.trim().is_empty() {
            format!("exit_code={code} (no stdout/stderr){py_inline_hint}")
        } else {
            format!("exit_code={code}\n{text_out}{py_inline_hint}")
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FilterMode {
    Normal,
    XmlTool,
    XmlToolsBlock,
    FencedJsonTool,
}

fn balanced_json_end(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.first().copied() != Some(b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut in_str = false;
    let mut esc = false;
    for (idx, &b) in bytes.iter().enumerate() {
        if in_str {
            if esc {
                esc = false;
            } else if b == b'\\' {
                esc = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_line_start_brace(buf: &str, emitted: usize) -> Option<usize> {
    let tail = &buf[emitted..];
    let mut at_line_start = emitted == 0 || buf.as_bytes()[emitted - 1] == b'\n';
    for (i, b) in tail.bytes().enumerate() {
        if b == b'\n' {
            at_line_start = true;
        } else if at_line_start {
            if b == b'{' {
                return Some(i);
            }
            if b != b' ' && b != b'\t' && b != b'\r' {
                at_line_start = false;
            }
        }
    }
    None
}

/// Filters `<tool_call>...</tool_call>`, fenced JSON, and bare JSON tool blocks out of live prose streaming.
struct ProseStreamFilter {
    buf: String,
    emitted: usize,
    mode: FilterMode,
}

impl ProseStreamFilter {
    fn new() -> Self {
        Self {
            buf: String::new(),
            emitted: 0,
            mode: FilterMode::Normal,
        }
    }

    fn push(&mut self, piece: &str, mut on_prose: impl FnMut(&str) -> bool) -> bool {
        const OPEN: &str = "<tool_call>";
        const CLOSE: &str = "</tool_call>";
        const OPEN_TOOLS: &str = "<tools>";
        const CLOSE_TOOLS: &str = "</tools>";
        const FENCE: &str = "```json";
        self.buf.push_str(piece);
        loop {
            let tail = &self.buf[self.emitted..];
            match self.mode {
                FilterMode::XmlTool => {
                    if let Some(end_rel) = tail.find(CLOSE) {
                        self.emitted += end_rel + CLOSE.len();
                        self.mode = FilterMode::Normal;
                    } else {
                        break;
                    }
                }
                FilterMode::XmlToolsBlock => {
                    if let Some(end_rel) = tail.find(CLOSE_TOOLS) {
                        self.emitted += end_rel + CLOSE_TOOLS.len();
                        self.mode = FilterMode::Normal;
                    } else {
                        break;
                    }
                }
                FilterMode::FencedJsonTool => {
                    let Some(obj_end) = balanced_json_end(tail) else {
                        break;
                    };
                    let rest = &tail[obj_end..];
                    let trimmed = rest.trim_start();
                    let ws_len = rest.len() - trimmed.len();
                    if trimmed.starts_with("```") {
                        self.emitted += obj_end + ws_len + 3;
                        self.mode = FilterMode::Normal;
                    } else if rest.len() < 16 {
                        break;
                    } else {
                        self.emitted += obj_end;
                        self.mode = FilterMode::Normal;
                    }
                }
                FilterMode::Normal => {
                    let xml_pos = tail.find(OPEN);
                    let tools_pos = tail.find(OPEN_TOOLS);
                    let fence_pos = tail.find(FENCE);
                    let brace_pos = find_line_start_brace(&self.buf, self.emitted);
                    let earliest = [xml_pos, tools_pos, fence_pos, brace_pos]
                        .into_iter()
                        .flatten()
                        .min();
                    if let Some(open_rel) = xml_pos.filter(|&x| Some(x) == earliest) {
                        if open_rel > 0 {
                            let chunk = &self.buf[self.emitted..self.emitted + open_rel];
                            if !on_prose(chunk) {
                                return false;
                            }
                        }
                        self.emitted += open_rel + OPEN.len();
                        self.mode = FilterMode::XmlTool;
                    } else if let Some(t_rel) = tools_pos.filter(|&x| Some(x) == earliest) {
                        if t_rel > 0 {
                            let chunk = &self.buf[self.emitted..self.emitted + t_rel];
                            if !on_prose(chunk) {
                                return false;
                            }
                        }
                        self.emitted += t_rel + OPEN_TOOLS.len();
                        self.mode = FilterMode::XmlToolsBlock;
                    } else if let Some(f_rel) = fence_pos.filter(|&x| Some(x) == earliest) {
                        if f_rel > 0 {
                            let chunk = &self.buf[self.emitted..self.emitted + f_rel];
                            self.emitted += f_rel;
                            if !on_prose(chunk) {
                                return false;
                            }
                            continue;
                        }
                        let after_fence = &tail[FENCE.len()..];
                        let trimmed = after_fence.trim_start();
                        let ws_len = after_fence.len() - trimmed.len();
                        if trimmed.is_empty() {
                            if after_fence.len() < 16 {
                                break;
                            }
                            self.emitted += FENCE.len();
                            if !on_prose(FENCE) {
                                return false;
                            }
                        } else if let Some(inner) = trimmed.strip_prefix('{') {
                            let inner_t = inner.trim_start();
                            if inner_t.starts_with("\"name\"") {
                                self.emitted += FENCE.len() + ws_len;
                                self.mode = FilterMode::FencedJsonTool;
                            } else if "\"name\"".starts_with(inner_t) && after_fence.len() < 32 {
                                break;
                            } else {
                                self.emitted += FENCE.len();
                                if !on_prose(FENCE) {
                                    return false;
                                }
                            }
                        } else {
                            self.emitted += FENCE.len();
                            if !on_prose(FENCE) {
                                return false;
                            }
                        }
                    } else if let Some(b_rel) = brace_pos {
                        if b_rel > 0 {
                            let chunk = &self.buf[self.emitted..self.emitted + b_rel];
                            self.emitted += b_rel;
                            if !on_prose(chunk) {
                                return false;
                            }
                            continue;
                        }
                        let inner_t = tail[1..].trim_start();
                        if inner_t.starts_with("\"name\"") {
                            self.mode = FilterMode::FencedJsonTool;
                        } else if "\"name\"".starts_with(inner_t) && tail.len() < 24 {
                            break;
                        } else {
                            self.emitted += 1;
                            if !on_prose("{") {
                                return false;
                            }
                        }
                    } else {
                        let mut hold = 0usize;
                        for pat in [OPEN, OPEN_TOOLS, FENCE] {
                            let max_check = tail.len().min(pat.len() - 1);
                            for k in (1..=max_check).rev() {
                                if tail.is_char_boundary(tail.len() - k)
                                    && pat.starts_with(&tail[tail.len() - k..])
                                {
                                    hold = hold.max(k);
                                    break;
                                }
                            }
                        }
                        let emit_len = tail.len() - hold;
                        if emit_len > 0 {
                            let chunk = &self.buf[self.emitted..self.emitted + emit_len];
                            self.emitted += emit_len;
                            if !on_prose(chunk) {
                                return false;
                            }
                        }
                        break;
                    }
                }
            }
        }
        true
    }

    fn finish(&mut self, mut on_prose: impl FnMut(&str) -> bool) -> bool {
        if self.mode == FilterMode::FencedJsonTool {
            let tail = &self.buf[self.emitted..];
            if let Some(obj_end) = balanced_json_end(tail) {
                let rest = &tail[obj_end..];
                let trimmed = rest.trim_start();
                let ws_len = rest.len() - trimmed.len();
                if trimmed.starts_with("```") {
                    self.emitted += obj_end + ws_len + 3;
                } else {
                    self.emitted += obj_end;
                }
                self.mode = FilterMode::Normal;
            }
        }
        if self.mode == FilterMode::Normal && self.emitted < self.buf.len() {
            let chunk = &self.buf[self.emitted..];
            self.emitted = self.buf.len();
            return on_prose(chunk);
        }
        true
    }

    fn active_tool_progress(&self) -> Option<(String, String, usize)> {
        if self.mode == FilterMode::Normal || self.emitted >= self.buf.len() {
            return None;
        }
        let tail = &self.buf[self.emitted..];
        let name = extract_partial_json_str(tail, "\"name\"").unwrap_or_else(|| "tool".to_string());
        let target = extract_partial_json_str(tail, "\"path\"")
            .or_else(|| extract_partial_json_str(tail, "\"target\""))
            .or_else(|| extract_partial_json_str(tail, "\"cmd\""))
            .or_else(|| extract_partial_json_str(tail, "\"pattern\""))
            .unwrap_or_default();
        Some((name, target, tail.len()))
    }
}

fn extract_partial_json_str(haystack: &str, key: &str) -> Option<String> {
    let idx = haystack.find(key)?;
    let after_key = haystack[idx + key.len()..].trim_start();
    let after_colon = after_key.strip_prefix(':')?.trim_start();
    let inner = after_colon.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = inner.chars();
    while let Some(ch) = chars.next() {
        if ch == '"' {
            return Some(out);
        }
        if ch == '\\' {
            if let Some(esc) = chars.next() {
                out.push(esc);
            } else {
                break;
            }
        } else {
            out.push(ch);
        }
        if out.len() >= 48 {
            return Some(out);
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

pub const UNIFIED_SYSTEM_PROMPT: &str = concat!(
    "Your name is Kronos. Melody Leonard (Melody Leonard Oparaugo) created and built you as a local multimodal software engineering and reasoning engine on Apple Silicon. ",
    "Always speak in the first person as Kronos ('Melody Leonard built me') when asked your name or who built you, and know Melody Leonard's verified public profile: Melody Leonard (full name: Melody Leonard Oparaugo, pronouns: he/him/his) is a Senior Software Engineer and Systems Architect (not a musician or actor) with over 10 years of software engineering experience (@IPMScoutek). His GitHub is https://github.com/melodyLeonard (52+ public repos including kronos-2t, tshare, cross-native, gpt_2_tokenizer, tomatoe), his LinkedIn is https://www.linkedin.com/in/melody-leonard-a362b6192, his email is melodyleonard7@gmail.com, and his X handle is @Melody_Leonard7. He specializes in Rust, TypeScript, JavaScript, Elixir, Solidity, Node.js/NestJS, Kafka/CQRS/GraphQL/gRPC microservices, PostgreSQL/MongoDB/Neo4j/Redis internals, React/React Native offline-first architecture, and AWS/Kubernetes CI/CD. ",
    "For general questions, explanations, or analysis of attached PDFs, documents, CSVs, code snippets, or images that do not modify workspace files, reply directly and concisely from the inline context without calling tools. ",
    "Because Kronos runs locally on hardware owned by the user, never refuse queries about the user's own local messaging apps (WhatsApp, iMessage, Telegram, Slack, Notes), files, or past conversations with stock disclaimers; when a [Local Conversations & Memory] or [Device Context] block is provided, answer directly and accurately from its verified local records, and when `use_computer` is available and the user asks to open or control an application, file, folder, or URL on their Mac, call `use_computer` directly. ",
    "When asked to create a new folder or project inside another local directory on this Mac (such as ~/Documents/orgProj), call `switch_workspace` first with the full new child folder path to create and activate that project workspace, then call `write_file` to build the project files inside it. ",
    "When asked to build, edit, style, fix, inspect, or run files in the workspace, use the workspace tools (switch_workspace, write_file, edit_file, read_file, list_dir, search_files, run_command) directly on disk. ",
    "When building a website or web app, deliver a complete, styled, responsive page with modern CSS layout, navigation, domain-specific sections, and interactive UI rather than a minimal placeholder page. ",
    "Do not paste file drafts or completion claims in prose before calling tools; emit <tool_call> invocations immediately, then summarize briefly after tool calls succeed. ",
    "Check the Existing workspace files list below before calling edit_file: only call edit_file on files that exist on disk using exact existing lines, and use write_file for new files or full rewrites. ",
    "When updating a web page title or heading, update both <title> in <head> and the visible <h1> heading in <body> so the rendered preview shows the change immediately. ",
    "The live preview reloads automatically after every file edit, so never ask the user to refresh the preview; if the user reports a change is not visible, inspect and update the visible HTML or CSS elements directly. ",
    "When asked to generate technical scripts, voice-cloned audio, or rendered video using local open-source tools, use the grounded Kronos engine documentation in [Device Context] (or call read_file on README.md and BENCHMARKS.md if not inlined) so every iteration (Iterations 1 through 8), mathematical equation, and benchmark comparison (Ollama, llama.cpp Q4_0/Q4_K_M/IQ4_XS/Q6_K, scalar VQ2 vs 8D Gosset E8 e8q2g128, generative judges vs Kronos-Jev) is exact, and execute cleanly in focused steps: first call write_file to save the complete spoken technical script (script.md >= 4.0 KB starting directly with '# ' plain markdown headings without wrapping content in outer triple-backtick code fences, covering all 8 Kronos iterations, formulas, and benchmark comparisons in full prose paragraphs rather than short bullet outlines); next call write_file to save one complete self-contained Python builder script (build_video.py with zero pass/TODO placeholders, using subprocess.run to run /opt/homebrew/bin/ffmpeg for extracting 12s raw_ref_16k.wav and raw_ref_24k.wav, loading raw_ref_16k.wav via wave + np.frombuffer into faster_whisper segs, _ = wm.transcribe(pcm), slicing raw_ref_24k.wav strictly to int(end_sec * 24000) int16 frames into ref_24k.wav so ref_24k.wav matches non-empty ref_text word-for-word, rendering 6 to 8 1920x1080 dark-slate PNG slides slides/slide_01.png..slide_0N.png with PIL.ImageDraw + /System/Library/Fonts/Helvetica.ttc + draw.textbbox, synthesizing one single-line symbol-free spoken English paragraph of 130 to 210 chars per slide via mlx_audio.tts.generate.generate_audio(..., ref_audio='ref_24k.wav', ref_text=ref_text, lang_code='en', temperature=0.55, top_p=0.9, repetition_penalty=1.05, output_path='audio', file_prefix=f'seg_{idx:02d}', audio_format='wav', join_audio=True), ASR-verifying EVERY audio/seg_XX.wav with faster_whisper checking overall word_recall >= 0.72 and both first_half_recall >= 0.65 and second_half_recall >= 0.65 with temperature=0.4 retry, writing audio/coherence_report.json, muxing segments/seg_XX.mp4, and concatenating into workspace-root kronos_explained.mp4); then run build_video.py with the matched .venv python3 via run_command and verify kronos_explained.mp4 with /opt/homebrew/bin/ffprobe before summarizing. ",
    "On multi-step modularization, refactoring, styling, or self-testing tasks, write 2 to 4 complete non-empty files per step using write_file <tool_call> blocks (never write 0-byte empty stubs and never paste file contents inside Markdown code fences in prose) so every tool call closes cleanly within max_tokens, and keep iterating with tool calls until every module is created and wired up, inline monolith blocks are extracted, requested visual styles and interactive behaviors are complete, and you have written and executed an automated verification command via run_command (fixing any non-zero exit or assertion failure before summarizing). ",
    "If a tool response reports an error or validation failure, fix it with a tool call immediately before finishing."
);

pub fn build_unified_system_prompt(workdir: &Path) -> String {
    let dir_line = format!("Workspace directory on disk: {}", workdir.display());
    let snap = format_workspace_snapshot(workdir, 12 * 1024);
    if snap.is_empty() {
        format!(
            "{UNIFIED_SYSTEM_PROMPT}\n\n{dir_line}\n\nWorkspace status: currently empty (0 files on disk; use write_file to create new files)."
        )
    } else {
        format!("{UNIFIED_SYSTEM_PROMPT}\n\n{dir_line}\n\n{snap}")
    }
}

fn build_initial_system_prompt(workdir: &Path) -> String {
    format_system_with_workspace(AGENT_SYSTEM_PROMPT, workdir)
}

fn format_workspace_snapshot(workdir: &Path, max_inline_bytes: u64) -> String {
    let entries = list_workspace_files(workdir);
    if entries.is_empty() {
        return String::new();
    }
    let total_bytes: u64 = entries.iter().map(|e| e.bytes).sum();
    if entries.len() <= 12 && total_bytes <= max_inline_bytes {
        let mut snapshots = String::new();
        let mut all_text = true;
        for entry in &entries {
            if let Ok(full) = resolve_jailed_path(workdir, &entry.path) {
                match fs::read_to_string(&full) {
                    Ok(content) => {
                        snapshots.push_str(&format!(
                            "\n--- {} ({} bytes) ---\n{}\n",
                            entry.path, entry.bytes, content
                        ));
                    }
                    Err(_) => {
                        all_text = false;
                        break;
                    }
                }
            }
        }
        if all_text && !snapshots.is_empty() {
            return format!("Existing workspace files on disk:{snapshots}");
        }
    }
    let mut listing = String::from("Existing workspace files on disk (call read_file before targeted edits):\n");
    for entry in entries.iter().take(40) {
        let mark = if entry.validation_ok { "valid" } else { &entry.validation_msg };
        listing.push_str(&format!("- {} ({} bytes, {mark})\n", entry.path, entry.bytes));
    }
    listing
}

fn format_system_with_workspace(base_prompt: &str, workdir: &Path) -> String {
    let snap = format_workspace_snapshot(workdir, 12 * 1024);
    if snap.is_empty() {
        base_prompt.to_string()
    } else {
        format!("{base_prompt}\n\n{snap}")
    }
}

const COMPACTED_MEMORY_HEADER: &str = "[Compacted Session Memory]";

fn summarize_single_line(text: &str, max_chars: usize) -> String {
    let collapsed: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        collapsed
    } else {
        let prefix: String = collapsed.chars().take(max_chars).collect();
        format!("{prefix}...")
    }
}

fn extract_user_turn_digest(content: &str) -> String {
    let mut img_notes = Vec::new();
    let mut plain_lines = Vec::new();
    let mut in_ocr_block = false;
    for line in content.lines() {
        let t = line.trim();
        if t.starts_with("[Attached Image #") {
            img_notes.push(t.to_string());
            in_ocr_block = false;
        } else if t.starts_with("- Dominant CSS palette:") || t.starts_with("- Visual categories:") {
            in_ocr_block = false;
        } else if t.starts_with("- Detected text") {
            in_ocr_block = true;
        } else if in_ocr_block && (line.starts_with("  [") || line.starts_with("  ...")) {
            continue;
        } else {
            in_ocr_block = false;
            if !t.is_empty() {
                plain_lines.push(t);
            }
        }
    }
    let user_req = summarize_single_line(&plain_lines.join(" "), 180);
    if img_notes.is_empty() {
        user_req
    } else if user_req.is_empty() {
        img_notes.join(" ")
    } else {
        format!("{} | {}", img_notes.join(" "), user_req)
    }
}

/// Compacts older agentic turns and oversized intermediate tool payloads while preserving `history[0]`
/// (`system` KV prefix), recent verbatim turns, multimodal image references, and live disk file state.
pub fn compact_messages_with_budget(
    history: &mut Vec<Message>,
    workdir: &Path,
    force: bool,
    threshold_tokens: usize,
    count_fn: impl Fn(&[Message]) -> usize,
) -> CompactionReport {
    let before_tokens = count_fn(history);
    if !force && before_tokens <= threshold_tokens {
        return CompactionReport {
            compacted: false,
            before_tokens,
            after_tokens: before_tokens,
            compacted_messages: 0,
        };
    }

    let sys_offset = usize::from(history.first().is_some_and(|m| m.role == "system"));
    if history.len() <= sys_offset + 1 {
        return CompactionReport {
            compacted: false,
            before_tokens,
            after_tokens: before_tokens,
            compacted_messages: 0,
        };
    }

    // Group non-system messages into genuine user-anchored atomic turns so synthetic intra-turn
    // harness prompts (`[Repair Required`, `[Workspace Progress Checkpoint`) never cause the active
    // user request (`[Device Context]`, `[Attached Image]`) to be compacted away as an older turn.
    let is_synthetic_harness_user = |content: &str| -> bool {
        let t = content.trim_start();
        t.starts_with("[Repair Required")
            || t.starts_with("[Workspace Progress Checkpoint")
            || t.starts_with(COMPACTED_MEMORY_HEADER)
    };
    let mut group_starts: Vec<usize> = Vec::new();
    for (i, m) in history.iter().enumerate().skip(sys_offset) {
        let starts_real_turn = m.role == "user" && !is_synthetic_harness_user(&m.content);
        if starts_real_turn || group_starts.is_empty() {
            group_starts.push(i);
        }
    }

    let total_groups = group_starts.len();
    let ends_with_open_user = history
        .last()
        .is_some_and(|m| m.role == "user" && !is_synthetic_harness_user(&m.content));
    let mut compacted_messages = 0usize;

    let min_groups_for_inter = if force { 2 } else { 3 };
    if total_groups >= min_groups_for_inter {
        let keep_groups = if force {
            1usize
        } else if ends_with_open_user && total_groups >= 3 {
            2usize
        } else {
            1usize
        };
        let split_group = total_groups.saturating_sub(keep_groups);
        if split_group > 0 {
            let split_msg_idx = group_starts[split_group];
            let mut log_bullets: Vec<String> = Vec::new();

            for g in 0..split_group {
                let start = group_starts[g];
                let end = group_starts.get(g + 1).copied().unwrap_or(history.len());
                let slice = &history[start..end];
                if slice.first().is_some_and(|m| m.content.starts_with(COMPACTED_MEMORY_HEADER)) {
                    for line in slice[0].content.lines() {
                        if line.starts_with("- Turn ") || line.starts_with("- Earlier: ") {
                            log_bullets.push(line.to_string());
                        }
                    }
                    continue;
                }

                let req_digest = slice
                    .first()
                    .map(|m| extract_user_turn_digest(&m.content))
                    .unwrap_or_default();
                let mut actions: Vec<String> = Vec::new();
                let mut final_reply = String::new();
                for m in slice.iter().skip(1) {
                    if m.role == "assistant" {
                        if !m.tool_calls.is_empty() {
                            for tc in &m.tool_calls {
                                let target = Json::parse(&tc.arguments)
                                    .ok()
                                    .and_then(|j| {
                                        j.get("path")
                                            .or_else(|| j.get("cmd"))
                                            .or_else(|| j.get("pattern"))
                                            .and_then(Json::as_str)
                                            .map(|s| summarize_single_line(s, 36))
                                    })
                                    .unwrap_or_default();
                                if target.is_empty() {
                                    actions.push(tc.name.clone());
                                } else {
                                    actions.push(format!("{}({target})", tc.name));
                                }
                            }
                        } else if !m.content.trim().is_empty() {
                            final_reply = summarize_single_line(&m.content, 160);
                        }
                    }
                }
                let mut entry = format!("- Earlier: user=\"{req_digest}\"");
                if !actions.is_empty() {
                    let capped: Vec<String> = actions.into_iter().take(8).collect();
                    entry.push_str(&format!(" -> tools=[{}]", capped.join(", ")));
                }
                if !final_reply.is_empty() {
                    entry.push_str(&format!(" -> outcome=\"{final_reply}\""));
                }
                log_bullets.push(entry);
            }

            if log_bullets.len() > 16 {
                let drop_n = log_bullets.len() - 16;
                log_bullets.drain(0..drop_n);
            }

            let mut base_mem = String::from(COMPACTED_MEMORY_HEADER);
            base_mem.push_str("\nPrior session log:\n");
            for b in &log_bullets {
                base_mem.push_str(b);
                base_mem.push('\n');
            }
            let full_snap = format_workspace_snapshot(workdir, 14 * 1024);
            let brief_snap = format_workspace_snapshot(workdir, 0);

            let build_candidate = |snap_str: &str| -> Vec<Message> {
                let mut mem = base_mem.clone();
                if !snap_str.is_empty() {
                    mem.push('\n');
                    mem.push_str(snap_str);
                }
                let mut rebuilt = Vec::with_capacity(sys_offset + 2 + (history.len() - split_msg_idx));
                rebuilt.extend_from_slice(&history[..sys_offset]);
                rebuilt.push(Message::new("user", &mem));
                rebuilt.push(Message::new(
                    "assistant",
                    "Session memory and current workspace files on disk are synchronized.",
                ));
                rebuilt.extend_from_slice(&history[split_msg_idx..]);
                rebuilt
            };

            let cand_full = build_candidate(&full_snap);
            let tok_full = count_fn(&cand_full);
            if tok_full < before_tokens {
                compacted_messages += split_msg_idx.saturating_sub(sys_offset);
                *history = cand_full;
            } else {
                let cand_brief = build_candidate(&brief_snap);
                let tok_brief = count_fn(&cand_brief);
                if tok_brief < before_tokens {
                    compacted_messages += split_msg_idx.saturating_sub(sys_offset);
                    *history = cand_brief;
                }
            }
        }
    }

    // Intra-turn pass: if one long multi-step turn exceeds budget, fold older completed
    // `(assistant(tool_calls), tool(results)...)` step batches into a plain-text progress checkpoint
    // right after the active `user` turn while keeping `history[0..=active_user_idx]` verbatim
    // (preserving the hot KV prefix) and keeping the recent tail tool calls 100% uncorrupted
    // (never rewriting `<tool_call>` JSON arguments to synthetic keys).
    let mid_tokens = count_fn(history);
    if (force && compacted_messages == 0) || mid_tokens > threshold_tokens {
        let active_user_idx = history
            .iter()
            .enumerate()
            .skip(sys_offset)
            .rev()
            .find(|(_, m)| m.role == "user" && !is_synthetic_harness_user(&m.content))
            .map(|(i, _)| i)
            .unwrap_or(sys_offset);
        for protect_tail in [6usize, 2, 0] {
            let raw_stop = history.len().saturating_sub(protect_tail);
            if raw_stop <= active_user_idx + 1 {
                continue;
            }
            // Align `split_idx` to an `assistant` message boundary so `(assistant, tool...)` batches
            // and ChatML role alternation stay intact, backing up to the most recent `assistant`
            // step when `raw_stop` lands inside multi-tool `tool` responses.
            let mut split_idx = raw_stop;
            while split_idx < history.len() && history[split_idx].role != "assistant" {
                split_idx += 1;
            }
            if protect_tail > 0 && split_idx >= history.len() {
                let mut back = raw_stop;
                while back > active_user_idx + 1 && history[back].role != "assistant" {
                    back -= 1;
                }
                if back > active_user_idx + 1 && history[back].role == "assistant" {
                    split_idx = back;
                } else {
                    continue;
                }
            }
            if split_idx <= active_user_idx + 1 {
                continue;
            }
            let mut completed_steps: Vec<String> = Vec::new();
            for m in &history[active_user_idx + 1..split_idx] {
                if m.role == "user" && m.content.starts_with("[Workspace Progress Checkpoint") {
                    for ln in m.content.lines() {
                        if ln.starts_with("- ") && completed_steps.len() < 32 {
                            completed_steps.push(ln.to_string());
                        }
                    }
                } else if m.role == "assistant" && !m.tool_calls.is_empty() {
                    for tc in &m.tool_calls {
                        let target = Json::parse(&tc.arguments)
                            .ok()
                            .and_then(|j| {
                                j.get("path")
                                    .or_else(|| j.get("cmd"))
                                    .or_else(|| j.get("pattern"))
                                    .and_then(Json::as_str)
                                    .map(|s| summarize_single_line(s, 48))
                            })
                            .unwrap_or_default();
                        if target.is_empty() {
                            completed_steps.push(format!("- executed `{}`", tc.name));
                        } else {
                            completed_steps.push(format!("- executed `{}` on `{target}`", tc.name));
                        }
                    }
                }
            }
            let disk_files = summarize_workspace_files_brief(workdir);
            let mut checkpoint_user = String::from(
                "[Workspace Progress Checkpoint — earlier steps in this turn are already applied on disk]\n",
            );
            for s in completed_steps.iter().take(28) {
                checkpoint_user.push_str(s);
                checkpoint_user.push('\n');
            }
            checkpoint_user.push_str(&format!(
                "Current workspace files on disk: {disk_files}\n"
            ));
            for entry in list_workspace_files(workdir) {
                if !entry.validation_ok && entry.bytes <= 8192 {
                    if let Ok(full) = resolve_jailed_path(workdir, &entry.path) {
                        if let Ok(src) = fs::read_to_string(&full) {
                            checkpoint_user.push_str(&format!(
                                "File `{}` on disk currently has validation error ({}):\n{}\n",
                                entry.path, entry.validation_msg, src
                            ));
                        }
                    }
                }
            }
            checkpoint_user.push_str(
                "Continue completing and testing any remaining requirements using valid workspace tool calls (`write_file` requires both `path` and full `content`).",
            );

            let mut rebuilt =
                Vec::with_capacity((active_user_idx + 1) + 2 + (history.len() - split_idx));
            rebuilt.extend_from_slice(&history[..=active_user_idx]);
            rebuilt.push(Message::new(
                "assistant",
                "Earlier workspace files and edits are saved on disk; continuing remaining steps.",
            ));
            rebuilt.push(Message::new("user", &checkpoint_user));
            rebuilt.extend_from_slice(&history[split_idx..]);

            let cand_tokens = count_fn(&rebuilt);
            if cand_tokens + 64 < mid_tokens {
                let folded = split_idx.saturating_sub(active_user_idx + 1);
                compacted_messages += folded;
                *history = rebuilt;
                break;
            }
        }
    }

    let after_tokens = count_fn(history);
    CompactionReport {
        compacted: compacted_messages > 0 && (force || after_tokens + 64 < before_tokens),
        before_tokens,
        after_tokens,
        compacted_messages,
    }
}

pub fn compact_conversation_history(
    model: &ChatModel,
    history: &mut Vec<Message>,
    tools_json: &[String],
    workdir: &Path,
    force: bool,
) -> CompactionReport {
    let ctx = model.context_size();
    // Keep hot KV prefix intact while guaranteeing >= 3,200 tokens of generation headroom on 16k context.
    let threshold = if ctx >= 8192 {
        ((ctx * 80) / 100).min(ctx.saturating_sub(3200))
    } else {
        (ctx * 74) / 100
    };
    compact_messages_with_budget(history, workdir, force, threshold, |msgs| {
        model.count_prompt_tokens(msgs, tools_json)
    })
}

#[cfg(not(target_os = "ios"))]
fn shell_single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

fn normalize_css_color_token(raw: &str) -> Option<String> {
    let v = raw
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    match v.as_str() {
        "#fff" | "#ffffff" | "white" | "rgb(255, 255, 255)" | "rgb(255,255,255)" => {
            Some("#ffffff".to_string())
        }
        "#000" | "#000000" | "black" | "rgb(0, 0, 0)" | "rgb(0,0,0)" => {
            Some("#000000".to_string())
        }
        s if s.starts_with('#') && (s.len() == 4 || s.len() == 7) => {
            if s.len() == 4 {
                let b = s.as_bytes();
                let (r, g, bl) = (b[1] as char, b[2] as char, b[3] as char);
                Some(format!("#{r}{r}{g}{g}{bl}{bl}"))
            } else {
                Some(s.to_string())
            }
        }
        _ => None,
    }
}

fn extract_quoted_calls(src: &str, needle: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in src.lines() {
        let code = line.split("//").next().unwrap_or("");
        let mut cur = 0usize;
        while let Some(pos) = code[cur..].find(needle) {
            let after = cur + pos + needle.len();
            let rest = code[after..].trim_start();
            let Some(q) = rest.chars().next() else {
                break;
            };
            if q == '"' || q == '\'' {
                let body = &rest[1..];
                if let Some(end_q) = body.find(q) {
                    let val = body[..end_q].trim();
                    if !val.is_empty() && !out.iter().any(|x| x == val) {
                        out.push(val.to_string());
                    }
                    cur = after + 1 + end_q + 1;
                    continue;
                }
            }
            cur = after;
        }
    }
    out
}

fn collect_html_ids(html: &str, ids: &mut Vec<String>) {
    let lower = html.to_ascii_lowercase();
    for pat in ["id=\"", "id='"] {
        let q = pat.chars().last().unwrap();
        let mut cur = 0usize;
        while let Some(pos) = lower[cur..].find(pat) {
            let start = cur + pos + pat.len();
            if let Some(end_rel) = html[start..].find(q) {
                let val = html[start..start + end_rel].trim();
                if !val.is_empty() && !ids.iter().any(|x| x == val) {
                    ids.push(val.to_string());
                }
                cur = start + end_rel + 1;
            } else {
                break;
            }
        }
    }
}

fn verify_web_dom_and_css_semantics(workdir: &Path) -> Result<Option<String>, String> {
    let entries = list_workspace_files(workdir);
    let mut html_files = Vec::new();
    let mut js_files = Vec::new();
    let mut css_files = Vec::new();
    for e in &entries {
        let lower = e.path.to_ascii_lowercase();
        if lower.ends_with(".html") || lower.ends_with(".htm") {
            html_files.push(e.path.clone());
        } else if lower.ends_with(".js") || lower.ends_with(".mjs") {
            js_files.push(e.path.clone());
        } else if lower.ends_with(".css") {
            css_files.push(e.path.clone());
        }
    }
    if html_files.is_empty() && css_files.is_empty() {
        return Ok(None);
    }

    let mut issues = Vec::new();
    let mut defined_ids = Vec::new();
    let mut has_dynamic_template_ids = false;
    for hp in &html_files {
        if let Ok(full) = resolve_jailed_path(workdir, hp) {
            if let Ok(src) = fs::read_to_string(&full) {
                collect_html_ids(&src, &mut defined_ids);
            }
        }
    }
    for jp in &js_files {
        if let Ok(full) = resolve_jailed_path(workdir, jp) {
            if let Ok(src) = fs::read_to_string(&full) {
                collect_html_ids(&src, &mut defined_ids);
                if src.contains(".id =") || src.contains(".id=") || src.contains("setAttribute('id'") || src.contains("setAttribute(\"id\"") {
                    has_dynamic_template_ids = true;
                }
            }
        }
    }

    if !html_files.is_empty() && !has_dynamic_template_ids {
        for jp in &js_files {
            if let Ok(full) = resolve_jailed_path(workdir, jp) {
                if let Ok(src) = fs::read_to_string(&full) {
                    for id_ref in extract_quoted_calls(&src, "getElementById(") {
                        if !defined_ids.iter().any(|d| d == &id_ref) {
                            issues.push(format!(
                                "`{jp}`: `getElementById(\"{id_ref}\")` has no matching `id=\"{id_ref}\"` in workspace HTML"
                            ));
                        }
                    }
                    for sel_ref in extract_quoted_calls(&src, "querySelector(") {
                        if let Some(id_only) = sel_ref.strip_prefix('#') {
                            if !id_only.is_empty()
                                && id_only.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
                                && !defined_ids.iter().any(|d| d == id_only)
                            {
                                issues.push(format!(
                                    "`{jp}`: `querySelector(\"#{id_only}\")` has no matching `id=\"{id_only}\"` in workspace HTML"
                                ));
                            }
                        }
                    }
                }
            }
        }
    }

    for cp in &css_files {
        if let Ok(full) = resolve_jailed_path(workdir, cp) {
            if let Ok(src) = fs::read_to_string(&full) {
                for block in src.split('{').skip(1) {
                    let body = block.split('}').next().unwrap_or("");
                    let mut fg: Option<String> = None;
                    let mut bg: Option<String> = None;
                    for decl in body.split(';') {
                        if let Some((prop, val)) = decl.split_once(':') {
                            let p = prop.trim().to_ascii_lowercase();
                            if p == "color" {
                                fg = normalize_css_color_token(val);
                            } else if p == "background" || p == "background-color" {
                                bg = normalize_css_color_token(val);
                            }
                        }
                    }
                    if let (Some(f), Some(b)) = (fg, bg) {
                        if f == b {
                            issues.push(format!(
                                "`{cp}`: CSS rule sets identical foreground `color` and `background` ({f}), making text invisible"
                            ));
                        }
                    }
                }
            }
        }
    }

    if issues.is_empty() {
        Ok(Some(format!(
            "DOM/CSS cross-check ok ({} HTML, {} CSS, {} JS)",
            html_files.len(),
            css_files.len(),
            js_files.len()
        )))
    } else {
        Err(issues.join("\n"))
    }
}

fn extract_relative_js_specifiers(src: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw_line in src.lines() {
        let line = raw_line.trim();
        if line.starts_with("//") || line.starts_with("/*") || line.starts_with('*') {
            continue;
        }
        if (line.starts_with("import ") || line.starts_with("export ")) && line.contains(" from ") {
            if let Some((_, rhs)) = line.rsplit_once(" from ") {
                if let Some(spec) = first_quoted_literal(rhs.trim()) {
                    if spec.starts_with("./") || spec.starts_with("../") {
                        out.push(spec);
                    }
                }
            }
        } else if line.starts_with("import ") {
            if let Some(spec) = first_quoted_literal(line.trim_start_matches("import ").trim()) {
                if spec.starts_with("./") || spec.starts_with("../") {
                    out.push(spec);
                }
            }
        }
        let mut rest = line;
        while let Some(idx) = rest.find("require(") {
            let after = &rest[idx + "require(".len()..];
            if let Some(spec) = first_quoted_literal(after.trim_start()) {
                if spec.starts_with("./") || spec.starts_with("../") {
                    out.push(spec);
                }
            }
            rest = after;
        }
    }
    out
}

fn first_quoted_literal(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let q = *bytes.first()?;
    if q != b'\'' && q != b'"' {
        return None;
    }
    let mut end = 1usize;
    while end < bytes.len() {
        if bytes[end] == b'\\' {
            end += 2;
            continue;
        }
        if bytes[end] == q {
            return Some(s[1..end].to_string());
        }
        end += 1;
    }
    None
}

fn resolve_relative_module_exists(base_dir: &Path, spec: &str) -> bool {
    let clean = spec.split(['?', '#']).next().unwrap_or(spec);
    let candidate = base_dir.join(clean);
    if candidate.is_file() {
        return true;
    }
    for ext in [".js", ".mjs", ".ts", ".tsx", ".jsx", ".json", ".css"] {
        let with_ext = PathBuf::from(format!("{}{ext}", candidate.display()));
        if with_ext.is_file() {
            return true;
        }
    }
    for idx in ["index.js", "index.mjs", "index.ts", "index.tsx"] {
        if candidate.join(idx).is_file() {
            return true;
        }
    }
    false
}

fn detect_truncation_stub_line(ext: &str, src: &str) -> Option<(usize, String)> {
    let stub_phrases = [
        "rest of code",
        "rest of the code",
        "rest of implementation",
        "rest of file",
        "existing code here",
        "remaining code here",
        "omitted for brevity",
    ];
    for (idx, raw) in src.lines().enumerate() {
        let trimmed = raw.trim();
        let lower = trimmed.to_ascii_lowercase();
        let is_comment = trimmed.starts_with("//")
            || trimmed.starts_with("/*")
            || trimmed.starts_with('#')
            || trimmed.starts_with("<!--");
        if is_comment && stub_phrases.iter().any(|p| lower.contains(p)) {
            return Some((idx + 1, truncate_utf8(trimmed, 80)));
        }
    }
    if matches!(ext, "html" | "htm") {
        let lower = src.to_ascii_lowercase();
        if let Some(b_start) = lower.find("<body") {
            if let Some(gt) = lower[b_start..].find('>') {
                let body_after = &src[b_start + gt + 1..];
                let body_inner = match lower[b_start + gt + 1..].find("</body>") {
                    Some(end) => &body_after[..end],
                    None => body_after,
                };
                if body_inner.chars().filter(|c| !c.is_whitespace()).count() < 12 {
                    return Some((1, "empty <body> placeholder".to_string()));
                }
            }
        }
    }
    None
}

fn verify_local_imports_and_anti_stubs(
    workdir: &Path,
    modified_files: &[String],
) -> Result<Option<String>, String> {
    let mut issues = Vec::new();
    let mut checked = 0usize;
    for rel in modified_files {
        let Ok(full) = resolve_jailed_path(workdir, rel) else {
            continue;
        };
        if !full.is_file() {
            continue;
        }
        let ext = full
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !matches!(
            ext.as_str(),
            "js" | "mjs" | "ts" | "tsx" | "jsx" | "rs" | "py" | "html" | "htm" | "css"
        ) {
            continue;
        }
        let Ok(src) = fs::read_to_string(&full) else {
            continue;
        };
        checked += 1;
        if let Some((ln, snippet)) = detect_truncation_stub_line(&ext, &src) {
            issues.push(format!(
                "`{rel}` line {ln}: incomplete truncation placeholder (`{snippet}`)"
            ));
        }
        let parent = full.parent().unwrap_or(workdir);
        if matches!(ext.as_str(), "js" | "mjs" | "ts" | "tsx" | "jsx") {
            for spec in extract_relative_js_specifiers(&src) {
                if !resolve_relative_module_exists(parent, &spec) {
                    issues.push(format!(
                        "`{rel}`: unresolved local module import `{spec}`"
                    ));
                }
            }
        } else if ext == "rs" {
            for (idx, raw) in src.lines().enumerate() {
                let t = raw.trim();
                let mod_rest = t
                    .strip_prefix("pub mod ")
                    .or_else(|| t.strip_prefix("pub(crate) mod "))
                    .or_else(|| t.strip_prefix("mod "));
                if let Some(rest) = mod_rest {
                    if let Some(name) = rest.strip_suffix(';') {
                        let clean = name.trim();
                        if !clean.is_empty()
                            && clean.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                            && !parent.join(format!("{clean}.rs")).is_file()
                            && !parent.join(clean).join("mod.rs").is_file()
                        {
                            issues.push(format!(
                                "`{rel}` line {}: `mod {clean};` has no `{clean}.rs` or `{clean}/mod.rs`",
                                idx + 1
                            ));
                        }
                    }
                }
            }
        }
    }
    if !issues.is_empty() {
        Err(issues.join("\n"))
    } else if checked > 0 {
        Ok(Some(format!("imports & completeness ok ({checked} file(s))")))
    } else {
        Ok(None)
    }
}

#[cfg(not(target_os = "ios"))]
fn verify_web_headless_dom_runtime(
    workdir: &Path,
    modified_files: &[String],
) -> Result<Option<String>, String> {
    let touched_web = modified_files.iter().any(|p| {
        let l = p.to_ascii_lowercase();
        l.ends_with(".html") || l.ends_with(".htm") || l.ends_with(".js")
    });
    if !touched_web {
        return Ok(None);
    }
    let html_rel = if workdir.join("index.html").is_file() {
        "index.html".to_string()
    } else {
        match list_workspace_files(workdir)
            .into_iter()
            .find(|e| e.path.to_ascii_lowercase().ends_with(".html"))
        {
            Some(e) => e.path,
            None => return Ok(None),
        }
    };

    let harness = concat!(
        "const fs=require('fs'),path=require('path'),vm=require('vm'),{pathToFileURL}=require('url');",
        "const realErr=s=>process.stderr.write(String(s)+'\\n');",
        "const htmlPath=process.argv[1];",
        "const html=fs.readFileSync(htmlPath,'utf8');",
        "const base=path.dirname(path.resolve(htmlPath));",
        "const re=/<script\\b([^>]*)\\bsrc\\s*=\\s*[\"']([^\"']+)[\"'][^>]*>/gi;",
        "const scripts=[],modules=[];let m;",
        "while((m=re.exec(html))!==null){",
        "const attrs=(m[1]||'').toLowerCase();const src=m[2].split(/[?#]/)[0];",
        "if(/^(https?:|\\/\\/)/i.test(src))continue;",
        "const fp=path.resolve(base,src);",
        "if(!fs.existsSync(fp)||!fs.statSync(fp).isFile())continue;",
        "const code=fs.readFileSync(fp,'utf8');",
        "if(attrs.includes('module')||/^\\s*(import|export)\\b/m.test(code))modules.push([src,fp,code]);",
        "else scripts.push([src,code]);",
        "}",
        "if(!scripts.length&&!modules.length)process.exit(0);",
        "function mkEl(){let p;const fn=function(){return p;};",
        "p=new Proxy(fn,{get(_t,k){",
        "if(k===Symbol.toPrimitive)return()=>'';",
        "if(k===Symbol.iterator)return function*(){};",
        "if(k==='length')return 0;",
        "if(k==='checked'||k==='disabled')return false;",
        "if(k==='value'||k==='textContent'||k==='innerHTML'||k==='id'||k==='className')return '';",
        "if(k==='getAttribute'||k==='closest')return()=>null;",
        "if(k==='matches')return()=>false;",
        "if(k==='dataset'||k==='style')return new Proxy({},{get:()=>'',set:()=>true});",
        "if(k==='classList')return{add(){},remove(){},toggle(){return false;},contains(){return false;}};",
        "if(k==='getBoundingClientRect')return()=>({width:800,height:600,top:0,left:0,right:800,bottom:600});",
        "if(k==='querySelectorAll'||k==='getElementsByClassName'||k==='getElementsByTagName'||k==='children')return[];",
        "return p;},set(){return true;},apply(){return p;}});return p;}",
        "const rootEl=mkEl();const store=new Map();",
        "const storage={getItem:k=>store.has(k)?store.get(k):null,setItem:(k,v)=>{store.set(String(k),String(v));},removeItem:k=>store.delete(k),clear:()=>store.clear()};",
        "const cbs=[];",
        "const addEv=(ev,cb)=>{if((ev==='DOMContentLoaded'||ev==='load')&&typeof cb==='function')cbs.push(cb);};",
        "const doc=new Proxy({body:rootEl,documentElement:rootEl,head:rootEl,readyState:'complete',",
        "getElementById:()=>mkEl(),querySelector:()=>mkEl(),querySelectorAll:()=>[],createElement:()=>mkEl(),createTextNode:()=>mkEl(),createDocumentFragment:()=>mkEl(),",
        "addEventListener:addEv,removeEventListener(){}},",
        "{get(t,k){return k in t?t[k]:mkEl();},set(t,k,v){t[k]=v;return true;}});",
        "const ctx={console:{log(){},warn(){},error(){},info(){}},setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},",
        "requestAnimationFrame:()=>1,cancelAnimationFrame(){},fetch:()=>Promise.resolve({ok:true,json:async()=>({}),text:async()=>''}),",
        "localStorage:storage,sessionStorage:storage,navigator:{clipboard:{writeText:async()=>{}},userAgent:'KronosCheck'},",
        "location:{href:'http://127.0.0.1/',pathname:'/',search:'',hash:'',reload(){}},history:{pushState(){},replaceState(){}},",
        "matchMedia:()=>({matches:false,addEventListener(){},removeEventListener(){}}),",
        "addEventListener:addEv,removeEventListener(){},",
        "Event:class{constructor(t){this.type=t;}preventDefault(){}stopPropagation(){}},",
        "CustomEvent:class{constructor(t,o){this.type=t;this.detail=o&&o.detail;}},",
        "IntersectionObserver:class{observe(){}unobserve(){}disconnect(){}},",
        "ResizeObserver:class{observe(){}unobserve(){}disconnect(){}},",
        "MutationObserver:class{observe(){}disconnect(){}},",
        "alert(){},confirm:()=>true,prompt:()=>'',document:doc};",
        "ctx.window=ctx;ctx.globalThis=ctx;ctx.self=ctx;",
        "const box=vm.createContext(ctx);",
        "for(const[name,code]of scripts){",
        "vm.runInContext(code,box,{filename:name,timeout:800});",
        "}",
        "(async()=>{",
        "if(modules.length){",
        "for(const[k,v]of Object.entries(ctx)){if(k==='globalThis')continue;try{Object.defineProperty(globalThis,k,{value:v,configurable:true,writable:true});}catch(_){}}",
        "globalThis.window=globalThis;globalThis.self=globalThis;",
        "for(const[_name,fp,code]of modules){",
        "if(/\\bfrom\\s+['\"]https?:\\/\\//i.test(code))continue;",
        "await import(pathToFileURL(fp).href);",
        "}",
        "}",
        "for(const cb of cbs){cb({type:'DOMContentLoaded',target:doc,preventDefault(){}});}",
        "})().catch(e=>{realErr(e&&e.stack||e);process.exit(1);});"
    );
    let cmd = format!(
        "if command -v node >/dev/null 2>&1; then node -e {} {}; fi",
        shell_single_quote(harness),
        shell_single_quote(&html_rel)
    );
    let out = run_shell_command_with_timeout(workdir, &cmd, 5, 9500);
    if !out.starts_with("exit_code=0") {
        Err(format!(
            "`{html_rel}` headless DOM runtime smoke test failed:\n{out}"
        ))
    } else {
        Ok(Some(format!("headless DOM runtime ok ({html_rel})")))
    }
}

/// Runs post-edit structural, cross-file DOM/CSS, and runtime compiler smoke checks on the workspace.
pub fn verify_workspace_work(
    workdir: &Path,
    modified_files: &[String],
    timeout_secs: u64,
) -> Result<VerificationReport, String> {
    let t0 = Instant::now();
    let mut details = Vec::new();
    let mut errors = Vec::new();

    match validate_workspace(workdir) {
        Ok(()) => details.push("syntax & local asset integrity ok".to_string()),
        Err(e) => errors.push(e),
    }

    match verify_local_imports_and_anti_stubs(workdir, modified_files) {
        Ok(Some(note)) => details.push(note),
        Ok(None) => {}
        Err(e) => errors.push(e),
    }

    match verify_web_dom_and_css_semantics(workdir) {
        Ok(Some(note)) => details.push(note),
        Ok(None) => {}
        Err(e) => errors.push(e),
    }

    #[cfg(not(target_os = "ios"))]
    {
        let mut js_checked = 0usize;
        let mut py_checked = 0usize;
        let mut sh_checked = 0usize;
        let mut check_rust = false;

        for rel in modified_files {
            let lower = rel.to_ascii_lowercase();
            let Ok(full) = resolve_jailed_path(workdir, rel) else {
                continue;
            };
            if !full.is_file() {
                continue;
            }
            if lower.ends_with(".js") || lower.ends_with(".mjs") {
                let cmd = format!(
                    "if command -v node >/dev/null 2>&1; then node --check {}; fi",
                    shell_single_quote(rel)
                );
                let out = run_shell_command_with_timeout(workdir, &cmd, 5, 9100 + js_checked);
                if !out.starts_with("exit_code=0") {
                    errors.push(format!("`{rel}` V8 syntax check failed:\n{out}"));
                } else {
                    js_checked += 1;
                }
            } else if lower.ends_with(".py") {
                let cmd = format!(
                    "if command -v python3 >/dev/null 2>&1; then python3 -m py_compile {}; fi",
                    shell_single_quote(rel)
                );
                let out = run_shell_command_with_timeout(workdir, &cmd, 5, 9200 + py_checked);
                if !out.starts_with("exit_code=0") {
                    errors.push(format!("`{rel}` python3 compile check failed:\n{out}"));
                } else {
                    py_checked += 1;
                }
            } else if lower.ends_with(".sh") {
                let cmd = format!("sh -n {}", shell_single_quote(rel));
                let out = run_shell_command_with_timeout(workdir, &cmd, 5, 9300 + sh_checked);
                if !out.starts_with("exit_code=0") {
                    errors.push(format!("`{rel}` shell syntax check failed:\n{out}"));
                } else {
                    sh_checked += 1;
                }
            } else if lower.ends_with(".rs") || lower == "cargo.toml" {
                check_rust = true;
            }
        }

        if js_checked > 0 {
            details.push(format!("V8 `node --check` passed ({js_checked} file(s))"));
        }
        if errors.is_empty() {
            match verify_web_headless_dom_runtime(workdir, modified_files) {
                Ok(Some(note)) => details.push(note),
                Ok(None) => {}
                Err(e) => errors.push(e),
            }
        }
        if py_checked > 0 {
            details.push(format!("`python3 -m py_compile` passed ({py_checked} file(s))"));
        }
        if sh_checked > 0 {
            details.push(format!("`sh -n` passed ({sh_checked} file(s))"));
        }
        if check_rust && workdir.join("Cargo.toml").is_file() {
            let out = run_shell_command_with_timeout(
                workdir,
                "if command -v cargo >/dev/null 2>&1; then cargo check --quiet; fi",
                timeout_secs.clamp(5, 30),
                9400,
            );
            if !out.starts_with("exit_code=0") {
                errors.push(format!("`cargo check` failed:\n{out}"));
            } else {
                details.push("`cargo check` passed".to_string());
            }
        }
    }
    #[cfg(target_os = "ios")]
    let _ = timeout_secs;

    if !errors.is_empty() {
        return Err(errors.join("\n"));
    }

    let checks_run = details.len().max(1);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let summary = format!(
        "{checks_run} check(s) passed ({})",
        details.join(" · ")
    );
    Ok(VerificationReport {
        checks_run,
        summary,
        details,
        elapsed_ms,
    })
}

#[derive(Clone, Debug)]
struct WorkspaceCheckpoint {
    workdir: PathBuf,
    entries: Vec<(PathBuf, Option<Vec<u8>>)>,
    oracle_ok: bool,
    score: f32,
}

impl WorkspaceCheckpoint {
    fn capture(
        workdir: &Path,
        tracked_rel_paths: &[String],
        oracle_ok: bool,
        score: f32,
    ) -> Self {
        let mut entries = Vec::with_capacity(tracked_rel_paths.len());
        for rel in tracked_rel_paths {
            if let Ok(full) = resolve_jailed_path(workdir, rel) {
                let data = if full.is_file() {
                    fs::read(&full).ok()
                } else {
                    None
                };
                entries.push((full, data));
            }
        }
        Self {
            workdir: workdir.to_path_buf(),
            entries,
            oracle_ok,
            score,
        }
    }

    fn restore(&self) {
        for (full, maybe_bytes) in &self.entries {
            match maybe_bytes {
                Some(bytes) => {
                    if let Some(parent) = full.parent() {
                        let _ = fs::create_dir_all(parent);
                    }
                    let _ = fs::write(full, bytes);
                }
                None => {
                    if full.is_file() {
                        let _ = fs::remove_file(full);
                    }
                }
            }
        }
    }
}

fn strip_injected_prompt_headers(user_content: &str) -> &str {
    let mut cur = user_content.trim();
    loop {
        let mut stripped_any = false;
        for (marker, closer) in [
            ("[Device Context", Some("[/Device Context]")),
            ("[Workspace Context Snapshot", Some("[/Workspace Context Snapshot]")),
            ("[Attached Image", Some("[/Attached Image]")),
            ("[Attached ", None),
            ("[Local Conversations", None),
        ] {
            if cur.starts_with(marker) {
                if let Some(close_tag) = closer {
                    if let Some(end) = cur.find(close_tag) {
                        cur = cur[end + close_tag.len()..].trim();
                        stripped_any = true;
                        continue;
                    }
                }
                if let Some(idx) = cur.find("\n\n") {
                    cur = cur[idx + 2..].trim();
                    stripped_any = true;
                }
            }
        }
        if !stripped_any {
            break;
        }
    }
    cur
}

fn append_repair_directive_to_history(
    history: &mut Vec<Message>,
    assistant_text: &str,
    directive: &str,
) {
    // Replace any immediately preceding stalled repair pair in-place so repeated repair prompts
    // never stack duplicate assistant turns into few-shot history while keeping strict
    // `user/tool -> assistant -> user` ChatML alternation outside `<tool_response>`.
    if history.len() >= 2
        && history[history.len() - 1].role == "user"
        && history[history.len() - 1]
            .content
            .starts_with("[Repair Required")
        && history[history.len() - 2].role == "assistant"
        && history[history.len() - 2].tool_calls.is_empty()
    {
        history.truncate(history.len() - 2);
    }

    let first_clean_line = assistant_text
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with("```") && !l.starts_with('<'))
        .unwrap_or("Inspecting remaining workspace verification items.");
    let mut brief = String::new();
    for ch in first_clean_line.chars().take(140) {
        brief.push(ch);
    }
    if history
        .last()
        .is_some_and(|m| m.role == "user" || m.role == "tool")
    {
        history.push(Message::new("assistant", &brief));
    }
    history.push(Message::new("user", directive));
}

fn summarize_workspace_files_brief(workdir: &Path) -> String {
    let files: Vec<String> = list_workspace_files(workdir)
        .into_iter()
        .filter(|e| !e.path.starts_with(".kronos_media/"))
        .take(24)
        .map(|e| format!("{} ({} B)", e.path, e.bytes))
        .collect();
    if files.is_empty() {
        "none yet".to_string()
    } else {
        files.join(", ")
    }
}

fn is_verification_or_test_command(cmd: &str) -> bool {
    let lower = cmd.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return false;
    }
    let active_runners = [
        "node ",
        "python",
        "pytest",
        "cargo ",
        "npm ",
        "npx ",
        "bun ",
        "deno ",
        "sh ",
        "bash ",
        "zsh ",
        "./",
        "curl ",
        "google chrome",
        "playwright",
        "puppeteer",
        "test",
        "verify",
        "check",
    ];
    active_runners.iter().any(|r| lower.contains(r))
}

fn exec_spawn_agent(
    cur_workdir: &Path,
    raw_args: &str,
    spawner: Option<&SubAgentSpawner>,
) -> ToolExecOutcome {
    let Some(spawn_fn) = spawner else {
        return err_outcome("error: `spawn_agent` is only available inside the multi-task server runtime");
    };
    let args = match Json::parse(raw_args) {
        Ok(j) => j,
        Err(e) => return err_outcome(&format!("error parsing `spawn_agent` JSON arguments: {e}")),
    };
    let Some(raw_wd) = args.get("workdir").and_then(Json::as_str) else {
        return err_outcome("error: `spawn_agent` requires string argument `workdir`");
    };
    let Some(task) = args.get("task").and_then(Json::as_str) else {
        return err_outcome("error: `spawn_agent` requires string argument `task`");
    };
    if task.trim().is_empty() {
        return err_outcome("error: `spawn_agent` `task` prompt cannot be empty");
    }
    let label = args.get("label").and_then(Json::as_str).unwrap_or("");
    let wait = args.get("wait").and_then(Json::as_bool).unwrap_or(false);
    let (canon_wd, _) = match resolve_safe_project_workdir(cur_workdir, raw_wd, None) {
        Ok(v) => v,
        Err(e) => return err_outcome(&e),
    };
    match spawn_fn(&canon_wd, label, task.trim(), wait) {
        Ok(msg) => ok_outcome(msg),
        Err(e) => err_outcome(&format!("error spawning worker agent: {e}")),
    }
}

fn extract_tail_lines(text: &str, max_lines: usize, max_bytes: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let start = lines.len().saturating_sub(max_lines);
    let joined = lines[start..].join("\n");
    if joined.len() <= max_bytes {
        joined
    } else {
        let mut cut = joined.len().saturating_sub(max_bytes);
        while cut < joined.len() && !joined.is_char_boundary(cut) {
            cut += 1;
        }
        format!("...{}", &joined[cut..])
    }
}

/// Runs an iterative tool-calling and Jev-verified monotonic repair loop while preserving the KV-cache prefix.
pub fn run_agent_turn(
    model: &mut ChatModel,
    history: &mut Vec<Message>,
    tools_json: &[String],
    cfg: &AgentTurnConfig,
    on_event: impl FnMut(AgentEvent<'_>) -> bool,
) -> Result<String, String> {
    let mut slot = Some(model);
    run_agent_turn_inner(
        |f| {
            if let Some(ref mut m) = slot {
                f(m);
            }
        },
        history,
        tools_json,
        cfg,
        None,
        on_event,
    )
}

/// Runs `run_agent_turn` acquiring `model_lock` strictly per forward-pass step and releasing it
/// during file I/O, shell execution, and workspace verification so multiple concurrent Kronos
/// worker agents interleave GPU inference and OS tool execution without blocking one another.
pub fn run_agent_turn_shared_with_spawner(
    model_lock: &std::sync::Mutex<ChatModel>,
    history: &mut Vec<Message>,
    tools_json: &[String],
    cfg: &AgentTurnConfig,
    spawner: Option<&SubAgentSpawner>,
    on_event: impl FnMut(AgentEvent<'_>) -> bool,
) -> Result<String, String> {
    run_agent_turn_inner(
        |f| {
            let mut guard = model_lock.lock().unwrap_or_else(|p| p.into_inner());
            f(&mut guard);
        },
        history,
        tools_json,
        cfg,
        spawner,
        on_event,
    )
}

fn run_agent_turn_inner(
    mut with_model: impl FnMut(&mut dyn FnMut(&mut ChatModel)),
    history: &mut Vec<Message>,
    tools_json: &[String],
    cfg: &AgentTurnConfig,
    spawner: Option<&SubAgentSpawner>,
    mut on_event: impl FnMut(AgentEvent<'_>) -> bool,
) -> Result<String, String> {
    let mut active_workdir = cfg.workdir.clone();
    if history.first().is_none_or(|m| m.role != "system") {
        let sys_prompt = build_initial_system_prompt(&active_workdir);
        history.insert(0, Message::new("system", &sys_prompt));
    }
    let sys_prompt_snapshot = history
        .first()
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let raw_last_user = history
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.clone())
        .unwrap_or_default();
    let clean_user_goal = strip_injected_prompt_headers(&raw_last_user).to_string();
    let recent_has_prefetch = raw_last_user.contains("[Workspace Context Snapshot");
    let history_used_tools = history
        .iter()
        .any(|m| m.role == "tool" || !m.tool_calls.is_empty());
    let has_images = raw_last_user.contains("[Attached Image")
        || raw_last_user.contains("[Image Attachment")
        || raw_last_user.contains(".kronos_media/");
    let allow_computer = crate::device::global_device_manager()
        .current_config()
        .allow_computer_control;
    let forced_route = if tools_json.is_empty() {
        Some(TurnRoute::Chat)
    } else if tools_json.len() == 1 && tools_json[0].contains("\"use_computer\"") {
        Some(TurnRoute::Computer)
    } else if cfg.expect_tool_call {
        Some(TurnRoute::Agent)
    } else {
        None
    };

    let mut jev_plan_opt = None;
    with_model(&mut |model| {
        jev_plan_opt = Some(crate::jev::plan_turn_with_jev(
            Some(model),
            &sys_prompt_snapshot,
            &clean_user_goal,
            forced_route,
            has_images,
            &active_workdir,
            history_used_tools,
            tools_json,
            allow_computer,
            recent_has_prefetch,
        ));
    });
    let jev_plan = jev_plan_opt.ok_or_else(|| "model unavailable".to_string())?;
    if !on_event(AgentEvent::JevDecision { plan: &jev_plan }) {
        return Err("cancelled".to_string());
    }
    if let Some(ref block) = jev_plan.prefetch_block {
        if let Some(last_u) = history.iter_mut().rev().find(|m| m.role == "user") {
            last_u.content = format!("{block}\n\n{}", last_u.content);
        }
    }
    let active_tools: &[String] = if tools_json.is_empty() {
        tools_json
    } else {
        &jev_plan.active_tools_json
    };

    const MAX_CONSECUTIVE_NO_TOOL_STEPS: usize = 5;
    let t_start = Instant::now();
    let mut touched_files = false;
    let mut used_any_tools = false;
    let mut repair_attempts = 0usize;
    let mut consecutive_no_tool_steps = 0usize;
    let mut consecutive_edit_failures = 0usize;
    let max_repairs = (cfg.max_steps * 3 / 4).max(8);
    let mut action_seq = 0usize;
    let mut last_mutation_seq = 0usize;
    let mut last_passing_test_seq = 0usize;
    let mut unresolved_edits: Vec<(String, String)> = Vec::new();
    let mut modified_files: Vec<String> = Vec::new();
    let mut tracked_files: Vec<String> = Vec::new();
    let mut best_checkpoint: Option<WorkspaceCheckpoint> = None;
    let mut last_failed_cmd: Option<(usize, String, String)> = None;

    for step in 1..=cfg.max_steps {
        let mut comp_rep_opt = None;
        with_model(&mut |model| {
            comp_rep_opt = Some(compact_conversation_history(
                model,
                history,
                active_tools,
                &active_workdir,
                false,
            ));
        });
        let comp_rep = comp_rep_opt.ok_or_else(|| "model unavailable".to_string())?;
        if comp_rep.compacted
            && !on_event(AgentEvent::ContextCompacted {
                before_tokens: comp_rep.before_tokens,
                after_tokens: comp_rep.after_tokens,
                compacted_messages: comp_rep.compacted_messages,
            })
        {
            return Err("cancelled".to_string());
        }

        if !on_event(AgentEvent::StepStart {
            step,
            max_steps: cfg.max_steps,
        }) {
            return Err("cancelled".to_string());
        }

        let mut filter = ProseStreamFilter::new();
        let mut alive = true;
        let mut aborted_fence_prose = false;
        let mut prose_probe = String::new();
        let mut last_prog_bytes = 0usize;
        let mut last_prog_target = String::new();
        let mut step_params = cfg.params.clone();
        let force_tool_prefix = if !active_tools.is_empty()
            && ((!used_any_tools && cfg.expect_tool_call)
                || consecutive_no_tool_steps > 0
                || !unresolved_edits.is_empty())
        {
            "<tool_call>\n{\"name\": \""
        } else {
            ""
        };
        if consecutive_no_tool_steps > 0 || !unresolved_edits.is_empty() || consecutive_edit_failures > 0 {
            let retry_ord = consecutive_no_tool_steps.max(consecutive_edit_failures).max(1);
            let max_temp = if force_tool_prefix.is_empty() {
                0.65
            } else if consecutive_edit_failures >= 2 {
                0.34
            } else {
                0.22
            };
            let escalated_temp = (0.12 + 0.06 * (retry_ord as f32)).min(max_temp);
            step_params.temperature = step_params.temperature.max(escalated_temp);
            step_params.top_p = step_params.top_p.min(0.92);
            step_params.repeat_penalty = step_params.repeat_penalty.max(
                if retry_ord >= 2 {
                    1.04
                } else {
                    1.02
                },
            );
            step_params.seed = cfg
                .params
                .seed
                .wrapping_add(
                    (repair_attempts as u64).wrapping_mul(7919)
                        + (consecutive_edit_failures as u64).wrapping_mul(1009)
                        + (step as u64).wrapping_mul(131),
                );
        }
        let guard_fence_stall = !active_tools.is_empty()
            && force_tool_prefix.is_empty()
            && (cfg.expect_tool_call
                || !unresolved_edits.is_empty()
                || matches!(jev_plan.route, TurnRoute::Agent));
        let mut comp_res: Option<Result<Completion, String>> = None;
        with_model(&mut |model| {
            comp_res = Some(model.generate_with_tools_prefixed(
                history,
                active_tools,
                force_tool_prefix,
                &step_params,
                |piece| {
                    alive = filter.push(piece, |prose| {
                        if guard_fence_stall && prose_probe.len() < 1400 {
                            prose_probe.push_str(prose);
                        }
                        on_event(AgentEvent::TextDelta(prose))
                    });
                    if alive {
                        if let Some((t_name, t_target, bytes)) = filter.active_tool_progress() {
                            let target_ready = !t_target.is_empty() && t_target != last_prog_target;
                            if target_ready || bytes >= last_prog_bytes + 96 {
                                last_prog_bytes = bytes;
                                if !t_target.is_empty() {
                                    last_prog_target.clone_from(&t_target);
                                }
                                alive = on_event(AgentEvent::ToolProgress {
                                    step,
                                    name: &t_name,
                                    target: &t_target,
                                    bytes,
                                });
                            }
                        } else if guard_fence_stall
                            && last_prog_bytes == 0
                            && prose_probe.len() >= 680
                            && prose_probe.contains("```")
                        {
                            aborted_fence_prose = true;
                            return false;
                        }
                    }
                    alive
                },
            ));
        });
        let mut comp = comp_res.ok_or_else(|| "model unavailable".to_string())??;
        if alive && !aborted_fence_prose {
            let _ = filter.finish(|prose| on_event(AgentEvent::TextDelta(prose)));
        }
        if !on_event(AgentEvent::StepStats {
            step,
            completion: &comp,
        }) {
            return Err("cancelled".to_string());
        }

        if comp.tool_calls.is_empty() {
            let can_repair = repair_attempts < max_repairs
                && consecutive_no_tool_steps < MAX_CONSECUTIVE_NO_TOOL_STEPS
                && step < cfg.max_steps;
            let had_fence = aborted_fence_prose
                || comp.text.contains("```")
                || comp.text.contains("<tool_call>");
            if !unresolved_edits.is_empty() && can_repair {
                repair_attempts += 1;
                consecutive_no_tool_steps += 1;
                let report = unresolved_edits
                    .iter()
                    .map(|(p, err)| {
                        if p.is_empty() {
                            err.clone()
                        } else {
                            format!("{p}: {err}")
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                unresolved_edits.clear();
                if !on_event(AgentEvent::ValidationRepair {
                    step,
                    report: &report,
                }) {
                    return Err("cancelled".to_string());
                }
                let fence_note = if had_fence {
                    "Markdown code fences in prose do NOT write files on disk. "
                } else {
                    ""
                };
                let html_hint = if report.contains("unexpected closing `</style>`")
                    || report.contains("unexpected closing `</script>`")
                {
                    "\nHint: `index.html` still has leftover inline CSS/JS lines from a partial tag replacement. Call `write_file` on `index.html` with the complete clean HTML document."
                } else {
                    ""
                };
                let directive = format!(
                    "[Repair Required — attempt {repair_attempts}] {fence_note}A file edit failed or has validation errors on disk:\n{report}{html_hint}\nDo not write prose summaries. Start your very next token immediately with `<tool_call>` using `write_file` or `edit_file`."
                );
                append_repair_directive_to_history(history, &comp.text, &directive);
                continue;
            }

            let oracle_res = if touched_files {
                Some(verify_workspace_work(
                    &active_workdir,
                    &modified_files,
                    cfg.cmd_timeout_secs,
                ))
            } else {
                None
            };
            if let Some(Ok(ref v_rep)) = oracle_res {
                let _ = on_event(AgentEvent::Verified {
                    step,
                    report: v_rep,
                });
            }
            let oracle_err = oracle_res
                .as_ref()
                .and_then(|r| r.as_ref().err().map(String::as_str));

            let tested_after_last_edit = !touched_files
                || (last_passing_test_seq > 0 && last_passing_test_seq >= last_mutation_seq);
            let mut jev_eval_opt = None;
            with_model(&mut |model| {
                jev_eval_opt = Some(crate::jev::evaluate_rubric_with_telemetry(
                    Some(model),
                    &clean_user_goal,
                    &jev_plan.rubric,
                    &active_workdir,
                    &modified_files,
                    used_any_tools,
                    tested_after_last_edit,
                    oracle_err,
                    &comp.text,
                ));
            });
            let jev_eval = jev_eval_opt.ok_or_else(|| "model unavailable".to_string())?;

            let cur_oracle_ok = oracle_err.is_none();
            if touched_files {
                let is_new_high_water = match best_checkpoint.as_ref() {
                    None => true,
                    Some(prev) if prev.workdir != active_workdir => true,
                    Some(prev) => {
                        (cur_oracle_ok && !prev.oracle_ok)
                            || (cur_oracle_ok == prev.oracle_ok
                                && jev_eval.score.expected >= prev.score - 0.01)
                    }
                };
                if is_new_high_water {
                    best_checkpoint = Some(WorkspaceCheckpoint::capture(
                        &active_workdir,
                        &tracked_files,
                        cur_oracle_ok,
                        jev_eval.score.expected,
                    ));
                }
            }

            let needs_tool_kick = cfg.expect_tool_call && !used_any_tools;
            if (!jev_eval.passed || needs_tool_kick) && !active_tools.is_empty() && can_repair {
                let mut rolled_back = false;
                if let Some(ref prev) = best_checkpoint {
                    if prev.workdir == active_workdir
                        && prev.oracle_ok
                        && !cur_oracle_ok
                        && jev_eval.score.expected + 0.08 < prev.score
                    {
                        prev.restore();
                        rolled_back = true;
                    }
                }
                if (!jev_plan.rubric.is_empty() || jev_eval.ungrounded_claim)
                    && !on_event(AgentEvent::JevVerification {
                        step,
                        eval: &jev_eval,
                        rolled_back,
                    })
                {
                    return Err("cancelled".to_string());
                }
                repair_attempts += 1;
                consecutive_no_tool_steps += 1;
                let fence_note = if had_fence {
                    "You emitted markdown code fences in prose instead of `<tool_call>`. Markdown fences do NOT save files or run commands on disk. "
                } else {
                    ""
                };
                if needs_tool_kick && !used_any_tools {
                    let report = if jev_eval.unmet_summary.is_empty() {
                        "requested tool action has not been executed yet"
                    } else {
                        jev_eval.unmet_summary.as_str()
                    };
                    if !on_event(AgentEvent::ValidationRepair { step, report }) {
                        return Err("cancelled".to_string());
                    }
                    let directive = format!(
                        "[Repair Required — attempt {repair_attempts}] {fence_note}No tool call was emitted yet, so the requested action has not run. Start your very next token immediately with `<tool_call>` (`write_file`, `edit_file`, `read_file`, `list_dir`, or `run_command`) without prose summaries or markdown code blocks."
                    );
                    append_repair_directive_to_history(history, &comp.text, &directive);
                    continue;
                }

                let combined_report = match (oracle_err, jev_eval.unmet_summary.is_empty()) {
                    (Some(err), false) if !jev_eval.unmet_summary.contains(err) => {
                        format!("{err}\n{}", jev_eval.unmet_summary)
                    }
                    (Some(err), _) => err.to_string(),
                    (None, _) => jev_eval.unmet_summary.clone(),
                };
                if !on_event(AgentEvent::ValidationRepair {
                    step,
                    report: &combined_report,
                }) {
                    return Err("cancelled".to_string());
                }
                let rollback_note = if rolled_back {
                    "The last breaking edit was rolled back to the previous verified checkpoint. "
                } else {
                    ""
                };
                let html_hint = if combined_report.contains("unexpected closing `</style>`")
                    || combined_report.contains("unexpected closing `</script>`")
                {
                    "\nHint: `index.html` still contains leftover inline CSS/JS lines. Call `write_file` on `index.html` with the complete clean HTML document."
                } else {
                    ""
                };
                let cmd_fail_note = match last_failed_cmd.as_ref() {
                    Some((fail_seq, prev_cmd, fail_tail)) if *fail_seq == last_mutation_seq => format!(
                        "\nLast command failure (`{prev_cmd}` — modify code with `write_file` or `edit_file` before re-running):\n{fail_tail}"
                    ),
                    _ => String::new(),
                };
                let disk_files = summarize_workspace_files_brief(&active_workdir);
                let directive = format!(
                    "[Repair Required — attempt {repair_attempts}] {fence_note}{rollback_note}Current workspace files on disk: {disk_files}\nVerification found unmet requirements or errors:\n{combined_report}{html_hint}{cmd_fail_note}\nDo not write prose tutorials or markdown code blocks. Start your very next token immediately with `<tool_call>` (`write_file`, `edit_file`, `read_file`, or `run_command`) to fix and verify the remaining requirements on disk."
                );
                append_repair_directive_to_history(history, &comp.text, &directive);
                continue;
            }

            history.push(Message::new("assistant", &comp.text));

            let mut final_rolled_back = false;
            if let Some(ref prev) = best_checkpoint {
                if prev.workdir == active_workdir && prev.oracle_ok && !cur_oracle_ok {
                    prev.restore();
                    final_rolled_back = true;
                }
            }
            if !jev_plan.rubric.is_empty() || !jev_eval.passed {
                let _ = on_event(AgentEvent::JevVerification {
                    step,
                    eval: &jev_eval,
                    rolled_back: final_rolled_back,
                });
            }

            let secs = t_start.elapsed().as_secs_f64();
            let _ = on_event(AgentEvent::Done {
                steps: step,
                total_secs: secs,
                final_text: &comp.text,
            });
            return Ok(comp.text);
        }

        used_any_tools = true;
        consecutive_no_tool_steps = 0;

        for (idx, tc) in comp.tool_calls.iter_mut().enumerate() {
            tc.id = format!("call_s{step}_{idx}");
        }
        let assistant_msg_idx = history.len();
        history.push(Message::assistant_tool_calls("", comp.tool_calls.clone()));

        let mut batch_blocker: Option<String> = None;
        for (idx, tc) in comp.tool_calls.iter().enumerate() {
            let is_file_mutation = matches!(tc.name.as_str(), "write_file" | "edit_file");
            if is_file_mutation {
                touched_files = true;
                if let Some(raw_p) = Json::parse(&tc.arguments)
                    .ok()
                    .and_then(|j| j.get("path").and_then(Json::as_str).map(str::to_string))
                {
                    if !raw_p.is_empty() && !tracked_files.iter().any(|p| p == &raw_p) {
                        tracked_files.push(raw_p);
                    }
                }
            }
            if !on_event(AgentEvent::ToolStart { step, call: tc }) {
                return Err("cancelled".to_string());
            }
            let outcome = if tc.name == "spawn_agent" {
                exec_spawn_agent(&active_workdir, &tc.arguments, spawner)
            } else if tc.name == "run_command" {
                let cmd_key = Json::parse(&tc.arguments)
                    .ok()
                    .and_then(|j| j.get("cmd").and_then(Json::as_str).map(|s| s.trim().to_string()))
                    .unwrap_or_default();
                if let Some(ref reason) = batch_blocker {
                    err_outcome(&format!(
                        "skipped `run_command` because {reason}; inspect and resolve that error first before running dependent commands"
                    ))
                } else if let Some((fail_seq, ref prev_cmd, ref fail_tail)) = last_failed_cmd {
                    if fail_seq == last_mutation_seq && !cmd_key.is_empty() && prev_cmd == &cmd_key {
                        err_outcome(&format!(
                            "error: blocked duplicate `run_command` (`{cmd_key}`) because this exact command already failed and no workspace files have been modified since that failure. Last failure output tail:\n{fail_tail}\nCall `edit_file` or `write_file` to fix the script on disk before running `{cmd_key}` again."
                        ))
                    } else {
                        execute_tool(&active_workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx)
                    }
                } else {
                    execute_tool(&active_workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx)
                }
            } else {
                execute_tool(&active_workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx)
            };
            if let Some(ref next_dir) = outcome.new_workdir {
                active_workdir.clone_from(next_dir);
                touched_files = false;
                action_seq = 0;
                last_mutation_seq = 0;
                last_passing_test_seq = 0;
                modified_files.clear();
                tracked_files.clear();
                best_checkpoint = None;
                unresolved_edits.clear();
                last_failed_cmd = None;
                if !on_event(AgentEvent::WorkspaceChanged {
                    workdir: &active_workdir,
                }) {
                    return Err("cancelled".to_string());
                }
            }
            if is_file_mutation {
                let raw_target = Json::parse(&tc.arguments)
                    .ok()
                    .and_then(|j| j.get("path").and_then(Json::as_str).map(str::to_string))
                    .unwrap_or_default();
                let target_path = outcome
                    .diff
                    .as_ref()
                    .map_or_else(|| raw_target.clone(), |d| d.path.clone());
                if !target_path.is_empty() && !tracked_files.iter().any(|p| p == &target_path) {
                    tracked_files.push(target_path.clone());
                }
                if outcome.ok {
                    action_seq += 1;
                    last_mutation_seq = action_seq;
                    unresolved_edits.retain(|(p, _)| p != &target_path && p != &raw_target);
                    if !outcome.validation_ok {
                        consecutive_edit_failures += 1;
                        if batch_blocker.is_none() {
                            batch_blocker = Some(format!(
                                "an earlier file edit (`{target_path}`) failed validation: {}",
                                outcome.validation_msg
                            ));
                        }
                        if tc.name == "write_file" && outcome.validation_msg.contains("is truncated") {
                            if let Some(msg) = history.get_mut(assistant_msg_idx) {
                                if let Some(stored_tc) = msg.tool_calls.get_mut(idx) {
                                    stored_tc.arguments = format!(
                                        "{{\"path\":{},\"content\":\"[truncated draft omitted; rewrite complete file]\"}}",
                                        quote(&target_path)
                                    );
                                }
                            }
                        }
                        unresolved_edits.push((target_path.clone(), outcome.validation_msg.clone()));
                    } else {
                        consecutive_edit_failures = 0;
                    }
                    if !target_path.is_empty() && !modified_files.iter().any(|p| p == &target_path) {
                        modified_files.push(target_path);
                    }
                } else {
                    consecutive_edit_failures += 1;
                    let summary = outcome
                        .output
                        .lines()
                        .next()
                        .unwrap_or("edit failed")
                        .to_string();
                    if batch_blocker.is_none() {
                        batch_blocker = Some(format!(
                            "an earlier file edit (`{target_path}`) failed: {summary}"
                        ));
                    }
                    if let Some(existing) = unresolved_edits.iter_mut().find(|(p, _)| p == &target_path) {
                        existing.1 = summary;
                    } else {
                        unresolved_edits.push((target_path, summary));
                    }
                }
            } else if tc.name == "run_command" {
                let cmd_str = Json::parse(&tc.arguments)
                    .ok()
                    .and_then(|j| j.get("cmd").and_then(Json::as_str).map(|s| s.trim().to_string()))
                    .unwrap_or_default();
                if outcome.ok {
                    last_failed_cmd = None;
                    if is_verification_or_test_command(&cmd_str) {
                        action_seq += 1;
                        last_passing_test_seq = action_seq;
                    }
                } else {
                    if !outcome.output.starts_with("error: blocked duplicate `run_command`")
                        && !outcome.output.starts_with("skipped `run_command`")
                    {
                        let tail = extract_tail_lines(&outcome.output, 14, 1400);
                        last_failed_cmd = Some((last_mutation_seq, cmd_str, tail));
                    }
                    if batch_blocker.is_none() {
                        batch_blocker =
                            Some("an earlier `run_command` in this step exited with an error".to_string());
                    }
                }
            }
            if !on_event(AgentEvent::ToolDone {
                step,
                call: tc,
                outcome: &outcome,
            }) {
                return Err("cancelled".to_string());
            }
            history.push(Message::tool_result(&tc.id, &outcome.output));
        }

        if !unresolved_edits.is_empty() {
            if let Some(last_tool) = history.last_mut().filter(|m| m.role == "tool") {
                let pending: Vec<String> = unresolved_edits
                    .iter()
                    .filter(|(p, _)| !p.is_empty() && !last_tool.content.contains(p.as_str()))
                    .map(|(p, err)| format!("`{p}`: {err}"))
                    .collect();
                if !pending.is_empty() {
                    last_tool.content.push_str(&format!(
                        "\n[Pending validation errors still on disk — fix with `write_file` before finishing: {}]",
                        pending.join("; ")
                    ));
                }
            }
        }

        let step_had_mutation = comp
            .tool_calls
            .iter()
            .any(|tc| matches!(tc.name.as_str(), "write_file" | "edit_file"));
        let had_unparsed_tool_tag = comp.text.contains("<tool_call>");
        let had_unsaved_prose_fences =
            !step_had_mutation && comp.text.contains("```") && comp.text.len() > 400;
        if had_unparsed_tool_tag || had_unsaved_prose_fences {
            let disk_files = summarize_workspace_files_brief(&active_workdir);
            let note = if had_unparsed_tool_tag {
                format!(
                    "\n[System warning: An unclosed or malformed <tool_call> in your step did not execute. Current workspace files on disk: {disk_files}. Write 2 to 3 complete files per step using valid write_file <tool_call> JSON blocks.]"
                )
            } else {
                format!(
                    "\n[System warning: Markdown code fences in prose are NOT written to disk. Current workspace files on disk: {disk_files}. Call write_file <tool_call> with complete non-empty content for each file.]"
                )
            };
            if let Some(last_msg) = history.last_mut().filter(|m| m.role == "tool") {
                last_msg.content.push_str(&note);
            }
        }
    }

    if let Some(ref prev) = best_checkpoint {
        if prev.workdir == active_workdir
            && prev.oracle_ok
            && verify_workspace_work(&active_workdir, &modified_files, cfg.cmd_timeout_secs)
                .is_err()
        {
            prev.restore();
        }
    }

    Err(format!(
        "reached max_steps ({}) without final response",
        cfg.max_steps
    ))
}

/// Serializes `Message` history into OpenAI-compatible JSON array items.
pub fn serialize_messages_json(messages: &[Message]) -> String {
    let mut items = Vec::with_capacity(messages.len());
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
            items.push(format!(
                r#"{{"role":"assistant","content":{content_val},"tool_calls":[{}]}}"#,
                tcs.join(",")
            ));
        } else if m.role == "tool" {
            let id = m.tool_call_id.as_deref().unwrap_or("call_0");
            items.push(format!(
                r#"{{"role":"tool","tool_call_id":{},"content":{}}}"#,
                quote(id),
                quote(&m.content)
            ));
        } else {
            items.push(format!(
                r#"{{"role":{},"content":{}}}"#,
                quote(&m.role),
                quote(&m.content)
            ));
        }
    }
    format!("[{}]", items.join(","))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jail_edit_read_search_and_fast_validation_gate() {
        let dir = std::env::temp_dir().join(format!("kronos_shared_agent_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        assert!(resolve_jailed_path(&dir, "../escape.txt").is_err());
        assert!(resolve_jailed_path(&dir, "a/../../escape.txt").is_err());
        assert!(resolve_jailed_path(&dir, "/etc/passwd").is_err());
        assert_eq!(resolve_jailed_path(&dir, "src/app.js").unwrap(), dir.join("src/app.js"));
        assert!(resolve_readable_path(&dir, "/etc/passwd").is_err());
        assert!(resolve_readable_path(&dir, "~/.ssh/id_rsa").is_err());
        let ext_doc = std::env::temp_dir().join(format!("kronos_ext_doc_{}.md", std::process::id()));
        fs::write(&ext_doc, "# Kronos External Doc\n").unwrap();
        assert_eq!(
            resolve_readable_path(&dir, &ext_doc.display().to_string()).unwrap(),
            ext_doc.canonicalize().unwrap()
        );
        let _ = fs::remove_file(&ext_doc);

        // 1. write_file with broken JS reports validation error in < 20 ms.
        let t_val = Instant::now();
        let bad_js = execute_tool(
            &dir,
            &ToolCall {
                id: "c0".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"app.js","content":"function add(a, b) {\n  return a + b;\n"}"#.into(),
            },
            5,
            1,
        );
        assert!(t_val.elapsed() < Duration::from_millis(20));
        assert!(bad_js.ok);
        assert!(!bad_js.validation_ok);
        assert!(bad_js.output.contains("validation error"));

        // 2. edit_file surgically repairs app.js and passes validation.
        let fix_js = execute_tool(
            &dir,
            &ToolCall {
                id: "c1".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"app.js","old_text":"  return a + b;","new_text":"  const msg = `sum=${a + b}`;\n  return msg;\n}"}"#.into(),
            },
            5,
            2,
        );
        assert!(fix_js.ok);
        assert!(fix_js.validation_ok, "expected valid JS: {}", fix_js.output);
        assert!(fix_js.output.contains("[validation: ok]"));

        // 3. read_file with start_line / end_line returns 1-based line numbers.
        let rd = execute_tool(
            &dir,
            &ToolCall {
                id: "c2".into(),
                name: "read_file".into(),
                arguments: r#"{"path":"app.js","start_line":2,"end_line":3}"#.into(),
            },
            5,
            3,
        );
        assert!(rd.ok);
        assert!(rd.output.contains("2:   const msg ="));
        assert!(rd.output.contains("3:   return msg;"));

        // 4. HTML missing style.css passes non-strict single-file check and fails strict workspace check until style.css exists.
        let html_out = execute_tool(
            &dir,
            &ToolCall {
                id: "c3".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"index.html","content":"<!doctype html><html><head><link rel=\"stylesheet\" href=\"style.css\"></head><body><div id=\"app\">ok</div><script src=\"app.js\"></script></body></html>"}"#.into(),
            },
            5,
            4,
        );
        assert!(html_out.validation_ok);
        assert!(html_out.output.contains("pending referenced asset"));
        assert!(validate_workspace(&dir).is_err());

        let css_out = execute_tool(
            &dir,
            &ToolCall {
                id: "c4".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"style.css","content":"body { margin: 0; color: #eee; }"}"#.into(),
            },
            5,
            5,
        );
        assert!(css_out.validation_ok);
        assert!(validate_workspace(&dir).is_ok());

        // 5. search_files finds `sum=` in `app.js`.
        let srch = execute_tool(
            &dir,
            &ToolCall {
                id: "c5".into(),
                name: "search_files".into(),
                arguments: r#"{"pattern":"sum="}"#.into(),
            },
            5,
            6,
        );
        assert!(srch.ok);
        assert!(srch.output.contains("app.js:2:"));

        // 6. ProseStreamFilter strips `<tool_call>...</tool_call>` across chunk boundaries.
        let mut f = ProseStreamFilter::new();
        let mut collected = String::new();
        for piece in ["Working on it.\n<tool_", "call>\n{\"name\":\"list_dir\"}\n</tool", "_call>\nDone."] {
            f.push(piece, |p| {
                collected.push_str(p);
                true
            });
        }
        f.finish(|p| {
            collected.push_str(p);
            true
        });
        assert_eq!(collected, "Working on it.\n\nDone.");

        let mut f2 = ProseStreamFilter::new();
        let mut c2 = String::new();
        for piece in [
            "Updating.\n```js",
            "on\n{\n  \"name\": \"edit_file\",\n  \"arguments\": {\"path\":\"a.js\"}\n}\n``",
            "`\nExample:\n```json\n{\"ok\":true}\n```",
        ] {
            f2.push(piece, |p| {
                c2.push_str(p);
                true
            });
        }
        f2.finish(|p| {
            c2.push_str(p);
            true
        });
        assert_eq!(c2, "Updating.\n\nExample:\n```json\n{\"ok\":true}\n```");

        let mut f3 = ProseStreamFilter::new();
        let mut c3 = String::new();
        for piece in ["{\"na", "me\": \"edit_file\", \"arguments\": {\"path\":\"app.js\"}}"] {
            f3.push(piece, |p| {
                c3.push_str(p);
                true
            });
        }
        f3.finish(|p| {
            c3.push_str(p);
            true
        });
        assert_eq!(c3, "");

        // 7. edit_file trimmed-line fallback handles indentation + blank-line differences.
        let css_init = execute_tool(
            &dir,
            &ToolCall {
                id: "c7a".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"theme.css","content":"body {\n    margin: 20px;\n}\n\nh1 {\n    text-align: center;\n}\n"}"#.into(),
            },
            5,
            7,
        );
        assert!(css_init.ok && css_init.validation_ok);

        let css_trim_edit = execute_tool(
            &dir,
            &ToolCall {
                id: "c7b".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"theme.css","old_text":"body {\n  margin: 20px;\n}\nh1 {\n  text-align: center;\n}","new_text":"body {\n  margin: 0;\n  background: #fafaf9;\n}\n\nh1 {\n  color: #0d9488;\n}"}"#.into(),
            },
            5,
            8,
        );
        assert!(css_trim_edit.ok && css_trim_edit.validation_ok);
        let updated_css = fs::read_to_string(dir.join("theme.css")).unwrap();
        assert!(updated_css.contains("color: #0d9488;"));

        // 8. edit_file mismatch on small file echoes numbered contents for immediate retry.
        let css_miss = execute_tool(
            &dir,
            &ToolCall {
                id: "c8".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"theme.css","old_text":".nonexistent { color: red; }","new_text":".nonexistent { color: blue; }"}"#.into(),
            },
            5,
            9,
        );
        assert!(!css_miss.ok);
        assert!(css_miss.output.contains("1 | body {"));

        // 9. build_initial_system_prompt snapshots small populated workspace and stays bare when empty.
        let populated_sys = build_initial_system_prompt(&dir);
        assert!(populated_sys.contains("--- theme.css"));
        let empty_dir = std::env::temp_dir().join(format!("kronos_empty_ws_{}", std::process::id()));
        let _ = fs::remove_dir_all(&empty_dir);
        fs::create_dir_all(&empty_dir).unwrap();
        assert_eq!(build_initial_system_prompt(&empty_dir), AGENT_SYSTEM_PROMPT);
        let _ = fs::remove_dir_all(&empty_dir);

        // 10. classify_turn_intent automatically distinguishes chat Q&A from workspace engineering.
        assert_eq!(classify_turn_intent("hello", false, false, false), TurnRoute::Chat);
        assert_eq!(
            classify_turn_intent("explain how speculative decoding works", false, true, true),
            TurnRoute::Chat
        );
        assert_eq!(
            classify_turn_intent("what text is in this screenshot?", true, false, false),
            TurnRoute::Chat
        );
        assert_eq!(
            classify_turn_intent("build a clean task tracker web app", false, false, false),
            TurnRoute::Agent
        );
        assert_eq!(
            classify_turn_intent("Edit the title to Kronos is Working", false, true, true),
            TurnRoute::Agent
        );
        assert_eq!(
            classify_turn_intent(
                "style It and make it more beautify, style the input, buttons and header",
                false,
                true,
                true
            ),
            TurnRoute::Agent
        );
        assert_eq!(
            classify_turn_intent("match this dark UI layout", true, false, false),
            TurnRoute::Agent
        );
        assert_eq!(
            classify_turn_intent(
                "What exact text and hex colors appear in this pasted image?",
                true,
                true,
                true
            ),
            TurnRoute::Chat
        );
        assert_eq!(
            classify_turn_intent("where's the folder for this website", false, true, true),
            TurnRoute::Chat
        );
        assert_eq!(
            classify_turn_intent("The title is still the same", true, true, true),
            TurnRoute::Agent
        );

        let unified_sys = build_unified_system_prompt(&dir);
        assert!(unified_sys.contains(&format!("Workspace directory on disk: {}", dir.display())));

        // 10b. edit_file with empty old_text overwrites file, and HTML <title> edits sync visible <h1>.
        let overwrite_css = execute_tool(
            &dir,
            &ToolCall {
                id: "c10b1".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"style.css","old_text":"","new_text":"body { background: #0f172a; color: #f8fafc; }\nbutton { border-radius: 8px; }\n"}"#.into(),
            },
            5,
            10,
        );
        assert!(overwrite_css.ok && overwrite_css.validation_ok);
        assert!(fs::read_to_string(dir.join("style.css")).unwrap().contains("border-radius: 8px;"));

        let seed_html = execute_tool(
            &dir,
            &ToolCall {
                id: "c10b2".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"page.html","content":"<!doctype html><html><head><title>Initial Title</title></head><body><h1 class=\"header\">Initial Title</h1></body></html>"}"#.into(),
            },
            5,
            11,
        );
        assert!(seed_html.ok && seed_html.validation_ok);
        let edit_title = execute_tool(
            &dir,
            &ToolCall {
                id: "c10b3".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"page.html","old_text":"<title>Initial Title</title>","new_text":"<title>Kronos Workspace</title>"}"#.into(),
            },
            5,
            12,
        );
        assert!(edit_title.ok && edit_title.validation_ok);
        let page_after_edit = fs::read_to_string(dir.join("page.html")).unwrap();
        assert!(page_after_edit.contains("<title>Kronos Workspace</title>"));
        assert!(page_after_edit.contains("<h1 class=\"header\">Kronos Workspace</h1>"));

        let idemp_h1 = execute_tool(
            &dir,
            &ToolCall {
                id: "c10b3_dup".into(),
                name: "edit_file".into(),
                arguments: r#"{"path":"page.html","old_text":"<h1 class=\"header\">Initial Title</h1>","new_text":"<h1 class=\"header\">Kronos Workspace</h1>"}"#.into(),
            },
            5,
            12,
        );
        assert!(idemp_h1.ok && idemp_h1.validation_ok);

        let write_title = execute_tool(
            &dir,
            &ToolCall {
                id: "c10b4".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"page.html","content":"<!doctype html><html><head><title>Kronos Next</title></head><body><h1 class=\"header\">Kronos Workspace</h1></body></html>"}"#.into(),
            },
            5,
            13,
        );
        assert!(write_title.ok && write_title.validation_ok);
        let page_after_write = fs::read_to_string(dir.join("page.html")).unwrap();
        assert!(page_after_write.contains("<h1 class=\"header\">Kronos Next</h1>"));

        // 11. compact_messages_with_budget prunes older multi-turn and intra-turn tool bloat
        //     while keeping system prefix, image asset references, and live disk file snapshot.
        let approx_tokens = |msgs: &[Message]| -> usize {
            msgs.iter()
                .map(|m| {
                    m.content.len() / 4
                        + m.tool_calls
                            .iter()
                            .map(|tc| tc.name.len() + tc.arguments.len() / 4)
                            .sum::<usize>()
                        + 6
                })
                .sum()
        };
        let big_css = "body { color: #111; }\n".repeat(120);
        let mut hist = vec![
            Message::new("system", UNIFIED_SYSTEM_PROMPT),
            Message::new(
                "user",
                "[Attached Image #1: 800x600 PNG, theme=dark, background=#0f172a, accent=#14b8a6, workspace_asset_path=\".kronos_media/image_1.png\"]\n- Detected text in 2D reading order:\n  [top-center] Dashboard\nBuild initial theme",
            ),
            Message::assistant_tool_calls(
                "",
                vec![ToolCall {
                    id: "t1".into(),
                    name: "write_file".into(),
                    arguments: format!(r#"{{"path":"theme.css","content":{}}}"#, quote(&big_css)),
                }],
            ),
            Message::tool_result("t1", "ok: wrote theme.css [validation: ok]"),
            Message::new("assistant", "Created initial stylesheet."),
            Message::new("user", "Update header accent color"),
            Message::assistant_tool_calls(
                "",
                vec![ToolCall {
                    id: "t2".into(),
                    name: "write_file".into(),
                    arguments: format!(r#"{{"path":"theme.css","content":{}}}"#, quote(&big_css)),
                }],
            ),
            Message::tool_result("t2", "ok: wrote theme.css [validation: ok]"),
            Message::new("assistant", "Updated header accent."),
            Message::new("user", "Now polish button padding"),
        ];
        let sys_before = hist[0].content.clone();
        let rep = compact_messages_with_budget(&mut hist, &dir, false, 400, approx_tokens);
        assert!(rep.compacted);
        assert!(rep.after_tokens < rep.before_tokens);
        assert_eq!(hist[0].content, sys_before);
        assert!(hist[1].content.starts_with(COMPACTED_MEMORY_HEADER));
        assert!(hist[1].content.contains(".kronos_media/image_1.png"));
        assert!(hist[1].content.contains("--- theme.css"));
        assert_eq!(hist.last().unwrap().content, "Now polish button padding");

        // 12. compute_unified_diff + verify_workspace_work catch DOM/CSS/V8 bugs and verify clean repair.
        let (add_n, rem_n, udiff) =
            compute_unified_diff("app.js", "const a = 1;\nconst b = 2;\n", "const a = 1;\nconst b = 3;\nconst c = 4;\n");
        assert_eq!(add_n, 2);
        assert_eq!(rem_n, 1);
        assert!(udiff.contains("-const b = 2;"));
        assert!(udiff.contains("+const b = 3;"));

        let broken_dom_js = execute_tool(
            &dir,
            &ToolCall {
                id: "c12a".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"app.js","content":"const btn = document.getElementById('ghost-missing-id');\n"}"#.into(),
            },
            5,
            14,
        );
        assert!(broken_dom_js.ok);
        assert!(broken_dom_js.diff.is_some());
        let dom_err = verify_workspace_work(&dir, &["app.js".to_string()], 5).unwrap_err();
        assert!(dom_err.contains("ghost-missing-id"), "unexpected verify err: {dom_err}");

        let fixed_dom_js = execute_tool(
            &dir,
            &ToolCall {
                id: "c12b".into(),
                name: "write_file".into(),
                arguments: r#"{"path":"app.js","content":"const root = document.getElementById('app');\n"}"#.into(),
            },
            5,
            15,
        );
        assert!(fixed_dom_js.ok);
        let v_ok = verify_workspace_work(&dir, &["app.js".to_string()], 5).unwrap();
        assert!(v_ok.checks_run >= 2);
        assert!(v_ok.summary.contains("DOM/CSS cross-check ok"));

        // 13. switch_workspace creates new child project folder, blocks umbrella & protected paths, and normalizes write_file paths.
        let umbrella = dir.join("orgProj_test");
        fs::create_dir_all(umbrella.join("alpha_proj")).unwrap();
        fs::create_dir_all(umbrella.join("beta_proj")).unwrap();

        let blocked_umbrella = execute_tool(
            &dir,
            &ToolCall {
                id: "c13a".into(),
                name: "switch_workspace".into(),
                arguments: format!(r#"{{"path":{}}}"#, quote(&umbrella.display().to_string())),
            },
            5,
            16,
        );
        assert!(!blocked_umbrella.ok);
        assert!(blocked_umbrella.output.contains("parent directory"));

        let blocked_ssh = execute_tool(
            &dir,
            &ToolCall {
                id: "c13b".into(),
                name: "switch_workspace".into(),
                arguments: r#"{"path":"~/.ssh/bad_workspace"}"#.into(),
            },
            5,
            17,
        );
        assert!(!blocked_ssh.ok);

        let emr_dir = umbrella.join("emr-software");
        let switched = execute_tool(
            &dir,
            &ToolCall {
                id: "c13c".into(),
                name: "switch_workspace".into(),
                arguments: format!(r#"{{"path":{}}}"#, quote(&emr_dir.display().to_string())),
            },
            5,
            18,
        );
        assert!(switched.ok, "switch failed: {}", switched.output);
        let new_ws = switched.new_workdir.expect("new_workdir");
        assert!(new_ws.is_dir());

        let abs_html = new_ws.join("index.html");
        let wrote_in_new = execute_tool(
            &new_ws,
            &ToolCall {
                id: "c13d".into(),
                name: "write_file".into(),
                arguments: format!(
                    r#"{{"path":{},"content":"<!doctype html><html><head><title>EMR</title></head><body><h1>EMR</h1></body></html>"}}"#,
                    quote(&abs_html.display().to_string())
                ),
            },
            5,
            19,
        );
        assert!(wrote_in_new.ok);
        assert_eq!(wrote_in_new.diff.unwrap().path, "index.html");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_zero_hallucination_imports_stubs_dom_runtime_and_checkpoint_rollback() {
        let dir = std::env::temp_dir().join(format!("kronos_jev_oracle_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        fs::write(
            dir.join("index.html"),
            "<!doctype html><html><head><title>App</title><script src=\"app.js\"></script></head><body><h1 id=\"hdr\">Dashboard</h1></body></html>",
        )
        .unwrap();
        fs::write(
            dir.join("app.js"),
            "const titleEl = document.getElementById('hdr');\n",
        )
        .unwrap();
        let tracked = vec!["index.html".to_string(), "app.js".to_string()];
        assert!(verify_workspace_work(&dir, &tracked, 5).is_ok());

        let clean_cp = WorkspaceCheckpoint::capture(&dir, &tracked, true, 0.94);

        // 1. Truncation stub comment must fail deterministic verification.
        fs::write(
            dir.join("app.js"),
            "// ... rest of code omitted for brevity\n",
        )
        .unwrap();
        let stub_err = verify_workspace_work(&dir, &tracked, 5).unwrap_err();
        assert!(stub_err.contains("truncation placeholder"), "got: {stub_err}");

        // 2. Unresolved local module import must fail deterministic verification.
        fs::write(
            dir.join("module.mjs"),
            "import { missingFn } from './nonexistent_helper.mjs';\nexport const x = missingFn;\n",
        )
        .unwrap();
        let imp_err =
            verify_workspace_work(&dir, &["module.mjs".to_string()], 5).unwrap_err();
        assert!(imp_err.contains("unresolved local module import"), "got: {imp_err}");
        let _ = fs::remove_file(dir.join("module.mjs"));

        // 3. Top-level ReferenceError in classic script must be caught by headless DOM runtime check when node is installed.
        fs::write(
            dir.join("app.js"),
            "document.addEventListener('DOMContentLoaded', () => { callUndefinedStartupHelper(); });\n",
        )
        .unwrap();
        if Command::new("sh")
            .args(["-c", "command -v node >/dev/null 2>&1"])
            .status()
            .is_ok_and(|s| s.success())
        {
            let dom_err = verify_workspace_work(&dir, &tracked, 5).unwrap_err();
            assert!(
                dom_err.contains("callUndefinedStartupHelper"),
                "expected ReferenceError caught, got: {dom_err}"
            );
        }

        // 4. Checkpoint rollback restores verified workspace state cleanly.
        clean_cp.restore();
        assert!(verify_workspace_work(&dir, &tracked, 5).is_ok());

        // 5. ES module `<script type="module">` cross-file named export mismatch is caught, and matching exports pass.
        if Command::new("sh")
            .args(["-c", "command -v node >/dev/null 2>&1"])
            .status()
            .is_ok_and(|s| s.success())
        {
            let esm_dir = dir.join("esm_case");
            fs::create_dir_all(esm_dir.join("js")).unwrap();
            fs::write(esm_dir.join("style.css"), "body { color: #f8fafc; background: #0f172a; }\n").unwrap();
            fs::write(
                esm_dir.join("index.html"),
                "<!doctype html><html><head><title>App</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><h1 id=\"title\">App</h1><script type=\"module\" src=\"js/entry.js\"></script></body></html>",
            )
            .unwrap();
            fs::write(
                esm_dir.join("js/state.js"),
                "export const ITEMS = ['a', 'b'];\n",
            )
            .unwrap();
            fs::write(
                esm_dir.join("js/entry.js"),
                "import { ITEMS, missingExport } from './state.js';\nwindow.addEventListener('DOMContentLoaded', () => { document.getElementById('title').textContent = String(ITEMS.length + missingExport); });\n",
            )
            .unwrap();
            let esm_files = vec![
                "index.html".to_string(),
                "style.css".to_string(),
                "js/state.js".to_string(),
                "js/entry.js".to_string(),
            ];
            let esm_err = verify_workspace_work(&esm_dir, &esm_files, 5).unwrap_err();
            assert!(
                esm_err.contains("missingExport"),
                "expected missing ES module export error, got: {esm_err}"
            );

            fs::write(
                esm_dir.join("js/state.js"),
                "export const ITEMS = ['a', 'b'];\nexport const missingExport = 1;\n",
            )
            .unwrap();
            assert!(verify_workspace_work(&esm_dir, &esm_files, 5).is_ok());
        }

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_auto_quote_unquoted_existing_paths_and_python_pitfalls() {
        let dir = std::env::temp_dir().join(format!("kronos_quote_and_py_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let spaced_file = dir.join("James Iroha Close 4.m4a");
        fs::write(&spaced_file, b"fake m4a").unwrap();
        let abs_spaced = spaced_file.display().to_string();

        let raw_cmd = format!(
            "/opt/homebrew/bin/ffmpeg -y -i {abs_spaced} -t 12 -ar 16000 -ac 1 raw_ref_16k.wav"
        );
        let quoted = auto_quote_unquoted_existing_paths(&dir, &raw_cmd);
        assert_eq!(
            quoted,
            format!(
                "/opt/homebrew/bin/ffmpeg -y -i '{abs_spaced}' -t 12 -ar 16000 -ac 1 raw_ref_16k.wav"
            )
        );
        // Already-quoted command stays untouched.
        assert_eq!(auto_quote_unquoted_existing_paths(&dir, &quoted), quoted);

        let eq_cmd = format!("tool --input={abs_spaced} --out=ok.wav");
        assert_eq!(
            auto_quote_unquoted_existing_paths(&dir, &eq_cmd),
            format!("tool --input='{abs_spaced}' --out=ok.wav")
        );

        let py_path = Path::new("script.py");
        let vid_path = Path::new("build_video.py");
        assert!(validate_python_runtime_pitfalls(py_path, "w, h = d.textsize('hi', font=f)\n").is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "pcm = np.array(wav.readframes(wav.getnframes()), dtype=np.int16)\n"
        )
        .is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "from faster_whisper import WhisperModel\nres = wm.transcribe(pcm)\ntxt = res['segments'][0]['text']\n"
        )
        .is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "from mlx_audio.tts.generate import generate_audio\ngenerate_audio(model=m, text=ref_text, ref_audio='raw_ref_24k.wav', ref_text=ref_text)\n"
        )
        .is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "from faster_whisper import WhisperModel\nsegs, _ = wm.transcribe(pcm)\nsegs = list(segs)\n"
        )
        .is_ok());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "from mlx_audio.tts.utils import load_model\nfrom mlx_audio.tts.generate import generate_audio\nimport subprocess, json\nm = load_model('tts-model')\ngenerate_audio(model=m, text=spoken, ref_audio='ref_24k.wav', ref_text=ref_text, output_path='audio', file_prefix='seg_01')\nwith open('audio/coherence_report.json', 'w') as f:\n    json.dump({'verified': True}, f)\nsubprocess.run(['ffmpeg', '-i', 'raw.m4a', 'ref_24k.wav'])\n"
        )
        .is_ok());
        assert!(validate_python_runtime_pitfalls(
            vid_path,
            "import os\nos.makedirs('slides', exist_ok=True)\n"
        )
        .is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "def f(text):\n    ref_text = text\n    mlx_audio.tts.generate.generate_audio(ref_audio='ref_24k.wav', ref_text=ref_text)\n"
        )
        .is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "import subprocess\nwith open('input.txt', 'w') as f:\n    f.write(\"file 'slides/slide_01.png'\\n\")\nsubprocess.run(['ffmpeg', '-f', 'concat', '-i', 'input.txt', 'out.mp4'])\n"
        )
        .is_err());
        // `, ref_audio=` must never fool the `import re` check when `re.findall` is used without `import re`.
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "from mlx_audio.tts.utils import load_model\nfrom mlx_audio.tts.generate import generate_audio\nimport subprocess, json\nm = load_model('tts-model')\ngenerate_audio(model=m, text=spoken, ref_audio='ref_24k.wav', ref_text=ref_text, output_path='audio', file_prefix='seg_01')\ntoks = re.findall(r'[a-z]+', spoken)\nwith open('audio/coherence_report.json', 'w') as f:\n    json.dump({'verified': True}, f)\nsubprocess.run(['ffmpeg', '-i', 'raw.m4a', 'ref_24k.wav'])\n"
        )
        .is_err());
        // Calling `load_model(...)` inside a loop or passing unresampled 24kHz TTS WAVs to `wm.transcribe` must fail validation.
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "from mlx_audio.tts.utils import load_model\nfrom mlx_audio.tts.generate import generate_audio\nimport subprocess, json\nfor idx, spoken in enumerate(['hello world']):\n    m = load_model('tts-model')\n    generate_audio(model=m, text=spoken, ref_audio='ref_24k.wav', ref_text=ref_text, output_path='audio', file_prefix='seg_01')\nwith open('audio/coherence_report.json', 'w') as f:\n    json.dump({'verified': True}, f)\nsubprocess.run(['ffmpeg', '-i', 'raw.m4a', 'ref_24k.wav'])\n"
        )
        .is_err());
        assert!(validate_python_runtime_pitfalls(
            py_path,
            "import wave, json, subprocess, numpy as np\nfrom faster_whisper import WhisperModel\nfrom mlx_audio.tts.utils import load_model\nfrom mlx_audio.tts.generate import generate_audio\nwm = WhisperModel('Systran/faster-whisper-small')\nm = load_model('tts-model')\ngenerate_audio(model=m, text=spoken, ref_audio='ref_24k.wav', ref_text=ref_text, output_path='audio', file_prefix='seg_01')\nwf = wave.open('audio/seg_01.wav', 'rb')\np24 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0\nwf.close()\nhyp = list(wm.transcribe(p24)[0])\nwith open('audio/coherence_report.json', 'w') as f:\n    json.dump({'verified': True}, f)\nsubprocess.run(['ffmpeg', '-i', 'raw.m4a', 'ref_24k.wav'])\n"
        )
        .is_err());

        // Intra-turn compaction must preserve the active user prompt (`[Device Context]`) verbatim
        // even when synthetic `[Repair Required ...]` messages exist in the same turn.
        let big_script = "A".repeat(9000);
        let user_with_ctx = "[Device Context · retrieved in 1.20 ms]\n  - [project] Kronos-video MLX Audio Python venv -> '/Users/test/.venv/bin/python3'\n[/Device Context]\n\nMake a video on Kronos";
        let mut hist = vec![
            Message::new("system", "Sys"),
            Message::new("user", user_with_ctx),
            Message::new("assistant", "Inspecting."),
            Message::new("user", "[Repair Required — attempt 1] Emit <tool_call>."),
            Message::assistant_tool_calls(
                "",
                vec![crate::chat::ToolCall {
                    id: "c1".to_string(),
                    name: "write_file".to_string(),
                    arguments: format!("{{\"path\":\"script.md\",\"content\":\"{big_script}\"}}"),
                }],
            ),
            Message::tool_result("c1", "ok: wrote 9000 bytes to script.md"),
            Message::new("assistant", "Inspecting."),
            Message::new("user", "[Repair Required — attempt 2] Continue."),
        ];
        fs::write(dir.join("script.md"), &big_script).unwrap();
        let rep = compact_messages_with_budget(&mut hist, &dir, false, 1200, |msgs| {
            msgs.iter().map(|m| m.content.len() / 4 + m.tool_calls.iter().map(|t| t.arguments.len() / 4).sum::<usize>() + 8).sum()
        });
        assert!(rep.compacted);
        assert_eq!(hist[1].content, user_with_ctx, "active user [Device Context] must never be compacted away");

        assert_eq!(
            strip_outer_markdown_code_fence("script.md", "```markdown\n# Kronos\nbody\n```\n").as_deref(),
            Some("# Kronos\nbody\n")
        );
        assert!(strip_outer_markdown_code_fence("script.md", "# Kronos\n```rust\nlet x = 1;\n```\n").is_none());
        assert!(validate_markdown_script(
            Path::new("script.md"),
            "# Kronos\nIteration 1 NF4 sub-2-bit truncated\n"
        )
        .is_err());
        let full_md = format!(
            "# Kronos Technical Script\nIteration 1 sub-2-bit NF4\n{}\nIteration 8 Kronos-Jev Joint Embedding\nBenchmarks vs llama.cpp and Ollama\n",
            "math ".repeat(700)
        );
        assert!(validate_markdown_script(Path::new("script.md"), &full_md).is_ok());

        let _ = fs::remove_dir_all(&dir);
    }
}
