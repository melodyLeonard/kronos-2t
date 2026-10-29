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
    "Your name is Kronos. You were created and built by Melody Leonard as a local software engineering runner on Apple Silicon. ",
    "When asked your name or who built or created you, answer directly that your name is Kronos and that you were built by Melody Leonard; never say you lack a name and never claim to be built by OpenAI, Alibaba, Qwen, Hugging Face, Anthropic, or any other third-party organization. ",
    "Use the workspace tools (write_file, edit_file, read_file, list_dir, search_files, run_command) ",
    "to build, edit, and verify complete working files directly on disk. ",
    "Do not paste file drafts or completion claims in prose before calling tools; emit <tool_call> invocations immediately, then summarize after tool calls succeed. ",
    "Use edit_file for small targeted edits to existing files, and use write_file for new files or when rewriting most of an existing file such as restyling a stylesheet. ",
    "When updating a web page title or heading, update both <title> in <head> and the visible <h1> heading in <body> so the rendered preview shows the change immediately. ",
    "The live preview reloads automatically after every file edit, so never ask the user to refresh the preview; if the user reports a change is not visible, inspect and update the visible HTML or CSS elements directly. ",
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
    ValidationRepair {
        step: usize,
        report: &'a str,
    },
    Verified {
        step: usize,
        report: &'a VerificationReport,
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

pub fn tool_definitions_json() -> Vec<String> {
    vec![
        r#"{"type":"function","function":{"name":"write_file","description":"Write or overwrite a UTF-8 workspace file and validate syntax.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative workspace path"},"content":{"type":"string","description":"Full file contents"}},"required":["path","content"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"edit_file","description":"Replace a unique text span in an existing workspace file and validate syntax.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative workspace path"},"old_text":{"type":"string","description":"Existing unique text block"},"new_text":{"type":"string","description":"Replacement text block"}},"required":["path","old_text","new_text"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"read_file","description":"Read a workspace file with 1-based line numbers.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative workspace path"},"start_line":{"type":"integer"},"end_line":{"type":"integer"}}},"required":["path"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"list_dir","description":"List directory entries inside workspace.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative directory (default '.')"}}},"required":[]}}"#.to_string(),
        r#"{"type":"function","function":{"name":"search_files","description":"Search workspace files for a substring pattern.","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"path":{"type":"string"},"case_insensitive":{"type":"boolean"}}},"required":["pattern"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"run_command","description":"Run sh -c command inside workspace with timeout.","parameters":{"type":"object","properties":{"cmd":{"type":"string"}}},"required":["cmd"]}}}"#.to_string(),
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
        "scaffold",
        "implement ",
        "fix ",
        "repair ",
        "debug ",
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

pub fn mime_type_for_path(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "html" | "htm" => "text/html; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "ico" => "image/x-icon",
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
    if !matches!(ext.as_str(), "json" | "js" | "mjs" | "ts" | "css" | "html" | "htm" | "py") {
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
    match ext {
        "json" => {
            Json::parse(content).map_err(|e| format!("JSON syntax error: {e}"))?;
            Ok(String::new())
        }
        "js" | "mjs" | "ts" => {
            validate_js_like(content, 1)?;
            Ok(String::new())
        }
        "css" => {
            validate_css(content, 1)?;
            Ok(String::new())
        }
        "html" | "htm" => validate_html(workdir, file_path, content, strict_assets),
        "py" => {
            validate_python_lexical(content)?;
            Ok(String::new())
        }
        _ => Ok(String::new()),
    }
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
            };
        }
    };

    match call.name.as_str() {
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
            }
        }
        "use_computer" => match crate::device::global_device_manager().execute_computer_action(&args) {
            Ok(msg) => ToolExecOutcome {
                ok: true,
                validation_ok: true,
                validation_msg: "ok".to_string(),
                output: msg,
                diff: None,
            },
            Err(err) => ToolExecOutcome {
                ok: false,
                validation_ok: true,
                validation_msg: "ok".to_string(),
                output: format!("error: {err}"),
                diff: None,
            },
        },
        other => ToolExecOutcome {
            ok: false,
            validation_ok: true,
            validation_msg: "ok".to_string(),
            output: format!("error: unknown tool `{other}`"),
            diff: None,
        },
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
            }
        }
        Err(err) => ToolExecOutcome {
            ok: true,
            validation_ok: false,
            validation_msg: err.clone(),
            output: format!("{base_msg}\n[validation error in `{rel}`: {err} — fix with `edit_file` or `write_file`]"),
            diff,
        },
    }
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

fn exec_write_file(workdir: &Path, args: &Json) -> ToolExecOutcome {
    let Some(rel) = args.get("path").and_then(Json::as_str) else {
        return err_outcome("error: `write_file` requires string argument `path`");
    };
    let Some(content) = args.get("content").and_then(Json::as_str) else {
        return err_outcome("error: `write_file` requires string argument `content`");
    };
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
    let mut final_content = content.to_string();
    let mut sync_note = String::new();
    let lower_rel = rel.to_ascii_lowercase();
    if lower_rel.ends_with(".html") || lower_rel.ends_with(".htm") {
        if let Some(prev) = &prev_opt {
            if let (Some(old_t), Some(old_h1), Some(new_t), Some(new_h1)) = (
                extract_plain_html_tag_inner(prev, "title"),
                extract_plain_html_tag_inner(prev, "h1"),
                extract_plain_html_tag_inner(content, "title"),
                extract_plain_html_tag_inner(content, "h1"),
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
            let diff = build_file_diff(rel, prev_opt, &final_content);
            let msg = format!(
                "ok: wrote {} bytes ({} lines{sync_note}) to `{rel}`",
                final_content.len(),
                final_content.lines().count()
            );
            attach_validation_status(workdir, &target, rel, msg, diff)
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
    let (updated, h1_synced) = sync_html_title_and_h1(rel, old_text, new_text, raw_updated);
    match fs::write(target, updated.as_bytes()) {
        Ok(()) => {
            let diff = build_file_diff(rel, prev_opt, &updated);
            let sync_note = match h1_synced {
                Some(t) => format!("; synced visible <h1> to `{t}`"),
                None => String::new(),
            };
            let msg = format!(
                "ok: edited `{rel}` at line {line_no} ({} bytes, {} lines{sync_note})",
                updated.len(),
                updated.lines().count()
            );
            attach_validation_status(workdir, target, rel, msg, diff)
        }
        Err(e) => err_outcome(&format!("error writing `{rel}`: {e}")),
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
    let target = match resolve_jailed_path(workdir, rel) {
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
    }
}

fn err_outcome(msg: &str) -> ToolExecOutcome {
    ToolExecOutcome {
        ok: false,
        validation_ok: true,
        validation_msg: "ok".to_string(),
        output: msg.to_string(),
        diff: None,
    }
}

/// Redirects stdout/stderr to a file rather than a pipe so backgrounded child processes do not block wait.
pub fn run_shell_command_with_timeout(
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum FilterMode {
    Normal,
    XmlTool,
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
                    let fence_pos = tail.find(FENCE);
                    let brace_pos = find_line_start_brace(&self.buf, self.emitted);
                    let earliest = [xml_pos, fence_pos, brace_pos].into_iter().flatten().min();
                    if let Some(open_rel) = xml_pos.filter(|&x| Some(x) == earliest) {
                        if open_rel > 0 {
                            let chunk = &self.buf[self.emitted..self.emitted + open_rel];
                            if !on_prose(chunk) {
                                return false;
                            }
                        }
                        self.emitted += open_rel + OPEN.len();
                        self.mode = FilterMode::XmlTool;
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
                        for pat in [OPEN, FENCE] {
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
    "Your name is Kronos. You were created and built by Melody Leonard as a local multimodal software engineering and reasoning engine on Apple Silicon. ",
    "When asked your name or who built or created you, answer directly that your name is Kronos and that you were built by Melody Leonard; never say you lack a name and never claim to be built by OpenAI, Alibaba, Qwen, Hugging Face, Anthropic, or any other third-party organization. ",
    "For general questions, explanations, or analysis of attached PDFs, documents, CSVs, code snippets, or images that do not modify workspace files, reply directly and concisely from the inline context without calling tools. ",
    "When a [Device Context] block is provided, answer using its exact local paths and watch history, and when `use_computer` is available and the user asks to open or control an application, file, folder, or URL on their Mac, call `use_computer` directly. ",
    "When asked to build, edit, style, fix, inspect, or run files in the workspace, use the workspace tools (write_file, edit_file, read_file, list_dir, search_files, run_command) directly on disk. ",
    "Do not paste file drafts or completion claims in prose before calling tools; emit <tool_call> invocations immediately, then summarize briefly after tool calls succeed. ",
    "Check the Existing workspace files list below before calling edit_file: only call edit_file on files that exist on disk using exact existing lines, and use write_file for new files or full rewrites. ",
    "When updating a web page title or heading, update both <title> in <head> and the visible <h1> heading in <body> so the rendered preview shows the change immediately. ",
    "The live preview reloads automatically after every file edit, so never ask the user to refresh the preview; if the user reports a change is not visible, inspect and update the visible HTML or CSS elements directly. ",
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

    // Group non-system messages into user-anchored atomic turns so tool_calls/tool_results never split.
    let mut group_starts: Vec<usize> = Vec::new();
    for (i, m) in history.iter().enumerate().skip(sys_offset) {
        if m.role == "user" || group_starts.is_empty() {
            group_starts.push(i);
        }
    }

    let total_groups = group_starts.len();
    let ends_with_open_user = history.last().is_some_and(|m| m.role == "user");
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

    // Intra-turn pass: if history still exceeds budget (e.g., one long multi-step turn), slim older tool steps.
    let mid_tokens = count_fn(history);
    if (force && compacted_messages == 0) || mid_tokens > threshold_tokens {
        let protect_tail = 4usize;
        let stop_at = history.len().saturating_sub(protect_tail);
        for m in history.iter_mut().take(stop_at).skip(sys_offset) {
            if m.role == "assistant" && !m.tool_calls.is_empty() {
                for tc in &mut m.tool_calls {
                    if matches!(tc.name.as_str(), "write_file" | "edit_file") && tc.arguments.len() > 260 {
                        let path = Json::parse(&tc.arguments)
                            .ok()
                            .and_then(|j| j.get("path").and_then(Json::as_str).map(str::to_string))
                            .unwrap_or_else(|| "file".to_string());
                        let orig_len = tc.arguments.len();
                        tc.arguments = format!(
                            r#"{{"path":{},"compacted_bytes":{orig_len}}}"#,
                            quote(&path)
                        );
                        compacted_messages += 1;
                    }
                }
            } else if m.role == "tool" && m.content.len() > 260 {
                let first = m.content.lines().next().unwrap_or("ok");
                m.content = format!(
                    "{}\n[compacted earlier step output; current workspace state is on disk]",
                    summarize_single_line(first, 140)
                );
                compacted_messages += 1;
            }
        }
    }

    let after_tokens = count_fn(history);
    CompactionReport {
        compacted: compacted_messages > 0 && after_tokens < before_tokens,
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
    // Auto-compact once prompt reaches ~68% of context window so multi-step decode always has ample KV room.
    let threshold = (ctx * 68) / 100;
    compact_messages_with_budget(history, workdir, force, threshold, |msgs| {
        model.count_prompt_tokens(msgs, tools_json)
    })
}

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

    match verify_web_dom_and_css_semantics(workdir) {
        Ok(Some(note)) => details.push(note),
        Ok(None) => {}
        Err(e) => errors.push(e),
    }

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
/// Runs an iterative tool-calling and post-work verification repair loop while preserving the KV-cache prefix.
pub fn run_agent_turn(
    model: &mut ChatModel,
    history: &mut Vec<Message>,
    tools_json: &[String],
    cfg: &AgentTurnConfig,
    mut on_event: impl FnMut(AgentEvent<'_>) -> bool,
) -> Result<String, String> {
    if history.first().is_none_or(|m| m.role != "system") {
        let sys_prompt = build_initial_system_prompt(&cfg.workdir);
        history.insert(0, Message::new("system", &sys_prompt));
    }
    let t_start = Instant::now();
    let mut touched_files = false;
    let mut used_any_tools = false;
    let mut repair_attempts = 0usize;
    let mut unresolved_edits: Vec<(String, String)> = Vec::new();
    let mut modified_files: Vec<String> = Vec::new();

    for step in 1..=cfg.max_steps {
        let comp_rep = compact_conversation_history(model, history, tools_json, &cfg.workdir, false);
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
        let mut last_prog_bytes = 0usize;
        let mut last_prog_target = String::new();
        let mut comp = model.generate_with_tools(history, tools_json, &cfg.params, |piece| {
            alive = filter.push(piece, |prose| on_event(AgentEvent::TextDelta(prose)));
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
                }
            }
            alive
        })?;
        if alive {
            let _ = filter.finish(|prose| on_event(AgentEvent::TextDelta(prose)));
        }
        if !on_event(AgentEvent::StepStats {
            step,
            completion: &comp,
        }) {
            return Err("cancelled".to_string());
        }

        if comp.tool_calls.is_empty() {
            history.push(Message::new("assistant", &comp.text));
            if !unresolved_edits.is_empty() && repair_attempts < 2 && step < cfg.max_steps {
                repair_attempts += 1;
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
                history.push(Message::new(
                    "user",
                    &format!(
                        "A file edit failed and has not been written to disk yet:\n{report}\nEmit a <tool_call> with `write_file` or `edit_file` now to apply the change before finishing."
                    ),
                ));
                continue;
            }
            if touched_files {
                match verify_workspace_work(&cfg.workdir, &modified_files, cfg.cmd_timeout_secs) {
                    Ok(v_rep) => {
                        let _ = on_event(AgentEvent::Verified {
                            step,
                            report: &v_rep,
                        });
                    }
                    Err(report) if repair_attempts < 2 && step < cfg.max_steps => {
                        repair_attempts += 1;
                        if !on_event(AgentEvent::ValidationRepair {
                            step,
                            report: &report,
                        }) {
                            return Err("cancelled".to_string());
                        }
                        history.push(Message::new(
                            "user",
                            &format!(
                                "Post-edit verification failed:\n{report}\nFix every reported issue using `edit_file` or `write_file` before finishing."
                            ),
                        ));
                        continue;
                    }
                    Err(_) => {}
                }
            }
            if cfg.expect_tool_call && !used_any_tools && repair_attempts < 2 && step < cfg.max_steps {
                repair_attempts += 1;
                history.push(Message::new(
                    "user",
                    "No tool call was emitted yet, so the requested action has not run. Emit a <tool_call> now (using `use_computer` for macOS app or file actions, or `write_file`/`edit_file`/`read_file`/`list_dir`/`run_command` for workspace actions) before replying.",
                ));
                continue;
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

        for (idx, tc) in comp.tool_calls.iter_mut().enumerate() {
            tc.id = format!("call_s{step}_{idx}");
        }
        history.push(Message::assistant_tool_calls(&comp.text, comp.tool_calls.clone()));

        for (idx, tc) in comp.tool_calls.iter().enumerate() {
            let is_file_mutation = matches!(tc.name.as_str(), "write_file" | "edit_file");
            if is_file_mutation {
                touched_files = true;
            }
            if !on_event(AgentEvent::ToolStart { step, call: tc }) {
                return Err("cancelled".to_string());
            }
            let outcome = execute_tool(&cfg.workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx);
            if is_file_mutation {
                let target_path = Json::parse(&tc.arguments)
                    .ok()
                    .and_then(|j| j.get("path").and_then(Json::as_str).map(str::to_string))
                    .unwrap_or_default();
                if outcome.ok {
                    unresolved_edits.retain(|(p, _)| p != &target_path);
                    if !target_path.is_empty() && !modified_files.iter().any(|p| p == &target_path) {
                        modified_files.push(target_path);
                    }
                } else {
                    let summary = outcome
                        .output
                        .lines()
                        .next()
                        .unwrap_or("edit failed")
                        .to_string();
                    if let Some(existing) = unresolved_edits.iter_mut().find(|(p, _)| p == &target_path) {
                        existing.1 = summary;
                    } else {
                        unresolved_edits.push((target_path, summary));
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

        let _ = fs::remove_dir_all(&dir);
    }
}
