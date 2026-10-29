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
    "You are Kronos Agent, a local software engineering runner on Apple Silicon. ",
    "Use the workspace tools (write_file, edit_file, read_file, list_dir, search_files, run_command) ",
    "to build, edit, and verify complete working files directly on disk. ",
    "Do not paste file drafts in prose before calling tools; emit <tool_call> invocations immediately. ",
    "Prefer edit_file for targeted modifications to existing files and write_file for new files. ",
    "If a tool response reports a validation error, fix it immediately before finishing."
);

#[derive(Clone, Debug)]
pub struct WorkspaceFileEntry {
    pub path: String,
    pub bytes: u64,
    pub validation_ok: bool,
    pub validation_msg: String,
}

#[derive(Clone, Debug)]
pub struct ToolExecOutcome {
    pub ok: bool,
    pub validation_ok: bool,
    pub validation_msg: String,
    pub output: String,
}

pub enum AgentEvent<'a> {
    StepStart {
        step: usize,
        max_steps: usize,
    },
    TextDelta(&'a str),
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
}

pub fn tool_definitions_json() -> Vec<String> {
    vec![
        r#"{"type":"function","function":{"name":"write_file","description":"Create or overwrite a UTF-8 file inside the workspace directory (parent directories are created automatically) and validate its syntax.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative file path inside the workspace, e.g. index.html or src/app.js"},"content":{"type":"string","description":"Complete file contents to write"}},"required":["path","content"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"edit_file","description":"Replace an exact unique substring in an existing UTF-8 workspace file and validate the updated file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative file path inside the workspace"},"old_text":{"type":"string","description":"Exact existing text block to replace (must match uniquely)"},"new_text":{"type":"string","description":"Replacement text block"}},"required":["path","old_text","new_text"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"read_file","description":"Read a UTF-8 file inside the workspace with 1-based line numbers and optional line range.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative file path inside the workspace"},"start_line":{"type":"integer","description":"Optional 1-based start line (inclusive)"},"end_line":{"type":"integer","description":"Optional 1-based end line (inclusive)"}}},"required":["path"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"list_dir","description":"List files and subdirectories at a relative path inside the workspace directory.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Relative directory path (use '.' for workspace root)"}}},"required":[]}}"#.to_string(),
        r#"{"type":"function","function":{"name":"search_files","description":"Search UTF-8 files in the workspace for a text pattern, returning `path:line: snippet` matches.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"Substring pattern to search for"},"path":{"type":"string","description":"Optional relative subdirectory or file to search (default '.')"},"case_insensitive":{"type":"boolean","description":"Optional case-insensitive match flag"}}},"required":["pattern"]}}}"#.to_string(),
        r#"{"type":"function","function":{"name":"run_command","description":"Execute a shell command (`sh -c`) inside the workspace directory with a timeout. Redirect stdout/stderr if starting a background server (e.g. `nohup python3 -m http.server 18123 > server.log 2>&1 &`).","parameters":{"type":"object","properties":{"cmd":{"type":"string","description":"Shell command line to run in the workspace"}}},"required":["cmd"]}}}"#.to_string(),
    ]
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
                };
            };
            let out = run_shell_command_with_timeout(workdir, cmd, cmd_timeout_secs, step_tag);
            let ok = out.starts_with("exit_code=0");
            ToolExecOutcome {
                ok,
                validation_ok: true,
                validation_msg: "ok".to_string(),
                output: out,
            }
        }
        other => ToolExecOutcome {
            ok: false,
            validation_ok: true,
            validation_msg: "ok".to_string(),
            output: format!("error: unknown tool `{other}`"),
        },
    }
}

fn attach_validation_status(workdir: &Path, target: &Path, rel: &str, base_msg: String) -> ToolExecOutcome {
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
            }
        }
        Err(err) => ToolExecOutcome {
            ok: true,
            validation_ok: false,
            validation_msg: err.clone(),
            output: format!("{base_msg}\n[validation error in `{rel}`: {err} — fix with `edit_file` or `write_file`]"),
        },
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
    match fs::write(&target, content.as_bytes()) {
        Ok(()) => {
            let msg = format!(
                "ok: wrote {} bytes ({} lines) to `{rel}`",
                content.len(),
                content.lines().count()
            );
            attach_validation_status(workdir, &target, rel, msg)
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
    if old_text.is_empty() {
        return err_outcome("error: `edit_file` `old_text` must not be empty");
    }
    let target = match resolve_jailed_path(workdir, rel) {
        Ok(p) => p,
        Err(e) => return err_outcome(&format!("error: {e}")),
    };
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
        // Fallback check for CRLF vs LF mismatch.
        let norm_existing = existing.replace("\r\n", "\n");
        let norm_old = old_text.replace("\r\n", "\n");
        if let Some(pos) = norm_existing.find(&norm_old) {
            if !norm_existing[pos + norm_old.len()..].contains(&norm_old) {
                let updated = format!(
                    "{}{}{}",
                    &norm_existing[..pos],
                    new_text,
                    &norm_existing[pos + norm_old.len()..]
                );
                return match fs::write(&target, updated.as_bytes()) {
                    Ok(()) => {
                        let (line_no, _) = line_col_at(&norm_existing, pos, 1);
                        let msg = format!("ok: edited `{rel}` at line {line_no} ({} bytes)", updated.len());
                        attach_validation_status(workdir, &target, rel, msg)
                    }
                    Err(e) => err_outcome(&format!("error writing `{rel}`: {e}")),
                };
            }
        }
        let first_old_line = old_text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
        let mut hint = String::new();
        if !first_old_line.is_empty() {
            for (idx, line) in existing.lines().enumerate() {
                if line.contains(first_old_line) {
                    hint = format!(" (nearest match for `{first_old_line}` is at line {}; call `read_file` to inspect exact text)", idx + 1);
                    break;
                }
            }
        }
        return err_outcome(&format!(
            "error: `old_text` not found in `{rel}`{hint}"
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
    match fs::write(&target, updated.as_bytes()) {
        Ok(()) => {
            let msg = format!(
                "ok: edited `{rel}` at line {line_no} ({} bytes, {} lines)",
                updated.len(),
                updated.lines().count()
            );
            attach_validation_status(workdir, &target, rel, msg)
        }
        Err(e) => err_outcome(&format!("error writing `{rel}`: {e}")),
    }
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
    }
}

fn err_outcome(msg: &str) -> ToolExecOutcome {
    ToolExecOutcome {
        ok: false,
        validation_ok: true,
        validation_msg: "ok".to_string(),
        output: msg.to_string(),
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
}

/// Runs an iterative tool-calling and validation repair loop while preserving the KV-cache prefix.
pub fn run_agent_turn(
    model: &mut ChatModel,
    history: &mut Vec<Message>,
    tools_json: &[String],
    cfg: &AgentTurnConfig,
    mut on_event: impl FnMut(AgentEvent<'_>) -> bool,
) -> Result<String, String> {
    if history.first().is_none_or(|m| m.role != "system") {
        history.insert(0, Message::new("system", AGENT_SYSTEM_PROMPT));
    }
    let t_start = Instant::now();
    let mut touched_files = false;
    let mut repair_attempts = 0usize;

    for step in 1..=cfg.max_steps {
        if !on_event(AgentEvent::StepStart {
            step,
            max_steps: cfg.max_steps,
        }) {
            return Err("cancelled".to_string());
        }

        let mut filter = ProseStreamFilter::new();
        let mut alive = true;
        let mut comp = model.generate_with_tools(history, tools_json, &cfg.params, |piece| {
            alive = filter.push(piece, |prose| on_event(AgentEvent::TextDelta(prose)));
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
            if touched_files && repair_attempts < 2 && step < cfg.max_steps {
                if let Err(report) = validate_workspace(&cfg.workdir) {
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
                            "Workspace validation failed:\n{report}\nFix every reported file using `edit_file` or `write_file` before finishing."
                        ),
                    ));
                    continue;
                }
            }
            let secs = t_start.elapsed().as_secs_f64();
            let _ = on_event(AgentEvent::Done {
                steps: step,
                total_secs: secs,
                final_text: &comp.text,
            });
            return Ok(comp.text);
        }

        for (idx, tc) in comp.tool_calls.iter_mut().enumerate() {
            tc.id = format!("call_s{step}_{idx}");
        }
        history.push(Message::assistant_tool_calls(&comp.text, comp.tool_calls.clone()));

        for (idx, tc) in comp.tool_calls.iter().enumerate() {
            if matches!(tc.name.as_str(), "write_file" | "edit_file") {
                touched_files = true;
            }
            if !on_event(AgentEvent::ToolStart { step, call: tc }) {
                return Err("cancelled".to_string());
            }
            let outcome = execute_tool(&cfg.workdir, tc, cfg.cmd_timeout_secs, step * 100 + idx);
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

        let _ = fs::remove_dir_all(&dir);
    }
}
