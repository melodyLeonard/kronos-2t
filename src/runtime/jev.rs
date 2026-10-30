#![forbid(unsafe_code)]
//! Single-forward-pass calibrated decision primitives, workspace symbol prefetch, and grounded rubric verification.

use crate::runtime::agent::{truncate_utf8, TurnRoute};
use crate::runtime::chat::ChatModel;
use crate::util::json::Json;
use std::fs;
use std::path::Path;
use std::time::Instant;
use crate::config::prompts::{JEV_CLASSIFY_SYSTEM, JEV_VERIFY_SYSTEM};



#[derive(Clone, Debug)]
pub struct CalibratedChoice {
    pub selected: usize,
    pub probs: Vec<f32>,
    pub entropy: f32,
    pub confidence: f32,
    pub margin: f32,
    pub used_neural: bool,
    pub elapsed_ms: f64,
}

impl CalibratedChoice {
    pub fn from_logits(
        logits: &[f32],
        temperature: f32,
        used_neural: bool,
        elapsed_ms: f64,
    ) -> Self {
        assert!(!logits.is_empty());
        let inv_t = 1.0f32 / temperature.max(0.05);
        let mut max_z = f32::NEG_INFINITY;
        for &z in logits {
            let s = z * inv_t;
            if s > max_z {
                max_z = s;
            }
        }
        let mut sum = 0.0f64;
        let mut probs = Vec::with_capacity(logits.len());
        for &z in logits {
            let e = (((z * inv_t) - max_z) as f64).exp();
            sum += e;
            probs.push(e as f32);
        }
        let norm = (sum.max(1e-12)) as f32;
        for p in &mut probs {
            *p /= norm;
        }
        let mut best_idx = 0usize;
        let mut top1 = -1.0f32;
        let mut top2 = -1.0f32;
        let mut h = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            if p > 1e-12 {
                h -= p * p.ln();
            }
            if p > top1 {
                top2 = top1;
                top1 = p;
                best_idx = i;
            } else if p > top2 {
                top2 = p;
            }
        }
        let max_h = (probs.len() as f32).ln().max(1e-6);
        let confidence = if probs.len() == 1 {
            1.0
        } else {
            (1.0 - (h / max_h)).clamp(0.0, 1.0)
        };
        let margin = if probs.len() == 1 {
            1.0
        } else {
            (top1 - top2.max(0.0)).clamp(0.0, 1.0)
        };
        Self {
            selected: best_idx,
            probs,
            entropy: h.max(0.0),
            confidence,
            margin,
            used_neural,
            elapsed_ms,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CalibratedBool {
    pub value: bool,
    pub prob_true: f32,
    pub entropy: f32,
    pub confidence: f32,
    pub used_neural: bool,
    pub elapsed_ms: f64,
}

impl CalibratedBool {
    pub fn from_log_odds(log_odds: f32, threshold: f32, used_neural: bool, elapsed_ms: f64) -> Self {
        let clamped = log_odds.clamp(-20.0, 20.0);
        let p = 1.0f32 / (1.0f32 + (-clamped).exp());
        let q = (1.0f32 - p).clamp(1e-9, 1.0);
        let pc = p.clamp(1e-9, 1.0);
        let h = -(pc * pc.ln() + q * q.ln());
        let confidence = (1.0 - h / std::f32::consts::LN_2).clamp(0.0, 1.0);
        Self {
            value: p >= threshold,
            prob_true: p,
            entropy: h.max(0.0),
            confidence,
            used_neural,
            elapsed_ms,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CalibratedScore {
    pub expected: f32,
    pub bins: [f32; 5],
    pub entropy: f32,
    pub confidence: f32,
    pub used_neural: bool,
    pub elapsed_ms: f64,
}

impl CalibratedScore {
    pub fn from_bin_logits(logits: [f32; 5], used_neural: bool, elapsed_ms: f64) -> Self {
        let choice = CalibratedChoice::from_logits(&logits, 1.0, used_neural, elapsed_ms);
        let mut bins = [0.0f32; 5];
        bins.copy_from_slice(&choice.probs[..5]);
        let centers = [0.0f32, 0.25, 0.50, 0.75, 1.0];
        let mut expected = 0.0f32;
        for (b, c) in bins.iter().zip(centers.iter()) {
            expected += b * c;
        }
        Self {
            expected: expected.clamp(0.0, 1.0),
            bins,
            entropy: choice.entropy,
            confidence: choice.confidence,
            used_neural,
            elapsed_ms,
        }
    }

    pub fn from_mean_and_concentration(
        mean: f32,
        concentration: f32,
        used_neural: bool,
        elapsed_ms: f64,
    ) -> Self {
        let m = mean.clamp(0.0, 1.0);
        let k = concentration.clamp(1.0, 24.0);
        let centers = [0.0f32, 0.25, 0.50, 0.75, 1.0];
        let mut logits = [0.0f32; 5];
        for (i, &c) in centers.iter().enumerate() {
            let d = m - c;
            logits[i] = -k * d * d * 4.0;
        }
        Self::from_bin_logits(logits, used_neural, elapsed_ms)
    }
}

#[inline]
pub fn fuse_binary_log_odds(
    prior_log_odds: f32,
    neural_yes_no: Option<(&[f32], f32)>,
    neural_temp: f32,
) -> (f32, bool) {
    if let Some((lp, mass)) = neural_yes_no {
        if lp.len() >= 2 {
            let reliability = (mass / 0.12).clamp(0.05, 1.0);
            let delta = ((lp[0] - lp[1]) / neural_temp.max(0.1)).clamp(-6.0, 6.0);
            return (prior_log_odds + reliability * delta, true);
        }
    }
    (prior_log_odds, false)
}

#[inline]
pub fn fuse_choice_logits(
    prior_logits: &[f32],
    neural_logprobs: Option<(&[f32], f32)>,
    neural_temp: f32,
) -> (Vec<f32>, bool) {
    if let Some((lp, mass)) = neural_logprobs {
        if lp.len() == prior_logits.len() && !lp.is_empty() {
            let reliability = (mass / 0.12).clamp(0.05, 1.0);
            let mean_lp = lp.iter().copied().sum::<f32>() / (lp.len() as f32);
            let inv_t = 1.0f32 / neural_temp.max(0.1);
            let fused = prior_logits
                .iter()
                .zip(lp.iter())
                .map(|(&pr, &nl)| pr + reliability * ((nl - mean_lp) * inv_t).clamp(-5.0, 5.0))
                .collect();
            return (fused, true);
        }
    }
    (prior_logits.to_vec(), false)
}

#[derive(Clone, Debug)]
pub struct IndexedFileSummary {
    pub rel_path: String,
    pub bytes: u64,
    pub line_count: usize,
    pub symbols: Vec<(usize, String)>,
}

#[derive(Clone, Debug, Default)]
pub struct WorkspaceSymbolIndex {
    pub files: Vec<IndexedFileSummary>,
}

impl WorkspaceSymbolIndex {
    pub fn build(workdir: &Path, max_files: usize) -> Self {
        let mut files = Vec::new();
        Self::walk(workdir, workdir, max_files.max(1), &mut files);
        files.sort_by(|a, b| a.rel_path.cmp(&b.rel_path));
        Self { files }
    }

    fn skip_name(name: &str) -> bool {
        name == ".git"
            || name == "node_modules"
            || name == "target"
            || name == "__pycache__"
            || name == ".DS_Store"
            || name == ".kronos_media"
            || name == crate::config::paths::STATE_DIR
            || name.starts_with(".kronos_cmd_")
            || name.ends_with(".k2tb")
            || name.ends_with(".k2tm")
            || name.ends_with(".k2di")
    }

    fn walk(root: &Path, cur: &Path, limit: usize, out: &mut Vec<IndexedFileSummary>) {
        if out.len() >= limit {
            return;
        }
        let Ok(rd) = fs::read_dir(cur) else {
            return;
        };
        let mut subdirs = Vec::new();
        for entry in rd.flatten() {
            if out.len() >= limit {
                break;
            }
            let name = entry.file_name().to_string_lossy().into_owned();
            if Self::skip_name(&name) {
                continue;
            }
            let Ok(meta) = entry.metadata() else {
                continue;
            };
            let path = entry.path();
            if meta.is_dir() {
                subdirs.push(path);
            } else if meta.is_file() {
                let Ok(rel) = path.strip_prefix(root) else {
                    continue;
                };
                let rel_path = rel.to_string_lossy().replace('\\', "/");
                let bytes = meta.len();
                let mut line_count = 0usize;
                let mut symbols = Vec::new();
                if bytes <= 128 * 1024 && is_indexable_text_ext(&path) {
                    if let Ok(text) = fs::read_to_string(&path) {
                        line_count = text.lines().count();
                        symbols = extract_file_symbols(&path, &text, 24);
                    }
                }
                out.push(IndexedFileSummary {
                    rel_path,
                    bytes,
                    line_count,
                    symbols,
                });
            }
        }
        for sub in subdirs {
            if out.len() >= limit {
                break;
            }
            Self::walk(root, &sub, limit, out);
        }
    }

    /// Pre-assembles a compact line-numbered workspace snapshot (`<= max_bytes`) for Turn 1.
    pub fn prefetch_context_block(
        &self,
        workdir: &Path,
        user_text: &str,
        max_bytes: usize,
    ) -> Option<(Vec<String>, String)> {
        if self.files.is_empty() || max_bytes < 256 {
            return None;
        }
        let lower = user_text.to_ascii_lowercase();
        let query_terms = extract_query_tokens(&lower);
        let wants_html = has_any(
            &lower,
            &[
                "html", "title", "heading", "header", "navbar", "hero", "footer", "section",
                "button", "form", "input", "table", "page", "modal", "sidebar",
            ],
        );
        let wants_css = has_any(
            &lower,
            &[
                "css", "style", "color", "theme", "dark", "light", "font", "layout", "grid",
                "flex", "padding", "margin", "center", "align", "gradient", "shadow", "rounded",
                "responsive", "bigger", "smaller",
            ],
        );
        let wants_code = has_any(
            &lower,
            &[
                "js", "script", "function", "click", "event", "filter", "search", "sort", "bug",
                "fix", "error", "broken", "state", "api", "fetch", "logic", "rust", "python",
            ],
        );

        let small_workspace = self.files.len() <= 4;
        let mut scored: Vec<(i32, &IndexedFileSummary)> = Vec::with_capacity(self.files.len());
        for f in &self.files {
            let path_l = f.rel_path.to_ascii_lowercase();
            let base_l = Path::new(&path_l)
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or(&path_l);
            let mut s = if small_workspace { 4 } else { 0 };
            if lower.contains(&path_l) || (!base_l.is_empty() && lower.contains(base_l)) {
                s += 18;
            }
            if wants_html && (path_l.ends_with(".html") || path_l.ends_with(".htm")) {
                s += 8;
            }
            if wants_css && path_l.ends_with(".css") {
                s += 8;
            }
            if wants_code
                && (path_l.ends_with(".js")
                    || path_l.ends_with(".ts")
                    || path_l.ends_with(".rs")
                    || path_l.ends_with(".py"))
            {
                s += 7;
            }
            for qt in &query_terms {
                if path_l.contains(qt.as_str()) {
                    s += 5;
                }
                if f.symbols
                    .iter()
                    .any(|(_, sym)| sym.to_ascii_lowercase().contains(qt.as_str()))
                {
                    s += 4;
                }
            }
            if s > 0 {
                scored.push((s, f));
            }
        }

        scored.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.rel_path.cmp(&b.1.rel_path)));
        let mut out = String::from(
            "[Workspace Context Snapshot — edit directly with edit_file or write_file without redundant discovery calls]\nFiles: ",
        );
        let listing: Vec<String> = self
            .files
            .iter()
            .take(16)
            .map(|f| format!("{} ({}B)", f.rel_path, f.bytes))
            .collect();
        out.push_str(&listing.join(", "));
        if self.files.len() > 16 {
            out.push_str(&format!(", +{} more", self.files.len() - 16));
        }
        out.push('\n');

        let mut picked = Vec::new();
        for (_, f) in scored.into_iter().take(3) {
            if out.len() + 160 >= max_bytes {
                break;
            }
            let full_path = workdir.join(&f.rel_path);
            let Ok(content) = fs::read_to_string(&full_path) else {
                continue;
            };
            let budget = ((max_bytes - out.len()) / 2).clamp(320, 1500);
            let excerpt = build_numbered_excerpt(&content, &f.symbols, &query_terms, budget);
            if excerpt.is_empty() {
                continue;
            }
            out.push_str(&format!("--- {} ({} lines) ---\n", f.rel_path, f.line_count));
            out.push_str(&excerpt);
            if !out.ends_with('\n') {
                out.push('\n');
            }
            picked.push(f.rel_path.clone());
        }

        if picked.is_empty() {
            None
        } else {
            Some((picked, truncate_utf8(&out, max_bytes)))
        }
    }
}

fn is_indexable_text_ext(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "rs" | "js"
            | "mjs"
            | "ts"
            | "tsx"
            | "jsx"
            | "py"
            | "html"
            | "htm"
            | "css"
            | "json"
            | "toml"
            | "md"
            | "txt"
            | "sh"
    )
}

fn extract_file_symbols(path: &Path, text: &str, max_syms: usize) -> Vec<(usize, String)> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut out = Vec::new();
    for (idx, raw) in text.lines().enumerate() {
        if out.len() >= max_syms {
            break;
        }
        let line_no = idx + 1;
        let t = raw.trim();
        if t.is_empty() || t.starts_with("//") || t.starts_with("/*") || t.starts_with('*') {
            continue;
        }
        let keep = match ext.as_str() {
            "rs" => {
                t.starts_with("pub fn ")
                    || t.starts_with("fn ")
                    || t.starts_with("pub struct ")
                    || t.starts_with("struct ")
                    || t.starts_with("pub enum ")
                    || t.starts_with("enum ")
                    || t.starts_with("impl ")
                    || t.starts_with("mod ")
            }
            "js" | "mjs" | "ts" | "tsx" | "jsx" => {
                t.starts_with("function ")
                    || t.starts_with("async function ")
                    || t.starts_with("export function ")
                    || t.starts_with("export const ")
                    || t.starts_with("export class ")
                    || t.starts_with("class ")
                    || (t.starts_with("const ") && (t.contains("=>") || t.contains("document.")))
            }
            "py" => t.starts_with("def ") || t.starts_with("async def ") || t.starts_with("class "),
            "html" | "htm" => {
                t.contains("<title")
                    || t.contains("<h1")
                    || t.contains("<h2")
                    || t.contains("<nav")
                    || t.contains("<main")
                    || t.contains("id=\"")
                    || t.contains("<script ")
                    || t.contains("<link ")
            }
            "css" => {
                t.ends_with('{')
                    && (t.starts_with(":root")
                        || t.starts_with("body")
                        || t.starts_with('.')
                        || t.starts_with('#')
                        || t.starts_with("@media"))
            }
            _ => false,
        };
        if keep {
            out.push((line_no, truncate_utf8(t, 96)));
        }
    }
    out
}

fn build_numbered_excerpt(
    content: &str,
    symbols: &[(usize, String)],
    query_terms: &[String],
    max_bytes: usize,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return String::new();
    }
    if content.len() + lines.len() * 6 <= max_bytes {
        let mut full = String::with_capacity(content.len() + lines.len() * 6);
        for (i, l) in lines.iter().enumerate() {
            let entry = format!("{:>4}: {}\n", i + 1, l);
            if full.len() + entry.len() > max_bytes {
                break;
            }
            full.push_str(&entry);
        }
        return full;
    }

    let mut focus_lines: Vec<usize> = vec![1];
    for (ln, sym) in symbols {
        let sl = sym.to_ascii_lowercase();
        if query_terms.iter().any(|q| sl.contains(q.as_str())) || focus_lines.len() < 4 {
            focus_lines.push(*ln);
        }
    }
    for (idx, raw) in lines.iter().enumerate() {
        if focus_lines.len() >= 8 {
            break;
        }
        let rl = raw.to_ascii_lowercase();
        if query_terms.iter().any(|q| rl.contains(q.as_str())) {
            focus_lines.push(idx + 1);
        }
    }
    focus_lines.sort_unstable();
    focus_lines.dedup();

    let mut included = vec![false; lines.len()];
    for center in focus_lines {
        let lo = center.saturating_sub(3).max(1);
        let hi = (center + 5).min(lines.len());
        for ln in lo..=hi {
            included[ln - 1] = true;
        }
    }

    let mut out = String::new();
    let mut prev_ln = 0usize;
    for (idx, &keep) in included.iter().enumerate() {
        if !keep {
            continue;
        }
        let ln = idx + 1;
        if prev_ln > 0 && ln > prev_ln + 1 {
            out.push_str("   ...\n");
        }
        let entry = format!("{:>4}: {}\n", ln, truncate_utf8(lines[idx], 140));
        if out.len() + entry.len() > max_bytes {
            break;
        }
        out.push_str(&entry);
        prev_ln = ln;
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArtifactKind {
    WebApp,
    PythonScript,
    RustCode,
    JsModule,
    ReadableScript,
    AudioMedia,
    VideoMedia,
}

impl ArtifactKind {
    pub fn label(self) -> &'static str {
        match self {
            ArtifactKind::WebApp => "web interface (.html + styles)",
            ArtifactKind::PythonScript => "Python script (.py)",
            ArtifactKind::RustCode => "Rust source (.rs)",
            ArtifactKind::JsModule => "JavaScript/TypeScript module (.js/.ts)",
            ArtifactKind::ReadableScript => "full narration script (.md/.txt, >= 3.2 KB spoken prose)",
            ArtifactKind::AudioMedia => "synthesized audio file (.wav/.m4a/.mp3/.flac) on disk",
            ArtifactKind::VideoMedia => "rendered video file (.mp4/.mov/.webm) on disk",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PredicateKind {
    ToolExecuted,
    WorkspaceSwitched { folder_hint: String },
    ExpectedArtifact { kind: ArtifactKind },
    TargetFilePresent { name_hint: String },
    DomainConceptsPresent { terms: Vec<String> },
    ModularWebArchitecture,
    SelfTestedAfterEdits,
    DeterministicOraclesClean,
}

#[derive(Clone, Debug)]
pub struct RequirementPredicate {
    pub id: String,
    pub description: String,
    pub threshold: f32,
    pub kind: PredicateKind,
}

#[derive(Clone, Debug)]
pub struct EvaluatedPredicate {
    pub id: String,
    pub description: String,
    pub threshold: f32,
    pub result: CalibratedBool,
}

#[derive(Clone, Debug)]
pub struct JevRubricEvaluation {
    pub passed: bool,
    pub score: CalibratedScore,
    pub predicates: Vec<EvaluatedPredicate>,
    pub unmet_summary: String,
    pub ungrounded_claim: bool,
    pub elapsed_ms: f64,
}

#[derive(Clone, Debug)]
pub struct JevPrepassPlan {
    pub route: TurnRoute,
    pub route_choice: CalibratedChoice,
    pub selected_tool_names: Vec<String>,
    pub active_tools_json: Vec<String>,
    pub pruned_tool_count: usize,
    pub prefetched_files: Vec<String>,
    pub prefetch_block: Option<String>,
    pub rubric: Vec<RequirementPredicate>,
    pub elapsed_ms: f64,
}

/// Computes structural log-odds over `[TurnRoute::Chat, TurnRoute::Agent, TurnRoute::Computer]`.
pub fn structural_route_logits(
    user_text: &str,
    has_images: bool,
    workspace_has_files: bool,
    history_used_tools: bool,
    allow_computer_control: bool,
) -> [f32; 3] {
    let lower = user_text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return [3.5, -2.5, -3.5];
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
        "who built you",
        "what is your name",
        "what can you do",
        "help",
    ];
    let stripped = lower.trim_end_matches(['?', '.', ',']);
    if greetings.contains(&stripped) {
        return [4.2, -3.0, -4.0];
    }

    if allow_computer_control && crate::runtime::device::is_computer_control_prompt(&lower) {
        return [-2.4, -1.6, 3.8];
    }
    if crate::runtime::device::is_workspace_folder_target_query(&lower) {
        return [-2.5, 4.0, -2.8];
    }

    let mut chat_z = 0.6f32;
    let mut agent_z = -0.4f32;
    let comp_z = -3.2f32;

    let file_markers = [
        ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".py", ".rs", ".json", ".svg", ".md",
        ".toml", "index.", "style.", "app.", "src/", "./",
    ];
    if file_markers.iter().any(|m| lower.contains(m)) {
        agent_z += 3.0;
        chat_z -= 1.5;
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
    if action_phrases.iter().any(|p| {
        lower.starts_with(p)
            || lower.contains(&format!(" {p}"))
            || lower.contains(&format!(", {p}"))
    }) {
        agent_z += 2.2;
        chat_z -= 1.0;
    }

    let artifact_nouns = [
        "webpage",
        "web page",
        "website",
        "web app",
        "landing page",
        "dashboard",
        "portal",
        "software",
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
        agent_z += 1.2;
    }

    if workspace_has_files || history_used_tools {
        let tweak_terms = [
            "darker",
            "lighter",
            "bigger",
            "smaller",
            "rounded",
            "modern",
            "cleaner",
            "gradient",
            "shadow",
            "padding",
            "margin",
            "font",
            "color",
            "colors",
            "theme",
            "layout",
            "responsive",
            "match this",
            "like this",
            "still the same",
            "not working",
            "doesn't work",
            "doesnt work",
            "didn't change",
            "didnt change",
            "nothing changed",
            "don't see",
            "dont see",
        ];
        if agent_z > 0.6 || tweak_terms.iter().any(|t| lower.contains(t)) {
            agent_z += 1.8;
            chat_z -= 0.8;
        }
    }

    if has_images
        && has_any(
            &lower,
            &[
                "build",
                "clone",
                "recreate",
                "replicate",
                "match",
                "style",
                "like this",
                "fix",
                "embed",
                "use this",
                "turn this into",
                "implement",
            ],
        )
    {
        agent_z += 2.6;
        chat_z -= 1.2;
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
        chat_z += 3.0;
        agent_z -= 2.4;
    }

    [chat_z, agent_z, comp_z]
}

/// Strips system-injected `[Device Context ...][/Device Context]` and `[Attached Image ...][/Attached Image]`
/// blocks so route classification, tool pruning, and rubric filename extraction inspect only the user's own prompt.
fn extract_raw_user_goal(s: &str) -> &str {
    let mut cur = s.trim();
    loop {
        if cur.starts_with("[Device Context") {
            if let Some(end) = cur.find("[/Device Context]") {
                cur = cur[end + "[/Device Context]".len()..].trim_start();
                continue;
            }
        }
        if cur.starts_with("[Attached Image") {
            if let Some(end) = cur.find("[/Attached Image]") {
                cur = cur[end + "[/Attached Image]".len()..].trim_start();
                continue;
            }
        }
        break;
    }
    for tail_marker in ["\n\n[Device Context", "\n[Device Context", "\n\n[Local Conversations", "\n[Local Conversations"] {
        if let Some(pos) = cur.find(tail_marker) {
            if pos > 0 {
                cur = cur[..pos].trim_end();
            }
        }
    }
    if cur.is_empty() {
        s.trim()
    } else {
        cur
    }
}

/// Runs the Turn-1 single-pass Jev pre-planner: calibrated route selection, KV-aware tool schema
/// selection, workspace file/symbol pre-fetching, and atomic requirement rubric decomposition.
#[allow(clippy::too_many_arguments)]
pub fn plan_turn_with_jev(
    mut model: Option<&mut ChatModel>,
    system_prompt: &str,
    user_text: &str,
    forced_route: Option<TurnRoute>,
    has_images: bool,
    workdir: &Path,
    history_used_tools: bool,
    full_tools_json: &[String],
    allow_computer_control: bool,
    recent_history_has_prefetch: bool,
) -> JevPrepassPlan {
    let user_text = extract_raw_user_goal(user_text);
    let t0 = Instant::now();
    let ws_index = WorkspaceSymbolIndex::build(workdir, 192);
    let workspace_has_files = !ws_index.files.is_empty();

    let route_choice = if let Some(r) = forced_route {
        let logits = match r {
            TurnRoute::Chat => [6.0, -4.0, -4.0],
            TurnRoute::Agent => [-4.0, 6.0, -4.0],
            TurnRoute::Computer => [-4.0, -4.0, 6.0],
        };
        CalibratedChoice::from_logits(&logits, 1.0, false, t0.elapsed().as_secs_f64() * 1000.0)
    } else {
        let prior = structural_route_logits(
            user_text,
            has_images,
            workspace_has_files,
            history_used_tools,
            allow_computer_control,
        );
        let prelim = CalibratedChoice::from_logits(&prior, 1.0, false, 0.0);
        if prelim.confidence < 0.68 && prelim.margin < 0.42 {
            let neural = model.as_deref_mut().and_then(|m| {
                let q = format!(
                    "Labels: Chat (answer directly), Code (create/edit/run project files), Mac (open/control macOS app).\nWorkspace has files: {workspace_has_files}.\nRequest: {}\nLabel:",
                    truncate_utf8(user_text.trim(), 320)
                );
                m.jev_score_anchors(JEV_CLASSIFY_SYSTEM, &q, &["Chat", "Code", "Mac"])
            });
            let (fused, used_neural) = fuse_choice_logits(
                &prior,
                neural.as_ref().map(|(v, mass)| (v.as_slice(), *mass)),
                1.25,
            );
            let mut adj = fused;
            if !allow_computer_control && adj.len() >= 3 {
                adj[2] = -8.0;
            }
            CalibratedChoice::from_logits(
                &adj,
                1.0,
                used_neural,
                t0.elapsed().as_secs_f64() * 1000.0,
            )
        } else {
            CalibratedChoice::from_logits(
                &prior,
                1.0,
                false,
                t0.elapsed().as_secs_f64() * 1000.0,
            )
        }
    };

    let route = match route_choice.selected {
        1 => TurnRoute::Agent,
        2 => TurnRoute::Computer,
        _ => TurnRoute::Chat,
    };

    let (selected_tool_names, active_tools_json, pruned_tool_count) = select_tools_kv_aware(
        model.as_deref(),
        system_prompt,
        user_text,
        route,
        workspace_has_files,
        full_tools_json,
    );

    let (prefetched_files, prefetch_block) =
        if route == TurnRoute::Agent && workspace_has_files && !recent_history_has_prefetch {
            match ws_index.prefetch_context_block(workdir, user_text, 3400) {
                Some((files, block)) => (files, Some(block)),
                None => (Vec::new(), None),
            }
        } else {
            (Vec::new(), None)
        };

    let rubric = extract_requirement_rubric(user_text, route, workspace_has_files);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    JevPrepassPlan {
        route,
        route_choice,
        selected_tool_names,
        active_tools_json,
        pruned_tool_count,
        prefetched_files,
        prefetch_block,
        rubric,
        elapsed_ms,
    }
}

fn tool_name_from_schema(schema: &str) -> Option<String> {
    let parsed = Json::parse(schema).ok()?;
    parsed
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(Json::as_str)
        .or_else(|| parsed.get("name").and_then(Json::as_str))
        .map(str::to_string)
}

fn select_tools_kv_aware(
    model: Option<&ChatModel>,
    system_prompt: &str,
    user_text: &str,
    route: TurnRoute,
    workspace_has_files: bool,
    full_tools_json: &[String],
) -> (Vec<String>, Vec<String>, usize) {
    if route == TurnRoute::Chat || full_tools_json.is_empty() {
        return (Vec::new(), Vec::new(), full_tools_json.len());
    }
    if route == TurnRoute::Computer {
        let mut names = Vec::new();
        let mut kept = Vec::new();
        for t in full_tools_json {
            let n = tool_name_from_schema(t).unwrap_or_default();
            if n == "use_computer" {
                names.push(n);
                kept.push(t.clone());
            }
        }
        if !kept.is_empty() {
            let pruned = full_tools_json.len().saturating_sub(kept.len());
            return (names, kept, pruned);
        }
    }

    let lower = user_text.to_ascii_lowercase();
    let wants_computer = crate::runtime::device::is_computer_control_prompt(&lower);

    // Preserve a pre-warmed target KV prefix on large-context sessions (`0 ms` reuse)
    // when no extraneous computer-control schema would pollute a workspace turn.
    let has_computer_schema = full_tools_json
        .iter()
        .any(|s| tool_name_from_schema(s).as_deref() == Some("use_computer"));
    let hot_prefix = (!has_computer_schema || wants_computer)
        && model.is_some_and(|m| {
            m.context_size() > 4096 && m.has_hot_prefix(system_prompt, full_tools_json)
        });
    if hot_prefix {
        let names = full_tools_json
            .iter()
            .filter_map(|s| tool_name_from_schema(s))
            .collect();
        return (names, full_tools_json.to_vec(), 0);
    }

    let wants_switch = crate::runtime::device::is_workspace_folder_target_query(&lower)
        || (has_any(&lower, &["new folder", "new directory", "new project", "switch workspace", "orgproj"]));
    let wants_cmd = has_any(
        &lower,
        &[
            "run ",
            "execute",
            "test",
            "verify",
            "verification",
            "check ",
            "validate",
            "assert",
            "modular",
            "refactor",
            "split ",
            "screenshot",
            "headless",
            "browser",
            "smoke",
            "suite",
            "iterate",
            "cargo",
            "pytest",
            "npm ",
            "node ",
            "python",
            "compile",
            "build ",
            "server",
            "curl",
            "git ",
            "command",
            "script",
            "video",
            "audio",
            "voice",
            "ffmpeg",
            "mlx",
            "tts",
            "render",
            "use tools",
            "venv",
        ],
    );
    let wants_search = workspace_has_files
        && has_any(
            &lower,
            &[
                "find",
                "search",
                "where",
                "grep",
                "locate",
                "across",
                "modular",
                "refactor",
                "split ",
                "monolith",
                "inspect",
                "audit",
                "structure",
            ],
        );

    let mut names = Vec::new();
    let mut kept = Vec::new();
    for schema in full_tools_json {
        let Some(name) = tool_name_from_schema(schema) else {
            kept.push(schema.clone());
            continue;
        };
        let keep = match name.as_str() {
            "write_file" => true,
            "edit_file" | "read_file" => workspace_has_files || wants_switch || wants_cmd,
            "list_dir" => (workspace_has_files && wants_search) || wants_cmd,
            "search_files" => wants_search,
            "switch_workspace" => wants_switch,
            "run_command" => wants_cmd,
            "use_computer" => wants_computer,
            _ => true,
        };
        if keep {
            names.push(name);
            kept.push(schema.clone());
        }
    }
    let pruned = full_tools_json.len().saturating_sub(kept.len());
    (names, kept, pruned)
}

/// Decomposes a user prompt into grounded, verifiable atomic predicates.
pub fn extract_requirement_rubric(
    user_text: &str,
    route: TurnRoute,
    workspace_has_files: bool,
) -> Vec<RequirementPredicate> {
    let trimmed = extract_raw_user_goal(user_text);
    if trimmed.is_empty() || route == TurnRoute::Chat {
        return Vec::new();
    }
    let lower = trimmed.to_ascii_lowercase();
    let mut preds = Vec::new();

    if route == TurnRoute::Computer {
        preds.push(RequirementPredicate {
            id: "tool_exec".to_string(),
            description: "execute the requested macOS action via `use_computer`".to_string(),
            threshold: 0.65,
            kind: PredicateKind::ToolExecuted,
        });
        return preds;
    }

    let is_action = has_any(
        &lower,
        &[
            "create", "build", "make", "write", "edit", "update", "modify", "change", "fix",
            "repair", "add", "remove", "delete", "style", "restyle", "redesign", "implement",
            "refactor", "modularize", "split", "extract", "separate", "run", "execute", "center",
            "align", "test", "verify",
        ],
    );
    if is_action {
        preds.push(RequirementPredicate {
            id: "tool_exec".to_string(),
            description: "execute workspace tool calls on disk rather than only describing changes"
                .to_string(),
            threshold: 0.70,
            kind: PredicateKind::ToolExecuted,
        });
    }

    if let Some(folder_hint) = extract_target_folder_hint(trimmed) {
        preds.push(RequirementPredicate {
            id: "workspace_dir".to_string(),
            description: format!(
                "switch active workspace into requested folder `{folder_hint}`"
            ),
            threshold: 0.65,
            kind: PredicateKind::WorkspaceSwitched { folder_hint },
        });
    }

    for (idx, kind) in detect_expected_artifact_kinds(&lower, workspace_has_files)
        .into_iter()
        .enumerate()
    {
        preds.push(RequirementPredicate {
            id: format!("artifact_kind_{idx}"),
            description: format!("produce complete {}", kind.label()),
            threshold: 0.65,
            kind: PredicateKind::ExpectedArtifact { kind },
        });
    }

    let wants_modular_web = has_any(
        &lower,
        &[
            "modularize",
            "modular",
            "split into",
            "separate css",
            "separate js",
            "separate files",
            "es module",
            "es modules",
            "break up the monolith",
            "refactor the monolith",
        ],
    ) && has_any(
        &lower,
        &[
            "website",
            "web page",
            "webpage",
            "web app",
            "dashboard",
            "portal",
            "html",
            "css",
            "index.html",
            "monolith",
            "page",
        ],
    );
    if wants_modular_web {
        preds.push(RequirementPredicate {
            id: "modular_web".to_string(),
            description: "extract inline monolith code into modular external `.css` stylesheets and `.js` modules linked from clean HTML".to_string(),
            threshold: 0.70,
            kind: PredicateKind::ModularWebArchitecture,
        });
    }

    let wants_self_test = is_action
        && has_any(
            &lower,
            &[
                "test its own",
                "test your own",
                "test your work",
                "test the work",
                "test afterwards",
                "write tests",
                "write a test",
                "run tests",
                "run a test",
                "test suite",
                "test script",
                "verification script",
                "automated test",
                "self-test",
                "and test it",
                "then test",
            ],
        );
    if wants_self_test {
        preds.push(RequirementPredicate {
            id: "self_tested".to_string(),
            description: "execute an automated test or verification command via `run_command` (`exit_code=0`) after the latest file edit".to_string(),
            threshold: 0.75,
            kind: PredicateKind::SelfTestedAfterEdits,
        });
    }

    for explicit_file in extract_explicit_filenames(trimmed) {
        preds.push(RequirementPredicate {
            id: format!("file_{explicit_file}"),
            description: format!("ensure `{explicit_file}` exists and passes validation"),
            threshold: 0.65,
            kind: PredicateKind::TargetFilePresent {
                name_hint: explicit_file,
            },
        });
    }

    let domain_terms = extract_domain_requirement_terms(&lower);
    if !domain_terms.is_empty() && is_action {
        preds.push(RequirementPredicate {
            id: "domain_features".to_string(),
            description: format!(
                "reflect requested domain elements ({}) in workspace files",
                domain_terms.join(", ")
            ),
            threshold: 0.55,
            kind: PredicateKind::DomainConceptsPresent { terms: domain_terms },
        });
    }

    let asks_kronos_technical_architecture = lower.contains("kronos")
        && has_any(
            &lower,
            &[
                "iteration",
                "math",
                "maths",
                "technical",
                "architecture",
                "compares",
                "compare",
            ],
        );
    if asks_kronos_technical_architecture && is_action {
        preds.push(RequirementPredicate {
            id: "kronos_grounded_math".to_string(),
            description: "ground technical explanation in real Kronos engine mathematics and benchmarks (Hadamard rotation, Cholesky GPTQ, NF4/E8Q2, Four-Russians LUT-GEMV, Kronos-Jev, and llama.cpp comparisons)".to_string(),
            threshold: 0.72,
            kind: PredicateKind::DomainConceptsPresent {
                terms: vec![
                    "hadamard".to_string(),
                    "gptq".to_string(),
                    "nf4".to_string(),
                    "lut".to_string(),
                    "jev".to_string(),
                    "llama.cpp".to_string(),
                ],
            },
        });
    }

    preds.push(RequirementPredicate {
        id: "oracles_clean".to_string(),
        description: "pass all syntax, asset/import resolution, anti-stub, and runtime checks"
            .to_string(),
        threshold: 0.75,
        kind: PredicateKind::DeterministicOraclesClean,
    });

    preds
}

/// Detects first-person past-tense claims of file or app execution when zero tools ran.
pub fn detect_ungrounded_action_claim(reply_text: &str, used_any_tools: bool) -> bool {
    if used_any_tools {
        return false;
    }
    let lower = reply_text.to_ascii_lowercase();
    let claim_patterns = [
        "i have created ",
        "i've created ",
        "i created the file",
        "i created `",
        "i have written ",
        "i've written ",
        "i have updated ",
        "i've updated ",
        "i updated `",
        "i modified `",
        "i have saved ",
        "i've saved ",
        "i have opened ",
        "i've opened ",
        "i launched ",
    ];
    claim_patterns.iter().any(|p| lower.contains(p))
}

/// Evaluates all atomic requirement predicates against physical workspace state and deterministic oracles.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rubric(
    model: Option<&mut ChatModel>,
    user_goal: &str,
    rubric: &[RequirementPredicate],
    workdir: &Path,
    modified_files: &[String],
    used_any_tools: bool,
    oracle_error: Option<&str>,
    reply_text: &str,
) -> JevRubricEvaluation {
    evaluate_rubric_with_telemetry(
        model,
        user_goal,
        rubric,
        workdir,
        modified_files,
        used_any_tools,
        true,
        oracle_error,
        reply_text,
    )
}

/// Evaluates all atomic requirement predicates with turn-level command verification telemetry.
#[allow(clippy::too_many_arguments)]
pub fn evaluate_rubric_with_telemetry(
    mut model: Option<&mut ChatModel>,
    user_goal: &str,
    rubric: &[RequirementPredicate],
    workdir: &Path,
    modified_files: &[String],
    used_any_tools: bool,
    tested_after_last_edit: bool,
    oracle_error: Option<&str>,
    reply_text: &str,
) -> JevRubricEvaluation {
    let t0 = Instant::now();
    let ungrounded_claim = detect_ungrounded_action_claim(reply_text, used_any_tools);
    if rubric.is_empty() && !ungrounded_claim && oracle_error.is_none() {
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        return JevRubricEvaluation {
            passed: true,
            score: CalibratedScore::from_mean_and_concentration(0.98, 16.0, false, elapsed_ms),
            predicates: Vec::new(),
            unmet_summary: String::new(),
            ungrounded_claim: false,
            elapsed_ms,
        };
    }

    let ws_index = WorkspaceSymbolIndex::build(workdir, 192);
    let combined_corpus = load_workspace_text_corpus(workdir, &ws_index, modified_files, 48 * 1024);
    let workdir_lower = workdir.to_string_lossy().to_ascii_lowercase();
    let mut evaluated = Vec::with_capacity(rubric.len() + 1);
    let mut any_neural = false;

    if ungrounded_claim {
        evaluated.push(EvaluatedPredicate {
            id: "grounded_claims".to_string(),
            description: "execute real tool calls before claiming files or apps were modified"
                .to_string(),
            threshold: 0.70,
            result: CalibratedBool::from_log_odds(-4.5, 0.70, false, 0.0),
        });
    }

    let mut artifact_notes: Vec<String> = Vec::new();
    for pred in rubric {
        let p_t0 = Instant::now();
        let eval_bool = match &pred.kind {
            PredicateKind::ToolExecuted => {
                let lo = if used_any_tools { 4.2 } else { -4.2 };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::WorkspaceSwitched { folder_hint } => {
                let hint_l = folder_hint.to_ascii_lowercase();
                let matched = workdir_lower.contains(&hint_l);
                let lo = if matched { 4.0 } else { -3.6 };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::ExpectedArtifact { kind } => {
                let sat = match check_artifact_kind_with_note(
                    workdir,
                    *kind,
                    &ws_index,
                    &combined_corpus,
                ) {
                    Ok(()) => true,
                    Err(note) => {
                        if !note.is_empty() {
                            artifact_notes.push(format!("{}: {note}", pred.description));
                        }
                        false
                    }
                };
                let lo = if sat { 3.8 } else { -3.8 };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::TargetFilePresent { name_hint } => {
                let nl = name_hint.to_ascii_lowercase();
                let exists = ws_index.files.iter().any(|f| {
                    let fl = f.rel_path.to_ascii_lowercase();
                    (fl == nl || fl.ends_with(&format!("/{nl}"))) && f.bytes > 0
                });
                let lo = if exists { 3.8 } else { -3.8 };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::DomainConceptsPresent { terms } => {
                let hits = terms
                    .iter()
                    .filter(|t| term_matches_corpus(t, &combined_corpus, &workdir_lower))
                    .count();
                let ratio = if terms.is_empty() {
                    1.0
                } else {
                    (hits as f32) / (terms.len() as f32)
                };
                let prior_lo = if combined_corpus.is_empty() {
                    -3.5
                } else if ratio >= 0.75 || hits >= 3 {
                    3.2
                } else if ratio >= 0.45 {
                    0.9
                } else if hits >= 1 {
                    -0.3
                } else {
                    -2.4
                };
                let (fused_lo, used_n) =
                    if (-1.4..=1.8).contains(&prior_lo) && !combined_corpus.is_empty() {
                        let neural = model.as_deref_mut().and_then(|m| {
                            let q = format!(
                                "Workspace excerpt:\n{}\n\nUser goal: {}\nQuestion: Does the workspace excerpt above implement the requested domain features ({})? Answer Yes or No.",
                                truncate_utf8(&combined_corpus, 1100),
                                truncate_utf8(user_goal, 260),
                                terms.join(", "),
                            );
                            m.jev_score_anchors(JEV_VERIFY_SYSTEM, &q, &["Yes", "No"])
                        });
                        fuse_binary_log_odds(
                            prior_lo,
                            neural.as_ref().map(|(v, mass)| (v.as_slice(), *mass)),
                            1.3,
                        )
                    } else {
                        (prior_lo, false)
                    };
                any_neural |= used_n;
                CalibratedBool::from_log_odds(
                    fused_lo,
                    pred.threshold,
                    used_n,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::ModularWebArchitecture => {
                let sat = check_modular_web_architecture_satisfied(workdir, &ws_index);
                let lo = if sat { 4.2 } else { -4.2 };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::SelfTestedAfterEdits => {
                let lo = if used_any_tools && tested_after_last_edit {
                    4.4
                } else {
                    -4.4
                };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
            PredicateKind::DeterministicOraclesClean => {
                let lo = if oracle_error.is_none() { 4.6 } else { -4.6 };
                CalibratedBool::from_log_odds(
                    lo,
                    pred.threshold,
                    false,
                    p_t0.elapsed().as_secs_f64() * 1000.0,
                )
            }
        };
        evaluated.push(EvaluatedPredicate {
            id: pred.id.clone(),
            description: pred.description.clone(),
            threshold: pred.threshold,
            result: eval_bool,
        });
    }

    let mut unmet_lines = Vec::new();
    let mut prob_sum = 0.0f32;
    let mut min_margin_ok = true;
    for ep in &evaluated {
        prob_sum += ep.result.prob_true;
        if ep.result.prob_true < ep.threshold {
            min_margin_ok = false;
            unmet_lines.push(format!(
                "- {} (p={:.2} < {:.2})",
                ep.description, ep.result.prob_true, ep.threshold
            ));
        }
    }
    if let Some(err) = oracle_error {
        min_margin_ok = false;
        unmet_lines.push(format!("- deterministic verification error:\n{}", truncate_utf8(err, 900)));
    }
    for note in &artifact_notes {
        unmet_lines.push(format!("- {note}"));
    }

    let mean_p = if evaluated.is_empty() {
        if oracle_error.is_none() { 0.96 } else { 0.25 }
    } else {
        prob_sum / (evaluated.len() as f32)
    };
    let concentration = if min_margin_ok { 14.0 } else { 8.0 };
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let score = CalibratedScore::from_mean_and_concentration(
        mean_p,
        concentration,
        any_neural,
        elapsed_ms,
    );
    let passed = min_margin_ok && oracle_error.is_none() && score.expected >= 0.72;

    JevRubricEvaluation {
        passed,
        score,
        predicates: evaluated,
        unmet_summary: unmet_lines.join("\n"),
        ungrounded_claim,
        elapsed_ms,
    }
}

fn measure_inline_tag_bytes(html: &str, tag_name: &str) -> usize {
    let lower = html.to_ascii_lowercase();
    let open_pat = format!("<{tag_name}");
    let close_pat = format!("</{tag_name}>");
    let mut total = 0usize;
    let mut cur = 0usize;
    while let Some(pos) = lower[cur..].find(&open_pat) {
        let open_start = cur + pos;
        let Some(gt_rel) = lower[open_start..].find('>') else {
            break;
        };
        let tag_header = &lower[open_start..open_start + gt_rel + 1];
        let body_start = open_start + gt_rel + 1;
        let Some(end_rel) = lower[body_start..].find(&close_pat) else {
            break;
        };
        if !(tag_name == "script" && tag_header.contains("src=")) {
            total += html[body_start..body_start + end_rel].trim().len();
        }
        cur = body_start + end_rel + close_pat.len();
    }
    total
}

fn check_modular_web_architecture_satisfied(
    workdir: &Path,
    ws_index: &WorkspaceSymbolIndex,
) -> bool {
    let mut html_paths = Vec::new();
    let mut css_count = 0usize;
    let mut app_js_count = 0usize;
    for f in &ws_index.files {
        let p = f.rel_path.to_ascii_lowercase();
        if (p.ends_with(".html") || p.ends_with(".htm")) && f.bytes >= 180 {
            html_paths.push(f.rel_path.as_str());
        } else if p.ends_with(".css") && f.bytes >= 100 {
            css_count += 1;
        } else if (p.ends_with(".js") || p.ends_with(".mjs")) && f.bytes >= 60 {
            let fname = Path::new(&p)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(&p);
            let is_test_helper = p.starts_with("test")
                || p.contains("/test")
                || fname.contains("test")
                || fname.contains("verify")
                || fname.contains("spec")
                || fname.contains("check");
            if !is_test_helper {
                app_js_count += 1;
            }
        }
    }
    if html_paths.is_empty() || css_count == 0 || app_js_count == 0 || (css_count + app_js_count) < 3 {
        return false;
    }
    let primary_html = html_paths
        .iter()
        .copied()
        .find(|p| p.eq_ignore_ascii_case("index.html"))
        .unwrap_or(html_paths[0]);
    let Ok(html_src) = fs::read_to_string(workdir.join(primary_html)) else {
        return false;
    };
    let html_lower = html_src.to_ascii_lowercase();
    let links_css = html_lower.contains(".css");
    let links_js = html_lower.contains("<script") && html_lower.contains("src=");
    let inline_style_bytes = measure_inline_tag_bytes(&html_src, "style");
    let inline_script_bytes = measure_inline_tag_bytes(&html_src, "script");
    links_css && links_js && inline_style_bytes <= 1200 && inline_script_bytes <= 1200
}

fn verify_wav_pcm_signal(path: &Path) -> Result<f64, String> {
    let bytes = fs::read(path).map_err(|e| format!("cannot read `{}`: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("`{}` is not a valid RIFF/WAVE file", path.display()));
    }
    let mut pos = 12usize;
    let mut channels = 1u16;
    let mut sample_rate = 24000u32;
    let mut bits_per_sample = 16u16;
    let mut audio_format = 1u16;
    let mut data_slice: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let sz = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        let body_start = pos + 8;
        let body_end = body_start.saturating_add(sz).min(bytes.len());
        if tag == b"fmt " && body_end >= body_start + 16 {
            let b = &bytes[body_start..body_end];
            audio_format = u16::from_le_bytes([b[0], b[1]]);
            channels = u16::from_le_bytes([b[2], b[3]]).max(1);
            sample_rate = u32::from_le_bytes([b[4], b[5], b[6], b[7]]).max(1);
            bits_per_sample = u16::from_le_bytes([b[14], b[15]]);
        } else if tag == b"data" {
            data_slice = Some(&bytes[body_start..body_end]);
            break;
        }
        pos = body_start.saturating_add(sz + (sz & 1));
    }
    let Some(pcm) = data_slice else {
        return Err(format!("`{}` is missing a WAVE `data` chunk", path.display()));
    };
    let bytes_per_sample = ((bits_per_sample as usize) / 8).max(2);
    let frame_bytes = bytes_per_sample * (channels as usize);
    let total_frames = pcm.len() / frame_bytes;
    let dur = (total_frames as f64) / (sample_rate as f64);
    if dur < 1.5 {
        return Err(format!(
            "`{}` duration is too short ({dur:.2}s < 1.50s)",
            path.display()
        ));
    }
    if (audio_format == 1 && bits_per_sample == 16) || (audio_format == 3 && bits_per_sample == 32)
    {
        let mid_frame = total_frames / 2;
        let mut sum_sq_first = 0.0f64;
        let mut sum_sq_second = 0.0f64;
        let mut clipped = 0usize;
        let win_frames = ((sample_rate as usize) / 10).max(1);
        let mut active_wins = 0usize;
        let mut total_wins = 0usize;
        let mut win_sq = 0.0f64;
        let mut win_count = 0usize;

        for f_idx in 0..total_frames {
            let off = f_idx * frame_bytes;
            let s: f64 = if audio_format == 3 && bits_per_sample == 32 {
                let v = f32::from_le_bytes([pcm[off], pcm[off + 1], pcm[off + 2], pcm[off + 3]]);
                if v.is_finite() { v as f64 } else { 0.0 }
            } else {
                let v = i16::from_le_bytes([pcm[off], pcm[off + 1]]);
                (v as f64) / 32768.0
            };
            if s.abs() >= 0.998 {
                clipped += 1;
            }
            let sq = s * s;
            if f_idx < mid_frame {
                sum_sq_first += sq;
            } else {
                sum_sq_second += sq;
            }
            win_sq += sq;
            win_count += 1;
            if win_count >= win_frames {
                total_wins += 1;
                if (win_sq / (win_count as f64)).sqrt() >= 0.0035 {
                    active_wins += 1;
                }
                win_sq = 0.0;
                win_count = 0;
            }
        }
        let rms_first = (sum_sq_first / (mid_frame.max(1) as f64)).sqrt();
        let rms_second =
            (sum_sq_second / ((total_frames.saturating_sub(mid_frame)).max(1) as f64)).sqrt();
        let clip_ratio = (clipped as f64) / (total_frames.max(1) as f64);
        let active_ratio = (active_wins as f64) / (total_wins.max(1) as f64);

        if rms_first < 0.0025 || rms_second < 0.0025 {
            return Err(format!(
                "`{}` has silent or collapsed speech energy (first_half_rms={rms_first:.4}, second_half_rms={rms_second:.4})",
                path.display()
            ));
        }
        if rms_second < 0.12 * rms_first || rms_first < 0.12 * rms_second {
            return Err(format!(
                "`{}` has unbalanced half-clip energy collapse (first_half_rms={rms_first:.4}, second_half_rms={rms_second:.4})",
                path.display()
            ));
        }
        if active_ratio < 0.22 {
            return Err(format!(
                "`{}` is mostly dead silence (active_speech_ratio={active_ratio:.2} < 0.22)",
                path.display()
            ));
        }
        if clip_ratio > 0.15 {
            return Err(format!(
                "`{}` has severe clipping distortion (clip_ratio={clip_ratio:.2})",
                path.display()
            ));
        }
    }
    Ok(dur)
}

fn is_reference_only_audio_stem(p_lower: &str) -> bool {
    let fname = Path::new(p_lower)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(p_lower);
    fname == "ref"
        || fname.starts_with("ref_")
        || fname.starts_with("raw_ref")
        || fname.ends_with("_ref")
        || fname.contains("voice_ref")
        || fname.contains("ref_voice")
        || fname.contains("reference")
}

fn verify_workspace_audio_coherence(
    workdir: &Path,
    ws_index: &WorkspaceSymbolIndex,
) -> Result<(), String> {
    let mut non_ref_audio_paths = Vec::new();
    let mut has_voice_clone_ref = false;
    for f in &ws_index.files {
        let p = f.rel_path.to_ascii_lowercase();
        let is_audio = p.ends_with(".wav")
            || p.ends_with(".m4a")
            || p.ends_with(".mp3")
            || p.ends_with(".flac")
            || p.ends_with(".aac");
        if !is_audio || f.bytes < 24 * 1024 {
            continue;
        }
        if is_reference_only_audio_stem(&p) {
            has_voice_clone_ref = true;
        } else {
            non_ref_audio_paths.push(f.rel_path.clone());
        }
    }
    if non_ref_audio_paths.is_empty() {
        let incomplete_py = ws_index
            .files
            .iter()
            .find(|f| f.rel_path.ends_with(".py"))
            .and_then(|f| {
                let src = fs::read_to_string(workdir.join(&f.rel_path)).ok()?;
                if !src.contains("generate_audio") {
                    Some(format!(
                        " (`{}` on disk currently omits `mlx_audio` `generate_audio(...)` voice-cloning synthesis — update `{}` to synthesize `audio/seg_XX.wav`, verify with `faster_whisper`, and mux `kronos_explained.mp4`, then run it via `run_command`)",
                        f.rel_path, f.rel_path
                    ))
                } else {
                    Some(format!(
                        " (run `{}` via `run_command` so it generates `audio/seg_XX.wav`, `audio/coherence_report.json`, and `kronos_explained.mp4`)",
                        f.rel_path
                    ))
                }
            })
            .unwrap_or_default();
        return Err(format!(
            "no synthesized narration audio file (outside `ref_*.wav`) exists on disk yet{incomplete_py}"
        ));
    }

    for rel in &non_ref_audio_paths {
        if rel.to_ascii_lowercase().ends_with(".wav") {
            let _ = verify_wav_pcm_signal(&workdir.join(rel))?;
        }
    }

    if has_voice_clone_ref {
        let report_path = if workdir.join("audio/coherence_report.json").is_file() {
            workdir.join("audio/coherence_report.json")
        } else if workdir.join("coherence_report.json").is_file() {
            workdir.join("coherence_report.json")
        } else {
            return Err(
                "voice-cloned WAVs exist on disk, but `audio/coherence_report.json` is missing — verify every `audio/seg_XX.wav` with `faster_whisper` ASR checking non-empty `ref_text` and `word_recall >= 0.70` across both first half and second half, then write `audio/coherence_report.json`"
                    .to_string(),
            );
        };
        let raw = fs::read_to_string(&report_path)
            .map_err(|e| format!("failed to read `{}`: {e}", report_path.display()))?;
        let parsed = crate::util::json::Json::parse(&raw)
            .map_err(|e| format!("invalid JSON in `{}`: {e}", report_path.display()))?;
        let ref_text = parsed
            .get("ref_text")
            .and_then(crate::util::json::Json::as_str)
            .unwrap_or("")
            .trim();
        if ref_text.split_whitespace().count() < 5 {
            return Err(
                "`ref_text` in `audio/coherence_report.json` is empty or < 5 words — trim `ref_24k.wav` to a complete Whisper sentence (4.0..8.5s) and pass its exact non-empty `ref_text` to `generate_audio` so voice cloning does not garble the opening half"
                    .to_string(),
            );
        }
        if parsed.get("verified").and_then(crate::util::json::Json::as_bool) == Some(false) {
            return Err(
                "`audio/coherence_report.json` has `verified: false` — regenerate incoherent segments with clean single-line symbol-free spoken prose"
                    .to_string(),
            );
        }
        let Some(segs) = parsed.get("segments").and_then(crate::util::json::Json::as_arr) else {
            return Err(
                "`audio/coherence_report.json` must contain a non-empty `segments` array"
                    .to_string(),
            );
        };
        if segs.is_empty() {
            return Err("`audio/coherence_report.json` `segments` array is empty".to_string());
        }
        let expected_seg_files: Vec<&String> = non_ref_audio_paths
            .iter()
            .filter(|p| {
                Path::new(p.as_str())
                    .file_name()
                    .and_then(|s| s.to_str())
                    .is_some_and(|n| n.to_ascii_lowercase().starts_with("seg_"))
            })
            .collect();
        for exp in &expected_seg_files {
            let covered = segs.iter().any(|s| {
                s.get("path")
                    .and_then(crate::util::json::Json::as_str)
                    .is_some_and(|sp| sp == exp.as_str())
            });
            if !covered {
                return Err(format!(
                    "synthesized segment `{exp}` on disk is missing from `audio/coherence_report.json` — every `seg_*.wav` must be ASR-verified across both halves"
                ));
            }
        }
        for (idx, seg) in segs.iter().enumerate() {
            let seg_rel = seg
                .get("path")
                .and_then(crate::util::json::Json::as_str)
                .unwrap_or("");
            if seg_rel.is_empty() || !workdir.join(seg_rel).is_file() {
                return Err(format!(
                    "segment #{idx} `{seg_rel}` in `coherence_report.json` is missing on disk"
                ));
            }
            let dur = verify_wav_pcm_signal(&workdir.join(seg_rel))?;
            if dur > 26.0 {
                return Err(format!(
                    "segment `{seg_rel}` is {dur:.1}s (> 26.0s zero-shot voice-cloning coherence window) — split into shorter 120-230 char per-slide segments"
                ));
            }
            let recall = seg
                .get("word_recall")
                .and_then(crate::util::json::Json::as_f64)
                .unwrap_or(0.0);
            let h1 = seg
                .get("first_half_recall")
                .and_then(crate::util::json::Json::as_f64)
                .unwrap_or(recall);
            let h2 = seg
                .get("second_half_recall")
                .and_then(crate::util::json::Json::as_f64)
                .unwrap_or(recall);
            if recall < 0.68 || h1 < 0.60 || h2 < 0.60 {
                return Err(format!(
                    "segment `{seg_rel}` failed ASR coherence gate (word_recall={recall:.2}, first_half={h1:.2}, second_half={h2:.2}; need >= 0.68 overall and >= 0.60 on both halves) — remove Markdown/LaTeX symbols and newlines from spoken text and regenerate"
                ));
            }
            let asr_text = seg
                .get("asr_text")
                .and_then(crate::util::json::Json::as_str)
                .unwrap_or("")
                .trim();
            let asr_words = asr_text.split_whitespace().count();
            let wps = (asr_words as f64) / dur.max(0.1);
            if asr_words < 5 || !(0.9..=5.2).contains(&wps) {
                return Err(format!(
                    "segment `{seg_rel}` has abnormal ASR speech rate ({asr_words} words in {dur:.1}s = {wps:.2} words/s) indicating babble or silence collapse"
                ));
            }
        }
    }
    Ok(())
}

fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

fn verify_workspace_video_container(
    workdir: &Path,
    ws_index: &WorkspaceSymbolIndex,
) -> Result<(), String> {
    let mut root_videos = Vec::new();
    let mut segment_videos = Vec::new();
    let mut has_any_audio = false;
    for f in &ws_index.files {
        let p = f.rel_path.to_ascii_lowercase();
        if p.ends_with(".wav") || p.ends_with(".m4a") || p.ends_with(".mp3") {
            has_any_audio = true;
        }
        if (p.ends_with(".mp4") || p.ends_with(".mov") || p.ends_with(".webm"))
            && f.bytes >= 64 * 1024
        {
            if p.starts_with("segments/") || p.contains("/segments/") {
                segment_videos.push(f.rel_path.clone());
            } else {
                root_videos.push(f.rel_path.clone());
            }
        }
    }
    if root_videos.is_empty() {
        if !segment_videos.is_empty() {
            return Err(
                "segment MP4 clips exist in `segments/`, but the final concatenated `.mp4` video has not been created at the workspace root yet — run `ffmpeg -y -f concat -safe 0 -i concat.txt -c copy kronos_explained.mp4`"
                    .to_string(),
            );
        }
        return Err("no rendered `.mp4`/`.mov`/`.webm` video (>= 64 KB) exists on disk yet".to_string());
    }
    let candidates = if !root_videos.is_empty() {
        &root_videos
    } else {
        &segment_videos
    };
    for rel in candidates {
        let lower = rel.to_ascii_lowercase();
        if lower.ends_with(".mp4") || lower.ends_with(".mov") {
            let bytes = fs::read(workdir.join(rel))
                .map_err(|e| format!("cannot read video `{rel}`: {e}"))?;
            if !contains_subslice(&bytes, b"moov") || !contains_subslice(&bytes, b"mdat") {
                return Err(format!(
                    "video `{rel}` is missing ISO-BMFF `moov`/`mdat` atoms (incomplete or corrupted mux)"
                ));
            }
            if !contains_subslice(&bytes, b"vide") {
                return Err(format!("video `{rel}` has no `vide` video track"));
            }
            if has_any_audio && !contains_subslice(&bytes, b"soun") {
                return Err(format!(
                    "video `{rel}` has no `soun` audio track — mux the verified voice narration WAVs into the video"
                ));
            }
        }
    }
    Ok(())
}

fn check_artifact_kind_with_note(
    workdir: &Path,
    kind: ArtifactKind,
    ws_index: &WorkspaceSymbolIndex,
    corpus: &str,
) -> Result<(), String> {
    match kind {
        ArtifactKind::WebApp => {
            let has_html = ws_index.files.iter().any(|f| {
                let p = f.rel_path.to_ascii_lowercase();
                (p.ends_with(".html") || p.ends_with(".htm")) && f.bytes >= 180
            });
            let has_styles = ws_index
                .files
                .iter()
                .any(|f| f.rel_path.to_ascii_lowercase().ends_with(".css") && f.bytes >= 80)
                || corpus.to_ascii_lowercase().contains("<style");
            if has_html && has_styles {
                Ok(())
            } else {
                Err("missing complete HTML document or CSS styles".to_string())
            }
        }
        ArtifactKind::PythonScript => {
            if ws_index
                .files
                .iter()
                .any(|f| f.rel_path.to_ascii_lowercase().ends_with(".py") && f.bytes >= 16)
            {
                Ok(())
            } else {
                Err("no `.py` script written on disk yet".to_string())
            }
        }
        ArtifactKind::RustCode => {
            if ws_index
                .files
                .iter()
                .any(|f| f.rel_path.to_ascii_lowercase().ends_with(".rs") && f.bytes >= 16)
            {
                Ok(())
            } else {
                Err("no `.rs` source file written on disk yet".to_string())
            }
        }
        ArtifactKind::JsModule => {
            if ws_index.files.iter().any(|f| {
                let p = f.rel_path.to_ascii_lowercase();
                (p.ends_with(".js") || p.ends_with(".mjs") || p.ends_with(".ts")) && f.bytes >= 16
            }) {
                Ok(())
            } else {
                Err("no `.js`/`.mjs`/`.ts` module written on disk yet".to_string())
            }
        }
        ArtifactKind::ReadableScript => {
            if ws_index.files.iter().any(|f| {
                let p = f.rel_path.to_ascii_lowercase();
                (p.ends_with(".md") || p.ends_with(".txt")) && f.bytes >= 3200
            }) {
                Ok(())
            } else {
                Err(
                    "narration script `.md`/`.txt` is missing or shorter than 3.2 KB — write the full technical script explaining every Kronos iteration, equation, and benchmark comparison"
                        .to_string(),
                )
            }
        }
        ArtifactKind::AudioMedia => verify_workspace_audio_coherence(workdir, ws_index),
        ArtifactKind::VideoMedia => verify_workspace_video_container(workdir, ws_index),
    }
}

fn load_workspace_text_corpus(
    workdir: &Path,
    ws_index: &WorkspaceSymbolIndex,
    modified_files: &[String],
    max_bytes: usize,
) -> String {
    let mut out = String::new();
    let mut seen = Vec::new();
    for rel in modified_files {
        if out.len() >= max_bytes {
            break;
        }
        let full = workdir.join(rel);
        if let Ok(txt) = fs::read_to_string(&full) {
            out.push_str(&truncate_utf8(&txt, max_bytes.saturating_sub(out.len())));
            out.push('\n');
            seen.push(rel.clone());
        }
    }
    for f in ws_index.files.iter().take(12) {
        if out.len() >= max_bytes {
            break;
        }
        if seen.iter().any(|s| s == &f.rel_path) || !is_indexable_text_ext(Path::new(&f.rel_path)) {
            continue;
        }
        let full = workdir.join(&f.rel_path);
        if let Ok(txt) = fs::read_to_string(&full) {
            out.push_str(&truncate_utf8(&txt, max_bytes.saturating_sub(out.len())));
            out.push('\n');
        }
    }
    out.to_ascii_lowercase()
}

fn term_matches_corpus(term: &str, corpus_lower: &str, workdir_lower: &str) -> bool {
    if corpus_lower.contains(term) || workdir_lower.contains(term) {
        return true;
    }
    if term.len() >= 5 {
        let stem = &term[..term.len() - 1];
        if corpus_lower.contains(stem) {
            return true;
        }
    }
    match term {
        "emr" => {
            corpus_lower.contains("electronic medical")
                || corpus_lower.contains("health record")
                || corpus_lower.contains("patient")
                || corpus_lower.contains("clinical")
        }
        "todo" | "to-do" => corpus_lower.contains("task"),
        "dark" => corpus_lower.contains("#0") || corpus_lower.contains("#1") || corpus_lower.contains("theme"),
        _ => false,
    }
}

fn extract_target_folder_hint(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    if !crate::runtime::device::is_workspace_folder_target_query(&lower) {
        return None;
    }
    for raw_tok in text.split_whitespace() {
        let clean = raw_tok.trim_matches(|c: char| {
            matches!(c, '"' | '\'' | '`' | ',' | '.' | ';' | ':' | '(' | ')')
        });
        if (clean.starts_with("~/") || clean.starts_with("/Users/")) && clean.len() > 4 {
            let p = Path::new(clean);
            if p.is_file() {
                continue;
            }
            if let Some(seg) = p.file_name().and_then(|s| s.to_str()) {
                if !seg.contains('.') && seg.len() >= 2 {
                    return Some(seg.to_string());
                }
            }
        }
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    for i in 0..words.len() {
        let wl = words[i].to_ascii_lowercase();
        if (wl == "documents" || wl == "desktop" || wl == "downloads") && i + 2 < words.len() {
            let next = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '_');
            let after = words[i + 2].to_ascii_lowercase();
            if !next.is_empty()
                && !matches!(next.to_ascii_lowercase().as_str(), "folder" | "directory" | "and" | "then")
                && (after.starts_with("folder") || after.starts_with("dir"))
            {
                return Some(next.to_string());
            }
        }
    }
    None
}

fn detect_expected_artifact_kinds(lower: &str, workspace_has_files: bool) -> Vec<ArtifactKind> {
    let mut kinds = Vec::new();
    let wants_video = has_any(
        lower,
        &[
            "make a video",
            "create a video",
            "build a video",
            "render a video",
            "generate a video",
        ],
    );
    let wants_audio = has_any(
        lower,
        &[
            "use my voice",
            "mlx audio",
            "mlx-audio",
            "mlx_audio",
            "voice clone",
            "voice-clone",
            "generate audio",
            "synthesize audio",
        ],
    );
    let wants_readable_script = has_any(
        lower,
        &[
            "script i can read",
            "make a script",
            "write a script",
            "narration script",
            "video script",
        ],
    ) && !lower.contains("python script")
        && !lower.contains("shell script")
        && !lower.contains("bash script");

    if wants_readable_script {
        kinds.push(ArtifactKind::ReadableScript);
    }
    if wants_audio {
        kinds.push(ArtifactKind::AudioMedia);
    }
    if wants_video {
        kinds.push(ArtifactKind::VideoMedia);
    }
    if !kinds.is_empty() {
        return kinds;
    }

    let wants_web_overhaul = has_any(
        lower,
        &[
            "modularize",
            "redesign",
            "restyle",
            "rebuild",
            "match the screenshot",
            "match this screenshot",
            "matching the screenshot",
        ],
    );
    if has_any(
        lower,
        &[
            "website",
            "web page",
            "webpage",
            "web app",
            "landing page",
            "dashboard",
            "portal",
        ],
    ) && (!workspace_has_files || wants_web_overhaul)
    {
        kinds.push(ArtifactKind::WebApp);
        return kinds;
    }
    if lower.contains("python script") || lower.contains(".py") {
        kinds.push(ArtifactKind::PythonScript);
        return kinds;
    }
    if lower.contains("rust ") && (lower.contains(".rs") || lower.contains("crate")) {
        kinds.push(ArtifactKind::RustCode);
        return kinds;
    }
    if lower.contains(".js") || lower.contains(".ts") {
        kinds.push(ArtifactKind::JsModule);
    }
    kinds
}

fn extract_explicit_filenames(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let exts = [".html", ".css", ".js", ".mjs", ".ts", ".py", ".rs", ".json", ".toml", ".sh"];
    for raw in text.split_whitespace() {
        let tok = raw.trim_matches(|c: char| {
            matches!(c, '`' | '"' | '\'' | ',' | ';' | ':' | '(' | ')' | '[' | ']')
        });
        let tok = tok.trim_end_matches('.');
        let lower = tok.to_ascii_lowercase();
        if exts.iter().any(|e| lower.ends_with(e))
            && !tok.contains("://")
            && !tok.contains('*')
            && !tok.contains('?')
            && !tok.contains('<')
            && !tok.contains('>')
            && !tok.starts_with('/')
            && !tok.starts_with("~/")
            && !tok.starts_with('.')
            && tok.len() <= 64
            && !out.iter().any(|x: &String| x.eq_ignore_ascii_case(tok))
        {
            out.push(tok.to_string());
        }
    }
    out
}

fn extract_domain_requirement_terms(lower: &str) -> Vec<String> {
    const STOP: &[&str] = &[
        "the", "and", "for", "with", "that", "this", "from", "into", "inside", "about", "then",
        "make", "build", "create", "write", "edit", "update", "modify", "change", "please",
        "kronos", "folder", "directory", "project", "file", "files", "code", "website", "webpage",
        "page", "site", "app", "application", "software", "new", "documents", "desktop",
        "downloads", "users", "orgproj", "using", "should", "have", "would", "could", "like",
        "also", "more", "some", "clean", "modern", "good", "nice", "fast", "full", "working",
        "complete", "html", "css", "javascript", "python", "rust", "index", "style", "main",
        "want", "help", "explain", "ensure", "behind", "everything", "every", "step", "why",
        "done", "whats", "what", "there", "out", "use", "tools", "tool", "too", "requires",
        "require", "open", "source", "already", "exist", "exists", "computer", "read", "can",
        "how", "all", "its", "compares", "compare", "https", "http", "github", "com", "m4a",
        "modularize", "modular", "monolith", "monolithic", "refactor", "split", "separate",
        "extract", "module", "modules", "attached", "attachment", "screenshot", "image",
        "picture", "reference", "match", "matching", "styles", "styled", "styling", "responsive",
        "test", "tests", "testing", "verify", "verification", "check", "checks", "suite",
        "script", "scripts", "headless", "browser", "iterate", "iterating", "until", "afterwards",
        "afterward", "work", "works", "own", "your", "our", "result", "results", "pass", "passes",
        "passing", "broken", "legacy", "single", "single-file", "multi", "multi-file", "multiple",
        "existing", "current", "target", "design", "visual", "layout", "automated",
    ];
    let mut terms = Vec::new();
    for raw in lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '-') {
        let w = raw.trim_matches('-');
        if w.len() < 3 || w.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if STOP.contains(&w) {
            continue;
        }
        if !terms.iter().any(|x: &String| x == w) {
            terms.push(w.to_string());
            if terms.len() >= 4 {
                break;
            }
        }
    }
    terms
}

fn extract_query_tokens(lower: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in lower.split(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '-') {
        let w = raw.trim_matches(|c: char| c == '_' || c == '-');
        if w.len() >= 3 && !out.iter().any(|x: &String| x == w) {
            out.push(w.to_string());
            if out.len() >= 12 {
                break;
            }
        }
    }
    out
}

#[inline]
fn has_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::agent::tool_definitions_json;

    #[test]
    fn test_calibrated_primitives_and_bayes_fusion() {
        let c = CalibratedChoice::from_logits(&[3.2, -1.0, -2.0], 1.0, false, 0.1);
        assert_eq!(c.selected, 0);
        let sum_p: f32 = c.probs.iter().sum();
        assert!((sum_p - 1.0).abs() < 1e-5);
        assert!(c.confidence > 0.80);
        assert!(c.margin > 0.90);

        let uncertain = CalibratedChoice::from_logits(&[0.1, 0.0, -0.1], 1.0, false, 0.1);
        assert!(uncertain.confidence < 0.05);

        let (fused, used) = fuse_binary_log_odds(0.2, Some((&[-0.3, -2.5], 0.45)), 1.0);
        assert!(used);
        let b = CalibratedBool::from_log_odds(fused, 0.65, used, 0.2);
        assert!(b.value);
        assert!(b.prob_true > 0.85);

        let sc = CalibratedScore::from_mean_and_concentration(0.92, 14.0, false, 0.1);
        assert!(sc.expected > 0.85);
        assert!(sc.confidence > 0.40);
    }

    #[test]
    fn test_workspace_symbol_index_and_jev_prepass_pruning() {
        let dir = std::env::temp_dir().join(format!("kronos_jev_idx_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("index.html"),
            "<!doctype html><html><head><title>Clinic</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><h1 id=\"hero-title\">Clinic EMR</h1><button id=\"theme-btn\">Theme</button></body></html>",
        )
        .unwrap();
        fs::write(
            dir.join("style.css"),
            ":root { --bg: #ffffff; }\nbody { background: var(--bg); color: #111; }\n#hero-title { font-size: 2rem; }\n",
        )
        .unwrap();
        fs::write(
            dir.join("app.js"),
            "function toggleDarkTheme() {\n  const btn = document.getElementById('theme-btn');\n  return btn;\n}\n",
        )
        .unwrap();

        let t0 = Instant::now();
        let full_tools = tool_definitions_json();
        let plan = plan_turn_with_jev(
            None,
            "sys",
            "update the hero-title color and dark theme button in app.js",
            None,
            false,
            &dir,
            false,
            &full_tools,
            false,
            false,
        );
        assert!(t0.elapsed().as_secs_f64() * 1000.0 < 25.0);
        assert_eq!(plan.route, TurnRoute::Agent);
        assert!(plan.pruned_tool_count >= 2, "expected cold-prefix tool pruning");
        assert!(!plan.prefetched_files.is_empty());
        let block = plan.prefetch_block.as_deref().unwrap();
        assert!(block.contains("app.js"));
        assert!(block.contains("toggleDarkTheme"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rubric_extraction_and_grounded_verification_catches_hallucinations() {
        let dir = std::env::temp_dir().join(format!("kronos_jev_rubric_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();

        let prompt = "create a website about an EMR software with patient charts";
        let rubric = extract_requirement_rubric(prompt, TurnRoute::Agent, false);
        assert!(rubric.len() >= 3);

        // 1. Hallucinated completion with zero tool calls must fail immediately.
        let fake_eval = evaluate_rubric(
            None,
            prompt,
            &rubric,
            &dir,
            &[],
            false,
            None,
            "I have created index.html and style.css for your EMR portal.",
        );
        assert!(!fake_eval.passed);
        assert!(fake_eval.ungrounded_claim);
        assert!(fake_eval.score.expected < 0.45);

        // 2. Grounded workspace with styled EMR HTML + CSS + clean oracles passes.
        fs::write(
            dir.join("index.html"),
            "<!doctype html><html><head><title>Pulse EMR</title><link rel=\"stylesheet\" href=\"style.css\"></head><body><header><h1>Pulse EMR Patient Charts</h1></header><main><section id=\"patients\">Patient chart records and clinical schedule</section></main></body></html>",
        )
        .unwrap();
        fs::write(
            dir.join("style.css"),
            ":root { --accent: #0f766e; }\nbody { margin: 0; font-family: system-ui; background: #f8fafc; }\nheader { padding: 1.5rem; }\n",
        )
        .unwrap();
        let real_eval = evaluate_rubric(
            None,
            prompt,
            &rubric,
            &dir,
            &["index.html".to_string(), "style.css".to_string()],
            true,
            None,
            "Built Pulse EMR with patient chart view.",
        );
        assert!(real_eval.passed, "expected pass, unmet:\n{}", real_eval.unmet_summary);
        assert!(real_eval.score.expected > 0.85);

        // 3. Multi-deliverable script + audio + video prompt with spaced input path and "open source tools"
        //    must not synthesize a bogus folder switch or keep `use_computer`.
        let media_prompt = "make a script I can read and make a video with this use tools and use my voice (/Users/melodyleonard/Downloads/James Iroha Close 4.m4a), use open source tools like MLX Audio";
        let mut tools_with_comp = tool_definitions_json();
        tools_with_comp.push(crate::runtime::device::use_computer_tool_definition_json());
        let media_plan = plan_turn_with_jev(
            None,
            "sys",
            media_prompt,
            Some(TurnRoute::Agent),
            false,
            &dir,
            false,
            &tools_with_comp,
            true,
            false,
        );
        assert!(!media_plan.selected_tool_names.iter().any(|n| n == "use_computer"));
        assert!(media_plan.selected_tool_names.iter().any(|n| n == "run_command"));
        assert!(!media_plan
            .rubric
            .iter()
            .any(|p| matches!(p.kind, PredicateKind::WorkspaceSwitched { .. })));
        assert!(media_plan.rubric.iter().any(|p| matches!(
            p.kind,
            PredicateKind::ExpectedArtifact {
                kind: ArtifactKind::ReadableScript
            }
        )));
        assert!(media_plan.rubric.iter().any(|p| matches!(
            p.kind,
            PredicateKind::ExpectedArtifact {
                kind: ArtifactKind::AudioMedia
            }
        )));
        assert!(media_plan.rubric.iter().any(|p| matches!(
            p.kind,
            PredicateKind::ExpectedArtifact {
                kind: ArtifactKind::VideoMedia
            }
        )));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_modular_web_architecture_and_self_test_rubric() {
        let dir = std::env::temp_dir().join(format!("kronos_jev_modweb_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("styles")).unwrap();
        fs::create_dir_all(dir.join("js")).unwrap();

        let prompt = "Modularize this monolithic website into styles/*.css and js/*.js modules, match the attached screenshot styles, and test your work afterwards.";
        let rubric = extract_requirement_rubric(prompt, TurnRoute::Agent, true);
        assert!(rubric.iter().any(|p| p.kind == PredicateKind::ModularWebArchitecture));
        assert!(rubric.iter().any(|p| p.kind == PredicateKind::SelfTestedAfterEdits));
        assert!(!rubric.iter().any(|p| matches!(&p.kind, PredicateKind::TargetFilePresent { name_hint } if name_hint.contains('*'))));

        // 1. Monolithic index.html with giant inline style/script fails ModularWebArchitecture.
        let big_css = "body { color: #e2e8f0; background: #0f172a; }\n".repeat(40);
        let big_js = "function renderAll() { return 42; }\n".repeat(50);
        fs::write(
            dir.join("index.html"),
            format!("<!doctype html><html><head><style>{big_css}</style></head><body><h1>Ops</h1><script>{big_js}</script></body></html>"),
        )
        .unwrap();
        let eval_monolith = evaluate_rubric_with_telemetry(
            None,
            prompt,
            &rubric,
            &dir,
            &["index.html".to_string()],
            true,
            true,
            None,
            "Updated index.html",
        );
        assert!(!eval_monolith.passed);

        // 2. Extracted modular CSS + JS passes ModularWebArchitecture, fails if self-test did not run after edits, passes once tested.
        fs::write(
            dir.join("index.html"),
            "<!doctype html><html><head><title>Ops Telemetry</title><link rel=\"stylesheet\" href=\"styles/tokens.css\"><link rel=\"stylesheet\" href=\"styles/layout.css\"></head><body><main id=\"app\"><h1>Ops Telemetry</h1></main><script type=\"module\" src=\"js/app.js\"></script></body></html>",
        )
        .unwrap();
        fs::write(
            dir.join("styles/tokens.css"),
            ":root { --bg: #0b1120; --panel: #111827; --accent: #14b8a6; --text: #f8fafc; }\nbody { background: var(--bg); color: var(--text); margin: 0; }\n",
        )
        .unwrap();
        fs::write(
            dir.join("styles/layout.css"),
            ".shell { display: grid; grid-template-columns: 240px 1fr; min-height: 100vh; gap: 1.25rem; }\n",
        )
        .unwrap();
        fs::write(
            dir.join("js/data.js"),
            "export const SERVICES = [{ id: 'auth', status: 'healthy', latencyMs: 14 }];\n",
        )
        .unwrap();
        fs::write(
            dir.join("js/app.js"),
            "import { SERVICES } from './data.js';\nconsole.log(SERVICES.length);\n",
        )
        .unwrap();

        let eval_untested = evaluate_rubric_with_telemetry(
            None,
            prompt,
            &rubric,
            &dir,
            &[
                "index.html".to_string(),
                "styles/tokens.css".to_string(),
                "styles/layout.css".to_string(),
                "js/data.js".to_string(),
                "js/app.js".to_string(),
            ],
            true,
            false,
            None,
            "Modularized into styles and js modules.",
        );
        assert!(!eval_untested.passed);
        assert!(eval_untested.unmet_summary.contains("run_command"));

        let eval_tested = evaluate_rubric_with_telemetry(
            None,
            prompt,
            &rubric,
            &dir,
            &[
                "index.html".to_string(),
                "styles/tokens.css".to_string(),
                "styles/layout.css".to_string(),
                "js/data.js".to_string(),
                "js/app.js".to_string(),
            ],
            true,
            true,
            None,
            "Modularized and verified all tests pass.",
        );
        assert!(eval_tested.passed, "expected pass, unmet:\n{}", eval_tested.unmet_summary);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_single_pass_neural_jev_preserves_kv_prefix_bit_exact() {
        use crate::runtime::chat::{GenParams, Message};
        let model_dir = Path::new("models/Qwen2.5-Coder-0.5B-Instruct");
        if !model_dir.join("kronos-nf4g128.k2tb").is_file() {
            return;
        }
        let mut chat = ChatModel::load(model_dir, "nf4g128", 512, 4, None).unwrap();
        let warmed = chat.prewarm_prefix("You are Kronos.", &[]);
        assert!(warmed > 0);
        assert!(chat.has_hot_prefix("You are Kronos.", &[]));

        let msgs = vec![
            Message::new("system", "You are Kronos."),
            Message::new("user", "Capital of France:"),
        ];
        let greedy = GenParams {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            max_tokens: 4,
            seed: 1,
        };
        let baseline = chat
            .generate_with_tools(&msgs, &[], &greedy, |_| true)
            .unwrap();

        // Reset back to the prewarmed system prefix, run single-pass Jev scoring on positive vs
        // negative workspace evidence, and verify greedy continuation from the restored prefix is token-identical.
        chat.prewarm_prefix("You are Kronos.", &[]);
        let t0 = Instant::now();
        let pos = chat
            .jev_score_anchors(
                JEV_VERIFY_SYSTEM,
                "Code:\nfunction addNumbers(a, b) { return a + b; }\n\nQuestion: Does the code above return the sum of a and b? Answer Yes or No.",
                &["Yes", "No"],
            )
            .expect("expected positive single-pass anchor log-probs");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let neg = chat
            .jev_score_anchors(
                JEV_VERIFY_SYSTEM,
                "Code:\nfunction addNumbers(a, b) { return 0; }\n\nQuestion: Does the code above return the sum of a and b? Answer Yes or No.",
                &["Yes", "No"],
            )
            .expect("expected negative single-pass anchor log-probs");
        let pos_lo = pos.0[0] - pos.0[1];
        let neg_lo = neg.0[0] - neg.0[1];
        assert!(
            pos.1 > 0.50 && neg.1 > 0.50,
            "expected >50% probability mass on Yes/No anchors, got pos={pos:?} neg={neg:?}"
        );
        assert!(
            pos_lo > neg_lo,
            "expected positive evidence log-odds ({pos_lo:.2}) > negative ({neg_lo:.2}) in {ms:.2}ms"
        );
        assert!(chat.has_hot_prefix("You are Kronos.", &[]));

        let after = chat
            .generate_with_tools(&msgs, &[], &greedy, |_| true)
            .unwrap();
        assert_eq!(
            after.prefilled_tokens, 1,
            "expected full conversation KV prefix reuse (only 1 tail token re-evaluated)"
        );
        assert_eq!(
            after.token_ids, baseline.token_ids,
            "expected bit-exact greedy continuation after KV-restored Jev pass"
        );
    }

    fn write_test_wav(path: &Path, sr: u32, first_half_amp: i16, second_half_amp: i16, secs: usize) {
        let total_samples = (sr as usize) * secs;
        let data_bytes = (total_samples * 2) as u32;
        let mut buf = Vec::with_capacity(44 + data_bytes as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVEfmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&sr.to_le_bytes());
        buf.extend_from_slice(&(sr * 2).to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        let mid = total_samples / 2;
        for i in 0..total_samples {
            let amp = if i < mid { first_half_amp } else { second_half_amp };
            let s = if (i / 40) % 2 == 0 { amp } else { -amp };
            buf.extend_from_slice(&s.to_le_bytes());
        }
        fs::write(path, buf).unwrap();
    }

    #[test]
    fn test_audio_coherence_first_and_second_half_and_mp4_dual_stream_verification() {
        let dir = std::env::temp_dir().join(format!("kronos_jev_audio_video_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("audio")).unwrap();
        fs::create_dir_all(dir.join("segments")).unwrap();

        let prompt = "make a script I can read and make a video with this use tools and use my voice, use open source tools like MLX Audio";
        let rubric = extract_requirement_rubric(prompt, TurnRoute::Agent, false);
        fs::write(dir.join("script.md"), "Kronos technical math narration. ".repeat(130)).unwrap();
        write_test_wav(&dir.join("ref_24k.wav"), 24000, 4200, 4200, 5);

        // 1. Half-collapsed audio (second half silent) must fail AudioMedia verification.
        write_test_wav(&dir.join("audio/seg_01.wav"), 24000, 4500, 20, 6);
        let eval_collapsed = evaluate_rubric_with_telemetry(
            None,
            prompt,
            &rubric,
            &dir,
            &["script.md".to_string()],
            true,
            true,
            None,
            "Generated audio.",
        );
        assert!(!eval_collapsed.passed);
        assert!(
            eval_collapsed.unmet_summary.contains("collapsed")
                || eval_collapsed.unmet_summary.contains("energy"),
            "unexpected unmet_summary: {}",
            eval_collapsed.unmet_summary
        );

        // 2. Balanced waveform with incoherent second-half ASR recall (< 0.60) must fail.
        write_test_wav(&dir.join("audio/seg_01.wav"), 24000, 4500, 4100, 6);
        fs::write(
            dir.join("audio/coherence_report.json"),
            r#"{"ref_text":"So I am building Kronos to run locally on your hardware","verified":true,"segments":[{"path":"audio/seg_01.wav","duration_secs":6.0,"word_count":14,"word_recall":0.55,"first_half_recall":0.86,"second_half_recall":0.28,"asr_text":"Kronos uses Cartesian product key routing and pre remotes babble"}]}"#,
        )
        .unwrap();
        let eval_bad_asr = evaluate_rubric_with_telemetry(
            None,
            prompt,
            &rubric,
            &dir,
            &["script.md".to_string()],
            true,
            true,
            None,
            "Generated audio.",
        );
        assert!(!eval_bad_asr.passed);
        assert!(eval_bad_asr.unmet_summary.contains("ASR coherence gate"));

        // 3. Coherent first-half and second-half ASR + merged dual-track MP4 passes.
        fs::write(
            dir.join("audio/coherence_report.json"),
            r#"{"ref_text":"So I am building Kronos to run locally on your hardware","verified":true,"segments":[{"path":"audio/seg_01.wav","duration_secs":6.0,"word_count":15,"word_recall":0.93,"first_half_recall":0.92,"second_half_recall":0.94,"asr_text":"Kronos uses Cartesian product key routing and four bit NormalFloat quantization for fast local inference"}]}"#,
        )
        .unwrap();
        let mut fake_mp4 = vec![0u8; 72 * 1024];
        fake_mp4[0..16].copy_from_slice(b"ftypisommoovmdat");
        fake_mp4[32..40].copy_from_slice(b"hdlrvide");
        fake_mp4[64..72].copy_from_slice(b"hdlrsoun");
        fs::write(dir.join("kronos_explained.mp4"), fake_mp4).unwrap();

        let eval_ok = evaluate_rubric_with_telemetry(
            None,
            prompt,
            &rubric,
            &dir,
            &["script.md".to_string()],
            true,
            true,
            None,
            "Completed coherent audio and video.",
        );
        assert!(eval_ok.passed, "expected pass, got unmet: {}", eval_ok.unmet_summary);

        let _ = fs::remove_dir_all(&dir);
    }
}
