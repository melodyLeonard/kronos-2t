//! Local user-owned conversation, messaging, and memory retrieval across macOS SQLite stores,
//! silent background window perception (`CoreGraphics` + `Vision`), and Kronos session history.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::json::{quote, Json};

#[cfg(target_os = "macos")]
const APPLE_COREDATA_EPOCH_OFFSET: i64 = 978_307_200;
const MAX_MEMORY_JSONL_LINES: usize = 1200;

#[derive(Clone, Debug)]
pub struct LocalConversationContext {
    pub prompt_block: String,
    pub sources_matched: Vec<String>,
    pub found_verified_match: bool,
    pub whatsapp_tcc_blocked: bool,
    pub elapsed_ms: f64,
}

pub fn is_conversation_or_memory_query(lower: &str) -> bool {
    let trimmed = lower.trim();
    if trimmed.is_empty() {
        return false;
    }

    let app_nouns = [
        "whatsapp",
        "whats app",
        "imessage",
        "telegram",
        "slack",
        "apple notes",
        "my notes",
    ];
    if app_nouns.iter().any(|k| trimmed.contains(k))
        && !trimmed.starts_with("open ")
        && !trimmed.starts_with("launch ")
        && !trimmed.starts_with("start ")
    {
        return true;
    }

    let msg_phrases = [
        "last message",
        "latest message",
        "recent message",
        "recent messages",
        "unread message",
        "unread messages",
        "message from ",
        "messages from ",
        "message to ",
        "messages to ",
        "my reply to ",
        "what did i send",
        "what did i reply",
        "what did i text",
        "what did i write",
        "messaged me",
        "texted me",
        "replied to me",
        "wrote to me",
        "sent me a message",
        "sent me on ",
        "said on ",
        "chat with ",
        "chats with ",
    ];
    if msg_phrases.iter().any(|p| trimmed.contains(p)) {
        return true;
    }

    let memory_phrases = [
        "remind me",
        "remind myself",
        "important conversation",
        "important conversations",
        "previous conversation",
        "previous conversations",
        "past conversation",
        "past conversations",
        "earlier conversation",
        "earlier conversations",
        "recent conversation",
        "recent conversations",
        "what did we talk about",
        "what did we discuss",
        "what we talked about",
        "what we discussed",
        "get context on ",
        "get context from ",
        "context of my conversation",
        "conversation history",
        "my conversations",
        "our last chat",
        "our previous chat",
    ];
    memory_phrases.iter().any(|p| trimmed.contains(p))
}

pub fn extract_conversation_query_terms(user_prompt: &str) -> Vec<String> {
    let stopwords: HashSet<&'static str> = [
        "what", "was", "were", "is", "are", "the", "a", "an", "last", "latest", "recent",
        "message", "messages", "chat", "chats", "conversation", "conversations", "on", "in",
        "from", "to", "with", "by", "about", "around", "for", "of", "and", "or", "whatsapp",
        "whats", "app", "imessage", "telegram", "slack", "notes", "note", "mail", "kronos",
        "said", "say", "says", "sent", "send", "text", "texted", "reply", "replied", "write",
        "wrote", "written", "messaged", "thing", "thread", "me", "my", "our", "his",
        "her", "their", "remind", "myself", "important", "context", "get", "show", "tell",
        "check", "read", "summarize", "summary", "please", "can", "could", "would", "you",
        "did", "does", "do", "have", "has", "had", "today", "yesterday", "earlier", "previous",
        "past", "any", "all", "who", "when", "where", "how", "why", "that", "this", "these",
        "those", "talk", "talked", "talking", "discuss", "discussed", "history",
    ]
    .into_iter()
    .collect();

    let mut proper_nouns = Vec::new();
    let mut regular_terms = Vec::new();

    for (idx, raw_tok) in user_prompt
        .split(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-')
        .enumerate()
    {
        let clean = raw_tok
            .trim_matches(|c: char| !c.is_alphanumeric())
            .trim_end_matches("'s");
        let clean = clean.trim_matches(|c: char| !c.is_alphanumeric());
        if clean.len() < 2 {
            continue;
        }
        let low = clean.to_ascii_lowercase();
        if stopwords.contains(low.as_str()) {
            continue;
        }
        let starts_upper = clean.chars().next().is_some_and(|c| c.is_uppercase());
        if starts_upper && idx > 0 {
            if !proper_nouns.contains(&low) {
                proper_nouns.push(low);
            }
        } else if !proper_nouns.contains(&low) && !regular_terms.contains(&low) {
            regular_terms.push(low);
        }
    }

    proper_nouns.extend(regular_terms);
    proper_nouns.truncate(6);
    proper_nouns
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageDirectionIntent {
    /// User wants the latest message authored/sent by the contact to the user
    /// (e.g. "what was Oliver-ke last message to me on WhatsApp", "what did Dera send me").
    IncomingFromContact,
    /// User wants the latest message authored/sent by the user to the contact
    /// (e.g. "what was my last message to Oliver-ke", "what did I send to Dera").
    OutgoingFromUser,
    /// User wants a bidirectional overview of recent messages in the thread
    /// (e.g. "show recent messages with Oliver-ke", "summarize my chat with Dera").
    ThreadOverview,
}

pub fn parse_message_direction_intent(user_prompt: &str) -> MessageDirectionIntent {
    let low = strip_injected_context_prefix(user_prompt)
        .to_ascii_lowercase()
        .replace(['\u{2018}', '\u{2019}'], "'");

    let outgoing_cues = [
        "my last message",
        "my latest message",
        "my recent message",
        "my message to ",
        "my messages to ",
        "my reply to ",
        "message i sent",
        "messages i sent",
        "message i wrote",
        "what did i send",
        "what did i write",
        "what did i text",
        "what did i message",
        "what did i reply",
        "what did i say",
        "what i sent ",
        "what i said to ",
        "last thing i sent",
        "last thing i said",
    ];
    if outgoing_cues.iter().any(|p| low.contains(p)) {
        return MessageDirectionIntent::OutgoingFromUser;
    }

    let incoming_cues = [
        "to me",
        "sent me",
        "send me",
        "messaged me",
        "texted me",
        "wrote to me",
        "wrote me",
        "replied to me",
        "said to me",
        "message from ",
        "messages from ",
    ];
    if incoming_cues.iter().any(|p| low.contains(p)) {
        return MessageDirectionIntent::IncomingFromContact;
    }

    let overview_cues = [
        "recent messages with ",
        "messages between ",
        "conversation with ",
        "chat history with ",
        "summarize my chat",
        "summarize my conversation",
        "thread with ",
    ];
    if overview_cues.iter().any(|p| low.contains(p)) {
        return MessageDirectionIntent::ThreadOverview;
    }

    // Singular "what was <Contact>'s last message" / "what did <Contact> send/say" asks for the contact's message.
    MessageDirectionIntent::IncomingFromContact
}

pub fn strip_injected_context_prefix(raw_user_text: &str) -> &str {
    let trimmed = raw_user_text.trim();
    if trimmed.starts_with("[Device Context") || trimmed.starts_with("[Local Conversations") {
        if let Some(pos) = trimmed.find("\n\n") {
            let rest = trimmed[pos + 2..].trim();
            if !rest.is_empty() {
                return strip_injected_context_prefix(rest);
            }
        }
    }
    trimmed
}

pub fn default_memory_jsonl_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    home.join(".kronos")
        .join("device_context")
        .join("conversation_memory.jsonl")
}

pub fn default_learned_memory_json_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    home.join(".kronos")
        .join("device_context")
        .join("learned_user_memory.json")
}

pub fn default_distilled_vault_json_path() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    home.join(".kronos")
        .join("device_context")
        .join("distilled_14b_vault.json")
}

pub fn read_learned_memory_json() -> String {
    let path = default_learned_memory_json_path();
    if let Ok(raw) = fs::read_to_string(&path) {
        if let Ok(parsed) = Json::parse(&raw) {
            return parsed.stringify();
        }
    }
    r#"{"facts":{},"notes":[],"reminders":[]}"#.to_string()
}

pub fn merge_learned_memory_json(incoming: &Json) -> String {
    let path = default_learned_memory_json_path();
    let existing = fs::read_to_string(&path)
        .ok()
        .and_then(|s| Json::parse(&s).ok())
        .unwrap_or_else(|| Json::Obj(Vec::new()));

    // Merge facts with last-write-wins timestamp resolution per canonical fact key.
    let mut facts_map: Vec<(String, Json)> = existing
        .get("facts")
        .and_then(Json::as_obj)
        .map(|o| o.to_vec())
        .unwrap_or_default();
    if let Some(inc_facts) = incoming.get("facts").and_then(Json::as_obj) {
        for (k, v) in inc_facts {
            let inc_ts = v.get("ts").and_then(Json::as_f64).unwrap_or(0.0);
            if let Some((_, cur)) = facts_map.iter_mut().find(|(ek, _)| ek == k) {
                let cur_ts = cur.get("ts").and_then(Json::as_f64).unwrap_or(0.0);
                if inc_ts >= cur_ts {
                    *cur = v.clone();
                }
            } else {
                facts_map.push((k.clone(), v.clone()));
            }
        }
    }

    let merge_text_list = |key: &str| -> Vec<Json> {
        let mut items: Vec<Json> = existing
            .get(key)
            .and_then(Json::as_arr)
            .map(|a| a.to_vec())
            .unwrap_or_default();
        if let Some(inc_arr) = incoming.get(key).and_then(Json::as_arr) {
            for item in inc_arr {
                let Some(txt) = item.get("text").and_then(Json::as_str) else {
                    continue;
                };
                let exists = items.iter().any(|e| {
                    e.get("text")
                        .and_then(Json::as_str)
                        .is_some_and(|et| et.eq_ignore_ascii_case(txt))
                });
                if !exists {
                    items.push(item.clone());
                }
            }
        }
        items
    };

    let merged = Json::Obj(vec![
        ("facts".to_string(), Json::Obj(facts_map)),
        ("notes".to_string(), Json::Arr(merge_text_list("notes"))),
        (
            "reminders".to_string(),
            Json::Arr(merge_text_list("reminders")),
        ),
    ]);
    let serialized = merged.stringify();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&path, &serialized);
    serialized
}

pub fn read_distilled_vault_json() -> String {
    let path = default_distilled_vault_json_path();
    if let Ok(raw) = fs::read_to_string(&path) {
        if let Ok(parsed) = Json::parse(&raw) {
            return parsed.stringify();
        }
    }
    "[]".to_string()
}

pub fn merge_distilled_vault_json(incoming: &Json) -> String {
    let path = default_distilled_vault_json_path();
    let mut items: Vec<Json> = fs::read_to_string(&path)
        .ok()
        .and_then(|s| Json::parse(&s).ok())
        .and_then(|j| j.as_arr().map(|a| a.to_vec()))
        .unwrap_or_default();
    if let Some(inc_arr) = incoming.as_arr() {
        for item in inc_arr {
            let Some(q) = item.get("q").and_then(Json::as_str) else {
                continue;
            };
            if let Some(existing) = items.iter_mut().find(|e| {
                e.get("q")
                    .and_then(Json::as_str)
                    .is_some_and(|eq| eq.eq_ignore_ascii_case(q))
            }) {
                *existing = item.clone();
            } else {
                items.push(item.clone());
            }
        }
    }
    if items.len() > 240 {
        let drop_n = items.len() - 240;
        items.drain(0..drop_n);
    }
    let serialized = Json::Arr(items).stringify();
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let _ = fs::write(&path, &serialized);
    serialized
}

pub fn record_conversation_exchange(user_text: &str, reply_text: &str, route: &str) {
    record_conversation_exchange_at(&default_memory_jsonl_path(), user_text, reply_text, route);
}

pub fn record_conversation_exchange_at(
    jsonl_path: &Path,
    user_text: &str,
    reply_text: &str,
    route: &str,
) {
    let clean_user = strip_injected_context_prefix(user_text).trim();
    let clean_reply = reply_text.trim();
    if clean_user.is_empty() || clean_reply.is_empty() || is_meta_memory_prompt(clean_user) {
        return;
    }
    if let Some(parent) = jsonl_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    let ts = now_epoch_secs();
    let user_trunc = truncate_chars(clean_user, 900);
    let reply_trunc = truncate_chars(clean_reply, 1400);
    let line = format!(
        r#"{{"ts":{ts},"route":{},"user":{},"reply":{}}}"#,
        quote(route),
        quote(&user_trunc),
        quote(&reply_trunc),
    );

    if let Ok(existing) = fs::read_to_string(jsonl_path) {
        let mut lines: Vec<&str> = existing.lines().filter(|l| !l.trim().is_empty()).collect();
        if let Some(last) = lines.last() {
            if let Ok(j) = Json::parse(last) {
                let prev_u = j.get("user").and_then(Json::as_str).unwrap_or("");
                if prev_u == user_trunc {
                    lines.pop();
                }
            }
        }
        if lines.len() >= MAX_MEMORY_JSONL_LINES {
            let drop_n = lines.len() + 1 - MAX_MEMORY_JSONL_LINES;
            lines.drain(0..drop_n);
        }
        let mut out = String::with_capacity(existing.len() + line.len() + 2);
        for l in lines {
            out.push_str(l);
            out.push('\n');
        }
        out.push_str(&line);
        out.push('\n');
        let _ = fs::write(jsonl_path, out);
        return;
    }

    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(jsonl_path) {
        let _ = writeln!(f, "{line}");
    }
}

pub fn retrieve_local_conversation_json(query: &str) -> String {
    let trimmed = query.trim();
    let terms = extract_conversation_query_terms(trimmed);
    let terms_json: Vec<String> = terms.iter().map(|t| quote(t)).collect();
    match retrieve_local_conversation_context_with_mode(trimmed, true) {
        Some(ctx) => {
            let src_json: Vec<String> = ctx.sources_matched.iter().map(|s| quote(s)).collect();
            format!(
                r#"{{"found_verified_match":{},"whatsapp_tcc_blocked":{},"elapsed_ms":{:.2},"terms":[{}],"sources":[{}],"context":{}}}"#,
                ctx.found_verified_match,
                ctx.whatsapp_tcc_blocked,
                ctx.elapsed_ms,
                terms_json.join(","),
                src_json.join(","),
                quote(&ctx.prompt_block),
            )
        }
        None => format!(
            r#"{{"found_verified_match":false,"whatsapp_tcc_blocked":false,"elapsed_ms":0.0,"terms":[{}],"sources":[],"context":""}}"#,
            terms_json.join(",")
        ),
    }
}

pub fn retrieve_local_conversation_context(user_prompt: &str) -> Option<LocalConversationContext> {
    retrieve_local_conversation_context_with_mode(user_prompt, false)
}

fn retrieve_local_conversation_context_with_mode(
    user_prompt: &str,
    force: bool,
) -> Option<LocalConversationContext> {
    let trimmed = user_prompt.trim();
    if trimmed.is_empty() {
        return None;
    }
    let lower = trimmed.to_ascii_lowercase();
    if !force && !is_conversation_or_memory_query(&lower) {
        return None;
    }

    let t0 = Instant::now();
    let terms = extract_conversation_query_terms(trimmed);
    #[cfg_attr(not(target_os = "macos"), allow(unused_variables))]
    let direction_intent = parse_message_direction_intent(trimmed);
    let mut sections: Vec<String> = Vec::new();
    let mut sources: Vec<String> = Vec::new();
    let mut found_verified_match = false;

    let wants_whatsapp = lower.contains("whatsapp") || lower.contains("whats app");
    let wants_imessage = lower.contains("imessage") || lower.contains("sms");
    let wants_telegram = lower.contains("telegram");
    let wants_slack = lower.contains("slack");
    let wants_notes = lower.contains("note");
    let wants_specific_external_messaging =
        wants_whatsapp || wants_imessage || wants_telegram || wants_slack;
    let asks_external_messaging = wants_specific_external_messaging
        || lower.contains("last message")
        || lower.contains("latest message")
        || lower.contains("recent message")
        || lower.contains("recent messages")
        || lower.contains("unread message")
        || lower.contains("message to ")
        || lower.contains("messages to ")
        || lower.contains("message from ")
        || lower.contains("messages from ")
        || lower.contains("message with ")
        || lower.contains("messages with ")
        || lower.contains("sent to ")
        || lower.contains("sent me")
        || lower.contains("send me")
        || lower.contains("messaged ")
        || lower.contains("texted ")
        || lower.contains("chat with ")
        || lower.contains("chats with ");
    let _wants_any_external = !wants_specific_external_messaging && !wants_notes;

    #[allow(unused_mut)]
    let mut whatsapp_tcc_blocked = false;
    #[allow(unused_mut)]
    let mut whatsapp_found_match_in_window = false;

    #[cfg(target_os = "macos")]
    {
        if wants_whatsapp || _wants_any_external {
            match macos_sources::query_whatsapp_sqlite(&terms, direction_intent) {
                macos_sources::SqliteOutcome::Rows(block) => {
                    sections.push(block);
                    sources.push("WhatsApp SQLite".to_string());
                    found_verified_match = true;
                }
                macos_sources::SqliteOutcome::PermissionDenied => {
                    whatsapp_tcc_blocked = true;
                }
                macos_sources::SqliteOutcome::NotFound | macos_sources::SqliteOutcome::Empty => {}
            }
        }

        if wants_imessage || _wants_any_external {
            if let macos_sources::SqliteOutcome::Rows(block) =
                macos_sources::query_imessage_sqlite(&terms, direction_intent)
            {
                sections.push(block);
                sources.push("Apple Messages SQLite".to_string());
                found_verified_match = true;
            }
        }

        if wants_notes || (_wants_any_external && !asks_external_messaging) {
            if let macos_sources::SqliteOutcome::Rows(block) =
                macos_sources::query_apple_notes_sqlite(&terms)
            {
                sections.push(block);
                sources.push("Apple Notes SQLite".to_string());
                found_verified_match = true;
            }
        }

        // Silent background window perception via CoreGraphics + Apple Vision 2D spatial OCR.
        let mut target_apps: Vec<&str> = Vec::new();
        if wants_whatsapp {
            target_apps.push("whatsapp");
        }
        if wants_imessage {
            target_apps.push("messages");
        }
        if wants_telegram {
            target_apps.push("telegram");
        }
        if wants_slack {
            target_apps.push("slack");
        }
        if wants_notes {
            target_apps.push("notes");
        }
        if target_apps.is_empty() {
            if asks_external_messaging {
                target_apps.extend(["whatsapp", "messages", "telegram", "slack"]);
            } else {
                target_apps.extend(["whatsapp", "messages", "telegram", "slack", "notes"]);
            }
        }

        let allow_bg_launch =
            wants_whatsapp || wants_imessage || wants_telegram || wants_slack || asks_external_messaging;
        for (app_label, win_block, matched_term) in macos_sources::perceive_live_messaging_windows(
            &target_apps,
            &terms,
            allow_bg_launch,
            direction_intent,
        ) {
            if app_label.eq_ignore_ascii_case("WhatsApp") && matched_term {
                whatsapp_found_match_in_window = true;
            }
            if matched_term {
                found_verified_match = true;
                sources.push(format!("{app_label} Live Window"));
            }
            sections.push(win_block);
        }
    }

    // Search Kronos persistent conversation memory + saved Web UI sessions only when the query is not
    // asking for an external messaging thread (never confuse Kronos user prompts with WhatsApp messages).
    if !asks_external_messaging {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let mem_jsonl = home
            .join(".kronos")
            .join("device_context")
            .join("conversation_memory.jsonl");
        let sessions_dir = home.join(".kronos").join("ui_state").join("sessions");
        if let Some(kronos_mem) =
            query_kronos_saved_conversations(&mem_jsonl, &sessions_dir, &terms, 6)
        {
            sections.push(kronos_mem);
            sources.push("Kronos Saved Conversations".to_string());
            found_verified_match = true;
        }
    }

    if whatsapp_tcc_blocked && !whatsapp_found_match_in_window && (wants_whatsapp || !terms.is_empty()) {
        let term_note = if terms.is_empty() {
            "older off-screen WhatsApp threads".to_string()
        } else {
            format!("off-screen WhatsApp history for `{}`", terms.join(", "))
        };
        sections.push(format!(
            "Local WhatsApp Archive Note: `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite` exists on this Mac, and macOS Privacy (Full Disk Access) has not yet been enabled for the Kronos server binary, so only the currently visible WhatsApp desktop window rows could be inspected. No verified message for {term_note} is visible on screen right now. Tell the user truthfully that this chat is scrolled off-screen and that enabling Full Disk Access in macOS System Settings -> Privacy & Security -> Full Disk Access (or scrolling that chat into view in WhatsApp) lets Kronos read it directly. Never guess 'Hello' or attribute another contact's message."
        ));
    }

    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if sections.is_empty() {
        let fallback = format!(
            "[Local Conversations & Memory · retrieved in {:.1} ms]\nKronos runs locally on this user-owned device. No verified conversation records matched query terms: {}. State clearly that no matching local message was found rather than guessing a placeholder message.",
            elapsed_ms,
            if terms.is_empty() {
                "(general summary)".to_string()
            } else {
                terms.join(", ")
            }
        );
        return Some(LocalConversationContext {
            prompt_block: fallback,
            sources_matched: sources,
            found_verified_match: false,
            whatsapp_tcc_blocked,
            elapsed_ms,
        });
    }

    let prompt_block = format!(
        "[Local Conversations & Memory on This User-Owned Device · retrieved in {:.1} ms]\n\
         Kronos runs locally on the user's own hardware with direct permission to assist with their conversations and context. \
         Never output stock refusal disclaimers about being unable to access messaging apps, and never fabricate or guess a message (such as 'Hello') or attribute another contact's message when a queried contact is scrolled off-screen. \
         Use strictly the verified local conversation data below to answer the user's question accurately and concisely.\n\n{}",
        elapsed_ms,
        sections.join("\n\n")
    );

    Some(LocalConversationContext {
        prompt_block,
        sources_matched: sources,
        found_verified_match,
        whatsapp_tcc_blocked,
        elapsed_ms,
    })
}

#[derive(Clone, Debug)]
struct ScoredExchange {
    score: f64,
    ts: u64,
    source_title: String,
    user: String,
    reply: String,
}

pub fn query_kronos_saved_conversations(
    memory_jsonl: &Path,
    sessions_dir: &Path,
    terms: &[String],
    limit: usize,
) -> Option<String> {
    let mut items: Vec<ScoredExchange> = Vec::new();
    let mut seen_users: HashSet<String> = HashSet::new();

    if let Ok(raw) = fs::read_to_string(memory_jsonl) {
        for line in raw.lines().rev().take(400) {
            let Ok(j) = Json::parse(line) else {
                continue;
            };
            let user = j
                .get("user")
                .and_then(Json::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            let reply = j
                .get("reply")
                .and_then(Json::as_str)
                .unwrap_or("")
                .trim()
                .to_string();
            if user.is_empty() || reply.is_empty() || is_meta_memory_prompt(&user) {
                continue;
            }
            let dedup_key = user.to_ascii_lowercase();
            if !seen_users.insert(dedup_key) {
                continue;
            }
            let ts = j.get("ts").and_then(Json::as_f64).unwrap_or(0.0) as u64;
            let score = score_conversation_item("", &user, &reply, ts, terms);
            if terms.is_empty() || score >= 10.0 {
                items.push(ScoredExchange {
                    score,
                    ts,
                    source_title: "Kronos Quick / Turn Memory".to_string(),
                    user,
                    reply,
                });
            }
        }
    }

    if let Ok(rd) = fs::read_dir(sessions_dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(raw) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(parsed) = Json::parse(&raw) else {
                continue;
            };
            let title = parsed
                .get("title")
                .and_then(Json::as_str)
                .unwrap_or("Saved Session")
                .to_string();
            let updated_at = parsed
                .get("updated_at")
                .or_else(|| parsed.get("created_at"))
                .and_then(Json::as_f64)
                .unwrap_or(0.0) as u64;
            let ts_secs = if updated_at > 10_000_000_000 {
                updated_at / 1000
            } else {
                updated_at
            };

            let Some(msgs) = parsed.get("messages").and_then(Json::as_arr) else {
                continue;
            };
            let mut pending_user = String::new();
            for m in msgs {
                let role = m.get("role").and_then(Json::as_str).unwrap_or("");
                let content = m
                    .get("displayText")
                    .or_else(|| m.get("content"))
                    .or_else(|| m.get("text"))
                    .and_then(Json::as_str)
                    .unwrap_or("")
                    .trim();
                if content.is_empty() {
                    continue;
                }
                if role == "user" {
                    pending_user = strip_injected_context_prefix(content).to_string();
                } else if role == "assistant" && !pending_user.is_empty() {
                    if !is_meta_memory_prompt(&pending_user) {
                        let key = pending_user.to_ascii_lowercase();
                        if seen_users.insert(key) {
                            let score =
                                score_conversation_item(&title, &pending_user, content, ts_secs, terms);
                            if terms.is_empty() || score >= 10.0 {
                                items.push(ScoredExchange {
                                    score,
                                    ts: ts_secs,
                                    source_title: title.clone(),
                                    user: pending_user.clone(),
                                    reply: content.to_string(),
                                });
                            }
                        }
                    }
                    pending_user.clear();
                }
            }
        }
    }

    if items.is_empty() {
        return None;
    }

    items.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.ts.cmp(&a.ts))
    });
    items.truncate(limit);

    let mut out = String::from("Saved Kronos Conversations on This Device:\n");
    for (idx, it) in items.iter().enumerate() {
        let age = format_relative_age(it.ts);
        out.push_str(&format!(
            "  {}. [{} · {}]\n     User: {}\n     Kronos: {}\n",
            idx + 1,
            truncate_chars(&it.source_title, 48),
            age,
            truncate_chars(&collapse_ws(&it.user), 220),
            truncate_chars(&collapse_ws(&it.reply), 320),
        ));
    }
    Some(out.trim_end().to_string())
}

fn is_meta_memory_prompt(user: &str) -> bool {
    let low = user.to_ascii_lowercase();
    low.contains("last message")
        || low.contains("latest message")
        || low.contains("message from ")
        || low.contains("sent me")
        || low.contains("send me")
        || low.starts_with("open whatsapp")
        || low.starts_with("message ")
        || low.contains("remind me of important conversations")
        || low.contains("what did we talk about")
}

fn score_conversation_item(
    title: &str,
    user: &str,
    reply: &str,
    ts_secs: u64,
    terms: &[String],
) -> f64 {
    let combined = format!("{title} {user} {reply}").to_ascii_lowercase();
    let mut score = 0.0;
    for t in terms {
        if combined.contains(t.as_str()) {
            score += 18.0;
            if title.to_ascii_lowercase().contains(t.as_str())
                || user.to_ascii_lowercase().contains(t.as_str())
            {
                score += 8.0;
            }
        }
    }

    let importance_cues = [
        "important",
        "deadline",
        "deploy",
        "release",
        "decision",
        "todo",
        "action",
        "fix",
        "schedule",
        "meeting",
        "architecture",
        "roadmap",
        "plan",
    ];
    for cue in importance_cues {
        if combined.contains(cue) {
            score += 1.5;
        }
    }

    let now = now_epoch_secs();
    if ts_secs > 0 && now >= ts_secs {
        let hours_ago = ((now - ts_secs) as f64) / 3600.0;
        score += 6.0 / (1.0 + hours_ago * 0.15);
    }
    score
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn format_relative_age(ts_secs: u64) -> String {
    if ts_secs == 0 {
        return "saved".to_string();
    }
    let now = now_epoch_secs();
    if now <= ts_secs {
        return "just now".to_string();
    }
    let diff = now - ts_secs;
    if diff < 60 {
        "just now".to_string()
    } else if diff < 3600 {
        format!("{}m ago", diff / 60)
    } else if diff < 86_400 {
        format!("{}h ago", diff / 3600)
    } else {
        format!("{}d ago", diff / 86_400)
    }
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let head: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

#[cfg(target_os = "macos")]
mod macos_sources {
    use super::*;
    use std::ffi::{c_char, c_void, CStr};
    use std::process::Command;
    use std::sync::atomic::{AtomicU64, Ordering};

    use crate::vision::{analyze_image_bytes, OcrRegion};

    static SNAPSHOT_SEQ: AtomicU64 = AtomicU64::new(1);

    pub enum SqliteOutcome {
        Rows(String),
        Empty,
        PermissionDenied,
        NotFound,
    }

    fn stage_sqlite_snapshot(db_path: &Path) -> Result<PathBuf, SqliteOutcome> {
        if !db_path.exists() {
            return Err(SqliteOutcome::NotFound);
        }
        match fs::File::open(db_path) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                return Err(SqliteOutcome::PermissionDenied);
            }
            Err(_) => return Err(SqliteOutcome::NotFound),
        }

        let seq = SNAPSHOT_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "kronos_sql_snap_{}_{}",
            std::process::id(),
            seq
        ));
        if fs::create_dir_all(&dir).is_err() {
            return Err(SqliteOutcome::NotFound);
        }
        let fname = db_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("db.sqlite");
        let dst = dir.join(fname);
        if fs::copy(db_path, &dst).is_err() {
            let _ = fs::remove_dir_all(&dir);
            return Err(SqliteOutcome::PermissionDenied);
        }
        for suffix in ["-wal", "-shm"] {
            let side_src = PathBuf::from(format!("{}{suffix}", db_path.display()));
            if side_src.exists() {
                let side_dst = PathBuf::from(format!("{}{suffix}", dst.display()));
                let _ = fs::copy(&side_src, &side_dst);
            }
        }
        Ok(dst)
    }

    fn run_sqlite_query(staged_db: &Path, sql: &str) -> Vec<Vec<String>> {
        let out = Command::new("/usr/bin/sqlite3")
            .arg("-readonly")
            .arg("-separator")
            .arg("\x1f")
            .arg(staged_db)
            .arg(sql)
            .output();
        let Ok(out) = out else {
            return Vec::new();
        };
        if !out.status.success() {
            return Vec::new();
        }
        let text = String::from_utf8_lossy(&out.stdout);
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.split('\x1f').map(|c| c.trim().to_string()).collect())
            .collect()
    }

    fn sql_escape_like(term: &str) -> String {
        term.chars()
            .filter(|c| c.is_alphanumeric() || *c == ' ' || *c == '-' || *c == '_')
            .collect::<String>()
            .to_ascii_lowercase()
    }

    fn split_contact_alnum_words(s: &str) -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_ascii_lowercase())
            .collect()
    }

    pub(super) fn contact_matches_query_term(contact: &str, term: &str) -> bool {
        let term_words = split_contact_alnum_words(term);
        if term_words.is_empty() {
            return false;
        }
        let contact_words = split_contact_alnum_words(contact);
        if contact_words.is_empty() {
            return false;
        }
        if term_words.len() == 1 {
            let tw = &term_words[0];
            return contact_words
                .iter()
                .any(|cw| cw == tw || (tw.len() >= 5 && cw.starts_with(tw.as_str())));
        }
        let all_words_present = term_words
            .iter()
            .all(|tw| contact_words.iter().any(|cw| cw == tw));
        if all_words_present {
            return true;
        }
        let collapsed: String = term_words.concat();
        collapsed.len() >= 4 && contact_words.iter().any(|cw| cw == &collapsed)
    }

    fn format_sqlite_directional_lead(
        rows: &[Vec<String>],
        intent: MessageDirectionIntent,
    ) -> Option<String> {
        let latest_in = rows.iter().find(|r| r.len() >= 4 && r[2] != "1");
        let latest_out = rows.iter().find(|r| r.len() >= 4 && r[2] == "1");
        match intent {
            MessageDirectionIntent::IncomingFromContact => {
                if let Some(inc) = latest_in {
                    let mut s = format!(
                        "  VERIFIED DIRECT ANSWER (Latest INCOMING message FROM `{}` TO You, {}): \"{}\"",
                        inc[0],
                        inc[1],
                        truncate_chars(&inc[3], 260)
                    );
                    if let Some(out_row) = latest_out {
                        if rows.first().is_some_and(|top| top.len() >= 4 && top[2] == "1") {
                            s.push_str(&format!(
                                " (Context note: \"{}\" at {} was YOUR outgoing reply to `{}`, not a message from `{}` to you.)",
                                truncate_chars(&out_row[3], 140),
                                out_row[1],
                                inc[0],
                                inc[0]
                            ));
                        }
                    }
                    Some(s)
                } else {
                    latest_out.map(|out_row| {
                        format!(
                            "  DIRECTIONALITY NOTE: All visible matching SQLite rows are YOUR outgoing messages to `{}` (latest outgoing at {}: \"{}\"); no incoming message from `{}` was found in the fetched window.",
                            out_row[0],
                            out_row[1],
                            truncate_chars(&out_row[3], 200),
                            out_row[0]
                        )
                    })
                }
            }
            MessageDirectionIntent::OutgoingFromUser => latest_out.map(|out_row| {
                format!(
                    "  VERIFIED DIRECT ANSWER (Latest OUTGOING message FROM You TO `{}`, {}): \"{}\"",
                    out_row[0],
                    out_row[1],
                    truncate_chars(&out_row[3], 260)
                )
            }),
            MessageDirectionIntent::ThreadOverview => None,
        }
    }

    pub fn query_whatsapp_sqlite(
        terms: &[String],
        intent: MessageDirectionIntent,
    ) -> SqliteOutcome {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return SqliteOutcome::NotFound;
        };
        let candidates = [
            home.join("Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite"),
            home.join("Library/Group Containers/group.net.whatsapp.WhatsAppSMB.shared/ChatStorage.sqlite"),
        ];

        let mut saw_perm_denied = false;
        for cand in &candidates {
            let staged = match stage_sqlite_snapshot(cand) {
                Ok(p) => p,
                Err(SqliteOutcome::PermissionDenied) => {
                    saw_perm_denied = true;
                    continue;
                }
                Err(other) => {
                    let _ = other;
                    continue;
                }
            };

            let mut lines = Vec::new();
            if !terms.is_empty() {
                let partner_preds: Vec<String> = terms
                    .iter()
                    .map(|t| sql_escape_like(t))
                    .filter(|t| !t.is_empty())
                    .map(|t| {
                        let sub_parts: Vec<&str> =
                            t.split(['-', '_', ' ']).filter(|p| !p.is_empty()).collect();
                        if sub_parts.len() >= 2 {
                            let and_parts: Vec<String> = sub_parts
                                .iter()
                                .map(|p| format!("LOWER(COALESCE(cs.ZPARTNERNAME,'')) LIKE '%{p}%'"))
                                .collect();
                            format!(
                                "(LOWER(COALESCE(cs.ZPARTNERNAME,'')) LIKE '%{t}%' OR ({}))",
                                and_parts.join(" AND ")
                            )
                        } else {
                            format!("LOWER(COALESCE(cs.ZPARTNERNAME,'')) LIKE '%{t}%'")
                        }
                    })
                    .collect();
                if !partner_preds.is_empty() {
                    let sql = format!(
                        "SELECT COALESCE(cs.ZPARTNERNAME, cs.ZCONTACTJID, 'Unknown'), \
                                datetime(CAST(m.ZMESSAGEDATE + {APPLE_COREDATA_EPOCH_OFFSET} AS INTEGER), 'unixepoch', 'localtime'), \
                                COALESCE(m.ZISFROMME, 0), \
                                REPLACE(REPLACE(COALESCE(m.ZTEXT, '[Media/Attachment]'), char(10), ' '), char(13), ' ') \
                         FROM ZWAMESSAGE m \
                         JOIN ZWACHATSESSION cs ON m.ZCHATSESSION = cs.Z_PK \
                         WHERE ({}) AND m.ZTEXT IS NOT NULL AND LENGTH(TRIM(m.ZTEXT)) > 0 \
                         ORDER BY m.ZMESSAGEDATE DESC LIMIT 20;",
                        partner_preds.join(" OR ")
                    );
                    let rows = run_sqlite_query(&staged, &sql);
                    if !rows.is_empty() {
                        lines.push(format!(
                            "WhatsApp Direct Thread History with `{}` (newest first; `You ->` = sent by you, `From` = sent to you):",
                            terms.join(", ")
                        ));
                        if let Some(lead) = format_sqlite_directional_lead(&rows, intent) {
                            lines.push(lead);
                        }
                        for r in &rows {
                            if r.len() >= 4 {
                                let dir = if r[2] == "1" { "You -> " } else { "From " };
                                lines.push(format!(
                                    "  - [{}] {}{}: {}",
                                    r[1],
                                    dir,
                                    r[0],
                                    truncate_chars(&r[3], 260)
                                ));
                            }
                        }
                    }
                }
            }

            if terms.is_empty() && lines.is_empty() {
                let recent_sql = format!(
                    "SELECT COALESCE(cs.ZPARTNERNAME, cs.ZCONTACTJID, 'Unknown'), \
                            datetime(CAST(m.ZMESSAGEDATE + {APPLE_COREDATA_EPOCH_OFFSET} AS INTEGER), 'unixepoch', 'localtime'), \
                            COALESCE(m.ZISFROMME, 0), \
                            REPLACE(REPLACE(COALESCE(m.ZTEXT, ''), char(10), ' '), char(13), ' ') \
                     FROM ZWAMESSAGE m \
                     JOIN ZWACHATSESSION cs ON m.ZCHATSESSION = cs.Z_PK \
                     WHERE m.ZTEXT IS NOT NULL AND LENGTH(TRIM(m.ZTEXT)) > 0 \
                     ORDER BY m.ZMESSAGEDATE DESC LIMIT 10;"
                );
                let recent_rows = run_sqlite_query(&staged, &recent_sql);
                if !recent_rows.is_empty() {
                    lines.push("Recent WhatsApp Messages on This Mac (newest first):".to_string());
                    for r in &recent_rows {
                        if r.len() >= 4 {
                            let dir = if r[2] == "1" { "You -> " } else { "From " };
                            lines.push(format!(
                                "  - [{}] {}{}: {}",
                                r[1],
                                dir,
                                r[0],
                                truncate_chars(&r[3], 240)
                            ));
                        }
                    }
                }
            }

            if let Some(parent) = staged.parent() {
                let _ = fs::remove_dir_all(parent);
            }
            if !lines.is_empty() {
                return SqliteOutcome::Rows(lines.join("\n"));
            }
            return SqliteOutcome::Empty;
        }

        if saw_perm_denied {
            SqliteOutcome::PermissionDenied
        } else {
            SqliteOutcome::NotFound
        }
    }

    pub fn query_imessage_sqlite(
        terms: &[String],
        intent: MessageDirectionIntent,
    ) -> SqliteOutcome {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return SqliteOutcome::NotFound;
        };
        let chat_db = home.join("Library/Messages/chat.db");
        let staged = match stage_sqlite_snapshot(&chat_db) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let where_clause = if terms.is_empty() {
            "m.text IS NOT NULL AND LENGTH(TRIM(m.text)) > 0".to_string()
        } else {
            let preds: Vec<String> = terms
                .iter()
                .map(|t| sql_escape_like(t))
                .filter(|t| !t.is_empty())
                .map(|t| {
                    format!(
                        "(LOWER(COALESCE(c.display_name,'')) LIKE '%{t}%' OR LOWER(COALESCE(h.id,'')) LIKE '%{t}%' OR LOWER(COALESCE(m.text,'')) LIKE '%{t}%')"
                    )
                })
                .collect();
            if preds.is_empty() {
                "m.text IS NOT NULL AND LENGTH(TRIM(m.text)) > 0".to_string()
            } else {
                format!(
                    "({}) AND m.text IS NOT NULL AND LENGTH(TRIM(m.text)) > 0",
                    preds.join(" OR ")
                )
            }
        };

        let sql = format!(
            "SELECT COALESCE(NULLIF(c.display_name,''), h.id, c.chat_identifier, 'Contact'), \
                    datetime(CAST((CASE WHEN m.date > 1000000000000 THEN m.date / 1000000000 ELSE m.date END) + {APPLE_COREDATA_EPOCH_OFFSET} AS INTEGER), 'unixepoch', 'localtime'), \
                    COALESCE(m.is_from_me, 0), \
                    REPLACE(REPLACE(m.text, char(10), ' '), char(13), ' ') \
             FROM message m \
             LEFT JOIN handle h ON m.handle_id = h.ROWID \
             LEFT JOIN chat_message_join cmj ON cmj.message_id = m.ROWID \
             LEFT JOIN chat c ON c.ROWID = cmj.chat_id \
             WHERE {where_clause} \
             ORDER BY m.date DESC LIMIT 16;"
        );
        let rows = run_sqlite_query(&staged, &sql);
        if let Some(parent) = staged.parent() {
            let _ = fs::remove_dir_all(parent);
        }
        if rows.is_empty() {
            return SqliteOutcome::Empty;
        }
        let mut lines = vec!["Apple Messages (iMessage / SMS) on This Mac (newest first):".to_string()];
        if !terms.is_empty() {
            if let Some(lead) = format_sqlite_directional_lead(&rows, intent) {
                lines.push(lead);
            }
        }
        for r in &rows {
            if r.len() >= 4 {
                let dir = if r[2] == "1" { "You -> " } else { "From " };
                lines.push(format!(
                    "  - [{}] {}{}: {}",
                    r[1],
                    dir,
                    r[0],
                    truncate_chars(&r[3], 240)
                ));
            }
        }
        SqliteOutcome::Rows(lines.join("\n"))
    }

    pub fn query_apple_notes_sqlite(terms: &[String]) -> SqliteOutcome {
        let Some(home) = std::env::var_os("HOME").map(PathBuf::from) else {
            return SqliteOutcome::NotFound;
        };
        let notes_db =
            home.join("Library/Group Containers/group.com.apple.notes/NoteStore.sqlite");
        let staged = match stage_sqlite_snapshot(&notes_db) {
            Ok(p) => p,
            Err(e) => return e,
        };

        let filter = if terms.is_empty() {
            String::new()
        } else {
            let preds: Vec<String> = terms
                .iter()
                .map(|t| sql_escape_like(t))
                .filter(|t| !t.is_empty())
                .map(|t| {
                    format!(
                        "(LOWER(COALESCE(ZTITLE1,'')) LIKE '%{t}%' OR LOWER(COALESCE(ZSNIPPET,'')) LIKE '%{t}%')"
                    )
                })
                .collect();
            if preds.is_empty() {
                String::new()
            } else {
                format!(" AND ({})", preds.join(" OR "))
            }
        };

        let sql = format!(
            "SELECT COALESCE(ZTITLE1, 'Untitled'), \
                    datetime(CAST(COALESCE(ZMODIFICATIONDATE1, 0) + {APPLE_COREDATA_EPOCH_OFFSET} AS INTEGER), 'unixepoch', 'localtime'), \
                    REPLACE(REPLACE(COALESCE(ZSNIPPET, ''), char(10), ' '), char(13), ' ') \
             FROM ZICCLOUDSYNCINGOBJECT \
             WHERE ZTITLE1 IS NOT NULL AND (ZMARKEDFORDELETION IS NULL OR ZMARKEDFORDELETION = 0){filter} \
             ORDER BY ZMODIFICATIONDATE1 DESC LIMIT 8;"
        );
        let rows = run_sqlite_query(&staged, &sql);
        if let Some(parent) = staged.parent() {
            let _ = fs::remove_dir_all(parent);
        }
        if rows.is_empty() {
            return SqliteOutcome::Empty;
        }
        let mut lines = vec!["Apple Notes on This Mac (most recently modified first):".to_string()];
        for r in &rows {
            if r.len() >= 3 {
                lines.push(format!(
                    "  - [{}] {}: {}",
                    r[1],
                    r[0],
                    truncate_chars(&r[2], 220)
                ));
            }
        }
        SqliteOutcome::Rows(lines.join("\n"))
    }

    type Id = *mut c_void;
    type Sel = *mut c_void;

    unsafe extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn CGWindowListCopyWindowInfo(option: u32, relative_to_window: u32) -> Id;
        fn CFRelease(cf: *const c_void);
        static kCGWindowNumber: Id;
        static kCGWindowLayer: Id;
        static kCGWindowBounds: Id;
        static kCGWindowOwnerName: Id;
    }

    macro_rules! msg_send {
        ($obj:expr, $sel_cstr:expr $(, $arg:expr => $aty:ty)* ; $ret:ty) => {{
            let sel = sel_registerName($sel_cstr.as_ptr());
            let f: unsafe extern "C" fn(Id, Sel $(, $aty)*) -> $ret =
                std::mem::transmute(objc_msgSend as *const ());
            f($obj, sel $(, $arg)*)
        }};
    }

    #[derive(Clone, Debug)]
    struct LiveWindowInfo {
        window_id: u32,
        owner_clean: String,
        area: f64,
    }

    fn strip_bidi_marks(s: &str) -> String {
        s.chars()
            .filter(|c| !matches!(*c, '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}'))
            .collect::<String>()
            .trim()
            .to_string()
    }

    fn list_candidate_app_windows() -> Vec<LiveWindowInfo> {
        let mut found = Vec::new();
        // SAFETY: CoreGraphics `CGWindowListCopyWindowInfo` returns a retained `CFArrayRef`
        // toll-free bridged to `NSArray *` containing `NSDictionary *` entries.
        unsafe {
            let arr = CGWindowListCopyWindowInfo(0, 0);
            if arr.is_null() {
                return found;
            }
            let ns_str_cls = objc_getClass(c"NSString".as_ptr());
            let w_key: Id = msg_send!(ns_str_cls, c"stringWithUTF8String:", c"Width".as_ptr() => *const c_char; Id);
            let h_key: Id = msg_send!(ns_str_cls, c"stringWithUTF8String:", c"Height".as_ptr() => *const c_char; Id);

            let count: usize = msg_send!(arr, c"count"; usize);
            for i in 0..count {
                let dict: Id = msg_send!(arr, c"objectAtIndex:", i => usize; Id);
                if dict.is_null() {
                    continue;
                }
                let layer_num: Id = msg_send!(dict, c"objectForKey:", kCGWindowLayer => Id; Id);
                if layer_num.is_null() {
                    continue;
                }
                let layer: i32 = msg_send!(layer_num, c"intValue"; i32);
                if layer != 0 {
                    continue;
                }
                let owner_ns: Id = msg_send!(dict, c"objectForKey:", kCGWindowOwnerName => Id; Id);
                if owner_ns.is_null() {
                    continue;
                }
                let c_ptr: *const c_char = msg_send!(owner_ns, c"UTF8String"; *const c_char);
                if c_ptr.is_null() {
                    continue;
                }
                let owner_clean = strip_bidi_marks(&CStr::from_ptr(c_ptr).to_string_lossy());
                if owner_clean.is_empty() {
                    continue;
                }
                let wid_num: Id = msg_send!(dict, c"objectForKey:", kCGWindowNumber => Id; Id);
                if wid_num.is_null() {
                    continue;
                }
                let wid: i32 = msg_send!(wid_num, c"intValue"; i32);
                if wid <= 0 {
                    continue;
                }
                let bounds: Id = msg_send!(dict, c"objectForKey:", kCGWindowBounds => Id; Id);
                if bounds.is_null() {
                    continue;
                }
                let w_num: Id = msg_send!(bounds, c"objectForKey:", w_key => Id; Id);
                let h_num: Id = msg_send!(bounds, c"objectForKey:", h_key => Id; Id);
                if w_num.is_null() || h_num.is_null() {
                    continue;
                }
                let width: f64 = msg_send!(w_num, c"doubleValue"; f64);
                let height: f64 = msg_send!(h_num, c"doubleValue"; f64);
                // Skip tiny status/splash auxiliary surfaces (like 500x500 empty helper layers when main window is larger).
                if width < 380.0 || height < 280.0 {
                    continue;
                }
                found.push(LiveWindowInfo {
                    window_id: wid as u32,
                    owner_clean,
                    area: width * height,
                });
            }
            CFRelease(arr.cast::<c_void>());
        }
        found
    }

    fn maybe_launch_app_in_background(target_lower: &str) {
        let app_name = match target_lower {
            "whatsapp" => "WhatsApp",
            "messages" => "Messages",
            "telegram" => "Telegram",
            "slack" => "Slack",
            _ => return,
        };
        let _ = Command::new("/usr/bin/open")
            .args(["-g", "-j", "-a", app_name])
            .status();
        std::thread::sleep(std::time::Duration::from_millis(650));
    }

    pub fn perceive_live_messaging_windows(
        target_apps: &[&str],
        terms: &[String],
        allow_bg_launch: bool,
        intent: MessageDirectionIntent,
    ) -> Vec<(String, String, bool)> {
        let mut windows = list_candidate_app_windows();
        if allow_bg_launch && target_apps.len() == 1 {
            let wanted = target_apps[0];
            let has_win = windows
                .iter()
                .any(|w| w.owner_clean.to_ascii_lowercase().contains(wanted));
            if !has_win {
                maybe_launch_app_in_background(wanted);
                windows = list_candidate_app_windows();
            }
        }

        let mut results = Vec::new();
        for &wanted in target_apps {
            let best = windows
                .iter()
                .filter(|w| {
                    let low = w.owner_clean.to_ascii_lowercase();
                    low.contains(wanted) && !low.contains("linkednotes")
                })
                .max_by(|a, b| {
                    a.area
                        .partial_cmp(&b.area)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            let Some(win) = best else {
                continue;
            };
            let seq = SNAPSHOT_SEQ.fetch_add(1, Ordering::Relaxed);
            let tmp_png = std::env::temp_dir().join(format!(
                "kronos_win_ocr_{}_{}_{}.png",
                std::process::id(),
                win.window_id,
                seq
            ));
            let status = Command::new("/usr/sbin/screencapture")
                .args(["-x", "-o", "-l"])
                .arg(win.window_id.to_string())
                .arg(&tmp_png)
                .status();
            let Ok(st) = status else {
                let _ = fs::remove_file(&tmp_png);
                continue;
            };
            if !st.success() {
                let _ = fs::remove_file(&tmp_png);
                continue;
            }
            let Ok(png_bytes) = fs::read(&tmp_png) else {
                let _ = fs::remove_file(&tmp_png);
                continue;
            };
            let _ = fs::remove_file(&tmp_png);
            if png_bytes.len() < 4096 {
                continue;
            }
            let Ok(perception) = analyze_image_bytes(&png_bytes, "png", None, 0) else {
                continue;
            };
            if perception.ocr_regions.is_empty() {
                continue;
            }
            let (formatted, matched_term) = format_two_pane_messaging_ocr_with_intent(
                &win.owner_clean,
                &perception.ocr_regions,
                terms,
                intent,
            );
            if !formatted.is_empty() {
                results.push((win.owner_clean.clone(), formatted, matched_term));
            }
        }
        results
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum SidebarPreviewDirection {
        IncomingOrUnknown,
        OutgoingFromYou,
        CallOrSystemEvent,
    }

    #[derive(Clone, Debug)]
    struct SidebarChatRow {
        contact: String,
        timestamp: String,
        clean_preview: String,
        direction: SidebarPreviewDirection,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BubbleSide {
        Incoming,
        Outgoing,
    }

    #[derive(Clone, Debug)]
    struct ParsedThreadBubble {
        side: BubbleSide,
        timestamp: String,
        body: String,
        quoted_reply: Option<(String, String)>,
        is_call_event: bool,
    }

    fn is_call_event_text(text: &str) -> bool {
        let low = text
            .trim()
            .trim_start_matches(['*', '\u{2022}', '-', '/'])
            .trim()
            .to_ascii_lowercase();
        low == "missed voice call"
            || low == "missed video call"
            || low == "click to call back"
            || low == "voice call"
            || low == "video call"
            || low == "no answer"
            || low.starts_with("missed voice call")
            || low.starts_with("missed video call")
    }

    fn is_centered_thread_separator_pill(text: &str, left_x: f32, width_n: f32, pane_left: f32) -> bool {
        let low = text.trim().to_ascii_lowercase();
        let exact_sep = matches!(
            low.as_str(),
            "today"
                | "yesterday"
                | "monday"
                | "tuesday"
                | "wednesday"
                | "thursday"
                | "friday"
                | "saturday"
                | "sunday"
                | "1 unread message"
        ) || low.ends_with(" unread messages")
            || low.contains("messages and calls are end-to-end encrypted");
        if exact_sep {
            return true;
        }
        let pane_center = pane_left + (1.0 - pane_left) * 0.5;
        let cx = left_x + width_n * 0.5;
        width_n > 0.0
            && width_n < 0.16
            && (cx - pane_center).abs() < 0.06
            && (low == "unread" || low.contains("unread message"))
    }

    /// Parses pure timestamp spans such as `"03:56"`, `"06:45 //"`, `"16:12 V"`, `"7:41 PM"`, `"Edited 17:03"`,
    /// returning `(clean_timestamp, has_outgoing_checkmark)` or `None` if the span also carries message words.
    fn parse_standalone_timestamp_span(raw: &str) -> Option<(String, bool)> {
        let mut s = raw.trim();
        if let Some(rest) = s.strip_prefix("Edited ").or_else(|| s.strip_prefix("edited ")) {
            s = rest.trim();
        }
        let mut has_tick = false;
        for suffix in [" //", " /", " V", " \u{2713}\u{2713}", " \u{2713}", "//", "\u{2713}\u{2713}", "\u{2713}"] {
            if let Some(stripped) = s.strip_suffix(suffix) {
                s = stripped.trim();
                has_tick = true;
                break;
            }
        }
        let core = s.strip_suffix(" AM")
            .or_else(|| s.strip_suffix(" PM"))
            .or_else(|| s.strip_suffix(" am"))
            .or_else(|| s.strip_suffix(" pm"))
            .unwrap_or(s)
            .trim();
        let (hh, mm) = core.split_once(':')?;
        if (1..=2).contains(&hh.len())
            && mm.len() == 2
            && hh.chars().all(|c| c.is_ascii_digit())
            && mm.chars().all(|c| c.is_ascii_digit())
        {
            Some((s.to_string(), has_tick))
        } else {
            None
        }
    }

    /// Strips trailing inline timestamp + checkmark when Vision merges `"Good is great 17:01 /"` into one span.
    fn split_trailing_inline_timestamp(raw: &str) -> (String, Option<(String, bool)>) {
        let trimmed = raw.trim();
        if parse_standalone_timestamp_span(trimmed).is_some() {
            return (String::new(), parse_standalone_timestamp_span(trimmed));
        }
        let tokens: Vec<&str> = trimmed.split_whitespace().collect();
        if tokens.len() >= 2 {
            for tail_len in [3usize, 2, 1] {
                if tokens.len() > tail_len {
                    let cand = tokens[tokens.len() - tail_len..].join(" ");
                    if let Some(ts) = parse_standalone_timestamp_span(&cand) {
                        let body = tokens[..tokens.len() - tail_len].join(" ");
                        if !body.is_empty() {
                            return (body, Some(ts));
                        }
                    }
                }
            }
        }
        (trimmed.to_string(), None)
    }

    fn classify_sidebar_preview(
        raw_preview: &str,
        contact_left_x: f32,
        preview_first_span: Option<&&OcrRegion>,
    ) -> (String, SidebarPreviewDirection) {
        let trimmed = raw_preview.trim();
        if trimmed.is_empty() {
            return (String::new(), SidebarPreviewDirection::IncomingOrUnknown);
        }
        if is_call_event_text(trimmed) {
            let clean = trimmed
                .trim_start_matches(['*', '\u{2022}', '-'])
                .trim()
                .to_string();
            return (clean, SidebarPreviewDirection::CallOrSystemEvent);
        }

        // Apple Vision OCR transcribes WhatsApp's outgoing single/double checkmarks in sidebar previews
        // as leading `/ `, `// `, `V `, `✓`, `✓✓`, or explicit `You: ` / `You reacted`.
        for prefix in [
            "// ",
            "/ ",
            "\u{2713}\u{2713} ",
            "\u{2713} ",
            "\u{2714} ",
            "V ",
            "You: ",
            "you: ",
        ] {
            if let Some(rest) = trimmed.strip_prefix(prefix) {
                return (rest.trim().to_string(), SidebarPreviewDirection::OutgoingFromYou);
            }
        }
        let low = trimmed.to_ascii_lowercase();
        if low.starts_with("you reacted ")
            || low.starts_with("you deleted ")
            || low.starts_with("you sent ")
        {
            return (trimmed.to_string(), SidebarPreviewDirection::OutgoingFromYou);
        }

        // Visual checkmark-icon fallback when Vision OCR drops the `✓✓` glyph itself:
        // WhatsApp indents outgoing sidebar previews by ~0.013..0.026 to make room for the tick glyph.
        if let Some(p_span) = preview_first_span {
            let indented_for_icon = p_span.left_x >= contact_left_x + 0.012;
            let starts_with_media_noun = [
                "photo", "video", "voice", "audio", "sticker", "gif", "document", "contact", "poll", "missed",
            ]
            .iter()
            .any(|m| low.starts_with(m));
            if p_span.has_left_tick_icon && indented_for_icon && !starts_with_media_noun {
                return (trimmed.to_string(), SidebarPreviewDirection::OutgoingFromYou);
            }
        }

        (trimmed.to_string(), SidebarPreviewDirection::IncomingOrUnknown)
    }

    fn classify_region_bubble_side(
        reg: &OcrRegion,
        pane_left: f32,
        inline_tick: Option<bool>,
        prev_side_and_y: Option<(BubbleSide, f32)>,
    ) -> BubbleSide {
        let [r, g, b] = [
            i32::from(reg.bg_rgb[0]),
            i32::from(reg.bg_rgb[1]),
            i32::from(reg.bg_rgb[2]),
        ];
        let has_pixel_sample = r != 0 || g != 0 || b != 0;

        // 1. Physical bubble tint: WhatsApp outgoing emerald/mint green (`G - R >= 15 && G - B >= 7`)
        //    or Apple Messages outgoing blue (`B - R >= 24 && B - G >= 14`).
        if has_pixel_sample {
            if (g - r >= 15 && g - b >= 7) || (b - r >= 24 && b - g >= 14) {
                return BubbleSide::Outgoing;
            }
            // Non-green, non-blue neutral gray/white bubble surface -> Incoming from contact.
            if (g - r) <= 10 && (b - r) <= 18 {
                if inline_tick == Some(true) {
                    return BubbleSide::Outgoing;
                }
                return BubbleSide::Incoming;
            }
        }

        // 2. Outgoing delivery/read checkmark on timestamp (`06:45 //`, `16:12 V`, `06:45 ✓✓`).
        if inline_tick == Some(true) {
            return BubbleSide::Outgoing;
        }

        // 3. Bare standalone timestamp (`03:56`) positioned at the right corner of a wide incoming bubble:
        //    inherit the immediately preceding bubble line's side when within `dy <= 0.028`.
        if inline_tick == Some(false) {
            if let Some((prev_side, prev_y)) = prev_side_and_y {
                if (reg.top_y - prev_y).abs() <= 0.028 {
                    return prev_side;
                }
            }
        }

        // 4. Geometry fallback when pixel buffer is absent (e.g. synthetic layout unit tests).
        let pane_width = (1.0 - pane_left).max(0.25);
        let right_x = reg.left_x + reg.width_n;
        if right_x >= 0.89 || reg.left_x >= pane_left + pane_width * 0.23 {
            BubbleSide::Outgoing
        } else {
            BubbleSide::Incoming
        }
    }

    fn parse_open_thread_bubbles(
        body_spans: &[&OcrRegion],
        pane_left: f32,
        contact_hint: &str,
        terms: &[String],
    ) -> Vec<ParsedThreadBubble> {
        #[derive(Clone)]
        struct RawSpan {
            side: BubbleSide,
            top_y: f32,
            text: String,
            ts: Option<String>,
            has_quote_bar: bool,
        }

        let mut cleaned: Vec<RawSpan> = Vec::with_capacity(body_spans.len());
        let mut prev_ctx: Option<(BubbleSide, f32)> = None;
        for &reg in body_spans {
            if is_centered_thread_separator_pill(&reg.text, reg.left_x, reg.width_n, pane_left) {
                continue;
            }
            let (body_part, ts_info) = split_trailing_inline_timestamp(&reg.text);
            let tick_flag = ts_info.as_ref().map(|(_, has_tick)| *has_tick);
            let side = classify_region_bubble_side(reg, pane_left, tick_flag, prev_ctx);
            if !body_part.is_empty() {
                prev_ctx = Some((side, reg.top_y));
            }
            cleaned.push(RawSpan {
                side,
                top_y: reg.top_y,
                text: body_part,
                ts: ts_info.map(|(t, _)| t),
                has_quote_bar: reg.has_left_quote_bar,
            });
        }

        // Group consecutive spans on the same side into discrete message bubbles.
        // A standalone or inline timestamp marks the bottom-right footer of a WhatsApp bubble.
        let mut groups: Vec<Vec<RawSpan>> = Vec::new();
        let mut prev_ended_with_ts = false;
        for sp in cleaned {
            let start_new = match groups.last() {
                None => true,
                Some(cur) => {
                    let last = cur.last().unwrap();
                    let dy = (sp.top_y - last.top_y).abs();
                    let pure_ts_attach = sp.text.is_empty() && sp.ts.is_some() && sp.side == last.side && dy <= 0.036;
                    if pure_ts_attach {
                        false
                    } else {
                        sp.side != last.side || prev_ended_with_ts || dy > 0.068
                    }
                }
            };
            prev_ended_with_ts = sp.ts.is_some();
            if start_new {
                groups.push(vec![sp]);
            } else if let Some(cur) = groups.last_mut() {
                cur.push(sp);
            }
        }

        let is_quote_author_header = |line: &str| -> bool {
            let t = line.trim();
            if t.is_empty() {
                return false;
            }
            if t.eq_ignore_ascii_case("you") {
                return true;
            }
            if !contact_hint.is_empty() && contact_matches_query_term(contact_hint, t) {
                return true;
            }
            if terms.iter().any(|term| contact_matches_query_term(t, term)) {
                return true;
            }
            t.starts_with('+') && t.contains('~')
        };

        let mut seen_bodies: Vec<String> = Vec::new();
        let mut bubbles: Vec<ParsedThreadBubble> = Vec::with_capacity(groups.len());

        for grp in groups {
            let side = grp[0].side;
            let mut timestamp = String::new();
            let mut lines: Vec<(String, bool)> = Vec::new();
            for sp in grp {
                if let Some(ts) = sp.ts {
                    timestamp = ts;
                }
                if !sp.text.is_empty() {
                    lines.push((sp.text, sp.has_quote_bar));
                }
            }
            if lines.is_empty() {
                continue;
            }

            // Separate embedded reply-quote card at the top of a bubble from the actual message text below it.
            // Pattern A: pixel-verified vertical quote bar (`has_quote_bar == true`) on top line(s), with >= 1 non-quote line below.
            // Pattern B: line[0] is a quote author header (`Oliver-ke` or `You`), line[1] is the quoted snippet
            //            (matching an earlier bubble or followed by >= 1 reply line), and line[2..] is the actual reply (`Sorry about that`).
            let mut quoted_reply: Option<(String, String)> = None;
            let mut body_start_idx = 0usize;

            if lines.len() >= 2 && lines[0].1 {
                let mut q_end = 1usize;
                while q_end + 1 < lines.len() && lines[q_end].1 {
                    q_end += 1;
                }
                let q_author = if q_end >= 2 && is_quote_author_header(&lines[0].0) {
                    lines[0].0.clone()
                } else {
                    "quoted message".to_string()
                };
                let q_text_start = usize::from(q_end >= 2 && is_quote_author_header(&lines[0].0));
                let q_text = lines[q_text_start..q_end]
                    .iter()
                    .map(|(s, _)| s.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");
                quoted_reply = Some((q_author, q_text));
                body_start_idx = q_end;
            } else if lines.len() >= 3 && is_quote_author_header(&lines[0].0) {
                let q_author = lines[0].0.clone();
                let q_snippet = lines[1].0.clone();
                quoted_reply = Some((q_author, q_snippet));
                body_start_idx = 2;
            } else if lines.len() >= 2 && seen_bodies.iter().any(|prev| prev.eq_ignore_ascii_case(lines[0].0.trim())) {
                quoted_reply = Some(("earlier message".to_string(), lines[0].0.clone()));
                body_start_idx = 1;
            }

            let actual_lines: Vec<&str> = lines[body_start_idx..]
                .iter()
                .map(|(s, _)| s.as_str())
                .collect();
            let call_event = actual_lines.iter().all(|l| is_call_event_text(l));
            let body = actual_lines.join(" ").trim().to_string();
            if body.is_empty() {
                continue;
            }
            if !call_event {
                seen_bodies.push(body.clone());
            }
            bubbles.push(ParsedThreadBubble {
                side,
                timestamp,
                body,
                quoted_reply,
                is_call_event: call_event,
            });
        }

        bubbles
    }

    #[cfg(test)]
    pub(super) fn format_two_pane_messaging_ocr(
        app_name: &str,
        regions: &[OcrRegion],
        terms: &[String],
    ) -> (String, bool) {
        format_two_pane_messaging_ocr_with_intent(
            app_name,
            regions,
            terms,
            MessageDirectionIntent::IncomingFromContact,
        )
    }

    pub(super) fn format_two_pane_messaging_ocr_with_intent(
        app_name: &str,
        regions: &[OcrRegion],
        terms: &[String],
        intent: MessageDirectionIntent,
    ) -> (String, bool) {
        // Exclude WhatsApp's far-left navigation icon rail (left_x < 0.105: Chats, Calls, Updates, Archived).
        let min_sidebar_x = if app_name.to_ascii_lowercase().contains("whatsapp") {
            0.105
        } else {
            0.04
        };
        let mut sidebar: Vec<&OcrRegion> = regions
            .iter()
            .filter(|r| r.left_x >= min_sidebar_x && r.left_x < 0.365 && r.top_y >= 0.065)
            .collect();

        // Detect narrow single-pane mode (when sidebar is collapsed/absent) vs standard two-pane desktop layout.
        let has_two_pane_sidebar = sidebar
            .iter()
            .filter(|r| r.top_y >= 0.12 && r.left_x < 0.28)
            .count()
            >= 2
            && regions.iter().any(|r| r.left_x >= 0.365);

        let pane_left = if has_two_pane_sidebar { 0.365 } else { 0.03 };
        if !has_two_pane_sidebar {
            sidebar.clear();
        }
        let mut main_pane: Vec<&OcrRegion> = regions
            .iter()
            .filter(|r| r.left_x >= pane_left)
            .collect();

        sidebar.sort_by(|a, b| {
            let ry_a = (a.top_y * 90.0).round() as i32;
            let ry_b = (b.top_y * 90.0).round() as i32;
            ry_a.cmp(&ry_b)
                .then_with(|| a.left_x.partial_cmp(&b.left_x).unwrap_or(std::cmp::Ordering::Equal))
        });
        main_pane.sort_by(|a, b| {
            let ry_a = (a.top_y * 85.0).round() as i32;
            let ry_b = (b.top_y * 85.0).round() as i32;
            ry_a.cmp(&ry_b)
                .then_with(|| a.left_x.partial_cmp(&b.left_x).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Group left-pane OCR spans into discrete contact cards (row pitch ~0.074).
        let mut cards: Vec<Vec<&OcrRegion>> = Vec::new();
        for reg in &sidebar {
            let start_new = match cards.last() {
                None => true,
                Some(cur) => {
                    let first_y = cur.first().map_or(reg.top_y, |c| c.top_y);
                    let last_y = cur.last().map_or(reg.top_y, |c| c.top_y);
                    (reg.top_y - first_y > 0.054) || (reg.top_y - last_y > 0.036)
                }
            };
            if start_new {
                cards.push(vec![*reg]);
            } else if let Some(cur) = cards.last_mut() {
                cur.push(*reg);
            }
        }

        let mut rows: Vec<SidebarChatRow> = Vec::with_capacity(cards.len());
        for card in cards {
            let Some(first) = card.first() else {
                continue;
            };
            let first_y = first.top_y;
            let mut contact_left_x = first.left_x;
            let mut contact_parts = Vec::new();
            let mut ts_parts = Vec::new();
            let mut preview_parts = Vec::new();
            let mut preview_first_span: Option<&&OcrRegion> = None;
            for span in &card {
                let dy = span.top_y - first_y;
                if dy <= 0.015 && span.left_x >= 0.265 {
                    ts_parts.push(span.text.as_str());
                } else if dy <= 0.014 && span.left_x < 0.265 {
                    if contact_parts.is_empty() {
                        contact_left_x = span.left_x;
                    }
                    contact_parts.push(span.text.as_str());
                } else {
                    if preview_first_span.is_none() {
                        preview_first_span = Some(span);
                    }
                    preview_parts.push(span.text.as_str());
                }
            }
            let contact = contact_parts.join(" ").trim().to_string();
            let timestamp = ts_parts.join(" ").trim().to_string();
            let raw_preview = preview_parts.join(" ").trim().to_string();
            let (clean_preview, direction) =
                classify_sidebar_preview(&raw_preview, contact_left_x, preview_first_span);
            if !contact.is_empty() || !clean_preview.is_empty() {
                rows.push(SidebarChatRow {
                    contact,
                    timestamp,
                    clean_preview,
                    direction,
                });
            }
        }

        let header_spans: Vec<&str> = main_pane
            .iter()
            .filter(|r| r.top_y < 0.105)
            .map(|r| r.text.as_str())
            .collect();
        let open_thread_header = header_spans.join(" · ");
        let open_header_low = open_thread_header.to_ascii_lowercase();

        let body_spans: Vec<&OcrRegion> = main_pane
            .iter()
            .copied()
            .filter(|r| r.top_y >= 0.105 && r.top_y <= 0.925)
            .collect();

        let mut out = Vec::new();
        out.push(format!("Live {app_name} Window on This Mac:"));

        if !terms.is_empty() {
            let mut matched_rows: Vec<&SidebarChatRow> = Vec::new();
            for row in &rows {
                if terms
                    .iter()
                    .any(|t| contact_matches_query_term(&row.contact, t))
                {
                    matched_rows.push(row);
                }
            }
            let right_header_matches = !open_header_low.is_empty()
                && terms
                    .iter()
                    .any(|t| contact_matches_query_term(&open_thread_header, t));

            if matched_rows.is_empty() && !right_header_matches {
                let visible_contacts: Vec<&str> = rows
                    .iter()
                    .map(|r| r.contact.as_str())
                    .filter(|c| !c.is_empty() && !c.eq_ignore_ascii_case("search"))
                    .take(12)
                    .collect();
                out.push(format!(
                    "  Verified Viewport Check: `{}` is NOT among the currently visible chat rows or open thread header in {app_name} (visible contacts on screen: {}). Do NOT guess 'Hello' or attribute another contact's message to `{}`.",
                    terms.join(", "),
                    if visible_contacts.is_empty() {
                        "(none)".to_string()
                    } else {
                        visible_contacts.join(", ")
                    },
                    terms.join(", ")
                ));
                return (out.join("\n"), false);
            }

            let contact_display = matched_rows
                .first()
                .map(|r| r.contact.as_str())
                .filter(|c| !c.is_empty())
                .unwrap_or_else(|| {
                    open_thread_header
                        .split(" · ")
                        .next()
                        .unwrap_or(terms[0].as_str())
                });

            if right_header_matches {
                let bubbles =
                    parse_open_thread_bubbles(&body_spans, pane_left, contact_display, terms);
                let latest_in = bubbles
                    .iter()
                    .rev()
                    .find(|b| !b.is_call_event && b.side == BubbleSide::Incoming);
                let latest_out = bubbles
                    .iter()
                    .rev()
                    .find(|b| !b.is_call_event && b.side == BubbleSide::Outgoing);
                let latest_any = bubbles.iter().rev().find(|b| !b.is_call_event);

                match intent {
                    MessageDirectionIntent::IncomingFromContact => {
                        if let Some(inc) = latest_in {
                            let ts_note = if inc.timestamp.is_empty() {
                                String::new()
                            } else {
                                format!(" at {}", inc.timestamp)
                            };
                            let mut lead = format!(
                                "  VERIFIED DIRECT ANSWER (Latest INCOMING message FROM `{contact_display}` TO You{ts_note}): \"{}\"",
                                inc.body
                            );
                            if latest_any.is_some_and(|last| last.side == BubbleSide::Outgoing) {
                                if let Some(out_b) = latest_out {
                                    let out_ts = if out_b.timestamp.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" at {}", out_b.timestamp)
                                    };
                                    lead.push_str(&format!(
                                        " (Context note: \"{}\" was YOUR outgoing reply to `{contact_display}`{out_ts}, NOT a message from `{contact_display}` to you.)",
                                        out_b.body
                                    ));
                                }
                            }
                            out.push(lead);
                        } else if let Some(out_b) = latest_out {
                            out.push(format!(
                                "  IMPORTANT DIRECTIONALITY CHECK: No incoming text message from `{contact_display}` is visible in the open viewport; the visible message \"{}\" was sent by YOU to `{contact_display}`, not by `{contact_display}` to you.",
                                out_b.body
                            ));
                        }
                    }
                    MessageDirectionIntent::OutgoingFromUser => {
                        if let Some(out_b) = latest_out {
                            let ts_note = if out_b.timestamp.is_empty() {
                                String::new()
                            } else {
                                format!(" at {}", out_b.timestamp)
                            };
                            out.push(format!(
                                "  VERIFIED DIRECT ANSWER (Latest OUTGOING message FROM You TO `{contact_display}`{ts_note}): \"{}\"",
                                out_b.body
                            ));
                        } else if let Some(inc) = latest_in {
                            out.push(format!(
                                "  IMPORTANT DIRECTIONALITY CHECK: No outgoing message from you to `{contact_display}` is visible in the open viewport (latest visible message is FROM `{contact_display}` TO you: \"{}\").",
                                inc.body
                            ));
                        }
                    }
                    MessageDirectionIntent::ThreadOverview => {}
                }

                out.push(format!(
                    "  Currently Open Thread Header in {app_name}: {open_thread_header}"
                ));
                if !bubbles.is_empty() {
                    out.push(format!(
                        "  Visible Messages in Open {app_name} Thread for `{contact_display}` (chronological, bottom = latest):"
                    ));
                    let start_idx = bubbles.len().saturating_sub(18);
                    for b in &bubbles[start_idx..] {
                        let ts_tag = if b.timestamp.is_empty() {
                            String::new()
                        } else {
                            format!(" ({})", b.timestamp)
                        };
                        if b.is_call_event {
                            out.push(format!("    [Call Event{ts_tag}] {}", b.body));
                        } else {
                            let dir_label = match b.side {
                                BubbleSide::Incoming => {
                                    format!("Incoming: {contact_display} -> You")
                                }
                                BubbleSide::Outgoing => {
                                    format!("Outgoing: You -> {contact_display}")
                                }
                            };
                            let quote_note = if let Some((qa, qt)) = &b.quoted_reply {
                                format!(
                                    " (replying to {qa}: \"{}\")",
                                    truncate_chars(qt, 96)
                                )
                            } else {
                                String::new()
                            };
                            out.push(format!(
                                "    [{dir_label}{ts_tag}] \"{}\"{quote_note}",
                                b.body
                            ));
                        }
                    }
                }
            } else {
                // Only left-sidebar contact row(s) are visible (another chat is open in the right pane).
                for mr in &matched_rows {
                    let ts_label = if mr.timestamp.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", mr.timestamp)
                    };
                    let preview_disp = if mr.clean_preview.is_empty() {
                        "[no text preview visible]"
                    } else {
                        mr.clean_preview.as_str()
                    };
                    let dir_tag = match mr.direction {
                        SidebarPreviewDirection::OutgoingFromYou => {
                            "[Outgoing: You -> Contact]"
                        }
                        SidebarPreviewDirection::CallOrSystemEvent => "[Call Event]",
                        SidebarPreviewDirection::IncomingOrUnknown => {
                            "[Incoming: Contact -> You]"
                        }
                    };
                    out.push(format!(
                        "  Matched Contact Row in {app_name} Chat List (`{}`{}) {dir_tag}: {}",
                        mr.contact, ts_label, preview_disp
                    ));

                    match (intent, mr.direction) {
                        (
                            MessageDirectionIntent::IncomingFromContact,
                            SidebarPreviewDirection::OutgoingFromYou,
                        ) => {
                            out.push(format!(
                                "  IMPORTANT DIRECTIONALITY CHECK: The visible sidebar preview for `{}`{} shows YOUR OWN OUTGOING message TO `{}` (\"{}\"), NOT a message from `{}` to you. Because a different chat is currently open in the right pane, `{}`'s earlier incoming message to you is hidden inside `{}`'s thread. Tell the user clearly that \"{}\" was their own message to `{}`, and ask them to click `{}`'s chat in {app_name} (or enable Full Disk Access for SQLite history) to read `{}`'s last incoming message to them.",
                                mr.contact,
                                ts_label,
                                mr.contact,
                                preview_disp,
                                mr.contact,
                                mr.contact,
                                mr.contact,
                                preview_disp,
                                mr.contact,
                                mr.contact,
                                mr.contact
                            ));
                        }
                        (
                            MessageDirectionIntent::IncomingFromContact,
                            SidebarPreviewDirection::IncomingOrUnknown,
                        ) if !mr.clean_preview.is_empty() => {
                            out.push(format!(
                                "  VERIFIED DIRECT ANSWER (Latest visible INCOMING message FROM `{}` TO You{}): \"{}\"",
                                mr.contact, ts_label, mr.clean_preview
                            ));
                        }
                        (
                            MessageDirectionIntent::OutgoingFromUser,
                            SidebarPreviewDirection::OutgoingFromYou,
                        ) if !mr.clean_preview.is_empty() => {
                            out.push(format!(
                                "  VERIFIED DIRECT ANSWER (Latest visible OUTGOING message FROM You TO `{}`{}): \"{}\"",
                                mr.contact, ts_label, mr.clean_preview
                            ));
                        }
                        (
                            MessageDirectionIntent::OutgoingFromUser,
                            SidebarPreviewDirection::IncomingOrUnknown,
                        ) if !mr.clean_preview.is_empty() => {
                            out.push(format!(
                                "  IMPORTANT DIRECTIONALITY CHECK: The visible sidebar preview for `{}`{} shows an INCOMING message FROM `{}` TO you (\"{}\"), not your outgoing message. Click `{}`'s chat in {app_name} (or enable Full Disk Access) to inspect your earlier outgoing messages.",
                                mr.contact, ts_label, mr.contact, mr.clean_preview, mr.contact
                            ));
                        }
                        _ => {}
                    }
                }
            }

            return (out.join("\n"), true);
        }

        // Unfiltered summary when no specific contact/term was queried.
        if !rows.is_empty() {
            out.push(format!("  {app_name} Recent Chats List (top to bottom):"));
            for row in rows.iter().take(16) {
                let ts_part = if row.timestamp.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", row.timestamp)
                };
                let dir_prefix = match row.direction {
                    SidebarPreviewDirection::OutgoingFromYou => "[You ->] ",
                    SidebarPreviewDirection::CallOrSystemEvent => "[Call] ",
                    SidebarPreviewDirection::IncomingOrUnknown => "",
                };
                out.push(format!(
                    "    - {}{}: {}{}",
                    row.contact, ts_part, dir_prefix, row.clean_preview
                ));
            }
        }

        if !main_pane.is_empty() {
            if !open_thread_header.is_empty() {
                out.push(format!(
                    "  Currently Open Thread Header in {app_name}: {open_thread_header}"
                ));
            }
            let header_contact = open_thread_header
                .split(" · ")
                .next()
                .unwrap_or("Contact");
            let bubbles =
                parse_open_thread_bubbles(&body_spans, pane_left, header_contact, terms);
            if !bubbles.is_empty() {
                out.push(format!(
                    "  Visible Messages in Open {app_name} Thread (chronological, bottom = latest):"
                ));
                let start_idx = bubbles.len().saturating_sub(18);
                for b in &bubbles[start_idx..] {
                    let ts_tag = if b.timestamp.is_empty() {
                        String::new()
                    } else {
                        format!(" ({})", b.timestamp)
                    };
                    if b.is_call_event {
                        out.push(format!("    [Call Event{ts_tag}] {}", b.body));
                    } else {
                        let side = match b.side {
                            BubbleSide::Incoming => "Incoming from Contact",
                            BubbleSide::Outgoing => "Outgoing from You",
                        };
                        out.push(format!("    [{side}{ts_tag}] \"{}\"", b.body));
                    }
                }
            }
        }

        let any_content = !rows.is_empty() || !main_pane.is_empty();
        (out.join("\n"), any_content)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_intent_and_entity_extraction() {
        assert!(is_conversation_or_memory_query(
            "what was dera last message on whatsapp about"
        ));
        assert!(is_conversation_or_memory_query(
            "remind me of important conversations today"
        ));
        assert!(is_conversation_or_memory_query(
            "what did we talk about earlier regarding quantization"
        ));
        assert!(!is_conversation_or_memory_query(
            "open whatsapp"
        ));
        assert!(!is_conversation_or_memory_query(
            "i want to make a video content on kronos, explain the maths behind every step"
        ));

        let terms =
            extract_conversation_query_terms("what was Dera last message on whatsapp about");
        assert_eq!(terms, vec!["dera".to_string()]);

        let terms2 =
            extract_conversation_query_terms("what did Dera's message on WhatsApp say about Kafka");
        assert_eq!(terms2[0], "dera");
        assert!(terms2.contains(&"kafka".to_string()));

        assert_eq!(
            parse_message_direction_intent("what was Oliver-ke last message to me on WhatsApp"),
            MessageDirectionIntent::IncomingFromContact
        );
        assert_eq!(
            parse_message_direction_intent("what was Dera's last message on WhatsApp"),
            MessageDirectionIntent::IncomingFromContact
        );
        assert_eq!(
            parse_message_direction_intent("what did Sandra send me on WhatsApp"),
            MessageDirectionIntent::IncomingFromContact
        );
        assert_eq!(
            parse_message_direction_intent("what was my last message to Oliver-ke on WhatsApp"),
            MessageDirectionIntent::OutgoingFromUser
        );
        assert_eq!(
            parse_message_direction_intent("what did I reply to Oliver-ke"),
            MessageDirectionIntent::OutgoingFromUser
        );
        assert_eq!(
            parse_message_direction_intent("show recent messages with Oliver-ke on WhatsApp"),
            MessageDirectionIntent::ThreadOverview
        );
    }

    #[test]
    fn test_persistent_conversation_memory_roundtrip_and_search() {
        let dir = std::env::temp_dir().join(format!("kronos_conv_mem_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("sessions")).unwrap();
        let mem_jsonl = dir.join("conversation_memory.jsonl");

        record_conversation_exchange_at(
            &mem_jsonl,
            "[Device Context · retrieved in 0.4 ms]\nfoo\n\nWe decided to ship the Kafka consumer migration on Friday",
            "Noted: shipping the Kafka consumer migration this Friday with zero-downtime consumer groups.",
            "chat",
        );
        record_conversation_exchange_at(
            &mem_jsonl,
            "Explain 2 + 2",
            "4",
            "chat",
        );

        let terms = extract_conversation_query_terms("remind me what we discussed about Kafka");
        let block = query_kronos_saved_conversations(&mem_jsonl, &dir.join("sessions"), &terms, 5)
            .expect("expected saved conversation match");
        assert!(block.contains("Kafka consumer migration"));
        assert!(!block.contains("[Device Context"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_two_pane_whatsapp_ocr_layout_pairing_and_directionality() {
        use crate::vision::OcrRegion;
        let regions = vec![
            OcrRegion {
                zone: "far-left-rail".into(),
                text: "Chats".into(),
                top_y: 0.14,
                left_x: 0.03,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-left".into(),
                text: "Alex Team".into(),
                top_y: 0.18,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-left".into(),
                text: "6:10 PM".into(),
                top_y: 0.18,
                left_x: 0.295,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-left".into(),
                text: "/ Pushed the branch".into(),
                top_y: 0.205,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-left".into(),
                text: "Dera".into(),
                top_y: 0.30,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-left".into(),
                text: "7:41 PM".into(),
                top_y: 0.30,
                left_x: 0.295,
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-left".into(),
                text: "Are we meeting at 9am tomorrow for the contract review".into(),
                top_y: 0.324,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-right".into(),
                text: "Engineering Group".into(),
                top_y: 0.06,
                left_x: 0.45,
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-right".into(),
                text: "I just got home.".into(),
                top_y: 0.48,
                left_x: 0.42,
                ..Default::default()
            },
        ];
        let (formatted, matched) =
            macos_sources::format_two_pane_messaging_ocr("WhatsApp", &regions, &["dera".into()]);
        assert!(matched);
        assert!(formatted.contains("Matched Contact Row in WhatsApp Chat List (`Dera` (7:41 PM))"));
        assert!(formatted.contains("Are we meeting at 9am tomorrow for the contract review"));
        // Strict contact attribution: unrelated right-pane thread ("Engineering Group") must NOT leak into Dera's lookup.
        assert!(!formatted.contains("I just got home."));

        // Sidebar outgoing checkmark (`/ Pushed the branch`) must be flagged as YOUR outgoing message, never incoming from Alex Team.
        let (alex_in_fmt, alex_matched) = macos_sources::format_two_pane_messaging_ocr_with_intent(
            "WhatsApp",
            &regions,
            &["alex".into()],
            MessageDirectionIntent::IncomingFromContact,
        );
        assert!(alex_matched);
        assert!(alex_in_fmt.contains("[Outgoing: You -> Contact]"));
        assert!(alex_in_fmt.contains("IMPORTANT DIRECTIONALITY CHECK"));
        assert!(alex_in_fmt.contains("YOUR OWN OUTGOING message"));

        // Off-screen contact query ("raldweb") must report matched == false and never leak another contact's message.
        let (unmatched_fmt, matched_rald) =
            macos_sources::format_two_pane_messaging_ocr("WhatsApp", &regions, &["raldweb".into()]);
        assert!(!matched_rald);
        assert!(unmatched_fmt.contains("is NOT among the currently visible chat rows"));
        assert!(!unmatched_fmt.contains("I just got home."));
        assert!(!unmatched_fmt.contains("Are we meeting at 9am"));

        // Hyphenated contact names ("Oliver-ke") must match hyphen, space, and en-dash OCR variants while preventing substring false positives.
        assert!(macos_sources::contact_matches_query_term("Oliver-ke", "oliver-ke"));
        assert!(macos_sources::contact_matches_query_term("Oliver ke", "oliver-ke"));
        assert!(macos_sources::contact_matches_query_term("Oliver \u{2013} ke", "oliver-ke"));
        assert!(!macos_sources::contact_matches_query_term("Cassandra", "sandra"));
        assert!(macos_sources::contact_matches_query_term("Sandra Okafor", "sandra"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn test_oliver_ke_open_thread_incoming_vs_outgoing_quoted_reply() {
        use crate::vision::OcrRegion;
        // Exact reproduction of Oliver-ke's thread:
        // - 03:55 Missed voice call / Click to call back (left gray card, RGB 29,30,30)
        // - 03:56 Incoming from Oliver-ke: "Please endeavour to return call, at least when you're available" (left gray bubble, RGB 36,38,38, timestamp at right end left_x=0.65)
        // - 06:45 Outgoing green bubble from You (RGB 21,77,56) with pink-bar reply quote ("Oliver-ke" / "Please endeavour...") + actual reply "Sorry about that"
        let regions = vec![
            OcrRegion {
                zone: "top-left".into(),
                text: "Oliver-ke".into(),
                top_y: 0.18,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-left".into(),
                text: "06:45".into(),
                top_y: 0.18,
                left_x: 0.295,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-left".into(),
                text: "// Sorry about that".into(),
                top_y: 0.205,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-left".into(),
                text: "Dera".into(),
                top_y: 0.29,
                left_x: 0.174,
                ..Default::default()
            },
            OcrRegion {
                zone: "top-right".into(),
                text: "Oliver-ke".into(),
                top_y: 0.06,
                left_x: 0.43,
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-right".into(),
                text: "Missed voice call".into(),
                top_y: 0.30,
                left_x: 0.41,
                bg_rgb: [29, 30, 30],
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-right".into(),
                text: "Click to call back".into(),
                top_y: 0.33,
                left_x: 0.41,
                bg_rgb: [29, 30, 30],
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-right".into(),
                text: "03:55".into(),
                top_y: 0.35,
                left_x: 0.51,
                bg_rgb: [29, 30, 30],
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-right".into(),
                text: "Please endeavour to return call, at least when you're available".into(),
                top_y: 0.43,
                left_x: 0.39,
                width_n: 0.28,
                bg_rgb: [36, 38, 38],
                ..Default::default()
            },
            OcrRegion {
                zone: "mid-right".into(),
                text: "03:56".into(),
                top_y: 0.44,
                left_x: 0.66,
                width_n: 0.04,
                bg_rgb: [36, 38, 38],
                ..Default::default()
            },
            // Wide green outgoing bubble whose left edge starts at 0.505 (< old 0.54 threshold)
            OcrRegion {
                zone: "bottom-right".into(),
                text: "Oliver-ke".into(),
                top_y: 0.56,
                left_x: 0.515,
                width_n: 0.08,
                bg_rgb: [17, 62, 45],
                has_left_quote_bar: true,
                ..Default::default()
            },
            OcrRegion {
                zone: "bottom-right".into(),
                text: "Please endeavour to return call, at least when you're available".into(),
                top_y: 0.60,
                left_x: 0.515,
                width_n: 0.41,
                bg_rgb: [17, 62, 45],
                has_left_quote_bar: true,
                ..Default::default()
            },
            OcrRegion {
                zone: "bottom-right".into(),
                text: "Sorry about that".into(),
                top_y: 0.66,
                left_x: 0.512,
                width_n: 0.14,
                bg_rgb: [21, 77, 56],
                has_left_quote_bar: false,
                ..Default::default()
            },
            OcrRegion {
                zone: "bottom-right".into(),
                text: "06:45 //".into(),
                top_y: 0.67,
                left_x: 0.91,
                width_n: 0.05,
                bg_rgb: [21, 77, 56],
                ..Default::default()
            },
        ];

        // 1. "what was Oliver-ke last message to me on WhatsApp" -> IncomingFromContact
        let (in_fmt, matched_in) = macos_sources::format_two_pane_messaging_ocr_with_intent(
            "WhatsApp",
            &regions,
            &["oliver-ke".into()],
            MessageDirectionIntent::IncomingFromContact,
        );
        assert!(matched_in);
        assert!(
            in_fmt.contains(
                "VERIFIED DIRECT ANSWER (Latest INCOMING message FROM `Oliver-ke` TO You at 03:56): \"Please endeavour to return call, at least when you're available\""
            ),
            "unexpected incoming format:\n{in_fmt}"
        );
        assert!(
            in_fmt.contains("\"Sorry about that\" was YOUR outgoing reply"),
            "expected outgoing clarification:\n{in_fmt}"
        );
        assert!(in_fmt.contains("[Call Event (03:55)] Missed voice call Click to call back"));
        assert!(in_fmt.contains("[Outgoing: You -> Oliver-ke (06:45)] \"Sorry about that\" (replying to Oliver-ke:"));

        // 2. "what was my last message to Oliver-ke on WhatsApp" -> OutgoingFromUser
        let (out_fmt, matched_out) = macos_sources::format_two_pane_messaging_ocr_with_intent(
            "WhatsApp",
            &regions,
            &["oliver-ke".into()],
            MessageDirectionIntent::OutgoingFromUser,
        );
        assert!(matched_out);
        assert!(
            out_fmt.contains(
                "VERIFIED DIRECT ANSWER (Latest OUTGOING message FROM You TO `Oliver-ke` at 06:45): \"Sorry about that\""
            ),
            "unexpected outgoing format:\n{out_fmt}"
        );
    }
}
