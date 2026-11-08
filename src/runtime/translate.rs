//! Translation on top of [`ChatModel`]: prompt building, decoding settings and output cleanup
//! shared by `kronos_server`'s `/translate` and the mobile FFI.

use crate::config::limits::{
    TRANSLATE_CONTEXT_MAX_BYTES, TRANSLATE_CONTEXT_MAX_ITEMS, TRANSLATE_TOKENS_BASE, TRANSLATE_TOKENS_MAX,
    TRANSLATE_TOKENS_MIN,
};
use crate::config::prompts::{translate_system, translate_user_hunyuan};
use crate::runtime::chat::{ChatModel, Completion, GenParams, Message, PromptStyle};

/// One translation job.
pub struct TranslateRequest<'a> {
    pub text: &'a str,
    /// Target language, English name.
    pub to: &'a str,
    /// Source language, English name or `auto`.
    pub from: &'a str,
    pub tone: &'a str,
    /// Earlier messages, oldest first.
    pub context: &'a [String],
    /// `0.0` is greedy, which keeps live retranslation stable.
    pub temperature: f32,
}

/// Newest `TRANSLATE_CONTEXT_MAX_ITEMS` non-empty items within the byte budget, oldest first.
pub fn trim_context<I: IntoIterator<Item = String>>(items: I) -> Vec<String> {
    let items: Vec<String> =
        items.into_iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let mut out = Vec::new();
    let mut bytes = 0;
    for item in items.into_iter().rev().take(TRANSLATE_CONTEXT_MAX_ITEMS) {
        bytes += item.len();
        if bytes > TRANSLATE_CONTEXT_MAX_BYTES {
            break;
        }
        out.push(item);
    }
    out.reverse();
    out
}

/// Messages, decoding parameters and assistant prefix for `req` on `model`.
pub fn prompt(model: &ChatModel, req: &TranslateRequest) -> (Vec<Message>, GenParams, &'static str) {
    let max_tokens =
        (req.text.len() / 2 + TRANSLATE_TOKENS_BASE).clamp(TRANSLATE_TOKENS_MIN, TRANSLATE_TOKENS_MAX);
    let temp = req.temperature.max(0.0);
    let (messages, params) = match model.prompt_style() {
        PromptStyle::HunYuan => (
            vec![Message::new("user", &translate_user_hunyuan(req.to, req.from, req.tone, req.context, req.text))],
            GenParams {
                temperature: temp,
                top_k: 20,
                top_p: 0.6,
                repeat_penalty: if temp <= 0.0 { 1.0 } else { 1.05 },
                max_tokens,
                seed: 0,
            },
        ),
        PromptStyle::ChatMl => (
            vec![
                Message::new("system", &translate_system(req.to, req.from, req.tone, req.context)),
                Message::new("user", req.text),
            ],
            GenParams {
                temperature: temp,
                top_k: 40,
                top_p: 0.9,
                repeat_penalty: if temp <= 0.0 { 1.0 } else { 1.02 },
                max_tokens,
                seed: 0,
            },
        ),
    };
    let prefix = if model.name.to_ascii_lowercase().contains("qwen3") { "<think>\n\n</think>\n\n" } else { "" };
    (messages, params, prefix)
}

/// Runs `req`, calling `on_text` with the cleaned translation so far whenever it changes.
/// Returns the completion; `clean_translation(&completion.text)` is the final result.
pub fn run(
    model: &mut ChatModel,
    req: &TranslateRequest,
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<Completion, String> {
    let (messages, params, prefix) = prompt(model, req);
    let mut acc = String::new();
    let mut last_sent = String::new();
    model.generate_with_tools_prefixed(&messages, &[], prefix, &params, |piece| {
        acc.push_str(piece);
        if !prefix.is_empty() && acc.len() <= prefix.len() {
            return true;
        }
        let cur = clean_translation(&acc);
        if !cur.is_empty() && cur != last_sent {
            last_sent.clone_from(&cur);
            return on_text(&cur);
        }
        true
    })
}

/// Strips think blocks, `Translation:`-style prefixes and wrapping quotes from model output.
pub fn clean_translation(raw: &str) -> String {
    let mut s = raw;
    if let Some(end) = s.find("</think>") {
        s = &s[end + "</think>".len()..];
    }
    let mut s = s.trim();
    loop {
        let before = s;
        for p in ["translation:", "translated text:", "translated:", "output:"] {
            if s.is_char_boundary(p.len()) && s[..p.len()].eq_ignore_ascii_case(p) {
                s = s[p.len()..].trim();
            }
        }
        for (open, close) in [('"', '"'), ('“', '”'), ('«', '»'), ('\'', '\''), ('`', '`')] {
            if s.starts_with(open) && s.ends_with(close) && s.chars().count() > 1 {
                s = s[open.len_utf8()..s.len() - close.len_utf8()].trim();
            }
        }
        if s == before {
            break;
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::{clean_translation, trim_context};

    #[test]
    fn cleans_prefixes_and_quotes() {
        assert_eq!(clean_translation("Translation: «Bonjour»"), "Bonjour");
        assert_eq!(clean_translation("<think>x</think>\n\"Salut\""), "Salut");
        assert_eq!(clean_translation("  Ça va ?  "), "Ça va ?");
    }

    #[test]
    fn context_keeps_newest_in_order() {
        let items = (0..12).map(|i| format!("m{i}")).collect::<Vec<_>>();
        let out = trim_context(items);
        assert_eq!(out.len(), 8);
        assert_eq!(out.first().map(String::as_str), Some("m4"));
        assert_eq!(out.last().map(String::as_str), Some("m11"));
        assert!(trim_context(vec![" ".to_string(), "".to_string()]).is_empty());
    }
}
