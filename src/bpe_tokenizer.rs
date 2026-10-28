#![forbid(unsafe_code)]
//! Pure-Rust Byte-Level BPE Tokenizer (`SmolLM2` / `GPT-2` `vocab.json` + `merges.txt`).
//!
//! Implements HuggingFace `Sequence([Digits(individual_digits=true), ByteLevel(use_regex=true)])`
//! pre-tokenization, byte-to-unicode mapping, lowest-rank BPE pair merging, and UTF-8 decoding
//! with **zero external crate dependencies**.

use crate::safetensors::JsonCursor;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// Byte-level BPE tokenizer loaded from `vocab.json` and `merges.txt`.
pub struct BpeTokenizer {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    merge_ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub eos_token_id: usize,
}

impl BpeTokenizer {
    /// Loads `vocab.json` and `merges.txt` from `model_dir`.
    pub fn from_model_dir(model_dir: &Path) -> io::Result<Self> {
        let vocab_text = fs::read_to_string(model_dir.join("vocab.json"))?;
        let merges_text = fs::read_to_string(model_dir.join("merges.txt"))?;
        Self::from_str_parts(&vocab_text, &merges_text)
            .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))
    }

    /// Parses `vocab.json` and `merges.txt` strings directly.
    pub fn from_str_parts(vocab_json: &str, merges_txt: &str) -> Result<Self, String> {
        let mut cur = JsonCursor::new(vocab_json);
        cur.expect_byte(b'{')?;
        let mut token_to_id = HashMap::new();
        let mut max_id = 0usize;

        if cur.peek() != Some(b'}') {
            loop {
                let tok = cur.parse_string()?;
                cur.expect_byte(b':')?;
                let id = cur.parse_u64()? as usize;
                max_id = max_id.max(id);
                token_to_id.insert(tok, id);
                if cur.peek() == Some(b',') {
                    cur.expect_byte(b',')?;
                } else {
                    break;
                }
            }
        }
        cur.expect_byte(b'}')?;

        let vocab_size = max_id + 1;
        let mut id_to_token = vec![String::new(); vocab_size];
        for (tok, &id) in &token_to_id {
            id_to_token[id] = tok.clone();
        }

        let mut merge_ranks = HashMap::new();
        for (rank, line) in merges_txt
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with("#version"))
            .enumerate()
        {
            if let Some((a, b)) = line.split_once(' ') {
                merge_ranks.insert((a.to_string(), b.to_string()), rank);
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_unicode_tables();
        let eos_token_id = token_to_id.get("<|endoftext|>").copied().unwrap_or(0);

        Ok(Self {
            token_to_id,
            id_to_token,
            merge_ranks,
            byte_to_char,
            char_to_byte,
            eos_token_id,
        })
    }

    /// Encodes arbitrary UTF-8 `text` into BPE token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        // Stage 1: `Digits(individual_digits=true)` splits ASCII digits into standalone 1-char spans.
        let mut span_start = 0usize;
        for (byte_idx, ch) in text.char_indices() {
            if ch.is_ascii_digit() {
                if span_start < byte_idx {
                    self.encode_non_digit_span(&text[span_start..byte_idx], &mut ids);
                }
                let digit_str = &text[byte_idx..byte_idx + 1];
                if let Some(&id) = self.token_to_id.get(digit_str) {
                    ids.push(id);
                }
                span_start = byte_idx + 1;
            }
        }
        if span_start < text.len() {
            self.encode_non_digit_span(&text[span_start..], &mut ids);
        }
        ids
    }

    fn encode_non_digit_span(&self, span: &str, out_ids: &mut Vec<usize>) {
        for piece in gpt2_regex_split(span) {
            self.encode_bpe_piece(piece.as_bytes(), out_ids);
        }
    }

    fn encode_bpe_piece(&self, raw_bytes: &[u8], out_ids: &mut Vec<usize>) {
        if raw_bytes.is_empty() {
            return;
        }
        let mut word: Vec<String> = raw_bytes
            .iter()
            .map(|&b| self.byte_to_char[b as usize].to_string())
            .collect();

        while word.len() >= 2 {
            let mut best_pair_idx: Option<usize> = None;
            let mut best_rank = usize::MAX;

            for i in 0..word.len() - 1 {
                if let Some(&rank) = self
                    .merge_ranks
                    .get(&(word[i].clone(), word[i + 1].clone()))
                {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair_idx = Some(i);
                    }
                }
            }

            let Some(first_idx) = best_pair_idx else {
                break;
            };
            let left = word[first_idx].clone();
            let right = word[first_idx + 1].clone();
            let merged = format!("{left}{right}");

            let mut next_word = Vec::with_capacity(word.len() - 1);
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
                    next_word.push(merged.clone());
                    i += 2;
                } else {
                    next_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = next_word;
        }

        for sym in word {
            if let Some(&id) = self.token_to_id.get(&sym) {
                out_ids.push(id);
            }
        }
    }

    /// Appends the raw UTF-8 bytes of a single token `id` to `raw_bytes` in `O(1)` time.
    #[inline]
    pub fn append_token_bytes(&self, id: usize, raw_bytes: &mut Vec<u8>) {
        let Some(tok) = self.id_to_token.get(id) else {
            return;
        };
        if tok.starts_with("<|") && tok.ends_with("|>") {
            raw_bytes.extend_from_slice(tok.as_bytes());
            return;
        }
        for ch in tok.chars() {
            if let Some(&b) = self.char_to_byte.get(&ch) {
                raw_bytes.push(b);
            } else {
                let mut buf = [0u8; 4];
                raw_bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            }
        }
    }

    /// Decodes a sequence of token IDs back into a UTF-8 `String`.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut raw_bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.append_token_bytes(id, &mut raw_bytes);
        }
        String::from_utf8_lossy(&raw_bytes).into_owned()
    }
}

fn build_gpt2_byte_unicode_tables() -> ([char; 256], HashMap<char, u8>) {
    let mut byte_to_char = ['\0'; 256];
    let mut char_to_byte = HashMap::with_capacity(256);
    let mut extra = 0u32;

    for b in 0u16..=255 {
        let direct = matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        let ch = if direct {
            char::from_u32(b as u32).unwrap()
        } else {
            let c = char::from_u32(256 + extra).unwrap();
            extra += 1;
            c
        };
        byte_to_char[b as usize] = ch;
        char_to_byte.insert(ch, b as u8);
    }
    (byte_to_char, char_to_byte)
}

/// Pure-Rust implementation of GPT-2's pre-tokenization regex:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
fn gpt2_regex_split(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0usize;

    let slice_chars = |start_c: usize, end_c: usize| -> &str {
        let sb = chars[start_c].0;
        let eb = if end_c < n { chars[end_c].0 } else { text.len() };
        &text[sb..eb]
    };

    while i < n {
        let ch = chars[i].1;

        // 1. Contractions: 's | 't | 're | 've | 'm | 'll | 'd
        if ch == '\'' && i + 1 < n {
            let c1 = chars[i + 1].1;
            if i + 2 < n {
                let c2 = chars[i + 2].1;
                if matches!((c1, c2), ('r' | 'v', 'e') | ('l', 'l')) {
                    out.push(slice_chars(i, i + 3));
                    i += 3;
                    continue;
                }
            }
            if matches!(c1, 's' | 't' | 'm' | 'd') {
                out.push(slice_chars(i, i + 2));
                i += 2;
                continue;
            }
        }

        // 2. Optional single space `' '` + letters / numbers / punctuation
        let has_lead_space = ch == ' ' && i + 1 < n && !chars[i + 1].1.is_whitespace();
        let next_idx = if has_lead_space { i + 1 } else { i };
        let head = chars[next_idx].1;

        if head.is_alphabetic() {
            let mut j = next_idx + 1;
            while j < n && chars[j].1.is_alphabetic() {
                j += 1;
            }
            out.push(slice_chars(i, j));
            i = j;
            continue;
        }

        if head.is_numeric() {
            let mut j = next_idx + 1;
            while j < n && chars[j].1.is_numeric() {
                j += 1;
            }
            out.push(slice_chars(i, j));
            i = j;
            continue;
        }

        if !head.is_whitespace() {
            let mut j = next_idx + 1;
            while j < n
                && !chars[j].1.is_whitespace()
                && !chars[j].1.is_alphabetic()
                && !chars[j].1.is_numeric()
            {
                j += 1;
            }
            out.push(slice_chars(i, j));
            i = j;
            continue;
        }

        // 3. Whitespace: `\s+(?!\S)|\s+`
        let mut j = i + 1;
        while j < n && chars[j].1.is_whitespace() {
            j += 1;
        }
        if j < n && (j - i) > 1 {
            // `\s+(?!\S)`: the last whitespace char starts the next token (`"\n\nx"` -> `"\n"`, `"\n"`, `"x"`).
            out.push(slice_chars(i, j - 1));
            i = j - 1;
        } else {
            out.push(slice_chars(i, j));
            i = j;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt2_regex_and_smollm2_roundtrip_if_present() {
        let parts = gpt2_regex_split("Hello,  world! don't stop\nnow");
        assert_eq!(
            parts,
            vec!["Hello", ",", " ", " world", "!", " don", "'t", " stop", "\n", "now"]
        );
        assert_eq!(gpt2_regex_split("a\n\nuser\n \tb  "), vec!["a", "\n", "\n", "user", "\n ", "\t", "b", "  "]);

        let model_dir = Path::new("/Users/melodyleonard/Documents/project/kronos-2t/models/SmolLM2-135M");
        if model_dir.join("vocab.json").exists() {
            let tk = BpeTokenizer::from_model_dir(model_dir).unwrap();
            let sample = "Alan Turing was born in 1912, and memory bandwidth limits LLMs!";
            let ids = tk.encode(sample);
            assert!(ids.len() > 8);
            assert_eq!(tk.decode(&ids), sample);
        }
    }
}
