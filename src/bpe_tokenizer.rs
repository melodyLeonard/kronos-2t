#![forbid(unsafe_code)]
//! Byte-level BPE tokenizer supporting GPT-2 / SmolLM2 and Qwen2 (`vocab.json`+`merges.txt` or `tokenizer.json`).

use crate::safetensors::JsonCursor;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

/// Pre-tokenization regex family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizerKind {
    /// `Sequence([Digits(individual_digits=true), ByteLevel(use_regex=true)])` (`SmolLM2` / `GPT-2`).
    SmolLm2DigitsGpt2,
    /// `Qwen2` / `Qwen2.5-Coder` `Split` regex + `ByteLevel(use_regex=false)`.
    Qwen2,
}

/// Byte-level BPE tokenizer.
pub struct BpeTokenizer {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    /// Literal tokens matched prior to BPE regex splitting (`<|im_start|>`, `<tool_call>`, etc.).
    added_tokens: Vec<(String, usize)>,
    /// Marks IDs whose `id_to_token` string is raw UTF-8 rather than GPT-2 byte-mapped characters.
    is_raw_special: Vec<bool>,
    added_first_bytes: [bool; 256],
    merge_ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pub pre_tokenizer: PreTokenizerKind,
    pub eos_token_id: usize,
    pub eos_token_ids: Vec<usize>,
}

impl BpeTokenizer {
    /// Loads tokenizer from `model_dir` (`vocab.json` + `merges.txt` and/or `tokenizer.json` + `tokenizer_config.json`).
    pub fn from_model_dir(model_dir: &Path) -> io::Result<Self> {
        let vocab_path = model_dir.join("vocab.json");
        let merges_path = model_dir.join("merges.txt");
        let tok_json_path = model_dir.join("tokenizer.json");
        let tok_cfg_path = model_dir.join("tokenizer_config.json");

        let mut tok = if vocab_path.exists() && merges_path.exists() {
            let vocab_text = fs::read_to_string(&vocab_path)?;
            let merges_text = fs::read_to_string(&merges_path)?;
            Self::from_str_parts(&vocab_text, &merges_text)
                .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?
        } else if tok_json_path.exists() {
            let tj = fs::read_to_string(&tok_json_path)?;
            Self::from_tokenizer_json_str(&tj)
                .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?
        } else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no vocab.json+merges.txt or tokenizer.json in {}", model_dir.display()),
            ));
        };

        if tok_json_path.exists() {
            if let Ok(tj) = fs::read_to_string(&tok_json_path) {
                tok.apply_tokenizer_json_metadata(&tj)
                    .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;
            }
        }
        if tok_cfg_path.exists() {
            if let Ok(tc) = fs::read_to_string(&tok_cfg_path) {
                tok.apply_tokenizer_config_metadata(&tc)
                    .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?;
            }
        }
        tok.finalize_special_tables();
        Ok(tok)
    }

    /// Parses `vocab.json` and `merges.txt` strings directly.
    pub fn from_str_parts(vocab_json: &str, merges_txt: &str) -> Result<Self, String> {
        let mut cur = JsonCursor::new(vocab_json);
        let (token_to_id, max_id) = parse_vocab_object(&mut cur)?;

        let vocab_size = max_id + 1;
        let mut id_to_token = vec![String::new(); vocab_size];
        let mut is_raw_special = vec![false; vocab_size];
        let mut added_tokens = Vec::new();
        for (tok, &id) in &token_to_id {
            id_to_token[id] = tok.clone();
            if tok.starts_with("<|") && tok.ends_with("|>") {
                is_raw_special[id] = true;
                added_tokens.push((tok.clone(), id));
            }
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
        let eos_opt = token_to_id.get("<|endoftext|>").copied();
        let mut out = Self {
            token_to_id,
            id_to_token,
            added_tokens,
            is_raw_special,
            added_first_bytes: [false; 256],
            merge_ranks,
            byte_to_char,
            char_to_byte,
            pre_tokenizer: PreTokenizerKind::SmolLm2DigitsGpt2,
            eos_token_id: eos_opt.unwrap_or(usize::MAX),
            eos_token_ids: eos_opt.into_iter().collect(),
        };
        out.finalize_special_tables();
        Ok(out)
    }

    /// Parses a standalone HuggingFace `tokenizer.json`.
    pub fn from_tokenizer_json_str(tok_json: &str) -> Result<Self, String> {
        let mut cur = JsonCursor::new(tok_json);
        cur.expect_byte(b'{')?;
        let mut token_to_id: Option<HashMap<String, usize>> = None;
        let mut max_id = 0usize;
        let mut merge_ranks = HashMap::new();
        let mut extra_added: Vec<(String, usize)> = Vec::new();
        let mut pre_kind = PreTokenizerKind::SmolLm2DigitsGpt2;

        if cur.peek() != Some(b'}') {
            loop {
                let key = cur.parse_string()?;
                cur.expect_byte(b':')?;
                match key.as_str() {
                    "added_tokens" => {
                        extra_added = parse_added_tokens_array(&mut cur)?;
                    }
                    "pre_tokenizer" => {
                        if parse_pre_tokenizer_is_qwen2(&mut cur)? {
                            pre_kind = PreTokenizerKind::Qwen2;
                        }
                    }
                    "model" => {
                        cur.expect_byte(b'{')?;
                        if cur.peek() != Some(b'}') {
                            loop {
                                let mkey = cur.parse_string()?;
                                cur.expect_byte(b':')?;
                                match mkey.as_str() {
                                    "vocab" => {
                                        let (v, m) = parse_vocab_object(&mut cur)?;
                                        max_id = max_id.max(m);
                                        token_to_id = Some(v);
                                    }
                                    "merges" => {
                                        merge_ranks = parse_merges_array(&mut cur)?;
                                    }
                                    _ => cur.skip_value()?,
                                }
                                if cur.peek() == Some(b',') {
                                    cur.expect_byte(b',')?;
                                } else {
                                    break;
                                }
                            }
                        }
                        cur.expect_byte(b'}')?;
                    }
                    _ => cur.skip_value()?,
                }
                if cur.peek() == Some(b',') {
                    cur.expect_byte(b',')?;
                } else {
                    break;
                }
            }
        }
        cur.expect_byte(b'}')?;

        let token_to_id = token_to_id.ok_or_else(|| "tokenizer.json missing model.vocab".to_string())?;
        for &(_, id) in &extra_added {
            max_id = max_id.max(id);
        }
        let vocab_size = max_id + 1;
        let mut id_to_token = vec![String::new(); vocab_size];
        let mut is_raw_special = vec![false; vocab_size];
        for (tok, &id) in &token_to_id {
            id_to_token[id] = tok.clone();
            if tok.starts_with("<|") && tok.ends_with("|>") {
                is_raw_special[id] = true;
            }
        }

        let (byte_to_char, char_to_byte) = build_gpt2_byte_unicode_tables();
        let eos_opt = token_to_id.get("<|endoftext|>").copied();
        let mut out = Self {
            token_to_id,
            id_to_token,
            added_tokens: Vec::new(),
            is_raw_special,
            added_first_bytes: [false; 256],
            merge_ranks,
            byte_to_char,
            char_to_byte,
            pre_tokenizer: pre_kind,
            eos_token_id: eos_opt.unwrap_or(usize::MAX),
            eos_token_ids: eos_opt.into_iter().collect(),
        };
        for (content, id) in extra_added {
            out.register_added_token(content, id);
        }
        out.finalize_special_tables();
        Ok(out)
    }

    fn apply_tokenizer_json_metadata(&mut self, tok_json: &str) -> Result<(), String> {
        let mut cur = JsonCursor::new(tok_json);
        cur.expect_byte(b'{')?;
        if cur.peek() != Some(b'}') {
            loop {
                let key = cur.parse_string()?;
                cur.expect_byte(b':')?;
                match key.as_str() {
                    "added_tokens" => {
                        for (content, id) in parse_added_tokens_array(&mut cur)? {
                            self.register_added_token(content, id);
                        }
                    }
                    "pre_tokenizer" => {
                        if parse_pre_tokenizer_is_qwen2(&mut cur)? {
                            self.pre_tokenizer = PreTokenizerKind::Qwen2;
                        }
                    }
                    _ => cur.skip_value()?,
                }
                if cur.peek() == Some(b',') {
                    cur.expect_byte(b',')?;
                } else {
                    break;
                }
            }
        }
        cur.expect_byte(b'}')?;
        Ok(())
    }

    fn apply_tokenizer_config_metadata(&mut self, cfg_json: &str) -> Result<(), String> {
        let mut cur = JsonCursor::new(cfg_json);
        cur.expect_byte(b'{')?;
        let mut eos_str: Option<String> = None;
        if cur.peek() != Some(b'}') {
            loop {
                let key = cur.parse_string()?;
                cur.expect_byte(b':')?;
                match key.as_str() {
                    "tokenizer_class" => {
                        let cls = cur.parse_string()?;
                        if cls.contains("Qwen2") || cls.contains("Qwen3") {
                            self.pre_tokenizer = PreTokenizerKind::Qwen2;
                        }
                    }
                    "eos_token" => {
                        if cur.peek() == Some(b'"') {
                            eos_str = Some(cur.parse_string()?);
                        } else {
                            cur.skip_value()?;
                        }
                    }
                    "added_tokens_decoder" => {
                        cur.expect_byte(b'{')?;
                        if cur.peek() != Some(b'}') {
                            loop {
                                let id_str = cur.parse_string()?;
                                cur.expect_byte(b':')?;
                                let id: usize = id_str.parse().unwrap_or(0);
                                let mut content: Option<String> = None;
                                cur.expect_byte(b'{')?;
                                if cur.peek() != Some(b'}') {
                                    loop {
                                        let f = cur.parse_string()?;
                                        cur.expect_byte(b':')?;
                                        if f == "content" {
                                            content = Some(cur.parse_string()?);
                                        } else {
                                            cur.skip_value()?;
                                        }
                                        if cur.peek() == Some(b',') {
                                            cur.expect_byte(b',')?;
                                        } else {
                                            break;
                                        }
                                    }
                                }
                                cur.expect_byte(b'}')?;
                                if let Some(c) = content {
                                    self.register_added_token(c, id);
                                }
                                if cur.peek() == Some(b',') {
                                    cur.expect_byte(b',')?;
                                } else {
                                    break;
                                }
                            }
                        }
                        cur.expect_byte(b'}')?;
                    }
                    _ => cur.skip_value()?,
                }
                if cur.peek() == Some(b',') {
                    cur.expect_byte(b',')?;
                } else {
                    break;
                }
            }
        }
        cur.expect_byte(b'}')?;
        if let Some(eos) = eos_str {
            if let Some(&id) = self.token_to_id.get(&eos) {
                self.eos_token_id = id;
                if !self.eos_token_ids.contains(&id) {
                    self.eos_token_ids.push(id);
                }
            }
        }
        Ok(())
    }

    fn register_added_token(&mut self, content: String, id: usize) {
        if content.is_empty() {
            return;
        }
        if id >= self.id_to_token.len() {
            self.id_to_token.resize(id + 1, String::new());
            self.is_raw_special.resize(id + 1, false);
        }
        self.id_to_token[id] = content.clone();
        self.is_raw_special[id] = true;
        self.token_to_id.insert(content.clone(), id);
        if !self.added_tokens.iter().any(|(_, existing_id)| *existing_id == id) {
            self.added_tokens.push((content, id));
        }
    }

    fn finalize_special_tables(&mut self) {
        // Longest-first tie-breaking at identical byte positions.
        self.added_tokens
            .sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.1.cmp(&b.1)));
        self.added_first_bytes = [false; 256];
        for (tok, _) in &self.added_tokens {
            if let Some(&b0) = tok.as_bytes().first() {
                self.added_first_bytes[b0 as usize] = true;
            }
        }
        for special_eos in ["<|im_end|>", "<|endoftext|>"] {
            if let Some(&id) = self.token_to_id.get(special_eos) {
                if self.eos_token_id == usize::MAX {
                    self.eos_token_id = id;
                }
                if !self.eos_token_ids.contains(&id) {
                    self.eos_token_ids.push(id);
                }
            }
        }
    }

    /// Drops literal added tokens rejected by `keep` so untrained zero-embedding entries fall back to BPE subwords.
    pub fn retain_added_tokens(&mut self, mut keep: impl FnMut(&str, usize) -> bool) {
        self.added_tokens.retain(|(s, id)| keep(s, *id));
        self.finalize_special_tables();
    }

    /// Encodes arbitrary UTF-8 `text` into BPE token IDs, matching `added_tokens` literally first.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        if self.added_tokens.is_empty() {
            self.encode_plain_segment(text, &mut ids);
            return ids;
        }

        let bytes = text.as_bytes();
        let mut seg_start = 0usize;
        let mut pos = 0usize;
        while pos < bytes.len() {
            if self.added_first_bytes[bytes[pos] as usize] {
                let tail = &text[pos..];
                if let Some((tok_str, tok_id)) = self
                    .added_tokens
                    .iter()
                    .find(|(pat, _)| tail.starts_with(pat.as_str()))
                {
                    if seg_start < pos {
                        self.encode_plain_segment(&text[seg_start..pos], &mut ids);
                    }
                    ids.push(*tok_id);
                    pos += tok_str.len();
                    seg_start = pos;
                    continue;
                }
            }
            pos += 1;
        }
        if seg_start < text.len() {
            self.encode_plain_segment(&text[seg_start..], &mut ids);
        }
        ids
    }

    fn encode_plain_segment(&self, text: &str, ids: &mut Vec<usize>) {
        match self.pre_tokenizer {
            PreTokenizerKind::SmolLm2DigitsGpt2 => {
                // Stage 1: `Digits(individual_digits=true)` splits ASCII digits into standalone 1-char spans.
                let mut span_start = 0usize;
                for (byte_idx, ch) in text.char_indices() {
                    if ch.is_ascii_digit() {
                        if span_start < byte_idx {
                            self.encode_non_digit_span(&text[span_start..byte_idx], ids);
                        }
                        let digit_str = &text[byte_idx..byte_idx + 1];
                        if let Some(&id) = self.token_to_id.get(digit_str) {
                            ids.push(id);
                        }
                        span_start = byte_idx + 1;
                    }
                }
                if span_start < text.len() {
                    self.encode_non_digit_span(&text[span_start..], ids);
                }
            }
            PreTokenizerKind::Qwen2 => {
                for piece in qwen2_regex_split(text) {
                    self.encode_bpe_piece(piece.as_bytes(), ids);
                }
            }
        }
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
        if self.is_raw_special.get(id).copied().unwrap_or(false)
            || (tok.starts_with("<|") && tok.ends_with("|>"))
        {
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

fn parse_vocab_object(cur: &mut JsonCursor<'_>) -> Result<(HashMap<String, usize>, usize), String> {
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
    Ok((token_to_id, max_id))
}

fn parse_merges_array(cur: &mut JsonCursor<'_>) -> Result<HashMap<(String, String), usize>, String> {
    cur.expect_byte(b'[')?;
    let mut ranks = HashMap::new();
    let mut rank = 0usize;
    if cur.peek() != Some(b']') {
        loop {
            if cur.peek() == Some(b'[') {
                cur.expect_byte(b'[')?;
                let a = cur.parse_string()?;
                cur.expect_byte(b',')?;
                let b = cur.parse_string()?;
                cur.expect_byte(b']')?;
                ranks.insert((a, b), rank);
            } else {
                let s = cur.parse_string()?;
                if let Some((a, b)) = s.split_once(' ') {
                    ranks.insert((a.to_string(), b.to_string()), rank);
                }
            }
            rank += 1;
            if cur.peek() == Some(b',') {
                cur.expect_byte(b',')?;
            } else {
                break;
            }
        }
    }
    cur.expect_byte(b']')?;
    Ok(ranks)
}

fn parse_added_tokens_array(cur: &mut JsonCursor<'_>) -> Result<Vec<(String, usize)>, String> {
    cur.expect_byte(b'[')?;
    let mut out = Vec::new();
    if cur.peek() != Some(b']') {
        loop {
            cur.expect_byte(b'{')?;
            let mut id: Option<usize> = None;
            let mut content: Option<String> = None;
            if cur.peek() != Some(b'}') {
                loop {
                    let f = cur.parse_string()?;
                    cur.expect_byte(b':')?;
                    match f.as_str() {
                        "id" => id = Some(cur.parse_u64()? as usize),
                        "content" => content = Some(cur.parse_string()?),
                        _ => cur.skip_value()?,
                    }
                    if cur.peek() == Some(b',') {
                        cur.expect_byte(b',')?;
                    } else {
                        break;
                    }
                }
            }
            cur.expect_byte(b'}')?;
            if let (Some(c), Some(i)) = (content, id) {
                out.push((c, i));
            }
            if cur.peek() == Some(b',') {
                cur.expect_byte(b',')?;
            } else {
                break;
            }
        }
    }
    cur.expect_byte(b']')?;
    Ok(out)
}

fn parse_pre_tokenizer_is_qwen2(cur: &mut JsonCursor<'_>) -> Result<bool, String> {
    if cur.peek() == Some(b'n') {
        cur.consume_literal(b"null")?;
        return Ok(false);
    }
    let mut is_qwen2 = false;
    fn scan_obj(cur: &mut JsonCursor<'_>, is_qwen2: &mut bool) -> Result<(), String> {
        match cur.peek() {
            Some(b'{') => {
                cur.expect_byte(b'{')?;
                if cur.peek() != Some(b'}') {
                    loop {
                        let k = cur.parse_string()?;
                        cur.expect_byte(b':')?;
                        if k == "Regex" && cur.peek() == Some(b'"') {
                            let pat = cur.parse_string()?;
                            if pat.contains("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+")
                                || pat.contains("[^\r\n\\p{L}\\p{N}]?\\p{L}+")
                            {
                                *is_qwen2 = true;
                            }
                        } else {
                            scan_obj(cur, is_qwen2)?;
                        }
                        if cur.peek() == Some(b',') {
                            cur.expect_byte(b',')?;
                        } else {
                            break;
                        }
                    }
                }
                cur.expect_byte(b'}')
            }
            Some(b'[') => {
                cur.expect_byte(b'[')?;
                if cur.peek() != Some(b']') {
                    loop {
                        scan_obj(cur, is_qwen2)?;
                        if cur.peek() == Some(b',') {
                            cur.expect_byte(b',')?;
                        } else {
                            break;
                        }
                    }
                }
                cur.expect_byte(b']')
            }
            _ => cur.skip_value(),
        }
    }
    scan_obj(cur, &mut is_qwen2)?;
    Ok(is_qwen2)
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

/// Pure-Rust implementation of Qwen2 / Qwen2.5-Coder's pre-tokenization regex:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
pub fn qwen2_regex_split(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut i = 0usize;

    let slice_chars = |start_c: usize, end_c: usize| -> &str {
        let sb = chars[start_c].0;
        let eb = if end_c < n { chars[end_c].0 } else { text.len() };
        &text[sb..eb]
    };
    let is_punct_sym = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();

    while i < n {
        let ch = chars[i].1;

        // 1. Case-insensitive contractions: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if ch == '\'' && i + 1 < n {
            let c1 = chars[i + 1].1.to_ascii_lowercase();
            if i + 2 < n {
                let c2 = chars[i + 2].1.to_ascii_lowercase();
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

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        if ch.is_alphabetic() {
            let mut j = i + 1;
            while j < n && chars[j].1.is_alphabetic() {
                j += 1;
            }
            out.push(slice_chars(i, j));
            i = j;
            continue;
        }
        if ch != '\r'
            && ch != '\n'
            && !ch.is_numeric()
            && i + 1 < n
            && chars[i + 1].1.is_alphabetic()
        {
            let mut j = i + 2;
            while j < n && chars[j].1.is_alphabetic() {
                j += 1;
            }
            out.push(slice_chars(i, j));
            i = j;
            continue;
        }

        // 3. \p{N} (single numeric character)
        if ch.is_numeric() {
            out.push(slice_chars(i, i + 1));
            i += 1;
            continue;
        }

        // 4. ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        if is_punct_sym(ch) || (ch == ' ' && i + 1 < n && is_punct_sym(chars[i + 1].1)) {
            let mut j = if ch == ' ' { i + 2 } else { i + 1 };
            while j < n && is_punct_sym(chars[j].1) {
                j += 1;
            }
            while j < n && matches!(chars[j].1, '\r' | '\n') {
                j += 1;
            }
            out.push(slice_chars(i, j));
            i = j;
            continue;
        }

        // 5-7. Whitespace: `\s*[\r\n]+|\s+(?!\S)|\s+`
        let mut j = i + 1;
        let mut last_nl: Option<usize> = matches!(ch, '\r' | '\n').then_some(i);
        while j < n && chars[j].1.is_whitespace() {
            if matches!(chars[j].1, '\r' | '\n') {
                last_nl = Some(j);
            }
            j += 1;
        }
        if let Some(nl_idx) = last_nl {
            out.push(slice_chars(i, nl_idx + 1));
            i = nl_idx + 1;
        } else if j < n && (j - i) > 1 {
            out.push(slice_chars(i, j - 1));
            i = j - 1;
        } else {
            out.push(slice_chars(i, j));
            i = j;
        }
    }

    out
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
    fn test_gpt2_and_qwen2_regex_and_special_tokens() {
        let parts = gpt2_regex_split("Hello,  world! don't stop\nnow");
        assert_eq!(
            parts,
            vec!["Hello", ",", " ", " world", "!", " don", "'t", " stop", "\n", "now"]
        );
        assert_eq!(
            gpt2_regex_split("a\n\nuser\n \tb  "),
            vec!["a", "\n", "\n", "user", "\n ", "\t", "b", "  "]
        );

        let qparts = qwen2_regex_split("def foo(x = 42):\n    return DON'T\n\n");
        assert_eq!(
            qparts,
            vec![
                "def", " foo", "(x", " =", " ", "4", "2", "):\n", "   ", " return", " DON", "'T",
                "\n\n"
            ]
        );

        let model_dir = Path::new("/Users/melodyleonard/Documents/project/kronos-2t/models/SmolLM2-135M");
        if model_dir.join("vocab.json").exists() {
            let tk = BpeTokenizer::from_model_dir(model_dir).unwrap();
            assert_eq!(tk.pre_tokenizer, PreTokenizerKind::SmolLm2DigitsGpt2);
            let sample = "Alan Turing was born in 1912, and memory bandwidth limits LLMs!";
            let ids = tk.encode(sample);
            assert!(ids.len() > 8);
            assert_eq!(tk.decode(&ids), sample);
        }

        // Qwen2.5 maps ID 0 to literal exclamation in vocab.json; ID 0 must not enter eos_token_ids.
        let mut qtk = BpeTokenizer::from_str_parts(
            r#"{"!": 0, "<": 1, "a": 2}"#,
            "",
        )
        .unwrap();
        qtk.apply_tokenizer_config_metadata(
            r#"{
                "tokenizer_class": "Qwen2Tokenizer",
                "eos_token": "<|im_end|>",
                "added_tokens_decoder": {
                    "151643": {"content": "<|endoftext|>"},
                    "151644": {"content": "<|im_start|>"},
                    "151645": {"content": "<|im_end|>"},
                    "151657": {"content": "<tool_call>"},
                    "151658": {"content": "</tool_call>"}
                }
            }"#,
        )
        .unwrap();
        qtk.finalize_special_tables();
        assert_eq!(qtk.pre_tokenizer, PreTokenizerKind::Qwen2);
        assert!(!qtk.eos_token_ids.contains(&0));
        assert!(qtk.eos_token_ids.contains(&151645));
        assert!(qtk.eos_token_ids.contains(&151643));
        assert_eq!(
            qtk.encode("<tool_call>!a</tool_call><|im_end|>"),
            vec![151657, 0, 2, 151658, 151645]
        );
    }
}
