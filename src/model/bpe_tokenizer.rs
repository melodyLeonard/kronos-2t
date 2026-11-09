#![forbid(unsafe_code)]
//! Byte-level BPE tokenizer supporting GPT-2 / SmolLM2 and Qwen2 (`vocab.json`+`merges.txt` or `tokenizer.json`).

use crate::storage::safetensors::JsonCursor;
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
    /// HunYuan (`Hy-MT2`): `Split(\p{N}{1,3})`, CJK runs, then a letters/punct/space regex.
    HunYuan,
}

/// Byte-level BPE tokenizer.
pub struct BpeTokenizer {
    vocab: Vocab,
    /// Literal tokens matched prior to BPE regex splitting (`<|im_start|>`, `<tool_call>`, etc.).
    added_tokens: Vec<(String, usize)>,
    /// Marks IDs whose `id_to_token` string is raw UTF-8 rather than GPT-2 byte-mapped characters.
    is_raw_special: Vec<bool>,
    added_first_bytes: [bool; 256],
    /// `(left id, right id)` → `(rank, merged id)`; keyed by ids so a 120k-merge vocab costs a few MB.
    merge_ranks: HashMap<(u32, u32), (u32, u32)>,
    /// Vocab id of each GPT-2 byte character (`u32::MAX` when the byte has no token).
    byte_char_ids: [u32; 256],
    char_to_byte: HashMap<char, u8>,
    pub pre_tokenizer: PreTokenizerKind,
    pub eos_token_id: usize,
    pub eos_token_ids: Vec<usize>,
}

/// Token strings in one arena plus an open-addressing id table: a few MB for a 120k vocab where
/// `HashMap<String, usize>` + `Vec<String>` cost ~30 MB of heap.
#[derive(Clone, Debug, Default)]
pub struct Vocab {
    text: String,
    /// `(start, len)` into `text` per id; `(0, 0)` for ids with no token.
    spans: Vec<(u32, u32)>,
    /// Linear-probing table of ids (`u32::MAX` = empty), at most half full.
    table: Vec<u32>,
}

fn hash_bytes(b: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &x in b {
        h = (h ^ u64::from(x)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h ^ (h >> 29)
}

impl Vocab {
    fn ensure_table(&mut self) {
        if self.table.is_empty() {
            self.table = vec![u32::MAX; 16];
        }
    }

    /// Grows the id range to at least `n` ids (new ids hold no token).
    pub fn reserve_ids(&mut self, n: usize) {
        if n > self.spans.len() {
            self.spans.resize(n, (0, 0));
        }
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn get(&self, id: usize) -> Option<&str> {
        let &(start, len) = self.spans.get(id)?;
        Some(&self.text[start as usize..(start + len) as usize])
    }

    pub fn id(&self, s: &str) -> Option<usize> {
        if s.is_empty() || self.table.is_empty() {
            return None;
        }
        let mask = self.table.len() - 1;
        let mut i = hash_bytes(s.as_bytes()) as usize & mask;
        loop {
            let id = self.table[i];
            if id == u32::MAX {
                return None;
            }
            if self.get(id as usize) == Some(s) {
                return Some(id as usize);
            }
            i = (i + 1) & mask;
        }
    }

    /// Sets the string of `id` (growing the vocab if needed); an earlier string for `id` stops resolving.
    pub fn set(&mut self, id: usize, s: &str) {
        if id >= self.spans.len() {
            self.spans.resize(id + 1, (0, 0));
        }
        let start = self.text.len() as u32;
        self.text.push_str(s);
        self.spans[id] = (start, s.len() as u32);
        if s.is_empty() {
            return;
        }
        self.ensure_table();
        if (self.spans.len() + 1) * 2 > self.table.len() {
            self.table = vec![u32::MAX; (self.spans.len() * 4).next_power_of_two()];
            for i in 0..self.spans.len() {
                if self.spans[i].1 != 0 && i != id {
                    self.insert(i);
                }
            }
        }
        self.insert(id);
    }

    fn insert(&mut self, id: usize) {
        let (start, len) = self.spans[id];
        let mask = self.table.len() - 1;
        let mut i = hash_bytes(&self.text.as_bytes()[start as usize..(start + len) as usize]) as usize & mask;
        loop {
            let cur = self.table[i];
            if cur == u32::MAX || cur as usize == id || self.get(cur as usize) == self.get(id) {
                self.table[i] = id as u32;
                return;
            }
            i = (i + 1) & mask;
        }
    }
}

impl BpeTokenizer {
    /// Id of the exact token string `s`, if it is in the vocabulary.
    pub fn token_id(&self, s: &str) -> Option<usize> {
        self.vocab.id(s)
    }

    /// The vocabulary string of `id` (byte-mapped for regular tokens).
    pub fn token_str(&self, id: usize) -> Option<&str> {
        self.vocab.get(id)
    }

    pub fn vocab_len(&self) -> usize {
        self.vocab.len()
    }

    /// True when the first `n` ids of both vocabularies hold identical strings.
    pub fn same_vocab_prefix(&self, other: &Self, n: usize) -> bool {
        (0..n).all(|i| self.vocab.get(i) == other.vocab.get(i))
    }

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
            // Mapped, not read: a 9 MB `tokenizer.json` on the heap would otherwise stay in the
            // footprint (the allocator keeps freed large blocks) on memory-capped hosts.
            let mapped = MappedText::open(&tok_json_path)?;
            Self::from_tokenizer_json_str(mapped.as_str())
                .map_err(|msg| io::Error::new(io::ErrorKind::InvalidData, msg))?
        } else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no vocab.json+merges.txt or tokenizer.json in {}", model_dir.display()),
            ));
        };

        if tok_json_path.exists() {
            if let Ok(mapped) = MappedText::open(&tok_json_path) {
                tok.apply_tokenizer_json_metadata(mapped.as_str())
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
        let mut vocab = Vocab::default();
        parse_vocab_object(&mut cur, &mut vocab)?;
        let vocab_size = vocab.len();
        let mut is_raw_special = vec![false; vocab_size];
        let mut added_tokens = Vec::new();
        for id in 0..vocab_size {
            let tok = vocab.get(id).unwrap_or("");
            if tok.starts_with("<|") && tok.ends_with("|>") {
                is_raw_special[id] = true;
                added_tokens.push((tok.to_string(), id));
            }
        }

        let merges = merges_txt
            .lines()
            .filter(|l| !l.is_empty() && !l.starts_with("#version"))
            .filter_map(|l| l.split_once(' '))
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let merge_ranks = build_merge_ranks(merges, &vocab);

        let (byte_to_char, char_to_byte) = build_gpt2_byte_unicode_tables();
        let eos_opt = vocab.id("<|endoftext|>");
        let mut out = Self {
            byte_char_ids: build_byte_char_ids(&byte_to_char, &vocab),
            vocab,
            added_tokens,
            is_raw_special,
            added_first_bytes: [false; 256],
            merge_ranks,
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
        let mut vocab: Option<Vocab> = None;
        let mut merges = Vec::new();
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
                        if let Some(k) = parse_pre_tokenizer_kind(&mut cur)? {
                            pre_kind = k;
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
                                        let mut v = Vocab::default();
                                        parse_vocab_object(&mut cur, &mut v)?;
                                        crate::model::memory_budget::mem_trace("tok: vocab parsed");
                                        vocab = Some(v);
                                    }
                                    "merges" => match &vocab {
                                        // Usual order (vocab first): rank the merges as they stream by.
                                        Some(v) => merge_ranks = parse_merges_array(&mut cur, Some(v))?.1,
                                        None => merges = parse_merges_array(&mut cur, None)?.0,
                                    },
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

        crate::model::memory_budget::mem_trace("tok: json scanned (vocab + ranks built)");
        let mut vocab = vocab.ok_or_else(|| "tokenizer.json missing model.vocab".to_string())?;
        if let Some(max_added) = extra_added.iter().map(|&(_, id)| id).max() {
            vocab.reserve_ids(max_added + 1);
        }
        let vocab_size = vocab.len();
        let mut is_raw_special = vec![false; vocab_size];
        for id in 0..vocab_size {
            let tok = vocab.get(id).unwrap_or("");
            if tok.starts_with("<|") && tok.ends_with("|>") {
                is_raw_special[id] = true;
            }
        }

        if !merges.is_empty() {
            merge_ranks = build_merge_ranks(merges, &vocab);
        }
        let (byte_to_char, char_to_byte) = build_gpt2_byte_unicode_tables();
        let eos_opt = vocab.id("<|endoftext|>");
        let mut out = Self {
            byte_char_ids: build_byte_char_ids(&byte_to_char, &vocab),
            vocab,
            added_tokens: Vec::new(),
            is_raw_special,
            added_first_bytes: [false; 256],
            merge_ranks,
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
                        if let Some(k) = parse_pre_tokenizer_kind(&mut cur)? {
                            self.pre_tokenizer = k;
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
            if let Some(id) = self.vocab.id(&eos) {
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
        if id >= self.is_raw_special.len() {
            self.is_raw_special.resize(id + 1, false);
        }
        self.vocab.set(id, &content);
        self.is_raw_special[id] = true;
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
        for special_eos in [crate::config::markup::IM_END, crate::config::markup::END_OF_TEXT] {
            if let Some(id) = self.vocab.id(special_eos) {
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
                        if let Some(id) = self.vocab.id(digit_str) {
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
            PreTokenizerKind::HunYuan => {
                for piece in hunyuan_regex_split(text) {
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
        // Every intermediate merge result is itself a vocab token, so the word is a list of ids.
        let mut word: Vec<u32> =
            raw_bytes.iter().map(|&b| self.byte_char_ids[b as usize]).filter(|&id| id != u32::MAX).collect();

        while word.len() >= 2 {
            let mut best: Option<(usize, u32, u32)> = None; // (index, rank, merged id)
            for i in 0..word.len() - 1 {
                if let Some(&(rank, merged)) = self.merge_ranks.get(&(word[i], word[i + 1])) {
                    if best.is_none_or(|(_, r, _)| rank < r) {
                        best = Some((i, rank, merged));
                    }
                }
            }
            let Some((first_idx, _, merged)) = best else {
                break;
            };
            let (left, right) = (word[first_idx], word[first_idx + 1]);
            let mut next_word = Vec::with_capacity(word.len() - 1);
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
                    next_word.push(merged);
                    i += 2;
                } else {
                    next_word.push(word[i]);
                    i += 1;
                }
            }
            word = next_word;
        }
        out_ids.extend(word.iter().map(|&id| id as usize));
    }

    /// Appends the raw UTF-8 bytes of a single token `id` to `raw_bytes` in `O(1)` time.
    #[inline]
    pub fn append_token_bytes(&self, id: usize, raw_bytes: &mut Vec<u8>) {
        let Some(tok) = self.vocab.get(id) else {
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

/// Streams `{"token": id, ...}` straight into `vocab`: one transient `String` at a time, so the
/// heap never holds a second copy of the vocabulary.
fn parse_vocab_object(cur: &mut JsonCursor<'_>, vocab: &mut Vocab) -> Result<(), String> {
    cur.expect_byte(b'{')?;
    if cur.peek() != Some(b'}') {
        loop {
            let tok = cur.parse_string()?;
            cur.expect_byte(b':')?;
            let id = cur.parse_u64()? as usize;
            vocab.set(id, &tok);
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

/// A read-only mapping of a UTF-8 file.
struct MappedText(crate::storage::mmap_pager::sys::RawMmap);

impl MappedText {
    fn open(path: &Path) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty file"));
        }
        let mapped = crate::storage::mmap_pager::sys::RawMmap::map_read_only(&file, len)?;
        std::str::from_utf8(mapped.as_slice()).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self(mapped))
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(self.0.as_slice()).unwrap_or("")
    }
}

/// Merge pairs in rank order; keyed by token ids, which is what the merge loop compares.
type MergeRanks = HashMap<(u32, u32), (u32, u32)>;

fn add_merge(ranks: &mut MergeRanks, vocab: &Vocab, rank: usize, a: &str, b: &str) {
    let (Some(l), Some(r)) = (vocab.id(a), vocab.id(b)) else { return };
    let Some(m) = vocab.id(&format!("{a}{b}")) else { return };
    ranks.entry((l as u32, r as u32)).or_insert((rank as u32, m as u32));
}

fn build_merge_ranks(merges: Vec<(String, String)>, vocab: &Vocab) -> MergeRanks {
    let mut ranks = HashMap::with_capacity(merges.len());
    for (rank, (a, b)) in merges.into_iter().enumerate() {
        add_merge(&mut ranks, vocab, rank, &a, &b);
    }
    ranks
}

fn build_byte_char_ids(byte_to_char: &[char; 256], vocab: &Vocab) -> [u32; 256] {
    let mut ids = [u32::MAX; 256];
    let mut buf = [0u8; 4];
    for (b, &c) in byte_to_char.iter().enumerate() {
        if let Some(id) = vocab.id(c.encode_utf8(&mut buf) as &str) {
            ids[b] = id as u32;
        }
    }
    ids
}

/// Parses the merges array; with `vocab` known they are ranked on the fly (returned as `.1`),
/// otherwise collected as string pairs (`.0`).
fn parse_merges_array(cur: &mut JsonCursor<'_>, vocab: Option<&Vocab>) -> Result<(Vec<(String, String)>, MergeRanks), String> {
    cur.expect_byte(b'[')?;
    let mut merges = Vec::new();
    let mut ranks = HashMap::with_capacity(vocab.map_or(0, Vocab::len));
    let mut rank = 0usize;
    let mut push = |merges: &mut Vec<(String, String)>, rank: usize, a: &str, b: &str| match vocab {
        Some(v) => add_merge(&mut ranks, v, rank, a, b),
        None => merges.push((a.to_string(), b.to_string())),
    };
    if cur.peek() != Some(b']') {
        loop {
            if cur.peek() == Some(b'[') {
                cur.expect_byte(b'[')?;
                let a = cur.parse_string()?;
                cur.expect_byte(b',')?;
                let b = cur.parse_string()?;
                cur.expect_byte(b']')?;
                push(&mut merges, rank, &a, &b);
            } else {
                let s = cur.parse_string()?;
                if let Some((a, b)) = s.split_once(' ') {
                    push(&mut merges, rank, a, b);
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
    drop(push);
    Ok((merges, ranks))
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

/// Pre-tokenizer family from the `Split` regexes in `tokenizer.json` (`None` = GPT-2 / SmolLM2).
fn parse_pre_tokenizer_kind(cur: &mut JsonCursor<'_>) -> Result<Option<PreTokenizerKind>, String> {
    if cur.peek() == Some(b'n') {
        cur.consume_literal(b"null")?;
        return Ok(None);
    }
    let mut kind = None;
    fn scan_obj(cur: &mut JsonCursor<'_>, kind: &mut Option<PreTokenizerKind>) -> Result<(), String> {
        match cur.peek() {
            Some(b'{') => {
                cur.expect_byte(b'{')?;
                if cur.peek() != Some(b'}') {
                    loop {
                        let k = cur.parse_string()?;
                        cur.expect_byte(b':')?;
                        if k == "Regex" && cur.peek() == Some(b'"') {
                            let pat = cur.parse_string()?;
                            // Qwen3.5 widens letters to `[\p{L}\p{M}]+`; the splitter is the same for unmarked text.
                            if pat.contains("\\p{N}{1,3}") || pat.contains("\\p{P}\\p{S}]?[\\p{L}\\p{M}]+") {
                                *kind = Some(PreTokenizerKind::HunYuan);
                            } else if kind.is_none()
                                && (pat.contains("[^\\r\\n\\p{L}\\p{N}]?\\p{L}+")
                                    || pat.contains("[^\r\n\\p{L}\\p{N}]?\\p{L}+")
                                    || pat.contains("[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+"))
                            {
                                *kind = Some(PreTokenizerKind::Qwen2);
                            }
                        } else {
                            scan_obj(cur, kind)?;
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
                        scan_obj(cur, kind)?;
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
    scan_obj(cur, &mut kind)?;
    Ok(kind)
}

/// Combining marks (`\p{M}`): the generic combining blocks plus the common script-specific ones.
fn is_mark(c: char) -> bool {
    matches!(
        c as u32,
        0x0300..=0x036F
            | 0x0483..=0x0489
            | 0x0591..=0x05BD
            | 0x05BF
            | 0x05C1..=0x05C2
            | 0x05C4..=0x05C5
            | 0x05C7
            | 0x0610..=0x061A
            | 0x064B..=0x065F
            | 0x0670
            | 0x06D6..=0x06DC
            | 0x06DF..=0x06E4
            | 0x06E7..=0x06E8
            | 0x06EA..=0x06ED
            | 0x0900..=0x0903
            | 0x093A..=0x093C
            | 0x093E..=0x094F
            | 0x0951..=0x0957
            | 0x0962..=0x0963
            | 0x0981..=0x0983
            | 0x09BC
            | 0x09BE..=0x09CD
            | 0x0E31
            | 0x0E34..=0x0E3A
            | 0x0E47..=0x0E4E
            | 0x1AB0..=0x1AFF
            | 0x1DC0..=0x1DFF
            | 0x20D0..=0x20FF
            | 0x302A..=0x302F
            | 0x3099..=0x309A
            | 0xFE00..=0xFE0F
            | 0xFE20..=0xFE2F
    )
}

/// `[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]`: CJK ideographs, hiragana and katakana.
fn is_cjk_run_char(c: char) -> bool {
    matches!(c as u32, 0x4E00..=0x9FA5 | 0x3040..=0x309F | 0x30A0..=0x30FF)
}

/// Pure-Rust HunYuan pre-tokenizer (`Hy-MT2`, `HunYuan-7B`): the `tokenizer.json` sequence
/// `Split(\p{N}{1,3})`, `Split(CJK+)`, then
/// `[ascii-punct][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
/// all `Isolated` (text between matches stays as its own piece).
pub fn hunyuan_regex_split(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    // Stage 1: numbers in groups of at most three digits; everything else in maximal runs.
    let mut run_start = 0usize;
    let mut digit_start: Option<(usize, usize)> = None;
    for (bi, ch) in text.char_indices() {
        if ch.is_numeric() {
            match digit_start {
                Some((start, count)) if count < 3 => digit_start = Some((start, count + 1)),
                Some((start, _)) => {
                    out.push(&text[start..bi]);
                    digit_start = Some((bi, 1));
                }
                None => {
                    if run_start < bi {
                        split_cjk_then_main(&text[run_start..bi], &mut out);
                    }
                    digit_start = Some((bi, 1));
                }
            }
            run_start = bi + ch.len_utf8();
        } else if let Some((start, _)) = digit_start.take() {
            out.push(&text[start..bi]);
            run_start = bi;
        }
    }
    if let Some((start, _)) = digit_start {
        out.push(&text[start..]);
    } else if run_start < text.len() {
        split_cjk_then_main(&text[run_start..], &mut out);
    }
    out
}

/// Stage 2 (`CJK+` runs isolated) feeding stage 3 for the rest.
fn split_cjk_then_main<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut run_start = 0usize;
    let mut cjk_start: Option<usize> = None;
    for (bi, ch) in text.char_indices() {
        if is_cjk_run_char(ch) {
            if cjk_start.is_none() {
                if run_start < bi {
                    hunyuan_main_split(&text[run_start..bi], out);
                }
                cjk_start = Some(bi);
            }
        } else if let Some(start) = cjk_start.take() {
            out.push(&text[start..bi]);
            run_start = bi;
        }
    }
    match cjk_start {
        Some(start) => out.push(&text[start..]),
        None if run_start < text.len() => hunyuan_main_split(&text[run_start..], out),
        None => {}
    }
}

/// Stage 3 of [`hunyuan_regex_split`].
fn hunyuan_main_split<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let slice_chars = |start_c: usize, end_c: usize| -> &'a str {
        let sb = chars[start_c].0;
        let eb = if end_c < n { chars[end_c].0 } else { text.len() };
        &text[sb..eb]
    };
    let is_lm = |c: char| c.is_alphabetic() || is_mark(c);
    let is_punct_sym = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric() && !is_mark(c) && !c.is_control();
    // End (exclusive, in chars) of the leftmost-alternative match starting at `i`, if any.
    let match_at = |i: usize| -> Option<usize> {
        let ch = chars[i].1;
        // 1. `[ascii-punct][A-Za-z]+`
        if ch.is_ascii_punctuation() && i + 1 < n && chars[i + 1].1.is_ascii_alphabetic() {
            let mut j = i + 2;
            while j < n && chars[j].1.is_ascii_alphabetic() {
                j += 1;
            }
            return Some(j);
        }
        // 2. `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`
        if is_lm(ch) {
            let mut j = i + 1;
            while j < n && is_lm(chars[j].1) {
                j += 1;
            }
            return Some(j);
        }
        if ch != '\r' && ch != '\n' && !is_punct_sym(ch) && i + 1 < n && is_lm(chars[i + 1].1) {
            let mut j = i + 2;
            while j < n && is_lm(chars[j].1) {
                j += 1;
            }
            return Some(j);
        }
        // 3. ` ?[\p{P}\p{S}]+[\r\n]*`
        if is_punct_sym(ch) || (ch == ' ' && i + 1 < n && is_punct_sym(chars[i + 1].1)) {
            let mut j = if ch == ' ' { i + 2 } else { i + 1 };
            while j < n && is_punct_sym(chars[j].1) {
                j += 1;
            }
            while j < n && matches!(chars[j].1, '\r' | '\n') {
                j += 1;
            }
            return Some(j);
        }
        // 4-6. `\s*[\r\n]+|\s+(?!\S)|\s+`
        if ch.is_whitespace() {
            let mut j = i + 1;
            let mut last_nl: Option<usize> = matches!(ch, '\r' | '\n').then_some(i);
            while j < n && chars[j].1.is_whitespace() {
                if matches!(chars[j].1, '\r' | '\n') {
                    last_nl = Some(j);
                }
                j += 1;
            }
            return Some(match last_nl {
                Some(nl) => nl + 1,
                None if j < n && j - i > 1 => j - 1,
                None => j,
            });
        }
        None
    };
    let mut i = 0usize;
    while i < n {
        match match_at(i) {
            Some(end) => {
                out.push(slice_chars(i, end));
                i = end;
            }
            None => {
                let mut j = i + 1;
                while j < n && match_at(j).is_none() {
                    j += 1;
                }
                out.push(slice_chars(i, j));
                i = j;
            }
        }
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

        assert_eq!(
            hunyuan_regex_split("Hello world 12345, don't stop! 今天天气\n\nok"),
            vec![
                "Hello", " world", " ", "123", "45", ",", " don", "'t", " stop", "!", " ", "今天天气", "\n\n", "ok"
            ]
        );
        assert_eq!(hunyuan_regex_split("à bientôt .com"), vec!["à", " bientôt", " .", "com"]);

        // Reference ids from HF `tokenizers` on the same `tokenizer.json`.
        let hy_dir = Path::new("/Users/melodyleonard/Documents/project/kronos-2t/models/Hy-MT2-1.8B");
        if hy_dir.join("tokenizer.json").exists() {
            let tk = BpeTokenizer::from_model_dir(hy_dir).unwrap();
            assert_eq!(tk.pre_tokenizer, PreTokenizerKind::HunYuan);
            assert!(tk.eos_token_ids.contains(&120020));
            let cases: [(&str, &[usize]); 4] = [
                ("see you tomorrow", &[6759, 444, 32104]),
                (
                    "Hello world 12345, don't stop! 今天天气\n\nok",
                    &[16883, 2385, 206, 7827, 2445, 11, 2475, 1638, 4800, 0, 206, 7331, 12523, 286, 821],
                ),
                (
                    "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate the following text into Spanish:\n\nsee you tomorrow<｜hy_Assistant｜>",
                    &[120000, 120006, 26403, 252, 1873, 2602, 1079, 12920, 1015, 6759, 444, 32104, 120007],
                ),
                (
                    "à bientôt .com Ẹ káàbọ̀ 😂 ça va?",
                    &[4116, 273, 1116, 44890, 1102, 1004, 206, 3018, 116, 633, 2451, 4116, 65, 16912, 74317, 82457, 210, 72941, 21106, 30],
                ),
            ];
            for (text, ids) in cases {
                assert_eq!(tk.encode(text), ids, "{text:?}");
                assert_eq!(tk.decode(ids), text);
            }
        }

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
